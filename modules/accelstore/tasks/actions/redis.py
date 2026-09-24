#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Redis persistence runs: one point per boot, on one arm.

Two arms: ``accelstore`` (``redis-server --accelstore-config <spdk.json>
--accelstore-device <bdev>`` inside the usual SPDK device lifecycle) and a block
stack id -- the same binary with the ``--accelstore-*`` flags omitted and
``--dir`` on a filesystem ``actions/posix_storage.py`` provides. Both arms start
from a fresh store or filesystem and a freshly created data directory.

``redis-server`` runs under a ``taskset`` holding the Redis cores, the rewrite
core and (on the accelstore arm) the reactor cores, ``redis-benchmark`` under a
disjoint one. The rewrite core also goes to ``redis-server`` itself on every arm
as ``--aof-rewrite-cpulist``/``--bgsave-cpulist``. ``tasks/redis.py``'s
:func:`~modules.accelstore.tasks.redis.core_layout` derives the sets, this action
applies and records them.

Each run writes the server's own log, ``redis-benchmark --csv`` per phase, ``INFO
persistence`` around each phase, a ``.json`` with the parsed rows and
the timed walls, and a ``*_flags.json`` recording the arm, the command lines, the
core layout and what an acknowledged write guarantees on each side.
"""

import json
import re
import shlex
import subprocess
import uuid
from dataclasses import asdict, is_dataclass
from typing import Optional

from core.tasks.actions import ActionContext
from core.tasks.actions.registry import register_action

from modules.accelstore.tasks import redis as redis_grid
from modules.accelstore.tasks.block_stacks import check_capacity
from modules.accelstore.tasks.storage_stacks import BLOCK_STACKS
from modules.accelstore.tasks.actions.accelstore import _save_output_to_file
from modules.accelstore.tasks.actions.bench import _write_flags_sidecar
from modules.accelstore.tasks.actions.posix_storage import posix_stack
from modules.accelstore.tasks.actions.storage_env import (
    precondition_device,
    spdk_device,
)

ACCELSTORE_ARM = redis_grid.ACCELSTORE_ARM
REDIS_ARMS = redis_grid.redis_arms(BLOCK_STACKS)

SERVER_BIN = "redis-server"
BENCH_BIN = "redis-benchmark"
CLI_BIN = "redis-cli"

# /var/tmp, not /tmp: a tmpfs would lose accelstore.tc across the restart.
ACCELSTORE_DATA_DIR = "/var/tmp/redis-accelstore"
POSIX_SUBDIR = "redis"

SERVER_START_TIMEOUT_S = 900  # minutes: SPDK start plus an AOF counter-cache build
PERSISTENCE_OP_TIMEOUT_S = 3600
SHUTDOWN_TIMEOUT_S = 300  # the accelstore exit runs a metadata epoch and SPDK teardown
POLL_INTERVAL_S = 0.005

ACCELSTORE_ACK = {
    "summary": "barrier",
    "detail": (
        "an acknowledged SET is in server.aof_buf; what makes it durable is the "
        "AOF flush's LOG append plus the policy's barrier -- appendfsync always "
        "is an as_fsync epoch on the main thread before the reply, everysec is "
        "one on the bio thread once a second. The epoch commits the trusted "
        "counter (--sync-trigger on-barrier, which the fork sets), so a "
        "surviving generation is attested and a rollback to an earlier one is "
        "detected at load"
    ),
}
REDIS_POSIX_ACK = {
    "summary": "fdatasync",
    "detail": (
        "an acknowledged SET is in server.aof_buf; appendfsync always is an "
        "fdatasync on the main thread before the reply and everysec one on the "
        "bio thread once a second. Whether that fdatasync is durable on the "
        "medium is then the block stack's promise, below -- and no stack here "
        "offers freshness: a rolled-back generation of the AOF loads clean"
    ),
}


def _variant_tokens(cfg: dict) -> list:
    """Return leaf tokens for every knob that departs from the point's defaults."""
    tokens = []
    point = redis_grid.REDIS_POINTS.get(cfg.get("point", ""))
    if point is not None:
        for key, token, default in (
            ("requests", "n", point.requests),
            ("fill_requests", "fill", point.fill_requests or None),
        ):
            value = cfg.get(key)
            if value is not None and default is not None and int(value) != default:
                tokens.append(f"{token}-{int(value)}")

    rtt = cfg.get("tc_rtt_us")
    if rtt:
        tokens.append(f"tcrtt-{int(rtt)}us")
    workers = cfg.get("workers")
    if workers is not None and int(workers) != redis_grid.DEFAULT_WORKERS:
        tokens.append(f"workers-{int(workers)}")
    return tokens


def _redis_path(name: str, cfg: dict) -> tuple:
    components = ["redis", name, cfg.get("arm", ACCELSTORE_ARM), cfg.get("point", "")]
    tokens = _variant_tokens(cfg)
    if tokens:
        components.append("_".join(tokens))
    return tuple(components)


@register_action("redis", path_fn=_redis_path)
def run_redis(
    ctx: ActionContext,
    point: str,
    arm: str = ACCELSTORE_ARM,
    config: str = None,
    pci_device: str = None,
    dev_path: str = None,
    device: str = "Nvme1n1",
    hugepages_gb: int = 32,
    guest_cores: Optional[int] = None,
    guest_cores_source: Optional[str] = None,
    workers: int = redis_grid.DEFAULT_WORKERS,
    tc_rtt_us: Optional[int] = None,
    requests: Optional[int] = None,
    fill_requests: Optional[int] = None,
    capacity_gb: Optional[int] = None,
    skip_precondition: bool = False,
    precondition_scheme: str = None,
):
    """Run one Redis persistence point on one arm.

    Args:
        ctx: Action context (VM/host runner, output dirs, timestamp)
        point: Point name from ``tasks/redis.py``'s table (always_c1,
               always_c8p4, everysec_c8p4, everysec_c50p16, dataset)
        arm: "accelstore", or a block stack id from BLOCK_STACKS (the POSIX
             baseline's filesystem sits on that stack)
        config: SPDK bdev configuration as JSON string (accelstore arm only)
        pci_device: PCI address the bdev config attaches (accelstore arm only)
        dev_path: Guest block device path -- pre-conditioning target on both
             arms, and what a POSIX arm's stack formats
        device: SPDK block device name created by the bdev config
        hugepages_gb: Hugepages reserved for the accelstore arm's SPDK app
        guest_cores: The guest's vCPU count, for the core layout. Read from the
             guest's nproc when absent
        guest_cores_source: Where that count came from, for the run's record
        workers: SPDK worker threads the store runs (--accelstore-workers,
             accelstore arm only)
        tc_rtt_us: Trusted-counter round trip to model, in microseconds
             (--accelstore-tc-rtt-us). 0/absent keeps the local counter
        requests / fill_requests: Override the point's own bounds. Each one
             that departs from the table gets a token in the result leaf
        capacity_gb: Window the POSIX arm's stack is given, in GB (10^9). None
             hands over the whole device
        skip_precondition: Skip secure erase + pre-conditioning
        precondition_scheme: Pre-conditioning scheme to use

    Output structure:
        bench-result/redis/{name}/{accelstore | stack}/{point}/[variant]/{timestamp}.log
                                                                       /{timestamp}.json
                                                                       /{timestamp}_flags.json
                                                                       /{timestamp}_server.log

    ``.json`` holds the parsed ``redis-benchmark --csv`` row of every phase, the
    ``INFO persistence`` counters around it and -- on ``dataset`` -- the BGSAVE,
    BGREWRITEAOF and restart-load walls. ``_flags.json`` holds the arm, the stack
    and its acknowledgement semantics, the point's parameters and derived store
    geometry, the core layout and the command lines. ``_server.log`` is
    redis-server's own output.
    """
    if arm not in REDIS_ARMS:
        raise ValueError(
            f"Invalid arm '{arm}'. Must be one of: {', '.join(REDIS_ARMS)}"
        )
    is_accelstore = arm == ACCELSTORE_ARM
    stack = None if is_accelstore else BLOCK_STACKS[arm]

    resolved = redis_grid.redis_point(point).with_overrides(
        requests=requests, fill_requests=fill_requests
    )

    vm = ctx.vm
    cores = int(guest_cores) if guest_cores else _guest_core_count(vm)
    source = guest_cores_source or (
        "passed by the caller" if guest_cores else "the guest's nproc"
    )
    layout = redis_grid.core_layout(cores, arm, source=source)
    print(
        f"Core layout ({cores} vCPUs, {source}, arm {arm}): "
        f"server taskset={layout.server_cpu_list} "
        f"rewrite={layout.rewrite_cpu_list} "
        f"client taskset={layout.client_cpu_list}"
        + (f" reactor mask={layout.core_mask}" if is_accelstore else "")
    )

    # Checked before anything is erased, so a refused run leaves the device intact.
    if not is_accelstore:
        for warning in check_capacity(arm, capacity_gb, None):
            print(f"NOTE: {warning}")

    date = ctx.timestamp
    output_log = ctx.outputdir_guest / f"{date}.log"
    server_log = ctx.outputdir_guest / f"{date}_server.log"

    precondition_device(
        ctx, vm, dev_path, skip=skip_precondition, scheme=precondition_scheme
    )

    record = {
        "schema": 1,
        "timestamp": date,
        "client": SERVER_BIN,
        "arm": arm,
        "point": resolved.name,
        "point_config": resolved.as_dict(),
        "overrides": {
            key: value
            for key, value in (
                ("requests", requests),
                ("fill_requests", fill_requests),
            )
            if value is not None
        },
        "core_layout": layout.as_dict(),
        "server_args": resolved.server_args(
            "<data-dir>",
            layout.rewrite_cpu_list,
        ),
    }

    if is_accelstore:
        record.update(
            {
                "backend": "accelstore",
                "device": device,
                "pci_device": pci_device,
                "workers": workers,
                "tc_rtt_us": tc_rtt_us or 0,
                "tc_backend": "mock-remote" if tc_rtt_us else "local",
                "store_geometry": {
                    "cluster_size": resolved.cluster_size,
                    "md_pages": resolved.md_pages,
                },
                "ack_semantics": ACCELSTORE_ACK,
            }
        )
        results, commands = _run_accelstore_arm(
            ctx,
            resolved,
            layout,
            config,
            pci_device,
            device,
            hugepages_gb,
            workers,
            tc_rtt_us,
            output_log,
            server_log,
        )
    else:
        record.update(
            {
                "backend": "posix",
                "stack": _stack_record(stack),
                "capacity_gb": capacity_gb,
                "ack_semantics": {
                    "redis": REDIS_POSIX_ACK,
                    "stack": _stack_ack(stack),
                },
            }
        )
        results, commands = _run_posix_arm(
            ctx, resolved, layout, arm, dev_path, capacity_gb, output_log, server_log
        )

    record["commands"] = commands
    _write_metrics(ctx, resolved, arm, layout, results)
    _write_flags_sidecar(ctx, record)

    failures = [phase for phase, row in results["phases"].items() if row.get("error")]
    if results.get("error"):
        raise RuntimeError(
            f"redis point '{resolved.name}' on arm '{arm}' failed: "
            f"{results['error']}; see {output_log} and {server_log}"
        )
    if failures:
        raise RuntimeError(
            f"redis point '{resolved.name}' on arm '{arm}': "
            f"{', '.join(failures)} failed; see {output_log}"
        )


def _run_accelstore_arm(
    ctx,
    point,
    layout,
    config,
    pci_device,
    device,
    hugepages_gb,
    workers,
    tc_rtt_us,
    output_log,
    server_log,
):
    if not config or not pci_device:
        raise ValueError("config and pci_device are required on the accelstore arm")
    try:
        json.loads(config)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON config: {e}")

    vm = ctx.vm

    # spdk_device destroys the blobstore on entry, so the data dir has to go too.
    with spdk_device(
        vm, config, pci_device, device, hugepages_gb=hugepages_gb
    ) as config_file:
        data_dir = _fresh_data_dir(vm, ACCELSTORE_DATA_DIR)
        accel_args = point.accelstore_args(
            config_file, device, layout.core_mask, workers, tc_rtt_us
        )
        return _run_point(
            ctx, point, layout, data_dir, accel_args, output_log, server_log
        )


def _run_posix_arm(
    ctx, point, layout, stack_id, dev_path, capacity_gb, output_log, server_log
):
    if not dev_path:
        raise ValueError(f"dev_path is required on the '{stack_id}' arm")

    vm = ctx.vm

    with posix_stack(vm, stack_id, dev_path, capacity_gb=capacity_gb) as mount_dir:
        data_dir = _fresh_data_dir(vm, f"{mount_dir}/{POSIX_SUBDIR}")
        # No --accelstore-* flags: same binary, backend left inactive.
        return _run_point(ctx, point, layout, data_dir, [], output_log, server_log)


def _run_point(ctx, point, layout, data_dir, accel_args, output_log, server_log):
    """Drive one point end to end and return its results and command lines."""
    vm = ctx.vm
    transcript = []
    commands = []
    results = {"phases": {}, "data_dir": data_dir}

    server_argv = [
        SERVER_BIN,
        *point.server_args(data_dir, layout.rewrite_cpu_list),
        *accel_args,
    ]
    commands.append(_taskset(layout.server_cpu_list, server_argv))

    # Recorded rather than raised, so the sidecars still land when a phase dies.
    try:
        _run_phases(
            ctx, point, layout, server_argv, server_log, results, transcript, commands
        )
    except Exception as exc:  # noqa: BLE001
        results["error"] = str(exc)
        print(f"ERROR: {exc}")

    _save_output_to_file(vm, output_log, "\n".join(transcript), "", 0)
    return results, commands


def _run_phases(
    ctx, point, layout, server_argv, server_log, results, transcript, commands
):
    """Run fill, run, bgsave, bgrewriteaof and restart_load, in that order."""
    vm = ctx.vm
    server = _start_server(vm, layout, server_argv, server_log, transcript)
    results["server_start_s"] = server["wall_s"]
    try:
        if point.fill_requests:
            results["phases"]["fill"] = _benchmark_phase(
                vm, layout, "fill", point.fill_args(), transcript, commands
            )
            _quiesce(vm, layout, transcript)

        results["phases"]["run"] = _benchmark_phase(
            vm, layout, "run", point.benchmark_args(), transcript, commands
        )

        if point.persistence_ops:
            results["phases"]["bgsave"] = _persistence_phase(
                vm, layout, "BGSAVE", "rdb_bgsave_in_progress", transcript, commands
            )
            results["phases"]["bgrewriteaof"] = _persistence_phase(
                vm,
                layout,
                "BGREWRITEAOF",
                "aof_rewrite_in_progress",
                transcript,
                commands,
            )
            results["info_before_restart"] = _info(vm, layout, "persistence")
    finally:
        _shutdown(vm, layout, transcript)

    if point.persistence_ops:
        # Outside the try/finally above: this phase starts and stops its own server.
        results["phases"]["restart_load"] = _restart_phase(
            vm, layout, server_argv, server_log, transcript, commands
        )


def _start_server(vm, layout, server_argv, server_log, transcript):
    """Launch redis-server in the background under taskset, wait for PONG."""
    script_path = f"/tmp/redis_launch_{uuid.uuid4().hex[:8]}.sh"
    pid_file = f"/tmp/redis_{uuid.uuid4().hex[:8]}.pid"
    argv = _taskset(layout.server_cpu_list, server_argv)
    script = f"""#!/usr/bin/env bash
set -u
# setsid so the server outlives this ssh session.
setsid {shlex.join(argv)} >> {shlex.quote(str(server_log))} 2>&1 &
echo $! > {shlex.quote(pid_file)}
"""
    vm.create_file(script_path, script)
    result = vm.ssh_cmd(["bash", script_path], check=True, stderr=subprocess.PIPE)
    transcript.append(f"$ {shlex.join(argv)}")
    if result.stdout:
        transcript.append(result.stdout.rstrip())
    vm.ssh_cmd(["rm", "-f", script_path], check=False)

    wall_s = _wait_for_pong(vm, layout, server_log, transcript)
    return {"pid_file": pid_file, "wall_s": wall_s}


def _wait_for_pong(vm, layout, server_log, transcript):
    """Poll PING inside the guest until it answers and return the wall in seconds."""
    ping = shlex.join(
        _taskset(
            layout.client_cpu_list,
            [CLI_BIN, "-p", str(redis_grid.REDIS_PORT), "ping"],
        )
    )
    probe = f"""#!/usr/bin/env bash
set -u
# Nanoseconds and shell arithmetic because the guest image has no bc.
start=$(date +%s%N)
deadline=$(( start + {int(SERVER_START_TIMEOUT_S * 1_000_000_000)} ))
while :; do
  if {ping} 2>/dev/null | grep -q PONG; then
    end=$(date +%s%N)
    echo "PONG $(( end - start ))"
    exit 0
  fi
  if [ "$(date +%s%N)" -gt "$deadline" ]; then
    echo "TIMEOUT"
    exit 1
  fi
  sleep {POLL_INTERVAL_S}
done
"""
    out = _run_probe(vm, probe, "redis_pong")
    transcript.append(f"[wait-for-pong] {out.strip()}")
    if not out.startswith("PONG"):
        tail = vm.ssh_cmd(["tail", "-n", "40", str(server_log)], check=False)
        raise RuntimeError(
            f"redis-server did not answer PING within {SERVER_START_TIMEOUT_S}s. "
            f"Server log tail:\n{tail.stdout}"
        )
    return int(out.split()[1]) / 1e9


def _benchmark_phase(vm, layout, phase, bench_args, transcript, commands):
    """One redis-benchmark invocation under the client taskset, parsed."""
    argv = _taskset(layout.client_cpu_list, [BENCH_BIN, *bench_args])
    commands.append(argv)
    transcript.append(f"$ {shlex.join(argv)}")
    info_before = _info(vm, layout, "persistence")
    result = vm.ssh_cmd(argv, check=False, stderr=subprocess.PIPE)
    transcript.append(result.stdout or "")
    if result.stderr:
        transcript.append(result.stderr)
    info_after = _info(vm, layout, "persistence")

    row = {
        "phase": phase,
        "exit_code": result.returncode,
        "csv": result.stdout,
        "tests": _parse_benchmark_csv(result.stdout),
        "info_before": info_before,
        "info_after": info_after,
    }
    if result.returncode != 0:
        row["error"] = f"{BENCH_BIN} exited {result.returncode}"
    elif not row["tests"]:
        row["error"] = f"{BENCH_BIN} produced no parseable --csv row"
    return row


def _persistence_phase(vm, layout, command, in_progress_field, transcript, commands):
    """Issue BGSAVE / BGREWRITEAOF and time it out of INFO persistence."""
    cli = _taskset(layout.client_cpu_list, [CLI_BIN, "-p", str(redis_grid.REDIS_PORT)])
    commands.append(cli + [command])
    cli_s = shlex.join(cli)
    probe = f"""#!/usr/bin/env bash
set -u
# Wait out anything already running, so the wall is this command's alone.
while {cli_s} info persistence | tr -d '\\r' | grep -q '^{in_progress_field}:1$'; do
  sleep {POLL_INTERVAL_S}
done
start=$(date +%s%N)
{cli_s} {shlex.quote(command)} > /dev/null || {{ echo "ISSUE_FAILED"; exit 1; }}
deadline=$(( start + {int(PERSISTENCE_OP_TIMEOUT_S * 1_000_000_000)} ))
while :; do
  if ! {cli_s} info persistence | tr -d '\\r' | grep -q '^{in_progress_field}:1$'; then
    end=$(date +%s%N)
    echo "DONE $(( end - start ))"
    exit 0
  fi
  if [ "$(date +%s%N)" -gt "$deadline" ]; then
    echo "TIMEOUT"
    exit 1
  fi
  sleep {POLL_INTERVAL_S}
done
"""
    out = _run_probe(vm, probe, f"redis_{command.lower()}")
    transcript.append(f"[{command}] {out.strip()}")
    info_after = _info(vm, layout, "persistence")
    row = {"phase": command.lower(), "info_after": info_after}
    if out.startswith("DONE"):
        row["wall_s"] = int(out.split()[1]) / 1e9
        # A failed child also leaves `*_in_progress` at 0, so read the status too.
        status_field = (
            "rdb_last_bgsave_status"
            if command == "BGSAVE"
            else "aof_last_bgrewrite_status"
        )
        status = info_after.get(status_field)
        row["status"] = status
        if status != "ok":
            row["error"] = f"{command} reported {status_field}={status}"
    else:
        row["error"] = f"{command} did not complete: {out.strip()}"
    return row


def _restart_phase(vm, layout, server_argv, server_log, transcript, commands):
    """Time a cold start over the persisted store: launch to first PONG."""
    transcript.append("[restart] starting a fresh server over the persisted store")
    commands.append(_taskset(layout.server_cpu_list, server_argv) + ["# restart"])
    try:
        server = _start_server(vm, layout, server_argv, server_log, transcript)
        row = {"phase": "restart_load", "wall_s": server["wall_s"]}
        row["info_after"] = _info(vm, layout, "persistence")
        dbsize = vm.ssh_cmd(
            _taskset(
                layout.client_cpu_list,
                [CLI_BIN, "-p", str(redis_grid.REDIS_PORT), "dbsize"],
            ),
            check=False,
        )
        row["dbsize"] = (dbsize.stdout or "").strip()
        transcript.append(f"[restart] dbsize={row['dbsize']}")
    except Exception as exc:
        row = {"phase": "restart_load", "error": str(exc)}
    finally:
        _shutdown(vm, layout, transcript)
    return row


def _taskset(cpu_list: str, argv: list) -> list:
    return ["taskset", "-c", cpu_list, *argv]


def _fresh_data_dir(vm, path: str) -> str:
    """Remove and recreate `path`, so the run starts on an empty data directory."""
    vm.ssh_cmd(["rm", "-rf", path], check=True)
    vm.ssh_cmd(["mkdir", "-p", path], check=True)
    return path


def _guest_core_count(vm) -> int:
    """The guest's online vCPU count, when the sweep did not pass it."""
    result = vm.ssh_cmd(["nproc"], check=True)
    return int(result.stdout.strip())


def _run_probe(vm, script: str, tag: str) -> str:
    """Write a probe script into the guest, run it, return its stdout."""
    path = f"/tmp/{tag}_{uuid.uuid4().hex[:8]}.sh"
    vm.create_file(path, script)
    result = vm.ssh_cmd(["bash", path], check=False, stderr=subprocess.PIPE)
    vm.ssh_cmd(["rm", "-f", path], check=False)
    if result.stderr:
        print(f"{tag}: {result.stderr.strip()}")
    return (result.stdout or "").strip()


def _info(vm, layout, section: str) -> dict:
    """``INFO <section>`` as a flat dict, or {} if the server is not answering."""
    result = vm.ssh_cmd(
        _taskset(
            layout.client_cpu_list,
            [CLI_BIN, "-p", str(redis_grid.REDIS_PORT), "info", section],
        ),
        check=False,
    )
    if result.returncode != 0:
        return {}
    fields = {}
    for line in (result.stdout or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        fields[key] = value
    return fields


def _quiesce(vm, layout, transcript) -> None:
    """Wait until no save, rewrite or load is in progress or scheduled."""
    cli = _taskset(layout.client_cpu_list, [CLI_BIN, "-p", str(redis_grid.REDIS_PORT)])
    cli_s = shlex.join(cli)
    probe = f"""#!/usr/bin/env bash
set -u
deadline=$(( $(date +%s%N) + {int(PERSISTENCE_OP_TIMEOUT_S * 1_000_000_000)} ))
while :; do
  info=$({cli_s} info persistence | tr -d '\\r')
  if ! echo "$info" | grep -qE '^(rdb_bgsave_in_progress|aof_rewrite_in_progress|aof_rewrite_scheduled|loading):1$'; then
    echo "QUIESCENT"
    exit 0
  fi
  if [ "$(date +%s%N)" -gt "$deadline" ]; then
    echo "TIMEOUT"
    exit 1
  fi
  sleep 0.05
done
"""
    out = _run_probe(vm, probe, "redis_quiesce")
    transcript.append(f"[quiesce] {out.strip()}")


def _shutdown(vm, layout, transcript) -> None:
    """SHUTDOWN NOSAVE -- a save here would land inside a measured sequence."""
    vm.ssh_cmd(
        _taskset(
            layout.client_cpu_list,
            [CLI_BIN, "-p", str(redis_grid.REDIS_PORT), "shutdown", "nosave"],
        ),
        check=False,
    )
    # The device stays bound to vfio until the accelstore exit path finishes.
    probe = f"""#!/usr/bin/env bash
set -u
deadline=$(( $(date +%s%N) + {int(SHUTDOWN_TIMEOUT_S * 1_000_000_000)} ))
while pgrep -x {shlex.quote(SERVER_BIN)} > /dev/null; do
  if [ "$(date +%s%N)" -gt "$deadline" ]; then
    echo "STILL_RUNNING"
    exit 0
  fi
  sleep 0.05
done
echo "EXITED"
"""
    out = _run_probe(vm, probe, "redis_shutdown")
    if out.startswith("STILL_RUNNING"):
        print(
            f"WARNING: {SERVER_BIN} still running {SHUTDOWN_TIMEOUT_S}s after "
            "SHUTDOWN NOSAVE; killing it"
        )
        vm.ssh_cmd(["pkill", "-x", SERVER_BIN], check=False)
    transcript.append(f"[shutdown] SHUTDOWN NOSAVE -> {out.strip()}")


_CSV_FIELD = re.compile(r'"([^"]*)"')

# The header redis-benchmark --csv prints before its rows (src/redis-benchmark.c).
BENCHMARK_CSV_COLUMNS = (
    "test",
    "rps",
    "avg_latency_ms",
    "min_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "max_latency_ms",
)


def _parse_benchmark_csv(output: str) -> list:
    """Parse redis-benchmark ``--csv`` rows into dicts, dropping the header."""
    rows = []
    for line in (output or "").splitlines():
        fields = _CSV_FIELD.findall(line.strip())
        if len(fields) != len(BENCHMARK_CSV_COLUMNS):
            continue
        if fields[0] == "test":
            continue
        row = dict(zip(BENCHMARK_CSV_COLUMNS, fields))
        for key in BENCHMARK_CSV_COLUMNS[1:]:
            try:
                row[key] = float(row[key])
            except (TypeError, ValueError):
                row[key] = None
        rows.append(row)
    return rows


def _write_metrics(ctx, point, arm, layout, results) -> None:
    """The run's own metrics sidecar, next to the log."""
    path = ctx.outputdir_host / f"{ctx.timestamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "timestamp": ctx.timestamp,
        "arm": arm,
        "point": point.name,
        "point_config": point.as_dict(),
        "core_layout": layout.as_dict(),
        **results,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Recorded the run's metrics in {path}")


def _json_safe(value):
    """Render a stack-table field json-serializable, falling back to ``str``."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _json_safe(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _stack_ack(stack) -> object:
    """What an acknowledged write guarantees on this stack, as the table says."""
    if hasattr(stack, "ack_record"):
        return _json_safe(stack.ack_record())
    return _json_safe(getattr(stack, "ack_semantics", None))


def _stack_record(stack) -> dict:
    """The block stack's own fields, for the flags sidecar."""
    return {
        "id": stack.id,
        "label": _json_safe(getattr(stack, "label", None)),
        "kernel": getattr(stack, "kernel", None),
        "default_capacity_gb": getattr(stack, "default_capacity_gb", None),
        "is_stacked": bool(getattr(stack, "is_stacked", False)),
        "notes": _json_safe(getattr(stack, "notes", ()) or ()),
    }
