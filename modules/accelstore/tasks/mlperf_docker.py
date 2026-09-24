#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Command lines for an MLPerf run: the container layout, the docker argv and
the guest-side helpers around it.

Nothing here imports ``core`` or ``invoke``, so a run's command line can be
built and asserted without a harness, a VM or a docker daemon.
"""

import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

from modules.accelstore.tasks.mlperf_variants import (
    ACCELSTORE_URI_SCHEME,
    DEFAULT_STACK,
    MlperfRunContext,
    SESSION_MAX_OP_BYTES_DEFAULT,
    SESSION_WORKERS_DEFAULT,
    stack_for,
    variant as _variant,
)
from modules.accelstore.tasks.sync_modes import SyncMode

DOCKER_BIN = "docker"

MLPERF_IMAGE = "mlperf:nix"
MLPERF_SERVICE = "mlperf.service"

CONTAINER_RESULTS = "/results"
CONTAINER_DATA = "/data"
# Guest-local, never the 9p share: a slow -f destination drops trace entries.
CONTAINER_TRACES = "/traces"
# Shared by the run's containers: a counter restarting at zero reads as a rollback.
CONTAINER_TC = "/tc"
CONTAINER_TC_FILE = f"{CONTAINER_TC}/counter.bin"
CONTAINER_BDEV_CONFIG = "/run/accelstore/bdev.json"
CONTAINER_CHECKPOINT_DIR = f"{CONTAINER_DATA}/checkpoints"

# A namespace and not a path: the scheme is what routes DLIO to the AccelStore writer.
ACCELSTORE_CHECKPOINT_URI = f"{ACCELSTORE_URI_SCHEME}://checkpoints"

# Next to the bdev config, not in /dev/shm, which --ipc=host makes the guest's.
CONTAINER_IPC_SOCKET = "/run/accelstore/master.sock"
CONTAINER_MASTER_LOG = "/tmp/pyaccelstore-master.log"
MASTER_ENTRY = ("python3", "-m", "pyaccelstore")
MASTER_READY_LINE = "pyaccelstore: master ready"
MASTER_FAILED_LINE = "pyaccelstore: master failed to start"
MASTER_READY_TIMEOUT_S = 300

# Mounted read-only into a traced run: spdk_trace_record is not in the image.
NIX_STORE = "/nix/store"


@dataclass(frozen=True)
class Mount:
    """One bind mount, source and target as the guest sees them."""

    source: str
    target: str
    readonly: bool = False

    def args(self) -> list[str]:
        spec = f"{self.source}:{self.target}"
        if self.readonly:
            spec += ":ro"
        return ["-v", spec]


SPDK_DEVICE_MOUNTS = (
    Mount("/dev/hugepages", "/dev/hugepages"),
    Mount("/dev/vfio", "/dev/vfio"),
)

NIX_STORE_MOUNT = Mount(NIX_STORE, NIX_STORE, readonly=True)


def docker_run_argv(
    image: str = MLPERF_IMAGE,
    *,
    entrypoint: Sequence[str] = (),
    args: Sequence[str] = (),
    env: Optional[dict] = None,
    mounts: Sequence[Mount] = (),
    privileged: bool = True,
    ipc_host: bool = True,
    network_host: bool = True,
    workdir: Optional[str] = None,
    docker_bin: str = DOCKER_BIN,
) -> list[str]:
    """The whole ``docker run`` command line for one phase of a run.

    ``entrypoint`` empty runs the image's own ENTRYPOINT (``mlpstorage``). A
    multi-word entrypoint is split the way docker requires: ``--entrypoint``
    takes exactly one program and the rest becomes the first words of the
    command. The environment is emitted sorted, so two identical runs record
    the same command line.
    """
    argv = [docker_bin, "run", "--rm"]
    if privileged:
        argv.append("--privileged")
    if ipc_host:
        argv.append("--ipc=host")
    if network_host:
        argv.append("--network=host")
    if workdir:
        argv += ["--workdir", workdir]
    for mount in mounts:
        argv += mount.args()
    for name in sorted(env or {}):
        argv += ["-e", f"{name}={env[name]}"]

    entry = list(entrypoint)
    if entry:
        argv += ["--entrypoint", entry[0]]
    argv.append(image)
    return argv + entry[1:] + list(args)


def master_argv(
    *,
    module: str,
    device: str,
    config: str = CONTAINER_BDEV_CONFIG,
    socket: str = CONTAINER_IPC_SOCKET,
    tc_path: str = CONTAINER_TC_FILE,
    workers: Optional[int] = None,
    max_op_bytes: Optional[int] = None,
    crypto_chunk_size: Optional[int] = None,
    pipeline_bytes: Optional[int] = None,
    cluster_size: Optional[int] = None,
    md_pages: Optional[int] = None,
    sync: Optional[SyncMode] = None,
    silent: bool = True,
) -> list[str]:
    """The store owner's command line, in the container's coordinates.

    The ranks run ``MODE=client`` and drop every owner-only setting, so the
    geometry, the trusted counter and the sync policy reach the store here or
    not at all. ``workers`` and ``max_op_bytes`` are always emitted, so the
    master matches the session's defaults rather than the runner's.
    """
    argv = [
        *MASTER_ENTRY,
        "--module",
        module,
        "--config",
        config,
        "--device",
        device,
        "--socket",
        socket,
        "--workers",
        str(workers if workers is not None else SESSION_WORKERS_DEFAULT),
        "--max-op-bytes",
        str(max_op_bytes if max_op_bytes is not None else SESSION_MAX_OP_BYTES_DEFAULT),
    ]
    for flag, value in (
        ("--crypto-chunk-size", crypto_chunk_size),
        ("--pipeline-bytes", pipeline_bytes),
        ("--cluster-size", cluster_size),
        ("--md-pages", md_pages),
    ):
        if value is not None:
            argv += [flag, str(value)]
    if tc_path:
        argv += ["--tc-path", tc_path]
    argv += (sync or SyncMode()).client_args()
    if silent:
        argv.append("--silent")
    return argv


def with_ipc_master(
    *,
    master: Sequence[str],
    command: Sequence[str],
    socket: str = CONTAINER_IPC_SOCKET,
    log: str = CONTAINER_MASTER_LOG,
    ready_timeout_s: int = MASTER_READY_TIMEOUT_S,
) -> tuple[list[str], list[str]]:
    """``(entrypoint, args)`` running ``command`` against a dedicated master.

    The master starts first and the benchmark waits for its ready line, since a
    rank that connects earlier gets ECONNREFUSED. A master that never comes up
    fails the run -- there is deliberately no fallback to local mode.
    mlpstorage's exit status is the run's, but the master's own failure is not
    swallowed: its stop is the store's final metadata sync.
    """
    script = f"""set -eu
sock={shlex.quote(socket)}
log={shlex.quote(log)}
rm -f "$sock"
: > "$log"
{shlex.join(master)} >"$log" 2>&1 &
master=$!
ready=0
waited=0
while [ "$waited" -lt {int(ready_timeout_s)} ]; do
    if grep -q {shlex.quote(MASTER_READY_LINE)} "$log" 2>/dev/null; then
        ready=1
        break
    fi
    if grep -q {shlex.quote(MASTER_FAILED_LINE)} "$log" 2>/dev/null; then
        break
    fi
    if ! kill -0 "$master" 2>/dev/null; then
        break
    fi
    sleep 1
    waited=$((waited + 1))
done
if [ "$ready" -ne 1 ]; then
    echo "mlperf: the AccelStore store master did not come up within {int(ready_timeout_s)}s;" \\
         "every rank of this run is a client of it" >&2
    sed 's/^/master: /' "$log" >&2
    kill -TERM "$master" 2>/dev/null || true
    wait "$master" 2>/dev/null || true
    exit 1
fi
echo "mlperf: AccelStore store master is serving $sock"
status=0
"$@" || status=$?
kill -TERM "$master" 2>/dev/null || true
master_status=0
wait "$master" || master_status=$?
sed 's/^/master: /' "$log"
if [ "$status" -ne 0 ]; then
    exit "$status"
fi
if [ "$master_status" -ne 0 ]; then
    echo "mlperf: the store master exited $master_status; its stop is the store's" \\
         "final metadata sync, so this run did not end cleanly" >&2
    exit "$master_status"
fi
"""
    return ["sh", "-c", script, "mlperf-ipc-master"], list(command)


def wait_for_image_argv(
    image: str = MLPERF_IMAGE,
    *,
    timeout_s: int = 900,
    service: str = MLPERF_SERVICE,
) -> list[str]:
    """Poll until the boot-time ``docker load`` has finished.

    Polls for the image rather than the unit: a ``RemainAfterExit`` oneshot is
    'active' the moment it succeeds, and the image may also be loaded by hand.
    """
    return [
        "sh",
        "-c",
        f"""set -e
deadline=$(( $(date +%s) + {int(timeout_s)} ))
while ! {DOCKER_BIN} image inspect {shlex.quote(image)} >/dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "timeout waiting for {image}" >&2
        systemctl status --no-pager {shlex.quote(service)} >&2 || true
        exit 1
    fi
    sleep 5
done
""",
    ]


def image_id_argv(image: str = MLPERF_IMAGE) -> list[str]:
    """The image digest, for the sidecar: which build produced these numbers."""
    return [DOCKER_BIN, "image", "inspect", "-f", "{{.Id}}", image]


def resolve_binary_argv(name: str) -> list[str]:
    """Absolute path of a guest binary, dereferenced out of the nix profile."""
    return ["sh", "-c", f'readlink -f "$(command -v {shlex.quote(name)})"']


def list_traces_argv(trace_dir: str) -> list[str]:
    """Every dump in the trace directory, one per line (empty if none)."""
    quoted = shlex.quote(trace_dir)
    return ["sh", "-c", f"ls -1 {quoted}/*.trace 2>/dev/null || true"]


def trace_convert_argv(dump: str) -> list[str]:
    """Convert one dump to JSON, guest-side and after the container is gone."""
    quoted = shlex.quote(dump)
    return ["sh", "-c", f"spdk_trace -f {quoted} -j > {quoted}.json"]


def trace_missed_scan_argv(trace_dir: str) -> list[str]:
    """Grep the recorders' logs for the overrun line that invalidates a dump."""
    quoted = shlex.quote(trace_dir)
    return [
        "sh",
        "-c",
        f"grep -H -i 'missed' {quoted}/*.log 2>/dev/null || true",
    ]


def build_run_context(
    arm,
    *,
    systemname: str,
    client_host_memory_gb: float = 0.0,
    extra: Sequence[str] = (),
) -> MlperfRunContext:
    """The run context for one arm, in the container's coordinates.

    An accelstore arm has no filesystem to point at, so its checkpoint folder is
    an ``accelstore://`` namespace URI rather than a mounted directory.
    """
    return MlperfRunContext(
        results_dir=CONTAINER_RESULTS,
        systemname=systemname,
        client_host_memory_gb=client_host_memory_gb,
        checkpoint_folder=(
            ACCELSTORE_CHECKPOINT_URI if arm.is_accelstore else CONTAINER_CHECKPOINT_DIR
        ),
        extra=tuple(extra),
    )


def mlperf_path_components(name: str, cfg: dict) -> tuple:
    """Result-directory components for one (variant, arm, stack) point.

    ``mlperf/<vm name>/<family>/<variant>/<arm>`` plus a leaf token per
    dimension that is not at its default -- the block stack, then the sync
    policy -- the same shape ``actions/bench.py`` uses, so a sweep point never
    overwrites the default run.
    """
    entry = _variant(cfg["variant"])
    components = [
        "mlperf",
        name,
        *entry.result_path(
            cfg["arm"], stack_for(cfg["arm"], cfg.get("stack", DEFAULT_STACK))
        ),
    ]
    leaf = SyncMode.from_config(cfg).path_component()
    if leaf:
        components.append(leaf)
    return tuple(components)


def flags_record(
    *,
    timestamp: str,
    variant,
    arm,
    sync: SyncMode,
    run_context,
    tracing: Optional[str],
    trace_entries: Optional[int],
    image: str,
    image_id: str,
    phases: Sequence[dict],
    device: Optional[dict] = None,
    stack=None,
) -> dict:
    """What this row ran, next to the row.

    A phase entry is ``{"phase", "argv", "env", "exit"}``, the argv being the
    whole docker command line. ``stack`` is the block stack under the arm's
    filesystem, and its geometry rides in ``device``.
    """
    resolved_stack = stack_for(arm.id, stack or DEFAULT_STACK)
    return {
        "schema": 1,
        "timestamp": timestamp,
        "variant": variant.id,
        "family": variant.family,
        "division": variant.division,
        "label": variant.label,
        "arm": arm.id,
        "arm_kind": arm.kind,
        "module": arm.module,
        "stack": resolved_stack.id,
        "deviations": list(variant.deviations(arm.id)) + list(resolved_stack.notes),
        "sync_mode": sync.label(),
        "sync_args": sync.client_args(),
        "tracing": tracing,
        "trace_entries": trace_entries if tracing else None,
        "image": image,
        "image_id": image_id,
        "device": device,
        # Only what this run set: the context's defaults are the empty values.
        "run_context": {
            key: value
            for key, value in asdict(run_context).items()
            if value not in ("", 0, None, ())
        },
        # Named keys only: a phase dict also carries the phase's whole output.
        "phases": [
            {key: phase[key] for key in ("phase", "argv", "env", "exit")}
            for phase in phases
        ],
    }


def posix_data_subdirs(data_dir: str) -> list[str]:
    """The guest paths behind the POSIX arms' container directories."""
    return [data_dir + CONTAINER_CHECKPOINT_DIR[len(CONTAINER_DATA) :]]


def guest_run_dirs(root: str, timestamp: str) -> tuple[str, str]:
    """The guest-local (trace, trusted counter) directories of one run.

    Under ``/var/tmp`` rather than the result directory, which is a 9p mount.
    """
    base = Path(root) / timestamp
    return str(base / "traces"), str(base / "tc")
