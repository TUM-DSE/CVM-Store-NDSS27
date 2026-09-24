#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""MLPerf Storage runs: one (variant, arm) per VM boot.

Runs the benchmark in the ``mlperf:nix`` container the ``--mlperf`` guest loads
at boot, wrapped in the usual AccelStore preamble: precondition the device,
then bind it to SPDK or format a POSIX stack on it. A point is two container
runs -- the ``init`` that stamps the results directory, then the measured
``run`` -- over one VM boot, one SPDK lifecycle and one trusted counter file.

What each run is comes from the variant table (``tasks/mlperf_variants.py``);
the container command lines come from ``tasks/mlperf_docker.py``. A variant
whose store has a master (more than one rank process) runs ``python3 -m
pyaccelstore`` alongside the ranks in the same container, composed by
``mlperf_docker.with_ipc_master``.
"""

import json
import shlex
import subprocess
from typing import Optional

from core.tasks.actions import ActionContext
from core.tasks.actions.registry import register_action

from modules.accelstore.tasks import mlperf_docker as docker
from modules.accelstore.tasks.block_stacks import check_capacity
from modules.accelstore.tasks.mlperf_variants import DEFAULT_STACK, ENV_PREFIX
from modules.accelstore.tasks.mlperf_variants import MLPSTORAGE_BIN
from modules.accelstore.tasks.mlperf_variants import stack_for
from modules.accelstore.tasks.mlperf_variants import variant as mlperf_variant
from modules.accelstore.tasks.actions.accelstore import (
    DEFAULT_TRACE_ENTRIES,
    _save_output_to_file,
)
from modules.accelstore.tasks.actions.posix_storage import (
    posix_stack,
    posix_storage_record,
)
from modules.accelstore.tasks.actions.storage_env import (
    precondition_device,
    spdk_device,
)
from modules.accelstore.tasks.sync_modes import SyncMode

TRACING_MODES = {"tracing": "1", "small-trace": "small"}

# Guest-local, so the trace dumps and the counter are not written to the 9p share.
GUEST_RUN_ROOT = "/var/tmp/mlperf"

# Own subdirectory: `mlpstorage init` refuses a non-empty directory with no sentinel.
RESULTS_SUBDIR = "mlperf"

# mlpstorage builds the mpirun command line itself and only passes
# --allow-run-as-root when told to (utils.py:646-647).
MPI_DEPLOYMENT_ARGS = ("--allow-run-as-root", "--mpi-btl", "tcp")


def _mlperf_path(name: str, cfg: dict) -> tuple:
    return docker.mlperf_path_components(name, cfg)


@register_action("mlperf", path_fn=_mlperf_path)
def run_mlperf(
    ctx: ActionContext,
    variant: str,
    arm: str,
    stack: str = DEFAULT_STACK,
    stack_capacity_gb: Optional[int] = None,
    config: str = None,
    pci_device: str = None,
    dev_path: str = None,
    device: str = "Nvme1n1",
    orgname: str = "TUM-DSE",
    systemname: str = "cvm-store",
    client_host_memory_gb: float = 0.0,
    mlpstorage_args: Optional[str] = None,
    image: str = docker.MLPERF_IMAGE,
    image_wait_s: int = 900,
    hugepages_gb: int = 32,
    workers: Optional[int] = None,
    max_op_bytes: Optional[int] = None,
    crypto_chunk_size: Optional[int] = None,
    pipeline_bytes: Optional[int] = None,
    cluster_size: Optional[int] = None,
    md_pages: Optional[int] = None,
    silent: bool = True,
    tracing: Optional[str] = None,
    trace_entries: Optional[int] = DEFAULT_TRACE_ENTRIES,
    trace_crypto_chunks: Optional[bool] = None,
    skip_precondition: bool = False,
    precondition_scheme: str = None,
    trigger: Optional[str] = None,
    every_n: Optional[int] = None,
    interval_us: Optional[int] = None,
    coupling: Optional[str] = None,
    commit_every: Optional[int] = None,
    blocking: bool = False,
    metadata_only: bool = False,
    tc_backend: Optional[str] = None,
    tc_rtt_us: Optional[int] = None,
    tc_rtt_model: Optional[str] = None,
    tc_rtt_interval: Optional[int] = None,
):
    """Run one MLPerf Storage (variant, arm) point in the guest's container.

    Args:
        ctx: Action context (VM/host runner, output dirs, timestamp)
        variant: Variant id from the table (tasks/mlperf_variants.py)
        arm: Arm id — accelstore-log, accelstore-fs, or a POSIX baseline:
             posix-direct, posix-buffered, or direct (the s3dlio direct://
             path). A pair the variant table marks blocked is refused here
        stack: Block stack under a POSIX arm's filesystem — ext4 (default),
             ext4-aes-xts-plain (LUKS2 dm-crypt, AES-XTS), rollbaccine or
             rollbaccine-f1m. An accelstore arm resolves to `raw` regardless:
             it owns the device through vfio
        stack_capacity_gb: Window the POSIX stack is given, in GB (10^9); None
             takes the whole device. Ignored on an accelstore arm, and checked
             against what the stack can work at (tasks/block_stacks.py)
        config: SPDK bdev configuration as JSON string (accelstore arms)
        pci_device: PCI address the bdev config attaches (accelstore arms)
        dev_path: Guest block device path — pre-conditioning target on both arm
             kinds, and what a POSIX arm formats
        device: SPDK block device name created by the bdev config
        orgname: Organization name `mlpstorage init` stamps the results
             directory with
        systemname: --systemname, required by every emitting mlpstorage command
        client_host_memory_gb: --client-host-memory-in-gb, the guest's memory;
             the checkpoint sizing is computed from it
        mlpstorage_args: Extra arguments appended to every mlpstorage command of
             this run, after the deployment knobs
        image: Container image (default mlperf:nix)
        image_wait_s: Seconds to wait for the boot-time `docker load`
        hugepages_gb: Hugepages reserved in the guest for the accelstore arms
        workers / max_op_bytes / crypto_chunk_size / pipeline_bytes: AccelStore
             store geometry, passed through MLPS_ACCELSTORE_* (accelstore arms
             only). Unset, the variant's own values decide; on a row with a
             store master the resolved geometry is what the master is started
             with
        cluster_size / md_pages: the blobstore's own geometry — cluster size and
             the metadata pages reserved at init. Unset, the variant decides
        silent: MLPS_ACCELSTORE_SILENT — suppress the library's per-operation
             output
        tracing: SPDK trace recording on the accelstore arms — "tracing",
             "small-trace", or None. The dumps are converted and copied into the
             output directory afterwards; refused on a row with a store master
        trace_entries: Trace ring entries per lcore; a power of two in
             [32768, 67108864], default DEFAULT_TRACE_ENTRIES
        trace_crypto_chunks: Record the per-AEAD-chunk crypto spans (default:
             the session's own default)
        skip_precondition / precondition_scheme: as in the other actions
        trigger .. tc_rtt_interval: One SyncMode, accelstore arms only; a
             non-default mode is refused on a POSIX arm

    Output structure:
        bench-result/mlperf/{name}/{family}/{variant}/{arm}/[stack]/[sync]/{timestamp}.log
                                                                 /{timestamp}_init.log
                                                                 /{timestamp}_flags.json
                                                                 /{timestamp}_tc.bin
                                                                 /mlperf/      (mlpstorage's own tree)
                                                                 /traces/      (if tracing)
    """
    entry = mlperf_variant(variant)
    entry.validate()
    resolved_arm = entry.arm(arm)
    resolved_stack = stack_for(resolved_arm, stack)
    # An accelstore arm owns the device through vfio: nothing to window.
    capacity_gb = None if resolved_arm.is_accelstore else stack_capacity_gb
    for warning in check_capacity(
        resolved_stack.id, capacity_gb, client_host_memory_gb
    ):
        print(f"NOTE: {warning}")
    blocked = entry.blocked_arms
    if resolved_arm.id in blocked:
        raise ValueError(
            f"{entry.id} on {resolved_arm.id} cannot run: {blocked[resolved_arm.id]}"
        )

    sync = SyncMode(
        trigger=trigger,
        every_n=every_n,
        interval_us=interval_us,
        coupling=coupling,
        commit_every=commit_every,
        blocking=blocking,
        metadata_only=metadata_only,
        tc_backend=tc_backend,
        tc_rtt_us=tc_rtt_us,
        tc_rtt_model=tc_rtt_model,
        tc_rtt_interval=tc_rtt_interval,
    )
    sync.validate()
    if not resolved_arm.is_accelstore and not sync.is_default():
        raise ValueError(
            f"Sync mode '{sync.label()}' has no meaning on the "
            f"'{resolved_arm.id}' arm; it has no metadata epochs"
        )
    for warning in sync.warnings():
        print(f"WARNING: {warning}")

    trace_mode = _validate_tracing(
        resolved_arm, tracing, trace_entries, trace_crypto_chunks
    )
    if trace_mode and entry.ipc_master(resolved_arm):
        raise ValueError(
            f"tracing a point whose store has a master ({entry.id} on "
            f"{resolved_arm.id}, {entry.store_processes()} rank processes) is "
            "not wired: the ring is the store master's, and the master runner "
            "has no auto-record. Trace a single-process "
            "row, or drop --tracing"
        )
    # A hand-written action_config can carry the ring size as a string.
    trace_entries = None if trace_entries is None else int(trace_entries)

    if resolved_arm.is_accelstore and not (config and pci_device):
        raise ValueError("config and pci_device are required on an accelstore arm")
    if resolved_arm.is_accelstore:
        try:
            json.loads(config)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON config: {exc}") from None
    if not dev_path and not skip_precondition:
        raise ValueError("dev_path is required unless pre-conditioning is skipped")
    if not resolved_arm.is_accelstore and not dev_path:
        raise ValueError(f"dev_path is required on the '{resolved_arm.id}' arm")

    extra = tuple(MPI_DEPLOYMENT_ARGS) + tuple(
        shlex.split(mlpstorage_args) if mlpstorage_args else ()
    )
    run_ctx = docker.build_run_context(
        resolved_arm,
        systemname=systemname,
        client_host_memory_gb=client_host_memory_gb,
        extra=extra,
    )
    entry.validate_context(run_ctx, resolved_arm)

    vm = ctx.vm
    _require_image(vm, image, image_wait_s)
    image_id = _image_id(vm, image)

    results_dir = ctx.outputdir_guest / RESULTS_SUBDIR
    trace_dir, tc_dir = docker.guest_run_dirs(GUEST_RUN_ROOT, ctx.timestamp)
    vm.ssh_cmd(["mkdir", "-p", str(results_dir), tc_dir], check=True)
    if trace_mode:
        vm.ssh_cmd(["mkdir", "-p", trace_dir], check=True)

    precondition_device(
        ctx, vm, dev_path, skip=skip_precondition, scheme=precondition_scheme
    )

    store = {
        "workers": workers,
        "max_op_bytes": max_op_bytes,
        "crypto_chunk_size": crypto_chunk_size,
        "pipeline_bytes": pipeline_bytes,
        "cluster_size": cluster_size,
        "md_pages": md_pages,
        "silent": silent,
        "tracing": trace_mode,
        "trace_dir": docker.CONTAINER_TRACES if trace_mode else None,
        "trace_entries": trace_entries if trace_mode else None,
        "trace_crypto_chunks": trace_crypto_chunks if trace_mode else None,
    }

    mounts = [docker.Mount(str(results_dir), docker.CONTAINER_RESULTS)]
    phases = []

    def phases_over(mounts):
        return _run_phases(
            ctx,
            entry,
            resolved_arm,
            run_ctx,
            sync=sync,
            mounts=mounts,
            image=image,
            device=device,
            orgname=orgname,
            store=store,
        )

    try:
        if resolved_arm.is_accelstore:
            with spdk_device(
                vm, config, pci_device, device, hugepages_gb=hugepages_gb
            ) as config_file:
                mounts += [
                    docker.Mount(
                        config_file, docker.CONTAINER_BDEV_CONFIG, readonly=True
                    ),
                    docker.Mount(tc_dir, docker.CONTAINER_TC),
                    *docker.SPDK_DEVICE_MOUNTS,
                ]
                if trace_mode:
                    # The image carries the AccelStore clients, not SPDK's tools.
                    mounts += [
                        docker.Mount(trace_dir, docker.CONTAINER_TRACES),
                        docker.NIX_STORE_MOUNT,
                    ]
                    store["record_bin"] = _resolve_binary(vm, "spdk_trace_record")
                phases = phases_over(mounts)
        else:
            with posix_stack(vm, resolved_stack.id, dev_path, capacity_gb) as data_dir:
                vm.ssh_cmd(
                    ["mkdir", "-p", *docker.posix_data_subdirs(data_dir)], check=True
                )
                mounts.append(docker.Mount(data_dir, docker.CONTAINER_DATA))
                phases = phases_over(mounts)
    finally:
        if trace_mode:
            _collect_traces(ctx, vm, trace_dir)
        if resolved_arm.is_accelstore:
            _collect_tc(ctx, vm, tc_dir)
        _write_flags_sidecar(
            ctx,
            docker.flags_record(
                timestamp=ctx.timestamp,
                variant=entry,
                arm=resolved_arm,
                sync=sync,
                run_context=run_ctx,
                tracing=tracing,
                trace_entries=trace_entries,
                image=image,
                image_id=image_id,
                phases=phases,
                stack=resolved_stack,
                device=(
                    {"device": device, "pci_device": pci_device}
                    if resolved_arm.is_accelstore
                    else posix_storage_record(resolved_stack.id, dev_path, capacity_gb)
                ),
            ),
        )

    failed = [p for p in phases if p["exit"] != 0]
    if failed:
        raise RuntimeError(
            f"{entry.id} on {resolved_arm.id}: "
            + ", ".join(f"{p['phase']} exited {p['exit']}" for p in failed)
        )


def _validate_tracing(
    arm, tracing, trace_entries, trace_crypto_chunks
) -> Optional[str]:
    """Return the session's spelling of the trace mode, or None, and refuse the rest."""
    if tracing is not None and tracing not in TRACING_MODES:
        raise ValueError(
            f"Invalid tracing mode '{tracing}'. Must be "
            f"{', '.join(repr(m) for m in TRACING_MODES)}, or None"
        )
    if not arm.is_accelstore:
        if tracing or trace_crypto_chunks is not None:
            raise ValueError(
                f"SPDK tracing only exists on an accelstore arm; "
                f"'{arm.id}' runs no SPDK application"
            )
        return None

    # The client indexes the ring with `offset & (num_entries - 1)`.
    if trace_entries is not None:
        trace_entries = int(trace_entries)
        if not 32768 <= trace_entries <= 67108864:
            raise ValueError(
                f"trace_entries {trace_entries} out of range [32768, 67108864]"
            )
        if trace_entries & (trace_entries - 1):
            raise ValueError(f"trace_entries {trace_entries} is not a power of two")
    return TRACING_MODES.get(tracing)


def _require_image(vm, image: str, wait_s: int) -> None:
    """Wait out the boot-time `docker load`, or say which guest is missing."""
    if vm.ssh_cmd(["sh", "-c", "command -v docker"], check=False).returncode != 0:
        raise RuntimeError(
            "No docker in the guest. The MLPerf container ships in the "
            "--mlperf guest image: build one with "
            "`oma-arvio accelstore.build-guest --mlperf`"
        )
    print(f"Waiting for {image} (the boot-time docker load moves ~10 GB)")
    result = vm.ssh_cmd(
        docker.wait_for_image_argv(image, timeout_s=wait_s),
        check=False,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{image} is not loaded in the guest after {wait_s}s "
            f"({(result.stderr or '').strip()}). Build the guest with "
            "`oma-arvio accelstore.build-guest --mlperf`"
        )


def _image_id(vm, image: str) -> str:
    """The image digest, so a row says which container produced it."""
    result = vm.ssh_cmd(docker.image_id_argv(image), check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _resolve_binary(vm, name: str) -> str:
    result = vm.ssh_cmd(docker.resolve_binary_argv(name), check=False)
    path = result.stdout.strip()
    if result.returncode != 0 or not path:
        raise RuntimeError(
            f"{name} is not in the guest PATH; it comes from the dpdk_spdk "
            "module and is what a traced run forks per process"
        )
    return path


def _run_phases(
    ctx,
    entry,
    arm,
    run_ctx,
    *,
    sync,
    mounts,
    image,
    device,
    orgname,
    store,
) -> list:
    """Run every container phase of this point in order, stopping at the first failure."""
    phases = []
    _resolve_geometry(entry, arm, store)
    # The sentinel gate exempts only init/version/lockfile/rules-coverage
    # (main.py:57-60, :294-334).
    phases.append(
        _run_phase(
            ctx,
            "init",
            entrypoint=(),
            args=["init", orgname, docker.CONTAINER_RESULTS],
            env={},
            mounts=mounts,
            image=image,
        )
    )
    if phases[-1]["exit"] != 0:
        return phases

    entrypoint, args = _with_master(
        entry,
        arm,
        sync,
        device,
        store,
        args=entry.run_args(arm.id, run_ctx),
    )

    phases.append(
        _run_phase(
            ctx,
            "run",
            entrypoint=entrypoint,
            args=args,
            env=_store_env(entry, arm, sync, device, store, "run"),
            mounts=mounts,
            image=image,
        )
    )
    return phases


def _resolve_geometry(entry, arm, store) -> None:
    """Fix this point's store geometry into ``store`` and print what was asked for."""
    geometry = entry.geometry_for(
        chunk_size=store["crypto_chunk_size"],
        cluster_size=store["cluster_size"],
        md_pages=store["md_pages"],
        pipeline_bytes=store["pipeline_bytes"],
    )
    store["geometry"] = geometry
    if arm.is_accelstore and not geometry.is_default():
        print(
            f"mlperf: store geometry requested {geometry.describe()}"
            " (what the store came up with is the rank log's `pyaccelstore:"
            " store geometry: cluster-size=... requested=...` line)"
        )
        for warning in geometry.warnings():
            print(f"WARNING: {warning}")


def _with_master(entry, arm, sync, device, store, *, args):
    """``(entrypoint, args)`` for the measured run, wrapped in a store master or not."""
    if not entry.ipc_master(arm):
        return (), args
    # The wrapper replaces the image's ENTRYPOINT, so the program has to be named.
    return docker.with_ipc_master(
        master=_master_argv(entry, arm, sync, device, store),
        command=[MLPSTORAGE_BIN] + list(args),
    )


def _master_argv(entry, arm, sync, device, store) -> list:
    """The store master's command line for this point."""
    return docker.master_argv(
        module=arm.module,
        device=device,
        workers=entry.workers_for(store["workers"]),
        max_op_bytes=entry.max_op_bytes_for(store["max_op_bytes"]),
        crypto_chunk_size=store["geometry"].chunk_size,
        pipeline_bytes=store["geometry"].pipeline_bytes,
        cluster_size=store["geometry"].cluster_size,
        md_pages=store["geometry"].md_pages,
        sync=sync,
        silent=bool(store["silent"]),
    )


def _store_env(entry, arm, sync, device, store, role) -> dict:
    """The AccelStore environment for one phase; empty on a POSIX arm."""
    env = entry.env(
        arm.id,
        config=docker.CONTAINER_BDEV_CONFIG,
        device=device,
        tc_path=docker.CONTAINER_TC_FILE,
        # Set unconditionally, read only where the row runs a master.
        ipc_socket=docker.CONTAINER_IPC_SOCKET,
        sync=sync,
        workers=store["workers"],
        max_op_bytes=store["max_op_bytes"],
        crypto_chunk_size=store["geometry"].chunk_size,
        pipeline_bytes=store["geometry"].pipeline_bytes,
        cluster_size=store["geometry"].cluster_size,
        md_pages=store["geometry"].md_pages,
        tracing=store["tracing"],
        trace_dir=store["trace_dir"],
        trace_entries=store["trace_entries"],
        trace_crypto_chunks=store["trace_crypto_chunks"],
        trace_role=role if store["tracing"] else None,
        silent=store["silent"],
    )
    if env and store.get("record_bin"):
        env[ENV_PREFIX + "TRACE_RECORD_BIN"] = store["record_bin"]
    return env


def _run_phase(ctx, phase, *, entrypoint, args, env, mounts, image) -> dict:
    """One `docker run`, its log, and what it exited with."""
    argv = docker.docker_run_argv(
        image,
        entrypoint=entrypoint,
        args=args,
        env=env,
        mounts=mounts,
    )
    suffix = "" if phase == "run" else f"_{phase}"
    output_log = ctx.outputdir_guest / f"{ctx.timestamp}{suffix}.log"
    print(f"mlperf: {phase} phase -> {output_log}")
    result = ctx.vm.ssh_cmd(argv, check=False, stderr=subprocess.PIPE)
    _save_output_to_file(
        ctx.vm, output_log, result.stdout, result.stderr, result.returncode
    )
    if result.returncode != 0:
        print(f"ERROR: the {phase} phase exited {result.returncode}")
    return {
        "phase": phase,
        "argv": argv,
        "env": env,
        "exit": result.returncode,
    }


def _collect_traces(ctx, vm, trace_dir: str) -> None:
    """Convert the trace dumps, report overruns, then copy them to the output dir."""
    listing = vm.ssh_cmd(docker.list_traces_argv(trace_dir), check=False)
    dumps = [line.strip() for line in listing.stdout.splitlines() if line.strip()]
    if not dumps:
        print(
            f"WARNING: tracing was on but {trace_dir} holds no dump; the "
            "session records only while a store is open"
        )
    for dump in dumps:
        result = vm.ssh_cmd(docker.trace_convert_argv(dump), check=False)
        if result.returncode != 0:
            print(f"WARNING: spdk_trace could not convert {dump}")

    missed = vm.ssh_cmd(docker.trace_missed_scan_argv(trace_dir), check=False)
    for line in missed.stdout.splitlines():
        print(f"WARNING: trace ring overrun: {line.strip()}")

    target = ctx.outputdir_guest / "traces"
    vm.ssh_cmd(["mkdir", "-p", str(target)], check=True)
    vm.ssh_cmd(
        ["sh", "-c", f"cp -a {shlex.quote(trace_dir)}/. {shlex.quote(str(target))}/"],
        check=False,
    )
    vm.ssh_cmd(["rm", "-rf", trace_dir], check=False)
    print(f"Copied {len(dumps)} trace dump(s) to {target}")


def _collect_tc(ctx, vm, tc_dir: str) -> None:
    """Keep the trusted counter the phases shared, for provenance."""
    source = f"{tc_dir}/counter.bin"
    target = ctx.outputdir_guest / f"{ctx.timestamp}_tc.bin"
    vm.ssh_cmd(
        [
            "sh",
            "-c",
            f"test -f {shlex.quote(source)} && "
            f"cp {shlex.quote(source)} {shlex.quote(str(target))}",
        ],
        check=False,
    )
    vm.ssh_cmd(["rm", "-rf", tc_dir], check=False)


def _write_flags_sidecar(ctx, record: dict) -> None:
    """What this row ran, next to the row (as in actions/bench.py)."""
    path = ctx.outputdir_host / f"{ctx.timestamp}_flags.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"Recorded the run's flag set in {path}")
