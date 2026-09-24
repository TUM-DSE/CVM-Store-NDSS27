#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The YCSB suite: ``accelstore.run-ycsb`` and ``accelstore.parse-ycsb``.

One VM boot per (setup, size, arm, point), with the points and every derivation
they need taken from ``tasks/ycsb.py``. Imports of ``tasks/accelstore.py``,
which re-exports both tasks, are deferred into the functions that need them so
the two modules do not form an import cycle.
"""

import itertools
import time
from pathlib import Path
from typing import Optional

from invoke import task

from core.tasks import vm as vm_tasks
from core.tasks.utils.device import Devices
from core.tasks.utils.pci import check_speed

from modules.accelstore.tasks import ycsb as ycsb_grid
from modules.accelstore.tasks.block_stacks import (
    check_capacity,
    default_capacity_gb,
    sweep_window_gb,
)
from modules.accelstore.tasks.crypto_backend import CryptoBackend
from modules.accelstore.tasks.storage_stacks import BLOCK_STACKS
from modules.accelstore.tasks.sync_modes import SyncMode
from modules.accelstore.tasks.actions.ycsb import (
    ACCELSTORE_ARM,
    DEFAULT_HUGEPAGES_GB,
    YCSB_ARMS,
    check_sync_for_arm,
)

GIB = 1 << 30

# Slack for the OS and the RocksDB arm on top of hugepages and pinned buffers.
GUEST_MEMORY_SLACK_GIB = 4

YCSB_DEFAULT_ARMS = (ACCELSTORE_ARM, "ext4")


def _parse_arms(arms: Optional[str]) -> list:
    """The ``--arms`` argument: every arm is accepted, not every one defaults."""
    arm_list = (
        [a.strip() for a in arms.split(",") if a.strip()]
        if arms
        else list(YCSB_DEFAULT_ARMS)
    )
    for arm in arm_list:
        if arm not in YCSB_ARMS:
            raise ValueError(f"Unknown arm '{arm}'. Valid: {', '.join(YCSB_ARMS)}")
    return arm_list


def _require_stack_guest(setups: list, stack_ids: list) -> None:
    """Refuse a stack whose kernel is not the one the installed guest image has."""
    from core.tasks.config import BUILD_DIR

    from modules.accelstore.tasks.accelstore import _mlperf_guest_image

    for stack_id in sorted(set(stack_ids)):
        kernel = BLOCK_STACKS[stack_id].kernel
        if not kernel:
            continue
        for image in sorted({_mlperf_guest_image(s) for s in setups}):
            marker = Path(BUILD_DIR) / "image" / f"{image}.kernel"
            installed = marker.read_text().strip() if marker.exists() else None
            if installed != kernel:
                raise RuntimeError(
                    f"stack '{stack_id}' needs the linux {kernel} guest; "
                    f"{image} has "
                    + (f"linux {installed}" if installed else "no kernel marker")
                    + ". Build it with `oma-arvio accelstore.build-guest --kernel "
                    f"{kernel}`"
                )


def _sweep_capacity_gb(
    stack_ids: list, points: list, override, guest_gb
) -> Optional[int]:
    """The window every RocksDB arm of this sweep is given, or None.

    One figure for the whole sweep, so the arms of a row see the same capacity.
    ``None`` (no stacked arm, nothing asked for) hands over the whole device.
    """
    # Only the stacks with a default window are sized for, ext4 takes the device.
    stacked = [s for s in stack_ids if default_capacity_gb(s) is not None]
    needed = ycsb_grid.sweep_capacity_needed_gb(points)
    if override is not None:
        capacity = int(override)
    elif not stacked:
        return None
    else:
        capacity = sweep_window_gb(set(stacked), needed)

    for stack_id in sorted(set(stack_ids)):
        for warning in check_capacity(stack_id, capacity, guest_gb):
            print(f"NOTE: {warning}")
    if needed > capacity:
        offenders = ", ".join(
            f"{p.name} ({ycsb_grid.required_capacity_gb(p)} GB)"
            for p in points
            if ycsb_grid.required_capacity_gb(p) > capacity
        )
        # The large half is 256 GiB of live values per point, the usual overflow.
        large = [p.name for p in points if p.name in ycsb_grid.LARGE_POINTS]
        remedy = (
            "run `--points grid,small` (the 10k-key families, which fit), "
            "raise --stack-capacity-gb if every stack in the sweep allows it"
            if large
            else "raise --stack-capacity-gb if every stack in the sweep allows it"
        )
        raise ValueError(
            f"the sweep's RocksDB arms need up to {needed} GB but the window is "
            f"{capacity} GB: {offenders}. " + remedy + ", drop the point, or "
            "lower --num-keys"
        )
    return capacity


def _check_client_memory(points, arms, threads, hostname, sizes, skips=None) -> None:
    """Refuse a sweep whose client buffers do not fit the guest, before booting.

    ycsb_client mlocks one pattern buffer plus one read buffer per worker, each
    the value size, which binds on the A sweep's large half. ``skips`` is
    :func:`_rocksdb_skips`' verdict, so a pair the sweep will not run cannot
    refuse it.
    """
    from modules.accelstore.tasks.accelstore import _vm_memory_gb

    workers = 1 if threads is None else int(threads)
    skips = skips or {}
    for arm in arms:
        # Only the accelstore arm reserves hugepages.
        reserved = DEFAULT_HUGEPAGES_GB if arm == ACCELSTORE_ARM else 0
        for point in points:
            if arm != ACCELSTORE_ARM and point.name in skips:
                continue
            wanted = ycsb_grid.client_buffer_bytes(point, workers) / GIB
            needed = wanted + reserved + GUEST_MEMORY_SLACK_GIB
            for size in sizes:
                have = _vm_memory_gb(hostname, size)
                if have >= needed:
                    continue
                raise ValueError(
                    f"{point.name} on arm '{arm}' pins {wanted:.0f} GiB of "
                    f"client buffers at {workers} worker(s) "
                    f"({ycsb_grid.format_bytes(point.value_size)} x "
                    f"(threads + 1)), which with "
                    f"{reserved} GiB of hugepages and {GUEST_MEMORY_SLACK_GIB} "
                    f"GiB of slack wants ~{needed:.0f} GiB; size '{size}' has "
                    f"{have:.0f}. Lower --threads, run at a larger --sizes, or "
                    "drop the point"
                )


def _rocksdb_skips(points, arms) -> dict:
    """Point name -> why no RocksDB arm can run it, for the points where one is.

    Skipped per (arm, point) rather than refusing the sweep. No point of the
    table reaches RocksDB's limit, so only a ``--value-size`` override can.
    """
    if not [a for a in arms if a != ACCELSTORE_ARM]:
        return {}
    reasons = {}
    for point in points:
        reason = ycsb_grid.rocksdb_rejects(point)
        if reason:
            reasons[point.name] = reason
    return reasons


def _gib(value):
    return int(value) * (1 << 30) if value else None


@task(name="run-ycsb")
def run_ycsb(
    ctx,
    setups: str = "snp",
    sizes: str = "medium",
    arms: str = None,
    points: str = None,
    crypto: str = "aegis",
    hostname: str = None,
    threads: Optional[int] = None,
    seed: Optional[int] = None,
    uniform: bool = False,
    zipf_theta: Optional[float] = None,
    validate: bool = False,
    per_worker: bool = True,
    value_size: Optional[int] = None,
    num_keys: Optional[int] = None,
    num_ops: Optional[int] = None,
    direction_bytes_gb: Optional[int] = None,
    large_total_gb: Optional[int] = None,
    quick: bool = False,
    stack_capacity_gb: Optional[int] = None,
    ycsb_args: str = None,
    sar_enabled: bool = True,
    sar_interval: int = 1,
    sar_options: str = "-u -r -n DEV",
    continue_on_error: bool = True,
    kvm_perf_enabled: bool = False,
    vfio_trace: bool = False,
    precondition_scheme: str = "null",
    sync_trigger: Optional[str] = None,
    sync_every_n: Optional[int] = None,
    sync_interval_us: Optional[int] = None,
    sync_coupling: Optional[str] = None,
    sync_commit_every: Optional[int] = None,
    sync_blocking: bool = False,
    sync_metadata_only: bool = False,
    tc_backend: Optional[str] = None,
    tc_rtt_us: Optional[int] = None,
    tc_rtt_model: Optional[str] = None,
    tc_rtt_interval: Optional[int] = None,
):
    """Run YCSB points across VM setups and arms.

    One VM boot per (setup, size, arm, point). The accelstore arm runs the KVS
    module in the usual SPDK lifecycle, once per crypto build named by
    ``--crypto``. A stack arm formats the passthrough NVMe for that stack and
    runs ``ycsb_client --backend rocksdb`` against a fresh DB directory on it.
    Both arms take the same ``--sync-*`` group, but the counter-specific
    settings have no RocksDB counterpart and are refused when a stack arm is in
    the sweep, and ``on-barrier`` is refused outright: a YCSB mix has no
    barrier operations.

    Args:
        setups: Comma-separated VM setups (default: snp)
        sizes: Comma-separated VM sizes (default: medium). An accelstore arm
               gives every worker a reactor core of its own, so size the VM for
               ``--threads`` + 1 cores
        arms: Comma-separated arms. Valid: "accelstore" plus every block stack
              (ext4, ext4-aes-xts-plain, rollbaccine); the default is
              accelstore + ext4
        points: Comma-separated point names (A_4k .. A_1g, B_4k .. F_1m),
                bare mix letters (``A`` selects the whole size sweep, both
                halves), a family name -- ``grid`` (B/C/F at 4k..1m), ``small``
                (workload A at 4k..1m), ``large`` (workload A at 64m..1g) --
                or "all" (default: all three, 23 points)
        crypto: Crypto build variants for the accelstore arm (comma-separated
                ids or "all"); a RocksDB arm always runs the default build
        hostname: Host to run benchmarks on (default: current host)
        threads: Workers running the mix concurrently, applied to every arm of
                 the sweep. The accelstore arm takes at most 63
        seed: Stream seed, fixing the whole run's op sequence on both arms
              (default: the client's 42)
        uniform / zipf_theta: Key distribution (default: Zipfian, theta 0.99)
        validate: Walk every read byte against the pattern (default False)
        per_worker: Pass --per-worker on every arm (default True)
        value_size / num_keys / num_ops: Override the grid's own numbers on
                  every point of this sweep. Each override that departs from
                  the point gets a token in the result leaf
        direction_bytes_gb: Run the A sweep's small half at this many GiB per
                run-phase direction instead of the table's 128 GB, by scaling
                num_ops
        large_total_gb: Run the A sweep's large half at this many GiB of load
                phase instead of 256, by scaling the key count
        quick: Mark the rows as `paper.* --quick`'s, in a result leaf of their
                own so they never overwrite a full-fidelity row
        stack_capacity_gb: Window the RocksDB arms' stacks are given, in GB
                  (10^9). Default: the sweep's own demand between the floor and
                  the ceiling its stacks allow; None for a sweep with no
                  stacked arm, i.e. the whole device
        ycsb_args: Extra ycsb_client arguments appended verbatim on every arm
        sar_* / kvm_perf_enabled / vfio_trace / precondition_scheme: as run-bench
        continue_on_error: Continue running points even if one fails
        sync_* / tc_*: One SyncMode (see above)

    Example:
        oma-arvio accelstore.run-ycsb
        oma-arvio accelstore.run-ycsb --points A --arms accelstore,ext4,rollbaccine
        oma-arvio accelstore.run-ycsb --points C_1m --threads 8 --sizes large
        oma-arvio accelstore.run-ycsb --points grid      # B/C/F at 4k..1m only
        oma-arvio accelstore.run-ycsb --points small     # workload A at 4k..1m
        oma-arvio accelstore.run-ycsb --points large     # workload A at 64m..1g
    """
    from modules.accelstore.tasks.accelstore import (
        _spdk_bdev_config,
        _validate_threads,
        _vm_memory_gb,
    )

    setup_list = [s.strip() for s in setups.split(",") if s.strip()]
    size_list = [s.strip() for s in sizes.split(",") if s.strip()]
    arm_list = _parse_arms(arms)
    point_list = [ycsb_grid.ycsb_point(n) for n in ycsb_grid.parse_points(points)]
    point_list = [
        ycsb_grid.with_byte_targets(
            p,
            direction_bytes=_gib(direction_bytes_gb),
            large_total_bytes=_gib(large_total_gb),
        ).with_overrides(value_size=value_size, num_keys=num_keys, num_ops=num_ops)
        for p in point_list
    ]
    crypto_backends = CryptoBackend.parse_list(crypto)
    threads = _validate_threads(threads, arm_list)

    sync = SyncMode.from_task_kwargs(
        sync_trigger=sync_trigger,
        sync_every_n=sync_every_n,
        sync_interval_us=sync_interval_us,
        sync_coupling=sync_coupling,
        sync_commit_every=sync_commit_every,
        sync_blocking=sync_blocking,
        sync_metadata_only=sync_metadata_only,
        tc_backend=tc_backend,
        tc_rtt_us=tc_rtt_us,
        tc_rtt_model=tc_rtt_model,
        tc_rtt_interval=tc_rtt_interval,
    )
    sync.validate()
    # Checked once per arm kind so a refusal arrives before the first boot.
    for arm in arm_list:
        check_sync_for_arm(sync, arm == ACCELSTORE_ARM)
    for warning in sync.warnings():
        print(f"WARNING: {warning}")

    stack_ids = [a for a in arm_list if a != ACCELSTORE_ARM]
    if stack_ids:
        _require_stack_guest(setup_list, stack_ids)
    # A rollbaccine window's checksum tree comes out of the guest's own RAM.
    guest_gb = min((_vm_memory_gb(hostname, size) for size in size_list), default=None)
    capacity_gb = _sweep_capacity_gb(stack_ids, point_list, stack_capacity_gb, guest_gb)
    rocksdb_skips = _rocksdb_skips(point_list, arm_list)
    for reason in rocksdb_skips.values():
        print(f"NOTE: {reason}")
    _check_client_memory(
        point_list, arm_list, threads, hostname, size_list, rocksdb_skips
    )

    # The accelstore arm fans out over the crypto builds, a stack arm is one run.
    runs = []
    for arm in arm_list:
        if arm == ACCELSTORE_ARM:
            runs.extend((arm, cb) for cb in crypto_backends)
        else:
            runs.append((arm, None))

    devices = Devices(hostname)
    combinations = list(itertools.product(setup_list, size_list, runs))

    print(
        f"Starting YCSB suite: {len(combinations)} configurations x "
        f"{len(point_list)} points"
    )

    all_results = []
    overall_start = time.time()

    for setup, size, (arm, crypto_backend) in combinations:
        arm_label = (
            crypto_backend.client_binary("ycsb_client")
            if arm == ACCELSTORE_ARM
            else arm
        )
        target = devices.storage_target(setup)
        vfio_pcie_arg = [target.vfio_device] if target.vfio_device else None

        action_config_base = {
            "arm": arm,
            "dev_path": target.dev_path,
            "threads": threads,
            "seed": seed,
            "uniform": uniform,
            "zipf_theta": zipf_theta,
            "validate": validate,
            "per_worker": per_worker,
            "skip_precondition": False,
            "precondition_scheme": precondition_scheme,
            "ycsb_args": ycsb_args,
            **sync.as_config(),
        }
        if arm == ACCELSTORE_ARM:
            action_config_base.update(
                {
                    "crypto": crypto_backend.value,
                    "config": _spdk_bdev_config(target.pci_dev),
                    "device": "Nvme1n1",
                    "pci_device": target.pci_dev,
                }
            )
        else:
            action_config_base.update(
                {
                    "capacity_gb": capacity_gb,
                    "guest_memory_gb": _vm_memory_gb(hostname, size),
                }
            )

        for point in point_list:
            if arm != ACCELSTORE_ARM and point.name in rocksdb_skips:
                all_results.append(
                    {
                        "config": f"{setup}-{size}-{arm_label}-{point.name}",
                        "status": "SKIPPED",
                        "duration": 0.0,
                        "error": rocksdb_skips[point.name],
                    }
                )
                continue

            print(f"Running: {setup}-{size} arm={arm_label} point={point.name}")

            start_time = time.time()

            try:
                with check_speed(
                    target.host_pci_dev,
                    verbose=True,
                    valid_speeds=target.valid_pcie_speeds,
                ):
                    vm_tasks.start(
                        ctx,
                        type=setup,
                        size=size,
                        hostname=hostname,
                        direct=False,
                        action="ycsb",
                        sar_enabled=sar_enabled,
                        sar_interval=sar_interval,
                        sar_options=sar_options,
                        vfio_pcie=vfio_pcie_arg,
                        kvm_perf_enabled=kvm_perf_enabled,
                        vfio_trace=vfio_trace,
                        action_config={
                            **action_config_base,
                            "point": point.name,
                            # The resolved numbers, not the table's own.
                            "value_size": point.value_size,
                            "num_keys": point.num_keys,
                            "num_ops": point.num_ops,
                            "quick": quick,
                        },
                    )

                all_results.append(
                    {
                        "config": f"{setup}-{size}-{arm_label}-{point.name}",
                        "status": "SUCCESS",
                        "duration": time.time() - start_time,
                    }
                )

            except Exception as e:
                all_results.append(
                    {
                        "config": f"{setup}-{size}-{arm_label}-{point.name}",
                        "status": "FAILED",
                        "duration": time.time() - start_time,
                        "error": str(e),
                    }
                )
                print(f"ERROR: {e}")

                if not continue_on_error:
                    raise

    total_duration = time.time() - overall_start
    success_count = sum(1 for r in all_results if r["status"] == "SUCCESS")
    failed_count = sum(1 for r in all_results if r["status"] == "FAILED")
    skipped_count = sum(1 for r in all_results if r["status"] == "SKIPPED")

    print(f"\nCompleted in {total_duration/60:.1f} minutes")
    print(
        f"Results: {success_count} success, {failed_count} failed, "
        f"{skipped_count} skipped"
    )

    return all_results


@task(name="parse-ycsb")
def parse_ycsb(
    _ctx,
    ycsb_dir: str = "bench-result/ycsb",
    csv_file: str = "bench-result/parsed/ycsb-summary.csv",
    kinds_csv: str = "bench-result/parsed/ycsb-kinds.csv",
    table: bool = True,
):
    """Parse run-ycsb results into CSVs comparable across arms.

    Reads ycsb_client's own metrics -- the ``--json`` sidecar next to a log
    where there is one, the metrics printer's lines otherwise -- for the latest
    run of every (config, arm, point, variant) point under ``ycsb_dir``, and
    merges the action's ``*_flags.json``.

    Summary CSV, one row per run, the load and the run phase side by side:
        config, arm, backend, point, variant, status, mix, value_size,
        num_keys, num_ops, threads, seed, dist, theta, sync_config, stack,
        crypto, crypto_chunk, cluster_size, md_pages, capacity_gb, ack,
        load_ops, load_wall_ms, load_ops_per_s, load_write_mib,
        load_write_mibs, load_write_per_request_mibs, load_write_in_flight,
        load_write_service_ms, load_effective_concurrency, load_disk_bytes,
        run_ops, run_wall_ms, run_ops_per_s,
        run_read_mib, run_read_mibs, run_read_per_request_mibs,
        run_read_in_flight, run_read_service_ms,
        run_write_mib, run_write_mibs, run_write_per_request_mibs,
        run_write_in_flight, run_write_service_ms,
        run_effective_concurrency, run_disk_bytes,
        read_p50_ms, read_p99_ms, update_p50_ms, update_p99_ms,
        rmw_p50_ms, rmw_p99_ms, insert_p50_ms, insert_p99_ms,
        failed_ops, mismatches, exit_code, log

    Each direction of each phase carries three rates: ``*_per_request_mibs``
    (what one request achieved, so it does not move with the thread count),
    ``*_in_flight`` (how many were outstanding on average) and the aggregate
    ``*_mibs`` the two multiply back to.

    Per-kind CSV, one row per (phase, op kind):
        config, arm, point, variant, phase, kind, count, failed, avg_ms,
        min_ms, max_ms, p50_ms, p99_ms

    Args:
        ycsb_dir: Result tree run-ycsb wrote (default bench-result/ycsb)
        csv_file: Per-run summary CSV
        kinds_csv: Per-kind CSV, empty string to skip it
        table: Also print the per-run comparison table (default True)

    Example:
        oma-arvio accelstore.parse-ycsb
    """
    from modules.accelstore.tasks.results import ycsb_results

    root = Path(ycsb_dir)
    if not root.exists():
        print(f"Error: Benchmark directory not found: {ycsb_dir}")
        return

    runs = ycsb_results.export_points_csv(
        root,
        Path(csv_file),
        Path(kinds_csv) if kinds_csv else None,
    )
    if not runs:
        print(f"No ycsb_client results found under {ycsb_dir}")
        return
    if table:
        ycsb_results.print_ycsb_table(runs)
