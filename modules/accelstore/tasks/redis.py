#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The Redis application benchmark: the points, the core layout, and the sweep.

``oma-arvio accelstore.run-redis`` runs Redis with the AccelStore persistence
backend against the same binary on a block stack, over five points, each load
point once with the automatic AOF rewrite off and once with it on.
"""

import itertools
import time
from dataclasses import dataclass, field, replace
from typing import Optional

from invoke import task

ACCELSTORE_ARM = "accelstore"

REDIS_PORT = 6379

VALUE_SIZE_SMALL = 64
VALUE_SIZE_THROUGHPUT = 512
VALUE_SIZE_FILL = 1024

DATASET_FILL_REQUESTS = 1_500_000  # x 1 KiB values, ~1.5 GB live

# Slack over a point's request count, for snapshot segments and churn.
MD_PAGES_SLACK = 131_072

# Where a blob's data region starts, so a cluster tiles a small object exactly.
CLUSTER_SIZE = 16384

DEFAULT_WORKERS = 3

ACCELSTORE_CORES = 1  # the store's reactors never yield, so they get whole cores

REDIS_CORES = 2  # the event loop, plus the bio threads that do the everysec fsync

REWRITE_CORES = 1

# Taken only if at least MIN_CLIENT_CORES remain for redis-benchmark.
HOUSEKEEPING_CORES = 2
MIN_CLIENT_CORES = 2

MIN_GUEST_CORES = REDIS_CORES + ACCELSTORE_CORES + REWRITE_CORES + MIN_CLIENT_CORES


@dataclass(frozen=True)
class RedisPoint:
    """One point of the sweep: a durability policy and a load shape.

    ``requests`` is fixed per point and identical on every arm.
    ``--requests-scale`` divides every point and ``--requests`` overrides one.
    """

    name: str
    appendfsync: str
    clients: int
    pipeline: int
    value_size: int
    requests: int
    fill_requests: int = 0  # untimed ops before the timed section
    persistence_ops: bool = False  # BGSAVE / BGREWRITEAOF / restart-load walls
    summary: str = ""
    benchmark_tests: tuple = field(default=("set",))

    def __post_init__(self) -> None:
        if self.appendfsync not in APPENDFSYNC_POLICIES:
            raise ValueError(
                f"{self.name}: unknown appendfsync '{self.appendfsync}'. "
                f"Known: {', '.join(APPENDFSYNC_POLICIES)}"
            )
        for field_name in ("clients", "pipeline", "value_size", "requests"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{self.name}: {field_name} must be a positive int")
        if self.fill_requests < 0:
            raise ValueError(f"{self.name}: fill_requests must not be negative")

    def with_overrides(self, requests=None, fill_requests=None, scale=None):
        """This point with whichever bound a run replaced.

        The name is kept. ``scale`` divides both bounds (floored at 1) and is
        applied after an explicit override.
        """
        changes = {}
        if requests is not None:
            changes["requests"] = int(requests)
        if fill_requests is not None and self.fill_requests:
            changes["fill_requests"] = int(fill_requests)
        point = replace(self, **changes) if changes else self
        if scale is not None and float(scale) != 1.0:
            scale = float(scale)
            if scale <= 0:
                raise ValueError(f"requests_scale {scale} must be positive")
            point = replace(
                point,
                requests=max(1, int(point.requests / scale)),
                fill_requests=(
                    max(1, int(point.fill_requests / scale))
                    if point.fill_requests
                    else 0
                ),
            )
        return point

    @property
    def keyspace(self) -> int:
        """``redis-benchmark -r``: one distinct key per request. Without it
        every SET overwrites the same key and the dataset stays at one entry."""
        return max(self.fill_requests, self.requests)

    def benchmark_args(self, requests=None) -> list:
        """The redis-benchmark half of the command line, identical per arm."""
        return [
            "-p",
            str(REDIS_PORT),
            "-t",
            ",".join(self.benchmark_tests),
            "-n",
            str(int(requests if requests is not None else self.requests)),
            "-c",
            str(self.clients),
            "-P",
            str(self.pipeline),
            "-d",
            str(self.value_size),
            "-r",
            str(self.keyspace),
            "--csv",
        ]

    def fill_args(self) -> list:
        """The untimed dataset fill, for the point that has one."""
        if not self.fill_requests:
            return []
        return [
            "-p",
            str(REDIS_PORT),
            "-t",
            "set",
            "-n",
            str(self.fill_requests),
            "-c",
            "50",
            "-P",
            "16",
            "-d",
            str(VALUE_SIZE_FILL),
            "-r",
            str(self.fill_requests),
            "--csv",
        ]

    @property
    def md_pages(self) -> int:
        """``--accelstore-md-pages``: the blob ceiling this point needs.

        It has to be passed: SPDK's default reservation is one page per
        cluster, and the store would fail to initialize.
        """
        return int(self.fill_requests + self.requests + MD_PAGES_SLACK)

    @property
    def cluster_size(self) -> int:
        return CLUSTER_SIZE

    def accelstore_args(self, config_file, device, core_mask, workers, tc_rtt_us):
        """The ``--accelstore-*`` half of redis-server's command line."""
        args = [
            "--accelstore-config",
            str(config_file),
            "--accelstore-device",
            str(device),
            "--accelstore-core-mask",
            str(core_mask),
            "--accelstore-workers",
            str(int(workers)),
            "--accelstore-cluster-size",
            str(self.cluster_size),
            "--accelstore-md-pages",
            str(self.md_pages),
        ]
        if tc_rtt_us:
            args += ["--accelstore-tc-rtt-us", str(int(tc_rtt_us))]
        return args

    def server_args(self, data_dir, rewrite_cpulist) -> list:
        """The arm-independent half of redis-server's command line.

        ``--save ''`` leaves the ``dataset`` point's BGSAVE as the only
        snapshot. ``--daemonize no`` is required: the store does not survive
        the daemonize fork. ``rewrite_cpulist`` is
        :attr:`CoreLayout.rewrite_cpu_list`, passed on every arm.
        """
        return [
            "--port",
            str(REDIS_PORT),
            "--dir",
            str(data_dir),
            "--appendonly",
            "yes",
            "--appendfsync",
            self.appendfsync,
            "--save",
            "",
            "--daemonize",
            "no",
            "--aof-rewrite-cpulist",
            str(rewrite_cpulist),
            "--bgsave-cpulist",
            str(rewrite_cpulist),
        ]

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "appendfsync": self.appendfsync,
            "clients": self.clients,
            "pipeline": self.pipeline,
            "value_size": self.value_size,
            "requests": self.requests,
            "fill_requests": self.fill_requests,
            "keyspace": self.keyspace,
            "persistence_ops": self.persistence_ops,
            "benchmark_tests": list(self.benchmark_tests),
            "md_pages": self.md_pages,
            "cluster_size": self.cluster_size,
            "summary": self.summary,
        }


APPENDFSYNC_POLICIES = ("always", "everysec", "no")


def _build_points() -> dict:
    """The point table."""
    points = (
        RedisPoint(
            name="always_c1",
            appendfsync="always",
            clients=1,
            pipeline=1,
            value_size=VALUE_SIZE_SMALL,
            requests=200_000,
            summary="one fsync per command: barrier latency with no group commit",
        ),
        RedisPoint(
            name="always_c8p4",
            appendfsync="always",
            clients=8,
            pipeline=4,
            value_size=VALUE_SIZE_SMALL,
            # Long enough that an AOF rewrite inside it cannot swing the rate.
            requests=5_000_000,
            summary="group commit: one fsync covers ~32 commands",
        ),
        RedisPoint(
            name="everysec_c8p4",
            appendfsync="everysec",
            clients=8,
            pipeline=4,
            value_size=VALUE_SIZE_SMALL,
            requests=5_000_000,
            summary="the same load with the fsync off the command path",
        ),
        RedisPoint(
            name="everysec_c50p16",
            appendfsync="everysec",
            clients=50,
            pipeline=16,
            value_size=VALUE_SIZE_THROUGHPUT,
            requests=10_000_000,
            summary="append throughput: the per-append floor, not the barrier",
        ),
        RedisPoint(
            name="dataset",
            appendfsync="everysec",
            clients=50,
            pipeline=16,
            value_size=VALUE_SIZE_FILL,
            # Small on purpose: this point measures the three walls, not a rate.
            requests=200_000,
            fill_requests=DATASET_FILL_REQUESTS,
            persistence_ops=True,
            summary="~1.5 GB fill, then timed BGSAVE / BGREWRITEAOF / restart-load",
        ),
    )
    return {point.name: point for point in points}


REDIS_POINTS = _build_points()
REDIS_POINT_NAMES = tuple(REDIS_POINTS)

FAMILY_LOAD = "load"
FAMILY_PERSIST = "persist"
FAMILIES = {
    FAMILY_LOAD: tuple(n for n, p in REDIS_POINTS.items() if not p.persistence_ops),
    FAMILY_PERSIST: tuple(n for n, p in REDIS_POINTS.items() if p.persistence_ops),
}


def redis_point(name) -> RedisPoint:
    """Resolve a point name (or point) against the table."""
    if isinstance(name, RedisPoint):
        return name
    if name not in REDIS_POINTS:
        raise ValueError(
            f"Unknown redis point '{name}'. Known: {', '.join(REDIS_POINT_NAMES)}, "
            f"families: {', '.join(FAMILIES)}, or 'all'"
        )
    return REDIS_POINTS[name]


def parse_points(spec) -> list:
    """``--points``: names, family names, or "all" (the default). Order
    follows the table rather than the argument."""
    if not spec or spec.strip() in ("all", ""):
        return list(REDIS_POINT_NAMES)
    selected = []
    for token in (t.strip() for t in spec.split(",")):
        if not token:
            continue
        if token == "all":
            selected.extend(REDIS_POINT_NAMES)
        elif token in FAMILIES:
            selected.extend(FAMILIES[token])
        else:
            selected.append(redis_point(token).name)
    ordered = [n for n in REDIS_POINT_NAMES if n in set(selected)]
    if not ordered:
        raise ValueError(f"--points '{spec}' selected nothing")
    return ordered


def redis_arms(block_stacks) -> tuple:
    """The arms a redis sweep can run, from the block-stack table."""
    return (ACCELSTORE_ARM, *block_stacks)


@dataclass(frozen=True)
class CoreLayout:
    """Which guest cores each of the four roles gets.

    ``server`` is what ``taskset -c`` gets. It includes the reactor cores on
    the accelstore arm, because DPDK refuses a core mask outside the process
    affinity mask, and the rewrite core on both arms, because the child's
    ``sched_setaffinity`` cannot leave the parent's mask.
    """

    total: int
    housekeeping: tuple
    redis: tuple
    accelstore: tuple
    rewrite: tuple
    client: tuple
    arm_is_accelstore: bool
    source: str = ""  # which config.toml table ``total`` came from

    @property
    def server(self) -> tuple:
        cores = self.redis + self.accelstore if self.arm_is_accelstore else self.redis
        return tuple(sorted(cores + self.rewrite))

    @property
    def core_mask(self) -> str:
        """``--accelstore-core-mask``, as the hex DPDK wants."""
        mask = 0
        for cpu in self.accelstore:
            mask |= 1 << cpu
        return f"0x{mask:x}"

    @property
    def server_cpu_list(self) -> str:
        return _cpu_list(self.server)

    @property
    def client_cpu_list(self) -> str:
        return _cpu_list(self.client)

    @property
    def rewrite_cpu_list(self) -> str:
        """``--aof-rewrite-cpulist`` / ``--bgsave-cpulist``, as Redis wants it."""
        return _cpu_list(self.rewrite)

    def as_dict(self) -> dict:
        return {
            "total_vcpus": self.total,
            "total_vcpus_source": self.source or None,
            "housekeeping": list(self.housekeeping),
            "redis": list(self.redis),
            "accelstore_reactors": list(self.accelstore),
            "rewrite": list(self.rewrite),
            "client": list(self.client),
            "server_taskset": self.server_cpu_list,
            "client_taskset": self.client_cpu_list,
            "rewrite_cpulist": self.rewrite_cpu_list,
            "accelstore_core_mask": self.core_mask if self.arm_is_accelstore else None,
        }


def _cpu_list(cpus) -> str:
    """A sorted cpu tuple as taskset's list syntax, collapsing runs."""
    cpus = sorted(cpus)
    if not cpus:
        raise ValueError("empty cpu set")
    ranges = []
    start = prev = cpus[0]
    for cpu in cpus[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append((start, prev))
        start = prev = cpu
    ranges.append((start, prev))
    return ",".join(str(lo) if lo == hi else f"{lo}-{hi}" for lo, hi in ranges)


def core_layout(
    total_cores: int, arm: str = ACCELSTORE_ARM, source: str = ""
) -> CoreLayout:
    """Split ``total_cores`` guest vCPUs into the four disjoint roles.

    From the bottom up: housekeeping (dropped unless
    :data:`MIN_CLIENT_CORES` still remain), Redis, the store's reactors, the
    rewrite child, and redis-benchmark on everything above. The reactor and
    rewrite cores are carved out on every arm so the client's set is identical
    across a row. At 12 vCPUs: housekeeping ``0,1``, redis ``2,3``, reactors
    ``4,5``, rewrite ``6``, client ``7-11``.

    ``total_cores`` has to be the count the guest actually boots with
    (:func:`_resolve_vm_cpus`).
    """
    total = int(total_cores)
    if total < MIN_GUEST_CORES:
        raise ValueError(
            f"the redis sweep needs at least {MIN_GUEST_CORES} guest vCPUs "
            f"({REDIS_CORES} for redis, {ACCELSTORE_CORES} for the store's "
            f"reactors, {REWRITE_CORES} for the rewrite child, "
            f"{MIN_CLIENT_CORES} for redis-benchmark, all disjoint); "
            f"this VM size has {total}. Run at a larger --sizes"
        )
    fixed = REDIS_CORES + ACCELSTORE_CORES + REWRITE_CORES
    housekeeping = (
        HOUSEKEEPING_CORES
        if total - HOUSEKEEPING_CORES - fixed >= MIN_CLIENT_CORES
        else 0
    )
    cursor = housekeeping
    redis_cores = tuple(range(cursor, cursor + REDIS_CORES))
    cursor += REDIS_CORES
    accel_cores = tuple(range(cursor, cursor + ACCELSTORE_CORES))
    cursor += ACCELSTORE_CORES
    rewrite_cores = tuple(range(cursor, cursor + REWRITE_CORES))
    cursor += REWRITE_CORES
    return CoreLayout(
        total=total,
        housekeeping=tuple(range(housekeeping)),
        redis=redis_cores,
        accelstore=accel_cores,
        rewrite=rewrite_cores,
        client=tuple(range(cursor, total)),
        arm_is_accelstore=arm == ACCELSTORE_ARM,
        source=source,
    )


def _resolve_vm_cpus(hostname, size):
    """The vCPU count a size boots with on this host, and which table said so.

    ``hostname`` must already be resolved: an unresolved ``None`` reads
    ``[defaults.vm_resources]`` rather than the host entry the VM boots from.
    """
    from core.tasks.config import load_config

    from modules.accelstore.tasks.accelstore import _vm_resources

    host = load_config().hosts.get(hostname)
    table = (
        f"hosts.{hostname}.vm_resources"
        if host and host.vm_resources
        else "defaults.vm_resources"
    )
    return int(_vm_resources(hostname, size).cpu), f"config.toml [{table}].{size}"


REDIS_DEFAULT_ARMS = (ACCELSTORE_ARM, "ext4")


def _parse_arms(arms, valid) -> list:
    arm_list = (
        [a.strip() for a in arms.split(",") if a.strip()]
        if arms
        else list(REDIS_DEFAULT_ARMS)
    )
    for arm in arm_list:
        if arm not in valid:
            raise ValueError(f"Unknown arm '{arm}'. Valid: {', '.join(valid)}")
    return arm_list


@task(name="run-redis")
def run_redis(
    ctx,
    setups: str = "snp",
    sizes: str = "medium",
    arms: str = None,
    points: str = None,
    hostname: str = None,
    workers: int = DEFAULT_WORKERS,
    tc_rtt_us: Optional[int] = None,
    requests: Optional[int] = None,
    fill_requests: Optional[int] = None,
    requests_scale: Optional[float] = None,
    stack_capacity_gb: Optional[int] = None,
    sar_enabled: bool = True,
    sar_interval: int = 1,
    sar_options: str = "-u -r -n DEV",
    continue_on_error: bool = True,
    kvm_perf_enabled: bool = False,
    vfio_trace: bool = False,
    precondition_scheme: str = "null",
    skip_precondition: bool = False,
):
    """Run the Redis persistence points across VM setups and arms.

    One VM boot per (setup, size, arm, point). The accelstore arm starts
    ``redis-server`` with the ``--accelstore-*`` flags against a freshly
    destroyed blobstore on the passthrough NVMe. A stack arm formats that NVMe
    for the stack and starts the same binary with those flags omitted. The
    core layout is derived from the guest's vCPU count (:func:`core_layout`).

    Args:
        setups: Comma-separated VM setups (default: snp)
        sizes: Comma-separated VM sizes (default: medium). The layout needs at
               least 7 vCPUs, and what a size name is worth is per host
        arms: Comma-separated arms. Valid: "accelstore" plus every block stack
              (ext4, the LUKS stacks, rollbaccine); the default is
              accelstore + ext4
        points: Comma-separated point names (always_c1, always_c8p4,
                everysec_c8p4, everysec_c50p16, dataset), a family -- `load`
                or `persist` -- or "all" (default)
        hostname: Host to run benchmarks on (default: current host)
        workers: SPDK worker threads the store runs (--accelstore-workers)
        tc_rtt_us: Trusted-counter round trip to model, in microseconds
                 (--accelstore-tc-rtt-us). 0/absent keeps the local counter
        requests / fill_requests: Override a point's own bounds on every point
                 of this sweep. Each override gets a token in the result leaf
        requests_scale: Divide every point's bounds by this factor (a smoke run
                 is `--requests-scale 100`). Applied after --requests
        stack_capacity_gb: Window a POSIX arm's stack is given, in GB (10^9).
                 Default: the whole device. A stacked arm (rollbaccine) needs
                 one -- tasks/block_stacks.py owns the limits
        sar_* / kvm_perf_enabled / vfio_trace / precondition_scheme: as run-bench
        skip_precondition: Skip the secure erase and pre-conditioning pass
        continue_on_error: Continue running points even if one fails

    Example:
        oma-arvio accelstore.run-redis
        oma-arvio accelstore.run-redis --points load --arms accelstore,ext4
        oma-arvio accelstore.run-redis --points dataset --arms accelstore,rollbaccine \\
            --stack-capacity-gb 512
        oma-arvio accelstore.run-redis --points always_c8p4 --tc-rtt-us 2000
        oma-arvio accelstore.run-redis --requests-scale 100   # smoke run
    """
    # Deferred: tasks/accelstore.py imports this module to register the task.
    from core.tasks import vm as vm_tasks
    from core.tasks.utils.device import Devices
    from core.tasks.utils.pci import check_speed

    from modules.accelstore.tasks.accelstore import (
        _mlperf_guest_image,
        _spdk_bdev_config,
    )
    from modules.accelstore.tasks.block_stacks import check_capacity
    from modules.accelstore.tasks.storage_stacks import BLOCK_STACKS

    setup_list = [s.strip() for s in setups.split(",") if s.strip()]
    size_list = [s.strip() for s in sizes.split(",") if s.strip()]
    arm_list = _parse_arms(arms, redis_arms(BLOCK_STACKS))
    point_list = [
        redis_point(n).with_overrides(
            requests=requests, fill_requests=fill_requests, scale=requests_scale
        )
        for n in parse_points(points)
    ]

    stack_ids = [a for a in arm_list if a != ACCELSTORE_ARM]
    if stack_ids:
        _require_stack_guest(setup_list, stack_ids, _mlperf_guest_image, BLOCK_STACKS)
        for stack_id in sorted(set(stack_ids)):
            for warning in check_capacity(stack_id, stack_capacity_gb, None):
                print(f"NOTE: {warning}")

    # Built first for its hostname, which the layout has to be resolved against.
    devices = Devices(hostname)

    # Refused before the first boot, once per size.
    layouts = {}
    for size in size_list:
        cores, cores_source = _resolve_vm_cpus(devices.hostname, size)
        layouts[size] = core_layout(cores, ACCELSTORE_ARM, source=cores_source)
        print(
            f"Core layout at size '{size}' ({cores} vCPUs, {cores_source}): "
            f"redis={_cpu_list(layouts[size].redis)} "
            f"reactors={_cpu_list(layouts[size].accelstore)} "
            f"(mask {layouts[size].core_mask}) "
            f"rewrite={layouts[size].rewrite_cpu_list} "
            f"client={layouts[size].client_cpu_list}"
        )
    combinations = list(itertools.product(setup_list, size_list, arm_list))

    print(
        f"Starting Redis suite: {len(combinations)} configurations x "
        f"{len(point_list)} points"
    )

    all_results = []
    overall_start = time.time()

    for setup, size, arm in combinations:
        target = devices.storage_target(setup)
        vfio_pcie_arg = [target.vfio_device] if target.vfio_device else None

        action_config_base = {
            "arm": arm,
            "dev_path": target.dev_path,
            "workers": workers,
            "tc_rtt_us": tc_rtt_us,
            "guest_cores": layouts[size].total,
            "guest_cores_source": layouts[size].source,
            "skip_precondition": skip_precondition,
            "precondition_scheme": precondition_scheme,
        }
        if arm == ACCELSTORE_ARM:
            action_config_base.update(
                {
                    "config": _spdk_bdev_config(target.pci_dev),
                    "device": "Nvme1n1",
                    "pci_device": target.pci_dev,
                }
            )
        else:
            action_config_base["capacity_gb"] = stack_capacity_gb

        for point in point_list:
            print(f"Running: {setup}-{size} arm={arm} point={point.name}")
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
                        action="redis",
                        sar_enabled=sar_enabled,
                        sar_interval=sar_interval,
                        sar_options=sar_options,
                        vfio_pcie=vfio_pcie_arg,
                        kvm_perf_enabled=kvm_perf_enabled,
                        vfio_trace=vfio_trace,
                        action_config={
                            **action_config_base,
                            "point": point.name,
                            "requests": point.requests,
                            "fill_requests": point.fill_requests or None,
                        },
                    )
                all_results.append(
                    {
                        "config": f"{setup}-{size}-{arm}-{point.name}",
                        "status": "SUCCESS",
                        "duration": time.time() - start_time,
                    }
                )
            except Exception as e:
                all_results.append(
                    {
                        "config": f"{setup}-{size}-{arm}-{point.name}",
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

    print(f"\nCompleted in {total_duration/60:.1f} minutes")
    print(f"Results: {success_count} success, {failed_count} failed")

    return all_results


@task(name="parse-redis")
def parse_redis(
    _ctx,
    redis_dir: str = "bench-result/redis",
    csv_file: str = "bench-result/parsed/redis-summary.csv",
    tests_csv: str = "bench-result/parsed/redis-tests.csv",
    table: bool = True,
):
    """Parse run-redis results into CSVs comparable across arms.

    Reads the ``.json`` metrics sidecar of the latest run of every
    (config, arm, point, variant) leaf under ``redis_dir`` and merges the
    action's ``*_flags.json`` (``results/redis_results.py`` documents every
    column).

    Summary CSV, one row per run:
        config, arm, point, variant, status, backend, appendfsync, clients,
        pipeline, value_size, requests, fill_requests, tc_backend, tc_rtt_us,
        workers, stack, stack_kernel, capacity_gb, cluster_size, md_pages,
        ack, server_start_s, fill_rps, fill_aof_bytes, run_rps, run_avg_ms,
        run_min_ms, run_p50_ms, run_p95_ms, run_p99_ms, run_max_ms,
        run_aof_bytes, run_delayed_fsync, bgsave_wall_s, bgsave_status,
        bgrewriteaof_wall_s, bgrewriteaof_status, restart_load_wall_s,
        restart_dbsize, error, json

    Per-test CSV, one row per (phase, redis-benchmark test):
        config, arm, point, variant, phase, test, rps, avg_ms, min_ms,
        p50_ms, p95_ms, p99_ms, max_ms

    A leaf whose latest run left no sidecar still gets a summary row
    (``status=no-results``).

    Args:
        redis_dir: Result tree run-redis wrote (default bench-result/redis)
        csv_file: Per-run summary CSV
        tests_csv: Per-test CSV; empty string to skip it
        table: Also print the per-run comparison table (default True)

    Example:
        oma-arvio accelstore.parse-redis
    """
    from pathlib import Path

    from modules.accelstore.tasks.results import redis_results

    root = Path(redis_dir)
    if not root.exists():
        print(f"Error: Benchmark directory not found: {redis_dir}")
        return

    runs = redis_results.export_points_csv(
        root,
        Path(csv_file),
        Path(tests_csv) if tests_csv else None,
    )
    if not runs:
        print(f"No run-redis results found under {redis_dir}")
        return
    if table:
        redis_results.print_redis_table(runs)


def _require_stack_guest(setups, stack_ids, guest_image_fn, block_stacks) -> None:
    """Refuse a stack whose module is not in the installed guest image."""
    from pathlib import Path

    from core.tasks.config import BUILD_DIR

    for stack_id in sorted(set(stack_ids)):
        kernel = block_stacks[stack_id].kernel
        if not kernel:
            continue
        for image in sorted({guest_image_fn(s) for s in setups}):
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
