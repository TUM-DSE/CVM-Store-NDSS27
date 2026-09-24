#!/usr/bin/env python3
"""AccelStore benchmark commands."""

import csv
import itertools
import time
from pathlib import Path
from typing import Optional

from invoke import task

from core.tasks import vm as vm_tasks
from core.tasks.config import PROJECT_ROOT
from core.tasks.utils.device import Devices
from core.tasks.utils.pci import check_speed

from modules.accelstore.tasks import sync_modes
from modules.accelstore.tasks.ablation import Ablation
from modules.accelstore.tasks.actions.accelstore import DEFAULT_TRACE_ENTRIES
from modules.accelstore.tasks.actions.bench import (
    BENCH_BACKENDS,
    BENCH_DEFAULT_BACKENDS,
)
from modules.accelstore.tasks.crypto_backend import CryptoBackend
from modules.accelstore.tasks.io_path import IoPath
from modules.accelstore.tasks import mlperf_docker
from modules.accelstore.tasks import store_memory
from modules.accelstore.tasks.block_stacks import (
    check_capacity,
    rollbaccine_tree_gb,
    sweep_window_gb,
)
from modules.accelstore.tasks.mlperf_variants import (
    DEFAULT_ARMS,
    DEFAULT_STACK,
    FAMILIES as MLPERF_FAMILIES,
    MLPERF_ARMS,
    MlperfRunContext,
    OPTIONAL_ARMS,
    RAW_STACK,
    SESSION_WORKERS_DEFAULT,
    stack_of as mlperf_stack,
    stacks_for_arm as mlperf_stacks_for_arm,
    variant as mlperf_variant,
    variants_for as mlperf_variants_for,
)
from modules.accelstore.tasks import pipeline_opts
from modules.accelstore.tasks.pipeline_opts import PipelineOpts

# Imported for task registration. They defer their own imports to avoid a cycle.
from modules.accelstore.tasks.redis import parse_redis, run_redis  # noqa: F401
from modules.accelstore.tasks.ycsb_tasks import parse_ycsb, run_ycsb  # noqa: F401
from modules.accelstore.tasks.workloads import (  # noqa: F401
    generate_log_workloads,
    generate_quick_jobs,
)
from modules.accelstore.tasks.sync_modes import SyncMode
from modules.accelstore.tasks.results.seal_unseal_analyzer import (
    analyze_seal_unseal_benchmarks,
    export_to_csv,
)
from modules.fio.tasks import fio as fio_tasks

ACCELSTORE_CLIENTS = [
    "kvs_client",
    "log_client",
    "fs_client",
]

ACCELSTORE_SEAL_UNSEAL_JOBS = {
    "kvs_client": [
        ["kvs_workload_100mb_seal.txt", "kvs_workload_100mb_unseal.txt"],
        ["kvs_workload_1gb_seal.txt", "kvs_workload_1gb_unseal.txt"],
        ["kvs_workload_10gb_seal.txt", "kvs_workload_10gb_unseal.txt"],
        ["kvs_workload_100gb_seal.txt", "kvs_workload_100gb_unseal.txt"],
        ["kvs_workload_1000gb_seal.txt", "kvs_workload_1000gb_unseal.txt"],
    ],
    "log_client": [
        ["log_workload_100mb_seal.txt", "log_workload_100mb_unseal.txt"],
        ["log_workload_1gb_seal.txt", "log_workload_1gb_unseal.txt"],
        ["log_workload_10gb_seal.txt", "log_workload_10gb_unseal.txt"],
        ["log_workload_100gb_seal.txt", "log_workload_100gb_unseal.txt"],
        ["log_workload_1000gb_seal.txt", "log_workload_1000gb_unseal.txt"],
    ],
    "fs_client": [
        ["fs_workload_100mb_seal.txt", "fs_workload_100mb_unseal.txt"],
        ["fs_workload_1gb_seal.txt", "fs_workload_1gb_unseal.txt"],
        ["fs_workload_10gb_seal.txt", "fs_workload_10gb_unseal.txt"],
        ["fs_workload_100gb_seal.txt", "fs_workload_100gb_unseal.txt"],
        ["fs_workload_1000gb_seal.txt", "fs_workload_1000gb_unseal.txt"],
    ],
}

ACCELSTORE_REGULAR_JOBS = {
    "kvs_client": [
        "kvs_workload_A_64m.txt",
        "kvs_workload_A_256m.txt",
        "kvs_workload_A_1g.txt",
        "kvs_workload_A_4g.txt",
    ],
    "log_client": [
        "log_workload_64MB_write_read_seq.txt",
        "log_workload_256MB_write_read_seq.txt",
        "log_workload_1GB_write_read_seq.txt",
        "log_workload_4GB_write_read_seq.txt",
    ],
    "fs_client": [
        "fs_workload_write_read_64m.txt",
        "fs_workload_write_read_256m.txt",
        "fs_workload_write_read_1g.txt",
        "fs_workload_write_read_4g.txt",
    ],
}

ACCELSTORE_EPOCH_JOBS = {
    "kvs_client": [
        "kvs_workload_A_256m.txt",
    ],
    "log_client": [
        "log_workload_256MB_write_read_seq.txt",
    ],
    "fs_client": [
        "fs_workload_write_read_256m.txt",
    ],
}

BENCH_RANDOM_READ_PERSONALITIES = [
    "fs_random_read_4k",
    "fs_random_read_16k",
    "fs_random_read_64k",
    "fs_random_read_256k",
    "fs_random_read_1m",
]

BENCH_LOG_SMALL_PERSONALITIES = [
    "log_small_16k",
    "log_small_64k",
    "log_small_256k",
    "log_small_1m",
]

BENCH_LOG_LARGE_PERSONALITIES = [
    "log_write_read_64mb",
    "log_write_read_256mb",
    "log_write_read_1gb",
    "log_write_read_4gb",
]

BENCH_FS_LARGE_PERSONALITIES = [
    "fs_write_read_64m",
    "fs_write_read_256m",
    "fs_write_read_1g",
    "fs_write_read_4g",
]

BENCH_APPS_PERSONALITIES = [
    "varmail_log",
    "videoaudit_log_64m_fsync",
    "trainload_fs_paced",
    "webserver_fs",
]

BENCH_SUITES = {
    "random-read": BENCH_RANDOM_READ_PERSONALITIES,
    "log-small": BENCH_LOG_SMALL_PERSONALITIES,
    "log-large": BENCH_LOG_LARGE_PERSONALITIES,
    "fs-large": BENCH_FS_LARGE_PERSONALITIES,
    "apps": BENCH_APPS_PERSONALITIES,
}

BENCH_SUITES_NEEDING_GEOMETRY = {"log-small"}

# Iteration bounds here are run totals, not per-worker counts: need --split-loops.
BENCH_SPLIT_LOOPS_PERSONALITIES = frozenset(
    BENCH_LOG_SMALL_PERSONALITIES
    + BENCH_LOG_LARGE_PERSONALITIES
    + BENCH_RANDOM_READ_PERSONALITIES
)

# Above eight the split leaves workers with no iterations at all.
BENCH_SPLIT_LOOPS_MAX_THREADS = 8


def filter_jobs_for_client(job_list: list, client: str) -> list:
    """Filter jobs to only include those matching the client prefix."""
    prefix_map = {
        "kvs_client": "kvs_",
        "log_client": "log_",
        "fs_client": "fs_",
    }

    prefix = prefix_map.get(client)
    if not prefix:
        return job_list

    return [job.strip() for job in job_list if job.strip().startswith(prefix)]


def _spdk_bdev_config(pci_device: str) -> str:
    """SPDK bdev configuration attaching the NVMe controller as Nvme1."""
    return IoPath().bdev_config(pci_device)


@task(name="run-all")
def run_all(
    ctx,
    setups: str = "amd,snp,host",
    sizes: str = "large",
    clients: str = None,
    hostname: str = None,
    sar_enabled: bool = True,
    sar_interval: int = 1,
    sar_options: str = "-u -r -n DEV",
    jobs: str = None,
    job_sequences: str = None,
    crypto: str = "aegis",
    tracing: Optional[str] = None,
    trace_entries: Optional[int] = DEFAULT_TRACE_ENTRIES,
    continue_on_error: bool = True,
    kvm_perf_enabled: bool = False,
    vfio_trace: bool = False,
    precondition_scheme: str = None,
    ablation: Optional[str] = None,
    io_path: Optional[str] = None,
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
    chunk_size: Optional[int] = None,
    pipeline_bytes: Optional[int] = None,
    io_depth: Optional[int] = None,
    reorder_window: Optional[int] = None,
    burst_bytes: Optional[str] = None,
    burst_adaptive: bool = False,
    io_span_bytes: Optional[int] = None,
    cluster_size: Optional[int] = None,
    md_pages: Optional[int] = None,
):
    """Run AccelStore benchmarks across VM setups, clients, jobs and crypto builds.

    Args:
        setups: Comma-separated VM setups — amd, snp, host
        sizes: Comma-separated VM sizes — small, medium, large
        clients: Comma-separated clients, unprefixed (default: all of
                 ACCELSTORE_CLIENTS). Each runs in every build --crypto names
        hostname: Host to run benchmarks on (default: current host)
        sar_enabled: Enable SAR system activity recording
        sar_interval: SAR sampling interval in seconds
        sar_options: SAR command options
        jobs: Override the per-client regular jobs (comma-separated)
        job_sequences: Semicolon-separated sequences of comma-separated jobs,
                       each sharing a persisted trusted counter (recovery /
                       seal-unseal scenarios). Not combinable with --jobs
        crypto: Comma-separated crypto backend ids (aegis, aesgcm, none, or
                "all"), or a list of CryptoBackend members
        tracing: SPDK trace recording ("tracing", "small-trace", or None)
        trace_entries: Trace ring entries per lcore (power of two, 32768 to
                       67108864). None falls back to the client's own default
        continue_on_error: Continue running jobs even if one fails
        kvm_perf_enabled: Enable KVM performance event collection
        vfio_trace: Enable VFIO region read/write tracing
        precondition_scheme: Preconditioning scheme for storage
        ablation: Comma-separated subset of
                  no-overlap|bounce-copy|no-encrypt|no-verify|zero-copy, or
                  "none" for the unablated arm of a sweep (no flag, own result
                  directory). Validated against every build --crypto names
        io_path: NVMe driver the bdev sits on — "spdk" (bdev_nvme on vfio-pci)
                 or "kernel" (bdev_aio on the guest's namespace node, skipping
                 the vfio bind). Omit it outside a bypass sweep
        sync_trigger: What starts a metadata epoch — on-shutdown (the library
                      default), on-barrier, every-n, every-t, per-op
        sync_every_n: Mutating operations between epochs (operand of every-n)
        sync_interval_us: Durability window in µs (operand of every-t)
        sync_coupling: ordered (default) or parallel — how the epoch's counter
                       commit is scheduled against its metadata blob write
        sync_commit_every: Counter commit every K epochs (K > 1 is refused)
        sync_blocking: Hold mutators until the epoch that covers them is
                       durable (every-n/every-t only)
        sync_metadata_only: Persist metadata without committing a counter value
        tc_backend: local (default) or mock-remote counter service
        tc_rtt_us: Round trip a paying commit waits out under the mock
        tc_rtt_model: every (default) | periodic | elapsed
        tc_rtt_interval: Operand of the periodic/elapsed RTT models
        chunk_size: --crypto-chunk-size in bytes (None keeps the library's
                    1 MiB constant), checked against the job's own reads
        pipeline_bytes: --pipeline-bytes, the in-flight crypto byte budget the
                    command depth is derived from (0 = the 64 MiB default)
        io_depth: --io-depth, chunk I/Os in flight per operation (1..1024)
        reorder_window: --reorder-window, chunks in progress on the in-order
                    read paths (KVS/LOG; 1..1024)
        burst_bytes: --burst-bytes, crypto per submit/drain pass, or "max"
        burst_adaptive: --burst-adaptive, let the two-watermark controller
                    resize the read drain's burst at run time
        io_span_bytes: --io-span-bytes, the device command span the engine
                    coalesces chunks into (clipped at the cluster)
        cluster_size: --cluster-size, blobstore cluster of a freshly created
                    store (multiple of 4096, at or above the chunk size)
        md_pages: --md-pages, blobstore metadata pages reserved at init

        The geometry arguments form one PipelineOpts and the sync/tc ones one
        SyncMode, each also a result-path component.

    Example:
        oma-arvio accelstore.run-all --setups snp --sizes medium --clients kvs_client
        oma-arvio accelstore.run-all --job-sequences "kvs_a_seal.txt,kvs_a_unseal.txt"
    """
    setup_list = [s.strip() for s in setups.split(",")]
    size_list = [s.strip() for s in sizes.split(",")]

    backends = CryptoBackend.parse_list(crypto)

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

    # Per backend: the same token list can be legal for aegis and rejected for none.
    abl = Ablation.parse(ablation)
    for backend in backends:
        abl.validate(backend)

    io = IoPath.parse(io_path)
    pipe = PipelineOpts.from_task_kwargs(
        chunk_size=chunk_size,
        pipeline_bytes=pipeline_bytes,
        io_depth=io_depth,
        reorder_window=reorder_window,
        burst_bytes=burst_bytes,
        burst_adaptive=burst_adaptive,
        io_span_bytes=io_span_bytes,
        cluster_size=cluster_size,
        md_pages=md_pages,
    )
    pipe.validate()
    for warning in pipe.warnings():
        print(f"WARNING: {warning}")

    client_list = (
        [c.strip() for c in clients.split(",")] if clients else ACCELSTORE_CLIENTS
    )

    if jobs and job_sequences:
        raise ValueError("Cannot specify both --jobs and --job-sequences")

    if job_sequences:
        job_list_raw = [
            [j.strip() for j in seq_str.split(",")]
            for seq_str in job_sequences.split(";")
        ]
    elif jobs:
        job_list_raw = [[j.strip()] for j in jobs.split(",")]
    else:
        job_list_raw = None

    devices = Devices(hostname)
    combinations = list(itertools.product(setup_list, size_list, client_list, backends))

    print(
        f"Starting AccelStore benchmark suite: {len(combinations)} client configurations"
    )

    all_results = []
    overall_start = time.time()

    for setup, size, client, backend in combinations:
        client_binary = backend.client_binary(client)

        if job_list_raw:
            filtered_sequences = []
            for job_or_seq in job_list_raw:
                filtered = filter_jobs_for_client(job_or_seq, client)
                if filtered:
                    filtered_sequences.append(filtered)

            if not filtered_sequences:
                print(f"Skipping {setup}-{size}-{client_binary}: No matching jobs")
                continue

            client_jobs = filtered_sequences
        else:
            discovered = ACCELSTORE_REGULAR_JOBS.get(client, [])
            client_jobs = [[job] for job in discovered]

        target = devices.storage_target(setup)
        device = io.device
        config = io.bdev_config(target.pci_dev, target.dev_path)

        vfio_pcie_arg = [target.vfio_device] if target.vfio_device else None

        for job_or_seq in client_jobs:
            is_sequence = len(job_or_seq) > 1
            job_display = f"[{', '.join(job_or_seq)}]" if is_sequence else job_or_seq[0]

            print(
                f"Running: {setup}-{size} client={client} "
                f"crypto={backend.label} "
                f"sync={sync.label()} "
                f"ablation={abl.label()} "
                f"io={io.label()} "
                f"pipeline={pipe.label()} "
                f"job={job_display} device={device}"
            )

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
                        action="run-accelstore",
                        sar_enabled=sar_enabled,
                        sar_interval=sar_interval,
                        sar_options=sar_options,
                        vfio_pcie=vfio_pcie_arg,
                        kvm_perf_enabled=kvm_perf_enabled,
                        vfio_trace=vfio_trace,
                        action_config={
                            "client": client,
                            "crypto": backend.value,
                            "job": job_or_seq,  # list (single or sequence)
                            "config": config,
                            "device": device,
                            "pci_device": target.pci_dev,
                            "dev_path": target.dev_path,
                            "tracing": tracing,
                            "trace_entries": trace_entries,
                            "skip_precondition": False,
                            "precondition_scheme": precondition_scheme,
                            **sync.as_config(),
                            **abl.as_config(),
                            **io.as_config(),
                            **pipe.as_config(),
                        },
                    )

                all_results.append(
                    {
                        "config": f"{setup}-{size}-{client_binary}-{job_display}",
                        "crypto": backend.value,
                        "status": "SUCCESS",
                        "duration": time.time() - start_time,
                    }
                )

            except Exception as e:
                all_results.append(
                    {
                        "config": f"{setup}-{size}-{client_binary}-{job_display}",
                        "crypto": backend.value,
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


@task(name="run-bench")
def run_bench(
    ctx,
    setups: str = "snp",
    sizes: str = "medium",
    backends: str = None,
    suite: str = None,
    personalities: str = None,
    crypto: str = "aegis",
    hostname: str = None,
    sar_enabled: bool = True,
    sar_interval: int = 1,
    sar_options: str = "-u -r -n DEV",
    tracing: Optional[str] = None,
    trace_entries: Optional[int] = DEFAULT_TRACE_ENTRIES,
    continue_on_error: bool = True,
    kvm_perf_enabled: bool = False,
    vfio_trace: bool = False,
    precondition_scheme: str = "null",
    bench_args: str = None,
    threads: Optional[int] = None,
    split_loops: bool = False,
    per_worker: bool = True,
    direct: bool = True,
    fsync_mode: Optional[str] = None,
    barrier_mode: Optional[str] = None,
    log_layout: Optional[str] = None,
    chunk_size: Optional[int] = None,
    derive_chunk_size: bool = True,
    intra_op_depth: int = 1,
    match_io_depth: bool = False,
    derive_store_geometry: Optional[bool] = None,
    pipeline_bytes: Optional[int] = None,
    io_depth: Optional[int] = None,
    reorder_window: Optional[int] = None,
    burst_bytes: Optional[str] = None,
    burst_adaptive: bool = False,
    io_span_bytes: Optional[int] = None,
    cluster_size: Optional[int] = None,
    md_pages: Optional[int] = None,
    sampler: bool = False,
    sampler_interval: float = 1.0,
    sampler_qd_period_us: int = 0,
    quick: bool = False,
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
    """Run bench_client personalities across VM setups and storage backends.

    The bench_client counterpart of run-all: the same .wl personality runs on
    the accelstore backend, once per crypto build named by --crypto, and on
    each POSIX storage backend, where the passthrough NVMe is formatted first
    and the client runs `--backend posix --dir` against the fresh directory.
    One VM boot per (setup, size, backend arm, personality).

    Args:
        setups: Comma-separated VM setups (default: snp)
        sizes: Comma-separated VM sizes (default: medium)
        backends: Comma-separated storage backends — "accelstore" plus the
                  block stacks (ext4, ext4-aes-xts-plain, rollbaccine).
                  Default: every arm that lives in the default guest image
        suite: Named personality set from BENCH_SUITES: random-read |
               log-small | log-large | fs-large | apps. No default — without
               --suite or --personalities the task refuses before anything boots
        personalities: Comma-separated personality names or guest paths,
                       overriding --suite
        crypto: Crypto build variants for the accelstore arm (comma-separated
                ids or "all"); POSIX arms always run the default build's binary
        hostname: Host to run benchmarks on (default: current host)
        sar_enabled: Enable SAR system activity recording
        sar_interval: SAR sampling interval in seconds
        sar_options: SAR command options
        tracing: SPDK trace recording, accelstore arm only ("tracing",
                 "small-trace", or None)
        trace_entries: Trace ring entries per lcore, as in run-all
        continue_on_error: Continue running points even if one fails
        kvm_perf_enabled: Enable KVM performance event collection
        vfio_trace: Enable VFIO region read/write tracing
        precondition_scheme: Preconditioning scheme for storage
        bench_args: Extra bench_client arguments appended verbatim on every arm
        threads: bench_client worker count (--threads) on every arm. None runs
                 each personality's own `threads` declaration; a value — 1
                 included — lands in a `threads-<n>` variant token. The
                 accelstore arm takes at most 63
        split_loops: Pass --split-loops on both arms (default off), making an
                iteration bound the run's total rather than a per-worker count.
                Rows land in their own `split-loops` leaf. Turned on
                automatically, thread count capped at 8, when the selection
                holds a BENCH_SPLIT_LOOPS_PERSONALITIES member
        per_worker: Pass --per-worker on both arms (default True)
        direct: O_DIRECT on the POSIX arms (default True)
        fsync_mode / barrier_mode / log_layout: POSIX-arm durability knobs
                (defaults none / dirty-fsync / files), in the variant token
                when they are not the default
        chunk_size: --crypto-chunk-size for the accelstore arm; None derives it
                from the personality's read geometry
        derive_chunk_size: Derive the chunk size per personality (default True)
        intra_op_depth: Intra-operation parallelism on BOTH arms (default 1 =
                no flag): the derived chunk cut N ways, resolved per
                personality. Refused together with an explicit --chunk-size or
                --derive-chunk-size false. Rows land in an `intra-op-<n>` leaf
        match_io_depth: Give the POSIX arms the accelstore arm's in-flight byte
                budget instead of bench_client's depth-1 default. The segment
                size is not matched: the baseline keeps its own, floored at its
                1 MiB default. Rows land in a `qd-matched` leaf. Refused beside
                --intra-op-depth
        derive_store_geometry: Derive the small-object store geometry per
                personality. Default: on for a suite in
                BENCH_SUITES_NEEDING_GEOMETRY, off otherwise
        pipeline_bytes / io_depth / reorder_window / burst_bytes /
        burst_adaptive / io_span_bytes / cluster_size / md_pages: the rest of
                the chunk-pipeline geometry (accelstore arm), as in run-all.
                None of them is derived, and each gets a result-leaf token
        sampler: Wrap the accelstore arm in bench_rpc_sample.py and write its
                per-interval CSV next to the log
        sampler_interval: Sampler poll interval in seconds (default 1.0)
        sampler_qd_period_us: bdev queue-depth sampling period (default 0 =
                off); non-zero gives the avg_qd / io_time_pct rows
        quick: Mark the rows as `paper.* --quick`'s, in a result leaf of their own
        sync_* / tc_*: One SyncMode for the accelstore arm, as in run-all.
                 A non-default mode is refused when a POSIX backend is in the
                 sweep, since a POSIX arm has no metadata epochs

    Example:
        oma-arvio accelstore.run-bench --suite apps
        oma-arvio accelstore.run-bench --backends ext4 --personalities varmail_log
        oma-arvio accelstore.run-bench --suite log-small --threads 8
        oma-arvio accelstore.run-bench --suite random-read --intra-op-depth 4
        oma-arvio accelstore.run-bench --suite fs-large
        oma-arvio accelstore.run-bench --setups amd,snp,host --sizes large
    """
    if personalities:
        personality_list = [p.strip() for p in personalities.split(",")]
    elif suite is None:
        raise ValueError(
            "run-bench needs --suite or --personalities; there is no default "
            f"sweep. Suites: {', '.join(BENCH_SUITES)}"
        )
    elif suite in BENCH_SUITES:
        personality_list = list(BENCH_SUITES[suite])
    else:
        raise ValueError(f"Unknown suite '{suite}'. Valid: {', '.join(BENCH_SUITES)}")
    backend_list = _parse_backends(backends)

    if derive_store_geometry is None:
        derive_store_geometry = (
            not personalities and suite in BENCH_SUITES_NEEDING_GEOMETRY
        )
    crypto_backends = CryptoBackend.parse_list(crypto)
    threads = _validate_threads(threads, backend_list)
    split_loops = _resolve_split_loops(personality_list, threads, split_loops)

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

    posix_backends = [b for b in backend_list if b != "accelstore"]
    if posix_backends and not sync.is_default():
        raise ValueError(
            f"Sync mode '{sync.label()}' has no meaning on POSIX backends "
            f"({', '.join(posix_backends)}); run them separately"
        )
    if tracing and posix_backends:
        raise ValueError(
            "SPDK tracing only exists on the accelstore backend; "
            "drop --tracing or the POSIX backends"
        )

    pipe = PipelineOpts.from_task_kwargs(
        pipeline_bytes=pipeline_bytes,
        io_depth=io_depth,
        reorder_window=reorder_window,
        burst_bytes=burst_bytes,
        burst_adaptive=burst_adaptive,
        io_span_bytes=io_span_bytes,
        cluster_size=cluster_size,
        md_pages=md_pages,
    )
    # Without the chunk, which the action resolves and validates per personality.
    pipe.validate()
    for warning in pipe.warnings():
        print(f"WARNING: {warning}")

    def arm_config(backend, crypto_backend, target):
        return _bench_arm_config(
            backend=backend,
            crypto_backend=crypto_backend,
            target=target,
            sync=sync,
            threads=threads,
            split_loops=split_loops,
            per_worker=per_worker,
            bench_args=bench_args,
            precondition_scheme=precondition_scheme,
            tracing=tracing,
            trace_entries=trace_entries,
            direct=direct,
            fsync_mode=fsync_mode,
            barrier_mode=barrier_mode,
            log_layout=log_layout,
            chunk_size=chunk_size,
            derive_chunk_size=derive_chunk_size,
            intra_op_depth=intra_op_depth,
            match_io_depth=match_io_depth,
            derive_store_geometry=derive_store_geometry,
            pipe=pipe,
            sampler=sampler,
            sampler_interval=sampler_interval,
            sampler_qd_period_us=sampler_qd_period_us,
            quick=quick,
        )

    return _run_bench_sweep(
        ctx,
        setup_list=[s.strip() for s in setups.split(",")],
        size_list=[s.strip() for s in sizes.split(",")],
        backend_list=backend_list,
        crypto_backends=crypto_backends,
        personality_list=personality_list,
        arm_config=arm_config,
        hostname=hostname,
        sar_enabled=sar_enabled,
        sar_interval=sar_interval,
        sar_options=sar_options,
        kvm_perf_enabled=kvm_perf_enabled,
        vfio_trace=vfio_trace,
        continue_on_error=continue_on_error,
    )


def _parse_backends(backends: Optional[str]) -> list:
    """The --backends argument, defaulting to BENCH_DEFAULT_BACKENDS."""
    backend_list = (
        [b.strip() for b in backends.split(",")]
        if backends
        else list(BENCH_DEFAULT_BACKENDS)
    )
    for b in backend_list:
        if b not in BENCH_BACKENDS:
            raise ValueError(
                f"Unknown backend '{b}'. Valid: {', '.join(BENCH_BACKENDS)}"
            )
    return backend_list


def _validate_threads(threads: Optional[int], backend_list: list) -> Optional[int]:
    """Refuse a thread count no accelstore row could be paired with, up front."""
    if threads is None:
        return None
    threads = int(threads)
    if not 1 <= threads <= 64:
        raise ValueError(f"threads {threads} out of range [1, 64]")
    if threads > 63 and "accelstore" in backend_list:
        raise ValueError(
            f"threads {threads} exceeds the accelstore arm's 63 workers "
            "(the library admits the owner plus 63 attached threads); the "
            "sweep would run its POSIX arms at a concurrency no accelstore "
            "row could be paired with"
        )
    return threads


def _resolve_split_loops(
    personality_list: list, threads: Optional[int], split_loops: bool
) -> bool:
    """The `--split-loops` this selection has to run with."""
    requiring = sorted(
        {
            Path(p).stem
            for p in personality_list
            if Path(p).stem in BENCH_SPLIT_LOOPS_PERSONALITIES
        }
    )
    if not requiring:
        return split_loops

    cap = BENCH_SPLIT_LOOPS_MAX_THREADS
    if threads is not None and int(threads) > cap:
        raise ValueError(
            f"threads {threads} exceeds the {cap}-worker ceiling "
            f"{', '.join(requiring)} set: their smallest iteration-bounded block is a "
            f"total of {cap}, so above it the split leaves the rest of the workers "
            "with no iterations at all and the row does not measure the concurrency "
            f"its label claims. Run them at --threads {cap} or below"
        )

    if not split_loops:
        print(
            f"NOTE: --split-loops implied by {', '.join(requiring)}: their run "
            "blocks are iteration-bounded TOTALS over a fileset whose size is a "
            "property of the file rather than of --threads, so "
            "without it every worker runs each block whole -- N times the bytes "
            "onto the same entries, and on a fileset that is not preallocated a "
            "DSL fileset-capacity error before anything is timed"
        )
    return True


def _bench_arm_config(
    *,
    backend,
    crypto_backend,
    target,
    sync,
    threads,
    split_loops,
    per_worker,
    bench_args,
    precondition_scheme,
    tracing,
    trace_entries,
    direct,
    fsync_mode,
    barrier_mode,
    log_layout,
    chunk_size,
    derive_chunk_size,
    intra_op_depth,
    match_io_depth,
    derive_store_geometry,
    pipe,
    sampler,
    sampler_interval,
    sampler_qd_period_us,
    quick,
) -> dict:
    """The bench action's config for one arm of a sweep."""
    config = {
        "backend": backend,
        "dev_path": target.dev_path,
        "skip_precondition": False,
        "precondition_scheme": precondition_scheme,
        "bench_args": bench_args,
        "threads": threads,
        "split_loops": split_loops,
        "per_worker": per_worker,
        "intra_op_depth": intra_op_depth,
        "quick": quick,
    }
    if backend == "accelstore":
        config.update(
            {
                "crypto": crypto_backend.value,
                "config": _spdk_bdev_config(target.pci_dev),
                "device": "Nvme1n1",
                "pci_device": target.pci_dev,
                "tracing": tracing,
                "trace_entries": trace_entries,
                "chunk_size": chunk_size,
                "derive_chunk_size": derive_chunk_size,
                "derive_store_geometry": derive_store_geometry,
                # The two keys above already carry the chunk.
                **pipe.with_chunk_size(None).as_config(),
                "sampler": sampler,
                "sampler_interval": sampler_interval,
                "sampler_qd_period_us": sampler_qd_period_us,
                **sync.as_config(),
            }
        )
    else:
        config.update(
            {
                "direct": direct,
                "fsync_mode": fsync_mode,
                "barrier_mode": barrier_mode,
                "log_layout": log_layout,
                "match_io_depth": match_io_depth,
            }
        )
        if match_io_depth:
            # The accelstore arm's geometry, to re-derive what it would run at.
            config.update(
                {
                    "chunk_size": chunk_size,
                    "derive_chunk_size": derive_chunk_size,
                    "derive_store_geometry": derive_store_geometry,
                    **pipe.with_chunk_size(None).as_config(),
                }
            )
    return config


def _run_bench_sweep(
    ctx,
    *,
    setup_list,
    size_list,
    backend_list,
    crypto_backends,
    personality_list,
    arm_config,
    hostname,
    sar_enabled,
    sar_interval,
    sar_options,
    kvm_perf_enabled,
    vfio_trace,
    continue_on_error,
):
    """One VM boot per (setup, size, arm, personality), in that order."""
    arms = []
    for b in backend_list:
        if b == "accelstore":
            arms.extend((b, cb) for cb in crypto_backends)
        else:
            arms.append((b, None))

    devices = Devices(hostname)
    combinations = list(itertools.product(setup_list, size_list, arms))

    print(
        f"Starting bench_client suite: {len(combinations)} configurations x "
        f"{len(personality_list)} personalities"
    )

    all_results = []
    overall_start = time.time()

    for setup, size, (backend, crypto_backend) in combinations:
        arm_label = (
            crypto_backend.client_binary("bench_client")
            if backend == "accelstore"
            else backend
        )

        target = devices.storage_target(setup)
        vfio_pcie_arg = [target.vfio_device] if target.vfio_device else None
        action_config_base = arm_config(backend, crypto_backend, target)

        for personality in personality_list:
            print(
                f"Running: {setup}-{size} backend={arm_label} "
                f"personality={personality}"
            )

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
                        action="run-bench",
                        sar_enabled=sar_enabled,
                        sar_interval=sar_interval,
                        sar_options=sar_options,
                        vfio_pcie=vfio_pcie_arg,
                        kvm_perf_enabled=kvm_perf_enabled,
                        vfio_trace=vfio_trace,
                        action_config={
                            **action_config_base,
                            "personality": personality,
                        },
                    )

                all_results.append(
                    {
                        "config": f"{setup}-{size}-{arm_label}-{personality}",
                        "status": "SUCCESS",
                        "duration": time.time() - start_time,
                    }
                )

            except Exception as e:
                all_results.append(
                    {
                        "config": f"{setup}-{size}-{arm_label}-{personality}",
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


def _mlperf_guest_image(setup: str) -> str:
    """Which guest image a setup boots (core.tasks.qemu_builder.get_vm_config)."""
    return "direct-guest-image" if "direct" in setup else "snp-guest-image"


def _require_mlperf_guest(setups: list) -> None:
    """Refuse before the first boot if the installed image has no container."""
    from core.tasks.config import BUILD_DIR

    for image in sorted({_mlperf_guest_image(s) for s in setups}):
        marker = Path(BUILD_DIR) / "image" / f"{image}.mlperf"
        if not marker.exists():
            raise FileNotFoundError(
                f"{image} was not built with the MLPerf container ({marker} is "
                "absent). Build it with `oma-arvio accelstore.build-guest "
                "--mlperf`"
            )


def _mlperf_selection(variants: Optional[str]) -> list:
    """The variants a --variants argument names: "smoke", "all", or a list of
    family names and variant ids."""
    tokens = [t.strip() for t in (variants or "smoke").split(",") if t.strip()]
    if tokens == ["smoke"]:
        return list(mlperf_variants_for(smoke=True))
    if tokens == ["all"]:
        return list(mlperf_variants_for())

    selected = []
    for token in tokens:
        found = (
            mlperf_variants_for([token])
            if token in MLPERF_FAMILIES
            else [mlperf_variant(token)]
        )
        for entry in found:
            if entry not in selected:
                selected.append(entry)
    return selected


def _mlperf_stacks(stacks: Optional[str]) -> list:
    """The --stacks argument: which block stacks a POSIX arm is run on."""
    stack_list = (
        [s.strip() for s in stacks.split(",") if s.strip()]
        if stacks
        else [DEFAULT_STACK]
    )
    for stack_id in stack_list:
        mlperf_stack(stack_id)  # raises with the known ids
        if stack_id == RAW_STACK:
            raise ValueError(
                f"'{RAW_STACK}' is not a selectable stack: it is what an "
                "accelstore arm runs on, and every accelstore point of a sweep "
                "already uses it"
            )
    return stack_list


def _mlperf_points(
    variants: Optional[str], arms: Optional[str], stacks: Optional[str] = None
) -> list:
    """Every (variant, arm, stack) triple the selection runs.

    An accelstore arm is emitted once however many stacks were asked for: it
    drives the NVMe through vfio, so no stack applies to it.
    """
    tokens = [a.strip() for a in arms.split(",") if a.strip()] if arms else []
    named = bool(tokens) and tokens != ["all"]
    for arm_id in tokens if named else ():
        if arm_id not in MLPERF_ARMS:
            raise ValueError(
                f"Unknown MLPerf arm '{arm_id}'. Known: "
                f"{', '.join(sorted(MLPERF_ARMS))} (or 'all')"
            )
    if named:
        wanted = set(tokens)
    elif tokens:  # ["all"], the way --variants spells it
        wanted = None
    else:
        wanted = set(DEFAULT_ARMS)
    stack_list = _mlperf_stacks(stacks)

    points, skipped = [], set()
    for entry in _mlperf_selection(variants):
        blocked = entry.blocked_arms
        for arm_id in entry.arms:
            if wanted is not None and arm_id not in wanted:
                if not named:
                    skipped.add(arm_id)
                continue
            if arm_id in blocked:
                if named:
                    print(f"NOTE: {entry.id} on {arm_id} is blocked: {blocked[arm_id]}")
                continue
            arm = MLPERF_ARMS[arm_id]
            for stack in mlperf_stacks_for_arm(arm, stack_list):
                points.append((entry, arm, stack))

    if skipped:
        names = ",".join(a for a in OPTIONAL_ARMS if a in skipped)
        print(
            f"NOTE: not running the opt-in arms {names} -- they are reference"
            " rows rather than storage-stack rows (their notes in the table say"
            f" why). Add them with --arms {names}, or run every arm each row"
            " declares with --arms all"
        )

    if not points:
        raise ValueError(
            f"No runnable MLPerf point in --variants {variants or 'smoke'}"
            + (f" --arms {arms}" if arms else "")
        )
    return points


def _require_mlperf_stack_guest(setups: list, points: list) -> None:
    """Refuse a stack whose module is not in the installed guest image."""
    from core.tasks.config import BUILD_DIR

    stacks = {stack for _, _, stack in points if stack.kernel}
    for stack in sorted(stacks, key=lambda s: s.id):
        for image in sorted({_mlperf_guest_image(s) for s in setups}):
            marker = Path(BUILD_DIR) / "image" / f"{image}.kernel"
            installed = marker.read_text().strip() if marker.exists() else None
            if installed != stack.kernel:
                raise RuntimeError(
                    f"stack '{stack.id}' needs the linux {stack.kernel} guest; "
                    f"{image} has "
                    + (f"linux {installed}" if installed else "no kernel marker")
                    + ". Build it with `oma-arvio accelstore.build-guest --kernel "
                    f"{stack.kernel} --mlperf`"
                )


def _mlperf_capacity_gb(plan: list, guest_gb: dict, override):
    """The window every POSIX arm of this sweep is given, or None.

    One figure for the whole sweep, so the arms of a row see the same capacity.
    `None` hands over the whole device. Checked in both directions: a window
    past a stack's ceiling and one below what the sweep's largest point writes
    are both silent failures hours in.
    """
    stacked = [s for _, _, s, _ in plan if s.is_stacked]
    per_variant = {}
    for entry, arm, _stack, _size in plan:
        if arm.is_accelstore:
            continue
        per_variant[entry.id] = max(
            per_variant.get(entry.id, 0), entry.capacity_needed_gb()
        )
    needed = max(per_variant.values(), default=0)
    # Only a stack that declares a window derives one. LUKS declares none.
    bounded = {s.id for s in stacked if s.default_capacity_gb is not None}
    if override is not None:
        capacity = int(override)
    elif not bounded:
        return None  # nothing windowed: the whole device
    else:
        capacity = sweep_window_gb(bounded, needed)
    smallest_guest = min(guest_gb.values()) if guest_gb else None
    for stack in sorted({s.id for s in stacked}):
        for warning in check_capacity(stack, capacity, smallest_guest):
            print(f"NOTE: {warning}")
    if needed > capacity:
        offenders = ", ".join(
            f"{variant_id} ({want} GB)"
            for variant_id, want in sorted(per_variant.items())
            if want > capacity
        )
        raise ValueError(
            f"the sweep's POSIX arms need up to {needed} GB but the window is "
            f"{capacity} GB: {offenders}. "
            + (
                "Raise --stack-capacity-gb if every stack in the sweep allows "
                "it, drop the point, or lower what it writes"
                if override is not None
                else "The window is already as large as the sweep's stacks "
                "allow, so this needs a stack whose limit is not a hard one "
                "(--stacks ext4,rollbaccine), a dropped point, or less written"
            )
        )
    return capacity


def _vm_resources(hostname: str, size: str):
    """The config.toml vm_resources entry one VM size resolves to."""
    from core.tasks.config import load_config

    cfg = load_config()
    host = cfg.hosts.get(hostname)
    resources = (host.vm_resources if host else None) or cfg.default_vm_resources
    if size not in resources:
        raise ValueError(
            f"No vm_resources entry for size '{size}' on {hostname}; "
            f"known: {', '.join(sorted(resources))}"
        )
    return resources[size]


def _vm_memory_gb(hostname: str, size: str) -> float:
    """The guest's memory for one VM size, from config.toml."""
    return float(_vm_resources(hostname, size).memory)


def _check_mlperf_cores(plan: list, hostname: str, workers) -> None:
    """Refuse a VM that cannot hold the run's processes, before the first boot.

    An undersized guest does not fail the run, it produces a number for a
    topology that never had the parallelism its label claims. The requirement
    is the variant's, not the (variant, arm) pair's, so the arms of a row stay
    comparable.
    """
    for entry, size in {(v.id, size): (v, size) for v, _, _, size in plan}.values():
        needed = entry.cores_needed(workers=workers)
        cpus = int(_vm_resources(hostname, size).cpu)
        if cpus < needed:
            ranks = max(entry.rank_processes, 1)
            shape = f"{ranks} rank process(es)"
            if any(entry.ipc_master(a) for a in entry.arms):
                shape += (
                    f" + {entry.workers_for(workers) or SESSION_WORKERS_DEFAULT}"
                    " store-master workers + 1 owner core"
                )
            raise ValueError(
                f"{entry.id} wants {needed} guest cores ({shape}), and size "
                f"{size} on {hostname} has {cpus}. Run it at a larger --sizes."
            )


def _mlperf_hugepages(
    plan: list,
    hostname: str,
    *,
    workers,
    crypto_chunk_size,
    cluster_size,
    md_pages,
    pipeline_bytes,
    requested,
) -> dict:
    """The hugepage reservation per VM size, refusing an impossible one here.

    Derived from the guest's own memory and the resolved store geometry
    (`store_memory`), since the reservation has to fit the guest and cover the
    store at once. Every size in the plan gets an entry, POSIX-only ones
    included, where the value is simply unused.
    """

    def demand(entry):
        # The resolved geometry: a row declares its own, the flags override fields.
        geometry = entry.geometry_for(
            chunk_size=crypto_chunk_size,
            cluster_size=cluster_size,
            md_pages=md_pages,
            pipeline_bytes=pipeline_bytes,
        )
        return (
            entry.workers_for(workers) or SESSION_WORKERS_DEFAULT,
            int(geometry.chunk_size or store_memory.CHUNK_SIZE_DEFAULT),
            int(geometry.pipeline_bytes or store_memory.PIPELINE_BYTES_DEFAULT),
        )

    reservations = {}
    for size in dict.fromkeys(size for *_, size in plan):
        demands = [
            demand(entry)
            for entry, arm, _stack, at in plan
            if at == size and arm.is_accelstore
        ]
        reservations[size] = (
            store_memory.hugepages_gb(
                _vm_memory_gb(hostname, size),
                *max(demands, key=lambda d: store_memory.required_gb(*d)),
                requested=None if requested is None else int(requested),
            )
            if demands
            else None
        )
    return reservations


@task(name="run-mlperf")
def run_mlperf(
    ctx,
    variants: str = "smoke",
    arms: str = None,
    stacks: str = None,
    stack_capacity_gb: Optional[int] = None,
    setups: str = "snp",
    sizes: str = None,
    hostname: str = None,
    orgname: str = "TUM-DSE",
    systemname: str = None,
    client_host_memory_gb: Optional[float] = None,
    mlpstorage_args: str = None,
    image: str = mlperf_docker.MLPERF_IMAGE,
    image_wait_s: int = 900,
    workers: Optional[int] = None,
    max_op_bytes: Optional[int] = None,
    crypto_chunk_size: Optional[int] = None,
    pipeline_bytes: Optional[int] = None,
    cluster_size: Optional[int] = None,
    md_pages: Optional[int] = None,
    hugepages_gb: Optional[int] = None,
    tracing: Optional[str] = None,
    trace_entries: Optional[int] = DEFAULT_TRACE_ENTRIES,
    trace_crypto_chunks: Optional[bool] = None,
    skip_precondition: bool = False,
    precondition_scheme: str = "null",
    sar_enabled: bool = True,
    sar_interval: int = 1,
    sar_options: str = "-u -r -n DEV",
    continue_on_error: bool = True,
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
    """Run MLPerf Storage points in the guest's container, one boot per point.

    A point is one (variant, arm, stack) triple of the declarative table in
    ``modules/accelstore/tasks/mlperf_variants.py``; this task only chooses
    which of its rows to run. Every point is validated (shape, arm, run
    context, geometry, cores) before the first VM boots. Needs the MLPerf
    guest (`build-guest --mlperf`), which is asserted here.

    Args:
        variants: "smoke" (default: the marked iteration subset), "all", or a
                  comma-separated list of family names and variant ids
        arms: Comma-separated arm filter (accelstore-log, accelstore-fs,
              posix-direct, posix-buffered, direct), or `all`. Default is the
              arms that measure the storage stack: the AccelStore modules and
              `posix-direct`. An arm the table blocks is reported and skipped
        stacks: Comma-separated block stacks the POSIX arms are run on (ext4,
              ext4-aes-xts-plain, rollbaccine; default ext4). Arms and stacks
              cross, except that each accelstore arm runs once however many
              stacks were named. The guest image is asserted to carry the
              stack's kernel before the first boot
        stack_capacity_gb: Window every POSIX arm of the sweep is given, in GB
              (10^9), plain ext4 included. Default is the smallest ceiling
              among the sweep's stacks, or the whole device when no selected
              stack has one. See tasks/block_stacks.py for which limit each
              stack hits
        setups: Comma-separated VM setups (default: snp)
        sizes: Comma-separated VM sizes, applied to every selected point.
               Default is each row's own (`MlperfVariant.vm_size`)
        hostname: Host to run on (default: current host)
        orgname: Organization the results directory is pinned to by
                 `mlpstorage init`
        systemname: --systemname (default: the host the sweep runs on)
        client_host_memory_gb: --client-host-memory-in-gb; default is the VM
                 size's memory from config.toml
        mlpstorage_args: Extra arguments appended to every mlpstorage command
                 (e.g. "--skip-fs-separation-gate --quiet")
        image / image_wait_s: Container image and how long to wait for the
                 boot-time `docker load`
        workers / max_op_bytes / crypto_chunk_size / pipeline_bytes /
                 cluster_size / md_pages: AccelStore store geometry for the
                 accelstore arms. Each overrides the field of the variant's own
                 `store_geometry` it names and leaves the rest standing
        hugepages_gb: Hugepages reserved in the guest for the accelstore arms
                 (default: derived from the resolved geometry and the guest's
                 memory)
        tracing: SPDK trace recording on the accelstore arms ("tracing",
                 "small-trace", None)
        trace_entries / trace_crypto_chunks: ring size and whether the
                 per-AEAD-chunk spans are recorded
        skip_precondition / precondition_scheme: as in run-bench
        sar_enabled / sar_interval / sar_options: SAR recording, as in run-all
        continue_on_error: Keep going when a point fails (default True)
        sync_* / tc_*: One SyncMode for the accelstore arms; a POSIX arm has no
                 metadata epochs, so a non-default mode is refused on it

    Example:
        oma-arvio accelstore.run-mlperf
        oma-arvio accelstore.run-mlperf --variants ckpt-llama3-8b-1rank --arms all
        oma-arvio accelstore.run-mlperf --variants ckpt-llama3-8b-1rank \\
            --stacks ext4,rollbaccine
    """
    setup_list = [s.strip() for s in setups.split(",")]
    size_list = [s.strip() for s in sizes.split(",")] if sizes else None
    points = _mlperf_points(variants, arms, stacks)
    _require_mlperf_guest(setup_list)
    _require_mlperf_stack_guest(setup_list, points)

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
    posix_points = [f"{v.id}/{a.id}" for v, a, _ in points if not a.is_accelstore]
    if posix_points and not sync.is_default():
        raise ValueError(
            f"Sync mode '{sync.label()}' has no meaning on the baseline arms "
            f"({', '.join(posix_points)}); they have no metadata epochs. Run "
            "them separately, or restrict --arms"
        )
    if tracing and posix_points:
        raise ValueError(
            "SPDK tracing only exists on an accelstore arm; drop --tracing or "
            f"the baseline arms ({', '.join(posix_points)})"
        )

    devices = Devices(hostname)
    plan = [
        (entry, arm, stack, size)
        for entry, arm, stack in points
        for size in (size_list or [entry.vm_size])
    ]
    # guest_gb is the RAM the VM gets, declared_gb only --client-host-memory-in-gb.
    guest_gb = {size: _vm_memory_gb(devices.hostname, size) for *_, size in plan}
    declared_gb = {
        (entry.id, size): (
            float(client_host_memory_gb)
            if client_host_memory_gb is not None
            else guest_gb[size]
        )
        for entry, _arm, _stack, size in plan
    }

    for entry, arm, _stack, size in plan:
        entry.validate()
        if arm.is_accelstore:
            # entry.validate() checks the declaration, not the resolved geometry.
            try:
                entry.geometry_for(
                    chunk_size=crypto_chunk_size,
                    cluster_size=cluster_size,
                    md_pages=md_pages,
                    pipeline_bytes=pipeline_bytes,
                )
            except ValueError as exc:
                raise ValueError(f"{entry.id}/{arm.id}: {exc}") from None
        entry.validate_context(
            mlperf_docker.build_run_context(
                arm,
                systemname=systemname or devices.hostname,
                client_host_memory_gb=declared_gb[(entry.id, size)],
            ),
            arm,
        )
    _check_mlperf_cores(plan, devices.hostname, workers)
    guest_hugepages = _mlperf_hugepages(
        plan,
        devices.hostname,
        workers=workers,
        crypto_chunk_size=crypto_chunk_size,
        cluster_size=cluster_size,
        md_pages=md_pages,
        pipeline_bytes=pipeline_bytes,
        requested=hugepages_gb,
    )
    capacity_gb = _mlperf_capacity_gb(plan, guest_gb, stack_capacity_gb)

    combinations = [
        (setup, size, (entry, arm, stack))
        for setup, (entry, arm, stack, size) in itertools.product(setup_list, plan)
    ]
    print(f"Starting MLPerf Storage suite: {len(combinations)} runs")
    if capacity_gb is not None:
        stacked = sorted({s.id for _, _, s in points if s.is_stacked})
        print(
            f"POSIX arms get a {capacity_gb} GB window"
            + (f" (stacks: {', '.join(stacked)})" if stacked else "")
        )
        if "rollbaccine" in stacked:
            print(
                f"  rollbaccine keeps a "
                f"{rollbaccine_tree_gb(capacity_gb):.1f} GB checksum tree in "
                "guest RAM at this window (disk_pages_for_merkle_tree=0)"
            )

    all_results = []
    overall_start = time.time()

    for setup, size, (entry, arm, stack) in combinations:
        label = f"{setup}-{size}-{entry.id}-{arm.id}" + (
            f"-{stack.id}" if stack.is_stacked else ""
        )
        target = devices.storage_target(setup)
        action_config = {
            "variant": entry.id,
            "arm": arm.id,
            "stack": stack.id,
            "stack_capacity_gb": capacity_gb,
            "dev_path": target.dev_path,
            "device": "Nvme1n1",
            "orgname": orgname,
            "systemname": systemname or devices.hostname,
            "client_host_memory_gb": declared_gb[(entry.id, size)],
            "mlpstorage_args": mlpstorage_args,
            "image": image,
            "image_wait_s": image_wait_s,
            "skip_precondition": skip_precondition,
            "precondition_scheme": precondition_scheme,
            **sync.as_config(),
        }
        if arm.is_accelstore:
            action_config.update(
                {
                    "config": _spdk_bdev_config(target.pci_dev),
                    "pci_device": target.pci_dev,
                    "workers": workers,
                    "max_op_bytes": max_op_bytes,
                    "crypto_chunk_size": crypto_chunk_size,
                    "pipeline_bytes": pipeline_bytes,
                    "cluster_size": cluster_size,
                    "md_pages": md_pages,
                    "hugepages_gb": guest_hugepages[size],
                    "tracing": tracing,
                    "trace_entries": trace_entries,
                    "trace_crypto_chunks": trace_crypto_chunks,
                }
            )

        print(
            f"Running: {label} ({entry.label} on {arm.label}"
            + (f", {stack.label}" if stack.is_stacked else "")
            + ")"
        )
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
                    action="run-mlperf",
                    sar_enabled=sar_enabled,
                    sar_interval=sar_interval,
                    sar_options=sar_options,
                    vfio_pcie=[target.vfio_device] if target.vfio_device else None,
                    action_config=action_config,
                    kvm_perf_enabled=False,
                )
            all_results.append(
                {
                    "config": label,
                    "status": "SUCCESS",
                    "duration": time.time() - start_time,
                }
            )
        except Exception as e:
            all_results.append(
                {
                    "config": label,
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


def _seal_unseal_pairs(client: str, data_sizes: Optional[str]) -> list:
    """The client's seal/unseal pairs, narrowed to `data_sizes` if given."""
    pairs = ACCELSTORE_SEAL_UNSEAL_JOBS[client]
    if not data_sizes:
        return pairs

    available = [p[0].split("_")[2] for p in pairs]
    wanted = [s.strip().lower() for s in data_sizes.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in available]
    if unknown:
        raise ValueError(
            f"Unknown seal/unseal size(s) {', '.join(unknown)}."
            f" Sizes: {', '.join(available)}"
        )
    return [p for p, size in zip(pairs, available) if size in wanted]


@task(name="run-seal-unseal")
def run_seal_unseal(
    ctx, hostname: str = None, crypto: str = "aegis", data_sizes: str = None
):
    """Run AccelStore paired workload benchmarks (seal→unseal sequences).

    Paired seal-then-unseal workloads for all three clients, each pair a
    sequence with fresh blobstore initialization and a trusted counter that
    persists between the two jobs. Hardcoded: SNP VM, medium size, small-trace,
    no preconditioning between sequences.

    Args:
        hostname: Host to run benchmarks on (default: current host)
        crypto: Crypto build variants to run (default: the AEGIS build only)
        data_sizes: Comma-separated workload sizes — 100mb, 1gb, 10gb, 100gb,
                    1000gb (default: all five). Not the VM size

    Output:
        bench-result/accelstore/snp-disk-medium/{client}/{seal}#{unseal}/{timestamp}_seq{N}.log
        bench-result/accelstore/snp-disk-medium/{client}/{seal}#{unseal}/{timestamp}_tc.bin
    """
    for client_name in ACCELSTORE_SEAL_UNSEAL_JOBS:
        pairs = _seal_unseal_pairs(client_name, data_sizes)
        job_sequences = ";".join([",".join(pair) for pair in pairs])

        run_all(
            ctx,
            setups="snp",
            sizes="medium",
            clients=client_name,
            hostname=hostname,
            job_sequences=job_sequences,
            crypto=crypto,
            tracing="small-trace",
            continue_on_error=True,
            precondition_scheme="null",
        )


def _run_sync_point(
    ctx,
    sync: SyncMode,
    hostname: Optional[str],
    crypto: str,
    jobs: dict,
) -> None:
    """One sweep point: one representative workload per client, one mode."""
    for client_name, job_list in jobs.items():
        run_all(
            ctx,
            setups="snp",
            sizes="medium",
            clients=client_name,
            hostname=hostname,
            job_sequences=";".join(job_list),
            crypto=crypto,
            tracing="small-trace",
            continue_on_error=True,
            precondition_scheme="null",
            **sync.as_task_kwargs(),
        )


@task(name="run-epoch-sweep")
def run_epoch_sweep(
    ctx,
    hostname: str = None,
    crypto: str = "aegis",
    trigger: str = "every-t",
    min_epoch_ms: int = sync_modes.EPOCH_MS_MIN,
    max_epoch_ms: int = sync_modes.EPOCH_MS_MAX,
    step_epoch_ms: int = sync_modes.EPOCH_MS_STEP,
    rtt_us: int = sync_modes.PAPER_RTT_US,
    arms: str = "full,metadata-only",
    blocking: str = "on",
    baseline: bool = True,
    per_op: bool = True,
):
    """Sweep the metadata epoch (§VII-B's figure).

    One representative workload per client (ACCELSTORE_EPOCH_JOBS) at each
    point of an epoch ladder, against a mock counter service whose commits are
    asynchronous and cost `rtt_us` each. Two arms, both read against the same
    baseline: `full` (epochs plus counter commits) and `metadata-only` (the
    same epochs with no commit), their difference the freshness cost.

    Args:
        hostname: Host to run benchmarks on (default: current host)
        crypto: Crypto build variants (default: the AEGIS build alone)
        trigger: "every-t" (a wall-clock durability window, the paper's x axis),
                 "every-n" (mutating operations between epochs, the
                 sync_modes.EVERY_N_LADDER), or "both"
        min_epoch_ms/max_epoch_ms/step_epoch_ms: the every-t ladder
        rtt_us: Counter round trip, default the paper's 5 ms
        arms: Comma-separated subset of "full,metadata-only"
        blocking: Whether the ladder is run with the durability gate shut —
                  "both" (default: each point without and with it, the
                  non-blocking ladder first), "off" or "on". Ladder points
                  only: per-op and the baseline are never doubled
        baseline: Also run the default mode (on-shutdown, local counter) once
                  per client, which is what every overhead here is relative to
        per_op: Also run --sync-trigger per-op once per arm: the worst case,
                and the upper bound the sweep's overheads are read against

    Output:
        bench-result/accelstore/snp-disk-medium/{client}/{job}/            (baseline)
        bench-result/accelstore/snp-disk-medium/{client}/{job}/{sync-slug}/
    """
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    unknown = set(arm_list) - {"full", "metadata-only"}
    if unknown:
        raise ValueError(
            f"unknown arm(s) {sorted(unknown)}; valid: full, metadata-only"
        )
    if trigger not in ("every-t", "every-n", "both"):
        raise ValueError(f"trigger must be every-t, every-n or both (got '{trigger}')")
    gates = {"off": (False,), "on": (True,), "both": (False, True)}.get(blocking)
    if gates is None:
        raise ValueError(f"blocking must be off, on or both (got '{blocking}')")

    if baseline:
        _run_sync_point(ctx, SyncMode(), hostname, crypto, ACCELSTORE_EPOCH_JOBS)

    modes: list[SyncMode] = []
    for arm in arm_list:
        # metadata-only issues no commit, so it configures no counter service.
        counter = (
            {}
            if arm == "metadata-only"
            else {"tc_backend": "mock-remote", "tc_rtt_us": rtt_us}
        )
        metadata_only = arm == "metadata-only"

        for gate in gates:
            if trigger in ("every-t", "both"):
                for epoch_ms in sync_modes.epoch_ladder_ms(
                    min_epoch_ms, max_epoch_ms, step_epoch_ms
                ):
                    modes.append(
                        SyncMode(
                            trigger="every-t",
                            interval_us=epoch_ms * 1000,
                            blocking=gate,
                            metadata_only=metadata_only,
                            **counter,
                        )
                    )
            if trigger in ("every-n", "both"):
                for every_n in sync_modes.EVERY_N_LADDER:
                    modes.append(
                        SyncMode(
                            trigger="every-n",
                            every_n=every_n,
                            blocking=gate,
                            metadata_only=metadata_only,
                            **counter,
                        )
                    )

        # No gate: per-op already delivers at durability, and rejects the flag.
        if per_op:
            modes.append(
                SyncMode(
                    trigger="per-op",
                    metadata_only=metadata_only,
                    **counter,
                )
            )

    for mode in modes:
        mode.validate()

    print(
        f"Epoch sweep: {len(modes)} modes x {len(ACCELSTORE_EPOCH_JOBS)} clients"
        f" = {len(modes) * len(ACCELSTORE_EPOCH_JOBS)} runs"
        f" (gate: {blocking})"
        f"{' + baselines' if baseline else ''}"
    )
    for mode in modes:
        _run_sync_point(ctx, mode, hostname, crypto, ACCELSTORE_EPOCH_JOBS)


# All three are *_PATTERN-only on write, which is what --ablation zero-copy needs.
ACCELSTORE_ABLATION_JOBS = dict(ACCELSTORE_EPOCH_JOBS)

# The same 256 MB objects at a quarter of the records, for `--quick`.
ACCELSTORE_ABLATION_JOBS_QUICK = {
    "kvs_client": ["kvs_workload_A_256m_quick.txt"],
    "log_client": ["log_workload_256MB_write_read_seq_quick.txt"],
    "fs_client": ["fs_workload_write_read_256m_quick.txt"],
}


def ablation_jobs(jobs: Optional[str]) -> dict:
    """The client -> job map an ablation run, parse or plot works over."""
    if not jobs:
        return ACCELSTORE_ABLATION_JOBS

    job_list = [j.strip() for j in jobs.split(",") if j.strip()]
    resolved = {
        client: filter_jobs_for_client(job_list, client)
        for client in ACCELSTORE_ABLATION_JOBS
    }
    empty = [client for client, found in resolved.items() if not found]
    if empty:
        raise ValueError(
            f"--jobs names nothing for {', '.join(sorted(empty))}:"
            f" every ablation client needs one job, got {jobs}"
        )
    return resolved


# (ablation spec, crypto build id): a token is only legal in some builds.
ABLATION_ARMS = [
    ("none", "aegis"),
    ("no-overlap", "aegis"),
    ("bounce-copy", "aegis"),
    ("no-encrypt", "aegis"),
    ("no-verify", "aegis"),
    ("none", "aesgcm"),
    ("no-overlap", "aesgcm"),
    ("bounce-copy", "aesgcm"),
    ("no-encrypt", "aesgcm"),
    ("no-verify", "aesgcm"),
    ("none", "none"),
    ("zero-copy", "none"),
]


def _ablation_points(arms: Optional[str], clients: Optional[str]) -> list:
    """Resolve --arms/--clients into validated (Ablation, backend, client)."""
    selected = ABLATION_ARMS
    if arms:
        wanted = {a.strip() for a in arms.split(",") if a.strip()}
        known = {f"{cid}:{spec}" for spec, cid in ABLATION_ARMS}
        known |= {spec for spec, _ in ABLATION_ARMS}
        unknown = wanted - known
        if unknown:
            raise ValueError(
                f"Unknown arm(s) {sorted(unknown)}. Known:"
                f" {', '.join(sorted(f'{cid}:{spec}' for spec, cid in ABLATION_ARMS))}"
            )
        selected = [
            (spec, cid)
            for spec, cid in ABLATION_ARMS
            if f"{cid}:{spec}" in wanted or spec in wanted
        ]

    client_list = (
        [c.strip() for c in clients.split(",")] if clients else ACCELSTORE_CLIENTS
    )
    for client in client_list:
        if client not in ACCELSTORE_ABLATION_JOBS:
            raise ValueError(
                f"No ablation workload for client '{client}'. Known:"
                f" {', '.join(sorted(ACCELSTORE_ABLATION_JOBS))}"
            )

    points = []
    for spec, cid in selected:
        abl = Ablation.parse(spec)
        backend = CryptoBackend.parse(cid)
        abl.validate(backend)
        for client in client_list:
            points.append((abl, backend, client))

    # Baselines first, so an interrupted sweep still leaves a readable comparison.
    points.sort(key=lambda p: bool(p[0].tokens))
    return points


@task(name="run-ablation")
def run_ablation(
    ctx,
    hostname: str = None,
    clients: str = None,
    arms: str = None,
    repeat: int = 1,
    tracing: Optional[str] = "small-trace",
    trace_entries: Optional[int] = DEFAULT_TRACE_ENTRIES,
    io_path: Optional[str] = None,
    print_only: bool = False,
    jobs: Optional[str] = None,
):
    """Sweep the data-path ablation switches on the 256 MB workloads.

    One arm takes away one property of the data path — the pipeline's overlap,
    its avoided copy, the AEAD, the read-path checks, or every copy at once —
    and what it costs is that run's end-to-end time against the unablated run
    of the same build. Every build's baseline is part of the sweep, since it is
    the denominator of every ratio read off it. `parse-ablation` reads the arms
    back. 12 arms x 3 clients = 36 VM boots per repeat. Tracing is not
    decorative: the parser reads the runtime and the per-bucket breakdown out
    of the trace.

    Args:
        hostname: Host to run benchmarks on (default: current host)
        clients: Comma-separated clients (default: all three)
        arms: Comma-separated arm names, either "<crypto>:<spec>"
              (aegis:no-overlap, none:zero-copy) or a bare spec matching every
              build it appears in. Default: all of ABLATION_ARMS
        repeat: Passes over the whole arm list (default 1), as the outer loop.
                The parser aggregates the repeats it finds
        tracing: SPDK trace mode; the runtime is read out of the trace
        trace_entries: Trace ring entries per lcore
        io_path: Run the whole sweep over one NVMe driver arm — "spdk"
                 (bdev_nvme) or "kernel" (bdev_aio on the namespace node).
                 Omitted, the sweep lands where it always has
        print_only: Resolve and validate every point, print the plan, run nothing
        jobs: Comma-separated job files, one per client (default: the 256 MB
              ACCELSTORE_ABLATION_JOBS). Parse and plot need the same list

    Output:
        bench-result/accelstore/snp-disk-medium/{client}/{job}/abl-{arm}/
        bench-result/accelstore/snp-disk-medium/{client}/{job}/io-{arm}/abl-{arm}/
        (the second form when --io-path is given)
    """
    repeat = int(repeat)
    if repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")

    points = _ablation_points(arms, clients)
    io = IoPath.parse(io_path)
    workloads = ablation_jobs(jobs)

    print(
        f"Ablation sweep: {len(points)} points x {repeat} repeat(s)"
        f" = {len(points) * repeat} runs, io path {io.label()}"
    )
    for abl, backend, client in points:
        job = workloads[client][0]
        args = " ".join(abl.client_args()) or "(no flag)"
        print(
            f"  {backend.value:6} {client:10} {abl.path_component():18}"
            f" {job:34} {args}"
        )
    if print_only:
        return

    for iteration in range(repeat):
        if repeat > 1:
            print(f"\n=== Ablation pass {iteration + 1}/{repeat} ===")
        for abl, backend, client in points:
            run_all(
                ctx,
                setups="snp",
                sizes="medium",
                clients=client,
                hostname=hostname,
                job_sequences=";".join(workloads[client]),
                crypto=[backend],
                tracing=tracing,
                trace_entries=trace_entries,
                continue_on_error=True,
                precondition_scheme="null",
                ablation=abl.spec,
                **io.as_task_kwargs(),
            )


KERNEL_VARIANTS = ["6.17"]


def _kernel_suffix(variant: str) -> str:
    return variant.replace(".", "-")


MLPERF_BASE_IMAGE_TAR = "mlperf-base-image.tar"
MLPERF_BASE_IMAGE_INPUT = "mod-accelstore/mlperf-base-image"


def _guest_target(image: str, kernel: str, mlperf: bool) -> str:
    """Flake attribute for one point of the (kernel x mlperf) guest grid."""
    return f"{image}-linux-{_kernel_suffix(kernel)}" + ("-mlperf" if mlperf else "")


def _mlperf_base_image_path(tar: Optional[str] = None) -> Path:
    from core.tasks.config import BUILD_DIR

    return (
        Path(tar).resolve()
        if tar
        else Path(BUILD_DIR).resolve() / MLPERF_BASE_IMAGE_TAR
    )


def _nix_build(target: str, extra_args: Optional[list] = None):
    """`nix build --json` with extra flags, which core's nix_build does not take."""
    import json

    from core.tasks.config import BUILD_DIR
    from core.tasks.procs import run

    Path(BUILD_DIR).mkdir(parents=True, exist_ok=True)
    outpath = Path(BUILD_DIR) / target.lstrip(".#")
    cmd = ["nix", "build", "--out-link", str(outpath), "--json"]
    cmd += extra_args or []
    cmd += [target]
    return json.loads(run(cmd, cwd=PROJECT_ROOT).stdout)


@task(name="build-mlperf-image")
def build_mlperf_image(_ctx, out: str = None):
    """Build the MLPerf Storage base container image with the host's docker.

    `docker build` cannot run in a nix builder, so the base image is built
    outside nix and handed back to the flake as the `mlperf-base-image` input.
    `build-guest --mlperf` runs this first unless told not to. Needs a
    reachable docker daemon on the host.

    Args:
        out: Where to write the docker-archive tar
             (default: build/mlperf-base-image.tar)

    Example:
        oma-arvio accelstore.build-mlperf-image
    """
    from core.tasks.procs import run

    tar = _mlperf_base_image_path(out)
    tar.parent.mkdir(parents=True, exist_ok=True)
    print(f"Building the MLPerf base image with the host docker -> {tar}")
    run(
        ["nix", "run", ".#build-mlperf-image", "--", str(tar)],
        cwd=PROJECT_ROOT,
        stdout=None,
    )
    return tar


@task(name="build-guest")
def build_guest(
    ctx,
    kernel: str = "6.17",
    image: str = "snp-guest-image",
    force: bool = True,
    mlperf: bool = False,
    build_mlperf_base_image: bool = True,
    mlperf_base_image: str = None,
):
    """Build a guest image with a specific guest kernel and make it the active one.

    6.17 carries the CVM patches (MAP_CVM_SHARED, user-space MMIO,
    walk_page_range export). The built image is installed as the canonical
    build/image/<image>.raw that `vm.start` boots, and the active variant is
    recorded in build/image/<image>.kernel (plus build/image/<image>.mlperf
    when the MLPerf container is in it).

    Args:
        kernel: Kernel variant — "6.17" (default, matches the submitted
                measurements)
        image: "snp-guest-image" (default) or "direct-guest-image"
        force: Overwrite an already-installed image (default: True — otherwise
               switching variants would silently keep the old image)
        mlperf: Include docker and the MLPerf Storage container (default: off)
        build_mlperf_base_image: Rebuild the base container image with the
                host's docker before building the guest (default: True).
                --no-build-mlperf-base-image reuses the existing tar, and
                fails if there is none. Ignored without --mlperf
        mlperf_base_image: Path to the base image tar
                (default: build/mlperf-base-image.tar)

    Example:
        oma-arvio accelstore.build-guest --kernel 6.17
        oma-arvio accelstore.build-guest --mlperf
        oma-arvio accelstore.build-guest --mlperf --no-build-mlperf-base-image
    """
    import shutil

    from core.tasks.config import BUILD_DIR
    from core.tasks.procs import run

    if kernel not in KERNEL_VARIANTS:
        raise ValueError(
            f"Unknown kernel variant '{kernel}'. Known: {', '.join(KERNEL_VARIANTS)}"
        )
    if image not in ("snp-guest-image", "direct-guest-image"):
        raise ValueError(f"Unknown image '{image}'")

    dst = Path(BUILD_DIR) / "image"
    dst_file = dst / f"{image}.raw"
    # Before anything is built: under --mlperf the first step is a slow docker build.
    if dst_file.exists() and not force:
        print(f"{dst_file} already exists. Skipping build.")
        return

    extra_args = []
    if mlperf:
        tar = _mlperf_base_image_path(mlperf_base_image)
        if build_mlperf_base_image:
            build_mlperf_image(ctx, out=str(tar))
        elif not tar.exists():
            raise FileNotFoundError(
                f"No MLPerf base image at {tar}. Build one with "
                f"`oma-arvio accelstore.build-mlperf-image`, or drop "
                f"--no-build-mlperf-base-image to have this task do it."
            )
        # file+file:// keeps the tar as-is: plain file://…tar would be unpacked.
        extra_args = [
            "--override-input",
            MLPERF_BASE_IMAGE_INPUT,
            f"file+file://{tar}",
        ]
    elif mlperf_base_image or not build_mlperf_base_image:
        print("Note: the --mlperf-base-image flags do nothing without --mlperf")

    target = _guest_target(image, kernel, mlperf)
    print(
        f"Building {image} with linux {kernel}"
        f"{' + mlperf' if mlperf else ''} ({target})"
    )

    result = _nix_build(f".#{target}", extra_args)

    dst.mkdir(parents=True, exist_ok=True)
    marker = dst / f"{image}.kernel"
    mlperf_marker = dst / f"{image}.mlperf"

    script = result[0]["outputs"]["out"]
    run([script], cwd=dst)
    shutil.move(dst / "main.raw", dst_file)
    marker.write_text(f"{kernel}\n")
    if mlperf:  # always rewritten: a stale marker would claim an image that is gone
        mlperf_marker.write_text("yes\n")
    else:
        mlperf_marker.unlink(missing_ok=True)

    print(f"Installed {dst_file} (linux {kernel}{', mlperf' if mlperf else ''})")


@task(name="active-guest-kernel")
def active_guest_kernel(_ctx, image: str = "snp-guest-image"):
    """Print which guest variant the installed image was built with."""
    from core.tasks.config import BUILD_DIR

    dst = Path(BUILD_DIR) / "image"
    marker = dst / f"{image}.kernel"
    if not marker.exists():
        print(
            f"No kernel marker for {image}. The image was either not built via "
            f"accelstore.build-guest, or not built at all."
        )
        return
    mlperf = " +mlperf" if (dst / f"{image}.mlperf").exists() else ""
    print(f"{image}: linux {marker.read_text().strip()}{mlperf}")


FIO_IOPS_JOBS = (
    "spdk",
    "libaio",
    "libaio-luks-ext4-aes",
    "libaio-luks-ext4-aes-xts",
    "libaio-luks-ext4-aegis128",
)


@task(name="run-fio-iops")
def run_fio_iops(ctx, hostname: str = None, iterations: int = 1):
    """Run the storage configuration motivation benchmarks across all VM types.

    Storage configs: spdk, libaio, luks-aes, luks-aes-xts, luks-aegis128
    on snp, host, and amd — medium VM, SD_CUSTOM preconditioning.
    """
    return fio_tasks.run_all(
        ctx,
        setups="snp,host,amd",
        sizes="medium",
        hostname=hostname,
        sar_enabled=True,
        sar_interval=1,
        sar_options="-u -r -n DEV",
        jobs=",".join(FIO_IOPS_JOBS),
        device_size="3500G",
        block_sizes="512",
        iterations=iterations,
        continue_on_error=True,
        precondition_scheme="sd_custom",
    )


@task(name="run-fio-bandwidth")
def run_fio_bandwidth(ctx, hostname: str = None, iterations: int = 1):
    """Run bandwidth-focused FIO tests across VM types and encryption configs."""
    readwrite_jobs = [
        "libaio_bandwidth",
        "libaio-ext4_bandwidth",
        "libaio-luks-ext4-aes_bandwidth",
        "libaio-luks-ext4-aes-xts_bandwidth",
        "libaio-luks-ext4-aegis128_bandwidth",
    ]
    readonly_jobs = [
        "libaio-dmverity-ext4_bandwidth",
    ]
    all_jobs = ",".join(readwrite_jobs + readonly_jobs)

    return fio_tasks.run_all(
        ctx,
        setups="snp,amd",
        sizes="medium",
        hostname=hostname,
        sar_enabled=True,
        sar_interval=1,
        sar_options="-u -r -n DEV",
        jobs=all_jobs,
        device_size="3500G",
        block_sizes="512",
        iterations=iterations,
        continue_on_error=True,
        precondition_scheme="sd_custom",
    )


@task(name="run-crypto")
def run_crypto(ctx, hostname: str = None):
    """Run crypto performance benchmarks using tcrypt and openssl.

    Output:
        bench-result/tcrypt/{setup}-disk-medium/{timestamp}.json
        bench-result/openssl/{setup}-disk-medium/{timestamp}.json
    """
    setups = ["snp", "host", "amd"]
    size = "medium"

    for action in ("run-tcrypt", "run-openssl"):
        for setup in setups:
            print(f"Running {action}: setup={setup} size={size}")
            try:
                vm_tasks.start(
                    ctx,
                    type=setup,
                    size=size,
                    hostname=hostname,
                    direct=False,
                    action=action,
                    sar_enabled=False,
                    kvm_perf_enabled=False,
                    action_config={},
                )
            except Exception as e:
                print(f"ERROR in {action} {setup}-{size}: {e}")


@task(name="parse-crypto")
def parse_crypto(ctx, output_file="bench-result/parsed/crypto-summary.csv"):
    """Parse and compare crypto benchmarks between VM, CVM, and Host.

    Discovers the latest tcrypt and openssl results per setup and writes a CSV
    with combined kernel/openssl throughput per algorithm.
    """
    from modules.crypto.tasks.utils.crypto import (
        ALGORITHMS,
        find_latest_json,
        parse_openssl_results,
        parse_tcrypt_results,
    )

    base_path = PROJECT_ROOT / "bench-result"
    setups = ["amd-disk-medium", "snp-disk-medium", "host-medium"]

    print("Discovering latest benchmark results...")

    tcrypt_files = {}
    openssl_files = {}

    for setup in setups:
        tcrypt_path = find_latest_json(base_path, setup, "tcrypt")
        openssl_path = find_latest_json(base_path, setup, "openssl")

        if tcrypt_path:
            print(f"  {setup} (tcrypt): {tcrypt_path.name}")
            tcrypt_files[setup] = tcrypt_path
        else:
            print(f"  {setup} (tcrypt): NOT FOUND")

        if openssl_path:
            print(f"  {setup} (openssl): {openssl_path.name}")
            openssl_files[setup] = openssl_path
        else:
            print(f"  {setup} (openssl): NOT FOUND")

    print("\nParsing benchmark results...")

    parsed_data = {}
    for setup in setups:
        tcrypt_results = parse_tcrypt_results(tcrypt_files.get(setup), ALGORITHMS)
        openssl_algos = [spec["openssl"] for spec in ALGORITHMS if spec.get("openssl")]
        openssl_results = parse_openssl_results(openssl_files.get(setup), openssl_algos)
        parsed_data[setup] = {"tcrypt": tcrypt_results, "openssl": openssl_results}

    csv_data = [["Algorithm", "VM Throughput", "CVM Throughput", "Host Throughput"]]

    for spec in ALGORITHMS:
        row = [spec["name"]]
        tcrypt_names = spec.get("tcrypt", [])
        openssl_name = spec.get("openssl")

        for setup in setups:
            tcrypt_throughput = None
            if tcrypt_names:
                tcrypt_throughputs = [
                    parsed_data[setup]["tcrypt"].get(tn)
                    for tn in tcrypt_names
                    if tn in parsed_data[setup]["tcrypt"]
                ]
                if tcrypt_throughputs:
                    tcrypt_throughput = max(tcrypt_throughputs)

            openssl_throughput = None
            if openssl_name:
                openssl_throughput = parsed_data[setup]["openssl"].get(openssl_name)

            tcrypt_str = f"{tcrypt_throughput:.2f}" if tcrypt_throughput else "-"
            openssl_str = f"{openssl_throughput:.2f}" if openssl_throughput else "-"
            row.append(f"{tcrypt_str} / {openssl_str}")

        csv_data.append(row)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)

    print(f"CSV written to: {output_path}")


@task(name="parse-seal-unseal")
def parse_seal_unseal(
    _ctx,
    bench_dir="bench-result/accelstore",
    csv_file="bench-result/parsed/seal-unseal-summary.csv",
):
    """Parse seal/unseal benchmark traces and export timing metrics to CSV.

    Output CSV columns:
        config, client, data_size, write_total_s, read_total_s, read_init_s
    """
    bench_path = Path(bench_dir)
    if not bench_path.exists():
        print(f"Error: Benchmark directory not found: {bench_dir}")
        return

    print(f"Analyzing benchmarks in: {bench_dir}")

    results = analyze_seal_unseal_benchmarks(bench_path)

    if not results:
        print("No benchmark results found.")
        return

    csv_path = Path(csv_file)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_csv(results, csv_path)
    print(f"Analyzed {len(results)} workload pairs")
    print(f"Results exported to: {csv_file}")


@task(name="parse-bench")
def parse_bench(
    _ctx,
    bench_dir: str = "bench-result/bench",
    csv_file: str = "bench-result/parsed/bench-summary.csv",
    flowops_csv: str = "bench-result/parsed/bench-flowops.csv",
    table: bool = True,
    all_runs: bool = False,
):
    """Parse run-bench results into CSVs comparable across backend arms.

    Reads bench_client's own metrics — the --json sidecar next to a log where
    there is one, the metrics printer's lines otherwise — for the newest run
    of every point under bench_dir. What counts as a repeat of one measurement
    is `BenchRunMetrics.cell_key`, and every run a newer one displaces is named
    on the console. `--all-runs` writes every log in the tree instead, told
    apart by the `log` column.

    Summary CSV, one row per measured phase:
        config, arm, backend, type, personality, variant, status, source,
        dsl, seed, threads, bound, sync_config,
        direct, fsync_mode, barrier_mode, log_layout, crypto_chunk, sampled,
        ack, phase, phase_bound, ops,
        wall_ms, ops_per_s, read_mib, read_mibs, write_mib, write_mibs,
        effective_concurrency, pacing_share, prealloc_ops, prealloc_ms,
        failed_ops, lease_wait_ms, log

    Flowop CSV, one row per flowop of a phase (`delay` has no latencies, it
    carries the time it slept in total_ms):
        config, arm, backend, type, personality, variant, phase, flowop, ops,
        avg_ms, min_ms, max_ms, p50_ms, p99_ms, total_ms

    Every point keeps a row, with the gate's verdict in `status`: no-results
    for a run that died before its metrics printer, exit-nonzero for one that
    failed an operation, starved for one whose workers spent the wall clock
    waiting. The flag columns (`direct` .. `ack`) come from the
    ``*_flags.json`` the bench action writes next to each log, and are blank
    for runs collected before that sidecar existed.

    Args:
        bench_dir: Result tree run-bench wrote (default bench-result/bench)
        csv_file: Per-phase summary CSV
        flowops_csv: Per-flowop CSV; empty string to skip it
        table: Also print the per-run comparison table (default True)
        all_runs: Every run of every point, not only the newest (default False)

    Example:
        oma-arvio accelstore.parse-bench
        oma-arvio accelstore.parse-bench --bench-dir bench-result/bench
        oma-arvio accelstore.parse-bench --all-runs --csv-file bench-result/parsed/bench-history.csv
    """
    from modules.accelstore.tasks.results import bench_results

    bench_path = Path(bench_dir)
    if not bench_path.exists():
        print(f"Error: Benchmark directory not found: {bench_dir}")
        return

    runs = bench_results.export_bench_csv(
        bench_path,
        Path(csv_file),
        Path(flowops_csv) if flowops_csv else None,
        all_runs=all_runs,
    )
    if not runs:
        print(f"No bench_client results found under {bench_dir}")
        return
    if table:
        bench_results.print_bench_table(runs)


@task(name="parse-ablation")
def parse_ablation(
    _ctx,
    bench_dir: str = "bench-result/accelstore",
    csv_file: str = "bench-result/parsed/ablation-summary.csv",
    aggregate: str = "median",
    table: bool = True,
    jobs: Optional[str] = None,
):
    """Parse run-ablation's arms into relative end-to-end runtime.

    Reads the workload window of every `abl-*` run under bench_dir — the time
    between the client's WORKLOAD_START/END markers — and divides each arm by
    the unablated run of its own build. `relative_cif` additionally puts every
    row on the AEGIS axis, so the no-CIF baseline's own row reports the cost of
    CIF rather than 1.000. Runtime comes from the trace where a run has one and
    from the log where it does not; a row whose trace and log disagree is
    marked `runtime-mismatch` rather than averaged.

    CSV, one row per arm, including the ones that produced nothing:
        config, client, crypto, workload, arm, tokens, slug, status, runs,
        runtime_ms, runtime_ms_min, runtime_ms_max, runtime_source,
        window_source, baseline_ms, relative, overhead_pct, cif_baseline_ms,
        relative_cif, overhead_cif_pct, encryption_ms, metadata_ms, merkle_ms,
        tc_ms, setup_ms, teardown_ms, total_ops, failed_ops, path

    Args:
        bench_dir: Result tree run-ablation wrote (default bench-result/accelstore)
        csv_file: Summary CSV
        aggregate: How repeats of one arm combine — `median` (default) or
                   `latest`, the harness's usual latest-wins convention
        table: Also print the comparison tables (default True)
        jobs: The --jobs run-ablation ran with, if not the default

    Example:
        oma-arvio accelstore.parse-ablation
    """
    from modules.accelstore.tasks.results import ablation_results

    bench_path = Path(bench_dir)
    if not bench_path.exists():
        print(f"Error: Benchmark directory not found: {bench_dir}")
        return

    if aggregate not in ("median", "latest"):
        raise ValueError(f"aggregate must be 'median' or 'latest', got '{aggregate}'")

    points = ablation_results.discover_ablation_points(bench_path, ablation_jobs(jobs))
    if not points:
        print(f"No ablation runs found under {bench_dir}")
        return

    rows = ablation_results.relative_runtimes(points, aggregate)
    ablation_results.export_ablation_csv(rows, Path(csv_file))
    print(f"Wrote {len(rows)} rows to {csv_file}")
    if table:
        ablation_results.print_ablation_tables(rows)


@task(name="plot-ablation")
def plot_ablation(
    _ctx,
    bench_dir: str = "bench-result/accelstore",
    output_dir: str = "bench-result/plots",
    config: str = None,
    formats: str = "pdf",
    table: bool = True,
    jobs: Optional[str] = None,
):
    """The single-panel ablation figure.

    Reads the same result tree `parse-ablation` does and writes ablation.{fmt},
    a grouped bar chart of what each design property adds to end-to-end
    runtime, in percent of CVM-Store without any CIF. One figure per machine
    config, since the ratios are device properties; with several in the tree,
    `config` picks the one that keeps the plain filename and the others get
    their config appended.

    Args:
        bench_dir: Result tree run-ablation wrote (default bench-result/accelstore)
        output_dir: Where the figure goes (default bench-result/plots)
        config: Machine config for the unsuffixed file (default: the only one)
        formats: Comma-separated output formats (default pdf)
        table: Also print the figure's numbers as a table (default True)
        jobs: The --jobs run-ablation ran with, if not the default

    Example:
        oma-arvio accelstore.plot-ablation
    """
    from modules.accelstore.tasks.plots import ablation
    from modules.accelstore.tasks.results import ablation_results

    bench_path = Path(bench_dir)
    if not bench_path.exists():
        print(f"Error: Benchmark directory not found: {bench_dir}")
        return

    points = ablation_results.discover_ablation_points(bench_path, ablation_jobs(jobs))
    if not points:
        print(f"No ablation runs found under {bench_dir}")
        return

    rows = ablation_results.relative_runtimes(points)
    per_config = ablation.by_config(rows)
    if config is not None and config not in per_config:
        raise ValueError(
            f"No ablation rows for config '{config}'."
            f" Found: {', '.join(sorted(per_config))}"
        )
    plain = config or (sorted(per_config)[0] if len(per_config) == 1 else None)

    out_dir = Path(output_dir)
    if table:
        ablation.print_ablation_summary(rows)
    for name, config_rows in sorted(per_config.items()):
        stem = "ablation" + ("" if name == plain else f"-{name}")
        for fmt in formats.split(","):
            target = out_dir / f"{stem}.{fmt}"
            if ablation.plot_ablation(config_rows, target):
                print(f"Wrote {target}")


@task(name="parse-mlperf")
def parse_mlperf(
    _ctx,
    bench_dir: str = "bench-result/mlperf",
    out_dir: str = "bench-result/parsed",
    prefix: str = "mlperf",
    aggregate: str = "latest",
    table: bool = True,
):
    """Parse run-mlperf results into one CSV per benchmark family.

    Reads mlpstorage's own ``summary.json`` for every point under bench_dir and
    nothing else: a metric re-derived here would be a second definition of a
    published number.

    The checkpointing family's table opens with the identity prefix (config,
    family, division, variant, label, arm, arm_kind, module, stack, sync,
    model, status, run_ts, wall_s, hosts, image_id, device, path) and then:

        ranks, checkpoint_size_gb, save_gbs[_stdev], save_s[_stdev],
        load_gbs[_stdev], load_s[_stdev]

    Every point that ran keeps a row with the status it earned: `no-summary`
    for a point that produced nothing, `phase-failed:<phase>` from the run's
    own ``*_flags.json``, `no-metric`, and `shape-mismatch` where the arms of
    one variant disagree on the checkpoint size. No ratios: a row here is what
    mlpstorage measured.

    Args:
        bench_dir: Result tree run-mlperf wrote (default bench-result/mlperf)
        out_dir: Where the CSVs are written (default bench-result/parsed)
        prefix: CSV stem; each family appends its own name
        aggregate: `latest` (default, one row per point) or `all` (every run)
        table: Also print the per-family comparison tables (default True)

    Example:
        oma-arvio accelstore.parse-mlperf
        oma-arvio accelstore.parse-mlperf --bench-dir bench-result-mlperf/mlperf
    """
    from modules.accelstore.tasks.results import mlperf_results

    if aggregate not in ("latest", "all"):
        raise ValueError(f"aggregate must be 'latest' or 'all', got '{aggregate}'")

    bench_path = Path(bench_dir)
    if not bench_path.exists():
        print(f"Error: Benchmark directory not found: {bench_dir}")
        return

    by_family, warnings = mlperf_results.parse_mlperf_tree(
        bench_path, aggregate=aggregate
    )
    for warning in warnings:
        print(f"NOTE: {warning}")

    total = 0
    for family, rows in sorted(by_family.items()):
        if not rows:
            continue
        total += len(rows)
        csv_path = Path(out_dir) / f"{prefix}-{family}.csv"
        mlperf_results.export_family_csv(rows, family, csv_path)
        print(f"Wrote {len(rows)} {family} rows to {csv_path}")
        if table:
            mlperf_results.print_family_table(rows, family)
    if not total:
        print(f"No MLPerf results found under {bench_dir}")


@task(name="parse-fio-bandwidth")
def parse_fio_bandwidth(
    _ctx,
    bench_dir: str = "bench-result/fio",
    output_file: str = "bench-result/parsed/fio-bandwidth.csv",
):
    """Parse bandwidth FIO test results with guest SAR CPU/memory utilization.

    CSV columns:
        vm_type, size, storage_config, jobname,
        read_bw_mibs, write_bw_mibs, cpu_used_pct, mem_used_gib
    """
    from modules.fio.tasks.plotting.fio_parser import (
        correlate_with_sar,
        find_bandwidth_fio_files,
        parse_config_from_path,
        parse_fio_file,
    )

    bench_path = Path(bench_dir)
    if not bench_path.exists():
        print(f"Error: Benchmark directory not found: {bench_dir}")
        return

    print(f"Scanning for bandwidth FIO results in: {bench_dir}")
    bandwidth_files = find_bandwidth_fio_files(bench_path)

    if not bandwidth_files:
        print("No bandwidth test results found.")
        return

    rows = []
    header = [
        "vm_type",
        "size",
        "storage_config",
        "jobname",
        "read_bw_mibs",
        "write_bw_mibs",
        "cpu_used_pct",
        "mem_used_gib",
    ]

    for config_name, job_dirs in sorted(bandwidth_files.items()):
        for job_dir_name, json_path in sorted(job_dirs.items()):
            vm_type, size, _, _ = parse_config_from_path(json_path)

            storage_config = job_dir_name.split("_bandwidth")[0]

            timestamp = json_path.stem
            guest_sar = json_path.parent / f"{timestamp}_guest_sar.txt"

            try:
                fio_jobs = parse_fio_file(json_path)
            except Exception as e:
                print(f"  Warning: Failed to parse {json_path}: {e}")
                continue

            if guest_sar.exists():
                try:
                    correlate_with_sar(fio_jobs, guest_sar)
                except Exception as e:
                    print(f"  Warning: SAR correlation failed for {guest_sar}: {e}")
            else:
                print(f"  No guest SAR file: {guest_sar.name}")

            for jobname, d in fio_jobs.items():
                read_bw_mibs = d.read_bw_mean / 1024.0
                write_bw_mibs = d.write_bw_mean / 1024.0
                cpu_pct = f"{d.cpu_used_pct:.2f}" if d.cpu_used_pct is not None else ""
                mem_gib = (
                    f"{d.mem_used_kb / 1024 / 1024:.3f}"
                    if d.mem_used_kb is not None
                    else ""
                )
                rows.append(
                    [
                        vm_type,
                        size,
                        storage_config,
                        jobname,
                        f"{read_bw_mibs:.2f}",
                        f"{write_bw_mibs:.2f}",
                        cpu_pct,
                        mem_gib,
                    ]
                )

    if not rows:
        print("No data to write.")
        return

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to: {output_path}")


@task(name="plot-fio-iops")
def plot_fio_iops(ctx, output_dir="bench-result/plots"):
    """Generate IOPS comparison plot for VM, CVM, and Host with libaio and SPDK."""
    from modules.accelstore.tasks.plots import fio

    print("Generating VM comparison IOPS plot...")

    setups = [
        {
            "name": "Host (libaio)",
            "path": "bench-result/fio/host-medium/libaio_bs512",
            "color_label": "Host",
            "hatch_label": "libaio",
        },
        {
            "name": "Host (SPDK)",
            "path": "bench-result/fio/host-medium/spdk_bs512",
            "color_label": "Host",
            "hatch_label": "SPDK",
        },
        {
            "name": "VM (libaio)",
            "path": "bench-result/fio/amd-disk-medium/libaio_bs512",
            "color_label": "VM",
            "hatch_label": "libaio",
        },
        {
            "name": "VM (SPDK)",
            "path": "bench-result/fio/amd-disk-medium/spdk_bs512",
            "color_label": "VM",
            "hatch_label": "SPDK",
        },
        {
            "name": "CVM (libaio)",
            "path": "bench-result/fio/snp-disk-medium/libaio_bs512",
            "color_label": "CVM",
            "hatch_label": "libaio",
        },
        {
            "name": "CVM (SPDK)",
            "path": "bench-result/fio/snp-disk-medium/spdk_bs512",
            "color_label": "CVM",
            "hatch_label": "SPDK",
        },
    ]

    fio.generate_all_plots(
        setups=setups,
        output_dir=output_dir,
        output_name="fio-iops",
        formats=["pdf"],
        plot_types=["iops"],
    )

    print(f"VM comparison plot saved to {output_dir}/fio-iops_iops.*")


@task(name="plot-fio-luks")
def plot_fio_luks(ctx, output_dir="bench-result/plots"):
    """Generate IOPS comparison plot for LUKS encryption variants in VM/CVM."""
    from modules.accelstore.tasks.plots import fio

    print("Generating LUKS variants IOPS plot...")

    setups = [
        {
            "name": "VM (none)",
            "path": "bench-result/fio/amd-disk-medium/libaio_bs512",
            "color_label": "VM",
            "hatch_label": "none",
        },
        {
            "name": "CVM (none)",
            "path": "bench-result/fio/snp-disk-medium/libaio_bs512",
            "color_label": "CVM",
            "hatch_label": "none",
        },
        {
            "name": "VM (aes-xts)",
            "path": "bench-result/fio/amd-disk-medium/libaio-luks-ext4-aes_bs512",
            "color_label": "VM",
            "hatch_label": "AES-XTS",
        },
        {
            "name": "CVM (aes-xts)",
            "path": "bench-result/fio/snp-disk-medium/libaio-luks-ext4-aes_bs512",
            "color_label": "CVM",
            "hatch_label": "AES-XTS",
        },
        {
            "name": "VM (aes-xts+sha256-mac)",
            "path": "bench-result/fio/amd-disk-medium/libaio-luks-ext4-aes-xts_bs512",
            "color_label": "VM",
            "hatch_label": "AES-XTS+SHA256",
        },
        {
            "name": "CVM (aes-xts+sha256-mac)",
            "path": "bench-result/fio/snp-disk-medium/libaio-luks-ext4-aes-xts_bs512",
            "color_label": "CVM",
            "hatch_label": "AES-XTS+SHA256",
        },
        {
            "name": "VM (aegis128)",
            "path": "bench-result/fio/amd-disk-medium/libaio-luks-ext4-aegis128_bs512",
            "color_label": "VM",
            "hatch_label": "AEGIS128",
        },
        {
            "name": "CVM (aegis128)",
            "path": "bench-result/fio/snp-disk-medium/libaio-luks-ext4-aegis128_bs512",
            "color_label": "CVM",
            "hatch_label": "AEGIS128",
        },
    ]

    fio.generate_all_plots(
        setups=setups,
        output_dir=output_dir,
        output_name="fio-luks",
        formats=["pdf"],
        plot_types=["iops"],
    )

    print(f"LUKS comparison plot saved to {output_dir}/fio-luks_iops.*")


@task(name="plot-apps")
def plot_apps(
    ctx,
    results_root: str = ".",
    output: str = "bench-result/plots/apps.pdf",
    personalities: Optional[str] = None,
    redis_points: Optional[str] = None,
    config: Optional[str] = None,
    redis_config: Optional[str] = None,
    bench_config: Optional[str] = None,
    module: Optional[str] = None,
    height: Optional[float] = None,
    fill_missing: bool = False,
    error_bars: bool = False,
    png: bool = False,
):
    """Paper application figure: checkpointing + redis + personalities, one column.

    Which personalities and which redis points appear is `apps.PERSONALITIES`
    and `redis_panels.POINTS`; the two arguments override them for a one-off.
    Every redis point is drawn against ext4, so a point the ext4 arm did not
    run is dropped with a note — `run-redis --arms ext4` fills such a hole.

    Args:
        results_root: Checkout or result tree holding mlperf/, redis/ and bench/
        personalities: Comma-separated personality groups, in bar order
        redis_points: Comma-separated redis points, in bar order
        config: MLPerf {setup}-disk-{size} config (default: fullest one)
        redis_config: redis result config (default: the one with most cells)
        bench_config: bench result config (default: the one with the most runs)
        module: AccelStore module CVM-Store's MLPerf bar is drawn from
        height: Figure height in inches (default: the module's own)
        fill_missing: Mark cells with no run so the full layout can be judged
        error_bars: Add the spread whiskers to the MLPerf bars (off by default:
            see apps.ERROR_BARS for why)
        png: Also write a PNG next to the PDF
    """
    from modules.accelstore.tasks.plots import apps

    kwargs = {}
    if personalities:
        kwargs["personalities"] = [
            p.strip() for p in personalities.split(",") if p.strip()
        ]
    if redis_points:
        kwargs["redis_points"] = [
            p.strip() for p in redis_points.split(",") if p.strip()
        ]
    if height:
        kwargs["height_in"] = height
    ok = apps.generate(
        Path(results_root),
        Path(output),
        config=config,
        redis_config=redis_config,
        bench_config=bench_config,
        module=module,
        fill_missing=fill_missing,
        error_bars=apps.ERROR_BARS or error_bars,
        png=png,
        **kwargs,
    )
    if not ok:
        raise SystemExit(1)


@task(name="plot-synthetic")
def plot_synthetic(
    ctx,
    results_root: str = ".",
    output: str = "bench-result/plots/synthetic.pdf",
    value_sizes: str = "",
    threads: str = "",
    fill_missing: bool = False,
    png: bool = False,
    quick: bool = False,
):
    """Paper figure C: YCSB-A rel. ops/s beside its per-op GiB/s, the
    FS/LOG size sweep, and the FS random reads' IOPS — one page-wide
    three-panel stack over a shared key.

    Args:
        results_root: Checkout holding bench-result/ (default .)
        value_sizes: Panel (a)'s x axis, comma separated (default 4K..1G)
        threads: Thread counts shown as the hatch pair (default "1,8")
        fill_missing: Mark cells with no run so the full layout can be judged
        png: Also write a PNG next to the PDF
        quick: Draw the figure from the `paper.* --quick` leaves, over the
               reduced point set that suite runs
    """
    from modules.accelstore.tasks.plots import synthetic

    synthetic.generate(
        results_root=Path(results_root),
        output=Path(output),
        value_sizes=synthetic.parse_size_list(
            value_sizes
            or (synthetic.QUICK_VALUE_SIZES if quick else synthetic.DEFAULT_VALUE_SIZES)
        ),
        threads=[
            int(t)
            for t in (threads or synthetic.DEFAULT_THREADS).split(",")
            if t.strip()
        ],
        fill_missing=fill_missing,
        png=png,
        quick=quick,
    )


@task(name="parse-trace")
def parse_trace(
    ctx,
    json_file,
    bdev_block_size: int = None,
    output_json: str = None,
):
    """Parse and analyze a single AccelStore SPDK trace JSON file.

    Extracts performance metrics including encryption overhead, throughput,
    metadata operations, and trusted counter operations.
    """
    from modules.accelstore.tasks.results.trace_parser import (
        TraceParser,
        dump_metrics,
        save_metrics_to_json,
    )

    # int(): a None default leaves invoke no type to cast to.
    bdev_block_size = (
        TraceParser.BDEV_BLOCK_SIZE if bdev_block_size is None else int(bdev_block_size)
    )

    trace_path = Path(json_file)

    if not trace_path.exists():
        print(f"ERROR: File not found: {trace_path}")
        return

    try:
        parser = TraceParser(trace_path, bdev_block_size)
        metrics = parser.parse_and_calculate_metrics()
        dump_metrics(metrics)

        if output_json:
            output_path = Path(output_json)
            save_metrics_to_json(metrics, output_path)
            print(f"\nSaved parsed metrics to: {output_path}")

    except ValueError as e:
        print(f"ERROR: Invalid trace format in {trace_path}: {e}")
    except Exception as e:
        print(f"ERROR: Failed to parse trace: {e}")


AMPLIFICATION_COLUMNS = [
    "config",
    "client",
    "job",
    "variant",
    "crypto",
    "api_read_bytes",
    "api_write_bytes",
    "bdev_read_bytes",
    "bdev_write_bytes",
    "read_amplification",
    "write_amplification",
    "total_amplification",
    "trace",
]


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    """bdev bytes over api bytes, or None where nothing was asked for."""
    return numerator / denominator if denominator > 0 else None


def _amplification_rows(root: Path) -> list:
    """One row per leaf of a run-all tree, from its newest trace.

    A leaf with no trace dump (or cache) is not a row, and a trace that cannot
    be parsed is reported and skipped rather than aborting the table.
    """
    from modules.accelstore.tasks.results.discovery import (
        find_latest_traces,
        iter_point_leaves,
    )
    from modules.accelstore.tasks.results.trace_parser import load_or_parse_trace

    rows = []
    for config, client_binary, job, variant, leaf in iter_point_leaves(root):
        traces = find_latest_traces(leaf, 1)
        if not traces:
            continue
        trace = traces[0]
        try:
            metrics = load_or_parse_trace(trace)
        except Exception as exc:  # noqa: BLE001 - one bad dump is not the table
            print(f"  ⚠ {trace}: {exc}")
            continue
        client, backend = CryptoBackend.split_client_binary(client_binary)
        api, bdev = metrics.api, metrics.bdev
        rows.append(
            {
                "config": config,
                "client": client,
                "job": job,
                "variant": variant or "default",
                "crypto": backend.value,
                "api_read_bytes": api.read_bytes,
                "api_write_bytes": api.write_bytes,
                "bdev_read_bytes": bdev.read_bytes,
                "bdev_write_bytes": bdev.write_bytes,
                "read_amplification": _ratio(bdev.read_bytes, api.read_bytes),
                "write_amplification": _ratio(bdev.write_bytes, api.write_bytes),
                "total_amplification": _ratio(bdev.total_bytes, api.total_bytes),
                "trace": trace.name,
            }
        )
    return rows


def _print_amplification_table(rows: list) -> None:
    from modules.accelstore.tasks.results.tables import fmt

    print("\nRead/Write Amplification (bdev bytes / api bytes)")
    print("-" * 128)
    print(
        f"{'config':<18}{'client':<12}{'job':<36}{'variant':<30}{'crypto':<8}"
        f"{'read':>8}{'write':>8}{'total':>8}"
    )
    for r in rows:
        print(
            f"{r['config']:<18}{r['client']:<12}{r['job']:<36}{r['variant']:<30}"
            f"{r['crypto']:<8}"
            f"{fmt(r['read_amplification'], 3):>8}"
            f"{fmt(r['write_amplification'], 3):>8}"
            f"{fmt(r['total_amplification'], 3):>8}"
        )


@task(name="parse-amplification")
def parse_amplification(
    _ctx,
    bench_dir: str = "bench-result/accelstore",
    csv_file: str = "bench-result/parsed/amplification.csv",
    table: bool = True,
):
    """The read/write amplification table, from the traced run-all runs.

    Reads the newest `*_trace.dump.json` (through its `_parsed.json` cache) of
    every leaf under bench_dir — the job directory itself as the default sync
    mode, each `sync-…`/`abl-…`/`pipe-…` subdirectory as its own row — and
    writes the API bytes the client asked for, the bdev bytes the device saw,
    and their ratios per direction and in total.

    CSV, one row per traced leaf:
        config, client, job, variant, crypto, api_read_bytes, api_write_bytes,
        bdev_read_bytes, bdev_write_bytes, read_amplification,
        write_amplification, total_amplification, trace

    `variant` is `default` for the job directory itself. A ratio is blank
    where the client moved no bytes in that direction.

    Args:
        bench_dir: Result tree run-all wrote (default bench-result/accelstore)
        csv_file: Output CSV (default bench-result/parsed/amplification.csv)
        table: Also print the ratios as a table (default True)

    Example:
        oma-arvio accelstore.parse-amplification
        oma-arvio paper.amplification --print-only   # the runs this table reads
    """
    from modules.accelstore.tasks.results.tables import fmt, write_csv

    bench_path = Path(bench_dir)
    if not bench_path.exists():
        print(f"Error: Benchmark directory not found: {bench_dir}")
        return

    rows = _amplification_rows(bench_path)
    if not rows:
        print(f"No traced runs found under {bench_dir}")
        return

    csv_rows = [
        {
            **row,
            "read_amplification": fmt(row["read_amplification"], 3),
            "write_amplification": fmt(row["write_amplification"], 3),
            "total_amplification": fmt(row["total_amplification"], 3),
        }
        for row in rows
    ]
    write_csv(Path(csv_file), AMPLIFICATION_COLUMNS, csv_rows)
    if table:
        _print_amplification_table(rows)


@task(name="plot-epoch")
def plot_epoch(
    _ctx,
    output_dir: str = None,
    formats: str = "pdf",
    average_runs: int = None,
):
    """Metadata epoch figure and tables.

    Reads what run-epoch-sweep left in bench-result/accelstore and writes
    epoch-overhead.{fmt} (overhead vs the durability window), plus the
    overhead grid as a console table. Every point is the mean of its
    `average_runs` most recent runs; a point with fewer runs than asked for is
    the mean of the ones it has.

    Args:
        output_dir: Output directory (default: bench-result/plots)
        formats: Comma-separated output formats
        average_runs: Most recent runs to average per sweep point
            (default: sync_overhead.EPOCH_AVERAGE_RUNS_DEFAULT, 2)
    """
    from modules.accelstore.tasks.plots import sync_overhead

    out_dir = Path(output_dir) if output_dir else Path("bench-result/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt_list = formats.split(",")

    workloads = dict(ACCELSTORE_EPOCH_JOBS)

    sync_data = sync_overhead.discover_sync_metrics(
        bench_dir=Path("bench-result/accelstore"),
        sweep_workloads=workloads,
        # int(): a None default leaves invoke no type to cast to.
        average_runs=(
            sync_overhead.EPOCH_AVERAGE_RUNS_DEFAULT
            if average_runs is None
            else int(average_runs)
        ),
    )

    sync_overhead.print_epoch_table(sync_data)

    for fmt in fmt_list:
        sync_overhead.plot_epoch_overhead(
            sync_data,
            output_file=out_dir / f"epoch-overhead.{fmt}",
            axis="every-t",
        )
