#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""bench_client runs: one DSL personality on one backend.

The same .wl file runs either on ``accelstore`` (the library inside the usual
SPDK lifecycle, one run per crypto build variant) or on a block stack, as
``bench_client --backend posix --dir <path>`` against a directory set up by
``actions/posix_storage.py`` from a stack in ``tasks/storage_stacks.py``.

Both arms need a fresh target per run: bench_client models an empty fileset at
startup, so a kept store or non-empty directory fails prealloc with -EEXIST.

Every run records its mechanism alongside the number: ``--per-worker --json``
on both arms, the accelstore arm optionally wrapped in the
``bench_rpc_sample.py`` sampler, and a ``*_flags.json`` sidecar carrying the
resolved flag set, the arm's acknowledgement semantics and the personality's
read geometry.
"""

import json
import shlex
import subprocess
from pathlib import Path
from typing import Optional

from core.tasks.actions import ActionContext
from core.tasks.actions.registry import register_action

from modules.accelstore.tasks import personality_shape
from modules.accelstore.tasks.pipeline_opts import PipelineOpts
from modules.accelstore.tasks.actions.accelstore import (
    DEFAULT_TRACE_ENTRIES,
    _run_client_with_tracing,
    _save_output_to_file,
)
from modules.accelstore.tasks.actions.posix_storage import (
    POSIX_DEFAULT_BACKENDS,
    posix_stack,
)
from modules.accelstore.tasks.actions.storage_env import (
    precondition_device,
    spdk_device,
)
from modules.accelstore.tasks.crypto_backend import CryptoBackend
from modules.accelstore.tasks.storage_stacks import BLOCK_STACKS
from modules.accelstore.tasks.sync_modes import SyncMode

BENCH_CLIENT = "bench_client"

BENCH_SAMPLER = "bench_rpc_sample.py"

# Passed explicitly rather than left to the client's default: the sampler has
# to name the same path.
BENCH_RPC_SOCKET = "/var/tmp/bench_client.sock"

BENCH_BACKENDS = ("accelstore", *BLOCK_STACKS)

BENCH_DEFAULT_BACKENDS = ("accelstore", *POSIX_DEFAULT_BACKENDS)

# On by default: a buffered POSIX arm would serve the fileset from the guest
# page cache, which the SPDK arm has no equivalent of.
DEFAULT_DIRECT = True

# bench_client's own `--io-chunk` default, and the floor the matched geometry
# may never go under: a smaller segment only handicaps the baseline.
POSIX_DEFAULT_IO_CHUNK = 1 << 20

POSIX_FSYNC_MODES = ("none", "per-op", "close")
POSIX_BARRIER_MODES = ("dirty-fsync", "syncfs", "none")
POSIX_LOG_LAYOUTS = ("files", "single")
DEFAULT_FSYNC_MODE = "none"
DEFAULT_BARRIER_MODE = "dirty-fsync"
DEFAULT_LOG_LAYOUT = "files"

ACCELSTORE_ACK = {
    "summary": "device",
    "detail": (
        "the data of an acknowledged operation has reached the device; its "
        "metadata durability point is whatever --sync-* configured (default: "
        "the epoch at as_free(), plus the module's own space threshold)"
    ),
}

ACK_SEMANTICS = {
    "accelstore": ACCELSTORE_ACK,
    **{stack.id: stack.ack_record() for stack in BLOCK_STACKS.values()},
}


def _resolve_personality(vm, binary: str, personality: str) -> str:
    """Return the guest path of a personality: verbatim if it has a slash, else
    resolved against the client package's installed .wl files."""
    if "/" in personality:
        return personality
    name = personality if personality.endswith(".wl") else f"{personality}.wl"
    result = vm.ssh_cmd(
        ["sh", "-c", f'dirname "$(readlink -f "$(command -v {binary})")"'],
        check=True,
    )
    bindir = result.stdout.strip()
    if not bindir:
        raise RuntimeError(f"Cannot locate {binary} in the guest PATH")
    return f"{bindir}/../share/accelstore/workloads/personalities/{name}"


def _bench_path(name: str, cfg: dict) -> tuple:
    backend = cfg.get("backend", "accelstore")
    if backend == "accelstore":
        crypto = CryptoBackend.parse(cfg.get("crypto") or CryptoBackend.AEGIS)
        arm = crypto.client_binary(BENCH_CLIENT)
    else:
        arm = backend
    components = ["bench", name, arm, Path(cfg.get("personality", "")).stem]
    # Folded into one component: the results parser reads exactly one directory
    # level under the personality as the variant leaf.
    tokens = []
    leaf = SyncMode.from_config(cfg).path_component()
    if leaf:
        tokens.append(leaf)

    threads = cfg.get("threads")
    if threads is not None:
        # Every override, `--threads 1` included: a forced-1 run must not be
        # label-identical to a default run.
        tokens.append(f"threads-{int(threads)}")

    if cfg.get("split_loops"):
        tokens.append("split-loops")

    depth = int(cfg.get("intra_op_depth") or 1)
    if depth > 1:
        # The requested depth, not the resolved one: the path is named before
        # the personality it resolves against is read.
        tokens.append(f"intra-op-{depth}")

    if backend == "accelstore":
        # Only an explicit chunk lands in the path: the derived one is constant
        # per personality, an explicit one is a sweep axis.
        chunk = cfg.get("chunk_size")
        if chunk:
            tokens.append(f"chunk-{personality_shape.format_bytes(int(chunk))}")
        # include_chunk=False: the line above already labelled it, on the
        # explicit-only terms this leaf wants.
        tokens.extend(PipelineOpts.from_config(cfg).path_tokens(include_chunk=False))
    else:
        # Tokenized although it is the default: rows taken before it became one
        # sit in leaves without the token.
        tokens.append("direct" if cfg.get("direct", DEFAULT_DIRECT) else "buffered")
        for key, default in (
            ("fsync_mode", DEFAULT_FSYNC_MODE),
            ("barrier_mode", DEFAULT_BARRIER_MODE),
            ("log_layout", DEFAULT_LOG_LAYOUT),
        ):
            value = cfg.get(key) or default
            if value != default:
                tokens.append(f"{key.replace('_', '-')}-{value}")
        if cfg.get("match_io_depth"):
            # A marker rather than the number: this function has no `shape` and
            # so cannot see the depth a derived chunk resolves to.
            tokens.append("qd-matched")

    if cfg.get("quick"):
        tokens.append("quick")

    if tokens:
        components.append("_".join(tokens))
    return tuple(components)


@register_action("bench", path_fn=_bench_path)
def run_bench(
    ctx: ActionContext,
    personality: str,
    backend: str = "accelstore",
    crypto: str = CryptoBackend.AEGIS.value,
    config: str = None,
    pci_device: str = None,
    dev_path: str = None,
    device: str = "Nvme1n1",
    tracing: Optional[str] = None,
    trace_entries: Optional[int] = DEFAULT_TRACE_ENTRIES,
    skip_precondition: bool = False,
    precondition_scheme: str = None,
    bench_args: Optional[str] = None,
    threads: Optional[int] = None,
    split_loops: bool = False,
    per_worker: bool = True,
    direct: bool = DEFAULT_DIRECT,
    fsync_mode: Optional[str] = None,
    barrier_mode: Optional[str] = None,
    log_layout: Optional[str] = None,
    chunk_size: Optional[int] = None,
    derive_chunk_size: bool = True,
    intra_op_depth: int = 1,
    match_io_depth: bool = False,
    pipeline_bytes: Optional[int] = None,
    io_depth: Optional[int] = None,
    reorder_window: Optional[int] = None,
    burst_bytes: Optional[str] = None,
    burst_adaptive: bool = False,
    io_span_bytes: Optional[int] = None,
    cluster_size: Optional[int] = None,
    md_pages: Optional[int] = None,
    derive_store_geometry: bool = False,
    sampler: bool = False,
    sampler_interval: float = 1.0,
    sampler_qd_period_us: int = 0,
    quick: bool = False,
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
    """Run one bench_client personality on one backend.

    Args:
        ctx: Action context (VM runner, output dirs, timestamp)
        personality: Personality name (varmail_log, webserver_fs, ...) resolved
                     against the client package's installed .wl files, or a
                     guest path containing a slash, used verbatim
        backend: "accelstore" or a block stack id from
                 storage_stacks.BLOCK_STACKS
        crypto: Crypto build variant of the accelstore arm. A POSIX arm always
                runs the default build's binary
        config: SPDK bdev configuration as JSON string (accelstore arm, required)
        pci_device: PCI address the bdev config attaches (accelstore arm,
                 required)
        dev_path: Guest block device path — pre-conditioning target on both
                  arms, and what a POSIX arm formats (required there)
        device: SPDK block device name created by the bdev config
        tracing: SPDK trace recording, accelstore arm only ("tracing",
                 "small-trace", None)
        trace_entries: Trace ring entries per lcore, a power of two in
                 [32768, 67108864]
        skip_precondition: Skip secure erase + pre-conditioning
        precondition_scheme: Pre-conditioning scheme to use
        bench_args: Extra bench_client arguments appended verbatim on every arm
        threads: --threads on BOTH arms, 1..64 (the accelstore arm caps at 63).
                 None runs the personality's own `threads` declaration, and any
                 explicit override — 1 included — gets its own result leaf
        split_loops: Pass --split-loops on BOTH arms, so an iteration bound is
                 the run's total rather than a per-worker count. It is the
                 identity at one worker and gets its own variant token
        per_worker: Pass --per-worker (default True), adding the per-worker rows
                 to the log as well as the --json sidecar
        direct: POSIX arms only, default True — O_DIRECT data I/O, so the arm
                 measures the device rather than the guest page cache. False is
                 refused on the accelstore arm
        fsync_mode: POSIX arms only, one of POSIX_FSYNC_MODES (default "none")
        barrier_mode: POSIX arms only, one of POSIX_BARRIER_MODES (default
                 "dirty-fsync", forced to "none" for a personality that places
                 no `fsync` flowop)
        log_layout: POSIX arms only, one of POSIX_LOG_LAYOUTS (default "files")
        chunk_size: --crypto-chunk-size for the accelstore arm. None derives it
                 from the personality's read geometry (see derive_chunk_size).
                 An explicit value that amplifies the personality's ranged reads
                 is refused
        derive_chunk_size: Derive the chunk size from the personality when
                 chunk_size is None (default True). False runs the library
                 default, which is how the amplification is measured on purpose
        intra_op_depth: Cut one operation into N overlapping pieces on BOTH arms
                 (default 1 = no flag on either side): the accelstore arm gets a
                 crypto chunk of `derived / N`, the POSIX arm `--io-chunk
                 <derived/N> --io-depth N`. It resolves against the personality
                 (power of two, floored by AccelStore's 4096 B chunk minimum),
                 and is refused beside chunk_size or derive_chunk_size=False.
                 NOT io_depth, which is the library's chunk-pipeline command
                 depth. Depth > 1 lands in an `intra-op-N` variant leaf
        match_io_depth: Give the POSIX arm the accelstore arm's in-flight byte
                 budget (--io-chunk/--io-depth) instead of bench_client's
                 depth-1 default, so a read row does not compare an overlapped
                 arm against a serialized one. Default False, lands in a
                 `qd-matched` leaf, and is refused beside intra_op_depth > 1
        pipeline_bytes .. md_pages: The rest of the chunk-pipeline geometry,
                 accelstore arm only — one PipelineOpts with the resolved chunk
                 size (modules/accelstore/tasks/pipeline_opts.py). None of them
                 is derived, so each one that is set gets a token in the leaf
        derive_store_geometry: Derive chunk/cluster_size/md_pages from the
                 personality's object size (default False, accelstore arm only).
                 Explicitly passed values still win
        sampler: Wrap the accelstore arm in bench_rpc_sample.py (the S1
                 sampler) and write its per-interval CSV next to the log.
                 Needs the sampler in the guest image, and is refused with
                 tracing
        sampler_interval: Sampler poll interval in seconds (default 1.0)
        sampler_qd_period_us: bdev queue-depth sampling period (default 0 = off,
                 since it installs a poller inside the process being measured)
        quick: Label the row a `paper.* --quick` one. It changes nothing the
                 client does, only the result leaf the row lands in
        trigger .. tc_rtt_interval: One SyncMode, accelstore arm only — a
                 non-default mode is refused on a POSIX arm rather than dropped

    Output structure:
        bench-result/bench/{name}/{crypto-prefixed bench_client | backend}/{personality}/[variant]/{timestamp}.log
                                                                                                  /{timestamp}.json
                                                                                                  /{timestamp}_flags.json
                                                                                                  /{timestamp}_sample.csv

    ``.json`` is bench_client's own metrics sidecar (--json), ``_flags.json``
    this action's record of the resolved flag set, acknowledgement semantics and
    read geometry, ``_sample.csv`` the sampler's CSV when ``sampler`` was set.
    """
    if backend not in BENCH_BACKENDS:
        raise ValueError(
            f"Invalid backend '{backend}'. Must be one of: {', '.join(BENCH_BACKENDS)}"
        )
    is_accelstore = backend == "accelstore"

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
    if not is_accelstore and not sync.is_default():
        raise ValueError(
            f"Sync mode '{sync.label()}' has no meaning on the '{backend}' backend"
        )
    for warning in sync.warnings():
        print(f"WARNING: {warning}")

    posix_opts = {
        "fsync_mode": (fsync_mode, POSIX_FSYNC_MODES, DEFAULT_FSYNC_MODE),
        "barrier_mode": (barrier_mode, POSIX_BARRIER_MODES, DEFAULT_BARRIER_MODE),
        "log_layout": (log_layout, POSIX_LOG_LAYOUTS, DEFAULT_LOG_LAYOUT),
    }
    for name, (value, valid, _default) in posix_opts.items():
        if value is None:
            continue
        if value not in valid:
            raise ValueError(
                f"Invalid {name} '{value}'. Must be one of: {', '.join(valid)}"
            )
        if is_accelstore:
            raise ValueError(
                f"{name} is a POSIX-arm knob; the accelstore arm's durability is "
                "the --sync-* policy"
            )
    if is_accelstore and not direct:
        raise ValueError(
            "direct is a POSIX-arm knob (O_DIRECT); the accelstore arm never "
            "goes through the guest page cache"
        )

    if tracing is not None:
        if not is_accelstore:
            raise ValueError("SPDK tracing only exists on the accelstore backend")
        if tracing not in ["tracing", "small-trace"]:
            raise ValueError(
                f"Invalid tracing mode '{tracing}'. Must be 'tracing', 'small-trace', or None"
            )

    if trace_entries is not None:
        trace_entries = int(trace_entries)
        if not 32768 <= trace_entries <= 67108864:
            raise ValueError(
                f"trace_entries {trace_entries} out of range [32768, 67108864]"
            )
        if trace_entries & (trace_entries - 1):
            raise ValueError(f"trace_entries {trace_entries} is not a power of two")

    if sampler:
        if not is_accelstore:
            raise ValueError(
                "the RPC sampler reads an SPDK application's socket; a POSIX arm "
                "runs no SPDK app. Its mechanism column comes from --per-worker "
                "--json and the guest's SAR capture"
            )
        if tracing:
            raise ValueError(
                "sampler and tracing both measure where the run's time went, at "
                "very different cost; run them separately so neither pays for "
                "the other"
            )
        if sampler_interval <= 0:
            raise ValueError("sampler_interval must be positive")

    if threads is not None:
        threads = int(threads)
        if not 1 <= threads <= 64:
            raise ValueError(f"threads {threads} out of range [1, 64]")
        if threads > 63 and is_accelstore:
            # Refused here rather than by the client, after the VM booted and
            # the device was erased.
            raise ValueError(
                f"threads {threads} exceeds the accelstore arm's 63 workers"
            )

    intra_op_depth = int(intra_op_depth)
    # Shape None checks the operand alone, before the device is erased.
    personality_shape.derive_intra_op(None, intra_op_depth)
    if intra_op_depth != 1:
        if chunk_size is not None:
            raise ValueError(
                "chunk_size and intra_op_depth both set the accelstore arm's "
                "--crypto-chunk-size: the depth IS the chunk, expressed as a "
                "division of the one the personality derives. Pass one or the "
                "other"
            )
        if not derive_chunk_size:
            raise ValueError(
                "intra_op_depth divides the chunk the read geometry derives, so "
                "there is nothing to divide with derive_chunk_size=False"
            )

    extra_args = shlex.split(bench_args) if bench_args else []
    if threads is not None:
        extra_args += ["--threads", str(threads)]
    if split_loops:
        extra_args.append("--split-loops")
    if per_worker:
        extra_args.append("--per-worker")

    vm = ctx.vm
    date = ctx.timestamp
    output_log = ctx.outputdir_guest / f"{date}.log"
    extra_args += ["--json", str(ctx.outputdir_guest / f"{date}.json")]

    client_binary = (
        CryptoBackend.parse(crypto).client_binary(BENCH_CLIENT)
        if is_accelstore
        # Unprefixed: the posix backend is raw syscalls in every build.
        else BENCH_CLIENT
    )
    wl_path = _resolve_personality(vm, client_binary, personality)

    # Read before the device is preconditioned: a run refused for an
    # amplifying chunk should leave it intact.
    shape = _read_personality_shape(vm, wl_path)
    if split_loops:
        workers = threads if threads is not None else (shape.threads if shape else 1)
        if workers == 1:
            print(
                "NOTE: --split-loops over one worker is the identity; this row "
                "measures what the unsplit one does, in its own result leaf"
            )

    split = personality_shape.derive_intra_op(shape, intra_op_depth)
    if intra_op_depth != 1:
        # The leaf carries the requested depth, so the resolved one is said out
        # loud here and recorded in the sidecar below.
        print(f"NOTE: intra-op depth {intra_op_depth} -> {split.reason}")

    chunk_bytes, chunk_reason = _resolve_chunk_size(
        shape, chunk_size, derive_chunk_size, is_accelstore, split
    )
    # Opt-in: deriving it by default would re-geometry stores whose rows were
    # taken at the library defaults.
    chunk_bytes, cluster_size, md_pages, geometry_reason = _resolve_store_geometry(
        shape,
        derive_store_geometry and is_accelstore,
        chunk_bytes,
        chunk_size,
        cluster_size,
        md_pages,
        threads,
        split_loops,
    )
    if geometry_reason:
        chunk_reason = geometry_reason
    barrier_mode, barrier_reason = _resolve_barrier_mode(
        shape, barrier_mode, is_accelstore
    )
    if barrier_reason:
        print(f"NOTE: --barrier-mode none: {barrier_reason}")
        posix_opts["barrier_mode"] = (
            barrier_mode,
            POSIX_BARRIER_MODES,
            DEFAULT_BARRIER_MODE,
        )
    # with_chunk_size wins: the resolved chunk reaches the command line, not
    # the argument.
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
    ).with_chunk_size(chunk_bytes)
    pipe.validate()
    for warning in pipe.warnings():
        print(f"WARNING: {warning}")

    precondition_device(
        ctx, vm, dev_path, skip=skip_precondition, scheme=precondition_scheme
    )

    record = {
        "schema": 1,
        "timestamp": date,
        "personality": personality,
        "personality_path": wl_path,
        "personality_shape": shape.as_dict() if shape else None,
        "backend": backend,
        "arm": client_binary if is_accelstore else backend,
        "threads_override": threads,
        "split_loops": split_loops,
        "per_worker": per_worker,
        "ack_semantics": ACK_SEMANTICS.get(backend),
        "extra_args": extra_args,
    }
    if intra_op_depth != 1:
        # Only the vocabulary this arm spelled the split in: an --io-depth
        # beside an accelstore row would read as a command depth it never set.
        record["intra_op"] = split.as_dict()
        record["intra_op"].update(
            {"io_chunk": None, "io_depth": None}
            if is_accelstore
            else {"chunk_size": None}
        )

    if is_accelstore:
        record.update(
            {
                "crypto": crypto,
                "device": device,
                "pci_device": pci_device,
                "sync_mode": sync.label(),
                "sync_args": sync.client_args(),
                "crypto_chunk_size": chunk_bytes,
                "crypto_chunk_reason": chunk_reason,
                "pipeline_opts": pipe.as_dict(),
                "pipeline_args": pipe.client_args(),
                "tracing": tracing,
                "sampler": (
                    {
                        "binary": CryptoBackend.parse(crypto).client_binary(
                            BENCH_SAMPLER
                        ),
                        "interval_s": sampler_interval,
                        "qd_period_us": sampler_qd_period_us,
                        "socket": BENCH_RPC_SOCKET,
                    }
                    if sampler
                    else None
                ),
            }
        )
        cmd = _run_accelstore_arm(
            ctx,
            wl_path,
            client_binary,
            crypto,
            config,
            pci_device,
            device,
            tracing,
            trace_entries,
            extra_args,
            sync,
            pipe,
            sampler,
            sampler_interval,
            sampler_qd_period_us,
            output_log,
        )
    else:
        resolved_posix = {
            name: (value if value is not None else default)
            for name, (value, _valid, default) in posix_opts.items()
        }
        if match_io_depth and not split.is_identity:
            raise ValueError(
                "match_io_depth and intra_op_depth both set the POSIX arm's "
                "--io-chunk/--io-depth: the axis cuts the operation on both "
                "arms and names its own depth, matching leaves the cut alone "
                "and takes the accelstore arm's. Pick one."
            )
        if match_io_depth:
            io_chunk, io_depth = _matched_posix_geometry(
                shape,
                chunk_size,
                derive_chunk_size,
                derive_store_geometry,
                cluster_size,
                md_pages,
                threads,
                split_loops,
                split,
                pipe,
            )
            resolved_posix["io_depth"] = io_depth
            resolved_posix["io_chunk"] = io_chunk
            record["match_io_depth"] = True
            print(
                f"NOTE: posix arm at --io-depth {io_depth} --io-chunk"
                f" {io_chunk} = {io_depth * io_chunk >> 20} MiB in flight, the"
                " accelstore arm's own budget. The SEGMENT is"
                f" the baseline's own, floored at {POSIX_DEFAULT_IO_CHUNK >> 20}"
                " MiB: the match adds overlap, it does not impose a crypto chunk"
            )
        record.update({"direct": direct, **resolved_posix})
        if barrier_reason:
            record["barrier_mode_reason"] = barrier_reason
        cmd = _run_posix_arm(
            ctx,
            wl_path,
            backend,
            dev_path,
            extra_args,
            direct,
            resolved_posix,
            split,
            output_log,
        )

    record["command"] = cmd
    _write_flags_sidecar(ctx, record)


def _read_personality_shape(vm, wl_path: str):
    """Parse the .wl file that will actually run, read out of the guest."""
    result = vm.ssh_cmd(["cat", wl_path], check=False)
    if result.returncode != 0 or not result.stdout.strip():
        print(
            f"WARNING: cannot read {wl_path} in the guest; the run keeps the "
            "library's chunk size and its flags sidecar carries no read geometry"
        )
        return None
    return personality_shape.parse_personality(result.stdout)


def _matched_posix_geometry(
    shape,
    chunk_size,
    derive_chunk_size,
    derive_store_geometry,
    cluster_size,
    md_pages,
    threads,
    split_loops,
    split,
    pipe,
):
    """Return (--io-chunk, --io-depth) handing the POSIX arm the accelstore
    arm's in-flight byte budget, at a segment size of the baseline's own."""
    chunk, _why = _resolve_chunk_size(shape, chunk_size, derive_chunk_size, True, split)
    chunk, _cluster, _md, _geometry_why = _resolve_store_geometry(
        shape,
        derive_store_geometry,
        chunk,
        chunk_size,
        cluster_size,
        md_pages,
        threads,
        split_loops,
    )
    # The depth comes from the floored chunk, so what matches across the arms
    # is the in-flight byte budget rather than the segment size.
    matched = pipe.with_chunk_size(chunk)
    io_chunk = max(matched.effective_chunk_size(), POSIX_DEFAULT_IO_CHUNK)
    io_depth = pipe.with_chunk_size(io_chunk).effective_io_depth()
    return io_chunk, io_depth


def _resolve_chunk_size(shape, chunk_size, derive, is_accelstore, split=None):
    """(chunk bytes or None, why) for this run's --crypto-chunk-size."""
    if not is_accelstore:
        return None, "POSIX arm: AccelStore's chunk size is not on this path"
    if chunk_size is not None:
        chunk_size = int(chunk_size)
        problem = personality_shape.check_chunk(shape, chunk_size) if shape else None
        if problem:
            raise ValueError(
                f"--chunk-size {chunk_size} amplifies this personality's reads: "
                f"{problem}. Drop it to derive the chunk from the personality, or "
                "set derive_chunk_size=False to measure the amplification on purpose"
            )
        return chunk_size, "explicit"
    if not derive or shape is None:
        return None, "library default (derivation off)"
    if split is not None and not split.is_identity:
        # The intra-op split is itself a chunk derivation: the derived chunk
        # divided by the depth.
        return split.chunk_size, split.reason
    return personality_shape.derive_chunk(shape)


def _resolve_store_geometry(
    shape,
    derive,
    chunk_bytes,
    explicit_chunk,
    cluster_size,
    md_pages,
    threads,
    split_loops=False,
):
    """Fill unset chunk/cluster/md_pages from the personality's object size,
    plus a chunk reason to override with or None to keep the read-side one."""
    if not derive or shape is None:
        return chunk_bytes, cluster_size, md_pages, None

    geom, why = personality_shape.derive_store_geometry(shape, threads, split_loops)
    if geom is None:
        print(f"WARNING: no small-object geometry derived: {why}")
        return chunk_bytes, cluster_size, md_pages, None

    if explicit_chunk is None:
        chunk_bytes = geom.chunk_size
    if cluster_size is None:
        cluster_size = geom.cluster_size
    if md_pages is None:
        md_pages = geom.md_pages
    return chunk_bytes, cluster_size, md_pages, f"store-geometry: {geom.reason}"


def _resolve_barrier_mode(shape, barrier_mode, is_accelstore):
    """Force `--barrier-mode none` on a POSIX arm whose personality places no
    `fsync` flowop, where the dirty-fsync default dominates the write phase."""
    if is_accelstore or barrier_mode is not None or shape is None:
        return barrier_mode, None
    if shape.has_barrier:
        return barrier_mode, None
    return "none", (
        "personality places no `fsync` flowop, so there is no durability point"
        " for dirty-fsync to defer to"
    )


def _write_flags_sidecar(ctx, record: dict) -> None:
    """Write the run's resolved flag set next to the row it describes."""
    path = ctx.outputdir_host / f"{ctx.timestamp}_flags.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"Recorded the arm's flag set in {path}")


def _run_accelstore_arm(
    ctx,
    wl_path,
    client_binary,
    crypto,
    config,
    pci_device,
    device,
    tracing,
    trace_entries,
    extra_args,
    sync,
    pipe,
    sampler,
    sampler_interval,
    sampler_qd_period_us,
    output_log,
):
    if not config or not pci_device:
        raise ValueError("config and pci_device are required on the accelstore backend")

    try:
        json.loads(config)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON config: {e}")

    vm = ctx.vm

    # spdk_device destroys the blobstore: bench_client refuses a kept fileset
    # with -EEXIST on prealloc.
    with spdk_device(vm, config, pci_device, device) as config_file:
        cmd = [
            client_binary,
            wl_path,
            "--backend",
            "accelstore",
            "--config",
            config_file,
            "--device",
            device,
            "--rpc-socket",
            BENCH_RPC_SOCKET,
        ]
        cmd.extend(pipe.client_args())
        cmd.extend(sync.client_args())
        cmd.extend(extra_args)

        if tracing:
            trace_output = ctx.outputdir_guest / f"{ctx.timestamp}_trace.dump"
            print(f"Running {client_binary} ({wl_path}) with {tracing} enabled")
            cmd.append(f"--{tracing}")
            if trace_entries is not None:
                cmd.extend(["--trace-entries", str(trace_entries)])
            _run_client_with_tracing(vm, cmd, output_log, trace_output, BENCH_CLIENT)
            return cmd

        if sampler:
            cmd = _wrap_in_sampler(
                ctx,
                vm,
                crypto,
                cmd,
                sampler_interval,
                sampler_qd_period_us,
            )
        print(f"Running {client_binary} ({wl_path})")
        result = vm.ssh_cmd(cmd, check=False, stderr=subprocess.PIPE)
        _save_output_to_file(
            vm, output_log, result.stdout, result.stderr, result.returncode
        )
        return cmd


def _wrap_in_sampler(ctx, vm, crypto, cmd, interval, qd_period_us):
    """Wrap the client command in the S1 sampler, which spawns it, polls its RPC
    socket and inherits its stdout and exit status."""
    sampler_binary = CryptoBackend.parse(crypto).client_binary(BENCH_SAMPLER)
    probe = vm.ssh_cmd(["sh", "-c", f"command -v {sampler_binary}"], check=False)
    if probe.returncode != 0:
        raise RuntimeError(
            f"{sampler_binary} is not in the guest PATH. It ships with AccelStore "
            "(benchmarking/bench_rpc_sample.py, installed into bin/); build "
            "the guest image from an AccelStore revision that carries it, or run "
            "with --sampler false and lose the mechanism column"
        )
    sample_csv = ctx.outputdir_guest / f"{ctx.timestamp}_sample.csv"
    wrapper = [
        sampler_binary,
        "-s",
        BENCH_RPC_SOCKET,
        "-o",
        str(sample_csv),
        "-i",
        str(interval),
        # Only has to cover DPDK init on a 32 GB hugepage reserve.
        "--wait-for-socket",
        "120",
    ]
    if qd_period_us:
        wrapper += ["--qd-period-us", str(int(qd_period_us))]
    print(f"Sampling the run into {sample_csv} every {interval}s")
    return wrapper + ["--"] + cmd


def _run_posix_arm(
    ctx, wl_path, backend, dev_path, extra_args, direct, posix_opts, split, output_log
):
    if not dev_path:
        raise ValueError(f"dev_path is required on the '{backend}' backend")

    vm = ctx.vm

    with posix_stack(vm, backend, dev_path) as data_dir:
        cmd = [BENCH_CLIENT, wl_path, "--backend", "posix", "--dir", data_dir]
        # Spelled out even at their defaults: this command line is the record
        # of what the arm ran.
        cmd += ["--direct"] if direct else ["--no-direct"]
        cmd += ["--fsync-mode", posix_opts["fsync_mode"]]
        cmd += ["--barrier-mode", posix_opts["barrier_mode"]]
        cmd += ["--log-layout", posix_opts["log_layout"]]
        if not split.is_identity:
            cmd += ["--io-chunk", str(split.io_chunk)]
            cmd += ["--io-depth", str(split.io_depth)]
        elif posix_opts.get("io_depth"):
            # match_io_depth: the accelstore arm's in-flight geometry instead
            # of bench_client's depth-1 default.
            cmd += ["--io-chunk", str(posix_opts["io_chunk"])]
            cmd += ["--io-depth", str(posix_opts["io_depth"])]
        cmd.extend(extra_args)

        print(f"Running {BENCH_CLIENT} ({wl_path}) on {backend} at {data_dir}")
        result = vm.ssh_cmd(cmd, check=False, stderr=subprocess.PIPE)
        _save_output_to_file(
            vm, output_log, result.stdout, result.stderr, result.returncode
        )
        return cmd
