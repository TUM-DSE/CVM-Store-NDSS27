#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parsing for run-bench results (bench_client).

Numbers come from the ``--json`` sidecar where one exists and from the
metrics printer's log lines otherwise, under::

    bench-result/bench/{setup}-disk-{size}/{arm}/{personality}/[variant]/{ts}.{log,json}
"""

import json
import re
import shlex
import warnings
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from core.tasks.plotting import common

from ..crypto_backend import CryptoBackend
from . import discovery, tables
from ..plots.style import crypto_color

BENCH_CLIENT = "bench_client"

# Slots 0-2 belong to the accelstore build variants (plots/style.py).
POSIX_ARM_COLORS = {
    "ext4": common.PALETTE_PASTEL[3],
    "ext4-aes-xts-plain": common.PALETTE_PASTEL[6],
    "rollbaccine": common.PALETTE_PASTEL[4],
}
POSIX_ARM_HATCHES = {
    "ext4": "\\\\",
    "ext4-aes-xts-plain": "--",
    "rollbaccine": "...",
}

# Busy-or-pacing share of the wall clock below which a run is marked as waiting.
MIN_WORKER_OCCUPANCY = 0.5

UNDEROCCUPIED_MARK = "*"
FAILED_OPS_MARK = "!"


_MARK_NOTES = {
    FAILED_OPS_MARK: f"{FAILED_OPS_MARK} operations failed in this run",
    UNDEROCCUPIED_MARK: (
        f"{UNDEROCCUPIED_MARK} workers busy or pacing < "
        f"{MIN_WORKER_OCCUPANCY:.0%} of the wall clock: the bar measures waiting"
    ),
}


_PHASE_RE = re.compile(r"^Phase (\d+) \((.*)\): (\d+) ops in ([\d.]+) ms wall$")
_READ_RE = re.compile(r"^\s+Read:\s+([\d.]+) MiB")
_WRITE_RE = re.compile(r"^\s+Write:\s+([\d.]+) MiB")
_PREALLOC_RE = re.compile(r"^Prealloc: (\d+) ops in ([\d.]+) ms$")
_FAILED_RE = re.compile(r"^Failed operations: (\d+)$")
_EFFECTIVE_RE = re.compile(r"^\s+Effective concurrency: ([\d.]+) of (\d+)")
_PACING_RE = re.compile(r"^\s+Pacing share: ([\d.]+) of (\d+)")
_LEASE_RE = re.compile(r"^Lease wait: ([\d.]+) ms total, ([\d.]+) ms max worker$")
# The tail is read by name, so logs predating the percentiles still parse.
_FLOWOP_RE = re.compile(r"^ {2}(\w+) *: (\d+) ops(.*)$")
_FLOWOP_STAT_RE = re.compile(r"(avg|min|max|p50|p99) ([\d.]+) ms")
_FLOWOP_TOTAL_RE = re.compile(r"([\d.]+) ms total")
_SYNC_RE = re.compile(r"^Metadata sync: (.+)$")
_BANNER_RE = re.compile(r"^bench: (.+)$")
# Written by _save_output_to_file (actions/accelstore.py).
_EXIT_RE = re.compile(r"^=== EXIT: (-?\d+) ===$")


def _share(worker_ms: float, wall_ms: float) -> float:
    """Worker-equivalents: worker time over wall clock."""
    return worker_ms / wall_ms if wall_ms > 0 else 0.0


#: Flowops that move bytes: the denominator of `mean_op_bytes`.
DATA_FLOWOPS = frozenset({"write", "append", "readwhole", "readentry", "readrange"})


@dataclass
class BenchFlowop:
    """One flowop of a phase: its op count, its rate and its latency spread."""

    name: str
    ops: int
    ops_per_s: Optional[float] = None  # None for `delay`, which does no work
    avg_ms: Optional[float] = None
    min_ms: Optional[float] = None
    max_ms: Optional[float] = None
    p50_ms: Optional[float] = None
    p99_ms: Optional[float] = None
    total_ms: Optional[float] = None  # `delay` only


@dataclass
class BenchPhase:
    """One `loop` block of the personality, measured barrier to barrier."""

    index: int  # 1-based, as the printer numbers them
    bound: str  # the block's own bound ("duration 60s", "1000 iterations")
    ops: int
    wall_ms: float
    read_mib: float = 0.0
    written_mib: float = 0.0
    busy_ms: float = 0.0
    delay_ms: float = 0.0
    flowops: List[BenchFlowop] = field(default_factory=list)

    @property
    def ops_per_s(self) -> float:
        return tables.rate(self.ops, self.wall_ms)

    @property
    def read_mibs(self) -> float:
        return tables.rate(self.read_mib, self.wall_ms)

    @property
    def write_mibs(self) -> float:
        return tables.rate(self.written_mib, self.wall_ms)

    @property
    def effective_concurrency(self) -> float:
        return _share(self.busy_ms, self.wall_ms)

    @property
    def pacing_share(self) -> float:
        return _share(self.delay_ms, self.wall_ms)


@dataclass
class BenchRunMetrics:
    """One bench_client run, as parsed from its sidecar or its log."""

    config: str  # "{setup}-disk-{size}" result directory
    arm: str  # backend arm directory: "ext4", "aesgcm_bench_client", ...
    personality: str
    variant: Optional[
        str
    ]  # variant leaf (sync mode, threads token), None for the default
    path: Path
    phases: List[BenchPhase]
    failed_ops: int
    # Resolved by the run (banner or sidecar), never read off the directory name.
    threads: Optional[int] = None
    bound: Optional[str] = None
    placement: Optional[str] = None
    cores: Optional[int] = None
    lease_wait_ms: float = 0.0
    prealloc_ops: int = 0
    prealloc_ms: float = 0.0
    backend: Optional[str] = None
    workload_type: Optional[str] = None
    # True for O_DIRECT, False for buffered, None on the accelstore arm.
    direct: Optional[bool] = None
    seed: Optional[int] = None
    dsl_version: Optional[int] = None
    sync_config: Optional[str] = None
    exit_code: Optional[int] = None
    source: str = "log"  # "json" when the sidecar was read
    banner: Dict[str, str] = field(default_factory=dict)

    @property
    def total_ops(self) -> int:
        return sum(p.ops for p in self.phases)

    @property
    def total_wall_ms(self) -> float:
        return sum(p.wall_ms for p in self.phases)

    @property
    def read_mib(self) -> float:
        return sum(p.read_mib for p in self.phases)

    @property
    def written_mib(self) -> float:
        return sum(p.written_mib for p in self.phases)

    @property
    def busy_ms(self) -> float:
        return sum(p.busy_ms for p in self.phases)

    @property
    def delay_ms(self) -> float:
        return sum(p.delay_ms for p in self.phases)

    @property
    def ops_per_s(self) -> float:
        return tables.rate(self.total_ops, self.total_wall_ms)

    @property
    def data_ops(self) -> int:
        """Operations that moved bytes — the `DATA_FLOWOPS` of every phase."""
        return sum(
            f.ops for p in self.phases for f in p.flowops if f.name in DATA_FLOWOPS
        )

    @property
    def mean_op_bytes(self) -> Optional[float]:
        """Mean bytes per data operation, None without per-kind lines to divide by."""
        ops = self.data_ops
        if not ops:
            return None
        return (self.read_mib + self.written_mib) * (1 << 20) / ops

    @property
    def effective_concurrency(self) -> float:
        """Worker-equivalents inside ops: Σ op latency / wall."""
        return _share(self.busy_ms, self.total_wall_ms)

    @property
    def pacing_share(self) -> float:
        """Worker-equivalents in `delay` flowops: Σ delay time / wall."""
        return _share(self.delay_ms, self.total_wall_ms)

    @property
    def occupancy(self) -> Optional[float]:
        """Occupancy against the worker count, None without threads or busy time."""
        if not self.threads or self.busy_ms <= 0:
            return None
        return (self.busy_ms + self.delay_ms) / self.total_wall_ms / self.threads

    @property
    def underoccupied(self) -> bool:
        """Whether the run mostly waited."""
        occupancy = self.occupancy
        return occupancy is not None and occupancy < MIN_WORKER_OCCUPANCY

    def marks(self) -> str:
        """The bar/table annotation suffix of this row, "" for a clean one."""
        return (FAILED_OPS_MARK if self.failed_ops else "") + (
            UNDEROCCUPIED_MARK if self.underoccupied else ""
        )

    @property
    def arm_key(self) -> str:
        """Arm plus variant: what one bar of the figure represents."""
        return f"{self.arm}#{self.variant}" if self.variant else self.arm

    @property
    def pair_key(self) -> tuple:
        """What two arms of one personality must agree on to be comparable."""
        return (self.threads, self.bound)

    @property
    def cell_key(self) -> tuple:
        """What makes two runs repeats: the leaf minus thread tokens, plus pair_key."""
        tokens = [
            t
            for t in (self.variant or "").split("_")
            if t and t != "split-loops" and not re.fullmatch(r"threads-\d+", t)
        ]
        return (
            self.config,
            self.arm,
            self.personality,
            "_".join(tokens),
            *self.pair_key,
        )


def _parse_banner(line: str) -> Dict[str, str]:
    """`bench: k=v k="v with spaces" ...` into a dict, {} if it is not one."""
    m = _BANNER_RE.match(line)
    if not m:
        return {}
    tokens = {}
    try:
        parts = shlex.split(m.group(1))
    except ValueError:
        return {}
    for part in parts:
        key, sep, value = part.partition("=")
        if sep:
            tokens[key] = value
    return tokens


def _phase_from_json(block: dict) -> BenchPhase:
    """One `blocks` entry of the sidecar as a phase, flowops included."""
    wall_ms = float(block.get("wall_ms", 0.0))
    return BenchPhase(
        index=int(block.get("index", 0)),
        bound=block.get("bound", ""),
        ops=int(block.get("ops", 0)),
        wall_ms=wall_ms,
        read_mib=int(block.get("bytes_read", 0)) / (1024.0 * 1024.0),
        written_mib=int(block.get("bytes_written", 0)) / (1024.0 * 1024.0),
        busy_ms=float(block.get("busy_ms", 0.0)),
        delay_ms=float(block.get("delay_ms", 0.0)),
        flowops=[
            BenchFlowop(
                name=name,
                ops=int(kind.get("count", 0)),
                ops_per_s=(
                    None
                    if name == "delay"
                    else tables.rate(int(kind.get("count", 0)), wall_ms)
                ),
                avg_ms=kind.get("avg_ms"),
                min_ms=kind.get("min_ms"),
                max_ms=kind.get("max_ms"),
                p50_ms=kind.get("p50_ms"),
                p99_ms=kind.get("p99_ms"),
                total_ms=kind.get("total_ms"),
            )
            for name, kind in (block.get("kinds") or {}).items()
        ],
    )


def parse_bench_json(path: Path) -> Optional[dict]:
    """One run from its --json sidecar, or None if unreadable."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        warnings.warn(f"Cannot read the metrics sidecar {path}: {e}")
        return None

    placement = data.get("placement") or {}
    prealloc = data.get("prealloc") or {}
    posix = data.get("posix") or {}  # only the POSIX arm writes this object
    return {
        "phases": [_phase_from_json(b) for b in data.get("blocks", [])],
        "failed_ops": int(data.get("failed_ops", 0)),
        "threads": int(data["threads"]) if "threads" in data else None,
        "bound": data.get("bound"),
        "placement": placement.get("kind"),
        "cores": placement.get("cores"),
        "prealloc_ops": int(prealloc.get("ops", 0)),
        "prealloc_ms": float(prealloc.get("wall_ms", 0.0)),
        "backend": data.get("backend"),
        "workload_type": data.get("type"),
        "seed": int(data["seed"]) if "seed" in data else None,
        "dsl_version": int(data["dsl_version"]) if "dsl_version" in data else None,
        "direct": bool(posix["direct"]) if "direct" in posix else None,
        "lease_wait_ms": sum(
            float(w.get("lease_wait_ms", 0.0)) for w in data.get("workers", [])
        ),
        "source": "json",
    }


def _parse_flowop(m: re.Match, wall_ms: float) -> BenchFlowop:
    """A matched flowop line: its count, and whichever statistics it carries."""
    stats = {name: float(v) for name, v in _FLOWOP_STAT_RE.findall(m.group(3))}
    total = _FLOWOP_TOTAL_RE.search(m.group(3))
    name = m.group(1)
    ops = int(m.group(2))
    return BenchFlowop(
        name=name,
        ops=ops,
        ops_per_s=None if name == "delay" else tables.rate(ops, wall_ms),
        avg_ms=stats.get("avg"),
        min_ms=stats.get("min"),
        max_ms=stats.get("max"),
        p50_ms=stats.get("p50"),
        p99_ms=stats.get("p99"),
        total_ms=float(total.group(1)) if total else None,
    )


def _parse_phase_line(phase: BenchPhase, line: str) -> None:
    """A line belonging to the phase that is open: its bytes, shares, flowops."""
    if m := _READ_RE.match(line):
        phase.read_mib += float(m.group(1))
    elif m := _WRITE_RE.match(line):
        phase.written_mib += float(m.group(1))
    elif m := _EFFECTIVE_RE.match(line):
        # Printed as a share of the phase wall, so worker time is share x wall.
        phase.busy_ms = float(m.group(1)) * phase.wall_ms
    elif m := _PACING_RE.match(line):
        phase.delay_ms = float(m.group(1)) * phase.wall_ms
    elif m := _FLOWOP_RE.match(line):
        phase.flowops.append(_parse_flowop(m, phase.wall_ms))


def parse_bench_log(path: Path) -> Optional[dict]:
    """One run log, phase by phase, or None if it holds no bench results."""
    phases: List[BenchPhase] = []
    prealloc_ops, prealloc_ms = 0, 0.0
    lease_wait_ms = 0.0
    failed_ops = None
    exit_code = None
    sync_config = None
    banner: Dict[str, str] = {}
    seen_results = False

    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if m := _EXIT_RE.match(line):
                exit_code = int(m.group(1))
                continue
            if line.startswith("bench: "):
                banner.update(_parse_banner(line))
                continue
            if m := _SYNC_RE.match(line):
                sync_config = m.group(1)
                continue
            if line == "=== Bench Results ===":
                seen_results = True
                continue
            if not seen_results:
                continue
            if m := _PHASE_RE.match(line):
                phases.append(
                    BenchPhase(
                        index=int(m.group(1)),
                        bound=m.group(2),
                        ops=int(m.group(3)),
                        wall_ms=float(m.group(4)),
                    )
                )
            elif m := _PREALLOC_RE.match(line):
                prealloc_ops, prealloc_ms = int(m.group(1)), float(m.group(2))
            elif m := _LEASE_RE.match(line):
                lease_wait_ms = float(m.group(1))
            elif m := _FAILED_RE.match(line):
                failed_ops = int(m.group(1))
            elif phases:
                _parse_phase_line(phases[-1], line)

    resolved = {
        "exit_code": exit_code,
        "banner": banner,
        "sync_config": sync_config,
        "backend": banner.get("backend"),
        "workload_type": banner.get("type"),
        "seed": int(banner["seed"]) if banner.get("seed") else None,
        "dsl_version": int(banner["dsl"]) if banner.get("dsl") else None,
        "threads": int(banner["threads"]) if banner.get("threads") else None,
        "bound": banner.get("bound"),
        "placement": banner.get("placement"),
        "cores": int(banner["cores"]) if banner.get("cores") else None,
        "direct": (banner["direct"] == "1") if "direct" in banner else None,
    }
    if not seen_results or failed_ops is None:
        # No metrics, but still a record when the log says how the run ended.
        if exit_code is None and not banner:
            return None
        return {"phases": [], "failed_ops": 0, **resolved}
    return {
        "phases": phases,
        "failed_ops": failed_ops,
        "prealloc_ops": prealloc_ops,
        "prealloc_ms": prealloc_ms,
        "lease_wait_ms": lease_wait_ms,
        **resolved,
    }


def _parse_run(log: Path) -> Optional[dict]:
    """One run: the sidecar's numbers where there is one, the log's otherwise."""
    parsed = parse_bench_log(log)
    sidecar = log.with_suffix(".json")
    if sidecar.exists():
        from_json = parse_bench_json(sidecar)
        if from_json is not None:
            merged = dict(parsed or {})
            merged.update(from_json)
            return merged
    return parsed


def discover_bench_logs(bench_dir: Path, all_runs: bool = False) -> List[tuple]:
    """Latest log per (config, arm, personality, variant), failed runs included.

    Returns (config, arm, personality, variant, path) tuples. ``all_runs``
    returns every log of every leaf instead, oldest first.
    """
    leaves = []
    for config, arm, personality, variant, leaf in discovery.iter_point_leaves(
        bench_dir
    ):
        logs = sorted(leaf.glob("*.log"))
        for log in logs if all_runs else logs[-1:]:
            leaves.append((config, arm, personality, variant, log))
    return leaves


def latest_per_cell(results: List[BenchRunMetrics]) -> List[BenchRunMetrics]:
    """Newest run per :attr:`BenchRunMetrics.cell_key`, naming the superseded ones."""
    newest: Dict[tuple, BenchRunMetrics] = {}
    for r in sorted(results, key=lambda r: r.path.stem, reverse=True):
        kept = newest.setdefault(r.cell_key, r)
        if kept is not r:
            print(
                f"{r.config}/{r.arm}/{r.personality}: {r.path.name} "
                f"({r.variant or 'default leaf'}) is superseded by "
                f"{kept.path.name} ({kept.variant or 'default leaf'})"
            )
    return list(newest.values())


def discover_bench_results(bench_dir: Path) -> List[BenchRunMetrics]:
    """Latest run per (config, arm, personality, variant) that parsed at all."""
    results = []
    for config, arm, personality, variant, path in discover_bench_logs(bench_dir):
        parsed = _parse_run(path)
        if parsed is None:
            warnings.warn(f"No bench results in {path}")
            continue
        results.append(
            BenchRunMetrics(
                config=config,
                arm=arm,
                personality=personality,
                variant=variant,
                path=path,
                **parsed,
            )
        )
    return results


def gate_verdict(r: BenchRunMetrics) -> tuple:
    """(status, detail) of a run: ``exit-nonzero``, ``no-results``, ``starved``, ``ok``.

    ``starved`` is not a drop: the row is kept and marked.
    """
    if r.exit_code not in (None, 0):
        return ("exit-nonzero", f"the run exited {r.exit_code}")
    if r.total_wall_ms <= 0 or r.total_ops == 0:
        return ("no-results", "no phase metrics in the log")
    if r.underoccupied:
        return (
            "starved",
            f"its {r.threads} workers were busy or pacing {r.occupancy:.0%} of the "
            f"wall clock (< {MIN_WORKER_OCCUPANCY:.0%}); effective concurrency "
            f"{r.effective_concurrency:.2f}, lease wait {r.lease_wait_ms:.0f} ms "
            "— the row measures waiting",
        )
    return ("ok", "")


def gate_bench_results(results: List[BenchRunMetrics]) -> List[BenchRunMetrics]:
    """Drop rows that are not measurements and warn about the starved ones."""
    kept = []
    for r in results:
        status, detail = gate_verdict(r)
        if status == "starved":
            warnings.warn(
                f"Marking {r.path} '{UNDEROCCUPIED_MARK}': {detail}, read it as such"
            )
        elif status != "ok":
            warnings.warn(f"Dropping {r.path}: {detail}")
            continue
        kept.append(r)
    return kept


def enforce_arm_parity(results: List[BenchRunMetrics]) -> List[BenchRunMetrics]:
    """Drop whole personality groups whose arms disagree on thread count or bound."""
    kept = []
    groups: Dict[tuple, List[BenchRunMetrics]] = {}
    for r in results:
        groups.setdefault((r.config, r.personality), []).append(r)

    for (config, personality), group in sorted(groups.items()):
        # A run predating the banner has no resolved values to compare.
        known = [r for r in group if r.pair_key != (None, None)]
        distinct = {r.pair_key for r in known}
        if len(distinct) > 1:
            detail = ", ".join(
                f"{r.arm_key}: threads={r.threads} bound={r.bound!r}"
                for r in sorted(known, key=lambda r: r.arm_key)
            )
            warnings.warn(
                f"Refusing to pair the arms of {config}/{personality}: they did not "
                f"run the same workload ({detail}). Re-run the odd arm with matching "
                f"--threads/bound; the whole group is left out of the figure."
            )
            continue
        kept.extend(group)
    return kept


def arm_label(arm: str, variant: Optional[str] = None) -> str:
    """Legend label of a backend arm, with the variant leaf in brackets."""
    if arm.endswith(BENCH_CLIENT):
        _, crypto = CryptoBackend.split_client_binary(arm)
        label = f"accelstore ({crypto.label})"
    else:
        label = arm
    return f"{label} [{variant}]" if variant else label


def arm_color(arm: str):
    """Bar colour of a backend arm, stable under arms joining or leaving."""
    if arm.endswith(BENCH_CLIENT):
        _, crypto = CryptoBackend.split_client_binary(arm)
        return crypto_color(crypto)
    if arm in POSIX_ARM_COLORS:
        return POSIX_ARM_COLORS[arm]
    slot = zlib.crc32(arm.encode()) % len(common.PALETTE_PASTEL)
    return common.PALETTE_PASTEL[slot]


def _arm_sort_key(arm_key: str):
    """Accelstore arms in build-variant order, then the POSIX arms."""
    arm, _, variant = arm_key.partition("#")
    if arm.endswith(BENCH_CLIENT):
        _, crypto = CryptoBackend.split_client_binary(arm)
        return (0, list(CryptoBackend).index(crypto), variant)
    return (1, arm, variant)


def print_bench_table(results: List[BenchRunMetrics]) -> None:
    """Console table of the gated rows, one per (config, arm, personality)."""
    header = (
        f"{'config':<20} {'arm':<28} {'personality':<18} {'thr':>4} {'eff':>6} {'pace':>6} "
        f"{'ops/s':>10} {'read MiB':>10} {'wrote MiB':>10} {'failed':>7}"
    )
    print()
    print(header)
    print("-" * len(header))
    for r in sorted(
        results, key=lambda r: (r.config, r.personality, _arm_sort_key(r.arm_key))
    ):
        arm, _, variant = r.arm_key.partition("#")
        threads = str(r.threads) if r.threads else "?"
        effective = f"{r.effective_concurrency:.2f}" if r.busy_ms > 0 else "?"
        effective += UNDEROCCUPIED_MARK if r.underoccupied else ""
        pacing = f"{r.pacing_share:.2f}" if r.busy_ms > 0 else "?"
        print(
            f"{r.config:<20} {arm_label(arm, variant or None):<28} {r.personality:<18} "
            f"{threads:>4} {effective:>6} {pacing:>6} "
            f"{r.ops_per_s:>10.1f} {r.read_mib:>10.2f} {r.written_mib:>10.2f} "
            f"{r.failed_ops:>7}"
        )
    if any(r.underoccupied for r in results):
        print(_MARK_NOTES[UNDEROCCUPIED_MARK])


SUMMARY_COLUMNS = [
    "config",
    "arm",
    "backend",
    "type",
    "personality",
    "variant",
    "status",
    "source",
    "dsl",
    "seed",
    "threads",
    "bound",
    "sync_config",
    # The arm's own flag set, out of the run's flags sidecar.
    "direct",
    "fsync_mode",
    "barrier_mode",
    "log_layout",
    "crypto_chunk",
    "pipeline_bytes",
    # io_depth is the accelstore pipeline depth or the POSIX intra-op depth,
    # told apart by `backend`.
    "intra_op_depth",
    "io_depth",
    "io_chunk",
    "burst_bytes",
    "cluster_size",
    "sampled",
    "ack",
    "phase",
    "phase_bound",
    "ops",
    "wall_ms",
    "ops_per_s",
    "read_mib",
    "read_mibs",
    "write_mib",
    "write_mibs",
    "effective_concurrency",
    "pacing_share",
    "prealloc_ops",
    "prealloc_ms",
    "failed_ops",
    "lease_wait_ms",
    "log",
]

FLOWOP_COLUMNS = [
    "config",
    "arm",
    "backend",
    "type",
    "personality",
    "variant",
    "phase",
    "flowop",
    "ops",
    "ops_per_s",
    "avg_ms",
    "min_ms",
    "max_ms",
    "p50_ms",
    "p99_ms",
    "total_ms",
]


def read_flags_sidecar(log_path: Path) -> dict:
    """Flag set from the ``*_flags.json`` beside ``log_path``, {} if absent."""
    record = discovery.flags_record(log_path)
    if record is None:
        return {}
    chunk = record.get("crypto_chunk_size")
    geometry = record.get("pipeline_opts") or {}
    intra = record.get("intra_op") or {}
    ack = record.get("ack_semantics") or {}
    # Never both: pipeline_opts is the accelstore arm, intra_op the POSIX one.
    io_depth = geometry.get("io_depth")
    if io_depth is None:
        io_depth = intra.get("io_depth")
    return {
        "direct": record.get("direct", ""),
        "fsync_mode": record.get("fsync_mode", ""),
        "barrier_mode": record.get("barrier_mode", ""),
        "log_layout": record.get("log_layout", ""),
        "crypto_chunk": chunk if chunk else "",  # blank means the library default
        **{
            f"{key}": (geometry.get(key) if geometry.get(key) is not None else "")
            for key in ("pipeline_bytes", "burst_bytes", "cluster_size")
        },
        "intra_op_depth": intra.get("depth", ""),  # resolved, not requested
        "io_depth": io_depth if io_depth is not None else "",
        "io_chunk": (
            intra.get("io_chunk") if intra.get("io_chunk") is not None else ""
        ),
        "sampled": bool(record.get("sampler")) if "sampler" in record else "",
        "ack": ack.get("summary", "") if isinstance(ack, dict) else ack,
    }


def export_bench_csv(
    bench_dir: Path,
    summary_csv: Path,
    flowops_csv: Optional[Path] = None,
    all_runs: bool = False,
) -> List[BenchRunMetrics]:
    """Write the result tree to ``summary_csv``, one row per phase, and return the runs.

    Every point gets a row, carrying its :func:`gate_verdict` status. Returned
    are only the runs that produced metrics.

    ``flowops_csv``: also write the per-flowop latency breakdown of the same phases.
    ``all_runs``: write every log in the tree instead of the newest run per cell.
    """
    runs, summary_rows, flowop_rows = [], [], []
    parsed_runs: List[BenchRunMetrics] = []

    for config, arm, personality, variant, path in discover_bench_logs(
        bench_dir, all_runs
    ):
        parsed = _parse_run(path)
        if parsed is None:
            warnings.warn(f"No bench results in {path}")
            summary_rows.append(
                {
                    "config": config,
                    "arm": arm,
                    "personality": personality,
                    "variant": variant or "",
                    "status": "no-results",
                    "log": path,
                }
            )
            continue
        parsed_runs.append(
            BenchRunMetrics(
                config=config,
                arm=arm,
                personality=personality,
                variant=variant,
                path=path,
                **parsed,
            )
        )
    if not all_runs:
        parsed_runs = latest_per_cell(parsed_runs)

    for run in sorted(
        parsed_runs,
        key=lambda r: (r.config, r.arm, r.personality, r.variant or "", r.path),
    ):
        path = run.path
        ident = {
            "config": run.config,
            "arm": run.arm,
            "personality": run.personality,
            "variant": run.variant or "",
        }
        status, _ = gate_verdict(run)
        ident |= {
            "backend": run.backend,
            "type": run.workload_type,
            "source": run.source,
            "dsl": run.dsl_version,
            "seed": run.seed,
            "threads": run.threads,
            "bound": run.bound,
            "sync_config": run.sync_config,
            **read_flags_sidecar(path),
        }
        common_row = {
            **ident,
            "status": status,
            "prealloc_ops": run.prealloc_ops,
            "prealloc_ms": tables.fmt(run.prealloc_ms, 2),
            "failed_ops": run.failed_ops,
            "lease_wait_ms": tables.fmt(run.lease_wait_ms, 2),
            "log": path,
        }
        if not run.phases:
            summary_rows.append(common_row)
            continue
        runs.append(run)
        for phase in run.phases:
            summary_rows.append(
                {
                    **common_row,
                    "phase": phase.index,
                    "phase_bound": phase.bound,
                    "ops": phase.ops,
                    "wall_ms": tables.fmt(phase.wall_ms, 2),
                    "ops_per_s": tables.fmt(phase.ops_per_s, 2),
                    "read_mib": tables.fmt(phase.read_mib, 2),
                    "read_mibs": tables.fmt(phase.read_mibs, 2),
                    "write_mib": tables.fmt(phase.written_mib, 2),
                    "write_mibs": tables.fmt(phase.write_mibs, 2),
                    "effective_concurrency": tables.fmt(phase.effective_concurrency, 2),
                    "pacing_share": tables.fmt(phase.pacing_share, 2),
                }
            )
            flowop_rows += [
                {
                    **ident,
                    "phase": phase.index,
                    "flowop": op.name,
                    "ops": op.ops,
                    "ops_per_s": tables.fmt(op.ops_per_s, 2),
                    "avg_ms": tables.fmt(op.avg_ms, 3),
                    "min_ms": tables.fmt(op.min_ms, 3),
                    "max_ms": tables.fmt(op.max_ms, 3),
                    "p50_ms": tables.fmt(op.p50_ms, 3),
                    "p99_ms": tables.fmt(op.p99_ms, 3),
                    "total_ms": tables.fmt(op.total_ms, 2),
                }
                for op in phase.flowops
            ]

    tables.write_csv(summary_csv, SUMMARY_COLUMNS, summary_rows)
    if flowops_csv is not None:
        tables.write_csv(flowops_csv, FLOWOP_COLUMNS, flowop_rows)

    return runs
