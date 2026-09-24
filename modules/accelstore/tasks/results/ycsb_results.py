#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parsing for run-ycsb results (ycsb_client).

Numbers come from the ``--json`` sidecar next to a log where there is one and
from the metrics printer's lines otherwise, plus the ``*_flags.json`` the ycsb
action writes. The result tree run-ycsb writes is::

    bench-result/ycsb/{setup}-disk-{size}/{arm}/{point}/[variant]/{ts}.log
                                                                 /{ts}.json
                                                                 /{ts}_flags.json

The load and run phases sit side by side in a row and are never summed: the
load phase is a pure insert rate, the run phase the mix's.
"""

import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from . import discovery, tables

YCSB_CLIENT = "ycsb_client"

PHASES = ("load", "run")

KINDS = ("insert", "read", "update", "rmw")  # insert is load-phase only

# Restated from bench_results rather than imported, which would pull in matplotlib.
MIN_WORKER_OCCUPANCY = 0.5

MIB = 1024.0 * 1024.0

_EXIT_RE = re.compile(r"^=== EXIT: (-?\d+) ===$")
_BANNER_RE = re.compile(r"^ycsb: (.+)$")
_PHASE_RE = re.compile(r"^Phase (\w+): (\d+) ops in ([\d.]+) ms wall$")
_KIND_RE = re.compile(r"^ {2}(\w+) *: (\d+) ops,(.*)$")
_KIND_STAT_RE = re.compile(r"(avg|min|max|p50|p99) ([\d.]+) ms")
_KIND_FAILED_RE = re.compile(r"(\d+) failed")
# The tail is optional: a direction that moved nothing prints the bytes alone.
_DIR_RE = re.compile(r"^ {2}(Read|Write):\s+([\d.]+) MiB(.*)$")
_PER_REQUEST_RE = re.compile(r"([\d.]+) MiB/s per request")
_EFFECTIVE_RE = re.compile(r"^ {2}Effective concurrency: ([\d.]+) of (\d+)")
_FAILED_RE = re.compile(r"^ {2}Failed operations: (\d+)$")
_MISMATCH_RE = re.compile(r"^ {2}Validation mismatches: (\d+)$")
_DISK_RE = re.compile(r"^ {2}Store size after this phase: ([\d.]+) MiB$")


@dataclass
class YcsbKind:
    """One op kind of one phase: its count and its latency spread."""

    name: str
    count: int
    failed: int = 0
    avg_ms: Optional[float] = None
    min_ms: Optional[float] = None
    max_ms: Optional[float] = None
    p50_ms: Optional[float] = None
    p99_ms: Optional[float] = None


@dataclass
class YcsbDirection:
    """One direction of one phase: its bytes and the service time behind them.

    ``aggregate = per request x in flight`` exactly.
    """

    name: str
    bytes: int = 0
    service_ms: Optional[float] = None  # None, not 0, where it was never reported

    @property
    def mib(self) -> float:
        return self.bytes / MIB

    @property
    def per_request_mibs(self) -> Optional[float]:
        """Bytes over service time: one request's rate, thread count aside."""
        return (
            None if self.service_ms is None else tables.rate(self.mib, self.service_ms)
        )

    def in_flight(self, wall_ms: float) -> Optional[float]:
        """The direction's measured concurrency: service time over the wall."""
        if self.service_ms is None or wall_ms <= 0:
            return None
        return self.service_ms / wall_ms

    def aggregate_mibs(self, wall_ms: float) -> float:
        """Bytes over the phase wall: the rate the phase actually delivered."""
        return tables.rate(self.mib, wall_ms)


@dataclass
class YcsbPhase:
    """One phase, measured barrier to barrier by the coordinator."""

    name: str
    ops: int
    wall_ms: float
    read: YcsbDirection = field(default_factory=lambda: YcsbDirection("read"))
    write: YcsbDirection = field(default_factory=lambda: YcsbDirection("write"))
    busy_ms: float = 0.0
    concurrency: Optional[float] = None
    ceiling: Optional[int] = None  # worker count
    failed_ops: int = 0
    mismatches: int = 0
    disk_bytes: Optional[int] = None  # RocksDB arm only
    kinds: Dict[str, YcsbKind] = field(default_factory=dict)

    @property
    def ops_per_s(self) -> float:
        return tables.rate(self.ops, self.wall_ms)

    @property
    def read_mib(self) -> float:
        return self.read.mib

    @property
    def written_mib(self) -> float:
        return self.write.mib

    @property
    def read_mibs(self) -> float:
        """The aggregate rate: bytes over the wall."""
        return self.read.aggregate_mibs(self.wall_ms)

    @property
    def write_mibs(self) -> float:
        return self.write.aggregate_mibs(self.wall_ms)

    @property
    def effective_concurrency(self) -> float:
        """Worker-equivalents inside ops: the client's figure, or busy/wall."""
        if self.concurrency is not None:
            return self.concurrency
        return self.busy_ms / self.wall_ms if self.wall_ms > 0 else 0.0


@dataclass
class YcsbRunMetrics:
    """One ycsb_client run, as parsed from its sidecar and its log."""

    config: str  # "{setup}-disk-{size}" result directory
    arm: str  # arm directory: "ext4", "aesgcm_ycsb_client", ...
    point: str  # point directory: "A_4k" .. "F_1m", or "A_64m" .. "A_1g"
    variant: Optional[str]  # variant leaf, None for the default run
    path: Path
    phases: Dict[str, YcsbPhase] = field(default_factory=dict)
    failed_ops: int = 0
    mismatches: int = 0
    exit_code: Optional[int] = None
    # What the run resolved to, not what the directory name asked for.
    backend: Optional[str] = None
    mix: Optional[str] = None
    value_size: Optional[int] = None
    num_keys: Optional[int] = None
    num_ops: Optional[int] = None
    threads: Optional[int] = None
    seed: Optional[int] = None
    dist: Optional[str] = None
    theta: Optional[float] = None
    sync_config: Optional[str] = None
    placement: Optional[str] = None
    cores: Optional[int] = None
    source: str = "log"  # "json" when the sidecar was read
    flags: Dict[str, object] = field(default_factory=dict)

    def phase(self, name: str) -> Optional[YcsbPhase]:
        return self.phases.get(name)

    @property
    def total_ops(self) -> int:
        return sum(p.ops for p in self.phases.values())

    @property
    def underoccupied(self) -> bool:
        """Whether the run phase mostly waited."""
        phase = self.phase("run")
        if phase is None or not self.threads or phase.wall_ms <= 0:
            return False
        return phase.effective_concurrency / self.threads < MIN_WORKER_OCCUPANCY

    @property
    def status(self) -> str:
        """The verdict this row carries into the CSV."""
        if not self.phases:
            return "no-results"
        if self.exit_code not in (None, 0):
            return "exit-nonzero"
        if self.failed_ops or self.mismatches:
            return "failed-ops"
        if self.underoccupied:
            return "starved"
        return "ok"


def _kind_from_json(name: str, block: dict) -> YcsbKind:
    return YcsbKind(
        name=name,
        count=int(block.get("count", 0)),
        failed=int(block.get("failed", 0)),
        avg_ms=block.get("avg_ms"),
        min_ms=block.get("min_ms"),
        max_ms=block.get("max_ms"),
        p50_ms=block.get("p50_ms"),
        p99_ms=block.get("p99_ms"),
    )


def _direction_from_json(name: str, phase: dict, bytes_key: str) -> YcsbDirection:
    """One direction, from the phase block or the legacy phase-level ``bytes_key``."""
    block = phase.get(name)
    if not isinstance(block, dict):
        return YcsbDirection(name=name, bytes=int(phase.get(bytes_key, 0)))
    service = block.get("service_ms")
    return YcsbDirection(
        name=name,
        bytes=int(block.get("bytes", phase.get(bytes_key, 0))),
        service_ms=None if service is None else float(service),
    )


def _phase_from_json(block: dict) -> YcsbPhase:
    return YcsbPhase(
        name=block.get("name", ""),
        ops=int(block.get("ops", 0)),
        wall_ms=float(block.get("wall_ms", 0.0)),
        read=_direction_from_json("read", block, "bytes_read"),
        write=_direction_from_json("write", block, "bytes_written"),
        busy_ms=float(block.get("busy_ms", 0.0)),
        concurrency=block.get("effective_concurrency"),
        ceiling=block.get("concurrency_ceiling"),
        failed_ops=int(block.get("failed_ops", 0)),
        mismatches=int(block.get("mismatches", 0)),
        disk_bytes=block.get("disk_bytes"),
        kinds={
            name: _kind_from_json(name, kind)
            for name, kind in (block.get("kinds") or {}).items()
        },
    )


def parse_ycsb_json(path: Path) -> Optional[dict]:
    """One run from its --json sidecar, or None if unreadable."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        warnings.warn(f"Cannot read the metrics sidecar {path}: {e}")
        return None

    cfg = data.get("config") or {}
    placement = cfg.get("placement") or {}
    return {
        "phases": {p.name: p for p in map(_phase_from_json, data.get("phases", []))},
        "failed_ops": int(data.get("failed_ops", 0)),
        "mismatches": int(data.get("mismatches", 0)),
        "exit_code": data.get("exit_status"),
        "backend": cfg.get("backend"),
        "mix": cfg.get("mix"),
        "value_size": cfg.get("value_size"),
        "num_keys": cfg.get("keys"),
        "num_ops": cfg.get("ops"),
        "threads": cfg.get("threads"),
        "seed": cfg.get("seed"),
        "dist": cfg.get("dist"),
        "theta": cfg.get("theta"),
        "sync_config": cfg.get("sync"),
        "placement": placement.get("kind"),
        "cores": placement.get("cores"),
        "source": "json",
    }


def _parse_banner(line: str) -> Dict[str, str]:
    """``ycsb: k=v k=v ...`` into a dict, ``{}`` if the line is not one."""
    m = _BANNER_RE.match(line)
    if not m:
        return {}
    rest = m.group(1)
    if rest.startswith("sync="):
        return {"sync": rest[len("sync=") :].strip()}
    tokens = {}
    for part in rest.split():
        key, sep, value = part.partition("=")
        if sep:
            tokens[key] = value
    return tokens


def _parse_kind(m: re.Match) -> YcsbKind:
    stats = {name: float(v) for name, v in _KIND_STAT_RE.findall(m.group(3))}
    failed = _KIND_FAILED_RE.search(m.group(3))
    return YcsbKind(
        name=m.group(1),
        count=int(m.group(2)),
        failed=int(failed.group(1)) if failed else 0,
        avg_ms=stats.get("avg"),
        min_ms=stats.get("min"),
        max_ms=stats.get("max"),
        p50_ms=stats.get("p50"),
        p99_ms=stats.get("p99"),
    )


def _parse_direction_line(direction: YcsbDirection, mib: float, tail: str) -> None:
    """The client's rate line, back into bytes and service time."""
    direction.bytes = int(mib * MIB)
    m = _PER_REQUEST_RE.search(tail)
    if m and float(m.group(1)) > 0:
        direction.service_ms = mib / float(m.group(1)) * 1000.0


def _parse_phase_line(phase: YcsbPhase, line: str) -> None:
    """One indented line of a phase block, folded into the phase."""
    m = _DIR_RE.match(line)
    if m:
        direction = phase.read if m.group(1) == "Read" else phase.write
        _parse_direction_line(direction, float(m.group(2)), m.group(3))
        return
    for regex, apply in (
        (_FAILED_RE, lambda v: setattr(phase, "failed_ops", int(v))),
        (_MISMATCH_RE, lambda v: setattr(phase, "mismatches", int(v))),
        (_DISK_RE, lambda v: setattr(phase, "disk_bytes", int(float(v) * MIB))),
    ):
        m = regex.match(line)
        if m:
            apply(m.group(1))
            return
    m = _EFFECTIVE_RE.match(line)
    if m:
        phase.concurrency = float(m.group(1))
        phase.ceiling = int(m.group(2))
        return
    m = _KIND_RE.match(line)
    if m:
        kind = _parse_kind(m)
        phase.kinds[kind.name] = kind
        phase.busy_ms += (kind.avg_ms or 0.0) * kind.count


def parse_ycsb_log(path: Path) -> Optional[dict]:
    """One run from the metrics printer's lines, for a log with no sidecar."""
    try:
        text = path.read_text(errors="replace")
    except OSError as e:
        warnings.warn(f"Cannot read {path}: {e}")
        return None

    phases: Dict[str, YcsbPhase] = {}
    current: Optional[YcsbPhase] = None
    banner: Dict[str, str] = {}
    exit_code = None

    for line in text.splitlines():
        m = _EXIT_RE.match(line)
        if m:
            exit_code = int(m.group(1))
            continue
        tokens = _parse_banner(line)
        if tokens:
            banner.update(tokens)
            continue
        m = _PHASE_RE.match(line)
        if m:
            current = YcsbPhase(
                name=m.group(1), ops=int(m.group(2)), wall_ms=float(m.group(3))
            )
            phases[current.name] = current
            continue
        if current is not None and line.startswith("  "):
            _parse_phase_line(current, line)
        elif line and not line.startswith(" "):
            current = None

    def _int(key):
        return int(banner[key]) if key in banner else None

    return {
        "phases": phases,
        "failed_ops": sum(p.failed_ops for p in phases.values()),
        "mismatches": sum(p.mismatches for p in phases.values()),
        "exit_code": exit_code,
        "backend": banner.get("backend"),
        "mix": banner.get("mix"),
        "value_size": _int("value"),
        "num_keys": _int("keys"),
        "num_ops": _int("ops"),
        "threads": _int("threads"),
        "seed": _int("seed"),
        "dist": banner.get("dist"),
        "sync_config": banner.get("sync"),
        "placement": banner.get("placement"),
        "cores": _int("cores"),
        "source": "log",
    }


def read_flags_sidecar(log_path: Path) -> dict:
    """The arm's flag set, from the ``*_flags.json`` next to ``log_path``."""
    record = discovery.flags_record(log_path)
    if record is None:
        return {}

    geometry = record.get("point_config") or {}
    stack = record.get("stack") or {}
    return {
        "arm_id": record.get("arm", ""),
        "stack": stack.get("id", ""),
        "stack_kernel": stack.get("kernel", ""),
        "crypto": record.get("crypto", ""),
        "crypto_chunk": geometry.get("chunk_size", ""),
        "cluster_size": geometry.get("cluster_size", ""),
        "md_pages": geometry.get("md_pages", ""),
        "capacity_gb": record.get("capacity_gb", ""),
        "store_gb": geometry.get("store_footprint_gb", ""),
        "validate": record.get("validate", ""),
        "ack": tables.ack_summary(record, ("rocksdb", "stack")),
    }


def parse_run(log: Path) -> Optional[dict]:
    """One run: the sidecar's numbers where there is one, the log's otherwise."""
    parsed = parse_ycsb_log(log)
    sidecar = log.with_suffix(".json")
    if sidecar.exists():
        from_json = parse_ycsb_json(sidecar)
        if from_json is not None:
            merged = dict(parsed or {})
            # The log's exit code wins: the sidecar records only the intended one.
            log_exit = merged.get("exit_code")
            merged.update(from_json)
            if log_exit is not None:
                merged["exit_code"] = log_exit
            return merged
    return parsed


def discover_ycsb_runs(root: Path) -> List[tuple]:
    """Latest log per (config, arm, point, variant), runs without metrics included."""
    leaves = []
    for config, arm, point, variant, leaf in discovery.iter_point_leaves(root):
        logs = sorted(leaf.glob("*.log"))
        if logs:
            leaves.append((config, arm, point, variant, logs[-1]))
    return leaves


def discover_ycsb_results(root: Path) -> List[YcsbRunMetrics]:
    """Latest run per (config, arm, point, variant) that parsed at all."""
    results = []
    for config, arm, point, variant, path in discover_ycsb_runs(root):
        parsed = parse_run(path)
        if parsed is None:
            warnings.warn(f"No ycsb results in {path}")
            continue
        results.append(
            YcsbRunMetrics(
                config=config,
                arm=arm,
                point=point,
                variant=variant,
                path=path,
                flags=read_flags_sidecar(path),
                **parsed,
            )
        )
    return results


SUMMARY_COLUMNS = [
    "config",
    "arm",
    "backend",
    "point",
    "variant",
    "status",
    "source",
    "mix",
    "value_size",
    "num_keys",
    "num_ops",
    "threads",
    "seed",
    "dist",
    "theta",
    "sync_config",
    "stack",
    "crypto",
    "crypto_chunk",
    "cluster_size",
    "md_pages",
    "capacity_gb",
    "store_gb",
    "ack",
    "load_ops",
    "load_wall_ms",
    "load_ops_per_s",
    "load_write_mib",
    "load_write_mibs",
    "load_write_per_request_mibs",
    "load_write_in_flight",
    "load_write_service_ms",
    "load_effective_concurrency",
    "load_disk_bytes",
    "run_ops",
    "run_wall_ms",
    "run_ops_per_s",
    "run_read_mib",
    "run_read_mibs",
    "run_read_per_request_mibs",
    "run_read_in_flight",
    "run_read_service_ms",
    "run_write_mib",
    "run_write_mibs",
    "run_write_per_request_mibs",
    "run_write_in_flight",
    "run_write_service_ms",
    "run_effective_concurrency",
    "run_disk_bytes",
    "insert_p50_ms",
    "insert_p99_ms",
    "read_p50_ms",
    "read_p99_ms",
    "update_p50_ms",
    "update_p99_ms",
    "rmw_p50_ms",
    "rmw_p99_ms",
    "failed_ops",
    "mismatches",
    "exit_code",
    "log",
]

KIND_COLUMNS = [
    "config",
    "arm",
    "point",
    "variant",
    "phase",
    "kind",
    "count",
    "failed",
    "avg_ms",
    "min_ms",
    "max_ms",
    "p50_ms",
    "p99_ms",
]


def _direction_columns(prefix: str, direction: YcsbDirection, wall_ms: float) -> dict:
    """One direction's bytes and every rate that can be read off them."""
    return {
        f"{prefix}_mib": tables.fmt(direction.mib, 2),
        f"{prefix}_mibs": tables.fmt(direction.aggregate_mibs(wall_ms), 2),
        f"{prefix}_per_request_mibs": tables.fmt(direction.per_request_mibs, 2),
        f"{prefix}_in_flight": tables.fmt(direction.in_flight(wall_ms), 2),
        f"{prefix}_service_ms": tables.fmt(direction.service_ms, 2),
    }


def _phase_columns(run: YcsbRunMetrics, name: str) -> dict:
    """The ``load_*`` or ``run_*`` half of a row, blank for a missing phase."""
    phase = run.phase(name)
    if phase is None:
        return {}
    row = {
        f"{name}_ops": phase.ops,
        f"{name}_wall_ms": tables.fmt(phase.wall_ms, 2),
        f"{name}_ops_per_s": tables.fmt(phase.ops_per_s, 2),
        **_direction_columns(f"{name}_write", phase.write, phase.wall_ms),
        f"{name}_effective_concurrency": tables.fmt(phase.effective_concurrency, 2),
        f"{name}_disk_bytes": (
            "" if phase.disk_bytes is None else int(phase.disk_bytes)
        ),
    }
    if name == "run":
        row |= _direction_columns("run_read", phase.read, phase.wall_ms)
    return row


def _percentile_columns(run: YcsbRunMetrics) -> dict:
    """One p50/p99 pair per op kind, taken from the phase that runs it."""
    columns = {}
    for phase in run.phases.values():
        for name, kind in phase.kinds.items():
            if name not in KINDS:
                continue
            columns[f"{name}_p50_ms"] = tables.fmt(kind.p50_ms, 3)
            columns[f"{name}_p99_ms"] = tables.fmt(kind.p99_ms, 3)
    return columns


def export_points_csv(
    root: Path, summary_csv: Path, kinds_csv: Optional[Path] = None
) -> List[YcsbRunMetrics]:
    """Write the result tree as CSV and return the runs that produced metrics.

    One row per (config, arm, point, variant), load and run phases side by side,
    including points that produced no metrics (their ``status`` says so).
    ``kinds_csv``, if given, takes the per-kind latency breakdown.
    """
    runs, summary_rows, kind_rows = [], [], []

    for config, arm, point, variant, path in discover_ycsb_runs(root):
        ident = {
            "config": config,
            "arm": arm,
            "point": point,
            "variant": variant or "",
        }
        parsed = parse_run(path)
        if parsed is None:
            warnings.warn(f"No ycsb results in {path}")
            summary_rows.append({**ident, "status": "no-results", "log": path})
            continue

        run = YcsbRunMetrics(
            config=config,
            arm=arm,
            point=point,
            variant=variant,
            path=path,
            flags=read_flags_sidecar(path),
            **parsed,
        )
        row = {
            **ident,
            "backend": run.backend,
            "status": run.status,
            "source": run.source,
            "mix": run.mix,
            "value_size": run.value_size,
            "num_keys": run.num_keys,
            "num_ops": run.num_ops,
            "threads": run.threads,
            "seed": run.seed,
            "dist": run.dist,
            "theta": run.theta,
            "sync_config": run.sync_config,
            **{k: v for k, v in run.flags.items() if k in SUMMARY_COLUMNS},
            **_phase_columns(run, "load"),
            **_phase_columns(run, "run"),
            **_percentile_columns(run),
            "failed_ops": run.failed_ops,
            "mismatches": run.mismatches,
            "exit_code": run.exit_code,
            "log": path,
        }
        summary_rows.append(row)
        if not run.phases:
            continue
        runs.append(run)
        for phase_name, phase in run.phases.items():
            kind_rows += [
                {
                    **ident,
                    "phase": phase_name,
                    "kind": kind.name,
                    "count": kind.count,
                    "failed": kind.failed,
                    "avg_ms": tables.fmt(kind.avg_ms, 3),
                    "min_ms": tables.fmt(kind.min_ms, 3),
                    "max_ms": tables.fmt(kind.max_ms, 3),
                    "p50_ms": tables.fmt(kind.p50_ms, 3),
                    "p99_ms": tables.fmt(kind.p99_ms, 3),
                }
                for kind in phase.kinds.values()
            ]

    tables.write_csv(summary_csv, SUMMARY_COLUMNS, summary_rows)
    if kinds_csv is not None:
        tables.write_csv(kinds_csv, KIND_COLUMNS, kind_rows)

    return runs


def print_ycsb_table(runs: List[YcsbRunMetrics]) -> None:
    """The per-run comparison table: one line per run, both phases on it.

    The MiB/s columns are the run phase's per-request rates. The aggregates and
    the load phase's own rate are in the CSV.
    """
    header = (
        f"{'point':<8} {'arm':<20} {'thr':>4} "
        f"{'load ops/s':>12} {'run ops/s':>12} "
        f"{'rd MiB/s':>10} {'wr MiB/s':>10} {'read p50':>9} {'read p99':>9} "
        f"{'conc':>6} {'status':<12}"
    )
    print("\n" + header)
    print("-" * len(header))
    for run in sorted(runs, key=lambda r: (r.point, r.arm, r.variant or "")):
        load = run.phase("load")
        mix = run.phase("run")
        read = (mix.kinds.get("read") if mix else None) or YcsbKind("read", 0)
        blank = YcsbDirection("")
        print(
            f"{run.point:<8} {run.arm:<20} {run.threads or 0:>4} "
            f"{(load.ops_per_s if load else 0):>12,.0f} "
            f"{(mix.ops_per_s if mix else 0):>12,.0f} "
            f"{tables.fmt((mix.read if mix else blank).per_request_mibs, 2):>10} "
            f"{tables.fmt((mix.write if mix else blank).per_request_mibs, 2):>10} "
            f"{tables.fmt(read.p50_ms, 3):>9} {tables.fmt(read.p99_ms, 3):>9} "
            f"{(mix.effective_concurrency if mix else 0):>6.2f} "
            f"{run.status:<12}"
        )
