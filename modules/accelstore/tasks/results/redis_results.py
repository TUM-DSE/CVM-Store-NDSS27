#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parsing for accelstore.run-redis results (the Redis application row).

Everything comes from the run's own sidecars: the ``.json`` the redis action
writes (the parsed ``redis-benchmark --csv`` row of every phase, the ``INFO
persistence`` counters around it and the timed walls of the ``dataset`` point)
plus the ``*_flags.json`` recording what the arm ran. The result tree run-redis
writes is::

    bench-result/redis/{setup}-disk-{size}/{arm}/{point}/[variant]/{ts}.json
                                                                  /{ts}_flags.json
                                                                  /{ts}.log ...

Phases are never summed: ``run`` is the row, ``fill`` the dataset load, and
``bgsave``, ``bgrewriteaof`` and ``restart_load`` are walls in seconds.
"""

import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from . import discovery, tables

ACCELSTORE_ARM = "accelstore"

# `run` is the only phase every point has, the rest belong to the `dataset` point.
BENCHMARK_PHASES = ("fill", "run")
WALL_PHASES = ("bgsave", "bgrewriteaof", "restart_load")

# Groups the files of one run and orders the runs of one leaf, lexicographically.
_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})")

MIB = 1024.0 * 1024.0


def _opt_float(value) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _opt_int(value) -> Optional[int]:
    try:
        return None if value in (None, "") else int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class RedisTest:
    """One redis-benchmark --csv row: one test's rate and latency spread."""

    name: str
    rps: Optional[float] = None
    avg_ms: Optional[float] = None
    min_ms: Optional[float] = None
    p50_ms: Optional[float] = None
    p95_ms: Optional[float] = None
    p99_ms: Optional[float] = None
    max_ms: Optional[float] = None


@dataclass
class RedisBenchmarkPhase:
    """A phase that ran redis-benchmark: ``fill`` or ``run``."""

    name: str
    exit_code: Optional[int] = None
    error: Optional[str] = None
    tests: Dict[str, RedisTest] = field(default_factory=dict)
    aof_delta_bytes: Optional[int] = None  # None, not 0, where INFO is missing
    delayed_fsync: Optional[int] = None

    def test(self, name: str) -> Optional[RedisTest]:
        """One test's row, by redis-benchmark's (upper-case) name."""
        return self.tests.get(name.upper())

    @property
    def primary(self) -> Optional[RedisTest]:
        """The first test of the phase: the row, for the one-test points."""
        return next(iter(self.tests.values()), None)


@dataclass
class RedisWallPhase:
    """A phase that is a wall: ``bgsave``, ``bgrewriteaof``, ``restart_load``."""

    name: str
    wall_s: Optional[float] = None
    status: Optional[str] = None  # Redis's own verdict on the child
    error: Optional[str] = None


@dataclass
class RedisRunMetrics:
    """One run-redis run, as parsed from its two sidecars."""

    config: str  # "{setup}-disk-{size}" result directory
    arm: str  # arm directory: "accelstore", "ext4", "rollbaccine", ...
    point: str  # point directory: "always_c1" .. "dataset"
    variant: Optional[str]  # variant leaf, None for the default run
    path: Path  # the run's .json sidecar (which may not exist)
    benchmark_phases: Dict[str, RedisBenchmarkPhase] = field(default_factory=dict)
    wall_phases: Dict[str, RedisWallPhase] = field(default_factory=dict)
    server_start_s: Optional[float] = None
    restart_dbsize: Optional[int] = None  # keys the restarted server loaded
    error: Optional[str] = None  # the action's own top-level failure record
    # What the run resolved to, not what the directory name asked for.
    point_config: Dict[str, object] = field(default_factory=dict)
    flags: Dict[str, object] = field(default_factory=dict)
    source: str = "json"

    def benchmark(self, name: str) -> Optional[RedisBenchmarkPhase]:
        return self.benchmark_phases.get(name)

    def wall(self, name: str) -> Optional[RedisWallPhase]:
        return self.wall_phases.get(name)

    @property
    def phase_errors(self) -> List[str]:
        """Every phase that recorded a failure, in phase order."""
        errors = []
        for name in BENCHMARK_PHASES:
            phase = self.benchmark_phases.get(name)
            if phase is not None and phase.error:
                errors.append(name)
        for name in WALL_PHASES:
            phase = self.wall_phases.get(name)
            if phase is not None and phase.error:
                errors.append(name)
        return errors

    @property
    def status(self) -> str:
        """The verdict this row carries into the CSV."""
        if self.error or self.phase_errors:
            return "failed"
        run = self.benchmark("run")
        if run is None or not run.tests:
            return "no-results"
        return "ok"


def _test_from_json(row: dict) -> RedisTest:
    """One parsed --csv row, from the action's own field names."""
    return RedisTest(
        name=str(row.get("test", "")).upper(),
        rps=_opt_float(row.get("rps")),
        avg_ms=_opt_float(row.get("avg_latency_ms")),
        min_ms=_opt_float(row.get("min_latency_ms")),
        p50_ms=_opt_float(row.get("p50_latency_ms")),
        p95_ms=_opt_float(row.get("p95_latency_ms")),
        p99_ms=_opt_float(row.get("p99_latency_ms")),
        max_ms=_opt_float(row.get("max_latency_ms")),
    )


def _info_delta(block: dict, counter: str) -> Optional[int]:
    """The growth of one INFO counter across a phase, or None if not bracketed."""
    before = _opt_int((block.get("info_before") or {}).get(counter))
    after = _opt_int((block.get("info_after") or {}).get(counter))
    if before is None or after is None:
        return None
    return after - before


def _benchmark_phase_from_json(name: str, block: dict) -> RedisBenchmarkPhase:
    tests = [_test_from_json(row) for row in block.get("tests") or []]
    return RedisBenchmarkPhase(
        name=name,
        exit_code=_opt_int(block.get("exit_code")),
        error=block.get("error"),
        tests={t.name: t for t in tests},
        aof_delta_bytes=_info_delta(block, "aof_current_size"),
        delayed_fsync=_info_delta(block, "aof_delayed_fsync"),
    )


def _wall_phase_from_json(name: str, block: dict) -> RedisWallPhase:
    return RedisWallPhase(
        name=name,
        wall_s=_opt_float(block.get("wall_s")),
        status=block.get("status"),
        error=block.get("error"),
    )


def parse_redis_json(path: Path) -> Optional[dict]:
    """One run from its ``.json`` sidecar, or None if unreadable."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        warnings.warn(f"Cannot read the metrics sidecar {path}: {e}")
        return None

    phases = data.get("phases") or {}
    benchmark_phases = {
        name: _benchmark_phase_from_json(name, phases[name])
        for name in BENCHMARK_PHASES
        if isinstance(phases.get(name), dict)
    }
    wall_phases = {
        name: _wall_phase_from_json(name, phases[name])
        for name in WALL_PHASES
        if isinstance(phases.get(name), dict)
    }
    restart = phases.get("restart_load") or {}
    return {
        "benchmark_phases": benchmark_phases,
        "wall_phases": wall_phases,
        "server_start_s": _opt_float(data.get("server_start_s")),
        "restart_dbsize": _opt_int(restart.get("dbsize")),
        "error": data.get("error"),
        "point_config": data.get("point_config") or {},
        "source": "json",
    }


def read_flags_sidecar(json_path: Path) -> dict:
    """The arm's flag set, from the ``*_flags.json`` next to ``json_path``."""
    record = discovery.flags_record(json_path)
    if record is None:
        return {}

    stack = record.get("stack") or {}
    geometry = record.get("store_geometry") or {}
    return {
        "backend": record.get("backend", ""),
        "tc_backend": record.get("tc_backend", ""),
        "tc_rtt_us": record.get("tc_rtt_us", ""),
        "workers": record.get("workers", ""),
        "stack": stack.get("id", ""),
        "stack_kernel": stack.get("kernel", ""),
        "capacity_gb": record.get("capacity_gb", ""),
        "cluster_size": geometry.get("cluster_size", ""),
        "md_pages": geometry.get("md_pages", ""),
        "ack": tables.ack_summary(record, ("redis", "stack")),
    }


def _leaf_timestamps(leaf: Path) -> List[str]:
    """Every run timestamp with any file in this leaf, sorted ascending."""
    stamps = set()
    for entry in leaf.iterdir():
        if not entry.is_file():
            continue
        m = _TIMESTAMP_RE.match(entry.name)
        if m:
            stamps.add(m.group(1))
    return sorted(stamps)


def discover_redis_runs(root: Path) -> List[tuple]:
    """Latest run per (config, arm, point, variant) under the result tree.

    Returns (config, arm, point, variant, json_path) tuples. ``json_path`` is
    where the sidecar belongs and may not exist.
    """
    leaves = []
    for config, arm, point, variant, leaf in discovery.iter_point_leaves(root):
        stamps = _leaf_timestamps(leaf)
        if stamps:
            leaves.append((config, arm, point, variant, leaf / f"{stamps[-1]}.json"))
    return leaves


def discover_redis_results(root: Path) -> List[RedisRunMetrics]:
    """Latest run per (config, arm, point, variant) that parsed at all."""
    results = []
    for config, arm, point, variant, path in discover_redis_runs(root):
        if not path.exists():
            warnings.warn(f"No redis metrics sidecar in {path.parent}")
            continue
        parsed = parse_redis_json(path)
        if parsed is None:
            continue
        results.append(
            RedisRunMetrics(
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
    "point",
    "variant",
    "status",
    "backend",
    "appendfsync",
    "clients",
    "pipeline",
    "value_size",
    "requests",
    "fill_requests",
    "tc_backend",
    "tc_rtt_us",
    "workers",
    "stack",
    "stack_kernel",
    "capacity_gb",
    "cluster_size",
    "md_pages",
    "ack",
    "server_start_s",
    "fill_rps",
    "fill_aof_bytes",
    "run_rps",
    "run_avg_ms",
    "run_min_ms",
    "run_p50_ms",
    "run_p95_ms",
    "run_p99_ms",
    "run_max_ms",
    "run_aof_bytes",
    "run_delayed_fsync",
    "bgsave_wall_s",
    "bgsave_status",
    "bgrewriteaof_wall_s",
    "bgrewriteaof_status",
    "restart_load_wall_s",
    "restart_dbsize",
    "error",
    "json",
]

TESTS_COLUMNS = [
    "config",
    "arm",
    "point",
    "variant",
    "phase",
    "test",
    "rps",
    "avg_ms",
    "min_ms",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "max_ms",
]


def _point_columns(run: RedisRunMetrics) -> dict:
    """The point's resolved parameters, from the sidecar's own record."""
    cfg = run.point_config
    return {
        key: cfg.get(key, "")
        for key in (
            "appendfsync",
            "clients",
            "pipeline",
            "value_size",
            "requests",
            "fill_requests",
        )
    }


def _run_phase_columns(run: RedisRunMetrics) -> dict:
    """The ``run`` phase's half of a row: the point's own benchmark numbers."""
    phase = run.benchmark("run")
    if phase is None:
        return {}
    row = {
        "run_aof_bytes": "" if phase.aof_delta_bytes is None else phase.aof_delta_bytes,
        "run_delayed_fsync": (
            "" if phase.delayed_fsync is None else phase.delayed_fsync
        ),
    }
    test = phase.primary
    if test is not None:
        row |= {
            "run_rps": tables.fmt(test.rps, 2),
            "run_avg_ms": tables.fmt(test.avg_ms, 3),
            "run_min_ms": tables.fmt(test.min_ms, 3),
            "run_p50_ms": tables.fmt(test.p50_ms, 3),
            "run_p95_ms": tables.fmt(test.p95_ms, 3),
            "run_p99_ms": tables.fmt(test.p99_ms, 3),
            "run_max_ms": tables.fmt(test.max_ms, 3),
        }
    return row


def _fill_phase_columns(run: RedisRunMetrics) -> dict:
    """The ``dataset`` point's fill: its insert rate and what it persisted."""
    phase = run.benchmark("fill")
    if phase is None:
        return {}
    test = phase.primary
    return {
        "fill_rps": tables.fmt(test.rps if test else None, 2),
        "fill_aof_bytes": (
            "" if phase.aof_delta_bytes is None else phase.aof_delta_bytes
        ),
    }


def _wall_columns(run: RedisRunMetrics) -> dict:
    """The ``dataset`` point's three walls, blank on every other point."""
    columns = {}
    for name in ("bgsave", "bgrewriteaof"):
        phase = run.wall(name)
        if phase is not None:
            columns[f"{name}_wall_s"] = tables.fmt(phase.wall_s, 3)
            columns[f"{name}_status"] = phase.status or ""
    restart = run.wall("restart_load")
    if restart is not None:
        columns["restart_load_wall_s"] = tables.fmt(restart.wall_s, 3)
        columns["restart_dbsize"] = (
            "" if run.restart_dbsize is None else run.restart_dbsize
        )
    return columns


def _error_column(run: RedisRunMetrics) -> str:
    """The run's failure record: the action's own, or the failed phases'."""
    if run.error:
        return str(run.error)
    return ", ".join(f"{name} failed" for name in run.phase_errors)


def export_points_csv(
    root: Path, summary_csv: Path, tests_csv: Optional[Path] = None
) -> List[RedisRunMetrics]:
    """Write the result tree as CSV and return the runs that produced metrics.

    One row per (config, arm, point, variant), the run phase's benchmark numbers
    and the ``dataset`` point's walls side by side, including leaves that
    produced no metrics (their ``status`` says so). ``tests_csv``, if given,
    takes the per-test breakdown of every benchmark phase.
    """
    runs, summary_rows, test_rows = [], [], []

    for config, arm, point, variant, path in discover_redis_runs(root):
        ident = {
            "config": config,
            "arm": arm,
            "point": point,
            "variant": variant or "",
        }
        parsed = parse_redis_json(path) if path.exists() else None
        if parsed is None:
            warnings.warn(f"No redis results in {path.parent}")
            summary_rows.append({**ident, "status": "no-results", "json": path})
            continue

        run = RedisRunMetrics(
            config=config,
            arm=arm,
            point=point,
            variant=variant,
            path=path,
            flags=read_flags_sidecar(path),
            **parsed,
        )
        summary_rows.append(
            {
                **ident,
                "status": run.status,
                **{k: v for k, v in run.flags.items() if k in SUMMARY_COLUMNS},
                **_point_columns(run),
                "server_start_s": tables.fmt(run.server_start_s, 3),
                **_fill_phase_columns(run),
                **_run_phase_columns(run),
                **_wall_columns(run),
                "error": _error_column(run),
                "json": path,
            }
        )
        runs.append(run)
        for phase_name in BENCHMARK_PHASES:
            phase = run.benchmark(phase_name)
            if phase is None:
                continue
            test_rows += [
                {
                    **ident,
                    "phase": phase_name,
                    "test": test.name,
                    "rps": tables.fmt(test.rps, 2),
                    "avg_ms": tables.fmt(test.avg_ms, 3),
                    "min_ms": tables.fmt(test.min_ms, 3),
                    "p50_ms": tables.fmt(test.p50_ms, 3),
                    "p95_ms": tables.fmt(test.p95_ms, 3),
                    "p99_ms": tables.fmt(test.p99_ms, 3),
                    "max_ms": tables.fmt(test.max_ms, 3),
                }
                for test in phase.tests.values()
            ]

    tables.write_csv(summary_csv, SUMMARY_COLUMNS, summary_rows)
    if tests_csv is not None:
        tables.write_csv(tests_csv, TESTS_COLUMNS, test_rows)

    return runs


def print_redis_table(runs: List[RedisRunMetrics]) -> None:
    """The per-run comparison table: one line per run.

    ops/s and the percentiles are the run phase's. The three wall columns are
    the ``dataset`` point's and blank everywhere else.
    """
    header = (
        f"{'point':<16} {'arm':<14} {'variant':<14} {'fsync':<9} "
        f"{'ops/s':>10} {'p50 ms':>8} {'p99 ms':>8} "
        f"{'bgsave':>8} {'rewrite':>8} {'restart':>8} "
        f"{'ack':<18} {'status':<10}"
    )
    print("\n" + header)
    print("-" * len(header))
    for run in sorted(runs, key=lambda r: (r.point, r.arm, r.variant or "")):
        phase = run.benchmark("run")
        test = (phase.primary if phase else None) or RedisTest("")

        def wall(name):
            w = run.wall(name)
            return tables.fmt(w.wall_s if w else None, 2)

        print(
            f"{run.point:<16} {run.arm:<14} {(run.variant or ''):<14} "
            f"{run.point_config.get('appendfsync', ''):<9} "
            f"{(test.rps or 0):>10,.0f} "
            f"{tables.fmt(test.p50_ms, 3):>8} {tables.fmt(test.p99_ms, 3):>8} "
            f"{wall('bgsave'):>8} {wall('bgrewriteaof'):>8} "
            f"{wall('restart_load'):>8} "
            f"{str(run.flags.get('ack', '')):<18} {run.status:<10}"
        )
