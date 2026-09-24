#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read `run-ablation`'s result tree back as relative end-to-end runtime.

The runtime is the workload window between the client's WORKLOAD_START and
WORKLOAD_END markers, falling back to the log's wall clock. Each arm is
divided by the `abl-none` point of the same (config, client, crypto, workload,
I/O path), and by the AEGIS `abl-none` of the same workload for `relative_cif`.
"""

import concurrent.futures
import csv
import re
import statistics
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from modules.accelstore.tasks.ablation import (
    Ablation,
    is_ablation_component,
    parse_path_component,
)
from modules.accelstore.tasks.crypto_backend import CryptoBackend
from modules.accelstore.tasks.io_path import (
    IoPath,
    is_io_path_component,
    parse_path_component as parse_io_path_component,
)
from modules.accelstore.tasks.results import discovery
from modules.accelstore.tasks.results.trace_parser import load_or_parse_trace

_WALL_RE = re.compile(r"^Wall Clock Time \(START to END\): ([\d.]+) ms$", re.M)
_TOTAL_OPS_RE = re.compile(r"^Total Operations: (\d+)$", re.M)
_FAILED_RE = re.compile(r"^Failed Operations: (\d+)$", re.M)
# Only written when the runner recorded one, so absence is "not recorded".
_EXIT_RE = re.compile(r"^=== EXIT: (-?\d+) ===$", re.M)

RUNTIME_MISMATCH_TOLERANCE = 0.01  # 1%


@dataclass
class AblationRun:
    """One run of one arm: a `{ts}.log`, its `{ts}_trace.dump*`, or both."""

    path: Path
    stem: str
    log: Optional[Path] = None
    trace: Optional[Path] = None
    runtime_ms: Optional[float] = None
    runtime_source: str = "none"  # "trace" | "log" | "none"
    window_source: str = ""
    wall_ms: Optional[float] = None
    total_ops: Optional[int] = None
    failed_ops: Optional[int] = None
    exit_code: Optional[int] = None
    buckets: Dict[str, float] = field(default_factory=dict)
    status: str = "ok"


@dataclass
class AblationPoint:
    """One arm of one workload in one build, over however many repeats it has."""

    config: str
    client: str  # unprefixed
    crypto: CryptoBackend
    workload: str
    ablation: Ablation
    # Part of the identity: an arm's cost is what it costs on the path it ran on.
    io: IoPath = field(default_factory=IoPath)
    runs: List[AblationRun] = field(default_factory=list)

    @property
    def key(self) -> Tuple[str, str, CryptoBackend, str, str]:
        """What a point shares with its baseline, the I/O path as a driver name."""
        return (
            self.config,
            self.client,
            self.crypto,
            self.workload,
            self.io.spec or "spdk",
        )

    @property
    def ok_runs(self) -> List[AblationRun]:
        return [r for r in self.runs if r.status == "ok"]

    @property
    def status(self) -> str:
        """`ok` if anything is usable, else why not."""
        if self.ok_runs:
            return "ok"
        return self.runs[0].status if self.runs else "no-runs"

    def runtime_ms(self, aggregate: str = "median") -> Optional[float]:
        values = [r.runtime_ms for r in self.ok_runs if r.runtime_ms is not None]
        if not values:
            return None
        if aggregate == "latest":
            return values[-1]
        return statistics.median(values)

    def bucket_ms(self, name: str) -> Optional[float]:
        values = [r.buckets[name] for r in self.ok_runs if name in r.buckets]
        return statistics.median(values) if values else None


def parse_ablation_log(path: Path) -> dict:
    """The numbers a client log carries about its own run."""
    text = path.read_text(errors="replace")

    def _one(pattern, cast):
        match = pattern.search(text)
        return cast(match.group(1)) if match else None

    return {
        "wall_ms": _one(_WALL_RE, float),
        "total_ops": _one(_TOTAL_OPS_RE, int),
        "failed_ops": _one(_FAILED_RE, int),
        "exit_code": _one(_EXIT_RE, int),
    }


_trace_for = discovery.trace_dump_if_present


def _run_stems(leaf: Path) -> List[str]:
    """The runs in a leaf, newest last, identified by their timestamp."""
    stems = set()
    for log in leaf.glob("*.log"):
        stems.add(log.stem)
    for suffix in discovery.TRACE_SUFFIXES:
        for trace in leaf.glob(f"*{suffix}"):
            stems.add(trace.name[: -len(suffix)])
    # A sequence run (seal → unseal) has no single end-to-end window.
    kept = sorted(s for s in stems if "_seq" not in s)
    dropped = sorted(s for s in stems if "_seq" in s)
    if dropped:
        warnings.warn(f"{leaf}: skipping {len(dropped)} sequence run(s)")
    return kept


def load_run(leaf: Path, stem: str) -> AblationRun:
    """One run's runtime, its gate verdict, and the breakdown behind it."""
    log = leaf / f"{stem}.log"
    run = AblationRun(
        path=leaf,
        stem=stem,
        log=log if log.exists() else None,
        trace=_trace_for(leaf, stem),
    )

    if run.log is not None:
        for name, value in parse_ablation_log(run.log).items():
            setattr(run, name, value)

    if run.trace is not None:
        try:
            metrics = load_or_parse_trace(run.trace)
        except Exception as exc:  # noqa: BLE001 - a bad dump is a status, not a crash
            warnings.warn(f"{run.trace}: {exc}")
        else:
            run.runtime_ms = metrics.total_execution_time_us / 1000.0
            run.runtime_source = "trace"
            run.window_source = metrics.workload_window_source
            # `encryption` is the data path only: metadata AEAD/MAC sit in `metadata`.
            run.buckets = {
                "encryption": metrics.encryption.total_time_us / 1000.0,
                "metadata": metrics.metadata.total_time_us / 1000.0,
                "merkle": metrics.merkle.total_time_us / 1000.0,
                "tc": metrics.trusted_counter.total_time_us / 1000.0,
                "setup": metrics.setup_time_us / 1000.0,
                "teardown": metrics.teardown_time_us / 1000.0,
            }

    if run.runtime_ms is None and run.wall_ms is not None:
        run.runtime_ms = run.wall_ms
        run.runtime_source = "log"

    run.status = _gate(run)
    return run


def _load_run_task(task: Tuple[Path, str]) -> Tuple[AblationRun, List[str]]:
    """:func:`load_run` in a worker process, returning its warnings as data."""
    leaf, stem = task
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = load_run(leaf, stem)
    return run, [str(entry.message) for entry in caught]


def _load_runs(tasks: List[Tuple[Path, str]]) -> List[AblationRun]:
    """Every run of a sweep, in the order asked for, parsed in a process pool."""
    if len(tasks) < 2:
        return [load_run(leaf, stem) for leaf, stem in tasks]

    runs: List[AblationRun] = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for run, caught in executor.map(_load_run_task, tasks):
            for message in caught:
                warnings.warn(message)
            runs.append(run)
    return runs


def _gate(run: AblationRun) -> str:
    """Why a run does not contribute, or `ok`."""
    if run.exit_code not in (None, 0):
        return "exit-nonzero"
    if run.failed_ops:
        return "failed-ops"
    if run.runtime_ms is None:
        return "no-runtime"
    if run.runtime_source == "trace" and run.window_source != "markers":
        # The number is then the capture window, which moves with the tpoint mask.
        return "stale-window"
    if run.runtime_source == "trace" and run.wall_ms:
        drift = abs(run.runtime_ms - run.wall_ms) / run.wall_ms
        if drift > RUNTIME_MISMATCH_TOLERANCE:
            return "runtime-mismatch"
    return "ok"


def _io_path_dirs(workload_dir: Path) -> List[Tuple[IoPath, Path]]:
    """Where a workload's ablation arms live, with and without an `io-*` level."""
    found: List[Tuple[IoPath, Path]] = [(IoPath(), workload_dir)]
    for child in sorted(p for p in workload_dir.iterdir() if p.is_dir()):
        if is_io_path_component(child.name):
            found.append((parse_io_path_component(child.name), child))
    return found


def discover_ablation_points(
    bench_dir: Path,
    sweep_workloads: Optional[Dict[str, List[str]]] = None,
) -> List[AblationPoint]:
    """Find every ablation arm under a result tree, reading every run in a leaf.

    Walks `{config}/{client}/{workload}/[io-*/]abl-*`. `sweep_workloads` maps a
    client to its job files and restricts the walk. Without it, every workload
    that has an arm is returned.
    """
    wanted = (
        {
            client: {Path(j).stem for j in jobs}
            for client, jobs in sweep_workloads.items()
        }
        if sweep_workloads
        else None
    )

    points: List[AblationPoint] = []
    counts: List[int] = []  # runs per point, in the same order
    tasks: List[Tuple[Path, str]] = []
    for config_dir in sorted(p for p in bench_dir.glob("*") if p.is_dir()):
        for client_dir in sorted(p for p in config_dir.glob("*") if p.is_dir()):
            client, crypto = CryptoBackend.split_client_binary(client_dir.name)
            if wanted is not None and client not in wanted:
                continue

            for workload_dir in sorted(p for p in client_dir.glob("*") if p.is_dir()):
                if wanted is not None and workload_dir.name not in wanted[client]:
                    continue

                for io_arm, arms_dir in _io_path_dirs(workload_dir):
                    for leaf in sorted(p for p in arms_dir.iterdir() if p.is_dir()):
                        if not is_ablation_component(leaf.name):
                            continue
                        try:
                            arm = parse_path_component(leaf.name)
                        except ValueError as exc:
                            warnings.warn(f"{leaf}: {exc}")
                            continue

                        stems = _run_stems(leaf)
                        if not stems:
                            warnings.warn(f"{leaf}: no runs")
                            continue
                        points.append(
                            AblationPoint(
                                config=config_dir.name,
                                client=client,
                                crypto=crypto,
                                workload=workload_dir.name,
                                ablation=arm,
                                io=io_arm,
                                runs=[],
                            )
                        )
                        counts.append(len(stems))
                        tasks.extend((leaf, stem) for stem in stems)

    runs = _load_runs(tasks)
    offset = 0
    for point, count in zip(points, counts):
        point.runs = runs[offset : offset + count]
        offset += count
    return _merge_repeats(points)


def _merge_repeats(points: List[AblationPoint]) -> List[AblationPoint]:
    """Fold leaves that describe the same experiment into one point."""
    merged: Dict[Tuple, AblationPoint] = {}
    for point in points:
        identity = (*point.key, point.ablation)
        first = merged.get(identity)
        if first is None:
            merged[identity] = point
        else:
            first.runs.extend(point.runs)

    for point in merged.values():
        # A differing operation count means the job file changed between sweeps.
        ops = {r.total_ops for r in point.ok_runs if r.total_ops is not None}
        if len(ops) > 1:
            warnings.warn(
                f"{point.client}/{point.workload}/{point.ablation.label()}:"
                f" repeats disagree on operation count {sorted(ops)} --"
                " they are not runs of the same workload"
            )
    return list(merged.values())


def baselines(points: List[AblationPoint]) -> Dict[Tuple, AblationPoint]:
    """The unablated arm of each (config, client, crypto, workload)."""
    return {p.key: p for p in points if not p.ablation.tokens}


def _cif_key(point: AblationPoint) -> Tuple:
    return (
        point.config,
        point.client,
        CryptoBackend.AEGIS,
        point.workload,
        point.io.spec or "spdk",
    )


def relative_runtimes(
    points: List[AblationPoint], aggregate: str = "median"
) -> List[dict]:
    """One row per point: its runtime, its two denominators, and the ratios.

    `relative` is against the unablated run of the point's own build,
    `relative_cif` against the AEGIS unablated run of the same workload.
    `aggregate` is `median` or `latest`. A missing denominator blanks its ratio
    and marks the row rather than dropping it.
    """
    by_key = baselines(points)
    rows = []
    for point in sorted(
        points,
        key=lambda p: (
            p.config,
            p.workload,
            p.client,
            p.crypto.value,
            p.io.spec or "",
            p.ablation.label(),
        ),
    ):
        runtime = point.runtime_ms(aggregate)
        base = by_key.get(point.key)
        base_ms = base.runtime_ms(aggregate) if base else None
        cif = by_key.get(_cif_key(point))
        cif_ms = cif.runtime_ms(aggregate) if cif else None

        status = point.status
        if status == "ok" and base_ms is None:
            status = "no-baseline"

        def ratio(denominator):
            if runtime is None or not denominator:
                return None
            return runtime / denominator

        values = [r.runtime_ms for r in point.ok_runs if r.runtime_ms is not None]
        rows.append(
            {
                "config": point.config,
                "client": point.client,
                "crypto": point.crypto.value,
                "workload": point.workload,
                "arm": point.ablation.label(),
                "tokens": ",".join(point.ablation.tokens),
                "slug": point.ablation.path_component(),
                # A run from before this axis existed ran on the userspace driver.
                "io_path": point.io.spec or "spdk",
                "status": status,
                "runs": len(point.ok_runs),
                "runtime_ms": runtime,
                "runtime_ms_min": min(values) if values else None,
                "runtime_ms_max": max(values) if values else None,
                "runtime_source": (
                    point.ok_runs[0].runtime_source if point.ok_runs else "none"
                ),
                "window_source": (
                    point.ok_runs[0].window_source if point.ok_runs else ""
                ),
                "baseline_ms": base_ms,
                "relative": ratio(base_ms),
                "cif_baseline_ms": cif_ms,
                "relative_cif": ratio(cif_ms),
                "encryption_ms": point.bucket_ms("encryption"),
                "metadata_ms": point.bucket_ms("metadata"),
                "merkle_ms": point.bucket_ms("merkle"),
                "tc_ms": point.bucket_ms("tc"),
                "setup_ms": point.bucket_ms("setup"),
                "teardown_ms": point.bucket_ms("teardown"),
                "total_ops": point.runs[0].total_ops,
                "failed_ops": sum(r.failed_ops or 0 for r in point.runs),
                "path": ";".join(sorted({str(r.path) for r in point.runs})),
            }
        )
    return rows


SUMMARY_COLUMNS = [
    "config",
    "client",
    "crypto",
    "workload",
    "arm",
    "tokens",
    "slug",
    "io_path",
    "status",
    "runs",
    "runtime_ms",
    "runtime_ms_min",
    "runtime_ms_max",
    "runtime_source",
    "window_source",
    "baseline_ms",
    "relative",
    "overhead_pct",
    "cif_baseline_ms",
    "relative_cif",
    "overhead_cif_pct",
    "encryption_ms",
    "metadata_ms",
    "merkle_ms",
    "tc_ms",
    "setup_ms",
    "teardown_ms",
    "total_ops",
    "failed_ops",
    "path",
]


def _fmt(value, digits=2):
    """Blank, not zero, for what was never measured."""
    if value is None:
        return ""
    return f"{value:.{digits}f}" if isinstance(value, float) else value


def export_ablation_csv(rows: List[dict], csv_path: Path) -> None:
    """Write one row per arm, every arm, whatever its status."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key, source in (
                ("overhead_pct", "relative"),
                ("overhead_cif_pct", "relative_cif"),
            ):
                value = row.get(source)
                out[key] = None if value is None else (value - 1.0) * 100.0
            writer.writerow({k: _fmt(out.get(k), 3) for k in SUMMARY_COLUMNS})


def print_ablation_tables(rows: List[dict]) -> None:
    """The relative table, the absolute one behind it, and the trace breakdown."""
    if not rows:
        print("No ablation runs found.")
        return

    arms = []
    for row in rows:
        if row["arm"] not in arms:
            arms.append(row["arm"])
    arms.sort(key=lambda a: (a != "unablated", a))

    show_io = len({row["io_path"] for row in rows}) > 1
    io_head = f"{'I/O':<8}" if show_io else ""

    def io_cell(value):
        return f"{value:<8}" if show_io else ""

    groups: Dict[Tuple, Dict[str, dict]] = {}
    for row in rows:
        groups.setdefault((row["workload"], row["crypto"], row["io_path"]), {})[
            row["arm"]
        ] = row

    label = {b.value: b.label for b in CryptoBackend}
    width = max(len(w) for w, _, _ in groups) + 2

    print("\nEnd-to-end runtime relative to the unablated run of the same build")
    print(
        "(1.000 = that build's baseline; N/A = the arm does not exist in that"
        " build, a status = it ran but cannot be read as a ratio."
        " The last column is every row against the CIF baseline instead.)"
    )
    header = (
        f"{'Workload':<{width}}{'Build':<10}{io_head}"
        + "".join(f"{a:>14}" for a in arms)
        + f"{'| vs CIF':>14}"
    )
    print(header)
    print("─" * len(header))
    for (workload, crypto, io_arm), by_arm in sorted(groups.items()):
        cells = ""
        for arm in arms:
            row = by_arm.get(arm)
            if row is None:
                cell = "N/A"
            elif row["relative"] is None:
                cell = row["status"]
            else:
                cell = f"{row['relative']:.3f}"
            cells += f"{cell:>14}"
        # What CIF costs on this workload.
        base = by_arm.get("unablated")
        vs_cif = base["relative_cif"] if base else None
        cells += f"{'|':>3}{'N/A' if vs_cif is None else f'{vs_cif:.3f}':>11}"
        print(
            f"{workload:<{width}}{label.get(crypto, crypto):<10}"
            f"{io_cell(io_arm)}{cells}"
        )

    print("\nAbsolute end-to-end runtime")
    print(
        f"{'Workload':<{width}}{'Build':<10}{io_head}{'Arm':<16}{'n':>3}"
        f"{'median ms':>12}{'min':>12}{'max':>12}  status"
    )
    for row in rows:
        print(
            f"{row['workload']:<{width}}{label.get(row['crypto'], row['crypto']):<10}"
            f"{io_cell(row['io_path'])}{row['arm']:<16}{row['runs']:>3}"
            f"{_fmt(row['runtime_ms']):>12}{_fmt(row['runtime_ms_min']):>12}"
            f"{_fmt(row['runtime_ms_max']):>12}  {row['status']}"
        )
        if row["status"] == "no-baseline":
            # The baseline a bypass arm is missing is its own path's.
            on_path = f" --io-path {row['io_path']}" if show_io else ""
            print(
                f"  ⚠ no unablated run for {row['crypto']}/{row['workload']}"
                f" — run `accelstore.run-ablation --arms {row['crypto']}:none"
                f"{on_path}`"
            )

    print("\nWhere the trace says the time went (ms, median)")
    print(
        f"{'Workload':<{width}}{'Build':<10}{io_head}{'Arm':<16}"
        f"{'encrypt':>12}{'metadata':>12}{'merkle':>12}{'counter':>12}"
    )
    for row in rows:
        print(
            f"{row['workload']:<{width}}{label.get(row['crypto'], row['crypto']):<10}"
            f"{io_cell(row['io_path'])}{row['arm']:<16}{_fmt(row['encryption_ms']):>12}"
            f"{_fmt(row['metadata_ms']):>12}{_fmt(row['merkle_ms']):>12}"
            f"{_fmt(row['tc_ms']):>12}"
        )
