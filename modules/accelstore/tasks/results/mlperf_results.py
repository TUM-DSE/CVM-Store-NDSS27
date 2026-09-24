#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parsing for run-mlperf results (MLPerf Storage v3.0).

Every number comes out of mlpstorage's own ``summary.json``. The result tree
``actions/mlperf.py`` writes::

    bench-result/mlperf/{setup}/{family}/{variant}/{arm}/[stack-*]/[sync]/
        {ts}.log  {ts}_init.log  {ts}_flags.json  {ts}_tc.bin
        mlperf/{division}/{org}/results/{system}/{family}/{model}/[{phase}/]{ts}/summary.json

The leaf tokens are present only off their default, so ``{arm}`` sits at a
fixed depth, and unknown directories are reported rather than misread.

One CSV per family. Every declared point keeps a row (``no-summary`` for one
that produced none), and a row whose shape disagrees with its variant
siblings' is marked ``shape-mismatch``. Ratios are left to whatever draws a
figure.
"""

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..mlperf_variants import (
    FAMILIES,
    MlperfArm,
    MlperfStack,
    MlperfVariant,
    parse_arm_component,
    stack_of,
)
from ..mlperf_variants import parse_path_component as parse_variant_component
from ..sync_modes import SyncMode, is_sync_component
from ..sync_modes import parse_path_component as parse_sync_component

NON_LEAF_DIRS = frozenset({"mlperf", "precondition", "traces"})

IDENTITY_COLUMNS = [
    "config",
    "family",
    "division",
    "variant",
    "label",
    "arm",
    "arm_kind",
    "module",
    "stack",
    "sync",
    "model",
    "status",
    "run_ts",
    "wall_s",
    "hosts",
    "image_id",
    "device",
    "path",
]

CHECKPOINTING_COLUMNS = IDENTITY_COLUMNS + [
    "ranks",
    "checkpoint_size_gb",
    "save_gbs",
    "save_gbs_stdev",
    "save_s",
    "save_s_stdev",
    "load_gbs",
    "load_gbs_stdev",
    "load_s",
    "load_s_stdev",
]

FAMILY_COLUMNS = {
    "checkpointing": CHECKPOINTING_COLUMNS,
}

# Spelled out, not derived: a key renamed upstream surfaces as a blank column.
CHECKPOINTING_METRICS = {
    "save_gbs": "save_checkpoint_io_mean_GB_per_second",
    "save_gbs_stdev": "save_checkpoint_io_stdev_GB_per_second",
    "save_s": "save_checkpoint_duration_mean_seconds",
    "save_s_stdev": "save_checkpoint_duration_stdev_seconds",
    "load_gbs": "load_checkpoint_io_mean_GB_per_second",
    "load_gbs_stdev": "load_checkpoint_io_stdev_GB_per_second",
    "load_s": "load_checkpoint_duration_mean_seconds",
    "load_s_stdev": "load_checkpoint_duration_stdev_seconds",
    "checkpoint_size_gb": "checkpoint_size_GB",
}

# The summary fields that say what workload a run actually performed.
SHAPE_KEYS = ("checkpoint_size_GB",)


@dataclass(frozen=True)
class MlperfLeaf:
    """One measured point: a (config, variant, arm, stack, sync) directory."""

    config: str
    variant: MlperfVariant
    arm: MlperfArm
    stack: MlperfStack
    sync: SyncMode
    path: Path

    @property
    def key(self) -> Tuple:
        # The default sync mode has no slug, and "" keeps the key sortable.
        return (
            self.config,
            self.variant.id,
            self.arm.id,
            self.stack.id,
            self.sync.path_component() or "",
        )


@dataclass
class MlperfRun:
    """One mlpstorage run under a leaf, and what it reported."""

    leaf: MlperfLeaf
    summary_path: Path
    run_ts: str
    division: str
    model: str
    summary: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)
    status: str = "ok"

    @property
    def metric(self) -> dict:
        return self.summary.get("metric") or {}

    @property
    def shape(self) -> Tuple:
        return tuple(self.summary.get(key) for key in SHAPE_KEYS)


def _normalize_ts(text: str) -> str:
    """Both timestamp spellings as one comparable digit string."""
    return "".join(ch for ch in text if ch.isdigit())


def _leaves(
    config: str,
    variant: MlperfVariant,
    arm: MlperfArm,
    directory: Path,
    stack: MlperfStack,
    sync: SyncMode,
    warnings: List[str],
) -> List[MlperfLeaf]:
    """Walk an arm directory's optional stack and sync tokens down to points."""
    found: List[MlperfLeaf] = []
    if (directory / "mlperf").is_dir():
        found.append(MlperfLeaf(config, variant, arm, stack, sync, directory))
    for child in sorted(p for p in directory.iterdir() if p.is_dir()):
        if child.name in NON_LEAF_DIRS:
            continue
        if child.name.startswith("stack-"):
            try:
                child_stack = stack_of(child.name[len("stack-") :])
            except (KeyError, ValueError) as exc:
                warnings.append(f"{child}: {exc}")
                continue
            found += _leaves(config, variant, arm, child, child_stack, sync, warnings)
        elif is_sync_component(child.name):
            try:
                child_sync = parse_sync_component(child.name)
            except ValueError as exc:
                warnings.append(f"{child}: {exc}")
                continue
            found += _leaves(config, variant, arm, child, stack, child_sync, warnings)
        else:
            warnings.append(f"unknown directory under {directory}: {child.name}")
    return found


def discover_leaves(bench_dir: Path) -> Tuple[List[MlperfLeaf], List[str]]:
    """Every point under a run-mlperf tree, and what could not be decoded."""
    leaves: List[MlperfLeaf] = []
    warnings: List[str] = []
    if not bench_dir.is_dir():
        return leaves, [f"no such directory: {bench_dir}"]

    for config_dir in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
        for family_dir in sorted(p for p in config_dir.iterdir() if p.is_dir()):
            if family_dir.name not in FAMILIES:
                warnings.append(f"unknown family directory: {family_dir}")
                continue
            for variant_dir in sorted(p for p in family_dir.iterdir() if p.is_dir()):
                try:
                    variant = parse_variant_component(variant_dir.name)
                except ValueError as exc:
                    warnings.append(f"{variant_dir}: {exc}")
                    continue
                if variant.family != family_dir.name:
                    warnings.append(
                        f"{variant_dir}: variant is family '{variant.family}', "
                        f"filed under '{family_dir.name}'"
                    )
                    continue
                for arm_dir in sorted(p for p in variant_dir.iterdir() if p.is_dir()):
                    try:
                        arm = parse_arm_component(arm_dir.name)
                    except ValueError as exc:
                        warnings.append(f"{arm_dir}: {exc}")
                        continue
                    # An accelstore arm owns the NVMe through vfio, under no stack.
                    base_stack = stack_of("raw" if arm.is_accelstore else "ext4")
                    leaves += _leaves(
                        config_dir.name,
                        variant,
                        arm,
                        arm_dir,
                        base_stack,
                        SyncMode(),
                        warnings,
                    )
    return leaves, warnings


def _division_and_model(summary_path: Path) -> Tuple[str, str]:
    """The division and model a summary sits under, counted forwards from ``results``."""
    parts = summary_path.parts
    try:
        results = len(parts) - 1 - parts[::-1].index("results")
    except ValueError:
        return "", ""
    division = parts[results - 2] if results >= 2 else ""
    model = parts[results + 3] if len(parts) > results + 3 else ""
    return division, model


def _flags_for(leaf: MlperfLeaf, run_ts: str) -> dict:
    """The newest flags file stamped at or before this run."""
    stamp = _normalize_ts(run_ts)
    best: Optional[Tuple[str, Path]] = None
    for path in leaf.path.glob("*_flags.json"):
        own = _normalize_ts(path.name.split("_flags.json")[0])
        if own <= stamp and (best is None or own > best[0]):
            best = (own, path)
    if best is None:
        return {}
    try:
        return json.loads(best[1].read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _device(flags: dict) -> str:
    """The storage this run was given (accelstore ``device``, POSIX ``dev_path``)."""
    device = flags.get("device") or {}
    return device.get("device") or device.get("dev_path") or ""


def _wall_s(summary: dict) -> Optional[float]:
    """The measured window mlpstorage itself reports, in seconds."""
    start, end = summary.get("start"), summary.get("end")
    if not start or not end:
        return None
    try:
        span = datetime.fromisoformat(end) - datetime.fromisoformat(start)
    except (TypeError, ValueError):
        return None
    return span.total_seconds()


def _gate(run: MlperfRun) -> str:
    """What this run is, before anything is compared against it."""
    if not run.summary:
        return "unreadable-summary"
    if not run.metric:
        return "no-metric"
    failed = [
        phase.get("phase", "?")
        for phase in run.flags.get("phases", ())
        if phase.get("exit")
    ]
    if failed:
        return "phase-failed:" + ",".join(failed)
    return "ok"


def load_runs(leaves: Sequence[MlperfLeaf]) -> Tuple[List[MlperfRun], List[str]]:
    """Every mlpstorage run under every point, and a row for points that made none."""
    runs: List[MlperfRun] = []
    warnings: List[str] = []
    for leaf in leaves:
        found = sorted((leaf.path / "mlperf").rglob("summary.json"))
        if not found:
            runs.append(
                MlperfRun(
                    leaf=leaf,
                    summary_path=leaf.path,
                    run_ts="",
                    division="",
                    model="",
                    status="no-summary",
                )
            )
            continue
        for path in found:
            try:
                summary = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                warnings.append(f"{path}: {exc}")
                summary = {}
            division, model = _division_and_model(path)
            run_ts = path.parent.name
            run = MlperfRun(
                leaf=leaf,
                summary_path=path,
                run_ts=run_ts,
                division=division,
                model=model,
                summary=summary,
                flags=_flags_for(leaf, run_ts),
            )
            run.status = _gate(run)
            recorded = run.flags.get("sync_mode")
            if recorded and recorded != leaf.sync.label():
                warnings.append(
                    f"{path}: sits under sync policy '{leaf.sync.label()}' but its"
                    f" flags recorded '{recorded}'"
                )
            runs.append(run)
    return runs, warnings


def _latest_per_point(runs: Sequence[MlperfRun]) -> List[MlperfRun]:
    """The harness's usual latest-wins: one row per point, its newest run."""
    best: Dict[Tuple, MlperfRun] = {}
    for run in runs:
        key = run.leaf.key
        current = best.get(key)
        if current is None or _normalize_ts(run.run_ts) > _normalize_ts(current.run_ts):
            best[key] = run
    return sorted(best.values(), key=lambda run: run.leaf.key)


def _mark_shape_mismatches(runs: Sequence[MlperfRun]) -> None:
    """Mark rows whose arm did not run its variant's shape, a tie marking all of them."""
    groups: Dict[Tuple[str, str], List[MlperfRun]] = {}
    for run in runs:
        if run.status == "ok":
            groups.setdefault((run.leaf.config, run.leaf.variant.id), []).append(run)
    for members in groups.values():
        shapes = [member.shape for member in members]
        if len(set(shapes)) < 2:
            continue
        counts = {shape: shapes.count(shape) for shape in set(shapes)}
        top = max(counts.values())
        winners = [shape for shape, count in counts.items() if count == top]
        majority = winners[0] if len(winners) == 1 else None
        for member in members:
            if majority is None or member.shape != majority:
                member.status = "shape-mismatch"


def _identity_row(run: MlperfRun) -> dict:
    leaf = run.leaf
    return {
        "config": leaf.config,
        "family": leaf.variant.family,
        "division": run.division or leaf.variant.division,
        "variant": leaf.variant.id,
        "label": leaf.variant.label,
        "arm": leaf.arm.id,
        "arm_kind": leaf.arm.kind,
        "module": leaf.arm.module or "",
        "stack": leaf.stack.id,
        # The path's policy, not the sidecar's, which `load_runs` checks against it.
        "sync": leaf.sync.label(),
        "model": run.model or leaf.variant.model,
        "status": run.status,
        "run_ts": run.run_ts,
        "wall_s": _wall_s(run.summary),
        "hosts": run.summary.get("num_hosts"),
        "image_id": run.flags.get("image_id", ""),
        "device": _device(run.flags),
        "path": str(run.summary_path),
    }


def rows_for(runs: Sequence[MlperfRun], family: str) -> List[dict]:
    """The rows of one family's CSV."""
    rows: List[dict] = []
    for run in runs:
        if run.leaf.variant.family != family:
            continue
        row = _identity_row(run)
        metric = run.metric
        if family == "checkpointing":
            row["ranks"] = run.summary.get("num_accelerators")
            row.update({k: metric.get(v) for k, v in CHECKPOINTING_METRICS.items()})
            rows.append(row)
    return rows


def _fmt(value, digits=3):
    """Blank, not zero, for what was never measured."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return f"{value:.{digits}f}" if isinstance(value, float) else value


def export_family_csv(rows: Sequence[dict], family: str, csv_path: Path) -> None:
    """Write one family's table, every row it has, whatever its status."""
    columns = FAMILY_COLUMNS[family]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row.get(key)) for key in columns})


def _cell(value, digits=2, width=12):
    if value is None:
        text = ""
    elif isinstance(value, float):
        text = f"{value:.{digits}f}"
    else:
        text = str(value)
    return f"{text:>{width}}"


TABLE_HEADS = {
    "checkpointing": (
        (
            ("save GiB/s", "save_gbs", 2),
            ("save s", "save_s", 2),
            ("load GiB/s", "load_gbs", 2),
            ("load s", "load_s", 2),
            ("ckpt GB", "checkpoint_size_gb", 1),
        ),
        None,
    ),
}


def print_family_table(rows: Sequence[dict], family: str) -> None:
    """One table per family, one line per row, statuses included."""
    if not rows:
        return
    heads, note = TABLE_HEADS[family]
    arm_width = max(len(f"{r['arm']}/{r['stack']}") for r in rows) + 2
    var_width = max(len(r["variant"]) for r in rows) + 2

    print(f"\n{family}")
    header = (
        f"{'variant':<{var_width}}{'arm/stack':<{arm_width}}"
        + "".join(_cell(name) for name, _, _ in heads)
        + "  status"
    )
    print(header)
    print("─" * len(header))
    for row in sorted(rows, key=lambda r: (r["variant"], r["arm"], r["stack"])):
        cells = "".join(_cell(row.get(key), digits) for _, key, digits in heads)
        arm = f"{row['arm']}/{row['stack']}"
        print(
            f"{row['variant']:<{var_width}}{arm:<{arm_width}}{cells}"
            f"  {row['status']}"
        )
    if note:
        print(f"  ({note})")


def parse_mlperf_tree(
    bench_dir: Path,
    *,
    aggregate: str = "latest",
) -> Tuple[Dict[str, List[dict]], List[str]]:
    """Discover, load, gate and lay out a run-mlperf tree, family by family.

    ``aggregate`` is ``latest`` (one row per point, its newest run) or ``all``
    (every run a point ever produced).
    """
    leaves, warnings = discover_leaves(bench_dir)
    runs, load_warnings = load_runs(leaves)
    warnings += load_warnings
    if aggregate == "latest":
        runs = _latest_per_point(runs)
    else:
        runs = sorted(runs, key=lambda run: (run.leaf.key, _normalize_ts(run.run_ts)))
    _mark_shape_mismatches(runs)
    return {family: rows_for(runs, family) for family in FAMILIES}, warnings
