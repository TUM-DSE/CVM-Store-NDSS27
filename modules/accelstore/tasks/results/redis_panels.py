#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The Redis half of the application figure: relative SET throughput.

Owns the ``accelstore.run-redis`` result format and hands :mod:`apps` one
:class:`mlperf_panels.Panel` and draws no figure of its own. ext4 is the
denominator, so a point ext4 did not run has no group at all.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..block_stacks import ROLLBACCINE_F1M_STACK
from . import discovery, mlperf_panels, redis_results
from .mlperf_panels import Cell, Panel

#: The points the half draws, left to right.
POINTS: Tuple[str, ...] = (
    "always_c1",
    "always_c8p4",
    "everysec_c8p4",
    "everysec_c50p16",
)

#: Short labels for the group ticks. A point without an entry draws under its
#: own name.
POINT_LABELS: Dict[str, str] = {
    "always_c1": "alw c1",
    "always_c8p4": "alw c8",
    "everysec_c8p4": "sec c8",
    "everysec_c50p16": "sec c50",
    "dataset": "data",
}

#: Current and legacy directory names for the same stack. A leaf keeps the
#: name it was written with.
ROLLBACCINE_ARMS: Tuple[str, ...] = (ROLLBACCINE_F1M_STACK, "accelstore-replicate")

#: Not :data:`mlperf_panels.CHECKPOINT_STACK_SYSTEMS`: this half's Rollbaccine
#: bar is the f=1m configuration, not the other halves' f=0 one.
ARM_SYSTEMS: Dict[str, str] = {
    redis_results.ACCELSTORE_ARM: "cvm-store",
    "ext4": "ext4",
    "ext4-aes-xts-plain": "luks",
    **{arm: "rollbaccine" for arm in ROLLBACCINE_ARMS},
}

#: Arms that ran but are not this half's bar, each with the reason, which is
#: printed rather than left to a missing group.
EXCLUDED_ARMS: Dict[str, str] = {
    "rollbaccine": (
        "f=0, no replica and so no rollback resistance; this half draws"
        f" '{ROLLBACCINE_F1M_STACK}', the configuration that makes the freshness"
        " claim a redis barrier is measured against"
    ),
}

#: redis-benchmark's name for the test the points run, asked for by name so a
#: point that grows a second test still draws this one.
TEST = "SET"

#: The timed invocation. ``fill`` is the ``dataset`` point's untimed load.
PHASE = "run"


def tree(root: Path) -> Path:
    """The run-redis tree under a results root, however the caller named it."""
    return discovery.resolve_result_tree(root, "redis")


def _system_key(run: redis_results.RedisRunMetrics) -> Optional[str]:
    """The system this run is a measurement of, or None for one not drawn."""
    return ARM_SYSTEMS.get(run.arm)


def _rate(run: redis_results.RedisRunMetrics) -> Optional[float]:
    """The run phase's SET rate, or None where the run does not carry one."""
    phase = run.benchmark(PHASE)
    if phase is None:
        return None
    test = phase.test(TEST)
    return None if test is None else test.rps


def _usable(
    runs: Sequence[redis_results.RedisRunMetrics],
    config: str,
    notes: List[str],
) -> List[redis_results.RedisRunMetrics]:
    """The runs of one config that are a drawable measurement."""
    kept = []
    excluded = set()
    for run in runs:
        if run.config != config:
            continue
        if run.arm in EXCLUDED_ARMS:
            if run.arm not in excluded:
                excluded.add(run.arm)
                notes.append(
                    f"redis: not drawing arm '{run.arm}' — {EXCLUDED_ARMS[run.arm]}"
                )
            continue
        if _system_key(run) is None:
            notes.append(f"not in this figure: redis arm '{run.arm}' ({run.point})")
            continue
        if run.variant:
            notes.append(
                f"redis: skipping variant leaf {run.arm}/{run.point}/{run.variant}"
            )
            continue
        if run.status != "ok":
            notes.append(f"dropped redis {run.arm}/{run.point}: status '{run.status}'")
            continue
        if _rate(run) is None:
            notes.append(f"dropped redis {run.arm}/{run.point}: no {TEST} row")
            continue
        kept.append(run)
    return kept


def _panel(
    runs: Sequence[redis_results.RedisRunMetrics],
    config: str,
    notes: List[str],
    points: Sequence[str],
) -> Optional[Panel]:
    """The relative ops/s panel: one group per point, ext4 as denominator."""
    usable = _usable(runs, config, notes)
    if not usable:
        return None

    cells: Dict[Tuple[str, str], Cell] = {}
    sizes: Dict[str, set] = {}
    for run in usable:
        key = (run.point, _system_key(run))
        if key in cells:
            notes.append(f"redis: two runs for {key[0]}/{key[1]}; keeping the first")
            continue
        cells[key] = Cell(value=_rate(run), stdev=0.0)
        size = run.point_config.get("value_size")
        if size:
            sizes.setdefault(run.point, set()).add(int(size))

    found = {group for group, _ in cells}
    groups: List[str] = []
    baselines: Dict[str, Cell] = {}
    for point in points:
        if point not in found:
            notes.append(f"redis: no run for point '{point}' under {config}")
            continue
        baseline = cells.get((point, mlperf_panels.BASELINE_KEY))
        if baseline is None or not baseline.value:
            notes.append(f"redis: no ext4 baseline for '{point}' — group dropped")
            continue
        groups.append(point)
        baselines[point] = baseline
    if not groups:
        return None
    return Panel(
        groups=groups,
        cells=cells,
        caption="redis",
        baselines=baselines,
        group_notes={point: _size_note(sizes.get(point, set())) for point in groups},
    )


def _size_note(sizes: set) -> str:
    """The value size every arm of this point ran, or "? B" when they disagree."""
    if not sizes:
        return ""
    if len(sizes) > 1:
        return "? B"
    size = sizes.pop()
    return f"{size // 1024} KiB" if size >= 1024 else f"{size} B"


def load_panel(
    root: Path,
    *,
    config: Optional[str] = None,
    points: Sequence[str] = POINTS,
) -> Tuple[Optional[Panel], str, List[str]]:
    """The half's panel, its config name, and the notes worth printing.

    ``config`` picks the machine config; without one the config with the most
    drawable cells wins. ``points`` are the groups to draw, left to right.
    """
    redis_dir = tree(root)
    runs = redis_results.discover_redis_results(redis_dir)
    notes: List[str] = []
    if not runs:
        return None, "", [f"redis: no parseable runs under {redis_dir}"]

    configs = sorted({run.config for run in runs} if config is None else {config})
    best: Optional[Tuple[int, str, Optional[Panel], List[str]]] = None
    for name in configs:
        own: List[str] = []
        panel = _panel(runs, name, own, points)
        filled = len(panel.cells) if panel else 0
        if best is None or filled > best[0]:
            best = (filled, name, panel, own)
    _, name, panel, own = best
    if config is None and len(configs) > 1:
        notes.append(
            f"redis config '{name}' has the most measured cells; also present:"
            f" {', '.join(c for c in configs if c != name)}"
        )
    return panel, name, notes + own


def missing_cells(panel: Optional[Panel]) -> List[str]:
    """Every (group, system) slot the tree holds no run for, ext4 aside."""
    if panel is None:
        return ["missing: the whole redis panel — no usable run"]
    systems = tuple(
        s for s in mlperf_panels.SYSTEMS if s.key != mlperf_panels.BASELINE_KEY
    )
    return [
        f"missing: redis / {group} / {system.label}"
        for group in panel.groups
        for system in systems
        if (group, system.key) not in panel.cells
    ]
