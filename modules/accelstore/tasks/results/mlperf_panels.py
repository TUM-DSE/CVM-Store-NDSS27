#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The MLPerf Storage half of the paper's application figure.

Draws nothing: owns the MLPerf result format — usable rows, module choice,
cell collection and averaging — and hands :mod:`plots.apps` the checkpointing
panel it draws, absolute GiB/s in a save and a load group. DLIO's
``..._GB_per_second`` fields are binary, so the figures say GiB. Missing cells
are named by :func:`missing_cells`, never drawn as zeros.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..crypto_backend import CryptoBackend
from . import discovery, mlperf_results
from .bench_results import POSIX_ARM_COLORS, POSIX_ARM_HATCHES
from ..plots.style import crypto_color

# Draw order: gridlines under the bars.
GRID_ZORDER = 1
BAR_ZORDER = 3

# Headroom above the tallest bar for its annotation.
AXIS_HEADROOM = 1.20
# Relative panels stay zero-based, so parity does not read as a visible gap.
RELATIVE_AXIS_MIN_TOP = 1.25


@dataclass(frozen=True)
class System:
    """One bar of every group: a storage system, and how it is drawn."""

    key: str
    label: str
    color: tuple
    hatch: str


SYSTEMS: Tuple[System, ...] = (
    System("cvm-store", "CVM-Store", crypto_color(CryptoBackend.AEGIS), ""),
    System(
        "rollbaccine",
        "Rollbaccine",
        POSIX_ARM_COLORS["rollbaccine"],
        POSIX_ARM_HATCHES["rollbaccine"],
    ),
    System(
        "luks",
        "LUKS (AES-XTS)",
        POSIX_ARM_COLORS["ext4-aes-xts-plain"],
        POSIX_ARM_HATCHES["ext4-aes-xts-plain"],
    ),
    System("ext4", "ext4", POSIX_ARM_COLORS["ext4"], POSIX_ARM_HATCHES["ext4"]),
)
SYSTEM_BY_KEY = {system.key: system for system in SYSTEMS}
BASELINE_KEY = "ext4"

# Each POSIX stack is drawn once per I/O mode, keyed by this system-key suffix.
CHECKPOINT_POSIX_ARMS: Dict[str, str] = {
    "posix-direct": "direct",
    "posix-buffered": "buffered",
}

IO_MODE_SEP = "/"


def io_mode_key(system: str, mode: Optional[str]) -> str:
    """One system measured through one I/O mode, as a cell key."""
    return f"{system}{IO_MODE_SEP}{mode}" if mode else system


def split_io_mode(key: str) -> Tuple[str, Optional[str]]:
    """A cell key back into (system, I/O mode); mode None where it carries none."""
    base, sep, mode = key.partition(IO_MODE_SEP)
    return (base, mode) if sep else (key, None)


def checkpoint_system_keys() -> Tuple[str, ...]:
    """Every system key the checkpointing panel can hold, in bar order."""
    modes = list(dict.fromkeys(CHECKPOINT_POSIX_ARMS.values()))
    keys = []
    for system in SYSTEMS:
        if system.key == "cvm-store":
            keys.append(system.key)
            continue
        keys.extend(io_mode_key(system.key, mode) for mode in modes)
    return tuple(keys)


# Rollbaccine is the f=1 colocated variant, because f=0 replicates nothing.
CHECKPOINT_STACK_SYSTEMS = {
    "ext4": "ext4",
    "ext4-aes-xts-plain": "luks",
    "rollbaccine-f1m": "rollbaccine",
}

# Which AccelStore module stands for CVM-Store when a config ran several.
MODULE_ORDER = ("log", "fs")

CHECKPOINT_DIRECTIONS = (("save", "save_gbs"), ("load", "load_gbs"))


@dataclass(frozen=True)
class Cell:
    """One (group, system) measurement: the bar's height and its error bar."""

    value: float
    stdev: float


@dataclass
class Panel:
    """One panel's groups in x order, its cells, and what it is about."""

    groups: List[str]
    cells: Dict[Tuple[str, str], Cell]
    caption: str
    # Relative panels only: the denominator every bar of a group is divided by.
    baselines: Optional[Dict[str, Cell]] = None
    # Which AccelStore module CVM-Store's bar was drawn from.
    module: Optional[str] = None
    # What each group ran, printed under the group's name.
    group_notes: Optional[Dict[str, str]] = None
    # Drawn once, centred, in the `group_notes` band.
    subtitle: Optional[str] = None


def _tree(root: Path) -> Path:
    """The run-mlperf tree under a results root, however the caller named it."""
    return discovery.resolve_result_tree(root, "mlperf")


def _system_key(
    row: dict,
    stack_systems: Dict[str, str],
    posix_arms: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """The system a row is a measurement of, or None for one this figure omits."""
    if row["arm_kind"] == "accelstore":
        return "cvm-store"
    system = stack_systems.get(row["stack"])
    if system is None or not posix_arms:
        return system
    mode = posix_arms.get(row["arm"])
    return io_mode_key(system, mode) if mode else None


def _usable(
    rows: Sequence[dict],
    config: str,
    notes: List[str],
    *,
    stack_systems: Dict[str, str],
    posix_arms: Optional[Dict[str, str]] = None,
) -> List[dict]:
    """The rows of one config that are a measurement of a drawable system."""
    kept = []
    for row in rows:
        if row["config"] != config:
            continue
        if _system_key(row, stack_systems, posix_arms) is None:
            known = stack_systems.get(row["stack"]) is not None
            why = "not a drawn I/O arm" if known else "not in this figure"
            notes.append(f"{why}: {row['arm']}/{row['stack']} ({row['model']})")
            continue
        if row["status"] != "ok":
            notes.append(
                f"dropped {row['arm']}/{row['stack']} {row['model']}:"
                f" status '{row['status']}'"
            )
            continue
        kept.append(row)
    return kept


def _module_for(
    rows: Sequence[dict],
    notes: List[str],
    family: str,
    requested: Optional[str] = None,
) -> Optional[str]:
    """The AccelStore module CVM-Store's bar is drawn from, of those that ran."""
    present = sorted({row["module"] for row in rows if row["arm_kind"] == "accelstore"})
    if not present:
        return None
    if requested and requested not in present:
        notes.append(
            f"{family}: no {requested} run; drawing what there is"
            f" ({', '.join(present)})"
        )
        requested = None
    chosen = requested or next((m for m in MODULE_ORDER if m in present), present[0])
    if len(present) > 1:
        notes.append(
            f"{family}: CVM-Store is the {chosen} module; also ran"
            f" {', '.join(m for m in present if m != chosen)}"
        )
    return chosen


def _collect(
    rows: Sequence[dict],
    group_of,
    value_key: str,
    stdev_key: str,
    module: Optional[str],
    notes: List[str],
    family: str,
    stack_systems: Dict[str, str] = None,
    posix_arms: Optional[Dict[str, str]] = None,
) -> Dict[Tuple[str, str], Cell]:
    """Fold rows into one cell per (group, system), averaging repeated runs."""
    samples: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    for row in rows:
        system = _system_key(row, stack_systems, posix_arms)
        if system == "cvm-store" and row["module"] != module:
            continue
        group = group_of(row)
        value = row.get(value_key)
        if group is None or value is None:
            continue
        samples.setdefault((group, system), []).append(
            (float(value), float(row.get(stdev_key) or 0.0))
        )

    cells: Dict[Tuple[str, str], Cell] = {}
    for (group, system), found in samples.items():
        runs = len(found)
        cells[(group, system)] = Cell(
            value=sum(value for value, _ in found) / runs,
            stdev=sum(stdev for _, stdev in found) / runs,
        )
        if runs > 1:
            notes.append(
                f"{family}: {system} {group} is the mean of {runs} runs"
                f" ({', '.join(f'{value:.3f}' for value, _ in found)})"
            )
    return cells


def _checkpoint_panel(
    rows: Sequence[dict],
    config: str,
    notes: List[str],
    module: Optional[str] = None,
) -> Optional[Panel]:
    """The absolute GiB/s panel: save and load of one checkpointing variant."""
    usable = _usable(
        rows,
        config,
        notes,
        stack_systems=CHECKPOINT_STACK_SYSTEMS,
        posix_arms=CHECKPOINT_POSIX_ARMS,
    )
    if not usable:
        return None
    variants = sorted({row["variant"] for row in usable})
    if len(variants) > 1:
        # One panel is one workload: rates compare only within one model.
        counts = {v: sum(1 for r in usable if r["variant"] == v) for v in variants}
        variant = max(variants, key=lambda v: (counts[v], v))
        notes.append(
            f"checkpointing: drawing variant '{variant}'; also present"
            f" {', '.join(v for v in variants if v != variant)}"
        )
        usable = [row for row in usable if row["variant"] == variant]
    module = _module_for(usable, notes, "checkpointing", module)

    cells: Dict[Tuple[str, str], Cell] = {}
    for direction, column in CHECKPOINT_DIRECTIONS:
        cells.update(
            _collect(
                usable,
                lambda _row, direction=direction: direction,
                column,
                f"{column}_stdev",
                module,
                notes,
                "checkpointing",
                stack_systems=CHECKPOINT_STACK_SYSTEMS,
                posix_arms=CHECKPOINT_POSIX_ARMS,
            )
        )
    model = usable[0]["model"] or usable[0]["variant"]
    return Panel(
        groups=[direction for direction, _ in CHECKPOINT_DIRECTIONS],
        cells=cells,
        caption=f"checkpointing ({model})",
        module=module,
        subtitle=_checkpoint_subtitle(usable),
    )


def _checkpoint_subtitle(rows: Sequence[dict]) -> str:
    """What was checkpointed: the model, its size in GiB, and the rank count."""

    def agreed(field, cast):
        found = {cast(r[field]) for r in rows if r.get(field) is not None}
        return found.pop() if len(found) == 1 else None

    model = agreed("model", str)
    size = agreed("checkpoint_size_gb", lambda v: round(float(v)))
    ranks = agreed("ranks", int)
    detail = []
    if size is not None:
        detail.append(f"{size} GiB")
    if ranks is not None:
        detail.append(f"{ranks} rank" + ("s" if ranks != 1 else ""))
    return ", ".join(part for part in (model or "", ", ".join(detail)) if part)


def load_panels(
    bench_dir: Path,
    *,
    config: Optional[str] = None,
    module: Optional[str] = None,
) -> Tuple[Optional[Panel], str, List[str]]:
    """The checkpointing panel of one machine config, plus notes about it.

    Without ``config`` the one with the most drawable cells wins. Without
    ``module``, :data:`MODULE_ORDER` decides which AccelStore module is drawn.
    """
    by_family, warnings = mlperf_results.parse_mlperf_tree(bench_dir, aggregate="all")
    notes = [f"NOTE: {warning}" for warning in warnings]

    configs = sorted(
        {row["config"] for rows in by_family.values() for row in rows}
        if config is None
        else {config}
    )
    if not configs:
        return None, "", notes

    best: Optional[Tuple[int, str, Panel, List[str]]] = None
    for name in configs:
        own: List[str] = []
        checkpoint = _checkpoint_panel(by_family["checkpointing"], name, own, module)
        filled = len(checkpoint.cells) if checkpoint else 0
        if best is None or filled > best[0]:
            best = (filled, name, checkpoint, own)
    _, name, checkpoint, own = best
    if config is None and len(configs) > 1:
        notes.append(
            f"config '{name}' has the most measured cells; also present:"
            f" {', '.join(c for c in configs if c != name)}"
        )
    return checkpoint, name, notes + own


def missing_cells(checkpoint: Optional[Panel]) -> List[str]:
    """Every (group, system) slot the tree holds no run for."""
    labels = {s.key: s.label for s in SYSTEMS}
    missing = []
    if checkpoint is None:
        return ["missing: the whole checkpointing panel — no usable run"]
    # Expected slots are per I/O mode, not per `SYSTEMS` entry.
    for group in checkpoint.groups:
        for key in checkpoint_system_keys():
            if (group, key) in checkpoint.cells:
                continue
            base, mode = split_io_mode(key)
            name = labels.get(base, base) + (f" ({mode})" if mode else "")
            missing.append(f"missing: checkpointing / {group} / {name}")
    return missing
