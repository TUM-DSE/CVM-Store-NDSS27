#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The synthetic-microbenchmark figure: three panels over one shared legend.

Each panel is two plots on one row, divided at the same x down the whole figure
(:data:`SPLIT_LEFT_IN`):

* (a) KVS, YCSB-A: run-phase op/s relative to RocksDB on ext4, beside the same
  runs' per-operation throughput, both at both thread counts.
* (b) FS and LOG over the size sweep, in absolute GiB/s, one plot per family.
  Read rates have the open segment taken out of their wall clock, see
  :func:`bench_read_gibs_less_open`.
* (c) FS random reads: IOPS relative to ext4, beside those runs' absolute read
  throughput.

The figure spans both columns of the paper's IEEEtran4 template, so it is laid
out at exactly :data:`FIG_WIDTH_IN` and saved without ``bbox_inches="tight"``.

A cell with no run is left empty and reported by name on the console, and
``--fill-missing`` marks it with a small x instead. A system with no run at all
in a panel gets no slot there (:func:`_panel_systems`). Where several runs cover
one cell the newest is taken (:func:`_pick_runs`) and printed with its timestamp
(:func:`print_bench_provenance`).

Drawn by ``accelstore.plot-synthetic``, which hands the task's arguments
straight to :func:`generate`.
"""

import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from core.tasks.plotting import common
from core.tasks.plotting.common import DenseHatchHandler

from ..crypto_backend import CryptoBackend
from ..results import bench_results, ycsb_results
from ..results.bench_results import POSIX_ARM_COLORS
from .style import crypto_color

FIG_WIDTH_IN = 504.0 / 72.0
FIG_HEIGHT_MAX_IN = 3.5

MIB = 1024.0 * 1024.0
GIB = 1024.0 * 1024.0 * 1024.0

DEFAULT_VALUE_SIZES = "16K,256K,1M,64M,256M,1G"
DEFAULT_THREADS = "1,8"

QUICK_VALUE_SIZES = "16K,1M,256M,1G"

C1_MIX = "A"  # the 50/50 read/update mix


@dataclass(frozen=True)
class System:
    """One series of the figure: a storage system, on one palette slot."""

    key: str
    label: str
    slot: int


SYSTEMS: Dict[str, System] = {
    s.key: s
    for s in (
        System("cvm_cif", "CVM-Store", 0),
        System("cvm_nocif", "CVM-Store w/o CIF", 1),
        System("ext4", "none", 3),
        System("rollbaccine", "Rollbaccine", 4),
    )
}

_LIGHT_FROM_SIBLINGS = {
    "cvm_cif": crypto_color(CryptoBackend.AEGIS),
    "cvm_nocif": crypto_color(CryptoBackend.NONE),
    "ext4": POSIX_ARM_COLORS["ext4"],
    "rollbaccine": POSIX_ARM_COLORS["rollbaccine"],
}
for _key, _colour in _LIGHT_FROM_SIBLINGS.items():
    assert tuple(common.PALETTE_PASTEL[SYSTEMS[_key].slot]) == tuple(_colour), _key

# ext4 is the ratio denominator, so it carries no bar in a rel block.
C1_SYSTEMS = ("cvm_cif", "cvm_nocif", "rollbaccine")
C1_BASELINE = "ext4"
C1_TPUT_SYSTEMS = ("cvm_cif", "cvm_nocif", "ext4", "rollbaccine")
C2_SYSTEMS = ("cvm_cif", "cvm_nocif", "ext4", "rollbaccine")
C3_SYSTEMS = ("cvm_cif", "cvm_nocif", "rollbaccine")
C3_TPUT_SYSTEMS = ("cvm_cif", "cvm_nocif", "ext4", "rollbaccine")
C3_BASELINE = "ext4"

DIRECTIONS = ("read", "write")

C1_GROUP_WIDTH = 0.90
C1_BLOCK_GAP = 0.018
C1_REL_BLOCKS = ("rel",)
C1_TPUT_BLOCKS = DIRECTIONS
C1_BLOCK_LABELS = {"read": "R", "write": "W"}
C1_HEADROOM_DECADES = 0.12
C2_DIRECTIONS = ("read", "write")

C2_FAMILY_GROUPS: Sequence[Tuple[str, Sequence[Tuple[str, str]]]] = (
    (
        "FS",
        (
            ("64M", "fs_write_read_64m"),
            ("256M", "fs_write_read_256m"),
            ("1G", "fs_write_read_1g"),
        ),
    ),
    (
        "LOG",
        (
            ("16K", "log_small_16k"),
            ("64K", "log_small_64k"),
            ("256K", "log_small_256k"),
            ("1M", "log_small_1m"),
            ("64M", "log_write_read_64mb"),
            ("256M", "log_write_read_256mb"),
            ("1G", "log_write_read_1gb"),
        ),
    ),
)

C2_FAMILY_GROUPS_QUICK: Sequence[Tuple[str, Sequence[Tuple[str, str]]]] = (
    (
        "FS",
        (
            ("64M", "fs_write_read_64m"),
            ("256M", "fs_write_read_256m"),
            ("1G", "fs_write_read_1g"),
        ),
    ),
    (
        "LOG",
        (
            ("16K", "log_small_16k"),
            ("1M", "log_small_1m"),
            ("256M", "log_write_read_256mb"),
            ("1G", "log_write_read_1gb"),
        ),
    ),
)

C2_GROUP_WIDTH = 0.90
C2_BLOCK_GAP = 0.02
C2_EDGE_PAD = 0.62  # in group pitches
C2_FAMILY_UNIT = {"FS": "file", "LOG": "entry"}
C2_BAND_COLOR = "0.92"
# Repeat count sets the stroke spacing: matplotlib strokes every 12/n pt.
THREAD_HATCH = ("", "\\" * 6, "\\" * 12)
# Denser than THREAD_HATCH, which a few-point swatch renders as one slash.
THREAD_HATCH_LEGEND = ("", "\\" * 10, "\\" * 14)

# fio bw_mean on the NVMe, in KiB/s.
C2_CEILING_KIBS = {"read": 14189969.4, "write": 6985415.1}
C2_CEILING_GIBS = {d: v / 1024.0**2 for d, v in C2_CEILING_KIBS.items()}
CEILING_LINEWIDTH = 0.4
CEILING_LABEL_PAD_PT = 0.5
C2_CEILING_STYLE = {
    "read": ("#00707a", (0, (4, 2))),
    "write": ("#a0157d", (0, (4, 1.6, 1, 1.6))),
}
CEILING_MARGIN_PAD_PT = 1.5

C3_GROUPS: Sequence[Tuple[str, str]] = (
    ("4K", "fs_random_read_4k"),
    ("16K", "fs_random_read_16k"),
    ("64K", "fs_random_read_64k"),
    ("256K", "fs_random_read_256k"),
    ("1M", "fs_random_read_1m"),
)

C3_GROUPS_QUICK: Sequence[Tuple[str, str]] = (
    ("4K", "fs_random_read_4k"),
    ("64K", "fs_random_read_64k"),
    ("1M", "fs_random_read_1m"),
)

C3_GROUP_WIDTH = 0.90
C3_HEADROOM_FRACTION = 0.12  # of the bars' own span
# (c)'s ratio axis starts here rather than at zero, where half of it is empty.
C3_FLOOR = 0.40
# A bar under C3_FLOOR drops the floor to a multiple of this, never clips it.
C3_FLOOR_STEP = 0.05
C3_RATIO_MINOR = 0.2
GIBS_NAMED = (0.01, 1.0, 10.0)
# Air over the device line, as a factor, for the label that sits on it.
C3_CEILING_HEADROOM = 1.9

# Layout, in inches against the fixed figure box.
LEFT_IN = 0.315
RIGHT_IN = 0.04
LEGEND_IN = 0.132
TAG_IN = 0.085
BOTTOM_PAD_IN = 0.03
PANEL_RIGHT_IN = 0.28
# Where the left plot of every row ends, sized by (a)'s row of absolutes.
SPLIT_LEFT_IN = 2.15
SPLIT_GAP_IN = LEFT_IN + 0.03
FIG_HEIGHT_IN = 2.267
BLOCK_ROW_IN = 0.074
BLOCK_LETTER_PAD_IN = 0.008
SPLIT_ROWS = frozenset({"c1", "c2", "c3"})
PANEL_LAYOUT: Sequence[Tuple[str, float, float]] = (
    # name, label strip under it, letter row over it
    ("c1", 0.152, BLOCK_ROW_IN),
    ("c2", 0.088, BLOCK_ROW_IN),
    ("c3", 0.152, BLOCK_ROW_IN),
)
PANEL_AXES_IN = (
    FIG_HEIGHT_IN
    - LEGEND_IN
    - BOTTOM_PAD_IN
    - sum(TAG_IN + row + label for _, label, row in PANEL_LAYOUT)
) / len(PANEL_LAYOUT)

# Axes height a decade needs before _log_ticks labels every one, not every second.
LOG_LABEL_PITCH_IN = 0.11

# Matplotlib's 1.0 pt default is one slab across a bar this narrow.
HATCH_LINEWIDTH = 0.4
# "base op/s" is as wide as the left margin takes, one character more is clipped.
BASELINE_ROW_IN = 0.095
BASELINE_ROW_LABEL = "base op/s"
MISSING_MARK_COLOR = "0.55"
BAR_ZORDER = 3
FRAME_ZORDER = BAR_ZORDER + 3
DIRECTION_NOTE = "higher is better \u2191"
BLOCK_LABEL_COLOR = "0.25"


def system_color(key: str):
    """Bar colour of a system: its pastel slot."""
    return common.PALETTE_PASTEL[SYSTEMS[key].slot]


def thread_hatch(level: int) -> str:
    """The hatch for thread level `level`, empty past what the key documents."""
    return THREAD_HATCH[level] if level < len(THREAD_HATCH) else ""


def thread_hatch_legend(level: int) -> str:
    """The same hatch at the density a legend swatch needs to show it."""
    return THREAD_HATCH_LEGEND[level] if level < len(THREAD_HATCH_LEGEND) else ""


PANEL_NAMES = {
    "c1": "(a) KVS YCSB-A",
    "c2": "(b) FS/LOG size sweep",
    "c3": "(c) FS random reads",
}


def _panel_systems(
    candidates: Sequence[str],
    present: set,
    panel: str,
    missing: "MissingCells",
) -> List[str]:
    """The systems a panel reserves bar slots for: those it has a run of."""
    drawn = [key for key in candidates if key in present]
    if not drawn:
        return list(candidates)
    for key in candidates:
        if key not in drawn:
            missing.add(panel, f"{SYSTEMS[key].label}: no run in this panel, no slot")
    return drawn


class MissingCells:
    """Every cell of the layout the data did not cover, by panel."""

    def __init__(self) -> None:
        self.rows: List[Tuple[str, str]] = []

    def add(self, panel: str, cell: str) -> None:
        self.rows.append((panel, cell))

    def mark(self, ax, x: float, fill: bool) -> None:
        """The ``--fill-missing`` mark: a small x on the baseline, never a bar."""
        if fill:
            ax.plot(
                [x],
                [0.0],
                transform=ax.get_xaxis_transform(),  # y=0 is off a log axis
                marker="x",
                markersize=1.7,
                markeredgewidth=0.35,
                color=MISSING_MARK_COLOR,
                clip_on=False,
                zorder=4,
            )

    def report(self) -> None:
        if not self.rows:
            print("\nEvery cell of the figure has a run behind it.")
            return
        print(f"\n{len(self.rows)} cells have no run and are left empty:")
        panel = None
        for name, cell in self.rows:
            if name != panel:
                panel = name
                print(f"  {PANEL_NAMES.get(name, name)}:")
            print(f"    - {cell}")


_SIZE_UNITS = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


def parse_size(token: str) -> int:
    """``256M`` -> bytes. Binary multipliers, as the harness names its points."""
    token = token.strip()
    if not token:
        raise ValueError("empty size")
    unit = _SIZE_UNITS.get(token[-1].upper())
    return int(token[:-1]) * unit if unit else int(token)


def parse_size_list(spec: str) -> List[Tuple[str, int]]:
    """``256K,16K`` -> [(label, bytes)] ascending, the labels as written."""
    parsed = [(t.strip(), parse_size(t)) for t in spec.split(",") if t.strip()]
    return sorted(parsed, key=lambda item: item[1])


def _fmt_si(value: float) -> str:
    """A rate in three characters or so, for a 4 pt annotation."""
    for scale, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if value >= scale:
            return f"{value / scale:.1f}{suffix}".replace(".0", "")
    return f"{value:.0f}"


def _gibs(mib_per_s: float) -> float:
    """MiB/s as the figure's GiB/s."""
    return mib_per_s * MIB / GIB


@dataclass(frozen=True)
class YcsbPoint:
    """One ycsb_client run, reduced to what panels (a) and (b) plot."""

    system: str
    mix: str
    value_size: int
    threads: int
    ops_per_s: float
    read_gibs: Optional[float]  # per-request, i.e. over service time
    write_gibs: Optional[float]
    path: Path


def _ycsb_system(path: Path, backend: Optional[str], flags: dict) -> Optional[str]:
    """Which series a ycsb run belongs to, or None for one outside the figure."""
    parts = [p.name for p in path.parents]
    binary = next((p for p in parts if p.endswith(ycsb_results.YCSB_CLIENT)), None)
    if backend == "accelstore" or binary:
        crypto = flags.get("crypto")
        if not crypto:
            _, parsed = CryptoBackend.split_client_binary(
                binary or ycsb_results.YCSB_CLIENT
            )
            crypto = parsed.value
        return {"aegis": "cvm_cif", "none": "cvm_nocif"}.get(crypto)
    stack = flags.get("stack") or next((p for p in parts if p in SYSTEMS), None)
    return stack if stack in SYSTEMS else None


def _ycsb_direction_rates(path: Path) -> Dict[str, Dict[str, float]]:
    """Per-phase ``{"read"/"write": {...}}`` blocks of a ycsb sidecar."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        phase.get("name", ""): {
            d: phase[d] for d in DIRECTIONS if isinstance(phase.get(d), dict)
        }
        for phase in data.get("phases", [])
    }


def _is_ycsb_sidecar(path: Path) -> bool:
    """Whether a JSON file is a ycsb_client sidecar, by what it says it is."""
    try:
        with open(path) as f:
            return json.load(f).get("client") == "ycsb"
    except (OSError, json.JSONDecodeError):
        return False


def _is_quick_leaf(path: Path) -> bool:
    """Whether a run landed in a `paper.* --quick` variant leaf."""
    return "quick" in path.parent.name.split("_")


def discover_ycsb_points(root: Path) -> List[YcsbPoint]:
    """Every ycsb run under ``bench-result/ycsb``, reduced to plottable points."""
    ycsb_dir = root / "bench-result" / "ycsb"
    if not ycsb_dir.is_dir():
        print(f"No ycsb results under {ycsb_dir}")
        return []

    points = []
    for path in sorted(ycsb_dir.rglob("*.json")):
        if path.name.endswith("_flags.json") or not _is_ycsb_sidecar(path):
            continue
        parsed = ycsb_results.parse_ycsb_json(path)
        if parsed is None:
            continue
        phase = (parsed["phases"] or {}).get("run")
        system = _ycsb_system(
            path, parsed.get("backend"), ycsb_results.read_flags_sidecar(path)
        )
        if phase is None or system is None:
            warnings.warn(
                f"Skipping {path}: no run phase, or an arm outside the figure"
            )
            continue
        if parsed.get("value_size") is None or parsed.get("threads") is None:
            warnings.warn(f"Skipping {path}: it records no resolved value size/threads")
            continue
        rates = _ycsb_direction_rates(path).get("run", {})
        points.append(
            YcsbPoint(
                system=system,
                mix=parsed.get("mix") or "",
                value_size=int(parsed["value_size"]),
                threads=int(parsed["threads"]),
                ops_per_s=phase.ops_per_s,
                read_gibs=_maybe_gibs(rates.get("read")),
                write_gibs=_maybe_gibs(rates.get("write")),
                path=path,
            )
        )
    return points


def _maybe_gibs(block: Optional[dict]) -> Optional[float]:
    if not block or block.get("per_request_mib_s") in (None, 0):
        return None
    return _gibs(float(block["per_request_mib_s"]))


def index_ycsb(points: Sequence[YcsbPoint], mix: str) -> Dict[tuple, YcsbPoint]:
    """``(system, value_size, threads) -> point`` for one mix."""
    return {(p.system, p.value_size, p.threads): p for p in points if p.mix == mix}


def _bench_system(arm: str) -> Optional[str]:
    """Which series a bench_client arm directory belongs to."""
    if arm.endswith(bench_results.BENCH_CLIENT):
        _, crypto = CryptoBackend.split_client_binary(arm)
        return {
            CryptoBackend.AEGIS: "cvm_cif",
            CryptoBackend.NONE: "cvm_nocif",
        }.get(crypto)
    return arm if arm in SYSTEMS else None


def discover_bench_runs(root: Path) -> List[bench_results.BenchRunMetrics]:
    """Gated bench_client runs whose arm is one of the figure's systems."""
    bench_dir = root / "bench-result" / "bench"
    if not bench_dir.is_dir():
        print(f"No bench results under {bench_dir}")
        return []
    runs = bench_results.discover_bench_results(bench_dir)
    return [
        r
        for r in bench_results.gate_bench_results(runs)
        if _bench_system(r.arm) is not None
    ]


def _split_loops(run: bench_results.BenchRunMetrics) -> bool:
    """Whether the run divided its iteration bounds across its workers."""
    return run.banner.get("split-loops") not in (None, "", "0")


def _newest_first(run: bench_results.BenchRunMetrics) -> str:
    """Sort key that puts the later run first, a log's stem being its timestamp."""
    return run.path.stem


def _pick_runs(
    runs: Sequence[bench_results.BenchRunMetrics], threads: int
) -> List[bench_results.BenchRunMetrics]:
    """One run per ``(personality, system)``: the newest of the candidates."""
    index: Dict[tuple, bench_results.BenchRunMetrics] = {}
    for run in sorted(runs, key=_newest_first, reverse=True):
        key = (run.personality, _bench_system(run.arm))
        kept = index.setdefault(key, run)
        if kept is run:
            continue
        detail = (
            f"{key[1]} @ {run.personality}: two {threads}-thread runs, "
            f"split-loops={int(_split_loops(kept))} ({kept.path.stem}) and "
            f"split-loops={int(_split_loops(run))} ({run.path.stem})"
        )
        if threads > 1 and _split_loops(run) != _split_loops(kept):
            warnings.warn(
                f"{detail} — above one worker a split bound is a total and an "
                "unsplit one is per worker, so those are not the same workload; "
                "plotting the newer one and saying so"
            )
        else:
            print(f"{detail}; plotting the newer one")
    return list(index.values())


def index_bench(
    runs: Sequence[bench_results.BenchRunMetrics],
    personalities: Sequence[str],
    threads: int,
) -> Dict[tuple, bench_results.BenchRunMetrics]:
    """``(personality, system) -> run`` at one thread count, arms paired."""
    slice_ = [
        r for r in runs if r.personality in personalities and r.threads == threads
    ]
    return {
        (r.personality, _bench_system(r.arm)): r
        for r in bench_results.enforce_arm_parity(_pick_runs(slice_, threads))
    }


# How far apart the arms of one group may be taken and still be one measurement.
VINTAGE_SPAN = timedelta(hours=24)


def _run_time(run: bench_results.BenchRunMetrics) -> datetime:
    """When a run was taken; a log's stem is its timestamp."""
    return datetime.strptime(run.path.stem, "%Y-%m-%d-%H-%M-%S")


def print_bench_provenance(
    panel: str,
    groups: Sequence[Tuple[str, str]],
    indexed: Dict[int, Dict[tuple, bench_results.BenchRunMetrics]],
    threads: Sequence[int],
    systems: Sequence[str],
) -> None:
    """The run behind every bar of a bench panel, by its timestamp."""
    print(f"\n{PANEL_NAMES.get(panel, panel)} — the run behind each bar")
    header = f"{'personality':<22}{'size':<6}{'T':>3}" + "".join(
        f"{s:>21}" for s in systems
    )
    print(header)
    print("-" * len(header))
    for label, personality in groups:
        for t in threads:
            runs = {s: indexed.get(t, {}).get((personality, s)) for s in systems}
            cells = "".join(
                f"{(r.path.stem if r is not None else '-'):>21}" for r in runs.values()
            )
            print(f"{personality:<22}{label:<6}{t:>3}{cells}")
            taken = [_run_time(r) for r in runs.values() if r is not None]
            if taken and max(taken) - min(taken) > VINTAGE_SPAN:
                detail = ", ".join(
                    f"{s} {r.path.stem}" for s, r in runs.items() if r is not None
                )
                warnings.warn(
                    f"{PANEL_NAMES.get(panel, panel)} {personality} at {t} threads "
                    f"pairs runs more than {VINTAGE_SPAN} apart ({detail}); re-run "
                    "the older arms so the group is one vintage"
                )


def _direction_phase(
    run: bench_results.BenchRunMetrics, direction: str
) -> Optional[bench_results.BenchPhase]:
    """The phase that moved bytes in `direction`, or None if none did."""
    bytes_of = (
        (lambda p: p.written_mib) if direction == "write" else (lambda p: p.read_mib)
    )
    best = max(run.phases, key=bytes_of, default=None)
    return best if best is not None and bytes_of(best) > 0 else None


def bench_gibs(run: bench_results.BenchRunMetrics, direction: str) -> Optional[float]:
    """GiB/s of `run`'s `direction` phase: its bytes over its own wall clock."""
    phase = _direction_phase(run, direction)
    if phase is None:
        return None
    return _gibs(phase.write_mibs if direction == "write" else phase.read_mibs)


def _open_wall_ms(block: dict) -> float:
    """The largest single worker's `open` time in a phase, in ms."""
    per_worker = [
        (w.get("kinds", {}).get("open", {}).get("avg_ms") or 0.0)
        * w.get("kinds", {}).get("open", {}).get("count", 0)
        for w in block.get("workers", [])
    ]
    if per_worker:
        return max(per_worker)
    kind = block.get("kinds", {}).get("open") or {}
    return (kind.get("avg_ms") or 0.0) * kind.get("count", 0)


def bench_read_gibs_less_open(
    run: bench_results.BenchRunMetrics,
) -> Optional[float]:
    """GiB/s of `run`'s read phase with the open segment out of the wall."""
    plain = bench_gibs(run, "read")
    if plain is None:
        return None
    try:
        data = json.loads(run.path.with_suffix(".json").read_text())
    except (OSError, json.JSONDecodeError):
        return plain
    blocks = [b for b in data.get("blocks", []) if b.get("bytes_read", 0) > 0]
    if not blocks:
        return plain
    block = max(blocks, key=lambda b: b["bytes_read"])
    wall_ms = block["wall_ms"] - _open_wall_ms(block)
    if wall_ms <= 0:
        return plain
    return (block["bytes_read"] / GIB) / (wall_ms / 1000.0)


def bench_read_ops(run: bench_results.BenchRunMetrics) -> Optional[float]:
    """IOPS of `run`'s read phase, which is what panel (c) is measured in."""
    phase = _direction_phase(run, "read")
    return phase.ops_per_s if phase is not None else None


def _style_panel(ax, ylabel: str) -> None:
    """The house axis style, with the x grid off and the frame opened up."""
    common.apply_axis_style(ax, ylabel=ylabel)
    ax.set_ylabel(ylabel, fontsize=common.TICKS_FONTSIZE + 0.5, labelpad=1.5)
    ax.grid(False, axis="x")
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.tick_params(axis="y", pad=1.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    _raise_frame(ax)


def _raise_frame(ax) -> None:
    """Put the axes' spines over its bars, and leave its grid under them."""
    for spine in ax.spines.values():
        spine.set_zorder(FRAME_ZORDER)


def _bar(ax, x: float, value: float, width: float, color, hatch: str = "") -> None:
    ax.bar(
        x,
        value,
        width,
        color=color,
        hatch=hatch,
        edgecolor=common.BAR_EDGECOLOR,
        linewidth=common.BAR_LINEWIDTH,
        alpha=common.BAR_ALPHA,
        zorder=BAR_ZORDER,
    )


def _set_groups(ax, labels: Sequence[str], pitch: float) -> None:
    """Group ticks at `pitch` apart, one label each, and the x limits to match."""
    ax.set_xticks([g * pitch for g in range(len(labels))])
    ax.set_xticklabels(labels, fontsize=common.TICKS_FONTSIZE)
    half = 0.5 * pitch
    ax.set_xlim(-half, (len(labels) - 1) * pitch + half)


def _annotate_baseline(
    ax, xs: Sequence[float], texts: Sequence[str], row_label: str
) -> List:
    """The denominator's absolute value under each tick, and the texts it drew."""
    height_in = ax.get_position().height * ax.get_figure().get_figheight()
    y = -BASELINE_ROW_IN / height_in
    ax.text(
        -0.012,
        y,
        row_label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=common.ANNOTATION_SIZE - 0.5,
        color="dimgray",
    )
    return [
        ax.text(
            x,
            y,
            text,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=common.ANNOTATION_SIZE,
            color="dimgray",
        )
        for x, text in zip(xs, texts)
    ]


def _check_baseline_row_fits(ax, drawn: Sequence) -> None:
    """Warn when the row of absolutes is wider than the group it sits under."""
    if not drawn:
        return
    fig = ax.get_figure()
    renderer = fig.canvas.get_renderer()
    widest = max(t.get_window_extent(renderer).width for t in drawn) / fig.dpi
    pitch_in = ax.get_position().width * fig.get_figwidth() / len(drawn)
    if widest > pitch_in:
        warnings.warn(
            f"the row of absolutes is {widest:.2f} in wide against a"
            f" {pitch_in:.2f} in group: widen SPLIT_LEFT_IN"
        )


def _hairline(ax) -> None:
    """The 1.0 line a ratio plot is read against, ruled across it."""
    ax.axhline(1.0, color="0.30", linewidth=0.55, zorder=2, dashes=(3, 2))


def _c1_bar_slots(
    levels: int, systems: Sequence[str], blocks: Sequence[str]
) -> Tuple[float, List[Tuple[str, str, int, float]]]:
    """Bar width and every bar's ``(block, system, thread level, offset)``."""
    members = [
        (
            block,
            (
                [key for key in systems if key != C1_BASELINE]
                if block == "rel"
                else list(systems)
            ),
        )
        for block in blocks
    ]
    count = sum(len(block_systems) for _, block_systems in members) * levels
    width = (C1_GROUP_WIDTH - C1_BLOCK_GAP * (len(members) - 1)) / count
    slots: List[Tuple[str, str, int, float]] = []
    x = -C1_GROUP_WIDTH / 2.0
    for index, (block, block_systems) in enumerate(members):
        if index:
            x += C1_BLOCK_GAP
        for level in range(levels):
            for system in block_systems:
                slots.append((block, system, level, x + width / 2.0))
                x += width
    return width, slots


def _c1_block_span(slots, block: str, width: float) -> Tuple[float, float]:
    """A block's extent in group-relative x, padded by most of a bar."""
    xs = [x for name, _, _, x in slots if name == block]
    return min(xs) - width * 0.7, max(xs) + width * 0.7


def _c1_read_band(ax, slots, pitch: float, groups: int, width: float) -> None:
    """The grey ground under the per-op plot's read block, as in (b) and (c)."""
    low, high = _c1_block_span(slots, "read", width)
    for g in range(groups):
        ax.axvspan(
            g * pitch + low,
            g * pitch + high,
            facecolor=C2_BAND_COLOR,
            linewidth=0,
            zorder=0,
        )


def _c1_ceilings(ax_tput) -> None:
    """(b)'s device bandwidth lines, on (a)'s per-op axis, and the pair named."""
    _ceilings(ax_tput, sides={"read": "bottom", "write": "top"})


def _subtitle(ax, text: str) -> None:
    """What this plot draws, centred over its own frame."""
    ax.text(
        0.5,
        1.0 + BLOCK_LETTER_PAD_IN / PANEL_AXES_IN,
        text,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=common.ANNOTATION_SIZE + 0.5,
        color=BLOCK_LABEL_COLOR,
        clip_on=False,
        zorder=FRAME_ZORDER,
    )


def _block_letters(ax, spans: Sequence[Tuple[str, float, float]]) -> None:
    """Name each block over the first group, just above the frame."""
    for text, low, high in spans:
        ax.text(
            (low + high) / 2.0,
            1.0 + BLOCK_LETTER_PAD_IN / PANEL_AXES_IN,
            text,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=common.ANNOTATION_SIZE,
            color=BLOCK_LABEL_COLOR,
            clip_on=False,
            zorder=FRAME_ZORDER,
        )


def _log_decade_limits(values: Sequence[float]) -> Tuple[float, float, float]:
    """``(bottom, top, decades)`` enclosing `values`."""
    finite = [v for v in values if v and v > 0]
    if not finite:
        return 0.1, 10.0, 2.0
    bottom = 10.0 ** math.floor(math.log10(min(finite)))
    top = 10.0 ** (math.log10(max(finite)) + C1_HEADROOM_DECADES)
    return bottom, top, math.log10(top / bottom)


def _fmt_decade(value: float, _pos=None) -> str:
    """A decade tick in as few characters as a 5 pt label can spare."""
    if value <= 0:
        return ""
    exponent = round(math.log10(value))
    if -3 <= exponent < 0:
        return f"{value:.{-exponent}f}".lstrip("0")
    if exponent < 0:
        return f"1e{exponent}"
    return _fmt_si(value)


def _fmt_pct(value: float, _pos=None) -> str:
    """A ratio as a percentage, in as few characters as a 5 pt label can spare."""
    if value < 0:
        return ""
    if value == 0:
        return "0"
    pct = value * 100.0
    if pct >= 1000.0:
        return f"{_fmt_si(pct)}%"
    if pct >= 1.0:
        return f"{pct:.0f}%"
    return f"{pct:g}%"


def _log_ticks(ax, bottom: float, formatter=_fmt_decade) -> None:
    """Decade gridlines, with a label on as many of them as the height admits."""
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=12))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    low, high = ax.get_ylim()
    height_in = ax.get_position().height * ax.get_figure().get_figheight()
    decades = max(math.log10(high / low), 1e-9)
    step = 1 if height_in / decades >= LOG_LABEL_PITCH_IN else 2
    base = round(math.log10(bottom))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda value, pos: (
                formatter(value, pos)
                if value > 0 and (round(math.log10(value)) - base) % step == 0
                else ""
            )
        )
    )


def plot_c1(
    ax,
    ax_tput,
    indexed: Dict[tuple, YcsbPoint],
    sizes: Sequence[Tuple[str, int]],
    threads: Sequence[int],
    systems: Sequence[str],
    missing: MissingCells,
    fill_missing: bool,
) -> str:
    """Draw (a) on `ax` (ratio) and `ax_tput` (per-op GiB/s), and return its tag."""
    pitch = 1.0
    rel_width, rel_slots = _c1_bar_slots(len(threads), systems, C1_REL_BLOCKS)
    tput_width, tput_slots = _c1_bar_slots(len(threads), systems, C1_TPUT_BLOCKS)
    _c1_read_band(ax_tput, tput_slots, pitch, len(sizes), tput_width)

    absolutes: List[str] = []
    drawn: Dict[str, List[float]] = {"rel": [], "tput": []}
    for g, (label, size) in enumerate(sizes):
        base = [indexed.get((C1_BASELINE, size, t)) for t in threads]
        absolutes.append("/".join(_fmt_si(b.ops_per_s) if b else "--" for b in base))
        for _, system, level, offset in rel_slots:
            x = g * pitch + offset
            point = indexed.get((system, size, threads[level]))
            cell = f"{SYSTEMS[system].label} @ {label}, {threads[level]}T"
            baseline = base[level]
            if point is None or baseline is None or baseline.ops_per_s <= 0:
                reason = "no run" if point is None else "no RocksDB/ext4 baseline"
                missing.add("c1", f"{cell}: {reason}")
                missing.mark(ax, x, fill_missing)
                continue
            value = point.ops_per_s / baseline.ops_per_s
            drawn["rel"].append(value)
            _bar(ax, x, value, rel_width, system_color(system), thread_hatch(level))
        for block, system, level, offset in tput_slots:
            x = g * pitch + offset
            point = indexed.get((system, size, threads[level]))
            cell = f"{SYSTEMS[system].label} @ {label}, {threads[level]}T"
            value = None if point is None else getattr(point, f"{block}_gibs")
            if value is None:
                reason = "no run" if point is None else f"no {block} service time"
                missing.add("c1", f"{cell}, per-op {block}: {reason}")
                missing.mark(ax_tput, x, fill_missing)
                continue
            drawn["tput"].append(value)
            _bar(
                ax_tput,
                x,
                value,
                tput_width,
                system_color(system),
                thread_hatch(level),
            )

    # The ratio axis has to hold the hairline whether or not a bar reaches it.
    left = _log_decade_limits(drawn["rel"] + [1.0])[:2]
    right = _log_decade_limits(drawn["tput"] + list(C2_CEILING_GIBS.values()))[:2]
    ax.set_yscale("log")
    ax.set_ylim(*left)
    _log_ticks(ax, left[0], _fmt_pct)
    ax_tput.set_yscale("log")
    ax_tput.set_ylim(*right)
    _gibs_ticks(ax_tput, right[0], "c1")

    _hairline(ax)
    _block_letters(
        ax_tput,
        [
            (C1_BLOCK_LABELS[block], *_c1_block_span(tput_slots, block, tput_width))
            for block in C1_TPUT_BLOCKS
        ],
    )
    for axes in (ax, ax_tput):
        _set_groups(axes, [label for label, _ in sizes], pitch)
    _subtitle(ax, "run-phase op/s, relative to RocksDB on ext4")
    _subtitle(ax_tput, "per-operation throughput")
    _c1_ceilings(ax_tput)
    _check_baseline_row_fits(
        ax,
        _annotate_baseline(
            ax,
            [g * pitch for g in range(len(sizes))],
            absolutes,
            BASELINE_ROW_LABEL,
        ),
    )
    _style_panel(ax, "rel. op/s")
    _style_panel(ax_tput, "GiB/s per op")
    return "(a) KVS, YCSB-A: 50% read / 50% update"


def _c2_bar_slots(
    levels: int, systems: Sequence[str]
) -> Tuple[float, List[Tuple[str, str, int, float]]]:
    """Bar width and every bar's (direction, system, thread level, offset)."""
    count = len(C2_DIRECTIONS) * len(systems) * levels
    width = (C2_GROUP_WIDTH - C2_BLOCK_GAP) / count
    span = count * width + C2_BLOCK_GAP
    slots: List[Tuple[str, str, int, float]] = []
    x = -span / 2.0
    for index, direction in enumerate(C2_DIRECTIONS):
        if index:
            x += C2_BLOCK_GAP
        for level in range(levels):
            for system in systems:
                slots.append((direction, system, level, x + width / 2.0))
                x += width
    return width, slots


def _c2_x_limits(count: int) -> Tuple[float, float]:
    """A family plot's x limits: `count` groups a unit apart, plus its edges."""
    return -C2_EDGE_PAD, (count - 1) + C2_EDGE_PAD


def _c2_bands(ax, count: int, width: float, slots) -> None:
    """A band behind every read block, so the blocks never merge."""
    reads = [x for direction, _, _, x in slots if direction == "read"]
    for group in range(count):
        ax.axvspan(
            group + min(reads) - width * 0.7,
            group + max(reads) + width * 0.7,
            facecolor=C2_BAND_COLOR,
            linewidth=0,
            zorder=0,
        )


def _ceilings(
    ax,
    sides: Optional[Dict[str, str]] = None,
    compact: bool = False,
    directions: Sequence[str] = DIRECTIONS,
    labelled: bool = True,
) -> None:
    """The device's fio bandwidth, one dashed line per direction, labelled.

    `directions` picks the lines, `compact` prints the value without the
    direction, `sides` maps a direction to ``"bottom"`` or ``"top"`` to push its
    label off its own line, and `labelled` False draws the lines alone.
    """
    for direction in directions:
        y = C2_CEILING_GIBS[direction]
        color, dashes = C2_CEILING_STYLE[direction]
        ax.axhline(
            y,
            color=color,
            linestyle=dashes,
            linewidth=CEILING_LINEWIDTH,
            zorder=BAR_ZORDER + 1,
        )
        if not labelled:
            continue
        text = f"{y:.1f}" if compact else f"{y:.1f} {direction}"
        va = (sides or {}).get(direction, "center")
        offset = {"bottom": CEILING_LABEL_PAD_PT, "top": -CEILING_LABEL_PAD_PT}
        ax.annotate(
            text,
            xy=(1.0, y),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            xytext=(CEILING_MARGIN_PAD_PT, offset.get(va, 0.0)),
            ha="left",
            va=va,
            fontsize=common.ANNOTATION_SIZE,
            color=color,
            annotation_clip=False,
            zorder=BAR_ZORDER + 2,
        )


def plot_c2(
    axes,
    indexed: Dict[int, Dict[tuple, bench_results.BenchRunMetrics]],
    threads: Sequence[int],
    systems: Sequence[str],
    missing: MissingCells,
    fill_missing: bool,
    groups_table: Sequence[Tuple[str, Sequence[Tuple[str, str]]]] = C2_FAMILY_GROUPS,
) -> str:
    """Draw (b), one plot per family on shared y limits, and return its tag."""
    width, slots = _c2_bar_slots(len(threads), systems)

    for (family, groups), ax in zip(groups_table, axes):
        _c2_bands(ax, len(groups), width, slots)
        for group, (label, personality) in enumerate(groups):
            for direction, system, level, offset in slots:
                x = group + offset
                run = indexed[threads[level]].get((personality, system))
                if run is None:
                    value = None
                elif direction == "read":
                    value = bench_read_gibs_less_open(run)
                else:
                    value = bench_gibs(run, direction)
                if value is None:
                    reason = "no run" if run is None else f"no {direction} phase"
                    missing.add(
                        "c2",
                        f"{family} {SYSTEMS[system].label} {direction} @ {label}"
                        f", {threads[level]}T ({personality}): {reason}",
                    )
                    missing.mark(ax, x, fill_missing)
                    continue
                _bar(ax, x, value, width, system_color(system), thread_hatch(level))

        _block_letters(
            ax,
            [
                (
                    C1_BLOCK_LABELS[direction],
                    min(x for d, _, _, x in slots if d == direction) - width,
                    max(x for d, _, _, x in slots if d == direction) + width,
                )
                for direction in C2_DIRECTIONS
            ],
        )
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([size for size, _ in groups], fontsize=common.TICKS_FONTSIZE)
        ax.set_xlim(*_c2_x_limits(len(groups)))
        ax.set_ylim(0, C2_CEILING_GIBS["read"] * 1.06)
        _style_panel(ax, "GiB/s")
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, integer=True))
        _c2_rotate_crowded(ax)
        _subtitle(ax, f"{family}, over the {C2_FAMILY_UNIT[family]} size")

    # Only the right-hand plot's frame reaches the margin, so only it names them.
    for ax, labelled in zip(axes, (False,) * (len(axes) - 1) + (True,)):
        _ceilings(ax, compact=True, labelled=labelled)

    return "(b) FS and LOG, read and write throughput"


def _c2_rotate_crowded(ax) -> None:
    """Rotate the size ticks only if they would otherwise touch."""
    figure = ax.get_figure()
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    labels = ax.get_xticklabels()
    if not labels:
        return
    widest = max(t.get_window_extent(renderer).width for t in labels) / figure.dpi
    pitch = (
        ax.get_window_extent(renderer).width
        / figure.dpi
        / (ax.get_xlim()[1] - ax.get_xlim()[0])
    )
    if widest <= 0.92 * pitch:
        return
    for label in labels:
        label.set_rotation(30)
        label.set_ha("right")
        label.set_rotation_mode("anchor")


def _c3_bar_slots(
    levels: int, systems: Sequence[str]
) -> Tuple[float, List[Tuple[str, int, float]]]:
    """Bar width and every bar's ``(system, thread level, offset)``."""
    count = len(systems) * levels
    width = C3_GROUP_WIDTH / count
    slots: List[Tuple[str, int, float]] = []
    x = -C3_GROUP_WIDTH / 2.0
    for level in range(levels):
        for system in systems:
            slots.append((system, level, x + width / 2.0))
            x += width
    return width, slots


def _c3_floor(values: Sequence[float]) -> float:
    """The bottom of (c)'s ratio axis: :data:`C3_FLOOR`, or under the bars."""
    finite = [v for v in values if v is not None and v > 0]
    if not finite or min(finite) >= C3_FLOOR:
        return C3_FLOOR
    floor = max(0.0, math.floor(min(finite) / C3_FLOOR_STEP) * C3_FLOOR_STEP)
    print(
        f"(c): a bar sits at {_fmt_pct(min(finite))}, under the {_fmt_pct(C3_FLOOR)}"
        f" floor this panel is drawn from; starting it at {_fmt_pct(floor)} instead"
    )
    return floor


def _c3_ratio_ticks(ax, floor: float) -> None:
    """(c)'s ratio ticks: the floor and 100% named, the rest stepped and thin."""
    top = ax.get_ylim()[1]
    ax.yaxis.set_major_locator(mticker.FixedLocator([floor, 1.0]))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_pct))
    steps = int(math.floor(top / C3_RATIO_MINOR))
    minors = [
        step * C3_RATIO_MINOR
        for step in range(1, steps + 1)
        if floor < step * C3_RATIO_MINOR < top
        and not math.isclose(step * C3_RATIO_MINOR, 1.0)
    ]
    ax.yaxis.set_minor_locator(mticker.FixedLocator(minors))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(
        axis="y", which="minor", length=1.2, width=common.BORDER_LINE_WIDTH * 0.6
    )


def _gibs_ticks(ax, bottom: float, panel: str) -> None:
    """A GiB/s axis' ticks: decade lines, :data:`GIBS_NAMED` labelled."""
    if bottom < min(GIBS_NAMED):
        print(
            f"{PANEL_NAMES.get(panel, panel)}: the GiB/s axis runs down to "
            f"{_fmt_decade(bottom)}, below the lowest label {_fmt_decade(min(GIBS_NAMED))}, "
            "to keep the bars that reach it; that decade is ruled but not labelled"
        )
    low, high = bottom, ax.get_ylim()[1]
    decades = [
        10.0**exponent
        for exponent in range(
            int(math.floor(math.log10(low))), int(math.ceil(math.log10(high))) + 1
        )
        if low <= 10.0**exponent <= high
    ]
    named = GIBS_NAMED
    ax.yaxis.set_major_locator(mticker.FixedLocator(decades))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda value, pos: (
                _fmt_decade(value, pos)
                if any(math.isclose(value, n, rel_tol=1e-9) for n in named)
                else ""
            )
        )
    )


def _c3_ratio_grid(ax) -> None:
    """(c)'s ratio grid: the stepped minors, and nothing at the two majors."""
    ax.grid(False, axis="y", which="major")
    ax.grid(
        True,
        axis="y",
        which="minor",
        linestyle="--",
        alpha=common.GRID_ALPHA,
        linewidth=common.GRID_LINE_WIDTH,
    )


def plot_c3(
    ax,
    ax_tput,
    indexed: Dict[int, Dict[tuple, bench_results.BenchRunMetrics]],
    threads: Sequence[int],
    systems: Sequence[str],
    tput_systems: Sequence[str],
    missing: MissingCells,
    fill_missing: bool,
    groups_table: Sequence[Tuple[str, str]] = C3_GROUPS,
) -> Tuple[str, Dict[str, Dict[Tuple[str, int], float]]]:
    """Draw (c): ratios on `ax`, GiB/s on `ax_tput`. Returns its tag and the IOPS."""
    pitch = 1.0
    rel_width, rel_slots = _c3_bar_slots(len(threads), systems)
    tput_width, tput_slots = _c3_bar_slots(len(threads), tput_systems)

    plotted: Dict[str, Dict[Tuple[str, int], float]] = {}
    absolutes: List[str] = []
    ratios: List[float] = [1.0]
    gibs: List[float] = []
    for g, (name, personality) in enumerate(groups_table):
        size = parse_size(name)
        plotted[name] = {}
        base_ops = [
            bench_read_ops(run) if run else None
            for run in (
                indexed.get(t, {}).get((personality, C3_BASELINE)) for t in threads
            )
        ]
        absolutes.append("/".join(_fmt_si(v) if v else "--" for v in base_ops))
        for t, value in zip(threads, base_ops):
            if value:
                plotted[name][(C3_BASELINE, t)] = value
        for system, level, offset in rel_slots:
            x = g * pitch + offset
            run = indexed.get(threads[level], {}).get((personality, system))
            iops = bench_read_ops(run) if run else None
            cell = f"{SYSTEMS[system].label} @ {name}, {threads[level]}T"
            baseline = base_ops[level]
            if iops is None or not baseline:
                reason = "no run" if iops is None else "no ext4 baseline"
                missing.add("c3", f"{cell} ({personality}): {reason}")
                missing.mark(ax, x, fill_missing)
                continue
            plotted[name][(system, threads[level])] = iops
            ratio = iops / baseline
            ratios.append(ratio)
            _bar(ax, x, ratio, rel_width, system_color(system), thread_hatch(level))
        for system, level, offset in tput_slots:
            x = g * pitch + offset
            run = indexed.get(threads[level], {}).get((personality, system))
            iops = bench_read_ops(run) if run else None
            cell = f"{SYSTEMS[system].label} @ {name}, {threads[level]}T"
            if iops is None:
                missing.add("c3", f"{cell}, device use ({personality}): no run")
                missing.mark(ax_tput, x, fill_missing)
                continue
            plotted[name][(system, threads[level])] = iops
            value = iops * size / GIB
            gibs.append(value)
            _bar(
                ax_tput,
                x,
                value,
                tput_width,
                system_color(system),
                thread_hatch(level),
            )

    floor = _c3_floor(ratios)
    top = max(ratios)
    ax.set_ylim(floor, top + C3_HEADROOM_FRACTION * max(top - floor, 0.1))
    _c3_ratio_ticks(ax, floor)
    for axes in (ax, ax_tput):
        _set_groups(axes, [name for name, _ in groups_table], pitch)

    # Logarithmic: linear to the device leaves the two smallest groups unreadable.
    bottom, top_gibs, _ = _log_decade_limits(
        gibs + [C2_CEILING_GIBS["read"] * C3_CEILING_HEADROOM]
    )
    ax_tput.set_yscale("log")
    ax_tput.set_ylim(bottom, top_gibs)
    _gibs_ticks(ax_tput, bottom, "c3")

    _hairline(ax)
    _check_baseline_row_fits(
        ax,
        _annotate_baseline(
            ax,
            [g * pitch for g in range(len(groups_table))],
            absolutes,
            BASELINE_ROW_LABEL,
        ),
    )
    _style_panel(ax, "rel. IOPS")
    _style_panel(ax_tput, "GiB/s")
    _subtitle(ax, "read IOPS, relative to ext4")
    _subtitle(ax_tput, "absolute read throughput")
    _c3_ratio_grid(ax)
    _ceilings(ax_tput, directions=("read",), compact=True)
    return "(c) FS random reads, 8 x 1 GiB files", plotted


def print_c3_table(
    plotted: Dict[str, Dict[Tuple[str, int], float]],
    threads: Sequence[int],
    systems: Sequence[str],
) -> None:
    """(c)'s absolute IOPS behind its ratios, and the thread scaling under them."""
    if not any(plotted.values()):
        return
    listed = [C3_BASELINE, *systems]
    header = f"{'size':<6}{'T':>4}" + "".join(f"{SYSTEMS[s].label:>20}" for s in listed)
    print(f"\n(c) FS random reads — IOPS, x {SYSTEMS[C3_BASELINE].label} in ()")
    print(header)
    print("-" * len(header))
    for label, row in plotted.items():
        for level in threads:
            reference = row.get((C3_BASELINE, level))
            cells = ""
            for system in listed:
                value = row.get((system, level))
                if value is None:
                    cells += f"{'--':>20}"
                elif reference and system != C3_BASELINE:
                    cells += f"{f'{value:,.0f} ({value / reference:.2f})':>20}"
                else:
                    cells += f"{f'{value:,.0f}':>20}"
            print(f"{label:<6}{level:>4}{cells}")
        if len(threads) > 1:
            low, high = threads[0], threads[-1]
            cells = ""
            for system in listed:
                first, last = row.get((system, low)), row.get((system, high))
                cells += (
                    f"{f'x{last / first:.2f}':>20}" if first and last else f"{'--':>20}"
                )
            print(f"{'':<6}{f'{low}>{high}':>4}{cells}")
    print(f"{'':<6}{'':>4}  (linear would be x{threads[-1] / threads[0]:.2f})")


def _panel_axes(fig) -> Dict[str, object]:
    """The panels' boxes, top to bottom, plus the y each tag is drawn at."""
    left_frac = LEFT_IN / FIG_WIDTH_IN
    span_in = FIG_WIDTH_IN - LEFT_IN - RIGHT_IN - PANEL_RIGHT_IN
    axes: Dict[str, object] = {}
    tags: Dict[str, float] = {}
    top_in = FIG_HEIGHT_IN - LEGEND_IN

    for name, label_in, row_in in PANEL_LAYOUT:
        ax_in = PANEL_AXES_IN
        top_in -= TAG_IN
        tags[name] = top_in / FIG_HEIGHT_IN
        top_in -= row_in
        bottom = (top_in - ax_in) / FIG_HEIGHT_IN
        height = ax_in / FIG_HEIGHT_IN
        if name in SPLIT_ROWS:
            rest_in = span_in - SPLIT_LEFT_IN - SPLIT_GAP_IN
            axes[name] = fig.add_axes(
                [left_frac, bottom, SPLIT_LEFT_IN / FIG_WIDTH_IN, height]
            )
            axes[f"{name}_tput"] = fig.add_axes(
                [
                    (LEFT_IN + SPLIT_LEFT_IN + SPLIT_GAP_IN) / FIG_WIDTH_IN,
                    bottom,
                    rest_in / FIG_WIDTH_IN,
                    height,
                ]
            )
        else:
            axes[name] = fig.add_axes(
                [left_frac, bottom, span_in / FIG_WIDTH_IN, height]
            )
        top_in -= ax_in + label_in

    axes["_tags"] = tags
    return axes


def _legend(fig, threads: Sequence[int], systems: Sequence[str]) -> None:
    """One key for the whole stack: the systems, the thread hatches, the bands."""
    system_handles = [
        mpatches.Patch(
            facecolor=system_color(key),
            edgecolor=common.BAR_EDGECOLOR,
            linewidth=common.BAR_LINEWIDTH,
            label=SYSTEMS[key].label,
        )
        for key in systems
    ]
    option_handles = [
        mpatches.Patch(
            facecolor="white",
            hatch=thread_hatch_legend(i),
            edgecolor=common.BAR_EDGECOLOR,
            linewidth=common.BAR_LINEWIDTH,
            label=f"{t} thread" if t == 1 else f"{t} threads",
        )
        for i, t in enumerate(threads)
    ]
    option_handles += [
        mpatches.Patch(
            facecolor=colour,
            edgecolor=common.BAR_EDGECOLOR,
            linewidth=common.BAR_LINEWIDTH,
            label=label,
        )
        for colour, label in (
            (C2_BAND_COLOR, f"{C1_BLOCK_LABELS['read']} = read"),
            ("white", f"{C1_BLOCK_LABELS['write']} = write"),
        )
    ]
    handles = list(system_handles) + list(option_handles)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.999),
        ncol=len(handles),
        fontsize=common.LEGEND_FONTSIZE - 0.5,
        frameon=True,
        edgecolor="black",
        borderaxespad=0,
        borderpad=0.3,
        columnspacing=0.6,
        labelspacing=0.15,
        handletextpad=0.3,
        handlelength=1.0,
        # density=1: these hatches are already written out at the swatch's density.
        handler_map={mpatches.Patch: DenseHatchHandler(density=1)},
    )
    fig.legends[0].get_frame().set_linewidth(common.BORDER_LINE_WIDTH)
    _check_legend_fits(fig)


def _check_legend_fits(fig) -> None:
    """Warn if the one row of the key is wider than the page it is drawn on."""
    fig.canvas.draw()
    width_in = (
        fig.legends[0].get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    )
    if width_in > FIG_WIDTH_IN:
        warnings.warn(
            f"the key is {width_in:.2f} in wide against a {FIG_WIDTH_IN:.2f} in"
            f" figure: {len(fig.legends[0].get_texts())} entries do not fit one row"
        )


def _tag(fig, y: float, text: str) -> None:
    """A panel's caption, with the direction its bars are read in."""
    fig.text(
        LEFT_IN / FIG_WIDTH_IN,
        y + 0.004,
        f"{text} ({DIRECTION_NOTE})",
        ha="left",
        va="bottom",
        fontsize=common.TITLE_FONTSIZE - 1,
        color=common.TITLE_COLOR,
    )


def generate(
    results_root: Path,
    output: Path,
    value_sizes: Sequence[Tuple[str, int]],
    threads: Sequence[int],
    png: bool = False,
    fill_missing: bool = False,
    quick: bool = False,
) -> Path:
    """Draw the three-panel synthetic figure and report the cells it could not fill.

    `results_root` is the tree ``bench-result/{ycsb,bench}`` sits under, `output`
    the PDF path, `value_sizes` panel (a)'s ``(label, bytes)`` groups and
    `threads` the thread counts every panel draws. `png` writes a PNG beside the
    PDF, `fill_missing` marks empty cells with a small x, and `quick` draws the
    figure from the ``paper.* --quick`` runs and their smaller size sets.
    """
    common.apply_mpl_settings()
    plt.rcParams["hatch.linewidth"] = HATCH_LINEWIDTH

    # A quick run and a full one measure the same point, so never mix the two.
    ycsb_points = [
        point
        for point in discover_ycsb_points(results_root)
        if _is_quick_leaf(point.path) == quick
    ]
    bench_runs = [
        run
        for run in discover_bench_runs(results_root)
        if _is_quick_leaf(run.path) == quick
    ]
    c2_groups = C2_FAMILY_GROUPS_QUICK if quick else C2_FAMILY_GROUPS
    c3_groups = C3_GROUPS_QUICK if quick else C3_GROUPS
    missing = MissingCells()

    if FIG_HEIGHT_IN > FIG_HEIGHT_MAX_IN:
        warnings.warn(
            f"{FIG_HEIGHT_IN:.3f} in is taller than the {FIG_HEIGHT_MAX_IN:.2f} in"
            " ceiling this figure is laid out against"
        )
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    axes = _panel_axes(fig)
    tags = axes["_tags"]

    c1_indexed = index_ycsb(ycsb_points, C1_MIX)
    c1_systems = _panel_systems(
        C1_TPUT_SYSTEMS, {system for system, _, _ in c1_indexed}, "c1", missing
    )

    c2_personalities = [p for _, groups in c2_groups for _, p in groups]
    c2_indexed = {t: index_bench(bench_runs, c2_personalities, t) for t in threads}
    c2_systems = _panel_systems(
        C2_SYSTEMS,
        {system for index in c2_indexed.values() for _, system in index},
        "c2",
        missing,
    )

    c3_personalities = [personality for _, personality in c3_groups]
    c3_indexed = {t: index_bench(bench_runs, c3_personalities, t) for t in threads}
    c3_tput_systems = _panel_systems(
        C3_TPUT_SYSTEMS,
        {system for index in c3_indexed.values() for _, system in index},
        "c3",
        missing,
    )
    c3_systems = [key for key in C3_SYSTEMS if key in c3_tput_systems]

    _legend(
        fig,
        threads,
        [
            key
            for key in SYSTEMS
            if key in set(c1_systems) | set(c2_systems) | set(c3_tput_systems)
        ],
    )

    tag = plot_c1(
        axes["c1"],
        axes["c1_tput"],
        c1_indexed,
        value_sizes,
        threads,
        c1_systems,
        missing,
        fill_missing,
    )
    _tag(fig, tags["c1"], tag)

    tag = plot_c2(
        (axes["c2"], axes["c2_tput"]),
        c2_indexed,
        threads,
        c2_systems,
        missing,
        fill_missing,
        c2_groups,
    )
    _tag(fig, tags["c2"], tag)

    tag, c3_plotted = plot_c3(
        axes["c3"],
        axes["c3_tput"],
        c3_indexed,
        threads,
        c3_systems,
        c3_tput_systems,
        missing,
        fill_missing,
        c3_groups,
    )
    _tag(fig, tags["c3"], tag)

    output.parent.mkdir(parents=True, exist_ok=True)
    # No bbox_inches="tight": the page admits exactly FIG_WIDTH_IN.
    fig.savefig(output, dpi=600)
    if png:
        fig.savefig(output.with_suffix(".png"), dpi=600)
    plt.close(fig)

    print(f"✓ Generated: {output}")
    print_bench_provenance(
        "c2",
        [group for _, groups in c2_groups for group in groups],
        c2_indexed,
        threads,
        c2_systems,
    )
    print_bench_provenance("c3", c3_groups, c3_indexed, threads, c3_tput_systems)
    print_c3_table(c3_plotted, threads, c3_systems)
    missing.report()
    return output
