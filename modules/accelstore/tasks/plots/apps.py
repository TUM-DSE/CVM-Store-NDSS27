#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The paper's application figure: MLPerf Storage checkpointing, Redis and the
application personalities, in one column.

Two panels, split by unit: checkpointing alone against a GiB/s spine on the
left, Redis and the personalities together against one shared ``x ext4`` spine
on the right. :mod:`mlperf_panels`, :mod:`redis_panels` and
:mod:`personality_results` load and gate the runs and draw no figure of their
own.

Missing cells are holes, never zeros: a system with no run for a group leaves
its slot empty and is named on stdout.

Drawn by ``accelstore.plot-apps``, which hands the task's arguments straight to
:func:`generate`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.text import Text

from core.tasks.plotting import common

from ..results import mlperf_panels, personality_results, redis_panels
from .style import group_bar_offsets

#: Drawn left to right. Each needs a `personality_results.NATIVE_THREADS` entry.
PERSONALITIES: Tuple[str, ...] = (
    "varmail_log",
    "videoaudit_log_64m_fsync",
    "trainload_fs_paced",
    "webserver_fs",
)

PERSONALITY_LABELS: Dict[str, str] = {
    "webserver_fs": "web",
    "webserver_log": "web-log",
    "varmail_log": "mail",
    "fileserver_log": "file",
    "videoserver_fs": "video",
    "oltp_log": "oltp",
    "modelload_fs": "model",
    "imagepull_fs": "pull",
    "ledger_log": "ledger",
    "datascan_fs": "scan",
    "trainload_fs": "train",
    "trainload_fs_paced": "train",
    "videoaudit_log": "audit",
    "videoaudit_log_fsync": "audit",
    "videoaudit_log_64m_fsync": "audit",
    "videoserver_fs_1g_session": "video",
}

IO_MODES: Tuple[str, ...] = ("direct", "buffered")
#: Diagonal on purpose: a horizontal hatch on a ~4 pt bar reads as part of the fill.
IO_MODE_HATCH: Dict[str, str] = {"direct": "", "buffered": "///"}

#: The same pattern, denser: one stroke in a 9 pt legend swatch reads as a scratch.
IO_MODE_HATCH_KEY: Dict[str, str] = {"direct": "", "buffered": "/////"}

IO_MODE_LABELS: Dict[str, str] = {"direct": "O_DIRECT", "buffered": "buffered"}
IO_MODE_SHORT: Dict[str, str] = {"direct": "dir", "buffered": "buf"}

#: Which ext4 arm the relative panel's 1.0 is: the one every other bar's ack matches.
BENCH_BASELINE_MODE: str = "direct"

#: f=1m, not f=0: f=0 replicates nothing and makes no freshness claim.
BENCH_ROLLBACCINE_ARM: str = personality_results.ROLLBACCINE_F1M_ARM

REDIS_POINTS: Tuple[str, ...] = redis_panels.POINTS

ROLLBACCINE_LABEL: str = "Rbc f=1m"

#: A constant, not read off a run: no Redis sidecar carries a module.
REDIS_API: str = "log"

API_LABELS: Dict[str, str] = {"log": "Log", "fs": "FS", "kvs": "KVS"}

CKPT_MODULES: Tuple[str, ...] = ("fs", "log")

#: Empty on purpose: MODULE_MARKS tells the modules apart, and hatch is the mode.
MODULE_HATCH: Tuple[str, ...] = ("", "")

#: One mark per CKPT_MODULES entry, on the bar's value label. Not an operator like "+".
MODULE_MARKS: Tuple[str, ...] = ("*", "\u2020")
#: Hatch stroke width in points, set both as an rcParam and per artist.
HATCH_LINEWIDTH = 0.3

#: Whether an MLPerf bar carries its spread. Off: the whiskers are sub-point here.
ERROR_BARS: bool = False

FIG_WIDTH_IN = 252.0 / 72.0
# The two figures this replaces were 1.42 in and 1.50 in.
FIG_HEIGHT_MAX_IN = 2.92
FIG_HEIGHT_IN = 1.00

# A little taller than the legend needs, so its frame does not touch the panels.
LEGEND_IN = 0.19
# The group ticks, the one-line group notes, and the half-captions under those.
BOTTOM_IN = 0.30
# The checkpointing panel's own GiB/s tick labels and axis label.
LEFT_IN = 0.30
GUTTER_IN = 0.12
#: The right margin of `ablation`, the paper's other single-column figure.
ABLATION_RIGHT_IN = 0.03
#: Width of the relative panel's tick block, measured off the rendered figure.
RIGHT_TICK_IN = 0.138
# Not the left margin: this lands the last ink where ablation's right spine does.
RIGHT_IN = ABLATION_RIGHT_IN + RIGHT_TICK_IN

# Fraction of the group pitch the bars occupy. The rest is the gap between groups.
GROUP_WIDTH = 0.82

#: How far a group's arms may disagree on mean op size before _op_size_note says so.
OP_SIZE_SPREAD = 1.10

GRID_ZORDER = mlperf_panels.GRID_ZORDER
BAR_ZORDER = mlperf_panels.BAR_ZORDER
#: Behind the bars but above the gridlines.
BASELINE_ZORDER = GRID_ZORDER + 1

HEADING = common.HIGHER_BETTER_TITLE

#: Where the half-captions sit, in axes fractions below the axis.
CAPTION_Y = -0.435
#: The rule between the relative panel's two halves, and how it is drawn.
DIVIDER_COLOR = "0.55"
DIVIDER_WIDTH = 0.5
#: Gap between a bar's error-bar cap and its value, as a fraction of the axis.
ANNOTATION_PAD = 0.02
#: Where a group's own note sits, under its tick label.
GROUP_NOTE_Y = -0.19
GROUP_NOTE_SIZE = common.ANNOTATION_SIZE
#: What a note may shrink to when a group's pitch cannot hold it.
GROUP_NOTE_MIN_SIZE = common.ANNOTATION_SIZE - 1.0
GROUP_NOTE_COLOR = "black"

AXIS_HEADROOM = mlperf_panels.AXIS_HEADROOM
RELATIVE_AXIS_MIN_TOP = mlperf_panels.RELATIVE_AXIS_MIN_TOP
#: Extra headroom for the two-line rate label above the tallest bar.
RELATIVE_TOP_HEADROOM = 1.32


@dataclass(frozen=True)
class Series:
    """One system as this figure draws it, in both halves and in the legend."""

    key: str
    label: str
    color: tuple
    #: The device path, and nothing else (:data:`IO_MODE_HATCH`).
    hatch: str = ""
    #: The `personality_results.System` whose `matches()` decides a bench arm.
    bench: Optional[personality_results.System] = None
    #: A one-character mark appended to this series' value labels.
    mark: str = ""
    #: Which device path this series is, where it is a POSIX baseline.
    io_mode: str = ""


#: The bench arm directory behind each system.
BENCH_ARMS: Dict[str, str] = {
    "cvm-store": personality_results.BENCH_CLIENT,
    "rollbaccine": BENCH_ROLLBACCINE_ARM,
    "luks": "ext4-aes-xts-plain",
    "ext4": "ext4",
}

#: Whether each arm is optional, read off :data:`personality_results.SYSTEMS`.
BENCH_OPTIONAL: Dict[str, bool] = {
    key: next(
        (s.optional for s in personality_results.SYSTEMS if s.key == arm),
        False,
    )
    for key, arm in BENCH_ARMS.items()
}


def _identity_series() -> Tuple[Series, ...]:
    """One entry per system: the legend's list, and the Redis half's bars."""
    out: List[Series] = [
        Series(key=system.key, label=system.label, color=system.color)
        for system in mlperf_panels.SYSTEMS
    ]
    # CVM-Store has no device-path axis, so it carries its own bench arm.
    out[0] = Series(
        key=out[0].key,
        label=out[0].label,
        color=out[0].color,
        bench=personality_results.System(
            personality_results.BENCH_CLIENT, out[0].label
        ),
    )
    return tuple(out)


def _io_series(base: Series) -> Tuple[Series, ...]:
    """One POSIX baseline, once per device path, in bar order."""
    arm = BENCH_ARMS[base.key]
    return tuple(
        Series(
            key=mlperf_panels.io_mode_key(base.key, mode),
            label=f"{base.label} ({IO_MODE_SHORT[mode]})",
            color=base.color,
            hatch=IO_MODE_HATCH[mode],
            bench=personality_results.System(
                arm, base.label, optional=BENCH_OPTIONAL[base.key], io_mode=mode
            ),
            io_mode=mode,
        )
        for mode in IO_MODES
    )


def _series() -> Tuple[Series, ...]:
    """Every series either kind of half can draw: identities, then modes."""
    identities = _identity_series()
    out = list(identities)
    for base in identities:
        if base.key == "cvm-store":
            continue
        out.extend(_io_series(base))
    return tuple(out)


SERIES: Tuple[Series, ...] = _series()
SERIES_BY_KEY = {s.key: s for s in SERIES}
BASELINE_KEY = mlperf_panels.BASELINE_KEY
#: The relative panel's 1.0.
BENCH_BASELINE_KEY: str = mlperf_panels.io_mode_key(BASELINE_KEY, BENCH_BASELINE_MODE)


def _ckpt_key(module: str) -> str:
    """The cell key of CVM-Store's checkpointing bar for one module."""
    return f"cvm-store@{module}"


def _checkpoint_by_module(
    tree: Path,
    config: str,
    fallback,
    requested: Optional[str],
) -> Tuple[object, Tuple[str, ...]]:
    """One checkpointing panel holding a CVM-Store bar per module that ran."""
    if fallback is None or requested:
        module = getattr(fallback, "module", None)
        if fallback is not None and module:
            _rekey_cvm_store(fallback, module)
            return fallback, (module,)
        return fallback, ()

    merged, found = None, []
    for module in CKPT_MODULES:
        panel, _, _ = mlperf_panels.load_panels(tree, config=config, module=module)
        if panel is None or panel.module != module:
            # `load_panels` falls back to what did run, so this pass found nothing.
            continue
        _rekey_cvm_store(panel, module)
        found.append(module)
        if merged is None:
            merged = panel
        else:
            merged.cells.update(panel.cells)
    if merged is None:
        return fallback, ()
    merged.module = None  # the bars name it now, so the caption must not
    return merged, tuple(found)


def _rekey_cvm_store(panel, module: str) -> None:
    """Move this panel's CVM-Store cells onto that module's own key."""
    for (group, system), cell in list(panel.cells.items()):
        if system == "cvm-store":
            del panel.cells[(group, system)]
            panel.cells[(group, _ckpt_key(module))] = cell


def _left_series(relative: bool) -> Tuple[Series, ...]:
    """The systems the MLPerf half draws, per side (the relative side omits ext4)."""
    keys = [s.key for s in mlperf_panels.SYSTEMS]
    if relative:
        keys = [k for k in keys if k != BASELINE_KEY]
    return tuple(SERIES_BY_KEY[k] for k in keys)


def _ckpt_series(modules: Sequence[str]) -> Tuple[Series, ...]:
    """The checkpointing half's bars: one CVM-Store per module, then the rest."""
    cvm = SERIES_BY_KEY["cvm-store"]
    out = [
        Series(
            key=_ckpt_key(module),
            label=f"{cvm.label} ({_api_label(module)})",
            color=cvm.color,
            hatch=MODULE_HATCH[index % len(MODULE_HATCH)],
            mark=MODULE_MARKS[index % len(MODULE_MARKS)],
        )
        for index, module in enumerate(modules)
    ]
    # The baselines twice each, once per device path.
    for base in _left_series(relative=False):
        if base.key != "cvm-store":
            out.extend(_io_series(base))
    return tuple(out)


def _ckpt_series_for(panel) -> Tuple[Series, ...]:
    """The checkpointing half's bars, read off the cells the panel holds."""
    if panel is None:
        return ()
    present = {
        key[len("cvm-store@") :]
        for _, key in panel.cells
        if key.startswith("cvm-store@")
    }
    ordered = [m for m in CKPT_MODULES if m in present]
    ordered += sorted(present - set(ordered))
    return _ckpt_series(ordered)


def _draw_bar(
    ax,
    x: float,
    height: float,
    width: float,
    s: Series,
    stdev: float,
    error_bars: bool,
):
    _thin_hatch(
        ax.bar(
            x,
            height,
            width,
            color=s.color,
            alpha=common.BAR_ALPHA,
            hatch=s.hatch or None,
            edgecolor=common.BAR_EDGECOLOR,
            linewidth=common.BAR_LINEWIDTH,
            yerr=(stdev or None) if error_bars else None,
            error_kw={
                "elinewidth": 0.4,
                "capsize": common.ERROR_BAR_CAP_SIZE,
                "capthick": 0.4,
            },
            zorder=BAR_ZORDER,
        )
    )


def _thin_hatch(container) -> None:
    """Set a bar's hatch stroke width, where the per-artist setter exists."""
    for patch in container:
        if hasattr(patch, "set_hatch_linewidth"):
            patch.set_hatch_linewidth(HATCH_LINEWIDTH)


def _fmt_rate(value: float) -> str:
    """A rate as a caption would quote it: three significant digits, no more."""
    for suffix, scale in (("M", 1e6), ("k", 1e3)):
        if value >= scale:
            scaled = value / scale
            digits = 0 if scaled >= 100 else (1 if scaled >= 10 else 2)
            return f"{scaled:.{digits}f}{suffix}"
    return f"{value:.0f}"


def _annotate(ax, x: float, y: float, text: str) -> None:
    """A bar's own value, above it, rotated because the bars are ~5 pt wide."""
    ax.text(
        x,
        y,
        text,
        fontsize=common.ANNOTATION_SIZE - 0.5,
        color="black",
        ha="left",
        va="center",
        rotation=90,
        rotation_mode="anchor",
        zorder=BAR_ZORDER + 1,
    )


def _half_grid(ax, xmin: float, xmax: float) -> None:
    """Gridlines over one half of the panel: the panel carries two y scales."""
    low, high = ax.get_ylim()
    ax.hlines(
        [t for t in ax.get_yticks() if low < t < high],
        xmin,
        xmax,
        linestyles="--",
        colors="0.5",
        alpha=common.GRID_ALPHA,
        linewidth=common.GRID_LINE_WIDTH,
        zorder=GRID_ZORDER,
    )


def _mark_missing(ax, x: float) -> None:
    """The slot's mark: on the baseline, and not a quantity."""
    ax.text(
        x,
        0.012,
        "×",
        transform=ax.get_xaxis_transform(),
        fontsize=common.ANNOTATION_SIZE,
        color="0.5",
        ha="center",
        va="bottom",
        zorder=BAR_ZORDER,
    )


def _style_linear(ax, ylabel: str, *, side: str) -> None:
    """One of the MLPerf half's two linear axes, ticked on the `side` spine."""
    other = "right" if side == "left" else "left"
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=common.TICKS_FONTSIZE, labelpad=1.0)
        ax.yaxis.set_label_position(side)
    ax.yaxis.set_ticks_position(side)
    # No `ax.grid`: a gridline here would span both scales (see `_half_grid`).
    ax.grid(False)
    ax.tick_params(axis="x", length=0, pad=2.25, labelsize=common.TICKS_FONTSIZE)
    ax.tick_params(
        axis="y",
        which="major",
        length=2.2,
        width=common.BORDER_LINE_WIDTH,
        pad=1.2,
        labelsize=common.TICKS_FONTSIZE,
    )
    ax.spines["top"].set_visible(False)
    ax.spines[other].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(common.BORDER_LINE_WIDTH)
        # Over the bars: a spine's default zorder is below them.
        spine.set_zorder(BAR_ZORDER + 1)


def _measured(panel, systems: Sequence[Series]) -> List[Series]:
    """The series with a cell somewhere in this panel, in bar order."""
    present = {key for _, key in panel.cells}
    return [system for system in systems if system.key in present]


def _ckpt_modules(systems: Sequence[Series]) -> List[Tuple[str, str]]:
    """(module name, mark) for each CVM-Store bar of the checkpointing half."""
    return [
        (series.label.split("(")[-1].rstrip(")"), series.mark)
        for series in systems
        if series.key.startswith("cvm-store@")
    ]


def _bar_slots(panel, systems: Sequence[Series], fill_missing: bool) -> int:
    """How many bar slots one group of a panel takes."""
    if panel is None:
        return 0
    return len(systems if fill_missing else _measured(panel, systems))


def _plot_checkpointing(
    ax,
    panel,
    xs: Sequence[float],
    fill_missing: bool,
    error_bars: bool,
    systems: Sequence[Series],
) -> Tuple[float, List[Tuple[float, float, str]]]:
    """The absolute half, in GiB/s. Returns the tallest whisker and the value labels."""
    width, offsets = group_bar_offsets(GROUP_WIDTH, len(systems))
    top = 0.0
    labels: List[Tuple[float, float, str]] = []
    for x, group in zip(xs, panel.groups):
        for system, offset in zip(systems, offsets):
            cell = panel.cells.get((group, system.key))
            if cell is None:
                if fill_missing:
                    _mark_missing(ax, x + offset)
                continue
            _draw_bar(ax, x + offset, cell.value, width, system, cell.stdev, error_bars)
            # Anchored on what was drawn: with no whisker there is nothing to clear.
            head = cell.value + cell.stdev if error_bars else cell.value
            labels.append((x + offset, head, f"{cell.value:.1f}{system.mark}"))
            top = max(top, head)
    return top, labels


def _plot_redis(
    ax, panel, xs: Sequence[float], fill_missing: bool
) -> Tuple[float, List[Tuple[float, str, str]], List[Text]]:
    """Redis groups as SET ops/s over ext4's. Returns top, rates and notes."""
    systems = _left_series(relative=True)
    if not fill_missing:
        systems = _measured(panel, systems)
    width, offsets = group_bar_offsets(GROUP_WIDTH, len(systems))
    top = 1.0
    rates: List[Tuple[float, str, str]] = []
    notes: List[Text] = []
    for x, group in zip(xs, panel.groups):
        base = panel.baselines[group].value
        for system, offset in zip(systems, offsets):
            cell = panel.cells.get((group, system.key))
            if cell is None:
                if fill_missing:
                    _mark_missing(ax, x + offset)
                continue
            ratio = cell.value / base
            _draw_bar(ax, x + offset, ratio, width, system, 0.0, False)
            top = max(top, ratio)
        # ext4's own rate goes on the shared rate line instead (`rates`).
        note = (panel.group_notes or {}).get(group, "")
        notes.append(_group_text(ax, x, note))
        rates.append((x, _fmt_rate(base), "op/s"))
    return top, rates, notes


def _api_label(module: str) -> str:
    """How to write one module's name, or the name itself if it is unmapped."""
    return API_LABELS.get(module, module)


def _pitch_in(ax, groups: int) -> float:
    """How many inches of figure one group of `ax` gets, ticks to ticks."""
    width_in = ax.get_position().width * ax.figure.get_figwidth()
    return width_in / max(groups, 1)


def _fit_texts(items: Sequence[Tuple[Text, float]], *, size: float, what: str) -> None:
    """Shrink one row of labels, together, to one size that fits every budget."""
    if not items:
        return
    fig = items[0][0].figure
    renderer = fig.canvas.get_renderer()
    scale = 1.0
    tightest = ""
    for text, budget_in in items:
        width_pt = text.get_window_extent(renderer=renderer).width / fig.dpi * 72.0
        budget_pt = budget_in * 72.0
        if width_pt > 0 and budget_pt / width_pt < scale:
            scale = budget_pt / width_pt
            tightest = text.get_text().splitlines()[0]
    if scale >= 1.0:
        return
    fitted = max(GROUP_NOTE_MIN_SIZE, size * scale)
    print(
        f"{what}: {fitted:.1f} pt instead of {size:.1f} —"
        f" {tightest!r} does not fit the {items[0][1]:.2f} in it has"
    )
    for text, _ in items:
        text.set_fontsize(fitted)


def _group_text(ax, x: float, text: str) -> Text:
    """A relative group's note, under its name. Returns the artist _fit_texts sizes."""
    return ax.text(
        x,
        GROUP_NOTE_Y,
        text,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=GROUP_NOTE_SIZE,
        color=GROUP_NOTE_COLOR,
        linespacing=1.1,
        clip_on=False,
    )


def _annotate_rate(ax, x: float, y: float, value: str, unit: str) -> None:
    """A relative group's ext4 rate, on the one line every group's shares."""
    ax.text(
        x,
        y,
        f"{value}\n{unit}",
        fontsize=common.ANNOTATION_SIZE - 0.5,
        color="black",
        ha="center",
        va="top",
        linespacing=1.1,
        clip_on=False,
        zorder=BAR_ZORDER + 1,
    )


def _half_caption(ax, x: float, text: str) -> Text:
    """A half's caption, under its notes: what the half is, and what those notes are."""
    return ax.text(
        x,
        CAPTION_Y,
        text,
        transform=ax.get_xaxis_transform(),
        fontsize=common.ANNOTATION_SIZE,
        color="black",
        ha="center",
        va="top",
        style="italic",
        clip_on=False,
    )


def plot_checkpoint_panel(
    ax, checkpoint, fill_missing: bool, error_bars: bool
) -> List[Tuple[Text, float]]:
    """MLPerf checkpointing alone, in GiB/s, against the left spine."""
    groups = list(checkpoint.groups) if checkpoint else []
    if not groups:
        ax.set_axis_off()
        return []

    xs = list(range(len(groups)))
    systems = _ckpt_series_for(checkpoint)
    if not fill_missing:
        systems = _measured(checkpoint, systems)
    top, labels = _plot_checkpointing(
        ax, checkpoint, xs, fill_missing, error_bars, systems
    )

    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(0, (top or 1.0) * AXIS_HEADROOM)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    _style_linear(ax, "GiB/s", side="left")

    # Now that the range is final: their clearance is a fraction of it.
    low, high = ax.get_ylim()
    for x, y, text in labels:
        _annotate(ax, x, y + (high - low) * ANNOTATION_PAD, text)

    _half_grid(ax, -0.5, len(groups) - 0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(groups, fontsize=common.TICKS_FONTSIZE)

    # The footnote for the duplicated CVM-Store bars, keyed by their value labels.
    modules = _ckpt_modules(systems)
    if len(modules) > 1:
        named = ", ".join(f"{mark}{name}" for name, mark in modules)
        suffix = f" (CVM-Store: {named})"
    else:
        suffix = f" ({modules[0][0]})" if modules else ""
    caption = _half_caption(ax, sum(xs) / len(xs), f"Checkpointing{suffix}")

    # What was checkpointed, under the group ticks. One note, centred.
    if checkpoint.subtitle:
        # Its budget is the whole panel, not one group's pitch.
        _fit_texts(
            [
                (
                    _group_text(ax, sum(xs) / len(xs), checkpoint.subtitle),
                    _pitch_in(ax, 1),
                )
            ],
            size=GROUP_NOTE_SIZE,
            what="checkpointing note",
        )
    return [(caption, _pitch_in(ax, 1))]


def _bench_series() -> Tuple[Series, ...]:
    """The series the filebench half collects, in bar order, both ext4 modes."""
    return tuple(s for s in SERIES if s.bench is not None and not s.bench.optional)


def _bench_drawn(grid: Dict[Tuple[str, str], personality_results.Cell]) -> List[Series]:
    """The filebench series with a run somewhere in the figure, in bar order."""
    measured = {key for _, key in grid}
    return [s for s in _bench_series() if s.key in measured]


def _bench_bars(grid, fill_missing: bool) -> List[Series]:
    """The filebench series drawn as *bars*: everything but the two ext4 lines."""
    systems = [
        s
        for s in _bench_series()
        if mlperf_panels.split_io_mode(s.key)[0] != BASELINE_KEY
    ]
    if fill_missing:
        return systems
    # The grid is not a Panel, so `_measured`'s rule is applied to it directly.
    measured = {key for _, key in grid}
    return [s for s in systems if s.key in measured]


def _collect_bench(
    cells: Sequence[personality_results.Cell],
    personalities: Sequence[str],
    systems: Sequence[Series],
) -> Tuple[Dict[Tuple[str, str], personality_results.Cell], List[str]]:
    """One cell per (personality, series), and the slots with no run."""
    grid: Dict[Tuple[str, str], personality_results.Cell] = {}
    missing: List[str] = []
    for personality in personalities:
        threads = personality_results.NATIVE_THREADS.get(personality)
        group: Dict[str, personality_results.Cell] = {}
        for series in systems:
            found, reason = (
                personality_results._pick(cells, series.bench, personality, threads)
                if threads is not None
                else personality_results._pick_any(cells, series.bench, personality)
            )
            if found is None:
                missing.append(f"{personality} / {series.label} ({reason})")
                continue
            grid[(personality, series.key)] = found
            group[series.label] = found
        personality_results._check_bounds(group, f"{personality} bars")
    return grid, missing


def _fmt_bytes(size: float) -> str:
    """A byte count as a figure annotation: two significant figures, one unit."""
    for unit, scale in (("MiB", 1 << 20), ("KiB", 1 << 10), ("B", 1)):
        if size >= scale or scale == 1:
            value = size / scale
            digits = 0 if value >= 10 else 1
            return f"{value:.{digits}f} {unit}"
    return ""


def _op_size_note(personality: str, cells: Sequence[personality_results.Cell]) -> str:
    """What one operation of this group moves, as the median over its drawn arms."""
    sizes = sorted(size for size in (c.run.mean_op_bytes for c in cells) if size)
    if not sizes:
        return ""
    if sizes[-1] > sizes[0] * OP_SIZE_SPREAD:
        print(
            f"{personality}: arms disagree on the mean op size"
            f" ({_fmt_bytes(sizes[0])}..{_fmt_bytes(sizes[-1])});"
            " the note is the median"
        )
    return _fmt_bytes(sizes[len(sizes) // 2])


def _group_note(
    grid: Dict[Tuple[str, str], personality_results.Cell],
    personality: str,
    systems: Sequence[Series],
) -> str:
    """What one op of this group is, and what ran it, as two lines."""
    cells = [
        grid[(personality, s.key)] for s in systems if (personality, s.key) in grid
    ]
    if not cells:
        return ""
    modules = {c.run.workload_type for c in cells if c.run.workload_type}
    threads = {c.threads for c in cells if c.threads}
    module = _api_label(modules.pop()) if len(modules) == 1 else "?"
    count = f"{threads.pop()} thr" if len(threads) == 1 else "? thr"
    size = _op_size_note(personality, cells)
    return f"{size}\n{module} {count}" if size else f"{module} {count}"


def _plot_personalities(
    ax,
    grid: Dict[Tuple[str, str], personality_results.Cell],
    personalities: Sequence[str],
    xs: Sequence[float],
    fill_missing: bool,
) -> Tuple[float, List[Tuple[float, str, str]], List[Text]]:
    """The personality groups of the relative panel: ops/s over ext4's."""
    systems = _bench_bars(grid, fill_missing)
    top = 1.0
    rates: List[Tuple[float, str, str]] = []
    notes: List[Text] = []
    width, offsets = group_bar_offsets(GROUP_WIDTH, len(systems))
    for x, personality in zip(xs, personalities):
        base = grid[(personality, BENCH_BASELINE_KEY)].ops_per_s
        for system, offset in zip(systems, offsets):
            cell = grid.get((personality, system.key))
            if cell is None:
                if fill_missing:
                    _mark_missing(ax, x + offset)
                continue
            ratio = cell.ops_per_s / base
            _draw_bar(ax, x + offset, ratio, width, system, 0.0, False)
            top = max(top, ratio)
            # The gate marks stay in the console table, not on the bars.
        # What the group ran. ext4's own rate goes on the shared rate line.
        note = _group_note(grid, personality, systems)
        notes.append(_group_text(ax, x, note))
        rates.append((x, _fmt_rate(base), "op/s"))
    return top, rates, notes


def plot_relative_panel(
    ax,
    redis,
    grid: Dict[Tuple[str, str], personality_results.Cell],
    personalities: Sequence[str],
    fill_missing: bool,
) -> List[Tuple[Text, float]]:
    """Redis and the personalities in one axes: both halves are ops/s over ext4's."""
    redis_groups = list(redis.groups) if redis else []
    drawable = [p for p in personalities if (p, BENCH_BASELINE_KEY) in grid]
    for personality in personalities:
        if personality not in drawable:
            print(
                f"no ext4 ({BENCH_BASELINE_MODE}) baseline for '{personality}'"
                " — group dropped"
            )

    n = len(redis_groups) + len(drawable)
    if n == 0:
        ax.set_axis_off()
        return []

    redis_xs = list(range(len(redis_groups)))
    bench_xs = [len(redis_groups) + i for i in range(len(drawable))]

    top = 1.0
    rate_labels: List[Tuple[float, str, str]] = []
    notes: List[Text] = []
    if redis_groups:
        redis_top, redis_rates, redis_notes = _plot_redis(
            ax, redis, redis_xs, fill_missing
        )
        top = max(top, redis_top)
        rate_labels.extend(redis_rates)
        notes.extend(redis_notes)
    if drawable:
        bench_top, bench_rates, bench_notes = _plot_personalities(
            ax, grid, drawable, bench_xs, fill_missing
        )
        top = max(top, bench_top)
        rate_labels.extend(bench_rates)
        notes.extend(bench_notes)
    # One size for both halves: the notes are one row across the panel.
    pitch_in = _pitch_in(ax, n)
    _fit_texts(
        [(note, pitch_in) for note in notes],
        size=GROUP_NOTE_SIZE,
        what="group notes",
    )

    # One line across both halves, named once at the panel's left edge.
    baseline_color = SERIES_BY_KEY[BASELINE_KEY].color
    ax.hlines(
        1.0,
        -0.5,
        n - 0.5,
        colors=[baseline_color],
        linestyles="--",
        linewidth=0.8,
        zorder=BASELINE_ZORDER,
    )
    ax.text(
        -0.5,
        1.0,
        "ext4",
        color=baseline_color,
        fontsize=common.TICKS_FONTSIZE,
        ha="left",
        va="bottom",
        zorder=BASELINE_ZORDER,
    )
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, max(top * RELATIVE_TOP_HEADROOM, RELATIVE_AXIS_MIN_TOP))
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xticks(redis_xs + bench_xs)
    ax.set_xticklabels(
        [redis_panels.POINT_LABELS.get(p, p) for p in redis_groups]
        + [PERSONALITY_LABELS.get(p, p) for p in drawable],
        fontsize=common.TICKS_FONTSIZE,
    )

    # Ext4's rate, on one shared line, now that the range is final.
    low, high = ax.get_ylim()
    rate_y = high - (high - low) * 0.03
    for x, value, unit in rate_labels:
        _annotate_rate(ax, x, rate_y, value, unit)

    if redis_groups and drawable:
        ax.axvline(
            len(redis_groups) - 0.5,
            color=DIVIDER_COLOR,
            linewidth=DIVIDER_WIDTH,
            zorder=BASELINE_ZORDER,
        )
    captions: List[Tuple[Text, float]] = []
    if redis_groups:
        captions.append(
            (
                _half_caption(
                    ax,
                    sum(redis_xs) / len(redis_xs),
                    f"Redis ({_api_label(REDIS_API)}), value size",
                ),
                len(redis_groups) * pitch_in,
            )
        )
    if drawable:
        captions.append(
            (
                _half_caption(
                    ax,
                    sum(bench_xs) / len(bench_xs),
                    "Workloads (mean op size, threads)",
                ),
                len(drawable) * pitch_in,
            )
        )

    # Ticks only: the unit is the label on the line, not a second axis label.
    _style_linear(ax, "", side="right")
    _half_grid(ax, -0.5, n - 0.5)
    return captions


#: Gap between the legend and the heading, as a fraction of the figure width.
LEGEND_HEADING_GAP = 0.015


def _legend_and_heading(
    fig, height_in: float, drawn: Sequence[Series], io_modes: bool
) -> None:
    """The key and "(Higher is better)" as one row. `io_modes` adds the textures."""
    drawn = [s for s in drawn if not s.key.startswith("cvm-store@")]
    handles = [
        mpatches.Patch(
            facecolor=s.color,
            hatch=s.hatch or None,
            edgecolor=common.BAR_EDGECOLOR,
            linewidth=0.5,
            alpha=common.BAR_ALPHA,
            label=s.label,
        )
        for s in drawn
    ]
    if io_modes:
        # Grey: these two say nothing about which system.
        handles += [
            mpatches.Patch(
                facecolor="0.85",
                hatch=IO_MODE_HATCH_KEY[mode] or None,
                edgecolor=common.BAR_EDGECOLOR,
                linewidth=0.5,
                label=IO_MODE_LABELS[mode],
            )
            for mode in IO_MODES
        ]
    _thin_hatch(handles)

    # First pass, anchored at the left edge only so there is something to measure.
    legend = fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.999),
        ncol=len(handles),
        fontsize=common.LEGEND_FONTSIZE - 0.5,
        frameon=True,
        edgecolor="black",
        borderaxespad=0,
        borderpad=0.25,
        columnspacing=0.5,
        labelspacing=0.15,
        handletextpad=0.25,
        handlelength=0.9,
    )
    legend.get_frame().set_linewidth(common.BORDER_LINE_WIDTH)
    y = (height_in - LEGEND_IN * 0.5) / height_in
    heading_text = fig.text(
        0.0,
        y,
        f"({HEADING})",
        ha="left",
        va="center",
        fontsize=common.TITLE_FONTSIZE,
        color=common.TITLE_COLOR,
    )

    # Widths by pixel width, not `transFigure.inverted()`, which carries an offset.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_w_px = fig.bbox.width
    legend_w = legend.get_window_extent(renderer).width / fig_w_px
    heading_w = heading_text.get_window_extent(renderer).width / fig_w_px
    total_w = legend_w + LEGEND_HEADING_GAP + heading_w
    left_x = (1.0 - total_w) / 2.0
    legend.set_bbox_to_anchor((left_x, 0.999), transform=fig.transFigure)
    heading_text.set_position((left_x + legend_w + LEGEND_HEADING_GAP, y))


def _drawn_keys(
    checkpoint,
    redis,
    grid: Dict[Tuple[str, str], personality_results.Cell],
) -> set:
    """Every cell key the figure holds a measurement under."""
    keys = set()
    for panel in (checkpoint, redis):
        if panel is not None:
            keys |= {system for _, system in panel.cells}
    return keys | {series for _, series in grid}


def _legend_series(
    checkpoint,
    redis,
    grid: Dict[Tuple[str, str], personality_results.Cell],
) -> List[Series]:
    """The systems with at least one bar in the figure, in legend order."""
    bases = set()
    for key in _drawn_keys(checkpoint, redis, grid):
        # "cvm-store@log" -> "cvm-store", "ext4/buffered" -> "ext4".
        bases.add(mlperf_panels.split_io_mode(key.split("@")[0])[0])
    return [s for s in SERIES if not s.io_mode and s.key in bases]


def plot_apps(
    checkpoint,
    redis,
    grid: Dict[Tuple[str, str], personality_results.Cell],
    personalities: Sequence[str],
    output_file: Path,
    *,
    fill_missing: bool = False,
    error_bars: bool = ERROR_BARS,
    png: bool = False,
    height_in: float = FIG_HEIGHT_IN,
) -> bool:
    """Write the figure. False, having said why, if there is nothing to draw."""
    common.apply_mpl_settings()
    # matplotlib takes the hatch stroke width from an rcParam, not from the artist.
    plt.rcParams["hatch.linewidth"] = HATCH_LINEWIDTH

    if checkpoint is None and redis is None and not grid:
        print("No usable runs in any half: nothing to plot")
        return False
    if height_in > FIG_HEIGHT_MAX_IN:
        print(
            f"{output_file.name}: {height_in:.3f} in exceeds the"
            f" {FIG_HEIGHT_MAX_IN:.3f} in the two figures it replaced took"
        )

    ckpt_groups = len(checkpoint.groups) if checkpoint else 0
    redis_groups = len(redis.groups) if redis else 0
    bench_groups = sum(1 for p in personalities if (p, BENCH_BASELINE_KEY) in grid)
    relative_groups = redis_groups + bench_groups

    # The relative panel's pitch is set by whichever half draws more bars.
    ckpt_bars = _bar_slots(checkpoint, _ckpt_series_for(checkpoint), fill_missing)
    relative_bars = max(
        _bar_slots(redis, _left_series(relative=True), fill_missing),
        len(_bench_bars(grid, fill_missing)) if bench_groups else 0,
        1,
    )

    fig = plt.figure(figsize=(FIG_WIDTH_IN, height_in))
    plot_w = FIG_WIDTH_IN - LEFT_IN - GUTTER_IN - RIGHT_IN
    left_slots = ckpt_groups * ckpt_bars
    total = left_slots + relative_groups * relative_bars
    # An empty panel still gets its axes, so the split falls back to even.
    left_w = plot_w * (left_slots / total if total else 0.5)
    right_w = plot_w - left_w
    axes_h = height_in - LEGEND_IN - BOTTOM_IN

    def box(x_in: float, w_in: float):
        return [
            x_in / FIG_WIDTH_IN,
            BOTTOM_IN / height_in,
            w_in / FIG_WIDTH_IN,
            axes_h / height_in,
        ]

    ax_ckpt = fig.add_axes(box(LEFT_IN, left_w))
    ax_relative = fig.add_axes(box(LEFT_IN + left_w + GUTTER_IN, right_w))

    captions = plot_checkpoint_panel(ax_ckpt, checkpoint, fill_missing, error_bars)
    captions += plot_relative_panel(
        ax_relative, redis, grid, personalities, fill_missing
    )
    # One size for all three captions: they are one row across the figure.
    _fit_texts(captions, size=common.ANNOTATION_SIZE, what="half-captions")
    keys = _drawn_keys(checkpoint, redis, grid)
    _legend_and_heading(
        fig,
        height_in,
        _legend_series(checkpoint, redis, grid),
        any(mlperf_panels.IO_MODE_SEP in key for key in keys),
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    # No bbox_inches="tight": the column fixes the MediaBox.
    fig.savefig(output_file, dpi=600)
    print(f"Wrote {output_file}")
    if png:
        preview = output_file.with_suffix(".png")
        fig.savefig(preview, dpi=600)
        print(f"Wrote {preview}")
    plt.close(fig)
    return True


def _col_width(columns: Sequence[Series]) -> int:
    """A console table's column width: its longest header, floored."""
    return max(12, max((len(c.label) for c in columns), default=12))


def _ratio_columns(value, baselines: Sequence[Tuple[str, float]]) -> str:
    """The trailing "vs X" cells of a table row."""
    out = []
    for _, base in baselines:
        out.append(
            f"{value / base:.2f}x".rjust(10) if value and base else "—".rjust(10)
        )
    return "".join(out)


def print_summary(
    checkpoint,
    redis,
    grid: Dict[Tuple[str, str], personality_results.Cell],
    personalities: Sequence[str],
    config: str,
    redis_config: str,
    bench_config: str,
) -> None:
    """The numbers behind the bars: one table per panel-half."""
    name_w = 22

    if checkpoint is not None:
        columns = _measured(checkpoint, _ckpt_series_for(checkpoint))
        width = _col_width(columns)
        print(f"\nMLPerf checkpointing, GiB/s ± stdev — {config}")
        print("group".ljust(name_w) + "".join(c.label.rjust(width) for c in columns))
        print("-" * (name_w + width * len(columns)))
        for group in checkpoint.groups:
            row = []
            for series in columns:
                cell = checkpoint.cells.get((group, series.key))
                row.append(
                    (
                        "—" if cell is None else f"{cell.value:.2f}±{cell.stdev:.2f}"
                    ).rjust(width)
                )
            print(group.ljust(name_w) + "".join(row))

    if redis is not None:
        columns = _measured(redis, _left_series(relative=True))
        width = _col_width(columns)
        print(f"\nredis SET ops/s — {redis_config}")
        header = "point".ljust(16) + "".join(c.label.rjust(width) for c in columns)
        print(header + "vs ext4".rjust(10) + "vs Rbc".rjust(10))
        print("-" * len(header + "x" * 20))
        for group in redis.groups:
            row = [group.ljust(16)]
            for series in columns:
                cell = redis.cells.get((group, series.key))
                row.append((_fmt_rate(cell.value) if cell else "—").rjust(width))
            ours = redis.cells.get((group, "cvm-store"))
            row.append(
                _ratio_columns(
                    ours.value if ours else 0.0,
                    [
                        (
                            (key, redis.cells[(group, key)].value)
                            if (group, key) in redis.cells
                            else (key, 0.0)
                        )
                        for key in ("ext4", "rollbaccine")
                    ],
                )
            )
            print("".join(row))
        print(
            "  (ratios > 1.00 are CVM-Store ahead; ext4 is the 1.0 the bars are"
            f" drawn against; Rollbaccine is {ROLLBACCINE_LABEL})"
        )

    columns = _bench_drawn(grid)
    width = _col_width(columns)
    # Its own name column: a personality name is longer than a checkpoint direction.
    if personalities:
        name_w = max(name_w, max(len(name) for name in personalities) + 2)
    print(f"\npersonality ops/s — {bench_config}")
    header = "personality".ljust(name_w) + "".join(
        c.label.rjust(width) for c in columns
    )
    print(header + "vs ext4".rjust(10) + "vs Rbc".rjust(10))
    print("-" * len(header + "x" * 20))
    rollbaccine_key = mlperf_panels.io_mode_key("rollbaccine", BENCH_BASELINE_MODE)
    for personality in personalities:
        row = [personality.ljust(name_w)]
        for series in columns:
            cell = grid.get((personality, series.key))
            row.append(
                (personality_results._fmt_value(cell.ops_per_s) + cell.marks).rjust(
                    width
                )
                if cell
                else "—".rjust(width)
            )
        ours = grid.get((personality, "cvm-store"))
        row.append(
            _ratio_columns(
                ours.ops_per_s if ours else 0.0,
                [
                    (
                        (key, grid[(personality, key)].ops_per_s)
                        if (personality, key) in grid
                        else (key, 0.0)
                    )
                    for key in (BENCH_BASELINE_KEY, rollbaccine_key)
                ],
            )
        )
        print("".join(row))
    print(
        f"  (ratios > 1.00 are CVM-Store ahead, against the {BENCH_BASELINE_MODE}"
        " arms; * a run whose workers waited)"
    )


def generate(
    results_root: Path,
    output: Path,
    *,
    personalities: Sequence[str] = PERSONALITIES,
    redis_points: Sequence[str] = REDIS_POINTS,
    config: Optional[str] = None,
    redis_config: Optional[str] = None,
    bench_config: Optional[str] = None,
    module: Optional[str] = None,
    fill_missing: bool = False,
    error_bars: bool = ERROR_BARS,
    png: bool = False,
    height_in: float = FIG_HEIGHT_IN,
) -> bool:
    """Collect all three halves from one result tree and write the figure.

    Each half is collected by the module that owns its format. False, having
    said why, if there was nothing to draw.

    Args:
        results_root: Checkout or result tree holding mlperf/, redis/ and bench/
        output: Path of the PDF to write
        personalities: Personality groups, in bar order
        redis_points: Redis points, in bar order
        config: MLPerf {setup}-disk-{size} config (default: the fullest one)
        redis_config: Redis result config (default: the one with most cells)
        bench_config: Bench result config (default: the one with the most runs)
        module: AccelStore module the checkpointing bars come from (default: one
            bar per module that ran)
        fill_missing: Mark the slots with no run instead of leaving them empty
        error_bars: Draw the spread whiskers on the MLPerf bars
        png: Also write a PNG next to the PDF
        height_in: Figure height in inches
    """
    tree = mlperf_panels._tree(results_root)
    checkpoint, config_name, notes = mlperf_panels.load_panels(
        tree, config=config, module=module
    )
    for note in notes:
        print(note)
    for line in mlperf_panels.missing_cells(checkpoint):
        print(line)
    print(f"Reading {tree} — config {config_name or '(none)'}")
    # Pinned to the resolved config so the extra passes cannot pick another machine.
    checkpoint, ckpt_modules = _checkpoint_by_module(
        tree, config_name, checkpoint, module
    )

    redis_dir = redis_panels.tree(results_root)
    redis, redis_name, redis_notes = redis_panels.load_panel(
        results_root, config=redis_config, points=redis_points
    )
    for note in redis_notes:
        print(note)
    for line in redis_panels.missing_cells(redis):
        print(line)
    print(f"Reading {redis_dir} — config {redis_name or '(none)'}")

    bench_dir = personality_results.resolve_bench_dir(results_root)
    bench_name, cells = personality_results.load_runs(bench_dir, bench_config)
    print(f"Reading {bench_dir} — config {bench_name}, {len(cells)} runs")
    grid, missing = _collect_bench(cells, personalities, _bench_series())
    for gap in missing:
        print(f"missing: personality / {gap}")

    ok = plot_apps(
        checkpoint,
        redis,
        grid,
        personalities,
        output,
        fill_missing=fill_missing,
        error_bars=error_bars,
        png=png,
        height_in=height_in,
    )
    if ok:
        print_summary(
            checkpoint,
            redis,
            grid,
            personalities,
            config_name,
            redis_name,
            bench_name,
        )
    return ok
