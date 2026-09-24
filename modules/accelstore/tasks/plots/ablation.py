#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The single-panel ablation figure, from `parse-ablation`'s rows.

Every bar is what one property adds to end-to-end runtime, in percent of
CVM-Store without any CIF, on a logarithmic axis.
"""

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from core.tasks.plotting import common

FIG_WIDTH_IN = 252.0 / 72.0  # \columnwidth of the paper's IEEEtran4 template
FIG_HEIGHT_MAX_IN = 103.9579355496 / 72.0  # ceiling: the slot this figure replaces
FIG_HEIGHT_IN = 1.32

NOISE_BAND_PCT = 1.0

GROUP_WIDTH = 0.8

FLOOR_PCT = 0.05  # axis floor, a log axis has no room for zero
BAR_BASE_PCT = FLOOR_PCT / 2.0  # below the floor, so the axes clip the bar's base
STUB_TOP_PCT = FLOOR_PCT * 1.8

# gridlines under the bars, the axis frame over them
GRID_ZORDER = 1
BAR_ZORDER = 3
AXIS_TOP_PCT = 300  # headroom for the range annotation above the tallest bar

CLIENT_LABELS = [
    ("kvs_client", "KVS"),
    ("fs_client", "FS"),
    ("log_client", "Log"),
]

CLIENT_COLORS = {
    "kvs_client": common.PALETTE_PASTEL[0],
    "fs_client": common.PALETTE_PASTEL[1],
    "log_client": common.PALETTE_PASTEL[2],
}
WITHIN_NOISE_COLOR = "0.72"

MIN_BAR_WIDTH_IN = 0.02  # below this a bar is thinner than its own outline

LEGEND_MAX_COLS = 4
LEGEND_ROW_IN = 0.085

HALF_CAPTION_IN = 0.10
GROUP_LABEL_FRAC = 0.26  # reach of the group labels, as a fraction of axes height


@dataclass(frozen=True)
class Arm:
    """One measured point: the build, the driver and the ablation switch."""

    crypto: str
    io_path: str
    arm: str


NO_CIF = Arm("none", "spdk", "unablated")  # the figure's denominator
CVM_STORE = Arm("aegis", "spdk", "unablated")


@dataclass(frozen=True)
class Series:
    """One bar of every group: a client, and the workload it ran."""

    client: str
    workload: str

    @property
    def client_label(self) -> str:
        return dict(CLIENT_LABELS).get(self.client, self.client)


@dataclass(frozen=True)
class Group:
    """One bar group: its label, and how its value is derived per series."""

    label: str
    value: Callable[["Runtimes", Series], Optional[float]]


GUARANTEE_GROUPS = [
    Group("CVM-Store\nw/ CIF", lambda r, s: r.added_pct(s, CVM_STORE)),
    Group(
        "AEGIS\nen/decryption",
        lambda r, s: r.share_pct(s, Arm("aegis", "spdk", "no-encrypt")),
    ),
]
DESIGN_GROUPS = [
    Group(
        "staging\ncopy",
        lambda r, s: r.added_pct(s, Arm("aegis", "spdk", "bounce-copy")),
    ),
    Group(
        "AES-256\nGCM",
        lambda r, s: r.added_pct(s, Arm("aesgcm", "spdk", "unablated")),
    ),
    Group(
        "no I/O\noverlap",
        lambda r, s: r.added_pct(s, Arm("aegis", "spdk", "no-overlap")),
    ),
    Group(
        "kernel\nNVMe drv",
        lambda r, s: r.added_pct(s, Arm("aegis", "kernel", "unablated")),
    ),
]
ALL_GROUPS = tuple(GUARANTEE_GROUPS) + tuple(DESIGN_GROUPS)

HALF_LABELS = ("what the guarantees cost", "what the design avoids")

# object size in a workload directory name: `_256m`, `_256MB_`, `_1g`
_SIZE_RE = re.compile(r"(?<![0-9a-z])(\d+)\s*(m|mb|g|gb|k|kb)(?![0-9a-z])", re.I)
_SIZE_UNIT_BYTES = {"k": 2**10, "m": 2**20, "g": 2**30}


def _size_of(workload: str) -> Optional[Tuple[int, str]]:
    """The object size a workload name carries, as (bytes, label), or None."""
    match = _SIZE_RE.search(workload)
    if not match:
        return None
    value, unit = int(match.group(1)), match.group(2).lower()[0]
    return value * _SIZE_UNIT_BYTES[unit], f"{value} {unit.upper()}iB"


class Runtimes:
    """The rows of one config, indexed by (client, workload, arm), in ms."""

    def __init__(self, rows: Sequence[dict]):
        self._ms: Dict[Tuple[str, str, Arm], float] = {}
        self._spans: Dict[Tuple[str, str, Arm], Dict[str, float]] = {}
        duplicated = []
        for row in rows:
            if row.get("status") != "ok":
                continue
            runtime = _float(row.get("runtime_ms"))
            if runtime is None:
                continue
            key = (
                row["client"],
                row["workload"],
                Arm(row["crypto"], row["io_path"], row["arm"]),
            )
            if key in self._ms:
                duplicated.append(key)
            self._ms[key] = runtime
            self._spans[key] = {
                name: value
                for name in ("encryption_ms", "metadata_ms", "merkle_ms", "tc_ms")
                if (value := _float(row.get(name))) is not None
            }
        if duplicated:
            warnings.warn(
                f"{len(duplicated)} identity/-ies appear twice in the rows"
                f" (e.g. {duplicated[0]}); the later row wins. Re-run"
                " `accelstore.parse-ablation`, whose repeats are merged."
            )

    def ms(self, series: Series, arm: Arm) -> Optional[float]:
        return self._ms.get((series.client, series.workload, arm))

    def span_ms(self, series: Series, arm: Arm, name: str) -> Optional[float]:
        """A trace bucket of one run, in ms. A span sum, not wall clock."""
        return self._spans.get((series.client, series.workload, arm), {}).get(name)

    def series(self) -> List[Series]:
        """Every (client, workload) with a baseline, by client then size."""
        found = {
            Series(client, workload)
            for client, workload, arm in self._ms
            if arm == NO_CIF
        }
        clients = [c for c, _ in CLIENT_LABELS]

        def order(item: Series):
            client_rank = (
                clients.index(item.client) if item.client in clients else len(clients)
            )
            size = _size_of(item.workload)
            return (client_rank, item.client, size[0] if size else 0, item.workload)

        return sorted(found, key=order)

    def added_pct(self, series: Series, arm: Arm) -> Optional[float]:
        """What running `arm` instead of the no-CIF baseline adds, in %."""
        base, value = self.ms(series, NO_CIF), self.ms(series, arm)
        if base is None or value is None:
            return None
        return 100.0 * (value - base) / base

    def share_pct(self, series: Series, switched_off: Arm) -> Optional[float]:
        """Full CIF minus the run without one mechanism, in % of the baseline."""
        base = self.ms(series, NO_CIF)
        full, without = self.ms(series, CVM_STORE), self.ms(series, switched_off)
        if base is None or full is None or without is None:
            return None
        return 100.0 * (full - without) / base


def _float(value) -> Optional[float]:
    """A CSV cell or an already-parsed number, or None for a blank."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def by_config(rows: Sequence[dict]) -> Dict[str, List[dict]]:
    """Split rows per machine config."""
    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        grouped.setdefault(row["config"], []).append(row)
    return grouped


def _series_style(
    series: Sequence[Series],
) -> Tuple[Dict[Series, tuple], Dict[Series, str]]:
    """Bar colour and legend label per series."""
    per_client: Dict[str, List[Series]] = {}
    for item in series:
        per_client.setdefault(item.client, []).append(item)

    colors: Dict[Series, tuple] = {}
    labels: Dict[Series, str] = {}
    for client, items in per_client.items():
        base = CLIENT_COLORS.get(client, common.PALETTE_PASTEL[7])
        for index, item in enumerate(items):
            if len(items) == 1:
                colors[item] = base
                labels[item] = item.client_label
                continue
            factor = 1.0 - 0.42 * (index / (len(items) - 1))
            colors[item] = tuple(c * factor for c in mcolors.to_rgb(base))
            size = _size_of(item.workload)
            labels[item] = f"{item.client_label} {size[1]}" if size else item.workload
    return colors, labels


def _column_major(handles: List, ncol: int) -> List:
    """Reorder handles so a column-filled legend reads in the given order."""
    nrows = -(-len(handles) // ncol)
    ordered = []
    for col in range(ncol):
        for row in range(nrows):
            index = row * ncol + col
            if index < len(handles):
                ordered.append(handles[index])
    return ordered


def _range_label(values: Sequence[float]) -> str:
    """The group's value range, as the annotation above its tallest bar."""
    low, high = min(values), max(values)
    digits = 0 if high >= 10 else 1
    if high - low < 0.05 * high:
        return f"{high:.{digits}f}%"
    return f"{low:.{digits}f}–{high:.{digits}f}%"


def plot_ablation(
    rows: Sequence[dict],
    output_file: Path,
    groups: Sequence[Group] = ALL_GROUPS,
    split_after: int = len(GUARANTEE_GROUPS),
    height_in: float = FIG_HEIGHT_IN,
) -> bool:
    """Write the ablation figure for one config's rows.

    `groups` are the bar groups in order, `split_after` how many of them belong
    to the first half (where the dividing rule goes) and `height_in` the figure
    height, capped at FIG_HEIGHT_MAX_IN. Returns False, having warned, when the
    rows carry no no-CIF baseline.
    """
    common.apply_mpl_settings()

    data = Runtimes(rows)
    series = data.series()
    if not series:
        warnings.warn(
            f"{output_file.name}: no DISABLE_CIF baseline run (crypto=none,"
            " io_path=spdk, arm=unablated) -- every bar is measured against it."
            " Run `accelstore.run-ablation --arms none:none`"
        )
        return False

    colors, labels = _series_style(series)

    values = {
        (index, item): group.value(data, item)
        for index, group in enumerate(groups)
        for item in series
    }
    saw_within_noise = any(
        value is not None and value <= FLOOR_PCT for value in values.values()
    )

    entries = len(series) + 1 + (1 if saw_within_noise else 0)
    ncol = min(entries, LEGEND_MAX_COLS)
    legend_rows = -(-entries // ncol)

    left, right = 0.30, 0.03
    bottom = 0.26 + HALF_CAPTION_IN
    top = 0.30 + (legend_rows - 1) * LEGEND_ROW_IN
    height_in = min(
        max(height_in, height_in + (legend_rows - 1) * LEGEND_ROW_IN),
        FIG_HEIGHT_MAX_IN,
    )
    if height_in > FIG_HEIGHT_MAX_IN:
        warnings.warn(
            f"{output_file.name}: {height_in:.3f} in is taller than the"
            f" {FIG_HEIGHT_MAX_IN:.3f} in slot this figure replaces"
        )
    fig = plt.figure(figsize=(FIG_WIDTH_IN, height_in))
    ax = fig.add_axes(
        [
            left / FIG_WIDTH_IN,
            bottom / height_in,
            (FIG_WIDTH_IN - left - right) / FIG_WIDTH_IN,
            (height_in - bottom - top) / height_in,
        ]
    )

    ax.set_yscale("log")
    ax.set_ylim(FLOOR_PCT, AXIS_TOP_PCT)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.axhspan(
        FLOOR_PCT,
        NOISE_BAND_PCT,
        color=common.PALETTE_PASTEL[7],
        alpha=0.35,
        lw=0,
        zorder=0,
    )

    step = GROUP_WIDTH / len(series)
    offsets = [-GROUP_WIDTH / 2 + step * (i + 0.5) for i in range(len(series))]
    group_pitch_in = (FIG_WIDTH_IN - left - right) / len(groups)
    if step * group_pitch_in < MIN_BAR_WIDTH_IN:
        warnings.warn(
            f"{output_file.name}: {len(series)} bars x {len(groups)} groups is"
            f" {step * group_pitch_in:.3f} in per bar at this width -- the bars"
            " are thinner than their own outline"
        )

    for index, group in enumerate(groups):
        tops: List[float] = []
        for item, offset in zip(series, offsets):
            value = values[(index, item)]
            if value is None:
                continue
            x = index + offset
            if value <= FLOOR_PCT:
                # at or below the baseline: a stub, never read as a quantity
                ax.bar(
                    x,
                    STUB_TOP_PCT - BAR_BASE_PCT,
                    bottom=BAR_BASE_PCT,
                    width=step,
                    color=WITHIN_NOISE_COLOR,
                    edgecolor=common.BAR_EDGECOLOR,
                    linewidth=common.BAR_LINEWIDTH,
                    zorder=BAR_ZORDER,
                )
            else:
                tops.append(value)
                ax.bar(
                    x,
                    value - BAR_BASE_PCT,
                    bottom=BAR_BASE_PCT,
                    width=step,
                    color=colors[item],
                    edgecolor=common.BAR_EDGECOLOR,
                    linewidth=common.BAR_LINEWIDTH,
                    zorder=BAR_ZORDER,
                )
        if tops:
            ax.text(
                index,
                max(tops) * 1.4,
                _range_label(tops),
                fontsize=common.ANNOTATION_SIZE,
                ha="center",
                va="bottom",
                color="0.2",
                zorder=6,
            )

    if 0 < split_after < len(groups):
        ax.axvline(split_after - 0.5, color="0.6", lw=0.5, ls=(0, (2.5, 2)), zorder=2)
        centres = (
            split_after / 2 - 0.5,
            split_after + (len(groups) - split_after) / 2 - 0.5,
        )
        for centre, label in zip(centres, HALF_LABELS):
            ax.text(
                centre,
                -GROUP_LABEL_FRAC,
                label,
                transform=ax.get_xaxis_transform(),
                fontsize=common.ANNOTATION_SIZE + 0.5,
                color="0.35",
                ha="center",
                va="top",
                style="italic",
                clip_on=False,
            )

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g.label for g in groups], fontsize=common.TICKS_FONTSIZE)
    ax.set_yticks([0.1, 1, 10, 100])
    ax.set_yticklabels(["0.1", "1", "10", "100"], fontsize=common.TICKS_FONTSIZE)
    # FixedLocator: a LogLocator prunes its subdivisions away over four decades
    ax.yaxis.set_minor_locator(
        mticker.FixedLocator(
            [
                mantissa * 10.0**exponent
                for exponent in range(-3, 4)
                for mantissa in (2, 5)
                if FLOOR_PCT < mantissa * 10.0**exponent < AXIS_TOP_PCT
            ]
        )
    )
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="y", which="minor", length=1.3, width=common.BORDER_LINE_WIDTH)
    ax.grid(
        True,
        axis="y",
        which="minor",
        linestyle="-",
        color="0.85",
        alpha=common.GRID_ALPHA,
        linewidth=common.GRID_LINE_WIDTH * 0.6,
        zorder=GRID_ZORDER,
    )
    ax.set_ylabel("added (%, log)", fontsize=common.TICKS_FONTSIZE + 0.5, labelpad=1.0)
    ax.set_title(
        f"Runtime vs. CVM-Store without CIF ({common.LOWER_BETTER_TITLE})",
        fontsize=common.TITLE_FONTSIZE,
        pad=2.5,
        color=common.TITLE_COLOR,
    )
    ax.grid(
        True,
        axis="y",
        linestyle="--",
        alpha=common.GRID_ALPHA,
        linewidth=common.GRID_LINE_WIDTH,
        zorder=GRID_ZORDER,
    )
    ax.grid(False, axis="x")
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.tick_params(
        axis="y", which="major", length=2.4, width=common.BORDER_LINE_WIDTH, pad=1.5
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(common.BORDER_LINE_WIDTH)
        spine.set_zorder(BAR_ZORDER + 1)  # else the bars overdraw the bottom spine

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            fc=colors[item],
            ec=common.BAR_EDGECOLOR,
            lw=common.BAR_LINEWIDTH,
            label=labels[item],
        )
        for item in series
    ]
    handles.append(
        plt.Rectangle(
            (0, 0),
            1,
            1,
            fc=common.PALETTE_PASTEL[7],
            alpha=0.35,
            ec="none",
            label=f"within noise (≤{NOISE_BAND_PCT:g}%)",
        )
    )
    if saw_within_noise:
        handles.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                fc=WITHIN_NOISE_COLOR,
                ec=common.BAR_EDGECOLOR,
                lw=common.BAR_LINEWIDTH,
                label="at/below baseline",
            )
        )
    fig.legend(
        handles=_column_major(handles, ncol),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.999),
        ncol=ncol,
        fontsize=common.LEGEND_FONTSIZE - 0.5,
        frameon=True,
        edgecolor="black",
        borderaxespad=0,
        borderpad=0.3,
        columnspacing=0.6,
        labelspacing=0.15,
        handletextpad=0.3,
        handlelength=1.0,
    )
    fig.legends[0].get_frame().set_linewidth(common.BORDER_LINE_WIDTH)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    # no bbox_inches="tight": the MediaBox must stay the fixed slot
    fig.savefig(output_file, dpi=600)
    plt.close(fig)
    return True


def print_ablation_summary(rows: Sequence[dict]) -> None:
    """The figure's own numbers, per config, as a table."""
    for config, config_rows in sorted(by_config(rows).items()):
        data = Runtimes(config_rows)
        series = data.series()
        if not series:
            continue
        _, labels = _series_style(series)
        width = max(max(len(labels[s]) for s in series) + 2, 10)
        print(f"\nAdded end-to-end runtime vs. CVM-Store without CIF — {config}")
        header = f"{'Property':<20}" + "".join(f"{labels[s]:>{width}}" for s in series)
        print(header)
        print("─" * len(header))
        for group in ALL_GROUPS:
            cells = ""
            for item in series:
                value = group.value(data, item)
                cells += (
                    f"{'—':>{width}}" if value is None else f"{value:>{width - 1}.2f}%"
                )
            print(f"{group.label.replace(chr(10), ' '):<20}{cells}")
        base = "".join(
            f"{data.ms(s, NO_CIF) / 1000.0:>{width - 1}.1f}s" for s in series
        )
        print(f"{'baseline (no CIF)':<20}{base}")
        _print_span_table(data, series, labels, width)


def _print_span_table(
    data: "Runtimes",
    series: Sequence[Series],
    labels: Dict[Series, str],
    width: int,
) -> None:
    """The trace buckets behind the CVM-Store bar."""
    buckets = (
        ("encryption span", "encryption_ms", CVM_STORE),
        ("  of no-encrypt", "encryption_ms", Arm("aegis", "spdk", "no-encrypt")),
        ("  of no-CIF build", "encryption_ms", NO_CIF),
        ("merkle span", "merkle_ms", CVM_STORE),
        ("metadata span", "metadata_ms", CVM_STORE),
        ("counter span", "tc_ms", CVM_STORE),
    )
    print("\nTrace buckets of the same runs (ms; a span sum, not wall clock)")
    for label, name, arm in buckets:
        cells = ""
        for item in series:
            value = data.span_ms(item, arm, name)
            cells += f"{'—':>{width}}" if value is None else f"{value:>{width}.1f}"
        print(f"{label:<20}{cells}")
    print(
        "  encryption/runtime:".ljust(20)
        + "".join(
            _ratio_cell(
                data.span_ms(s, CVM_STORE, "encryption_ms"),
                data.ms(s, CVM_STORE),
                width,
            )
            for s in series
        )
    )


def _ratio_cell(
    numerator: Optional[float], denominator: Optional[float], width: int
) -> str:
    if numerator is None or not denominator:
        return f"{'—':>{width}}"
    return f"{100.0 * numerator / denominator:>{width - 1}.0f}%"
