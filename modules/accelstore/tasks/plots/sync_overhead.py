"""Metadata epoch sweep: the synchronization-overhead figure and its tables."""

import concurrent.futures
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import seaborn as sns

from core.tasks.plotting import common
from ..crypto_backend import CryptoBackend
from ..results import discovery, sync_log
from ..results.trace_parser import (
    OperationMetrics,
    OutputMetrics,
    average_metrics,
    load_or_parse_trace,
)
from ..sync_modes import SyncMode, is_sync_component, parse_path_component


def _workload_sort_key(workload_name: str) -> Tuple[int, str, str]:
    """Sort key grouping workloads by client type, then by size and name."""
    if workload_name.startswith("kvs_"):
        client_order = 0
    elif workload_name.startswith("log_"):
        client_order = 1
    elif workload_name.startswith("fs_"):
        client_order = 2
    else:
        client_order = 3

    size_match = re.search(r"(\d+)\s*(mb|gb|m|g)", workload_name, re.IGNORECASE)
    if size_match:
        num = int(size_match.group(1))
        unit = size_match.group(2).lower()
        if unit in ("gb", "g"):
            num *= 1024
        size_key = f"{num:012d}"
    else:
        size_key = workload_name

    return (client_order, size_key, workload_name)


def _workload_label(workload_name: str) -> str:
    """Human-readable x-axis label for a workload directory name."""
    name = workload_name

    if name.startswith("kvs_workload_"):
        remainder = name[len("kvs_workload_") :]
        match = re.search(r"(\d+\s*(?:mb|gb|m|g)b?)", remainder, re.IGNORECASE)
        if match:
            size = match.group(1).upper()
            if size.endswith("M"):
                size += "B"
            return f"KVS {size}"
        return f"KVS {remainder}"

    elif name.startswith("log_workload_"):
        remainder = name[len("log_workload_") :]
        match = re.search(r"(\d+\s*(?:mb|gb|m|g)b?)", remainder, re.IGNORECASE)
        if match:
            size = match.group(1).upper()
            if size.endswith("M"):
                size += "B"
            return f"Log {size}"
        return f"Log {remainder}"

    elif name.startswith("fs_workload_"):
        remainder = name[len("fs_workload_") :]
        match = re.search(r"(\d+\s*(?:mb|gb|m|g)b?)", remainder, re.IGNORECASE)
        if match:
            size = match.group(1).upper()
            if size.endswith("M"):
                size += "B"
            return f"FS {size}"
        return f"FS {remainder}"

    return workload_name


EPOCH_AVERAGE_RUNS_DEFAULT = 2


@dataclass
class SyncPoint:
    """One sweep point: a workload run under one metadata persistence mode."""

    client_type: str  # "kvs", "fs", "log"
    crypto: CryptoBackend
    workload_name: str
    mode: SyncMode
    metrics: OutputMetrics
    log: Optional[sync_log.RunLog] = None  # fallback for dumps without barrier tpoints
    runs: int = 1

    @property
    def throughput_mibps(self) -> float:
        return self.metrics.api.throughput_mibps

    def _phase(self, name: str) -> Optional[OperationMetrics]:
        """One epoch phase's spans, or None when the run captured none."""
        return self.metrics.epoch.sub_operations.get(name)

    @property
    def epoch_count(self) -> int:
        """Epochs the run did, counted from the snapshot each one takes."""
        snapshot = self._phase("snapshot")
        return snapshot.operation_count if snapshot else 0

    @property
    def snapshot_ms(self) -> float:
        """Reactor stall at the head of the epochs: the synchronous part."""
        snapshot = self._phase("snapshot")
        return snapshot.total_time_us / 1000.0 if snapshot else 0.0

    @property
    def gate_ms(self) -> float:
        """What the epochs cost the operations they blocked, summed."""
        gate = self._phase("gate")
        return gate.total_time_us / 1000.0 if gate else 0.0

    @property
    def commit_ms(self) -> float:
        """Counter round trips the epochs paid: the freshness line item."""
        commit = self._phase("commit")
        return commit.total_time_us / 1000.0 if commit else 0.0

    @property
    def drain_ms(self) -> float:
        """What the epochs waited for the operations already in flight."""
        drain = self._phase("drain")
        return drain.total_time_us / 1000.0 if drain else 0.0

    @property
    def barrier_ms(self) -> float:
        """The pre-superblock ordering barrier: the epoch's last wait."""
        barrier = self._phase("barrier")
        return barrier.total_time_us / 1000.0 if barrier else 0.0

    @property
    def epoch_wall_ms(self) -> float:
        """Wall time inside the epochs, from the enclosing META_EPOCH span.

        The phases nest inside this span and overlap, so it is not their sum.
        """
        epoch = self._phase("epoch")
        return epoch.total_time_us / 1000.0 if epoch else 0.0

    @property
    def epoch_wall_count(self) -> int:
        """Epoch attempts the enclosing span saw, where epoch_count counts snapshots."""
        epoch = self._phase("epoch")
        return epoch.operation_count if epoch else 0

    @property
    def epoch_wall_mean_ms(self) -> Optional[float]:
        """Mean wall time of one epoch, or None when no epoch span was seen."""
        count = self.epoch_wall_count
        return self.epoch_wall_ms / count if count else None

    @property
    def metadata_ms(self) -> float:
        """Metadata sync time: the durability line item."""
        return self.metrics.metadata.total_time_us / 1000.0

    @property
    def barrier_verdicts(self) -> Dict[str, Dict[str, float]]:
        """Per-outcome barrier counts and margins, empty for an old dump."""
        return self.metrics.epoch_barrier or {}

    @property
    def has_barrier_verdicts(self) -> bool:
        """True when the trace itself says how the epochs' barriers went."""
        return bool(self.metrics.epoch_barrier)

    def barrier_count(self, outcome: str) -> int:
        """Epochs whose barrier reached ``outcome`` (trace_parser's names)."""
        return int(self.barrier_verdicts.get(outcome, {}).get("count", 0))

    @property
    def race_won(self) -> Optional[int]:
        """Epochs whose commit landed after their metadata blob was durable."""
        return self.barrier_count("race_won") if self.has_barrier_verdicts else None

    @property
    def race_lost(self) -> Optional[int]:
        """Epochs whose commit landed *before* their own blob was durable."""
        if self.has_barrier_verdicts:
            return self.barrier_count("race_lost")
        return self.log.race_lost if self.log else None

    @property
    def race_source(self) -> str:
        """Where the ordering verdicts came from: trace, log, or nothing."""
        if self.has_barrier_verdicts:
            return "trace"
        return "log" if self.log else "none"

    @property
    def race_barriers(self) -> Optional[int]:
        """Barriers the verdicts were taken over: the share's denominator."""
        if self.has_barrier_verdicts:
            return sum(int(s.get("count", 0)) for s in self.barrier_verdicts.values())
        return self.log.commits if self.log else None

    @property
    def race_lost_share(self) -> Optional[float]:
        """Fraction of the barriers whose ordering did not hold."""
        lost, total = self.race_lost, self.race_barriers
        if lost is None or not total:
            return None
        return lost / total

    @property
    def race_lost_margin_mean_us(self) -> Optional[float]:
        """Mean margin by which a lost race was lost, in µs."""
        if self.has_barrier_verdicts:
            stats = self.barrier_verdicts.get("race_lost")
            count = int(stats.get("count", 0)) if stats else 0
            if not stats or not count:
                return None
            return stats.get("margin_us_total", 0.0) / count
        if self.log is None:
            return None
        return (
            float(self.log.race_lost_mean_margin_us)
            if self.log.race_lost_mean_margin_us is not None
            else None
        )

    @property
    def race_lost_margin_max_us(self) -> Optional[float]:
        """Worst single lost-race margin, µs. The trace alone carries it."""
        stats = self.barrier_verdicts.get("race_lost")
        if not stats or not int(stats.get("count", 0)):
            return None
        return stats.get("margin_us_max", 0.0)

    @property
    def commits_local(self) -> int:
        """Commits a service answered on the caller's stack: no round trip."""
        return self.metrics.trusted_counter.commits_local

    @property
    def commits_remote(self) -> int:
        """Commits the mock remote service answered, i.e. paying ones."""
        return self.metrics.trusted_counter.commits_remote

    @property
    def commit_service(self) -> str:
        """The service the epochs' commits were actually answered by."""
        local, remote = self.commits_local, self.commits_remote
        if local and remote:
            return "mixed"
        if remote:
            return "remote"
        if local:
            return "local"
        return "unknown"


def _process_sync_trace_task(args):
    """Load one sweep point (module-level so it pickles for the pool)."""
    client_full, client_type, crypto, workload_name, trace_paths, slug = args
    label = f"{workload_name}/{slug}" if slug else f"{workload_name} (baseline)"
    asked = f" (mean of {len(trace_paths)} runs)" if len(trace_paths) > 1 else ""
    print(f"  Loading: {client_full}/{label}{asked}")
    try:
        mode = parse_path_component(slug)
    except ValueError as e:
        print(f"  ✗ Skipping {client_full}/{label}: {e}")
        return None

    runs: List[OutputMetrics] = []
    logs: List[sync_log.RunLog] = []
    for trace_path in trace_paths:
        try:
            runs.append(load_or_parse_trace(trace_path))
        except Exception as e:
            print(f"  ✗ Error ({client_full}/{label}, {trace_path.name}): {e}")
            continue
        log_path = sync_log.find_run_log(trace_path)
        if log_path:
            logs.append(sync_log.parse_run_log(log_path))

    if not runs:
        return None

    return SyncPoint(
        client_type=client_type,
        crypto=crypto,
        workload_name=workload_name,
        mode=mode,
        metrics=average_metrics(runs),
        log=sync_log.average_run_logs(logs),
        runs=len(runs),
    )


def discover_sync_metrics(
    bench_dir: Path,
    sweep_workloads: Dict[str, List[str]],
    average_runs: int = EPOCH_AVERAGE_RUNS_DEFAULT,
) -> Dict[str, List[SyncPoint]]:
    """Discover the epoch sweeps' runs, keyed by workload.

    Scans each workload directory (the baseline, default mode) and every sync
    subdirectory under it (one sweep point each).

    Args:
        bench_dir: root benchmark directory, e.g. bench-result/accelstore
        sweep_workloads: client name (e.g. "kvs_client") → workload .txt names
        average_runs: most recent runs per point to fold into one mean

    Returns:
        workload name → its points, baseline included when present.
    """
    average_runs = max(1, int(average_runs))
    print(f"\nDiscovering metadata epoch runs in {bench_dir}...")
    if average_runs > 1:
        print(f"  Each point is the mean of its {average_runs} most recent runs")

    wanted = {
        client: {Path(job).stem for job in jobs}
        for client, jobs in sweep_workloads.items()
    }

    tasks_args = []
    for config_dir in sorted(bench_dir.glob("*")):
        if not config_dir.is_dir():
            continue

        for client_dir in sorted(config_dir.glob("*")):
            if not client_dir.is_dir():
                continue
            base_client, crypto = CryptoBackend.split_client_binary(client_dir.name)
            if base_client not in wanted:
                continue
            client_type = base_client.removesuffix("_client")

            for workload_dir in sorted(client_dir.glob("*")):
                if not workload_dir.is_dir():
                    continue
                if workload_dir.name not in wanted[base_client]:
                    continue

                scan = [(workload_dir, None)]
                scan += [
                    (sub, sub.name)
                    for sub in sorted(workload_dir.iterdir())
                    if sub.is_dir() and is_sync_component(sub.name)
                ]

                for scan_dir, slug in scan:
                    traces = discovery.find_latest_traces(scan_dir, average_runs)
                    if not traces:
                        continue
                    tasks_args.append(
                        (
                            client_dir.name,
                            client_type,
                            crypto,
                            workload_dir.name,
                            traces,
                            slug,
                        )
                    )

    result: Dict[str, List[SyncPoint]] = {}
    if tasks_args:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for point in executor.map(_process_sync_trace_task, tasks_args):
                if point is not None:
                    result.setdefault(point.workload_name, []).append(point)

    points = sum(len(v) for v in result.values())
    loaded = sum(point.runs for v in result.values() for point in v)
    print(f"\nLoaded {loaded} runs as {points} points across {len(result)} workloads")
    short = sum(1 for v in result.values() for point in v if point.runs < average_runs)
    if short:
        print(
            f"  ⚠ {short} of {points} points had fewer than {average_runs} runs"
            f" to average — those rows are not the same measurement as the rest"
        )
    return result


def _baseline_of(points: List[SyncPoint]) -> Optional[SyncPoint]:
    """The default-mode run the overheads of a workload are relative to."""
    for point in points:
        if point.mode.is_default():
            return point
    return None


def _relative_overhead(point: SyncPoint, baseline: SyncPoint) -> Optional[float]:
    """Throughput lost against the baseline, in percent (0 = none)."""
    if baseline.throughput_mibps <= 0:
        return None
    return (1.0 - point.throughput_mibps / baseline.throughput_mibps) * 100.0


def _arm(mode: SyncMode) -> str:
    """The arm a point measures: durability, durability+freshness, blocking or not."""
    if mode.metadata_only:
        arm = "durability"
    elif mode.rtt_us:
        arm = "durability+freshness"
    else:
        arm = "durability (local counter)"
    return f"{arm}, blocking" if mode.blocking else arm


def _epoch_axis(
    mode: SyncMode, target_axis: Optional[str] = None
) -> Optional[Tuple[str, int]]:
    """The knob a point varies, as (axis, value), or None if it varies none."""
    if mode.effective_trigger == "per-op":  # worst case: 1 op per epoch, no window
        if target_axis == "every-n":
            return ("every-n", 1)
        return ("every-t", 0)
    if mode.epoch_us is not None:
        return ("every-t", mode.epoch_us)
    if mode.epoch_ops is not None:
        return ("every-n", mode.epoch_ops)
    return None


def _averaging_note(sync_data: Dict[str, List[SyncPoint]]) -> Optional[str]:
    """One line on how many runs the points are means of, or None for one each."""
    counts = {point.runs for points in sync_data.values() for point in points}
    if not counts or counts == {1}:
        return None
    low, high = min(counts), max(counts)
    if low == high:
        return f"Every cell is the mean of its point's {high} most recent runs"
    return (
        "Every cell is the mean of its point's most recent runs"
        f" ({low}-{high} of them, per point)"
    )


def print_epoch_table(sync_data: Dict[str, List[SyncPoint]]) -> None:
    """Print overhead against the baseline, rows = workload × arm, columns = epoch.

    One table per epoch axis, since every-t counts milliseconds and every-n
    counts operations.
    """
    if not sync_data:
        print("No metadata epoch data available.")
        return

    for axis, unit in (("every-t", "ms"), ("every-n", "ops")):
        rows = []
        values: set = set()
        for workload_name, points in sorted(
            sync_data.items(), key=lambda x: _workload_sort_key(x[0])
        ):
            baseline = _baseline_of(points)
            if baseline is None:
                print(
                    f"  ⚠ {workload_name}: no baseline run (the default mode, in the"
                    f" workload directory itself) — overheads not computable"
                )
                continue

            by_arm: Dict[str, Dict[int, float]] = {}
            for point in points:
                found = _epoch_axis(point.mode, target_axis=axis)
                if not found or found[0] != axis:
                    continue
                overhead = _relative_overhead(point, baseline)
                if overhead is None:
                    continue
                by_arm.setdefault(_arm(point.mode), {})[found[1]] = overhead
                values.add(found[1])

            for arm, per_value in sorted(by_arm.items()):
                rows.append((workload_name, arm, per_value))

        if not rows:
            continue

        ordered = sorted(values)
        print()
        print(f"Metadata epoch overhead vs baseline — trigger {axis} (lower is better)")
        note = _averaging_note(sync_data)
        if note:
            print(note)
        # 32 fits the longest arm name, "durability+freshness, blocking".
        header = f"{'Workload':<28}{'Arm':<32}" + "".join(
            f"{(v / 1000 if axis == 'every-t' else v):>{13 - len(unit)}.0f}{unit}"
            for v in ordered
        )
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for workload_name, arm, per_value in rows:
            line = f"{_workload_label(workload_name):<28}{arm:<32}"
            for value in ordered:
                overhead = per_value.get(value)
                line += f"{'N/A':>13}" if overhead is None else f"{overhead:>12.1f}%"
            print(line)
        print("-" * len(header))


def plot_epoch_overhead(
    sync_data: Dict[str, List[SyncPoint]],
    output_file: Path,
    axis: str = "every-t",
) -> None:
    """Plot relative overhead against the epoch, one line per workload and arm.

    Only blocking variants are plotted, with per-op as the worst-case boundary.
    Color is per client, marker shape per arm.

    Args:
        sync_data: output of discover_sync_metrics()
        output_file: output file path
        axis: "every-t" (epoch window in ms) or "every-n" (mutating ops per epoch)
    """
    PER_OP_OFFSET_MS = -5.0

    common.apply_mpl_settings()

    series: Dict[Tuple[str, str, str], List[Tuple[float, float]]] = {}
    for workload_name, points in sync_data.items():
        baseline = _baseline_of(points)
        if baseline is None:
            warnings.warn(f"No baseline run for {workload_name}, skipping")
            continue
        for point in points:
            arm = _arm(point.mode)
            if "local counter" in arm:
                continue
            # per-op is blocking by definition
            if not point.mode.blocking and point.mode.trigger != "per-op":
                continue

            found = _epoch_axis(point.mode, target_axis=axis)
            if not found or found[0] != axis:
                continue
            overhead = _relative_overhead(point, baseline)
            if overhead is None:
                continue
            x = found[1] / 1000.0 if axis == "every-t" else float(found[1])
            series.setdefault((point.client_type, workload_name, arm), []).append(
                (x, overhead)
            )

    if not series:
        warnings.warn(f"No metadata epoch data on axis {axis} to plot")
        return

    fig, ax = common.create_figure(
        ax_width=common.AX_WIDTH,
        ax_height=common.AX_HEIGHT_SMALL2,
        left_margin=common.AX_LEFT_MARGIN,
        right_margin=common.AX_RIGHT_MARGIN,
        top_margin=common.AX_TOP_MARGIN - 1,  # room for the legend
        bottom_margin=common.AX_BOTTOM_MARGIN,
    )

    unique_clients = sorted({ct for ct, _, _ in series.keys()})
    colors = sns.color_palette("deep", n_colors=max(len(unique_clients), 1))

    def _get_client_color(client_type: str) -> str:
        idx = unique_clients.index(client_type) if client_type in unique_clients else 0
        return colors[idx % len(colors)]

    arm_configs = {
        "durability+freshness, blocking": {
            "label": "Durability + Freshness (blocking)",
            "marker": "o",
            "style": "-",
        },
        "durability, blocking": {
            "label": "Durability only (blocking)",
            "marker": "s",
            "style": "--",
        },
        "durability+freshness": {
            "label": "Durability + Freshness (blocking)",
            "marker": "o",
            "style": "-",
        },
        "durability": {
            "label": "Durability only (blocking)",
            "marker": "s",
            "style": "--",
        },
    }

    unique_arms = sorted({arm for _, _, arm in series.keys()})
    fallback_markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]

    for (client_type, workload_name, arm), values in sorted(
        series.items(), key=lambda kv: (_workload_sort_key(kv[0][1]), kv[0][2])
    ):
        values.sort(key=lambda v: v[0])
        cfg = arm_configs.get(arm, {})
        marker = cfg.get(
            "marker",
            fallback_markers[
                (
                    unique_arms.index(arm) % len(fallback_markers)
                    if arm in unique_arms
                    else 0
                )
            ],
        )
        linestyle = cfg.get("style", "-")
        color = _get_client_color(client_type)

        per_op_points = [v for v in values if v[0] == 0.0]
        ladder_points = [v for v in values if v[0] > 0.0]

        if ladder_points:
            ax.plot(
                [v[0] for v in ladder_points],
                [v[1] for v in ladder_points],
                color=color,
                marker=marker,
                markersize=common.MARKER_SIZE,
                linewidth=common.LINE_WIDTH,
                linestyle=linestyle,
            )

        # per-op has no window, so it is drawn off-axis at PER_OP_OFFSET_MS
        if per_op_points:
            ax.plot(
                [PER_OP_OFFSET_MS for _ in per_op_points],
                [v[1] for v in per_op_points],
                color=color,
                marker=marker,
                markersize=common.MARKER_SIZE,
                linestyle="None",
                zorder=10,
                clip_on=False,
            )

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    # ±1% band and the 2.5% line separating low- from high-overhead regimes
    ax.axhspan(-1.0, 1.0, color="gray", alpha=0.15, zorder=0, linewidth=0)
    ax.axhline(y=2.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.8, zorder=1)
    ax.set_yticks([0, 2.5, 5, 7.5, 10])

    # the 3pt offset below matches the y-tick label pad apply_axis_style uses
    label_trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for y_value, label in ((0.0, "±1%"), (2.5, "2.5%")):
        ax.annotate(
            label,
            xy=(1.0, y_value),
            xycoords=label_trans,
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=common.TICKS_FONTSIZE,
            color="gray",
            zorder=11,
            annotation_clip=False,
        )

    common.apply_axis_style(
        ax,
        title=f"Metadata Epoch Overhead ({common.LOWER_BETTER_TITLE})",
        xlabel=(
            "Epoch window (ms)"
            if axis == "every-t"
            else "Mutating operations per epoch"
        ),
        ylabel="Relative Overhead (%)",
        grid=True,
    )
    if axis == "every-n":
        ax.set_xscale("log")
        ax.set_xlim(left=0.8)
    else:
        ax.set_xlim(PER_OP_OFFSET_MS)

    ax.set_axisbelow(True)

    client_label_map = {"kvs": "KVS", "log": "Log", "fs": "FS"}
    client_handles = [
        mlines.Line2D(
            [],
            [],
            color=_get_client_color(ct),
            linewidth=common.LINE_WIDTH * 1.5,
            label=client_label_map.get(ct.lower(), ct.upper()),
        )
        for ct in unique_clients
    ]

    mode_handles = [
        mlines.Line2D(
            [],
            [],
            color="black",
            marker="o",
            markersize=common.MARKER_SIZE,
            linewidth=common.LINE_WIDTH,
            linestyle="-",
            label="Durability + Freshness",
        ),
        mlines.Line2D(
            [],
            [],
            color="black",
            marker="s",
            markersize=common.MARKER_SIZE,
            linewidth=common.LINE_WIDTH,
            linestyle="--",
            label="Durability only",
        ),
    ]

    legend_handles = client_handles + mode_handles

    leg = ax.legend(
        handles=legend_handles,
        loc=common.LEGEND_ALIGN,
        bbox_to_anchor=(0.5, common.LEGEND_BBOX_HEIGHT + 0.05),
        ncol=min(len(legend_handles), 5),
        fontsize=common.LEGEND_FONTSIZE,
        edgecolor="black",
        borderaxespad=0,
        frameon=True,
        columnspacing=0.3,
        labelspacing=0.2,
        borderpad=0.5,
        handletextpad=0.4,
    )
    leg.set_zorder(1)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✓ Generated: {output_file}")
