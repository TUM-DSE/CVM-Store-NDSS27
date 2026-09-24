"""Paper-quality FIO IOPS and bandwidth bar charts."""

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

from modules.fio.tasks.plotting import fio_parser
from core.tasks.plotting import common
from core.tasks.plotting.common import DenseHatchHandler


def find_latest_json(directory: Path) -> Optional[Path]:
    """Return the lexicographically last *.json in directory, or None."""
    directory = Path(directory)

    if not directory.exists():
        return None

    json_files = list(directory.glob("*.json"))

    if not json_files:
        return None

    latest = max(json_files, key=lambda f: f.stem)
    return latest


def load_fio_data_from_setups(
    setups: List[Dict[str, str]],
) -> Tuple[List[str], Dict[str, Dict[str, fio_parser.FIOData]]]:
    """Parse the latest JSON in each setup path into per-job FIO data."""
    setup_names = []
    fio_data = {}

    for setup in setups:
        name = setup["name"]
        path = Path(setup["path"])

        json_file = find_latest_json(path)

        if json_file is None:
            print(f"Warning: Skipping {name} - no valid JSON file found in {path}")
            continue

        print(f"Loading {name} from {json_file}")

        try:
            data = fio_parser.read_fio_json(json_file)
            parsed_jobs = fio_parser.parse_fio_file(json_file)

            setup_names.append(name)
            fio_data[name] = parsed_jobs

        except Exception as e:
            print(f"Error parsing {json_file}: {e}")
            continue

    return setup_names, fio_data


def prepare_iops_data(
    setup_names: List[str], fio_data: Dict[str, Dict[str, fio_parser.FIOData]]
) -> Tuple[List[str], List[str], np.ndarray]:
    """Build a [workload x setup] IOPS matrix and the workload labels."""
    workload_mapping = {
        "iops randread": "RandR",
        "iops randwrite": "RandW",
        "iops rwmixread": "RRW70",
        "iops rwmixwrite": "RRW30",
    }

    iops_jobs = fio_parser.IOPS_JOBS
    workload_labels = [workload_mapping[job] for job in iops_jobs]

    num_workloads = len(iops_jobs)
    num_setups = len(setup_names)
    iops_matrix = np.zeros((num_workloads, num_setups))

    for setup_idx, setup_name in enumerate(setup_names):
        if setup_name not in fio_data:
            continue

        job_data = fio_data[setup_name]

        for workload_idx, job in enumerate(iops_jobs):
            if job not in job_data:
                continue

            data = job_data[job]

            if job in ["iops randread", "iops rwmixread"]:
                total_iops = data.read_iops_mean
            elif job == "iops randwrite":
                total_iops = data.write_iops_mean
            else:  # "iops rwmixwrite"
                total_iops = data.read_iops_mean + data.write_iops_mean

            iops_matrix[workload_idx, setup_idx] = total_iops

    return workload_labels, setup_names, iops_matrix


def prepare_bw_data(
    setup_names: List[str], fio_data: Dict[str, Dict[str, fio_parser.FIOData]]
) -> Tuple[List[str], List[str], np.ndarray]:
    """Build a [workload x setup] bandwidth matrix in KiB/s and the workload labels."""
    workload_mapping = {"bw read": "Read", "bw write": "Write"}

    bw_jobs = fio_parser.BW_JOBS
    workload_labels = [workload_mapping[job] for job in bw_jobs]

    num_workloads = len(bw_jobs)
    num_setups = len(setup_names)
    bw_matrix = np.zeros((num_workloads, num_setups))

    for setup_idx, setup_name in enumerate(setup_names):
        if setup_name not in fio_data:
            continue

        job_data = fio_data[setup_name]

        for workload_idx, job in enumerate(bw_jobs):
            if job not in job_data:
                continue

            data = job_data[job]

            if job == "bw read":
                bw_value = data.read_bw_mean
            else:  # "bw write"
                bw_value = data.write_bw_mean

            bw_matrix[workload_idx, setup_idx] = bw_value

    return workload_labels, setup_names, bw_matrix


def _build_split_legend_handles(
    setups: List[Dict[str, str]],
    setup_colors: list,
    setup_hatches: list,
) -> List[mpatches.Patch]:
    """One legend patch per unique color and per unique hatch."""
    handles: List[mpatches.Patch] = []
    seen_colors: set = set()
    seen_hatches: set = set()

    for setup, color in zip(setups, setup_colors):
        color_key = str(color)
        if color_key in seen_colors:
            continue
        seen_colors.add(color_key)
        label = setup.get("color_label", setup.get("name", ""))
        handles.append(
            mpatches.Patch(
                facecolor=color,
                edgecolor=common.BAR_EDGECOLOR,
                linewidth=0.5,
                alpha=common.BAR_ALPHA,
                label=label,
            )
        )

    for setup, hatch in zip(setups, setup_hatches):
        if hatch in seen_hatches:
            continue
        seen_hatches.add(hatch)
        label = setup.get("hatch_label", setup.get("name", ""))
        handles.append(
            mpatches.Patch(
                facecolor="white",
                hatch=hatch,
                edgecolor=common.BAR_EDGECOLOR,
                linewidth=0.5,
                label=label,
            )
        )

    return handles


def plot_iops_comparison(
    workload_labels: List[str],
    setup_names: List[str],
    iops_matrix: np.ndarray,
    setups: List[Dict[str, str]],
    output_path: Path,
):
    """Draw the grouped IOPS bar chart and save it to output_path."""
    fig, ax = common.create_figure()

    setup_styles = [common.get_style_from_setup(setup) for setup in setups]
    setup_colors = [style[0] for style in setup_styles]
    setup_hatches = [style[1] for style in setup_styles]

    x_positions_base = np.arange(len(workload_labels))
    bar_width = 0.8 / len(setup_names)

    for setup_idx, setup_name in enumerate(setup_names):
        x_positions = (
            x_positions_base + (setup_idx - len(setup_names) / 2 + 0.5) * bar_width
        )

        values = iops_matrix[:, setup_idx]

        bars = ax.bar(
            x_positions,
            values,
            bar_width,
            color=setup_colors[setup_idx],
            alpha=common.BAR_ALPHA,
            hatch=setup_hatches[setup_idx],
            edgecolor=common.BAR_EDGECOLOR,
            linewidth=common.BAR_LINEWIDTH,
        )

        for x_pos, value in zip(x_positions, values):
            if value > 0:
                ax.text(
                    x_pos,
                    value,
                    f"{value/1000:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=common.ANNOTATION_SIZE,
                    color="black",
                )

    ax.set_xticks(range(len(workload_labels)))
    ax.set_xticklabels(workload_labels, fontsize=common.TICKS_FONTSIZE)

    max_iops = np.max(iops_matrix)
    if max_iops > 0:
        ax.yaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda val, pos: f"{val/1000:g}")
        )
        ax.set_ylim(0, max_iops * 1.1)

    common.apply_axis_style(
        ax,
        title=f"IOPS ({common.HIGHER_BETTER_TITLE})",
        xlabel="Workload",
        ylabel="Throughput [K IOPS]",
        grid=True,
    )

    legend_handles = _build_split_legend_handles(setups, setup_colors, setup_hatches)

    ax.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc=common.LEGEND_ALIGN,
        bbox_to_anchor=(0.5, common.LEGEND_BBOX_HEIGHT),
        ncol=len(legend_handles),
        fontsize=common.LEGEND_FONTSIZE,
        edgecolor="black",
        borderaxespad=0,
        frameon=True,
        columnspacing=0.3,
        labelspacing=0.2,
        borderpad=0.5,
        handletextpad=0.5,
        handler_map={mpatches.Patch: DenseHatchHandler(density=2)},
    )

    plt.savefig(
        output_path, format=output_path.suffix[1:], dpi=300, bbox_inches="tight"
    )
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_bw_comparison(
    workload_labels: List[str],
    setup_names: List[str],
    bw_matrix: np.ndarray,
    setups: List[Dict[str, str]],
    output_path: Path,
):
    """Draw the grouped bandwidth bar chart and save it to output_path."""
    fig, ax = common.create_figure()

    setup_styles = [common.get_style_from_setup(setup) for setup in setups]
    setup_colors = [style[0] for style in setup_styles]
    setup_hatches = [style[1] for style in setup_styles]

    x_positions_base = np.arange(len(workload_labels))
    bar_width = 0.8 / len(setup_names)

    for setup_idx, setup_name in enumerate(setup_names):
        x_positions = (
            x_positions_base + (setup_idx - len(setup_names) / 2 + 0.5) * bar_width
        )

        values = bw_matrix[:, setup_idx]

        bars = ax.bar(
            x_positions,
            values,
            bar_width,
            color=setup_colors[setup_idx],
            alpha=common.BAR_ALPHA,
            hatch=setup_hatches[setup_idx],
            edgecolor=common.BAR_EDGECOLOR,
            linewidth=common.BAR_LINEWIDTH,
        )

        for x_pos, value in zip(x_positions, values):
            if value > 0:
                ax.text(
                    x_pos,
                    value,
                    f"{value/1_048_576:.2f}",  # bw_mean is KiB/s
                    ha="center",
                    va="bottom",
                    fontsize=common.ANNOTATION_SIZE,
                    color="black",
                )

    ax.set_xticks(range(len(workload_labels)))
    ax.set_xticklabels(workload_labels, fontsize=common.TICKS_FONTSIZE)

    max_bw = np.max(bw_matrix)
    if max_bw > 0:
        ax.yaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda val, pos: f"{val/1_048_576:g}")
        )
        ax.set_ylim(0, max_bw * 1.1)

    common.apply_axis_style(
        ax,
        title=f"Bandwidth ({common.HIGHER_BETTER_TITLE})",
        xlabel="Workload",
        ylabel="Bandwidth [GiB/s]",
        grid=True,
    )

    legend_handles = _build_split_legend_handles(setups, setup_colors, setup_hatches)

    ax.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc=common.LEGEND_ALIGN,
        bbox_to_anchor=(0.5, common.LEGEND_BBOX_HEIGHT),
        ncol=len(legend_handles),
        fontsize=common.LEGEND_FONTSIZE,
        edgecolor="black",
        borderaxespad=0,
        frameon=True,
        columnspacing=0.3,
        labelspacing=0.2,
        borderpad=0.5,
        handletextpad=0.5,
        handler_map={mpatches.Patch: DenseHatchHandler(density=2)},
    )

    plt.savefig(
        output_path, format=output_path.suffix[1:], dpi=300, bbox_inches="tight"
    )
    print(f"Saved plot to {output_path}")
    plt.close()


def generate_all_plots(
    setups: Optional[List[Dict[str, str]]] = None,
    output_dir: Optional[str] = None,
    output_name: str = "fio_comparison",
    formats: List[str] = ["pdf"],
    plot_types: List[str] = ["iops", "bw"],
):
    """Generate the FIO IOPS and bandwidth plots.

    Args:
        setups: Setup dicts with "name" and "path" keys, plus optional
                "color_label"/"hatch_label". None uses the host/VM/CVM SPDK runs.
        output_dir: Output directory, default bench-result/fio-plots-paper.
        output_name: Base filename, written as {output_name}_iops.{fmt} and
                     {output_name}_bw.{fmt}.
        formats: File formats to write, e.g. ["pdf"].
        plot_types: Any of "iops" and "bw".
    """
    common.apply_mpl_settings()

    if output_dir is None:
        output_dir = Path("bench-result/fio-plots-paper")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if setups is None:
        setups = [
            {"name": "Host (SPDK)", "path": "bench-result/fio/host-large/spdk"},
            {"name": "VM (SPDK)", "path": "bench-result/fio/amd-large/spdk"},
            {"name": "CVM (SPDK)", "path": "bench-result/fio/snp-large/spdk"},
        ]

    print(f"Generating FIO paper plots...")
    print(f"Output directory: {output_path}")
    print(f"Plot types: {', '.join(plot_types)}")
    print(f"Formats: {', '.join(formats)}")

    setup_names, fio_data = load_fio_data_from_setups(setups)

    if not fio_data:
        print("Error: No data loaded. Cannot generate plots.")
        sys.exit(1)

    print(f"\nLoaded data for {len(setup_names)} setups: {', '.join(setup_names)}")

    if "iops" in plot_types:
        print("\nGenerating IOPS plots...")
        workload_labels, setup_names_out, iops_matrix = prepare_iops_data(
            setup_names, fio_data
        )

        for fmt in formats:
            output_file = output_path / f"{output_name}_iops.{fmt}"
            plot_iops_comparison(
                workload_labels, setup_names_out, iops_matrix, setups, output_file
            )

        print("\nIOPS Results [K IOPS]:")
        col_w = max(len(n) for n in setup_names_out) + 2
        header = f"{'Workload':<12}" + "".join(f"{n:>{col_w}}" for n in setup_names_out)
        print(header)
        print("-" * len(header))
        for i, label in enumerate(workload_labels):
            row = f"{label:<12}" + "".join(
                f"{iops_matrix[i, j]/1000:>{col_w}.1f}"
                for j in range(len(setup_names_out))
            )
            print(row)

    if "bw" in plot_types:
        print("\nGenerating bandwidth plots...")
        workload_labels, setup_names_out, bw_matrix = prepare_bw_data(
            setup_names, fio_data
        )

        for fmt in formats:
            output_file = output_path / f"{output_name}_bw.{fmt}"
            plot_bw_comparison(
                workload_labels, setup_names_out, bw_matrix, setups, output_file
            )

        print("\nBandwidth Results [GiB/s]:")
        col_w = max(len(n) for n in setup_names_out) + 2
        header = f"{'Workload':<12}" + "".join(f"{n:>{col_w}}" for n in setup_names_out)
        print(header)
        print("-" * len(header))
        for i, label in enumerate(workload_labels):
            row = f"{label:<12}" + "".join(
                f"{bw_matrix[i, j]/1_048_576:>{col_w}.2f}"
                for j in range(len(setup_names_out))
            )
            print(row)

    print(f"\nAll plots saved to {output_path}/")
