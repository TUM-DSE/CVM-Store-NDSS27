#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Workload generation tasks.

Generates job files for AccelStore benchmarks (log, kvs, fs) into
modules/accelstore/jobs/ using the generators shipped in the accelstore nix
package. New files need a `git add` for the flake sharedData to pick them up.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Optional

from invoke import task

from core.tasks.config import PROJECT_ROOT

WORKLOAD_OUTPUT_DIR = PROJECT_ROOT / "modules" / "accelstore" / "jobs"

# (size_bytes, human_readable)
LOG_SIZE_CONFIGS = [
    (64 * 1024 * 1024, "64MB"),
    (128 * 1024 * 1024, "128MB"),
    (256 * 1024 * 1024, "256MB"),
    (512 * 1024 * 1024, "512MB"),
    (1024 * 1024 * 1024, "1GB"),
    (2 * 1024 * 1024 * 1024, "2GB"),
    (4 * 1024 * 1024 * 1024, "4GB"),
]


def get_accelstore_bin() -> Path:
    """Get the path to the accelstore bin directory from nix build."""
    result = subprocess.run(
        ["nix", "build", ".#accelstore", "--json", "--no-link"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to build accelstore: {result.stderr}")

    build_info = json.loads(result.stdout)
    store_path = Path(build_info[0]["outputs"]["out"])
    return store_path / "bin"


def run_workload_generator(generator_script: str, args: list[str]) -> None:
    """Run a workload generator script with arguments."""
    bin_path = get_accelstore_bin()
    generator_path = bin_path / generator_script

    if not generator_path.exists():
        raise RuntimeError(f"Generator not found: {generator_path}")

    cmd = ["python3", str(generator_path)] + args
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Generator failed: {result.stderr}")

    print(result.stdout)


@task
def generate_log_workloads(
    ctx: Any,
    total_data_gb: int = 256,
    output_dir: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Generate log workloads with append-then-sequential-read pattern.

    One workload per entry size (64MB to 4GB), each with as many entries as
    the requested total needs.

    Args:
        total_data_gb: Total data to write and read back, in GB (default: 256)
        output_dir: Output directory (default: modules/accelstore/jobs)
        dry_run: If True, only print what would be generated
    """
    total_bytes = total_data_gb * 1024 * 1024 * 1024
    out_dir = Path(output_dir) if output_dir else WORKLOAD_OUTPUT_DIR

    print("Generating log workloads:")
    print(
        f"  Total data per workload: {total_data_gb}GB write + {total_data_gb}GB read"
    )
    print(f"  Output directory: {out_dir}")
    print()

    for size_bytes, size_human in LOG_SIZE_CONFIGS:
        num_entries = total_bytes // size_bytes
        log_name = f"benchmark_log_{size_human}"
        output_file = out_dir / f"log_workload_{size_human}_write_read_seq.txt"

        print(
            f"  {size_human}: {num_entries} entries × {size_human} = {total_data_gb}GB"
        )
        print(f"    Output: {output_file.name}")

        if not dry_run:
            args = [
                "append-retrieve",
                "-n",
                str(num_entries),
                "-s",
                str(size_bytes),
                "-o",
                str(output_file),
                "--log-name",
                log_name,
                "--retrieve-pattern",
                "sequential",
            ]
            run_workload_generator("log_workload_generator.py", args)

    print()
    if dry_run:
        print("Dry run complete. No files were created.")
    else:
        print(f"Generated {len(LOG_SIZE_CONFIGS)} log workload files.")
        print("Remember to `git add` new files so the flake sharedData includes them.")


# The --quick ablation workloads: a quarter of the full runs' records.
QUICK_ABLATION_RECORDS = 256
QUICK_ABLATION_ENTRY_BYTES = 256 * 1024 * 1024


@task(name="generate-quick-jobs")
def generate_quick_jobs(
    ctx: Any,
    records: int = QUICK_ABLATION_RECORDS,
    output_dir: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Generate the *_quick.txt ablation workloads (`paper.ablation --quick`).

    Args:
        records: Objects per workload (default: 256, i.e. 64 GB per direction)
        output_dir: Output directory (default: modules/accelstore/jobs)
        dry_run: If True, only print what would be generated
    """
    out_dir = Path(output_dir) if output_dir else WORKLOAD_OUTPUT_DIR
    size = QUICK_ABLATION_ENTRY_BYTES

    print(f"Generating quick ablation jobs: {records} x 256MB per direction")
    print(f"  Output directory: {out_dir}")

    kvs = out_dir / "kvs_workload_A_256m_quick.txt"
    log = out_dir / "log_workload_256MB_write_read_seq_quick.txt"
    fs = out_dir / "fs_workload_write_read_256m_quick.txt"
    for target in (kvs, log, fs):
        print(f"    {target.name}")

    if dry_run:
        print("Dry run complete. No files were created.")
        return

    run_workload_generator(
        "kvs_workload_generator.py",
        [
            "A",
            "-o",
            str(kvs),
            "-n",
            str(records),
            "-r",
            str(records),
            "--data-size",
            str(size),
        ],
    )
    run_workload_generator(
        "log_workload_generator.py",
        [
            "append-retrieve",
            "-n",
            str(records),
            "-s",
            str(size),
            "-o",
            str(log),
            "--log-name",
            "benchmark_log_256MB",
            "--retrieve-pattern",
            "sequential",
        ],
    )
    # No generator emits this shape, so truncate the (phase-structured) full job.
    _truncate_fs_job(out_dir / "fs_workload_write_read_256m.txt", fs, records)

    print("Remember to `git add` new files so the flake sharedData includes them.")


def _truncate_fs_job(source: Path, target: Path, records: int) -> None:
    """Write `source` to `target`, keeping only its first `records` files."""
    kept = [
        line
        for line in source.read_text().splitlines()
        if not (
            (m := re.search(r"wr_file_256m_(\d{6})\.txt", line))
            and int(m.group(1)) >= records
        )
    ]
    text = "\n".join(kept) + "\n"
    text = text.replace("# Files: 1024", f"# Files: {records}")
    text = text.replace(
        "# Total data per direction: 274877906944 bytes (256g)",
        f"# Total data per direction: {records * QUICK_ABLATION_ENTRY_BYTES}"
        f" bytes ({records * 256 // 1024}g)",
    )
    target.write_text(text)
