"""SEV-SNP attestation benchmark: repeated attestation runs and their summary."""

import csv
from pathlib import Path
from typing import Any, Optional

from invoke import task

from core.tasks import vm as vm_tasks
from core.tasks.config import PROJECT_ROOT

from modules.attestation.tasks import veraison as veraison_tasks
from modules.attestation.tasks.utils.attestation import (
    average_times,
    latest_runs,
    load_runs,
    summary_rows,
)

DEFAULT_SIZE = "small"
DEFAULT_RUNS = 3
SUMMARY_CSV = "bench-result/attestation-summary.csv"


def result_dir(size: str) -> Path:
    return PROJECT_ROOT / "bench-result" / "attestation" / f"snp-disk-{size}"


@task
def run(
    ctx: Any,
    hostname: Optional[str] = None,
    size: str = DEFAULT_SIZE,
    iterations: int = DEFAULT_RUNS,
    config_size: str = "1M",
    processor_model: Optional[str] = None,
    setup: bool = False,
) -> None:
    """Boot an SNP guest `iterations` times and attest it (--setup clones and builds Veraison first)."""
    if setup:
        veraison_tasks.setup(ctx)
    veraison_tasks.require_deployment()
    for i in range(iterations):
        print(f"Attestation run {i + 1}/{iterations}")
        vm_tasks.start(
            ctx,
            type="snp",
            size=size,
            hostname=hostname,
            direct=False,
            action="run-attestation",
            attestation=True,
            action_config={
                "config_size": config_size,
                "processor_model": processor_model,
            },
        )


@task
def parse(
    ctx: Any,
    size: str = DEFAULT_SIZE,
    runs: int = DEFAULT_RUNS,
    output_file: str = SUMMARY_CSV,
) -> None:
    """Average the latest `runs` attestation runs into the summary CSV."""
    results = result_dir(size)
    timestamps = latest_runs(results, runs)
    if not timestamps:
        raise FileNotFoundError(f"No attestation runs under {results}")
    if len(timestamps) < runs:
        print(f"Warning: found {len(timestamps)} runs, expected {runs}")
    for ts in timestamps:
        print(f"  {ts}")

    rows = summary_rows(average_times(*load_runs(results, timestamps)))
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"CSV written to {output}")
    for row in rows:
        print(",".join(row))
