#!/usr/bin/env python3
"""Timing metrics from AccelStore seal/unseal workload pair traces."""

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from ..crypto_backend import CryptoBackend
from . import discovery


@dataclass
class SealUnsealPairMetrics:
    """Metrics for a seal/unseal workload pair (seal→unseal sequence)."""

    config: str
    vm_type: str
    size: str
    client: str
    crypto: CryptoBackend
    data_size: str
    seal_job: str
    unseal_job: str

    pair_dir: Path
    seq0_trace: Path
    seq1_trace: Path

    seq0_total_time_us: float
    seq0_total_time_s: float

    seq1_init_time_us: float
    seq1_init_time_s: float
    seq1_total_time_us: float
    seq1_total_time_s: float


def extract_data_size(workload_name: str) -> str:
    """The data size in a workload name, e.g. "100mb", else "unknown"."""
    match = re.search(r"_(\d+(?:mb|gb))_", workload_name.lower())
    return match.group(1) if match else "unknown"


def parse_config_name(config: str) -> Tuple[str, str]:
    """(vm_type, size) from a config name: "snp-disk-medium" → ("snp", "medium")."""
    parts = config.split("-")
    if len(parts) < 2:
        return ("unknown", "unknown")

    vm_type = parts[0]

    if "disk" in parts:
        disk_idx = parts.index("disk")
        size = parts[disk_idx + 1] if len(parts) > disk_idx + 1 else "unknown"
    else:
        size = parts[1]

    return (vm_type, size)


def latest_trace_pair(pair_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """The latest timestamp's (seq0, seq1) traces, or (None, None).

    A run counts as present when either its dump or the ``_parsed.json`` cache
    beside it exists, but the returned path is always the dump's, because
    ``load_or_parse_trace`` derives the cache's name from it.
    """
    timestamps = set()
    for suffix in discovery.TRACE_SUFFIXES:
        for f in pair_dir.glob(f"*_seq*{suffix}"):
            match = re.match(r"(.+)_seq\d+", discovery.trace_stem(f.name))
            if match:
                timestamps.add(match.group(1))

    if not timestamps:
        return (None, None)

    latest = max(timestamps)  # lexicographic sort works for ISO timestamps

    def _present(seq: int) -> Optional[Path]:
        return discovery.trace_dump_if_present(pair_dir, f"{latest}_seq{seq}")

    seq0_trace, seq1_trace = _present(0), _present(1)
    if seq0_trace is None or seq1_trace is None:
        return (None, None)
    return (seq0_trace, seq1_trace)


def parse_seal_unseal_pair(pair_dir: Path) -> Optional[SealUnsealPairMetrics]:
    """Parse one ``{config}/{client}/{seal}#{unseal}`` pair, or None if unusable."""
    # imported here to break a circular import
    from modules.accelstore.tasks.results.trace_parser import load_or_parse_trace

    parts = pair_dir.parts

    if len(parts) < 3:
        print(f"Warning: Unexpected path structure for {pair_dir}")
        return None

    workload_pair = parts[-1]
    client = parts[-2]
    config_dir = parts[-3]

    if "#" not in workload_pair:
        print(f"Warning: Workload pair does not contain '#' separator: {workload_pair}")
        return None

    jobs = workload_pair.split("#")
    if len(jobs) != 2:
        print(f"Warning: Expected 2 jobs, got {len(jobs)}: {workload_pair}")
        return None

    seal_job, unseal_job = jobs

    vm_type, size = parse_config_name(config_dir)
    data_size = extract_data_size(seal_job)
    seq0_trace, seq1_trace = latest_trace_pair(pair_dir)

    if seq0_trace is None or seq1_trace is None:
        print(f"Warning: Missing trace files in {pair_dir}")
        return None

    try:
        metrics0 = load_or_parse_trace(seq0_trace)
        seq0_total_us = metrics0.total_execution_time_us
        seq0_total_s = seq0_total_us / 1e6

        metrics1 = load_or_parse_trace(seq1_trace)
        seq1_init_us = metrics1.init.total_time_us if metrics1.init else 0.0
        seq1_total_us = metrics1.total_execution_time_us
        seq1_init_s = seq1_init_us / 1e6
        seq1_total_s = seq1_total_us / 1e6

        _, backend = CryptoBackend.split_client_binary(client)

        return SealUnsealPairMetrics(
            config=config_dir,
            vm_type=vm_type,
            size=size,
            client=client,
            crypto=backend,
            data_size=data_size,
            seal_job=seal_job,
            unseal_job=unseal_job,
            pair_dir=pair_dir,
            seq0_trace=seq0_trace,
            seq1_trace=seq1_trace,
            seq0_total_time_us=seq0_total_us,
            seq0_total_time_s=seq0_total_s,
            seq1_init_time_us=seq1_init_us,
            seq1_init_time_s=seq1_init_s,
            seq1_total_time_us=seq1_total_us,
            seq1_total_time_s=seq1_total_s,
        )

    except (TypeError, AttributeError, ImportError, NameError):
        raise  # a programming error, not a bad dump
    except Exception as e:  # noqa: BLE001 - a bad dump is one skipped pair
        print(f"Error parsing traces in {pair_dir}: {e}")
        return None


def discover_seal_unseal_pairs(bench_dir: Path) -> List[Path]:
    """Every ``{config}/{client}/{seal}#{unseal}`` directory under ``bench_dir``."""
    pairs = []

    if not bench_dir.exists():
        return pairs

    for config_dir in bench_dir.iterdir():
        if not config_dir.is_dir():
            continue

        for client_dir in config_dir.iterdir():
            if not client_dir.is_dir():
                continue

            for pair_dir in client_dir.iterdir():
                if not pair_dir.is_dir():
                    continue

                if "#" in pair_dir.name:
                    pairs.append(pair_dir)

    return pairs


def analyze_seal_unseal_benchmarks(bench_dir: Path) -> List[SealUnsealPairMetrics]:
    """Every parseable pair under ``bench_dir``, sorted by config, client, size."""
    print(f"Discovering workload pairs in {bench_dir}...")
    pair_dirs = discover_seal_unseal_pairs(bench_dir)
    print(f"Found {len(pair_dirs)} workload pairs")

    results = []
    for pair_dir in pair_dirs:
        metrics = parse_seal_unseal_pair(pair_dir)
        if metrics:
            results.append(metrics)
            print(
                f"  Parsed: {metrics.config}/{metrics.client}"
                f" [{metrics.crypto.label}]/{metrics.data_size} "
                f"(seal={metrics.seq0_total_time_s:.2f}s, unseal={metrics.seq1_total_time_s:.2f}s, init={metrics.seq1_init_time_s:.2f}s)"
            )

    results.sort(key=lambda m: (m.config, m.client, m.data_size))

    return results


def export_to_csv(results: List[SealUnsealPairMetrics], output: Path) -> None:
    """Write the pairs to ``output`` as CSV, one row each."""
    if not results:
        print("No results to export")
        return

    with open(output, "w", newline="") as csvfile:
        fieldnames = [
            "config",
            "client",
            "crypto",
            "data_size",
            "seal_total_s",
            "unseal_total_s",
            "unseal_init_s",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for m in results:
            writer.writerow(
                {
                    "config": m.config,
                    "client": m.client,
                    "crypto": m.crypto.value,
                    "data_size": m.data_size,
                    "seal_total_s": f"{m.seq0_total_time_s:.2f}",
                    "unseal_total_s": f"{m.seq1_total_time_s:.2f}",
                    "unseal_init_s": f"{m.seq1_init_time_s:.2f}",
                }
            )
