"""SPDK trace parser: AccelStore metrics from a `spdk_record -j` dump.

Extracts encryption, throughput, metadata, trusted-counter, blobstore and
execution-time metrics. START and END records are paired by (tpoint family,
object value) and their values read by argument index.

`encryption`, `metadata` and `merkle` are disjoint addends of the workload
window. Every other category is a breakdown of something already counted
elsewhere. The `*_whole_run` fields repeat a category over the whole capture,
because under the default `--sync-trigger on-shutdown` every epoch happens
after WORKLOAD_END.

Usage:
    parser = TraceParser(Path("trace.dump"))
    metrics = parser.parse_and_calculate_metrics()
    save_metrics_to_json(metrics, Path("metrics.json"))
"""

import json
import ijson
from collections import defaultdict
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
from enum import Enum, IntEnum
import warnings


class SpdkBdevIoType(IntEnum):
    """SPDK block device I/O type enumeration."""

    INVALID = 0
    READ = 1
    WRITE = 2
    UNMAP = 3
    FLUSH = 4
    RESET = 5
    NVME_ADMIN = 6
    NVME_IO = 7
    NVME_IO_MD = 8
    WRITE_ZEROES = 9
    ZCOPY = 10
    GET_ZONE_INFO = 11
    ZONE_MANAGEMENT = 12
    ZONE_APPEND = 13
    COMPARE = 14
    COMPARE_AND_WRITE = 15
    ABORT = 16
    SEEK_HOLE = 17
    SEEK_DATA = 18
    COPY = 19
    NVME_IOV_MD = 20
    NVME_NSSR = 21


# Bumped whenever a parser change alters the meaning of a cached field.
METRICS_SCHEMA_VERSION = 8


class StaleMetricsCache(Exception):
    """A cached *_parsed.json was written by an older parser schema."""


# META_BARRIER_END `outcome` (enum as_trace_sync_outcome).
BARRIER_OUTCOMES: Dict[int, str] = {
    0: "ordered",
    1: "race_won",
    2: "race_lost",
    3: "no_round_trip",
    4: "commit_failed",
    5: "metadata_only",
}

# TC_COMMIT_START `backend` (enum tc_backend_kind).
TC_BACKENDS: Dict[int, str] = {
    0: "local",
    1: "mock_remote",
}

# BLOB_OPEN_END `outcome` (enum as_trace_blob_open_outcome).
BLOB_OPEN_OUTCOMES: Dict[int, str] = {
    0: "hit",
    1: "hit_slow",
    2: "coalesced",
    3: "miss",
    4: "reopen",
    5: "failed",
}

# Blob I/O span `kind` (enum as_blob_io_kind), not all of the device traffic.
BLOB_IO_KINDS: Dict[int, str] = {
    0: "unspec",
    1: "superblock",
    2: "merkle",
}


@dataclass
class OperationMetrics:
    """Unified metrics for all operation types (API, BDEV, GPU, TOTAL, encryption, metadata, etc.)."""

    total_time_us: float = 0.0
    read_time_us: float = 0.0
    write_time_us: float = 0.0

    total_bytes: int = 0
    read_bytes: int = 0
    write_bytes: int = 0

    throughput_mibps: float = 0.0
    read_throughput_mibps: float = 0.0
    write_throughput_mibps: float = 0.0

    operation_count: int = 0
    read_count: int = 0
    write_count: int = 0

    # API calls that move no payload, so they carry no bytes.
    other_count: int = 0
    other_time_us: float = 0.0

    throughput_iops: float = 0.0

    avg_latency_us: float = 0.0
    read_avg_latency_us: float = 0.0
    write_avg_latency_us: float = 0.0

    operation_times_us: List[float] = field(default_factory=list)

    error_count: int = 0
    read_error_count: int = 0
    write_error_count: int = 0
    other_error_count: int = 0

    # Counter service that answered the epochs' commits (trusted_counter only).
    commits_local: int = 0
    commits_remote: int = 0

    client_type: str = ""  # "kvs", "fs", "log", etc.

    bytes_input: int = 0
    bytes_output: int = 0
    bytes_overhead: int = 0
    read_bytes_input: int = 0
    read_bytes_output: int = 0
    write_bytes_input: int = 0
    write_bytes_output: int = 0

    sub_operations: Dict[str, "OperationMetrics"] = field(default_factory=dict)


@dataclass
class OutputMetrics:
    """Performance metrics extracted from SPDK trace."""

    api: OperationMetrics = field(default_factory=OperationMetrics)
    bdev: OperationMetrics = field(default_factory=OperationMetrics)
    gpu: OperationMetrics = field(default_factory=OperationMetrics)
    total: OperationMetrics = field(default_factory=OperationMetrics)

    encryption: OperationMetrics = field(default_factory=OperationMetrics)
    metadata: OperationMetrics = field(default_factory=OperationMetrics)
    trusted_counter: OperationMetrics = field(default_factory=OperationMetrics)
    merkle: OperationMetrics = field(default_factory=OperationMetrics)
    # Epoch phases, nested inside the categories above and in the operations.
    epoch: OperationMetrics = field(default_factory=OperationMetrics)
    # The metadata path's own AEAD and MAC, nested inside `metadata`.
    metadata_crypto: OperationMetrics = field(default_factory=OperationMetrics)
    # The LOG counter cache build, nested inside the LOG open or retrieve.
    log_cache: OperationMetrics = field(default_factory=OperationMetrics)
    # `metadata` and `epoch` over the whole capture (see WHOLE_RUN_CATEGORIES).
    metadata_whole_run: OperationMetrics = field(default_factory=OperationMetrics)
    epoch_whole_run: OperationMetrics = field(default_factory=OperationMetrics)
    # Blobstore lifecycle spans: create, handle get, sync_md, delete. No bytes.
    blob_lifecycle: OperationMetrics = field(default_factory=OperationMetrics)
    # Device bytes bs_manager_read/write moved, by kind (see BLOB_IO_KINDS).
    blob_io: OperationMetrics = field(default_factory=OperationMetrics)
    # Both of the above over the whole capture.
    blob_lifecycle_whole_run: OperationMetrics = field(default_factory=OperationMetrics)
    blob_io_whole_run: OperationMetrics = field(default_factory=OperationMetrics)

    init: OperationMetrics = field(default_factory=OperationMetrics)
    cleanup: OperationMetrics = field(default_factory=OperationMetrics)

    # total_execution_time_us is the workload window, full_trace_time_us the capture.
    total_execution_time_us: float = 0.0
    full_trace_time_us: float = 0.0
    # Capture time outside the workload window, kept out of the runtime.
    setup_time_us: float = 0.0
    teardown_time_us: float = 0.0
    trace_start_tsc: int = 0
    trace_end_tsc: int = 0
    workload_start_tsc: int = 0
    workload_end_tsc: int = 0
    # "markers" | "total_spans" | "capture" - only "markers" excludes a load phase.
    workload_window_source: str = "none"

    # {outcome_name: {"count", "margin_us_total", "margin_us_max"}}.
    epoch_barrier: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # What the parse could not account for, keyed by kind and then by family.
    diagnostics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    trace_source: str = ""
    analysis_timestamp: str = ""
    schema_version: int = METRICS_SCHEMA_VERSION


def _mean(values: List[float]) -> float:
    """Arithmetic mean of a run-level quantity."""
    return sum(float(v) for v in values) / len(values)


def _mean_count(values: List[float]) -> int:
    """Mean of a count, rounded half up so a rare outcome is not averaged away."""
    return int(_mean(values) + 0.5)


def _average_stat_dicts(
    runs: List[Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """Mean of `{outcome: {stat: value}}` dicts, over the runs where a key is present."""
    out: Dict[str, Dict[str, float]] = {}
    for key in sorted({k for run in runs for k in run}):
        present = [run[key] for run in runs if key in run]
        stats: Dict[str, float] = {}
        for name in sorted({k for stat in present for k in stat}):
            values = [float(stat[name]) for stat in present if name in stat]
            if name.endswith("_max"):
                stats[name] = max(values)
            elif name == "count" or name.endswith("_count"):
                stats[name] = float(_mean_count(values))
            else:
                stats[name] = _mean(values)
        out[key] = stats
    return out


def _average_operation_metrics(runs: List[OperationMetrics]) -> OperationMetrics:
    """Mean of one category's metrics across the runs."""
    out = OperationMetrics()
    for f in fields(OperationMetrics):
        values = [getattr(run, f.name) for run in runs]
        if f.name == "sub_operations":
            names = {name for value in values for name in value}
            setattr(
                out,
                f.name,
                {
                    name: _average_operation_metrics(
                        [value[name] for value in values if name in value]
                    )
                    for name in sorted(names)
                },
            )
        elif f.name == "operation_times_us":
            # Pooled: the mean of a distribution is not a distribution.
            setattr(out, f.name, [x for value in values for x in value])
        elif f.type is int:
            setattr(out, f.name, _mean_count(values))
        elif f.type is float:
            setattr(out, f.name, _mean(values))
        else:
            setattr(out, f.name, values[0])
    return out


def average_metrics(runs: List[OutputMetrics]) -> OutputMetrics:
    """Fold a point's repeat runs into one mean `OutputMetrics`, newest first.

    Non-numeric and `*_tsc` fields come from the newest run, and `trace_source`
    names every dump that went into the mean. A single run is returned as it is.
    """
    if not runs:
        raise ValueError("average_metrics() needs at least one run")
    if len(runs) == 1:
        return runs[0]

    out = OutputMetrics()
    for f in fields(OutputMetrics):
        values = [getattr(run, f.name) for run in runs]
        if f.type is OperationMetrics:
            setattr(out, f.name, _average_operation_metrics(values))
        elif f.name in ("epoch_barrier", "diagnostics"):
            setattr(out, f.name, _average_stat_dicts(values))
        elif f.name.endswith("_tsc"):
            setattr(out, f.name, values[0])
        elif f.type is int:
            setattr(out, f.name, _mean_count(values))
        elif f.type is float:
            setattr(out, f.name, _mean(values))
        else:
            setattr(out, f.name, values[0])

    out.trace_source = ", ".join(str(run.trace_source) for run in runs)
    return out


def _dict_to_operation_metrics(data: Dict) -> OperationMetrics:
    """Convert a dict to an OperationMetrics, recursing into sub_operations."""
    if not isinstance(data, dict):
        return data

    sub_ops = data.get("sub_operations", {})
    if sub_ops and isinstance(sub_ops, dict):
        data = data.copy()
        data["sub_operations"] = {
            key: _dict_to_operation_metrics(value) for key, value in sub_ops.items()
        }

    return OperationMetrics(**data)


def _dict_to_output_metrics(data: Dict) -> OutputMetrics:
    """Convert a dict to an OutputMetrics, rebuilding its OperationMetrics fields."""
    if not isinstance(data, dict):
        return data

    operation_fields = [
        "api",
        "bdev",
        "gpu",
        "total",
        "encryption",
        "metadata",
        "trusted_counter",
        "merkle",
        "epoch",
        "metadata_crypto",
        "log_cache",
        "metadata_whole_run",
        "epoch_whole_run",
        "blob_lifecycle",
        "blob_io",
        "blob_lifecycle_whole_run",
        "blob_io_whole_run",
        "init",
        "cleanup",
    ]

    data = data.copy()
    for field in operation_fields:
        if field in data and isinstance(data[field], dict):
            data[field] = _dict_to_operation_metrics(data[field])

    return OutputMetrics(**data)


def load_metrics_from_json(filepath: Path) -> OutputMetrics:
    """Load OutputMetrics from JSON, raising StaleMetricsCache for an older schema."""
    with open(filepath, "r") as f:
        data = json.load(f)
    version = data.get("schema_version", 1)
    if version != METRICS_SCHEMA_VERSION:
        raise StaleMetricsCache(
            f"{filepath.name} was written by schema v{version}, "
            f"this parser produces v{METRICS_SCHEMA_VERSION}"
        )
    return _dict_to_output_metrics(data)


def save_metrics_to_json(metrics: OutputMetrics, filepath: Path) -> None:
    """Save OutputMetrics to JSON file."""
    with open(filepath, "w") as f:
        json.dump(asdict(metrics), f, indent=2)


def load_or_parse_trace(trace_path: Path) -> OutputMetrics:
    """Return metrics for `trace_path`, parsing it if the cache beside it is unusable.

    The cache is `<stem>_parsed.json` in the same directory and answers without
    the dump being present. Raises FileNotFoundError if neither exists, and
    RuntimeError if the dump is there and cannot be parsed.
    """
    trace_path = Path(trace_path)
    parsed_path = trace_path.parent / f"{trace_path.stem}_parsed.json"

    if parsed_path.exists():
        try:
            metrics = load_metrics_from_json(parsed_path)
            print(f"  ✓ Loaded cached: {parsed_path.name}")
            return metrics
        except StaleMetricsCache as e:
            print(f"  ↻ Stale cache ({e}), reparsing: {parsed_path.name}")
        except Exception as e:
            warnings.warn(
                f"Failed to load cached metrics from {parsed_path}: {e}, reparsing..."
            )

    if not trace_path.exists():
        raise FileNotFoundError(
            f"Trace file not found: {trace_path} "
            f"(and no cached parsed file {parsed_path.name})"
        )

    print(f"  ✓ Parsing: {trace_path.name}")
    try:
        parser = TraceParser(str(trace_path))
        metrics = parser.parse_and_calculate_metrics()
    except Exception as e:
        raise RuntimeError(f"Failed to parse {trace_path}: {e}") from e

    try:
        save_metrics_to_json(metrics, parsed_path)
        print(f"  ✓ Cached to: {parsed_path.name}")
    except Exception as e:
        warnings.warn(f"Failed to cache metrics to {parsed_path}: {e}")

    return metrics


def dump_metrics(metrics: OutputMetrics) -> None:
    """Print performance metrics in a human-readable format."""
    print("\n" + "=" * 70)
    print("SPDK TRACE METRICS SUMMARY")
    print("=" * 70)

    print(f"\nTrace File: {metrics.trace_source}")
    print(f"Analysis Time: {metrics.analysis_timestamp}")

    print("\n" + "-" * 70)
    print("Execution & Throughput")
    print("-" * 70)
    duration_sec = metrics.total_execution_time_us / 1e6
    print(
        f"  Workload Time:           {metrics.total_execution_time_us:>15,.0f} µs ({duration_sec:>8.2f}s)"
    )
    print(
        f"  Setup Time (pre-workload):{metrics.setup_time_us:>14,.0f} µs "
        f"({metrics.setup_time_us / 1e6:>8.2f}s)"
    )
    print(
        f"  Teardown Time:           {metrics.teardown_time_us:>15,.0f} µs "
        f"({metrics.teardown_time_us / 1e6:>8.2f}s)"
    )
    print(
        f"  Trace Capture Window:    {metrics.full_trace_time_us:>15,.0f} µs "
        f"({metrics.full_trace_time_us / 1e6:>8.2f}s)"
    )
    print(f"  API Throughput (IOPS):   {metrics.api.throughput_iops:>15,.1f} IOPS")
    print(f"  API Throughput (MiB/s):  {metrics.api.throughput_mibps:>15,.2f} MiB/s")
    print(
        f"  API Read Throughput (MiB/s):  {metrics.api.read_throughput_mibps:>15,.2f} MiB/s"
    )
    print(
        f"  API Write Throughput (MiB/s): {metrics.api.write_throughput_mibps:>15,.2f} MiB/s"
    )
    print(f"  API Call Count:          {metrics.api.operation_count:>15,}")
    if metrics.api.other_count:
        print(f"    - of which zero-byte:  {metrics.api.other_count:>15,}")

    print("\n" + "-" * 70)
    print("API Operation Latency")
    print("-" * 70)
    print(f"  Total API Time:          {metrics.api.total_time_us:>15,.0f} µs")
    print(f"  Read API Time:           {metrics.api.read_time_us:>15,.0f} µs")
    print(f"  Write API Time:          {metrics.api.write_time_us:>15,.0f} µs")
    print(f"  Average Latency:         {metrics.api.avg_latency_us:>15,.2f} µs/op")
    print(f"    - Read Avg Latency:    {metrics.api.read_avg_latency_us:>15,.2f} µs/op")
    print(
        f"    - Write Avg Latency:   {metrics.api.write_avg_latency_us:>15,.2f} µs/op"
    )

    print("\n" + "-" * 70)
    print("Operation Errors")
    print("-" * 70)
    print(f"  API Errors:              {metrics.api.error_count:>15,}")
    print(f"    - Read Errors:         {metrics.api.read_error_count:>15,}")
    print(f"    - Write Errors:        {metrics.api.write_error_count:>15,}")

    print("\n" + "-" * 70)
    client_type_labels = {
        "kvs": ("KVS", "PUT+GET"),
        "fs": ("Filesystem", "WRITE+READ"),
        "log": ("Log", "APPEND+RETRIEVE"),
    }
    client_label, api_label = client_type_labels.get(
        metrics.api.client_type, ("Storage", "API")
    )
    print(f"{client_label} Read/Write Amplification")
    print("-" * 70)
    print(f"  Base Bytes ({api_label}): {metrics.api.total_bytes:>15,}")
    print(f"    - Base Read Bytes:     {metrics.api.read_bytes:>15,}")
    print(f"    - Base Write Bytes:    {metrics.api.write_bytes:>15,}")
    print(f"  Total Bytes (BDEV I/O):  {metrics.bdev.total_bytes:>15,}")
    print(f"    - BDEV Read Bytes:     {metrics.bdev.read_bytes:>15,}")
    print(f"    - BDEV Write Bytes:    {metrics.bdev.write_bytes:>15,}")

    if metrics.api.total_bytes > 0:
        ratio = metrics.bdev.total_bytes / metrics.api.total_bytes
        print(f"  Amplification Ratio:     {ratio:>15.3f}x")

        if metrics.api.read_bytes > 0:
            read_ratio = metrics.bdev.read_bytes / metrics.api.read_bytes
            print(f"    - Read Amplification:  {read_ratio:>15.3f}x")
        if metrics.api.write_bytes > 0:
            write_ratio = metrics.bdev.write_bytes / metrics.api.write_bytes
            print(f"    - Write Amplification: {write_ratio:>15.3f}x")

        _dump_amplification_causes(metrics)

    print("\n" + "-" * 70)
    print("GPU Operations")
    print("-" * 70)
    print(f"  Total Time:              {metrics.gpu.total_time_us:>15,.0f} µs")
    print(f"  Read Time:               {metrics.gpu.read_time_us:>15,.0f} µs")
    print(f"  Write Time:              {metrics.gpu.write_time_us:>15,.0f} µs")
    print(f"  Total Bytes:             {metrics.gpu.total_bytes:>15,}")
    print(f"    - Read Bytes:          {metrics.gpu.read_bytes:>15,}")
    print(f"    - Write Bytes:         {metrics.gpu.write_bytes:>15,}")
    print(f"  Operation Count:         {metrics.gpu.operation_count:>15,}")
    print(f"  Throughput:              {metrics.gpu.throughput_mibps:>15,.2f} MiB/s")

    print("\n" + "-" * 70)
    print("Total Pipeline Operations")
    print("-" * 70)
    print(f"  Total Time:              {metrics.total.total_time_us:>15,.0f} µs")
    print(f"  Read Time:               {metrics.total.read_time_us:>15,.0f} µs")
    print(f"  Write Time:              {metrics.total.write_time_us:>15,.0f} µs")
    print(f"  Total Bytes:             {metrics.total.total_bytes:>15,}")
    print(f"    - Read Bytes:          {metrics.total.read_bytes:>15,}")
    print(f"    - Write Bytes:         {metrics.total.write_bytes:>15,}")
    print(f"  Operation Count:         {metrics.total.operation_count:>15,}")
    print(f"  Throughput:              {metrics.total.throughput_mibps:>15,.2f} MiB/s")

    print("\n" + "-" * 70)
    print("Encryption Operations")
    print("-" * 70)
    print(f"  Total Time:              {metrics.encryption.total_time_us:>15,.0f} µs")
    print(f"  Total Operations:        {metrics.encryption.operation_count:>15,}")

    enc = metrics.encryption.sub_operations.get("encrypt", OperationMetrics())
    print(f"\n  Encrypt Time:            {enc.total_time_us:>15,.0f} µs")
    print(f"  Encrypt Count:           {enc.operation_count:>15,}")
    print(f"  Encrypt Input Bytes:     {enc.bytes_input:>15,}")
    print(f"  Encrypt Output Bytes:    {enc.bytes_output:>15,}")
    print(f"  Encrypt Overhead:        {enc.bytes_overhead:>15,}")

    dec = metrics.encryption.sub_operations.get("decrypt", OperationMetrics())
    print(f"\n  Decrypt Time:            {dec.total_time_us:>15,.0f} µs")
    print(f"  Decrypt Count:           {dec.operation_count:>15,}")
    print(f"  Decrypt Input Bytes:     {dec.bytes_input:>15,}")
    print(f"  Decrypt Output Bytes:    {dec.bytes_output:>15,}")
    print(f"  Decrypt Overhead:        {dec.bytes_overhead:>15,}")

    print("\n" + "-" * 70)
    print("Metadata Operations")
    print("-" * 70)
    print(f"  Total Time:              {metrics.metadata.total_time_us:>15,.0f} µs")
    print(f"  Total Operations:        {metrics.metadata.operation_count:>15,}")
    print(f"  Total Bytes:             {metrics.metadata.total_bytes:>15,}")

    for op_name in METADATA_SUBCATEGORIES:
        sub = metrics.metadata.sub_operations.get(op_name, OperationMetrics())
        nested = " (nested in Sync)" if op_name in METADATA_NESTED else ""
        print(
            f"\n  {op_name.capitalize()} Count:           "
            f"{sub.operation_count:>15,}{nested}"
        )
        print(
            f"  {op_name.capitalize()} Time:            {sub.total_time_us:>15,.0f} µs"
        )
        if sub.total_bytes > 0:
            print(f"  {op_name.capitalize()} Bytes:           {sub.total_bytes:>15,}")
        if sub.bytes_input > 0:
            # META_SYNC only: the occupied prefix its snapshots covered.
            print(f"  {op_name.capitalize()} Prefix Bytes:    {sub.bytes_input:>15,}")

    _dump_whole_run_metadata(metrics)

    print("\n" + "-" * 70)
    print("Metadata Epoch (a breakdown of what an epoch costs, not an addend)")
    print("-" * 70)
    # Phases only - the two enclosing spans are excluded from the total.
    print(f"  Phase Time:              {metrics.epoch.total_time_us:>15,.0f} µs")
    print(f"  Phase Operations:        {metrics.epoch.operation_count:>15,}")
    for op_name in EPOCH_SUBCATEGORIES:
        sub = metrics.epoch.sub_operations.get(op_name, OperationMetrics())
        label = op_name.capitalize().ljust(9)
        enclosing = (
            " (enclosing, excluded from the phase total)"
            if op_name in EPOCH_ENCLOSING
            else ""
        )
        print(f"\n  {label} Count:         {sub.operation_count:>15,}{enclosing}")
        print(f"  {label} Time:          {sub.total_time_us:>15,.0f} µs")
        if sub.total_bytes > 0:
            print(f"  {label} Bytes:         {sub.total_bytes:>15,}")

    whole_epoch = metrics.epoch_whole_run
    scoped_epochs = metrics.epoch.sub_operations.get("epoch", OperationMetrics())
    whole_epochs = whole_epoch.sub_operations.get("epoch", OperationMetrics())
    if (
        whole_epoch.operation_count != metrics.epoch.operation_count
        or whole_epochs.operation_count != scoped_epochs.operation_count
    ):
        print(
            f"\n  Whole run: epoch count {whole_epochs.operation_count:,}, "
            f"phase operations {whole_epoch.operation_count:,}, "
            f"{whole_epoch.total_time_us:,.0f} µs of phases - the epochs above "
            "plus every one that ran outside the timed window"
        )

    if metrics.epoch_barrier:
        print("\n  Barrier verdicts (outcome: count, summed margin, worst margin):")
        for name, stats in sorted(metrics.epoch_barrier.items()):
            print(
                f"    {name:<16} {int(stats.get('count', 0)):>10,}"
                f"  {stats.get('margin_us_total', 0.0):>12,.0f} µs"
                f"  {stats.get('margin_us_max', 0.0):>12,.0f} µs"
            )

    print("\n" + "-" * 70)
    print("Metadata Crypto (nested in Metadata)")
    print("-" * 70)
    print(
        f"  Total Time:              {metrics.metadata_crypto.total_time_us:>15,.0f} µs"
    )
    print(f"  Total Operations:        {metrics.metadata_crypto.operation_count:>15,}")
    for op_name in METADATA_CRYPTO_SUBCATEGORIES:
        sub = metrics.metadata_crypto.sub_operations.get(op_name, OperationMetrics())
        if not sub.operation_count:
            continue
        # The chunk families report a per-chunk length, the MAC families none.
        print(
            f"    {op_name:<12} {sub.operation_count:>12,} ops "
            f"{sub.total_time_us:>12,.0f} µs {sub.bytes_input:>15,} B"
        )

    print("\n" + "-" * 70)
    print("Trusted Counter Operations")
    print("-" * 70)
    print(
        f"  Total Time:              {metrics.trusted_counter.total_time_us:>15,.0f} µs"
    )
    print(f"  Total Operations:        {metrics.trusted_counter.operation_count:>15,}")

    nxt = metrics.trusted_counter.sub_operations.get("next", OperationMetrics())
    print(f"\n  Next Count:              {nxt.operation_count:>15,}")
    print(f"  Next Time:               {nxt.total_time_us:>15,.0f} µs")

    inc = metrics.trusted_counter.sub_operations.get("inc", OperationMetrics())
    if inc.operation_count:
        # Only a legacy dump (TC_INC era) has these.
        print(f"\n  Inc Count (legacy):      {inc.operation_count:>15,}")
        print(f"  Inc Time:                {inc.total_time_us:>15,.0f} µs")

    read = metrics.trusted_counter.sub_operations.get("read", OperationMetrics())
    print(f"\n  Read Count:              {read.operation_count:>15,}")
    print(f"  Read Time:               {read.total_time_us:>15,.0f} µs")

    tc = metrics.trusted_counter
    if tc.commits_local or tc.commits_remote:
        print(f"\n  Commits (local service): {tc.commits_local:>15,}")
        print(f"  Commits (remote service):{tc.commits_remote:>15,}")

    print("\n" + "-" * 70)
    print("Merkle Tree Operations")
    print("-" * 70)
    print(f"  Total Time:              {metrics.merkle.total_time_us:>15,.0f} µs")
    print(f"  Total Operations:        {metrics.merkle.operation_count:>15,}")

    add = metrics.merkle.sub_operations.get("add", OperationMetrics())
    print(f"\n  Add Count:               {add.operation_count:>15,}")
    print(f"  Add Time:                {add.total_time_us:>15,.0f} µs")
    if add.operation_count > 0:
        print(
            f"  Add Avg Latency:         {add.total_time_us / add.operation_count:>15,.2f} µs/op"
        )

    ver = metrics.merkle.sub_operations.get("verify", OperationMetrics())
    print(f"\n  Verify Count:            {ver.operation_count:>15,}")
    print(f"  Verify Time:             {ver.total_time_us:>15,.0f} µs")
    if ver.operation_count > 0:
        print(
            f"  Verify Avg Latency:      {ver.total_time_us / ver.operation_count:>15,.2f} µs/op"
        )

    print("\n" + "-" * 70)
    print("Initialization Operations")
    print("-" * 70)
    print(f"  Total Time:              {metrics.init.total_time_us:>15,.0f} µs")
    print(f"  Operation Count:         {metrics.init.operation_count:>15,}")
    if metrics.init.operation_count > 0:
        print(f"  Average Time:            {metrics.init.avg_latency_us:>15,.2f} µs/op")

    if metrics.log_cache.operation_count:
        print("\n" + "-" * 70)
        print("Log Counter Cache Build (nested in the LOG open/retrieve)")
        print("-" * 70)
        print(
            f"  Total Time:              {metrics.log_cache.total_time_us:>15,.0f} µs"
        )
        print(f"  Operation Count:         {metrics.log_cache.operation_count:>15,}")

    _dump_blobstore(metrics)

    print("\n" + "-" * 70)
    print("Cleanup Operations")
    print("-" * 70)
    print(f"  Total Time:              {metrics.cleanup.total_time_us:>15,.0f} µs")
    print(f"  Total Elements:          {metrics.cleanup.total_bytes:>15,}")
    print(f"  Operation Count:         {metrics.cleanup.operation_count:>15,}")
    if metrics.cleanup.operation_count > 0:
        print(
            f"  Average Time:            {metrics.cleanup.avg_latency_us:>15,.2f} µs/op"
        )

    _dump_diagnostics(metrics)

    print("\n" + "=" * 70)


def _dump_amplification_causes(metrics: OutputMetrics) -> None:
    """The measured share of the bdev-minus-api gap, over the whole run."""
    blob_io = metrics.blob_io_whole_run
    life = metrics.blob_lifecycle_whole_run
    if not blob_io.operation_count and not life.operation_count:
        # No blobstore tracegroups in this dump.
        return

    print("\n  Measured causes of the gap (whole run):")
    for kind, sub_metrics in sorted(blob_io.sub_operations.items()):
        if not sub_metrics.operation_count:
            continue
        print(
            f"    blob I/O {kind:<12} R={sub_metrics.read_bytes:>14,} B"
            f"  W={sub_metrics.write_bytes:>14,} B"
            f"  ({sub_metrics.operation_count:,} I/Os)"
        )

    if life.operation_count:
        # The events carry no bytes - what one costs the device is an md page.
        counts = {
            name: life.sub_operations.get(name, OperationMetrics()).operation_count
            for name in BLOB_LIFECYCLE_SUBCATEGORIES
        }
        print(
            f"    blob lifecycle       create={counts['create']:,}"
            f"  delete={counts['delete']:,}  sync_md={counts['sync_md']:,}"
            f"  open={counts['open']:,}"
        )
        print(
            f"      of which opens that read an md page: miss={counts['open_miss']:,}"
            f"  reopen={counts['open_reopen']:,}"
            f"  (hit={counts['open_hit']:,}"
            f" hit_slow={counts['open_hit_slow']:,}"
            f" coalesced={counts['open_coalesced']:,}"
            f" failed={counts['open_failed']:,})"
        )


def _dump_blobstore(metrics: OutputMetrics) -> None:
    """The blobstore's own spans: what it did, and what it moved doing it."""
    life, life_all = metrics.blob_lifecycle, metrics.blob_lifecycle_whole_run
    blob_io, blob_io_all = metrics.blob_io, metrics.blob_io_whole_run
    if not life_all.operation_count and not blob_io_all.operation_count:
        return

    print("\n" + "-" * 70)
    print("Blobstore Operations")
    print("-" * 70)
    print(f"  Total Time:              {life_all.total_time_us:>15,.0f} µs")
    print(
        f"  Lifecycle Events:        {life_all.operation_count:>15,}"
        f"   ({life.operation_count:,} in window)"
    )
    for name in BLOB_LIFECYCLE_SUBCATEGORIES:
        whole = life_all.sub_operations.get(name, OperationMetrics())
        if not whole.operation_count:
            continue
        scoped = life.sub_operations.get(name, OperationMetrics())
        label = f"    {name:<16}"
        print(
            f"{label} {whole.operation_count:>12,} events "
            f"{whole.total_time_us:>12,.0f} µs "
            f"({scoped.operation_count:,} in window)"
        )

    if blob_io_all.operation_count:
        print(
            f"\n  Blob I/O Bytes:          R={blob_io_all.read_bytes:>14,}"
            f"  W={blob_io_all.write_bytes:>14,}"
        )
        print(
            f"    in window:             R={blob_io.read_bytes:>14,}"
            f"  W={blob_io.write_bytes:>14,}"
        )
        for kind in sorted(blob_io_all.sub_operations):
            whole = blob_io_all.sub_operations[kind]
            if not whole.operation_count:
                continue
            print(
                f"    {kind:<16} {whole.operation_count:>12,} I/Os "
                f"R={whole.read_bytes:>14,} B  W={whole.write_bytes:>14,} B"
            )


def _dump_whole_run_metadata(metrics: OutputMetrics) -> None:
    """The metadata category over the whole capture, beside the scoped one."""
    whole, scoped = metrics.metadata_whole_run, metrics.metadata
    print("\n  Whole run (the workload window and everything outside it):")
    print(f"  Total Time:              {whole.total_time_us:>15,.0f} µs")
    print(f"  Total Operations:        {whole.operation_count:>15,}")
    print(f"  Total Bytes:             {whole.total_bytes:>15,}")
    if whole.bytes_input > 0:
        print(f"  Prefix Bytes:            {whole.bytes_input:>15,}")

    ops = whole.operation_count - scoped.operation_count
    byts = whole.total_bytes - scoped.total_bytes
    time_us = whole.total_time_us - scoped.total_time_us
    if ops or byts or abs(time_us) > 0.5:
        print(
            f"    - outside the timed window: {ops:,} operations, "
            f"{byts:,} bytes, {time_us:,.0f} µs"
        )
    else:
        print("    - identical: every metadata operation fell inside the window")


def _dump_diagnostics(metrics: OutputMetrics) -> None:
    """One line per diagnostic that says something went unaccounted for."""
    diag = metrics.diagnostics or {}

    layout = diag.get("chunk_layout")
    if layout:
        parts = ", ".join(f"{k}={int(v):,}" for k, v in sorted(layout.items()))
        print(f"\n  Chunk crypto span layout:  {parts}")

    owners = diag.get("owners")
    if owners:
        # Raw owner strings - the dump carries no table for the thread names.
        parts = ", ".join(
            f"{k}={int(v):,}" for k, v in sorted(owners.items(), key=lambda kv: -kv[1])
        )
        print(f"  Spans started per owner:   {parts}")

    # Expected at the tail of a capture, which is stopped at an arbitrary point.
    unfinished = diag.get("unfinished_start")
    if unfinished:
        print(f"  Spans still open at end of trace: {int(sum(unfinished.values())):,}")

    alarms = {
        key: sub
        for key, sub in diag.items()
        if key
        in (
            "unmatched_start",
            "unmatched_end",
            "arg_out_of_range",
            "unhandled_accelstore_tpoint",
            "missing_object_key",
            "chunk_size_field_missing",
            "unknown_tc_backend",
        )
        and sub
    }
    if not alarms:
        return

    summary = "; ".join(
        f"{key}: {int(sum(sub.values())):,} "
        f"({', '.join(f'{k}={int(v):,}' for k, v in sorted(sub.items())[:4])}"
        f"{', …' if len(sub) > 4 else ''})"
        for key, sub in sorted(alarms.items())
    )
    print(f"\n  WARNING - unaccounted trace records: {summary}")


class ExtractMode(Enum):
    """Argument layout of a pair: fixed indices, or the per-chunk crypto record."""

    ARGS = "args"
    CHUNK = "chunk"


@dataclass
class OpExtractConfig:
    """Where each value sits in a START/END pair: ('start'|'end', arg_index).

    `outcome` and `margin_us` are META_BARRIER's, `backend` is TC_COMMIT's,
    `kind` is a blob I/O span's (BLOB_IO_KINDS) and `open_outcome` BLOB_OPEN's
    (BLOB_OPEN_OUTCOMES). `size_optional` marks a size older dumps do not carry.
    """

    size: Optional[Tuple[str, int]] = None
    bytes_in: Optional[Tuple[str, int]] = None
    bytes_out: Optional[Tuple[str, int]] = None
    io_type: Optional[Tuple[str, int]] = None
    op_type: Optional[Tuple[str, int]] = None
    status: Optional[Tuple[str, int]] = None
    op_id: Optional[Tuple[str, int]] = None
    outcome: Optional[Tuple[str, int]] = None
    margin_us: Optional[Tuple[str, int]] = None
    backend: Optional[Tuple[str, int]] = None
    kind: Optional[Tuple[str, int]] = None
    open_outcome: Optional[Tuple[str, int]] = None
    size_optional: bool = False


@dataclass
class OperationPair:
    """How to extract one START/END family's metrics.

    `rw` is 'read', 'write', 'other' (an API call that moves no payload) or
    None. `name` is the OPERATION_PAIRS key, filled in below, and half of a
    span's pairing key.
    """

    category: str
    subcategory: str
    rw: Optional[str]
    extract: OpExtractConfig
    mode: ExtractMode = ExtractMode.ARGS
    name: str = ""


def _api_other(subcategory: str, op_id_index: int) -> OperationPair:
    """An API call that moves no payload: open, create, close, delete, exists."""
    return OperationPair(
        category="api",
        subcategory=subcategory,
        rw="other",
        extract=OpExtractConfig(status=("end", 1), op_id=("start", op_id_index)),
    )


OPERATION_PAIRS: Dict[str, OperationPair] = {
    "KVS_PUT": OperationPair(
        category="api",
        subcategory="kvs",
        rw="write",
        extract=OpExtractConfig(
            size=("start", 1), status=("end", 1), op_id=("start", 2)
        ),
    ),
    "KVS_GET": OperationPair(
        category="api",
        subcategory="kvs",
        rw="read",
        extract=OpExtractConfig(size=("end", 2), status=("end", 1), op_id=("start", 2)),
    ),
    "KVS_EXISTS": _api_other("kvs", 2),
    "KVS_DELETE": _api_other("kvs", 2),
    "FS_WRITE": OperationPair(
        category="api",
        subcategory="fs",
        rw="write",
        extract=OpExtractConfig(
            size=("start", 1), status=("end", 1), op_id=("start", 2)
        ),
    ),
    "FS_READ": OperationPair(
        category="api",
        subcategory="fs",
        rw="read",
        extract=OpExtractConfig(size=("end", 2), status=("end", 1), op_id=("start", 2)),
    ),
    # Create and open carry a name length, so their op_id sits one index later.
    "FS_CREATE": _api_other("fs", 2),
    "FS_OPEN": _api_other("fs", 2),
    "FS_DELETE": _api_other("fs", 1),
    "FS_CLOSE": _api_other("fs", 1),
    "LOG_APPEND": OperationPair(
        category="api",
        subcategory="log",
        rw="write",
        extract=OpExtractConfig(
            size=("start", 1), status=("end", 1), op_id=("start", 2)
        ),
    ),
    # The START's number is the entry position, not a byte count.
    "LOG_RETRIEVE": OperationPair(
        category="api",
        subcategory="log",
        rw="read",
        extract=OpExtractConfig(size=("end", 2), status=("end", 1), op_id=("start", 2)),
    ),
    "LOG_TRUNCATE": _api_other("log", 2),
    "LOG_CREATE": _api_other("log", 1),
    "LOG_OPEN": _api_other("log", 1),
    "LOG_DELETE": _api_other("log", 1),
    "LOG_CLOSE": _api_other("log", 1),
    # Nested inside the LOG open or retrieve that triggered it, so not in `api`.
    "LOG_CACHE_BUILD": OperationPair(
        category="log_cache",
        subcategory="build",
        rw=None,
        extract=OpExtractConfig(status=("end", 1)),
    ),
    "BDEV_IO": OperationPair(
        category="bdev",
        subcategory="io",
        rw=None,
        extract=OpExtractConfig(io_type=("start", 0)),
    ),
    # op_type: 0=read, 1=write, 2=other.
    "GPU": OperationPair(
        category="gpu",
        subcategory="compute",
        rw=None,
        extract=OpExtractConfig(size=("start", 1), op_type=("start", 2)),
    ),
    "TOTAL": OperationPair(
        category="total",
        subcategory="pipeline",
        rw=None,
        extract=OpExtractConfig(size=("start", 1), op_type=("start", 2)),
    ),
    # The data path only - the metadata path has tpoints of its own below.
    "CRYPTO_ENCRYPT": OperationPair(
        category="encryption",
        subcategory="encrypt",
        rw="write",
        extract=OpExtractConfig(),
        mode=ExtractMode.CHUNK,
    ),
    "CRYPTO_DECRYPT": OperationPair(
        category="encryption",
        subcategory="decrypt",
        rw="read",
        extract=OpExtractConfig(),
        mode=ExtractMode.CHUNK,
    ),
    # Time and count only: these spans never carried a byte count.
    "CRYPTO_MAC_CREATE": OperationPair(
        category="encryption",
        subcategory="mac_create",
        rw="write",
        extract=OpExtractConfig(status=("end", 1)),
    ),
    "CRYPTO_MAC_VERIFY": OperationPair(
        category="encryption",
        subcategory="mac_verify",
        rw="read",
        extract=OpExtractConfig(status=("end", 1)),
    ),
    # META_SYNC: END args[2] is the bytes issued, START args[1] the occupied prefix.
    "META_SYNC": OperationPair(
        category="metadata",
        subcategory="sync",
        rw=None,
        extract=OpExtractConfig(
            size=("end", 2), bytes_in=("start", 1), status=("end", 1)
        ),
    ),
    "META_LOAD": OperationPair(
        category="metadata",
        subcategory="load",
        rw=None,
        extract=OpExtractConfig(status=("end", 1)),
    ),
    # `sz` was appended after the epoch generation, so it is at index 2 and optional.
    "META_WRITE": OperationPair(
        category="metadata",
        subcategory="write",
        rw=None,
        extract=OpExtractConfig(
            size=("start", 2), status=("end", 1), size_optional=True
        ),
    ),
    "META_VERIFY": OperationPair(
        category="metadata",
        subcategory="verify",
        rw=None,
        extract=OpExtractConfig(status=("end", 1)),
    ),
    # Nested inside META_SYNC and META_LOAD, so a breakdown of `metadata`.
    "META_ENCRYPT": OperationPair(
        category="metadata_crypto",
        subcategory="encrypt",
        rw="write",
        extract=OpExtractConfig(),
        mode=ExtractMode.CHUNK,
    ),
    "META_DECRYPT": OperationPair(
        category="metadata_crypto",
        subcategory="decrypt",
        rw="read",
        extract=OpExtractConfig(),
        mode=ExtractMode.CHUNK,
    ),
    "META_MAC_CREATE": OperationPair(
        category="metadata_crypto",
        subcategory="mac_create",
        rw="write",
        extract=OpExtractConfig(status=("end", 1)),
    ),
    "META_MAC_VERIFY": OperationPair(
        category="metadata_crypto",
        subcategory="mac_verify",
        rw="read",
        extract=OpExtractConfig(status=("end", 1)),
    ),
    # All nested: inside the epoch's META_SYNC, or in the operation a gate blocks.
    "META_SNAPSHOT": OperationPair(
        category="epoch",
        subcategory="snapshot",
        rw=None,
        extract=OpExtractConfig(size=("start", 1)),
    ),
    "META_GATE": OperationPair(
        category="epoch",
        subcategory="gate",
        rw=None,
        extract=OpExtractConfig(status=("end", 1)),
    ),
    # The END's number is a mutator count, not a size.
    "META_DRAIN": OperationPair(
        category="epoch",
        subcategory="drain",
        rw=None,
        extract=OpExtractConfig(status=("end", 1)),
    ),
    # The verdict is what tells a `parallel` run that paid an `ordered` cost.
    "META_BARRIER": OperationPair(
        category="epoch",
        subcategory="barrier",
        rw=None,
        extract=OpExtractConfig(
            status=("end", 1), outcome=("end", 2), margin_us=("end", 3)
        ),
    ),
    # Encloses every phase above, so it is kept out of the category total.
    "META_EPOCH": OperationPair(
        category="epoch",
        subcategory="epoch",
        rw=None,
        extract=OpExtractConfig(status=("end", 1)),
    ),
    # Regularly spans two epochs, so it is out of the category total too.
    "FSYNC": OperationPair(
        category="epoch",
        subcategory="fsync",
        rw=None,
        extract=OpExtractConfig(status=("end", 1), op_id=("start", 1)),
    ),
    # The START's first argument is the counter value, not a byte count.
    "TC_COMMIT": OperationPair(
        category="epoch",
        subcategory="commit",
        rw=None,
        extract=OpExtractConfig(status=("end", 1), backend=("start", 2)),
    ),
    # TC_INC was renamed TC_NEXT and kept its tpoint ids, so a dump names either.
    "TC_NEXT": OperationPair(
        category="trusted_counter",
        subcategory="next",
        rw=None,
        extract=OpExtractConfig(),
    ),
    "TC_INC": OperationPair(
        category="trusted_counter",
        subcategory="inc",
        rw=None,
        extract=OpExtractConfig(),
    ),
    "TC_READ": OperationPair(
        category="trusted_counter",
        subcategory="read",
        rw=None,
        extract=OpExtractConfig(),
    ),
    "INIT": OperationPair(
        category="init",
        subcategory="init",
        rw=None,
        extract=OpExtractConfig(),
    ),
    "CLEANUP": OperationPair(
        category="cleanup",
        subcategory="cleanup",
        rw=None,
        extract=OpExtractConfig(size=("start", 1)),
    ),
    "FS_MERKLE_ADD": OperationPair(
        category="merkle",
        subcategory="add",
        rw="write",
        extract=OpExtractConfig(status=("end", 1)),
    ),
    "FS_MERKLE_VER": OperationPair(
        category="merkle",
        subcategory="verify",
        rw="read",
        extract=OpExtractConfig(status=("end", 1)),
    ),
    # Counts and time only: what one costs the device is an md page no argument has.
    "BLOB_CREATE": OperationPair(
        category="blob_lifecycle",
        subcategory="create",
        rw=None,
        extract=OpExtractConfig(status=("end", 1)),
    ),
    # `outcome` separates the opens that read an md page from cache hits.
    "BLOB_OPEN": OperationPair(
        category="blob_lifecycle",
        subcategory="open",
        rw=None,
        extract=OpExtractConfig(status=("end", 1), open_outcome=("end", 2)),
    ),
    "BLOB_SYNC_MD": OperationPair(
        category="blob_lifecycle",
        subcategory="sync_md",
        rw=None,
        extract=OpExtractConfig(status=("end", 1)),
    ),
    "BLOB_DELETE": OperationPair(
        category="blob_lifecycle",
        subcategory="delete",
        rw=None,
        extract=OpExtractConfig(status=("end", 1)),
    ),
    # Size comes off the END - the START's is the requested length, 0 meaning whole blob.
    "BLOB_READ": OperationPair(
        category="blob_io",
        subcategory="read",
        rw="read",
        extract=OpExtractConfig(size=("end", 2), status=("end", 1), kind=("end", 3)),
    ),
    "BLOB_WRITE": OperationPair(
        category="blob_io",
        subcategory="write",
        rw="write",
        extract=OpExtractConfig(size=("end", 2), status=("end", 1), kind=("end", 3)),
    ),
}

# A pair knows its own name, which is half of a span's pairing key.
for _name, _pair in OPERATION_PAIRS.items():
    _pair.name = _name
del _name, _pair

# The subcategories each builder reports, in print order.
METADATA_SUBCATEGORIES = ["sync", "load", "write", "verify"]
# META_WRITE sits inside the META_SYNC span, so it is a line item and no addend.
METADATA_NESTED = frozenset({"write"})
EPOCH_SUBCATEGORIES = [
    "drain",
    "snapshot",
    "gate",
    "commit",
    "barrier",
    "epoch",
    "fsync",
]
# META_EPOCH encloses the phases and an fsync can span two epochs.
EPOCH_ENCLOSING = frozenset({"epoch", "fsync"})
METADATA_CRYPTO_SUBCATEGORIES = ["encrypt", "decrypt", "mac_create", "mac_verify"]

# The `open_*` entries are the spans `open` already counts, so they are line items.
BLOB_LIFECYCLE_SUBCATEGORIES = ["create", "open", "sync_md", "delete"] + [
    f"open_{outcome}" for outcome in BLOB_OPEN_OUTCOMES.values()
]
BLOB_LIFECYCLE_NESTED = frozenset(
    f"open_{outcome}" for outcome in BLOB_OPEN_OUTCOMES.values()
)
# A kind this table lacks lands in `unknown_<n>` and stays in the category total.
BLOB_IO_SUBCATEGORIES = list(BLOB_IO_KINDS.values())

# AccelStore's tracegroups are 0x13..0x1B, and a tpoint id is (group << 6) | index.
AS_TPOINT_ID_FIRST = 0x13 << 6
AS_TPOINT_ID_LAST = ((0x1B + 1) << 6) - 1


class GenericMetricsAccumulator:
    """Accumulates time, counts, bytes and errors by [category][subcategory]."""

    def __init__(self):
        self.time_us: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        self.count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # [category][subcategory][type], type in total/read/write/in/out
        self.bytes: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )

        self.errors: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        self.client_ops: Dict[str, int] = defaultdict(int)

        # {(op_name, object value): (start_entry, op_config)} - see _pair_key.
        self.pending: Dict[Tuple[str, object], Tuple[Dict, OperationPair]] = {}

        # {backend_name: count}
        self.tc_commit_backends: Dict[str, int] = defaultdict(int)

        # {outcome_name: {"count", "margin_us_total", "margin_us_max"}}
        self.barrier_outcomes: Dict[str, Dict[str, float]] = {}

        self.client_type: str = ""

    def accumulate(
        self,
        category: str,
        subcategory: str,
        rw: Optional[str],
        time_us: float = 0.0,
        size: int = 0,
        bytes_in: int = 0,
        bytes_out: int = 0,
        error: bool = False,
    ):
        """Accumulate one completed operation.

        `rw` is 'read', 'write', 'other' (an API call that moves no payload) or None.
        """
        directed = rw in ("read", "write", "other")
        keyed = f"{subcategory}_{rw}" if directed else None

        self.time_us[category][subcategory] += time_us
        if keyed:
            self.time_us[category][keyed] += time_us

        self.count[category][subcategory] += 1
        if keyed:
            self.count[category][keyed] += 1

        # 'other' operations move no bytes, so they get no direction bucket.
        if size > 0:
            self.bytes[category][subcategory]["total"] += size
            if rw == "read":
                self.bytes[category][subcategory]["read"] += size
            elif rw == "write":
                self.bytes[category][subcategory]["write"] += size

        if bytes_in > 0:
            self.bytes[category][subcategory]["in"] += bytes_in
        if bytes_out > 0:
            self.bytes[category][subcategory]["out"] += bytes_out

        if error:
            self.errors[category][subcategory] += 1
            if keyed:
                self.errors[category][keyed] += 1

    def record_barrier(self, outcome: int, margin_us: float) -> None:
        """One epoch's barrier verdict, by name (see BARRIER_OUTCOMES)."""
        name = BARRIER_OUTCOMES.get(outcome, f"unknown_{outcome}")
        stats = self.barrier_outcomes.setdefault(
            name, {"count": 0.0, "margin_us_total": 0.0, "margin_us_max": 0.0}
        )
        stats["count"] += 1
        stats["margin_us_total"] += margin_us
        stats["margin_us_max"] = max(stats["margin_us_max"], float(margin_us))

    def detect_client_type(self):
        if self.client_ops["kvs"] > 0:
            self.client_type = "kvs"
        elif self.client_ops["fs"] > 0:
            self.client_type = "fs"
        elif self.client_ops["log"] > 0:
            self.client_type = "log"
        else:
            self.client_type = "unknown"


class TraceParser:
    """Single-pass parser for SPDK JSON trace dumps."""

    BDEV_BLOCK_SIZE = 512

    # Outside the window by construction, so scoping would zero them.
    OUTSIDE_WINDOW_CATEGORIES = frozenset({"init", "cleanup"})

    # Also read unscoped - the default on-shutdown trigger puts every epoch after
    # WORKLOAD_END.
    WHOLE_RUN_CATEGORIES = frozenset({"metadata", "epoch", "blob_lifecycle", "blob_io"})

    def __init__(
        self,
        filepath: Path,
        bdev_block_size: int = BDEV_BLOCK_SIZE,
    ):
        self.filepath = Path(filepath)
        self.bdev_block_size = bdev_block_size

        self.tsc_rate: int = 0
        self.tpoint_name_to_id: Dict[str, int] = {}
        self.tpoint_id_to_name: Dict[int, str] = {}

        self._start_tpoint_ids: Set[int] = set()
        self._end_tpoint_ids: Set[int] = set()
        self._tpoint_config: Dict[int, OperationPair] = {}
        # AccelStore tpoints this parser has no configuration for.
        self._as_unhandled_ids: Set[int] = set()

        self._workload_start_id: Optional[int] = None
        self._workload_end_id: Optional[int] = None

        self._acc = GenericMetricsAccumulator()

        # The unscoped pass over WHOLE_RUN_CATEGORIES, with its own `pending`.
        self._acc_whole_run = GenericMetricsAccumulator()

        # A bin for a non-primary pass's diagnostics, so alarms are not doubled.
        self._diag_discard: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Unaccounted records and layout facts, keyed by kind and then by family.
        self._diag: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def tsc_to_microseconds(self, tsc_ticks: int) -> float:
        """Convert TSC ticks to microseconds."""
        if self.tsc_rate <= 0:
            raise ValueError("tsc_rate not set or invalid")
        return (tsc_ticks / self.tsc_rate) * 1_000_000.0

    def load_trace(self) -> None:
        """Load trace file and build tpoint mappings using ijson."""
        self.tsc_rate = 0
        self.tpoint_name_to_id = {}
        self.tpoint_id_to_name = {}

        with open(self.filepath, "rb") as f:
            parser = ijson.parse(f)
            current_tpoint = {}
            for prefix, event, value in parser:
                if prefix == "tsc_rate" and event == "number":
                    self.tsc_rate = value
                elif prefix == "tpoints.item":
                    if event == "start_map":
                        current_tpoint = {}
                    elif event == "end_map":
                        tp_id = current_tpoint.get("id")
                        tp_name = current_tpoint.get("name", "UNKNOWN")
                        if tp_id is not None:
                            self.tpoint_id_to_name[tp_id] = tp_name
                            self.tpoint_name_to_id[tp_name] = tp_id
                elif prefix == "tpoints.item.name" and event == "string":
                    current_tpoint["name"] = value
                elif prefix == "tpoints.item.id" and event == "number":
                    current_tpoint["id"] = value
                elif prefix == "entries" and event == "start_array":
                    break

        if self.tsc_rate <= 0:
            raise ValueError(f"tsc_rate must be positive, got {self.tsc_rate}")

        if not self.tpoint_id_to_name:
            raise ValueError("No tpoints found in trace file")

        self._build_tpoint_config()

    def _build_tpoint_config(self) -> None:
        """Build tpoint ID -> OperationPair mapping from OPERATION_PAIRS."""
        for op_name, op_config in OPERATION_PAIRS.items():
            start_id = self.tpoint_name_to_id.get(f"{op_name}_START")
            if op_name == "BDEV_IO":
                end_id = self.tpoint_name_to_id.get("BDEV_IO_DONE")
            else:
                end_id = self.tpoint_name_to_id.get(f"{op_name}_END")

            if start_id is not None:
                self._start_tpoint_ids.add(start_id)
                self._tpoint_config[start_id] = op_config

            if end_id is not None:
                self._end_tpoint_ids.add(end_id)
                self._tpoint_config[end_id] = op_config

        self._workload_start_id = self.tpoint_name_to_id.get("WORKLOAD_START")
        self._workload_end_id = self.tpoint_name_to_id.get("WORKLOAD_END")

        # Fallback for traces captured before every client had WORKLOAD_* markers.
        self._total_start_id = self.tpoint_name_to_id.get("TOTAL_START")
        self._total_end_id = self.tpoint_name_to_id.get("TOTAL_END")

        # AccelStore tpoints with no configuration and no separate handling.
        handled = set(self._tpoint_config)
        handled.update(
            tid
            for tid in (self._workload_start_id, self._workload_end_id)
            if tid is not None
        )
        self._as_unhandled_ids = {
            tid
            for tid in self.tpoint_id_to_name
            if AS_TPOINT_ID_FIRST <= tid <= AS_TPOINT_ID_LAST and tid not in handled
        }

    def parse_and_calculate_metrics(self) -> OutputMetrics:
        """Parse the trace in a single pass and return the metrics."""
        if not self.tpoint_id_to_name:
            self.load_trace()

        # Accumulate optimistically and discard the pre-workload part if a marker
        # turns up.
        in_workload = True
        saw_marker_start = False

        first_tsc = None
        last_tsc = None

        # The workload window, from the markers or else from the TOTAL spans.
        marker_start_tsc = None
        marker_end_tsc = None
        op_start_tsc = None
        op_end_tsc = None

        with open(self.filepath, "rb") as f:
            for entry in ijson.items(f, "entries.item"):
                tpoint_id = entry.get("tpoint")
                tsc = entry.get("tsc", 0)

                if first_tsc is None:
                    first_tsc = tsc
                last_tsc = tsc

                # An unregistered tpoint's id is None, which an entry with no
                # tpoint field would match.
                if tpoint_id is None:
                    continue
                elif tpoint_id == self._workload_start_id:
                    if marker_start_tsc is None:
                        marker_start_tsc = tsc
                elif tpoint_id == self._workload_end_id:
                    marker_end_tsc = tsc
                elif tpoint_id == self._total_start_id:
                    if op_start_tsc is None:
                        op_start_tsc = tsc
                elif tpoint_id == self._total_end_id:
                    op_end_tsc = tsc

                if tpoint_id == self._workload_start_id:
                    if not saw_marker_start:
                        saw_marker_start = True
                        self._discard_pre_workload()
                    in_workload = True
                    continue
                elif tpoint_id == self._workload_end_id:
                    in_workload = False
                    continue

                # An unconfigured AccelStore tpoint lands in no category at all.
                if tpoint_id not in self._tpoint_config:
                    if tpoint_id in self._as_unhandled_ids:
                        name = self.tpoint_id_to_name.get(tpoint_id, str(tpoint_id))
                        self._diag["unhandled_accelstore_tpoint"][name] += 1
                    continue

                # The unscoped pass, before the window check discards anything.
                if self._tpoint_config[tpoint_id].category in self.WHOLE_RUN_CATEGORIES:
                    self._process_entry(entry, tpoint_id, self._acc_whole_run)

                # Keeps the buckets on the same region as total_execution_time_us.
                if (
                    not in_workload
                    and self._tpoint_config[tpoint_id].category
                    not in self.OUTSIDE_WINDOW_CATEGORIES
                ):
                    continue

                self._process_entry(entry, tpoint_id)

        # A START still open here is one whose END the capture did not reach.
        for op_name, _obj in self._acc.pending:
            self._diag["unfinished_start"][op_name] += 1

        self._acc.detect_client_type()

        workload_start_tsc, workload_end_tsc, source = self._resolve_window(
            marker_start_tsc, marker_end_tsc, op_start_tsc, op_end_tsc
        )

        return self._build_output_metrics(
            first_tsc, last_tsc, workload_start_tsc, workload_end_tsc, source
        )

    def _resolve_window(
        self,
        marker_start_tsc: Optional[int],
        marker_end_tsc: Optional[int],
        op_start_tsc: Optional[int],
        op_end_tsc: Optional[int],
    ) -> Tuple[Optional[int], Optional[int], str]:
        """Pick the workload window and say where it came from."""
        if marker_start_tsc is not None and marker_end_tsc is not None:
            return marker_start_tsc, marker_end_tsc, "markers"

        if op_start_tsc is not None and op_end_tsc is not None:
            warnings.warn(
                f"{self.filepath}: no WORKLOAD_START/END markers; falling back "
                "to first TOTAL_START..last TOTAL_END. This spans every "
                "operation, so any load phase the workload runs before START "
                "is counted. Re-capture with a client that emits the markers "
                "to scope it."
            )
            return op_start_tsc, op_end_tsc, "total_spans"

        return None, None, "none"

    def _discard_pre_workload(self) -> None:
        """Drop everything accumulated before WORKLOAD_START, keeping init."""
        carried = GenericMetricsAccumulator()
        for category in self.OUTSIDE_WINDOW_CATEGORIES:
            if category in self._acc.time_us:
                carried.time_us[category] = self._acc.time_us[category]
            if category in self._acc.count:
                carried.count[category] = self._acc.count[category]
            if category in self._acc.bytes:
                carried.bytes[category] = self._acc.bytes[category]
            if category in self._acc.errors:
                carried.errors[category] = self._acc.errors[category]
        self._acc = carried

    @staticmethod
    def _pair_key(entry: Dict, op_config: OperationPair):
        """The key a START and its END are matched on, or None if there is none.

        (family, object.value): the family because nested spans share a context
        pointer, and `value` because `object.id` is unique per object type only.
        """
        obj = entry.get("object") or {}
        key = obj.get("value")
        if key is None:
            key = obj.get("id")
        if key is None:
            return None
        return (op_config.name, key)

    def _diag_for(self, acc: "GenericMetricsAccumulator") -> Dict[str, Dict[str, int]]:
        """Where this pass's diagnostics go - only the scoped pass reports."""
        return self._diag if acc is self._acc else self._diag_discard

    def _process_entry(
        self,
        entry: Dict,
        tpoint_id: int,
        acc: Optional["GenericMetricsAccumulator"] = None,
    ) -> None:
        """Process one START or END entry into `acc`, the scoped pass by default."""
        acc = self._acc if acc is None else acc
        diag = self._diag_for(acc)

        op_config = self._tpoint_config.get(tpoint_id)
        if op_config is None:
            return

        key = self._pair_key(entry, op_config)
        if key is None:
            diag["missing_object_key"][op_config.name] += 1
            return

        if tpoint_id in self._start_tpoint_ids:
            # Which thread began the span, as the raw owner string the dump has.
            owner = entry.get("poller")
            if owner:
                diag["owners"][str(owner)] += 1

            # A START already pending is one whose END never arrived.
            if key in acc.pending:
                diag["unmatched_start"][op_config.name] += 1
            acc.pending[key] = (entry, op_config)

        elif tpoint_id in self._end_tpoint_ids:
            pending = acc.pending.pop(key, None)
            if pending is None:
                diag["unmatched_end"][op_config.name] += 1
                return

            start_entry, op_config = pending
            self._accumulate_pair(start_entry, entry, op_config, acc)

    def _extract_value(
        self,
        start_entry: Dict,
        end_entry: Dict,
        location: Optional[Tuple[str, int]],
        op_name: str = "",
        optional: bool = False,
        diag: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> int:
        """Extract a value from `location`, or 0 if the argument list is shorter.

        A short list is counted as layout drift unless `optional`.
        """
        if location is None:
            return 0

        which, idx = location
        entry = start_entry if which == "start" else end_entry
        args = entry.get("args", [])
        if len(args) <= idx:
            if not optional:
                diag = self._diag if diag is None else diag
                diag["arg_out_of_range"][f"{op_name}.{which}[{idx}]"] += 1
            return 0
        return args[idx]

    def _extract_time(self, end_entry: Dict) -> float:
        """Extract time in microseconds from the object.time."""
        time_tsc = end_entry.get("object", {}).get("time")
        if time_tsc is None:
            tpoint_name = self.tpoint_id_to_name.get(end_entry.get("tpoint"), "UNKNOWN")
            warnings.warn(f"Time object not found for tpoint {tpoint_name}")
            return 0.0
        return self.tsc_to_microseconds(time_tsc)

    def _accumulate_chunk_pair(
        self,
        start_entry: Dict,
        end_entry: Dict,
        op_config: OperationPair,
        acc: Optional["GenericMetricsAccumulator"] = None,
    ) -> None:
        """A per-chunk crypto span, in either of its two layouts.

            old   START (caller_ctx, i_sz)   END (caller_ctx, status, o_sz)
            new   START (chunk_id,   op_id)  END (status)

        The END's argument count is the discriminator, told apart per entry.
        """
        acc = self._acc if acc is None else acc
        diag = self._diag_for(acc)
        end_args = end_entry.get("args", [])
        legacy = len(end_args) >= 3

        if legacy:
            diag["chunk_layout"]["pre_206"] += 1
            bytes_in = self._extract_value(
                start_entry, end_entry, ("start", 1), op_config.name, diag=diag
            )
            bytes_out = self._extract_value(
                start_entry, end_entry, ("end", 2), op_config.name, diag=diag
            )
            status = self._extract_value(
                start_entry, end_entry, ("end", 1), op_config.name, diag=diag
            )
        else:
            diag["chunk_layout"]["v206"] += 1
            # An AEAD chunk's ciphertext is as long as its plaintext, so in == out.
            chunk_bytes = start_entry.get("size") or end_entry.get("size") or 0
            if not chunk_bytes:
                diag["chunk_size_field_missing"][op_config.name] += 1
            bytes_in = bytes_out = chunk_bytes
            status = self._extract_value(
                start_entry, end_entry, ("end", 0), op_config.name, diag=diag
            )
            # args[0] is the chunk_id and args[1] the op_id, neither aggregated yet.
            if len(start_entry.get("args", [])) > 1:
                diag["op_id_spans"][op_config.name] += 1

        acc.accumulate(
            category=op_config.category,
            subcategory=op_config.subcategory,
            rw=op_config.rw,
            time_us=self._extract_time(end_entry),
            bytes_in=bytes_in,
            bytes_out=bytes_out,
            error=status != 0,
        )

    def _accumulate_pair(
        self,
        start_entry: Dict,
        end_entry: Dict,
        op_config: OperationPair,
        acc: Optional["GenericMetricsAccumulator"] = None,
    ) -> None:
        """Accumulate metrics from a paired START/END entry."""
        acc = self._acc if acc is None else acc
        diag = self._diag_for(acc)
        extract = op_config.extract
        category = op_config.category
        subcategory = op_config.subcategory
        rw = op_config.rw

        if category == "api":
            acc.client_ops[subcategory] += 1

        if op_config.mode is ExtractMode.CHUNK:
            self._accumulate_chunk_pair(start_entry, end_entry, op_config, acc)
            return

        if category == "bdev":
            io_type = self._extract_value(
                start_entry, end_entry, extract.io_type, op_config.name, diag=diag
            )
            size_blocks = start_entry.get("size", 0)
            size_bytes = size_blocks * self.bdev_block_size

            if io_type == SpdkBdevIoType.READ:
                acc.bytes["bdev"]["io"]["total"] += size_bytes
                acc.bytes["bdev"]["io"]["read"] += size_bytes
            elif io_type in [SpdkBdevIoType.WRITE]:
                acc.bytes["bdev"]["io"]["total"] += size_bytes
                acc.bytes["bdev"]["io"]["write"] += size_bytes
            return

        # An unnamed kind keeps its number, so its bytes stay in the category total.
        if category == "blob_io":
            kind = self._extract_value(
                start_entry, end_entry, extract.kind, op_config.name, diag=diag
            )
            kind_name = BLOB_IO_KINDS.get(kind)
            if kind_name is None:
                diag["unknown_blob_io_kind"][f"{op_config.name}[{kind}]"] += 1
                kind_name = f"unknown_{kind}"
            size = self._extract_value(
                start_entry, end_entry, extract.size, op_config.name, diag=diag
            )
            status = self._extract_value(
                start_entry, end_entry, extract.status, op_config.name, diag=diag
            )
            acc.accumulate(
                category=category,
                subcategory=kind_name,
                rw=rw,
                time_us=self._extract_time(end_entry),
                size=size,
                error=status != 0,
            )
            return

        if category in ("gpu", "total") and extract.op_type:
            op_type = self._extract_value(
                start_entry, end_entry, extract.op_type, op_config.name, diag=diag
            )
            if op_type == 2:  # Skip "other" operations
                return

            if op_type == 0:
                actual_rw = "read"
            elif op_type == 1:
                actual_rw = "write"
            else:
                actual_rw = None

            size = self._extract_value(
                start_entry, end_entry, extract.size, op_config.name, diag=diag
            )
            # A read's size may only be known at the END (e.g. KVS GET).
            if (
                size == 0
                and actual_rw == "read"
                and extract.size
                and extract.size[0] == "start"
            ):
                fallback_loc = ("end", extract.size[1])
                size = self._extract_value(
                    start_entry, end_entry, fallback_loc, op_config.name, diag=diag
                )

            time_us = self._extract_time(end_entry)

            acc.accumulate(
                category=category,
                subcategory=subcategory,
                rw=actual_rw,
                time_us=time_us,
                size=size,
            )
            return

        name = op_config.name
        size = self._extract_value(
            start_entry,
            end_entry,
            extract.size,
            name,
            optional=extract.size_optional,
            diag=diag,
        )
        time_us = self._extract_time(end_entry)
        bytes_in = self._extract_value(
            start_entry, end_entry, extract.bytes_in, name, diag=diag
        )
        bytes_out = self._extract_value(
            start_entry, end_entry, extract.bytes_out, name, diag=diag
        )

        error = False
        if extract.status:
            status = self._extract_value(
                start_entry, end_entry, extract.status, name, diag=diag
            )
            error = status != 0

        # Recorded as availability only - nothing joins chunks to operations yet.
        if extract.op_id is not None:
            which, idx = extract.op_id
            args = (start_entry if which == "start" else end_entry).get("args", [])
            if len(args) > idx:
                diag["op_id_spans"][name] += 1

        # A sub-operation of its own, so it is scoped and cached like any line item.
        if extract.open_outcome is not None:
            outcome = self._extract_value(
                start_entry, end_entry, extract.open_outcome, name, diag=diag
            )
            outcome_name = BLOB_OPEN_OUTCOMES.get(outcome)
            if outcome_name is None:
                diag["unknown_blob_open_outcome"][str(outcome)] += 1
                outcome_name = f"unknown_{outcome}"
            acc.accumulate(
                category=category,
                subcategory=f"open_{outcome_name}",
                rw=None,
                time_us=time_us,
            )

        if extract.outcome is not None:
            outcome = self._extract_value(
                start_entry, end_entry, extract.outcome, name, diag=diag
            )
            margin_us = self._extract_value(
                start_entry, end_entry, extract.margin_us, name, diag=diag
            )
            acc.record_barrier(outcome, margin_us)

        if extract.backend is not None:
            backend = self._extract_value(
                start_entry, end_entry, extract.backend, name, diag=diag
            )
            if backend not in TC_BACKENDS:
                # Otherwise the commit counts would be short with nothing to say why.
                diag["unknown_tc_backend"][str(backend)] += 1
            acc.tc_commit_backends[TC_BACKENDS.get(backend, f"unknown_{backend}")] += 1

        acc.accumulate(
            category=category,
            subcategory=subcategory,
            rw=rw,
            time_us=time_us,
            size=size,
            bytes_in=bytes_in,
            bytes_out=bytes_out,
            error=error,
        )

    def _span_us(self, start: Optional[int], end: Optional[int]) -> float:
        """Duration between two tsc marks, or 0.0 if the pair is unusable."""
        if start is None or end is None or end <= start:
            return 0.0
        return self.tsc_to_microseconds(end - start)

    def _build_output_metrics(
        self,
        first_tsc: Optional[int],
        last_tsc: Optional[int],
        workload_start_tsc: Optional[int] = None,
        workload_end_tsc: Optional[int] = None,
        workload_window_source: str = "none",
    ) -> OutputMetrics:
        """Build OutputMetrics from accumulated dictionary values."""
        full_trace_time_us = self._span_us(first_tsc, last_tsc)

        # A trace with no operation spans at all falls back to the capture window.
        if workload_start_tsc is None or workload_end_tsc is None:
            warnings.warn(
                f"{self.filepath}: no WORKLOAD_*/TOTAL_* spans found, falling back "
                "to the trace capture window for total_execution_time_us. This "
                "includes blobstore load and client setup and is not comparable "
                "against runs with a different tpoint mask."
            )
            total_execution_time_us = full_trace_time_us
            workload_window_source = "capture"
        else:
            total_execution_time_us = self._span_us(
                workload_start_tsc, workload_end_tsc
            )

        metrics = OutputMetrics(
            trace_source=str(self.filepath),
            analysis_timestamp=datetime.now().isoformat(),
            trace_start_tsc=first_tsc or 0,
            trace_end_tsc=last_tsc or 0,
            workload_start_tsc=workload_start_tsc or 0,
            workload_end_tsc=workload_end_tsc or 0,
            workload_window_source=workload_window_source,
            total_execution_time_us=total_execution_time_us,
            full_trace_time_us=full_trace_time_us,
            setup_time_us=self._span_us(first_tsc, workload_start_tsc),
            teardown_time_us=self._span_us(workload_end_tsc, last_tsc),
            epoch_barrier={
                name: dict(stats)
                for name, stats in sorted(self._acc.barrier_outcomes.items())
            },
            diagnostics={
                key: dict(sub) for key, sub in sorted(self._diag.items()) if sub
            },
        )

        builders = {
            "api": self._build_api_metrics,
            "bdev": self._build_bdev_metrics,
            "gpu": lambda: self._build_rw_category_metrics("gpu", "compute"),
            "total": lambda: self._build_rw_category_metrics("total", "pipeline"),
            "encryption": self._build_encryption_metrics,
            "metadata": lambda: self._build_simple_metrics(
                "metadata", METADATA_SUBCATEGORIES, exclude_from_total=METADATA_NESTED
            ),
            "trusted_counter": self._build_trusted_counter_metrics,
            "epoch": lambda: self._build_simple_metrics(
                "epoch", EPOCH_SUBCATEGORIES, exclude_from_total=EPOCH_ENCLOSING
            ),
            "metadata_crypto": lambda: self._build_simple_metrics(
                "metadata_crypto", METADATA_CRYPTO_SUBCATEGORIES
            ),
            "log_cache": lambda: self._build_simple_metrics("log_cache", ["build"]),
            "metadata_whole_run": lambda: self._build_simple_metrics(
                "metadata",
                METADATA_SUBCATEGORIES,
                exclude_from_total=METADATA_NESTED,
                acc=self._acc_whole_run,
            ),
            "epoch_whole_run": lambda: self._build_simple_metrics(
                "epoch",
                EPOCH_SUBCATEGORIES,
                exclude_from_total=EPOCH_ENCLOSING,
                acc=self._acc_whole_run,
            ),
            "merkle": lambda: self._build_simple_metrics("merkle", ["add", "verify"]),
            "blob_lifecycle": lambda: self._build_simple_metrics(
                "blob_lifecycle",
                BLOB_LIFECYCLE_SUBCATEGORIES,
                exclude_from_total=BLOB_LIFECYCLE_NESTED,
            ),
            "blob_io": self._build_blob_io_metrics,
            "blob_lifecycle_whole_run": lambda: self._build_simple_metrics(
                "blob_lifecycle",
                BLOB_LIFECYCLE_SUBCATEGORIES,
                exclude_from_total=BLOB_LIFECYCLE_NESTED,
                acc=self._acc_whole_run,
            ),
            "blob_io_whole_run": lambda: self._build_blob_io_metrics(
                acc=self._acc_whole_run
            ),
            "init": lambda: self._build_simple_metrics("init", ["init"]),
            "cleanup": lambda: self._build_simple_metrics("cleanup", ["cleanup"]),
        }
        for field, builder in builders.items():
            setattr(metrics, field, builder())

        return metrics

    @staticmethod
    def _mibps(bytes_count: int, time_us: float) -> float:
        """Convert bytes + microseconds to MiB/s."""
        return bytes_count / (time_us / 1_000_000.0) / float(1 << 20)

    def _build_api_metrics(self) -> OperationMetrics:
        """Build API OperationMetrics from accumulator."""
        acc = self._acc
        api = OperationMetrics(client_type=acc.client_type)
        subs = ["kvs", "fs", "log"]

        api.total_time_us = sum(acc.time_us["api"].get(s, 0) for s in subs)
        api.total_bytes = sum(acc.bytes["api"][s]["total"] for s in subs)
        api.operation_count = sum(acc.count["api"].get(s, 0) for s in subs)

        api.read_time_us = sum(acc.time_us["api"].get(f"{s}_read", 0) for s in subs)
        api.read_bytes = sum(acc.bytes["api"][s]["read"] for s in subs)
        api.read_count = sum(acc.count["api"].get(f"{s}_read", 0) for s in subs)
        api.read_error_count = sum(acc.errors["api"].get(f"{s}_read", 0) for s in subs)

        api.write_time_us = sum(acc.time_us["api"].get(f"{s}_write", 0) for s in subs)
        api.write_bytes = sum(acc.bytes["api"][s]["write"] for s in subs)
        api.write_count = sum(acc.count["api"].get(f"{s}_write", 0) for s in subs)
        api.write_error_count = sum(
            acc.errors["api"].get(f"{s}_write", 0) for s in subs
        )

        # Zero-byte calls are counted but stay out of the byte rates (see _api_other).
        api.other_time_us = sum(acc.time_us["api"].get(f"{s}_other", 0) for s in subs)
        api.other_count = sum(acc.count["api"].get(f"{s}_other", 0) for s in subs)
        api.other_error_count = sum(
            acc.errors["api"].get(f"{s}_other", 0) for s in subs
        )

        api.error_count = (
            api.read_error_count + api.write_error_count + api.other_error_count
        )

        rw_time_us = api.read_time_us + api.write_time_us

        if api.total_time_us > 0:
            api.throughput_iops = api.operation_count / (
                api.total_time_us / 1_000_000.0
            )
        if rw_time_us > 0:
            api.throughput_mibps = self._mibps(api.total_bytes, rw_time_us)
        if api.read_time_us > 0:
            api.read_throughput_mibps = self._mibps(api.read_bytes, api.read_time_us)
            if api.read_count > 0:
                api.read_avg_latency_us = api.read_time_us / api.read_count
        if api.write_time_us > 0:
            api.write_throughput_mibps = self._mibps(api.write_bytes, api.write_time_us)
            if api.write_count > 0:
                api.write_avg_latency_us = api.write_time_us / api.write_count
        if api.operation_count > 0:
            api.avg_latency_us = api.total_time_us / api.operation_count

        return api

    def _build_bdev_metrics(self) -> OperationMetrics:
        """Build BDEV OperationMetrics from accumulator."""
        b = self._acc.bytes["bdev"]["io"]
        return OperationMetrics(
            total_bytes=b["total"], read_bytes=b["read"], write_bytes=b["write"]
        )

    def _build_rw_category_metrics(
        self, category: str, subcategory: str
    ) -> OperationMetrics:
        """Build OperationMetrics for a category with read/write split (GPU, TOTAL)."""
        acc = self._acc
        m = OperationMetrics()
        sub, sub_r, sub_w = subcategory, f"{subcategory}_read", f"{subcategory}_write"

        m.total_time_us = acc.time_us[category].get(sub, 0)
        m.read_time_us = acc.time_us[category].get(sub_r, 0)
        m.write_time_us = acc.time_us[category].get(sub_w, 0)
        m.total_bytes = acc.bytes[category][sub]["total"]
        m.read_bytes = acc.bytes[category][sub]["read"]
        m.write_bytes = acc.bytes[category][sub]["write"]
        m.operation_count = acc.count[category].get(sub, 0)
        m.read_count = acc.count[category].get(sub_r, 0)
        m.write_count = acc.count[category].get(sub_w, 0)

        if m.total_time_us > 0:
            m.throughput_mibps = self._mibps(m.total_bytes, m.total_time_us)
        if m.read_time_us > 0:
            m.read_throughput_mibps = self._mibps(m.read_bytes, m.read_time_us)
        if m.write_time_us > 0:
            m.write_throughput_mibps = self._mibps(m.write_bytes, m.write_time_us)

        return m

    def _build_encryption_metrics(self) -> OperationMetrics:
        """Build encryption OperationMetrics from accumulator."""
        acc = self._acc
        enc_cat = acc.bytes["encryption"]
        enc_time = acc.time_us["encryption"]
        enc_count = acc.count["encryption"]

        groups = {
            "encrypt": ["encrypt", "mac_create"],
            "decrypt": ["decrypt", "mac_verify"],
        }
        subs: Dict[str, OperationMetrics] = {}
        for name, parts in groups.items():
            s = OperationMetrics()
            s.total_time_us = sum(enc_time.get(p, 0) for p in parts)
            s.operation_count = sum(enc_count.get(p, 0) for p in parts)
            s.bytes_input = sum(enc_cat[p]["in"] for p in parts)
            s.bytes_output = sum(enc_cat[p]["out"] for p in parts)
            s.bytes_overhead = abs(s.bytes_output - s.bytes_input)
            subs[name] = s

        encrypt, decrypt = subs["encrypt"], subs["decrypt"]
        e = OperationMetrics(
            total_time_us=encrypt.total_time_us + decrypt.total_time_us,
            write_time_us=encrypt.total_time_us,
            read_time_us=decrypt.total_time_us,
            bytes_input=encrypt.bytes_input + decrypt.bytes_input,
            bytes_output=encrypt.bytes_output + decrypt.bytes_output,
            bytes_overhead=encrypt.bytes_overhead + decrypt.bytes_overhead,
            operation_count=encrypt.operation_count + decrypt.operation_count,
            write_count=encrypt.operation_count,
            read_count=decrypt.operation_count,
            write_bytes_input=encrypt.bytes_input,
            write_bytes_output=encrypt.bytes_output,
            read_bytes_input=decrypt.bytes_input,
            read_bytes_output=decrypt.bytes_output,
            sub_operations=subs,
        )
        return e

    def _build_blob_io_metrics(
        self, acc: Optional["GenericMetricsAccumulator"] = None
    ) -> OperationMetrics:
        """The blob I/O spans' device bytes, split by kind, unknown kinds included."""
        acc = self._acc if acc is None else acc
        m = OperationMetrics()

        seen = {
            sub
            for sub in acc.count["blob_io"]
            if not sub.endswith(("_read", "_write", "_other"))
        }
        for kind in list(BLOB_IO_SUBCATEGORIES) + sorted(
            seen.difference(BLOB_IO_SUBCATEGORIES)
        ):
            b = acc.bytes["blob_io"][kind]
            s = OperationMetrics(
                total_time_us=acc.time_us["blob_io"].get(kind, 0),
                read_time_us=acc.time_us["blob_io"].get(f"{kind}_read", 0),
                write_time_us=acc.time_us["blob_io"].get(f"{kind}_write", 0),
                total_bytes=b["total"],
                read_bytes=b["read"],
                write_bytes=b["write"],
                operation_count=acc.count["blob_io"].get(kind, 0),
                read_count=acc.count["blob_io"].get(f"{kind}_read", 0),
                write_count=acc.count["blob_io"].get(f"{kind}_write", 0),
                error_count=acc.errors["blob_io"].get(kind, 0),
            )
            m.sub_operations[kind] = s
            m.total_time_us += s.total_time_us
            m.read_time_us += s.read_time_us
            m.write_time_us += s.write_time_us
            m.total_bytes += s.total_bytes
            m.read_bytes += s.read_bytes
            m.write_bytes += s.write_bytes
            m.operation_count += s.operation_count
            m.read_count += s.read_count
            m.write_count += s.write_count
            m.error_count += s.error_count

        if m.operation_count > 0:
            m.avg_latency_us = m.total_time_us / m.operation_count
        return m

    def _build_simple_metrics(
        self,
        category: str,
        subcategories: List[str],
        exclude_from_total: frozenset = frozenset(),
        acc: Optional["GenericMetricsAccumulator"] = None,
    ) -> OperationMetrics:
        """Aggregate a category's subcategories into one OperationMetrics.

        `exclude_from_total` names subcategories recorded as line items but left
        out of the total. `acc` defaults to the window-scoped pass.
        """
        acc = self._acc if acc is None else acc
        m = OperationMetrics()
        for sub in subcategories:
            b = acc.bytes[category][sub]
            s = OperationMetrics(
                total_time_us=acc.time_us[category].get(sub, 0),
                operation_count=acc.count[category].get(sub, 0),
                total_bytes=b["total"],
                # Set by the chunk crypto families, which report a per-chunk length.
                bytes_input=b["in"],
                bytes_output=b["out"],
            )
            if len(subcategories) > 1:
                m.sub_operations[sub] = s
            if sub in exclude_from_total:
                continue
            m.total_time_us += s.total_time_us
            m.operation_count += s.operation_count
            m.total_bytes += s.total_bytes
            m.bytes_input += s.bytes_input
            m.bytes_output += s.bytes_output
        if m.operation_count > 0:
            m.avg_latency_us = m.total_time_us / m.operation_count
        return m

    def _build_trusted_counter_metrics(self) -> OperationMetrics:
        """The counter's own spans, plus which service answered the commits."""
        m = self._build_simple_metrics("trusted_counter", ["next", "inc", "read"])
        backends = self._acc.tc_commit_backends
        m.commits_local = backends.get("local", 0)
        m.commits_remote = backends.get("mock_remote", 0)
        return m
