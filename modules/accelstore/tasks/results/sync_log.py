#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""What a run's own log says about its metadata epochs.

Five lines that no trace carries: the client's configuration, the chunk
pipeline, the blobstore geometry, the parallel-coupling summary and the lag
summary.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from . import discovery

# clients/client_common.c, printed just before as_init().
_CONFIG_RE = re.compile(r"^Metadata sync: (?P<config>trigger=.*)$", re.MULTILINE)

# clients/client_common.c, common_print_crypto_pipeline(). Only the chunk size is read.
_CHUNK_RE = re.compile(r"^Crypto pipeline: chunk-size=(?P<chunk>\d+)\b", re.MULTILINE)

# clients/client_common.c, common_print_bs_geometry(). A loaded store keeps its
# superblock geometry regardless of --cluster-size/--md-pages.
_GEOMETRY_RE = re.compile(
    r"^Blobstore geometry: cluster-size=(?P<cluster>\d+) io-unit=(?P<io_unit>\d+)"
    r".*? store=(?P<store>\w+)$",
    re.MULTILINE,
)

# lib/util/meta_sync_engine.c, at engine release, WARNLOG.
_RACE_RE = re.compile(
    r"Metadata sync coupling parallel: (?P<lost>\d+) of (?P<commits>\d+)"
    r" counter commits landed before their own epoch's metadata blob was durable"
    r" \(mean margin (?P<margin>\d+) us\)"
)

# lib/util/meta_sync_engine.c, at engine release. Printed only when non-zero.
_LAG_RE = re.compile(
    r"Metadata sync lag: (?P<uncovered>\d+) mutating operations were answered"
    r" while an epoch was in flight"
)


@dataclass
class RunLog:
    """The epoch-relevant facts in one client run's log."""

    path: Path
    sync_config: Optional[str] = None
    # --crypto-chunk-size in force, in bytes. None when the log lacks the line.
    crypto_chunk_size: Optional[int] = None
    # The store's own geometry. None when the log lacks the line.
    cluster_size: Optional[int] = None
    io_unit_size: Optional[int] = None
    # True when this run initialized the store, False when it loaded one.
    store_created: Optional[bool] = None
    commits: Optional[int] = None
    race_lost: int = 0
    race_lost_mean_margin_us: Optional[int] = None
    # Mutations answered while an epoch was in flight, zero when no lag line.
    ops_uncovered: int = 0

    @property
    def race_lost_share(self) -> Optional[float]:
        """Fraction of this run's commits whose ordering did not hold."""
        if not self.commits:
            return None
        return self.race_lost / self.commits


def parse_run_log(path: Path) -> RunLog:
    """Read the epoch facts out of a client run log; missing ones stay None."""
    log = RunLog(path=path)
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return log

    config = _CONFIG_RE.search(text)
    if config:
        log.sync_config = config.group("config").strip()

    chunk = _CHUNK_RE.search(text)
    if chunk:
        log.crypto_chunk_size = int(chunk.group("chunk"))

    geometry = _GEOMETRY_RE.search(text)
    if geometry:
        log.cluster_size = int(geometry.group("cluster"))
        log.io_unit_size = int(geometry.group("io_unit"))
        log.store_created = geometry.group("store") == "created"

    race = _RACE_RE.search(text)
    if race:
        log.race_lost = int(race.group("lost"))
        log.commits = int(race.group("commits"))
        log.race_lost_mean_margin_us = int(race.group("margin"))

    lag = _LAG_RE.search(text)
    if lag:
        log.ops_uncovered = int(lag.group("uncovered"))

    return log


def average_run_logs(logs: List[RunLog]) -> Optional[RunLog]:
    """Fold a point's repeat run logs (newest first) into one, or None if empty.

    Counts average over every log, the lost-race margin only over runs that
    lost one, weighted by losses. Configuration fields come from the newest
    log, and disagreements between runs are reported.
    """
    if not logs:
        return None
    if len(logs) == 1:
        return logs[0]

    configs = {log.sync_config for log in logs if log.sync_config}
    if len(configs) > 1:
        print(
            f"  ⚠ {logs[0].path.parent}: averaging runs with different sync"
            f" configurations: {sorted(configs)}"
        )

    chunks = {log.crypto_chunk_size for log in logs if log.crypto_chunk_size}
    if len(chunks) > 1:
        print(
            f"  ⚠ {logs[0].path.parent}: averaging runs with different crypto"
            f" chunk sizes: {sorted(chunks)}"
        )

    geometries = {
        (log.cluster_size, log.io_unit_size)
        for log in logs
        if log.cluster_size or log.io_unit_size
    }
    if len(geometries) > 1:
        print(
            f"  ⚠ {logs[0].path.parent}: averaging runs with different blobstore"
            f" geometries (cluster, io unit): {sorted(geometries)}"
        )

    commits = [log.commits for log in logs if log.commits is not None]
    lost = [log.race_lost for log in logs]
    margins = [
        (log.race_lost, float(log.race_lost_mean_margin_us))
        for log in logs
        if log.race_lost and log.race_lost_mean_margin_us is not None
    ]
    lost_total = sum(count for count, _ in margins)

    return RunLog(
        path=logs[0].path,
        sync_config=logs[0].sync_config,
        crypto_chunk_size=logs[0].crypto_chunk_size,
        cluster_size=logs[0].cluster_size,
        io_unit_size=logs[0].io_unit_size,
        store_created=logs[0].store_created,
        commits=int(sum(commits) / len(commits) + 0.5) if commits else None,
        race_lost=int(sum(lost) / len(lost) + 0.5),
        race_lost_mean_margin_us=(
            int(sum(count * margin for count, margin in margins) / lost_total + 0.5)
            if lost_total
            else None
        ),
        ops_uncovered=int(sum(log.ops_uncovered for log in logs) / len(logs) + 0.5),
    )


def find_run_log(trace_path: Path) -> Optional[Path]:
    """The client log belonging to a trace dump, by their shared timestamp."""
    stem = discovery.trace_stem(trace_path.name)
    if stem is None:
        return None

    candidate = trace_path.parent / f"{stem}.log"
    if candidate.exists():
        return candidate

    matches = sorted(trace_path.parent.glob(f"{stem}*.log"))
    return matches[0] if matches else None
