#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The bench-personality half of the paper's application figure.

Owns the bench result format and hands :mod:`plots.apps` the cells it draws.
Every personality is read at its own native thread count
(:data:`NATIVE_THREADS`), ops/s is total ops over total wall clock with phases
summed, and missing cells are reported gaps rather than estimates.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..crypto_backend import CryptoBackend
from . import bench_results
from .bench_results import BENCH_CLIENT, BenchRunMetrics

# Each personality's declared `.wl` thread count. A run is a cell only at that count.
NATIVE_THREADS = {
    "webserver_fs": 63,
    "webserver_log": 63,
    "videoserver_fs": 48,
    "varmail_log": 16,
    "fileserver_log": 50,
    "oltp_log": 63,
    "modelload_fs": 16,
    "imagepull_fs": 16,
    "ledger_log": 8,
    "datascan_fs": 8,
    "trainload_fs": 4,
    "trainload_fs_paced": 4,
    "videoaudit_log": 16,
    "videoaudit_log_fsync": 8,
    "videoaudit_log_64m_fsync": 8,
    "videoserver_fs_1g_session": 48,
}


@dataclass(frozen=True)
class System:
    """One series: a storage stack, and how a result directory names it.

    `arm` is the arm directory, except for CVM-Store, whose directory is the
    crypto-prefixed client binary. On the POSIX arms a series is also an I/O
    mode, so O_DIRECT and buffered runs are never pooled.
    """

    key: str
    label: str
    optional: bool = False
    #: Which client build, for CVM-Store series.
    crypto: CryptoBackend = CryptoBackend.AEGIS
    #: ``"direct"``, ``"buffered"``, or None for "either".
    io_mode: Optional[str] = None

    def matches(self, arm: str) -> bool:
        """Whether an arm *directory* is this series, I/O mode aside."""
        if self.key == BENCH_CLIENT:
            # One build only: the AES-GCM binary belongs to the ablation figure.
            base, crypto = CryptoBackend.split_client_binary(arm)
            return base == BENCH_CLIENT and crypto is self.crypto
        return arm == self.key

    def selects(self, run: BenchRunMetrics) -> bool:
        """Whether a run is a measurement of this series: arm and I/O mode."""
        if not self.matches(run.arm):
            return False
        if self.io_mode is None:
            return True
        return run.direct == (self.io_mode == "direct")

    @property
    def color(self):
        return bench_results.arm_color(self.key)

    @property
    def hatch(self) -> str:
        return bench_results.POSIX_ARM_HATCHES.get(self.key, "")


# Bar and legend order: system under test first, block-level baselines in
# decreasing order of what they protect, unprotected filesystem last.
SYSTEMS = (
    System(BENCH_CLIENT, "CVM-Store"),
    System("rollbaccine", "Rollbaccine"),
    System("ext4", "ext4"),
    System("ext4-aes-xts-plain", "LUKS", optional=True),
)

#: The f=1, checksums-only, colocated Rollbaccine arm, same key/label/colour
#: as the f=0 one. Never both in one figure.
ROLLBACCINE_F1M_ARM = "rollbaccine-f1m"


@dataclass
class Cell:
    """One plotted point: the run behind it, and what it measured."""

    run: BenchRunMetrics

    @property
    def ops_per_s(self) -> float:
        return self.run.ops_per_s

    @property
    def marks(self) -> str:
        return self.run.marks()

    @property
    def threads(self) -> Optional[int]:
        return self.run.threads


def resolve_bench_dir(results_root: Path) -> Path:
    """The ``bench`` result tree under a results root.

    Accepts ``<root>/bench-result/bench``, ``<root>/bench`` or the bench
    directory itself.
    """
    for candidate in (
        results_root / "bench-result" / "bench",
        results_root / "bench",
        results_root,
    ):
        if candidate.is_dir() and any(p.is_dir() for p in candidate.glob("*/*/*")):
            return candidate
    raise SystemExit(
        f"No bench result tree under {results_root}: expected"
        " <root>/bench-result/bench/<config>/<arm>/<personality>"
    )


def load_runs(bench_dir: Path, config: Optional[str] = None) -> Tuple[str, List[Cell]]:
    """Gated runs of one config, and the config they belong to.

    Configs are never mixed: `config` picks one, and without it the one with
    the most runs wins, said out loud.
    """
    runs = bench_results.gate_bench_results(
        bench_results.discover_bench_results(bench_dir)
    )
    configs = sorted({r.config for r in runs})
    if not configs:
        raise SystemExit(f"No bench_client results under {bench_dir}")
    if config is None:
        config = max(configs, key=lambda c: sum(1 for r in runs if r.config == c))
        if len(configs) > 1:
            print(
                f"note: {len(configs)} configs under {bench_dir}"
                f" ({', '.join(configs)}); plotting {config}"
            )
    elif config not in configs:
        raise SystemExit(f"No results for config {config}; have {', '.join(configs)}")
    return config, [Cell(r) for r in runs if r.config == config]


def _pick(
    cells: Sequence[Cell], system: System, personality: str, threads: int
) -> Tuple[Optional[Cell], str]:
    """The run of one (system, personality, thread count), and why not.

    The thread count is the one the run resolved to, not what was asked for.
    """
    candidates = [
        c for c in cells if c.run.personality == personality and system.selects(c.run)
    ]
    if not candidates:
        return None, "no run"
    matching = [c for c in candidates if c.run.threads == threads]
    if not matching:
        found = sorted({str(c.run.threads) for c in candidates})
        return None, f"no run at {threads} threads (found {', '.join(found)})"
    if len(matching) > 1:
        newest = max(matching, key=lambda c: c.run.path.name)
        others = ", ".join(
            str(c.run.path.parent.name) for c in matching if c is not newest
        )
        warnings.warn(
            f"{personality}/{system.label} at {threads} threads has"
            f" {len(matching)} runs; taking {newest.run.path.name} over {others}"
        )
        return newest, ""
    return matching[0], ""


def _check_bounds(cells: Dict[str, Cell], what: str) -> None:
    """Warn when the systems compared in one group did not run the same bound."""
    bounds = {label: cell.run.bound for label, cell in cells.items() if cell.run.bound}
    if len(set(bounds.values())) > 1:
        detail = ", ".join(f"{label}: {bound!r}" for label, bound in bounds.items())
        warnings.warn(f"{what}: the systems did not run the same bound ({detail})")


def _pick_any(
    cells: Sequence[Cell], system: System, personality: str
) -> Tuple[Optional[Cell], str]:
    """The newest run of a personality whose native thread count is unknown."""
    candidates = [
        c for c in cells if c.run.personality == personality and system.selects(c.run)
    ]
    if not candidates:
        return None, "no run"
    return max(candidates, key=lambda c: c.run.path.name), ""


def _fmt_value(value: float) -> str:
    """A measured rate for the console table, to three significant digits."""
    for suffix, scale in (("M", 1e6), ("k", 1e3)):
        if value >= scale:
            return f"{value / scale:.2f}{suffix}"
    return f"{value:.1f}"
