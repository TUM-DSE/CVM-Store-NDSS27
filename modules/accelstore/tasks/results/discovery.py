"""Shared discovery helpers for reading a ``bench-result/`` tree.

The config/arm/point walk, the trace dump's pairing with its ``_parsed.json``
cache, and the ``*_flags.json`` sidecar read. What a run is inside a leaf stays
with each parser. Read-side only.
"""

import json
import warnings
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

# A run counts as present when either the raw dump or the parsed-metrics cache
# exists, so an exported tree without its dumps still analyses.
TRACE_DUMP_SUFFIX = "_trace.dump.json"
TRACE_CACHE_SUFFIX = "_trace.dump_parsed.json"
TRACE_SUFFIXES = (TRACE_CACHE_SUFFIX, TRACE_DUMP_SUFFIX)


def trace_stem(name: str) -> Optional[str]:
    """The run stem of a trace dump's or cache's name, or None otherwise."""
    for suffix in TRACE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def trace_dump_path(directory: Path, stem: str) -> Path:
    """Where a run's trace dump lives, whether or not it is on disk."""
    return directory / f"{stem}{TRACE_DUMP_SUFFIX}"


def trace_dump_if_present(directory: Path, stem: str) -> Optional[Path]:
    """The run's dump path, returned whenever the dump or its cache exists.

    Always the dump's path, because ``load_or_parse_trace`` derives the cache's
    name from it.
    """
    dump = trace_dump_path(directory, stem)
    if dump.exists() or (directory / f"{stem}{TRACE_CACHE_SUFFIX}").exists():
        return dump
    return None


def subdirs(path: Path) -> List[Path]:
    """The subdirectories of ``path``, sorted by name."""
    return sorted(p for p in path.glob("*") if p.is_dir())


def iter_point_leaves(
    root: Path,
) -> Iterator[Tuple[str, str, str, Optional[str], Path]]:
    """The leaves of a ``{config}/{arm}/{point}`` result tree, in walk order.

    Yields ``(config, arm, point, variant, leaf_dir)``: the point directory as
    the default-mode leaf (``variant`` None), each subdirectory as a variant
    leaf, except ``precondition``. Empty leaves are yielded too.
    """
    for config_dir in subdirs(root):
        for arm_dir in subdirs(config_dir):
            for point_dir in subdirs(arm_dir):
                yield (config_dir.name, arm_dir.name, point_dir.name, None, point_dir)
                for d in sorted(point_dir.glob("*")):
                    if d.is_dir() and d.name != "precondition":
                        yield (
                            config_dir.name,
                            arm_dir.name,
                            point_dir.name,
                            d.name,
                            d,
                        )


def flags_record(run_path: Path) -> Optional[dict]:
    """The raw ``*_flags.json`` record next to a run's log or metrics file.

    None for a run without a readable sidecar, so an old or partial tree still
    parses.
    """
    sidecar = run_path.with_name(f"{run_path.stem}_flags.json")
    if not sidecar.exists():
        return None
    try:
        return json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(f"Cannot read {sidecar}: {exc}")
        return None


def resolve_result_tree(root: Path, name: str) -> Path:
    """The ``{name}`` result tree under a results root, however it was named.

    Accepts ``<root>/bench-result/{name}`` and ``<root>/{name}``, and otherwise
    walks ``root`` itself.
    """
    for candidate in (root / "bench-result" / name, root / name):
        if candidate.is_dir():
            return candidate
    return root


def find_latest_traces(scan_dir: Path, count: int = 1) -> List[Path]:
    """Up to `count` most recent trace runs in a directory, newest first.

    Dump and cache deduplicate to one run, and the filename sort is the run
    order because names are timestamp-prefixed. Fewer runs than asked for
    returns the ones present.
    """
    all_bases = set()
    for p in scan_dir.glob(f"*{TRACE_CACHE_SUFFIX}"):
        all_bases.add(trace_stem(p.name) + TRACE_DUMP_SUFFIX)
    for p in scan_dir.glob(f"*{TRACE_DUMP_SUFFIX}"):
        all_bases.add(p.name)

    newest = sorted(all_bases, reverse=True)[: max(1, count)]
    return [scan_dir / base for base in newest]
