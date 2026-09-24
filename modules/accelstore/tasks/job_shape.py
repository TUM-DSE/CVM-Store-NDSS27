#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""What a legacy-client job file asks of the store, read off its .txt.

The ``run-all`` counterpart of :mod:`personality_shape`: it recovers what a
chunk size is chosen against -- the object sizes written and any ranged reads.
Nothing is derived here, only checked (:func:`check_chunk_for_job`) and
described (:func:`describe`).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from modules.accelstore.tasks.personality_shape import (
    DEFAULT_CHUNK_BYTES,
    PersonalityShape,
    check_chunk,
    unaligned_penalty,
)

# Which client a job file is for, by the ops it uses, not the file-name prefix.
_CLIENT_OPS = {
    "kvs": {"INSERT", "INSERT_PATTERN", "UPDATE", "UPDATE_PATTERN"},
    "log": {"APPEND", "APPEND_PATTERN", "RETRIEVE", "TRUNCATE", "DUMP"},
    "fs": {"WRITE", "WRITE_PATTERN"},
}

# Ops whose last token is a byte count of the object written.
_SIZED_WRITE_OPS = {
    "INSERT_PATTERN",
    "UPDATE_PATTERN",
    "WRITE_PATTERN",
    "APPEND_PATTERN",
}


@dataclass
class JobShape:
    """The parts of a job file the harness has to act on."""

    client: str = "unknown"  # "unknown" when no op names a client
    ops: Dict[str, int] = field(default_factory=dict)
    write_sizes: List[int] = field(default_factory=list)  # distinct, ascending
    # One (length, align, line number) per FS `READ <f> <offset> <length>`.
    ranged_reads: List[Tuple[int, Optional[int], int]] = field(default_factory=list)
    open_ended_reads: List[int] = field(default_factory=list)  # line numbers

    @property
    def has_ranged_reads(self) -> bool:
        return bool(self.ranged_reads)

    @property
    def max_write_size(self) -> Optional[int]:
        return max(self.write_sizes) if self.write_sizes else None

    @property
    def min_write_size(self) -> Optional[int]:
        return min(self.write_sizes) if self.write_sizes else None

    def chunks_per_object(self, chunk: int = DEFAULT_CHUNK_BYTES) -> Optional[float]:
        """How many chunks the job's largest object spans at ``chunk``.

        Below 1 the chunk is a cap and the object moves un-pipelined.
        """
        largest = self.max_write_size
        return None if largest is None else largest / chunk

    def as_read_shape(self) -> PersonalityShape:
        """This job in the form :mod:`personality_shape`'s rules take.

        Job files declare no alignment, so every ranged read carries
        ``align=None``.
        """
        return PersonalityShape(
            meta_type=self.client,
            aligns={1: None},
            ranged_reads=list(self.ranged_reads),
            flowops=sorted(self.ops),
        )

    def as_dict(self) -> dict:
        """JSON-friendly form for a run's record of what it ran."""
        return {
            "client": self.client,
            "ops": dict(sorted(self.ops.items())),
            "write_sizes": self.write_sizes,
            "ranged_reads": [
                {"length": ln, "line": line} for ln, _al, line in self.ranged_reads
            ],
            "open_ended_reads": self.open_ended_reads,
        }


def parse_job(text: str) -> JobShape:
    """Parse a legacy job file into its shape.

    Deliberately lenient: an unknown op is counted, never an error.
    """
    shape = JobShape()
    sizes = set()
    seen_ops = set()

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        op = tokens[0]
        shape.ops[op] = shape.ops.get(op, 0) + 1
        seen_ops.add(op)

        if op in _SIZED_WRITE_OPS and len(tokens) >= 3:
            try:
                sizes.add(int(tokens[-1]))
            except ValueError:
                pass
        elif op == "READ" and len(tokens) == 4:
            # FS `READ <file> <offset> <length>` -- KVS READ takes a key alone.
            try:
                length = int(tokens[3])
            except ValueError:
                continue
            if length:
                shape.ranged_reads.append((length, None, lineno))
        elif op == "READ" and len(tokens) == 3:
            try:
                int(tokens[2])
            except ValueError:
                continue
            shape.open_ended_reads.append(lineno)

    for client, markers in _CLIENT_OPS.items():
        if seen_ops & markers:
            shape.client = client
            break

    shape.write_sizes = sorted(sizes)
    return shape


def check_chunk_for_job(shape: JobShape, chunk: int) -> Optional[str]:
    """Why ``chunk`` amplifies this job's reads, or None if it does not."""
    return check_chunk(shape.as_read_shape(), chunk)


def describe(shape: JobShape, chunk: Optional[int] = None) -> str:
    """One line on what this job asks of the store at ``chunk``.

    ``chunk`` of None means the library default.
    """
    effective = chunk or DEFAULT_CHUNK_BYTES
    largest = shape.max_write_size
    if largest is None:
        return f"{shape.client}: no sized writes (read-only or unseal job)"

    span = shape.chunks_per_object(effective)
    if span < 1:
        geometry = (
            f"object {largest} B < chunk {effective} B: one capped chunk per"
            " object, no intra-object pipelining"
        )
    else:
        geometry = f"object {largest} B = {span:.0f} x {effective} B chunks"
    if largest <= effective:
        fused = "fused single-I/O write applies"
    elif largest <= DEFAULT_CHUNK_BYTES:
        fused = (
            f"fused single-I/O write LOST at this chunk (applies at the"
            f" {DEFAULT_CHUNK_BYTES} B default)"
        )
    else:
        fused = "staged write at any chunk size"
    if shape.has_ranged_reads:
        reads = f"{len(shape.ranged_reads)} ranged read(s)"
        penalty = unaligned_penalty(shape.as_read_shape(), effective)
        if penalty:
            reads = f"{reads} -- {penalty}"
    else:
        reads = "whole-object reads only"
    return f"{shape.client}: {geometry}; {fused}; {reads}"
