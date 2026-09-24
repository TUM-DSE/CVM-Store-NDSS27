#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""What a bench_client personality asks of the store, read off its .wl file.

Parses the .wl file the run actually uses for the two things its name does not
carry: the read geometry that decides the crypto chunk size (:func:`derive_chunk`,
and :func:`derive_intra_op` for the axis that cuts one operation into N pieces)
and the small-object store sizing a whole-object personality wants
(:func:`derive_store_geometry`).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# AccelStore's bounds for --crypto-chunk-size (clients/client_common.c).
DEFAULT_CHUNK_BYTES = 1 << 20
MIN_CHUNK_BYTES = 4096

DEFAULT_SET = 1


@dataclass
class PersonalityShape:
    """The parts of a .wl file the harness has to act on."""

    meta_type: str = "fs"
    seed: Optional[int] = None
    threads: int = 1
    bound: Optional[str] = None
    # Fileset index (1-based, as the DSL's `set n`) -> declared align, or None.
    aligns: Dict[int, Optional[int]] = field(default_factory=dict)
    # One (length, align, line number) per FS ranged read: `read <h> <len>`.
    ranged_reads: List[Tuple[int, Optional[int], int]] = field(default_factory=list)
    flowops: List[str] = field(default_factory=list)
    # One fixed size per write/append flowop, None for a sampled `gamma` size.
    writes: List[Optional[int]] = field(default_factory=list)
    # Object writes per worker, None once any block is duration-bounded.
    write_ops: Optional[int] = 0
    # Whether any fileset declares a sampled size (`gamma`).
    fileset_sampled: bool = False

    @property
    def has_barrier(self) -> bool:
        """Whether any block places an `fsync` flowop."""
        return "fsync" in self.flowops

    @property
    def has_sampled_writes(self) -> bool:
        return self.fileset_sampled or any(w is None for w in self.writes)

    @property
    def fixed_writes(self) -> List[int]:
        return [w for w in self.writes if w]

    @property
    def max_write(self) -> Optional[int]:
        fixed = self.fixed_writes
        return max(fixed) if fixed else None

    @property
    def has_ranged_reads(self) -> bool:
        return bool(self.ranged_reads)

    @property
    def min_ranged_read(self) -> Optional[int]:
        return min(r[0] for r in self.ranged_reads) if self.ranged_reads else None

    @property
    def declared_aligns(self) -> List[int]:
        return sorted({a for a in self.aligns.values() if a})

    def as_dict(self) -> dict:
        """JSON-friendly form for the run's flags sidecar."""
        return {
            "type": self.meta_type,
            "seed": self.seed,
            "threads": self.threads,
            "bound": self.bound,
            "aligns": {str(k): v for k, v in sorted(self.aligns.items())},
            "ranged_reads": [
                {"length": ln, "align": al, "line": line}
                for ln, al, line in self.ranged_reads
            ],
            "flowops": sorted(set(self.flowops)),
            "writes": self.writes,
            "write_ops": self.write_ops,
            "has_barrier": self.has_barrier,
        }


def _int(token: str) -> Optional[int]:
    try:
        return int(token)
    except (TypeError, ValueError):
        return None


def parse_personality(text: str) -> PersonalityShape:
    """Parse the config lines and flowops of a .wl file.

    Lenient: bench_client's own parser is the authority, so an unknown keyword
    is recorded rather than refused.
    """
    shape = PersonalityShape()
    handle_set: Dict[str, int] = {}
    fileset_index = 0
    # (bound, writes) per run block, bound None = duration-bounded.
    blocks: List[Tuple[Optional[int], int]] = []
    block_bound: Optional[int] = None
    block_writes = 0
    in_block = False

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        head = tokens[0]

        if head == "type" and len(tokens) > 1:
            shape.meta_type = tokens[1]
        elif head == "seed" and len(tokens) > 1:
            shape.seed = _int(tokens[1])
        elif head == "threads" and len(tokens) > 1:
            shape.threads = _int(tokens[1]) or 1
        elif head in ("iterations", "duration") and len(tokens) > 1:
            shape.bound = f"{head} {tokens[1]}"
        elif head == "fileset":
            fileset_index += 1
            align = None
            if "align" in tokens:
                align = _int(tokens[tokens.index("align") + 1])
            shape.aligns[fileset_index] = align
            # A sampled size means the set's objects vary, whatever the writes say.
            if "gamma" in tokens or "maxsize" in tokens:
                shape.fileset_sampled = True
        elif head.startswith("loop"):
            if in_block:
                blocks.append((block_bound, block_writes))
            in_block = True
            block_writes = 0
            # `loop <n>:` iterations, `loop <t>s:` seconds, bare `loop:` the global
            operand = line[len("loop") :].strip().rstrip(":").strip()
            if not operand:
                global_bound = (shape.bound or "").split()
                block_bound = (
                    _int(global_bound[1])
                    if len(global_bound) == 2 and global_bound[0] == "iterations"
                    else None
                )
            else:
                block_bound = None if operand.endswith("s") else _int(operand)
        else:
            shape.flowops.append(head)
            set_index = DEFAULT_SET
            if "set" in tokens:
                set_index = _int(tokens[tokens.index("set") + 1]) or DEFAULT_SET

            if head in ("write", "append") and len(tokens) > 2:
                # `gamma` is a distribution rather than a size, recorded as None.
                shape.writes.append(None if tokens[2] == "gamma" else _int(tokens[2]))
                block_writes += 1

            if head in ("create", "open") and len(tokens) > 1:
                handle_set[tokens[1]] = set_index
            elif head == "read" and len(tokens) > 2 and shape.meta_type == "fs":
                # Only the FS `read <h> <len>` form is ranged, a LOG read has no length.
                length = _int(tokens[2])
                if length:
                    owner = handle_set.get(tokens[1], DEFAULT_SET)
                    shape.ranged_reads.append((length, shape.aligns.get(owner), lineno))

    if in_block:
        blocks.append((block_bound, block_writes))
    total = 0
    for bound, writes in blocks:
        if not writes:
            continue
        if bound is None:
            total = None  # a duration-bounded block writes an unknown count
            break
        total += bound * writes
    shape.write_ops = total

    return shape


def _largest_pow2_divisor(value: int) -> int:
    return value & -value


def _pow2_floor(value: int) -> int:
    return 1 << (value.bit_length() - 1)


def derive_chunk(shape: PersonalityShape) -> Tuple[Optional[int], str]:
    """The crypto chunk size this personality should run at, and why.

    Returns ``(chunk_bytes, reason)``, with None when the personality does not
    constrain the chunk and the library default applies.
    """
    if not shape.has_ranged_reads:
        return (
            None,
            "no ranged read in the personality: every read starts at offset 0 and "
            "runs to the end of its object, so a chunk larger than the object is a "
            "cap and not padding - the library default applies",
        )

    limit = DEFAULT_CHUNK_BYTES
    for length, align, _lineno in shape.ranged_reads:
        limit = min(limit, _pow2_floor(length))
        # No align means byte-exact offsets, so nothing above the floor is aligned.
        limit = min(limit, _largest_pow2_divisor(align) if align else MIN_CHUNK_BYTES)

    chunk = max(MIN_CHUNK_BYTES, limit)
    return (
        chunk,
        f"largest chunk with chunk <= the smallest ranged read "
        f"({shape.min_ranged_read} B) and align % chunk == 0 (declared align "
        f"{shape.declared_aligns or 'none'}), capped at the {DEFAULT_CHUNK_BYTES} B "
        f"library default and floored at {MIN_CHUNK_BYTES} B",
    )


def check_chunk(shape: PersonalityShape, chunk: int) -> Optional[str]:
    """Why ``chunk`` amplifies this personality's reads, or None if it does not."""
    if chunk < MIN_CHUNK_BYTES:
        return f"chunk size {chunk} is below AccelStore's {MIN_CHUNK_BYTES} B minimum"
    problems = {}
    for length, align, _lineno in shape.ranged_reads:
        if chunk > length:
            problems[(length, align)] = (
                f"a {length} B ranged read reads a whole {chunk} B chunk "
                f"({chunk / length:.1f}x amplification at best)"
            )
        elif align is not None and align % chunk:
            problems[(length, align)] = (
                f"offsets are multiples of {align} B, which is not a multiple of the "
                f"{chunk} B chunk, so a {length} B read straddles two chunks "
                "(2x amplification)"
            )
    # No declared alignment is deliberately not a problem here, see unaligned_penalty.
    return "; ".join(problems.values()) if problems else None


def unaligned_penalty(shape: PersonalityShape, chunk: int) -> Optional[str]:
    """What the ranged reads with no declared alignment cost at ``chunk``.

    Advisory, not a refusal: a byte-exact offset puts at most one extra chunk
    on a read.
    """
    worst = None
    for length, align, _lineno in shape.ranged_reads:
        if align is not None or chunk > length:
            continue
        factor = (length + chunk) / length
        if worst is None or factor > worst[0]:
            worst = (factor, length)
    if worst is None:
        return None
    factor, length = worst
    return (
        f"offsets are not declared aligned, so a {length} B read can straddle one"
        f" extra {chunk} B chunk ({factor:.2f}x at worst)"
    )


# bench_client's bound on the POSIX intra-op depth (BENCH_POSIX_IO_DEPTH_MAX).
MAX_INTRA_OP_DEPTH = 1024


@dataclass
class IntraOpSplit:
    """How one operation is cut up, and the flags that say so on each arm.

    ``depth`` is what the personality resolved to, ``requested`` what was asked
    for: the two differ where no depth is reachable.
    """

    requested: int
    depth: int = 1
    # --crypto-chunk-size for the accelstore arm, None when the axis is a no-op.
    chunk_size: Optional[int] = None
    io_chunk: Optional[int] = None
    io_depth: Optional[int] = None
    reason: str = ""

    @property
    def is_identity(self) -> bool:
        """Whether no flag reaches either arm."""
        return self.depth <= 1

    @property
    def collapsed(self) -> bool:
        """Whether the personality could not give the depth that was asked for."""
        return self.depth != self.requested

    def as_dict(self) -> dict:
        return {
            "requested": self.requested,
            "depth": self.depth,
            "chunk_size": self.chunk_size,
            "io_chunk": self.io_chunk,
            "io_depth": self.io_depth,
            "reason": self.reason,
        }


def derive_intra_op(shape: Optional[PersonalityShape], requested: int) -> IntraOpSplit:
    """Split one operation ``requested`` ways, on both arms, or say why not.

    ``requested == 1`` is the identity. A personality the axis cannot apply to
    resolves to the identity with a reason rather than raising.
    """
    requested = int(requested)
    if not 1 <= requested <= MAX_INTRA_OP_DEPTH:
        raise ValueError(
            f"intra-op depth {requested} is outside [1, {MAX_INTRA_OP_DEPTH}]"
            " (1 = one crypto chunk per ranged read, i.e. the derived geometry)"
        )
    if requested == 1:
        return IntraOpSplit(
            requested,
            reason="depth 1: one chunk per ranged read, the derived geometry",
        )
    if shape is None:
        return IntraOpSplit(
            requested,
            reason=(
                "the personality could not be read, so there is no derived chunk"
                " to divide and the run keeps the library's geometry"
            ),
        )

    base, _base_reason = derive_chunk(shape)
    if base is None:
        return IntraOpSplit(
            requested,
            reason=(
                "personality has no ranged read, so nothing derives a chunk and"
                " there is nothing to divide: a whole-object read already spans"
                " as many chunks as the object holds, and shrinking the chunk"
                " below the object would cost it AccelStore's fused single-I/O"
                " path for a split the POSIX arm's --io-chunk has no"
                " matching operation size to be paired against"
            ),
        )

    # Rounded down to a power of two, which keeps `align % chunk == 0` by construction.
    chunk = max(MIN_CHUNK_BYTES, base // _pow2_floor(requested))
    depth = base // chunk
    if depth <= 1:
        return IntraOpSplit(
            requested,
            reason=(
                f"the derived chunk is already {base} B, AccelStore's"
                f" {MIN_CHUNK_BYTES} B minimum, so a ranged read here is one chunk"
                " at every depth and the run is the depth-1 run"
            ),
        )

    reason = (
        f"depth {depth}: the derived {base} B chunk cut {depth} ways gives"
        f" {chunk} B, so the accelstore arm runs --crypto-chunk-size {chunk}"
        f" (a ranged read becomes {depth} chunk I/Os it decrypts as they land)"
        f" and the POSIX arm --io-chunk {chunk} --io-depth {depth} (the same cut,"
        f" {depth} segments in flight on the worker's ring)"
    )
    if depth != requested:
        reason += (
            f"; asked for {requested}, floored to a power of two at or below it"
            f" and at AccelStore's {MIN_CHUNK_BYTES} B chunk minimum"
        )
    return IntraOpSplit(
        requested,
        depth,
        chunk_size=chunk,
        io_chunk=chunk,
        io_depth=depth,
        reason=reason,
    )


# Where an AEAD blob's data region starts (BLOB_AEAD_DATA_ALIGN_DEFAULT).
DATA_ALIGN_BYTES = 16 * 1024

# An FS write creates the data blob and the file's Merkle blob, a LOG append one.
BLOBS_PER_OBJECT = {"fs": 2, "log": 1}

MD_PAGE_HEADROOM = 1.25


@dataclass
class StoreGeometry:
    """The chunk/cluster/md-page sizing a small-object personality asks for."""

    chunk_size: int
    cluster_size: int
    md_pages: Optional[int]
    reason: str

    def as_dict(self) -> dict:
        return {
            "chunk_size": self.chunk_size,
            "cluster_size": self.cluster_size,
            "md_pages": self.md_pages,
            "reason": self.reason,
        }


def derive_store_geometry(
    shape: PersonalityShape,
    threads: Optional[int] = None,
    split_loops: bool = False,
) -> Tuple[Optional[StoreGeometry], str]:
    """The small-object geometry for this personality, or None with the reason.

    Narrow by design: a personality that fails any condition below is left at
    the library defaults rather than given a geometry the zero-waste argument
    does not cover. ``split_loops`` is bench_client's ``--split-loops``, where
    an iteration bound is the run's total rather than a per-worker count.
    """
    if shape.has_ranged_reads:
        return None, (
            "personality has ranged reads: its chunk is set by the read geometry"
            " (derive_chunk), and a 16 KiB cluster under a large file would put"
            " many fragments on every chunk"
        )
    if shape.has_sampled_writes:
        return None, (
            "personality samples its object sizes (`gamma`), so there is no one"
            " object size to size the chunk to, and the blob would not tile the"
            " cluster exactly"
        )
    largest = shape.max_write
    if not largest:
        return None, "personality declares no fixed object write"
    if largest > DEFAULT_CHUNK_BYTES:
        return None, (
            f"largest object {largest} B exceeds the {DEFAULT_CHUNK_BYTES} B"
            " default chunk: the fused path is already out of reach and the"
            " default cluster is the right one"
        )
    # Every declared size, not just the largest: the cluster has to tile each blob.
    ragged = sorted({w for w in shape.fixed_writes if w % DATA_ALIGN_BYTES})
    if ragged:
        return None, (
            f"object sizes {ragged} are not multiples of {DATA_ALIGN_BYTES} B, so a"
            " 16 KiB cluster would not tile their blobs without waste"
        )

    # `write_ops` counts one worker's writes, unless the bounds were split.
    workers = 1 if split_loops else (threads or shape.threads)

    md_pages = None
    if shape.write_ops:
        blobs = shape.write_ops * BLOBS_PER_OBJECT.get(shape.meta_type, 1)
        md_pages = int(blobs * workers * MD_PAGE_HEADROOM)

    return (
        StoreGeometry(
            chunk_size=largest,
            cluster_size=DATA_ALIGN_BYTES,
            md_pages=md_pages,
            reason=(
                f"largest fixed object {largest} B: chunk = the object (smallest"
                " that still fuses), cluster = 16 KiB (the data-region start, so"
                f" the {DATA_ALIGN_BYTES + largest} B blob tiles it exactly and the"
                " Merkle blob costs one cluster)"
                + (
                    f", md_pages = {md_pages} for"
                    f" {shape.write_ops} x {BLOBS_PER_OBJECT.get(shape.meta_type, 1)}"
                    f" blobs x {workers}"
                    f" {'thread (bounds split across workers)' if split_loops else 'threads'}"
                    f" x {MD_PAGE_HEADROOM}"
                    if md_pages
                    else ", md_pages left unset: the object count is not bounded"
                    " by the personality"
                )
            ),
        ),
        "derived",
    )


def format_bytes(value: int) -> str:
    """Compact byte size for a result-directory token: 4096 -> 4k, 2**20 -> 1m."""
    for unit, size in (("g", 1 << 30), ("m", 1 << 20), ("k", 1 << 10)):
        if value >= size and value % size == 0:
            return f"{value // size}{unit}"
    return str(value)
