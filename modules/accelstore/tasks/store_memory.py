"""How much hugepage memory an AccelStore store needs.

Derived from the store geometry by mirroring the iobuf pool sizing in
`lib/util/as_iobuf.c`, scaled by a margin for what that derivation misses.
"""

import math

MIB = 1024 * 1024
GIB = 1024 * MIB

# include/accelstore.h
CHUNK_SIZE_DEFAULT = 1 * MIB
PIPELINE_BYTES_DEFAULT = 64 * MIB
CHUNK_IO_DEPTH_DEFAULT = 64
CRYPTO_WINDOW_MAX = 1024
# include/as_iobuf.h
IOBUF_SLACK = 16
IOBUF_THREAD_SMALL_CACHE = 256
IOBUF_THREAD_LARGE_CACHE = 32
# SPDK's defaults, which the untuned size class keeps (lib/thread/iobuf.c)
SPDK_SMALL_BUFSIZE = 8192
SPDK_LARGE_BUFSIZE = 132 * 1024
SPDK_SMALL_POOL_COUNT = 8192
SPDK_LARGE_POOL_COUNT = 1024

SAFETY_MARGIN = 1.5
FIXED_OVERHEAD_GB = 2
CEILING_GB = 32
GUEST_FRACTION = 0.5


def crypto_window(pipeline_bytes: int, command_bytes: int) -> int:
    """`as_crypto_default_depth()` (include/accelstore.h), in Python."""
    pipeline_bytes = pipeline_bytes or PIPELINE_BYTES_DEFAULT
    by_budget = pipeline_bytes // (command_bytes or 1)
    return max(1, min(by_budget, CHUNK_IO_DEPTH_DEFAULT, CRYPTO_WINDOW_MAX))


def iobuf_pool_bytes(
    workers: int,
    chunk_size: int = CHUNK_SIZE_DEFAULT,
    pipeline_bytes: int = PIPELINE_BYTES_DEFAULT,
) -> int:
    """Bytes the two iobuf size classes occupy, as `as_iobuf.c` derives them."""
    chunk_size = chunk_size or CHUNK_SIZE_DEFAULT
    pipeline_bytes = pipeline_bytes or PIPELINE_BYTES_DEFAULT
    threads = max(1, workers) + 1  # +1: the internal recovery/metadata operation

    need = -(-chunk_size // 4096) * 4096
    window = crypto_window(pipeline_bytes, need)
    depth = threads * window + IOBUF_SLACK
    small_channels = threads * IOBUF_THREAD_SMALL_CACHE
    large_channels = threads * IOBUF_THREAD_LARGE_CACHE

    if need <= SPDK_LARGE_BUFSIZE:
        # The chunks land in the small class, so that is the one resized.
        want = depth + small_channels
        if SPDK_SMALL_BUFSIZE < need:
            small_bufsize, small_count = need, want
        else:
            small_bufsize = SPDK_SMALL_BUFSIZE
            small_count = max(SPDK_SMALL_POOL_COUNT, want)
        large_bufsize = SPDK_LARGE_BUFSIZE
        large_count = max(SPDK_LARGE_POOL_COUNT, large_channels)
    else:
        large_bufsize, large_count = need, depth + large_channels
        small_bufsize = SPDK_SMALL_BUFSIZE
        small_count = max(SPDK_SMALL_POOL_COUNT, small_channels)

    return small_bufsize * small_count + large_bufsize * large_count


def required_gb(
    workers: int,
    chunk_size: int = CHUNK_SIZE_DEFAULT,
    pipeline_bytes: int = PIPELINE_BYTES_DEFAULT,
) -> int:
    """Hugepages this store geometry needs, pools plus margin and overhead."""
    pools = iobuf_pool_bytes(workers, chunk_size, pipeline_bytes)
    return math.ceil(pools * SAFETY_MARGIN / GIB) + FIXED_OVERHEAD_GB


def hugepages_gb(
    guest_memory_gb: float,
    workers: int,
    chunk_size: int = CHUNK_SIZE_DEFAULT,
    pipeline_bytes: int = PIPELINE_BYTES_DEFAULT,
    requested: int = None,
) -> int:
    """The reservation to pass `spdk_device`, or raise saying why there is none.

    Without ``requested``: the ceiling where the guest can spare it, its share
    where it cannot, never below :func:`required_gb`.
    """
    need = required_gb(workers, chunk_size, pipeline_bytes)
    share = int(guest_memory_gb * GUEST_FRACTION)

    if requested is not None:
        if requested < need:
            raise ValueError(
                f"--hugepages-gb {requested} is below the {need} GB this store "
                f"geometry needs ({workers} workers, "
                f"{chunk_size // MIB} MiB chunks, "
                f"{pipeline_bytes // MIB} MiB pipeline); SPDK would fail its "
                f"channel allocation inside the run instead"
            )
        if requested >= guest_memory_gb:
            raise ValueError(
                f"--hugepages-gb {requested} does not fit a "
                f"{guest_memory_gb:g} GB guest"
            )
        return requested

    reserve = min(CEILING_GB, share)
    if reserve < need:
        raise ValueError(
            f"a {guest_memory_gb:g} GB guest can spare {share} GB of hugepages "
            f"and this store geometry needs {need} ({workers} workers, "
            f"{chunk_size // MIB} MiB chunks, {pipeline_bytes // MIB} MiB "
            f"pipeline). Run it at a larger --sizes, or lower --workers / "
            f"--pipeline-bytes / --crypto-chunk-size"
        )
    return reserve
