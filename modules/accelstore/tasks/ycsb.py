#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The YCSB grid: the points a sweep can run and the flags each one derives.

Pure -- nothing here imports ``core`` or ``invoke`` -- so the action, the sweep,
the plotter and the tests read the same table. Three families: ``grid`` (mixes
B/C/F over the small value sizes at a fixed op count) plus ``small`` and
``large``, the two halves of workload A's size sweep, which hold a byte volume
instead. ``large`` runs on the library's store geometry, the other two on the
small-object one.
"""

from dataclasses import dataclass, replace
from typing import Optional

from modules.accelstore.tasks.block_stacks import BYTES_PER_GB, FILESYSTEM_HEADROOM

ACCELSTORE_ARM = "accelstore"

# ycsb_client implements A/B/C/F only: D and E are absent upstream.
YCSB_MIXES = ("A", "B", "C", "F")

MIX_WRITE_SHARE = {"A": 0.5, "B": 0.05, "C": 0.0, "F": 0.5}

AXIS_MIX = "A"

GRID_MIXES = tuple(mix for mix in YCSB_MIXES if mix != AXIS_MIX)

YCSB_VALUE_SIZES = (4096, 16384, 65536, 262144, 1048576)

LARGE_VALUE_SIZES = (64 << 20, 256 << 20, 1 << 30)

LARGE_TOTAL_BYTES = 256 << 30

DEFAULT_CLUSTER_BYTES = 1 << 20

# Library cap on --crypto-chunk-size (clients/client_common.c), not enforced here.
MAX_CHUNK_BYTES = 1 << 30

ROCKSDB_MAX_VALUE_BYTES = (1 << 32) - 1

DEFAULT_NUM_OPS = 100_000

MIX_A_TARGET_DIRECTION_BYTES = 128 * BYTES_PER_GB

DEFAULT_NUM_KEYS = 10_000

DATA_ALIGN_BYTES = 16 * 1024  # BLOB_AEAD_DATA_ALIGN_DEFAULT, lib/util/blob_aead.h
CLUSTER_BYTES = DATA_ALIGN_BYTES  # the same number: it tiles a 16 KiB + value blob

MIN_CHUNK_BYTES = 4096

MD_PAGE_HEADROOM = 1.25

MD_PAGES_FLOOR = 8192

ROCKSDB_SPACE_AMPLIFICATION = 2.0


def mix_a_num_ops(
    value_size: int, target_direction_bytes: int = MIX_A_TARGET_DIRECTION_BYTES
) -> int:
    """Op count whose mix-A run phase moves ``target_direction_bytes`` each way."""
    target = 2 * target_direction_bytes
    return (target + value_size // 2) // value_size


def format_bytes(value: int) -> str:
    """Compact byte size for a point name or a path token: 4096 -> 4k."""
    for unit, size in (("g", 1 << 30), ("m", 1 << 20), ("k", 1 << 10)):
        if value >= size and value % size == 0:
            return f"{value // size}{unit}"
    return str(value)


GEOMETRY_SMALL_OBJECT = "small-object"
GEOMETRY_LIBRARY_DEFAULT = "library-default"

FAMILY_GRID = "grid"
FAMILY_SMALL = "small"
FAMILY_LARGE = "large"
FAMILIES = (FAMILY_GRID, FAMILY_SMALL, FAMILY_LARGE)

FAMILY_GEOMETRY = {
    FAMILY_GRID: GEOMETRY_SMALL_OBJECT,
    FAMILY_SMALL: GEOMETRY_SMALL_OBJECT,
    FAMILY_LARGE: GEOMETRY_LIBRARY_DEFAULT,
}


@dataclass(frozen=True)
class YcsbPoint:
    """One (mix, value size) point of the grid, with its op and key counts."""

    name: str
    mix: str
    value_size: int
    num_keys: int = DEFAULT_NUM_KEYS
    num_ops: int = DEFAULT_NUM_OPS
    family: str = FAMILY_GRID

    def __post_init__(self) -> None:
        if self.mix not in YCSB_MIXES:
            raise ValueError(
                f"Unknown YCSB mix '{self.mix}'. Known: {', '.join(YCSB_MIXES)}"
            )
        if self.family not in FAMILIES:
            raise ValueError(
                f"{self.name}: unknown family '{self.family}'. "
                f"Known: {', '.join(FAMILIES)}"
            )
        for field_name in ("value_size", "num_keys", "num_ops"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{self.name}: {field_name} must be a positive int")

    def with_overrides(self, value_size=None, num_keys=None, num_ops=None):
        """This point with whichever of the three a run replaced, name kept."""
        changes = {
            key: int(value)
            for key, value in (
                ("value_size", value_size),
                ("num_keys", num_keys),
                ("num_ops", num_ops),
            )
            if value is not None
        }
        return replace(self, **changes) if changes else self

    @property
    def geometry(self) -> str:
        """The store geometry this point's family runs on."""
        return FAMILY_GEOMETRY[self.family]

    @property
    def write_ops(self) -> int:
        """Run-phase operations that write a value, from the mix's own split."""
        return int(self.num_ops * MIX_WRITE_SHARE[self.mix])

    def workload_args(self) -> list:
        """The four flags that define the workload, on either backend."""
        return [
            "--ycsb",
            self.mix,
            "--value-size",
            str(self.value_size),
            "--num-keys",
            str(self.num_keys),
            "--num-ops",
            str(self.num_ops),
        ]

    @property
    def chunk_size(self):
        """``--crypto-chunk-size``, or None under the library-default geometry."""
        if self.geometry == GEOMETRY_LIBRARY_DEFAULT:
            return None
        return max(MIN_CHUNK_BYTES, self.value_size)

    @property
    def cluster_size(self) -> int:
        """The blobstore cluster this point's store is created with."""
        if self.geometry == GEOMETRY_LIBRARY_DEFAULT:
            return DEFAULT_CLUSTER_BYTES
        return CLUSTER_BYTES

    @property
    def blob_bytes(self) -> int:
        """On-device bytes one key occupies: its blob, rounded to the cluster."""
        raw = DATA_ALIGN_BYTES + self.value_size
        cluster = self.cluster_size
        return -(-raw // cluster) * cluster

    @property
    def live_blobs(self) -> int:
        """Blobs held at once, worst case: one per key plus one per mutating op."""
        return self.num_keys + self.write_ops

    @property
    def md_pages(self):
        """``--md-pages``, or None to leave the library's reservation."""
        if self.geometry == GEOMETRY_LIBRARY_DEFAULT:
            return None
        return max(MD_PAGES_FLOOR, int(self.live_blobs * MD_PAGE_HEADROOM))

    @property
    def store_bytes(self) -> int:
        """The live store the load phase leaves on an AccelStore device."""
        return self.num_keys * self.blob_bytes

    def as_dict(self) -> dict:
        """The point and everything derived from it, for a flags sidecar."""
        return {
            "name": self.name,
            "mix": self.mix,
            "value_size": self.value_size,
            "num_keys": self.num_keys,
            "num_ops": self.num_ops,
            "write_ops": self.write_ops,
            "family": self.family,
            "geometry": self.geometry,
            "chunk_size": self.chunk_size,
            "cluster_size": self.cluster_size,
            "md_pages": self.md_pages,
            "blob_bytes": self.blob_bytes,
            "store_bytes": self.store_bytes,
            "store_footprint_gb": round(store_footprint_gb(self), 3),
            "required_capacity_gb": required_capacity_gb(self),
        }


def _small_point(size: int) -> YcsbPoint:
    """One point of the A sweep's 4 KiB -> 1 MiB half: fixed keys, scaled ops."""
    return YcsbPoint(
        name=f"{AXIS_MIX}_{format_bytes(size)}",
        mix=AXIS_MIX,
        value_size=size,
        num_keys=DEFAULT_NUM_KEYS,
        num_ops=mix_a_num_ops(size),
        family=FAMILY_SMALL,
    )


def _large_point(size: int) -> YcsbPoint:
    """One point of the A sweep's 64 MiB -> 1 GiB half: keys = 256 GiB / size."""
    keys = LARGE_TOTAL_BYTES // size
    return YcsbPoint(
        name=f"{AXIS_MIX}_{format_bytes(size)}",
        mix=AXIS_MIX,
        value_size=size,
        num_keys=keys,
        num_ops=keys,
        family=FAMILY_LARGE,
    )


def with_byte_targets(
    point: "YcsbPoint",
    direction_bytes: Optional[int] = None,
    large_total_bytes: Optional[int] = None,
) -> "YcsbPoint":
    """`point` re-derived at reduced byte targets: ops for the small half, keys
    for the large one. A grid point holds its op count and is returned as is.
    """
    if point.family == FAMILY_SMALL and direction_bytes:
        return point.with_overrides(
            num_ops=mix_a_num_ops(point.value_size, direction_bytes)
        )
    if point.family == FAMILY_LARGE and large_total_bytes:
        keys = large_total_bytes // point.value_size
        return point.with_overrides(num_keys=keys, num_ops=keys)
    return point


def _build_points() -> dict:
    """The three families in sweep order, the A sweep first and small to large."""
    points = {}
    for size in YCSB_VALUE_SIZES:
        point = _small_point(size)
        points[point.name] = point
    for size in LARGE_VALUE_SIZES:
        point = _large_point(size)
        points[point.name] = point
    for mix in GRID_MIXES:
        for size in YCSB_VALUE_SIZES:
            point = YcsbPoint(
                name=f"{mix}_{format_bytes(size)}",
                mix=mix,
                value_size=size,
                family=FAMILY_GRID,
            )
            points[point.name] = point
    return points


YCSB_POINTS = _build_points()


def family_points(family: str) -> tuple:
    """The names of one family, in the table's own order."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown family '{family}'. Known: {', '.join(FAMILIES)}")
    return tuple(p.name for p in YCSB_POINTS.values() if p.family == family)


GRID_POINTS = family_points(FAMILY_GRID)
SMALL_POINTS = family_points(FAMILY_SMALL)
LARGE_POINTS = family_points(FAMILY_LARGE)

AXIS_POINTS = SMALL_POINTS + LARGE_POINTS

YCSB_DEFAULT_POINTS = tuple(YCSB_POINTS)


def ycsb_point(name) -> YcsbPoint:
    """Resolve a point name (or a point) against the grid."""
    if isinstance(name, YcsbPoint):
        return name
    if name not in YCSB_POINTS:
        raise ValueError(
            f"Unknown YCSB point '{name}'. Known: {', '.join(YCSB_POINTS)}"
        )
    return YCSB_POINTS[name]


def parse_points(spec) -> list:
    """A ``--points`` argument: point names, a mix letter, a family name
    (``grid``/``small``/``large``) or "all". ``A`` is both halves of the sweep.
    """
    tokens = [t.strip() for t in (spec or "all").split(",") if t.strip()]
    if tokens == ["all"]:
        return list(YCSB_DEFAULT_POINTS)
    names = []
    for token in tokens:
        if token.upper() in YCSB_MIXES:
            matched = [n for n, p in YCSB_POINTS.items() if p.mix == token.upper()]
        elif token.lower() in FAMILIES:
            matched = list(family_points(token.lower()))
        else:
            matched = [ycsb_point(token).name]
        for name in matched:
            if name not in names:
                names.append(name)
    return names


def accelstore_point_args(point) -> list:
    """The store-geometry flags the accelstore arm needs for this point.

    Chunk, cluster and md-pages are one derivation and are passed together. A
    ``large``-family point takes the library's geometry and gets none of them.
    """
    point = ycsb_point(point)
    if point.geometry == GEOMETRY_LIBRARY_DEFAULT:
        return []
    return [
        "--crypto-chunk-size",
        str(point.chunk_size),
        "--cluster-size",
        str(point.cluster_size),
        "--md-pages",
        str(point.md_pages),
    ]


def rocksdb_rejects(point):
    """Why RocksDB cannot run this point (a value above uint32 max), or None."""
    point = ycsb_point(point)
    if point.value_size > ROCKSDB_MAX_VALUE_BYTES:
        return (
            f"{point.name}: RocksDB refuses a value above "
            f"{ROCKSDB_MAX_VALUE_BYTES} bytes ('value is too large', "
            f"db/write_batch.cc) and this point's is {point.value_size}. "
            "Run this point on the accelstore arm alone (--arms accelstore), "
            "or drop it from the sweep"
        )
    return None


def client_buffer_bytes(point, threads=1) -> int:
    """Guest RAM ycsb_client mlocks here: one pattern buffer plus one per worker."""
    return ycsb_point(point).value_size * (int(threads) + 1)


def store_footprint_gb(point) -> float:
    """The live store one point leaves behind, in GB (10^9, as everywhere)."""
    return ycsb_point(point).store_bytes / BYTES_PER_GB


def required_capacity_gb(point) -> int:
    """The window a RocksDB arm of this point has to be given, in GB, rounded up."""
    bytes_needed = (
        ycsb_point(point).store_bytes
        * ROCKSDB_SPACE_AMPLIFICATION
        * FILESYSTEM_HEADROOM
    )
    return max(1, -(-int(bytes_needed) // BYTES_PER_GB))


def sweep_capacity_needed_gb(points) -> int:
    """The largest window any point of a selection needs."""
    return max((required_capacity_gb(p) for p in points), default=0)


def ycsb_arms(block_stacks) -> tuple:
    """The arms a YCSB sweep can run: the accelstore arm plus the block stacks."""
    return (ACCELSTORE_ARM, *block_stacks)
