#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chunk-pipeline geometry of an AccelStore client run.

The clients expose the chunked I/O engine (``lib/util/chunk_pipeline.c``) as
flags. The knobs are not independent: the chunk size alone also moves the
command depth, the per-buffer DMA size and whether the fused single-I/O write
path applies, so a chunk arm needs the others pinned to isolate it. Bounds
mirror ``parse_common_arguments`` (``clients/client_common.c``).
"""

from dataclasses import dataclass
from typing import Optional

# Mirrors clients/client_common.c.
MIN_CHUNK_BYTES = 4096
MAX_CHUNK_BYTES = 1024**3
MAX_PIPELINE_BYTES = 4 * 1024**3
MIN_BURST_BYTES = 4096
MAX_BURST_BYTES = 4 * 1024**3
MIN_SPAN_BYTES = 4096
MAX_SPAN_BYTES = 1024**3
WINDOW_MAX = 1024
CLUSTER_MULTIPLE = 4096
MAX_CLUSTER_BYTES = 2**32 - 1
# uint32_t in both consumers, and only rejected after the device is erased.
MAX_MD_PAGES = 2**32 - 1

# The library's own defaults (include/accelstore.h, include/bs_manager.h).
DEFAULT_CHUNK_BYTES = 1 << 20
DEFAULT_PIPELINE_BYTES = 64 << 20
DEFAULT_CLUSTER_BYTES = 1 << 20

# Per-thread, AS_BS_MAX_CHANNEL_OPS - 64 (lib/util/chunk_pipeline.c).
FRAGMENT_BUDGET = 1024

BURST_MAX_TOKEN = "max"
SLUG_PREFIX = "pipe-"


def _format_bytes(value: int) -> str:
    """Compact byte size for a result-directory token: 4096 -> 4k, 2**20 -> 1m."""
    for unit, size in (("g", 1 << 30), ("m", 1 << 20), ("k", 1 << 10)):
        if value >= size and value % size == 0:
            return f"{value // size}{unit}"
    return str(value)


def _as_int(name: str, value) -> Optional[int]:
    """Coerce a task argument, which invoke may hand over as a string."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name}: '{value}' is not a byte/chunk count") from None


@dataclass(frozen=True)
class PipelineOpts:
    """The chunk-pipeline geometry of one run.

    Every field is ``None`` for "leave the library's default": no flag on the
    command line, no component in the result path.
    """

    chunk_size: Optional[int] = None
    pipeline_bytes: Optional[int] = None
    io_depth: Optional[int] = None
    reorder_window: Optional[int] = None
    burst_bytes: Optional[str] = None
    burst_adaptive: bool = False
    io_span_bytes: Optional[int] = None
    cluster_size: Optional[int] = None
    md_pages: Optional[int] = None

    @classmethod
    def from_task_kwargs(
        cls,
        chunk_size=None,
        pipeline_bytes=None,
        io_depth=None,
        reorder_window=None,
        burst_bytes=None,
        burst_adaptive: bool = False,
        io_span_bytes=None,
        cluster_size=None,
        md_pages=None,
    ) -> "PipelineOpts":
        """Build from the flat ``--chunk-size``/``--io-depth``/... task arguments."""
        burst = None
        if burst_bytes is not None:
            text = str(burst_bytes).strip()
            burst = (
                text if text == BURST_MAX_TOKEN else str(_as_int("burst_bytes", text))
            )
        return cls(
            chunk_size=_as_int("chunk_size", chunk_size),
            pipeline_bytes=_as_int("pipeline_bytes", pipeline_bytes),
            io_depth=_as_int("io_depth", io_depth),
            reorder_window=_as_int("reorder_window", reorder_window),
            burst_bytes=burst,
            burst_adaptive=bool(burst_adaptive),
            io_span_bytes=_as_int("io_span_bytes", io_span_bytes),
            cluster_size=_as_int("cluster_size", cluster_size),
            md_pages=_as_int("md_pages", md_pages),
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "PipelineOpts":
        """Pick this module's keys out of an action config, ignoring the rest."""
        return cls.from_task_kwargs(
            chunk_size=cfg.get("chunk_size"),
            pipeline_bytes=cfg.get("pipeline_bytes"),
            io_depth=cfg.get("io_depth"),
            reorder_window=cfg.get("reorder_window"),
            burst_bytes=cfg.get("burst_bytes"),
            burst_adaptive=cfg.get("burst_adaptive", False),
            io_span_bytes=cfg.get("io_span_bytes"),
            cluster_size=cfg.get("cluster_size"),
            md_pages=cfg.get("md_pages"),
        )

    def with_chunk_size(self, chunk_size: Optional[int]) -> "PipelineOpts":
        """This geometry with the chunk replaced by a resolved value."""
        return PipelineOpts(
            chunk_size=_as_int("chunk_size", chunk_size),
            pipeline_bytes=self.pipeline_bytes,
            io_depth=self.io_depth,
            reorder_window=self.reorder_window,
            burst_bytes=self.burst_bytes,
            burst_adaptive=self.burst_adaptive,
            io_span_bytes=self.io_span_bytes,
            cluster_size=self.cluster_size,
            md_pages=self.md_pages,
        )

    def effective_chunk_size(self) -> int:
        """The chunk the accelstore arm actually runs at, default resolved."""
        return self.chunk_size or DEFAULT_CHUNK_BYTES

    def effective_io_depth(self) -> int:
        """Chunk I/Os the accelstore arm keeps in flight per operation.

        ``as_crypto_window()`` (include/accelstore.h) in Python, so the POSIX
        arm can be given the same depth instead of its ``--io-depth 1`` default.
        """
        if self.io_depth is not None:
            return self.io_depth
        budget = (
            DEFAULT_PIPELINE_BYTES
            if self.pipeline_bytes is None
            else self.pipeline_bytes
        )
        chunk = self.effective_chunk_size()
        return max(1, min(WINDOW_MAX, budget // chunk))

    def is_default(self) -> bool:
        """True when the run leaves the whole geometry at the library's defaults."""
        return (
            self.chunk_size is None
            and self.pipeline_bytes is None
            and self.io_depth is None
            and self.reorder_window is None
            and self.burst_bytes is None
            and not self.burst_adaptive
            and self.io_span_bytes is None
            and self.cluster_size is None
            and self.md_pages is None
        )

    def validate(self) -> None:
        """Reject what ``parse_common_arguments`` would, before a VM boots."""
        if self.chunk_size is not None and self.chunk_size != 0:
            if not MIN_CHUNK_BYTES <= self.chunk_size <= MAX_CHUNK_BYTES:
                raise ValueError(
                    f"chunk_size {self.chunk_size} is outside the client's"
                    f" [{MIN_CHUNK_BYTES}, {MAX_CHUNK_BYTES}] range"
                    " (0 = the 1 MiB library default)"
                )
        if (
            self.pipeline_bytes is not None
            and not 0 <= self.pipeline_bytes <= MAX_PIPELINE_BYTES
        ):
            raise ValueError(
                f"pipeline_bytes {self.pipeline_bytes} is outside"
                f" [0, {MAX_PIPELINE_BYTES}] (0 = the 64 MiB default)"
            )
        for name, value in (
            ("io_depth", self.io_depth),
            ("reorder_window", self.reorder_window),
        ):
            if value is not None and not 1 <= value <= WINDOW_MAX:
                raise ValueError(f"{name} {value} is outside [1, {WINDOW_MAX}] chunks")
        if self.burst_bytes is not None and self.burst_bytes != BURST_MAX_TOKEN:
            burst = int(self.burst_bytes)
            if not MIN_BURST_BYTES <= burst <= MAX_BURST_BYTES:
                raise ValueError(
                    f"burst_bytes {burst} is outside"
                    f" [{MIN_BURST_BYTES}, {MAX_BURST_BYTES}]; pass"
                    f" '{BURST_MAX_TOKEN}' for an uncapped pass"
                )
        if (
            self.io_span_bytes is not None
            and not MIN_SPAN_BYTES <= self.io_span_bytes <= MAX_SPAN_BYTES
        ):
            raise ValueError(
                f"io_span_bytes {self.io_span_bytes} is outside"
                f" [{MIN_SPAN_BYTES}, {MAX_SPAN_BYTES}]"
            )
        if self.cluster_size is not None:
            if not 0 <= self.cluster_size <= MAX_CLUSTER_BYTES:
                raise ValueError(f"cluster_size {self.cluster_size} is out of range")
            if self.cluster_size % CLUSTER_MULTIPLE:
                raise ValueError(
                    f"cluster_size {self.cluster_size} is not a multiple of"
                    f" {CLUSTER_MULTIPLE} (0 = the 1 MiB default)"
                )
        if self.md_pages is not None and not 0 <= self.md_pages <= MAX_MD_PAGES:
            raise ValueError(
                f"md_pages {self.md_pages} is outside [0, {MAX_MD_PAGES}]"
                " (0 = SPDK's own default of one 4 KiB page per cluster)"
            )

    def warnings(self) -> list[str]:
        """What is legal but will not measure what its label claims."""
        out = []
        chunk = self.chunk_size or DEFAULT_CHUNK_BYTES
        budget = (
            DEFAULT_PIPELINE_BYTES
            if self.pipeline_bytes is None
            else self.pipeline_bytes
        )
        cluster = self.cluster_size or DEFAULT_CLUSTER_BYTES

        if self.chunk_size is not None and self.pipeline_bytes is None:
            out.append(
                f"chunk_size {_format_bytes(chunk)} without pipeline_bytes: the"
                f" in-flight byte budget stays at its {_format_bytes(budget)}"
                " default, so the arm moves the command depth and the iobuf"
                " footprint along with the chunk. Pin"
                " --pipeline-bytes across a chunk pair to isolate the chunk."
            )
        if budget and budget < chunk and self.io_depth is None:
            out.append(
                f"pipeline_bytes {_format_bytes(budget)} is below the chunk size"
                f" {_format_bytes(chunk)}: the derived depth floors at one chunk,"
                " so the run has no pipelining at all"
            )
        if self.cluster_size and self.cluster_size < chunk:
            # Only a warning: the fragments are submitted concurrently, so the
            # cost is more and smaller I/Os, not a broken run.
            out.append(
                f"cluster_size {_format_bytes(self.cluster_size)} is below the"
                f" {'' if self.chunk_size else 'default '}chunk size"
                f" {_format_bytes(chunk)}: every chunk spans"
                f" {-(-chunk // self.cluster_size)} clusters, so a chunk is that"
                " many concurrent device I/Os"
            )
        fragments = -(-chunk // cluster)
        demand = (self.io_depth or 0) * fragments
        if demand > FRAGMENT_BUDGET:
            out.append(
                f"io_depth {self.io_depth} x {fragments} cluster fragments per"
                f" command = {demand} exceeds the per-thread fragment budget of"
                f" {FRAGMENT_BUDGET}: the run loses throughput"
                " in proportion to the overcommit"
            )
        if (
            self.cluster_size
            and self.md_pages is None
            and self.cluster_size < DEFAULT_CLUSTER_BYTES
        ):
            out.append(
                f"cluster_size {_format_bytes(self.cluster_size)} without"
                " md_pages: SPDK reserves one 4 KiB metadata page per cluster,"
                f" i.e. {4096 / self.cluster_size:.2%} of the device."
                " Size --md-pages for the workload's object"
                " count, or init may fail with -ENOMEM"
            )
        return out

    def client_args(self) -> list[str]:
        """The client command-line flags for this geometry."""
        args: list[str] = []
        if self.chunk_size is not None:
            args += ["--crypto-chunk-size", str(self.chunk_size)]
        if self.pipeline_bytes is not None:
            args += ["--pipeline-bytes", str(self.pipeline_bytes)]
        if self.io_depth is not None:
            args += ["--io-depth", str(self.io_depth)]
        if self.reorder_window is not None:
            args += ["--reorder-window", str(self.reorder_window)]
        if self.burst_bytes is not None:
            args += ["--burst-bytes", str(self.burst_bytes)]
        if self.burst_adaptive:
            args.append("--burst-adaptive")
        if self.io_span_bytes is not None:
            args += ["--io-span-bytes", str(self.io_span_bytes)]
        if self.cluster_size is not None:
            args += ["--cluster-size", str(self.cluster_size)]
        if self.md_pages is not None:
            args += ["--md-pages", str(self.md_pages)]
        return args

    def as_config(self) -> dict:
        """The keys that were set, to merge into an action_config."""
        cfg = {}
        for key, value in (
            ("chunk_size", self.chunk_size),
            ("pipeline_bytes", self.pipeline_bytes),
            ("io_depth", self.io_depth),
            ("reorder_window", self.reorder_window),
            ("burst_bytes", self.burst_bytes),
            ("io_span_bytes", self.io_span_bytes),
            ("cluster_size", self.cluster_size),
            ("md_pages", self.md_pages),
        ):
            if value is not None:
                cfg[key] = value
        if self.burst_adaptive:
            cfg["burst_adaptive"] = True
        return cfg

    def as_task_kwargs(self) -> dict:
        """The ``run-all`` arguments for this geometry, for a sweep to splat."""
        return self.as_config()

    def as_dict(self) -> dict:
        """JSON-friendly form for a run's flags sidecar (every field, set or not)."""
        return {
            "chunk_size": self.chunk_size,
            "pipeline_bytes": self.pipeline_bytes,
            "io_depth": self.io_depth,
            "reorder_window": self.reorder_window,
            "burst_bytes": self.burst_bytes,
            "burst_adaptive": self.burst_adaptive,
            "io_span_bytes": self.io_span_bytes,
            "cluster_size": self.cluster_size,
            "md_pages": self.md_pages,
        }

    def path_component(self) -> Optional[str]:
        """Stable slug identifying this geometry in a result path.

        ``None`` at the library defaults. A field set to the value the library
        would have picked is a sweep arm and still appears.
        """
        tokens = self.path_tokens()
        return SLUG_PREFIX + "+".join(tokens) if tokens else None

    def path_tokens(self, include_chunk: bool = True) -> list[str]:
        """The result-path tokens for this geometry, one per field that was set.

        ``include_chunk=False`` is for the bench path, which labels its chunk
        itself.
        """
        parts = []
        if include_chunk and self.chunk_size is not None:
            parts.append(f"chunk-{_format_bytes(self.chunk_size)}")
        if self.pipeline_bytes is not None:
            parts.append(f"budget-{_format_bytes(self.pipeline_bytes)}")
        if self.io_depth is not None:
            parts.append(f"depth-{self.io_depth}")
        if self.reorder_window is not None:
            parts.append(f"reorder-{self.reorder_window}")
        if self.burst_bytes is not None:
            burst = (
                BURST_MAX_TOKEN
                if self.burst_bytes == BURST_MAX_TOKEN
                else _format_bytes(int(self.burst_bytes))
            )
            parts.append(f"burst-{burst}")
        if self.burst_adaptive:
            parts.append("burst-adaptive")
        if self.io_span_bytes is not None:
            parts.append(f"span-{_format_bytes(self.io_span_bytes)}")
        if self.cluster_size is not None:
            parts.append(f"cluster-{_format_bytes(self.cluster_size)}")
        if self.md_pages is not None:
            parts.append(f"mdpages-{self.md_pages}")
        return parts

    def label(self) -> str:
        slug = self.path_component()
        return "library-default" if slug is None else slug[len(SLUG_PREFIX) :]
