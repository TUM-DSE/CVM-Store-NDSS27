#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The MLPerf Storage runs we make, and how each one is driven.

:data:`MLPERF_VARIANTS` is the authority: a *variant* is a workload shape and an
*arm* is the storage it runs against. A run is one (variant, arm) pair, and its
argv, environment, result-path slugs and deviations are all derived here.
"""

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from modules.accelstore.tasks.pipeline_opts import PipelineOpts
from modules.accelstore.tasks.storage_stacks import (
    BLOCK_STACKS,
    DEFAULT_STACK,
    RAW_STACK,
    BlockStack,
)
from modules.accelstore.tasks.sync_modes import SyncMode

MLPSTORAGE_BIN = "mlpstorage"

ENV_PREFIX = "MLPS_ACCELSTORE_"

# Emitted either way, so a result's environment records which arm it was.
CHECKPOINT_PERSISTENT_WRITER_ENV = "MLPS_CHECKPOINT_PERSISTENT_WRITER"
CHECKPOINT_PERSISTENT_WRITER = True

# Mirrors the mlpstorage session's defaults: the harness starts the master.
SESSION_WORKERS_DEFAULT = 4
SESSION_MAX_OP_BYTES_DEFAULT = 64 << 20

MULTIRANK_WORKERS = 8

# A dispatch, not a label: it routes DLIO's `file` route to the AccelStore writer.
ACCELSTORE_URI_SCHEME = "accelstore"

FAMILIES = ("checkpointing",)
DIVISIONS = ("closed", "open", "whatif")

FAMILY_MODULES = {
    "checkpointing": ("log", "fs"),
}

VM_SIZE_DEFAULT = "medium"

# ClosedGPUs per model, from mlpstorage_py/config.py.
LLM_CLOSED_PROCS = {
    "llama3-8b": 8,
    "llama3-70b": 64,
    "llama3-405b": 512,
    "llama3-1t": 1024,
}
LLM_SUBSET_PROCS = 8


@dataclass(frozen=True)
class MlperfArm:
    """One storage a variant is run against."""

    id: str
    kind: str
    module: Optional[str]
    label: str
    notes: tuple[str, ...] = ()
    default: bool = True  # runs when no --arms is asked for
    family_notes: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def is_accelstore(self) -> bool:
        return self.kind == "accelstore"

    def notes_for(self, family: str) -> tuple[str, ...]:
        """This arm's deviations in one family."""
        return self.notes + tuple(self.family_notes.get(family, ()))

    def path_component(self) -> str:
        """Stable slug identifying this arm in a result path."""
        return self.id


MLPERF_ARMS = {
    arm.id: arm
    for arm in (
        MlperfArm(
            id="accelstore-log",
            kind="accelstore",
            module="log",
            label="AccelStore (LOG)",
        ),
        MlperfArm(
            id="accelstore-fs",
            kind="accelstore",
            module="fs",
            label="AccelStore (FS)",
        ),
        MlperfArm(
            id="direct",
            kind="posix",
            module=None,
            label="O_DIRECT (s3dlio direct://)",
            default=False,
            notes=(
                "mlpstorage --o-direct with the `file` positional rewrites the"
                " run to storage_type=direct_fs, library s3dlio, uri_scheme"
                " direct (benchmarks/dlio.py:231-263), so every file is opened"
                " O_DIRECT and the page cache is out of the comparison",
                "what that path is *worth* was measured standalone on the"
                " evaluation NVMe: ~0.49 GB/s write and ~0.46"
                " GB/s read, against fio of the same shape at 3.6-4.9 GB/s"
                " write and 2.7-6.1 GB/s read on the same device. A row from"
                " this arm therefore bounds the s3dlio library and not the"
                " storage stack, and must not be read as what the hardware does"
                " under O_DIRECT -- posix-direct is that row. Kept for"
                " reference",
            ),
        ),
        MlperfArm(
            id="posix-buffered",
            kind="posix",
            module=None,
            label="POSIX buffered (file backend + fadvise DONTNEED)",
            default=False,
            notes=(
                "not a durability equivalent of the two O_DIRECT arms -- see"
                " the family note below for what does, or does not, defeat the"
                " page cache on that family's I/O path; the two differ",
            ),
            family_notes={
                "checkpointing": (
                    "mlpstorage's own POSIX file backend: buffered writes and"
                    " one fsync at close, with the page cache defeated by a"
                    " per-chunk posix_fadvise(POSIX_FADV_DONTNEED) on the write"
                    " path *and* the read path rather than by O_DIRECT. Kernel"
                    " readahead is left on (POSIX_FADV_SEQUENTIAL), so a read"
                    " still comes off the device at full depth",
                    "that is semantically weaker than O_DIRECT -- a write is"
                    " acknowledged by the page cache and the hint only follows"
                    " it, so this arm is not a durability equivalent of the two"
                    " O_DIRECT ones -- and it is measured at device speed all"
                    " the same. Both halves are measurement-fairness context a"
                    " row here is read with, not an aside",
                ),
            },
        ),
        MlperfArm(
            id="posix-direct",
            kind="posix",
            module=None,
            label="POSIX O_DIRECT (native backend)",
            notes=(
                "it replaces the s3dlio direct:// path as *the* O_DIRECT"
                " baseline: that path measured ~0.5 GB/s on a local NVMe whose"
                " fio ceiling is 3.6-6 GB/s, i.e. it"
                " bounded the library rather than the device. s3dlio stays"
                " reachable as the `direct` arm, for reference",
            ),
            family_notes={
                "checkpointing": (
                    "mlpstorage's native O_DIRECT backend (`file_direct`):"
                    " aligned staging buffers, a small pwrite thread pool"
                    " (queue depth 4 by default, MLPS_POSIX_ODIRECT_QD) and"
                    " preadv into preallocated aligned buffers on the read"
                    " side",
                ),
            },
        ),
    )
}

POSIX_ARMS = tuple(a.id for a in MLPERF_ARMS.values() if a.kind == "posix")

DEFAULT_ARMS = tuple(a.id for a in MLPERF_ARMS.values() if a.default)
OPTIONAL_ARMS = tuple(a.id for a in MLPERF_ARMS.values() if not a.default)


# Alias kept: mlperf code and its tests use this name for the shared type.
MlperfStack = BlockStack


MLPERF_STACKS = {
    RAW_STACK: BlockStack(
        id=RAW_STACK,
        label="raw NVMe (SPDK/vfio)",
        kernel=None,
        default_capacity_gb=None,
        ack_summary="device",
        ack_semantics=(
            "the data of an acknowledged operation has reached the device; its "
            "metadata durability point is whatever --sync-* configured"
        ),
        notes=(
            "not a selectable stack: the accelstore arms own the device"
            " through vfio and never see the kernel block layer, so they"
            " are neither stacked nor windowed by --stack-capacity-gb",
        ),
    ),
    **BLOCK_STACKS,
}

POSIX_STACKS = tuple(s for s in MLPERF_STACKS if s != RAW_STACK)


def stack_of(stack) -> MlperfStack:
    """Resolve a stack id (or stack) against the table."""
    if isinstance(stack, MlperfStack):
        return stack
    if stack not in MLPERF_STACKS:
        raise ValueError(
            f"Unknown block stack '{stack}'. Known: {', '.join(POSIX_STACKS)}"
        )
    return MLPERF_STACKS[stack]


def stack_for(arm, stack) -> MlperfStack:
    """The stack a (arm, stack) pair runs on: `raw` for any accelstore arm."""
    arm = MLPERF_ARMS[arm] if isinstance(arm, str) else arm
    return MLPERF_STACKS[RAW_STACK] if arm.is_accelstore else stack_of(stack)


def stacks_for_arm(arm, stacks) -> tuple[MlperfStack, ...]:
    """Which stacks one arm contributes a point on: one for accelstore, one per
    stack for POSIX."""
    arm = MLPERF_ARMS[arm] if isinstance(arm, str) else arm
    if arm.is_accelstore:
        return (MLPERF_STACKS[RAW_STACK],)
    return tuple(stack_of(s) for s in stacks)


# An unmapped POSIX arm raises rather than silently running the default path.
CHECKPOINTING_POSIX_IO_ARGS = {
    "direct": ("--o-direct",),
    "posix-buffered": ("--posix-io", "buffered"),
    "posix-direct": ("--posix-io", "odirect"),
}


@dataclass(frozen=True)
class MlperfRunContext:
    """The per-run values a variant cannot know: paths, sizes, the host."""

    results_dir: str
    systemname: str = "cvm-store"
    client_host_memory_gb: float = 0.0  # the guest's memory
    # accelstore:// namespace URI on an accelstore arm, a directory on a POSIX one
    checkpoint_folder: str = ""
    hosts: tuple[str, ...] = ()
    extra: tuple[str, ...] = ()  # appended verbatim to every argv


@dataclass(frozen=True)
class StoreGeometry:
    """The store a variant's accelstore rows are laid out in.

    Every field is ``None`` for "leave the library's default". The bounds are
    :class:`PipelineOpts`'s.
    """

    chunk_size: Optional[int] = None
    cluster_size: Optional[int] = None
    md_pages: Optional[int] = None
    pipeline_bytes: Optional[int] = None

    def _opts(self) -> PipelineOpts:
        return PipelineOpts(
            chunk_size=self.chunk_size,
            pipeline_bytes=self.pipeline_bytes,
            cluster_size=self.cluster_size,
            md_pages=self.md_pages,
        )

    def validate(self) -> None:
        self._opts().validate()

    def warnings(self) -> list[str]:
        return self._opts().warnings()

    def is_default(self) -> bool:
        return self._opts().is_default()

    def overridden(
        self,
        *,
        chunk_size: Optional[int] = None,
        cluster_size: Optional[int] = None,
        md_pages: Optional[int] = None,
        pipeline_bytes: Optional[int] = None,
    ) -> "StoreGeometry":
        """This geometry with whatever the caller named replacing it."""
        return StoreGeometry(
            chunk_size=self.chunk_size if chunk_size is None else int(chunk_size),
            cluster_size=(
                self.cluster_size if cluster_size is None else int(cluster_size)
            ),
            md_pages=self.md_pages if md_pages is None else int(md_pages),
            pipeline_bytes=(
                self.pipeline_bytes if pipeline_bytes is None else int(pipeline_bytes)
            ),
        )

    def describe(self) -> str:
        """One log line: what the store was built with."""
        return " ".join(
            f"{name}={'default' if value is None else value}"
            for name, value in (
                ("chunk", self.chunk_size),
                ("cluster", self.cluster_size),
                ("md-pages", self.md_pages),
                ("pipeline-bytes", self.pipeline_bytes),
            )
        )


@dataclass(frozen=True)
class MlperfVariant:
    """One workload shape, and the arms it is measured on."""

    id: str
    family: str
    division: str
    label: str
    arms: tuple[str, ...]
    model: str = ""
    smoke: bool = False
    blocked: tuple[tuple[str, str], ...] = ()  # (arm id, why that pair cannot run)
    notes: tuple[str, ...] = ()

    # 0 leaves the session default. It also sizes the master's reactor mask.
    store_workers: int = 0

    # 0 leaves the session default. Too small grows a scratch buffer per read.
    max_op_bytes: int = 0

    store_geometry: StoreGeometry = StoreGeometry()

    # Live GB (10^9) left in the store, from mlpstorage's own capacity gate.
    min_capacity_gb: int = 0

    num_processes: int = 0
    checkpoint_subset: bool = False

    def path_component(self) -> str:
        """Stable slug identifying this variant in a result path."""
        return self.id

    def result_path(self, arm: str, stack=None) -> tuple[str, ...]:
        """Result-directory components: family / variant / arm [/ stack]."""
        components = (
            self.family,
            self.path_component(),
            self.arm(arm).path_component(),
        )
        token = stack_of(stack).path_component() if stack is not None else ""
        return components + (token,) if token else components

    def capacity_needed_gb(self) -> int:
        """Live GB this row leaves in the store: the declared figure."""
        return int(self.min_capacity_gb)

    def arm(self, arm) -> MlperfArm:
        """Resolve an arm id (or arm) and check it belongs to this variant."""
        if isinstance(arm, MlperfArm):
            arm = arm.id
        if arm not in self.arms:
            raise ValueError(
                f"variant {self.id} has no arm '{arm}'." f" Arms: {'|'.join(self.arms)}"
            )
        return MLPERF_ARMS[arm]

    @property
    def blocked_arms(self) -> dict[str, str]:
        """arm id -> the reason that pair cannot run today."""
        return dict(self.blocked)

    def runnable_arms(self) -> tuple[str, ...]:
        """The arms that are wired up today, in table order."""
        blocked = self.blocked_arms
        return tuple(a for a in self.arms if a not in blocked)

    def deviations(self, arm) -> tuple[str, ...]:
        """What has to be written next to this run's numbers: the variant's
        deviations plus the arm's, for this family."""
        return self.notes + self.arm(arm).notes_for(self.family)

    @property
    def vm_size(self) -> str:
        """The config.toml VM size this row runs at when `--sizes` names none."""
        return VM_SIZE_DEFAULT

    @property
    def rank_processes(self) -> int:
        """The rank count: checkpointing's ``--num-processes``."""
        return self.num_processes

    def store_processes(self) -> int:
        """Processes of one run that reach this row's store: the ranks."""
        return max(self.rank_processes, 1)

    def ipc_master(self, arm) -> bool:
        """Whether this (variant, arm) needs a dedicated store master.

        A single-process run is left in ``local`` mode deliberately: a master
        would put an IPC hop on a run with nothing to share the store with.
        """
        return self.arm(arm).is_accelstore and self.store_processes() > 1

    def workers_for(self, workers: Optional[int] = None) -> Optional[int]:
        """The store's worker count: the caller's, else this row's, else ``None``
        for 'say nothing'."""
        if workers is not None:
            return int(workers)
        return self.store_workers or None

    def max_op_bytes_for(self, max_op_bytes: Optional[int] = None) -> Optional[int]:
        """The store's per-operation buffer, with :meth:`workers_for`'s precedence."""
        if max_op_bytes is not None:
            return int(max_op_bytes)
        return self.max_op_bytes or None

    def geometry_for(
        self,
        *,
        chunk_size: Optional[int] = None,
        cluster_size: Optional[int] = None,
        md_pages: Optional[int] = None,
        pipeline_bytes: Optional[int] = None,
    ) -> StoreGeometry:
        """The store geometry of one run, with :meth:`workers_for`'s precedence
        applied field by field."""
        geometry = self.store_geometry.overridden(
            chunk_size=chunk_size,
            cluster_size=cluster_size,
            md_pages=md_pages,
            pipeline_bytes=pipeline_bytes,
        )
        geometry.validate()
        return geometry

    def cores_needed(self, arm=None, workers: Optional[int] = None) -> int:
        """Guest cores one run of this row wants: one per rank, plus the
        master's ``workers + 1`` reactors. Without an ``arm``, the largest over
        this row's arms.
        """
        if arm is None:
            return max(self.cores_needed(a, workers) for a in self.arms)
        ranks = max(self.rank_processes, 1)
        if not self.ipc_master(arm):
            return ranks
        return ranks + (self.workers_for(workers) or SESSION_WORKERS_DEFAULT) + 1

    def validate(self) -> None:
        """Reject a table entry that could not run, before any VM boots."""
        if self.family not in FAMILIES:
            raise ValueError(
                f"{self.id}: family '{self.family}' is unknown."
                f" Accepted: {'|'.join(FAMILIES)}"
            )
        if self.division not in DIVISIONS:
            raise ValueError(
                f"{self.id}: division '{self.division}' is unknown."
                f" Accepted: {'|'.join(DIVISIONS)}"
            )
        if not self.id or not _is_slug(self.id):
            raise ValueError(
                f"variant id '{self.id}' is not a path component:"
                " lowercase letters, digits and dashes only"
            )
        if not self.arms:
            raise ValueError(f"{self.id}: a variant with no arm measures nothing")

        for arm_id in self.arms:
            if arm_id not in MLPERF_ARMS:
                raise ValueError(f"{self.id}: unknown arm '{arm_id}'")
            arm = MLPERF_ARMS[arm_id]
            if arm.is_accelstore and arm.module not in FAMILY_MODULES[self.family]:
                raise ValueError(
                    f"{self.id}: the {self.family} session does not accept module"
                    f" '{arm.module}' (it takes"
                    f" {'|'.join(FAMILY_MODULES[self.family])})"
                )
        unknown_blocks = set(self.blocked_arms) - set(self.arms)
        if unknown_blocks:
            raise ValueError(
                f"{self.id}: blocked names arms it does not have:"
                f" {', '.join(sorted(unknown_blocks))}"
            )
        if not self.runnable_arms():
            raise ValueError(f"{self.id}: every arm is blocked; drop the entry")

        default_arms = tuple(a for a in self.runnable_arms() if a in DEFAULT_ARMS)
        if not default_arms:
            raise ValueError(
                f"{self.id}: every runnable arm is opt-in, so a default sweep"
                " would skip the row entirely"
            )
        if not any(MLPERF_ARMS[a].kind == "posix" for a in default_arms):
            raise ValueError(
                f"{self.id}: no baseline runs by default, so a default sweep"
                " produces accelstore numbers with nothing to divide by"
            )

        self.store_geometry.validate()
        if self.max_op_bytes < 0:
            raise ValueError(f"{self.id}: max_op_bytes is a size, not an offset")

        getattr(self, f"_validate_{self.family}")()

    def _validate_checkpointing(self) -> None:
        """What a checkpointing entry has to be, per division."""
        if self.model not in LLM_CLOSED_PROCS:
            raise ValueError(
                f"{self.id}: '{self.model}' is not an LLM checkpointing model"
            )
        if self.num_processes <= 0:
            raise ValueError(f"{self.id}: a checkpointing run needs --num-processes")
        if (
            self.division == "closed"
            and self.num_processes != LLM_CLOSED_PROCS[self.model]
        ):
            raise ValueError(
                f"{self.id}: closed checkpointing of {self.model} is exactly"
                f" {LLM_CLOSED_PROCS[self.model]} processes, not {self.num_processes}"
            )
        if self.checkpoint_subset and (
            self.model != "llama3-8b" or self.num_processes != LLM_SUBSET_PROCS
        ):
            raise ValueError(
                f"{self.id}: --checkpoint-subset is defined only for llama3-8b at"
                f" {LLM_SUBSET_PROCS} processes"
            )

    def validate_context(self, ctx: MlperfRunContext, arm=None) -> None:
        """Reject a run context this variant cannot be driven with.

        ``arm`` is optional: the rules that depend on it are then skipped.
        """
        if not ctx.results_dir:
            raise ValueError(f"{self.id}: a run needs a results directory")
        if not ctx.systemname:
            raise ValueError(
                f"{self.id}: mlpstorage requires --systemname on every emitting command"
            )
        if ctx.client_host_memory_gb <= 0:
            raise ValueError(
                f"{self.id}: --client-host-memory-in-gb is required and is the"
                " guest's memory, not a constant of the workload"
            )
        if self.family == "checkpointing":
            if not ctx.checkpoint_folder:
                raise ValueError(f"{self.id}: checkpointing needs --checkpoint-folder")
            resolved = self.arm(arm) if arm is not None else None
            if resolved is None:
                pass
            elif resolved.is_accelstore:
                # The scheme is the dispatch: without it DLIO writes a POSIX tree.
                if not ctx.checkpoint_folder.startswith(ACCELSTORE_URI_SCHEME + "://"):
                    raise ValueError(
                        f"{self.id}: checkpoint_folder '{ctx.checkpoint_folder}'"
                        f" has no {ACCELSTORE_URI_SCHEME}:// scheme; that scheme"
                        " is what sends the run to the AccelStore writer, and"
                        " without it the file route writes a POSIX tree"
                    )
            elif not ctx.checkpoint_folder.startswith("/"):
                # A bare name would resolve against the container's cwd.
                raise ValueError(
                    f"{self.id}: checkpoint_folder '{ctx.checkpoint_folder}' is"
                    " not an absolute path; a POSIX arm checkpoints into the"
                    " mounted directory"
                )

    def run_args(self, arm, ctx: MlperfRunContext) -> list[str]:
        """The measured command, as argv after the image's ``mlpstorage``
        ENTRYPOINT."""
        resolved = self._checked_arm(arm, ctx)
        return self._run_args_checkpointing(resolved, ctx) + list(ctx.extra)

    def _run_args_checkpointing(
        self, arm: MlperfArm, ctx: MlperfRunContext
    ) -> list[str]:
        args = [
            self.division,
            "checkpointing",
            "run",
            # `file` on every arm: checkpointing dispatches on the folder.
            "file",
            "--model",
            self.model,
            "--num-processes",
            str(self.num_processes),
            "--client-host-memory-in-gb",
            _num(ctx.client_host_memory_gb),
            "--checkpoint-folder",
            ctx.checkpoint_folder,
        ]
        if self.checkpoint_subset:
            args.append("--checkpoint-subset")
        if not arm.is_accelstore:
            args += list(
                self._storage_flag_args(
                    arm, CHECKPOINTING_POSIX_IO_ARGS, "CHECKPOINTING_POSIX_IO_ARGS"
                )
            )
        return args + self._universal(ctx)

    @staticmethod
    def _storage_flag_args(
        arm: MlperfArm, table: Mapping[str, tuple[str, ...]], table_name: str
    ) -> tuple[str, ...]:
        """This POSIX arm's storage flag; see the table above."""
        try:
            return table[arm.id]
        except KeyError:
            raise ValueError(
                f"the POSIX arm '{arm.id}' has no storage flag in {table_name}."
                " Give it one: an arm that emits none runs mlpstorage's default"
                " buffered path under its own label, which no note next to the"
                " row would correct"
            ) from None

    def _universal(self, ctx: MlperfRunContext) -> list[str]:
        args = ["--results-dir", ctx.results_dir, "--systemname", ctx.systemname]
        if ctx.hosts:
            args += ["--hosts", *ctx.hosts]
        return args

    def _checked_arm(self, arm, ctx: MlperfRunContext) -> MlperfArm:
        resolved = self.arm(arm)
        blocked = self.blocked_arms
        if resolved.id in blocked:
            raise ValueError(
                f"{self.id} on {resolved.id} cannot run: {blocked[resolved.id]}"
            )
        self.validate_context(ctx, resolved)
        return resolved

    def env(
        self,
        arm,
        *,
        config: str = "",
        device: str = "",
        tc_path: str = "",
        ipc_socket: str = "",
        sync: Optional[SyncMode] = None,
        workers: Optional[int] = None,
        max_op_bytes: Optional[int] = None,
        crypto_chunk_size: Optional[int] = None,
        pipeline_bytes: Optional[int] = None,
        cluster_size: Optional[int] = None,
        md_pages: Optional[int] = None,
        tracing: Optional[str] = None,
        trace_dir: Optional[str] = None,
        trace_entries: Optional[int] = None,
        trace_crypto_chunks: Optional[bool] = None,
        trace_role: Optional[str] = None,
        silent: Optional[bool] = None,
    ) -> dict[str, str]:
        """The container environment for one run.

        Empty for a POSIX arm -- those carry their storage in the argv.
        ``config`` (the SPDK bdev JSON) and ``device`` (the bdev name) are
        required on an accelstore arm, as is ``tc_path``: the session's own
        default lives in the container's temp directory and does not survive
        the run. ``ipc_socket`` is required exactly where :meth:`ipc_master`
        is true and is what makes a rank a client. ``workers``,
        ``max_op_bytes`` and the geometry knobs (``crypto_chunk_size``,
        ``pipeline_bytes``, ``cluster_size``, ``md_pages``) override this
        row's declarations. ``sync`` is emitted as the session's seven
        ``SYNC_*`` variables. ``tracing`` is ``0``, ``1`` or ``small`` and
        gates ``trace_dir``, ``trace_entries``, ``trace_crypto_chunks`` and
        ``trace_role``. ``silent`` sets ``SILENT``.
        """
        resolved = self.arm(arm)
        if not resolved.is_accelstore:
            return {}
        if not config or not device:
            raise ValueError(
                f"{self.id}: the accelstore arm needs {ENV_PREFIX}CONFIG (the SPDK"
                f" bdev JSON) and {ENV_PREFIX}DEVICE (the bdev name)"
            )
        if not tc_path:
            raise ValueError(
                f"{self.id}: set {ENV_PREFIX}TC_PATH explicitly; the session's own"
                " default lives in the container's temp directory and does not"
                " survive to the next phase of the run"
            )

        if self.ipc_master(resolved) and not ipc_socket:
            raise ValueError(
                f"{self.id}: {self.store_processes()} processes reach one"
                f" store, so every one of them is a client"
                f" and needs {ENV_PREFIX}IPC_SOCKET naming the master's socket"
            )

        sync = sync or SyncMode()
        sync.validate()

        env = {
            ENV_PREFIX + "CONFIG": config,
            ENV_PREFIX + "DEVICE": device,
            ENV_PREFIX + "MODULE": resolved.module,
            ENV_PREFIX + "TC_PATH": tc_path,
        }
        if self.ipc_master(resolved):
            env[ENV_PREFIX + "MODE"] = "client"
            env[ENV_PREFIX + "IPC_SOCKET"] = ipc_socket
        geometry = self.geometry_for(
            chunk_size=crypto_chunk_size,
            cluster_size=cluster_size,
            md_pages=md_pages,
            pipeline_bytes=pipeline_bytes,
        )
        for name, value in (
            ("WORKERS", self.workers_for(workers)),
            ("MAX_OP_BYTES", self.max_op_bytes_for(max_op_bytes)),
            ("CRYPTO_CHUNK_SIZE", geometry.chunk_size),
            ("PIPELINE_BYTES", geometry.pipeline_bytes),
            ("CLUSTER_SIZE", geometry.cluster_size),
            ("MD_PAGES", geometry.md_pages),
            ("TC_RTT_US", sync.tc_rtt_us),
            ("TC_RTT_INTERVAL", sync.tc_rtt_interval),
        ):
            if value is not None:
                env[ENV_PREFIX + name] = str(value)
        for name, value in (
            ("TC_BACKEND", sync.tc_backend),
            ("TC_RTT_MODEL", sync.tc_rtt_model),
        ):
            if value:
                env[ENV_PREFIX + name] = value

        env[CHECKPOINT_PERSISTENT_WRITER_ENV] = (
            "1" if CHECKPOINT_PERSISTENT_WRITER else "0"
        )

        env.update(self._sync_env(sync))
        env.update(
            _trace_env(
                tracing=tracing,
                trace_dir=trace_dir,
                trace_entries=trace_entries,
                trace_crypto_chunks=trace_crypto_chunks,
                trace_role=trace_role,
            )
        )
        if silent is not None:
            env[ENV_PREFIX + "SILENT"] = "1" if silent else "0"
        return env

    def _sync_env(self, sync: SyncMode) -> dict[str, str]:
        """The sync policy as the session's seven ``MLPS_ACCELSTORE_SYNC_*``
        variables."""
        env = {}
        for name, value in (
            ("SYNC_TRIGGER", sync.trigger),
            ("SYNC_EVERY_N", sync.every_n),
            ("SYNC_INTERVAL_US", sync.interval_us),
            ("SYNC_COUPLING", sync.coupling),
            ("SYNC_COMMIT_EVERY", sync.commit_every),
        ):
            if value is not None:
                env[ENV_PREFIX + name] = str(value)
        if sync.metadata_only:
            env[ENV_PREFIX + "SYNC_METADATA_ONLY"] = "1"
        if sync.blocking:
            env[ENV_PREFIX + "SYNC_BLOCKING"] = "1"
        return env


def _trace_env(
    *,
    tracing: Optional[str],
    trace_dir: Optional[str],
    trace_entries: Optional[int],
    trace_crypto_chunks: Optional[bool],
    trace_role: Optional[str],
) -> dict[str, str]:
    """The tracing variables. ``TRACE_DIR`` without ``TRACING`` enables no
    tpoint group and would dump nothing, so it is refused here rather than at
    the session, one VM boot earlier."""
    modes = ("0", "1", "small")
    if tracing is not None and tracing not in modes:
        raise ValueError(f"tracing must be one of {'|'.join(modes)}, not {tracing!r}")
    off = tracing in (None, "0")
    if off and trace_dir:
        raise ValueError(
            "trace_dir without tracing: no tpoint group would be enabled and"
            " every dump would be empty"
        )
    if off and (trace_entries is not None or trace_crypto_chunks is not None):
        raise ValueError("the trace knobs need tracing to be on")

    env = {}
    if tracing is not None:
        env[ENV_PREFIX + "TRACING"] = tracing
    if trace_dir:
        env[ENV_PREFIX + "TRACE_DIR"] = trace_dir
    if trace_entries is not None:
        env[ENV_PREFIX + "TRACE_ENTRIES"] = str(trace_entries)
    if trace_crypto_chunks is not None:
        env[ENV_PREFIX + "TRACE_CRYPTO_CHUNKS"] = "1" if trace_crypto_chunks else "0"
    if trace_role:
        env[ENV_PREFIX + "TRACE_ROLE"] = trace_role
    return env


def _is_slug(text: str) -> bool:
    return all(
        c.islower() or c.isdigit() or c == "-" for c in text
    ) and not text.startswith("-")


def _num(value: float) -> str:
    """Render a size/memory operand without a gratuitous ``.0``."""
    return str(int(value)) if float(value).is_integer() else str(value)


MLPERF_VARIANTS = (
    MlperfVariant(
        id="ckpt-llama3-8b-subset",
        family="checkpointing",
        division="closed",
        label="llama3-8b checkpoint save/restore, subset (8 ranks)",
        model="llama3-8b",
        num_processes=8,
        checkpoint_subset=True,
        store_workers=MULTIRANK_WORKERS,
        # 8 ranks x (15 + 90) GiB / ClosedGPUs=8 x 10 retained checkpoints.
        min_capacity_gb=1200,
        arms=(
            "accelstore-log",
            "accelstore-fs",
            "direct",
            "posix-buffered",
            "posix-direct",
        ),
        notes=(
            "the subset shape is the officially valid one for llama3-8b"
            " (Rules.md 4.3.5; --checkpoint-subset is legal only at 8"
            " processes, rules/run_checkers/checkpointing.py:127-176) and"
            " CLOSED requires exactly 8 processes (:89-90)",
            "the accelstore arms run the 8 ranks as clients of one dedicated"
            " store master: the harness starts"
            " `python3 -m pyaccelstore` in the run's container, waits for its"
            " ready line and stops it after mlpstorage, and every rank -- plus"
            " every writer subprocess a rank forks -- reaches it over the"
            " socket. That topology measured at 0.999x write / 1.007x read"
            " of the same concurrency in one process's threads, so the rank"
            " count is the shape's and not the store's",
            "the master's stop is the store's final metadata sync, so its clean"
            " exit is part of the run: a master that dies early or never comes"
            " up fails the run rather than letting it fall back to local mode",
            "the client write path stages one copy through a per-thread /dev/shm"
            " arena, which is copy-for-copy what the O_DIRECT baseline's writer"
            " does (its staging pool is page-aligned and the producer's shm"
            " buffers are not), so the arms are not separated by that copy",
            "the three POSIX baselines run their native 8-rank shape here, so"
            " this is the row where every arm is at the shape the rules name."
            " They are three different I/O paths and not three samples of one:"
            " posix-direct is the O_DIRECT device row, posix-buffered is device"
            " speed with a page-cache acknowledgement, and `direct` bounds the"
            " s3dlio library rather than the device (their arm notes carry the"
            " numbers)",
            "--num-checkpoints-read/-write are left at their default of 10,"
            " which is what CLOSED requires (cli/checkpointing_args.py:222-237)",
        ),
    ),
    MlperfVariant(
        id="ckpt-llama3-8b-1rank",
        family="checkpointing",
        division="whatif",
        label="llama3-8b checkpoint save/restore, 1 rank",
        model="llama3-8b",
        num_processes=1,
        checkpoint_subset=False,
        store_workers=MULTIRANK_WORKERS,
        # 1 rank x (15 + 90) GiB / ClosedGPUs=8 x 10 retained checkpoints.
        min_capacity_gb=160,
        arms=(
            "accelstore-log",
            "accelstore-fs",
            "direct",
            "posix-buffered",
            "posix-direct",
        ),
        smoke=True,
        notes=(
            "the iteration row, and the local-mode reference: one rank owns the"
            " store in its own process, so it is the same workload shape with"
            " no IPC in it. The 8-rank CLOSED row (ckpt-llama3-8b-subset) is the"
            " measurement one; this row keeps the"
            " same store geometry (8 workers) so the two differ in the rank"
            " count and the store mode alone. It is NOT a clean IPC ablation --"
            " that would be one rank in client mode, which nothing here asks"
            " for -- and the byte counts differ too (see the sizing note)",
            "the POSIX arms run the same 1-rank shape rather than their native"
            " 8, so every arm is the same workload",
            "whatif because 1 is neither CLOSED's exact ClosedGPUs (8,"
            " rules/run_checkers/checkpointing.py:82-125) nor an OPEN multiple"
            " of it; the row is our measurement shape, not a submittable one",
            "--checkpoint-subset is NOT passed: it is refused anywhere but 8"
            " processes (:162-174). mlpstorage sets checkpoint.mode=subset"
            " itself below ClosedGPUs (benchmarks/dlio.py:1262-1283) and warns"
            " that this is not the §4.3.5 claim -- which is the same thing this"
            " note says",
            "sizing: the per-rank share is (15 + 90) / ClosedGPUs = ~13.1 GiB"
            " per checkpoint and the denominator is ClosedGPUs rather than the"
            " rank count (config.py:85-92, benchmarks/dlio.py:1413-1416), so at"
            " the default 10 write + 10 read checkpoints this row moves ~131"
            " GiB each way -- not the ~105 GiB an 8-rank checkpoint is",
            "the accelstore arms' checkpoint folder is an accelstore:// URI (a"
            " store namespace); mlpstorage appends the model to it"
            " (benchmarks/dlio.py:1312) and skips the CAP-01/CAP-03 capacity"
            " gates for such a target (:1452-1470), which the POSIX arms still"
            " pay",
        ),
    ),
)

MLPERF_VARIANTS_BY_ID = {v.id: v for v in MLPERF_VARIANTS}

SMOKE_VARIANTS = tuple(v.id for v in MLPERF_VARIANTS if v.smoke)


def variant(variant_id: str) -> MlperfVariant:
    """Look one variant up by id, with the table in the error message."""
    try:
        return MLPERF_VARIANTS_BY_ID[variant_id]
    except KeyError:
        raise ValueError(
            f"unknown MLPerf variant '{variant_id}'."
            f" Known: {', '.join(sorted(MLPERF_VARIANTS_BY_ID))}"
        ) from None


def variants_for(
    families: Optional[Sequence[str]] = None,
    *,
    smoke: bool = False,
) -> tuple[MlperfVariant, ...]:
    """The table, filtered the ways a run task filters it. A variant with no
    runnable arm is dropped."""
    out = []
    for entry in MLPERF_VARIANTS:
        if families and entry.family not in families:
            continue
        if smoke and not entry.smoke:
            continue
        if not entry.runnable_arms():
            continue
        out.append(entry)
    return tuple(out)


def parse_path_component(slug: str) -> MlperfVariant:
    """Decode a result-directory slug back into a variant."""
    return variant(slug)


def parse_arm_component(slug: str) -> MlperfArm:
    """Decode a result-directory slug back into an arm."""
    try:
        return MLPERF_ARMS[slug]
    except KeyError:
        raise ValueError(
            f"unknown MLPerf arm '{slug}'. Known: {', '.join(sorted(MLPERF_ARMS))}"
        ) from None


def validate_table() -> None:
    """Check the whole table. Called at import; also a test."""
    seen = set()
    for entry in MLPERF_VARIANTS:
        if entry.id in seen:
            raise ValueError(f"duplicate variant id '{entry.id}'")
        seen.add(entry.id)
        entry.validate()


validate_table()
