#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The block stacks a POSIX benchmark arm can be put on: one declaration.

A *stack* is the block device under a POSIX benchmark's filesystem -- plain
ext4 on the NVMe, or a dm target with ext4 on top. What a stack costs lives in
``tasks/block_stacks.py``, how it is built in ``tasks/actions/posix_storage.py``
(which asserts at import that its registry matches this table). No OmaArvio
dependency, so this imports without a VM or a guest image.
"""

from dataclasses import dataclass
from typing import Optional

from modules.accelstore.tasks import block_stacks

# The accelstore arm has no kernel block device at all (SPDK drives via vfio).
# No entry in BLOCK_STACKS -- that one lives in the MLPerf table.
RAW_STACK = "raw"
DEFAULT_STACK = "ext4"

# `defaultKernel` in the module flake (nix/kernel.nix).
DEFAULT_GUEST_KERNEL = "6.17"


@dataclass(frozen=True)
class BlockStack:
    """One block stack a POSIX arm's filesystem can sit on.

    ``kernel`` is the guest kernel variant whose image carries the module, None
    for a stack that needs no module. ``default_capacity_gb`` is the *floor*
    under the window of a run that does not size itself, None for a stack that
    imposes no window of its own. ``ack_semantics`` (and its one-token
    ``ack_summary``) is what an acknowledged operation guarantees here, and
    ``notes`` what a row off this stack does *not* include -- both ride in every
    run's flags sidecar. ``in_default_sweep`` is False for a stack that runs on
    the default image but is a second configuration of one already swept.
    """

    id: str
    label: str
    kernel: Optional[str]
    default_capacity_gb: Optional[int]
    ack_semantics: str
    ack_summary: str = ""
    notes: tuple[str, ...] = ()
    in_default_sweep: bool = True

    @property
    def is_stacked(self) -> bool:
        """Does this stack put a device between the filesystem and the NVMe?"""
        return self.id not in (RAW_STACK, DEFAULT_STACK)

    @property
    def needs_default_guest(self) -> bool:
        """Does this stack run on the image a sweep boots by default?"""
        return self.kernel in (None, DEFAULT_GUEST_KERNEL)

    def path_component(self) -> str:
        """The result-path token, empty for `raw`/`ext4` so paths predating this
        dimension stay where they are."""
        return "" if not self.is_stacked else f"stack-{self.id}"

    def ack_record(self) -> dict:
        """The acknowledgement semantics as a run's flags sidecar carries them."""
        return {"summary": self.ack_summary, "detail": self.ack_semantics}


def _stack(
    stack_id: str,
    label: str,
    ack_summary: str,
    ack_semantics: str,
    notes: tuple[str, ...] = (),
    in_default_sweep: bool = True,
) -> BlockStack:
    """Declare a stack, taking its kernel and window from ``block_stacks``."""
    return BlockStack(
        id=stack_id,
        label=label,
        kernel=block_stacks.required_kernel(stack_id),
        default_capacity_gb=block_stacks.default_capacity_gb(stack_id),
        ack_semantics=ack_semantics,
        ack_summary=ack_summary,
        notes=notes,
        in_default_sweep=in_default_sweep,
    )


def _luks_stacks() -> tuple[BlockStack, ...]:
    """One stack per LUKS mode, generated from ``block_stacks.LUKS_MODES``."""
    return tuple(
        BlockStack(
            id=mode.id,
            label=mode.label,
            # dm-crypt is in every image and has no ceiling, so nothing to read.
            kernel=None,
            default_capacity_gb=None,
            ack_summary="device-if-direct",
            ack_semantics=(
                "dm-crypt adds no durability point of its own: it encrypts the "
                "bio and passes it down, flush and FUA included, so an "
                "acknowledged write means exactly what it means on plain ext4 "
                "-- the device only under --direct and/or --fsync-mode, with "
                "metadata journalled asynchronously either way"
            ),
            notes=mode.notes
            + (
                "no ceiling and no module of its own: dm-crypt is"
                " length-preserving, keeps nothing per block in guest RAM and"
                " lives in the default 6.17 image, so it takes whatever window"
                " the row's other arms are given and costs a sweep only its VM"
                " boot",
                "the LUKS2 header (16 MiB by default) comes off the top of"
                " whatever window the partition is given, so this arm's"
                " filesystem is that much smaller than the plain-ext4 arm's at"
                " the same capacity. The passphrase is a fixture like the other"
                " stacks' keys -- the arm measures the data path, it protects"
                " nothing",
            ),
        )
        for mode in block_stacks.LUKS_MODES.values()
    )


BLOCK_STACKS: dict[str, BlockStack] = {
    stack.id: stack
    for stack in (
        _stack(
            DEFAULT_STACK,
            "ext4 on the raw NVMe",
            "device-if-direct",
            "an acknowledged write has reached the device only under --direct "
            "and/or --fsync-mode; ext4 metadata is journalled asynchronously "
            "either way, so a barrier is what makes a generation durable",
        ),
        *_luks_stacks(),
        _stack(
            "rollbaccine",
            "ext4 on rollbaccine (dm)",
            "backup-memory",
            "rollbaccine acknowledges a write once it is replicated to the "
            "backup's MEMORY, not once it is persisted on the device - a weaker "
            "promise than either AccelStore arm makes, and it is not visible in "
            "the throughput number",
            notes=(
                "single node, f=0: no replica, so the arm gives confidentiality"
                " and integrity but *no rollback resistance* -- freshness in"
                " rollbaccine comes from the replicated log. The authors' own"
                " eval runs f=1 with a second node over TCP; a faithful"
                " freshness row would need that topology, which the harness has"
                " no shape for",
                "the whole AES-GCM checksum tree is in guest RAM at the"
                " authors' own disk_pages_for_merkle_tree=0, costing"
                " window/256 bytes -- 2.0 GB at the 512 GB default window. It"
                " comes out of the same guest memory the benchmark sizes itself"
                " from (--client-host-memory-in-gb), so a larger window is a"
                " question about the guest and not about the device: a sweep"
                " takes the window its rows need, states the tree next to the"
                " row, and refuses only one that would take a real bite out of"
                " that memory (~1.6 TB on a 64 GB guest, the whole device on a"
                " 256 GB one)",
                "flush and FUA are honoured (num_flush_bios=1, the default sync"
                " mode passes REQ_PREFLUSH|REQ_FUA through), which is what makes"
                " this the durability-comparable stacked arm. The accelstore"
                " arms run their sync gate non-blocking by default, so neither"
                " side of the pair claims per-op durability",
            ),
        ),
        _stack(
            block_stacks.ROLLBACCINE_F1M_STACK,
            "ext4 on rollbaccine, f=1 (checksum replica)",
            "backup-memory",
            "an acknowledged fsync has been replicated to a second rollbaccine "
            "node's MEMORY -- that node holds the checksum tree and nothing "
            "else, so what the ack means is that a rollback of this device "
            "would be *detected*, not that any second copy of the data exists. "
            "Non-fsync writes ack on the local device as on the single-node "
            "arm; only a barrier waits (rollbaccine.c:1845)",
            notes=(
                "f=1 with a real backup, which is what the single-node"
                " `rollbaccine` arm does not measure: at f=0 not even an fsync"
                " waits for anything but the local device, so no row off that"
                " arm ever paid for replication. The two arms are different"
                " questions and neither supersedes the other",
                "only_replicate_checksums on both nodes: the backup receives"
                " metadata and checksums, allocates no bio and submits nothing"
                " to its device, so it can *detect* a rolled-back primary but"
                " cannot repair one -- rollbaccine's recovery fetches pages"
                " from the counterpart, which this backup does not have. That"
                " is deliberately the guarantee the accelstore arms make"
                " (freshness detection against an external attestation"
                " service, no repair), which is what makes the two comparable",
                "both nodes run in ONE guest over loopback, so the arm's"
                " replication RTT is far below a datacenter's and its fsync"
                " latency is an *understatement* of what the topology"
                " rollbaccine's own evaluation uses would cost. The deviation"
                " runs in the baseline's favour. Checksums-only is what makes"
                " it defensible at all: the wire carries 0.5% of a 128 KiB"
                " write, so loopback is not standing in for a NIC that would"
                " have had to move GB/s",
                "colocation also means the arm does not *deploy* the security"
                " it measures -- one host rolls back both nodes. It measures"
                " what the configuration costs, which is the number the"
                " comparison needs",
                "the backup competes for guest CPU: one HMAC-SHA256 verify and"
                " one tree memcpy per message, on the cores the benchmark also"
                " runs on. No AES-GCM -- that is the pages path, which is off",
                "two full checksum trees in guest RAM rather than one"
                " (window/256 bytes each, 4.0 GB together at the 512 GB"
                " default window), out of the same memory the benchmark sizes"
                " itself from",
                "rollbaccine reads uninitialised slab memory on its replica"
                " path -- alloc_bio_data() does not zero, and the receive path"
                " never assigns bio_data->shallow_clone, which free_bio_data()"
                " then tests against NULL. It is safe here only because the"
                " guest kernel sets CONFIG_INIT_ON_ALLOC_DEFAULT_ON=y, which"
                " the setup asserts rather than assumes",
            ),
            in_default_sweep=False,  # a second configuration of `rollbaccine`
        ),
    )
}

# The ids, in declaration order: what a `--backends` / `--stacks` list may name.
BLOCK_STACK_IDS = tuple(BLOCK_STACKS)

# The stacks whose module lives in the image a sweep boots by default.
DEFAULT_GUEST_STACK_IDS = tuple(
    s.id for s in BLOCK_STACKS.values() if s.needs_default_guest
)

# What a sweep runs unasked. A run that names an excluded stack still gets it.
DEFAULT_SWEEP_STACK_IDS = tuple(
    s.id for s in BLOCK_STACKS.values() if s.needs_default_guest and s.in_default_sweep
)

# Stack id -> guest kernel variant, for the stacks needing a non-default image.
GUEST_VARIANTS = {
    s.id: s.kernel for s in BLOCK_STACKS.values() if not s.needs_default_guest
}


def block_stack(stack_id) -> BlockStack:
    """Resolve a stack id (or stack) against the table."""
    if isinstance(stack_id, BlockStack):
        return stack_id
    if stack_id not in BLOCK_STACKS:
        raise ValueError(
            f"Unknown block stack '{stack_id}'. Known: {', '.join(BLOCK_STACK_IDS)}"
        )
    return BLOCK_STACKS[stack_id]
