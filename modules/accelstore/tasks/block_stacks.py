#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Block-stack geometry: the capacity each stacked backend works at.

Separate from the code that drives the stacks (``actions/posix_storage.py``) so
the variant table and the tests can read it without a VM. rollbaccine keeps its
whole checksum tree in guest RAM (window / 256 bytes) and ``init_merkle_tree()``
drops its allocation failure, so a window is always sized, never the raw device.
"""

import math
from dataclasses import dataclass

# A GB is 10^9 here, as in every other capacity the harness passes around.
BYTES_PER_GB = 1_000_000_000

# ext4's journal, inode tables and 5% reserve take ~6.6% off a window's capacity.
FILESYSTEM_HEADROOM = 1.10
SECTORS_PER_PAGE = 8  # 4 KiB / 512 B, both dm targets address pages

# A floor and not a ceiling: rollbaccine's real limit is the guest RAM its tree takes.
ROLLBACCINE_DEFAULT_CAPACITY_GB = 512
ROLLBACCINE_F1M_STACK = "rollbaccine-f1m"
# Half of that, because the replicating arm keeps two trees in the same guest.
ROLLBACCINE_F1M_DEFAULT_CAPACITY_GB = 256
ROLLBACCINE_TREE_RAM_REFUSE_FRACTION = 0.10
# Above the default window's share (~3% of a 64 GB guest), so a normal run is quiet.
ROLLBACCINE_TREE_RAM_WARN_FRACTION = 0.05

DEFAULT_CAPACITIES_GB = {
    "rollbaccine": ROLLBACCINE_DEFAULT_CAPACITY_GB,
    ROLLBACCINE_F1M_STACK: ROLLBACCINE_F1M_DEFAULT_CAPACITY_GB,
}

# Windows a stack cannot work at whatever the guest looks like. None today.
HARD_CAPACITIES_GB: dict[str, int] = {}

# The kernel variant each backend's module is built against. Plain ext4 needs none.
REQUIRED_KERNEL = {
    "rollbaccine": "6.17",
    ROLLBACCINE_F1M_STACK: "6.17",
}

# The key is a fixture -- the arm measures the data path, it protects nothing.
ROLLBACCINE_DM_NAME = "rollbaccine1"
ROLLBACCINE_KEY = "abcdefghijklmnop"
ROLLBACCINE_PORT = 12340

# Both nodes share one network namespace, so the backup needs a port of its own.
ROLLBACCINE_BACKUP_DM_NAME = "rollbaccine2"
ROLLBACCINE_BACKUP_ZERO_DM_NAME = "rollbaccine-zero"
ROLLBACCINE_BACKUP_PORT = 12350
ROLLBACCINE_BACKUP_ADDR = "127.0.0.1"
ROLLBACCINE_PRIMARY_ID = 1
ROLLBACCINE_BACKUP_ID = 2
# Sockets per peer (`NUM_NICS`). `connect_to_server()` tries each exactly once.
ROLLBACCINE_NUM_NICS = 4

# In-RAM checksum trees a stack keeps in the guest, one per node it runs there.
ROLLBACCINE_TREE_COPIES = {
    "rollbaccine": 1,
    ROLLBACCINE_F1M_STACK: 2,
}

# A fixture like the two keys above -- these arms protect nothing.
LUKS_PASSPHRASE = "accelstore"


@dataclass(frozen=True)
class LuksMode:
    """One dm-crypt configuration ext4 can be put on.

    Everything a mode does not name is left at cryptsetup's defaults, so a
    deviation belongs in the declaration below and never in the setup code.
    `notes` is what a row off the mode does *not* include, in the shape the
    MlperfStack entries use.
    """

    id: str
    cipher: str
    label: str
    integrity: str | None = None
    key_size: int | None = None
    notes: tuple[str, ...] = ()


# Flag sets from OmaArvio's own LUKS formatters (modules/nvme/tasks/utils/storage.py).
LUKS_MODES = {
    mode.id: mode
    for mode in (
        LuksMode(
            id="ext4-aes-xts-plain",
            cipher="aes-xts-plain64",
            label="ext4 on LUKS2 dm-crypt (AES-XTS)",
            notes=(
                "confidentiality alone: XTS is unauthenticated and"
                " length-preserving, so a tampered ciphertext block decrypts to"
                " bytes ext4 cannot tell from data, and rolling the device back"
                " to an older image is undetectable. That is the point of the"
                " arm -- it is the encryption-only floor the integrity and"
                " freshness stacks are read against, not a system that claims"
                " what they claim",
                "cryptsetup's defaults throughout (LUKS2 header, 512 B sectors,"
                " per-CPU en/decrypt workqueues): the baseline is what an"
                " operator gets by typing the command, and a tuned dm-crypt"
                " would be a different row",
            ),
        ),
    )
}


def luks_mode(mode_id: str) -> LuksMode:
    """Resolve a LUKS backend/stack id against the table."""
    if mode_id not in LUKS_MODES:
        raise ValueError(
            f"Unknown LUKS mode '{mode_id}'. Known: {', '.join(LUKS_MODES)}"
        )
    return LUKS_MODES[mode_id]


def luks_dm_name(mode: LuksMode) -> str:
    """The /dev/mapper name a mode's container is opened under."""
    return f"luks-{mode.id}"


def luks_format_argv(mode: LuksMode, partition: str) -> list[str]:
    """The `cryptsetup luksFormat` command line a mode asks for."""
    argv = [
        "cryptsetup",
        "luksFormat",
        "--batch-mode",
        "--type=luks2",
        f"--cipher={mode.cipher}",
    ]
    if mode.integrity:
        argv.append(f"--integrity={mode.integrity}")
    if mode.key_size:
        argv.append(f"--key-size={mode.key_size}")
    argv.append(partition)
    return argv


def default_capacity_gb(backend: str):
    """The window `backend` is given when the run does not size one."""
    return DEFAULT_CAPACITIES_GB.get(backend)


def hard_capacity_gb(backend: str):
    """The window past which `backend` cannot work at all, None if it has none."""
    return HARD_CAPACITIES_GB.get(backend)


def required_kernel(backend: str):
    """The guest kernel variant `backend`'s module is built against, or None."""
    return REQUIRED_KERNEL.get(backend)


def window_sectors(capacity_gb) -> int:
    """A capacity in GB as a page-aligned sector count for a dm table."""
    sectors = int(capacity_gb) * BYTES_PER_GB // 512
    return sectors - (sectors % SECTORS_PER_PAGE)


def rollbaccine_tree_bytes(sectors: int) -> int:
    """The RAM `sectors` of rollbaccine mapping costs at disk_pages 0.

    init_merkle_tree()'s own arithmetic: one 4096 B checksum page per 2048
    sectors, all resident because no tree level is put on disk.
    """
    return ((sectors + 2047) // 2048) * 4096


def _rollbaccine_table(
    device: str,
    sectors,
    *,
    node_id: int,
    is_leader: bool,
    f: int,
    port: int,
    only_replicate_checksums: bool,
    counterpart_id: int,
    primary: tuple = (),
) -> str:
    """One rollbaccine dm table, positionals as ``rollbaccine_constructor()``.

    argv[0..13] of that constructor: device, node id, seen ballot, leader, the
    16-byte AES-GCM/HMAC key, f, merkle-tree disk pages, sync mode, listen port,
    checksum-only replication, recovering, counterpart id -- then, for a
    non-leader only, the primary's address and port. ``sectors`` and the
    disk-page count must be identical on both nodes of a pair.
    """
    return " ".join(
        str(token)
        for token in (
            0,
            sectors,
            "rollbaccine",
            device,
            node_id,
            1,
            "true" if is_leader else "false",
            ROLLBACCINE_KEY,
            f,
            0,
            "default",
            port,
            "true" if only_replicate_checksums else "false",
            "false",
            counterpart_id,
            *primary,
        )
    )


def rollbaccine_table(device: str, sectors) -> str:
    """The dm table for a standalone rollbaccine target of `sectors`.

    Node id 1, leader, f=0 -- no replicas, so the trailing arguments are inert,
    the arm gives no rollback resistance, and not even an fsync waits for
    anything but this node's own device.
    """
    return _rollbaccine_table(
        device,
        sectors,
        node_id=ROLLBACCINE_PRIMARY_ID,
        is_leader=True,
        f=0,
        port=ROLLBACCINE_PORT,
        only_replicate_checksums=False,
        counterpart_id=ROLLBACCINE_BACKUP_ID,
    )


def rollbaccine_primary_table(device: str, sectors) -> str:
    """The replicating arm's leader: f=1, checksums-only, on the real device.

    At f=1 an fsync is held until the backup acks it. The checksums-only flag
    must match the backup's -- the one value gates the sender's page send *and*
    the receiver's page receive, so a mismatched pair desynchronises the stream.
    """
    return _rollbaccine_table(
        device,
        sectors,
        node_id=ROLLBACCINE_PRIMARY_ID,
        is_leader=True,
        f=1,
        port=ROLLBACCINE_PORT,
        only_replicate_checksums=True,
        counterpart_id=ROLLBACCINE_BACKUP_ID,
    )


def rollbaccine_backup_table(device: str, sectors) -> str:
    """The replicating arm's backup, on `device` (a dm-zero of `sectors`).

    A non-leader takes two arguments the leader does not: the primary's address
    and port, which it dials once. Under checksums-only the backup's receive
    path never touches its device, so dm-zero is enough of a backing store.
    """
    return _rollbaccine_table(
        device,
        sectors,
        node_id=ROLLBACCINE_BACKUP_ID,
        is_leader=False,
        f=1,
        port=ROLLBACCINE_BACKUP_PORT,
        only_replicate_checksums=True,
        counterpart_id=ROLLBACCINE_PRIMARY_ID,
        primary=(ROLLBACCINE_BACKUP_ADDR, ROLLBACCINE_PORT),
    )


def rollbaccine_zero_table(sectors) -> str:
    """The dm-zero backing store under the backup: writes dropped, reads zero."""
    return f"0 {sectors} zero"


def sweep_window_gb(backends, needed_gb: int) -> int:
    """The window a sweep over `backends` gets for a demand of `needed_gb`.

    One figure serves every arm of every row: floored at the smallest default
    among the sweep's stacks, capped at the smallest hard ceiling, and raised
    by `FILESYSTEM_HEADROOM` in between.
    """
    defaults = [default_capacity_gb(b) for b in backends]
    if not defaults or any(d is None for d in defaults):
        raise ValueError(
            "sweep_window_gb wants stacked backends with a default window; "
            f"got {sorted(set(backends))}"
        )
    window = max(min(defaults), math.ceil(int(needed_gb or 0) * FILESYSTEM_HEADROOM))
    ceilings = [hard_capacity_gb(b) for b in backends]
    ceilings = [c for c in ceilings if c is not None]
    return min(window, min(ceilings)) if ceilings else window


def rollbaccine_tree_gb(capacity_gb, backend: str = "rollbaccine") -> float:
    """The resident checksum tree a `capacity_gb` mapping costs the guest, GB.

    Per `backend`, since the replicating arm runs both nodes in one guest and
    each keeps a full tree of its own.
    """
    copies = ROLLBACCINE_TREE_COPIES.get(backend, 1)
    return copies * rollbaccine_tree_bytes(window_sectors(capacity_gb)) / BYTES_PER_GB


def check_capacity(backend: str, capacity_gb, guest_memory_gb=None) -> list[str]:
    """Raise if `backend` cannot work at `capacity_gb`, return any warnings.

    A hard ceiling stops a stack working past a fixed size, while rollbaccine's
    window is bounded by the guest's memory instead.
    """
    warnings = []
    if capacity_gb is None:
        return warnings
    capacity_gb = int(capacity_gb)

    hard = hard_capacity_gb(backend)
    if hard is not None and capacity_gb > hard:
        raise ValueError(
            f"{backend} cannot be given {capacity_gb} GB: past {hard} GB the "
            "stack stops working at all (see tasks/block_stacks.py)"
        )
    default = default_capacity_gb(backend)

    if backend in ROLLBACCINE_TREE_COPIES:
        tree_gb = rollbaccine_tree_gb(capacity_gb, backend)
        trees = ROLLBACCINE_TREE_COPIES[backend]
        held = "checksum tree" if trees == 1 else f"{trees} checksum trees"
        if guest_memory_gb:
            fraction = tree_gb / float(guest_memory_gb)
            if fraction > ROLLBACCINE_TREE_RAM_REFUSE_FRACTION:
                raise ValueError(
                    f"{backend} at {capacity_gb} GB keeps {tree_gb:.1f} GB of "
                    f"{held} in RAM, {fraction:.0%} of a "
                    f"{guest_memory_gb:g} GB guest. That memory is also what "
                    "the benchmark sizes itself from "
                    "(--client-host-memory-in-gb), so the arms would not be "
                    "sized alike -- and an allocation failure is not "
                    "propagated by init_merkle_tree(), it just leaves a target "
                    "that dies on the first write. Lower the window or run a "
                    "larger guest (`--sizes large` is a 256 GB one, which holds "
                    "the tree for the whole device)"
                )
            if fraction > ROLLBACCINE_TREE_RAM_WARN_FRACTION:
                warnings.append(
                    f"{backend} at {capacity_gb} GB keeps {tree_gb:.1f} GB of "
                    f"{held} in RAM, {fraction:.0%} of the guest -- "
                    "state it next to the row"
                )
        elif capacity_gb > default:
            warnings.append(
                f"{backend} at {capacity_gb} GB keeps {tree_gb:.1f} GB of "
                f"{held} in RAM and no guest memory was given to check "
                "it against"
            )
    return warnings
