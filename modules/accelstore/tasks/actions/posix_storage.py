#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Setup/teardown for each block stack ``tasks/storage_stacks.py`` declares:
format (plus device stacking), mount, hand back an empty data directory.

The device is re-formatted on every run and wiped on cleanup, and
:func:`posix_stack` pairs the two. ``capacity_gb`` (GB, 10^9) windows the
storage, None hands over the whole device.
"""

from contextlib import contextmanager
from functools import partial
from typing import Iterator, Optional

from modules.accelstore.tasks.block_stacks import (
    LUKS_MODES,
    LUKS_PASSPHRASE,
    ROLLBACCINE_BACKUP_DM_NAME,
    ROLLBACCINE_BACKUP_ZERO_DM_NAME,
    ROLLBACCINE_DM_NAME,
    ROLLBACCINE_NUM_NICS,
    ROLLBACCINE_PORT,
    ROLLBACCINE_F1M_STACK,
    ROLLBACCINE_TREE_COPIES,
    luks_dm_name,
    luks_format_argv,
    luks_mode,
    rollbaccine_backup_table,
    rollbaccine_primary_table,
    rollbaccine_table,
    rollbaccine_tree_bytes,
    rollbaccine_zero_table,
    window_sectors,
)
from modules.accelstore.tasks.storage_stacks import (
    BLOCK_STACKS,
    DEFAULT_SWEEP_STACK_IDS,
    GUEST_VARIANTS,
    block_stack,
)
from modules.nvme.tasks.utils.storage import (
    create_partition,
    get_partition_name,
    make_filesystem,
)

# A subdirectory, not the mount point: lost+found would make that non-empty.
POSIX_MOUNT = "/mnt/bench"
POSIX_DATA_DIR = f"{POSIX_MOUNT}/data"


def _mount_data_dir(vm, blockdev: str) -> str:
    """Mount a freshly formatted device at POSIX_MOUNT, return the data dir."""
    vm.ssh_cmd(["mkdir", "-p", POSIX_MOUNT], check=True)
    vm.ssh_cmd(["mount", blockdev, POSIX_MOUNT], check=True)
    vm.ssh_cmd(["mkdir", POSIX_DATA_DIR], check=True)
    return POSIX_DATA_DIR


def _data_partition(vm, device: str, capacity_gb=None) -> str:
    """Create one GPT partition over the device or a `capacity_gb` window."""
    end = "100%" if capacity_gb is None else f"{int(capacity_gb)}GB"
    create_partition(vm, device, [("0%", end)])
    return get_partition_name(device, 1)


def _setup_ext4_dir(vm, device: str, capacity_gb=None) -> str:
    """Fresh ext4 on the device, mounted at POSIX_MOUNT."""
    partition = _data_partition(vm, device, capacity_gb)
    make_filesystem(vm, partition, "ext4")
    return _mount_data_dir(vm, partition)


def _setup_luks_dir(mode, vm, device: str, capacity_gb=None) -> str:
    """Set up `mode`'s LUKS2 container with ext4 on top, return the data dir."""
    partition = _data_partition(vm, device, capacity_gb)

    # Explicit though DM would autoload it: fails here, not inside cryptsetup.
    vm.ssh_cmd(["modprobe", "dm_crypt"], check=True)
    print(f"{mode.id}: cryptsetup {mode.cipher}, integrity {mode.integrity or 'none'}")
    # cryptsetup asks for the passphrase twice at luksFormat, once at open.
    vm.ssh_cmd(
        luks_format_argv(mode, partition),
        check=True,
        input=f"{LUKS_PASSPHRASE}\n{LUKS_PASSPHRASE}\n",
    )
    dm_name = luks_dm_name(mode)
    vm.ssh_cmd(
        ["cryptsetup", "open", partition, dm_name],
        check=True,
        input=f"{LUKS_PASSPHRASE}\n",
    )
    dm_path = f"/dev/mapper/{dm_name}"
    make_filesystem(vm, dm_path, "ext4")
    return _mount_data_dir(vm, dm_path)


def _teardown_luks(mode, vm) -> None:
    # The open container holds the partition, so the cleanup's wipefs would fail.
    vm.ssh_cmd(["cryptsetup", "close", luks_dm_name(mode)], check=False)


def _setup_rollbaccine_dir(vm, device: str, capacity_gb=None) -> str:
    """Set up a standalone rollbaccine dm target with ext4 on top."""
    vm.ssh_cmd(["modprobe", "rollbaccine"], check=True)
    if capacity_gb is None:
        result = vm.ssh_cmd(["blockdev", "--getsz", device], check=True)
        sectors = int(result.stdout.strip())
    else:
        sectors = window_sectors(capacity_gb)
    print(
        f"rollbaccine: {sectors} sectors mapped, "
        f"{rollbaccine_tree_bytes(sectors) / 1e9:.2f} GB of checksum tree in RAM"
    )
    table = rollbaccine_table(device, sectors)
    vm.ssh_cmd(["dmsetup", "create", ROLLBACCINE_DM_NAME, "--table", table], check=True)
    dm_path = f"/dev/mapper/{ROLLBACCINE_DM_NAME}"
    make_filesystem(vm, dm_path, "ext4")
    return _mount_data_dir(vm, dm_path)


def _teardown_rollbaccine(vm) -> None:
    vm.ssh_cmd(["dmsetup", "remove", ROLLBACCINE_DM_NAME], check=False)


# /proc/net/tcp writes ports as upper-case hex.
_ROLLBACCINE_PORT_HEX = f"{ROLLBACCINE_PORT:04X}"

# Local port in state 01 (ESTABLISHED, the listener is 0A) only: both nodes
# share this namespace, so matching either column would double-count.
_ROLLBACCINE_WAIT_CONNECTED = f"""
for _ in $(seq 1 120); do
  n=$(awk '$4 == "01" && $2 ~ /:{_ROLLBACCINE_PORT_HEX}$/' /proc/net/tcp | wc -l)
  [ "$n" -ge {ROLLBACCINE_NUM_NICS} ] && exit 0
  sleep 0.5
done
exit 1
"""


def _require_init_on_alloc(vm) -> None:
    """Fail unless the guest kernel sets CONFIG_INIT_ON_ALLOC_DEFAULT_ON=y."""
    probe = vm.ssh_cmd(
        [
            "sh",
            "-c",
            "zcat /proc/config.gz 2>/dev/null || cat /proc/config 2>/dev/null",
        ],
        check=False,
    )
    config = probe.stdout or ""
    if not config:
        print(
            "WARNING: cannot read the guest kernel config, so "
            "CONFIG_INIT_ON_ALLOC_DEFAULT_ON is unverified. rollbaccine's "
            "replica path reads uninitialised slab memory without it"
        )
        return
    if "CONFIG_INIT_ON_ALLOC_DEFAULT_ON=y" not in config:
        raise RuntimeError(
            "The guest kernel does not set CONFIG_INIT_ON_ALLOC_DEFAULT_ON=y. "
            "rollbaccine's replica path reads uninitialised slab memory "
            "(alloc_bio_data() does not zero, and free_bio_data() tests "
            "bio_data->shallow_clone, which the receive path never assigns), "
            "so a backup on this kernel would crash rather than replicate. "
            "Boot with init_on_alloc=1 or rebuild the guest with that config."
        )


def _setup_rollbaccine_f1m_dir(vm, device: str, capacity_gb=None) -> str:
    """Set up rollbaccine with replication: leader on the device, backup on dm-zero."""
    _require_init_on_alloc(vm)
    vm.ssh_cmd(["modprobe", "rollbaccine"], check=True)
    if capacity_gb is None:
        result = vm.ssh_cmd(["blockdev", "--getsz", device], check=True)
        sectors = int(result.stdout.strip())
    else:
        sectors = window_sectors(capacity_gb)
    trees = ROLLBACCINE_TREE_COPIES[ROLLBACCINE_F1M_STACK]
    print(
        f"{ROLLBACCINE_F1M_STACK}: {sectors} sectors mapped, "
        f"{trees * rollbaccine_tree_bytes(sectors) / 1e9:.2f} GB of checksum "
        f"tree in RAM ({trees} nodes, both in this guest)"
    )

    # Primary first: connect_to_server() makes exactly one attempt per socket.
    vm.ssh_cmd(
        [
            "dmsetup",
            "create",
            ROLLBACCINE_DM_NAME,
            "--table",
            rollbaccine_primary_table(device, sectors),
        ],
        check=True,
    )
    vm.ssh_cmd(
        [
            "dmsetup",
            "create",
            ROLLBACCINE_BACKUP_ZERO_DM_NAME,
            "--table",
            rollbaccine_zero_table(sectors),
        ],
        check=True,
    )
    # --noudev*: a udev blkid scan's readahead gets -EIO from dm-zero.
    vm.ssh_cmd(
        [
            "dmsetup",
            "create",
            "--noudevrules",
            "--noudevsync",
            ROLLBACCINE_BACKUP_DM_NAME,
            "--table",
            rollbaccine_backup_table(
                f"/dev/mapper/{ROLLBACCINE_BACKUP_ZERO_DM_NAME}", sectors
            ),
        ],
        check=True,
    )
    connected = vm.ssh_cmd(["sh", "-c", _ROLLBACCINE_WAIT_CONNECTED], check=False)
    if connected.returncode != 0:
        raise RuntimeError(
            f"The rollbaccine backup did not open its {ROLLBACCINE_NUM_NICS} "
            f"replication sockets to port {ROLLBACCINE_PORT} within 60 s. "
            "connect_to_server() does not retry, so this is terminal: the "
            "first fsync of mkfs would block forever. Check the guest's dmesg "
            "for the rollbaccine handshake"
        )

    dm_path = f"/dev/mapper/{ROLLBACCINE_DM_NAME}"
    make_filesystem(vm, dm_path, "ext4")
    return _mount_data_dir(vm, dm_path)


def _teardown_rollbaccine_f1m(vm) -> None:
    # Backup first: it holds the dm-zero device open.
    vm.ssh_cmd(["dmsetup", "remove", ROLLBACCINE_BACKUP_DM_NAME], check=False)
    vm.ssh_cmd(["dmsetup", "remove", ROLLBACCINE_BACKUP_ZERO_DM_NAME], check=False)
    vm.ssh_cmd(["dmsetup", "remove", ROLLBACCINE_DM_NAME], check=False)


# (vm, device, capacity_gb) -> data directory.
POSIX_STORAGE_SETUPS = {
    "ext4": _setup_ext4_dir,
    **{m.id: partial(_setup_luks_dir, m) for m in LUKS_MODES.values()},
    "rollbaccine": _setup_rollbaccine_dir,
    "rollbaccine-f1m": _setup_rollbaccine_f1m_dir,
}

# Between umount and wipefs: a dm target holds the NVMe open until removed.
POSIX_STORAGE_TEARDOWNS = {
    **{m.id: partial(_teardown_luks, m) for m in LUKS_MODES.values()},
    "rollbaccine": _teardown_rollbaccine,
    "rollbaccine-f1m": _teardown_rollbaccine_f1m,
}

assert set(POSIX_STORAGE_SETUPS) == set(BLOCK_STACKS), (
    "storage_stacks.BLOCK_STACKS and POSIX_STORAGE_SETUPS disagree: "
    f"declared-only {sorted(set(BLOCK_STACKS) - set(POSIX_STORAGE_SETUPS))}, "
    f"implemented-only {sorted(set(POSIX_STORAGE_SETUPS) - set(BLOCK_STACKS))}"
)
assert set(POSIX_STORAGE_TEARDOWNS) <= set(POSIX_STORAGE_SETUPS)

POSIX_STORAGE_GUEST_VARIANTS = dict(GUEST_VARIANTS)

POSIX_BACKENDS = tuple(POSIX_STORAGE_SETUPS)

POSIX_DEFAULT_BACKENDS = tuple(
    b for b in POSIX_BACKENDS if b in DEFAULT_SWEEP_STACK_IDS
)


def setup_posix_storage(vm, backend: str, device: str, capacity_gb=None) -> str:
    """Set up `backend` on `device`, return the empty data directory."""
    return POSIX_STORAGE_SETUPS[backend](vm, device, capacity_gb)


def posix_storage_record(backend: str, device: str, capacity_gb=None) -> dict:
    """Return what setup_posix_storage will build, for a run's flags sidecar.

    Pure: derived from the same helpers the setup calls, so it needs no VM.
    """
    stack = block_stack(backend)
    record = {
        "backend": backend,
        "dev_path": device,
        "capacity_gb": capacity_gb,
        "label": stack.label,
        "ack_semantics": stack.ack_record(),
    }
    if backend in LUKS_MODES:
        mode = luks_mode(backend)
        record.update(
            {
                "cipher": mode.cipher,
                "integrity": mode.integrity,
                "key_size": mode.key_size,
                "dm_name": luks_dm_name(mode),
            }
        )
    elif backend == "rollbaccine":
        sectors = None if capacity_gb is None else window_sectors(capacity_gb)
        record.update(
            {
                "window_sectors": sectors,
                "table": rollbaccine_table(device, sectors if sectors else "<device>"),
                "merkle_tree_ram_bytes": (
                    None if sectors is None else rollbaccine_tree_bytes(sectors)
                ),
                "replicas": 0,
            }
        )
    elif backend == ROLLBACCINE_F1M_STACK:
        sectors = None if capacity_gb is None else window_sectors(capacity_gb)
        zero = f"/dev/mapper/{ROLLBACCINE_BACKUP_ZERO_DM_NAME}"
        record.update(
            {
                "window_sectors": sectors,
                "table": rollbaccine_primary_table(
                    device, sectors if sectors else "<device>"
                ),
                "backup_table": rollbaccine_backup_table(
                    zero, sectors if sectors else "<device>"
                ),
                "backup_backing_table": rollbaccine_zero_table(
                    sectors if sectors else "<device>"
                ),
                # Per node, and both nodes live in this guest.
                "merkle_tree_ram_bytes": (
                    None if sectors is None else rollbaccine_tree_bytes(sectors)
                ),
                "replicas": 1,
                "only_replicate_checksums": True,
                "replica_colocated": True,
            }
        )
    return record


def cleanup_posix_storage(vm, backend: str, device: str) -> None:
    """Undo setup_posix_storage so the next run finds a bare device."""
    vm.ssh_cmd(["umount", POSIX_MOUNT], check=False)
    teardown = POSIX_STORAGE_TEARDOWNS.get(backend)
    if teardown is not None:
        teardown(vm)
    vm.ssh_cmd(["wipefs", "-a", device], check=False)


@contextmanager
def posix_stack(
    vm, stack_id: str, device: str, capacity_gb: Optional[int] = None
) -> Iterator[str]:
    """Yield the data directory of `stack_id` on `device`, torn down afterwards.

    The teardown also runs when the setup raised, since a half-built stack
    still holds the device open.
    """
    try:
        yield setup_posix_storage(vm, stack_id, device, capacity_gb)
    finally:
        cleanup_posix_storage(vm, stack_id, device)
