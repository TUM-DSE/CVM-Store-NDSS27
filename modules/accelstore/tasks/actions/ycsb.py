#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ycsb_client runs: one YCSB point (``tasks/ycsb.py``) per boot, on one arm.

Arms are ``accelstore`` (the KVS module in the usual SPDK lifecycle) or a block
stack id from ``tasks/storage_stacks.py``, which runs ``ycsb_client --backend
rocksdb`` against a directory ``actions/posix_storage.py`` sets up.

Every run writes the client's ``--json`` metrics sidecar and a ``*_flags.json``
recording the arm's resolved flags, store geometry, stack, and ack semantics.
"""

import json
import shlex
import subprocess
from dataclasses import asdict, is_dataclass
from typing import Optional

from core.tasks.actions import ActionContext
from core.tasks.actions.registry import register_action

from modules.accelstore.tasks import ycsb as ycsb_grid
from modules.accelstore.tasks.block_stacks import check_capacity
from modules.accelstore.tasks.crypto_backend import CryptoBackend
from modules.accelstore.tasks.storage_stacks import BLOCK_STACKS
from modules.accelstore.tasks.sync_modes import SyncMode
from modules.accelstore.tasks.actions.accelstore import _save_output_to_file
from modules.accelstore.tasks.actions.bench import _write_flags_sidecar
from modules.accelstore.tasks.actions.posix_storage import posix_stack
from modules.accelstore.tasks.actions.storage_env import (
    precondition_device,
    spdk_device,
)

YCSB_CLIENT = "ycsb_client"

# Passed explicitly so the samplers name the path this action chose.
YCSB_RPC_SOCKET = "/var/tmp/ycsb_client.sock"

# ycsb_client opens RocksDB with `error_if_exists`, and the data dir exists.
ROCKSDB_SUBDIR = "db"

# ycsb_client's own defaults, restated to tell an override from a default.
DEFAULT_SEED = 42
DEFAULT_ZIPF_THETA = 0.99

# `ycsb_tasks._check_client_memory` subtracts the same number.
DEFAULT_HUGEPAGES_GB = 32

YCSB_ARMS = ycsb_grid.ycsb_arms(BLOCK_STACKS)
ACCELSTORE_ARM = ycsb_grid.ACCELSTORE_ARM

ACCELSTORE_ACK = {
    "summary": "device",
    "detail": (
        "the value of an acknowledged PUT has reached the device; its metadata "
        "durability point is whatever --sync-* configured (default: the epoch "
        "at as_free(), plus the KVS module's own space threshold)"
    ),
}
ROCKSDB_ACK = {
    "summary": "wal",
    "detail": (
        "an acknowledged PUT is in RocksDB's WAL, which --sync-* decides the "
        "durability of: on-shutdown leaves it unsynced until close, per-op is "
        "WriteOptions.sync, every-n/every-t is an inline rocksdb_flush_wal(sync) "
        "on the worker crossing the boundary. Whether that WAL write is durable "
        "on the medium is then the block stack's promise, below"
    ),
}


def _variant_tokens(cfg: dict) -> list:
    """Return the non-default knobs as result-leaf tokens."""
    tokens = []
    leaf = SyncMode.from_config(cfg).path_component()
    if leaf:
        tokens.append(leaf)

    threads = cfg.get("threads")
    if threads is not None:
        # No default comparison: a forced-1 run is not a default run.
        tokens.append(f"threads-{int(threads)}")

    if cfg.get("uniform"):
        tokens.append("uniform")
    theta = cfg.get("zipf_theta")
    if theta is not None and float(theta) != DEFAULT_ZIPF_THETA:
        tokens.append(f"theta-{str(theta).replace('.', 'p')}")
    seed = cfg.get("seed")
    if seed is not None and int(seed) != DEFAULT_SEED:
        tokens.append(f"seed-{int(seed)}")

    point = ycsb_grid.YCSB_POINTS.get(cfg.get("point", ""))
    if point is not None:
        for key, token in (
            ("value_size", "value"),
            ("num_keys", "keys"),
            ("num_ops", "ops"),
        ):
            value = cfg.get(key)
            if value is not None and int(value) != getattr(point, key):
                rendered = (
                    ycsb_grid.format_bytes(int(value))
                    if key == "value_size"
                    else int(value)
                )
                tokens.append(f"{token}-{rendered}")
    return tokens


def _ycsb_path(name: str, cfg: dict) -> tuple:
    arm = cfg.get("arm", ACCELSTORE_ARM)
    if arm == ACCELSTORE_ARM:
        crypto = CryptoBackend.parse(cfg.get("crypto") or CryptoBackend.AEGIS)
        arm_component = crypto.client_binary(YCSB_CLIENT)
    else:
        arm_component = arm
    components = ["ycsb", name, arm_component, cfg.get("point", "")]
    tokens = _variant_tokens(cfg)
    if cfg.get("quick"):
        tokens.append("quick")
    if tokens:
        components.append("_".join(tokens))
    return tuple(components)


@register_action("ycsb", path_fn=_ycsb_path)
def run_ycsb(
    ctx: ActionContext,
    point: str,
    arm: str = ACCELSTORE_ARM,
    crypto: str = CryptoBackend.AEGIS.value,
    config: str = None,
    pci_device: str = None,
    dev_path: str = None,
    device: str = "Nvme1n1",
    hugepages_gb: int = DEFAULT_HUGEPAGES_GB,
    value_size: Optional[int] = None,
    num_keys: Optional[int] = None,
    num_ops: Optional[int] = None,
    quick: bool = False,
    threads: Optional[int] = None,
    seed: Optional[int] = None,
    uniform: bool = False,
    zipf_theta: Optional[float] = None,
    validate: bool = False,
    per_worker: bool = True,
    capacity_gb: Optional[int] = None,
    guest_memory_gb: float = 0.0,
    skip_precondition: bool = False,
    precondition_scheme: str = None,
    ycsb_args: Optional[str] = None,
    trigger: Optional[str] = None,
    every_n: Optional[int] = None,
    interval_us: Optional[int] = None,
    coupling: Optional[str] = None,
    commit_every: Optional[int] = None,
    blocking: bool = False,
    metadata_only: bool = False,
    tc_backend: Optional[str] = None,
    tc_rtt_us: Optional[int] = None,
    tc_rtt_model: Optional[str] = None,
    tc_rtt_interval: Optional[int] = None,
):
    """Run one YCSB point on one arm.

    Args:
        ctx: Action context (VM/host runner, output dirs, timestamp)
        point: Point name from ``tasks/ycsb.py``'s table (A_4k .. A_1g for the
            workload-A size sweep, B_4k .. F_1m for the grid)
        arm: "accelstore", or a block stack id from BLOCK_STACKS (ext4,
            ext4-aes-xts-plain, rollbaccine, rollbaccine-f1m) -- the stack the
            RocksDB baseline's filesystem sits on
        crypto: Crypto build variant of the accelstore arm (default aegis). A
            RocksDB arm always runs the default build's binary
        config: SPDK bdev configuration as JSON string (accelstore arm only)
        pci_device: PCI address the bdev config attaches (accelstore arm only)
        dev_path: Guest block device path -- pre-conditioning target on both
            arms, and what a RocksDB arm's stack formats
        device: SPDK block device name created by the bdev config
        hugepages_gb: Hugepages reserved for the accelstore arm's SPDK app
        value_size / num_keys / num_ops: Override the point's own numbers. Each
            one that departs from the grid adds a token to the result leaf
        quick: Add a "quick" token to the result leaf (``paper.* --quick`` runs
            a reduced point set). Affects the result path only
        threads: Workers running the mix concurrently (--threads), 1..64 and at
            most 63 on the accelstore arm
        seed: Stream seed (--seed, client default 42)
        uniform: Uniform key selection instead of Zipfian (--uniform)
        zipf_theta: Zipfian skew, strictly between 0 and 1 (--zipf-theta,
            client default 0.99). Mutually exclusive with uniform
        validate: Walk every read byte against the pattern (--validate). Off by
            default: the walk lands on the phase wall clock
        per_worker: Pass --per-worker (default True), adding the per-worker rows
            to the log
        capacity_gb: Window the RocksDB arm's stack is given, in GB (10^9).
            None hands over the whole device
        guest_memory_gb: The guest's RAM, for the rollbaccine window check
        skip_precondition: Skip secure erase + pre-conditioning
        precondition_scheme: Pre-conditioning scheme to use
        ycsb_args: Extra ycsb_client arguments appended verbatim on every arm
        trigger .. tc_rtt_interval: One SyncMode. The trigger group goes to BOTH
            arms, the counter-specific half only to the accelstore arm

    Output structure:
        bench-result/ycsb/{name}/{crypto-prefixed ycsb_client | stack}/{point}/[variant]/{timestamp}.log
                                                                                        /{timestamp}.json
                                                                                        /{timestamp}_flags.json

    ``.json`` is ycsb_client's own metrics sidecar, ``_flags.json`` this
    action's record of the run.
    """
    if arm not in YCSB_ARMS:
        raise ValueError(f"Invalid arm '{arm}'. Must be one of: {', '.join(YCSB_ARMS)}")
    is_accelstore = arm == ACCELSTORE_ARM
    stack = None if is_accelstore else BLOCK_STACKS[arm]

    resolved = ycsb_grid.ycsb_point(point).with_overrides(
        value_size=value_size, num_keys=num_keys, num_ops=num_ops
    )

    # On the resolved point: an override can exceed a limit no table entry does.
    if not is_accelstore:
        refusal = ycsb_grid.rocksdb_rejects(resolved)
        if refusal:
            raise ValueError(refusal)

    if uniform and zipf_theta is not None:
        raise ValueError(
            "uniform and zipf_theta name two different key distributions; pass one"
        )
    if zipf_theta is not None and not 0.0 < float(zipf_theta) < 1.0:
        raise ValueError(f"zipf_theta {zipf_theta} must be strictly between 0 and 1")

    if threads is not None:
        threads = int(threads)
        if not 1 <= threads <= 64:
            raise ValueError(f"threads {threads} out of range [1, 64]")
        if threads > 63 and is_accelstore:
            # Refused here: ycsb_client only refuses after the boot and erase.
            raise ValueError(
                f"threads {threads} exceeds the accelstore arm's 63 workers"
            )

    sync = SyncMode(
        trigger=trigger,
        every_n=every_n,
        interval_us=interval_us,
        coupling=coupling,
        commit_every=commit_every,
        blocking=blocking,
        metadata_only=metadata_only,
        tc_backend=tc_backend,
        tc_rtt_us=tc_rtt_us,
        tc_rtt_model=tc_rtt_model,
        tc_rtt_interval=tc_rtt_interval,
    )
    sync.validate()
    check_sync_for_arm(sync, is_accelstore)
    for warning in sync.warnings():
        print(f"WARNING: {warning}")

    # Checked before anything is erased.
    if not is_accelstore:
        for warning in check_capacity(arm, capacity_gb, guest_memory_gb or None):
            print(f"NOTE: {warning}")

    date = ctx.timestamp
    vm = ctx.vm
    output_log = ctx.outputdir_guest / f"{date}.log"

    workload_args = resolved.workload_args()
    if uniform:
        workload_args.append("--uniform")
    elif zipf_theta is not None:
        workload_args += ["--zipf-theta", str(zipf_theta)]
    if seed is not None:
        workload_args += ["--seed", str(int(seed))]
    if threads is not None:
        workload_args += ["--threads", str(threads)]
    if validate:
        workload_args.append("--validate")
    if per_worker:
        workload_args.append("--per-worker")
    workload_args += ["--json", str(ctx.outputdir_guest / f"{date}.json")]
    workload_args += sync.client_args()
    workload_args += shlex.split(ycsb_args) if ycsb_args else []

    precondition_device(
        ctx, vm, dev_path, skip=skip_precondition, scheme=precondition_scheme
    )

    record = {
        "schema": 1,
        "timestamp": date,
        "client": YCSB_CLIENT,
        "arm": arm,
        "point": resolved.name,
        "point_config": resolved.as_dict(),
        "overrides": {
            key: value
            for key, value in (
                ("value_size", value_size),
                ("num_keys", num_keys),
                ("num_ops", num_ops),
            )
            if value is not None
        },
        "threads": threads,
        "seed": seed if seed is not None else DEFAULT_SEED,
        "dist": "uniform" if uniform else "zipfian",
        "zipf_theta": None if uniform else (zipf_theta or DEFAULT_ZIPF_THETA),
        "validate": validate,
        "per_worker": per_worker,
        "sync_mode": sync.label(),
        "sync_args": sync.client_args(),
        "workload_args": workload_args,
    }

    if is_accelstore:
        client_binary = CryptoBackend.parse(crypto).client_binary(YCSB_CLIENT)
        geometry_args = ycsb_grid.accelstore_point_args(resolved)
        record.update(
            {
                "backend": "accelstore",
                "binary": client_binary,
                "crypto": crypto,
                "device": device,
                "pci_device": pci_device,
                "store_geometry_args": geometry_args,
                "ack_semantics": ACCELSTORE_ACK,
            }
        )
        cmd = _run_accelstore_arm(
            ctx,
            client_binary,
            config,
            pci_device,
            device,
            hugepages_gb,
            geometry_args,
            workload_args,
            output_log,
        )
    else:
        record.update(
            {
                "backend": "rocksdb",
                # Unprefixed: the RocksDB backend is the same in every build.
                "binary": YCSB_CLIENT,
                "stack": _stack_record(stack),
                "capacity_gb": capacity_gb,
                "ack_semantics": {
                    "rocksdb": ROCKSDB_ACK,
                    "stack": _stack_ack(stack),
                },
            }
        )
        cmd = _run_rocksdb_arm(
            ctx, arm, dev_path, capacity_gb, workload_args, output_log
        )

    record["command"] = cmd
    _write_flags_sidecar(ctx, record)


def check_sync_for_arm(sync: SyncMode, is_accelstore: bool) -> None:
    """Refuse the sync settings this arm cannot honour, before the boot.

    Mirrors ``ycsb_check_sync()`` in clients/ycsb_client.c: ``on-barrier`` is
    refused on both arms, the counter-specific knobs on the RocksDB arm.
    """
    if sync.trigger == "on-barrier":
        raise ValueError(
            "sync trigger 'on-barrier' has no meaning for a YCSB mix: it "
            "contains no barrier operations, so the run would measure "
            "'on-shutdown' under another name (both arms refuse it)"
        )
    if is_accelstore:
        return
    counter_knobs = {
        "coupling": sync.coupling,
        "commit_every": sync.commit_every if (sync.commit_every or 1) > 1 else None,
        "metadata_only": sync.metadata_only or None,
        "tc_backend": sync.tc_backend,
        "tc_rtt_us": sync.tc_rtt_us,
        "tc_rtt_model": sync.tc_rtt_model,
        "tc_rtt_interval": sync.tc_rtt_interval,
    }
    named = sorted(k for k, v in counter_knobs.items() if v is not None)
    if named:
        raise ValueError(
            f"{', '.join(named)} schedule a trusted-counter commit against a "
            "metadata write, and the RocksDB arm has neither. Run the "
            "counter-policy sweep on the accelstore arm alone; the trigger "
            "group (trigger/every_n/interval_us/blocking) is what both arms "
            "share and it stays passed to both"
        )


def _json_safe(value):
    """Render a stack-table field JSON-safe, falling back to ``str``."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _json_safe(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _stack_ack(stack) -> object:
    """Return the stack's ack semantics, preferring ``ack_record()``."""
    if hasattr(stack, "ack_record"):
        return _json_safe(stack.ack_record())
    return _json_safe(getattr(stack, "ack_semantics", None))


def _stack_record(stack) -> dict:
    """Return the block stack's own fields for the flags sidecar."""
    return {
        "id": stack.id,
        "label": _json_safe(getattr(stack, "label", None)),
        "kernel": getattr(stack, "kernel", None),
        "default_capacity_gb": getattr(stack, "default_capacity_gb", None),
        "is_stacked": bool(getattr(stack, "is_stacked", False)),
        "notes": _json_safe(getattr(stack, "notes", ()) or ()),
    }


def _run_accelstore_arm(
    ctx,
    client_binary,
    config,
    pci_device,
    device,
    hugepages_gb,
    geometry_args,
    workload_args,
    output_log,
):
    if not config or not pci_device:
        raise ValueError("config and pci_device are required on the accelstore arm")

    try:
        json.loads(config)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON config: {e}")

    vm = ctx.vm

    # spdk_device destroys the blobstore: the load phase inserts every key once.
    with spdk_device(
        vm, config, pci_device, device, hugepages_gb=hugepages_gb
    ) as config_file:
        cmd = [
            client_binary,
            "--backend",
            "accelstore",
            "--config",
            config_file,
            "--device",
            device,
            "--rpc-socket",
            YCSB_RPC_SOCKET,
            *geometry_args,
            *workload_args,
        ]
        print(f"Running {client_binary} on the accelstore backend")
        result = vm.ssh_cmd(cmd, check=False, stderr=subprocess.PIPE)
        _save_output_to_file(
            vm, output_log, result.stdout, result.stderr, result.returncode
        )
        return cmd


def _run_rocksdb_arm(ctx, stack_id, dev_path, capacity_gb, workload_args, output_log):
    if not dev_path:
        raise ValueError(f"dev_path is required on the '{stack_id}' arm")

    vm = ctx.vm

    with posix_stack(vm, stack_id, dev_path, capacity_gb=capacity_gb) as data_dir:
        db_dir = f"{data_dir}/{ROCKSDB_SUBDIR}"
        # No store-geometry flags: the RocksDB arm refuses all but --sync-*.
        cmd = [
            YCSB_CLIENT,
            "--backend",
            "rocksdb",
            "--dir",
            db_dir,
            *workload_args,
        ]
        print(f"Running {YCSB_CLIENT} on rocksdb over {stack_id} at {db_dir}")
        result = vm.ssh_cmd(cmd, check=False, stderr=subprocess.PIPE)
        _save_output_to_file(
            vm, output_log, result.stdout, result.stderr, result.returncode
        )
        return cmd
