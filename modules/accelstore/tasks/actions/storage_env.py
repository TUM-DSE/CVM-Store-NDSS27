#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The per-run device setup the SPDK actions share.

Precondition the device while the kernel still owns it, write the bdev config
into the guest, bind vfio-pci, reserve hugepages, destroy whatever blobstore
the previous run left, and give the device back afterwards.
"""

import uuid
from contextlib import contextmanager
from typing import Optional

from core.tasks.qemu import setup_hugepages

from modules.nvme.tasks.utils.nvme import nvme_secure_erase, run_precondition


@contextmanager
def spdk_device(
    vm,
    config_json: str,
    pci_device: str,
    device: str,
    *,
    hugepages_gb: int = 32,
    destroy_blobstore: bool = True,
    destroy_silent: bool = True,
    bind: bool = True,
    bind_note: Optional[str] = None,
):
    """Hold the device set up for SPDK for the length of the block.

    Yields the guest path of the bdev config file the application is pointed at.

    ``bind`` is False for a run on the in-kernel driver: hugepages are reserved
    either way, but the namespace stays on the nvme driver and the teardown
    undoes only a bind this run performed. ``bind_note`` is printed instead.
    The 32 GB hugepage default is for the iobuf pools, which are sized per SPDK
    worker thread before spdk_app_start and never resized.
    """
    config_file = _create_spdk_config_file(vm, config_json)

    try:
        if bind:
            vm.ssh_cmd(
                ["spdk-setup"], check=True, extra_env={"PCI_ALLOWED": pci_device}
            )
        else:
            print(
                bind_note
                or f"Leaving {pci_device} on the kernel driver; hugepages only"
            )
        setup_hugepages(vm, total_size_gb=hugepages_gb)

        if destroy_blobstore:
            _run_blobstore_destroyer(vm, config_file, device, destroy_silent)

        yield config_file
    finally:
        if bind:
            vm.ssh_cmd(
                ["spdk-setup", "reset"],
                check=True,
                extra_env={"PCI_ALLOWED": pci_device},
            )
        _cleanup_config_file(vm, config_file)


def precondition_device(ctx, vm, dev_path, *, skip: bool, scheme: str) -> None:
    """Secure-erase and pre-condition the device.

    Runs before spdk_device() binds anything: nvme-cli needs the namespace on
    the kernel driver.
    """
    if skip:
        return
    if not dev_path:
        raise ValueError("dev_path is required for pre-conditioning")

    precond_dir = ctx.outputdir_guest / "precondition"
    (ctx.outputdir_host / "precondition").mkdir(parents=True, exist_ok=True)

    print(f"Secure erasing NVMe: {dev_path}")
    nvme_secure_erase(vm, dev_path)

    print(f"Running pre-conditioning on {dev_path}")
    run_precondition(
        vm,
        dev_path,
        output_dir=precond_dir,
        timestamp=ctx.timestamp,
        scheme=scheme,
    )


def _create_spdk_config_file(vm, config: str) -> str:
    """Write the bdev config to a temporary file on the VM and return its path."""
    config_file = f"/tmp/accelstore_config_{uuid.uuid4().hex[:8]}.json"
    vm.create_file(config_file, config)
    print(f"Created config file: {config_file}")
    return config_file


def _cleanup_config_file(vm, config_file: str) -> None:
    """Remove the temporary config file from the VM."""
    vm.ssh_cmd(["rm", "-f", config_file], check=False)
    print(f"Cleaned up config file: {config_file}")


def _run_blobstore_destroyer(vm, config_file: str, device: str, silent: bool) -> None:
    """Wipe the blobstore on the device before the run."""
    print(f"Running blobstore destroyer on device {device}")
    # Unprefixed on purpose: the superblock it wipes is the same for every backend.
    cmd = ["blobstore_destroyer", "-c", config_file, "-b", device]
    if not silent:
        cmd.append("-v")

    result = vm.ssh_cmd(cmd, check=False)

    if result.returncode != 0:
        print(f"Warning: blobstore destroyer exited with code {result.returncode}")
        if result.stderr:
            print(f"stderr: {result.stderr}")
    else:
        print(f"Successfully destroyed blobstore on device {device}")
