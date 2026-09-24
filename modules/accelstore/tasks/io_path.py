#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Which NVMe driver a run's bdev sits on.

The kernel-bypass axis: the same binary, blobstore and workload, driven either
over SPDK's userspace driver (``bdev_nvme_attach_controller``) or the kernel's
(``bdev_aio_create`` on the raw namespace).
"""

import json
from dataclasses import dataclass
from typing import Optional

# Different on purpose: logs, traces and result dirs then record the arm.
SPDK_DEVICE = "Nvme1n1"
KERNEL_DEVICE = "KernelNvme1n1"

SLUG_PREFIX = "io-"

SPDK = "spdk"
KERNEL = "kernel"
KINDS = (SPDK, KERNEL)


@dataclass(frozen=True)
class IoPath:
    """The I/O-path arm of one run.

    ``explicit`` separates a run outside any bypass sweep (keeps today's result
    path) from the sweep's own SPDK arm (gets the ``io-spdk`` directory).
    """

    kernel: bool = False
    explicit: bool = False

    @classmethod
    def parse(cls, spec) -> "IoPath":
        """Coerce a task argument to an arm: ``None`` (outside a sweep),
        ``"spdk"`` or ``"kernel"``.

        >>> IoPath.parse(None).is_default()
        True
        >>> IoPath.parse("kernel").device
        'KernelNvme1n1'
        """
        if isinstance(spec, cls):
            return spec
        if spec is None:
            return cls()

        text = str(spec).strip()
        if text not in KINDS:
            raise ValueError(
                f"io_path: unknown arm '{spec}'. Valid: {'|'.join(KINDS)},"
                " or omit the argument outside a kernel-bypass sweep"
            )
        return cls(kernel=text == KERNEL, explicit=True)

    @classmethod
    def from_config(cls, cfg: dict) -> "IoPath":
        """Pick the io_path key out of an action config, ignoring the rest."""
        return cls.parse(cfg.get("io_path"))

    @property
    def spec(self) -> Optional[str]:
        """The task/action argument this arm round-trips through."""
        if not self.explicit:
            return None
        return KERNEL if self.kernel else SPDK

    @property
    def device(self) -> str:
        """The bdev name this arm's config creates, i.e. ``--device``."""
        return KERNEL_DEVICE if self.kernel else SPDK_DEVICE

    def is_default(self) -> bool:
        """True for a run that is not part of a bypass sweep at all."""
        return not self.kernel and not self.explicit

    def bdev_config(self, pci_device: str, dev_path: Optional[str] = None) -> str:
        """The SPDK bdev configuration for this arm, as a JSON string.

        ``dev_path`` (``StorageTarget.dev_path``) is needed by the kernel arm
        and ignored by the SPDK arm. ``block_size`` is deliberately absent:
        ``bdev_aio`` reads it via ``ioctl(BLKSSZGET)``.
        """
        if self.kernel:
            if not dev_path:
                raise ValueError(
                    "io_path 'kernel' needs the guest device node (dev_path):"
                    " bdev_aio opens the namespace by path, since the kernel"
                    " driver -- not SPDK -- owns the PCI function in this arm"
                )
            params = {"name": KERNEL_DEVICE, "filename": dev_path}
            method = "bdev_aio_create"
        else:
            params = {
                "name": "Nvme1",
                "trtype": "pcie",
                "traddr": pci_device,
            }
            method = "bdev_nvme_attach_controller"

        return (
            json.dumps(
                {
                    "subsystems": [
                        {
                            "subsystem": "bdev",
                            "config": [
                                {"method": method, "params": params},
                                {"method": "bdev_wait_for_examine"},
                            ],
                        }
                    ]
                },
                indent=2,
            )
            + "\n"
        )

    def as_config(self) -> dict:
        """The key to merge into an action_config; empty when not an arm."""
        return {} if self.spec is None else {"io_path": self.spec}

    def as_task_kwargs(self) -> dict:
        """The ``run-all`` argument for this arm, for a sweep to splat."""
        return self.as_config()

    def path_component(self) -> Optional[str]:
        """Stable slug identifying this arm in a result path, None outside a
        sweep."""
        if self.is_default():
            return None
        return SLUG_PREFIX + (KERNEL if self.kernel else SPDK)

    def label(self) -> str:
        return "kernel (bdev_aio)" if self.kernel else "spdk (bdev_nvme)"


def parse_path_component(slug: str) -> IoPath:
    """Decode a result-directory slug back into an arm.

    >>> parse_path_component("io-kernel").kernel
    True
    >>> parse_path_component("io-spdk").is_default()
    False
    """
    if not slug or not slug.startswith(SLUG_PREFIX):
        raise ValueError(f"not an io-path slug: '{slug}'")
    return IoPath.parse(slug[len(SLUG_PREFIX) :])


def is_io_path_component(name: str) -> bool:
    """Whether a directory name is (meant to be) one of this module's slugs."""
    if not name or not name.startswith(SLUG_PREFIX):
        return False
    try:
        parse_path_component(name)
    except ValueError:
        return False
    return True
