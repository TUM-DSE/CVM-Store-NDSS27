#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-backend colours and grouped bar geometry for the AccelStore plots."""

import zlib
from typing import List, Tuple

from core.tasks.plotting import common

from ..crypto_backend import CryptoBackend

CRYPTO_COLORS = {
    CryptoBackend.AEGIS: common.COLOR_CIF,
    CryptoBackend.NONE: common.COLOR_NO_CIF,
    CryptoBackend.AESGCM: common.PALETTE_PASTEL[2],
}


def crypto_color(backend: CryptoBackend):
    """Bar colour of a crypto backend."""
    if backend in CRYPTO_COLORS:
        return CRYPTO_COLORS[backend]
    slot = zlib.crc32(backend.value.encode()) % len(common.PALETTE_PASTEL)
    return common.PALETTE_PASTEL[slot]


def group_bar_offsets(
    total_group_width: float, n_bars: int
) -> Tuple[float, List[float]]:
    """Bar width and bar centre offsets for a group centred on its x tick."""
    bar_width = total_group_width / n_bars
    offsets = [-total_group_width / 2 + bar_width * (i + 0.5) for i in range(n_bars)]
    return bar_width, offsets
