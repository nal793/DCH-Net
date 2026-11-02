"""Spectral bands used by DCH-Net on SHD-25.

The cubes store 242 Hyperion bands. The network uses the 145-band
subset numbered 8–56, 81–118, 132–162, and 187–213. Those numbers are
1-based Hyperion band IDs; `paper_band_indices` returns the matching
0-based positions in the released `(256, 256, 242)` cube.
"""

from __future__ import annotations

import numpy as np

# Inclusive 1-based Hyperion band numbers.
PAPER_BAND_RANGES = ((8, 56), (81, 118), (132, 162), (187, 213))


def paper_band_ids() -> list[int]:
    """Return the 145 one-based band numbers in wavelength order."""
    ids: list[int] = []
    for start, end in PAPER_BAND_RANGES:
        ids.extend(range(start, end + 1))
    if len(ids) != 145 or len(set(ids)) != 145:
        raise RuntimeError("band list is not 145 unique ids")
    return ids


def paper_band_indices() -> np.ndarray:
    """0-based channel indices for ``cube[:, :, indices]``."""
    index = [band - 1 for band in paper_band_ids()]
    if min(index) < 0 or max(index) > 241:
        raise RuntimeError("band index falls outside the 242-band cube")
    return np.asarray(index, dtype=np.int64)
