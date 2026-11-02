"""SHD-25 reader for the DCH-Net release."""

from shd25.bands import paper_band_indices
from shd25.dataset import SHD25, load_hsi, load_rgb, load_yolo_label, select_paper_bands

__all__ = [
    "SHD25",
    "load_hsi",
    "load_rgb",
    "load_yolo_label",
    "paper_band_indices",
    "select_paper_bands",
]
