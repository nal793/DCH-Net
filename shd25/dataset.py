"""Read the SHD-25 dataset."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from shd25.bands import paper_band_indices

HSI_SHAPE = (256, 256, 242)
RGB_SIZE = (256, 256)


def load_yolo_label(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(class_ids [N], boxes [N, 4])``.

    Boxes are YOLO ``cx, cy, w, h`` normalized to the 256×256 frame.
    """
    classes: list[int] = []
    boxes: list[list[float]] = []
    text = Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path} has a label row that is not 'cls cx cy w h': {line}")
        classes.append(int(float(parts[0])))
        boxes.append([float(v) for v in parts[1:]])
    class_ids = np.asarray(classes, dtype=np.int64)
    box_arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return class_ids, box_arr


def load_hsi(path: Path) -> np.ndarray:
    """Load one cube as ``uint16`` array ``(256, 256, 242)``."""
    cube = tifffile.imread(path)
    if cube.shape != HSI_SHAPE or cube.dtype != np.uint16:
        raise ValueError(f"{path} is {cube.shape} {cube.dtype}, expected {HSI_SHAPE} uint16")
    return cube


def load_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.size != RGB_SIZE:
        raise ValueError(f"{path} is {image.size}, expected {RGB_SIZE}")
    return np.asarray(image)


class SHD25:
    """SHD-25 samples. Hyperspectral cubes are loaded on access.

    Parameters
    ----------
    root:
        Directory that contains ``hsi/``, ``rgb/``, ``labels/`` and
        ``meta/index.csv``.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        index_path = self.root / "meta" / "index.csv"
        if not index_path.is_file():
            raise FileNotFoundError(f"missing dataset index: {index_path}")
        with index_path.open(encoding="utf-8", newline="") as handle:
            self.samples = list(csv.DictReader(handle))
        if not self.samples:
            raise RuntimeError(f"empty dataset index: {index_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        row = self.samples[index]
        stem = row["file_name"]
        class_ids, boxes = load_yolo_label(self.root / "labels" / f"{stem}.txt")
        return {
            "id": row["sample_id"],
            "file_name": stem,
            "scene_id": row["scene_id"],
            "place_en": row["place_en"],
            "place_zh": row["place_zh"],
            "country_en": row["country_en"],
            "country_zh": row["country_zh"],
            "original_name": row["original_name"],
            "hsi": load_hsi(self.root / "hsi" / f"{stem}.tif"),
            "rgb": load_rgb(self.root / "rgb" / f"{stem}.png"),
            "class_ids": class_ids,
            "boxes": boxes,
        }

    def get(self, sample_id: str) -> dict:
        for index, row in enumerate(self.samples):
            if row["sample_id"] == sample_id or row["file_name"] == sample_id:
                return self[index]
        raise KeyError(sample_id)


def select_paper_bands(cube: np.ndarray) -> np.ndarray:
    """Slice a cube to the 145 bands used by DCH-Net."""
    if cube.ndim != 3 or cube.shape[-1] != 242:
        raise ValueError(f"expected HWC cube with 242 bands, got {cube.shape}")
    return cube[:, :, paper_band_indices()]
