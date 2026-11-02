# SHD-25

SHD-25 is the hyperspectral ship detection set for DCH-Net: 120 EO-1 Hyperion chips, \(256\times256\times242\), with YOLO ship boxes, RGB quicklooks, and yellow-box previews. The cubes are about 3.6 GB and are not stored in this code repository.

Download the dataset from <https://huggingface.co/datasets/fengyanzi/SHD25>.

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download fengyanzi/SHD25 --repo-type dataset --local-dir SHD25
```

```python
from huggingface_hub import snapshot_download

snapshot_download("fengyanzi/SHD25", repo_type="dataset", local_dir="SHD25")
```

The downloaded folder is the dataset root (`hsi/`, `rgb/`, `labels/`, `previews/`, `meta/`, `shd25.yaml`). Its README describes the file layout, bands, and labels. Point the reader at that directory:

```python
from shd25 import SHD25, select_paper_bands

dataset = SHD25("SHD25")
sample = dataset[0]
cube = select_paper_bands(sample["hsi"])  # (256, 256, 145)
```
