"""Torch dataset over the generated reaching benchmark (memory-mapped uint8 images + metadata)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

CF_VARIANTS = 4


class ReachingSplit(Dataset):
    """One split. Images are returned as uint8 HWC numpy (default) or float CHW tensors in [0,1]."""

    def __init__(self, root: str | Path, split: str, *, as_tensor: bool = False, load_images: bool = True):
        self.root, self.split = Path(root), split
        d = self.root / split
        if not d.exists():
            raise FileNotFoundError(d)
        self.images = np.load(d / "images.npy", mmap_mode="r") if load_images else None
        m = np.load(d / "meta.npz")
        self.meta = {k: m[k] for k in m.files}
        self.as_tensor = as_tensor
        self._nuisance = None

    @property
    def nuisance(self) -> list[dict]:
        if self._nuisance is None:
            with open(self.root / self.split / "nuisance.json") as f:
                self._nuisance = json.load(f)
        return self._nuisance

    def __len__(self):
        return len(self.meta["states"])

    def __getitem__(self, i):
        item = {k: torch.as_tensor(v[i]) for k, v in self.meta.items()}
        item["index"] = torch.tensor(i)
        if self.images is not None:
            img = np.array(self.images[i])  # copy out of the read-only memmap
            item["image"] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 if self.as_tensor else img
        return item


def load_manifest(root: str | Path) -> dict:
    with open(Path(root) / "manifest.json") as f:
        return json.load(f)


def cf_splits() -> list[str]:
    return [f"cf_test_v{v}" for v in range(CF_VARIANTS)]


def all_splits(root: str | Path) -> list[str]:
    return sorted(p.name for p in Path(root).iterdir() if (p / "meta.npz").exists())
