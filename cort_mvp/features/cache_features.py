"""
Extract and cache frozen backbone features for every split of a reaching dataset.

    python -m cort_mvp.features.cache_features --data data_cache/reaching_v1 --backbone dinov3_vits16

Output: results/features/<dataset_name>/<backbone>/<split>.npz   {cls, patch_mean, index}
        results/features/<dataset_name>/<backbone>/meta.json      backbone id/revision, preprocessing, shapes,
                                                                   per-split sha256, determinism check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cort_mvp.data.reaching_dataset import ReachingSplit, all_splits
from cort_mvp.features.frozen_dino import FrozenDino


def _collate(batch):
    return {"image": torch.from_numpy(np.stack([b["image"] for b in batch])), "index": torch.stack([b["index"] for b in batch])}


def extract_split(model: FrozenDino, ds: ReachingSplit, batch_size: int, workers: int) -> dict:
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=_collate, pin_memory=True)
    cls, pm, idx = [], [], []
    for b in dl:
        out = model(b["image"])
        cls.append(out["cls"].float().cpu()); pm.append(out["patch_mean"].float().cpu()); idx.append(b["index"])
    return {"cls": torch.cat(cls).numpy(), "patch_mean": torch.cat(pm).numpy(), "index": torch.cat(idx).numpy()}


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_cache/reaching_v1")
    ap.add_argument("--backbone", default="dinov3_vits16")
    ap.add_argument("--out", default="results/features")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--splits", nargs="*", default=None)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = Path(args.data)
    out = Path(args.out) / data.name / args.backbone
    out.mkdir(parents=True, exist_ok=True)
    model = FrozenDino(args.backbone, device="cuda" if torch.cuda.is_available() else "cpu")
    print("backbone:", asdict(model.info))

    meta = {"backbone": asdict(model.info), "dataset": str(data), "splits": {}, "feature_keys": ["cls", "patch_mean"],
            "primary_feature": "cls", "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    splits = args.splits or all_splits(data)
    t0 = time.time()
    for name in splits:
        ds = ReachingSplit(data, name)
        feats = extract_split(model, ds, args.batch_size, args.workers)
        assert np.isfinite(feats["cls"]).all() and np.isfinite(feats["patch_mean"]).all(), f"non-finite features in {name}"
        assert np.array_equal(feats["index"], np.arange(len(ds)))
        np.savez(out / f"{name}.npz", **feats)
        meta["splits"][name] = {"n": int(len(ds)), "cls_shape": list(feats["cls"].shape), "cls_sha256": sha(feats["cls"]),
                                "patch_mean_sha256": sha(feats["patch_mean"]),
                                "cls_norm_mean": float(np.linalg.norm(feats["cls"], axis=1).mean())}
        print(f"[{time.time() - t0:6.1f}s] {name:12s} {feats['cls'].shape}  sha={meta['splits'][name]['cls_sha256']}")

    # determinism check: re-extract the first 512 train images and compare
    ds = ReachingSplit(data, "train")
    sub = torch.utils.data.Subset(ds, range(min(512, len(ds))))
    sub.meta = ds.meta
    a = extract_split(model, sub, args.batch_size, 0)["cls"]
    b = extract_split(model, sub, args.batch_size, 0)["cls"]
    cached = np.load(out / "train.npz")["cls"][: len(a)]
    meta["determinism"] = {"max_abs_diff_repeat": float(np.abs(a - b).max()), "max_abs_diff_vs_cache": float(np.abs(a - cached).max()),
                           "tolerance": 1e-4, "passed": bool(np.abs(a - b).max() < 1e-4 and np.abs(a - cached).max() < 1e-4)}
    print("determinism:", meta["determinism"])
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote", out / "meta.json")


if __name__ == "__main__":
    main()
