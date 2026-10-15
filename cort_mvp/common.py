"""Shared helpers: config loading, feature/label loading, run bookkeeping, metrics."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

WS = Path(__file__).resolve().parents[1]
CFG_DIR = WS / "cort_mvp" / "configs"


def load_config(experiment: str | None, overrides: list[str]):
    cfg = OmegaConf.load(CFG_DIR / "base.yaml")
    if experiment:
        p = Path(experiment)
        if not p.exists():
            p = CFG_DIR / "experiment" / f"{experiment}.yaml"
        cfg = OmegaConf.merge(cfg, OmegaConf.load(p))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    if cfg.run_name is None:
        exp = Path(experiment).stem if experiment else cfg.model.kind
        cfg.run_name = f"{exp}_{cfg.projector}_{cfg.backbone}_s{cfg.seed}"
        if cfg.train_fraction < 1.0:
            cfg.run_name += f"_frac{cfg.train_fraction}"
    return cfg


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WS).decode().strip()
    except Exception:
        return "unknown"


def env_record() -> dict:
    import sklearn
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "numpy": np.__version__, "sklearn": sklearn.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "platform": platform.platform()}


def features_dir(cfg) -> Path:
    return WS / cfg.features_root / Path(cfg.data_root).name / cfg.backbone


def load_features(cfg, split: str) -> np.ndarray:
    return np.load(features_dir(cfg) / f"{split}.npz")[cfg.feature_key].astype(np.float32)


def load_meta(cfg, split: str) -> dict:
    m = np.load(WS / cfg.data_root / split / "meta.npz")
    return {k: m[k] for k in m.files}


def feature_meta(cfg) -> dict:
    with open(features_dir(cfg) / "meta.json") as f:
        return json.load(f)


def subset_indices(n: int, fraction: float, seed: int) -> np.ndarray:
    if fraction >= 1.0:
        return np.arange(n)
    rng = np.random.default_rng(1000 + seed)
    k = max(8, int(round(n * fraction)))
    return np.sort(rng.choice(n, size=k, replace=False))


def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- metrics
def action_metrics(a_hat: np.ndarray, a: np.ndarray, min_norm: float = 0.2) -> dict:
    """MSE (raw action units), per-dim RMSE, angular error (deg; targets with |a| < min_norm excluded), R^2."""
    err = a_hat - a
    mse = float((err ** 2).mean())
    na, nh = np.linalg.norm(a, axis=1), np.linalg.norm(a_hat, axis=1)
    keep = na >= min_norm
    cos = (a_hat[keep] * a[keep]).sum(1) / (na[keep] * nh[keep] + 1e-8)
    ang = float(np.degrees(np.arccos(np.clip(cos, -1, 1))).mean()) if keep.any() else float("nan")
    ss_res, ss_tot = ((a - a_hat) ** 2).sum(0), ((a - a.mean(0)) ** 2).sum(0)
    r2 = float((1 - ss_res / (ss_tot + 1e-12)).mean())
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "angular_err_deg": ang, "angular_n": int(keep.sum()), "r2": r2,
            "nmse": float(mse / (a.var(0).sum() / a.shape[1] + 1e-12))}


def to_jsonable(o):
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    return o
