"""Shared loading of exported representation stages for all model kinds."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from cort_mvp.common import WS

REP_ROOT = WS / "results" / "representations"
RUN_ROOT = WS / "results" / "runs"


def run_info(run_name: str) -> dict:
    rd = RUN_ROOT / run_name
    info = {"run": run_name}
    if (rd / "config.yaml").exists():
        cfg = OmegaConf.load(rd / "config.yaml")
        info.update(kind=cfg.model.kind, seed=int(cfg.seed), projector=cfg.projector, backbone=cfg.backbone,
                    train_fraction=float(cfg.train_fraction), exp=re.sub(r"_s\d+.*$", "", run_name).replace(f"_{cfg.projector}_{cfg.backbone}", ""),
                    n_steps=int(cfg.model.get("n_steps", 0)), pairing=cfg.model.get("pairing", "paired"),
                    lambda_fm=float(cfg.model.get("lambda_fm", 0)), lambda_action=float(cfg.model.get("lambda_action", 0)),
                    lambda_emb=float(cfg.model.get("lambda_emb", 0)), lambda_cf=float(cfg.model.get("lambda_cf", 0)))
    else:
        r = json.load(open(rd / "run.json"))
        info.update(kind=r["kind"], seed=0, projector="none", backbone=r["backbone"], train_fraction=1.0, exp="m0")
    return info


def load_stages(run_name: str, split: str) -> dict[str, np.ndarray]:
    """Return an ordered dict stage_name -> (N, D) float32 for the given run and split."""
    d = np.load(REP_ROOT / run_name / f"{split}.npz")
    if "traj" in d.files:
        traj, times = d["traj"], d["times"]
        return {f"t={t:.1f}": traj[i] for i, t in enumerate(times)}
    if "z0" in d.files:
        out = {"z0": d["z0"]}
        for i in range(d["hidden"].shape[0]):
            out[f"h{i + 1}"] = d["hidden"][i]
        out["z1"] = d["z1"]
        return out
    return {"h": d["h"]}


def load_index(run_name: str, split: str) -> np.ndarray:
    return np.load(REP_ROOT / run_name / f"{split}.npz")["index"]


def all_evaluated_runs() -> list[str]:
    return sorted(p.name for p in RUN_ROOT.iterdir() if (p / "eval.json").exists() and (REP_ROOT / p.name).exists())


def effective_rank(X: np.ndarray, center=True) -> tuple[float, np.ndarray]:
    """Entropy effective rank of the singular-value spectrum of (centered) X. Returns (erank, singular_values)."""
    if center:
        X = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    p = s ** 2 / (np.sum(s ** 2) + 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum())), s


def zscore_params(X_train: np.ndarray):
    return X_train.mean(0), X_train.std(0) + 1e-6
