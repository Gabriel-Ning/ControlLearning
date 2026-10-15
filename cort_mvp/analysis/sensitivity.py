"""
§6.3 Task-relevant response via finite differences (only for factors with a continuous parametrization).

For K counterfactual-test states (cf_test_v0 nuisance) we re-render:
    base image,  +eps perturbation of each state dim  (ee_x, ee_y, goal_x, goal_y),
    +eps perturbation of each continuous nuisance factor (hue, hue2, sat, val, freq, phase, angle).
Perturbations are expressed as a fraction of each factor's sampling range so J columns are comparable:
    J_x[:, j]   = (z(x + e_j) - z(x)) / (eps_j / range_j),   J_eta analogous.
Frozen features for all perturbed images are extracted once and cached (results/sensitivity/features_<backbone>.npz);
each run then only propagates features through its (frozen) model to obtain z(t) and J(t).

Reported per run / stage: mean Frobenius norms ||J_x||, ||J_eta||, their ratio, per-factor column norms, and entropy
effective rank of the stacked sensitivity directions (K*4 x D and K*7 x D). Discrete nuisance (background family,
distractor layout) is NOT differentiated here; it is covered by paired differences in invariance.py.

    python -m cort_mvp.analysis.sensitivity --k 200
Output: results/metrics/sensitivity.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from cort_mvp.common import WS
from cort_mvp.data import reaching_render as R
from cort_mvp.data.reaching_dataset import ReachingSplit
from cort_mvp.analysis.common import RUN_ROOT, all_evaluated_runs, effective_rank, run_info
from cort_mvp.evaluate import ANALYSIS_STEPS, load_run
from cort_mvp.models.transport import TransportModel

STATE_NAMES = ["ee_x", "ee_y", "goal_x", "goal_y"]
STATE_RANGE = 2 * R.WORKSPACE
ETA_RANGES = {"hue": 1.0, "hue2": 1.0, "sat": 0.7, "val": 0.65, "freq": 11.0, "phase": 2 * np.pi, "angle": np.pi}
FRAC_EPS = 0.01   # default perturbation = 1 % of the factor's range (override with --frac)


def build_perturbed_images(data_root: Path, k: int, frac: float, seed: int = 0):
    ds = ReachingSplit(data_root, "cf_test_v0", load_images=False)
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(len(ds), k, replace=False))
    states = ds.meta["states"][sel].astype(np.float64)
    etas = [R.Nuisance.from_dict(ds.nuisance[i]) for i in sel]
    images, tags = [], []       # tags: (sample, kind, factor)
    for n, (x, eta) in enumerate(zip(states, etas)):
        images.append(R.render(x, eta)); tags.append((n, "base", ""))
        for j, nm in enumerate(STATE_NAMES):
            xp = x.copy(); xp[j] += frac * STATE_RANGE
            images.append(R.render(xp, eta)); tags.append((n, "x", nm))
        v = eta.continuous_vector()
        for j, nm in enumerate(R.CONTINUOUS_NAMES):
            vp = v.copy(); vp[j] += frac * ETA_RANGES[nm]
            images.append(R.render(x, eta.with_continuous(vp))); tags.append((n, "eta", nm))
    return np.stack(images), tags, sel, [e.bg_family for e in etas]


def get_features(data_root: Path, backbone: str, k: int, frac: float, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / (f"features_{backbone}_k{k}.npz" if frac == 0.01 else f"features_{backbone}_k{k}_frac{frac}.npz")
    if p.exists():
        d = np.load(p, allow_pickle=True)
        return d["cls"], list(map(tuple, d["tags"])), d["sel"], d["fams"]
    from cort_mvp.features.frozen_dino import FrozenDino
    imgs, tags, sel, fams = build_perturbed_images(data_root, k, frac)
    model = FrozenDino(backbone, device="cuda" if torch.cuda.is_available() else "cpu")
    feats = []
    for i in range(0, len(imgs), 256):
        feats.append(model(torch.from_numpy(imgs[i:i + 256]))["cls"].cpu().numpy())
    cls = np.concatenate(feats)
    np.savez(p, cls=cls, tags=np.array(tags, dtype=object), sel=sel, fams=np.array(fams))
    return cls, tags, sel, np.array(fams)


def stages_from_features(run: str, H: np.ndarray, device) -> dict[str, np.ndarray]:
    info = run_info(run)
    if info["kind"] == "m0":
        return {"h": H}
    cfg, model, _ = load_run(RUN_ROOT / run, device)
    with torch.no_grad():
        h = torch.as_tensor(H, device=device)
        if isinstance(model, TransportModel):
            tr = model.trajectory(h, n_steps=ANALYSIS_STEPS)
            return {f"t={t:.1f}": tr["traj"][i].cpu().numpy() for i, t in enumerate(tr["times"].numpy())}
        tr = model.trajectory(h)
        out = {"z0": tr["z0"].cpu().numpy()}
        for i in range(tr["hidden"].shape[0]):
            out[f"h{i + 1}"] = tr["hidden"][i].cpu().numpy()
        out["z1"] = tr["z1"].cpu().numpy()
        return out


def sensitivity_run(run: str, H, tags, fams, device, frac: float) -> list[dict]:
    info = run_info(run)
    stages = stages_from_features(run, H, device)
    base_idx = {t[0]: i for i, t in enumerate(tags) if t[1] == "base"}
    rows = []
    for stage, Z in stages.items():
        Jx = {nm: [] for nm in STATE_NAMES}; Je = {nm: [] for nm in R.CONTINUOUS_NAMES}
        for i, (n, kind, nm) in enumerate(tags):
            if kind == "base":
                continue
            d = (Z[i] - Z[base_idx[n]]) / frac              # per unit of normalized range
            (Jx if kind == "x" else Je)[nm].append(d)
        Jx = {k: np.stack(v) for k, v in Jx.items()}; Je = {k: np.stack(v) for k, v in Je.items()}
        Jx_all = np.concatenate(list(Jx.values())); Je_all = np.concatenate(list(Je.values()))
        # per-sample Frobenius norms: sqrt(sum_j ||d_j||^2)
        nx = np.sqrt(sum((v ** 2).sum(1) for v in Jx.values())); ne = np.sqrt(sum((v ** 2).sum(1) for v in Je.values()))
        er_x, _ = effective_rank(Jx_all, center=False); er_e, _ = effective_rank(Je_all, center=False)
        row = dict(**info, stage=stage, frac=frac, Jx_norm=float(nx.mean()), Jeta_norm=float(ne.mean()),
                   ratio_eta_over_x=float(ne.mean() / (nx.mean() + 1e-12)), erank_Jx=er_x, erank_Jeta=er_e,
                   Jx_norm_rel=float(nx.mean() / (np.linalg.norm(Z, axis=1).mean() + 1e-12)))
        for k, v in Jx.items():
            row[f"Jx_{k}"] = float(np.linalg.norm(v, axis=1).mean())
        for k, v in Je.items():
            row[f"Jeta_{k}"] = float(np.linalg.norm(v, axis=1).mean())
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--frac", type=float, default=FRAC_EPS, help="perturbation as fraction of factor range")
    ap.add_argument("--out", default=None, help="default results/metrics/sensitivity.csv (or sensitivity_frac<frac>.csv)")
    args = ap.parse_args()
    out = args.out or ("results/metrics/sensitivity.csv" if args.frac == FRAC_EPS else f"results/metrics/sensitivity_frac{args.frac}.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = WS / "data_cache" / "reaching_v1"
    runs = args.runs or all_evaluated_runs()
    feats = {}
    rows = []
    for r in runs:
        info = run_info(r)
        bb = info["backbone"]
        if bb not in feats:
            feats[bb] = get_features(data_root, bb, args.k, args.frac, WS / "results" / "sensitivity")
        H, tags, sel, fams = feats[bb]
        rows += sensitivity_run(r, H, tags, fams, device, args.frac)
        print("sensitivity", r, flush=True)
    keys = sorted({k for row in rows for k in row})
    with open(WS / out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("wrote", out, len(rows), "rows")


if __name__ == "__main__":
    main()
