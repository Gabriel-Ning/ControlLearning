"""
Evaluate trained runs (and the M0 ridge baseline): action metrics on every test split, ODE step-count sweep,
counterfactual action consistency, and export of representation stages for identical evaluation samples.

    python -m cort_mvp.evaluate --runs results/runs/m3_linear_dinov3_vits16_s0 [...]     (default: all runs)
    python -m cort_mvp.evaluate --m0 --backbone dinov3_vits16

Per run:  results/runs/<run>/eval.json
          results/representations/<run>/<split>.npz   traj (T,N,D) for transport models at t=0,0.1,...,1.0 (n=10 Euler)
                                                        z0/z1/hidden for M1;  h (frozen features) for M0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from cort_mvp.common import WS, action_metrics, load_features, load_meta, to_jsonable
from cort_mvp.models.embedding import ActionNormalizer, FixedActionEmbedding
from cort_mvp.models.regression import DirectRegression, fit_ridge_probe
from cort_mvp.models.transport import TransportModel
from cort_mvp.train import build_model

EVAL_SPLITS = ["val", "id_test", "ood_bg", "geo_ood", "cf_test_v0", "cf_test_v1", "cf_test_v2", "cf_test_v3", "sp_test_v0", "sp_test_v1"]
REP_SPLITS = ["train", "id_test", "ood_bg", "geo_ood", "cf_test_v0", "cf_test_v1", "cf_test_v2", "cf_test_v3", "sp_test_v0", "sp_test_v1"]
STEP_SWEEP = [1, 4, 8, 16]
ANALYSIS_STEPS = 10          # t = 0.0, 0.1, ..., 1.0
TRAIN_REP_N = 4000           # subset of train states used to FIT probes/subspaces (seeded)


def load_run(run_dir: Path, device):
    cfg = OmegaConf.load(run_dir / "config.yaml")
    ck = torch.load(run_dir / "best.pt", map_location=device)
    normalizer = ActionNormalizer(np.zeros(cfg.action_dim), np.ones(cfg.action_dim))
    embedding = FixedActionEmbedding(cfg.action_dim, cfg.latent_dim, seed=cfg.embedding_seed)
    in_dim = load_features(cfg, "val").shape[1]
    model = build_model(cfg, normalizer, embedding, in_dim)
    model.load_state_dict(ck["model"])      # restores the training-set normalizer buffers too
    return cfg, model.to(device).eval(), ck


@torch.no_grad()
def predict_actions(model, H, device, n_steps=None, bs=4096):
    out = []
    for i in range(0, len(H), bs):
        h = torch.as_tensor(H[i:i + bs], device=device)
        _, _, a_hat = model.predict(h, n_steps=n_steps) if isinstance(model, TransportModel) else model.predict(h)
        out.append(model.normalizer.unnormalize(a_hat).cpu().numpy())
    return np.concatenate(out)


def cf_consistency(preds: dict) -> dict:
    """Mean over states of the mean pairwise squared distance between predicted actions across the 4 cf variants."""
    P = np.stack([preds[f"cf_test_v{v}"] for v in range(4)], 0)  # (4, N, 2)
    def pair(a, b): return float(((P[a] - P[b]) ** 2).sum(-1).mean())
    allp = [pair(i, j) for i in range(4) for j in range(i + 1, 4)]
    return {"all_pairs": float(np.mean(allp)), "seen_seen": pair(0, 1), "unseen_unseen": pair(2, 3),
            "seen_unseen": float(np.mean([pair(0, 2), pair(0, 3), pair(1, 2), pair(1, 3)]))}


def eval_run(run_dir: Path, device, rep_root: Path, skip_reps=False):
    cfg, model, ck = load_run(run_dir, device)
    res = {"run": run_dir.name, "kind": cfg.model.kind, "best_step": int(ck["step"]), "train_n_steps": int(cfg.model.get("n_steps", 0)),
           "splits": {}, "step_sweep": {}}
    preds = {}
    for s in EVAL_SPLITS:
        H, A = load_features(cfg, s), load_meta(cfg, s)["actions"]
        a_hat = predict_actions(model, H, device)
        preds[s] = a_hat
        res["splits"][s] = action_metrics(a_hat, A)
        res["splits"][s]["finite"] = bool(np.isfinite(a_hat).all())
    res["cf_action_consistency"] = cf_consistency(preds)
    res["id_to_ood_degradation"] = {k: res["splits"][k]["mse"] - res["splits"]["id_test"]["mse"] for k in ("ood_bg", "geo_ood")}
    if isinstance(model, TransportModel):
        for n in STEP_SWEEP:
            res["step_sweep"][n] = {s: action_metrics(predict_actions(model, load_features(cfg, s), device, n_steps=n), load_meta(cfg, s)["actions"])["mse"]
                                    for s in ("id_test", "ood_bg", "geo_ood")}
    if not skip_reps:
        out = rep_root / run_dir.name
        out.mkdir(parents=True, exist_ok=True)
        for s in REP_SPLITS:
            H = load_features(cfg, s)
            idx = np.arange(len(H))
            if s == "train":
                idx = np.sort(np.random.default_rng(0).choice(len(H), TRAIN_REP_N, replace=False))
            with torch.no_grad():
                h = torch.as_tensor(H[idx], device=device)
                if isinstance(model, TransportModel):
                    tr = model.trajectory(h, n_steps=ANALYSIS_STEPS)
                    arrays = {"traj": tr["traj"].cpu().numpy().astype(np.float32), "times": tr["times"].numpy(), "index": idx}
                    assert np.isfinite(arrays["traj"]).all(), f"non-finite trajectory {run_dir.name}/{s}"
                else:
                    tr = model.trajectory(h)
                    arrays = {"z0": tr["z0"].cpu().numpy(), "z1": tr["z1"].cpu().numpy(), "hidden": tr["hidden"].cpu().numpy(), "index": idx}
            np.savez(out / f"{s}.npz", **arrays)
        res["representations"] = str(out)
    with open(run_dir / "eval.json", "w") as f:
        json.dump(to_jsonable(res), f, indent=2)
    return res


def eval_m0(backbone: str, feature_key: str, rep_root: Path, data_root="data_cache/reaching_v1", features_root="results/features"):
    """M0: ridge regression from frozen features (alpha by 5-fold CV on train). Representations = frozen features."""
    cfg = OmegaConf.create({"data_root": data_root, "features_root": features_root, "backbone": backbone, "feature_key": feature_key})
    Xtr, Ytr = load_features(cfg, "train"), load_meta(cfg, "train")["actions"]
    model, mu, sd, alpha = fit_ridge_probe(Xtr, Ytr)
    name = f"m0_ridge_{backbone}_{feature_key}"
    res = {"run": name, "kind": "m0", "alpha": alpha, "splits": {}, "step_sweep": {}}
    preds = {}
    for s in EVAL_SPLITS:
        a_hat = model.predict((load_features(cfg, s) - mu) / sd)
        preds[s] = a_hat
        res["splits"][s] = action_metrics(a_hat, load_meta(cfg, s)["actions"]); res["splits"][s]["finite"] = True
    res["cf_action_consistency"] = cf_consistency(preds)
    res["id_to_ood_degradation"] = {k: res["splits"][k]["mse"] - res["splits"]["id_test"]["mse"] for k in ("ood_bg", "geo_ood")}
    out = rep_root / name; out.mkdir(parents=True, exist_ok=True)
    for s in REP_SPLITS:
        H = load_features(cfg, s); idx = np.arange(len(H))
        if s == "train":
            idx = np.sort(np.random.default_rng(0).choice(len(H), TRAIN_REP_N, replace=False))
        np.savez(out / f"{s}.npz", h=H[idx], index=idx)
    res["representations"] = str(out)
    rd = WS / "results" / "runs" / name; rd.mkdir(parents=True, exist_ok=True)
    with open(rd / "eval.json", "w") as f:
        json.dump(to_jsonable(res), f, indent=2)
    with open(rd / "run.json", "w") as f:
        json.dump({"run_name": name, "kind": "m0", "backbone": backbone, "feature_key": feature_key, "ridge_alpha": alpha,
                   "config": OmegaConf.to_container(cfg)}, f, indent=2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--runs_root", default="results/runs")
    ap.add_argument("--m0", action="store_true")
    ap.add_argument("--backbone", default="dinov3_vits16")
    ap.add_argument("--feature_key", default="cls")
    ap.add_argument("--skip_reps", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rep_root = WS / "results" / "representations"
    if args.m0:
        r = eval_m0(args.backbone, args.feature_key, rep_root)
        print(f"{r['run']}: id {r['splits']['id_test']['mse']:.4f} ood_bg {r['splits']['ood_bg']['mse']:.4f} geo {r['splits']['geo_ood']['mse']:.4f} alpha {r['alpha']}")
        return
    runs = [Path(p) for p in args.runs] if args.runs else sorted(p for p in (WS / args.runs_root).iterdir() if (p / "best.pt").exists())
    for rd in runs:
        if (rd / "eval.json").exists() and not args.force:
            continue
        r = eval_run(rd, device, rep_root, skip_reps=args.skip_reps)
        s = r["splits"]
        print(f"{rd.name}: id {s['id_test']['mse']:.4f} ood_bg {s['ood_bg']['mse']:.4f} geo {s['geo_ood']['mse']:.4f} "
              f"cf_cons {r['cf_action_consistency']['all_pairs']:.4f}", flush=True)


if __name__ == "__main__":
    main()
