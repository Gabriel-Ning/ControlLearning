"""
§6.4 Linear subspace diagnostics (descriptive statistics, not proof of disentanglement).

For every run / stage:
  delta_eta = z(cf_v0) - z(cf_v1)       nuisance-variation differences (seen families), N x D
  delta_x   = z(cf_v0) - z(sp_v1)       state-variation differences (same nuisance),   N x D
  C         = orthonormal basis (D x 2) of the control-predictive directions = ridge-probe weights for rel_goal,
              fitted on TRAIN-stage representations (separate data; not the model's own readout W).
Reported:
  * singular-value spectrum / effective rank of delta_eta and delta_x
  * frac_eta_in_C  = sum ||C^T delta_eta||^2 / sum ||delta_eta||^2   (nuisance energy inside the control subspace)
  * frac_x_in_C    = same for delta_x
  * selectivity    = frac_x_in_C / (frac_eta_in_C + eps)
  * principal-angle overlap between top-k PCs of delta_eta and C (mean cos^2), k = 2, 4, 8
  * the same with the model's own fixed readout W (transport models only) for comparison
Output: results/metrics/subspaces.csv
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
from omegaconf import OmegaConf
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from cort_mvp.common import WS, load_meta
from cort_mvp.models.embedding import FixedActionEmbedding
from cort_mvp.analysis.common import all_evaluated_runs, effective_rank, load_index, load_stages, run_info


def orth(M):
    Q, _ = np.linalg.qr(M)
    return Q


def pcs(X, k):
    Xc = X - X.mean(0)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k].T, s


def mean_cos2(A, B):
    """Mean squared cosine of principal angles between span(A) and span(B) (both orthonormal columns)."""
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    return float((s ** 2).mean())


def frac_in(delta, C):
    return float((np.linalg.norm(delta @ C, axis=1) ** 2).sum() / ((delta ** 2).sum() + 1e-12))


def subspace_run(run: str, cfg_like) -> list[dict]:
    info = run_info(run)
    tr = load_stages(run, "train"); tr_idx = load_index(run, "train")
    Y = load_meta(cfg_like, "train")["rel_goal"][tr_idx]
    cf0, cf1, sp1 = load_stages(run, "cf_test_v0"), load_stages(run, "cf_test_v1"), load_stages(run, "sp_test_v1")
    W = None
    if info["kind"] in ("m2", "m3", "m4"):
        cfg = OmegaConf.load(WS / "results" / "runs" / run / "config.yaml")
        W = FixedActionEmbedding(cfg.action_dim, cfg.latent_dim, seed=cfg.embedding_seed).W.numpy()
    rows = []
    for stage, Xtr in tr.items():
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        ridge = RidgeCV(alphas=np.logspace(-3, 4, 15), cv=KFold(5, shuffle=True, random_state=0)).fit((Xtr - mu) / sd, Y)
        C = orth((ridge.coef_ / sd[None, :]).T)          # control directions in RAW coordinates (D x 2)
        d_eta = cf0[stage] - cf1[stage]
        d_x = cf0[stage] - sp1[stage]
        er_eta, s_eta = effective_rank(d_eta, center=False)
        er_x, s_x = effective_rank(d_x, center=False)
        row = dict(**info, stage=stage, erank_eta=er_eta, erank_x=er_x,
                   sv_eta_top4=(s_eta[:4] ** 2 / (s_eta ** 2).sum()).tolist(), sv_x_top4=(s_x[:4] ** 2 / (s_x ** 2).sum()).tolist(),
                   frac_eta_in_C=frac_in(d_eta, C), frac_x_in_C=frac_in(d_x, C))
        row["selectivity_C"] = row["frac_x_in_C"] / (row["frac_eta_in_C"] + 1e-12)
        for k in (2, 4, 8):
            U, _ = pcs(d_eta, min(k, d_eta.shape[1]))
            row[f"overlap_eta_pc{k}_C"] = mean_cos2(U, C)
        if W is not None:
            Wq = orth(W)
            row.update(frac_eta_in_W=frac_in(d_eta, Wq), frac_x_in_W=frac_in(d_x, Wq), overlap_C_W=mean_cos2(C, Wq))
            row["selectivity_W"] = row["frac_x_in_W"] / (row["frac_eta_in_W"] + 1e-12)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--out", default="results/metrics/subspaces.csv")
    args = ap.parse_args()
    cfg_like = OmegaConf.create({"data_root": "data_cache/reaching_v1"})
    rows = []
    for r in args.runs or all_evaluated_runs():
        rows += subspace_run(r, cfg_like)
    keys = sorted({k for row in rows for k in row})
    with open(WS / args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("wrote", args.out, len(rows), "rows")


if __name__ == "__main__":
    main()
