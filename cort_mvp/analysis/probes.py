"""
§6.1 Control linear decodability. For every run and representation stage: freeze the model, fit a NEW ridge probe
(alpha by 5-fold CV on train-stage representations only) and evaluate on held-out states.
Targets: rel_goal (control variable), action (oracle), state_abs (ee_xy, goal_xy; arbitrary physical information).
Supplementary nonlinear probe: fixed-capacity MLP (64 hidden units) for rel_goal.

    python -m cort_mvp.analysis.probes [--runs ...] [--no_mlp]
Output: results/metrics/probes.csv
"""
from __future__ import annotations

import argparse
import csv
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor

from cort_mvp.common import WS, load_meta
from cort_mvp.analysis.common import all_evaluated_runs, load_index, load_stages, run_info

EVAL = ["id_test", "ood_bg", "geo_ood"]
TARGETS = {"rel_goal": lambda m: m["rel_goal"], "action": lambda m: m["actions"],
           "state_abs": lambda m: m["states"]}


def r2_nmse(y, yhat):
    ss_res = ((y - yhat) ** 2).sum(0); ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return float((1 - ss_res / (ss_tot + 1e-12)).mean()), float(((y - yhat) ** 2).mean() / (y.var(0).mean() + 1e-12))


def probe_run(run: str, do_mlp: bool, cfg_like) -> list[dict]:
    info = run_info(run)
    rows = []
    tr_idx = load_index(run, "train")
    meta_tr = load_meta(cfg_like, "train")
    stages_tr = load_stages(run, "train")
    stages_ev = {s: load_stages(run, s) for s in EVAL}
    metas_ev = {s: load_meta(cfg_like, s) for s in EVAL}
    for stage, Xtr in stages_tr.items():
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr_n = (Xtr - mu) / sd
        for tname, tf in TARGETS.items():
            Ytr = tf(meta_tr)[tr_idx]
            ridge = RidgeCV(alphas=np.logspace(-3, 4, 15), cv=KFold(5, shuffle=True, random_state=0)).fit(Xtr_n, Ytr)
            r2_tr, _ = r2_nmse(Ytr, ridge.predict(Xtr_n))
            for s in EVAL:
                Y = tf(metas_ev[s]); r2, nmse = r2_nmse(Y, ridge.predict((stages_ev[s][stage] - mu) / sd))
                rows.append(dict(**info, stage=stage, target=tname, probe="ridge", split=s, r2=r2, nmse=nmse, alpha=float(ridge.alpha_), r2_train=r2_tr))
            if do_mlp and tname == "rel_goal":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    mlp = MLPRegressor(hidden_layer_sizes=(64,), max_iter=400, early_stopping=True, random_state=0, alpha=1e-4).fit(Xtr_n, Ytr)
                for s in EVAL:
                    Y = tf(metas_ev[s]); r2, nmse = r2_nmse(Y, mlp.predict((stages_ev[s][stage] - mu) / sd))
                    rows.append(dict(**info, stage=stage, target=tname, probe="mlp64", split=s, r2=r2, nmse=nmse, alpha=1e-4, r2_train=float("nan")))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--no_mlp", action="store_true")
    ap.add_argument("--out", default="results/metrics/probes.csv")
    args = ap.parse_args()
    from omegaconf import OmegaConf
    cfg_like = OmegaConf.create({"data_root": "data_cache/reaching_v1"})
    runs = args.runs or all_evaluated_runs()
    rows = []
    for r in runs:
        rows += probe_run(r, not args.no_mlp, cfg_like)
        print("probed", r, flush=True)
    keys = sorted({k for row in rows for k in row})
    with open(WS / args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("wrote", args.out, len(rows), "rows")


if __name__ == "__main__":
    main()
