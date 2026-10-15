"""
§6.2 Nuisance sensitivity from counterfactual pairs (identical state, different nuisance) vs controlled state changes
(identical nuisance, different state).

  within_seen     = mean ||z(cf_v0) - z(cf_v1)||^2        (both seen background families)
  within_unseen   = mean ||z(cf_v2) - z(cf_v3)||^2        (both unseen families)
  within_cross    = mean over seen/unseen pairs
  within_all      = mean over all 6 pairs
  between_small   = mean ||z(cf_v0) - z(sp_v0)||^2        (state moved by 0.2, same nuisance)
  between_rand    = mean ||z(cf_v0) - z(sp_v1)||^2        (independent state, same nuisance)
  R = within / (between + eps); reported raw and after per-dimension z-scoring with TRAIN-stage statistics.
Also: trace of train covariance and entropy effective rank per stage (latent-collapse indicators).

Output: results/metrics/invariance.csv
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from cort_mvp.common import WS
from cort_mvp.analysis.common import all_evaluated_runs, effective_rank, load_stages, run_info

EPS = 1e-8


def msd(a, b):
    return float(((a - b) ** 2).sum(1).mean())


def invariance_run(run: str) -> list[dict]:
    info = run_info(run)
    cf = [load_stages(run, f"cf_test_v{v}") for v in range(4)]
    sp = [load_stages(run, f"sp_test_v{v}") for v in range(2)]
    tr = load_stages(run, "train")
    rows = []
    for stage in tr:
        Z = [c[stage] for c in cf]; S = [s[stage] for s in sp]
        mu, sd = tr[stage].mean(0), tr[stage].std(0) + 1e-6
        erank, svals = effective_rank(tr[stage])
        out = dict(**info, stage=stage, train_total_var=float(tr[stage].var(0).sum()), train_erank=erank,
                   train_top1_var_frac=float(svals[0] ** 2 / (svals ** 2).sum()))
        for tag, f in (("raw", lambda x: x), ("z", lambda x: (x - mu) / sd)):
            z = [f(x) for x in Z]; s = [f(x) for x in S]
            w_seen, w_unseen = msd(z[0], z[1]), msd(z[2], z[3])
            w_cross = np.mean([msd(z[0], z[2]), msd(z[0], z[3]), msd(z[1], z[2]), msd(z[1], z[3])])
            w_all = np.mean([msd(z[i], z[j]) for i in range(4) for j in range(i + 1, 4)])
            b_small, b_rand = msd(z[0], s[0]), msd(z[0], s[1])
            out.update({f"{tag}_within_seen": w_seen, f"{tag}_within_unseen": w_unseen, f"{tag}_within_cross": w_cross,
                        f"{tag}_within_all": w_all, f"{tag}_between_small": b_small, f"{tag}_between_rand": b_rand,
                        f"{tag}_R_all_rand": w_all / (b_rand + EPS), f"{tag}_R_seen_rand": w_seen / (b_rand + EPS),
                        f"{tag}_R_unseen_rand": w_unseen / (b_rand + EPS), f"{tag}_R_all_small": w_all / (b_small + EPS)})
        rows.append(out)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--out", default="results/metrics/invariance.csv")
    args = ap.parse_args()
    rows = []
    for r in args.runs or all_evaluated_runs():
        rows += invariance_run(r)
    keys = sorted({k for row in rows for k in row})
    with open(WS / args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("wrote", args.out, len(rows), "rows")


if __name__ == "__main__":
    main()
