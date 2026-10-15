"""
Parallel run driver. Runs `python -m cort_mvp.train --exp <exp> <overrides>` for a list of jobs, skipping jobs whose
run.json already exists. Job lists are defined here so every sweep is reproducible from the repo.

    python -m cort_mvp.sweep --suite lr_sweep -j 4
    python -m cort_mvp.sweep --suite main      -j 4
    python -m cort_mvp.sweep --suite ablations -j 4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cort_mvp.common import WS, load_config

SEEDS = [0, 1, 2]
LRS = ["3e-4", "1e-3", "3e-3", "1e-2"]   # 1e-2 added after 3e-3 was best for M1-M3 in the first pass
PRIMARY = ["m1", "m2", "m3", "m4"]


def best_lr_table(path=WS / "results" / "metrics" / "lr_sweep_selection.json") -> dict:
    return json.load(open(path)) if path.exists() else {}


def suite_lr_sweep():
    return [(m, [f"optim.lr={lr}", "seed=0", f"run_name=lrsweep_{m}_lr{lr}_s0", "out_root=results/lr_sweep"]) for m in PRIMARY for lr in LRS]


def suite_main():
    lrs = best_lr_table()
    jobs = []
    for m in PRIMARY:
        for s in SEEDS:
            jobs.append((m, [f"optim.lr={lrs.get(m, '1e-3')}", f"seed={s}"]))
    return jobs


def suite_ablations():
    lrs = best_lr_table()
    lr3, lr2, lr1 = lrs.get("m3", "1e-3"), lrs.get("m2", "1e-3"), lrs.get("m1", "1e-3")
    jobs = []
    for s in SEEDS:
        # B: loss decomposition on the same flow architecture (uses M3's LR for flow variants, M1's for the regression analogue)
        jobs += [("ablB_fm_only", [f"optim.lr={lr3}", f"seed={s}"]),
                 ("ablB_endpoint_emb", [f"optim.lr={lr2}", f"seed={s}"]),
                 ("ablB_m1_emb", [f"optim.lr={lr1}", f"seed={s}"])]
        # C: projector capacity, for M3 and the matched M2 control
        for pk in ("frozen_random", "mlp2"):
            jobs += [("m3", [f"optim.lr={lr3}", f"seed={s}", f"projector={pk}"]),
                     ("m2", [f"optim.lr={lr2}", f"seed={s}", f"projector={pk}"])]
        # D: DINOv2 backbone
        for m, lr in (("m1", lr1), ("m2", lr2), ("m3", lr3)):
            jobs.append((m, [f"optim.lr={lr}", f"seed={s}", "backbone=dinov2_vits14"]))
        # E: dataset size (fraction of training states)
        for frac in ("0.01", "0.1"):
            for m, lr in (("m1", lr1), ("m2", lr2), ("m3", lr3)):
                jobs.append((m, [f"optim.lr={lr}", f"seed={s}", f"train_fraction={frac}"]))
        # F: OT-CFM re-pairing
        jobs.append(("ablF_ot_pairing", [f"optim.lr={lr3}", f"seed={s}"]))
        # LR robustness for M1 (selected lr=1e-2 showed an unstable seed): same lr as M2/M3
        jobs.append(("m1", ["optim.lr=3e-3", f"seed={s}", f"run_name=m1_lr3e-3_linear_dinov3_vits16_s{s}"]))
    return jobs


def suite_ode_steps():
    """Train-time ODE step-count sweep for M2/M3 (evaluation step sweep is done in evaluate.py)."""
    lrs = best_lr_table()
    jobs = []
    for s in SEEDS:
        for n in (1, 4, 16):
            jobs += [("m3", [f"optim.lr={lrs.get('m3', '1e-3')}", f"seed={s}", f"model.n_steps={n}", f"run_name=m3_linear_dinov3_vits16_s{s}_nsteps{n}"]),
                     ("m2", [f"optim.lr={lrs.get('m2', '1e-3')}", f"seed={s}", f"model.n_steps={n}", f"run_name=m2_linear_dinov3_vits16_s{s}_nsteps{n}"])]
    return jobs


SUITES = {"lr_sweep": suite_lr_sweep, "main": suite_main, "ablations": suite_ablations, "ode_steps": suite_ode_steps}


def run_job(job):
    exp, ov = job
    cfg = load_config(exp, ov)
    run_dir = WS / cfg.out_root / "runs" / cfg.run_name
    if (run_dir / "run.json").exists():
        return f"skip  {cfg.run_name}"
    cmd = [sys.executable, "-m", "cort_mvp.train", "--exp", exp, *ov]
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "stdout.log", "w") as f:
        r = subprocess.run(cmd, cwd=WS, stdout=f, stderr=subprocess.STDOUT)
    return f"{'ok   ' if r.returncode == 0 else 'FAIL '} {cfg.run_name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=sorted(SUITES))
    ap.add_argument("-j", type=int, default=4)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    jobs = SUITES[args.suite]()
    print(f"{len(jobs)} jobs")
    if args.dry:
        for j in jobs:
            print(j)
        return
    with ThreadPoolExecutor(args.j) as ex:
        for msg in ex.map(run_job, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
