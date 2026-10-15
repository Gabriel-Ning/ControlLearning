"""
Generate the synthetic 2D reaching benchmark (P1).

Splits (default sizes):
  train           10000 states, seen background families, nuisance random
  val              2000 states (disjoint), seen backgrounds
  id_test          2000 states (disjoint), seen backgrounds
  ood_bg           2000 states (disjoint, same state distribution), UNSEEN background families
  cf_test          2000 states x 4 nuisance variants (variants 0,1: seen families; 2,3: unseen families)
                   -> identical physical state and oracle action across the 4 variants
  sp_test          2000 x 2 state-perturbation controls sharing the nuisance of cf_test variant 0:
                   variant 0: small state perturbation  x + dx  (|dx| ~ 0.2)
                   variant 1: independent random state  x'
  geo_ood          2000 states with goal inside the held-out box, seen backgrounds

Optional diagnostic (separate directory, clearly labelled): --spurious writes a dataset in which the
background family is a deterministic function of the goal quadrant in train/val, and decorrelated in tests.

Output layout  <out>/<split>/images.npy   uint8 (N,224,224,3)
               <out>/<split>/meta.npz     states, actions, rel_goal, state_id, variant, nuisance_seed, bg_family,
                                          n_distractors, halo, nuisance_cont (N,7), clipped (N,)
               <out>/<split>/nuisance.json  full nuisance dataclasses (for exact re-rendering)
               <out>/manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from cort_mvp.data import reaching_render as R

DEFAULT_SIZES = dict(train=10000, val=2000, id_test=2000, ood_bg=2000, cf_test=2000, sp_test=2000, geo_ood=2000)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2]).decode().strip()
    except Exception:
        return "unknown"


def _render_many(args):
    states, etas, draw_arm = args
    out = np.empty((len(states), R.IMG, R.IMG, 3), dtype=np.uint8)
    for i, (s, e) in enumerate(zip(states, etas)):
        out[i] = R.render(s, R.Nuisance.from_dict(e), draw_arm=draw_arm)
    return out


def render_split(states, etas, draw_arm, workers):
    n = len(states)
    chunk = max(64, n // (workers * 4) + 1)
    jobs = [(states[i:i + chunk], [asdict(e) for e in etas[i:i + chunk]], draw_arm) for i in range(0, n, chunk)]
    imgs = np.empty((n, R.IMG, R.IMG, 3), dtype=np.uint8)
    with ProcessPoolExecutor(workers) as ex:
        pos = 0
        for arr in ex.map(_render_many, jobs):
            imgs[pos:pos + len(arr)] = arr
            pos += len(arr)
    return imgs


def quadrant_family(goal: np.ndarray) -> int:
    """Spurious-correlation mapping goal quadrant -> seen background family (diagnostic only)."""
    q = (goal[0] >= 0) * 2 + (goal[1] >= 0)
    return [0, 1, 2, 3][int(q)]


def build_split(name, states, rng, families, *, state_ids, variant, spurious=False, max_distractors=4, halo_prob=0.5,
                reuse_etas=None):
    etas = []
    for k, s in enumerate(states):
        if reuse_etas is not None:
            etas.append(reuse_etas[k])
            continue
        fam = quadrant_family(s[2:4]) if spurious else int(rng.choice(families))
        etas.append(R.sample_nuisance(rng, fam, s, max_distractors=max_distractors, halo_prob=halo_prob))
    actions = R.oracle_action(states)
    rel = R.relative_goal(states)
    clipped = np.any(np.abs(R.K_GAIN * rel) > R.A_MAX, axis=1)
    meta = dict(
        states=states.astype(np.float32), actions=actions.astype(np.float32), rel_goal=rel.astype(np.float32),
        state_id=np.asarray(state_ids, dtype=np.int64), variant=np.full(len(states), variant, dtype=np.int64),
        nuisance_seed=np.array([e.seed for e in etas], dtype=np.int64),
        bg_family=np.array([e.bg_family for e in etas], dtype=np.int64),
        n_distractors=np.array([e.n_distractors for e in etas], dtype=np.int64),
        halo=np.array([e.halo for e in etas], dtype=np.int64),
        nuisance_cont=np.stack([e.continuous_vector() for e in etas]).astype(np.float32),
        clipped=clipped,
    )
    return etas, meta


def save_split(out_dir: Path, name: str, imgs, meta, etas):
    d = out_dir / name
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "images.npy", imgs)
    np.savez(d / "meta.npz", **meta)
    with open(d / "nuisance.json", "w") as f:
        json.dump([asdict(e) for e in etas], f)
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(imgs[:: max(1, len(imgs) // 256)]).tobytes())
    h.update(meta["states"].tobytes())
    return {"n": int(len(imgs)), "images_sha256_subsample": h.hexdigest()[:16],
            "clipped_fraction": float(meta["clipped"].mean()),
            "bg_family_counts": {int(k): int(v) for k, v in zip(*np.unique(meta["bg_family"], return_counts=True))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_cache/reaching_v1")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply all split sizes (for quick tests)")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    ap.add_argument("--no_arm", action="store_true")
    ap.add_argument("--spurious", action="store_true", help="write the spurious-correlation diagnostic variant")
    ap.add_argument("--max_distractors", type=int, default=4)
    ap.add_argument("--halo_prob", type=float, default=0.5)
    args = ap.parse_args()

    sizes = {k: max(8, int(v * args.scale)) for k, v in DEFAULT_SIZES.items()}
    out = Path(args.out)
    if args.spurious:
        out = out.parent / (out.name + "_SPURIOUS_DIAGNOSTIC")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    draw_arm = not args.no_arm
    t0 = time.time()

    # ---- disjoint physical states (float64 continuous sampling; verified below)
    n_main = sizes["train"] + sizes["val"] + sizes["id_test"] + sizes["ood_bg"] + sizes["cf_test"]
    pool = R.sample_states(rng, n_main, geo_holdout=False)
    keys = {tuple(np.round(s, 6)) for s in pool}
    assert len(keys) == len(pool), "duplicate physical states in pool"
    geo = R.sample_states(rng, sizes["geo_ood"], geo_holdout=True)
    assert R.in_geo_holdout(geo).all() and not R.in_geo_holdout(pool).any()

    off = 0
    parts = {}
    for name in ("train", "val", "id_test", "ood_bg", "cf_test"):
        parts[name] = (pool[off:off + sizes[name]], np.arange(off, off + sizes[name]))
        off += sizes[name]
    parts["geo_ood"] = (geo, np.arange(off, off + sizes["geo_ood"]))

    manifest = {
        "name": out.name, "seed": args.seed, "sizes": sizes, "git_sha": _git_sha(),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "python": platform.python_version(), "numpy": np.__version__,
        "image_size": R.IMG, "workspace": R.WORKSPACE, "k_gain": R.K_GAIN, "a_max": R.A_MAX,
        "geo_holdout_box": R.GEO_HOLDOUT_BOX, "seen_families": list(R.SEEN_FAMILIES), "unseen_families": list(R.UNSEEN_FAMILIES),
        "family_names": R.FAMILY_NAMES, "draw_arm": draw_arm, "max_distractors": args.max_distractors, "halo_prob": args.halo_prob,
        "spurious_diagnostic": bool(args.spurious), "continuous_nuisance_names": R.CONTINUOUS_NAMES,
        "state_layout": "[ee_x, ee_y, goal_x, goal_y]", "action": "clip(k*(goal-ee), -a_max, a_max)",
        "splits": {},
    }

    def gen(name, states, ids, families, variant=0, spurious=False, reuse=None):
        etas, meta = build_split(name, states, rng, families, state_ids=ids, variant=variant, spurious=spurious,
                                 max_distractors=args.max_distractors, halo_prob=args.halo_prob, reuse_etas=reuse)
        imgs = render_split(states, etas, draw_arm, args.workers)
        info = save_split(out, name, imgs, meta, etas)
        manifest["splits"][name] = info
        print(f"[{time.time() - t0:7.1f}s] {name:10s} n={info['n']:6d} clipped={info['clipped_fraction']:.4f} fam={info['bg_family_counts']}")
        return etas, meta

    seen, unseen = list(R.SEEN_FAMILIES), list(R.UNSEEN_FAMILIES)
    gen("train", *parts["train"], seen, spurious=args.spurious)
    gen("val", *parts["val"], seen, spurious=args.spurious)
    gen("id_test", *parts["id_test"], seen)                       # decorrelated even in the spurious diagnostic
    gen("ood_bg", *parts["ood_bg"], unseen)
    gen("geo_ood", *parts["geo_ood"], seen)

    # counterfactual: same states, 4 nuisance variants (2 seen, 2 unseen families)
    cf_states, cf_ids = parts["cf_test"]
    cf_etas = []
    for v, fams in enumerate([seen, seen, unseen, unseen]):
        e, _ = gen(f"cf_test_v{v}", cf_states, cf_ids, fams, variant=v)
        cf_etas.append(e)
    # state-perturbation controls sharing nuisance of cf variant 0
    sp_small = cf_states.copy()
    dx = rng.normal(size=(len(sp_small), 4))
    dx = dx / np.linalg.norm(dx, axis=1, keepdims=True) * 0.2
    sp_small = np.clip(sp_small + dx, -R.WORKSPACE, R.WORKSPACE)
    sp_rand = R.sample_states(rng, len(cf_states), geo_holdout=False)
    gen("sp_test_v0", sp_small, cf_ids, seen, variant=0, reuse=cf_etas[0])
    gen("sp_test_v1", sp_rand, cf_ids, seen, variant=1, reuse=cf_etas[0])

    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"done in {time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
