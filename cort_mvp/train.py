"""
Train M1 / M2 / M3 / M4 (and ablation variants) on cached frozen features.

    python -m cort_mvp.train --exp m3 seed=0 optim.lr=1e-3

Each run writes to results/runs/<run_name>/:
    config.yaml, run.json (git sha, env, params, dataset/backbone provenance, final metrics),
    curves.csv (loss components / val metrics / grad cosine), best.pt + last.pt (selected on validation only).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from cort_mvp.common import (WS, action_metrics, env_record, feature_meta, git_sha, load_config, load_features, load_meta,
                             set_seed, subset_indices, to_jsonable)
from cort_mvp.models.embedding import ActionNormalizer, FixedActionEmbedding
from cort_mvp.models.regression import DirectRegression
from cort_mvp.models.transport import TransportModel


def build_model(cfg, normalizer, embedding, in_dim):
    m = cfg.model
    common = dict(in_dim=in_dim, latent_dim=cfg.latent_dim, hidden_dim=cfg.net.hidden_dim, num_layers=cfg.net.num_layers,
                  projector_kind=cfg.projector, normalizer=normalizer, embedding=embedding, seed=cfg.seed)
    if m.kind == "m1":
        return DirectRegression(lambda_action=m.lambda_action, lambda_emb=m.lambda_emb, **common)
    return TransportModel(n_steps=m.n_steps, lambda_fm=m.lambda_fm, lambda_action=m.lambda_action, lambda_emb=m.lambda_emb,
                          lambda_cf=m.get("lambda_cf", 0.0), cf_times=tuple(m.get("cf_times", [1.0])), pairing=m.get("pairing", "paired"),
                          kind=m.kind, **common)


def lr_at(step, cfg):
    o = cfg.optim
    if step < o.warmup_steps:
        return o.lr * (step + 1) / o.warmup_steps
    if o.schedule == "cosine":
        p = (step - o.warmup_steps) / max(1, o.steps - o.warmup_steps)
        return o.lr * 0.5 * (1 + math.cos(math.pi * p))
    return o.lr


@torch.no_grad()
def evaluate(model, H, A, normalizer, device, bs=2048, n_steps=None):
    model.eval()
    out = []
    for i in range(0, len(H), bs):
        h = torch.as_tensor(H[i:i + bs], device=device)
        if isinstance(model, TransportModel):
            _, _, a_hat = model.predict(h, n_steps=n_steps)
        else:
            _, _, a_hat = model.predict(h)
        out.append(normalizer.unnormalize(a_hat).cpu().numpy())
    a_hat = np.concatenate(out)
    finite = bool(np.isfinite(a_hat).all())
    met = action_metrics(np.nan_to_num(a_hat), A) if finite else {"mse": float("nan"), "angular_err_deg": float("nan"), "r2": float("nan")}
    met["finite"] = finite
    return met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="m3", help="experiment yaml name in cort_mvp/configs/experiment or a path")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.exp, args.overrides)
    set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    run_dir = WS / cfg.out_root / "runs" / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")

    # ---- data (cached frozen features; training-set statistics only)
    H_tr_full, M_tr = load_features(cfg, cfg.train_split), load_meta(cfg, cfg.train_split)
    idx = subset_indices(len(H_tr_full), cfg.train_fraction, cfg.seed)
    H_tr, A_tr = H_tr_full[idx], M_tr["actions"][idx]
    H_val, A_val = load_features(cfg, cfg.val_split), load_meta(cfg, cfg.val_split)["actions"]
    H_pair = None
    if cfg.model.kind == "m4":
        H_pair = load_features(cfg, "train_cf")[idx]
        assert np.array_equal(load_meta(cfg, "train_cf")["state_id"][idx], M_tr["state_id"][idx])
    normalizer = ActionNormalizer.from_train_actions(A_tr).to(device)
    embedding = FixedActionEmbedding(cfg.action_dim, cfg.latent_dim, seed=cfg.embedding_seed).to(device)
    model = build_model(cfg, normalizer, embedding, H_tr.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params_net = sum(p.numel() for p in model.net.parameters())
    n_params_proj = sum(p.numel() for p in model.projector.parameters() if p.requires_grad)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.optim.lr,
                            betas=tuple(cfg.optim.betas), weight_decay=cfg.optim.weight_decay)
    Htr_t, Atr_t = torch.as_tensor(H_tr, device=device), torch.as_tensor(A_tr, device=device)
    Hp_t = torch.as_tensor(H_pair, device=device) if H_pair is not None else None
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)

    rows, best, best_step, nonfinite_steps = [], float("inf"), -1, 0
    t0 = time.time()
    for step in range(cfg.optim.steps):
        model.train()
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step, cfg)
        bi = torch.randint(0, len(Htr_t), (cfg.optim.batch_size,), generator=g).to(device)
        h, a = Htr_t[bi], Atr_t[bi]
        hp = Hp_t[bi] if Hp_t is not None else None
        loss, logs = model.losses(h, a, hp) if isinstance(model, TransportModel) else model.losses(h, a)
        if not torch.isfinite(loss):
            nonfinite_steps += 1
            opt.zero_grad(set_to_none=True)
            if nonfinite_steps > 20:
                print("ABORT: repeated non-finite loss"); break
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
        opt.step()

        row = {"step": step, "loss": loss.item(), "grad_norm": gn.item(), "lr": lr_at(step, cfg)}
        row.update({f"loss_{k}": float(v) for k, v in logs.items()})
        if (isinstance(model, TransportModel) and model.lambda_fm > 0 and model.lambda_action > 0
                and cfg.log_grad_cosine_every > 0 and step % cfg.log_grad_cosine_every == 0):
            g_fm, g_ac = model.separate_grad_vectors(h, a)
            row["grad_cos_fm_action"] = float(torch.nn.functional.cosine_similarity(g_fm, g_ac, dim=0))
            row["grad_norm_fm"], row["grad_norm_action"] = float(g_fm.norm()), float(g_ac.norm())
            opt.zero_grad(set_to_none=True)
        if step % cfg.eval_every == 0 or step == cfg.optim.steps - 1:
            vm = evaluate(model, H_val, A_val, normalizer, device)
            row.update({f"val_{k}": v for k, v in vm.items() if k in ("mse", "angular_err_deg", "r2")})
            sel = vm["mse"] if vm["finite"] else float("inf")
            if sel < best:
                best, best_step = sel, step
                torch.save({"model": model.state_dict(), "step": step, "val_mse": sel}, run_dir / "best.pt")
            print(f"[{cfg.run_name}] step {step:5d} loss {loss.item():.4f} " +
                  " ".join(f"{k}={float(v):.4f}" for k, v in logs.items()) +
                  f" | val mse {vm['mse']:.4f} ang {vm['angular_err_deg']:.2f} | {time.time() - t0:.0f}s", flush=True)
        rows.append(row)

    torch.save({"model": model.state_dict(), "step": cfg.optim.steps - 1}, run_dir / "last.pt")
    keys = sorted({k for r in rows for k in r})
    with open(run_dir / "curves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    fmeta = feature_meta(cfg)
    with open(WS / cfg.data_root / "manifest.json") as f:
        manifest = json.load(f)
    record = {
        "run_name": cfg.run_name, "git_sha": git_sha(), "config": OmegaConf.to_container(cfg, resolve=True), "seed": cfg.seed,
        "params": {"trainable_total": n_params, "net": n_params_net, "projector_trainable": n_params_proj},
        "dataset_manifest": {"name": manifest["name"], "seed": manifest["seed"], "git_sha": manifest["git_sha"],
                             "train_n_used": int(len(idx)), "train_sha": manifest["splits"]["train"]["images_sha256_subsample"]},
        "backbone": fmeta["backbone"], "feature_key": cfg.feature_key, "feature_cache_sha": fmeta["splits"]["train"]["cls_sha256"],
        "environment": env_record(), "action_norm": {"mean": normalizer.mean.cpu().tolist(), "std": normalizer.std.cpu().tolist()},
        "train_time_s": time.time() - t0, "nonfinite_loss_steps": nonfinite_steps,
        "best_val_mse": best, "best_step": best_step, "checkpoint": str(run_dir / "best.pt"), "last_checkpoint": str(run_dir / "last.pt"),
        "selection": cfg.select_on,
    }
    with open(run_dir / "run.json", "w") as f:
        json.dump(to_jsonable(record), f, indent=2)
    print(f"done {cfg.run_name}: best val mse {best:.5f} @ {best_step}, params {n_params}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
