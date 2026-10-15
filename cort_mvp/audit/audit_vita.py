"""
P0 audit script: instantiate the ORIGINAL VITA pipeline on the converted PushT subset
and record (without modifying upstream code):

  * module inventory and parameter counts
  * tensor shapes along the forward path
  * every active loss component and its weight
  * gradient paths (which modules receive gradient from which loss)
  * ODE sampling schedule (time grid, dt, finiteness of the trajectory)
  * whether the configured `exact` OT-CFM re-pairs (obs, action) within a minibatch

Usage (from the workspace root, with the venv activated and FLARE_DATASETS_DIR set):

    python -m cort_mvp.audit.audit_vita --out reports/audit_artifacts/vita_audit.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

WORKSPACE = Path(__file__).resolve().parents[2]
VITA_ROOT = WORKSPACE / "VITA"


def git_sha(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo).decode().strip()


def count_params(m: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def build_cfg(overrides: list[str]):
    with initialize_config_dir(config_dir=str(VITA_ROOT / "flare" / "configs"), version_base="1.3"):
        cfg = compose(config_name="default_policy", overrides=overrides)
    # Hydra runtime interpolation `${hydra:runtime.cwd}` is not resolvable under compose(); set explicitly.
    cfg.base_dir = str(VITA_ROOT)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(WORKSPACE / "reports" / "audit_artifacts" / "vita_audit.json"))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    sys.path.insert(0, str(VITA_ROOT))
    from flare.factory import get_policy_class
    from flare.utils.dataset_utils import create_dataloader, create_dataset, create_dataset_stats

    overrides = [
        "policy=vita",
        "task=pusht",
        "session=audit",
        "policy.flow_net.name=simple_flow_net",  # required: yaml ships `simple`, policy accepts `simple_flow_net`
        "wandb.enable=false",
        f"train.batch_size={args.batch_size}",
        "train.num_workers=0",
        f"device={args.device}",
    ]
    cfg = build_cfg(overrides)
    torch.manual_seed(0)
    np.random.seed(0)

    record: dict = {
        "vita_commit": git_sha(VITA_ROOT),
        "overrides": overrides,
        "torch": torch.__version__,
        "device": args.device,
    }

    # ------------------------------------------------------------------ data
    dataset_meta, stats = create_dataset_stats(cfg)
    policy_cls = get_policy_class(cfg.policy.name)
    policy = policy_cls(cfg, stats).to(args.device)
    policy.train()

    ds = create_dataset(policy, cfg)
    dl = create_dataloader(ds, cfg, is_training=True)
    batch = next(iter(dl))
    batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    record["dataset"] = {
        "repo_id": cfg.task.dataset_repo_id,
        "num_frames_in_zarr": ds.num_frames,
        "num_episodes": ds.num_episodes,
        "fps": ds.fps,
        "batch_keys": {k: list(v.shape) for k, v in batch.items() if torch.is_tensor(v)},
        "image_dtype_range": [float(batch[cfg.task.image_keys[0]].min()), float(batch[cfg.task.image_keys[0]].max())],
        "action_raw_range": [float(batch["action"].min()), float(batch["action"].max())],
        "stats_action": {k: np.asarray(v).tolist() for k, v in stats["action"].items() if k in ("mean", "std", "min", "max")},
    }

    # ------------------------------------------------------------------ modules
    record["modules"] = {
        "observer": {"class": type(policy.observer).__name__, "params": count_params(policy.observer),
                      "backbone": "torchvision resnet18 IMAGENET1K_V1, FrozenBatchNorm2d, layer4 -> AdaptiveAvgPool(1,1)",
                      "resize_shape": list(cfg.resize_shape), "crop_shape": list(cfg.crop_shape),
                      "note": "random crop in BaseObserver.get_images uses torch.randint unconditionally (also in eval mode)"},
        "obs_encoder": {"class": type(policy.obs_encoder).__name__, "in": policy.obs_dim, "out": policy.latent_dim,
                         "params": count_params(policy.obs_encoder)},
        "action_encoder": {"class": type(policy.action_encoder).__name__, "params": count_params(policy.action_encoder)},
        "action_decoder": {"class": type(policy.action_decoder).__name__, "params": count_params(policy.action_decoder)},
        "flow_net": {"class": type(policy.flow_net).__name__, "params": count_params(policy.flow_net),
                      "hidden_dim": cfg.policy.flow_net.hidden_dim, "num_layers": cfg.policy.flow_net.num_layers},
        "flow_matcher": {"wrapper": type(policy.FM).__name__, "torchcfm_inner": type(policy.FM.fm).__name__,
                          "sigma": cfg.policy.flow_matcher.sigma, "num_sampling_steps": cfg.policy.flow_matcher.num_sampling_steps},
        "policy_total": count_params(policy),
    }

    # ------------------------------------------------------------------ forward shapes
    nb = policy.normalize_inputs(batch)
    nb = policy.normalize_targets(nb)
    with torch.no_grad():
        obs_feat = policy.observer(nb)
        obs_lat = policy.obs_encoder(obs_feat)
        act_lat = policy.action_encoder(nb["action"])
        dec = policy.action_decoder(act_lat)
    record["shapes"] = {
        "obs_features": list(obs_feat.shape),
        "obs_latents_z0": list(obs_lat.shape),
        "action_latents_z1": list(act_lat.shape),
        "decoded_actions": list(dec.shape),
        "normalized_action_range": [float(nb["action"].min()), float(nb["action"].max())],
    }

    # ------------------------------------------------------------------ OT-CFM pairing check
    ot = policy.FM.fm.ot_sampler
    pi = ot.get_map(obs_lat, act_lat)
    i, j = ot.sample_map(pi, obs_lat.shape[0], replace=True)
    diag_mass = float(np.trace(pi) / pi.sum())
    record["ot_cfm_pairing"] = {
        "plan_shape": list(pi.shape),
        "fraction_of_plan_mass_on_diagonal": diag_mass,
        "fraction_sampled_pairs_with_i_eq_j": float(np.mean(i == j)),
        "unique_source_rows_in_sampled_batch": int(len(np.unique(i))),
        "unique_target_rows_in_sampled_batch": int(len(np.unique(j))),
        "sampling_with_replacement": True,
        "conclusion": "the configured `exact` matcher re-pairs obs/action latents by minibatch OT (pot.emd) and "
                      "samples pairs with replacement; data-aligned (obs_i, action_i) correspondence is NOT preserved.",
    }

    # ------------------------------------------------------------------ losses + gradient paths
    policy.zero_grad(set_to_none=True)
    loss, metrics = policy.compute_loss(nb)
    loss.backward()
    weights = {
        "flow_loss": 1.0,
        "consistency_loss": float(cfg.policy.vita.consistency_weight),
        "flow_action_recon_loss": float(cfg.policy.action_ae.flow_recon_weight),
        "enc_action_recon_loss": float(cfg.policy.action_ae.enc_recon_weight),
        "enc_contrastive_loss": float(cfg.policy.vita.enc_contrastive_weight),
        "flow_contrastive_loss": float(cfg.policy.vita.flow_contrastive_weight),
        "action_kl_loss": float(cfg.policy.action_ae.kl_weight),
    }
    record["losses"] = {
        "total": float(loss.item()),
        "components": {k: float(v) for k, v in metrics.items()},
        "weights_from_config": weights,
        "recon_loss_type": cfg.policy.action_ae.recon_loss_type,
        "decode_flow_latents": bool(cfg.policy.vita.decode_flow_latents),
        "all_finite": bool(torch.isfinite(loss).item()),
    }

    def grad_norm(m):
        g = [p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]
        return float(np.sqrt(sum(g))) if g else 0.0

    record["grad_norms_total_loss"] = {
        "observer.backbone": grad_norm(policy.observer.backbone),
        "obs_encoder": grad_norm(policy.obs_encoder),
        "action_encoder": grad_norm(policy.action_encoder),
        "action_decoder": grad_norm(policy.action_decoder),
        "flow_net": grad_norm(policy.flow_net),
    }

    # Gradient path per loss component (isolate by recomputing each term).
    def isolated(fn):
        policy.zero_grad(set_to_none=True)
        val = fn()
        val.backward()
        return {
            "observer.backbone": grad_norm(policy.observer.backbone),
            "obs_encoder": grad_norm(policy.obs_encoder),
            "action_encoder": grad_norm(policy.action_encoder),
            "action_decoder": grad_norm(policy.action_decoder),
            "flow_net": grad_norm(policy.flow_net),
        }

    def fm_only():
        of = policy.observer(nb); z0 = policy.obs_encoder(of); z1 = policy.action_encoder(nb["action"])
        l, _ = policy.FM.compute_loss(policy.flow_net, target=z1, start=z0)
        return l

    def endpoint_consistency_only():
        of = policy.observer(nb); z0 = policy.obs_encoder(of); z1 = policy.action_encoder(nb["action"])
        z1_hat = policy.FM.sample(policy.flow_net, shape=z0.shape, device=z0.device, start=z0, num_steps=policy.num_sampling_steps)
        return torch.nn.functional.mse_loss(z1_hat, z1)

    def flow_recon_only():
        of = policy.observer(nb); z0 = policy.obs_encoder(of)
        z1_hat = policy.FM.sample(policy.flow_net, shape=z0.shape, device=z0.device, start=z0, num_steps=policy.num_sampling_steps)
        return torch.nn.functional.l1_loss(policy.action_decoder(z1_hat), nb["action"])

    def enc_recon_only():
        z1 = policy.action_encoder(nb["action"])
        return torch.nn.functional.l1_loss(policy.action_decoder(z1), nb["action"])

    record["grad_paths_per_loss"] = {
        "flow_loss (FM velocity MSE)": isolated(fm_only),
        "consistency_loss (ODE endpoint vs action latent)": isolated(endpoint_consistency_only),
        "flow_action_recon_loss (decode ODE endpoint, L1)": isolated(flow_recon_only),
        "enc_action_recon_loss (decode encoder latent, L1)": isolated(enc_recon_only),
    }
    policy.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------ ODE sampling behaviour
    policy.eval()
    with torch.no_grad():
        of = policy.observer(nb); z0 = policy.obs_encoder(of)
        n = policy.num_sampling_steps
        z1_hat, (traj, vel) = policy.FM.sample(policy.flow_net, shape=z0.shape, device=z0.device,
                                               start=z0, num_steps=n, return_traces=True)
        acts = policy.action_decoder(z1_hat)
        acts_un = policy.unnormalize_outputs({"action": acts})["action"]
    record["ode_sampling"] = {
        "integrator": "explicit Euler, x <- x + v(x, t) * dt",
        "num_steps": n,
        "dt": 1.0 / n,
        "time_grid_fed_to_velocity_net": [k / n for k in range(n)],
        "note": "time grid is left-endpoint {0, 1/n, ..., (n-1)/n}; t=1 is never queried. Training t ~ U(0,1).",
        "trajectory_len_incl_start": len(traj),
        "trajectory_all_finite": bool(all(torch.isfinite(torch.as_tensor(x)).all().item() for x in traj)),
        "velocity_all_finite": bool(all(np.isfinite(np.asarray(v)).all() for v in vel)),
        "endpoint_shape": list(z1_hat.shape),
        "decoded_action_shape": list(acts.shape),
        "decoded_action_unnormalized_range": [float(acts_un.min()), float(acts_un.max())],
        "gradients_flow_through_sampler_during_training": True,
        "sampler_note": "FM.sample has no torch.no_grad(); endpoint losses back-propagate through all Euler steps "
                        "into flow_net, obs_encoder and the ResNet observer.",
    }

    # ------------------------------------------------------------------ validation-path quirk
    # BasePolicy.validate() normalizes inputs but NOT targets before compute_loss -> val `loss` uses raw actions.
    with torch.no_grad():
        v_in_only = policy.normalize_inputs(batch)
        l_raw, m_raw = policy.compute_loss(v_in_only)
        l_norm, m_norm = policy.compute_loss(policy.normalize_targets(v_in_only))
    record["validation_path_check"] = {
        "compute_loss_with_inputs_normalized_only (what BasePolicy.validate does)": float(l_raw),
        "compute_loss_with_inputs_and_targets_normalized (what training does)": float(l_norm),
        "conclusion": "BasePolicy.validate() omits normalize_targets, so reported offline val `loss`/recon terms are "
                      "computed against un-normalized actions and are not comparable to the training loss. "
                      "`action_mse` (pred un-normalized vs raw gt) is still in correct units.",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
