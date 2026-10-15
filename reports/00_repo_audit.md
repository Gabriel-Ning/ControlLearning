# P0 — VITA Repository Audit

- Upstream: https://github.com/ucd-dare/VITA
- Commit audited: `f9ce88edaafe33aa673ad8ad0e6431ea26dd3b81` (2026-05-13, "add meanflow/improved meanflow")
- Clone method: `git clone --depth 1 --filter=blob:none` (full clone repeatedly failed with `early EOF` on this network; the shallow clone contains the full working tree at the audited commit).
- Local path: `VITA/` (left unmodified; all new code lives in `cort_mvp/`).
- Machine-readable artifact: `reports/audit_artifacts/vita_audit.json` (produced by `python -m cort_mvp.audit.audit_vita`).

## 1. Component inventory (default `policy=vita task=pusht`)

| Component | File | Class | Params | Notes |
|---|---|---|---|---|
| Visual observer | `flare/policies/observers/resnet_observer.py` | `ResNetObserver` | 11.17 M (all trainable) | torchvision ResNet-18 (ImageNet1K_V1), `FrozenBatchNorm2d`, `layer4` → `AdaptiveAvgPool2d(1,1)` → 512-d per image; concatenated with the raw proprio state (dim 2 for PushT) → `obs_dim = 514`. Trained end-to-end (same LR as rest; `optimizer_lr_backbone` exists in config but is **not used** by `VitaPolicy.get_optimizer`). |
| Observation encoder | `vita_policy.py` | `nn.Linear(514, 512)` | 0.26 M | Maps observer features to flow source `z0`. |
| Action encoder | `flare/networks/vita/action_ae.py` | `CNNActionEncoder` | 4.20 M | 4× `Conv1d(k=5, s=2)` over the 16-step action chunk → flatten → `Linear` → 512-d `z1`. Trainable, orthogonal init. |
| Action decoder | `action_ae.py` | `SimpleActionDecoder` | 2.38 M | MLP `512 → 512 → (4× Mlp) → 16·2`. |
| Flow velocity net | `flare/policies/vita/simple_flow_net.py` | `SimpleFlowNet` | 12.87 M | `Linear(512→512)` → 4× `FlowNetLayer` (LayerNorm + Mlp(ratio 4) with adaLN-style `(gamma, scale, shift)` time modulation, zero-init) → LayerNorm → `Linear`. Sinusoidal time embedding (256-d) → MLP. |
| Flow matcher | `flare/flow/flow_matchers.py` | `ExactOptimalTransportConditionalFlowMatcher` (wraps `torchcfm`) | – | `sigma=0.0`, `num_sampling_steps=6`. |
| Total | | `VitaPolicy` | 30.88 M | |

### Configuration mismatch (confirmed)

`flare/configs/policy/vita.yaml` sets `flow_net.name: simple`, while `VitaPolicy.__init__` accepts only
`"simple_flow_net"` or `"simple_mean_flow_net"`. Running the documented command verbatim fails:

```
ValueError: Unsupported flow_net: simple
```

Workaround used everywhere in this project: append `policy.flow_net.name=simple_flow_net`.

## 2. Training losses (all active by default)

`VitaPolicy.compute_loss` (after `BasePolicy.forward` normalizes inputs **and** targets):

| Term | Formula | Weight (config key) | Measured value (step 0, bs=32) |
|---|---|---|---|
| `flow_loss` | `MSE(v_θ(z_t, t), u_t)` with `(z0, z1)` from OT-CFM | 1.0 (implicit) | 1.63 |
| `consistency_loss` (FLC) | `MSE(ODE_endpoint(z0), z1)` | 1.0 (`vita.consistency_weight`) | 1.59 |
| `flow_action_recon_loss` (FLD) | `L1(Dec(ODE_endpoint(z0)), a)` | 0.5 (`action_ae.flow_recon_weight`) | 0.67 |
| `enc_action_recon_loss` (AE) | `L1(Dec(z1), a)` | 0.5 (`action_ae.enc_recon_weight`) | 0.66 |
| `enc_contrastive_loss` | InfoNCE(z0, z1) | 0.0 | – |
| `flow_contrastive_loss` | InfoNCE(z0, ẑ1) | 0.0 | – |
| `action_kl_loss` | KL (VAE only) | 0.0 (`use_variational: false`) | – |

Total at step 0: 3.88, finite.

So the shipped "VITA" objective is **not** pure cross-modal flow matching: it is FM + endpoint-consistency + two
decoder-reconstruction losses, all back-propagated into the ResNet, the obs encoder, the action AE and the velocity net.
The MVP therefore treats "FM only" vs "FM + endpoint" vs "endpoint only" as explicit ablations (Ablation B).

## 3. Gradient paths (measured per isolated loss term)

Grad-norms when back-propagating a single term:

| Term | ResNet backbone | obs_encoder | action_encoder | action_decoder | flow_net |
|---|---|---|---|---|---|
| flow_loss | 28.1 | 4.66 | 0.95 | 0 | 3.42 |
| consistency_loss | 30.7 | 4.62 | 0.13 | 0 | 3.22 |
| flow_action_recon_loss | 1.22 | 0.43 | 0 | 0.81 | 0.30 |
| enc_action_recon_loss | 0 | 0 | 0.009 | 0.093 | 0 |

Key facts:

- `FM.sample` has **no** `torch.no_grad()`; the endpoint losses back-propagate through all 6 Euler steps into
  `flow_net`, `obs_encoder` and the ResNet observer. (Relevant for M2/M3: gradient through the ODE endpoint into the
  projector is the upstream behaviour.)
- The FM velocity loss itself also trains the *action encoder* (target `z1` is not detached) and the visual backbone.
- The action decoder only receives gradient from the two reconstruction terms.

## 4. OT-CFM source–target coupling (confirmed: pairs are re-assigned)

`flow_matcher.name: exact` → `torchcfm.ExactOptimalTransportConditionalFlowMatcher.sample_location_and_conditional_flow`
calls `OTPlanSampler.sample_plan(x0, x1)` which:

1. solves exact minibatch OT (`pot.emd`, squared-Euclidean cost) between the batch of obs latents and action latents,
2. samples `(i, j)` **with replacement** from the plan, returning `x0[i], x1[j]`.

Measured on a real PushT batch (bs=32, step-0 weights):

- fraction of plan mass on the diagonal: **0.125**
- fraction of sampled pairs with `i == j`: **0.031**
- unique source rows used: 22 / 32, unique target rows used: 22 / 32

Conclusion: the official configuration **does not preserve the data-aligned (image_i, action_i) correspondence** inside a
minibatch — the velocity target for `z0_i` is usually the action latent of a different sample, and ~30 % of the batch
is dropped/duplicated per step. The MVP primary model (M3) uses strictly paired interpolation
(`z_t = (1-t) z0_i + t z1_i`, `u = z1_i − z0_i`); the OT re-pairing is kept only as Ablation F.
(`flow_matcher.name=conditional` in the upstream repo gives paired `ConditionalFlowMatcher` behaviour.)

## 5. ODE integration and trajectory tracing

`TorchFlowMatcher.sample`: explicit Euler, `dt = 1/n`, velocity queried at the **left endpoints**
`t ∈ {0, 1/n, …, (n−1)/n}` (`t = 1` is never queried; training samples `t ~ U(0,1)`). With `return_traces=True` it returns
the full latent history (n+1 states incl. start) and velocity history; all finite on the audited batch.
`visualize()` decodes each intermediate latent with the action decoder for the "denoising" plots.

## 6. Dataset interface and preprocessing

- `gym-av-aloha/scripts/convert.py` → `create_av_aloha_dataset_from_lerobot` converts a LeRobot HF dataset to a local
  Zarr `ReplayBuffer` (images stored as uint8 HWC, resized to the task's `image_size`; stats aggregated from LeRobot
  episode stats).
- `AVAlohaDataset.__getitem__` returns `observation.image` as float `[0,1]` (`B,S,C,H,W`), `observation.state`,
  `action` chunk of `pred_horizon` = 16 (padded at episode end), plus `*_is_pad` masks.
- `BasePolicy` normalizes images with the task's `override_stats` (ImageNet mean/std), state and action with dataset
  mean/std. Observer resizes to `resize_shape` and random-crops to `crop_shape` (**also in eval mode**, see §8).
- Train/val split is by episode index (last `val.num_episodes` episodes = val). The log line
  `Val: 738 frames | 5 episodes` reports the *whole-dataset* counts (`num_frames` comes from meta), not the split.

## 7. Checkpoints and optimizer

- `AdamW(lr=1e-4, betas=(0.95, 0.999), eps=1e-8, wd=1e-6)`, cosine schedule with 500 warm-up steps, grad-clip 1.0,
  EMA (`power=0.75`, used only for online eval).
- Checkpoints: `model.safetensors` (via `PyTorchModelHubMixin.save_pretrained`) + `training_state.pt`
  (optimizer/scheduler/EMA/step). "best" checkpoints are selected on **online** env success — not applicable offline.

## 8. Other observations relevant to the MVP

1. **Validation-loss unit bug (upstream):** `BasePolicy.validate()` calls `normalize_inputs` but not
   `normalize_targets` before `compute_loss`. Offline val `loss`/recon metrics are therefore computed against raw
   (pixel-unit) actions: measured 289.3 vs 3.96 for the identical batch with targets normalized. `action_mse` is
   computed in raw units and is correct. Training is unaffected.
2. **Non-deterministic eval:** the random crop in `BaseObserver.get_images` is applied regardless of `self.training`.
3. `flare/utils/eval.py` (imported by `train.py`) globally sets `torch.use_deterministic_algorithms(True, warn_only=True)`,
   disables TF32 and sets `CUBLAS_WORKSPACE_CONFIG`.
4. `robomimic` from `requirements.txt` fails to build (`egl-probe` cmake error) on this machine; it is only needed for
   Robomimic tasks and was skipped.
5. `torchcodec==0.9.0` (pulled by LeRobot) cannot find FFmpeg shared libraries here; it was uninstalled so LeRobot falls
   back to the `pyav` decoder (`av==17.1.0`). Dataset conversion then works.

## 9. Smoke test (acceptance criterion for P0)

Dataset subset (5 PushT episodes, 738 frames) converted with:

```bash
unset ALL_PROXY all_proxy   # httpx rejects socks:// proxy URLs
export FLARE_DATASETS_DIR=$PWD/VITA/gym-av-aloha/outputs
python - <<'EOF'
from gym_av_aloha.datasets.av_aloha_dataset import create_av_aloha_dataset_from_lerobot
create_av_aloha_dataset_from_lerobot(episodes={"lerobot/pusht": list(range(5))},
                                     repo_id="lerobot/pusht", remove_keys=[], image_size=(96, 96))
EOF
```

Working original-VITA command (run from `VITA/`):

```bash
python flare/train.py policy=vita task=pusht session=smoke \
  policy.flow_net.name=simple_flow_net wandb.enable=false \
  val.val_online_freq=0 val.val_offline_freq=20 val.num_episodes=1 \
  train.steps=40 train.log_freq=10 train.save_freq=40 train.num_workers=0 \
  train.batch_size=32 val.batch_size=32
```

Result: train loss 3.93 → 2.09 over 40 steps (finite), offline validation ran twice (ODE sampling → decoded actions of
shape `(B, 16, 2)`, `action_mse` in pixel units ≈ 8.6e3 at this untrained stage), checkpoint saved to
`VITA/flare_outputs/lerobot_pusht/vita/smoke/checkpoints/step_0000000040/`.

**P0 acceptance: PASSED** — a valid batch passes through the original pipeline, FM loss is finite, ODE sampling returns
actions with the expected dimensions.
