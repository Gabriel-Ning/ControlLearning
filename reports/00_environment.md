# P0 — Software / Hardware Environment

## Hardware

- GPU: NVIDIA GeForce RTX 5090 Laptop GPU, 24 GB (compute capability 12.0 / sm_120)
- Driver 595.99.02, CUDA 13.2 runtime on host; PyTorch wheel built for CUDA 12.8
- CPU: 24 threads; RAM 125 GB
- OS: Ubuntu 24.04.4 LTS, kernel 7.0.0-38-generic

## Python environment

- Created with `uv venv --python 3.10 .venv` (CPython 3.10.19) at the workspace root.
- Install order (mirrors the upstream README; conda replaced by uv):

```bash
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install "torch==2.9.1" "torchvision==0.24.1"      # resolves to +cu128 wheels
uv pip install -e ./VITA/lerobot
uv pip install -e ./VITA
grep -v '^robomimic' VITA/requirements.txt > /tmp/req.txt && uv pip install -r /tmp/req.txt
uv pip install -e ./VITA/gym-av-aloha
uv pip uninstall torchcodec                                # no system FFmpeg libs -> fall back to pyav
uv pip install transformers matplotlib pandas pytest umap-learn   # CORT-MVP extras
```

- Full lock: `reports/audit_artifacts/pip_freeze.txt` (239 packages).

Key versions:

| Package | Version |
|---|---|
| torch / torchvision | 2.9.1+cu128 / 0.24.1+cu128 |
| torchcfm | 1.0.5 |
| pot (POT) | 0.9.7.post1 |
| timm | 1.0.20 |
| diffusers | 0.30.3 |
| hydra-core / omegaconf | 1.3.2 / 2.3.0 |
| numpy | 1.26.4 |
| einops | 0.8.1 |
| zarr | 2.18.3 |
| datasets | 3.3.2 |
| huggingface-hub | 2.2.0 |
| transformers | 5.19.0 |
| scikit-learn | 1.7.2 |
| av (pyav) | 17.1.0 |
| gymnasium / gym-pusht | 0.29.1 / 0.1.5 |

## Deviations from upstream instructions (documented)

1. `robomimic` skipped — its `egl-probe` dependency fails to compile (cmake error). Only needed for Robomimic tasks.
2. `torchcodec` removed — requires FFmpeg 4–7 shared libraries that are not present; LeRobot's `get_safe_default_codec`
   then selects `pyav`, which works for PushT conversion.
3. `huggingface-hub==2.2.0` (pulled by the current resolver) breaks `timm`'s hub import
   (`has_hf_hub(necessary=True)` raises). DINO weights are therefore downloaded with `hf_hub_download` directly and loaded
   via `load_state_dict` (see `cort_mvp/features/frozen_dino.py`). This does not affect upstream VITA training.
4. Shell proxy variable `ALL_PROXY=socks://127.0.0.1:7897/` is rejected by `httpx`; `unset ALL_PROXY all_proxy` is
   required before any Hugging Face download.

## Pretrained visual backbone availability (checked 2026-10-15)

| Repo | Gated | Status |
|---|---|---|
| `facebook/dinov3-vits16-pretrain-lvd1689m` | manual | **no access** for the logged-in account (GatedRepoError) |
| `timm/vit_small_patch16_dinov3.lvd1689m` (sha `3bf4720a…`) | no | **accessible**; weights load with zero missing/unexpected keys → used as the primary DINOv3 backbone |
| `timm/vit_base_patch16_dinov3.lvd1689m` (sha `c6a5fb7d…`) | no | accessible (optional larger backbone) |
| `facebook/dinov2-small` / `facebook/dinov2-base` | no | accessible (Ablation D fallback) |

DINOv3 ViT-S/16: 21.6 M params, 384-d features; at 224×224 → 201 tokens = 1 CLS + 4 registers + 196 patches.
Preprocessing per timm config: bicubic resize, ImageNet mean/std, `crop_pct=1.0`.
