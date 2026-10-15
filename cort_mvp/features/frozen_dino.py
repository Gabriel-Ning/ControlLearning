"""
Frozen pretrained visual backbones (P2).

Primary:  DINOv3 ViT-S/16  (timm `vit_small_patch16_dinov3.lvd1689m`, HF repo `timm/vit_small_patch16_dinov3.lvd1689m`)
Fallback: DINOv2 ViT-S/14  (timm `vit_small_patch14_dinov2.lvd142m`, run at 224px with resampled pos-embed)

Both are redistributed, ungated timm checkpoints (the `facebook/dinov3-*` repos are gated and were not accessible).

Feature strategies cached for every image:
  cls         : CLS token after the final norm                        (primary feature `h` for all models)
  patch_mean  : mean of patch tokens (register tokens excluded)       (diagnostic alternative)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import cort_mvp  # noqa: F401  (installs the huggingface_hub>=2 compatibility shim before timm is imported)
import huggingface_hub as _hf
import timm  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

BACKBONES = {
    "dinov3_vits16": dict(timm_name="vit_small_patch16_dinov3.lvd1689m", img_size=224),
    "dinov3_vitb16": dict(timm_name="vit_base_patch16_dinov3.lvd1689m", img_size=224),
    "dinov2_vits14": dict(timm_name="vit_small_patch14_dinov2.lvd142m", img_size=224),
    "dinov2_vitb14": dict(timm_name="vit_base_patch14_dinov2.lvd142m", img_size=224),
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class BackboneInfo:
    name: str
    timm_name: str
    hf_hub_id: str
    hf_revision: str
    img_size: int
    feature_dim: int
    num_prefix_tokens: int
    num_params: int
    preprocessing: str


class FrozenDino(nn.Module):
    def __init__(self, name: str = "dinov3_vits16", device: str = "cuda"):
        super().__init__()
        spec = BACKBONES[name]
        self.name = name
        self.model = timm.create_model(spec["timm_name"], pretrained=True, num_classes=0, img_size=spec["img_size"])
        self.model.eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.to(device)
        self.device = device
        pc = self.model.pretrained_cfg
        hub_id = pc.get("hf_hub_id", "")
        revision = "unknown"
        try:
            p = _hf.hf_hub_download(hub_id, "model.safetensors")
            m = re.search(r"/snapshots/([0-9a-f]{40})/", p)
            if m:
                revision = m.group(1)
        except Exception:
            pass
        self.info = BackboneInfo(
            name=name, timm_name=spec["timm_name"], hf_hub_id=hub_id, hf_revision=revision, img_size=spec["img_size"],
            feature_dim=int(self.model.num_features), num_prefix_tokens=int(self.model.num_prefix_tokens),
            num_params=sum(p.numel() for p in self.model.parameters()),
            preprocessing=f"uint8 RGB -> float [0,1] -> resize to {spec['img_size']} (bilinear, if needed) -> "
                          f"normalize ImageNet mean/std; eval mode, fp32, no augmentation",
        )

    def train(self, mode: bool = True):  # always frozen / eval
        return super().train(False)

    @torch.no_grad()
    def forward(self, images_uint8_hwc: torch.Tensor | None = None, *, images_float_chw: torch.Tensor | None = None) -> dict:
        if images_float_chw is None:
            x = images_uint8_hwc.to(self.device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
        else:
            x = images_float_chw.to(self.device, non_blocking=True).float()
        if x.shape[-1] != self.info.img_size:
            x = torch.nn.functional.interpolate(x, size=(self.info.img_size, self.info.img_size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        tokens = self.model.forward_features(x)           # (B, 1 + regs + P, D), after final norm
        npre = self.info.num_prefix_tokens
        return {"cls": tokens[:, 0], "patch_mean": tokens[:, npre:].mean(1)}
