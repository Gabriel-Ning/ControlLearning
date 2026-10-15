"""Shared, non-trainable action representation (P3 §5.1) and action normalization."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ActionNormalizer(nn.Module):
    """Standardize actions with TRAINING-SET statistics only. Stored as buffers so it travels with checkpoints."""

    def __init__(self, mean: np.ndarray | torch.Tensor, std: np.ndarray | torch.Tensor):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(np.asarray(mean), dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(np.asarray(std), dtype=torch.float32))

    @classmethod
    def from_train_actions(cls, actions: np.ndarray) -> "ActionNormalizer":
        a = np.asarray(actions, dtype=np.float64)
        return cls(a.mean(0), a.std(0) + 1e-8)

    def normalize(self, a: torch.Tensor) -> torch.Tensor:
        return (a - self.mean) / self.std

    def unnormalize(self, a_n: torch.Tensor) -> torch.Tensor:
        return a_n * self.std + self.mean


class FixedActionEmbedding(nn.Module):
    """
    za = W a_norm, with W in R^{latent_dim x action_dim} having orthonormal columns (seeded, fixed across runs).
    Readout is the transpose: a_hat = W^T z. Since W^T W = I, decode(encode(a)) == a exactly.
    """

    def __init__(self, action_dim: int = 2, latent_dim: int = 64, seed: int = 0, scale: float = 1.0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        A = torch.randn(latent_dim, action_dim, generator=g, dtype=torch.float64)
        Q, _ = torch.linalg.qr(A)                      # (latent_dim, action_dim), orthonormal columns
        self.register_buffer("W", (Q * scale).float())
        self.action_dim, self.latent_dim, self.scale = action_dim, latent_dim, scale

    def encode(self, a_norm: torch.Tensor) -> torch.Tensor:
        return a_norm @ self.W.T

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return (z @ self.W) / (self.scale ** 2)

    def forward(self, a_norm):
        return self.encode(a_norm)
