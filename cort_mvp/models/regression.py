"""
Projector (shared by all trainable models), M0 ridge probe, and M1 direct regression.

M1 uses the SAME network class as the velocity net of M2/M3 (VITA `SimpleFlowNet`, imported unmodified) evaluated at a
constant time t=0, so M1/M2/M3 have identical trainable parameter counts. The time-embedding parameters exist but
receive a constant input. Intermediate activations of the residual blocks are exposed for analysis; they are NOT
claimed to be equivalent to FM time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_VITA = Path(__file__).resolve().parents[2] / "VITA"
if str(_VITA) not in sys.path:
    sys.path.insert(0, str(_VITA))
from flare.policies.vita.simple_flow_net import SimpleFlowNet  # noqa: E402  (upstream, unmodified)

from cort_mvp.models.embedding import ActionNormalizer, FixedActionEmbedding  # noqa: E402


class Projector(nn.Module):
    """h (384) -> z0 (latent). kinds: linear | mlp2 | frozen_random (seeded linear, requires_grad=False)."""

    def __init__(self, in_dim: int, latent_dim: int, kind: str = "linear", hidden: int = 256, seed: int = 0):
        super().__init__()
        self.kind = kind
        g = torch.Generator().manual_seed(seed)
        if kind in ("linear", "frozen_random"):
            self.net = nn.Linear(in_dim, latent_dim)
            with torch.no_grad():
                self.net.weight.copy_(torch.randn(latent_dim, in_dim, generator=g) / np.sqrt(in_dim))
                self.net.bias.zero_()
            if kind == "frozen_random":
                self.net.requires_grad_(False)
        elif kind == "mlp2":
            self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, latent_dim))
        else:
            raise ValueError(kind)
        self.in_norm = nn.LayerNorm(in_dim, elementwise_affine=False)   # fixed standardization of frozen features

    def forward(self, h):
        return self.net(self.in_norm(h))


def make_velocity_net(latent_dim: int, hidden_dim: int, num_layers: int, mlp_ratio: float = 4.0) -> SimpleFlowNet:
    return SimpleFlowNet(input_dim=latent_dim, hidden_dim=hidden_dim, output_dim=latent_dim, num_layers=num_layers,
                         mlp_ratio=mlp_ratio, dropout=0.0)


def flownet_forward_with_hidden(net: SimpleFlowNet, x, t):
    """Same computation as SimpleFlowNet.forward but also returns the residual-stream activations after each block."""
    hs = []
    x = net.input_proj(x)
    te = net.time_embed(t)
    for block in net.layers:
        x = block(x, te)
        hs.append(x)
    x = net.norm(x)
    return net.out_proj(x), hs


class DirectRegression(nn.Module):
    """M1: z1_hat = z0 + f(z0, t=0). Loss = lambda_action * MSE(W^T z1_hat, a_norm) + lambda_emb * MSE(z1_hat, za)."""

    def __init__(self, in_dim, latent_dim, hidden_dim, num_layers, projector_kind, normalizer: ActionNormalizer,
                 embedding: FixedActionEmbedding, lambda_action=1.0, lambda_emb=1.0, seed=0, residual=True):
        super().__init__()
        self.projector = Projector(in_dim, latent_dim, projector_kind, seed=seed)
        self.net = make_velocity_net(latent_dim, hidden_dim, num_layers)
        self.normalizer, self.embedding = normalizer, embedding
        self.lambda_action, self.lambda_emb, self.residual = lambda_action, lambda_emb, residual
        self.kind = "m1"

    def predict(self, h, return_hidden=False):
        z0 = self.projector(h)
        t0 = torch.zeros(h.shape[0], device=h.device)
        out, hs = flownet_forward_with_hidden(self.net, z0, t0)
        z1 = z0 + out if self.residual else out
        a_hat = self.embedding.decode(z1)
        if return_hidden:
            return z0, z1, a_hat, hs
        return z0, z1, a_hat

    def losses(self, h, a_raw):
        a_n = self.normalizer.normalize(a_raw)
        za = self.embedding.encode(a_n)
        z0, z1, a_hat = self.predict(h)
        l_action = ((a_hat - a_n) ** 2).mean()
        l_emb = ((z1 - za) ** 2).sum(-1).mean()
        total = self.lambda_action * l_action + self.lambda_emb * l_emb
        return total, {"action": l_action.detach(), "emb": l_emb.detach()}

    @torch.no_grad()
    def trajectory(self, h, n_steps: int = 10):
        """'Representation stages' for M1: [z0, hidden_1..hidden_L (projected to latent dim via out_proj? no), z1].
        We expose z0 and z1 in latent space, plus raw residual-stream activations (hidden_dim) per block."""
        z0, z1, a_hat, hs = self.predict(h, return_hidden=True)
        return {"z0": z0, "z1": z1, "hidden": torch.stack(hs, 0), "a_hat": a_hat}


# ---------------------------------------------------------------------------- M0
def fit_ridge_probe(X_train: np.ndarray, Y_train: np.ndarray, alphas=np.logspace(-3, 4, 15), n_folds: int = 5, seed: int = 0):
    """Ridge regression with alpha selected by K-fold CV on the training set only. Returns (model, mu, sd, alpha)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold

    mu, sd = X_train.mean(0), X_train.std(0) + 1e-6
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    model = RidgeCV(alphas=alphas, cv=cv).fit((X_train - mu) / sd, Y_train)
    return model, mu, sd, float(model.alpha_)
