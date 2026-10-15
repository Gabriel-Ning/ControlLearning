"""
Transport models (P3 §5.2): M2 (endpoint-only residual ODE), M3 (paired FM + endpoint), M4 (M3 + counterfactual
invariance), plus the OT-CFM re-pairing variant for Ablation F.

    z0 = projector(h)                    za = W a_norm
    FM:        t ~ U(0,1), zt = (1-t) z0 + t za, u = za - z0,  L_fm = MSE(v(zt,t), u)      (strictly paired i<->i)
    Endpoint:  z1 = Euler(z0; n_steps),  L_action = MSE(W^T z1, a_norm),  L_emb = ||z1 - za||^2 (optional)
    M4:        L_cf = sum_{t in cf_times} MSE(z_t(h_a), z_t(h_b)) for counterfactual image pairs (a,b)

The Euler integrator mirrors upstream VITA (`TorchFlowMatcher.sample`): left-endpoint time grid, dt = 1/n.
Gradients flow through every Euler step (as upstream). All models share ActionNormalizer and FixedActionEmbedding.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from cort_mvp.models.embedding import ActionNormalizer, FixedActionEmbedding
from cort_mvp.models.regression import Projector, make_velocity_net


def euler_integrate(net: nn.Module, z0: torch.Tensor, n_steps: int, *, return_traj: bool = False):
    """x <- x + v(x, t_k) * dt with t_k = k/n, k = 0..n-1 (identical to upstream VITA)."""
    z = z0
    dt = 1.0 / n_steps
    traj = [z0]
    for k in range(n_steps):
        t = torch.full((z.shape[0],), k / n_steps, device=z.device, dtype=z.dtype)
        z = z + net(z, t) * dt
        if return_traj:
            traj.append(z)
    return (z, torch.stack(traj, 0)) if return_traj else z


def ot_repair(z0: torch.Tensor, za: torch.Tensor, rng: np.random.Generator):
    """Ablation F: upstream OT-CFM coupling (exact minibatch OT via POT, sampled with replacement)."""
    import ot as pot

    with torch.no_grad():
        M = torch.cdist(z0.detach(), za.detach()) ** 2
        a, b = pot.unif(z0.shape[0]), pot.unif(za.shape[0])
        pi = pot.emd(a, b, M.cpu().numpy())
        p = pi.flatten(); p = p / p.sum()
        choices = rng.choice(pi.size, p=p, size=z0.shape[0], replace=True)
        i, j = np.divmod(choices, pi.shape[1])
    i, j = torch.as_tensor(i, device=z0.device), torch.as_tensor(j, device=z0.device)
    return z0[i], za[j], float((i == j).float().mean())


class TransportModel(nn.Module):
    def __init__(self, in_dim, latent_dim, hidden_dim, num_layers, projector_kind, normalizer: ActionNormalizer,
                 embedding: FixedActionEmbedding, *, n_steps=8, lambda_fm=1.0, lambda_action=1.0, lambda_emb=0.0,
                 lambda_cf=0.0, cf_times=(1.0,), pairing="paired", seed=0, kind="m3"):
        super().__init__()
        self.projector = Projector(in_dim, latent_dim, projector_kind, seed=seed)
        self.net = make_velocity_net(latent_dim, hidden_dim, num_layers)
        self.normalizer, self.embedding = normalizer, embedding
        self.n_steps = n_steps
        self.lambda_fm, self.lambda_action, self.lambda_emb, self.lambda_cf = lambda_fm, lambda_action, lambda_emb, lambda_cf
        self.cf_times = tuple(cf_times)
        assert pairing in ("paired", "ot")
        self.pairing = pairing
        self._ot_rng = np.random.default_rng(seed)
        self.kind = kind

    # ------------------------------------------------------------------ inference
    def predict(self, h, n_steps: int | None = None):
        z0 = self.projector(h)
        z1 = euler_integrate(self.net, z0, n_steps or self.n_steps)
        return z0, z1, self.embedding.decode(z1)

    @torch.no_grad()
    def trajectory(self, h, n_steps: int = 10):
        """Numerical ODE trajectory: states at t = k/n for k = 0..n  -> (n+1, B, D). Default n=10 gives t=0.0..1.0."""
        z0 = self.projector(h)
        z1, traj = euler_integrate(self.net, z0, n_steps, return_traj=True)
        return {"traj": traj, "times": torch.linspace(0, 1, n_steps + 1), "a_hat": self.embedding.decode(z1)}

    # ------------------------------------------------------------------ losses
    def fm_loss(self, z0, za):
        if self.pairing == "ot":
            z0, za, frac_diag = ot_repair(z0, za, self._ot_rng)
        else:
            frac_diag = 1.0
        t = torch.rand(z0.shape[0], device=z0.device, dtype=z0.dtype)
        zt = (1 - t[:, None]) * z0 + t[:, None] * za
        u = za - z0
        v = self.net(zt, t)
        return ((v - u) ** 2).mean(), frac_diag

    def endpoint_losses(self, z0, za, a_n):
        z1, traj = euler_integrate(self.net, z0, self.n_steps, return_traj=True)
        a_hat = self.embedding.decode(z1)
        l_action = ((a_hat - a_n) ** 2).mean()
        l_emb = ((z1 - za) ** 2).sum(-1).mean()
        return l_action, l_emb, traj

    def losses(self, h, a_raw, h_pair: torch.Tensor | None = None):
        a_n = self.normalizer.normalize(a_raw)
        za = self.embedding.encode(a_n)
        z0 = self.projector(h)
        logs, total = {}, 0.0
        if self.lambda_fm > 0:
            l_fm, frac = self.fm_loss(z0, za)
            total = total + self.lambda_fm * l_fm
            logs["fm"] = l_fm.detach(); logs["pair_frac_diag"] = torch.tensor(frac)
        need_endpoint = self.lambda_action > 0 or self.lambda_emb > 0 or (self.lambda_cf > 0 and h_pair is not None)
        traj = None
        if need_endpoint:
            l_action, l_emb, traj = self.endpoint_losses(z0, za, a_n)
            total = total + self.lambda_action * l_action + self.lambda_emb * l_emb
            logs["action"] = l_action.detach(); logs["emb"] = l_emb.detach()
        if self.lambda_cf > 0 and h_pair is not None:
            z0b = self.projector(h_pair)
            _, trajb = euler_integrate(self.net, z0b, self.n_steps, return_traj=True)
            l_cf = 0.0
            for tt in self.cf_times:
                k = int(round(tt * self.n_steps))
                l_cf = l_cf + ((traj[k] - trajb[k]) ** 2).sum(-1).mean()
            l_cf = l_cf / len(self.cf_times)
            total = total + self.lambda_cf * l_cf
            logs["cf"] = l_cf.detach()
        return total, logs

    def separate_grad_vectors(self, h, a_raw):
        """For the gradient-conflict diagnostic: flattened grads of L_fm and L_action over shared params."""
        params = [p for p in self.parameters() if p.requires_grad]
        a_n = self.normalizer.normalize(a_raw); za = self.embedding.encode(a_n)
        z0 = self.projector(h)
        l_fm, _ = self.fm_loss(z0, za)
        g_fm = torch.autograd.grad(l_fm, params, retain_graph=False, allow_unused=True)
        z0 = self.projector(h)
        l_action, _, _ = self.endpoint_losses(z0, za, a_n)
        g_ac = torch.autograd.grad(l_action, params, allow_unused=True)
        flat = lambda gs: torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(gs, params)])
        return flat(g_fm), flat(g_ac)
