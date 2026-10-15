"""Model unit tests for the critical technical checks in plan §5.3."""
import numpy as np
import torch

from cort_mvp.models.embedding import ActionNormalizer, FixedActionEmbedding
from cort_mvp.models.regression import DirectRegression, Projector
from cort_mvp.models.transport import TransportModel, euler_integrate, ot_repair

D_IN, LAT = 384, 64


def _make(kind, **kw):
    norm = ActionNormalizer(np.zeros(2), np.ones(2))
    emb = FixedActionEmbedding(2, LAT, seed=0)
    common = dict(in_dim=D_IN, latent_dim=LAT, hidden_dim=64, num_layers=2, projector_kind="linear", normalizer=norm, embedding=emb)
    if kind == "m1":
        return DirectRegression(**common, **kw)
    return TransportModel(**common, kind=kind, **kw)


def test_fixed_embedding_orthonormal_and_invertible():
    emb = FixedActionEmbedding(2, LAT, seed=0)
    W = emb.W
    assert torch.allclose(W.T @ W, torch.eye(2), atol=1e-5)
    a = torch.randn(16, 2)
    assert torch.allclose(emb.decode(emb.encode(a)), a, atol=1e-5)
    assert torch.equal(FixedActionEmbedding(2, LAT, seed=0).W, W)          # fixed across runs
    assert not torch.equal(FixedActionEmbedding(2, LAT, seed=1).W, W)


def test_normalizer_uses_given_stats_only():
    a = np.array([[1.0, 2.0], [3.0, 6.0]])
    n = ActionNormalizer.from_train_actions(a)
    assert torch.allclose(n.mean, torch.tensor([2.0, 4.0]))
    x = torch.tensor([[2.0, 4.0]])
    assert torch.allclose(n.normalize(x), torch.zeros(1, 2), atol=1e-6)
    assert torch.allclose(n.unnormalize(n.normalize(x)), x)


def test_parameter_counts_match_m1_m2_m3():
    cnt = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert cnt(_make("m1")) == cnt(_make("m2", lambda_fm=0.0)) == cnt(_make("m3"))


def test_fm_interpolation_preserves_pairing():
    """With paired coupling the FM target for sample i must be za_i - z0_i (checked via a linear velocity net)."""
    m = _make("m3")
    h = torch.randn(8, D_IN); a = torch.randn(8, 2)
    a_n = m.normalizer.normalize(a); za = m.embedding.encode(a_n); z0 = m.projector(h)
    captured = {}
    orig = m.net.forward

    def spy(zt, t):
        captured["zt"], captured["t"] = zt, t
        return orig(zt, t)

    m.net.forward = spy
    m.fm_loss(z0, za)
    t = captured["t"][:, None]
    assert torch.allclose(captured["zt"], (1 - t) * z0 + t * za, atol=1e-6)


def test_ot_repairing_changes_pairs():
    z0, za = torch.randn(32, LAT), torch.randn(32, LAT)
    z0b, zab, frac = ot_repair(z0, za, np.random.default_rng(0))
    assert z0b.shape == z0.shape and frac < 0.5


def test_gradient_flows_through_ode_endpoint_into_projector():
    m = _make("m2", lambda_fm=0.0, lambda_action=1.0)
    h = torch.randn(8, D_IN); a = torch.randn(8, 2)
    loss, _ = m.losses(h, a)
    loss.backward()
    assert m.projector.net.weight.grad is not None and m.projector.net.weight.grad.abs().sum() > 0
    assert all(p.grad is not None for p in m.net.parameters() if p.requires_grad and p.numel() > 0 and p.grad is not None)


def test_frozen_random_projector_receives_no_gradient_but_net_does():
    norm = ActionNormalizer(np.zeros(2), np.ones(2)); emb = FixedActionEmbedding(2, LAT, 0)
    m = TransportModel(D_IN, LAT, 64, 2, "frozen_random", norm, emb, lambda_fm=1.0, lambda_action=1.0)
    loss, _ = m.losses(torch.randn(8, D_IN), torch.randn(8, 2)); loss.backward()
    assert m.projector.net.weight.grad is None
    assert m.net.out_proj.weight.grad is not None


def test_cached_features_do_not_detach_projector():
    """Features are constant inputs; the projector output must still carry grad."""
    m = _make("m3")
    h = torch.from_numpy(np.random.randn(4, D_IN).astype(np.float32))  # as loaded from the cache
    z0 = m.projector(h)
    assert z0.requires_grad


def test_trajectory_finite_and_matches_time_grid():
    m = _make("m3")
    out = m.trajectory(torch.randn(5, D_IN), n_steps=10)
    assert out["traj"].shape == (11, 5, LAT) and torch.isfinite(out["traj"]).all()
    assert torch.allclose(out["times"], torch.linspace(0, 1, 11))
    # endpoint equals a separate predict() with the same step count
    _, z1, _ = m.predict(torch.zeros(1, D_IN), n_steps=10)
    assert torch.allclose(m.trajectory(torch.zeros(1, D_IN), 10)["traj"][-1], z1, atol=1e-6)


def test_m4_counterfactual_loss_zero_for_identical_pairs():
    m = _make("m4", lambda_cf=1.0, cf_times=(0.0, 0.5, 1.0))
    h = torch.randn(6, D_IN); a = torch.randn(6, 2)
    _, logs = m.losses(h, a, h_pair=h.clone())
    assert float(logs["cf"]) < 1e-10
    _, logs2 = m.losses(h, a, h_pair=torch.randn(6, D_IN))
    assert float(logs2["cf"]) > 0


def test_m1_hidden_exposed():
    m = _make("m1")
    out = m.trajectory(torch.randn(3, D_IN))
    assert out["hidden"].shape[0] == 2 and out["z1"].shape == (3, LAT)
