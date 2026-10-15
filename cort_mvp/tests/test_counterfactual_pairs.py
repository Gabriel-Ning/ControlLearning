"""
P1 validation tests. Run with:  pytest cort_mvp/tests/test_counterfactual_pairs.py -q
Set CORT_DATA=<root> to test a generated dataset; otherwise a tiny dataset is generated into a temp dir.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cort_mvp.data import reaching_render as R
from cort_mvp.data.reaching_dataset import ReachingSplit, cf_splits, load_manifest

WS = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    env = os.environ.get("CORT_DATA")
    if env:
        return Path(env)
    out = tmp_path_factory.mktemp("reach") / "tiny"
    subprocess.check_call([sys.executable, "-m", "cort_mvp.data.generate_reaching", "--out", str(out), "--scale", "0.004",
                           "--workers", "2"], cwd=WS)
    return out


def test_oracle_action_formula():
    s = np.array([[0.1, -0.2, 0.5, 0.3], [0.0, 0.0, 2.0, -2.0]])
    a = R.oracle_action(s)
    assert np.allclose(a[0], [0.4, 0.5])
    assert np.allclose(a[1], [R.A_MAX, -R.A_MAX])


def test_render_deterministic_and_nuisance_roundtrip():
    rng = np.random.default_rng(3)
    st = R.sample_states(rng, 1, geo_holdout=False)[0]
    eta = R.sample_nuisance(rng, 6, st)
    a = R.render(st, eta)
    b = R.render(st, R.Nuisance.from_dict(json.loads(json.dumps(eta.to_dict()))))
    assert a.shape == (224, 224, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b)


def test_foreground_pixels_identical_across_nuisance():
    """Markers must be unchanged by nuisance: pixels strictly inside the ee disk and goal dot are identical."""
    rng = np.random.default_rng(5)
    st = R.sample_states(rng, 1, geo_holdout=False)[0]
    imgs = [R.render(st, R.sample_nuisance(rng, f, st)) for f in (0, 2, 6, 7)]
    ee_mask = R._disk_mask(st[0], st[1], R.EE_RADIUS - 0.02) > 0.99
    goal_mask = R._annulus_mask(st[2], st[3], R.GOAL_R_IN + 0.01, R.GOAL_R_OUT - 0.01) > 0.99
    assert ee_mask.sum() > 20 and goal_mask.sum() > 20
    for im in imgs[1:]:
        assert np.array_equal(im[ee_mask], imgs[0][ee_mask])
        assert np.array_equal(im[goal_mask], imgs[0][goal_mask])


def test_counterfactual_variants_share_state_and_action(data_root):
    base = ReachingSplit(data_root, "cf_test_v0", load_images=False)
    for name in cf_splits()[1:]:
        s = ReachingSplit(data_root, name, load_images=False)
        assert np.array_equal(s.meta["state_id"], base.meta["state_id"])
        assert np.array_equal(s.meta["states"], base.meta["states"])
        assert np.array_equal(s.meta["actions"], base.meta["actions"])
        assert np.array_equal(s.meta["rel_goal"], base.meta["rel_goal"])
        # nuisance must actually differ
        assert not np.array_equal(s.meta["nuisance_seed"], base.meta["nuisance_seed"])
    # recompute oracle actions from stored states
    assert np.allclose(R.oracle_action(base.meta["states"].astype(np.float64)), base.meta["actions"], atol=1e-5)
    # images differ between variants (nuisance applied)
    a = ReachingSplit(data_root, "cf_test_v0"); b = ReachingSplit(data_root, "cf_test_2".replace("2", "v2"))
    assert not np.array_equal(np.asarray(a.images[0]), np.asarray(b.images[0]))


def test_cf_family_assignment(data_root):
    man = load_manifest(data_root)
    seen, unseen = set(man["seen_families"]), set(man["unseen_families"])
    for v in (0, 1):
        assert set(ReachingSplit(data_root, f"cf_test_v{v}", load_images=False).meta["bg_family"].tolist()) <= seen
    for v in (2, 3):
        assert set(ReachingSplit(data_root, f"cf_test_v{v}", load_images=False).meta["bg_family"].tolist()) <= unseen


def test_state_perturbation_controls_share_nuisance(data_root):
    cf0 = ReachingSplit(data_root, "cf_test_v0", load_images=False)
    for v in (0, 1):
        sp = ReachingSplit(data_root, f"sp_test_v{v}", load_images=False)
        assert np.array_equal(sp.meta["nuisance_seed"], cf0.meta["nuisance_seed"])
        assert np.array_equal(sp.meta["bg_family"], cf0.meta["bg_family"])
        assert np.array_equal(sp.meta["state_id"], cf0.meta["state_id"])
        assert not np.allclose(sp.meta["states"], cf0.meta["states"])
    sp0 = ReachingSplit(data_root, "sp_test_v0", load_images=False)
    d = np.linalg.norm(sp0.meta["states"] - cf0.meta["states"], axis=1)
    assert d.max() <= 0.2 + 1e-5


def test_no_state_leakage_between_independent_splits(data_root):
    names = ["train", "val", "id_test", "ood_bg", "geo_ood", "cf_test_v0"]
    seen = {}
    for n in names:
        st = ReachingSplit(data_root, n, load_images=False).meta["states"]
        for row in np.round(st.astype(np.float64), 5):
            key = tuple(row)
            assert key not in seen, f"state leak between {seen.get(key)} and {n}"
            seen[key] = n


def test_split_family_membership_and_geo_holdout(data_root):
    man = load_manifest(data_root)
    seen, unseen = set(man["seen_families"]), set(man["unseen_families"])
    for n in ("train", "val", "id_test", "geo_ood"):
        assert set(ReachingSplit(data_root, n, load_images=False).meta["bg_family"].tolist()) <= seen
    assert set(ReachingSplit(data_root, "ood_bg", load_images=False).meta["bg_family"].tolist()) <= unseen
    geo = ReachingSplit(data_root, "geo_ood", load_images=False).meta["states"]
    assert R.in_geo_holdout(geo).all()
    for n in ("train", "val", "id_test", "ood_bg"):
        assert not R.in_geo_holdout(ReachingSplit(data_root, n, load_images=False).meta["states"]).any()


def test_dataset_item_and_clipping_rate(data_root):
    ds = ReachingSplit(data_root, "train", as_tensor=True)
    it = ds[0]
    assert it["image"].shape == (3, 224, 224) and it["image"].dtype.is_floating_point
    assert 0.0 <= float(it["image"].min()) and float(it["image"].max()) <= 1.0
    assert ds.meta["clipped"].mean() < 0.02
