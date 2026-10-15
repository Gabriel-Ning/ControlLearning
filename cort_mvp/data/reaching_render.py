"""
Deterministic NumPy renderer for the 2D reaching benchmark.

World frame: x, y in [-1, 1], y pointing up.  Fixed orthographic camera, 224x224 px.

Physical state x = [ee_x, ee_y, goal_x, goal_y]
Oracle action   a = clip(K * (goal - ee), -A_MAX, A_MAX)

Nuisance parameters (eta) are an explicit dataclass with continuous fields (hue, value, texture
frequency/phase/angle, distractor positions...) so that `render(x, eta)` is a deterministic function
and continuous nuisance factors can be finite-differenced.

Layering (bottom -> top):  background family -> distractors -> halo (foreground-adjacent nuisance)
                           -> arm -> goal marker -> ee marker
Distractors and the halo are placed so they never overlap the ee / goal markers.
"""
from __future__ import annotations

import colorsys
from dataclasses import dataclass, field, asdict

import numpy as np

IMG = 224
K_GAIN = 1.0
A_MAX = 1.5
WORKSPACE = 0.8          # states sampled in [-WORKSPACE, WORKSPACE]^2
GEO_HOLDOUT_BOX = (0.3, 0.8)   # goal in [0.3,0.8]^2 is held out for the geometry-OOD split

# Foreground geometry (world units)
EE_RADIUS = 0.07
GOAL_R_IN, GOAL_R_OUT = 0.045, 0.085
GOAL_DOT_R = 0.02
ARM_BASE = np.array([0.0, -1.12])
ARM_L1, ARM_L2 = 1.15, 1.15
ARM_WIDTH = 0.028
EE_COLOR = np.array([30, 90, 235], dtype=np.float32)
GOAL_COLOR = np.array([235, 45, 45], dtype=np.float32)
ARM_COLOR = np.array([95, 95, 100], dtype=np.float32)
JOINT_COLOR = np.array([60, 60, 65], dtype=np.float32)

# Marker hues to avoid for nuisance colours (blue ~0.63, red ~0.0)
_FORBIDDEN_HUES = (0.63, 0.0)
_HUE_MARGIN = 0.09

SEEN_FAMILIES = (0, 1, 2, 3, 4)
UNSEEN_FAMILIES = (5, 6, 7, 8)
FAMILY_NAMES = {
    0: "solid", 1: "hstripes", 2: "checker", 3: "radial", 4: "smooth_noise",
    5: "dstripes", 6: "blobs", 7: "white_noise", 8: "multi_octave",
}

_ys, _xs = np.mgrid[0:IMG, 0:IMG]
WX = (_xs + 0.5) / IMG * 2.0 - 1.0          # world x per pixel
WY = 1.0 - (_ys + 0.5) / IMG * 2.0          # world y per pixel


# --------------------------------------------------------------------------- physics
def oracle_action(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    delta = state[..., 2:4] - state[..., 0:2]
    return np.clip(K_GAIN * delta, -A_MAX, A_MAX)


def relative_goal(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    return state[..., 2:4] - state[..., 0:2]


def in_geo_holdout(state: np.ndarray) -> np.ndarray:
    lo, hi = GEO_HOLDOUT_BOX
    g = np.asarray(state)[..., 2:4]
    return (g[..., 0] >= lo) & (g[..., 0] <= hi) & (g[..., 1] >= lo) & (g[..., 1] <= hi)


def sample_states(rng: np.random.Generator, n: int, *, geo_holdout: bool, min_sep: float = 0.15) -> np.ndarray:
    """Sample physical states. geo_holdout=False excludes the hold-out goal box, True samples only inside it.
    ee and goal are kept at least `min_sep` apart so markers never overlap."""
    out = []
    while len(out) < n:
        m = 4 * (n - len(out)) + 16
        ee = rng.uniform(-WORKSPACE, WORKSPACE, size=(m, 2))
        if geo_holdout:
            lo, hi = GEO_HOLDOUT_BOX
            goal = rng.uniform(lo, hi, size=(m, 2))
        else:
            goal = rng.uniform(-WORKSPACE, WORKSPACE, size=(m, 2))
        st = np.concatenate([ee, goal], axis=1)
        ok = np.linalg.norm(goal - ee, axis=1) >= min_sep
        if not geo_holdout:
            ok &= ~in_geo_holdout(st)
        out.extend(st[ok].tolist())
    return np.asarray(out[:n], dtype=np.float64)


# --------------------------------------------------------------------------- nuisance
def _safe_hue(rng: np.random.Generator) -> float:
    while True:
        h = float(rng.uniform(0, 1))
        if all(min(abs(h - f), 1 - abs(h - f)) > _HUE_MARGIN for f in _FORBIDDEN_HUES):
            return h


@dataclass
class Nuisance:
    """All task-irrelevant rendering factors. Everything is explicit so render() is a pure function."""
    bg_family: int = 0
    seed: int = 0                       # drives discrete/random sub-structure (noise, blob layout)
    hue: float = 0.3                    # continuous: base colour hue      in [0,1)
    hue2: float = 0.8                   # continuous: secondary colour hue in [0,1)
    sat: float = 0.5                    # continuous: saturation           in [0.2,0.9]
    val: float = 0.7                    # continuous: brightness           in [0.35,1.0]
    freq: float = 6.0                   # continuous: texture frequency    in [3,14]
    phase: float = 0.0                  # continuous: texture phase        in [0,2pi)
    angle: float = 0.0                  # continuous: texture angle (rad)  in [0,pi)
    center: tuple[float, float] = (0.0, 0.0)   # continuous: radial centre
    n_distractors: int = 0
    distractor_xy: tuple = ()           # continuous positions, shape (n,2)
    distractor_shape: tuple = ()        # 0 square, 1 triangle, 2 circle
    distractor_hue: tuple = ()
    distractor_size: tuple = ()
    halo: int = 0                       # foreground-adjacent ring around ee (0/1)
    halo_hue: float = 0.4
    halo_radius: float = 0.12

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d["distractor_xy"] = tuple(tuple(float(c) for c in p) for p in d["distractor_xy"])
        for k in ("distractor_shape", "distractor_hue", "distractor_size"):
            d[k] = tuple(d[k])
        d["center"] = tuple(d["center"])
        return Nuisance(**d)

    def continuous_vector(self) -> np.ndarray:
        """Continuous factors usable for finite-difference sensitivity: [hue, hue2, sat, val, freq, phase, angle]."""
        return np.array([self.hue, self.hue2, self.sat, self.val, self.freq, self.phase, self.angle], dtype=np.float64)

    def with_continuous(self, vec: np.ndarray) -> "Nuisance":
        d = self.to_dict()
        d.update(hue=float(vec[0] % 1.0), hue2=float(vec[1] % 1.0), sat=float(np.clip(vec[2], 0.05, 1.0)),
                 val=float(np.clip(vec[3], 0.1, 1.0)), freq=float(max(vec[4], 0.5)), phase=float(vec[5]), angle=float(vec[6]))
        return Nuisance.from_dict(d)


CONTINUOUS_NAMES = ["hue", "hue2", "sat", "val", "freq", "phase", "angle"]


def sample_nuisance(rng: np.random.Generator, family: int, state: np.ndarray, *, max_distractors: int = 4,
                    halo_prob: float = 0.5, n_distractors: int | None = None) -> Nuisance:
    """Sample a nuisance configuration whose distractors/halo do not overlap the ee or goal marker of `state`."""
    ee, goal = np.asarray(state[0:2]), np.asarray(state[2:4])
    if n_distractors is None:
        n_distractors = int(rng.integers(0, max_distractors + 1))
    dxy, dshape, dhue, dsize = [], [], [], []
    tries = 0
    while len(dxy) < n_distractors and tries < 500:
        tries += 1
        p = rng.uniform(-0.95, 0.95, size=2)
        size = float(rng.uniform(0.04, 0.09))
        if np.linalg.norm(p - ee) < EE_RADIUS + size + 0.16:
            continue
        if np.linalg.norm(p - goal) < GOAL_R_OUT + size + 0.12:
            continue
        if any(np.linalg.norm(p - np.asarray(q)) < size + s + 0.03 for q, s in zip(dxy, dsize)):
            continue
        dxy.append((float(p[0]), float(p[1])))
        dshape.append(int(rng.integers(0, 3)))
        dhue.append(_safe_hue(rng))
        dsize.append(size)
    return Nuisance(
        bg_family=int(family),
        seed=int(rng.integers(0, 2**31 - 1)),
        hue=_safe_hue(rng), hue2=_safe_hue(rng),
        sat=float(rng.uniform(0.2, 0.9)), val=float(rng.uniform(0.35, 1.0)),
        freq=float(rng.uniform(3.0, 14.0)), phase=float(rng.uniform(0, 2 * np.pi)), angle=float(rng.uniform(0, np.pi)),
        center=(float(rng.uniform(-0.8, 0.8)), float(rng.uniform(-0.8, 0.8))),
        n_distractors=len(dxy), distractor_xy=tuple(dxy), distractor_shape=tuple(dshape),
        distractor_hue=tuple(dhue), distractor_size=tuple(dsize),
        halo=int(rng.uniform() < halo_prob), halo_hue=_safe_hue(rng), halo_radius=float(rng.uniform(0.115, 0.135)),
    )


# --------------------------------------------------------------------------- rendering helpers
def _hsv(h, s, v) -> np.ndarray:
    return np.array(colorsys.hsv_to_rgb(h % 1.0, float(np.clip(s, 0, 1)), float(np.clip(v, 0, 1))), dtype=np.float32) * 255.0


def _blend(img: np.ndarray, mask: np.ndarray, color: np.ndarray):
    """Alpha-blend `color` into img where mask (float in [0,1]) > 0."""
    m = mask[..., None].astype(np.float32)
    img *= (1.0 - m)
    img += m * color[None, None, :]


def _disk_mask(cx, cy, r, soft=0.5 / IMG * 2):
    d = np.sqrt((WX - cx) ** 2 + (WY - cy) ** 2)
    return np.clip((r - d) / soft + 0.5, 0.0, 1.0)


def _annulus_mask(cx, cy, r_in, r_out):
    return np.clip(_disk_mask(cx, cy, r_out) - _disk_mask(cx, cy, r_in), 0.0, 1.0)


def _segment_mask(p, q, width):
    p, q = np.asarray(p), np.asarray(q)
    d = q - p
    L2 = float(d @ d) + 1e-12
    t = np.clip(((WX - p[0]) * d[0] + (WY - p[1]) * d[1]) / L2, 0.0, 1.0)
    px, py = p[0] + t * d[0], p[1] + t * d[1]
    dist = np.sqrt((WX - px) ** 2 + (WY - py) ** 2)
    soft = 0.5 / IMG * 2
    return np.clip((width / 2 - dist) / soft + 0.5, 0.0, 1.0)


def _square_mask(cx, cy, half):
    return ((np.abs(WX - cx) <= half) & (np.abs(WY - cy) <= half)).astype(np.float32)


def _triangle_mask(cx, cy, half):
    # upward-pointing triangle inscribed in the square of half-size `half`
    y_rel = (WY - cy + half) / (2 * half)           # 0 at bottom, 1 at top
    x_rel = np.abs(WX - cx) / half                  # 0 at centre line
    return ((y_rel >= 0) & (y_rel <= 1) & (x_rel <= (1 - y_rel))).astype(np.float32)


def _two_link_ik(target: np.ndarray):
    """Elbow-up planar 2-link IK from ARM_BASE to target (clamped to reachable)."""
    d = target - ARM_BASE
    r = float(np.linalg.norm(d))
    r = float(np.clip(r, abs(ARM_L1 - ARM_L2) + 1e-3, ARM_L1 + ARM_L2 - 1e-3))
    cos_e = (r ** 2 - ARM_L1 ** 2 - ARM_L2 ** 2) / (2 * ARM_L1 * ARM_L2)
    elbow = np.arccos(np.clip(cos_e, -1, 1))
    base_ang = np.arctan2(d[1], d[0]) - np.arctan2(ARM_L2 * np.sin(elbow), ARM_L1 + ARM_L2 * np.cos(elbow))
    # choose the elbow configuration pointing away from the workspace center (deterministic sign flip for x<0)
    if target[0] < 0:
        elbow = -elbow
        base_ang = np.arctan2(d[1], d[0]) - np.arctan2(ARM_L2 * np.sin(elbow), ARM_L1 + ARM_L2 * np.cos(elbow))
    joint = ARM_BASE + ARM_L1 * np.array([np.cos(base_ang), np.sin(base_ang)])
    return joint


# --------------------------------------------------------------------------- background families
def _background(eta: Nuisance) -> np.ndarray:
    c1, c2 = _hsv(eta.hue, eta.sat, eta.val), _hsv(eta.hue2, eta.sat, eta.val * 0.75)
    f = eta.bg_family
    if f == 0:   # solid
        return np.broadcast_to(c1, (IMG, IMG, 3)).astype(np.float32).copy()
    if f == 1:   # horizontal stripes
        m = (np.sin(WY * eta.freq * np.pi + eta.phase) > 0).astype(np.float32)
    elif f == 2:  # checkerboard
        m = (((np.floor(WX * eta.freq / 2 + eta.phase) + np.floor(WY * eta.freq / 2)) % 2) == 0).astype(np.float32)
    elif f == 3:  # radial gradient
        d = np.sqrt((WX - eta.center[0]) ** 2 + (WY - eta.center[1]) ** 2)
        m = np.clip(d / 2.0, 0, 1).astype(np.float32)
    elif f == 4:  # smooth low-frequency noise (sum of sinusoids seeded)
        rng = np.random.default_rng(eta.seed)
        m = np.zeros((IMG, IMG), np.float32)
        for _ in range(4):
            kx, ky = rng.uniform(-eta.freq / 4, eta.freq / 4, size=2)
            m += np.sin(kx * WX * np.pi + ky * WY * np.pi + rng.uniform(0, 2 * np.pi))
        m = (m - m.min()) / (m.max() - m.min() + 1e-6)
    elif f == 5:  # diagonal stripes
        u = WX * np.cos(eta.angle) + WY * np.sin(eta.angle)
        m = (np.sin(u * eta.freq * np.pi + eta.phase) > 0).astype(np.float32)
    elif f == 6:  # coloured blobs on solid
        rng = np.random.default_rng(eta.seed)
        img = np.broadcast_to(c1, (IMG, IMG, 3)).astype(np.float32).copy()
        for _ in range(int(rng.integers(8, 20))):
            cx, cy = rng.uniform(-1.1, 1.1, size=2)
            r = rng.uniform(0.08, 0.3)
            col = _hsv(_safe_hue(rng), eta.sat, eta.val)
            _blend(img, _disk_mask(cx, cy, r), col)
        return img
    elif f == 7:  # high-frequency white noise, tinted
        rng = np.random.default_rng(eta.seed)
        n = rng.uniform(0, 1, size=(IMG, IMG)).astype(np.float32)
        return (c1[None, None] * (0.4 + 0.6 * n[..., None])).astype(np.float32)
    elif f == 8:  # multi-octave sinusoidal texture ("image-like")
        rng = np.random.default_rng(eta.seed)
        m = np.zeros((IMG, IMG), np.float32)
        amp = 1.0
        for o in range(5):
            kx, ky = rng.uniform(-1, 1, size=2) * eta.freq * (2 ** o) / 4
            m += amp * np.sin(kx * WX * np.pi + ky * WY * np.pi + rng.uniform(0, 2 * np.pi))
            amp *= 0.5
        m = (m - m.min()) / (m.max() - m.min() + 1e-6)
    else:
        raise ValueError(f"unknown bg family {f}")
    return (c1[None, None] * (1 - m[..., None]) + c2[None, None] * m[..., None]).astype(np.float32)


# --------------------------------------------------------------------------- full render
def render(state: np.ndarray, eta: Nuisance, *, draw_arm: bool = True) -> np.ndarray:
    """Render a 224x224x3 uint8 image for physical state `state` under nuisance `eta`."""
    state = np.asarray(state, dtype=np.float64)
    ee, goal = state[0:2], state[2:4]
    img = _background(eta)

    # distractors (never overlap markers by construction of sample_nuisance)
    for (cx, cy), shp, hue, size in zip(eta.distractor_xy, eta.distractor_shape, eta.distractor_hue, eta.distractor_size):
        col = _hsv(hue, 0.75, 0.85)
        if shp == 0:
            m = _square_mask(cx, cy, size)
        elif shp == 1:
            m = _triangle_mask(cx, cy, size)
        else:
            m = _disk_mask(cx, cy, size)
        _blend(img, m, col)

    # foreground-adjacent nuisance: halo ring around the ee marker (outside the marker)
    if eta.halo:
        _blend(img, _annulus_mask(ee[0], ee[1], eta.halo_radius, eta.halo_radius + 0.016), _hsv(eta.halo_hue, 0.8, 0.9))

    # arm (deterministic function of ee)
    if draw_arm:
        joint = _two_link_ik(ee)
        _blend(img, _segment_mask(ARM_BASE, joint, ARM_WIDTH), ARM_COLOR)
        _blend(img, _segment_mask(joint, ee, ARM_WIDTH), ARM_COLOR)
        _blend(img, _disk_mask(joint[0], joint[1], ARM_WIDTH * 0.9), JOINT_COLOR)
        _blend(img, _disk_mask(ARM_BASE[0], ARM_BASE[1], ARM_WIDTH * 1.4), JOINT_COLOR)

    # goal marker: ring + dot
    _blend(img, _annulus_mask(goal[0], goal[1], GOAL_R_IN, GOAL_R_OUT), GOAL_COLOR)
    _blend(img, _disk_mask(goal[0], goal[1], GOAL_DOT_R), GOAL_COLOR)
    # ee marker: filled disk
    _blend(img, _disk_mask(ee[0], ee[1], EE_RADIUS), EE_COLOR)

    return np.clip(img + 0.5, 0, 255).astype(np.uint8)
