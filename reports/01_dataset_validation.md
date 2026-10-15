# P1 — Synthetic Paired Reaching Benchmark: Validation

Generator: `cort_mvp/data/generate_reaching.py` (renderer `cort_mvp/data/reaching_render.py`),
dataset class `cort_mvp/data/reaching_dataset.py`, tests `cort_mvp/tests/test_counterfactual_pairs.py`.
Dataset root: `data_cache/reaching_v1` (seed 2026; manifest `data_cache/reaching_v1/manifest.json`).
Example figure: `results/figures/dataset_examples.png`.

## Physical model

- State `x = [ee_x, ee_y, goal_x, goal_y]`, sampled uniformly in `[-0.8, 0.8]^2 × [-0.8, 0.8]^2`
  with `|goal − ee| ≥ 0.15` (markers never overlap).
- Oracle action `a = clip(1.0 · (goal − ee), −1.5, 1.5)`; clipping occurs in 0.6–1.4 % of samples per split
  (recorded per split in the manifest as `clipped_fraction`; the `clipped` flag is stored per sample).
- Geometry hold-out: goals inside `[0.3, 0.8]^2` are excluded from every split except `geo_ood`.

## Rendering

224×224 RGB, fixed orthographic camera (world `[-1,1]^2` → pixels), pure NumPy, deterministic given `(x, eta)`.
Layer order: background family → distractors → halo → 2-link arm (deterministic IK from a fixed base below the
image) → goal marker (red ring + dot) → ee marker (blue disk). Distractors and the halo are sampled with exclusion radii
around both markers, so nuisance never occludes or alters the foreground (tested pixel-exactly).

Nuisance `eta` is an explicit dataclass with continuous fields `[hue, hue2, sat, val, freq, phase, angle]`
(stored as `nuisance_cont`), a seed for discrete sub-structure, distractor count/positions/shapes/hues/sizes, and the
foreground-adjacent halo ring (presence, hue, radius). Full nuisance records are saved in `nuisance.json` per split so any
image can be re-rendered exactly, and continuous factors can be finite-differenced.

Background families: seen = {0 solid, 1 hstripes, 2 checker, 3 radial, 4 smooth_noise};
unseen (OOD) = {5 dstripes, 6 blobs, 7 white_noise, 8 multi_octave}.

## Splits

| Split | N | States | Backgrounds | Purpose |
|---|---:|---|---|---|
| train | 10 000 | disjoint | seen | training |
| val | 2 000 | disjoint | seen | model selection only |
| id_test | 2 000 | disjoint | seen | ID test |
| ood_bg | 2 000 | disjoint | unseen | OOD-background test |
| geo_ood | 2 000 | goal in hold-out box | seen | geometry OOD |
| cf_test_v0..v3 | 4 × 2 000 | **identical** across variants | v0,v1 seen; v2,v3 unseen | counterfactual nuisance pairs |
| sp_test_v0 | 2 000 | `x + dx`, ‖dx‖ = 0.2 | nuisance **identical to cf_test_v0** | small state perturbation control |
| sp_test_v1 | 2 000 | independent `x'` | nuisance identical to cf_test_v0 | between-state control |

A separate, clearly labelled diagnostic set `data_cache/reaching_v1_SPURIOUS_DIAGNOSTIC` maps background family to
goal quadrant in train/val and decorrelates it in the test splits. It is **not** mixed into the primary dataset.

## Automatic checks (all pass; `CORT_DATA=data_cache/reaching_v1 pytest cort_mvp/tests/test_counterfactual_pairs.py`)

1. Oracle action formula and clipping bounds.
2. Render determinism and nuisance JSON round-trip (bit-identical images).
3. Foreground pixels strictly inside the ee disk and the goal ring are bit-identical across nuisance families.
4. The four `cf_test` variants share `state_id`, `states`, `actions`, `rel_goal` exactly while nuisance seeds differ
   and images differ; stored actions equal `oracle_action(states)` to 1e-5.
5. cf variant family assignment (v0/v1 seen, v2/v3 unseen).
6. `sp_test` controls share nuisance seed/family with `cf_test_v0`, differ in state, and `‖dx‖ ≤ 0.2` for v0.
7. No physical-state leakage (rounded to 1e-5) between train / val / id_test / ood_bg / geo_ood / cf_test.
8. Family membership per split and geometry hold-out box membership.
9. Dataset item shapes/ranges and train clipping rate < 2 %.

Generation time: 38.6 s for 30 000 images (12 workers); 4.3 GB per dataset (uint8 `.npy` memmaps).
