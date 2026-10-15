# CORT-MVP — is cross-modal Flow Matching a distinctive mechanism for control-oriented visual representations?

Controlled empirical study around [VITA](https://github.com/ucd-dare/VITA) (vision-to-action flow matching). Frozen DINO
features → parameter-matched heads (ridge M0, direct regression M1, endpoint-only ODE M2, paired FM M3, FM + counterfactual
invariance M4 as positive control) on a synthetic 2-D reaching benchmark with counterfactual nuisance pairs.

Reports (read in order): `reports/00_repo_audit.md`, `00_environment.md`, `01_dataset_validation.md`,
`02_baseline_results.md`, `03_mechanism_analysis.md` (answers Q1–Q5). Tables: `reports/tables/`; figures: `results/figures/`;
machine-readable metrics: `results/metrics/`; per-run artefacts (config, run.json, curves, eval.json): `results/runs/<run>/`.

## Reproduction (from the workspace root, `.venv` as in `reports/00_environment.md`)

```bash
source .venv/bin/activate
unset ALL_PROXY all_proxy                      # only needed if a socks proxy is set (httpx rejects it)

# P1 data (≈10 min, 8 workers)
python -m cort_mvp.data.generate_reaching --out data_cache/reaching_v1 --seed 2026 --workers 8 --extra_train_cf
CORT_DATA=data_cache/reaching_v1 python -m pytest cort_mvp/tests -q

# P2 frozen features (DINOv3 ViT-S/16 primary, DINOv2 ViT-S/14 for ablation D)
python -m cort_mvp.features.cache_features --backbone dinov3_vits16
python -m cort_mvp.features.cache_features --backbone dinov2_vits14

# P3 training: LR sweep (seed 0, val-selected) -> 3-seed main runs -> ablations -> ODE-step sweep
python -m cort_mvp.sweep --suite lr_sweep -j 4
python -m cort_mvp.sweep --suite main -j 4
python -m cort_mvp.sweep --suite ablations -j 4
python -m cort_mvp.sweep --suite ode_steps -j 4
python -m cort_mvp.sweep --suite lowdata_B -j 4

# P4 evaluation + representation export, analyses, report export
python -m cort_mvp.evaluate --m0 --backbone dinov3_vits16 --feature_key cls
python -m cort_mvp.evaluate --m0 --backbone dinov3_vits16 --feature_key patch_mean
python -m cort_mvp.evaluate --m0 --backbone dinov2_vits14 --feature_key cls
python -m cort_mvp.evaluate
python -m cort_mvp.analysis.invariance && python -m cort_mvp.analysis.subspaces
python -m cort_mvp.analysis.sensitivity --k 200 --frac 0.1 && python -m cort_mvp.analysis.sensitivity --k 200
python -m cort_mvp.analysis.probes
python -m cort_mvp.export_report
```

Single run: `python -m cort_mvp.train --exp m3 optim.lr=3e-3 seed=0` (OmegaConf dot-list overrides over
`cort_mvp/configs/base.yaml` + `configs/experiment/<exp>.yaml`). The upstream VITA clone lives in `VITA/` (unmodified,
git-ignored; commit recorded in `reports/00_repo_audit.md`).
