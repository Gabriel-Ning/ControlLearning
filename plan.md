# Implementation Plan: Control-Oriented Representation Learning through Cross-Modal Transport

**Project codename:** CORT-MVP  
**Reference:** https://github.com/ucd-dare/VITA  
**Research type:** Hypothesis-driven, controlled empirical study  
**Priority:** Mechanism verification before policy performance optimization

## 1. Research Objective

Investigate whether transporting frozen pretrained visual representations into action representations using Flow Matching induces more control-oriented, nuisance-robust representations than direct regression.

We are not proposing a new method at this stage. The purpose of this MVP is to systematically investigate the representation-learning mechanism of Vision-to-Action Transport, identify its limitations, and determine whether a new method is justified.

### Hypotheses

**H1 — Control relevance**

Cross-modal Flow Matching produces intermediate or terminal representations from which control-relevant variables are more easily recovered using low-capacity probes.

**H2 — Nuisance separation**

Transport reduces the relative sensitivity of control-relevant representations to background, lighting, and other task-irrelevant visual factors.

**H3 — Transport-specific improvement**

Any observed benefit is attributable to the transport objective rather than increased network capacity, extra losses, stronger visual backbones, or the action decoder.

**H4 — Generalization**

If the representation effect exists, it may improve out-of-distribution action prediction under nuisance shifts.

These are hypotheses, not expected results. Negative results must be preserved and reported.

## 2. Phase P0 — Repository Audit and Reproduction

### Tasks

Clone and inspect:

https://github.com/ucd-dare/VITA

Record the exact Git commit SHA and software environment.

Identify and document:

- Visual observer and observation encoder.
- Action encoder and decoder.
- Flow velocity network.
- Flow Matcher and its source-target coupling.
- All active training losses.
- ODE integration and trajectory tracing.
- Dataset interface and preprocessing.
- Checkpoint loading and optimizer configuration.

Relevant files already identified:

- `flare/train.py`
- `flare/policies/vita/vita_policy.py`
- `flare/policies/vita/simple_flow_net.py`
- `flare/flow/flow_matchers.py`
- `flare/networks/vita/action_ae.py`
- `flare/configs/policy/vita.yaml`

Inspect the configuration mismatch between `flow_net.name: simple` and the names accepted by `VitaPolicy`.

Verify whether the exact OT-CFM implementation changes paired vision-action correspondences within a minibatch. The primary MVP should use data-aligned source-target pairs without OT reassignment.

Run a short baseline smoke test using an officially supported configuration and a small available dataset.

Record actual loss components, shapes, gradient paths, and ODE sampling behavior.

**Deliverables**

- `reports/00_repo_audit.md`
- `reports/00_environment.md`
- A working original-VITA smoke-test command.
- Commit SHA and dependency versions.

**Acceptance criteria**

A valid batch passes through the original pipeline, FM loss is finite, and ODE sampling returns actions with expected dimensions.

Do not start the main experiments until this phase passes.

## 3. Phase P1 — Synthetic Paired Benchmark

Build a simple rendered 2D reaching dataset.

Avoid a complex robotics simulator in the first iteration.

### 3.1 Physical state

Define:

`x = [ee_x, ee_y, goal_x, goal_y]`

Generate an oracle action:

`a = clip(k * (goal_xy - ee_xy), -amax, amax)`

Prefer sampling configurations where clipping rarely occurs for the primary dataset.

Store:

- physical state x
- relative goal displacement
- ground-truth oracle action
- image
- nuisance parameters
- physical state ID
- rendering seed

### 3.2 Rendered image

Render at 224 × 224 pixels with a fixed camera and deterministic foreground geometry.

Include:

- robot end-effector marker
- target marker
- optional simple robot arm
- background texture
- configurable distractors

The background must not occlude or modify the physical foreground in the initial experiment.

### 3.3 Controlled nuisance factors

Support independently configurable:

- background color
- procedural background texture
- background images
- distractor locations
- irrelevant foreground-adjacent patterns

Keep nuisance generation deterministic using stored seeds.

Generate counterfactual image pairs:

`I_1 = render(x, eta_1)`

`I_2 = render(x, eta_2)`

Both images must have identical physical state and oracle action.

### 3.4 Dataset splits

Initial dataset:

| Split | Physical states | Notes |
|---|---:|---|
| Train | 10,000 | Limited background families |
| Validation | 2,000 | Disjoint states, seen backgrounds |
| ID Test | 2,000 | Disjoint states, seen backgrounds |
| OOD Background | 2,000 | Same state distribution, unseen background families |
| Counterfactual Test | 2,000 × 4 | Identical states with four nuisance variants |
| Geometry OOD | 2,000 | Held-out spatial configurations |

Ensure there is no exact physical-state leakage between train, validation, and independent test splits.

For the counterfactual test, repeated physical states are intentionally shared across nuisance variants within the test split.

Optionally create a spurious-correlation split in which background characteristics correlate with goal regions during training but are decorrelated at evaluation.

Do not mix this diagnostic into the primary dataset without labeling it separately.

### Validation

Run automatic checks proving that different nuisance renderings with identical state IDs have identical oracle actions.

**Deliverables**

- `data/generate_reaching.py`
- `data/reaching_dataset.py`
- `tests/test_counterfactual_pairs.py`
- `figures/dataset_examples.png`
- Dataset manifest and generation configuration.

## 4. Phase P2 — Frozen Visual Encoder

Use a pretrained DINOv3 model as the preferred backbone.

If DINOv3 weights are inaccessible or incompatible, permit DINOv2 as a documented fallback. Do not silently switch models.

Freeze the entire visual backbone.

Extract image embeddings in batches and cache them on local storage.

Baseline feature:

`h = DINO(image)`

Start with global pooled embeddings. Record the feature extraction strategy precisely, including CLS versus patch pooling if applicable.

Use the identical cached features for every competing model.

Store:

- pretrained model identifier and revision
- preprocessing transform
- embedding shape
- feature-cache checksum
- source image IDs

Add a lightweight trainable projection:

`z0 = Projector(h)`

Use latent dimension 64 initially.

The projector architecture must be identical for matched model comparisons.

Keep a separate baseline that probes frozen DINO features directly, without the trainable projector.

**Acceptance criteria**

The feature extractor runs in evaluation mode, all outputs are finite, and repeated extraction of the same image is deterministic within numerical tolerance.

## 5. Phase P3 — Transport and Regression Models

Implement a small research module without breaking original VITA behavior.

### 5.1 Shared action representation

Start with 2D oracle actions.

Standardize action values using training-set statistics only.

Construct a fixed embedding:

`za = W_action @ a`

where `W_action` is a seeded matrix with orthonormal columns and output dimension 64.

For this initial controlled study, action prediction is recovered using the fixed transpose mapping.

Keep W fixed across runs.

This removes trainable Action AE as a confounding variable.

### 5.2 Models

**M0 — Frozen DINO linear probe**

Fit ridge regression directly from frozen DINO embeddings to oracle actions.

**M1 — Direct regression**

Train the shared projector and a nonlinear MLP to predict action embedding or action.

Use the same fixed action readout and comparable reconstruction loss.

**M2 — Residual ODE, endpoint supervision**

Train a time-conditioned velocity network using the numerical ODE endpoint and action loss, without FM velocity supervision.

Use the same velocity architecture and integration method as M3.

**M3 — Vision-to-Action Flow Matching**

Initialize the flow at the projected visual representation.

`z0 = projector(h)`

`za = fixed_action_embedding(a)`

Sample:

`t ~ Uniform(0, 1)`

`zt = (1 - t) * z0 + t * za`

`target_velocity = za - z0`

Train the velocity network with:

`L_fm = MSE(v_theta(zt, t), target_velocity)`

Use the ODE integration:

`dz/dt = v_theta(z, t)`

Add an explicitly weighted endpoint action loss:

`L_action = MSE(W_action.T @ z1, a_normalized)`

Total:

`L = lambda_fm * L_fm + lambda_action * L_action`

Record each loss separately.

**M4 — FM + explicit counterfactual invariance**

Use the architecture and losses of M3.

Add a paired-image representation consistency loss, computed at specified transport times.

Keep this as an intervention/positive-control experiment, not part of the original FM hypothesis.

### 5.3 Required controls

Match projector architecture, latent dimensions, main network parameter count, training samples, and optimizer budgets wherever feasible.

Keep M2 and M3 especially closely matched: same time-conditioned velocity network, same integration method, same number of numerical steps, same endpoint loss.

Use fixed initial random seeds and report three independent seeds for the final comparison.

Run a small hyperparameter validation sweep fairly for every model.

Do not optimize only M3.

Test a few ODE integration step counts, such as 1, 4, 8, and 16, and report performance separately.

Record whether the FM-only velocity objective and endpoint objective produce conflicting or complementary gradients.

### Critical technical checks

- Gradients must propagate through the ODE endpoint into the projector when intended.
- Feature caching must not detach the trainable projector.
- All models must use exactly the same action normalization.
- FM training interpolation must preserve real source-target pairing.
- ODE trajectories must be finite at all logged timestamps.
- No test-set statistics may be used for normalization or hyperparameter selection.

**Deliverables**

- `cort/models/regression.py`
- `cort/models/transport.py`
- `cort/models/embedding.py`
- `cort/train.py`
- `configs/experiment/*.yaml`
- Model unit tests and saved checkpoints.

## 6. Phase P4 — Representation Analysis

This is the most important phase.

Do not restrict evaluation to action prediction accuracy.

Extract:

`z(t)` for `t = 0.0, 0.1, ..., 1.0`

Use numerical ODE trajectories for analysis.

Do not substitute analytical straight-line interpolation as if it were the learned trajectory.

Store representations for identical evaluation samples across all models.

For regression and residual-ODE baselines, expose reasonable intermediate activations or ODE states.

Do not claim a conventional MLP layer index is mathematically equivalent to FM time.

### 6.1 Control linear decodability

For each representation stage, freeze the model and fit a new regularized linear probe.

Predict:

- relative goal displacement
- oracle action
- optionally absolute state coordinates

Fit probes using only training samples and evaluate on held-out states.

Report test R², normalized MSE, and probe regularization.

Also measure nonlinear probe performance with a small fixed-capacity MLP as a supplementary test.

Control variables should be distinguished from arbitrary physical-state information.

### 6.2 Nuisance sensitivity

Using counterfactual pairs sharing the same state x:

`delta_eta(t) = z_t(x, eta_1) - z_t(x, eta_2)`

Measure within-state representation variation.

Measure between-state variation using controlled pairs of different physical states.

Report a normalized ratio:

`R_nuisance(t) = within_state_variance / (between_state_variance + epsilon)`

Additionally report the numerator and denominator separately to detect latent collapse.

Normalize or whiten representations consistently when comparing geometries across different models.

### 6.3 Task-relevant response

Perturb physical state while keeping nuisance fixed.

Compare representation response to state changes versus nuisance changes.

Where possible, estimate finite-difference sensitivity matrices:

`J_x(t) = dz_t / dx`

`J_eta(t) = dz_t / d_eta`

Use finite differences only for nuisance factors with meaningful continuous parametrization.

For discrete backgrounds, use paired representation differences rather than pretending a derivative exists.

Report sensitivity norms and effective ranks.

### 6.4 Linear subspace diagnostics

Perform SVD or PCA-based analysis of paired representation differences.

Measure whether nuisance variation becomes concentrated in directions that are less useful for decoding control variables.

Define the control-relevant directions using linear probes trained on separate data.

Avoid defining control relevance by the model's own action decoder alone.

Report principal-angle overlap between control-predictive and nuisance-variation subspaces.

Treat this as a descriptive statistic, not proof of globally identifiable disentanglement.

### 6.5 Action robustness

Evaluate ID and OOD action prediction.

Metrics:

- action MSE
- action angular error, excluding near-zero target magnitudes
- counterfactual action consistency
- ID-to-OOD degradation
- uncertainty across seeds

### 6.6 Visualization

Required figures:

1. Dataset counterfactual examples.
2. ID/OOD action-error comparison.
3. Control probe R² versus transport time.
4. Nuisance sensitivity ratio versus transport time.
5. Within-state and between-state variation separately.
6. Representation PCA/UMAP plots colored by physical state and nuisance.
7. Effective-rank and singular-value analysis.
8. Example learned ODE trajectories for matched counterfactual pairs.

PCA/UMAP plots must be treated as qualitative evidence only.

### Important theoretical caveat

A deterministic invertible transport cannot increase information about the control state beyond what exists in its source representation.

The question is whether it improves accessibility, geometry, low-capacity decodability, nuisance robustness, or downstream sample efficiency.

Do not interpret changes in probe accuracy as a direct measurement of mutual information.

## 7. Phase P5 — Main Ablations

Run these after M0–M4 training and evaluation are stable.

**Ablation A: Flow versus regression**

Compare M1, M2, M3 under matched architecture and training budgets.

**Ablation B: FM loss versus endpoint loss**

For the same flow architecture, compare endpoint only, FM only, and FM + endpoint.

**Ablation C: Trainable projector**

Compare frozen random projector, trainable linear projector, and trainable two-layer projector.

**Ablation D: Visual features**

Compare pretrained DINOv3 against frozen DINOv2 if both are available.

**Ablation E: Dataset size**

Evaluate 1%, 10%, and 100% of training states, with seeds fixed.

**Ablation F: Source-target pairing**

Compare genuine paired FM with the official OT-CFM coupling mechanism if the audit confirms possible rematching.

The purpose is to understand whether physically meaningful paired transport matters.

Do not run all ablations blindly before the principal hypotheses have been evaluated.

## 8. Phase P6 — Optional Realistic Benchmark

Only begin after the synthetic experiment produces a meaningful and reproducible finding.

Recommended first candidate: PushT, using the official VITA pipeline and supported LeRobot dataset conversion.

Keep the original VITA model as an additional baseline.

Introduce a frozen visual foundation encoder behind a configuration switch.

Evaluate:

- original VITA observer
- frozen DINO + VITA transport
- frozen DINO + matched regression
- frozen DINO + matched endpoint ODE

Construct controlled visual disturbances in simulation without changing physical state, rewards, or expert actions.

First evaluate offline action prediction and representation metrics.

Run closed-loop simulation only after offline evaluation is stable.

Use multiple evaluation seeds and report success-rate confidence intervals.

Do not treat successful synthetic transfer as guaranteed evidence of manipulation-level generalization.

## 9. Suggested Project Structure

```text
VITA/
├── flare/                      # Original implementation
├── cort_mvp/
│   ├── configs/
│   ├── data/
│   │   ├── generate_reaching.py
│   │   └── reaching_dataset.py
│   ├── features/
│   │   ├── frozen_dino.py
│   │   └── cache_features.py
│   ├── models/
│   │   ├── embedding.py
│   │   ├── regression.py
│   │   └── transport.py
│   ├── analysis/
│   │   ├── probes.py
│   │   ├── invariance.py
│   │   ├── sensitivity.py
│   │   └── subspaces.py
│   ├── tests/
│   ├── train.py
│   ├── evaluate.py
│   └── export_report.py
├── results/
│   ├── metrics/
│   ├── figures/
│   └── checkpoints/
└── reports/
    ├── 00_repo_audit.md
    ├── 01_dataset_validation.md
    ├── 02_baseline_results.md
    └── 03_mechanism_analysis.md
```

Use standard PyTorch, Hydra/YAML where compatible with the repo, NumPy, scikit-learn, and Matplotlib.

No need to introduce a new distributed training framework.

## 10. Reproducibility and Reporting

Every run must save:

- Git commit SHA
- experiment config
- seed
- model parameter count
- training dataset manifest
- pretrained backbone revision
- software environment
- training loss curves
- validation metrics
- independent test metrics
- generated figure paths
- model checkpoint path

Export aggregated metrics in machine-readable CSV or JSON format.

Use the same evaluation set and metric implementation across all models.

Report mean and standard deviation over seeds, and confidence intervals for key comparisons where possible.

Do not select checkpoints on OOD test data.

Record failures, NaNs, unstable training, and unsuccessful model configurations rather than quietly excluding them.

## 11. Acceptance Criteria

### Minimum technical completion

- Original VITA smoke test passes or a precise upstream blocker is documented.
- Synthetic paired dataset passes invariance-label tests.
- Frozen DINO feature caching works.
- M0, M1, M2, and M3 train successfully.
- Intermediate Flow states can be exported.
- Action and representation metrics run end-to-end.
- At least three seeds for the primary model comparison.
- Automated report generation completes.

### Scientific completion

The final report must directly answer:

**Q1:** Does cross-modal FM improve action prediction or control-state linear decodability over matched regression?

**Q2:** Does FM reduce relative nuisance sensitivity under matched physical states?

**Q3:** At which transport times, if any, do meaningful representation changes appear?

**Q4:** Can these changes be explained by endpoint supervision or architecture alone?

**Q5:** Does the observed representation effect generalize to unseen background distributions?

### Possible outcomes

**Positive:** FM demonstrates reproducible improvements in representation geometry, control probing, or OOD prediction.

**Mixed:** FM improves action prediction but not representation separation, or vice versa.

**Negative:** Properly matched regression or endpoint ODE achieves equivalent or better results.

All three outcomes are valid. The report must explain the evidence without claiming a new mechanism when none is demonstrated.

## 12. Execution Instructions for Coding Agent

Execute in stages.

First complete the repository audit and dataset validation. Then train the controlled baselines. Finally perform representation analysis.

Prefer small, testable commits.

Reuse original VITA utilities where reasonable, but avoid invasive modifications to the original policy implementation.

Do not introduce new theoretical losses, new decoder architectures, RL, world models, or complex robotic simulators in the initial MVP.

If an implementation detail is unclear, choose the simpler scientifically controlled version, document the decision, and continue.

Do not fabricate results or assume that the original paper's reported gains will reproduce under the new experimental setting.

**Final deliverable:** A reproducible research prototype and an empirical report that determine whether cross-modal Flow Matching acts as a distinctive mechanism for learning control-oriented visual representations.
