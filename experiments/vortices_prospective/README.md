# Prospective FIDE vortices experiment

This experiment tests whether FIDE can choose localized sensors *before* target
intermediate measurements are revealed.  It compares a geometry minimizing a
fixed aggregate scientific risk (`Law`), a geometry minimizing particle Tangent
action (`Tangent`), and a geometry minimizing physical Full action (`Full`)
inside a shared 2% relative risk allowance.

The physical system, sensor family, observation model, moment smoother,
endpoint-only flow reference, information projection, and Full-action geometry
follow `experiments/vortices_percentage`.  No skyrmion protocol is used and no
existing experiment is modified.

## Trust and data boundary

At full-law level, only independent target endpoint ensembles `P0` and `P1` are
trusted.  They train one box-constrained flow-matching reference, which is then
frozen for every candidate.

At intermediate times, selection receives only:

- a gridded mean and second-moment table for every candidate Gaussian response;
- predicted means of five predeclared global scientific QoIs;
- the finite sampling and detector-noise model with frozen common randomness.

The response table is evaluated at continuous sensor centers by differentiable
bilinear interpolation.  Its preprocessing uses the exact separability of the
Gaussian observable, so production construction is batched without a
particle-by-spatial-grid tensor.

The global QoIs are normalized horizontal and vertical centroids, their raw
second moments, and their cross moment.  They are distinct from all optimized
localized sensor channels.  Scientific risk is their time-integrated, fixed-scale
squared error under the information-projected reference law.  Hidden-law MMD is
never a selection objective.

The important distinction is:

```text
simulation state available internally during aggregate preprocessing
!=
intermediate law trusted by FIDE
```

The saved prospective artifact contains aggregates only.
`TargetProspectiveData` has no hidden-state path or loader.  Selection writes a
hash-bearing frozen manifest before validation is allowed to generate or load the
disjoint microscopic validation bank.  Validation reads the frozen geometries and
cannot rerun selection or mutate the manifest.

## Pipeline

For each candidate, aggregate predicted acquisitions are converted to finite
pseudo-observations with common random numbers, reconstructed with the established
endpoint-anchored cubic smoother, and information-projected relative to the same
frozen reference.  Law minimizes aggregate-QoI risk. Tangent minimizes the
established particle-space Tangent action. Full minimizes authoritative
physical-density weighted-Poisson action. Both action designs satisfy

```text
R(eta) <= (1 + epsilon) R(Law),  epsilon = 0.02.
```

Every risk-feasible candidate receives a reduced-trial/time/grid physical Full
proxy score; only a fixed shortlist receives authoritative Full rescoring at the
declared production grid, time grid, and selection-bank size. Law, Tangent, and
Full are all frozen before validation and receive reported authoritative Full
evaluations.

After freezing, validation uses independent simulator realizations, actual finite
particle subsampling, and detector noise.  It reports predicted and realized risk,
Full action, their gaps, uncertainty, projection/ESS/covariance diagnostics,
physical Poisson residuals, component compatibility, Full moment-rate residuals,
and valid-trial fractions.

## Prospective v4 robust-Full replicate

`prospective_v4_robust_full` is a new replicate and does not replace the legacy
Law/Tangent/Full production run above. Tangent is disabled for v4 candidate
generation. Full instead uses 32 independent Adam trajectories whose gradients
pass through the differentiable aggregate interpolation, fixed finite-sampling
and detector-noise CRNs, production anchored-cubic reconstruction, implicit
information projection and multiplier derivative, Full forcing, rasterization,
and implicit-CG weighted Poisson value. Leading candidates receive denser
Full-objective Adam/L-BFGS refinement before authoritative physical-direct
certification.

The production v4 result is frozen in
`outputs/prospective_v4_robust_full/results/frozen_manifest.json`. Its fresh
hidden state and observation banks were created only after that receipt and are
bound to its SHA-256. `certify_v4_covariance.py` is a post-freeze read-only audit;
it reports per-trial raw and ridge-regularized covariance condition numbers and
cannot alter geometries or ranking.

Run the v4 smoke pipeline:

```bash
.venv/bin/python experiments/vortices_prospective/run_v4_smoke.py
```

Run or reproduce the preregistered v4 stages explicitly:

```bash
.venv/bin/python experiments/vortices_prospective/run_v4.py \
  --config experiments/vortices_prospective/configs/production_v4.json \
  --output-dir experiments/vortices_prospective/outputs/prospective_v4_robust_full \
  --stage select

.venv/bin/python experiments/vortices_prospective/run_v4.py \
  --config experiments/vortices_prospective/configs/production_v4.json \
  --output-dir experiments/vortices_prospective/outputs/prospective_v4_robust_full \
  --stage validate

.venv/bin/python experiments/vortices_prospective/certify_v4_covariance.py \
  --config experiments/vortices_prospective/configs/production_v4.json \
  --output-dir experiments/vortices_prospective/outputs/prospective_v4_robust_full
```

Validation refuses to run without a compatible frozen manifest. A compatible
second `select` invocation reuses that manifest and cannot access validation data;
an incompatible invocation fails closed and requires a new run identity.

## Commands

Run the complete development pipeline from the repository root:

```bash
.venv/bin/python experiments/vortices_prospective/run_smoke.py
```

Run the production configuration:

```bash
.venv/bin/python experiments/vortices_prospective/run_experiment.py \
  --config experiments/vortices_prospective/configs/production.json \
  --output-dir experiments/vortices_prospective/outputs/production
```

Stages may also be invoked independently, in this order:

```bash
.venv/bin/python experiments/vortices_prospective/build_prospective_data.py --config CONFIG --output-dir OUTPUT
.venv/bin/python experiments/vortices_prospective/train_reference.py --config CONFIG --output-dir OUTPUT
.venv/bin/python experiments/vortices_prospective/select.py --config CONFIG --output-dir OUTPUT
.venv/bin/python experiments/vortices_prospective/validate.py --config CONFIG --output-dir OUTPUT
```

To reproduce a frozen result, retain the configuration and everything under
`endpoint_reference/`, `prospective/`, and `results/frozen_manifest.json`, then run
only `validate.py`.  It verifies the configuration/freeze hash before touching
hidden validation.

## Artifacts and caching

Each output root contains:

```text
endpoint_reference/endpoint_data.npz
endpoint_reference/reference_checkpoint.npz
endpoint_reference/reference_rollout.npz
prospective/aggregate_predictions.npz
prospective/selection_randomness.npz
results/frozen_manifest.json
hidden_validation/hidden_state_bank.npz
hidden_validation/hidden_observation_randomness.npz
results/validation_result.json
results/validation_trials.csv
results/report.md
results/*.png
results/runtime_breakdown.json
results/last_invocation_runtime.json
```

Every expensive stage has a configuration/content signature and is reused on a
second compatible invocation.  A frozen manifest with incompatible inputs fails
closed; use a new output directory for a changed experiment.  Smoke outputs are
development diagnostics and are not paper-authoritative.
