# Differentiable many-body completion

Reusable JAX simulators, reduced-statistic datasets, validation tooling, and two differentiable
solver Tesseract build contexts for the many-body completion methodology.

## Implemented components

- Smooth periodic chord distances for the pair energy and pair observables.
- A separate full-period oriented bond feature for local angular quantities. This is necessary because
  the half-period chord displacement is periodic only after squaring and cannot be used directly as a
  translation-invariant oriented bond vector.
- Pair-only and explicit exchangeable three-body/angular synthetic regimes.
- Fixed-step differentiable overdamped Langevin simulation.
- Gaussian radial pair coefficients averaged over replicas at the ensemble level.
- Held-out angular moments kept separate from the observed projection constraints.
- Deterministic S1--S3 problem fixtures.
- Dataset recomputation, wrapping, translation, permutation, rank, and ambiguity diagnostics.
- A proximal physical-relaxation solver with backtracking and an unrolled reverse-mode derivative.
- A complete physical-relaxation Tesseract build context with `apply`, `abstract_eval`, Jacobian, JVP,
  and VJP endpoints.
- A ridge-regularized SQP ensemble moment projector with diagonal whitening, exact-norm merit
  backtracking, rank monitoring, explicit basis pruning, correction clipping, and unrolled gradients.
- A complete moment-projection Tesseract build context with gradients with respect to both input
  coordinates and target moments.
- A compact pure-JAX conditional message-passing generator with permutation, toroidal translation,
  and square-box D4 equivariance.
- Explicit full-batch Adam training through both local solvers, including parameter-space finite-
  difference checks, gradient clipping, correction-burden metrics, and reproducible output archives.
- A common Base/Post-hoc/Relax-E2E/Full-E2E ablation API with compute-minimal solver routing, matched
  initialization, fair serving-stage evaluation, and deterministic comparison archives.
- Leakage-safe calibration training with stratified train/validation splits, deterministic fixed-shape
  minibatches, train-only normalization, and isolated JAX workers for solver-heavy modes.
- Evaluation-only angular distribution metrics, including whitened RMSE, RBF MMD, and hidden-regime
  centroid separation at the generated, relaxed, projected, and serving stages.

## Environment

CUDA host environment:

```bash
./scripts/bootstrap_cuda.sh
source .venv/bin/activate
```

CPU development environment:

```bash
./scripts/bootstrap_cpu.sh
source .venv/bin/activate
```

The repository deliberately retains the working integration pins:

```text
jax[cuda]==0.8.1
tesseract-core==1.10.0
tesseract-jax==0.2.3
```

Docker is required to build and serve Tesseracts.

> **CLI name collision:** some Linux distributions install the OCR program as `tesseract`. Always activate `.venv` before building. The build script verifies that it found Pasteur Labs Tesseract Core rather than the OCR executable.

## Generate and validate data

Run the statistically inspectable calibration configuration:

```bash
./scripts/generate_and_validate_data.sh
```

Equivalent explicit commands:

```bash
mbc-generate-data \
  --config configs/calibration_smoke.yaml \
  --output data/calibration_smoke.npz

mbc-validate-data data/calibration_smoke.npz \
  --output artifacts/calibration_validation.json

mbc-match-regimes data/calibration_smoke.npz \
  --output artifacts/calibration_matches.json
```

The included calibration archive has shape `(S, M, N, 2) = (16, 8, 8, 2)`. Its stored moments,
energies, distances, and overlap fractions recompute exactly; periodic translation and permutation
errors are at floating-point precision; and the six-dimensional pair basis has effective rank six.
The best current cross-regime match has whitened pair distance about `0.713` and standardized angular
distance about `3.431`. This is suitable for solver development and regime-calibration work, but it is
not yet the final paper-scale ambiguity showcase.

The CI fixture remains:

```bash
mbc-generate-data --config configs/tiny_smoke.yaml --output data/tiny_smoke.npz
mbc-validate-data data/tiny_smoke.npz --output artifacts/tiny_validation.json
```

Because it has only four samples for six pair coefficients, its covariance is intentionally rank
limited and the validation report marks it `calibration_required` rather than treating it as evidence
of matched hidden regimes.

## Dataset contract

- `coordinates`: `(S, M, N, 2)`
- `pair_moments`: `(S, R)`, averaged over ordered pairs and then replicas
- `angular_moments`: `(S, Q)`, held out from projection constraints
- `energy_per_replica`, `minimum_pair_distance`, `overlap_fraction`: `(S, M)`
- exact simulator, basis, regime, seed, backend, and schema metadata in the adjacent JSON file

Generate the fixed S1--S3 inputs with:

```bash
mbc-generate-problems --output data/smoke_problems.npz
```

## Test the local physical relaxation solver

```bash
pytest -q
```

The S1 test verifies increased separation, lower repulsive energy, lower proximal objective, retained
center of mass, translation/permutation equivariance, and a directional-derivative sweep against
centered finite differences.

## Build the physical-relaxation Tesseract

```bash
./scripts/build_physical_relaxation_tesseract.sh

tesseract run manybody-physical-relaxation check

tesseract run manybody-physical-relaxation apply \
  @tesseracts/physical_relaxation/examples/s1_payload.json
```

Check the exposed derivative endpoints:

```bash
tesseract run \
  --runtime-args "--input-paths coordinates --output-paths relaxed_coordinates --eps 1e-5" \
  manybody-physical-relaxation check-gradients \
  @tesseracts/physical_relaxation/examples/s1_payload.json
```

Then test host-side `tesseract-jax` composition and `jax.grad`:

```bash
python experiments/smoke_tests/run_physical_relaxation_tesseract.py
```

The first image uses `jax==0.8.1` on CPU for a clean correctness milestone while the host remains on
`jax[cuda]==0.8.1`. GPU-accelerating the Tesseract should be a separate change after the API and
VJP checks pass, so numerical and container issues are not debugged simultaneously.


## Test the local ensemble moment projector

Run the full test suite:

```bash
pytest -q
```

Run the deterministic S2 report directly:

```bash
python experiments/smoke_tests/run_moment_projection_local.py
cat artifacts/moment_projection_local_s2.json
```

The included S2 fixture starts with one whitened moment residual of about `7.56e-3`. The current SQP
core reduces it to about `4.57e-11` in three accepted iterations, with RMS per-particle correction
about `2.58e-3`. The exact final rank is `1/1`, the KKT stationarity diagnostic is about `1.41e-7`,
and gradients with respect to both coordinates and target moments are finite.

The projection solves the identity-coordinate-metric case `W_x = I`. `moment_scales` provide diagonal
coefficient whitening. `basis_mask` preserves fixed shapes while allowing known redundant coefficients
to be explicitly pruned. A rank-deficient unpruned basis returns finite regularized output together
with `rank_deficient=true`; it is never silently treated as well-conditioned.

## Build the moment-projection Tesseract

```bash
./scripts/build_moment_projection_tesseract.sh

tesseract run manybody-moment-projection check

tesseract run manybody-moment-projection apply \
  @tesseracts/moment_projection/examples/s2_payload.json
```

Check the exposed derivatives with respect to both differentiable inputs:

```bash
tesseract run \
  --runtime-args "--input-paths coordinates,target_moments --output-paths projected_coordinates --eps 1e-5" \
  manybody-moment-projection check-gradients \
  @tesseracts/moment_projection/examples/s2_payload.json
```

Then test the host-side `tesseract-jax` call and `jax.grad`:

```bash
python experiments/smoke_tests/run_moment_projection_tesseract.py
```

## Run S3 scalar composition and training

S3 uses the deterministic generator

```text
G_a(Z, c) = wrap(X_base + a Z)
```

and trains the single scalar `a` through the complete local composition

```text
scalar generator -> proximal relaxation -> ensemble moment projection -> outer loss
```

Run the validated local experiment:

```bash
./scripts/run_s3_local.sh
cat artifacts/s3_scalar_local.json
head artifacts/s3_scalar_trace.csv
```

The default fixture starts from `a=0.20` with a known target `a*=0.65`. With 30 projected-gradient
steps, the current implementation reaches approximately `a=0.6479`. The outer loss decreases by a
factor of about `19.6`, the final projected moment error is about `6.0e-7`, and the initial composed
gradient agrees with centered finite differences to about `5e-11` relative error. Both solver stages
reach their configured stopping criteria and the four-coefficient projection remains full rank.

The reusable training loop accepts any objective with the signature
`objective(a) -> (loss, scalar_metrics)`. The same loop is therefore used by the local JAX experiment
and the Tesseract-backed experiment; only the two solver callables change.

After building both images, run the container composition:

```bash
./scripts/build_physical_relaxation_tesseract.sh
./scripts/build_moment_projection_tesseract.sh
python experiments/smoke_tests/run_s3_tesseracts.py
```

The local S3 code is split into reusable modules:

- `composition.py`: scalar generator, periodic correction metric, and local two-solver composition;
- `scalar_training.py`: S3 objective, finite-difference sweep, and solver-agnostic scalar optimizer;
- `configs/s3_scalar.yaml`: all physical, solver, loss, training, and acceptance parameters.

## Train the compact native generator

The first neural milestone uses scalar node latents and reduced pair coefficients as inputs. Pairwise
messages depend only on scalar node states and smooth radial features. Coordinate updates are built
from full-period periodic direction vectors multiplied by symmetric learned scalar weights. This gives
particle-permutation equivariance, toroidal translation equivariance even across wrapping boundaries,
and D4 equivariance in a square box. It deliberately does not claim unrestricted E(2) equivariance.

Run the local two-condition smoke training:

```bash
./scripts/run_native_generator_smoke.sh
cat artifacts/native_generator_smoke.json
head artifacts/native_generator_smoke_trace.csv
```

The smoke configuration uses two targets from `tiny_smoke.npz`, but its latent anchors are independently
constructed translated periodic grids with jitter; no target microscopic coordinates are passed to the
generator. The current 3,014-parameter model reduces mean total correction RMS from about `0.06360` to
`0.05406` in 20 Adam steps, a reduction of about `15.0%`. Final projected moment error is about
`5.20e-6`; both final solver stages converge; the physical relaxation remains active with a small
nonzero displacement; the projection remains full rank; and a parameter-space directional derivative
agrees with centered finite differences to about `4.45e-8` relative error.

Outputs are stored separately so the run can be inspected without rerunning training:

- `native_generator_smoke.json`: configuration, acceptance metrics, and full history;
- `native_generator_smoke_trace.csv`: one row per optimization step;
- `native_generator_smoke_outputs.npz`: anchors and generated/relaxed/projected ensembles;
- `native_generator_smoke_parameters.npz`: flattened trained parameter arrays, restorable with
  `restore_generator_parameters` after initializing an architecture-compatible template.

This is an integration smoke test, not yet a population-completion result. It verifies that a reusable
conditional neural model can receive useful gradients through the complete local solver composition.

## Run the four training ablations

The ablation contract is explicit:

- **Base:** train and serve the native generator output; no scientific solver runs during training.
- **Post-hoc:** train the identical native objective as Base, then apply relaxation and projection only
  for evaluation/serving. Its trained parameters must therefore match Base exactly.
- **Relax-E2E:** train through physical relaxation; projection is evaluation/serving-only.
- **Full-E2E:** train through both physical relaxation and moment projection.

This implementation deliberately does not use an identity straight-through estimator for a stopped
solver. A solver is either part of the differentiated training stage or excluded from the training
objective. Complete three-stage diagnostics are evaluated separately after training.

Run the deterministic two-condition routing smoke test:

```bash
./scripts/run_generator_ablations.sh
cat artifacts/generator_ablation_smoke.json
head artifacts/generator_ablation_smoke_trace.csv
```

All modes use exactly the same initialized 3,014-parameter model, latent anchors, node latents, targets,
and optimizer settings. Base/Post-hoc parameters agree exactly. Independent parameter-space gradient
checks for Base, Relax-E2E, and Full-E2E currently have best relative errors below `3e-8`; all final
relaxation and projection evaluations converge and remain full rank.

The tiny fixture is intended to validate routing, not to establish the paper's scientific ablation
hypothesis. After 15 steps, Full-E2E's total correction RMS is about `0.05785`, versus `0.05789` for
Post-hoc—an improvement of only about `0.07%`. A stronger separation must be sought on the larger
calibration and matched-regime datasets rather than tuned into this smoke fixture.

Outputs:

- `generator_ablation_smoke.json`: configuration, routing semantics, gradient checks, and summaries;
- `generator_ablation_smoke_trace.csv`: long-form training traces for all four named modes;
- `generator_ablation_smoke_outputs.npz`: generated, relaxed, and projected arrays for each mode;
- `generator_ablation_smoke_parameters.npz`: path-keyed parameter archives for each mode.

## Run calibration-scale minibatch ablations

The calibration experiment uses all 16 samples in `calibration_smoke.npz`, with a deterministic
stratified split of 12 training and 4 validation samples. Each hidden regime contributes six training
and two validation samples. Condition normalization, projection scales, and angular descriptor scales
are fitted from the training split only. Hidden labels and angular moments never enter `GeneratorBatch`,
the condition vector, the projection constraints, or the optimizer objective.

Run the complete isolated-worker experiment:

```bash
./scripts/run_calibration_ablations.sh
cat artifacts/calibration_ablations.json
```

The driver trains Base, Relax-E2E, and Full-E2E in separate JAX processes, evaluates Post-hoc from the
exact Base parameter archive, and runs the Full-E2E finite-difference check in a fourth isolated
process. This bounds XLA compilation memory without changing initialization, minibatch order, or model
parameters. `--skip-workers` can re-aggregate completed worker artifacts after an interrupted reporting
step.

The default run uses two epochs and six total optimizer updates. Current validation results are:

- all relaxation and projection evaluations converge and all projection Jacobians remain full rank;
- projected pair-moment errors are approximately `1.30e-4` for Post-hoc, `1.21e-4` for Relax-E2E,
  and `9.40e-5` for Full-E2E;
- Full-E2E validation correction RMS is approximately `0.075575`, versus `0.075611` for Post-hoc,
  a small improvement of about `0.048%`;
- the Full-E2E parameter directional derivative agrees with centered finite differences to about
  `1.63e-4` relative error;
- held-out angular MMD and regime-separation diagnostics are reported but are not optimized.

This result validates the minibatch/data/evaluation machinery. The correction-burden separation is too
small to support a scientific claim, and the four-sample validation split is too small for a definitive
higher-order distribution comparison.

Artifacts:

- `calibration_ablations.json`: split metadata, train-only scales, solver metrics, angular diagnostics,
  gradient check, and four-mode summaries;
- `calibration_ablations_trace.csv`: minibatch optimization traces for all four named modes;
- `calibration_ablations_outputs.npz`: train/validation coordinates, pair moments, and angular moments
  at every pipeline stage;
- `calibration_ablations_parameters.npz`: path-keyed parameter archives for every mode.

## Run conditional equivariant flow matching

The first stochastic sampler uses conditional flow matching on the periodic box. A uniform torus
ensemble is coupled to each reference ensemble by a componentwise shortest periodic displacement. The
particle-mean displacement is removed independently in every replica, so the path reaches the target up
to a global translation and the velocity network never has to learn an unidentifiable translation gauge.
Smooth physical observables continue to use chord geometry; minimum-image geometry is used only for the
flow interpolation target.

The time-conditioned velocity field reuses the audited equivariant message-passing dynamics. Time and
reduced pair statistics are scalar node features, and vector velocities are assembled from periodic
relative directions with symmetric scalar edge weights. The resulting field is permutation equivariant,
toroidal-translation invariant as a velocity, square-box `D4` equivariant, and zero mean over particles.
Sampling uses a fixed-step wrapped Euler or Heun ODE integrator.

Run the toy calibration experiment:

```bash
./scripts/run_flow_matching_toy.sh
cat artifacts/flow_matching_toy.json
```

The default run uses 90 minibatch updates, four stochastic samples per validation condition, and a
16-step Heun solver. Current results are:

- fixed-key training flow loss decreases by about `0.8%`;
- fixed-key validation flow loss decreases by about `2.4%`;
- validation pair-statistic error decreases from about `9.58` to `4.83` in train-standard-deviation units;
- mean repulsive energy decreases from about `0.388` to `0.241`;
- overlap fraction decreases from about `0.0550` to `0.0343`;
- held-out angular MMD decreases from about `0.717` to `0.474`;
- a parameter-space directional derivative agrees with centered finite differences to about `1.25e-8`
  relative error.

These results establish a working stochastic sampler and correct torus/equivariance machinery, but not
yet multimodal recovery. The predicted hidden-regime centroid separation remains below the reference
separation, so the next scientific target is the exact homometric two-mode benchmark rather than more
tuning on this calibration archive.

Artifacts:

- `flow_matching_toy.json`: configuration, fixed-key losses, gradient check, and sampling diagnostics;
- `flow_matching_toy_trace.csv`: one row per stochastic minibatch update;
- `flow_matching_toy_outputs.npz`: initial and trained samples plus pair/angular descriptors;
- `flow_matching_toy_parameters.npz`: path-keyed trained parameter arrays.

## Run the exact homometric benchmark

The homometric benchmark uses two non-congruent four-point motifs on the `12 x 12` discrete torus.
After scaling into the unit periodic box, their six smooth chord distances agree as a multiset to
float64 precision, so every radial pair coefficient in the configured basis is identical. Exhaustive
checks confirm that the motifs are not related by a global torus translation, a square-box `D4`
transformation, or particle relabeling. Their held-out eight-coefficient angular descriptors differ by
approximately `1.4153` in Euclidean norm.

Generate and certify the benchmark:

```bash
./scripts/generate_homometric_benchmark.sh
cat artifacts/homometric_validation.json
```

The archive contains 64 symmetry-augmented ensemble samples from each mode, with eight replicas and four
particles per ensemble. Stored particle labels remain canonical; flow training applies a fresh shared
source-target permutation at each update, preserving exchangeability without injecting random target-label
noise. The mathematically common pair condition is explicitly coalesced to one exact all-zero normalized
condition after verifying that the archive deviation is below `1e-10`. Hidden mode labels and angular
descriptors remain evaluation-only.

Train the Base conditional flow on that single ambiguous condition:

```bash
./scripts/run_homometric_flow.sh
cat artifacts/flow_matching_homometric.json
```

The default CPU-safe run uses 224 batch-size-four updates, a 16-step Heun sampler, and 5,522 parameters.
Current deterministic results are:

- fixed-key validation flow loss decreases by about `4.6%`;
- mean pair-moment error decreases from about `0.425` to `0.218`;
- mean repulsive energy decreases from about `5.327` to `1.870`;
- held-out angular MMD decreases from about `0.958` to `0.862`;
- the fraction far from both reference angular modes decreases from `0.875` to `0.750`;
- both modes are present, with an estimated `A/B` split of `0.8125/0.1875` over 32 generated replicas;
- a parameter directional derivative agrees with centered finite differences to about `1.18e-10`.

The mode-frequency estimate is intentionally described as a small-sample smoke diagnostic, not a final
calibrated population result. The Base flow is not balanced yet; the next stochastic ablation must report
this mode bias rather than hiding it behind pair feasibility. Sampling is chunked and synchronized between
initial and trained models to bound XLA memory on CPU validation hosts.

Artifacts:

- `homometric_benchmark.npz/json`: exact benchmark coordinates, common pair condition, reference motifs,
  and held-out descriptors;
- `homometric_validation.json`: distance-multiset, non-congruence, recomputation, and separation checks;
- `flow_matching_homometric.json`: training, gradient, feasibility, and mode-coverage report;
- `flow_matching_homometric_trace.csv`: stochastic optimization history;
- `flow_matching_homometric_outputs.npz`: initial/trained samples, descriptors, labels, and mode distances;
- `flow_matching_homometric_parameters.npz`: flattened trained flow parameters.

## Next milestones

1. Add stochastic Base, Post-hoc, Relax-E2E, and Full-E2E routing around the flow ODE output.
2. Apply relaxation and projection to identical sampled priors and report feasibility, correction burden,
   mode frequencies, angular MMD, and mode-transition rates.
3. Fine-tune through each permitted solver path while keeping the Base/Post-hoc flow objective identical.
4. Increase isolated sampling statistics for the final showcase and attach confidence intervals to mode
   frequencies and correction metrics.
5. Run the stochastic ablation through Docker-backed Tesseracts and report solver and sampling costs.
6. Add the optional implicit KKT VJP while retaining the validated unrolled baseline.
