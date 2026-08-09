# MFSI experiment record and results

Last consolidated: 2026-08-09

This document records the MFSI experiments currently present in this
repository, how they were evaluated, the results produced so far, and the main
limitations of those results. It is a results snapshot, not a claim that every
number below is publication-ready.

The primary machine-readable sources are the JSON/CSV files under `results/`.
The commands for reproducing them are also listed in [`notes.md`](notes.md).

## 1. Evidence map

| Study | Main purpose | Backend/result mode | Replication | Status |
|---|---|---:|---:|---:|
| Experiment A / levels 0–1 | Exact 1D diagnostic, learned correction, matched baselines | JAX standard; Tesseract kernel checks | One main run | Passed validation gates |
| Experiment B / levels 0–1 | 2D ring-to-four-lobe distribution shift | JAX standard | One main run | Completed |
| Experiment B multiseed | Training/evaluation robustness | JAX full | 10 training × 10 evaluation crossed-seed cells | Completed |
| Level-1 ablation | Implicit vs unrolled vs stop-gradient schedule derivatives | JAX reference snapshot | Deterministic comparison | Completed |
| Scalar level 2 | Adapt one constant noise amplitude | JAX and served Tesseract, standard | Deterministic quadrature | Passed; backend parity passed |
| Advanced level 2: finite neural | Finite banks and a 3-parameter schedule in 2D | JAX and served Tesseract, standard | Independent train/fresh banks | Passed; backend parity passed |
| Advanced level 2: many-body | 16 particles, 32D state, radial features | JAX and served Tesseract, standard | Independent train/fresh banks | Passed; backend parity passed |
| Paper-facing level 2 | 32 particles, 64D, full MLP, baselines and CIs | JAX standard | Five independent banks | Passed declared gates |
| Paper-facing Tesseract check | Served full-MLP correction kernel | Tesseract quick | One seed | Passed; parity passed |

The metrics are not interchangeable across rows. Experiment A uses Wasserstein
and KS distances; Experiment B uses MMD; the level-2 schedule studies primarily
measure correction energy, forcing power, calibration, and ESS.

## 2. Implementation and backend conventions

The common pipeline is:

```text
endpoint samples
  -> stochastic-interpolant/reference path
  -> empirical exponential I-projection onto a moment fiber
  -> implicit multiplier derivative and reweighting source h_t
  -> conservative Ritz or neural correction
  -> corrected ODE generation
  -> comparison with an independently sampled projected law
```

Two execution routes are maintained:

- `jax`: runs the scientific kernels directly in-process.
- `tesseract`: builds a Tesseract Core Docker image, serves its `/apply`
  endpoint, calls it over HTTP, and tears it down after the experiment.

Neural optimization remains native JAX. Tesseract is used for the context-free
scientific component maps. This avoids treating a long training job as a
Tesseract component.

## 3. Experiment A: exact one-dimensional diagnostic

### 3.1 Setup

- State dimension: 1.
- Target moments: mean `0`, second moment `1`.
- Main evaluation: 2,048 particles at 11 times from `0` to `1`.
- Deterministic flows: 120 Heun steps, 240 velocity evaluations.
- MGD: interacting-particle predictor/corrector, 1,000 steps in the matched
  benchmark.
- Distribution metrics: W1, W2, KS, and fourth-moment error.
- The exact one-dimensional weighted-Poisson solution is reported only as an
  oracle floor.
- The active single-run validation loaded the existing reference and Ritz
  checkpoints; the empty training histories in its JSON mean it was an
  evaluation run, not a fresh retraining run.

### 3.2 Learned-pipeline validation

The active validation file reports all gates passing.

| Diagnostic | Result |
|---|---:|
| Flow-matching MSE | 1.6191 |
| Zero-predictor MSE | 2.0483 |
| FM/zero MSE ratio | 0.7905 |
| Implicit VJP relative error vs finite difference | `1.77e-10` |
| Maximum calibration residual | `4.44e-16` |
| Minimum ESS fraction | 0.5358 |
| Median / maximum weak-form residual | 0.0271 / 0.0531 |
| Median / maximum projected MMD | 0.0453 / 0.0772 |
| Maximum generated moment error, learned flow | 0.0544 |
| Maximum generated moment error, safety-corrected flow | `2.09e-7` |

The learned reference therefore beats the zero predictor, the implicit
derivative matches finite differences, and empirical projection is accurate.
The optional population-rate safety correction nearly eliminates moment drift,
but it is an extra correction and should not be confused with better full-law
accuracy by itself.

### 3.3 Matched method benchmark

| Method | Mean interior W1 | Mean interior W2 | Max second-moment error | Fourth-moment RMSE | NFE / steps |
|---|---:|---:|---:|---:|---:|
| Raw SI | 0.1760 | 0.2080 | 0.4999 | 1.5534 | 240 / 120 |
| Moment tangent | **0.0104** | **0.0133** | `8.03e-8` | 0.1047 | 240 / 120 |
| MGD | 0.0511 | 0.0670 | `9.21e-6` | 0.5421 | 1,000 stochastic steps |
| Learned MFSI | 0.0433 | 0.0529 | 0.0528 | 0.1934 | 240 / 120 |
| Learned MFSI + safety | 0.0338 | 0.0472 | `2.37e-7` | 0.3200 | 240 / 120 |
| Exact MFSI oracle | 0.000221 | 0.000266 | `1.66e-4` | 0.00986 | 240 / 120 |

Interpretation:

- Every correction substantially improves on raw SI.
- The simple instantaneous moment-tangent baseline is strongest in this exact
  low-dimensional setting.
- Learned MFSI is better than the tested MGD configuration on W1, W2, KS, and
  fourth-moment error, but it does not reach the tangent or oracle solutions.
- The oracle is a diagnostic floor, not a trainable baseline.
- Recorded CPU warm integration medians were approximately 92.7 ms (raw),
  131.5 ms (tangent), 1,590 ms (MGD), 173.4 ms (learned), 365.8 ms (safe), and
  72.6 ms (oracle). These are local microbenchmarks with faithful native
  discretizations, not equal wall-clock budgets.

Primary artifacts:

- [`results/learned_validation.json`](results/learned_validation.json)
- [`results/method_benchmark_metrics.json`](results/method_benchmark_metrics.json)
- [`results/method_benchmark.png`](results/method_benchmark.png)
- [`results/example_a_density_overlay.png`](results/example_a_density_overlay.png)

## 4. Experiment B: two-dimensional ring to four lobes

### 4.1 Setup

- State dimension: 2.
- Minus endpoint: noisy ring.
- Plus endpoint: four axis-aligned noisy lobes.
- Shared constrained quantities: two means and three quadratic moments.
- Held-out angular Fourier features measure structure not explicitly fixed by
  those five constraints.
- Main deterministic integrations: 160 Heun steps and 320 NFE.

The endpoint moment checks are close to their population target while the
angular statistics differ, so the problem tests whether a path can retain the
measured fiber while transporting unconstrained shape.

### 4.2 Single-run result

| Method | Mean interior MMD | Max interior MMD | Max moment error | Mean angular error | Runtime (s) |
|---|---:|---:|---:|---:|---:|
| Raw SI | 0.1546 | 0.2268 | 0.6972 | 0.1352 | 0.289 |
| Moment tangent | 0.0721 | 0.0932 | `4.63e-7` | 0.1173 | 0.946 |
| Learned MFSI | **0.0698** | **0.0879** | 0.0424 | 0.1161 | 0.738 |
| Learned MFSI + safety | 0.0715 | 0.0899 | `6.34e-7` | **0.1156** | 1.283 |
| MGD-style baseline | 0.1055 | 0.1287 | `5.75e-6` | 0.2245 | 1.361 |

Additional diagnostics:

- Reference FM/zero MSE ratio: 0.7250.
- Fresh-bank calibration residual: at most `4.02e-16` over the reported times.
- Fresh-bank ESS fraction: 0.276 at the most difficult reported time.
- Maximum reported weak-form residual: 0.0457.
- Integrated learned correction energy: 0.3055.
- Tesseract/JAX kernel differences: zero to `8.88e-16` for the recorded maps.

### 4.3 Full multiseed result

This is the strongest replication result for Experiment B: 10 independently
trained models crossed with 10 independent evaluation seeds. The resulting 100
cells are repeated measurements, not 100 iid scientific replications: cells in
a row share a trained model and cells in a column share an evaluation seed.
Intervals below independently resample the two seed dimensions (20,000
deterministic crossed-bootstrap replicates). Cell SDs are descriptive only.

| Method | Mean interior MMD, crossed 95% CI | Max moment error, cell mean ± SD | Angular error, cell mean ± SD | Runtime, cell mean ± SD (s) |
|---|---:|---:|---:|---:|
| Raw SI | 0.1685 (0.1608, 0.1773) | 0.7015 ± 0.0052 | 0.1214 ± 0.0145 | 0.303 ± 0.017 |
| Moment tangent | 0.0713 (0.0656, 0.0778) | `(4.48 ± 0.16)e-7` | 0.1004 ± 0.0103 | 0.880 ± 0.055 |
| Learned MFSI | 0.0666 (0.0619, 0.0720) | 0.0509 ± 0.0118 | 0.0942 ± 0.0079 | 0.732 ± 0.027 |
| Learned MFSI + safety | **0.0657 (0.0616, 0.0702)** | `(6.79 ± 1.02)e-7` | **0.0938 ± 0.0078** | 1.287 ± 0.065 |
| MGD-style baseline | 0.1196 (0.1150, 0.1243) | `(9.92 ± 1.57)e-6` | 0.2203 ± 0.0115 | 1.430 ± 0.152 |

The central paired contrast is
`Δ_MFSI,tangent = MMD_MFSI-safe - MMD_tangent = -0.00564`, with crossed-seed
95% bootstrap CI `(-0.00787, -0.00364)`. Thus the advantage over tangent is
robust in this design. The analogous MFSI-safe-minus-MGD and
MFSI-safe-minus-raw intervals are `(-0.05699, -0.05063)` and
`(-0.10775, -0.09800)`. Random-effects method-of-moments diagnostics show that
evaluation-seed variance dominates training-seed variance for the central
contrast, which explains why treating the 100 cells as iid was misleading.

The multiseed ordering supports the single-run conclusion: learned MFSI and
the tangent baseline strongly improve on raw SI, and learned MFSI has the best
mean full-law MMD in this tested group. The safety variant controls moments to
near machine integration accuracy without materially degrading MMD.

Primary artifacts:

- [`results/example_b/example_b_results.json`](results/example_b/example_b_results.json)
- [`results/multiseed/example_b/jax/aggregate.json`](results/multiseed/example_b/jax/aggregate.json)
- [`results/example_b/path_mmd.png`](results/example_b/path_mmd.png)
- [`results/example_b/projection_diagnostics.png`](results/example_b/projection_diagnostics.png)
- [`results/example_b/snapshots_t075.png`](results/example_b/snapshots_t075.png)

## 5. Level-1 derivative and schedule ablations

The reference ablation compares differentiation through the empirical
I-projection multiplier solve.

| Check | Relative discrepancy |
|---|---:|
| Unrolled distortion gradient vs finite difference | `3.36e-9` |
| Implicit distortion gradient vs finite difference | `3.36e-9` |
| Implicit vs unrolled correction gradient | `1.27e-11` |
| Implicit correction gradient vs finite difference | `2.94e-9` |
| Stop-gradient vs unrolled correction gradient | **0.2083** |

Optimized correction energy was `1.8336e-4` for implicit differentiation and
the same to numerical precision for unrolled differentiation. Stop-gradient
optimization reached `1.8081e-3`, about ten times larger. This supports the
level-1 claim that implicit differentiation reproduces the unrolled derivative
while avoiding the scientifically wrong stop-gradient schedule derivative.

Artifact: [`results/reference/ablation_metrics.json`](results/reference/ablation_metrics.json).

## 6. Scalar level-2 schedule adaptation

### 6.1 Setup

- Experiment-A path and target moments.
- One learned parameter: constant stochastic-interpolant noise amplitude
  `beta`.
- 17 time points and 801 quadrature points.
- Initial `beta = 0.25`.
- Objective: integrated exact correction energy plus an ESS penalty below 0.60.
- 128 optimization steps.
- Known controlled optimum: `beta = 1`, where the raw path already stays on
  the selected moment fiber.

### 6.2 Result

| Metric | Initial | Optimized |
|---|---:|---:|
| Beta | 0.25 | 1.00101 |
| Objective | 0.46010 | `1.26045e-6` |
| Integrated correction energy | 0.45372 | `1.26045e-6` |
| Minimum ESS fraction | 0.54985 | 0.9999995 |
| Maximum moment error | `6.66e-16` | `4.44e-16` |

The correction-energy reduction was 99.9997%. The implicit directional
gradient matched a central finite difference with relative error `3.67e-8`.
JAX and served Tesseract results passed parity; important metric differences
were between zero and approximately `1.94e-12`.

Artifacts:

- [`results/level2_schedule/jax/level2_results.json`](results/level2_schedule/jax/level2_results.json)
- [`results/level2_schedule/jax/level2_schedule_summary.png`](results/level2_schedule/jax/level2_schedule_summary.png)
- [`results/level2_schedule/jax/level2_density_paths.png`](results/level2_schedule/jax/level2_density_paths.png)

This is a controlled recovery test, not evidence by itself that schedule
adaptation improves an unknown real system.

## 7. Advanced level-2 suite

These experiments remain separate from the scalar study and from the newer
paper-facing study.

### 7.1 Finite-neural 2D study

Configuration:

- Nine time points.
- 512 configurations per training time and 1,024 per independent validation
  time.
- Three schedule parameters.
- 64 fixed random neural features with a fitted output layer.

| Metric | Initial | Optimized |
|---|---:|---:|
| Fresh-bank correction energy | 0.47451 | 0.04030 |
| Fresh-bank forcing power | 9.97366 | 0.07193 |
| Minimum fresh ESS | 0.24418 | 0.98536 |
| Beta range | 0.1396–0.4179 | 0.9027–0.9229 |

Fresh-bank correction energy fell 91.5%. Calibration residual was below
`8.89e-16`, and the schedule-gradient relative error was `3.03e-8`.

### 7.2 Many-body study

Configuration:

- 16 particles in 2D: 32-dimensional microscopic state.
- Seven time points.
- 72 training configurations and 160 independent validation configurations
  per time.
- Three schedule parameters.
- Periodic radial-pair features, three pair/gyration constraints, and held-out
  fourfold order.

| Metric | Initial | Optimized |
|---|---:|---:|
| Fresh-bank correction energy | 0.14836 | `1.779e-4` |
| Fresh-bank forcing power | 2.44780 | 0.01640 |
| Minimum fresh ESS | 0.85800 | 0.93733 |
| Beta range | 0.4469–0.6732 | 0.4994–0.5055 |

Fresh-bank energy fell 99.88%. Calibration residual was below `4.45e-16`, the
gradient relative error was `2.67e-7`, and the held-out structural-motion gate
passed.

### 7.3 Backend result

Both standard studies passed under direct JAX and served Tesseract. The largest
listed JAX/Tesseract discrepancies were approximately `9.62e-12` for the
finite-neural gradient check and `3.58e-15` for the many-body gradient check.

Limitations:

- These use fixed random hidden features rather than a fully trained MLP.
- They test schedule adaptation and fresh-bank Ritz quantities, not a complete
  multi-seed corrected ODE benchmark against independent projected laws.
- Each standard result uses one reproducible training/fresh-bank realization,
  so it is strong validation evidence but weak uncertainty evidence.

Artifacts:

- [`results/level2_suite/finite_neural/jax/results.json`](results/level2_suite/finite_neural/jax/results.json)
- [`results/level2_suite/finite_neural/jax/finite_neural_dashboard.png`](results/level2_suite/finite_neural/jax/finite_neural_dashboard.png)
- [`results/level2_suite/manybody/jax/results.json`](results/level2_suite/manybody/jax/results.json)
- [`results/level2_suite/manybody/jax/manybody_dashboard.png`](results/level2_suite/manybody/jax/manybody_dashboard.png)

## 8. Paper-facing level-2 N=32 study

This is the most comprehensive level-2 experiment currently in the repository.

### 8.1 The six design goals implemented

1. Compare a hand constant, an optimized scalar, and a three-parameter schedule
   over independent finite banks.
2. Train all weights of a two-hidden-layer invariant MLP potential and use its
   conservative correction in an ODE.
3. Compare generated samples with an independently sampled projected law using
   MMD on radial descriptors plus held-out q4.
4. Select the correction amplitude on a separate validation bank, with an exact
   zero-correction fallback, then measure Ritz gain on another test bank.
5. Use 32 particles in 2D, giving a 64-dimensional many-body state with radial
   pair constraints and energy-relaxed endpoints.
6. Report raw, instantaneous tangent, and neural methods with five-bank 95%
   confidence intervals, wall time, and matched NFE.

### 8.2 Bank and endpoint construction

- Standard seeds: `401, 402, 403, 404, 405`.
- The minus population is a low-q4 disordered phase.
- The plus population is a high-q4 fourfold phase.
- Both populations are briefly relaxed under compatible soft many-body
  energies.
- Three smooth radial-pair RBF observables are constrained.
- A linear program finds a common point in the relative interiors of the two
  finite empirical moment hulls; each bank is then independently exponentially
  calibrated to that point.
- No shared configurations are inserted to make the endpoint moments match.

Endpoint result across the five standard banks:

| Quantity | Result |
|---|---:|
| Mean weighted minus q4 | 0.0740 |
| Mean weighted plus q4 | 0.7730 |
| Hidden q4 gap | approximately 0.699 |
| Maximum endpoint calibration residual | `8.25e-16` |

Thus the measured radial moments match to numerical precision while a large
unconstrained angular structural change remains.

### 8.3 Schedule comparison

The scalar model is nested exactly in the three-parameter family. The
multi-parameter candidate is optimized on one bank, selected against the
nested scalar on a second bank, and reported on a third bank.

| Schedule | Correction energy, mean (95% CI) | Forcing power, mean (95% CI) | Minimum ESS, mean (95% CI) |
|---|---:|---:|---:|
| Hand constant | 1.3153 (1.1701, 1.4605) | 37.9802 (33.4813, 42.4791) | 0.3583 (0.2581, 0.4586) |
| Optimized scalar | 0.10585 (0.06271, 0.14899) | 9.1509 (7.8581, 10.4436) | 0.4200 (0.3614, 0.4787) |
| Selected multi | **0.09529 (0.05274, 0.13783)** | **8.7983 (7.5393, 10.0574)** | 0.4187 (0.3505, 0.4869) |

Relative to the hand schedule, the selected multi schedule reduced mean
correction energy by 92.8% and forcing power by 76.8%. Relative to the optimized
scalar, its mean energy was 10.0% lower and forcing power was 3.9% lower.

The paired multi-minus-scalar energy effect was `-0.01056`, with 95% CI
`(-0.02767, 0.00655)`. Therefore the mean favors the richer schedule, but the
five-bank interval includes zero. The multi candidate was selected on three of
five banks; the nested scalar was retained on two.

### 8.4 Full neural correction

- Input: eight permutation/translation/rotation-invariant radial descriptors
  plus three time features.
- Network: two hidden SiLU layers, width 18.
- Output: scalar potential; correction is its negative configuration gradient.
- All layers are trained with the empirical Deep-Ritz objective.
- A separate validation bank selects an amplitude in `[0, 1]`; negative
  validation gain activates the exact zero fallback.
- A third bank supplies the reported test gain.

| Metric | Five-bank result |
|---|---:|
| Selected gate amplitude | mean 0.834, 95% CI (0.627, 1.041) |
| Held-out Ritz gain | **0.04415**, 95% CI **(0.02083, 0.06747)** |
| Test correction energy | 0.12030, 95% CI (0.06185, 0.17874) |

The held-out Ritz interval is entirely positive. This is the strongest current
evidence that the full invariant MLP learns a useful conservative correction
on unseen finite banks.

### 8.5 End-to-end generated-law result

All methods use 24 Heun steps and therefore 48 velocity evaluations. The
primary MMD integrates interior times `0.25`, `0.50`, and `0.75`; endpoint MMD
is stored separately because the endpoint banks are supplied to the bridge.

| Method | Interior MMD², mean (95% CI) | Maximum moment error, mean (95% CI) | Wall time, mean (95% CI), s | NFE |
|---|---:|---:|---:|---:|
| Raw | 0.01813 (0.01257, 0.02369) | 0.02570 (0.01894, 0.03245) | 0.506 (0.195, 0.817) | 48 |
| Tangent | **0.00898 (0.00368, 0.01429)** | **0.00199 (-0.00002, 0.00400)** | 2.535 (1.555, 3.515) | 48 |
| Neural | 0.01830 (0.01313, 0.02347) | 0.02291 (0.01465, 0.03117) | 2.377 (1.520, 3.234) | 48 |

These are ordinary Student-t intervals and are not clipped to a metric's
physical range; that is why the tangent moment-error lower endpoint is slightly
negative even though every observed error is nonnegative.

Paired effects:

| Effect | Mean difference | 95% CI | Interpretation |
|---|---:|---:|---|
| Tangent minus raw interior MMD² | **-0.00915** | **(-0.01367, -0.00462)** | Clear improvement |
| Neural minus raw interior MMD² | 0.00017 | (-0.00602, 0.00636) | Statistically neutral |
| Neural minus tangent interior MMD² | **0.00932** | **(0.00673, 0.01190)** | Tangent is clearly better |
| Neural minus raw max moment error | -0.00279 | (-0.00583, 0.00025) | Mean improvement; interval touches zero |

The tangent correction reduces mean interior MMD² by 50.5% and has a paired
interval entirely below zero. The neural model has a convincingly positive
held-out Ritz gain but does not yet produce a statistically reliable full-law
MMD improvement after ODE integration. These are complementary findings, not
contradictory ones: the Ritz objective measures the local conservative source
fit, while end-to-end MMD also includes accumulated integration, finite-sample,
and representation errors. The neural-versus-tangent interval makes the
unresolved issue explicit: the cheaper instantaneous tangent method is
substantially better end to end in this experiment.

### 8.6 Local-to-rollout failure diagnosis

Five targeted probes were run on separate diagnostic banks while leaving the
primary random stream and primary result unchanged.

| Probe | Five-bank result | Reading |
|---|---:|---|
| Time-averaged Ritz gain on the original grid | 0.05809 (0.02740, 0.08878) | Positive local generalization |
| Time-averaged Ritz gain at off-grid times | 0.02284 (-0.00105, 0.04674) | Temporal interpolation loses most of the gain |
| Neural MMD², 24 / 48 / 96 Heun steps | 0.018300 / 0.018276 / 0.018270 | ODE discretization is not the bottleneck |
| Mean rollout feature Mahalanobis shift | 0.833 (0.740, 0.925) | Rollout leaves the center of the fitted fiber law |
| Rollout/fiber correction-energy ratio | 0.964 (0.828, 1.100) | No correction-amplitude explosion off distribution |
| Rollout-selected gate minus Ritz-gated test MMD² | -0.00173 (-0.00519, 0.00172) | Gate objective mismatch is plausible but not established |
| Angular-augmented minus radial-MLP test MMD² | -0.00348 (-0.01082, 0.00386) | Richer representation helps in mean only; interval crosses zero |

The angular diagnostic adds smooth `q2²`, `q4²`, and `q6²` invariants while
holding width, optimizer, and training bank fixed. Its held-out Ritz gain is
0.04103 (0.02030, 0.06175), and its test MMD² is 0.01482
(0.00690, 0.02274), but the paired improvement over the radial MLP is not yet
reliable. The rollout gate is selected on an independent validation rollout
and evaluated on the untouched primary test bank; its paired improvement also
includes zero.

The strongest diagnosis is therefore time approximation, with moderate
distribution shift as a secondary concern. Finer ODE integration is ruled out
at the tested resolutions. Representation and gate selection remain promising
follow-ups, but five banks do not support claiming either as the fix.

### 8.7 Tesseract result

The paper-facing Tesseract was built and called through a real served Docker
container in quick mode. On seed 401:

- particle count/state dimension: 32 / 64;
- correction relative error versus direct JAX: `6.35e-16`;
- descriptor relative error: `1.35e-16`;
- all quick acceptance gates passed.

The five-bank confidence-interval result is currently the direct-JAX standard
run. The Tesseract result validates deployment parity; it is not a second set
of five independent scientific replications.

Primary artifacts:

- [`results/level2_paper_study/jax/summary.json`](results/level2_paper_study/jax/summary.json)
- [`results/level2_paper_study/jax/paper_level2_summary.png`](results/level2_paper_study/jax/paper_level2_summary.png)
- [`results/level2_paper_study/jax/paper_level2_path_diagnostics.png`](results/level2_paper_study/jax/paper_level2_path_diagnostics.png)
- [`results/level2_paper_study/jax/paper_level2_failure_diagnostics.png`](results/level2_paper_study/jax/paper_level2_failure_diagnostics.png)
- [`results/level2_paper_study/tesseract/summary.json`](results/level2_paper_study/tesseract/summary.json)

Provenance note: `--aggregate-existing` was used after the complete five-seed
simulation to add derived interior-MMD effects and regenerate figures. That
operation replaced the top-level `elapsed_seconds` with aggregation time. Use
the recorded per-method and training timings for performance comparisons; do
not interpret the current top-level elapsed value as total simulation runtime.

## 9. MGD-specific numerical validation

The archived reference validation sweeps predictor/corrector noise levels
against the analytic Gaussian oracle for Experiment A. Across the displayed
cases, maximum W1 errors were roughly 0.0106–0.0152, maximum mean errors were
below `1.2e-10`, and maximum second-moment errors ranged from `7.5e-7` to about
`1.3e-5`. This verifies that the MGD implementation is actually simulated and
that its moment corrector is numerically effective.

Artifact: [`results/reference/mgd_validation_metrics.json`](results/reference/mgd_validation_metrics.json).

## 10. Tesseract validation summary

The core level-0/1 Tesseracts were also checked independently of the individual
experiment reports.

- Real Tesseract Core runtime was available through Docker.
- Reference-velocity runtime difference: `3.33e-16`.
- Fiber-weight runtime difference: `6.94e-18`.
- Fiber-velocity runtime difference: `4.44e-16`.
- Pure-JAX JVP/VJP and component parity checks were at numerical precision;
  the largest listed JVP discrepancy was `1.18e-11`.

Artifact: [`results/tesseract_validation.json`](results/tesseract_validation.json).

## 11. What the results currently support

Supported reasonably well:

- Empirical moment calibration converges to numerical precision on the tested
  finite banks.
- The implicit multiplier derivative matches unrolled and finite-difference
  derivatives, while a stop-gradient correction-energy derivative can be
  materially wrong.
- Schedule adaptation can dramatically reduce correction burden and improve
  finite-bank overlap.
- The method and its gradients agree between direct JAX and real served
  Tesseract components.
- In Experiment B, learned MFSI improves full-law MMD over raw SI, tangent, and
  the tested MGD-style baseline under a crossed 10-training-seed by
  10-evaluation-seed bootstrap; the 100 cells are not treated as iid.
- In the N=32 study, the tangent realization improves independent-law interior
  MMD with a paired interval below zero.
- The full invariant neural correction has positive held-out Ritz gain across
  five independent banks.

Not yet supported strongly enough:

- A statistically significant advantage of the three-parameter schedule over
  the optimized scalar schedule; its paired five-bank interval includes zero.
- An end-to-end N=32 neural MMD advantage; its paired interval is centered near
  zero.
- An N=32 neural advantage over tangent; neural-minus-tangent is positive with
  a 95% interval entirely above zero.
- Broad scaling claims beyond 32 particles or 64 state dimensions.
- Claims about realistic molecular systems: the current many-body endpoints
  are controlled synthetic soft-particle phases.
- Definitive wall-clock superiority: compilation, REST overhead, and native
  discretization choices differ across methods.

## 12. Reproduction commands

Run commands from the repository root after installing dependencies.

```bash
./scripts/install.sh
source .venv/bin/activate
```

Experiments A and B:

```bash
./scripts/run_example_a.sh --backend jax
./scripts/run_example_b.sh --backend jax

# Served Tesseract versions
./scripts/build_tesseracts.sh
./scripts/run_example_a.sh --backend tesseract
./scripts/run_example_b.sh --backend tesseract
```

Experiment-B multiseed sweep:

```bash
./scripts/run_multiseed_b.sh \
  --train-seeds "101 102 103 104 105 106 107 108 109 110" \
  --eval-seeds  "201 202 203 204 205 206 207 208 209 210" \
  --backend jax
```

Scalar level 2:

```bash
./scripts/run_level2.sh --backend jax
./scripts/run_level2_both.sh
```

Advanced level-2 suite:

```bash
./scripts/run_level2_suite.sh --backend jax
./scripts/run_level2_suite_both.sh
```

Paper-facing level 2:

```bash
# One-seed smoke test
./scripts/run_level2_paper_study.sh --backend jax --quick

# Five-bank standard experiment
./scripts/run_level2_paper_study.sh --backend jax

# Real served component check
./scripts/build_level2_paper_tesseract.sh
./scripts/run_level2_paper_study.sh --backend tesseract --quick

# Rebuild intervals and figures without rerunning complete seeds
./scripts/run_level2_paper_study.sh --backend jax --aggregate-existing
```

Validation and ablation commands:

```bash
./scripts/run_tesseracts.sh --build
./scripts/run_validations.sh --backend jax
./scripts/run_part0_ablations.sh
```

Quick modes are plumbing tests. Standard or full modes should be used for
scientific tables and figures.

## 13. Overall assessment

The repository now contains a coherent progression from an exact 1D test, to a
replicated 2D distribution-shift benchmark, to controlled schedule adaptation,
to finite-bank many-body studies with real Tesseract deployment parity.

The most defensible current paper results are:

- the Experiment-B 10×10 crossed-seed robustness result;
- the implicit-gradient correctness and stop-gradient ablation;
- the controlled scalar schedule recovery;
- the five-bank N=32 endpoint/calibration construction;
- the positive held-out full-MLP Ritz gain; and
- the paired improvement of the N=32 tangent correction on interior MMD.

The next scientific priority is to turn the positive neural Ritz result into a
reliable end-to-end neural law improvement, then increase the number of banks
and introduce less synthetic many-body endpoints.
