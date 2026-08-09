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
| Stage 3 rollout adaptation | Differentiate a 3-parameter time modulation through the frozen neural ODE | JAX standard | Five independent adaptation/selection/evaluation bank triples | Completed; surrogate improved, final-law interval crosses zero |
| Stage 3B confirmation | Confirm Stage 3 and isolate temporal structure and full credit assignment | JAX standard | Ten new model seeds and bank triples | Confirmed all three prespecified effects |
| Stage 4 fiber design | Differentiate a rank-three radial-observable subspace through the fixed moment-fiber construction | JAX standard | Five independent adaptation/selection/evaluation bank triples | Completed; mean improved, paired interval narrowly crosses zero |

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

### 8.7 Matched random-continuous-time training

The time-approximation hypothesis was tested directly without expanding the
network or tuning against the final law metric. The existing model trains on
six fixed times with 192 configurations per time. Its comparator uses 18
stratified uniform random times with 64 configurations per time. Both therefore
see 1,152 training configurations, take 420 optimizer steps, start from exactly
the same MLP initialization, and use the same untouched gate, off-grid test,
generation, and projected-law banks.

| Quantity | Fixed-grid training | Random-time training |
|---|---:|---:|
| Reference-grid time-averaged Ritz gain | 0.05809 (0.02740, 0.08878) | 0.04404 (0.00836, 0.07972) |
| Off-grid time-averaged Ritz gain | 0.02284 (-0.00105, 0.04673) | **0.04201 (0.01324, 0.07078)** |
| Mean reference-minus-off-grid degradation | 0.03525 | **0.00203** |
| Interior rollout MMD² | 0.01830 (0.01313, 0.02347) | 0.01853 (0.00388, 0.03318) |
| Neural minus tangent MMD² | 0.00932 (0.00673, 0.01190) | 0.00955 (-0.00057, 0.01967) |

Random-time training reduces the mean on/off-grid Ritz degradation by 94.2%.
Its off-grid interval is entirely positive, whereas the fixed-grid off-grid
interval touches zero. The paired random-time-minus-fixed-grid off-grid gain is
0.01917 with CI (-0.01327, 0.05161), so the between-model improvement itself is
not yet statistically resolved with five banks; the stronger evidence is the
near equality of reference-grid and off-grid gain within the random-time model.

This improvement does not translate into a mean rollout-MMD improvement:
random-time minus fixed-grid MMD² is 0.00023 with CI
(-0.01201, 0.01247). Nor does the random-time model beat tangent in mean. That
is not the progression criterion. Before examining the result,
rollout-adaptation readiness was defined as: off-grid gain must not decrease, and the mean
reference/off-grid degradation must fall by at least 50%. The observed 94.2%
reduction passes that criterion. Tangent MMD is explicitly not used by the
criterion, so the next scoped experiment can test differentiable rollout-aware
modulation while the remaining end-to-end neural limitation is reported
unchanged.

### 8.8 Tesseract result

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

### 8.9 Stage 3: rollout-aware differentiable correction adaptation

#### Scientific question and frozen setup

This single follow-up asks whether differentiation through the complete
generated trajectory can convert the locally useful random-time neural
correction into better global law-valued transport. It does not retrain or
redesign the potential. Each run restores the completed random-continuous-time
MLP and scalar Ritz gate from the five paper-facing seed artifacts and freezes
the model parameters. SHA-256 hashes before and after adaptation are identical.

Everything except the temporal amplitude was frozen:

- seeds `401, 402, 403, 404, 405`, with 32 particles and a 64-dimensional state;
- physical endpoint populations generated with seed `seed + 10000`, endpoint
  calibration, target moments, and the three measured radial-RBF observables;
- the deterministic polar-angle endpoint pairing used by the paper study;
- the selected three-parameter stochastic-interpolant schedule, reference
  velocity, invariant width-18 two-hidden-layer MLP, its random-time Deep-Ritz
  weights, and its held-out scalar gate;
- fixed-step Heun with 24 steps and 48 NFE, snapshots at
  `t = 0.25, 0.50, 0.75, 1.00`;
- the final radial-descriptor-plus-q4 weighted median-RBF MMD, moment-error
  calculation, q4 definition, and endpoint reporting.

The only new trainable object has three scalar parameters:

\[
a_\alpha(t)=\operatorname{sigmoid}\!\left(
\operatorname{logit}(g)+\alpha_0+\alpha_1\cos(2\pi t)
+\alpha_2\sin(2\pi t)\right),
\]

where `g` is the frozen scalar Ritz gate. Thus `alpha = (0,0,0)` exactly
reproduces the existing neural rollout, the modulation is smooth and bounded in
`(0,1)`, and no state-dependent or neural parameters are added.

Each seed uses three disjoint deterministic bank roles. Each role contains 64
generated trajectories and an independently sampled oracle with 256 particles
at each evaluation time. The RNG offsets are `81000`, `82000`, and `83000` for
adaptation, selection, and untouched evaluation; all fifteen recorded bank
fingerprints are role-distinct. Adaptation minimizes interior weighted
median-RBF MMD² on only the three measured `Phi` values at `0.25, 0.50, 0.75`.
The final radial-plus-q4 law feature and q4 itself are never evaluated by the
optimizer or candidate selector. Adam runs for 40 steps at learning rate
`0.04`; the fixed candidates at steps `0,5,...,40` are evaluated once on the
selection bank, including the exact frozen baseline at step zero. The selected
candidate is then evaluated once on the untouched bank against raw SI, tangent,
and frozen neural rollouts using identical particles and oracle samples.

The restored schedules, gates, selected parameters, and amplitudes were:

| seed | frozen schedule raw | gate | selected step | selected alpha | a(0), a(.25), a(.5), a(.75), a(1) |
|---:|---|---:|---:|---|---|
| 401 | `[-1.05488348, 0, 0]` | 0.53962691 | 40 | `[-1.26969417, -1.36359399, -1.39916482]` | `[0.07766904, 0.07515884, 0.56285093, 0.57158269, 0.07766904]` |
| 402 | `[-0.82104778, 0.16889917, 0.01738027]` | 0.67990922 | 40 | `[-1.41094248, -1.46022826, -1.44708283]` | `[0.10737721, 0.10864368, 0.69053901, 0.68772288, 0.10737721]` |
| 403 | `[-0.90906500, 0.13898768, 0.03180078]` | 0.78580193 | 40 | `[-0.29173878, -1.45897299, -1.60210980]` | `[0.38914487, 0.35570730, 0.92179850, 0.93151354, 0.38914487]` |
| 404 | `[-0.90164719, 0.10455292, -0.01122837]` | 0.60255511 | 40 | `[1.42313070, 0.75022799, -0.87796466]` | `[0.93018155, 0.72337978, 0.74819818, 0.93803504, 0.93018155]` |
| 405 | `[-0.95929780, 0, 0]` | 0.76010873 | 35 | `[-0.63131950, 1.35861310, -1.29403764]` | `[0.86767651, 0.31602903, 0.30223935, 0.86008473, 0.86767651]` |

The mean selected alpha was `[-0.43611285, -0.43479083, -1.32407195]`; the
mean amplitudes at the five displayed times were
`[0.47440984, 0.31578373, 0.64512519, 0.79778777, 0.47440984]`.

#### Adaptation and selection objectives

| seed | adaptation initial | adaptation selected | selection initial | selection selected |
|---:|---:|---:|---:|---:|
| 401 | 0.05863152 | 0.04528355 | 0.02561536 | 0.01838422 |
| 402 | 0.05036908 | 0.03533874 | 0.05732235 | 0.04340959 |
| 403 | 0.02191169 | 0.02068902 | 0.01190874 | 0.01079539 |
| 404 | 0.02513193 | 0.02143046 | 0.03198214 | 0.02652568 |
| 405 | 0.01821606 | 0.01360663 | 0.00626879 | 0.00348695 |
| mean (95% CI) | 0.03485206 (0.01208775, 0.05761637) | 0.02726968 (0.01139919, 0.04314017) | 0.02661948 (0.00176700, 0.05147196) | 0.02052037 (0.00139560, 0.03964513) |

The mean selected step was `39.0` with interval `(36.224, 41.776)`; four seeds
selected step 40 and seed 405 selected step 35. Every selected candidate lowered
both its adaptation-bank and selection-bank objective relative to alpha zero.

#### Untouched evaluation results

The primary `interior law MMD²` is the unchanged final MMD on eight radial
descriptors plus held-out q4, integrated over `0.25, 0.50, 0.75`. `Integrated
law MMD²` additionally includes the reported endpoint at `t=1`. Values are
five-seed means with 95% t intervals.

| method | interior law MMD² | integrated law MMD² | endpoint MMD² | maximum moment error | interior Phi MMD² | q4 change |
|---|---:|---:|---:|---:|---:|---:|
| raw SI | 0.01914905 (0.01098884, 0.02730927) | 0.02584914 (0.01719983, 0.03449846) | 0.01485990 (-0.00132650, 0.03104630) | 0.02683098 (0.01743078, 0.03623117) | 0.03350829 (0.01420223, 0.05281434) | 0.60280942 (0.57917483, 0.62644400) |
| tangent | **0.00980376** (0.00453648, 0.01507105) | 0.02637707 (0.01101505, 0.04173908) | 0.09098961 (0.03314209, 0.14883714) | **0.00167379** (0.00105651, 0.00229107) | **0.00987402** (0.00416023, 0.01558781) | 0.56233342 (0.54354022, 0.58112663) |
| frozen neural | 0.01968066 (0.00975290, 0.02960841) | 0.02890349 (0.01848875, 0.03931823) | 0.03414446 (0.01104101, 0.05724790) | 0.02493123 (0.01342103, 0.03644142) | 0.03155668 (0.01083869, 0.05227467) | 0.60587740 (0.58347166, 0.62828313) |
| rollout-adapted | 0.01602761 (0.00776599, 0.02428922) | **0.02364048** (0.01685750, 0.03042346) | 0.02888677 (-0.00690430, 0.06467784) | 0.02324640 (0.01415000, 0.03234281) | 0.02476925 (0.00885448, 0.04068402) | 0.60567336 (0.58580674, 0.62553998) |

Five-seed mean path values at each unchanged evaluation time were:

| method | MMD² .25 | MMD² .50 | MMD² .75 | MMD² 1.0 | moment error .25 | moment error .50 | moment error .75 | moment error 1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw SI | 0.03181126 | 0.04132016 | 0.03874084 | 0.01485990 | 0.02024833 | 0.02315727 | 0.02683098 | 0.00238356 |
| tangent | 0.01586412 | 0.01048458 | 0.04159682 | 0.09098961 | 0.00165465 | 0.00163999 | 0.00164680 | 0.00166042 |
| frozen neural | 0.03128508 | 0.04326097 | 0.03963823 | 0.03414446 | 0.01984766 | 0.02311119 | 0.02355032 | 0.01029990 |
| rollout-adapted | 0.02536421 | 0.03542021 | 0.03201621 | 0.02888677 | 0.01737063 | 0.02013459 | 0.02045794 | 0.00763132 |

| method | q4 .25 | q4 .50 | q4 .75 | q4 1.0 |
|---|---:|---:|---:|---:|
| raw SI | 0.16187317 | 0.23677430 | 0.40999435 | 0.76468258 |
| tangent | 0.16332127 | 0.23092082 | 0.38567999 | 0.72565470 |
| frozen neural | 0.15963565 | 0.23085568 | 0.40098408 | 0.76551305 |
| rollout-adapted | 0.16056222 | 0.23229761 | 0.40126000 | 0.76623558 |
| oracle | 0.16681301 | 0.24077210 | 0.42303026 | 0.77346496 |

Per-seed primary interior law MMD² values were:

| seed | raw SI | tangent | frozen neural | rollout-adapted |
|---:|---:|---:|---:|---:|
| 401 | 0.01914357 | 0.01260773 | 0.02804496 | 0.02044941 |
| 402 | 0.01479483 | 0.00357138 | 0.02079690 | 0.01484126 |
| 403 | 0.01592005 | 0.01434898 | 0.01774453 | 0.01716977 |
| 404 | 0.03051441 | 0.01066273 | 0.02463715 | 0.02236217 |
| 405 | 0.01537240 | 0.00782799 | 0.00717975 | 0.00531543 |

All paired contrasts below are left method minus right method:

| contrast | interior law MMD² | integrated law MMD² | endpoint MMD² | maximum moment error | interior Phi MMD² | q4 change |
|---|---:|---:|---:|---:|---:|---:|
| adapted - frozen | -0.00365305 (-0.00734731, **0.00004121**) | -0.00526301 (-0.01243602, 0.00190999) | -0.00525769 (-0.02997796, 0.01946258) | -0.00168482 (-0.00663284, 0.00326320) | **-0.00678743 (-0.01352129, -0.00005358)** | -0.00020404 (-0.00472932, 0.00432124) |
| adapted - tangent | 0.00622384 (-0.00127566, 0.01372335) | -0.00273659 (-0.01477697, 0.00930380) | **-0.06210285 (-0.11686270, -0.00734299)** | **0.02157261 (0.01294964, 0.03019559)** | **0.01489523 (0.00002053, 0.02976993)** | **0.04333994 (0.02216806, 0.06451181)** |
| adapted - raw | -0.00312144 (-0.00998198, 0.00373909) | -0.00220866 (-0.00898681, 0.00456948) | 0.01402687 (-0.01310475, 0.04115849) | -0.00358457 (-0.00811529, 0.00094615) | -0.00873904 (-0.02049191, 0.00301383) | 0.00286394 (-0.00615897, 0.01188685) |
| frozen - tangent | 0.00987689 (-0.00002560, 0.01977939) | 0.00252643 (-0.01538032, 0.02043317) | -0.05684516 (-0.11995511, 0.00626480) | **0.02325744 (0.01221890, 0.03429597)** | **0.02168266 (0.00144373, 0.04192160)** | **0.04354397 (0.02334993, 0.06373802)** |

The adapted-minus-frozen final-law effect is favorable in mean but its upper
interval endpoint is `4.12e-5`, just above zero. The experiment therefore does
not establish a replicated improvement in the final law-valued metric. It does
establish improvement in the held-out measured-Phi rollout target, whose paired
interval is entirely below zero. This is useful evidence that trajectory-level
differentiation changes the frozen field in the intended direction, but that
the measured-Phi surrogate still does not fully resolve global law mismatch.
The adapted method also does not beat tangent: its mean interior law MMD² is
`0.00622384` higher, with an interval crossing zero, and its moment error and
measured-Phi MMD are significantly higher than tangent.

#### Numerical validation and timing

- differentiated-rollout directional derivative: autodiff
  `0.0021100991006`, central finite difference `0.0021100990968`, relative
  error `1.78e-9` at step `1e-4`;
- functional JAX Heun versus the established Python Heun loop: maximum absolute
  trajectory error `8.88e-16` across five evaluation banks;
- static-shape differentiable weighted MMD versus the established MMD:
  absolute error `0.0`;
- all frozen neural hashes unchanged; all bank roles distinct; q4 and final
  evaluation were absent from optimization and selection;
- total wall time recorded by the full command: `125.11 s`; per-seed optimizer
  wall times were `9.054, 6.341, 6.231, 6.382, 6.310 s` (the first includes
  more compilation); every evaluated method used 48 NFE. Per-seed rollout wall
  times are retained in the machine-readable summary because compilation makes
  them unsuitable for a scientific ranking.

Primary artifacts:

- [`results/stage3_rollout_adaptation/summary.json`](results/stage3_rollout_adaptation/summary.json)
- [`results/stage3_rollout_adaptation/stage3_metrics.csv`](results/stage3_rollout_adaptation/stage3_metrics.csv)
- [`results/stage3_rollout_adaptation/REPORT.md`](results/stage3_rollout_adaptation/REPORT.md)
- [`results/stage3_rollout_adaptation/stage3_summary.png`](results/stage3_rollout_adaptation/stage3_summary.png)
- [`results/stage3_rollout_adaptation/stage3_optimization.png`](results/stage3_rollout_adaptation/stage3_optimization.png)
- [`results/stage3_rollout_adaptation/stage3_paths.png`](results/stage3_rollout_adaptation/stage3_paths.png)

### 8.10 Stage 3B: confirmatory replication and credit-assignment controls

#### Predeclaration and unchanged protocol

Stage 3B was predeclared in [`stage3b_protocol.json`](stage3b_protocol.json)
before executing any of the new seeds. It uses ten new scientific seeds
`406–415`; none overlaps the original Stage 3 seeds `401–405`. The primary
confirmatory estimand is full-rollout-adapted minus frozen-neural interior
radial-plus-q4 law MMD² on untouched evaluation banks. The predeclared
confirmation rule is an upper 95% paired-t interval endpoint below zero.

There were no changes to the original full-rollout method:

- the same three-parameter bounded harmonic modulation and alpha-zero frozen
  baseline;
- Adam, learning rate `0.04`, 40 steps, gradient clipping, and checkpoints
  `0,5,...,40` selected on the independent selection bank;
- adaptation/selection/evaluation offsets `91000/92000/93000`, with 64
  generated trajectories and 256 oracle particles per time in each role;
- the three measured-Phi weighted median-RBF interior MMD² adaptation and
  selection loss, with q4 and the final law MMD kept hidden;
- 24-step Heun, 48 NFE, and evaluation times
  `0.25, 0.50, 0.75, 1.00`;
- the endpoint construction, calibration, selected schedule, reference field,
  invariant MLP architecture, 18-time × 64-particle random-continuous-time
  training, 420 Deep-Ritz steps, and held-out scalar gate procedure.

The ten base objects were reconstructed by
[`prepare_stage3b_base_models.py`](prepare_stage3b_base_models.py), which runs
the unchanged paper pipeline exactly through schedule selection, random-time
MLP training, and gating. It skips only unconsumed fixed-grid, angular-network,
deployment, and diagnostic rollouts. A seed-406 comparison against the full
runner gave a schedule difference below `6.4e-15`, gate difference `2.5e-6`,
and model-parameter relative difference `3.64e-7`, consistent with numerical
GPU rerun variation. Base preparation took `178.20 s`; Stage 3B took `422.77 s`.

Two controls were added on the same new bank triples:

1. **Scalar full-rollout adaptation:** optimize only `alpha0`, with
   `alpha1=alpha2=0`, so `a(t)` is constant while retaining full trajectory
   differentiation.
2. **Stopped-state three-parameter adaptation:** use the identical
   time-dependent forward rollout, but apply `stop_gradient` to the incoming
   state and Heun proposal at every step. The direct dependence on the two
   current-step field calls remains, while accumulated state-to-state temporal
   credit is removed.

Across every seed, full and stopped-state forward trajectories agreed exactly
at the gradient probe (`0.0` maximum error), while the full-versus-stopped
gradient-difference norm ranged from `0.00048913` to `0.00592903` with mean
`0.00280268`. Thus this is a gradient-only control, not a changed simulator.
All frozen MLP hashes remained unchanged, all thirty bank-role fingerprints
were distinct within their seed, and no q4 or evaluation-bank value entered
adaptation or selection.

The new schedules and frozen gates were:

| seed | selected schedule raw | frozen gate |
|---:|---|---:|
| 406 | `[-0.76793131, 0.20624515, -0.02002081]` | 0.49972780 |
| 407 | `[-0.80991355, 0.17476388, 0.00089047]` | 0.68215865 |
| 408 | `[-0.76829742, 0.20404200, 0.00706254]` | 0.66565714 |
| 409 | `[-0.88245860, 0.17071452, -0.00618999]` | 0.75370248 |
| 410 | `[-0.82947210, 0.17352714, 0.00819035]` | 0.83047033 |
| 411 | `[-1.07051702, 0, 0]` | 0.55456209 |
| 412 | `[-1.11119563, 0.03500000, 0.03500000]` | 0.73335786 |
| 413 | `[-0.63393418, 0.25961182, 0.03494845]` | 0.37198000 |
| 414 | `[-1.03398873, 0, 0]` | 0.68072177 |
| 415 | `[-1.03528849, 0.03500000, 0.03500000]` | 0.68235628 |

#### Optimization behavior

Values below are ten-seed means with 95% intervals. All three controls start
from the same alpha-zero loss on each seed.

| control | adaptation initial | adaptation selected | selection initial | selection selected | mean selected checkpoint |
|---|---:|---:|---:|---:|---:|
| scalar full-gradient | 0.02257817 (0.01552354, 0.02963280) | 0.02092838 (0.01404510, 0.02781166) | 0.01990090 (0.01368739, 0.02611441) | 0.01860524 (0.01281843, 0.02439204) | 36.0 (26.952, 45.048) |
| stopped-state 3-param | 0.02257817 (0.01552354, 0.02963280) | 0.02086301 (0.01369045, 0.02803558) | 0.01990090 (0.01368739, 0.02611441) | 0.01837760 (0.01241883, 0.02433637) | 25.5 (11.756, 39.244) |
| full-rollout 3-param | 0.02257817 (0.01552354, 0.02963280) | **0.01917982** (0.01177787, 0.02658176) | 0.01990090 (0.01368739, 0.02611441) | **0.01669030** (0.01130776, 0.02207284) | 40.0 (40.0, 40.0) |

Mean selected full-alpha vectors were `[0.55021565, 0, 0]` for scalar,
`[0.75424258, -0.78713415, -0.32824718]` for stopped-state, and
`[0.57638199, 0.08688137, -0.45554226]` for full rollout. Mean amplitudes at
`t=(0,.25,.5,.75,1)` were respectively:

- scalar: `[0.70498660, 0.70498660, 0.70498660, 0.70498660, 0.70498660]`;
- stopped-state: `[0.63713923, 0.68163471, 0.85053519, 0.79941787, 0.63713923]`;
- full rollout: `[0.72346570, 0.58529805, 0.70722257, 0.79534595, 0.72346570]`.

#### Confirmatory untouched-bank results

| method | interior law MMD² | integrated law MMD² | endpoint MMD² | maximum moment error | interior Phi MMD² | q4 change |
|---|---:|---:|---:|---:|---:|---:|
| raw SI | 0.01616719 (0.01249643, 0.01983795) | 0.02105137 (0.01663573, 0.02546700) | 0.00599468 (0.00363871, 0.00835064) | 0.02380353 (0.01991622, 0.02769083) | 0.02480190 (0.02046183, 0.02914198) | 0.60753661 (0.59508819, 0.61998503) |
| tangent | **0.00867894** (0.00447191, 0.01288597) | 0.02001195 (0.01161057, 0.02841334) | 0.05658391 (0.03354898, 0.07961884) | **0.00207751** (0.00140184, 0.00275318) | **0.00690188** (0.00310673, 0.01069704) | 0.57037496 (0.55681256, 0.58393736) |
| frozen neural | 0.01396672 (0.01035677, 0.01757668) | 0.01907115 (0.01469433, 0.02344796) | 0.01353036 (0.00656067, 0.02050006) | 0.01928118 (0.01649331, 0.02206905) | 0.02013301 (0.01542044, 0.02484558) | 0.61012911 (0.59821664, 0.62204158) |
| scalar adapted | 0.01343881 (0.00972975, 0.01714787) | 0.01883303 (0.01423074, 0.02343532) | 0.01449500 (0.00696624, 0.02202375) | 0.01832717 (0.01567054, 0.02098381) | 0.01852957 (0.01402195, 0.02303719) | 0.60966924 (0.59774194, 0.62159654) |
| stopped-state adapted | 0.01332811 (0.00956716, 0.01708907) | 0.01868846 (0.01427941, 0.02309752) | 0.01755738 (0.00914102, 0.02597374) | 0.01842250 (0.01586007, 0.02098492) | 0.01844929 (0.01410786, 0.02279073) | 0.60898064 (0.59750130, 0.62045998) |
| **full rollout adapted** | **0.01238723** (0.00869682, 0.01607764) | **0.01781153** (0.01342017, 0.02220289) | 0.01886952 (0.00873614, 0.02900290) | **0.01771588** (0.01454453, 0.02088723) | **0.01655039** (0.01224662, 0.02085415) | 0.61056591 (0.59849046, 0.62264137) |

Per-seed primary MMDs and the three prespecified paired effects were:

| seed | frozen | scalar | stopped | full | full-frozen | full-scalar | full-stopped |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 406 | 0.01749975 | 0.01749975 | 0.01696997 | 0.01623942 | -0.00126032 | -0.00126032 | -0.00073054 |
| 407 | 0.01235737 | 0.01138773 | 0.01235737 | 0.01010975 | -0.00224763 | -0.00127798 | -0.00224763 |
| 408 | 0.01763463 | 0.01573210 | 0.01528886 | 0.01240788 | -0.00522675 | -0.00332422 | -0.00288098 |
| 409 | 0.00842658 | 0.00777649 | 0.00828588 | 0.00768465 | -0.00074193 | -0.00009184 | -0.00060123 |
| 410 | 0.02013874 | 0.02034301 | 0.02046835 | 0.01931914 | -0.00081960 | -0.00102387 | -0.00114921 |
| 411 | 0.01070429 | 0.01142380 | 0.01070429 | 0.01015848 | -0.00054581 | -0.00126531 | -0.00054581 |
| 412 | 0.01369390 | 0.01311780 | 0.01298332 | 0.01298889 | -0.00070500 | -0.00012891 | 0.00000558 |
| 413 | 0.00918308 | 0.00745046 | 0.00636600 | 0.00569125 | -0.00349183 | -0.00175920 | -0.00067475 |
| 414 | 0.00808210 | 0.00803108 | 0.00808210 | 0.00796257 | -0.00011954 | -0.00006851 | -0.00011954 |
| 415 | 0.02194679 | 0.02162593 | 0.02177501 | 0.02131026 | -0.00063652 | -0.00031566 | -0.00046474 |

The complete paired contrast table is:

| contrast | interior law MMD² | integrated law MMD² | endpoint MMD² | maximum moment error | interior Phi MMD² | q4 change |
|---|---:|---:|---:|---:|---:|---:|
| full - frozen | **-0.00157949 (-0.00273881, -0.00042018)** | -0.00125962 (-0.00286942, 0.00035018) | 0.00533916 (0.00164107, 0.00903724) | -0.00156530 (-0.00322180, 0.00009119) | **-0.00358262 (-0.00596396, -0.00120129)** | 0.00043680 (-0.00230426, 0.00317787) |
| full - scalar | **-0.00105158 (-0.00177035, -0.00033281)** | -0.00102150 (-0.00221775, 0.00017475) | 0.00437452 (0.00035724, 0.00839181) | -0.00061130 (-0.00218215, 0.00095956) | **-0.00197918 (-0.00350871, -0.00044966)** | 0.00089668 (-0.00133090, 0.00312425) |
| full - stopped-state | **-0.00094088 (-0.00160246, -0.00027931)** | -0.00087693 (-0.00200030, 0.00024644) | 0.00131214 (-0.00545600, 0.00808028) | -0.00070662 (-0.00203934, 0.00062611) | **-0.00189891 (-0.00340648, -0.00039133)** | 0.00158527 (-0.00028628, 0.00345683) |
| scalar - frozen | -0.00052791 (-0.00112134, 0.00006552) | -0.00023812 (-0.00121285, 0.00073661) | 0.00096463 (-0.00148069, 0.00340995) | **-0.00095401 (-0.00165921, -0.00024880)** | **-0.00160344 (-0.00265436, -0.00055251)** | -0.00045987 (-0.00179013, 0.00087038) |
| stopped-state - frozen | -0.00063861 (-0.00140398, 0.00012676) | -0.00038268 (-0.00103160, 0.00026623) | 0.00402702 (-0.00117407, 0.00922811) | **-0.00085868 (-0.00160979, -0.00010758)** | **-0.00168372 (-0.00316011, -0.00020732)** | -0.00114847 (-0.00259798, 0.00030104) |
| full - tangent | 0.00370829 (-0.00018266, 0.00759923) | -0.00220042 (-0.01019801, 0.00579716) | **-0.03771439 (-0.06524780, -0.01018098)** | **0.01563837 (0.01216110, 0.01911564)** | **0.00964850 (0.00370403, 0.01559297)** | **0.04019095 (0.03337834, 0.04700357)** |

#### Answers to the three prespecified questions

1. **Adapted vs frozen:** confirmed. Full rollout adaptation improved primary
   MMD on all `10/10` new seeds. The mean effect was `-0.00157949`, with 95%
   interval `(-0.00273881, -0.00042018)`.
2. **Time-dependent vs scalar:** supported. Full time-dependent adaptation was
   better on all `10/10` new seeds. The mean effect was `-0.00105158`, interval
   `(-0.00177035, -0.00033281)`.
3. **Full gradient vs stopped-state:** supported. Full credit assignment was
   better on `9/10` seeds and worse by only `5.58e-6` on seed 412. The mean
   effect was `-0.00094088`, interval
   `(-0.00160246, -0.00027931)`.

The original five and confirmatory ten are not used interchangeably for the
confirmatory decision. As a secondary descriptive estimate across all 15
independent model seeds, full minus frozen interior MMD² was `-0.00227068`,
with interval `(-0.00342938, -0.00111198)`. The confirmatory magnitude is
smaller than the original five-seed estimate (`-0.00158` versus `-0.00365`) but
has the same direction and a resolved interval. This supports the stronger
claim that both temporal modulation and full trajectory credit assignment
contribute to improving the frozen correction. It still does not establish
superiority to tangent: tangent has lower mean interior MMD and much lower
moment error, while the full-minus-tangent MMD interval crosses zero.

Primary artifacts:

- [`stage3b_protocol.json`](stage3b_protocol.json)
- [`results/stage3b_base_models/summary.json`](results/stage3b_base_models/summary.json)
- [`results/stage3b_confirmatory/summary.json`](results/stage3b_confirmatory/summary.json)
- [`results/stage3b_confirmatory/stage3b_metrics.csv`](results/stage3b_confirmatory/stage3b_metrics.csv)
- [`results/stage3b_confirmatory/REPORT.md`](results/stage3b_confirmatory/REPORT.md)
- [`results/stage3b_confirmatory/stage3b_summary.png`](results/stage3b_confirmatory/stage3b_summary.png)
- [`results/stage3b_confirmatory/stage3b_contrasts.png`](results/stage3b_confirmatory/stage3b_contrasts.png)
- [`results/stage3b_confirmatory/stage3b_selection.png`](results/stage3b_confirmatory/stage3b_selection.png)

### 8.11 Stage 4: differentiable moment-fiber design

Stage 4 isolates the choice of the three measured observables. For every
paper-facing seed `401–405`, the physical endpoint configurations and weights,
hidden q4 gap, deterministic angular-sort coupling, selected reference
schedule, construction times, calibration equations, and eight-function Ritz
realization dictionary were fixed. No rollout, schedule optimization, coupling
adaptation, or neural training was performed.

The learnable object is a rank-three subspace of an eleven-function radial RBF
dictionary that nests the original hand observables. The coefficients are
row-orthonormal, preventing scale or rank collapse, and lie in the nullspace of
the fixed weighted endpoint dictionary-mean difference. Thus every candidate
defines exactly three observables and keeps the two original endpoint laws in
the same equivalence class. q4 and all angular descriptors are excluded from
adaptation and selection.

Each seed used disjoint adaptation, checkpoint-selection, and untouched
evaluation banks. Step zero was the exact hand-observable span, so selection
could retain the control. The primary construction objective was the existing
integrated Ritz correction energy plus `0.02` times forcing power and the ESS
floor penalty. Calibration was iterated to convergence for both arms; maximum
evaluation residuals were at numerical precision. A directional derivative
check through calibration and the full fiber construction had relative error
`2.58e-8`.

Untouched-bank five-seed results were:

| fiber | construction objective | correction energy | forcing power | minimum ESS |
|---|---:|---:|---:|---:|
| hand | 0.64082 | 0.22281 | 20.6301 | 0.16402 |
| differentiably designed | **0.31943** | **0.12300** | **9.8216** | **0.36946** |

Designed checkpoints improved the primary evaluation objective on seeds 401,
402, 404, and 405; seed 403 was slightly worse. The paired designed-minus-hand
effect was `-0.32139`, with 95% interval `(-0.65720, 0.01441)`. Correction
energy and forcing power also improved in mean, but their paired intervals
crossed zero. Therefore differentiation found substantially better mean fibers,
but this five-seed experiment does **not** establish that the designed
three-observable equivalence class is better than the hand class.

Primary artifacts:

- [`stage4_protocol.json`](stage4_protocol.json)
- [`results/stage4_fiber_design/summary.json`](results/stage4_fiber_design/summary.json)
- [`results/stage4_fiber_design/stage4_metrics.csv`](results/stage4_fiber_design/stage4_metrics.csv)
- [`results/stage4_fiber_design/REPORT.md`](results/stage4_fiber_design/REPORT.md)
- [`results/stage4_fiber_design/stage4_summary.png`](results/stage4_fiber_design/stage4_summary.png)
- [`results/stage4_fiber_design/stage4_selection_q4.png`](results/stage4_fiber_design/stage4_selection_q4.png)

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
- Matched random-time training reduces mean reference/off-grid Ritz degradation
  by 94.2% while keeping the off-grid gain interval above zero; this passes the
  predefined rollout-adaptation-readiness criterion independently of tangent
  MMD.
- Stage 3 rollout differentiation lowers held-out measured-Phi MMD relative to
  the frozen correction by `-0.00678743`, with interval
  `(-0.01352129, -0.00005358)`, while freezing the MLP and every established
  transport component.
- Stage 3B confirms a final-law improvement on ten new seeds: full-rollout
  minus frozen MMD² is `-0.00157949` with interval
  `(-0.00273881, -0.00042018)`.
- Stage 3B supports nonconstant temporal structure: full time-dependent minus
  scalar rollout adaptation is `-0.00105158` with interval
  `(-0.00177035, -0.00033281)`.
- Stage 3B supports full temporal credit assignment: full-gradient minus
  identical-forward stopped-state MMD² is `-0.00094088` with interval
  `(-0.00160246, -0.00027931)`.

Not yet supported strongly enough:

- A Stage-4 advantage of the differentiably designed three-observable fiber:
  its mean construction objective is substantially lower, but the paired
  five-seed interval `(-0.65720, 0.01441)` narrowly includes zero.
- A statistically significant advantage of the three-parameter schedule over
  the optimized scalar schedule; its paired five-bank interval includes zero.
- An end-to-end N=32 neural MMD advantage; its paired interval is centered near
  zero.
- An N=32 neural advantage over tangent; neural-minus-tangent is positive with
  a 95% interval entirely above zero for the fixed-grid model, while the
  random-time model's interval includes zero but does not establish an
  advantage.
- Superiority of rollout-adapted MFSI to tangent: on the ten new seeds,
  full-minus-tangent interior MMD² is `0.00370829` with interval
  `(-0.00018266, 0.00759923)`, and tangent retains substantially lower moment
  error.
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

Stage 3 frozen-correction rollout adaptation:

```bash
./scripts/run_stage3_rollout_adaptation.sh
./scripts/run_stage3_rollout_adaptation.sh --aggregate-existing
./.venv/bin/python validate_stage3_rollout_adaptation.py
```

Stage 3B predeclared confirmation:

```bash
./scripts/prepare_stage3b_base_models.sh
./scripts/run_stage3b_confirmatory.sh
./scripts/run_stage3b_confirmatory.sh --aggregate-existing
./.venv/bin/python validate_stage3b_confirmatory.py
```

Stage 4 differentiable observable design:

```bash
./scripts/run_stage4_fiber_design.sh
./scripts/run_stage4_fiber_design.sh --aggregate-existing
./.venv/bin/python validate_stage4_fiber_design.py
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
- the positive held-out full-MLP Ritz gain;
- the matched random-time reduction in off-grid Ritz degradation; and
- the Stage 3 held-out measured-Phi rollout-MMD improvement from differentiable
  temporal modulation; and
- the ten-seed Stage 3B confirmation of improved final-law MMD over the frozen
  neural correction;
- the Stage 3B evidence that time-dependent modulation outperforms scalar
  rollout adaptation and that full temporal credit assignment outperforms an
  identical-forward stopped-state gradient; and
- the paired improvement of the N=32 tangent correction on interior MMD.

Stage 3B resolves the original five-seed uncertainty without changing the
method: the final-law improvement over the frozen neural correction replicates
on ten new model seeds. The two predeclared controls also supply the missing
mechanistic evidence. Nonconstant temporal structure and full trajectory
credit assignment each improve the primary law metric relative to their
matched controls. The effect is real but modest, and tangent remains the
stronger mean end-to-end comparator; no claim of tangent superiority reversal
is supported.
