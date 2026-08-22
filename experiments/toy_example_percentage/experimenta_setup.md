# Toy percentage-risk Pareto experiment

## Authoritative experiment setup, results, and paper record

This document records the corrected production Pareto experiment under
`outputs/pareto/`. It follows the same reporting structure as the active-nematic
and vortices experiment records and is intended to support a future Methods
section, supplement, results table, reproducibility statement, or figure
caption.

The active scientific result is the corrected positive-support physical-`q_h`
Full sweep. The pre-correction/mixed artifacts under
`outputs/old/pareto_pre_corrected_full/` are retained only as provenance and
must not be presented as the current result.

## 1. Experiment identity and status

| Item | Value |
|:---|:---|
| Experiment | `toy_example_percentage` |
| Source configuration | `toy_example` |
| Base seed | 20260813 |
| State domain | `[-3.2,3.2]^2` |
| Allowances | 0.5%, 1%, 2%, 3%, 4%, 5% |
| Production output | `outputs/pareto/` |
| Scientific Full evaluator | Positive-support physical-`q_h`, direct signed source |
| Scientific grid/time nodes | 101 x 101; all 21 times |
| Frozen scientific bandwidth | 0.417530106552 |
| Selection trials | 64 |
| Validation trials | 128, independent |
| Full selection curve nested | Yes |
| All certificates | Pass |
| Common-raster decomposition | Pass |
| Frozen inputs unchanged | Yes |
| Final status | **PASS** |

The `smoke` block in `config.json` was not active for the production result.
The experiment asks where two localized sensors should be placed when a
controlled increase in finite-data law risk may be exchanged for lower
measurement-conditioned transport action.

The 4% allowance is the first tested point on the final corrected Full-action
plateau. Its held-out Full action is 19.013149 versus 29.013980 for Law, a
34.47% ratio-of-means reduction. The same geometry remains the 5% winner.

## 2. Compared design rules

The design consists of two sensor angles on a radius-1.5 ring:

```text
eta = [theta_1, theta_2].
```

Angles are canonicalized and sorted. Their projective separation must be at
least 20 degrees.

- **Population** minimizes exact analytic population loss and defines the
  absolute population screen.
- **Law** minimizes finite noisy law risk and supplies one common anchor for
  all percentage allowances.
- **Tangent** minimizes particle-space tangent action under both screens.
- **Full** minimizes corrected physical-density weighted-Poisson action under
  both screens and is the primary Pareto rule.

Selection uses only the frozen selection bank. Validation is evaluated only
after the winner has been frozen and never participates in selection.

## 3. Analytic hidden population

The state is `x=(x_1,x_2)` on `[-3.2,3.2]^2`. Define

```text
d(alpha) = (cos(alpha), sin(alpha)),

g_alpha(x) = 1/2 N(x; 1.5 d(alpha), 0.3^2 I)
           + 1/2 N(x;-1.5 d(alpha), 0.3^2 I).
```

For normalized time `t in [0,1]`, the analytic path is

```text
rho_t^alpha = (1-t)^2 g_0
            + 2t(1-t) g_alpha
            + t^2 g_(pi/2).
```

The nuisance orientation `alpha` is uniformly averaged over 30--60 degrees
using five-point Gauss--Legendre quadrature. The path therefore moves from
horizontal antipodal lobes to vertical antipodal lobes while retaining
uncertain interior curvature.

| Population parameter | Value |
|:---|---:|
| Lobe radius | 1.5 |
| Gaussian sigma | 0.3 |
| Domain half-width | 3.2 |
| Nuisance-angle range | 30--60 degrees |
| Nuisance quadrature nodes | 5 |
| Scientific time nodes | 21 |

## 4. Endpoint-only learned reference

The reference velocity is trained only from the horizontal and vertical
endpoint laws. Its input contains two coordinates and five time features
`(t, sin(pi t), cos(pi t), sin(2 pi t), cos(2 pi t))`. Four hidden layers of
width 128 use SiLU activations and the output is a two-dimensional velocity.

Conditional flow matching uses

```text
x_t = (1-t)x_0 + t x_1 + 0.15 sin(pi t) z,
u_t = -x_0 + x_1 + 0.15 pi cos(pi t) z.
```

| Reference-training hyperparameter | Value |
|:---|:---|
| Seed | 20260813 |
| Hidden width / layers | 128 / 4 |
| Training steps | 12,000 |
| Batch size | 2,048 |
| Optimizer | Adam |
| Adam beta1 / beta2 / epsilon | 0.9 / 0.999 / 1e-8 |
| Initial learning rate | 1e-3 |
| Final learning-rate ratio | 0.05 |
| Schedule | Cosine decay |
| Gradient clipping norm | 10 |
| Bridge schedule / noise | Linear / 0.15 |
| Checkpoint reuse | Enabled after compatibility checks |
| Final logged training loss | 1.371587 |

The reference is integrated by RK4 with 16 substeps per scientific interval.
A tensor Gauss--Hermite rule of order 36 per coordinate and endpoint lobe gives
2,592 deterministic weighted reference particles. Frozen reference arrays have
shapes `[21,2592,2]` for positions and velocities and `[21,2592]` for weights.

## 5. Sensor and observation model

Sensor centers lie on the same radius-1.5 ring as the Gaussian lobes:

```text
s_j = 1.5 (cos(theta_j), sin(theta_j)),

Phi_j(x;theta_j)
    = exp(-||x-s_j||^2 / (2 * 0.45^2)).
```

| Measurement parameter | Value |
|:---|:---|
| Number of sensors | 2 |
| Sensor radius | 1.5 |
| Sensor width | 0.45 |
| Minimum projective separation | 20 degrees |
| Finite particles per acquisition | 100 |
| Acquisition nodes | 11 of 21 |
| Acquisition indices | 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 |
| Detector-noise standard deviation | 0.01 |
| Endpoint moments | Exact |

The selection bank contains 64 trials and the validation bank 128 independent
trials. Their sample-index arrays have shapes `[64,11,100]` and
`[128,11,100]`; detector-noise arrays have shapes `[64,11,2]` and
`[128,11,2]`. Each trial also freezes its nuisance angle and analytic
time-dependent population masses.

## 6. Moment reconstruction and projection

The 11 observations are reconstructed over all 21 time nodes by
endpoint-anchored quadratic generalized least squares.

| Reconstruction hyperparameter | Value |
|:---|---:|
| Relative ridge | 1e-12 |
| Variance floor | 1e-10 |

Reconstructed paths are screened against a common feasible moment polytope and
authoritative candidates receive exact convex-hull checks.

| Feasibility hyperparameter | Value |
|:---|---:|
| Polytope directions | 96 |
| Margin | 1e-6 |
| Feasibility tolerance | 1e-9 |

At each time, empirical information projection tilts the frozen reference
weights:

```text
q_i(lambda) = b_i exp(lambda^T Phi(x_i;eta)) / Z(lambda),
sum_i q_i Phi(x_i;eta) = c(t).
```

| Projection hyperparameter | Value |
|:---|---:|
| Maximum steps | 300 |
| Newton ridge | 1e-7 |
| Step cap | 20 |
| Multiplier clip | 1,000 |
| Search maximum steps | 80 |
| Search residual tolerance | 1e-7 |
| Search line-search steps | 6 |
| Trajectory backend | `tesseract_cpp` |
| Solver acceptance tolerance | 2e-6 |

Particle covariance and Tangent search ridges are 1e-7. Exact covariance and
Tangent ridges are zero, exact covariance minimum eigenvalue is 1e-12,
Tangent pseudoinverse `rcond` is 1e-10, and maximum Tangent compatibility
residual is 1e-7.

## 7. Risk and action definitions

### 7.1 Population and finite-law constraints

Population loss `L` compares projected laws based on exact analytic moments
with the hidden population. Finite-law risk `R` uses Gaussian-kernel MMD with
bandwidth 0.55 at held-out scientific time nodes.

The frozen anchors are

```text
L_star = 0.0687360118546043,
L_max  = L_star + 0.0005 = 0.0692360118546043,
R_star = 0.06618573672099662.
```

At allowance `p`, every candidate must satisfy

```text
L(eta) <= L_max,
R(eta) <= R_star + (p/100) abs(R_star).
```

The source configuration's `law.max_relative_risk_violation = 0.05` is a
default that is overwritten point-by-point by the Pareto driver.

### 7.2 Tangent action

The Tangent objective is the minimum local particle-space correction visible
to the sensor-moment rates. It is the unchanged optimization metric and is not
numerically substituted for the common-raster decomposition reported below.

### 7.3 Corrected Full action

The authoritative evaluator bilinearly deposits physical density `q_h` and
signed source `s_h`. Both are smoothed by the same strictly positive,
full-domain separable Gaussian using per-source-column boundary normalization.
The scientific solve is

```text
-div(q_h grad psi_h) = s_h,
delta*_h = -grad psi_h,
A_full,h = ||delta*_h||^2_(q_h).
```

There is no density floor in the scientific operator. Conductive components
are solved with a gauge constraint and equation-preserving sparse direct solve;
a preconditioned-CG fallback is available.

| Corrected scientific setting | Value |
|:---|:---|
| Raster | 101 x 101 |
| Time nodes | 21 |
| Frozen weighted Scott bandwidth | 0.41753010655193434 |
| Bandwidth scale | 1.0 |
| Density floor | 0 |
| Mass tolerance | 1e-12 |
| Source/component tolerance | 1e-12 |
| Moment/energy tolerance | 1e-6 |

The 51 x 51 grid, `operator_floor_rel = 2e-5`, and related CG settings in the
base configuration belong to the earlier/proxy route. They do not define the
corrected paper values.

## 8. Common-raster Tangent/hidden decomposition

In the same raster vector space, quadrature, density, and inner product as
Full, define

```text
delta_tan,h = argmin ||delta||^2_(q_h) subject to L_h delta = -r_h,
delta_hid,h = delta*_h - delta_tan,h.
```

The audit reports `A_tan,h`, `A_hid,h`,
`Gamma_h=A_hid,h/A_full,h`, and their weighted inner product. It checks Full
and Tangent moment feasibility, hidden-nullspace membership, orthogonality,
Pythagorean identity, and raw hierarchy without clipping.

## 9. Corrected selection protocol

Population, Law, Tangent, the endpoint reference, reconstruction, projection,
and observation banks were frozen before the corrected Full follow-up. For
each allowance, the previous tighter corrected Full winner is a mandatory
incumbent and is rechecked against the current exact L and R caps.

Candidate seeds include the incumbent, historical Full, saved Tangent, Law,
previously audited feasible Full candidates, local perturbations, and normal
multistarts. Archived geometries are seed/provenance inputs only and never
contribute saved action values to a fresh evaluation.

New basins are navigated on a 51 x 51 two-trial frozen prefix, promoted to a
12-trial 101 x 101 prescreen, then decided using all 64 frozen selection trials
at 101 x 101. A winner is replaced only if a feasible and fully certified
candidate lowers corrected selection action by more than 1e-6. Independent
128-trial validation begins only after selection is frozen.

Nested-stage candidate totals were:

| Allowance | Candidates | Feasible | Exact promoted | Incumbent replaced |
|---:|---:|---:|---:|:---:|
| 1% | 144 | 119 | 7 | Yes |
| 2% | 198 | 139 | 13 | Yes |
| 3% | 204 | 143 | 19 | Yes |
| 4% | 209 | 147 | 25 | Yes |
| 5% | 212 | 150 | 31 | No |

The isolated corrected 0.5% run was reproduced separately and then used as the
first mandatory incumbent.

## 10. Base optimization hyperparameters

These parameters describe the original Population/Law/Tangent/Full candidate
pipeline. The corrected Full follow-up adds the staged search above.

| Hyperparameter | Population | Law | Tangent | Full proxy |
|:---|---:|---:|---:|---:|
| Steps | 120 | 50 | 40 | 40 |
| Learning rate | 0.02 shared default | 0.015 | 0.012 | 0.01 |
| Starts | -- | 4 | 7 | 7 |
| Gradient trials | -- | 4 | 4 | 4 |
| Exact audit candidates | -- | 8 | 16 | 30 |
| Exact rescore candidates | -- | -- | 8 | 10 |
| Local starts | -- | -- | 12 | 16 |
| Local perturbation | -- | -- | 5 degrees | 6 degrees |

Shared settings are constraint penalty 10,000, start count 12, feasibility
tolerance 1e-6, invalid penalty 10,000, and JIT objectives enabled. Full has
eight additional random starts, 12 prescreen trials, four proxy gradient
trials, seven proxy time nodes, a 41 x 41 configured proxy grid, tolerance
1e-6, and at most 360 CG iterations.

The required minimum exact-valid/finalist counts are 8/6 for Tangent and 12/10
for Full.

## 11. Randomness and numerical validity

| Randomness setting | Value |
|:---|---:|
| Law trials | 32 |
| Action/selection trials | 64 |
| Validation trials | 128 |
| Selection namespace | 8890 |
| Validation namespace | 8891 |
| Bootstrap replicates | 5,000 |

| Validity gate | Threshold |
|:---|---:|
| Maximum finite calibration residual | 1e-3 |
| Minimum ESS fraction | 0.03 |
| Minimum in-domain base mass | 0.995 |
| Tangent lower-bound/decomposition tolerance | 1e-6 |

Corrected global numerical maxima across all final rules, allowances, and both
banks are:

| Quantity | Maximum |
|:---|---:|
| Raster mass error | 4.440892e-16 |
| Signed-source compatibility error | 5.171130e-16 |
| Physical Poisson relative residual | 1.189575e-11 |
| Full moment-rate residual | 6.617265e-14 |
| Tangent moment-rate residual | 8.082546e-16 |
| Hidden-nullspace residual | 6.617078e-14 |
| Absolute orthogonality residual | 1.460755e-13 |
| Absolute Pythagorean residual | 9.094947e-13 |
| Raw hierarchy value | -1.273685e-1 |

Every named selection and validation flag passes. The negative hierarchy
maximum is strict slack and was not clipped.

## 12. Selected sensor geometries

| Allowance | Rule | Sensor angles in degrees |
|---:|:---|:---|
| all | Law | (23.384916, 67.951787) |
| 0.5% | Tangent | (24.439611, 67.280877) |
| 0.5% | Full | (23.086777, 69.060851) |
| 1--5% | Tangent | (25.165658, 64.146445) |
| 1% | Full | (20.791665, 71.674319) |
| 2% | Full | (21.145219, 72.027872) |
| 3% | Full | (21.498772, 72.381426) |
| 4%, 5% | Full | (21.675549, 72.558202) |

The repeated 4--5% Full geometry is intentional: no feasible audited 5%
candidate improved on the 4% incumbent beyond tolerance.

## 13. Selection results

Every Full-action value below uses the corrected 101 x 101 scientific
evaluator on all 64 frozen selection trials.

| Allowance | Rule | Exact L | Exact R | R increase | Full action | Certified |
|---:|:---|---:|---:|---:|---:|:---:|
| 0.5% | Law | .068736012 | .066185737 | 0.000% | 30.201589 | Yes |
| 0.5% | Tangent | .068761244 | .066158993 | -.040% | 31.428738 | Yes |
| 0.5% | Full | .068759371 | .066211886 | .040% | 27.573951 | Yes |
| 1% | Law | .068736012 | .066185737 | 0.000% | 30.201589 | Yes |
| 1% | Tangent | .069002193 | .066373724 | .284% | 38.324836 | Yes |
| 1% | Full | .069174620 | .066698609 | .775% | 21.040259 | Yes |
| 2% | Law | .068736012 | .066185737 | 0.000% | 30.201589 | Yes |
| 2% | Tangent | .069002193 | .066373724 | .284% | 38.324836 | Yes |
| 2% | Full | .069192898 | .066683496 | .752% | 20.322432 | Yes |
| 3% | Law | .068736012 | .066185737 | 0.000% | 30.201589 | Yes |
| 3% | Tangent | .069002193 | .066373724 | .284% | 38.324836 | Yes |
| 3% | Full | .069217482 | .066674573 | .739% | 19.671402 | Yes |
| 4% | Law | .068736012 | .066185737 | 0.000% | 30.201589 | Yes |
| 4% | Tangent | .069002193 | .066373724 | .284% | 38.324836 | Yes |
| 4% | Full | .069232111 | .066672406 | .735% | 19.370319 | Yes |
| 5% | Law | .068736012 | .066185737 | 0.000% | 30.201589 | Yes |
| 5% | Tangent | .069002193 | .066373724 | .284% | 38.324836 | Yes |
| 5% | Full | .069232111 | .066672406 | .735% | 19.370319 | Yes |

The corrected Full consecutive selection-action changes are

```text
[-6.5336911153, -0.7178277686, -0.6510293732,
 -0.3010833072, 0.0].
```

The non-increasing selection gate passes at tolerance 1e-6.

## 14. Independent validation results

The standard error is the ordinary across-trial SE over 128 independent
validation trials. Reduction is the ratio-of-means change from the common Law
geometry.

| Allowance | Rule | Validation R +/- SE | Full action +/- SE | Action reduction vs Law | Valid |
|---:|:---|:---|:---|---:|---:|
| 0.5% | Law | .065816 +/- .000211 | 29.013980 +/- 1.033466 | 0.00% | 100% |
| 0.5% | Tangent | .065805 +/- .000209 | 30.332182 +/- 1.002152 | -4.54% | 100% |
| 0.5% | Full | .065835 +/- .000216 | 26.618646 +/- .856027 | 8.26% | 100% |
| 1% | Law | .065816 +/- .000211 | 29.013980 +/- 1.033466 | 0.00% | 100% |
| 1% | Tangent | .066045 +/- .000208 | 36.576956 +/- 1.537346 | -26.07% | 100% |
| 1% | Full | .066289 +/- .000248 | 20.310305 +/- .633813 | 30.00% | 100% |
| 2% | Law | .065816 +/- .000211 | 29.013980 +/- 1.033466 | 0.00% | 100% |
| 2% | Tangent | .066045 +/- .000208 | 36.576956 +/- 1.537346 | -26.07% | 100% |
| 2% | Full | .066281 +/- .000250 | 19.746996 +/- .542327 | 31.94% | 100% |
| 3% | Law | .065816 +/- .000211 | 29.013980 +/- 1.033466 | 0.00% | 100% |
| 3% | Tangent | .066045 +/- .000208 | 36.576956 +/- 1.537346 | -26.07% | 100% |
| 3% | Full | .066279 +/- .000253 | 19.243082 +/- .461129 | 33.68% | 100% |
| 4% | Law | .065816 +/- .000211 | 29.013980 +/- 1.033466 | 0.00% | 100% |
| 4% | Tangent | .066045 +/- .000208 | 36.576956 +/- 1.537346 | -26.07% | 100% |
| 4% | Full | .066280 +/- .000255 | 19.013149 +/- .425278 | 34.47% | 100% |
| 5% | Law | .065816 +/- .000211 | 29.013980 +/- 1.033466 | 0.00% | 100% |
| 5% | Tangent | .066045 +/- .000208 | 36.576956 +/- 1.537346 | -26.07% | 100% |
| 5% | Full | .066280 +/- .000255 | 19.013149 +/- .425278 | 34.47% | 100% |

The optimized Tangent metric does not predict Full-action ranking in this
example: the saved Tangent geometry is worse than Law on held-out Full action
at 1--5%, while corrected Full is better at every allowance.

## 15. Corrected Full decomposition

| Allowance | Selection Full | `A_tan,h` | `A_hid,h` | `Gamma_h` |
|---:|---:|---:|---:|---:|
| 0.5% | 27.573951 | .719447 | 26.854504 | .973908 |
| 1% | 21.040259 | .878956 | 20.161304 | .958225 |
| 2% | 20.322432 | .876657 | 19.445775 | .956863 |
| 3% | 19.671402 | .874379 | 18.797024 | .955551 |
| 4% | 19.370319 | .873247 | 18.497072 | .954918 |
| 5% | 19.370319 | .873247 | 18.497072 | .954918 |

The supported tangent-invisible fraction is approximately 95.5%--97.4% on the
selection bank. This statement uses the common-raster projection and is not a
subtraction between incompatible particle and raster objectives.

## 16. Paper-facing interpretation

A defensible primary statement is:

> In an analytic uncertain Gaussian-mixture transport problem, two sensors
> selected under a 4% Law-relative risk allowance reduced mean held-out
> physical-density weighted-Poisson action by 34.47% relative to the common Law
> geometry. The 4% point was the first tested point on the corrected Full-action
> plateau, and all population, law, projection, raster, Poisson, and
> common-discretization decomposition certificates passed.

Important qualifications:

- The observed risk increase at the 4% winner is only 0.735%; the allowance is
  an upper bound, not a target.
- The 4--5% plateau is a result of this nested candidate search, not an analytic
  global-optimum proof.
- Tangent's particle-space optimization metric does not provide the held-out
  Full-action ranking here.
- The population path is analytic and low-dimensional; external validity to
  physical systems must come from the vortices and active-nematic studies.
- Only one sensor radius, width, finite-particle count, and detector-noise level
  were evaluated in this authoritative sweep.

## 17. Reproduction

Run from the repository root with 64-bit JAX enabled:

```bash
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

.venv/bin/python experiments/toy_example_percentage/rerun_corrected_full_0p5.py \
  --pareto-dir experiments/toy_example_percentage/outputs/old/pareto_pre_corrected_full \
  --output-dir experiments/toy_example_percentage/outputs/pareto/risk_0p5pct/full_search \
  --grid-n 101 --bandwidth-scale 1.0

.venv/bin/python experiments/toy_example_percentage/run_corrected_nested_full_sweep.py \
  --pareto-dir experiments/toy_example_percentage/outputs/old/pareto_pre_corrected_full \
  --output-dir experiments/toy_example_percentage/outputs/pareto \
  --rerun-0p5-json experiments/toy_example_percentage/outputs/pareto/risk_0p5pct/full_search/corrected_full_rerun.json \
  --search-from-1pct --fresh-evaluations \
  --grid-n 101 --bandwidth-scale 1.0

.venv/bin/python experiments/toy_example_percentage/finalize_authoritative_corrected_pareto.py \
  --seed-pareto experiments/toy_example_percentage/outputs/old/pareto_pre_corrected_full \
  --source-run experiments/toy_example_percentage/outputs/run \
  --output experiments/toy_example_percentage/outputs/pareto \
  --previous-corrected experiments/toy_example_percentage/outputs/old/pareto_pre_corrected_full/corrected_nested_full_sweep/corrected_nested_full_sweep.json \
  --grid-n 101 --bandwidth-scale 1.0
```

All three commands are checkpointed. `--fresh-evaluations` prevents archived
action values from entering the active corrected cache; archived geometries
remain legitimate candidates.

Corrected end-to-end wall time and peak memory were not stored in the current
artifact schema and should be added before computational-cost reporting.

## 18. Software and provenance

| Item | Recorded value |
|:---|:---|
| Repository branch | `main` |
| Repository HEAD during documentation | `5a73a9fe91bc47087f54ba2439738707abdc5a56` |
| Working tree | Modified; source hashes are required in addition to HEAD |
| Python | 3.12.3 |
| NumPy / SciPy | 2.5.1 / 1.18.0 |
| JAX / jaxlib | 0.8.3 / 0.8.3 |
| Observed JAX backend | CPU (`TFRT_CPU_0`) |
| Host | Linux under WSL2, x86_64 |
| CPU | Intel Core Ultra 9 275HX; 24 visible cores |

Key source hashes:

| File | SHA-256 |
|:---|:---|
| `experiment.py` | `35bbce25a92790f152664c8304adcd837e117598213c0c80b821254b50e7dcd3` |
| `run_corrected_nested_full_sweep.py` | `cba9b1c0191184e6cb3f3a7ef7a4a93e4937fb0e17b90da268634d5046cdd3a3` |
| `finalize_authoritative_corrected_pareto.py` | `8963ca4bccc6d45da6fe5860dca25719de8911e1d50ead5e4be7b7e8eca53ba9` |

## 19. Artifact and hash index

| Artifact | Role |
|:---|:---|
| `outputs/pareto/corrected_nested_full_sweep.*` | Full-precision corrected nested sweep |
| `outputs/pareto/pareto_methods_selection.csv` | Population/Law/Tangent/Full selection metrics |
| `outputs/pareto/pareto_methods_validation.csv` | Independent validation metrics |
| `outputs/pareto/positive_raster_decomposition_diagnostics.*` | Common-raster audit |
| `outputs/pareto/validation_trial_summaries.csv` | Per-trial method actions |
| `outputs/pareto/authoritative_run_summary.json` | Global nesting and numerical summary |
| `outputs/pareto/frozen_inputs/manifest.json` | Frozen input hashes/source paths |
| `outputs/pareto/risk_*pct/` | Per-allowance candidates, audits, results, and trials |

| Authoritative artifact | SHA-256 |
|:---|:---|
| Corrected nested sweep JSON | `114df72191c0b519e6e45cf7c574060a47ac6c64201eba7ed7432f2f11fc2c7e` |
| Positive-raster decomposition JSON | `e57928bd779e5873bd9044a601574122d1c38a2d0a008dbdbc4716934f42eed4` |
| Authoritative run summary | `2e29a178e3850ccb35c067006fb565bcb4f5fb845740ab4b6d4e4859034b80db` |

Frozen input hashes are:

| Input | SHA-256 |
|:---|:---|
| Reference checkpoint | `4bca0ff23a4009a86ea92548d908cdff98be7289f28e7273faeb327ba65bff87` |
| Reference bank | `d721d5b7f21bfd389d9fc210252ebed4a4a93ce147b68d6529b430585b31fdef` |
| Selection bank | `a8c21c0a8d7b67b78d87fc73bb992ca9b9a2b0b5477153a7ae11973d694c628a` |
| Validation bank | `6f96e05cd365b37a15b50c2c0448a6d4e55b213717cb94af3c114374c72750bd` |

## 20. Suggested figure caption and future checks

> Corrected percentage-risk Pareto evaluation for two ring-constrained sensors
> in an analytic uncertain Gaussian-mixture transport path. Full designs were
> selected using 64 frozen trials subject to exact population and Law-relative
> finite-risk screens, then evaluated on 128 independent trials. Reported Full
> action uses a positive-support physical-density weighted-Poisson solve on a
> 101 x 101 grid at all 21 time nodes with a frozen bandwidth. The 4% allowance
> is the first point on the corrected Full-action plateau and reduces mean
> held-out action by 34.47% relative to Law. Common-raster decomposition assigns
> approximately 95.5%--97.4% of selected Full action to the tangent-invisible
> component.

Before a confirmatory paper run, preregister 4% as the primary operating point,
regenerate independent selection and validation banks, store exact runtime and
memory data, commit the source revision, and consider sensitivity sweeps over
particle count, sensor width, detector noise, and nuisance-angle quadrature.
