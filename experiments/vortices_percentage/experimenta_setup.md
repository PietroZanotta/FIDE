# Vortices percentage-risk Pareto experiment

## Authoritative experiment setup, results, and paper record

This document records the corrected production Pareto experiment under
`outputs/pareto/`. It follows the same reporting structure as the active-nematic
experiment record and is intended to support a future Methods section,
supplement, results table, reproducibility statement, or figure caption.

The active scientific result is the corrected physical-density (`q_h`)
weighted-Poisson evaluation. The pre-correction sweep is provenance only and
must not be quoted as the current result. The ignored `outputs/old/` tree is
absent from a normal checkout; the README documents how to reconstruct its
exact seed state from Git history for a full replay.

## 1. Experiment identity and status

| Item | Value |
|:---|:---|
| Experiment | `vortices_percentage` |
| Source configuration | `vortices_double_gyre` |
| Base seed | 20260815 |
| Domain | `[0,2] x [0,1]` |
| Allowances | 0.5%, 1%, 2%, 3%, 4%, 5% |
| Production output | `outputs/pareto/` |
| Scientific Full evaluator | Sparse physical-`q_h` weighted-Poisson direct solve |
| Scientific grid/time nodes | 128 x 64; all 21 times |
| Selection trials | 24 |
| Validation trials | 64, independent |
| Full selection curve nested | Yes |
| All final candidates certified | Yes |
| Common-raster decomposition | Pass |
| Frozen inputs unchanged | Yes |
| Final status | **PASS** |

The `smoke` block in `config.json` was not active for the production result.
The primary question was how much physical correction action can be removed by
allowing a small increase in finite-data law risk when four localized sensors
observe a noisy particle population in a time-dependent double-gyre flow.

The corrected Full designs reduce mean held-out Full action relative to the
common Law geometry by 59.17%--70.18%, depending on the allowance. The earlier
26.70% headline used the archived regularized evaluator and is superseded.

## 2. Compared design rules

The eight-dimensional design vector contains four labelled sensor centers:

```text
eta = [x1,y1,x2,y2,x3,y3,x4,y4].
```

- **Population** minimizes exact-oracle population risk and establishes the
  population-loss screen.
- **Law** minimizes finite noisy law risk and supplies one common anchor for
  all percentage allowances.
- **Tangent** minimizes the particle-space tangent action under the population
  and finite-law screens.
- **Full** minimizes the corrected physical-density weighted-Poisson action
  under the same screens.

Selection uses only the frozen selection bank. Validation is evaluated only
after the winner has been frozen and does not participate in selection.

## 3. Physical double-gyre model

Normalized experiment time `t in [0,1]` corresponds to physical time
`tau = H t` with `H = 10`. The velocity field is

```text
a(tau) = epsilon sin(omega tau)
b(tau) = 1 - 2 a(tau)
f(x,tau) = a(tau) x^2 + b(tau) x
omega = 2 pi / period

dx/dtau = -pi A sin(pi f) cos(pi y)
dy/dtau =  pi A cos(pi f) sin(pi y) df/dx
```

and normalized velocity is `dX/dt = H dX/dtau`.

| Physical parameter | Value |
|:---|---:|
| Amplitude `A` | 0.1 |
| Time dependence `epsilon` | 0.25 |
| Physical horizon `H` | 10.0 |
| Period | 10.0 |
| Domain | `[0,2] x [0,1]` |

### 3.1 Initial law

The initial law is a 10% uniform background plus four truncated Gaussian
components. The component weights below are the internal mixture weights for
the non-background portion.

| Component | Weight | Center | Standard deviations |
|---:|---:|:---|:---|
| 1 | 0.30 | (0.45, 0.25) | (0.07, 0.07) |
| 2 | 0.20 | (0.78, 0.72) | (0.07, 0.07) |
| 3 | 0.25 | (1.28, 0.28) | (0.07, 0.07) |
| 4 | 0.25 | (1.62, 0.68) | (0.07, 0.07) |

Out-of-domain Gaussian samples are rejection-resampled rather than clipped.

## 4. Truth and endpoint banks

The frozen truth bank has shape `[21, 50000, 2]`. It uses 50,000 particles,
truth seed offset 1001, and RK4 with 32 substeps per scientific time interval.
The scientific grid has 21 equally spaced normalized time nodes.

Endpoint-only reference training uses an independent 50,000-particle endpoint
bank with seed offset 2001 and 512 RK4 substeps over the endpoint trajectory.
The endpoint arrays have shapes `[50000,2]` at both `t=0` and `t=1`.

| Bank setting | Value |
|:---|:---|
| Truth particles | 50,000 |
| Truth seed | `base_seed + 1001` |
| Truth RK4 substeps per interval | 32 |
| Scientific time nodes | 21 |
| Endpoint particles | 50,000 |
| Endpoint seed | `base_seed + 2001` |
| Endpoint RK4 substeps | 512 |

## 5. Endpoint-only learned reference

The reference is trained only from the endpoint laws. It is represented in
box-logit latent coordinates, and its physical pushforward remains inside the
rectangle. The network input contains two latent coordinates and five time
features `(t, sin(pi t), cos(pi t), sin(2 pi t), cos(2 pi t))`. Four hidden
layers of width 128 use SiLU activations; the output is a two-dimensional
velocity.

| Reference-training hyperparameter | Value |
|:---|:---|
| Seed | 20260815 |
| Hidden width / layers | 128 / 4 |
| Training steps | 12,000 |
| Batch size | 2,048 |
| Optimizer | Adam |
| Adam beta1 / beta2 / epsilon | 0.9 / 0.999 / 1e-8 |
| Initial learning rate | 1e-3 |
| Final learning-rate ratio | 0.05 |
| Schedule | Cosine decay |
| Gradient clipping norm | 10 |
| Bridge | Linear, independent endpoint pairing |
| Bridge noise | 0.0 |
| Log interval | 500 steps |
| Final logged flow-matching loss | 4.634215 |

The rollout bank contains 32,768 reference particles at each of 21 time nodes
and has arrays `nodes`, `velocity`, and `weights` of shapes
`[21,32768,2]`, `[21,32768,2]`, and `[21,32768]`. It uses seed offset 3001 and
RK4 with 16 substeps per scientific interval.

## 6. Sensor and observation model

Each sensor is a localized Gaussian feature

```text
Phi_j(x;eta) = exp(-||x-s_j||^2 / (2 ell^2)),  ell = 0.12.
```

| Measurement parameter | Value |
|:---|:---|
| Number of sensors | 4 |
| Sensor width | 0.12 |
| Allowed x range | [0.24, 1.76] |
| Allowed y range | [0.24, 0.76] |
| Minimum pairwise separation | 0.24 |
| Finite particles per acquisition | 2,000 |
| Acquisition nodes | 9 of 21 |
| Acquisition indices | 0, 2, 5, 8, 10, 12, 15, 18, 20 |
| Detector-noise standard deviation | 0.005 |
| Endpoint moments | Exact |

The selection observation bank stores sample indices with shape
`[24,9,2000]` and detector noise with shape `[24,9,4]`. The independent
validation bank has corresponding shapes `[64,9,2000]` and `[64,9,4]`.

### 6.1 Moment reconstruction

Moment paths are reconstructed with an endpoint-anchored, bounded, C2 cubic
penalized least-squares spline.

| Hyperparameter | Value |
|:---|---:|
| Internal knots | 3 |
| Smoothing | 1e-4 |
| Relative ridge | 1e-10 |
| Roughness quadrature order | 8 |
| Feature bounds | [0,1] |
| Interior feature-bound margin | 0.002 |
| Bound transition width | 0.002 |

### 6.2 Information projection

At each time, the frozen reference weights are exponentially tilted to match
the reconstructed sensor moments. The production projection settings are:

| Projection hyperparameter | Value |
|:---|---:|
| Maximum steps | 300 |
| Newton ridge | 1e-7 |
| Step cap | 20 |
| Multiplier clip | 1,000 |
| Search maximum steps | 60 |
| Search residual tolerance | 1e-6 |
| Search line-search steps | 6 |
| Trajectory backend | `tesseract_cpp` |
| Solver acceptance tolerance | 2e-6 |
| Support-certificate tolerance | 1e-10 |
| Fallback Newton steps | 0 |
| L-BFGS maximum iterations | 800 |
| Retry clip multiplier / retries | 2.0 / 2 |

Particle-MFSI covariance and Tangent search ridges are 1e-7. Exact covariance
and Tangent ridges are zero; exact covariance minimum eigenvalue is 1e-6,
Tangent pseudoinverse `rcond` is 1e-10, and maximum Tangent compatibility
residual is 1e-7.

## 7. Risk and action definitions

### 7.1 Population and finite-law screens

Population loss `L` uses exact-oracle population information. Finite noisy law
risk `R` is a multiscale MMD with bandwidths 0.05, 0.1, 0.2, and 0.4.

The common constraints at allowance `p` are

```text
L(eta) <= L_star + epsilon_L,
R(eta) <= R_star + (p/100) abs(R_star),
```

where `epsilon_L = 0.025` and the frozen Law anchor is
`R_star = 0.03838744201119958`.

### 7.2 Corrected Full action

The authoritative Full evaluator uses physical raster density `q_h` in both
the operator and the vector-field inner product:

```text
-div(q_h grad psi_h) = s_h,
delta*_h = -grad psi_h,
A_full,h = ||delta*_h||^2_(q_h).
```

The scientific grid is 128 x 64 and all 21 time nodes are included. There is
no density floor in the scientific operator. The solver uses conductive
components, component compatibility checks, a constant-fixing pin, symmetric
variable scaling, and an equation-preserving sparse direct solve with residual
iterative refinement. A preconditioned-CG fallback is available.

The `operator_floor_rel = 2e-5`, 64 x 32 grid, and seven-node time grid retained
in `config.json` belong to the differentiable `tesseract_cpp` search proxy.
They do not define reported corrected Full action.

## 8. Common-raster Tangent/hidden decomposition

For each final Law, Tangent, and Full geometry, the moment-rate operator is
built in exactly the same raster vector-field space as Full:

```text
delta_tan,h = argmin ||delta||^2_(q_h) subject to L_h delta = -r_h,
delta_hid,h = delta*_h - delta_tan,h.
```

The audit reports

```text
A_tan,h = ||delta_tan,h||^2_(q_h),
A_hid,h = ||delta_hid,h||^2_(q_h),
Gamma_h = A_hid,h / A_full,h.
```

It independently checks moment feasibility, hidden-nullspace membership,
orthogonality, Pythagorean identity, and hierarchy. No residual, energy, or
hierarchy value is clipped.

## 9. Selection and validation protocol

One Law anchor is reused across the complete 0.5--5% sweep. Full selection is
sequential: each tighter-stage corrected Full winner is a mandatory candidate
at the next allowance and is replaced only by an exactly feasible candidate
whose corrected selection action is lower beyond 1e-6.

Candidate families include the previous corrected Full incumbent, archived
Full candidates, current/repaired Tangent candidates, Law, saved audited
feasible candidates, and new global/local multistarts. Archived geometries are
seeds only and are re-audited against current frozen banks and the corrected
evaluator.

The differentiable Full search uses a 64 x 32 grid, seven time nodes, two
gradient trials, tolerance 1e-6, and at most 360 CG iterations. Candidates then
receive four-trial exact prescreening and full corrected rescoring. Reported
selection values use all 24 selection trials; validation uses 64 disjoint
trials after selection is frozen.

## 10. Optimization hyperparameters

| Hyperparameter | Population | Law | Tangent | Full proxy |
|:---|---:|---:|---:|---:|
| Optimization steps | 100 | 50 | 50 | 30 |
| Learning rate | 0.01 | 0.008 | 0.006 | 0.004 |
| Normal starts | 8 | 6 | 4 | 3 |
| Gradient trials | -- | 4 | 4 | 2 |
| Exact audit budget | 20 | 24 | 30 | 30 |
| Minimum exact-valid | 6 | 8 | -- | -- |
| Exact rescores | -- | -- | 10 | 8 |
| Local starts / scale | -- | -- | 12 / 0.08 | 10 / 0.06 |

Shared settings are 64 generated starts from an oversampled pool of 128,
constraint penalty 10,000, feasibility tolerance 1e-6, and invalid penalty
1,000. Law permits two anchor-refinement passes with consistency tolerance
1e-5. Full uses four prescreen trials.

One historical Tangent seed and one historical Full seed are explicitly listed
in `config.json`, with provenance commit and artifact metadata. They are treated
as candidates and receive current exact audits rather than being accepted by
provenance alone.

## 11. Numerical validity and publication gates

| Validity gate | Threshold |
|:---|---:|
| Maximum population calibration residual | 1e-5 |
| Maximum finite calibration residual | 1e-3 |
| Maximum Poisson relative residual | 2e-7 |
| Minimum ESS fraction | 0.03 |
| Minimum in-domain base mass | 0.995 |
| Maximum action/median ratio | 5.0 |
| Tangent lower-bound/decomposition tolerance | 1e-6 |

Across all 18 final geometries and both frozen banks, the corrected
certification maxima are:

| Quantity | Maximum |
|:---|---:|
| Physical Poisson relative residual | 5.132997e-10 |
| Component compatibility residual | 5.390598e-15 |
| Full moment-rate residual | 8.004651e-11 |
| Tangent moment-rate residual | 3.619007e-14 |
| Hidden-nullspace residual | 8.004623e-11 |
| Absolute orthogonality residual | 3.262391e-11 |
| Absolute Pythagorean residual | 6.184564e-11 |
| Raw hierarchy value `A_tan,h-A_full,h` | -8.561291e-2 |

There are zero aggregate, trial, time/trial, hierarchy, or invalid-trial
violations. All physical solves converged. The negative hierarchy maximum is
strict slack, not a clipped zero.

## 12. Selected sensor geometries

The common Law centers are

```text
(1.077489,0.388963), (0.479798,0.760000),
(1.760000,0.240000), (0.257336,0.602956).
```

Selected Tangent and Full centers are:

| Allowance | Rule | Four centers `(x,y)` |
|---:|:---|:---|
| 0.5% | Tangent = Full | (1.072627,.396293) (.486242,.760000) (1.760000,.240000) (.271940,.618506) |
| 1% | Tangent | (1.054449,.390149) (.451830,.748630) (1.760000,.240000) (.262004,.577847) |
| 1% | Full | (1.054874,.402549) (.470011,.760000) (1.760000,.240000) (.240000,.610886) |
| 2%, 3% | Tangent | (1.058357,.379515) (.475429,.760000) (1.757857,.240000) (.289138,.571884) |
| 2%, 3% | Full | (1.083266,.433572) (.507291,.749698) (1.760000,.240000) (.280867,.639629) |
| 4% | Tangent | (1.070393,.426616) (.490176,.700052) (1.760000,.240000) (.263510,.629298) |
| 4% | Full | (1.063709,.455019) (.498537,.742034) (1.759762,.240000) (.249706,.646382) |
| 5% | Tangent = Full | (1.052135,.417897) (.504911,.725622) (1.749147,.246883) (.245617,.652486) |

## 13. Corrected selection results

Every value below is computed on the frozen 24-trial selection bank. Full
action is evaluated with the corrected scientific evaluator.

| Allowance | Rule | Exact R | R increase | Exact L | Full action | Certified |
|---:|:---|---:|---:|---:|---:|:---:|
| 0.5% | Law | .038387442 | 0.000% | .037957849 | 8.113410 | Yes |
| 0.5% | Tangent/Full | .038521091 | .348% | .038056816 | 7.343264 | Yes |
| 1% | Law | .038387442 | 0.000% | .037957849 | 8.113410 | Yes |
| 1% | Tangent | .038727057 | .885% | .037955747 | 8.037192 | Yes |
| 1% | Full | .038740166 | .919% | .038212162 | 6.688760 | Yes |
| 2% | Law | .038387442 | 0.000% | .037957849 | 8.113410 | Yes |
| 2% | Tangent | .039066942 | 1.770% | .038396153 | 9.040456 | Yes |
| 2% | Full | .039058745 | 1.749% | .038784770 | 6.510045 | Yes |
| 3% | Law | .038387442 | 0.000% | .037957849 | 8.113410 | Yes |
| 3% | Tangent | .039066942 | 1.770% | .038396153 | 9.040456 | Yes |
| 3% | Full | .039058745 | 1.749% | .038784770 | 6.510045 | Yes |
| 4% | Law | .038387442 | 0.000% | .037957849 | 8.113410 | Yes |
| 4% | Tangent | .039789168 | 3.652% | .039496702 | 6.301403 | Yes |
| 4% | Full | .039756850 | 3.567% | .039504166 | 5.748588 | Yes |
| 5% | Law | .038387442 | 0.000% | .037957849 | 8.113410 | Yes |
| 5% | Tangent/Full | .040072176 | 4.389% | .039821283 | 5.730633 | Yes |

The corrected Full selection actions are

```text
7.343264075, 6.688759712, 6.510045283,
6.510045283, 5.748588139, 5.730632961.
```

Consecutive differences are
`[-0.6545043623, -0.1787144297, 0.0, -0.7614571439,
-0.0179551773]`; nesting passes.

## 14. Independent validation results

The standard error is the ordinary across-trial SE over 64 independent frozen
validation trials. Reduction is the ratio-of-means change from the common Law
geometry.

| Allowance | Rule | Validation R +/- SE | Full action +/- SE | Action reduction vs Law | Valid |
|---:|:---|:---|:---|---:|---:|
| 0.5% | Law | .038378 +/- .000053 | 21.037210 +/- 13.499984 | 0.00% | 100% |
| 0.5% | Tangent/Full | .038491 +/- .000051 | 7.367562 +/- .370033 | 64.98% | 100% |
| 1% | Law | .038378 +/- .000053 | 21.037210 +/- 13.499985 | 0.00% | 100% |
| 1% | Tangent | .038729 +/- .000051 | 7.952425 +/- .277932 | 62.20% | 100% |
| 1% | Full | .038725 +/- .000048 | 8.589732 +/- 1.744830 | 59.17% | 100% |
| 2% | Law | .038378 +/- .000053 | 21.037211 +/- 13.499985 | 0.00% | 100% |
| 2% | Tangent | .039059 +/- .000053 | 8.587801 +/- .203604 | 59.18% | 100% |
| 2% | Full | .038997 +/- .000048 | 6.273495 +/- .460063 | 70.18% | 100% |
| 3% | Law | .038378 +/- .000053 | 21.037209 +/- 13.499983 | 0.00% | 100% |
| 3% | Tangent | .039059 +/- .000053 | 8.587801 +/- .203604 | 59.18% | 100% |
| 3% | Full | .038997 +/- .000048 | 6.273495 +/- .460062 | 70.18% | 100% |
| 4% | Law | .038378 +/- .000053 | 21.037209 +/- 13.499984 | 0.00% | 100% |
| 4% | Tangent | .039751 +/- .000047 | 5.872456 +/- .284154 | 72.09% | 100% |
| 4% | Full | .039693 +/- .000043 | 6.274276 +/- .901433 | 70.18% | 100% |
| 5% | Law | .038378 +/- .000053 | 21.037210 +/- 13.499984 | 0.00% | 100% |
| 5% | Tangent/Full | .040012 +/- .000046 | 6.288513 +/- .679615 | 70.11% | 100% |

The Law validation action has an unusually large mean and SE because a small
number of trials have high action. All 64 Law trials remain numerically valid;
the raw per-trial values are retained in `validation_trial_summaries.csv`.

Tangent has lower validation Full action at 1% and 4%; Full has lower action at
2% and 3%; the two geometries coincide at 0.5% and 5%. This mixed method ranking
must replace any claim that Full uniformly beats Tangent on validation.

## 15. Full common-raster decomposition

| Allowance | Selection Full | `A_tan,h` | `A_hid,h` | `Gamma_h` |
|---:|---:|---:|---:|---:|
| 0.5% | 7.343264 | .430744 | 6.912520 | .941342 |
| 1% | 6.688760 | .435773 | 6.252986 | .934850 |
| 2% | 6.510045 | .430471 | 6.079575 | .933876 |
| 3% | 6.510045 | .430471 | 6.079575 | .933876 |
| 4% | 5.748588 | .417009 | 5.331580 | .927459 |
| 5% | 5.730633 | .386964 | 5.343669 | .932474 |

The supported tangent-invisible fraction is approximately 92.7%--94.1% on the
selection bank. This statement relies on the common-raster orthogonality and
Pythagorean checks; it is not inferred by subtracting incompatible particle and
raster metrics.

## 16. Paper-facing interpretation

A defensible primary statement is:

> In the corrected double-gyre experiment, four sensor locations selected
> under 0.5%--5% Law-relative finite-risk allowances reduced mean held-out
> physical-density weighted-Poisson action by 59.17%--70.18% relative to the
> common Law geometry. All corrected designs passed population, law,
> projection, Poisson, and common-raster decomposition gates. The corrected
> decomposition attributed approximately 92.7%--94.1% of selected Full action
> to the tangent-invisible component.

Important qualifications:

- The large Law-action SE makes the ratio-of-means reduction sensitive to a
  small number of high-action Law trials; report the raw Law mean and SE.
- Tangent and Full have mixed validation rankings across allowances.
- The 2% geometry is retained at 3%; this is a search plateau, not an analytic
  proof of global optimality.
- Results cover one flow regime, sensor count, noise level, and initial law.
- The old regularized-evaluator reductions are not comparable as pure
  optimization changes because the scientific evaluator itself changed.

## 17. Reproduction

The commands below describe the original in-place production layout. For a
fresh checkout, use the README's three-level reproduction procedure: it adds
the required archive bootstrap and writes a replay outside the tracked
authoritative directory.

From the repository root:

```bash
.venv/bin/python experiments/vortices_percentage/run_pareto.py \
  --source-run experiments/vortices_percentage/outputs/old/pareto_pre_corrected_full/risk_0p5pct \
  --output experiments/vortices_percentage/outputs/pareto \
  --seed-pareto experiments/vortices_percentage/outputs/old/pareto_pre_corrected_full

.venv/bin/python experiments/vortices_percentage/finalize_authoritative_corrected_pareto.py \
  --pareto-dir experiments/vortices_percentage/outputs/pareto \
  --archive-pareto experiments/vortices_percentage/outputs/old/pareto_pre_corrected_full \
  --source-run experiments/vortices_percentage/outputs/old/pareto_pre_corrected_full/risk_0p5pct
```

The finalizer is fail-closed and does not optimize or alter winners.

The per-point timing receipts for the staged source sweep total approximately
104 minutes across six points on the recorded machine, but they do not include
all later corrected-evaluator searches and finalization. They should not be
reported as the total cost of the corrected study.

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
| `experiment.py` | `5bcd5b3c96668cabf6d7a8b2b1944f48f490635763b997172584328551a9a4c4` |
| `selection.py` | `baf15fceff7b926d25471560e3342fe7fe7aaaa3993998b3f2274af8094d99ed` |
| `run_pareto.py` | `351b14c5aa3ef41b929ffb548a3c47ab6f2136e7c78494e3043159225cefa62a` |
| `finalize_authoritative_corrected_pareto.py` | `e5c146dcd176d8c96937536b0c4575afdf659382f8220c30ecb71b6b3f1aaeb0` |

## 19. Artifact and hash index

| Artifact | Role |
|:---|:---|
| `outputs/pareto/corrected_authoritative_pareto.*` | Corrected Full table |
| `outputs/pareto/corrected_authoritative_decomposition_audit.*` | Selection/validation decomposition receipt |
| `outputs/pareto/pareto_methods_selection.csv` | Law/Tangent/Full selection table |
| `outputs/pareto/pareto_methods_validation.csv` | Independent validation table |
| `outputs/pareto/validation_trial_summaries.csv` | Per-trial validation values |
| `outputs/pareto/authoritative_certification_diagnostic.json` | Fail-closed certification detail |
| `outputs/pareto/authoritative_run_summary.json` | Concise scientific summary |
| `outputs/pareto/frozen_inputs/manifest.json` | Eight frozen-input hashes and source paths |
| `outputs/pareto/old_vs_corrected_full_comparison.*` | Archived/current comparison |
| `outputs/pareto/risk_*pct/` | Per-allowance candidates, banks, receipts, and timings |

| Authoritative artifact | SHA-256 |
|:---|:---|
| Corrected Pareto JSON | `cdf84cd0e8277c8b3f89bf950d82a349f18972fff9986a5b1a217e213fac89aa` |
| Certification diagnostic | `4bda8b411ed38f5078f22c343ec31f74c38336f1b5876d8d547a19f08f57af7a` |
| Authoritative run summary | `4414ec8308bd1121ce1d3bea28f4a4bd3bfe68defd15e7ed4ef25edfb8017daa` |

Frozen input SHA-256 values are:

| Input | SHA-256 |
|:---|:---|
| Truth bank | `d897ff7fc44c0b85d7bb5391c0cc25895b4301e9c2ce00184697a1899d853b5b` |
| Reference endpoints | `ad4006927e268c52f621c16c773f0600d803370bd21fb5e0816d82a70dbdfbba` |
| Reference checkpoint | `63619893b44d49f6fd89f210239e0428d3d41aabfb4789cff616973066d21138` |
| Reference rollout bank | `159515f930ab1b82c0a9ef42706705f192af295ca2c74a9611da1558464a0b2f` |
| Selection bank | `0ae52680ba66f07e36e02a0d85d25847fc11dc2554fcb63f95cb4e7aa0636ef9` |
| Validation bank | `63748a79d00bce58e6307f2070f29c480998de0a7c5c47b4fcb0788696dea894` |
| Frozen config | `8f57f167675718b19d7ffc1741a8175adbe22069ff4043634b62df8dcf100ed0` |

## 20. Suggested figure caption and future checks

> Corrected percentage-risk Pareto evaluation for four localized sensors in a
> time-dependent double-gyre flow. Full designs were selected on a frozen
> 24-trial bank subject to exact population and Law-relative risk screens and
> evaluated after selection on an independent 64-trial bank. Reported Full
> action uses a physical-density weighted-Poisson solve on a 128 x 64 grid at
> all 21 time nodes; error bars are across-trial standard errors. Corrected Full
> designs reduce mean held-out action by 59.17%--70.18% relative to Law, while
> common-raster decomposition attributes approximately 92.7%--94.1% of selected
> Full action to the tangent-invisible component.

Before a confirmatory paper run, preregister the allowance and primary contrast,
generate a new independent validation bank, record total corrected-run wall
time and peak memory, and commit the exact source revision. A larger validation
bank is especially valuable because of the heavy-tailed Law action.
