# Vortices / double-gyre percentage-risk Pareto experiment

This experiment studies sensor placement for a time-dependent double-gyre flow. Four localized Gaussian sensors observe a finite, noisy particle population. Their positions are optimized under four increasingly operational objectives—Population, Law, Tangent, and Full—then the Full design is swept across controlled percentage increases in finite-law risk.

The completed sweep uses allowances of `0.5%, 1%, 2%, 3%, 4%, 5%`. All 18 reported Law/Tangent/Full method points pass their exact selection-bank certificates, and every method has `64/64` valid independent validation trials at every allowance. The largest validated reduction occurs at the `4%` allowance: Full reduces mean validation action from `3.36292 ± 0.06331` to `2.46503 ± 0.02749`, a `26.70%` ratio-of-means reduction, while using an exact selection-risk increase of `3.652%`.

This README describes the current saved artifacts under `outputs/pareto/`. Exact values and audit metadata live in each point's `result.json`; the tables below are rounded for readability.

## Contents

- [Scientific question](#scientific-question)
- [Double-gyre truth model](#double-gyre-truth-model)
- [Sensor and observation model](#sensor-and-observation-model)
- [Boundary-preserving learned reference](#boundary-preserving-learned-reference)
- [Moment reconstruction and information projection](#moment-reconstruction-and-information-projection)
- [Population, Law, Tangent, and Full objectives](#population-law-tangent-and-full-objectives)
- [Optimization, auditing, and Pareto protocol](#optimization-auditing-and-pareto-protocol)
- [Complete hyperparameters](#complete-hyperparameters)
- [Results](#results)
- [Interpretation and limitations](#interpretation-and-limitations)
- [Reproduction](#reproduction)
- [Artifacts and implementation map](#artifacts-and-implementation-map)

## Scientific question

Let the design vector contain four sensor centers,

```text
eta = (x_1,y_1, ..., x_4,y_4).
```

The experiment asks whether sensor placements with slightly higher finite-data distributional risk can substantially lower the physical correction action required to reconcile a learned reference flow with the measured population moments.

At each percentage allowance `p`, transport designs must satisfy

```text
L(eta) <= L* + epsilon_L
R(eta) <= R* + (p / 100) |R*|,
```

where `L` is an exact-oracle population risk, `R` is finite noisy law risk, and `R*` is one common exact Law anchor frozen across the sweep. The sweep then compares:

- **Law:** minimizes finite-data law risk;
- **Tangent:** minimizes local tangent transport action;
- **Full:** minimizes the weighted-Poisson MFSI action.

All designs are evaluated on the same exact metrics and on a separate common validation bank.

## Double-gyre truth model

### Domain and velocity field

The physical domain is

```text
Omega = [0,2] x [0,1],
```

with normalized experiment time `t in [0,1]` corresponding to physical time `tau = Ht`, where `H = 10`. The double-gyre field is

```text
a(tau) = epsilon sin(omega tau),
b(tau) = 1 - 2a(tau),
f(x,tau) = a(tau)x^2 + b(tau)x,
omega = 2pi / period,

dx/dtau = -pi A sin(pi f) cos(pi y),
dy/dtau =  pi A cos(pi f) sin(pi y) d f/dx.
```

The implementation returns normalized-time velocity `dX/dt = H dX/dtau`. Active parameters are

```text
A = 0.1,
epsilon = 0.25,
H = 10,
period = 10.
```

The velocity is tangent to the rectangular boundary, and the saved truth bank has minimum in-domain fraction exactly `1.0`.

### Initial population

The initial law is a mixture of:

- `10%` uniform background on the full rectangle;
- `90%` distributed among four truncated Gaussian components with normalized internal weights `(0.30, 0.20, 0.25, 0.25)`;
- centers `(0.45,0.25)`, `(0.78,0.72)`, `(1.28,0.28)`, `(1.62,0.68)`;
- coordinate standard deviations `(0.07,0.07)`.

Gaussian draws outside the physical box are rejection-resampled rather than clipped. The hidden truth uses `50,000` particles, seed offset `1001`, 21 time nodes, and RK4 with `32` substeps per time interval. A separate high-accuracy endpoint rollout uses `512` RK4 substeps to build reference-training endpoints.

The hidden truth bank is frozen and hard-linked/copied into every compatible Pareto point. It is never resimulated independently per percentage.

## Sensor and observation model

### Gaussian point sensors

Each freely placed sensor has feature

```text
Phi_j(x; eta) = exp(-||x-s_j||^2 / (2 ell^2)),
ell = 0.12.
```

The four centers are labeled during differentiable optimization to avoid nondifferentiable sorting boundaries. Geometry constraints require:

- center box `x in [0.24,1.76]`, `y in [0.24,0.76]`;
- pairwise center separation at least `0.24`;
- optimizer iterates projected back into the center box.

The boundary margin is twice the sensor width, so the nominal one-standard-deviation sensing disk remains away from the physical boundary.

### Finite observations

Each trial uses:

- `2,000` sampled truth particles per acquisition time;
- `9` acquisition nodes nested in the 21-node scientific time grid;
- four sensor observables;
- additive independent Gaussian detector noise with standard deviation `0.005`;
- exact endpoint moments at the first and last nodes.

For an interior acquisition node,

```text
y_{k,j} = (1/2000) sum_i Phi_j(X_i(t_k);eta) + 0.005 z_{k,j},
z_{k,j} ~ N(0,1).
```

Sample indices and detector noises are pre-generated in frozen common-random-number banks. Law/action selection banks and validation banks are disjoint.

## Boundary-preserving learned reference

### Why a transformed reference is needed

The generic learned MLP velocity is defined on all of `R^2`, while the double-gyre physics is confined to a box. The experiment therefore maps physical coordinates to box-logit coordinates,

```text
s = (x-low)/(high-low),
z = log(s) - log(1-s),
```

trains and integrates the flow in latent `z` space, and maps back through

```text
x = low + (high-low) sigmoid(z).
```

Physical velocity is obtained by the chain-rule pushforward. The Jacobian factor tends to zero at the box boundary, and latent-coordinate integration guarantees that returned reference particles remain strictly inside the physical rectangle. The transform version saved in the checkpoint is `box_logit_reference_v1` with clipping epsilon `1e-6`.

### Endpoint data

The reference is endpoint-only. It receives `50,000` endpoint particles, generated with seed offset `2001`, from the hidden simulator's accurate initial/final laws. It does not train against intermediate double-gyre states.

The frozen rollout bank contains `32,768` particles with seed offset `3001` and uses RK4 with `16` substeps per scientific interval.

### MLP and flow matching

The latent velocity MLP has:

- seven inputs: two latent spatial coordinates plus `(t, sin(pi t), cos(pi t), sin(2pi t), cos(2pi t))`;
- four SiLU hidden layers of width `128`;
- a linear two-dimensional output;
- five affine parameter layers total.

Independent endpoint pairs use the deterministic linear bridge

```text
z_t = (1-t)z_0 + t z_1,
u_t = z_1 - z_0,
```

because bridge noise is `0.0` in this experiment. Training minimizes conditional-flow-matching squared error with Adam for `12,000` steps, batch size `2,048`, gradient clipping at `10`, and cosine learning-rate decay from `1e-3` to `5e-5`.

The saved checkpoint's logged minibatch loss decreases from `8.46044` at step 1 to `4.63421` at step 12,000. These are stochastic training diagnostics, not the downstream MMD risk or MFSI action.

## Moment reconstruction and information projection

### Anchored cubic reconstruction

Each sensor's nine noisy acquisitions are fit using an endpoint-anchored cubic penalized least-squares spline with:

- three internal knots;
- smoothing coefficient `1e-4`;
- relative ridge `1e-10`;
- roughness quadrature order `8`;
- `C2`-smooth reconstruction;
- feature bounds `[0,1]`, shrunk to the interior by `0.002`;
- smooth bound-transition width `0.002`.

The reconstruction supplies `c(t)` and `c_dot(t)` on all 21 nodes. Smooth saturation prevents finite noisy fits from demanding impossible Gaussian-sensor moments while retaining differentiability near the bounds.

### Empirical I-projection

At every time, frozen reference particles with base weights `b_i` are exponentially tilted to match the reconstructed moments:

```text
q_i(lambda) = b_i exp(lambda^T Phi(x_i;eta)) / Z(lambda),
sum_i q_i Phi(x_i;eta) = c(t).
```

The multiplier solve is warm-started along trajectories. Native `tesseract_cpp` projection is used for both search and exact evaluation. Candidate acceptance requires calibration residual, effective sample size, support feasibility, and reference in-domain mass checks.

## Population, Law, Tangent, and Full objectives

### Exact-oracle population risk `L`

The Population stage uses exact hidden moments—no finite sampling and no sparse observation layer—then projects the reference law and evaluates time-integrated multiscale MMD against the hidden truth. It establishes the exact-oracle baseline

```text
L*    = 0.0377392837017
L_max = 0.0627392837017
epsilon_L = 0.025.
```

### Finite law risk `R`

The Law stage repeats reconstruction and projection using sparse noisy finite observations. At each time it rasterizes the projected reference law onto a `128 x 64` physical grid and computes squared MMD against the truth mass. The kernel is an equal-weight mixture of Gaussian kernels with bandwidths

```text
0.05, 0.10, 0.20, 0.40.
```

Time values are trapezoid-integrated and trial values are averaged. The exact common Law anchor is

```text
R* = 0.0383874420112.
```

For allowance `p`,

```text
epsilon_R(p) = (p/100)|R*|,
R_max(p) = R* + epsilon_R(p).
```

At the representative 3% point, `epsilon_R = 0.00115162326034` and `R_max = 0.0395390652715`.

### Tangent action `T`

The tangent objective measures the minimum local correction seen through the sensor moments. Schematically,

```text
G_t = E_q[J Phi J Phi^T],
r_t = E_q[J Phi u_ref] - c_dot(t),
T = integral r_t^T (G_t + ridge I)^(-1) r_t dt.
```

It requires projected weights and sensor gradients but no raster forcing or Poisson solve. It is exact for the declared tangent definition on the frozen action bank.

### Full weighted-Poisson action `A_full`

The full calculation derives the measurement-consistent particle forcing, rasterizes density `q_t` and scalar forcing `h_t`, and solves

```text
-div(q_t grad psi_t) = q_t h_t
```

with a gauge constraint. The reported action is

```text
A_full = integral int q_t ||grad psi_t||^2 dx dt.
```

The authoritative evaluator uses the `128 x 64` grid and all 21 time nodes with the physical-density sparse direct solver. Stage-4 gradients use a lower-cost `64 x 32` grid, seven time nodes, two CRN trials, CG tolerance `1e-6`, and up to `360` iterations with the regularized native `tesseract_cpp` search proxy. Only authoritative exact-audit values are published.

## Optimization, auditing, and Pareto protocol

### Four-stage selection

1. **Population:** minimizes exact-oracle `L` over geometrically valid sensor layouts.
2. **Law:** minimizes finite noisy `R` under `L <= L_max`; exact auditing establishes `R*`.
3. **Tangent:** minimizes exact tangent action under the `L` and current `R` screens.
4. **Full:** minimizes a lower-resolution full-action search objective, then prescreens and rescoring candidates under exact full fidelity.

The global candidate generator samples `64` starts from an oversampled pool of `128`. The optimizer uses smaller deterministic subsets per stage, while every start remains eligible for ranking and exact audit. Configured historical Tangent and Full geometries are candidate seeds only; their provenance is recorded and they are always re-audited under the current frozen banks.

### Exact audit policy

Candidates must satisfy:

- sensor-center box and pairwise-separation constraints;
- exact `L` and `R` screens;
- full-bank projection/calibration validity;
- minimum effective sample size;
- in-domain reference mass;
- full-action Poisson residual and action-tail checks when applicable.

If a Tangent or Full candidate beats the claimed Law anchor risk, the pipeline refines/restarts the Law search up to two passes. A frozen Pareto anchor that is beaten causes an error rather than silently publishing a negative-excess point.

### Nested Pareto sweep

The runner uses methodology version `3`. It:

1. finds an archived exact-valid Law seed only as a candidate generator;
2. freezes the first completed point's exact Law design and `R*` for subsequent points;
3. seeds every point with identical truth, endpoint, reference, selection, and validation banks;
4. carries the best exact Full incumbent into the next, looser allowance;
5. rejects an exact selection action that increases relative to the incumbent beyond numerical tolerance;
6. validates all designs on the disjoint bank;
7. generates aggregate tables and figures after completion.

The saved run uses `16` Law-selection trials, `24` action-selection trials, and `64` independent validation trials. Selection and validation namespaces are `9890` and `9891`.

## Complete hyperparameters

The canonical machine-readable configuration is [`config.json`](config.json). Active full-run values are enumerated below.

### Truth, law, and measurement

| Group | Hyperparameter | Value |
|:---|:---|---:|
| global | seed | `20260815` |
| truth | amplitude / temporal epsilon | `0.1 / 0.25` |
| truth | horizon / period | `10 / 10` |
| truth | particles | `50000` |
| truth | truth seed offset | `1001` |
| truth | RK4 substeps / endpoint RK4 substeps | `32 / 512` |
| initial law | background weight | `0.1` |
| initial law | internal mixture weights | `0.30, 0.20, 0.25, 0.25` |
| initial law | component standard deviations | `0.07, 0.07` |
| law | MMD bandwidths | `0.05, 0.10, 0.20, 0.40` |
| law | `epsilon_L` | `0.025` |
| law | nominal relative risk allowance | `0.05` (overridden per point) |
| measurement | sensors / sensor width | `4 / 0.12` |
| measurement | boundary margin / minimum separation | `0.24 / 0.24` |
| measurement | finite particles / acquisition nodes | `2000 / 9` |
| measurement | detector-noise standard deviation | `0.005` |

### Moment reconstruction

| Hyperparameter | Value |
|:---|---:|
| kind | `endpoint_anchored_cubic_penalized_ls_c2_bounded` |
| feature bounds | `[0,1]` |
| interior margin / transition width | `0.002 / 0.002` |
| internal knots | `3` |
| smoothing / relative ridge | `1e-4 / 1e-10` |
| roughness quadrature order | `8` |

### Learned reference

| Hyperparameter | Value |
|:---|---:|
| training seed | `20260815` |
| hidden width / layers | `128 / 4` |
| training steps / batch size | `12000 / 2048` |
| initial learning rate / minimum ratio | `0.001 / 0.05` |
| Adam beta1 / beta2 / epsilon | `0.9 / 0.999 / 1e-8` |
| gradient clip norm | `10` |
| bridge schedule / noise std | `linear / 0` |
| logging interval | `500` |
| endpoint particles / seed offset | `50000 / 2001` |
| rollout particles / seed offset | `32768 / 3001` |
| rollout RK4 substeps per interval | `16` |
| coordinate transform | `box_logit_reference_v1` |

### Projection, particle MFSI, raster, and Poisson

| Group | Hyperparameter | Value |
|:---|:---|---:|
| projection | authoritative max steps | `300` |
| projection | search max steps / residual tol | `60 / 1e-6` |
| projection | Newton ridge / step cap | `1e-7 / 20` |
| projection | lambda clip / implicit ridge | `1000 / 0` |
| projection | search line-search steps | `6` |
| projection | backend / solver acceptance tol | `tesseract_cpp / 2e-6` |
| projection | support certificate tol | `1e-10` |
| projection | fallback Newton steps | `0` |
| projection | L-BFGS max iterations | `800` |
| projection | retry clip multiplier / retries | `2 / 2` |
| particle MFSI | covariance ridge / tangent ridge | `1e-7 / 1e-7` |
| exact MFSI | covariance ridge / tangent ridge | `0 / 0` |
| exact MFSI | minimum covariance eigenvalue | `1e-6` |
| exact MFSI | tangent pseudoinverse rcond | `1e-10` |
| exact MFSI | max tangent compatibility residual | `1e-7` |
| raster | bandwidth / truncation | `0 / 4` |
| Poisson | physical box | `[0,2] x [0,1]` |
| Poisson | grid / time nodes | `128 x 64 / 21` |
| Poisson | search-proxy floor / authoritative operator floor | `2e-5 / 0` (proxy preconditioning/regularization only) |
| Poisson | CG tolerance / max iterations | `1e-7 / 520` |
| Poisson | gauge strength | `1` |

### Validity and randomness

| Hyperparameter | Value |
|:---|---:|
| max population calibration residual | `1e-5` |
| max finite calibration residual | `1e-3` |
| max Poisson relative residual | `2e-7` |
| minimum ESS fraction | `0.03` |
| minimum in-domain base mass | `0.995` |
| maximum action/median ratio | `5` |
| tangent lower-bound tolerance | `1e-6` |
| Law / action / validation trials | `16 / 24 / 64` |
| selection / validation namespace | `9890 / 9891` |

### Optimization

| Hyperparameter | Value |
|:---|---:|
| Population / Law / Tangent / Full steps | `100 / 50 / 50 / 30` |
| default learning rate | `0.01` |
| Population / Law / Tangent / Full learning rate | `0.01 / 0.008 / 0.006 / 0.004` |
| constraint penalty / invalid penalty | `10000 / 1000` |
| feasibility tolerance | `1e-6` |
| start count / oversample | `64 / 128` |
| Population starts / exact audits / min valid | `8 / 20 / 6` |
| Law starts / gradient trials / exact audits / min valid | `6 / 4 / 24 / 8` |
| Law anchor refinement passes / consistency tol | `2 / 1e-5` |
| Tangent starts / gradient trials | `4 / 4` |
| Tangent local starts / scale | `12 / 0.08` |
| Tangent exact audits / rescores | `30 / 10` |
| Full starts / gradient trials | `3 / 2` |
| Full local starts / scale | `10 / 0.06` |
| Full gradient grid / time nodes | `64 x 32 / 7` |
| Full gradient CG tolerance / max iterations | `1e-6 / 360` |
| Full gradient proxy / authoritative exact Poisson backend | `tesseract_cpp / physical-q sparse direct` |
| Full prescreen trials / exact audits / rescores | `4 / 30 / 8` |
| exact batch trials | `4` |
| Pareto methodology version | `3` (set by runner) |

The Tangent candidate seed is

```text
(0.334482,0.726689), (1.248193,0.536980),
(0.957159,0.423838), (0.646333,0.348526).
```

The Full candidate seed is

```text
(0.262649,0.587315), (1.254508,0.520985),
(0.401796,0.387089), (0.638563,0.288270).
```

These are candidate generators, not fixed answers. Their recorded provenance is source commit `d5d5cb30b4b4718d20858f029198d9bec5ac28b6`, observation commit `7725d6ff92a35752a0871205016611e5ca2e57a1`, artifact `experiments/vortices/outputs/run/result.json`, with policy `candidate_only_reaudit_under_current_frozen_banks`.

### Smoke overrides

Smoke mode uses `2,048` truth particles, truth RK4 `4`, endpoint RK4 `32`, `64` finite particles, `5` acquisitions, a two-layer width-32 MLP trained for `10` steps with batch `128`, `2,048` endpoint particles, `1,024` reference particles, reference RK4 `2`, a `32 x 16 x 5` Poisson discretization, four candidate starts, zero optimization steps, one trial per objective, and relaxed in-domain/Poisson diagnostic tolerances. It tests integration and differentiability only.

## Results

### Complete method comparison

`Risk inc.` is exact selection-bank `100(R-R*)/|R*|`. Validation metrics are mean `±` standard error over 64 independent trials. `Delta A` is reduction in mean full action relative to Law at the same allowance.

| Allow. | Method | Risk inc. | Selection `A_full` | Validation `R` | Validation `A_full` | Delta A vs Law | Cert. |
|---:|:---|---:|---:|---:|---:|---:|:---:|
| 0.5% | Law | 0.000% | 3.4586 | 0.038378 ± 0.000053 | 3.3629 ± 0.0633 | 0.00% | yes |
| 0.5% | Tangent | 0.437% | 3.8576 | 0.038556 ± 0.000053 | 3.7918 ± 0.0662 | -12.75% | yes |
| 0.5% | Full | 0.370% | 3.1765 | 0.038527 ± 0.000051 | 3.0941 ± 0.0418 | 7.99% | yes |
| 1% | Law | 0.000% | 3.4586 | 0.038378 ± 0.000053 | 3.3629 ± 0.0633 | 0.00% | yes |
| 1% | Tangent | 0.885% | 3.5792 | 0.038729 ± 0.000051 | 3.5539 ± 0.0519 | -5.68% | yes |
| 1% | Full | 0.919% | 2.8258 | 0.038725 ± 0.000048 | 2.8287 ± 0.0412 | 15.89% | yes |
| 2% | Law | 0.000% | 3.4586 | 0.038378 ± 0.000053 | 3.3629 ± 0.0633 | 0.00% | yes |
| 2% | Tangent | 0.885% | 3.5792 | 0.038729 ± 0.000051 | 3.5539 ± 0.0519 | -5.68% | yes |
| 2% | Full | 0.919% | 2.8258 | 0.038725 ± 0.000048 | 2.8287 ± 0.0412 | 15.89% | yes |
| 3% | Law | 0.000% | 3.4586 | 0.038378 ± 0.000053 | 3.3629 ± 0.0633 | 0.00% | yes |
| 3% | Tangent | 2.415% | 3.0610 | 0.039329 ± 0.000050 | 3.0353 ± 0.0421 | 9.74% | yes |
| 3% | Full | 0.919% | 2.8258 | 0.038725 ± 0.000048 | 2.8287 ± 0.0412 | 15.89% | yes |
| 4% | Law | 0.000% | 3.4586 | 0.038378 ± 0.000053 | 3.3629 ± 0.0633 | 0.00% | yes |
| 4% | Tangent | 3.652% | 2.4720 | 0.039751 ± 0.000047 | 2.4650 ± 0.0275 | **26.70%** | yes |
| 4% | Full | 3.652% | 2.4720 | 0.039751 ± 0.000047 | 2.4650 ± 0.0275 | **26.70%** | yes |
| 5% | Law | 0.000% | 3.4586 | 0.038378 ± 0.000053 | 3.3629 ± 0.0633 | 0.00% | yes |
| 5% | Tangent | 4.389% | 2.4403 | 0.040012 ± 0.000046 | 2.4721 ± 0.0343 | **26.49%** | yes |
| 5% | Full | 4.541% | 2.4302 | 0.040072 ± 0.000046 | 2.4905 ± 0.0367 | 25.94% | yes |

### Sensor coordinates

The Law geometry is common to all allowances:

```text
(1.07749, 0.38896), (0.47980, 0.76000),
(1.76000, 0.24000), (0.25734, 0.60296).
```

Tangent and Full geometries are:

| Allow. | Tangent centers `(x,y)` | Full centers `(x,y)` |
|---:|:---|:---|
| 0.5% | `(1.08094,.40027) (.47466,.76000) (1.75380,.24000) (.26984,.58384)` | `(1.06075,.37792) (.46109,.76000) (1.76000,.24000) (.24719,.60619)` |
| 1% | `(1.05445,.39015) (.45183,.74863) (1.76000,.24000) (.26200,.57785)` | `(1.05487,.40255) (.47001,.76000) (1.76000,.24000) (.24000,.61089)` |
| 2% | same as 1% | same as 1% |
| 3% | `(1.04889,.39445) (.44516,.74504) (1.74186,.25639) (.25388,.58153)` | same as 1% |
| 4% | same as Full | `(1.07039,.42662) (.49018,.70005) (1.76000,.24000) (.26351,.62930)` |
| 5% | `(1.05214,.41790) (.50491,.72562) (1.74915,.24688) (.24562,.65249)` | `(1.05086,.41959) (.50196,.72778) (1.74501,.24553) (.24508,.65277)` |

### Visual summary

![Law, Tangent, and Full comparison across the vortex Pareto sweep](outputs/pareto/pareto_methods.png)

![Vortex sensor positions across allowances](outputs/pareto/pareto_sensor_layouts.png)

The representative 3% dashboard shows the hidden population evolution and velocity field, all four objective-specific sensor layouts, exact selection trade-off, validation distributions, and stage timings:

![Double-gyre experiment and sensor dashboard](outputs/pareto/experiment_sensors.png)

### Exact Tangent/Full decomposition audit

The saved Law, Tangent, and Full geometry at every allowance was re-evaluated with `VortexExperiment.evaluate_trials_exact` on the frozen 24-trial action-selection bank and all 21 time nodes, using the reported `128 x 64` Full-Poisson discretization. Tangent and Full share each reconstructed target and projected weight field. With the configured hierarchy tolerance `1e-6`, all 18 final candidates pass: there are zero aggregate, trial-level, or time/trial-level violations. The maximum raw value of `A_tan - A_full` is `-0.138640981045`; it is signed, unclipped hierarchy slack.

The complete table of `A_tan`, `A_full`, `A_hid = A_full - A_tan`, and `Gamma = 1 - A_tan/A_full` is in [the decomposition audit](outputs/pareto/action_decomposition_audit.md). The CSV is `outputs/pareto/action_decomposition_audit.csv`, with raw per-trial/per-time values in `outputs/pareto/action_decomposition_evaluations.json`.

The stronger independent correction-field audit does **not** validate an orthogonal decomposition at the same tolerance. It constructs the Full and Tangent correction potentials, forms `delta_hid = delta_* - delta_tan`, and directly evaluates the weighted hidden norm and Tangent–Hidden inner product at every time/trial. The result is **FAIL**: maximum absolute `A_full - A_tan - E||delta_hid||^2` is `0.214713097789`, and maximum absolute `E[delta_tan · delta_hid]` is `0.296941844225`. The underlying direct-field polarization identity is internally consistent to `1.92e-14`, isolating the discrepancy to the particle Tangent versus rasterized Full discretizations. See `outputs/pareto/orthogonal_decomposition_audit.csv` and `.json`.

### Common-raster orthogonal-decomposition audit

The follow-up audit removes that particle/raster mismatch. It constructs `L_h` from the four sensor potentials on the authoritative `128 x 64` Full grid, with the same physical-density edge weights and cell quadrature used by the reported Full action. At every time/trial it solves the raster Gram system for the minimum-norm correction satisfying `L_h delta_tan,h = -r_h`, computes `delta_hid,h = delta_h* - delta_tan,h`, and evaluates all three energies independently.

The result remains **FAIL** at the saved absolute tolerance `1e-6`, with Full moment-rate feasibility failing first. The maximum Full residual is `0.0163264953989`, versus `2.45427568961e-14` for raster Tangent feasibility. The maximum hidden-nullspace, absolute orthogonality, and absolute Pythagorean residuals are `0.0163264953989`, `0.0199222346691`, and `0.0398444693381`. The raw hierarchy never fails (maximum `A_tan,h - A_full,h = -0.138856148563`), and the raster Full energy reproduces the authoritative action exactly at reported precision.

The historical failure occurred before the projection: the then-authoritative linear solve used `q_h + q_floor` (`operator_floor_rel = 2e-5`) for stability, while the scientific action and requested common Hilbert space used physical `q_h`. Subtracting the floor and gauge moment contributions reduced the maximum Full residual to `3.32801017212e-8`, directly confirming the mismatch. Thus `A_hid,h/A_full,h` was **not** numerically supported as a genuine tangent-invisible fraction for those saved reports. All saved candidate and frozen-bank hashes were unchanged. See [the historical common-discretization report](outputs/pareto/common_discretization_decomposition_audit.md), its CSV table, and its JSON time/trial record. The corrected targeted diagnostic below supersedes its solver implementation without overwriting that artifact.

### Corrected physical-density solver: targeted gate

The authoritative evaluator now solves `-div(q_h grad psi_h) = q_h h_h` and evaluates action in the same physical-density metric. Equation-preserving stabilization consists of scaling by `max(q_h)`, restriction to conductive components, one constant-fixing pin per component, and sparse SuperLU with a preconditioned-CG fallback. The density floor is never part of the scientific operator. Exact Full validity additionally requires the independently constructed raster moment-rate residual, component compatibility, and solver convergence.

The mandated 3% Law/Tangent/Full targeted diagnostic **PASSES** locally. Across all three candidates, the maxima are: physical Poisson residual `7.86684769872e-9`, Full moment residual `1.80018663406e-10`, Tangent moment residual `1.27453187867e-14`, hidden-nullspace residual `1.80019158125e-10`, absolute orthogonality residual `3.36466965400e-11`, and absolute Pythagorean residual `6.72741862218e-11`. The maximum raw hierarchy value is `-0.141457602639`, so no hierarchy violation occurs. Thus `Gamma_h` has the requested genuine discrete geometric interpretation for this targeted vortex set. The paired rescore is nevertheless blocked by the toy prerequisite; no all-allowance corrected outputs or optimization changes were made. See [the corrected targeted audit](outputs/pareto/corrected_full_solver_targeted_audit.md) and its JSON record.

### Tangent optimizer repair at 4% and 5%

The corresponding Full geometry and previous Tangent geometry were added as mandatory Tangent seeds alongside the original configured multistarts. At 4%, the Full geometry remained the best exact feasible Tangent candidate and replaced the old, incorrectly labeled optimum (`A_tan: 0.4199644109 -> 0.3883788541`, 7.52% lower). At 5%, a nearby refinement improved further beyond the Full seed (`0.4199644109 -> 0.3877498844`, 7.67% lower; Full-seed `A_tan = 0.3878727661`). Full results and risk definitions were unchanged. The exact finalist receipt is `outputs/pareto/tangent_refinement_audit.json`.

### Runtime

The representative 3% point recorded `780.53 s` (`13 min 0.53 s`) total wall time:

| Phase/stage | Seconds |
|:---|---:|
| setup and cached inputs | 0.45 |
| stage 1 Population | 115.01 |
| stage 2 Law | 136.34 |
| stage 3 Tangent | 278.60 |
| stage 4 Full | 178.72 |
| validation and certification | 68.85 |

These timings are machine- and cache-dependent. They are recorded to explain the saved run, not as a portable benchmark.

## Interpretation and limitations

- **The risk/action trade-off is strong at 4%.** Allowing a certified 3.652% selection-risk increase produces a 26.70% validation-action reduction.
- **The loosest allowance is not validation-best.** The 5% design has the lowest selection action (`2.4302`), but its validation mean action (`2.4905`) is slightly higher than the 4% design (`2.4650`). Selection and validation banks are independent, so this is plausible out-of-sample variation rather than a contradiction.
- **The exact Full selection curve is nested.** Full selection action never increases as the allowed set expands: `3.1765`, `2.8258`, `2.8258`, `2.8258`, `2.4720`, `2.4302`. The runner explicitly carries forward and checks the incumbent.
- **Tangent becomes useful only after enough risk flexibility.** At 0.5–2% its designs have worse full action than Law; at 3–5% they reduce validation full action by 9.74–26.70%. The repaired 4% Tangent and Full geometries coincide; at 5% the Tangent refinement is nearby but distinct.
- **Repeated layouts are meaningful.** A looser constraint does not move the solution when the prior incumbent remains the best audited candidate.
- **Certification is selection-bank authoritative.** Validation estimates generalization and is not used to redefine the risk constraint after selection.
- **Standard errors are Monte Carlo uncertainty only.** They do not include uncertainty over the learned reference seed, truth parameters, alternative optimizer basins, or model misspecification.
- **This is a controlled oracle study.** The hidden double-gyre population and exact simulator are available for evaluation. Real deployments do not provide exact full-population MMD or exact endpoint laws.
- **The learned reference is seed- and architecture-dependent.** The repository includes a separate reference-seed sensitivity workflow; the Pareto table here uses seed `20260815` and one frozen checkpoint.
- **Native numerical backends are part of the declared method.** Changing projection or Poisson backends, grids, tolerances, or floating-point precision can change candidate ordering and must create a new configuration hash.

## Reproduction

Run from the repository root. The project requires Python `>=3.11`, JAX `>=0.4,<0.9`, NumPy `>=1.26`, Matplotlib, and the configured Tesseract C++/JAX extras.

```bash
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# 1. Generate/reuse truth, endpoint, learned-reference, and CRN artifacts.
.venv/bin/python experiments/vortices_percentage/run.py

# 2. Run or resume the common-anchor percentage sweep.
.venv/bin/python experiments/vortices_percentage/run_pareto.py \
  --percent 0.5 1 2 3 4 5 \
  --source-run experiments/vortices_percentage/outputs/run \
  --output experiments/vortices_percentage/outputs/pareto

# Use --force only to recompute otherwise compatible points.

# 3. Rebuild every table and figure from saved results without optimization.
MPLBACKEND=Agg .venv/bin/python \
  experiments/vortices_percentage/visualize_pareto.py

# 4. Re-audit saved final candidates on the frozen action bank (no optimization).
.venv/bin/python \
  experiments/vortices_percentage/audit_action_decomposition.py

# Targeted 4%/5% Tangent repair and independent correction-field audit.
.venv/bin/python experiments/vortices_percentage/refine_tangent_4_5.py
.venv/bin/python experiments/vortices_percentage/audit_orthogonal_decomposition.py
.venv/bin/python experiments/vortices_percentage/audit_common_discretization_decomposition.py
.venv/bin/python experiments/vortices_percentage/audit_corrected_full_solver_targeted.py
```

Smoke and gradient wiring checks:

```bash
.venv/bin/python experiments/vortices_percentage/verify_core.py
.venv/bin/python experiments/vortices_percentage/run.py --smoke
.venv/bin/python experiments/vortices_percentage/run_gradient_smoke.py
```

An isolated reference-seed run can be launched with:

```bash
.venv/bin/python experiments/vortices_percentage/run.py \
  --reference-seed 20260816 \
  --output-dir experiments/vortices_percentage/outputs/reference_seed_sensitivity/reference_seed_20260816
```

## Artifacts and implementation map

### Aggregate artifacts

| Artifact | Purpose |
|:---|:---|
| `outputs/pareto/pareto.csv`, `pareto.json` | one row per percentage allowance |
| `outputs/pareto/pareto_methods_selection.csv` | exact selection metrics and coordinates for Law/Tangent/Full |
| `outputs/pareto/pareto_methods_validation.csv` | validation means, SEs, reductions, and valid fractions |
| `outputs/pareto/pareto_methods_tables.md` | compact generated method tables |
| `outputs/pareto/pareto.png` | certified risk/action frontier |
| `outputs/pareto/pareto_methods.png` | four-panel method comparison |
| `outputs/pareto/pareto_sensor_layouts.png` | all sensor layouts over time-averaged occupancy |
| `outputs/pareto/experiment_sensors.png` | representative 3% scientific dashboard |
| `outputs/pareto/action_decomposition_audit.csv` | exact per-allowance/per-design Tangent/Full decomposition and hierarchy checks |
| `outputs/pareto/action_decomposition_audit.md` | human-readable decomposition table and concise PASS/FAIL summary |
| `outputs/pareto/action_decomposition_audit_summary.json` | machine-readable global hierarchy summary and frozen-bank hash |
| `outputs/pareto/action_decomposition_evaluations.json` | raw exact per-trial and per-time Tangent/Full values |
| `outputs/pareto/tangent_refinement_audit.json` | 4%/5% exact Tangent repair seeds, finalists, improvements, and immutability checks |
| `outputs/pareto/orthogonal_decomposition_audit.csv` | aggregate direct-field hidden-action and orthogonality checks |
| `outputs/pareto/orthogonal_decomposition_audit.json` | aggregate plus every per-time/per-trial direct-field diagnostic |
| `outputs/pareto/common_discretization_decomposition_audit.csv` | one row per allowance/design for the common-raster decomposition |
| `outputs/pareto/common_discretization_decomposition_audit.json` | complete aggregate, trial, and time/trial common-raster diagnostics and immutability hashes |
| `outputs/pareto/common_discretization_decomposition_audit.md` | human-readable common-raster PASS/FAIL report and tables |
| `outputs/pareto/corrected_full_solver_targeted_audit.md` | corrected physical-q 3% Law/Tangent/Full stage-gate report |
| `outputs/pareto/corrected_full_solver_targeted_audit.json` | raw corrected per-time/per-trial diagnostics, hashes, and paired-gate decision |

Each `outputs/pareto/risk_*pct/` directory includes a manifest, exact `result.json`, truth/reference/selection/validation banks, candidate summary, per-trial validation CSV, and timing JSON.

### Main implementation files

| File | Responsibility |
|:---|:---|
| `domain.py` | double-gyre velocity, initial mixture, truth simulation, endpoint source |
| `bounded_reference.py` | box-logit transformation and boundary-preserving learned reference |
| `experiment.py` | banks, spline reconstruction, I-projection, MMD, Tangent/Full actions, exact evaluation, validation |
| `selection.py` | geometry constraints, multistart optimization, exact audits, anchor refinement, four stages |
| `run.py` | full/smoke/reference-seed CLI |
| `run_pareto.py` | frozen anchor, nested incumbent, percentage sweep, aggregation |
| `visualize.py` | full single-run dashboard |
| `visualize_pareto.py` | Pareto tables, plots, atlas, and representative dashboard |
| `audit_action_decomposition.py` | frozen-candidate exact Tangent/Full decomposition audit |
| `audit_orthogonal_decomposition.py` | independent direct correction-field decomposition audit |
| `audit_common_discretization_decomposition.py` | common-raster moment feasibility, projection, nullspace, orthogonality, and energy audit |
| `audit_corrected_full_solver_targeted.py` | corrected physical-q targeted diagnostic on frozen candidates/bank |
| `refine_tangent_4_5.py` | targeted frozen-bank Tangent repair for 4% and 5% |
| `../percentage_pareto_visualization.py` | shared Law/Tangent/Full result extraction and plotting |
| `../../src/mfsi/measurements.py` | four Gaussian point-sensor features |
| `../../src/mfsi/flow_matching.py`, `reference.py` | MLP flow matching and rollout |
| `../../src/mfsi/projection.py` | empirical exponential-family I-projection |
| `../../src/mfsi/particles.py` | particle forcing and tangent MFSI |
| `../../src/mfsi/poisson.py` | weighted Poisson action |

The saved representative 3% point has configuration hash `2c1fe08102f0e869007870f3f493fe687213971fba8b0e319b7830c2058b841e`, truth signature `206d9d09bd5e9e2772db117dcb0f0a875879434ec0b056b2812ccb029805f9cd`, and manifest schema version `1`.
