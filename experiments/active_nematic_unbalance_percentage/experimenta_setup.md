# Active-nematic robust percentage-risk Pareto experiment

## Authoritative experiment setup, results, and paper record

This document records the completed production Pareto experiment in
`outputs/pareto_robust/`. It is intended to be the detailed source for a future
Methods section, results table, supplement, reproducibility statement, or
figure caption. Values are taken from the frozen run inputs and the
machine-readable authoritative outputs, not reconstructed from the compact
README table.

## 1. Experiment identity and status

| Item | Value |
|:---|:---|
| Experiment | `active_nematic_unbalance_robust_authoritative_pareto` |
| Source configuration name | `active_nematic_unbalance_percentage_two_species` |
| Pareto methodology version | 1 |
| Point-result schema | 2 |
| Final authoritative schema | 1 |
| Base random seed | 20260818 |
| Allowances | 0.5%, 1%, 2%, 3%, 4%, 5% |
| Production output | `outputs/pareto_robust/` |
| Selection completed before validation | Yes |
| Independent reconstruction/recomputation | Performed; passed |
| Final certification | **PASS** |
| Frozen input files | 30 |
| Selection failures | None |
| Validation-trial failures | None |
| Independent-recompute failures | None |

The `smoke` override block in `config.json` was not active. Every value in this
record refers to the full-resolution production configuration.

The central question was whether allowing a small, controlled increase in
finite-measure reconstruction risk relative to one common Law-optimal sensor
geometry could lower the unbalanced MFSI action, while remaining robust to both
physical-realization variation and learned-reference variation.

The primary result is that the 3% allowance is the smallest tested allowance
that reaches the selected Full-action plateau. On held-out physical views its
mean Full action is 18.157088 versus 21.761715 for Law, a 16.56% reduction.
The result is heterogeneous across held-out physical folds and is not, by
itself, a formal statistical-significance claim.

## 2. Compared design rules

All three rules optimize the same two movable sensor centers. The design vector
is

```text
eta = [x_1, y_1, x_2, y_2],    eta_i in [0, 32) periodically.
```

The two sensors are labelled during optimization, although swapping their
labels represents the same physical sensor set.

- **Law** minimizes the finite-measure RKHS risk. Its exactly audited geometry
  is frozen and shared by every allowance.
- **Tangent** minimizes a local/tangent unbalanced action subject to the same
  Law-relative risk ceiling. It is an intermediate approximation and a useful
  baseline.
- **Full** minimizes the screened full unbalanced action subject to the same
  ceiling. This is the primary Pareto design.

The selection statistic for action is the maximum over every frozen
physical/reference view. Validation reports the mean over the corresponding
held-out views, with a physical-fold jackknife standard error.

The configuration also retains the fixed geometry
`[4.0495504455, 30.5060085043, 28.4003983052, 2.8281974339]`, whose provenance
is recorded as the balanced seed-20260819 common geometry. It is marked
`evaluation only` and was not one of the three rules in this authoritative
Pareto selection/validation table; no paper comparison to that control should
be inferred from the present outputs.

## 3. Physical active-nematic model

The simulated state is a symmetric, traceless two-dimensional nematic tensor

```text
Q = [[q1, q2],
     [q2,-q1]].
```

The reduced Beris--Edwards/screened-Stokes model is

```text
partial_t Q + u . grad Q
    = S(Q, grad u) - (1/gamma) [delta F / delta Q]^TS,

eta_v Laplacian u - friction u
    = grad p + div(alpha Q),

div u = 0,

F = integral [A |Q|^2 + C |Q|^4 + L |grad Q|^2] dx.
```

Here `eta_v` denotes the physical viscosity and is distinct from the sensor
design vector `eta`. Passive nematic/back-flow stresses are omitted from the
momentum equation in this reduced model.

### 3.1 Physical and numerical parameters

| Parameter | Symbol/meaning | Value |
|:---|:---|---:|
| Grid size | `n` | 128 x 128 |
| Periodic box side | `box_size` | 32.0 |
| Bulk coefficient | `A` | -1.0 |
| Quartic coefficient | `C` | 1.0 |
| Elastic coefficient | `L` | 0.2 |
| Rotational viscosity | `gamma` | 1.0 |
| Flow-alignment coefficient |  | 0.7 |
| Fluid viscosity | `viscosity` | 1.0 |
| Substrate friction | `friction` | 0.1 |
| Activity | `activity` | 1.5 |
| Time step | `dt` | 0.02 |
| Initial director-angle noise |  | 0.18 |
| Initial amplitude noise |  | 0.03 |
| Initial smoothing length |  | 2.0 |

The solver revision is `dealiased-2x-etd2-v1`. The screened-Stokes equation is
solved mode-by-mode in Fourier space. Cubic nonlinearities are evaluated on a
2N x 2N padded grid and truncated to the resolved grid. The Q equation uses an
ETD2 update with the elastic Laplacian treated exactly. Saved physical times
are `t = 21, 22, ..., 31`, giving 11 snapshots over the evaluated interval.

## 4. Physical bank, defect state, and splits

The physical bank contains 64 independently initialized simulations:

```text
q1 shape: [64, 11, 128, 128]
q2 shape: [64, 11, 128, 128]
physical seeds: 20261819, 20261820, ..., 20261882
```

The seeds equal `base_seed + physical_bank.seed_offset + run_index`, with
`physical_bank.seed_offset = 1001`. Generation used 16 worker processes.

The deterministic train/design/validation split was created with NumPy RNG
seed `base_seed + splits.seed_offset = 20261919`:

| Split | Count | Physical-bank run indices |
|:---|---:|:---|
| Train | 32 | 31, 56, 42, 44, 57, 41, 12, 20, 54, 8, 39, 9, 2, 51, 15, 61, 11, 37, 45, 53, 29, 47, 33, 49, 63, 38, 50, 22, 58, 5, 25, 55 |
| Design | 16 | 32, 52, 14, 36, 24, 3, 40, 23, 7, 48, 34, 13, 21, 26, 43, 10 |
| Validation | 16 | 1, 28, 17, 62, 19, 35, 30, 18, 4, 60, 46, 27, 59, 16, 6, 0 |

### 4.1 Signed-defect representation

Both positive and negative defect populations are retained. Each accepted
state is `(x, y, beta)` on periods `(32, 32, 2*pi)`. For the negative species,
`beta` is a triatic phase and must not be interpreted as vector polarity. Each
physical defect has weight `1 / number_of_selected_physical_runs`, so the
finite-measure mass is the mean defect count per realization.

Defect-extraction hyperparameters were:

| Parameter | Value |
|:---|---:|
| Minimum orientation coherence, plus | 0.2 |
| Minimum orientation coherence, minus | 0.2 |
| Texture-fit inner radius | 2.0 cells |
| Texture-fit outer radius | 6.0 cells |
| Maximum core residual | Not imposed (`null`) |
| Minimum accepted species mass | 0.1 |
| Expected charge imbalance | 0.0 |
| Charge-balance tolerance | 0.25 |
| Enforce charge balance | Yes |

The final two-species bank contains 8,177 accepted states for each sign. No
defects were rejected by the configured coherence or core-residual gates. The
observed plus/minus mass imbalance is exactly zero at every saved time.

| Split | Mass at t=21 | Mass at t=31 | Range over saved times | Charge audit |
|:---|---:|---:|:---|:---|
| Train | 10.6875 | 12.34375 | [10.59375, 12.34375] | Pass |
| Design | 11.4375 | 11.875 | [11.3125, 12.4375] | Pass |
| Validation | 10.8125 | 11.75 | [10.8125, 12.0625] | Pass |
| All 64 runs | 10.90625 | 12.078125 | [10.90625, 12.078125] | Pass |

## 5. Endpoint-only learned reference

Reference shape dynamics were trained only from the train-split populations at
the endpoints `t=21` and `t=31`; no intermediate physical marginal was used in
training. Separate plus/minus periodic flow models were trained for three base
seeds. Minus-model seeds use an offset of 10,000.

The model input is periodic sine/cosine state encoding plus five time features.
The velocity network is a four-hidden-layer MLP with SiLU activations and 128
units per hidden layer. It predicts a three-dimensional velocity on
`(x,y,beta)`.

### 5.1 Reference-training hyperparameters

| Hyperparameter | Value |
|:---|:---|
| Base reference seeds | 20260818, 20260819, 20260820 |
| Plus seed offset | 0 |
| Minus seed offset | 10000 |
| Hidden width | 128 |
| Hidden layers | 4 |
| Training steps | 12,000 |
| Batch size | 2,048 |
| Optimizer | Adam |
| Adam beta1 / beta2 / epsilon | 0.9 / 0.999 / 1e-8 |
| Initial learning rate | 1e-3 |
| Final LR ratio | 0.05, cosine schedule |
| Gradient clipping norm | 10.0 |
| Bridge schedule | Linear shortest-periodic-arc bridge |
| Bridge noise standard deviation | 0.15 |
| Log interval | 500 steps |
| Endpoint samples used to form source | 50,000 |
| Reference-bank particles per species | 8,192 |
| Reference-bank position jitter | 1.0 |
| Reference-bank beta jitter | 0.25 |
| Endpoint-sampling seed offset | 2001 |
| Reference-bank sampling seed offset | 3001 |
| Rollout integrator | RK4, 16 substeps per saved-time interval |

Final logged conditional flow-matching losses were:

| Base seed | Plus loss at step 12,000 | Minus loss at step 12,000 |
|---:|---:|---:|
| 20260818 | 143.683450 | 139.831232 |
| 20260819 | 145.486556 | 141.103509 |
| 20260820 | 145.630196 | 145.222657 |

These losses are training diagnostics on the configured stochastic bridge, not
held-out scientific metrics.

The production Pareto runner reused compatible full-resolution reference
checkpoints already present in the source output after checking the relevant
configuration sections for exact equality. It copied those checkpoints,
reference banks, histories, and mass schedules into `frozen_inputs/`; it did
not silently retrain a different reference during the Pareto sweep.

### 5.2 Finite-mass reference

The normalized reference shape flow is combined with an analytic Fisher--Rao
pair-mass schedule. The training-split endpoint pair masses are 10.6875 and
12.34375, with zero charge imbalance. For normalized time `tau`, the square
root of pair mass is linearly interpolated. Both species therefore have equal
mass schedules in this run.

## 6. Measurement design and observation model

Two shared movable sensor locations are applied separately to the plus and
minus populations. Each sensor is a smooth periodic Gaussian-like window with
width 4.0. Sensor centers must have periodic chord separation at least 3.0.

Each species has six local observables: occupancy, cosine phase, and sine phase
for each of two sensors. The plus angular channels encode defect orientation;
the minus angular channels encode triatic phase. Species mass is additionally
treated as an exactly available global observation and does not count as a
third sensor.

| Observation parameter | Value |
|:---|:---|
| Movable sensors | 2 |
| Sensor width | 4.0 |
| Minimum periodic separation | 3.0 |
| Local channels per species | 6 |
| Global mass observable | Yes |
| Finite particles per acquisition | 256 per species |
| Acquisitions | 9 of 11 saved times |
| Acquisition indices | 0, 1, 2, 4, 5, 6, 8, 9, 10 |
| Acquisition times | 21, 22, 23, 25, 26, 27, 29, 30, 31 |
| Additive detector-noise standard deviation | 0.005 |
| Endpoint observation treatment | Exact at t=21 and t=31 |
| Truth particles per physical view/species | 2,048 |

Interior moment trajectories are reconstructed with an endpoint-anchored cubic
spline:

| Reconstruction hyperparameter | Value |
|:---|---:|
| Internal knots | 3 |
| Smoothing | 1e-4 |
| Relative ridge | 1e-10 |
| Roughness quadrature order | 8 |

The I-projection and particle-MFSI parameters were:

| Hyperparameter | Value |
|:---|---:|
| I-projection maximum steps | 300 |
| I-projection residual tolerance | 1e-8 |
| Newton ridge | 1e-7 |
| Step cap | 20.0 |
| Lambda clipping | 1,000 |
| Line-search steps | 8 |
| Particle covariance ridge | 1e-7 |
| Tangent ridge | 1e-7 |

## 7. Risk and action definitions

### 7.1 Finite-measure risk

For finite nonnegative measures with weighted samples `(x,w_x)` and `(y,w_y)`,
Law risk is the squared RKHS embedding distance

```text
R = w_x^T K_xx w_x + w_y^T K_yy w_y - 2 w_x^T K_xy w_y.
```

Unlike a probability-only MMD, the weights are not normalized before this
quantity is evaluated, so both distributional shape and total mass affect the
risk. Shape MMD and squared mass error are also recorded separately. The two
species are added with equal weights. Time integration uses trapezoidal weights
over `t=21,...,31`.

The periodic histogram grid is 48 x 48 x 24 on periods `(32,32,2*pi)`. The
multiscale periodic Gaussian kernel bandwidths are 0.5, 1.0, 2.0, and 4.0.
Plus and minus risk weights are both 1.0. Finite-mass smoothing is 1e-4, and
mass interpolation is Fisher--Rao as described above.

### 7.2 Full unbalanced action

For normalized shape `q`, physical mass `M`, shape residual `h_shape`, and
reference source rate `g_ref`, the unbalanced residual is

```text
h_ub = h_shape + dot(M)/M - g_ref.
```

The correction solves

```text
-div(q grad psi) + (q/kappa) psi = q h_ub,
```

corresponding to `delta = grad psi` and `alpha = psi/kappa`. The reported
species action is

```text
A_full = integral M q (|delta|^2 + kappa alpha^2),
```

and is decomposed into move and reaction terms. Species actions are added with
equal weights. The reaction parameter is `kappa = 1.0`.

Full-action numerical parameters were:

| Hyperparameter | Production value | Gradient-proxy value |
|:---|:---|:---|
| Grid | 48 x 48 x 24 | 24 x 24 x 12 |
| Backend | Native Tesseract C++/OpenMP IC(0) | Same differentiable interface |
| Native solver revision | `unbalanced-screened3d-cpp-openmp-ic0-v1` | Same |
| Raster bandwidth | 1.2 | 1.2 |
| Polarity metric radius | 1.0 | 1.0 |
| Operator floor relative to max q | 1e-14 | 1e-14 |
| CG tolerance | 1e-7 | 1e-6 |
| CG maximum iterations | 1,200 | 420 |

## 8. Percentage-risk constraint

The common Law geometry is optimized and exactly audited once. For each frozen
selection view `v`, its risk defines `R_star[v]`. At allowance `p`, Tangent and
Full must satisfy the view-specific constraint

```text
R(eta,v) <= R_star[v] + (p/100) * abs(R_star[v])
```

for all 12 selection views. Feasibility is therefore not based only on an
aggregate or worst-risk scalar. A candidate that fails any physical fold or
reference seed is rejected.

The source configuration's `law.max_relative_risk_violation = 0.05` is only a
default. `run_pareto.py` overwrites it with `p/100` independently for each of
the six point configurations before hashing and selection.

The common Law sensor centers are:

```text
sensor 1 = (28.5259155172, 4.0061079690)
sensor 2 = (24.4177052136, 9.1439526811)
```

Its worst-view risk is `R_star = 2.0222823612301424`.

## 9. Robust selection and held-out evaluation protocol

### 9.1 Physical/reference views

The 16 design runs are deterministically shuffled and divided into four folds
of four. Each selection physical view omits one fold and retains 12 runs. Each
physical view is crossed with all three learned-reference seeds, producing 12
selection views. The disjoint validation split is processed identically,
producing 12 held-out validation views.

The view-shuffle seeds are 20262019 for selection and 20262020 for validation
(`base_seed + robust_selection.seed_offset`, with an additional one for
validation). The per-view physical truth-resampling seeds are 20264918--20264921
for selection and 20265018--20265021 for validation. A given physical resample
is shared across the three reference seeds in its fold.

| Selection physical view | Included design-run indices |
|---:|:---|
| 0 | 13, 48, 21, 43, 32, 10, 7, 26, 52, 40, 24, 14 |
| 1 | 23, 3, 34, 36, 32, 10, 7, 26, 52, 40, 24, 14 |
| 2 | 23, 3, 34, 36, 13, 48, 21, 43, 52, 40, 24, 14 |
| 3 | 23, 3, 34, 36, 13, 48, 21, 43, 32, 10, 7, 26 |

| Validation physical view | Included validation-run indices |
|---:|:---|
| 0 | 35, 16, 59, 30, 0, 60, 6, 27, 4, 28, 46, 62 |
| 1 | 19, 18, 1, 17, 0, 60, 6, 27, 4, 28, 46, 62 |
| 2 | 19, 18, 1, 17, 35, 16, 59, 30, 4, 28, 46, 62 |
| 3 | 19, 18, 1, 17, 35, 16, 59, 30, 0, 60, 6, 27 |

The robust objective is the maximum metric over the complete 12-view selection
set. `upper_quantile = 0.75` remains in the configuration but is inactive
because the selected objective is `max`.

### 9.2 Common random numbers and evaluation ordering

Selection and validation use independent, frozen observation banks:

| Bank | Namespace | Trials | Shape per species |
|:---|---:|---:|:---|
| Selection | 9890 | 32 | `[32, 9, 256]` sample indices and `[32, 9, 6]` detector noise |
| Validation | 9891 | 32 | `[32, 9, 256]` sample indices and `[32, 9, 6]` detector noise |

The RNG is initialized with `SeedSequence([base_seed, namespace])`. The same
bank is reused across geometries and views to provide common random numbers.
Every allowance winner was frozen before any validation evaluation began.

Each selected validation design therefore has
`4 physical folds x 3 references x 32 trials = 384` trial rows. Across six
allowances and three rules, the serialized validation record contains 6,912
valid trial rows.

### 9.3 Candidate generation and exact certification

Differentiable candidate generation uses only physical view 0 crossed with all
three reference seeds (`gradient_physical_views = 1`). This reduces optimizer
cost but does not weaken selection: generated candidates are screened and
rescored across all 12 views using the complete 32-trial selection bank.

The stages are:

1. Optimize Law at the first allowance; exactly audit candidates and freeze the
   winning Law geometry for all later allowances.
2. Optimize Tangent under the normalized proxy risk constraint; explicitly
   reinsert the Law incumbent; screen risk and exactly audit Tangent action.
3. Optimize Full on the lower-resolution proxy; explicitly reinsert Law,
   Tangent, and the previous Pareto Full incumbent; run an exact Full
   pre-screen; rescore finalists on the production grid and complete bank.
4. Require the Full selection curve to be non-increasing within tolerance.
5. Only after all six points pass selection, evaluate Law, Tangent, and Full on
   the held-out validation views.

Optimizer endpoints are not assumed to preserve starts. Incumbents are added
back as explicit candidates before exact certification.

## 10. Optimization hyperparameters

All direct searches use Adam with beta1 0.9, beta2 0.999, and epsilon 1e-8.
Objectives are scaled by the relevant Law-incumbent value. Constraint
violations receive a quadratic penalty.

| Hyperparameter | Law | Tangent | Full |
|:---|---:|---:|---:|
| Adam steps | 60 | 50 | 40 |
| Learning rate | 0.008 | 0.006 | 0.004 |
| Gradient observation trials | 16 | 16 | 4 |
| Global starts used | 8 | 6 | 6 |
| Local starts | 0 | 8 around Law | 10 around Law/Tangent |
| Local perturbation scale | -- | 1.5 | 1.5 |
| Exact audit candidate budget | 12 | 16 | 16 |
| Exact Full pre-screen trials | -- | -- | 8 |
| Exact Full rescore finalists | -- | -- | 4 |

Shared settings:

| Hyperparameter | Value |
|:---|---:|
| Random global-start pool | 24 |
| Global-start oversampling | 96 |
| Constraint penalty | 10,000 |
| Feasibility tolerance | 1e-6 |
| Global-start RNG key | `base_seed + 17` |
| Tangent local-cloud seed | `base_seed + 401` |
| Full local-cloud seed | `base_seed + 501` |
| Pareto nesting tolerance | 1e-8 |

The number of serialized, exactly audited candidates varied because duplicate
geometries are removed and only feasible candidates survive the staged screens:

| Allowance | Law candidates | Tangent candidates | Full candidates |
|---:|---:|---:|---:|
| 0.5% | 12 | 1 | 1 |
| 1% | 1 | 6 | 6 |
| 2% | 1 | 12 | 6 |
| 3% | 1 | 16 | 7 |
| 4% | 1 | 16 | 6 |
| 5% | 1 | 16 | 6 |

## 11. Numerical validity gates

Every exact trial must satisfy:

| Gate | Required threshold | Worst observed selection | Worst observed validation |
|:---|---:|---:|---:|
| Maximum calibration residual | <= 1e-3 | 9.99993e-9 | 9.99980e-9 |
| Minimum ESS fraction | >= 0.03 | 0.741794 | 0.773776 |
| Maximum screened-PDE relative residual | <= 1e-5 | 9.999999e-8 | 9.999983e-8 |

There are 89,088 valid serialized selection-candidate trial rows and 6,912
valid validation rows. PDE residuals are applicable to 12,288 of the selection
rows and all validation rows. No recorded row failed its applicable gates.

The authoritative finalizer additionally verifies:

- exact selected-audit receipts for Law, Tangent, and Full;
- every view-specific percentage-risk ceiling;
- the common Law geometry and common per-view risk anchors;
- nested Full selection action;
- move plus reaction equals total action with the configured species weights;
- finite values in every required validation metric;
- selection-before-validation ordering;
- sizes and SHA-256 hashes of all frozen inputs; and
- independent reconstruction and recomputation of every unique selected
  geometry on both selection and validation data.

## 12. Selected sensor geometries

Coordinates are periodic on `[0,32)`.

| Allowance | Rule | Sensor 1 `(x,y)` | Sensor 2 `(x,y)` |
|---:|:---|:---|:---|
| all | Law | (28.525916, 4.006108) | (24.417705, 9.143953) |
| 0.5% | Tangent | (28.525916, 4.006108) | (24.417705, 9.143953) |
| 0.5% | Full | (28.525916, 4.006108) | (24.417705, 9.143953) |
| 1% | Tangent | (17.399722, 16.101378) | (27.637676, 4.924615) |
| 1% | Full | (16.233787, 15.779209) | (24.022979, 4.070354) |
| 2% | Tangent | (0.924288, 21.885475) | (29.666521, 2.198670) |
| 2% | Full | (31.714974, 22.792602) | (26.176669, 2.750354) |
| 3%, 4%, 5% | Tangent | (3.826467, 7.610768) | (22.666931, 21.619448) |
| 3%, 4%, 5% | Full | (4.572898, 6.594879) | (24.129322, 21.257897) |

The repeated 3--5% geometry is caused by explicit Pareto-incumbent retention:
no newly generated feasible candidate at 4% or 5% beat the exactly certified
3% incumbent.

## 13. Selection results

All action columns below use the production Full-action evaluator, including
the Tangent geometry. `Full risk` is the maximum risk over the 12 selection
views. The selection action is also a 12-view maximum.

| Allowance | Worst-view ceiling | Law Full action | Tangent-geometry Full action | Selected Full action | Full risk | Full action reduction vs Law |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5% | 2.032394 | 29.377607 | 29.377607 | 29.377607 | 2.022282 | 0.00% |
| 1% | 2.042505 | 29.377607 | 25.393960 | 23.037196 | 2.029423 | 21.58% |
| 2% | 2.062728 | 29.377607 | 23.956776 | 22.529704 | 2.037132 | 23.31% |
| 3% | 2.082951 | 29.377607 | 21.784316 | 21.552532 | 2.073642 | 26.64% |
| 4% | 2.103174 | 29.377607 | 21.784316 | 21.552532 | 2.073642 | 26.64% |
| 5% | 2.123396 | 29.377607 | 21.784316 | 21.552532 | 2.073642 | 26.64% |

Because every ceiling is view-specific, the displayed worst-view ceiling and
worst Full risk need not come from the same view. The exact per-view checks are
authoritative. The maximum percentage allowance actually used by the selected
Full design was 0%, 0.8828%, 0.8376%, 2.7773%, 2.7773%, and 2.7773% at the six
respective sweep points.

The exact Full selection-action differences between consecutive points are

```text
[-6.3404116236, -0.5074912598, -0.9771726303, 0.0, 0.0],
```

so the nested selection gate passes.

## 14. Authoritative held-out results

`View SE` is the leave-one-physical-fold-out jackknife standard error after
first averaging the three learned-reference seeds within each physical fold.
It is not the much smaller standard error over the 384 conditional observation
trials.

| Allowance | Law action | Tangent action +/- view SE | Tangent vs Law | Full action +/- view SE | Full vs Law |
|---:|---:|---:|---:|---:|---:|
| 0.5% | 21.761715 | 21.761715 +/- 5.361181 | 0.00% | 21.761715 +/- 5.361181 | 0.00% |
| 1% | 21.761715 | 21.759648 +/- 6.132718 | 0.01% | 20.869286 +/- 4.080682 | 4.10% |
| 2% | 21.761715 | 20.813186 +/- 6.090054 | 4.36% | 22.228235 +/- 8.968947 | -2.14% |
| 3% | 21.761715 | 18.779569 +/- 1.693147 | 13.70% | 18.157088 +/- 2.117093 | 16.56% |
| 4% | 21.761715 | 18.779569 +/- 1.693147 | 13.70% | 18.157088 +/- 2.117093 | 16.56% |
| 5% | 21.761715 | 18.779569 +/- 1.693147 | 13.70% | 18.157088 +/- 2.117093 | 16.56% |

The 2% point demonstrates why held-out evaluation is required: selection Full
action improves by 23.31%, but held-out mean action is 2.14% worse than Law.

Held-out finite-measure risks are shown for context. These are evaluation
metrics, not additional validation-side constraints:

| Allowance | Law risk | Tangent-geometry risk | Full-geometry risk |
|---:|---:|---:|---:|
| 0.5% | 1.939146 | 1.939146 | 1.939146 |
| 1% | 1.939146 | 1.955166 | 1.952588 |
| 2% | 1.939146 | 1.953143 | 1.953941 |
| 3% | 1.939146 | 1.947203 | 1.955304 |
| 4% | 1.939146 | 1.947203 | 1.955304 |
| 5% | 1.939146 | 1.947203 | 1.955304 |

### 14.1 Full-action decomposition

| Allowance | Full move action | Full reaction action | Reaction fraction | Plus action | Minus action |
|---:|---:|---:|---:|---:|---:|
| 0.5% | 3.485962 | 18.275754 | 83.98% | 12.862158 | 8.899557 |
| 1% | 3.348549 | 17.520737 | 83.95% | 12.313452 | 8.555834 |
| 2% | 3.417133 | 18.811101 | 84.63% | 12.286516 | 9.941719 |
| 3% | 3.063576 | 15.093512 | 83.13% | 9.816671 | 8.340418 |
| 4% | 3.063576 | 15.093512 | 83.13% | 9.816671 | 8.340418 |
| 5% | 3.063576 | 15.093512 | 83.13% | 9.816671 | 8.340418 |

At 3%, both parts of Full action are lower than Law:

| Geometry evaluated with Full action | Move | Reaction | Total | Plus total | Minus total |
|:---|---:|---:|---:|---:|---:|
| Law | 3.485962 | 18.275754 | 21.761715 | 12.862158 | 8.899557 |
| Tangent | 3.133688 | 15.645880 | 18.779569 | 10.423485 | 8.356084 |
| Full | 3.063576 | 15.093512 | 18.157088 | 9.816671 | 8.340418 |

### 14.2 Held-out physical-fold heterogeneity at 3%

The following values average the three reference seeds within each held-out
physical view:

| Validation physical view | Law | Tangent | Full | Full vs Law |
|---:|---:|---:|---:|---:|
| 0 | 23.406895 | 19.423751 | 18.119957 | 22.59% |
| 1 | 18.919003 | 20.034317 | 20.163083 | -6.58% |
| 2 | 26.022336 | 17.952573 | 17.093586 | 34.31% |
| 3 | 18.698627 | 17.707634 | 17.251727 | 7.74% |

Thus Full improves three of four physical-fold summaries but is worse in view
1. At the 12 physical/reference-view level, Full is better in 9 of 12 views.
This heterogeneity is the main reason the physical-view uncertainty must be
reported.

### 14.3 Per-view 3% validation action

| Physical fold | Reference seed | Law | Tangent | Full |
|---:|---:|---:|---:|---:|
| 0 | 20260818 | 23.734833 | 19.526419 | 18.258176 |
| 0 | 20260819 | 23.452800 | 19.321262 | 18.055900 |
| 0 | 20260820 | 23.033052 | 19.423573 | 18.045795 |
| 1 | 20260818 | 19.131436 | 20.289811 | 20.417725 |
| 1 | 20260819 | 18.990000 | 19.736942 | 19.859571 |
| 1 | 20260820 | 18.635575 | 20.076197 | 20.211953 |
| 2 | 20260818 | 26.532218 | 18.085306 | 17.251892 |
| 2 | 20260819 | 26.114671 | 17.742444 | 16.879066 |
| 2 | 20260820 | 25.420118 | 18.029969 | 17.149799 |
| 3 | 20260818 | 18.860337 | 17.887767 | 17.405168 |
| 3 | 20260819 | 18.700309 | 17.378929 | 16.928034 |
| 3 | 20260820 | 18.535235 | 17.856205 | 17.421980 |

### 14.4 Per-view risk receipt for the selected 3% Full geometry

| Physical fold | Reference seed | Law anchor risk | 3% ceiling | Selected Full risk | Slack |
|---:|---:|---:|---:|---:|---:|
| 0 | 20260818 | 2.020998 | 2.081628 | 2.022151 | 0.059477 |
| 0 | 20260819 | 2.014408 | 2.074840 | 2.022475 | 0.052366 |
| 0 | 20260820 | 2.017356 | 2.077877 | 2.030514 | 0.047363 |
| 1 | 20260818 | 1.990505 | 2.050220 | 2.027001 | 0.023219 |
| 1 | 20260819 | 1.990747 | 2.050469 | 2.025867 | 0.024602 |
| 1 | 20260820 | 1.988118 | 2.047762 | 2.034068 | 0.013694 |
| 2 | 20260818 | 1.918701 | 1.976262 | 1.953956 | 0.022306 |
| 2 | 20260819 | 1.916657 | 1.974157 | 1.958838 | 0.015319 |
| 2 | 20260820 | 1.910001 | 1.967301 | 1.955696 | 0.011605 |
| 3 | 20260818 | 2.020211 | 2.080818 | 2.064927 | 0.015891 |
| 3 | 20260819 | 2.022282 | 2.082951 | 2.070803 | 0.012148 |
| 3 | 20260820 | 2.017606 | 2.078134 | 2.073642 | 0.004492 |

All 12 view-specific constraints pass. The most heavily used allowance is
2.7773%, below the declared 3% ceiling.

## 15. Interpretation suitable for a paper

The defensible primary interpretation is:

> Under the frozen robust protocol, the 3% Law-relative risk allowance was the
> smallest tested allowance that attained the minimum selected Full action. Its
> sensor geometry reduced mean held-out Full action by 16.56% relative to the
> common Law geometry, with a leave-one-physical-fold-out jackknife SE of
> 2.1171 action units for the Full-action estimate. The improvement was
> heterogeneous, occurring in three of four held-out physical-fold summaries.

The following stronger statements are not supported by this experiment alone:

- that Full is better for every physical realization or physical fold;
- that the 16.56% reduction is statistically significant under a specified
  population-level hypothesis test;
- that 3% is globally optimal outside the tested allowance grid;
- that the result generalizes to other activity, friction, box size, sensor
  count, acquisition count, or noise regimes; or
- that reference-seed uncertainty has been exhausted with three seeds.

The 3--5% plateau should be described as a plateau within this candidate search
and exact incumbent-retention protocol, not proof that no better 4% or 5%
geometry exists.

## 16. Reproduction commands

From the repository root:

```bash
.venv/bin/python experiments/active_nematic_unbalance_percentage/run.py physical-bank \
  --output-dir experiments/active_nematic_unbalance_percentage/outputs/source

.venv/bin/python experiments/active_nematic_unbalance_percentage/run.py defects \
  --output-dir experiments/active_nematic_unbalance_percentage/outputs/source

.venv/bin/python experiments/active_nematic_unbalance_percentage/run.py defect-audit \
  --output-dir experiments/active_nematic_unbalance_percentage/outputs/source

.venv/bin/python experiments/active_nematic_unbalance_percentage/run.py reference \
  --output-dir experiments/active_nematic_unbalance_percentage/outputs/source

.venv/bin/python experiments/active_nematic_unbalance_percentage/run_pareto.py \
  --input-dir experiments/active_nematic_unbalance_percentage/outputs/source \
  --output experiments/active_nematic_unbalance_percentage/outputs/pareto_robust

.venv/bin/python experiments/active_nematic_unbalance_percentage/finalize_authoritative_pareto.py \
  --pareto-dir experiments/active_nematic_unbalance_percentage/outputs/pareto_robust
```

The default finalizer performs independent recomputation. `--skip-recompute`
is intended only for quick smoke checks and must not be used to create an
authoritative paper result.

Focused tests:

```bash
.venv/bin/python -m pytest \
  experiments/active_nematic_unbalance_percentage/test_percentage_budget.py \
  experiments/active_nematic_unbalance_percentage/test_robust_selection.py \
  experiments/active_nematic_unbalance_percentage/test_pareto_helpers.py
```

At the time of this record, all 12 focused tests passed.

## 17. Software and execution environment

| Item | Recorded value |
|:---|:---|
| Repository branch | `main` |
| Repository HEAD during documentation | `5a73a9fe91bc47087f54ba2439738707abdc5a56` |
| Working-tree qualification | Experiment changes were not yet committed; use source hashes below in addition to HEAD |
| Python | 3.12.3 |
| NumPy | 2.5.1 |
| SciPy | 1.18.0 |
| JAX / jaxlib | 0.8.3 / 0.8.3 |
| JAX backend observed during run/finalization | CPU (`TFRT_CPU_0`) |
| Host | Linux 6.6.87.2 under WSL2, x86_64 |
| CPU | Intel Core Ultra 9 275HX, 24 visible CPU cores |

CUDA plug-in initialization emitted a nonfatal error and JAX fell back to CPU.
The run and independent finalizer completed successfully on that backend. Wall
clock time and peak memory were not stored in the current artifact schema and
should be added to the manifest for future computational-cost reporting.

Key source-file SHA-256 values recorded with this document:

| Source file | SHA-256 |
|:---|:---|
| `active_nematic_solver.py` | `84c93c00b68fca5256fa11457f9f9ff6ba05ae0135a0eb93c44298895aaaac6a` |
| `percentage_selection.py` | `060d8ec8d1ddc8f43ec331f6ae8af89cc6b4ebae87a650252e23704589714b2f` |
| `robust_selection.py` | `8f47a5317f69d1f3a92638f992f40012dcb15960a09f2f47ddd797349837790c` |
| `run_pareto.py` | `ef43ffa5da333512c6d976d51fdbc17a14fccad1b590d0cfc675308148d7836b` |
| `finalize_authoritative_pareto.py` | `81a096478dcf27b45b42c0f1218e2b1a0b03292ce0a43c334f64d7c364a45044` |
| `unbalanced_experiment.py` | `8f97b74f7ae2b2b4db13a071247a7682449b14a37e03e239bf999d897ea5daf2` |
| `unbalanced_correction.py` | `f9aa13df12f94312418273d33a2cecf0726223482e33f332b1ed2d2c8e7e9207` |

## 18. Artifact and provenance index

| Artifact | Role |
|:---|:---|
| `outputs/pareto_robust/frozen_inputs/effective_config.json` | Exact production configuration before per-point allowance/incumbent injection |
| `outputs/pareto_robust/frozen_inputs/view_manifest.json` | Exact train/design/validation indices and robust views |
| `outputs/pareto_robust/frozen_inputs/manifest.json` | Sizes and SHA-256 hashes for all 30 frozen inputs |
| `outputs/pareto_robust/risk_*pct/result.json` | Candidate audits, selected designs, and validation trials for each allowance |
| `outputs/pareto_robust/pareto.json` | Sweep checkpoint summary |
| `outputs/pareto_robust/authoritative_pareto.json` | Final machine-readable paper table |
| `outputs/pareto_robust/authoritative_pareto.csv` | Final tabular export |
| `outputs/pareto_robust/authoritative_pareto.md` | Compact human-readable report |
| `outputs/pareto_robust/authoritative_pareto.png` | Pareto/action/decomposition figure |
| `outputs/pareto_robust/authoritative_certification_diagnostic.json` | Fail-closed certification receipt |

Important artifact hashes:

| Artifact | SHA-256 |
|:---|:---|
| Frozen-input manifest | `b461147557db26baf72f3082d0f36699da335c248343f53cb0b5de4ef6a62a7e` |
| Pareto manifest | `5c425394452e47eb9660582264b75c6a2ac6346b73328ddb9db3a20c9639c8d7` |
| Pareto checkpoint JSON | `8ef7b8c4cf27d0ee8ee07f33b40d67229ec04afc5d5646f61092bd4645bbd2a0` |
| Authoritative Pareto JSON | `828d2bc5ef451ec2ed9f4762e272e3793b6d79ad25865cb60eb64590ff2b4d43` |
| Certification diagnostic | `3ffe469db507d4fd47d9176f8d196f2d6b4e07bc5499e8976fe4dacc2f5c0a41` |

Per-point effective configuration hashes are:

| Allowance | Configuration SHA-256 fingerprint |
|---:|:---|
| 0.5% | `3c8d79ba2192b4496e72b1c3568d5c6968366273334635bfe56cf69fab532d29` |
| 1% | `5dc28a3da22edad6a55fac2e2cc31795bebc13b6404d0e0df7cf358433e387d4` |
| 2% | `68e78c21e9d42af79234a0d2c449a3489364c4a01594f5d3fb7e41ad1aacef69` |
| 3% | `9c8820998f12942120a9eddce814fe538d4734859973366deafd8f0ed3e72bcc` |
| 4% | `a65f219fcb2bcf7d21d6932b2dc6c6e97022a0da22d437fe6b2c6c4ebf8a0f2d` |
| 5% | `e68708cd6bd840c79d61c729024f580498bc06ff6ef3344ce2fe7dd48fba54f2` |

## 19. Suggested figure caption

> Robust percentage-risk Pareto evaluation for two-species active-nematic
> sensing. Sensor geometries were selected by minimizing the worst Full action
> over four leave-one-physical-fold-out design views crossed with three
> independently trained endpoint-reference seeds, subject to a separate
> Law-relative risk ceiling in every view. Held-out values were computed only
> after the complete allowance sweep was frozen and use a disjoint set of 16
> physical simulations. Error bars show leave-one-physical-fold-out jackknife
> standard errors after reference-seed averaging. The 3% allowance is the first
> point on the selected Full-action plateau and reduces mean held-out action by
> 16.56% relative to Law; the 2% selection improvement reverses on held-out
> views.

## 20. Known limitations and next paper-facing checks

1. Only one physical parameter regime was evaluated.
2. Four physical folds and three learned-reference seeds provide a robust
   diagnostic but limited population-level uncertainty resolution.
3. The fold views overlap by construction; the reported uncertainty uses the
   matching leave-one-fold-out jackknife and should not be replaced by an IID
   standard error over the 12 views or 384 conditional trials.
4. Validation mean action is not constrained by the selection risk ceiling;
   the ceiling is a selection-side design constraint.
5. The 2% reversal and the 3% fold-1 reversal should be retained in any
   supplement or robustness discussion.
6. A future confirmatory run should preregister 3% as the operating point and
   evaluate it on a newly generated physical bank without repeating selection.
7. Future manifests should record start/end timestamps, peak memory, CPU/GPU
   backend details, and a committed source revision.

This document describes the production run only. Historical runs under
`outputs/run/` and `outputs/run_strengthened_seed18/` are diagnostics and must
not be substituted for the authoritative artifacts above.
