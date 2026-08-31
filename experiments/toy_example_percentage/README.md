# Analytical Gaussian-Mixture Transport: Percentage-Risk Sensor Design

> **Naming convention.** This directory retains its historical repository name, `toy_example_percentage`. In the [project README](../../README.md) and the [paper](../../full_report.pdf), it is the **analytical Gaussian-mixture experiment**. **FIDE** denotes the complete design framework; **Full** is the FIDE-selected method in tables and figures because it minimizes the Full law-level action. Law and Tangent are the comparison methods.

This controlled benchmark asks where to place two localized sensors when the experiment reveals only aggregate information about an evolving population. Among sensor pairs that remain near-optimal for the finite-data scientific task, FIDE favors the pair whose complete measurement-implied law is easiest to realize relative to one shared, frozen endpoint-trained reference flow.

The benchmark is analytic on purpose: the hidden population path is available to us for evaluation, while the simulated experiment receives only noisy sensor averages. This makes it possible to distinguish what the sensors actually identify from what the benchmarker knows about the hidden truth.

## Experiment narrative

### The hidden system

The state is $x=(x_1,x_2)\in[-3.2,3.2]^2$. For $d(\alpha)=(\cos\alpha,\sin\alpha)$, let $g_\alpha$ be an equally weighted pair of isotropic Gaussian lobes centered at $\pm1.5d(\alpha)$ with standard deviation $0.3$. The population follows

$$
\rho_t^\alpha = (1-t)^2g_0+2t(1-t)g_\alpha+t^2g_{\pi/2},
\qquad t\in[0,1].
$$

The lobes begin horizontally separated, pass through an uncertain intermediate orientation $\alpha\in[30^\circ,60^\circ]$, and finish vertically separated. Only the common endpoint laws are supplied to the neural reference flow. The reference is trained once and frozen; its intermediate path is a common dynamical background, not a claim about the hidden intermediate dynamics.

### What the experiment observes

Two Gaussian sensors are constrained to a radius-$1.5$ ring. A sensor at angle $\theta_j$ has width $0.45$ and returns the population average $c_j(t)=\mathbb E[\Phi_j(X_t;\theta_j)]$. Each finite trial estimates these two averages from 100 particles at 11 acquisition times and adds detector noise with standard deviation $0.01$.

Consequently, the experiment does not observe a density. At each time it identifies only a moment fiber: the family of probability laws matching the two sensor averages. The complete density shown in the visualizations is available only because this is a controlled validation problem.

### From aggregate measurements to a complete law

The 11 noisy observations are reconstructed over 21 scientific time nodes. At every node, FIDE information-projects the same frozen reference law onto the reconstructed sensor moments. The result is the unique maximum-entropy exponential tilt of that reference at the law level.

This projected law is **measurement-implied**, not asserted to be the hidden truth. The observations determine the directions that must be matched; the frozen reference completes everything the two sensors leave unresolved. Full action then measures the minimum kinetic correction needed for this entire projected law path, not merely its two measured moments, to evolve consistently relative to the frozen reference velocity.

### What is being compared

The design variable is the sensor-angle pair $\eta=(\theta_1,\theta_2)$. All methods use the same reference, observations, risk definitions, selection bank, and independent validation bank.

| Method        | What selects the sensor geometry                               | Role in the experiment                                                           |
| :------------ | :------------------------------------------------------------- | :------------------------------------------------------------------------------- |
| **Law**       | Minimum finite-data scientific risk                            | Freezes the risk anchor and supplies the baseline geometry                       |
| **Tangent**   | Minimum correction visible through the two sensor-moment rates | Tests whether moment-level compatibility is sufficient                           |
| **Full/FIDE** | Minimum action of the complete information-projected law       | Tests law-level dynamical compatibility inside the scientifically admissible set |

For allowance $p\in\{0.5,1,2,3,4,5\}$, Tangent and Full must pass the same population screen and satisfy

$$
R(\eta)\leq\left(1+\frac{p}{100}\right)R_{\mathrm{Law}}.
$$

Full action cannot compensate for an uninformative experiment: it ranks only sensor pairs that have already passed the scientific-risk restrictions. Final geometries are frozen on the selection bank before being evaluated on 128 disjoint validation trials.

### Animated walkthrough

![Animated hidden population, corrected law, and sensor views](figures/toy_population_correction_sensors.gif)

The left panel is the analytic hidden population. The center panel is the maximum-entropy correction of the frozen endpoint reference that matches the two reconstructed sensor readings. The two panels on the right show the spatial contribution seen by each sensor and report its scalar response. The colored crosses mark sensor centers; dashed circles show one sensor width.

Differences between the hidden and corrected densities away from the sensor supports are expected. They visualize the unresolved directions of the moment fiber rather than a failed constraint: the corrected law must reproduce the two aggregate readings, not recover the hidden density pointwise. FIDE asks how dynamically expensive this reference-completed law is to realize.

## Authority and scope

The current results are the fresh, nested, corrected Full-design sweep produced with the accepted positive-support physical-`q_h` evaluator. Population, Law, Tangent, the endpoint reference model, observation banks, reconstruction, information projection, and risk definitions are frozen.

The authoritative corrected result is **PASS**. The Full selection curve is nonincreasing over `0.5%, 1%, 2%, 3%, 4%, 5%`, every selection and validation certificate passes, and the common-raster decomposition is resolved without clipping. The pre-correction/mixed result tree is not a current scientific result. It is intentionally not tracked at `outputs/old/pareto_pre_corrected_full/`; the reproduction section below reconstructs its seed state from repository commit `9250d083890fbc5b9938d46210b733b491a849f4` before a full search replay.

The publication result is the corrected percentage sweep under `outputs/pareto/`. In particular:

- `outputs/pareto/corrected_nested_full_sweep.json` freezes the selected Full geometries and their independent Full validation trials;
- `outputs/pareto/authoritative_run_summary.json` is the fail-closed global receipt;
- `outputs/pareto/positive_raster_decomposition_diagnostics.json` certifies Law, Tangent, and Full in the common raster space; and
- `outputs/run/result.json` is only the earlier source/base run used to freeze Population, Law, Tangent, the reference, and the banks. Its historical Full-action values are not publication values.

The saved artifacts establish a numerical multistart certificate under the declared candidate protocol. They are not an analytic proof of global optimality. The `smoke` section of `config.json` is a wiring-test profile and was not used for the production result.

## Headline result: risk-controlled action reduction

| Allowance | Full sensor angles     | Exact `L` | Exact `R` | Risk increase | Selection `A_full,h` | Validation `A_full,h` ± SE | Reduction vs Law | `A_tan,h` | `A_hid,h` | `Gamma_h` |
| --------: | :--------------------- | --------: | --------: | ------------: | -------------------: | -------------------------: | ---------------: | --------: | --------: | --------: |
|      0.5% | `(23.0868°, 69.0609°)` | 0.0687594 | 0.0662119 |       0.0395% |              27.5740 |           26.6186 ± 0.8560 |            8.26% |    0.7194 |   26.8545 |    0.9739 |
|        1% | `(20.7917°, 71.6743°)` | 0.0691746 | 0.0666986 |       0.7749% |              21.0403 |           20.3103 ± 0.6338 |           30.00% |    0.8790 |   20.1613 |    0.9582 |
|        2% | `(21.1452°, 72.0279°)` | 0.0691929 | 0.0666835 |       0.7521% |              20.3224 |           19.7470 ± 0.5423 |           31.94% |    0.8767 |   19.4458 |    0.9569 |
|        3% | `(21.4988°, 72.3814°)` | 0.0692175 | 0.0666746 |       0.7386% |              19.6714 |           19.2431 ± 0.4611 |           33.68% |    0.8744 |   18.7970 |    0.9556 |
|        4% | `(21.6755°, 72.5582°)` | 0.0692321 | 0.0666724 |       0.7353% |              19.3703 |           19.0131 ± 0.4253 |           34.47% |    0.8732 |   18.4971 |    0.9549 |
|        5% | `(21.6755°, 72.5582°)` | 0.0692321 | 0.0666724 |       0.7353% |              19.3703 |           19.0131 ± 0.4253 |           34.47% |    0.8732 |   18.4971 |    0.9549 |

The consecutive corrected selection-action changes are

```text
-6.5336911, -0.7178278, -0.6510294, -0.3010833, 0.0.
```

Thus `A_full(0.5%) >= A_full(1%) >= ... >= A_full(5%)` passes at tolerance `1e-6`. At 5%, no feasible audited candidate improved on the 4% incumbent beyond tolerance, so the repeated endpoint is intentional.

The tracked [complete method tables](outputs/pareto/pareto_methods_tables.md) give all 18 Law/Tangent/Full selection rows and all 18 independent validation rows, including exact risk use, action, SE, reduction, and valid fraction. The common Law geometry is `(23.384916°, 67.951787°)`; Tangent is `(24.439611°, 67.280877°)` at 0.5% and `(25.165658°, 64.146445°)` at 1–5%. Tangent is worse than Law in held-out Full action at every allowance, whereas corrected Full is better at every allowance. This is why particle Tangent action and corrected Full action must not be interpreted as interchangeable objectives.

### Result figures

![Law, Tangent, and Full comparison](outputs/pareto/pareto_methods.png)

This is the main quantitative comparison. Panel A expresses corrected Full action as a percentage of Law action on the same bank: `100%` is the Law baseline, solid lines are selection, and dashed lines are independent validation. Full improves from about `92%` of Law action at 0.5% allowance to about `65%` at 4–5%, while the Tangent-selected geometry is more expensive than Law in this common Full metric. Panel B shows how much of each allowed finite-risk budget is actually used. Only the solid selection curves are constrained by the allowance; dashed validation risk is an out-of-sample diagnostic, not a second selection criterion.

![Sensor positions across allowances](outputs/pareto/pareto_sensor_layouts.png)

Columns increase the allowance and rows compare the three objectives over the representative midpoint population. The dashed circle is the admissible sensor ring, numbered points are sensor centers, and thin circles show one sensor width. Law is fixed across the sweep; Tangent changes once between 0.5% and 1%; Full moves gradually toward its 4% geometry and then remains unchanged at 5%. The atlas shows that the action gain comes from a small but systematic geometric rebalancing, not from moving sensors to a different spatial scale.

![Experiment and representative sensor layout](outputs/pareto/experiment_sensors.png)

This dashboard explains one representative 3% point rather than all six allowances. Panel A shows the analytic path at `alpha=45°`; panel B overlays the Law, Tangent, and Full sensor pairs; panel C places their exact selection risk and corrected Full action in the same trade-off plane; and panel D shows independent validation means with 95% normal intervals. Its central visual message is that the nearby Full geometry sharply lowers Full action, whereas the Tangent geometry optimizes a different local quantity and performs poorly under the common corrected action.

### Paper observation-mechanism figure

![Hidden population, corrected law, and sensor views](figures/toy_population_correction_sensors.png)

`visualize_paper.py` uses the authoritative 5% Full geometry and the frozen reference/validation banks. It selects the frozen validation trial closest to the representative `45°` nuisance angle and displays four acquisition nodes:

- the analytic hidden population;
- the endpoint-reference law after maximum-entropy correction to the two saved sensor observations; and
- one sensor-weighted spatial view per sensor, annotated with the corresponding observed scalar value.

Read the columns as snapshots of one frozen observation trial. The upper row is the hidden law, the middle row is the maximum-entropy correction of the endpoint-only reference, and the bottom views show the spatial regions that contribute to each scalar sensor reading. The corrected law is required to match those moments, not to reconstruct the hidden density pointwise; visible differences away from the sensor supports are therefore expected and help illustrate why a large tangent-invisible correction can remain.

The figure is post-processing only: it performs no training, design optimization, or new validation. From the repository root, regenerate both the raster and vector versions with

```bash
.venv/bin/python experiments/toy_example_percentage/visualize_paper.py
```

This writes `figures/toy_population_correction_sensors.png` and `figures/toy_population_correction_sensors.pdf`. Use `--output-stem` to change the common destination while retaining both formats.

## Scientific setup

### Hidden population

The state is `x=(x_1,x_2)` on `[-3.2,3.2]^2`. Let

```text
d(alpha) = (cos(alpha), sin(alpha)),
g_alpha(x) = 1/2 N(x; 1.5 d(alpha), 0.3^2 I)
           + 1/2 N(x;-1.5 d(alpha), 0.3^2 I).
```

For normalized time `t in [0,1]`, the analytic hidden path is

```text
rho_t^alpha = (1-t)^2 g_0 + 2t(1-t) g_alpha + t^2 g_(pi/2).
```

The nuisance orientation `alpha` is uniformly averaged over `30–60°` using five-point Gauss–Legendre quadrature. The path moves from horizontal antipodal lobes to vertical antipodal lobes, with uncertain interior curvature.

### Sensors and observations

Two Gaussian sensors are constrained to the radius-`1.5` ring. For sensor angle `theta_j`,

```text
s_j = 1.5 (cos(theta_j), sin(theta_j)),
Phi_j(x;theta_j) = exp(-||x-s_j||^2 / (2 * 0.45^2)).
```

Angles are canonicalized and sorted. Their projective separation must be at least `20°`. Each finite observation trial uses 100 particles at 11 acquisition nodes nested in the 21-node scientific grid, detector noise standard deviation `0.01`, and exact endpoint moments. Selection and validation use disjoint frozen banks.

The scientific nodes are `t_i=i/20`, `i=0,...,20`. The acquisition indices are

```text
0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20.
```

Thus the finite-law risk is evaluated at the ten non-acquisition interior indices

```text
1, 3, 5, 7, 9, 11, 13, 15, 17, 19.
```

The selection bank stores sample indices of shape `[64,11,100]`, detector noise of shape `[64,11,2]`, and one frozen nuisance angle and analytic mass path per trial. The independent validation shapes are `[128,11,100]` and `[128,11,2]`.

### Endpoint-only reference flow

The reference velocity `v_psi(t,x)` is trained only from the endpoint laws. Its input contains two coordinates and five time features `(t, sin(pi t), cos(pi t), sin(2 pi t), cos(2 pi t))`. The MLP has four width-128 hidden layers with SiLU activations and a linear two-dimensional output. Conditional flow matching uses

```text
x_t = (1-t)x_0 + t x_1 + 0.15 sin(pi t) z,
u_t = -x_0 + x_1 + 0.15 pi cos(pi t) z.
```

Training uses Adam for 12,000 steps, batch size 2,048, gradient clipping at 10, and cosine learning-rate decay from `1e-3` to `5e-5`. The frozen reference is integrated by RK4 with 16 substeps per scientific interval. A tensor Gauss–Hermite rule of order 36 per coordinate and endpoint lobe gives 2,592 deterministic weighted reference particles.

### Reconstruction and projection

The 11 noisy moment observations are reconstructed over all 21 nodes by endpoint-anchored quadratic generalized least squares with relative ridge `1e-12` and variance floor `1e-10`. Reconstructed trajectories are screened against a common feasible moment polytope; authoritative candidates receive exact convex-hull feasibility checks.

At every time, the empirical information projection tilts frozen reference weights `b_i`:

```text
q_i(lambda) = b_i exp(lambda^T Phi(x_i;eta)) / Z(lambda),
sum_i q_i Phi(x_i;eta) = c(t).
```

The Newton solve is warm-started through time and must pass calibration, effective-sample-size, feasibility, and in-domain-mass checks.

## Where Tesseract enters this experiment

The outer sensor optimization is written in JAX, but each differentiable Full candidate evaluation contains two implicit inner problems. The empirical information-projection Tesseract solves the batched moment-calibration trajectories with a C++/OpenMP Newton method and exposes implicit JVPs/VJPs through the converged covariance system. The weighted-Poisson Tesseract solves the batched minimum-energy correction with matrix-free C++/OpenMP PCG and propagates derivatives through an adjoint solve. Their JAX adapters are [`src/mfsi/projection_tesseract.py`](../../src/mfsi/projection_tesseract.py) and [`src/mfsi/poisson_tesseract.py`](../../src/mfsi/poisson_tesseract.py).

For the configured differentiable proxy, one Full evaluation uses four common-random-number trials and seven time nodes. The information-projection backend calibrates the resulting trajectories over 2,592 weighted reference particles, and the Poisson backend receives all 28 systems of size `41 x 41` in one native call. Tesseract accelerates candidate generation without changing the scientific definition of the result: promoted geometries are still decided by the corrected `101 x 101`, 21-time-node authoritative evaluator and then tested on the disjoint validation bank.

See [Where Does Tesseract Enter the Picture?](../../README.md#where-does-tesseract-enter-the-picture) in the project README for the complete forward/backward path and the native-versus-JAX benchmark.

## Objectives and constraints

The population loss `L` compares projected laws based on exact analytic moments with the hidden population. It averages over the five nuisance-angle quadrature nodes and the 19 interior scientific times, using normalized trapezoid weights. The finite-law risk `R` uses Gaussian-kernel MMD with bandwidth `0.55` at the ten held-out time nodes listed above, again with normalized trapezoid weights. Both frozen risk definitions use the base `51 x 51` grid; the corrected `101 x 101` grid changes Full action only. Base-grid particle laws use the repository raster rule in `src/mfsi/raster.py`: cell deposition followed by Gaussian anti-aliasing, with `raster.bandwidth=0` selecting the implementation default `0.35 dx` and truncation `4`.

The frozen anchors are

```text
L*    = 0.0687360118546
L_max = L* + 0.0005 = 0.0692360118546
R*    = 0.0661857367210.
```

At percentage allowance `p`, every candidate must satisfy

```text
L(eta) <= L_max,
R(eta) <= R* + (p/100) |R*|.
```

The particle Tangent objective is the minimum local correction visible to sensor-moment rates. It remains the experiment's unchanged Tangent optimization metric.

The corrected Full objective is evaluated in a common raster Hilbert space. Bilinear deposition produces physical density `q_h` and signed source `s_h`; both are smoothed by the same strictly positive, full-domain separable Gaussian with per-source-column boundary normalization. The scientific solve is

```text
-div(q_h grad psi_h) = s_h,
delta_h* = -grad psi_h,
A_full,h = ||delta_h*||^2_(q_h).
```

No density floor is added to the scientific operator. Conductive components are solved with a gauge constraint and an equation-preserving sparse direct solve, with a preconditioned-CG fallback. The authoritative raster is `101 x 101`, uses all 21 time nodes, and freezes the median weighted Scott bandwidth at `0.417530106552`.

## Common-discretization decomposition

On the same raster vector space, quadrature, physical `q_h`, and inner product as Full, the moment-rate operator `L_h` defines

```text
delta_tan,h = argmin ||delta||^2_(q_h) subject to L_h delta = -r_h,
delta_hid,h = delta_h* - delta_tan,h.
```

The audit independently computes `A_tan,h`, `A_hid,h`, `Gamma_h=A_hid,h/A_full,h`, and `<delta_tan,h,delta_hid,h>_(q_h)`. It checks Full and Tangent moment feasibility, hidden nullspace membership, orthogonality, the Pythagorean identity, and raw hierarchy. The declared moment/energy tolerance is `1e-6`; values are never clipped.

| Check                              | Global maximum |
| :--------------------------------- | -------------: |
| mass error                         |     `4.44e-16` |
| signed-source compatibility error  |     `5.17e-16` |
| physical Poisson relative residual |     `1.19e-11` |
| Full moment-rate residual          |     `6.62e-14` |
| Tangent moment-rate residual       |     `8.08e-16` |
| hidden-nullspace residual          |     `6.62e-14` |
| absolute orthogonality residual    |     `1.46e-13` |
| absolute Pythagorean residual      |     `9.09e-13` |
| raw hierarchy violation            |      `-0.1274` |

Every gate passes. The negative raw hierarchy maximum is slack, not a clipped zero. Consequently `Gamma_h` is numerically supported as a genuine tangent-invisible action fraction for this corrected discretization.

## Frozen hyperparameters

The canonical base configuration is [`config.json`](config.json). The corrected Full follow-up changes only the Full raster/evaluation/search route described below.

| Group                  | Setting                                   | Value                               |
| :--------------------- | :---------------------------------------- | :---------------------------------- |
| global                 | experiment/training seed                  | `20260813`                          |
| population             | radius / sigma / domain half-width        | `1.5 / 0.3 / 3.2`                   |
| population             | alpha quadrature                          | `5` nodes over `30–60°`             |
| measurement            | sensor radius / width                     | `1.5 / 0.45`                        |
| measurement            | particles / acquisitions / detector noise | `100 / 11 / 0.01`                   |
| measurement            | minimum projective separation             | `20°`                               |
| law                    | MMD bandwidth / absolute `L` allowance    | `0.55 / 0.0005`                     |
| reference MLP          | hidden width / layers                     | `128 / 4`                           |
| training               | steps / batch / initial LR / final ratio  | `12000 / 2048 / 1e-3 / 0.05`        |
| training               | Adam / gradient clip                      | `(0.9,0.999,1e-8) / 10`             |
| bridge                 | schedule / noise                          | `linear / 0.15`                     |
| reference bank         | Gauss–Hermite order / particles           | `36 / 2592`                         |
| rollout                | RK4 substeps per interval                 | `16`                                |
| reconstruction         | method                                    | endpoint-anchored quadratic GLS     |
| reconstruction         | relative ridge / variance floor           | `1e-12 / 1e-10`                     |
| feasibility            | directions / margin / tolerance           | `96 / 1e-6 / 1e-9`                  |
| I-projection           | authoritative/search steps                | `300 / 80`                          |
| I-projection           | residual tolerance / ridge / step cap     | `1e-7 / 1e-7 / 20`                  |
| I-projection           | lambda clip / line-search steps           | `1000 / 6`                          |
| validity               | finite/population calibration             | `1e-3 / 1e-5`                       |
| validity               | minimum ESS / in-domain mass              | `0.03 / 0.995`                      |
| exact Tangent          | pseudoinverse rcond / compatibility       | `1e-10 / 1e-7`                      |
| exact particle solve   | covariance/Tangent ridge                  | `0 / 0`                             |
| exact particle solve   | covariance minimum eigenvalue             | `1e-12`                             |
| base risk/proxy raster | grid / bandwidth field / truncation       | `51 x 51 / 0 (0.35 dx default) / 4` |
| corrected Full         | raster / time nodes                       | `101 x 101 / 21`                    |
| corrected Full         | bandwidth                                 | `0.417530106552`                    |
| corrected Full         | density floor                             | `0` in scientific operator          |
| corrected Full         | Poisson tolerance                         | `1e-7`                              |
| corrected Full         | mass/source/component tolerance           | `1e-12`                             |
| corrected Full         | moment/decomposition tolerance            | `1e-6`                              |
| randomness             | Law/action/validation trials              | `32 / 64 / 128`                     |
| randomness             | selection/validation namespaces           | `8890 / 8891`                       |
| reporting              | bootstrap replicates                      | `5000`                              |

### Base Population/Law/Tangent/Full candidate pipeline

The corrected follow-up freezes Population, Law, and Tangent, but their original selection protocol remains part of the experiment definition.

| Setting                |  Population |   Law | Tangent | historical Full proxy |
| :--------------------- | ----------: | ----: | ------: | --------------------: |
| optimization steps     |         120 |    50 |      40 |                    40 |
| learning rate          |        0.02 | 0.015 |   0.012 |                  0.01 |
| normal starts          | shared pool |     4 |       7 |                     7 |
| gradient trials        |           — |     4 |       4 |                     4 |
| exact audit candidates |           — |     8 |      16 |                    30 |
| exact rescores         |           — |     — |       8 |                    10 |
| local starts           |           — |     — |      12 |                    16 |
| perturbation scale     |           — |     — |    `5°` |                  `6°` |

Shared settings are start count `12`, constraint penalty `10000`, feasibility tolerance `1e-6`, invalid penalty `10000`, and JIT-enabled objectives. Full also has eight random starts, uses 12 prescreen trials, a configured `41 x 41`/seven-time-node differentiable proxy, CG tolerance `1e-6`, and at most 360 CG iterations. Required exact-valid/finalist counts are `8/6` for Tangent and `12/10` for Full. These proxy quantities never replace the corrected reported Full action.

## Corrected Full search protocol

Population, Law, and Tangent are not rerun. For every allowance, the previous tighter corrected Full winner is a mandatory incumbent and is rechecked against the current exact `L` and `R` caps. Candidate seeds include the incumbent, historical Full, saved Tangent, Law, all feasible previously audited Full-search candidates, and normal multistarts.

New basins are navigated on a `51 x 51` two-trial frozen prefix, promoted to a 12-trial `101 x 101` prescreen, then decided using all 64 frozen selection trials at `101 x 101`. A winner is replaced only by a feasible, fully certified candidate whose corrected selection action is lower by more than `1e-6`. Only after selection is frozen is the independent 128-trial validation bank evaluated.

All 1–5% winners differ from the previously corrected sweep; their selection-action changes are `-2.68696`, `-2.15943`, `-2.09889`, `-1.59191`, and `-0.85405`. The fixed isolated 0.5% result reproduces within `1.5e-14`. Every current geometry differs from the older pre-correction saved endpoint. The corrected Full design has lower corrected Full action than the saved Tangent design at every allowance, so the central FIDE ranking survives. No further Full optimization is required by the declared protocol; this is a numerical multistart certificate, not an analytic global-optimum proof.

## Reproduction

There are three distinct reproducibility levels. The first verifies the exact published files and is fast. The second regenerates a new base run. The third replays the expensive corrected search from its frozen historical inputs.

### Environment and native backend

Python 3.11 or newer is required. From the repository root, create the environment, install only the analytical experiment's scientific and native-backend dependencies, and build both Tesseract accelerators requested by the production configuration:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[analytical,tesseract-cpp]'

.venv/bin/cmake \
  -S native/iprojection_tesseract \
  -B native/iprojection_tesseract/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python"
.venv/bin/cmake --build native/iprojection_tesseract/build --parallel "$(nproc)"

.venv/bin/cmake \
  -S native/poisson_tesseract \
  -B native/poisson_tesseract/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python"
.venv/bin/cmake --build native/poisson_tesseract/build --parallel "$(nproc)"

export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

The `analytical` extra does not install or opt into the skyrmion experiment. Native build details and backend tests are in the [`iprojection_tesseract`](../../native/iprojection_tesseract/README.md) and [`poisson_tesseract`](../../native/poisson_tesseract/README.md) READMEs. For reproducible CPU scheduling, also set an explicit `OMP_NUM_THREADS` no larger than the number of physical cores and avoid nested OpenMP.

### Level 1: verify the saved authoritative result

```bash
.venv/bin/python experiments/toy_example_percentage/eval_pareto.py
```

This is read-only. It verifies SHA-256 receipts, the six PASS rows, nesting, method tables, and 2,304 saved validation-trial records. It is the correct command for checking the publication result. The more general `eval.py` command reads the older tracked base/source run and is retained for pipeline diagnostics; its numbers are not the corrected Pareto result.

### Level 2: regenerate a base run

```bash
.venv/bin/python experiments/toy_example_percentage/run.py --smoke \
  --output-dir experiments/toy_example_percentage/outputs/reproduction/smoke
.venv/bin/python experiments/toy_example_percentage/run.py \
  --output-dir experiments/toy_example_percentage/outputs/reproduction/source
```

The first command is a small wiring test and does not reproduce production statistics. The second trains/generates Population, Law, Tangent, and the original Full candidate pipeline under `config.json`. It supplies frozen source-run inputs but does not perform the later corrected `101 x 101` nested Full sweep.

### Level 3: replay the corrected nested Full sweep

The ignored seed tree is deliberately absent from an ordinary checkout. It is still recoverable byte-for-byte from Git commit `9250d083890fbc5b9938d46210b733b491a849f4`. A shallow clone must fetch that commit before running the archive command:

```bash
toy_archive=experiments/toy_example_percentage/outputs/old/pareto_pre_corrected_full
test ! -e "$toy_archive"
toy_unpack="$(mktemp -d)"
git archive 9250d083890fbc5b9938d46210b733b491a849f4 \
  experiments/toy_example_percentage/outputs/pareto \
  | tar -x -C "$toy_unpack"
mkdir -p "$(dirname "$toy_archive")"
mv "$toy_unpack/experiments/toy_example_percentage/outputs/pareto" "$toy_archive"
```

Recreate the three corrected-candidate audit files expected by the nested driver, then write the replay to a separate ignored directory so the tracked authoritative files are not overwritten:

```bash
.venv/bin/python experiments/toy_example_percentage/audit_corrected_all_candidates.py \
  --pareto-dir "$toy_archive" --grid-n 101 --bandwidth-scale 1.0

toy_replay=experiments/toy_example_percentage/outputs/reproduction/pareto

.venv/bin/python experiments/toy_example_percentage/rerun_corrected_full_0p5.py \
  --pareto-dir "$toy_archive" \
  --output-dir "$toy_replay/risk_0p5pct/full_search" \
  --grid-n 101 --bandwidth-scale 1.0

.venv/bin/python experiments/toy_example_percentage/run_corrected_nested_full_sweep.py \
  --pareto-dir "$toy_archive" \
  --output-dir "$toy_replay" \
  --rerun-0p5-json "$toy_replay/risk_0p5pct/full_search/corrected_full_rerun.json" \
  --search-from-1pct --fresh-evaluations \
  --grid-n 101 --bandwidth-scale 1.0

.venv/bin/python experiments/toy_example_percentage/finalize_authoritative_corrected_pareto.py \
  --seed-pareto "$toy_archive" \
  --source-run experiments/toy_example_percentage/outputs/run \
  --output "$toy_replay" \
  --grid-n 101 --bandwidth-scale 1.0
```

The audit produces `toy_corrected_all_candidates_rescore.json`, `toy_corrected_validation_rescore.json`, and `toy_corrected_candidate_pool_audit.json` in the archive. The search commands are checkpointed. `--fresh-evaluations` prevents archived action values from entering the corrected cache; archived geometries remain valid seeds.

`--previous-corrected` is intentionally omitted from the finalizer command. That optional input affects only old-versus-previous-corrected provenance fields, and the precise intermediate file was never tracked. It does not affect the current winners, risks, actions, validation, or certificates. The saved authoritative directory already contains the original comparison receipt. Cross-platform sparse solves can differ in their last floating-point digits, so scientific tolerances—not byte identity of a fresh replay—define success.

### Regenerate figures

```bash
.venv/bin/python experiments/toy_example_percentage/visualize_pareto.py
.venv/bin/python experiments/toy_example_percentage/visualize_paper.py
.venv/bin/python experiments/toy_example_percentage/visualize_paper_gif.py
```

The paper command always writes PNG and PDF. These scripts are post-processing only and use the saved authoritative geometry and frozen banks.

## Current artifacts

| Path                                                                     | Contents                                                      |
| :----------------------------------------------------------------------- | :------------------------------------------------------------ |
| `outputs/pareto/corrected_nested_full_sweep.csv/.json/.md`               | full-precision corrected nested sweep and PASS summary        |
| `outputs/pareto/pareto.csv/.json`                                        | compact one-row-per-allowance Full table                      |
| `outputs/pareto/pareto_methods_selection.csv`                            | corrected Law/Tangent/Full selection table                    |
| `outputs/pareto/pareto_methods_validation.csv`                           | independent validation means, SEs, and validity               |
| `outputs/pareto/pareto_methods_tables.md`                                | generated human-readable method tables                        |
| `outputs/pareto/positive_raster_decomposition_diagnostics.csv/.json/.md` | common-discretization numerical audit                         |
| `outputs/pareto/validation_trial_summaries.csv`                          | all Full, Law, and Tangent validation trial actions           |
| `outputs/pareto/risk_*pct/result.json`                                   | current per-allowance result record                           |
| `outputs/pareto/risk_*pct/candidates.csv`                                | all preserved/generated seeds and stage outcomes              |
| `outputs/pareto/risk_*pct/audit.json`                                    | exact risk, selection, validation, and certificate record     |
| `outputs/pareto/risk_*pct/validation_trials.csv`                         | per-allowance Full validation trials                          |
| `outputs/pareto/frozen_inputs/manifest.json`                             | hashes and paths for the active frozen copies                 |
| `outputs/pareto/authoritative_run_summary.json`                          | global nesting, numerical maxima, and old/new comparison      |
| `outputs/pareto/*.png`                                                   | current Pareto, method, experiment, and sensor-layout figures |

The active frozen files are `reference.npz`, `reference_bank.npz`, `selection_bank.npz`, and `validation_bank.npz`. Their SHA-256 hashes match their source-run counterparts. `outputs/old/` is ignored and is not present in a normal checkout; the exact historical seed tree is reconstructed from the Git commit documented above.

## Software, provenance, and limits

The accepted run was documented with Python 3.12.3, NumPy 2.5.1, SciPy 1.18.0, JAX/jaxlib 0.8.3, and 64-bit JAX. The frozen-input manifest is the authority for input hashes. Key scientific source hashes are:

| File                                         | SHA-256                                                            |
| :------------------------------------------- | :----------------------------------------------------------------- |
| `experiment.py`                              | `35bbce25a92790f152664c8304adcd837e117598213c0c80b821254b50e7dcd3` |
| `run_corrected_nested_full_sweep.py`         | `cba9b1c0191184e6cb3f3a7ef7a4a93e4937fb0e17b90da268634d5046cdd3a3` |
| `finalize_authoritative_corrected_pareto.py` | `8963ca4bccc6d45da6fe5860dca25719de8911e1d50ead5e4be7b7e8eca53ba9` |

| Authoritative artifact             | SHA-256                                                            |
| :--------------------------------- | :----------------------------------------------------------------- |
| Corrected nested sweep JSON        | `114df72191c0b519e6e45cf7c574060a47ac6c64201eba7ed7432f2f11fc2c7e` |
| Positive-raster decomposition JSON | `e57928bd779e5873bd9044a601574122d1c38a2d0a008dbdbc4716934f42eed4` |
| Authoritative run summary          | `2e29a178e3850ccb35c067006fb565bcb4f5fb845740ab4b6d4e4859034b80db` |

| Frozen input          | SHA-256                                                            |
| :-------------------- | :----------------------------------------------------------------- |
| `reference.npz`       | `4bca0ff23a4009a86ea92548d908cdff98be7289f28e7273faeb327ba65bff87` |
| `reference_bank.npz`  | `d721d5b7f21bfd389d9fc210252ebed4a4a93ce147b68d6529b430585b31fdef` |
| `selection_bank.npz`  | `a8c21c0a8d7b67b78d87fc73bb992ca9b9a2b0b5477153a7ae11973d694c628a` |
| `validation_bank.npz` | `6f96e05cd365b37a15b50c2c0448a6d4e55b213717cb94af3c114374c72750bd` |

This is a one-configuration numerical multistart certificate, not a proof of a global optimum. End-to-end corrected wall time and peak memory were not saved. The current result does not establish sensitivity to particle count, sensor width, detector noise, nuisance quadrature, raster resolution, or new random banks. Those are appropriate confirmatory studies, not missing gates in the declared frozen protocol.

## Read-only saved-result evaluation

From the repository root:

```bash
.venv/bin/python experiments/toy_example_percentage/eval.py
.venv/bin/python experiments/toy_example_percentage/eval_pareto.py
```

The first command displays the older tracked base/source run and is not the publication result. The second displays and hash-verifies the corrected authoritative Pareto sweep. Neither command runs the experiment or writes outputs. Both use the repository-wide saved-evaluator table style, include Law/Tangent/Full, and report sample SDs from the saved independent validation trials (or from a saved ordinary SE and `n`).
