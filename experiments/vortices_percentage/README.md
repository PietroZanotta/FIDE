# Vortices percentage-risk Pareto experiment

## Authoritative status

This directory contains the corrected, authoritative percentage-risk sweep for the time-dependent double-gyre experiment. The active results are under `outputs/pareto/`. The pre-correction sweep is not a current scientific result and is not present in a normal checkout; when needed as replay input, it can be reconstructed byte-for-byte from the Git commit documented below.

The corrected sweep covers `0.5%, 1%, 2%, 3%, 4%, 5%`. Full selection uses the physical-density weighted-Poisson evaluator on a `128 x 64` grid at all 21 time nodes. Every final Law, Tangent, and Full geometry passes the exact common-raster audit on the frozen 24-trial selection bank and the independent frozen 64-trial validation bank. The corrected Full selection curve is nested.

The headline result survives: corrected Full designs reduce mean validation Full action relative to the common Law design by `59.17%` to `70.18%`, depending on allowance. The earlier `26.70%` headline belonged to the archived evaluator and must not be quoted as the current result.

Key artifacts:

- [corrected authoritative table](outputs/pareto/corrected_authoritative_pareto.md)
- [common-raster decomposition audit](outputs/pareto/corrected_authoritative_decomposition_audit.md)
- [old-versus-corrected comparison](outputs/pareto/old_vs_corrected_full_comparison.md)
- [complete Law/Tangent/Full tables](outputs/pareto/pareto_methods_tables.md)
- [Pareto figure](outputs/pareto/pareto_methods.png)
- [sensor-layout figure](outputs/pareto/pareto_sensor_layouts.png)
- [experiment and sensor illustration](outputs/pareto/experiment_sensors.png)

The publication authority is the corrected table, per-allowance audit files,
independent validation receipts, frozen-input manifest, and fail-closed
finalizer under `outputs/pareto/`. `outputs/run/`, `eval.py`, historical
reference-seed studies, proxy actions, and smoke runs are useful diagnostics
but do not supersede that result. “Authoritative” here means certified under
the declared frozen numerical protocol; it is a multistart certificate, not
an analytic global-optimum proof.

## Scientific question

Four localized sensors observe a finite noisy particle population in a double-gyre flow. Their locations are optimized while allowing a controlled percentage increase in finite-law risk. The experiment asks how much physical correction action can be removed by accepting that small risk increase, and how much of the correction is tangent-visible versus tangent-invisible.

The eight-dimensional design is

```text
eta = (x1,y1,x2,y2,x3,y3,x4,y4).
```

At allowance `p`, an eligible action design must satisfy the unchanged exact risk screens

```text
L(eta) <= L* + epsilon_L,
R(eta) <= R* + (p/100) |R*|.
```

Here `L` is exact-oracle population risk, `R` is finite noisy law risk, and the shared frozen Law anchor is

```text
L*    = 0.03773928370170817,
L_max = L* + 0.025 = 0.06273928370170817,
R*    = 0.03838744201119958.
```

Selection uses only the frozen selection bank and these exact screens. Validation is computed only after a winner is frozen and never participates in selection.

## Physical model

The domain is `Omega = [0,2] x [0,1]`. Normalized experiment time `t in [0,1]` corresponds to physical time `tau = Ht`, with `H = 10`. The double-gyre field is

```text
a(tau) = epsilon sin(omega tau)
b(tau) = 1 - 2 a(tau)
f(x,tau) = a(tau) x^2 + b(tau) x
df/dx = 2 a(tau) x + b(tau)
omega = 2 pi / period

dx/dtau = -pi A sin(pi f) cos(pi y)
dy/dtau =  pi A cos(pi f) sin(pi y) df/dx
```

with `A = 0.1`, `epsilon = 0.25`, `period = 10`, and normalized velocity `dX/dt = H dX/dtau`.

The initial law is a `10%` uniform background plus four truncated Gaussian components. Their internal mixture weights are `(0.30, 0.20, 0.25, 0.25)`, centers are `(0.45,0.25)`, `(0.78,0.72)`, `(1.28,0.28)`, `(1.62,0.68)`, and coordinate standard deviations are `0.07`. Out-of-domain samples are rejection-resampled, not clipped.

The base seed is `20260815`. The frozen truth bank contains `50,000`
particles on the equally spaced nodes `t_i=i/20`, generated with seed offset
`1001` and RK4 with 32 substeps per interval. Endpoint training data use
`50,000` particles, seed offset `2001`, and 512 RK4 substeps over the complete
endpoint trajectory.

## Sensors and observations

Each Gaussian sensor has feature

```text
Phi_j(x; eta) = exp(-||x-s_j||^2 / (2 ell^2)),  ell = 0.12.
```

There are four labeled sensors. Centers are constrained to `x in [0.24,1.76]`, `y in [0.24,0.76]`, with minimum pairwise separation `0.24`. Each noisy observation uses `2,000` truth particles at acquisition indices `(0,2,5,8,10,12,15,18,20)` of the 21-node scientific grid and independent detector noise with standard deviation `0.005`. Endpoint moments are exact.

The selection bank stores sample indices with shape `[24,9,2000]` and
detector noise with shape `[24,9,4]`; its first 16 trials define Law risk and
all 24 define action. The independent validation arrays have shapes
`[64,9,2000]` and `[64,9,4]`. Selection and validation use random namespaces
`9890` and `9891`, respectively.

The moment path is an endpoint-anchored, bounded, `C2` cubic penalized least-squares spline with three internal knots, smoothing `1e-4`, relative ridge `1e-10`, eighth-order roughness quadrature, feature bounds `[0,1]`, interior margin `0.002`, and transition width `0.002`.

At each scientific time, empirical information projection calibrates the frozen reference particles to the reconstructed moments. The projection settings are: 300 maximum steps, Newton ridge `1e-7`, step cap `20`, multiplier clip `1000`, search tolerance `1e-6`, acceptance tolerance `2e-6`, support-certificate tolerance `1e-10`, L-BFGS maximum 800 iterations, and at most two retries. Certification also requires adequate effective sample size and empirical-hull support.

## Learned reference model

The endpoint-only reference is trained in box-logit coordinates so its physical pushforward stays inside the rectangle. Its velocity MLP receives two latent coordinates plus `(t, sin(pi t), cos(pi t), sin(2pi t), cos(2pi t))`, has four SiLU hidden layers of width 128, and a linear two-dimensional output.

Training uses conditional flow matching with a deterministic linear bridge, zero bridge noise, Adam (`beta1=0.9`, `beta2=0.999`, `eps=1e-8`), batch size 2,048, 12,000 steps, gradient clipping at 10, and cosine learning-rate decay from `1e-3` to `5e-5`. The frozen reference rollout bank contains 32,768 particles, uses seed offset `3001`, and RK4 with 16 substeps per scientific interval.

The reference checkpoint, rollout bank, truth bank, endpoint data, selection bank, validation bank, source manifest, and configuration are frozen. Their active copies and hashes are recorded in `outputs/pareto/frozen_inputs/manifest.json`.

## Objectives

- **Population** minimizes exact-oracle population risk and is used to establish the population screen.
- **Law** minimizes finite noisy law risk and provides the one common anchor for all allowances.
- **Tangent** minimizes the experiment's unchanged particle-space tangent metric under the same risk screens.
- **Full** minimizes weighted-Poisson correction action under the same risk screens.

The Law risk is a multi-bandwidth MMD with bandwidths `0.05`, `0.1`, `0.2`, and `0.4`; the population slack is `epsilon_L = 0.025`. Both `L` and `R` are normalized trapezoidal averages over all 21 scientific time nodes. Projected and truth laws are compared after depositing cell mass on the `128 x 64` rectangular grid and applying the configured separable Gaussian anti-aliasing kernel. `raster.bandwidth=0` selects `0.35 min(dx,dy)`; the kernel truncation is four standard deviations. Mass is normalized and the deposited signed source is centered to floating-point compatibility before the Poisson solve.

### Correct authoritative Full evaluator

Reported Full action uses the physical raster density `q_h` in both the operator and the vector-field inner product:

```text
-div(q_h grad psi_h) = s_h,
delta*_h = -grad psi_h,
A_full,h = ||delta*_h||^2_{q_h}.
```

The authoritative grid is `128 x 64`, all 21 time nodes are included, and there is no density floor in the scientific operator. The solve is a sparse physical-`q_h` direct solve over conductive components, with component compatibility checks, a constant-fixing pin, symmetric variable scaling, and residual iterative refinement using the same factorization. These operations preserve the discrete equation. A preconditioned-CG fallback is available. Certification independently checks the physical Poisson residual and the sensor moment-rate equations.

The regularized `tesseract_cpp` solve remains a search proxy only: its `64 x 32`, seven-time-node gradient objective may guide candidates, but it never supplies a reported scientific Full-action value. The `operator_floor_rel` field retained in the frozen configuration belongs to that proxy path and is not used in the corrected scientific operator.

### Common-raster decomposition

For each final Law, Tangent, and Full geometry, the audit builds the moment-rate operator `L_h` in exactly the Full raster vector-field space and solves

```text
delta_tan,h = argmin ||delta||^2_{q_h}  subject to L_h delta = -r_h,
delta_hid,h = delta*_h - delta_tan,h.
```

It independently evaluates

```text
A_tan,h  = ||delta_tan,h||^2_{q_h},
A_hid,h  = ||delta_hid,h||^2_{q_h},
Gamma_h  = A_hid,h / A_full,h,
<delta_tan,h, delta_hid,h>_{q_h}.
```

`Gamma_h` is reported only because Full feasibility, Tangent feasibility, hidden nullspace, orthogonality, Pythagorean identity, and hierarchy all pass without clipping.

## Optimization and selection protocol

Base optimizer settings are 64 generated starts from an oversampled pool of 128. Population, Law, Tangent, and Full receive 100, 50, 50, and 30 optimization steps with learning rates `0.01`, `0.008`, `0.006`, and `0.004`. Constraint penalty is `10,000`, optimization feasibility tolerance is `1e-6`, and invalid penalty is `1,000`.

Population uses eight starts and audits 20 candidates, requiring at least six exact-valid candidates. Law uses six starts, four gradient trials, audits 24 candidates, requires at least eight exact-valid candidates, and permits two anchor-refinement passes with consistency tolerance `1e-5`.

The repaired Tangent workflow uses four normal starts, four gradient trials, 12 local starts of scale `0.08`, 30 exact audits, and ten exact rescores. Archived candidates, repaired 4%/5% candidates, Full geometries, and the Law geometry are included as seeds and are re-audited; no obsolete Tangent solution is restored.

Full uses three normal starts, two proxy gradient trials, ten local starts of scale `0.06`, four-trial prescreening, 30 authoritative exact audits, and eight exact rescores. Candidate seeds at every allowance include the mandatory previous corrected Full incumbent, archived Full candidates, current Tangent candidates, Law, saved audited feasible candidates, and normal multistarts.

The Full sweep is sequential. The 0.5% winner is carried into 1%, and so on through 5%. An incumbent is replaced only by a feasible candidate with lower corrected selection action beyond tolerance `1e-6`. Validation uses 64 disjoint trials only after the stage winner is frozen.

### Frozen production settings

| Group | Setting | Value |
|:---|:---|:---|
| global | base/reference seed | `20260815` |
| truth | particles / seed offset | `50000 / 1001` |
| truth | RK4 substeps per interval | `32` |
| endpoints | particles / seed offset / RK4 steps | `50000 / 2001 / 512` |
| reference | rollout particles / seed offset | `32768 / 3001` |
| reference | RK4 substeps per interval | `16` |
| randomness | Law/action/validation trials | `16 / 24 / 64` |
| randomness | selection/validation namespaces | `9890 / 9891` |
| validity | population/finite calibration | `1e-5 / 1e-3` |
| validity | minimum ESS / in-domain mass | `0.03 / 0.995` |
| validity | Poisson relative residual | `2e-7` |
| exact Tangent | covariance eigenvalue floor / pseudoinverse `rcond` | `1e-6 / 1e-10` |
| exact Tangent | compatibility tolerance | `1e-7` |
| scientific Full | grid / time nodes / density floor | `128 x 64 / 21 / 0` |
| scientific Full | CG tolerance / maximum iterations | `1e-7 / 520` |
| search proxy | grid / time nodes / operator floor | `64 x 32 / 7 / 2e-5` |
| search proxy | CG tolerance / maximum iterations | `1e-6 / 360` |

The `smoke` block in `config.json` replaces these values with small test values
and was not used for the authoritative sweep.

## Corrected authoritative results

The Full table below uses the frozen 24-trial selection bank for selection action and the independent 64-trial bank for validation. Reduction is the ratio-of-means reduction from the common Law geometry. Values are rounded; machine-readable precision is in `corrected_authoritative_pareto.csv` and `.json`.

| Allowance | Full centers `(x,y)` | Exact L | Exact R | R increase | Selection A_full | Validation A_full ± SE | Full vs Law | A_tan,h | A_hid,h | Gamma_h |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5% | `(1.07263,.39629) (.48624,.76000) (1.76000,.24000) (.27194,.61851)` | .038056816 | .038521091 | .348% | 7.343264 | 7.367562 ± .370033 | 64.98% | .430744 | 6.912520 | .941342 |
| 1% | `(1.05487,.40255) (.47001,.76000) (1.76000,.24000) (.24000,.61089)` | .038212162 | .038740166 | .919% | 6.688760 | 8.589732 ± 1.744830 | 59.17% | .435773 | 6.252986 | .934850 |
| 2% | `(1.08327,.43357) (.50729,.74970) (1.76000,.24000) (.28087,.63963)` | .038784770 | .039058745 | 1.749% | 6.510045 | 6.273495 ± .460063 | 70.18% | .430471 | 6.079575 | .933876 |
| 3% | same as 2% | .038784770 | .039058745 | 1.749% | 6.510045 | 6.273495 ± .460062 | 70.18% | .430471 | 6.079575 | .933876 |
| 4% | `(1.06371,.45502) (.49854,.74203) (1.75976,.24000) (.24971,.64638)` | .039504166 | .039756850 | 3.567% | 5.748588 | 6.274276 ± .901433 | 70.18% | .417009 | 5.331580 | .927459 |
| 5% | `(1.05214,.41790) (.50491,.72562) (1.74915,.24688) (.24562,.65249)` | .039821283 | .040072176 | 4.389% | 5.730633 | 6.288513 ± .679615 | 70.11% | .386964 | 5.343669 | .932474 |

The corrected Full selection sequence is

```text
7.343264075 >= 6.688759712 >= 6.510045283
             >= 6.510045283 >= 5.748588139 >= 5.730632961.
```

The 2% incumbent remains the 3% winner and is reported with the tighter-stage action; its independent repeated solve differed by only `1.63e-7`, below the declared `1e-6` tolerance.

The common Law centers are
`(1.077489,.388963)`, `(.479798,.760000)`, `(1.760000,.240000)`, and
`(.257336,.602956)`. The complete selected Tangent centers are:

| Allowance | Tangent centers `(x,y)` |
|---:|:---|
| 0.5% | `(1.072627,.396293) (.486242,.760000) (1.760000,.240000) (.271940,.618506)` |
| 1% | `(1.054449,.390149) (.451830,.748630) (1.760000,.240000) (.262004,.577847)` |
| 2%, 3% | `(1.058357,.379515) (.475429,.760000) (1.757857,.240000) (.289138,.571884)` |
| 4% | `(1.070393,.426616) (.490176,.700052) (1.760000,.240000) (.263510,.629298)` |
| 5% | `(1.052135,.417897) (.504911,.725622) (1.749147,.246883) (.245617,.652486)` |

Tangent and Full coincide at 0.5% and 5%. The tracked
[complete method tables](outputs/pareto/pareto_methods_tables.md) contain every
Law/Tangent/Full selection and independent-validation row, including risk,
action, SE, reduction, and valid fraction.

### Law, Tangent, and Full interpretation

The corrected Full geometry equals the Tangent geometry at 0.5% and 5%. On the validation bank, Tangent has lower Full action at 1% and 4%, while Full has lower action at 2% and 3%. This mixed finite-bank ranking replaces the older simpler story. It does not weaken the central result: every selected Full geometry yields a large positive Full-vs-Law reduction, and the common-raster audit supports a genuine tangent-invisible fraction of roughly `92.7%` to `94.1%` on the selection bank.

The unusually large Law validation mean and SE (`about 21.04 ± 13.50`) reflect a small number of high-action validation realizations. All 64 Law validation trials remain numerically valid; the ratio-of-means reduction and its raw trial summaries are retained for transparency.

## Numerical certification

The tolerance is `1e-6`. Across all 18 final geometries and both banks:

| Certification quantity | Maximum |
|:---|---:|
| physical Poisson relative residual | `5.132997e-10` |
| component compatibility residual | `5.390598e-15` |
| Full moment-rate residual | `8.004651e-11` |
| Tangent moment-rate residual | `3.619007e-14` |
| hidden-nullspace residual | `8.004623e-11` |
| absolute orthogonality residual | `3.262391e-11` |
| absolute Pythagorean residual | `6.184564e-11` |
| maximum raw hierarchy value `A_tan,h - A_full,h` | `-8.561291e-2` |

There are zero aggregate, trial, time/trial, hierarchy, or invalid-trial violations. Every physical solve converged. Moment calibration, ESS, support feasibility, in-domain mass, component compatibility, Full moment feasibility, and decomposition gates remain active. Residuals and hierarchy gaps are raw and unclipped.

## Comparison with the archived sweep

The corrected winner changes at 0.5%, 2%, 3%, 4%, and 5%; the 1% Full geometry is unchanged. Absolute old and corrected actions are not directly comparable as optimization-only changes because the scientific evaluator changed from the earlier regularized formulation to physical `q_h`. The complete coordinates and numbers are in `old_vs_corrected_full_comparison.csv`, `.json`, and `.md`.

The corrected validation reductions by allowance are `64.98%`, `59.17%`, `70.18%`, `70.18%`, `70.18%`, and `70.11%`; these supersede the archived `7.99%`, `15.89%`, `15.89%`, `15.89%`, `26.70%`, and `25.94%` values.

No additional Full or Tangent optimization is required for this declared sweep: all mandated seed families were considered, every selected candidate is exact-feasible and certified, the repaired Tangent search is retained, and the Full curve is nested.

## Figures

![Hidden population, corrected law, and four sensor views](figures/vortices_population_correction_sensors.png)

Each column is a time snapshot from frozen validation trial 0 at the
authoritative 5% Full geometry. The upper row shows the transported hidden
population and instantaneous double-gyre streamlines; the middle row shows the
moment-corrected endpoint reference; the small panels show what each localized
sensor weights and report its scalar observation. The correction matches four
moments at each reconstructed time, not the hidden density pointwise. The
remaining off-sensor differences visualize the underdetermination that the
Full action measures.

![Corrected Law, Tangent, and Full Pareto curves](outputs/pareto/pareto_methods.png)

Panel A normalizes corrected Full action to the Law value on the same bank;
solid curves are selection and dashed curves are independent validation. Full
is below Law at every allowance, while Tangent has a mixed common-action
ranking—especially at 1–3%—despite minimizing its own particle metric. Panel B
shows use of the Law-relative risk budget. Every selected point remains below
the 100% cap; validation risk is shown only as an out-of-sample diagnostic.
The large selection/validation separation warns that bank dependence matters;
the table and raw trial summary, rather than this normalized plot, quantify
the particularly large Law uncertainty.

![Corrected sensor layouts by method and allowance](outputs/pareto/pareto_sensor_layouts.png)

Rows compare Law, Tangent, and Full centers; columns increase the allowance.
The background is time-averaged particle occupancy, translucent disks show one
sensor width, and the dashed rectangle is the admissible-center region. Law is
fixed, whereas Tangent and Full redistribute sensors among the upper-left,
interior, and right-boundary transport structures. Tangent and Full coincide
at 0.5% and 5% but follow different intermediate geometries, consistent with
their mixed validation ranking.

![Double-gyre experiment and four-sensor geometry](outputs/pareto/experiment_sensors.png)

This multi-panel dashboard uses the representative 3% point. Panel A shows the
hidden double-gyre evolution; panel B compares Population, Law, Tangent, and
Full geometries; panel C shows the certified selection trade-off; and panel D
shows all 64 independent trial actions on a log scale. The high Law outlier is
why its validation mean and SE are much larger than its median. Panel E is a
timing breakdown for this representative staged run only—it is not the total
cost of the corrected six-allowance campaign or a cross-platform benchmark.

Regenerate the paper-style observation-mechanism figure from the frozen 5% Full
geometry and validation/reference banks (both PNG and PDF are always written):

```bash
.venv/bin/python experiments/vortices_percentage/visualize_paper.py
.venv/bin/python experiments/vortices_percentage/visualize_paper_gif.py
.venv/bin/python experiments/vortices_percentage/visualize_pareto.py
```

These are deterministic post-processing commands over saved results. The paper
command writes both PNG and PDF; the animation writes the corresponding GIF.

## Reproduction

### Environment and native backend

Python 3.11 or newer is required. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[skyrmions,tesseract-cpp]'

.venv/bin/cmake \
  -S native/poisson_tesseract \
  -B native/poisson_tesseract/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python"
.venv/bin/cmake --build native/poisson_tesseract/build -j "$(nproc)"

export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

The `skyrmions` extra is the repository's shared SciPy/Matplotlib extra; it
does not change this experiment. See
[`native/poisson_tesseract/README.md`](../../native/poisson_tesseract/README.md)
for native tests and backend semantics. Set an explicit `OMP_NUM_THREADS` no
larger than the number of physical cores and avoid nested OpenMP.

### Level 1: verify the saved authoritative result

```bash
.venv/bin/python experiments/vortices_percentage/eval_pareto.py
```

This read-only check verifies the saved hashes, six corrected rows, 1,536
validation-trial method records, and certification summary. It is the correct
publication-result verifier. `eval.py` instead displays the older tracked
base/source run and is retained only as a pipeline diagnostic.

### Level 2: regenerate a source run

```bash
.venv/bin/python experiments/vortices_percentage/run.py --smoke \
  --output-dir experiments/vortices_percentage/outputs/reproduction/smoke
.venv/bin/python experiments/vortices_percentage/run.py \
  --output-dir experiments/vortices_percentage/outputs/reproduction/source
```

The smoke command is a small wiring test and does not reproduce production
statistics. The full command trains the endpoint reference and generates the
base Population/Law/Tangent/Full run under `config.json`; it is not the later
corrected nested Pareto search.

### Level 3: replay the corrected sweep

The ignored pre-correction tree is recoverable from Git commit
`f4d955a55bebcedc84d2bb858e456dda4f7a66d0`. A shallow clone must fetch that
commit before running the archive command:

```bash
vortices_archive=experiments/vortices_percentage/outputs/old/pareto_pre_corrected_full
test ! -e "$vortices_archive"
vortices_unpack="$(mktemp -d)"
git archive f4d955a55bebcedc84d2bb858e456dda4f7a66d0 \
  experiments/vortices_percentage/outputs/pareto \
  | tar -x -C "$vortices_unpack"
mkdir -p "$(dirname "$vortices_archive")"
mv "$vortices_unpack/experiments/vortices_percentage/outputs/pareto" \
  "$vortices_archive"
```

Run the nested search into a separate ignored destination so the tracked
authoritative result is not overwritten, then finalize and independently
certify its frozen winners:

```bash
vortices_replay=experiments/vortices_percentage/outputs/reproduction/pareto

.venv/bin/python experiments/vortices_percentage/run_pareto.py \
  --source-run "$vortices_archive/risk_0p5pct" \
  --output "$vortices_replay" \
  --seed-pareto "$vortices_archive"

.venv/bin/python experiments/vortices_percentage/finalize_authoritative_corrected_pareto.py \
  --pareto-dir "$vortices_replay" \
  --archive-pareto "$vortices_archive" \
  --source-run "$vortices_archive/risk_0p5pct"
```

The finalizer fails closed if nesting, exact candidate certification, or
common-raster decomposition fails. It does not optimize or alter winners.
Cross-platform sparse and native solves can differ in their last
floating-point digits, so the declared tolerances—not byte identity of a fresh
replay—define scientific success. Existing per-stage timing receipts total
about 104 minutes over the six source points on the recorded machine, but omit
later corrected searches and finalization and are not a total-cost benchmark.

## Artifact map

| Path | Purpose |
|:---|:---|
| `outputs/pareto/pareto.csv`, `.json` | complete active Pareto sweep |
| `outputs/pareto/pareto_methods_selection.csv` | exact selection metrics for Law/Tangent/Full |
| `outputs/pareto/pareto_methods_validation.csv` | independent validation metrics |
| `outputs/pareto/corrected_authoritative_pareto.*` | final corrected Full table and summary |
| `outputs/pareto/corrected_authoritative_decomposition_audit.*` | aggregate and per-time/trial common-raster audit |
| `outputs/pareto/old_vs_corrected_full_comparison.*` | archived-versus-current comparison |
| `outputs/pareto/validation_trial_summaries.csv` | per-trial validation receipts |
| `outputs/pareto/authoritative_run_summary.json` | concise certification and scientific conclusions |
| `outputs/pareto/frozen_inputs/manifest.json` | frozen-input paths and SHA-256 hashes |
| `outputs/pareto/risk_*pct/` | per-allowance results, candidate audits, timings, banks, and manifests |
| `outputs/old/pareto_pre_corrected_full/` | ignored historical input reconstructed from the documented Git commit |

Core implementation files are `experiment.py` (experiment/evaluators), `selection.py` (candidate generation and exact selection), `run_pareto.py` (nested sweep), `finalize_authoritative_corrected_pareto.py` (publication audit), and `src/mfsi/poisson.py` (weighted-Poisson solvers).

## Software, provenance, and limitations

The accepted run was documented with Python 3.12.3, NumPy 2.5.1, SciPy 1.18.0,
JAX/jaxlib 0.8.3, and 64-bit JAX. The frozen-input manifest is authoritative
for the truth, endpoint, checkpoint, rollout, selection, validation, and
configuration hashes.

| File | SHA-256 |
|:---|:---|
| `experiment.py` | `5bcd5b3c96668cabf6d7a8b2b1944f48f490635763b997172584328551a9a4c4` |
| `selection.py` | `baf15fceff7b926d25471560e3342fe7fe7aaaa3993998b3f2274af8094d99ed` |
| `run_pareto.py` | `351b14c5aa3ef41b929ffb548a3c47ab6f2136e7c78494e3043159225cefa62a` |
| `finalize_authoritative_corrected_pareto.py` | `e5c146dcd176d8c96937536b0c4575afdf659382f8220c30ecb71b6b3f1aaeb0` |

| Authoritative artifact | SHA-256 |
|:---|:---|
| Corrected Pareto JSON | `cdf84cd0e8277c8b3f89bf950d82a349f18972fff9986a5b1a217e213fac89aa` |
| Certification diagnostic | `4bda8b411ed38f5078f22c343ec31f74c38336f1b5876d8d547a19f08f57af7a` |
| Authoritative run summary | `4414ec8308bd1121ce1d3bea28f4a4bd3bfe68defd15e7ed4ef25edfb8017daa` |

| Frozen input | SHA-256 |
|:---|:---|
| `truth_bank.npz` | `d897ff7fc44c0b85d7bb5391c0cc25895b4301e9c2ce00184697a1899d853b5b` |
| `reference_endpoints.npz` | `ad4006927e268c52f621c16c773f0600d803370bd21fb5e0816d82a70dbdfbba` |
| `reference.npz` | `63619893b44d49f6fd89f210239e0428d3d41aabfb4789cff616973066d21138` |
| `reference_bank.npz` | `159515f930ab1b82c0a9ef42706705f192af295ca2c74a9611da1558464a0b2f` |
| `selection_bank.npz` | `0ae52680ba66f07e36e02a0d85d25847fc11dc2554fcb63f95cb4e7aa0636ef9` |
| `validation_bank.npz` | `63748a79d00bce58e6307f2070f29c480998de0a7c5c47b4fcb0788696dea894` |
| `source_manifest.json` | `114610c3422cc791622e94bac36c3f589d77ba0e52a5a48dbe1284cb68de3967` |
| frozen `config.json` | `8f57f167675718b19d7ffc1741a8175adbe22069ff4043634b62df8dcf100ed0` |

[`reference_seed_sensitivity.md`](reference_seed_sensitivity.md) is a
historical three-seed audit performed with the earlier evaluator and a fixed
provenance-seeded Full geometry. Its `50.54%–54.15%` reductions must not be
mixed with the corrected `59.17%–70.18%` Pareto headline. It supports only the
narrow conclusion that the tested learned references gave similar conclusions
under that older controlled setup; it is not a corrected-Pareto rerun.

The authoritative study uses one frozen truth/reference realization and a
finite validation bank; the heavy-tailed Law action makes a larger independent
validation bank especially useful. The current result does not establish
global optimality or sensitivity to particle count, sensor width, detector
noise, grid resolution, or de-novo searches without archived candidate seeds.
End-to-end corrected wall time and peak memory were not stored.

## Read-only saved-result evaluation

From the repository root:

```bash
.venv/bin/python experiments/vortices_percentage/eval.py
.venv/bin/python experiments/vortices_percentage/eval_pareto.py
```

The first command displays the older tracked base/source run and is not the
publication result. The second displays and hash-verifies the corrected
authoritative Pareto sweep. Neither command runs the experiment or writes
outputs. Both use the repository-wide saved-evaluator table style, include
Law/Tangent/Full, and report sample SDs from the saved independent validation
trials (or from a saved ordinary SE and `n`).
