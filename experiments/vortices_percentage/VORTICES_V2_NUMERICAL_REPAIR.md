# Vortices V2 numerical repair

> **Historical first-stage record.** This document records the initial
> source-column-normalized V2 development method and its unresolved continuity
> gate. The subsequent commutator audit is
> [`VORTICES_V2_CONTINUITY_COMMUTATOR_AUDIT.md`](VORTICES_V2_CONTINUITY_COMMUTATOR_AUDIT.md),
> and the prospectively adopted reflection/Neumann pre-freeze result is
> [`VORTICES_V2_NUMERICAL_REPAIR_FINAL.md`](VORTICES_V2_NUMERICAL_REPAIR_FINAL.md).
> The tables below remain unchanged as historical development evidence.

## Status and decision

This is a **development-only numerical repair**. It does not replace, amend, or
reinterpret the immutable V1 confirmatory result.

The V2 Full action is grid-convergent on the declared development cases when
the density-estimation bandwidth is held fixed in physical units. The
`256 x 128` solve is valid in every case and bandwidth tested. Recalibrated
hard-fiber actions also stabilize over nested reference sizes `8,192`, `16,384`,
and `32,768`. The independent SciPy implementation agrees with production to
near machine precision.

The golden V1 event is removed by the numerical repair without changing its
hard information projection: its `t=0.5` action changes from
`179303.091252` to `28.453386`, and its integrated action changes from
`9000.510131` to `2.587036`. This is causal development evidence that the V1
catastrophe was dominated by its nonconvergent raster/PDE limit. It is not a V2
Full-vs-Law validation result.

One issue remains before the method should be frozen. The centered-difference
strong continuity check improves as `epsilon` decreases and has correlation
above `0.9998`, while weak errors are about `0.13%` at the smallest epsilon.
However, its strong relative L2 error is `0.94%`, `1.16%`, and `1.59%` on the
three successively finer grids, so strong-form grid convergence is not
demonstrated. A likely explanation is a commutator between particle motion,
cell-centered CIC deposition, and the source-column-normalized boundary
kernel. That explanation is plausible, not proven. The action convergence is a
PASS; readiness to freeze a final scientific experiment is therefore **NO**.

No soft moment fiber was implemented. The Phase 8 evidence does not justify
changing the information projection before the remaining raster-continuity
question is resolved.

The full-precision authority is
[`outputs/vortices_v2_development/convergence_summary.json`](outputs/vortices_v2_development/convergence_summary.json).
The flat analysis table is
[`outputs/vortices_v2_development/convergence_summary.csv`](outputs/vortices_v2_development/convergence_summary.csv).

## 1. Scientific separation from V1

V1 is preserved in `experiments/vortices_percentage/`. V2 lives entirely in
`experiments/vortices_percentage_v2/` and writes only to its own `outputs/`
tree. The V2 run verified the historical scientific inputs before evaluation:

| V1 file | SHA-256 |
|:---|:---|
| `experiments/vortices_percentage/config.json` | `8f57f167675718b19d7ffc1741a8175adbe22069ff4043634b62df8dcf100ed0` |
| `experiments/vortices_percentage/experiment.py` | `5bcd5b3c96668cabf6d7a8b2b1944f48f490635763b997172584328551a9a4c4` |

No file under `experiments/vortices_percentage/outputs/pareto/` was written by
the V2 harness. Namespace `19892` and the V1 selected geometries are used only
as an immutable mechanism/development set. They cannot confirm V2, and the old
sensor winners cannot be reused as final V2 winners.

## 2. Mathematical convention

The particle forcing is the positive continuity defect

```text
s = partial_t q + div(q u) = q h.
```

The correction and operator are

```text
delta = -grad psi,
K(q) = -div(q grad),
K(q) psi = -s.
```

Therefore

```text
partial_t q + div(q (u + delta)) = 0.
```

For a measured feature `Phi`, the weak identity is

```text
E_q[J_Phi delta]
  = integral Phi s
  = c_dot - E_q[J_Phi u].
```

V1 stored the same positive defect but described `delta=-grad psi` while
solving the opposite right-hand side. V2 solves `K psi=-s`. Flipping only the
source flips `psi` and `delta`; all three checked scalar actions change by zero
to numerical precision. Archived V1 scalar actions are untouched because the
quadratic energy is sign invariant.

## 3. Fixed continuum discretization

Let `D_G(x_i)` be bilinear cloud-in-cell deposition of particle `i` onto cell
centers. Both mass and signed source mass use the same weights:

```text
m_G       = sum_i w_i D_G(x_i),
sigma_G   = sum_i w_i h_i D_G(x_i).
```

Let `S_h = S_y,h (.) S_x,h^T` be the separable Gaussian map on physical cell
centers. Each source-cell column is normalized over the complete rectangular
domain. V2 forms

```text
M_h       = S_h m_G,
Sigma_h   = S_h sigma_G,
q_h       = M_h / cell_area,
s_h       = Sigma_h / cell_area.
```

The common mass normalization is applied to both fields. Only the residual
floating-point total source is removed, distributed as `M_h`; no independent
source rescaling is performed. There is no density floor. Across all 63
grid/bandwidth evaluations:

| Check | Global value |
|:---|---:|
| Maximum mass error | `3.33e-16` |
| Maximum absolute integrated source | `8.88e-16` |
| Minimum `q_h` | strictly positive in every evaluated field |
| Maximum component count | `1` |
| Maximum physical-Poisson relative residual | `1.62e-11` |

The same positive full-domain raster implementation is shared with the
corrected analytic benchmark through `mfsi.raster`; V2 supplies the rectangular
grid and a genuinely fixed physical bandwidth.

## 4. Frozen bandwidth rule

For each scientific time node, V2 computes

```text
scale_t = sqrt((weighted_var_x + weighted_var_y) / 2),
N_eff,t = 1 / sum_i b_ti^2,
h_t     = scale_t N_eff,t^(-1/6).
```

The one physical bandwidth is the median over the 21 frozen reference times:

```text
h_ref = 0.05883961987664522.
```

It depends only on the frozen reference rollout. It has no grid-cell floor and
does not depend on sensor geometry, allowance, observation trial, or action.
The predeclared sensitivity set is `0.75 h_ref`, `h_ref`, and `1.25 h_ref`.
All three converge; the default remains the unscaled Scott rule rather than
selecting the multiplier with the smallest action or condition proxy.

| Bandwidth | Worst `128 -> 256` relative action change | Largest diagonal condition proxy |
|---:|---:|---:|
| `0.75 h_ref` | `0.598%` | `3.02e11` |
| `1.00 h_ref` | `0.468%` | `1.36e8` |
| `1.25 h_ref` | `0.344%` | `3.24e6` |

Bandwidth changes the scientific smoothed continuum object, so the action
differences across these rows are sensitivity results—not discretization
error and not a basis for tuning Full-vs-Law performance.

## 5. Development cases

The cases were fixed before V2 actions were read. Trials 114 and 83 were drawn
from namespace `19892` with seed `20260830`, excluding declared tail cases; no
action was used in their selection.

| Case | Geometry | Trial | Role |
|:---|:---|---:|:---|
| golden Full 4% | `ea6c90af64ce4356` | 130 | dominant known V1 tail |
| golden Law | `f8fdd998b4627969` | 130 | paired Law control |
| golden Full 2%/3% | `ce783572fe3170da` | 130 | paired Full control |
| known Law tail | `f8fdd998b4627969` | 65 | additional V1 tail |
| known Full 1% tail | `41ca33ec45daa976` | 240 | additional V1 tail |
| ordinary Law 1 | `f8fdd998b4627969` | 114 | action-blind random sample |
| ordinary Law 2 | `f8fdd998b4627969` | 83 | action-blind random sample |

## 6. PDE-grid convergence

Every row uses the same 32,768-particle projected weights and forcing, the same
physical bandwidth `h_ref`, and a newly assembled grid operator.

| Case | `64 x 32` | `128 x 64` | `256 x 128` | `64 -> 128` | `128 -> 256` |
|:---|---:|---:|---:|---:|---:|
| golden Full 4% | 2.551104 | 2.579494 | 2.587036 | 1.101% | 0.292% |
| golden Law | 1.288281 | 1.308536 | 1.313824 | 1.548% | 0.403% |
| golden Full 2%/3% | 1.261694 | 1.279277 | 1.283897 | 1.374% | 0.360% |
| known Law tail | 2.935273 | 2.972016 | 2.981689 | 1.236% | 0.324% |
| known Full 1% tail | 1.570623 | 1.592498 | 1.598251 | 1.374% | 0.360% |
| ordinary Law 1 | 1.356548 | 1.376692 | 1.381944 | 1.463% | 0.380% |
| ordinary Law 2 | 1.482462 | 1.509647 | 1.516740 | 1.801% | 0.468% |

The declared medium-to-fine gate is 5%; the observed worst case is 0.468%.
Changes also shrink from coarse-to-medium to medium-to-fine for every case.
This is a genuine fixed-object convergence result, unlike V1 refinement where
the physical bandwidth contracted with `dx`.

![Action versus grid](outputs/vortices_v2_development/action_vs_grid_resolution.png)

## 7. Particle-count convergence

This is separate from the PDE-grid study. At each `N`, V2 takes a deterministic
nested prefix of the frozen rollout, renormalizes its base weights, and
recalibrates the information projection and multiplier derivative on that
empirical law. It does not project once at 32,768 and then subsample a raster.
All actions use the `256 x 128` grid and `h_ref`.

| Case | `N=8,192` | `N=16,384` | `N=32,768` | `16,384 -> 32,768` |
|:---|---:|---:|---:|---:|
| golden Full 4% | 2.440315 | 2.513204 | 2.587036 | 2.854% |
| golden Law | 1.358750 | 1.328337 | 1.313824 | 1.105% |
| golden Full 2%/3% | 1.309048 | 1.285074 | 1.283897 | 0.092% |

The declared final-step gate is 10%; the worst observed value is 2.854%.
No 65,536-particle bank was generated. The original sampling routine does not
guarantee that a newly generated 65,536 bank preserves the frozen 32,768 bank
as an exact prefix, so that optional independent-rollout study remains future
work rather than being silently mixed with the nested-prefix result.

![Action versus particle count](outputs/vortices_v2_development/action_vs_particle_count.png)

## 8. Causal V1/V2 comparison

Only the raster/PDE convention changes in this table. The noisy observations,
hard target path, projection, covariance, multiplier derivative, particle
weights, and forcing remain the same.

| Case | V1 integrated | V2 integrated | V1 at `t=.5` | V2 at `t=.5` | Max `||lambda_dot||` |
|:---|---:|---:|---:|---:|---:|
| golden Full 4% | 9000.510131 | 2.587036 | 179303.091252 | 28.453386 | 13562.91 |
| golden Law | 10.878760 | 1.313824 | 104.852491 | 1.133815 | 522.04 |
| golden Full 2%/3% | 56.989160 | 1.283897 | 625.603778 | 3.160944 | 1411.93 |
| known Law tail | 1118.726950 | 2.981689 | 22241.422400 | 21.447439 | 12725.24 |
| known Full 1% tail | 1082.947848 | 1.598251 | 21546.772807 | 6.388694 | 8951.01 |
| ordinary Law 1 | 7.512177 | 1.381944 | 53.810184 | .979880 | 872.01 |
| ordinary Law 2 | 8.672929 | 1.516740 | 61.404291 | 1.009937 | 405.72 |

The hard fiber remains ill-conditioned and produces large `lambda_dot` in the
known noisy cases. Yet none of the curated V2 instantaneous actions exceeds
`28.46`, versus the development-only catastrophic threshold `1000`. This
localizes the V1 explosion to how that forcing was represented by a shrinking,
zero-support conductivity. It does not prove that a broader fresh V2 bank has
no heavy tail.

![Bandwidth sensitivity](outputs/vortices_v2_development/bandwidth_sensitivity.png)

## 9. Continuity and moment checks

The golden case was independently recalibrated at `t-epsilon`, `t`, and
`t+epsilon`. All three densities use the same fixed physical kernel. A
finite-volume divergence of the independently deposited velocity flux was
compared with the deposited positive defect.

At the smallest `epsilon=2e-4`:

| Grid | Relative L2 | Correlation | Maximum weak relative error |
|:---|---:|---:|---:|
| `64 x 32` | 0.944% | .9999603 | .1303% |
| `128 x 64` | 1.160% | .9999358 | .1239% |
| `256 x 128` | 1.595% | .9998752 | .1328% |

For each grid, decreasing epsilon from `1e-3` to `2e-4` reduces the strong
error substantially. The particle weak moment identity

```text
E[J_Phi delta] = c_dot - E[J_Phi u]
```

has maximum absolute error `1.31e-13`. Nevertheless, the strong residual is
not monotone under grid refinement. This is why the report does not declare
the complete raster continuity representation frozen. The exact arrays and
all epsilon/grid rows are retained in the JSON authority.

## 10. Independent Poisson verification

The second solver assembles an unscaled SciPy incidence matrix, uses a
mean-zero KKT gauge, and does not reuse the production component pinning,
symmetric scaling, refinement, or fallback.

| Case | Grid | Production | Independent | Relative action error | Weighted-gradient error |
|:---|:---|---:|---:|---:|---:|
| golden Full 4% | `256 x 128` | 28.453386127 | 28.453386127 | `2.2e-15` | `6.9e-14` |
| golden Law | `128 x 64` | 1.132126640 | 1.132126640 | `7.8e-14` | `5.9e-14` |
| golden Full 2%/3% | `128 x 64` | 3.154681940 | 3.154681940 | `1.7e-15` | `1.5e-14` |

The maximum production residual in the complete development run is
`1.62e-11`. The independent residuals above are below `5e-13`. The energy
identity and sign-flip invariance tests also pass.

## 11. Figures and exact fields

![Golden density](outputs/vortices_v2_development/golden_q.png)

The log-density view shows the strictly positive boundary-normalized
conductivity used by the `256 x 128` golden solve. Unlike V1, there are no
zero-density islands or disconnected conductive components.

![Golden source](outputs/vortices_v2_development/golden_source.png)

The signed field is the smoothed positive continuity defect. Its integral is
zero to floating-point precision. Red and blue regions are not clipped.

![Golden energy density](outputs/vortices_v2_development/golden_energy_density.png)

The energy is no longer concentrated almost entirely in a few cells. At the
default bandwidth, the largest top-1% share across all time nodes of the
golden case is about `21.85%`, versus `99.971%` in V1.

The combined four-panel diagnostic is
[`golden_q_source_energy.png`](outputs/vortices_v2_development/golden_q_source_energy.png),
and the exact arrays are
[`golden_v2_fields.npz`](outputs/vortices_v2_development/golden_v2_fields.npz).

## 12. Implementation map

- `config.json` declares the V2 scientific identity, sign, bandwidth rule and
  value, development cases, convergence grids, particle counts, and gates.
- `core.py` loads immutable V1 inputs, recalibrates each empirical hard fiber,
  computes the reference-only Scott bandwidth, applies common CIC/Gaussian
  rasterization, solves `K psi=-s`, implements the independent solver, and
  performs continuity diagnostics.
- `diagnose_vortices_v2_convergence.py` fixes the case set, runs PDE-grid,
  bandwidth, particle, continuity, sign, energy, and independent-solver
  studies, then writes the JSON/CSV/plots.
- `test_v2.py` covers V1 hashes, sign and energy identities, physical bandwidth,
  common operators, mass/source conservation, positivity, axis orientation,
  independent solver agreement, hard-fiber `lambda_dot` finite differences,
  float64, frozen bank identity, and config fingerprint sensitivity.

No source under the V1 experiment was modified for V2. The only reused shared
primitive is the already implemented positive rectangular raster in
`src/mfsi/raster.py`.

## 13. Reproduction

From the repository root, with the existing environment and native projection
backend available:

```bash
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/src:$PWD/experiments/vortices_percentage_v2"
export OMP_NUM_THREADS=4

.venv/bin/python -m pytest \
  experiments/vortices_percentage_v2/test_v2.py \
  tests/test_positive_raster.py -q

.venv/bin/python \
  experiments/vortices_percentage_v2/diagnose_vortices_v2_convergence.py
```

The recorded focused test result is `11 passed`. The run records repository
HEAD, dirty-worktree status, code/config hashes, V1 bank hash, namespace,
projection type, raster definition, physical bandwidth, grid, and particle
count in the JSON artifact. `--bandwidth` is diagnostic-only; using it changes
the recorded V2 object. `--include-512` requests the optional `512 x 256` grid
but was not used for this report.

## 14. Remaining limitations

- The strong raster continuity error is small but not grid-convergent under the
  present flux check. This must be explained or repaired before freezing.
- The development set uses old namespace `19892`; it is mechanism evidence,
  not independent V2 validation.
- Only seven cases were studied. Absence of a catastrophic event here does not
  establish a light-tailed V2 action distribution.
- The 65,536-particle and optional `512 x 256` studies were not run.
- V2 has not trained three independent reference models, rerun Population/Law/
  Tangent/Full selection, or generated any fresh V2 validation bank.
- The bandwidth sensitivity changes action materially even though each fixed
  bandwidth converges. The rule—not a favorable multiplier—must remain frozen.
- The hard-fiber weak covariance amplification still exists. Soft projection
  may become justified only if catastrophic events persist after the remaining
  numerical issue is resolved and on a prospectively defined development set.

## Final decision gate

A. Legacy V1 preserved?
   PASS

B. Sign convention repaired and tested?
   PASS

C. Fixed-physical-bandwidth raster implemented?
   PASS

D. Grid convergence demonstrated?
   PASS

E. Empirical particle convergence demonstrated?
   PASS

F. Does hard-fiber V2 still show catastrophic noise-induced action tails?
   INCONCLUSIVE

G. Was soft fiber therefore scientifically justified?
   NO

H. If implemented, does soft-fiber derivative pass independent finite
   differences?
   NOT IMPLEMENTED

I. Is the V2 method ready to freeze before a new selection/validation study?
   NO

J. Remaining unresolved risks.
   Strong raster-continuity grid convergence is unresolved; only seven old
   development cases and one frozen reference were examined; `65,536`-particle
   and `512 x 256` studies are absent; and no fresh multi-reference V2 selection
   or validation study has been run.
