# Vortices V2 continuity-commutator audit

> **Post-audit status.** Candidate B was subsequently promoted prospectively
> into the V2 production development path and passed the complete pre-freeze
> suite. See
> [`VORTICES_V2_NUMERICAL_REPAIR_FINAL.md`](VORTICES_V2_NUMERICAL_REPAIR_FINAL.md).
> This report and its recorded hashes remain the immutable mechanism audit that
> motivated that choice; its column-normalized code survives only for audit
> replay, not as the V2 scientific action.

## Status and decision

The remaining V2 strong-continuity anomaly has been explained. It is not a
failure of the V2 weighted-Poisson action, but the pre-freeze continuity gate
is **not mathematically valid as written**.

The old diagnostic did deposit and smooth the natural particle flux

```text
j_h = S_CIC(q u),
```

not `q_h` times a velocity sampled on the raster. It then computed divergence
by averaging this cell-centered flux to faces and forcing all normal boundary
faces to zero. That last operation is inconsistent with the current
source-column-normalized Gaussian: its natural smoothed flux generally has
nonzero normal boundary values. The old gate also omitted the exact
boundary-normalization commutator.

For

```text
K(y|x) = g_h(y-x) / Z(x),
```

the correct regularized identity is

```text
partial_t q_h + div_y(j_h) = s_h + C_h,

C_h(y) = -integral K(y|x) q(x)
                     [u(x) dot grad_x log Z(x)] dx.
```

The sign in this formula was checked analytically and against an independent
finite difference. The maximum componentwise discrepancy in that check was
`7.21e-9`.

On the golden V1 event at `t=0.5` and `epsilon=2e-4`:

- replacing the artificial zero-face divergence by the analytic divergence
  of the natural column-kernel flux reduces relative L2 error from
  `0.944%--1.595%` to `0.395%--0.412%`;
- subtracting the analytic commutator reduces it further to
  `0.247%--0.254%`;
- the corrected error decreases from `0.2543%` to `0.2474%` over
  `64 x 32` through `256 x 128`;
- the corrected error also decreases strongly as `epsilon` decreases, from
  about `5.91%` at `1e-3`, through `1.53%` at `5e-4`, to `0.247%` at
  `2e-4`; and
- the corrected commutator integral error decreases by approximately four on
  each grid refinement: `.0022751`, `.0005520`, `.0001383`.

The commutator is therefore real and quantitatively important, but it is not
the only finite-resolution term. Centered time differentiation and the
piecewise-linear CIC deposition contribute the remaining error. The old
nonmonotone `0.944%--1.595%` trend was dominated by the inconsistent forced
zero-boundary closure.

An independent reflected/Neumann particle kernel conserves mass without an
`x`-dependent normalizer, has exactly zero normal reflected flux, and satisfies
manufactured continuity to relative L2 error `9.41e-11--3.74e-10`. On the
golden case it reaches the same finite-time residual floor as the corrected
column kernel.

The scientifically coherent recommendation is **candidate B: a
reflection/Neumann density/source kernel with its consistently odd-reflected
normal flux**. It preserves the ordinary no-flux continuity equation and keeps
`s_h` as the Poisson source. It has not been adopted here. Adopting it changes
the scientific raster and action and must be recorded prospectively before
any new selection or validation.

No selection or validation bank was created, no soft fiber was implemented,
and no bandwidth or tolerance was tuned. The validation protocol draft was
not changed.

## 1. Scope and frozen inputs

This was a development-only mechanism audit. It reused only the already
declared golden development case:

| Item | Value |
|:---|:---|
| V1 observation namespace | `19892` |
| Trial | `130` |
| Geometry | Full 4%, key `ea6c90af64ce4356` |
| Scientific time | `t=0.5` |
| Particles | `32,768` |
| Physical bandwidth | `0.05883961987664522` |
| Grids | `64 x 32`, `128 x 64`, `256 x 128` |
| Golden finite differences | `1e-3`, `5e-4`, `2e-4` |
| Manufactured finite difference | `2e-5` |

The hard information projection, reconstructed target path, reference
particles, reference velocities, multiplier derivative, particle forcing,
physical domain `[0,2] x [0,1]`, and V2 sign convention were unchanged.

The machine-readable development authority is
[`outputs/vortices_v2_continuity_commutator/continuity_commutator_summary.json`](outputs/vortices_v2_continuity_commutator/continuity_commutator_summary.json).
The flattened rows are in
[`continuity_commutator_summary.csv`](outputs/vortices_v2_continuity_commutator/continuity_commutator_summary.csv).
These files live under the ignored V2 `outputs/` tree and are reproducible from
the frozen bank and code.

## 2. Exact derivation and sign

Write the rectangular domain as

```text
Omega = [a_1,b_1] x [a_2,b_2]
```

and let `g_h` be an isotropic Gaussian. The current smoothing kernel is

```text
K(y|x) = g_h(y-x) / Z(x),
Z(x)   = integral_Omega g_h(z-x) dz.
```

Because `grad_x g_h(y-x) = -grad_y g_h(y-x)`, differentiation of the
quotient gives

```text
grad_x K(y|x)
  = -grad_y K(y|x) - K(y|x) grad_x log Z(x).
```

Assume the unsmoothed law obeys the positive-defect convention

```text
partial_t q + div_x(q u) = s
```

and that the underlying transport has no boundary contribution. Define

```text
q_h(y) = integral K(y|x) q(x) dx,
j_h(y) = integral K(y|x) q(x) u(x) dx,
s_h(y) = integral K(y|x) s(x) dx.
```

Then

```text
partial_t q_h
  = integral K [s - div_x(q u)] dx
  = s_h + integral grad_x K dot (q u) dx
  = s_h - div_y(j_h)
      - integral K q [u dot grad_x log Z] dx.
```

Therefore

```text
partial_t q_h + div_y(j_h) = s_h + C_h,

C_h(y) = -integral K(y|x) q(x)
                    [u(x) dot grad_x log Z(x)] dx.
```

The minus sign in `C_h` follows directly from differentiating `1/Z(x)`.
It is not a convention chosen from numerical results.

### Analytic rectangular normalizer

Separability gives `Z(x)=Z_1(x_1)Z_2(x_2)`. For coordinate interval `[a,b]`,

```text
Z_d(x) = Phi((b-x)/h) - Phi((a-x)/h),

d Z_d / dx
  = [phi((a-x)/h) - phi((b-x)/h)] / h,

d log Z_d / dx = (d Z_d / dx) / Z_d(x),
```

where `Phi` and `phi` are the standard-normal CDF and density. Near a lower
boundary the derivative is positive; near an upper boundary it is negative.
The analytic implementation was checked at four near-corner points and the
box center by an independent centered difference with step `1e-7`. The maximum
absolute component error was `7.2041644e-9`.

## 3. What the previous diagnostic actually computed

The previous implementation in `core.continuity_check` did the following:

1. independently recalibrated projected weights at `t-epsilon`, `t`, and
   `t+epsilon`;
2. rasterized the two endpoint particle laws and formed a centered `dq_h/dt`;
3. deposited the particle quantities `w_i u_{i,x}` and `w_i u_{i,y}` by CIC;
4. smoothed both deposited flux components by the same source-column-normalized
   Gaussian used for density and source; and
5. averaged the cell-centered fluxes to faces, setting the exterior normal
   faces to zero, before taking a finite-volume divergence.

Thus its flux was

```text
j_h = S_h(D_CIC(w u)),
```

not

```text
q_h u(reference velocity sampled on the grid).
```

The flux construction itself is the natural one. The problem was its
subsequent boundary closure and the omitted commutator.

The audit separately differentiates the actual discrete column kernel in the
target coordinate. That supplies the natural divergence of the cell-centered
smoothed particle flux without silently imposing zero normal boundary flux.
It also evaluates `grad_x log Z` both at particles and at CIC source cells.
Those two commutator versions agree closely in the corrected golden residual.

## 4. `j_h=S(q u)` is not `q_h u`

For a spatially varying velocity,

```text
S(q u) != S(q) u.
```

The velocity transported by the smoothed law is instead

```text
u_h(y) = j_h(y) / q_h(y),
```

which is well defined here because the V2 density is strictly positive. Then
`j_h=q_h u_h` by definition. Multiplying `q_h` by the original reference
velocity evaluated at raster cells would define a different flux and would
not follow from smoothing the particle continuity equation.

The present V2 diagnostics implicitly use the mathematically natural
`S(q u)` flux. The V2 Poisson solve uses `q_h` only as the conductivity for the
correction field; it does not reconstruct the reference flux as `q_h u`.

## 5. Golden source-column-normalized result

All relative L2 values below are normalized by `||s_h||_2`. `R_current` uses
the old forced-zero-face divergence. `R_naive` uses the analytic target
derivative of the natural column-kernel flux. `R_corrected=R_naive-C_h` uses
the particle analytic commutator.

### Smallest epsilon: complete strong and weak metrics

| Grid | Residual | Relative L2 | L1 | Maximum absolute | Maximum weak relative | `corr(R_naive,C_h)` |
|:---|:---|---:|---:|---:|---:|---:|
| `64 x 32` | `R_current` | .94424% | .236756 | 1.973668 | .92843% | -- |
| `64 x 32` | `R_naive` | .41242% | .116369 | .467304 | 1.06026% | .759917 |
| `64 x 32` | `R_corrected` | .25427% | .059853 | .425232 | .24323% | -- |
| `128 x 64` | `R_current` | 1.16017% | .182229 | 4.230726 | .62392% | -- |
| `128 x 64` | `R_naive` | .40115% | .108367 | .579076 | 1.03629% | .750463 |
| `128 x 64` | `R_corrected` | .25184% | .049383 | .579317 | .24218% | -- |
| `256 x 128` | `R_current` | 1.59460% | .172363 | 8.688769 | .54676% | -- |
| `256 x 128` | `R_naive` | .39534% | .105925 | .532513 | 1.02079% | .752363 |
| `256 x 128` | `R_corrected` | .24735% | .046345 | .532720 | .24085% | -- |

The correlation requested here is between the residual field and the
commutator field. It is not the old correlation between `dq_h/dt+div(j_h)` and
`s_h`, which exceeded `.9998` because both fields share a much larger common
signal.

The weak column in this audit uses a deliberately stricter seven-function
family: the constant plus six localized Gaussians near corners and the two
interior halves. It is therefore not numerically identical to the earlier
five-function weak check whose maximum was about `.13%`.

The commutator explains a substantial and spatially coherent part of the
natural residual, but `R_naive` is not pointwise equal to `C_h` at the finite
time step: the correlation is about `.75`, not approximately one. The
remaining corrected field is consistent with the finite-time and CIC terms
isolated below.

### Epsilon and grid trend

| Epsilon | Grid | `R_naive` relative L2 | `R_corrected` relative L2 |
|---:|:---|---:|---:|
| `1e-3` | `64 x 32` | 5.89221% | 5.89538% |
| `1e-3` | `128 x 64` | 5.90691% | 5.91053% |
| `1e-3` | `256 x 128` | 5.90538% | 5.90914% |
| `5e-4` | `64 x 32` | 1.54116% | 1.51784% |
| `5e-4` | `128 x 64` | 1.54816% | 1.52737% |
| `5e-4` | `256 x 128` | 1.54510% | 1.52498% |
| `2e-4` | `64 x 32` | .41242% | .25427% |
| `2e-4` | `128 x 64` | .40115% | .25184% |
| `2e-4` | `256 x 128` | .39534% | .24735% |

At the two larger time steps, centered-difference/recalibration error dominates
the fixed commutator. At `2e-4`, the commutator is resolved and its subtraction
produces the decreasing grid trend.

### Integral identity

At `epsilon=2e-4`, `integral s_h` and `integral partial_t q_h` are below
`1.4e-13` in absolute value. The nonzero integral is therefore the flux through
the regularized boundary.

| Grid | `integral div(j_h)` | `integral C_h` | Corrected difference |
|:---|---:|---:|---:|
| `64 x 32` | .07030325 | .06802812 | .00227513 |
| `128 x 64` | .06858015 | .06802812 | .00055203 |
| `256 x 128` | .06816643 | .06802812 | .00013832 |

The factor-of-four reduction in the final column is the expected quadrature
trend for the smooth boundary integral. It also proves why forcing the normal
boundary flux to zero cannot be part of the column-kernel identity.

![Golden commutator fields](outputs/vortices_v2_continuity_commutator/golden_commutator_fields.png)

The four panels show `R_naive`, the analytic particle commutator, and the two
corrected residuals. They make the common boundary-localized structure and the
smaller remaining finite-resolution field visible.

![Continuity grid trends](outputs/vortices_v2_continuity_commutator/continuity_grid_trends.png)

This plot separates the natural column residual, its commutator-corrected
version, and the reflected ordinary-identity residual at `epsilon=2e-4`.

## 6. Why a residual remains after subtracting `C_h`

The analytic formula above is exact for the continuous kernel
`g_h(y-x)/Z(x)`. The current implementation first maps a moving particle to
fixed cell centers by CIC and only then applies a column-normalized Gaussian.
Its effective kernel is

```text
K_eff(y|x) = sum_G S_h(y|x_G) D_G(x).
```

Differentiating this object includes derivatives of the piecewise-linear CIC
weights. Those terms are not the same as differentiating a Gaussian centered
directly at the particle. The audit therefore distinguishes:

- the exact boundary-normalization commutator of the Gaussian columns;
- the CIC transport/deposition commutator; and
- the centered time/reprojection truncation error.

The particle and source-cell versions of the Gaussian-normalization
commutator give essentially the same corrected golden residual. Neither is
expected to remove the independent CIC derivative or finite-time error.

Consequently the supported classification is precise: the boundary
commutator is real and explains an important part of the discrepancy; the
artificial zero-face divergence explains the earlier adverse grid trend; and
the remaining small residual is a finite time/CIC error, not evidence of a
failed Poisson action.

## 7. Manufactured tests

Three deterministic systems used known particle trajectories and exact
exponential weight evolution. The forcing was centered under the particle
weights, so the source preserves total mass. The systems were:

- constant velocity on particles separated from the boundary;
- particles placed `0.012` from every boundary and moving inward; and
- a divergence-free field tangent to all four boundaries.

The manufactured centered-difference step was `2e-5`.

### Column-normalized kernel

| Case | Grid | Naive relative L2 | Corrected relative L2 | `corr(R_naive,C_h)` |
|:---|:---|---:|---:|---:|
| constant interior | `64 x 32` | 9.8808% | 9.8810% | -.0528 |
| constant interior | `128 x 64` | 4.9971% | 4.9967% | .1126 |
| constant interior | `256 x 128` | 2.5887% | 2.5889% | -.0573 |
| near all boundaries | `64 x 32` | 253.888% | 204.376% | .5927 |
| near all boundaries | `128 x 64` | 273.446% | 8.5939% | .999621 |
| near all boundaries | `256 x 128` | 271.641% | 7.8020% | .999589 |
| tangent field | `64 x 32` | 10.0766% | 9.6191% | .2982 |
| tangent field | `128 x 64` | 6.1481% | 5.2575% | .5179 |
| tangent field | `256 x 128` | 4.0445% | 2.5085% | .7842 |

The interior case has negligible boundary commutator and exposes the
first-order CIC transport error. The boundary case makes the commutator
dominant on the two finer grids. The tangent field contains both effects and
the corrected residual decreases with refinement.

### Reflection/Neumann kernel

| Case | `64 x 32` | `128 x 64` | `256 x 128` |
|:---|---:|---:|---:|
| constant interior | `1.10e-10` | `1.18e-10` | `1.37e-10` |
| near all boundaries | `2.40e-10` | `2.72e-10` | `3.74e-10` |
| tangent field | `9.41e-11` | `1.06e-10` | `1.41e-10` |

These are relative L2 errors. Maximum weak relative errors over the same rows
are at most `3.87e-10`. The values are centered-difference roundoff/truncation,
not spatial inconsistency. This independently verifies the reflection
derivative and flux signs.

## 8. Reflection/Neumann diagnostic

For one interval `[a,b]` of length `L`, candidate B uses the even reflected
Gaussian for scalar mass/source and the odd reflected Gaussian for the normal
flux. Schematically,

```text
K_N(y,x) = sum_k [g(y-x+2kL) + g(y+x-2a+2kL)],
K_D(y,x) = sum_k [g(y-x+2kL) - g(y+x-2a+2kL)].
```

Cell-integrating `K_N` conserves every source column without a normalization
depending on `x`. `K_D` is zero on both boundary faces. The derivative identity

```text
partial_x K_N = -partial_y K_D
```

then gives ordinary continuity under the tensor-product two-dimensional
construction. Density/source use even reflection in both coordinates. The
`x` flux uses odd reflection in `x` and even reflection in `y`; the `y` flux
uses the converse.

The implementation integrates the scalar kernels over cells and evaluates
the flux kernels at faces, so its finite-volume divergence is the exact cell
integral of the reflected kernel derivative up to the truncated image sum.
Four image pairs were sufficient at the fixed bandwidth; the maximum scalar
column-mass error was `1.56e-15`.

### Golden reflected continuity

| Epsilon | Grid | Relative L2 | L1 | Maximum absolute | Maximum weak relative |
|---:|:---|---:|---:|---:|---:|
| `1e-3` | `64 x 32` | 5.90706% | 1.068642 | 12.715314 | 5.75332% |
| `1e-3` | `128 x 64` | 5.91001% | 1.072214 | 13.001941 | 5.74982% |
| `1e-3` | `256 x 128` | 5.91075% | 1.073146 | 13.063910 | 5.74877% |
| `5e-4` | `64 x 32` | 1.52494% | .277249 | 3.249132 | 1.48435% |
| `5e-4` | `128 x 64` | 1.52572% | .278176 | 3.319358 | 1.48341% |
| `5e-4` | `256 x 128` | 1.52592% | .278404 | 3.340271 | 1.48313% |
| `2e-4` | `64 x 32` | .24720% | .044940 | .526170 | .24076% |
| `2e-4` | `128 x 64` | .24733% | .045093 | .537513 | .24060% |
| `2e-4` | `256 x 128` | .24737% | .045130 | .540993 | .24056% |

At a fixed epsilon the values approach an epsilon-dependent plateau as the
grid is refined. Across epsilon they decrease by the expected centered-time
trend. The manufactured tests show that the reflected spatial identity itself
is accurate to about `1e-10`.

## 9. Candidate comparison

The comparison is numerical only. It does not compare Full and Law and does
not reuse these values as selection evidence.

Candidate A below reports the existing V2 homogeneous-Neumann action driven
by `s_h` alone. That action is a useful record of the current implementation,
but Section 10 explains why it is not the action of the complete
commutator-aware column-kernel continuity defect. Candidate B reports the
reflected density/source with the ordinary homogeneous-Neumann correction.

| Quantity | A: column normalized | B: reflection/Neumann |
|:---|:---|:---|
| Scalar mass conservation | `<= 2.22e-16` | `<= 1.11e-16` before common normalization |
| Integrated source | `<= 8.88e-16` | `<= 1.78e-15` before centering |
| Positivity | PASS on every grid | PASS on every grid |
| Conductive components | one | one |
| Continuity identity | requires `C_h` and nonzero boundary flux | ordinary identity; zero normal reflected flux |
| Manufactured continuity | CIC-limited after commutator correction | `9.41e-11--3.74e-10` relative L2 |
| Golden continuity | corrected `0.247%--0.254%` at `2e-4` | `0.247%` plateau at `2e-4` |
| Fine-grid condition proxy | `6.16e6` | `6.88e6` |

### Golden `t=0.5` action convergence

| Grid | A existing `s_h` action | Relative change | B reflected action | Relative change |
|:---|---:|---:|---:|---:|
| `64 x 32` | 28.132012 | -- | 28.283175 | -- |
| `128 x 64` | 28.384900 | .8909% | 28.450166 | .5870% |
| `256 x 128` | 28.453386 | .2407% | 28.495213 | .1581% |

Both actions are grid convergent and numerically well solved. Candidate B's
fine-grid physical-Poisson relative residual is `5.15e-13`; all three reflected
solves are compatible. The similarity of the action values was not used to
choose the recommendation.

![Candidate action convergence](outputs/vortices_v2_continuity_commutator/candidate_action_convergence.png)

The plot shows that both boundary constructions have stable golden action
limits. It is a numerical comparison, not an optimization or validation plot.

## 10. Which source belongs in the scientific Poisson problem?

The answer follows from continuity, not from which action is smaller.

The V2 correction convention is

```text
delta_h = -grad psi_h,
K(q_h)  = -div(q_h grad),
```

and the corrected smoothed flux is `j_h + q_h delta_h`. For candidate A,

```text
partial_t q_h + div(j_h) = s_h + C_h.
```

Therefore exact regularized continuity requires

```text
div(q_h delta_h) = -(s_h + C_h),
K(q_h) psi_h     = -(s_h + C_h).
```

So `s_h` alone is not the complete regularized continuity defect of the
column-normalized law relative to its natural flux.

There is an equally important boundary condition. Since the column-normalized
`j_h` generally has nonzero boundary flux, no-flux continuity of the corrected
law requires

```text
(j_h + q_h delta_h) dot n = 0,
q_h partial_n psi_h = j_h dot n.
```

The integral of `C_h` equals the outward flux of `j_h` when `integral s_h=0`.
Consequently `s_h+C_h` is generally incompatible with the existing
homogeneous-Neumann Poisson operator. Merely adding `C_h` to the current source
and recentering it would discard the boundary balance and would not implement
the derived equation.

Candidate A can be made coherent only as a new coupled definition containing
both:

1. the volume defect `s_h+C_h`; and
2. the matching nonhomogeneous correction boundary flux.

That would change the scientific action and is not implemented in this audit.
Keeping the existing action driven by `s_h` alone is mathematically possible
only if V2 is explicitly redefined as an ad hoc smoothed-source metric and no
claim is made that it is the continuity correction of `(q_h,j_h)`. That is not
the intended FIDE interpretation.

For candidate B, reflection gives

```text
partial_t q_h + div(j_h) = s_h,
j_h dot n = 0.
```

The existing homogeneous-Neumann correction then coherently solves

```text
K(q_h) psi_h = -s_h.
```

This is why candidate B is recommended. It retains `s_h` for a mathematical
reason, not because its golden action happens to be close to candidate A.

## 11. Validity of the current pre-freeze gate

The current gate in `VORTICES_V2_VALIDATION_PROTOCOL_DRAFT.md` requires the old
strong residual to be below `2%`, correlated above `.999`, and monotone under
grid refinement. As presently specified, that gate is **invalid** for the
column-normalized method because it tests

```text
dq_h/dt + div_zero_face(S_CIC(q u)) = s_h
```

instead of either mathematically valid identity:

```text
dq_h/dt + div_natural(S_CIC(q u)) = s_h + C_h + CIC terms
```

or, for candidate B,

```text
dq_h/dt + div_reflected(j_h) = s_h.
```

Passing the old `2%` threshold would therefore not certify the intended PDE,
and failure of its monotonicity does not diagnose a failed scientific action.
The protocol draft was deliberately left unchanged; it must be amended only
after a boundary-kernel definition is prospectively selected and frozen.

## 12. Recommendation and freeze consequence

The audit recommends the following prospective V2 definition:

1. use the cell-integrated even-reflection/Neumann Gaussian for density and
   signed source;
2. use the matching odd-reflection kernel for each normal flux component;
3. retain the fixed physical bandwidth `0.05883961987664522` without tuning;
4. retain `s_h` as the positive defect and solve the current sign convention
   `K(q_h) psi_h=-s_h` with homogeneous Neumann correction flux;
5. replace the invalid strong gate by a reflected ordinary-continuity test,
   including manufactured sign/conservation tests and the golden epsilon/grid
   study; and
6. treat the change as a new V2 numerical-method amendment before any fresh
   selection or validation.

Candidate B is recommended, not adopted. This audit does not authorize a new
selection, a new validation bank, reuse of V1 winners, soft fibers, bandwidth
tuning, tolerance tuning, or retrospective changes to the draft protocol.

## 13. Implementation references

The diagnostic implementation is intentionally separate from `core.py`:

- [`continuity_commutator.py`](continuity_commutator.py) implements analytic
  rectangular `grad log Z` at line 28, the discrete source-column derivative
  at line 64, natural `S_CIC(q u)` and both commutators at line 90, the
  reflected scalar/flux kernels at lines 156--244, reflected divergence at
  line 275, and the reflected action diagnostic at line 286.
- [`audit_v2_continuity_commutator.py`](audit_v2_continuity_commutator.py)
  runs the column and reflection golden checks at lines 167 and 249, defines
  and evaluates the manufactured systems at lines 315 and 377, and writes the
  complete audit from line 469.
- [`core.py`](core.py), line 477, is the audited previous `continuity_check`.
  The current raster, solve, arbitrary-field smoothing, and forced-zero-face
  divergence start at lines 235, 273, 394, and 412 respectively.
- [`src/mfsi/raster.py`](../../src/mfsi/raster.py), functions
  `_full_support_gaussian_matrix_1d` (line 121),
  `_bilinear_cell_center_deposition_rect` (line 171), and
  `rasterize_projected_particles_positive_rect` (line 203), contains the
  current common CIC/source-column raster primitives.
- [`test_v2.py`](test_v2.py), lines 231 and 266, contains regression tests for
  the exact commutator sign/derivative and reflected manufactured continuity,
  including zero normal reflected face flux.

No production V2 raster function was replaced. The new operators are audit
code until the recommendation is prospectively accepted.

## 14. Reproduction

From the repository root:

```bash
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/src:$PWD/experiments/vortices_percentage_v2"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

.venv/bin/python experiments/vortices_percentage_v2/audit_v2_continuity_commutator.py

.venv/bin/python -m pytest \
  experiments/vortices_percentage_v2/test_v2.py \
  tests/test_positive_raster.py -q
```

The recorded focused test result is `13 passed`.

Generated development artifacts:

| Path | Contents |
|:---|:---|
| `outputs/vortices_v2_continuity_commutator/continuity_commutator_summary.json` | Full-precision authority and provenance |
| `outputs/vortices_v2_continuity_commutator/continuity_commutator_summary.csv` | Flattened numerical rows |
| `outputs/vortices_v2_continuity_commutator/golden_commutator_fields.npz` | Fine-grid exact fields at `epsilon=2e-4` |
| `outputs/vortices_v2_continuity_commutator/golden_commutator_fields.png` | Residual and commutator fields |
| `outputs/vortices_v2_continuity_commutator/continuity_grid_trends.png` | Golden grid comparison |
| `outputs/vortices_v2_continuity_commutator/candidate_action_convergence.png` | Candidate action convergence only |

Code and authority hashes for this run:

| File | SHA-256 |
|:---|:---|
| `continuity_commutator.py` | `35139ec81801dfe03693189886df35023cfc453c91d4d9c7f73411da94f65d78` |
| `audit_v2_continuity_commutator.py` | `2d59b6a3692cb946cec940137b1add58993ae4b39bb5bd1393baa4357d64f1a1` |
| `test_v2.py` | `811ad2101c5222e773f3900fc8c81a218216cfca9fe067daf78dccc36d5bc1a4` |
| `continuity_commutator_summary.json` | `2a8fe92e630da27a6ffc5a76b6676f48b78c3affcefc65eea2a936629f1f7c97` |
| `continuity_commutator_summary.csv` | `57535c1e00d00f6ae2bce14cc75147370032aa16db448f25853ca5e3d8d3606c` |
| Frozen development bank | `b25fe9be6a467c451671cad110f44a63e24b9f7787a9af2b34b16aed096bc5bf` |

## Final scientific statement

> The Vortices V2 action remains numerically convergent and independently
> verified. The former strong-continuity gate was mis-specified for its
> source-column-normalized boundary kernel: it omitted the exact negative
> normalization commutator and imposed zero boundary flux on a natural
> smoothed flux that is not zero there. Correcting those definitions removes
> the adverse grid trend, while independent reflected manufactured systems
> satisfy continuity to about `1e-10`. A column-normalized scientific action
> would require both the volume defect `s_h+C_h` and a matching
> nonhomogeneous boundary condition. The recommended prospective V2 method is
> instead the reflection/Neumann kernel, for which the ordinary no-flux
> identity and the existing `s_h` source are coherent. That recommendation is
> not adopted here, and no selection, validation, soft-fiber, bandwidth, or
> tolerance change has been made.
