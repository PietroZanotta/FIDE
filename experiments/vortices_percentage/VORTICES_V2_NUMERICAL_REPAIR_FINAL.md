# Vortices V2 reflection/Neumann numerical repair — final pre-freeze record

## Status and answer

The reflection/Neumann V2 numerical method is **ready to freeze before new
reference training, sensor selection, or validation**.

Every required pre-freeze gate passes:

- the shared reflected scalar and matched-flux kernels pass conservation,
  positivity, derivative, no-flux, image-saturation, and float64 tests;
- all seven fixed development cases pass action grid convergence;
- all three nested empirical-law studies pass particle convergence;
- manufactured reflected continuity passes at approximately `1e-10`;
- the real golden finite-time identity passes all prospectively amended
  thresholds;
- five independent SciPy Poisson comparisons agree to approximately
  `3.35e-14` relative action error or better;
- the bilinear and edge-energy actions agree to `1.96e-16` or better; and
- all seven common-raster Full/Tangent/Hidden decompositions pass without
  clipping.

This is a numerical-method decision, not a V2 scientific-performance result.
No new reference model, selection bank, validation bank, sensor winner, Law
anchor, risk cap, or Full-vs-Law reduction was produced. No soft moment fiber,
density floor, bandwidth tuning, action cap, trimming, or winsorization was
introduced.

The full-precision authority was recorded as `outputs/vortices_v2_reflection_prefreeze/convergence_summary.json`, with a flattened `convergence_summary.csv`. Those development outputs are not inputs to the published result or its visualizations and now reside in the ignored `old_stuff/vortices_percentage_v2_1_development/outputs/` archive.

## 1. Scientific history and separation from V1

The records must remain distinct:

1. [`VORTICES_V2_NUMERICAL_REPAIR.md`](VORTICES_V2_NUMERICAL_REPAIR.md)
   records the first fixed-bandwidth V2 development method, which still used
   CIC followed by a source-column-normalized Gaussian.
2. [`VORTICES_V2_CONTINUITY_COMMUTATOR_AUDIT.md`](VORTICES_V2_CONTINUITY_COMMUTATOR_AUDIT.md)
   proves why that kernel requires a volume commutator and nonhomogeneous
   correction boundary flux, and tests reflection/Neumann as candidate B.
3. This document records the prospective promotion of candidate B and the
   complete replay performed before any new V2 scientific bank was generated.

Specifically, the old normalized kernel was

```text
K(y|x) = g_h(y-x) / Z(x).
```

Differentiating its source-dependent `1/Z(x)` factor adds the volume term

```text
C_h(y) = -integral K(y|x) q(x)
                    [u(x) dot grad_x log Z(x)] dx,
```

so its exact regularized identity is
`partial_t q_h+div(j_h)=s_h+C_h`, not the former identity with `s_h` alone.
Its natural smoothed reference flux also generally has nonzero normal boundary
values. A coherent correction would consequently require both source
`s_h+C_h` and the nonhomogeneous condition
`q_h partial_n psi_h=j_h dot n`. Forcing the reference flux to zero at the
boundary while omitting `C_h` broke that coupled identity. Reflection was
selected prospectively because it removes the source-dependent normalization,
makes the matched normal flux exactly zero, and restores the ordinary
no-flux continuity equation—not because its development actions were smaller.

The V1 result under `experiments/vortices_percentage/` is immutable. Its
2,048-trial numerical PASS/statistical FAIL remains the authoritative V1
conclusion. V2 does not delete its rare trials, reinterpret its estimand, or
replace its historical artifacts.

The replay rechecked the frozen V1 scientific input hashes:

| V1 input | SHA-256 |
|:---|:---|
| `experiments/vortices_percentage/config.json` | `8f57f167675718b19d7ffc1741a8175adbe22069ff4043634b62df8dcf100ed0` |
| `experiments/vortices_percentage/experiment.py` | `5bcd5b3c96668cabf6d7a8b2b1944f48f490635763b997172584328551a9a4c4` |
| Frozen development bank, namespace `19892` | `b25fe9be6a467c451671cad110f44a63e24b9f7787a9af2b34b16aed096bc5bf` |

The seven old trials are development/mechanism data only. Their actions cannot
be used as V2 validation or to select a new geometry.

## 2. Final prospective numerical definition

The unsmoothed hard projected law uses the unchanged positive-defect convention

```text
s = partial_t q + div(q u).
```

The scientific scalar raster is the direct particle, cell-integrated
reflection/Neumann Gaussian:

```text
q_h = S_reflect(q),
s_h = S_reflect(s).
```

Density and signed source use exactly the same even-reflection scalar kernel.
There is no CIC step and no source-location-dependent normalization. The
matched reference flux is

```text
j_h = S_reflect(q u),
u_h = j_h / q_h.
```

For the `x` component, the flux kernel is odd in `x` and even in `y`; for the
`y` component it is even in `x` and odd in `y`. It is not `q_h` multiplied by
the original reference velocity sampled on raster cells. In general,
`S(q u) != S(q)u`.

The reflected construction satisfies

```text
partial_t q_h + div(j_h) = s_h,
j_h dot n = 0.
```

The V2 Full correction therefore remains

```text
delta_h = -grad psi_h,
K(q_h) = -div(q_h grad),
K(q_h) psi_h = -s_h,
```

with homogeneous Neumann correction boundary conditions. The instantaneous
and design actions are

```text
A_h(t) = integral q_h |grad psi_h|^2,
A_h    = normalized_trapezoidal_integral_0^1 A_h(t) dt.
```

This is the minimum-energy correction of the reflected regularized law
relative to its reflected regularized reference flux.

## 3. Shared reflected raster primitive

The production primitive now lives in `src/mfsi/raster.py`. On an interval
`[a,b]`, with `L=b-a`, the scalar and normal-flux kernels are

```text
K_N(y,x) = sum_k [g_h(y-x+2kL) + g_h(y+x-2a+2kL)],
K_D(y,x) = sum_k [g_h(y-x+2kL) - g_h(y+x-2a+2kL)].
```

`K_N` is integrated analytically over every target cell. `K_D` is evaluated on
faces and is set to its exact zero value on both domain boundaries. Their
finite-volume derivative relation is

```text
partial_x integral_cell K_N(y,x) dy
  = -[K_D(right_face,x)-K_D(left_face,x)].
```

The implementation uses four translated image pairs on either side of the
central image. At the frozen bandwidth, comparison with five image pairs is
bitwise identical for scalar and face kernels on both fine-grid axes. The
maximum tested scalar column-mass error is `1.554313e-15`; the smallest tested
fine-axis kernel entry remains strictly positive (`2.77e-251`).

The raster does not divide each source column by a truncated-domain
normalizer. It does not add a density floor. It removes only the residual
global floating-point mean of the already centered signed source.

## 4. Hard projection and reference-flux semantics

The empirical information projection and `lambda_dot` calculation are
unchanged. Every particle-convergence row recalibrates the projection,
covariance, multiplier derivative, weights, and forcing on that exact nested
prefix.

The action realizes `q_h` relative to `j_h=S_reflect(q u)`. No decomposition,
continuity check, or final documentation uses `q_h*u_original(grid)`.

The common-raster decomposition uses the same reflected `q_h`, the same
reflected `s_h`, the same fine grid, and the same weighted vector-field inner
product as Full. The shared decomposition helper historically expresses its
target as `L(delta)=-r(source)`. V2 solves `K psi=-s`, so the V2 wrapper supplies
`-s_h` to that helper; this makes both Full and Tangent satisfy the V2 target
`L(delta)=+r(s_h)`. No particle-space sign or forcing was changed.

## 5. Frozen bandwidth policy

The development replay keeps the previously declared reference-only value

```text
h_ref = 0.05883961987664522.
```

It is the median weighted two-dimensional Scott bandwidth over the 21 nodes of
the one frozen development reference. It has no grid floor and does not depend
on geometry, allowance, observation trial, risk, or action.

The `.75`, `1.00`, and `1.25` multipliers remain sensitivity diagnostics. They
were not used to choose the scientific bandwidth.

| Multiplier | Worst `128 -> 256` action change | Largest condition proxy | Smallest `q_h` |
|---:|---:|---:|---:|
| `.75` | .4393% | `3.5223e11` | `1.3237e-11` |
| `1.00` | .3397% | `1.5020e8` | `2.5253e-8` |
| `1.25` | .2460% | `2.4670e6` | `1.2431e-6` |

For the future three-reference experiment, the code supports computing one
Scott value from each independently qualified frozen reference and freezing
their median as one common physical bandwidth. That value has not been
computed because the three prospective references do not yet exist.

## 6. Fixed seven-case development set

| Case | Geometry key | Trial | Role |
|:---|:---|---:|:---|
| golden Full 4% | `ea6c90af64ce4356` | 130 | dominant known V1 tail |
| golden Law | `f8fdd998b4627969` | 130 | paired Law mechanism control |
| golden Full 2%/3% | `ce783572fe3170da` | 130 | paired Full mechanism control |
| known Law tail | `f8fdd998b4627969` | 65 | additional known V1 tail |
| known Full 1% tail | `41ca33ec45daa976` | 240 | additional known V1 tail |
| ordinary Law 1 | `f8fdd998b4627969` | 114 | action-blind seeded sample |
| ordinary Law 2 | `f8fdd998b4627969` | 83 | action-blind seeded sample |

Trials 114 and 83 were fixed with seed `20260830` before their V2 actions were
read. No development case is evidence of expected Full-vs-Law improvement.

## 7. Reflected action grid convergence

Every value uses 32,768 projected particles, all 21 scientific nodes, fixed
`h_ref`, and a newly assembled reflected raster and weighted-Poisson operator.

| Case | `64 x 32` | `128 x 64` | `256 x 128` | `64 -> 128` | `128 -> 256` |
|:---|---:|---:|---:|---:|---:|
| golden Full 4% | 2.566432 | 2.586170 | 2.591376 | .7632% | .2009% |
| golden Law | 1.297813 | 1.312571 | 1.316394 | 1.1243% | .2904% |
| golden Full 2%/3% | 1.270311 | 1.282988 | 1.286288 | .9881% | .2565% |
| known Law tail | 2.954159 | 2.981048 | 2.988056 | .9020% | .2345% |
| known Full 1% tail | 1.581329 | 1.597208 | 1.601339 | .9942% | .2580% |
| ordinary Law 1 | 1.366457 | 1.381169 | 1.384982 | 1.0652% | .2753% |
| ordinary Law 2 | 1.494660 | 1.514595 | 1.519758 | 1.3162% | .3397% |

The worst final change is `.3397%`, versus the `5%` gate. Every final change
is smaller than its corresponding coarse-to-medium change.

### Fine-grid numerical diagnostics

Each mass/source/residual/concentration entry is the worst over the 21 nodes.

| Case | `A(t=.5)` | Max `A(t)` | Mass error | `|int s_h|` | Min `q_h` | Components | Poisson residual | Condition proxy | Max top-1% energy share |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| golden Full 4% | 28.495213 | 28.495213 | `8.22e-15` | `8.88e-16` | `2.14e-7` | 1 | `5.31e-12` | `1.237e7` | 21.85% |
| golden Law | 1.134881 | 5.993249 | `8.22e-15` | `1.67e-16` | `2.53e-8` | 1 | `1.42e-11` | `1.502e8` | 21.97% |
| golden Full 2%/3% | 3.166571 | 3.207758 | `8.33e-15` | `2.22e-16` | `4.04e-8` | 1 | `4.31e-12` | `7.734e7` | 21.57% |
| known Law tail | 21.485880 | 21.485880 | `8.33e-15` | `8.88e-16` | `6.85e-8` | 1 | `1.79e-11` | `4.822e7` | 19.18% |
| known Full 1% tail | 6.400213 | 6.400213 | `8.44e-15` | `4.44e-16` | `1.02e-7` | 1 | `8.56e-12` | `3.577e7` | 21.50% |
| ordinary Law 1 | .980930 | 6.161953 | `8.22e-15` | `1.11e-16` | `8.68e-8` | 1 | `1.13e-11` | `4.032e7` | 17.30% |
| ordinary Law 2 | 1.011061 | 12.459119 | `7.99e-15` | `1.11e-16` | `1.14e-4` | 1 | `7.23e-12` | `2.226e4` | 17.71% |

Every density is strictly positive, every solve has one conductive component,
and every physical solve converged. The maximum bilinear/edge-energy identity
error anywhere in these default rows is `4.94e-13`.

## 8. Complete fine-grid instantaneous actions

These are all 21 reflected actions at the default bandwidth on `256 x 128`.
The compact JSON retains additional full-precision digits.

| `t` | Golden Full 4% | Golden Law | Golden Full 2%/3% | Known Law tail | Known Full 1% tail | Ordinary Law 1 | Ordinary Law 2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| .00 | 1.832871 | 5.993249 | 2.336162 | 8.919641 | 4.344547 | 6.161953 | 12.459119 |
| .05 | 2.060460 | 3.648976 | 2.638928 | 4.596357 | 3.645764 | 4.249245 | 4.727645 |
| .10 | 3.156122 | 2.872447 | 2.908325 | 3.490071 | 3.435738 | 2.732648 | 3.135908 |
| .15 | 2.330775 | 1.749266 | 1.983860 | 1.840116 | 2.115547 | 1.608167 | 1.730740 |
| .20 | 1.181854 | .867118 | 1.001063 | .835954 | 1.030761 | .778672 | .847081 |
| .25 | .296469 | .231485 | .257391 | .209447 | .266551 | .188110 | .220509 |
| .30 | .165291 | .202714 | .147309 | .202745 | .239587 | .154020 | .191035 |
| .35 | .453360 | .490857 | .398891 | .497356 | .576703 | .415663 | .467863 |
| .40 | .821982 | .831392 | .762967 | .841474 | .901275 | .705437 | .758572 |
| .45 | 1.257470 | 1.310415 | 1.207361 | 1.326141 | 1.290772 | 1.021072 | 1.070523 |
| .50 | 28.495213 | 1.134881 | 3.166571 | 21.485880 | 6.400213 | .980930 | 1.011061 |
| .55 | .895665 | 1.039562 | 1.026436 | 1.127574 | 1.035721 | .948190 | .913960 |
| .60 | 2.822325 | 1.549105 | 1.737790 | 1.443873 | 1.424325 | .964610 | .924088 |
| .65 | 1.295074 | 1.294962 | 1.288132 | 2.158574 | 1.285556 | 1.237478 | .974035 |
| .70 | .785110 | .775965 | 1.084968 | 2.261595 | .966961 | 1.406012 | 1.050196 |
| .75 | .601705 | .396757 | .690989 | .874747 | .481542 | .417234 | .346857 |
| .80 | .508683 | .386206 | .540127 | .774270 | .449030 | .620176 | .349885 |
| .85 | .309118 | .243456 | .369179 | 8.151219 | 1.017658 | 3.040621 | 1.191025 |
| .90 | .682731 | .760812 | .702428 | .697711 | .612599 | .694617 | .602594 |
| .95 | 1.080208 | 1.264008 | 1.041083 | .958557 | 1.019601 | .970847 | 1.196163 |
| 1.00 | 3.422927 | 4.561735 | 3.207758 | 3.055273 | 3.317187 | 2.969831 | 4.911721 |

The large values at selected times are development observations, not a tail
distribution estimate. None exceeds the mechanism-only threshold `1000`.

## 9. Nested particle convergence

Every row uses the fine grid and fixed `h_ref`. Each `N` has a newly calibrated
hard empirical projection.

| Case | `N=8,192` | `N=16,384` | `N=32,768` | `16,384 -> 32,768` |
|:---|---:|---:|---:|---:|
| golden Full 4% | 2.444615 | 2.517219 | 2.591376 | 2.8617% |
| golden Law | 1.361695 | 1.330940 | 1.316394 | 1.1050% |
| golden Full 2%/3% | 1.311673 | 1.287448 | 1.286288 | .0902% |

The worst final step is `2.8617%`, below the `10%` gate. A 65,536-particle
study was not required because no controlled frozen extension of the present
nested bank was created.

## 10. Reflected continuity verification

### Manufactured spatial identity

The systems use known trajectories and exactly centered exponential weight
evolution. The finite-difference step is `2e-5`.

| Case | Grid | Relative L2 | Maximum weak relative | Mass error | Maximum normal boundary flux |
|:---|:---|---:|---:|---:|---:|
| constant interior | `64 x 32` | `1.27e-10` | `1.04e-10` | `0` | `0` |
| constant interior | `128 x 64` | `1.30e-10` | `1.08e-10` | `0` | `0` |
| constant interior | `256 x 128` | `1.34e-10` | `1.07e-10` | `0` | `0` |
| near all boundaries | `64 x 32` | `2.40e-10` | `5.42e-10` | `0` | `0` |
| near all boundaries | `128 x 64` | `2.50e-10` | `5.36e-10` | `0` | `0` |
| near all boundaries | `256 x 128` | `2.73e-10` | `5.35e-10` | `1.11e-16` | `0` |
| tangent to all boundaries | `64 x 32` | `1.04e-10` | `1.54e-11` | `1.11e-16` | `0` |
| tangent to all boundaries | `128 x 64` | `1.08e-10` | `1.92e-11` | `1.11e-16` | `0` |
| tangent to all boundaries | `256 x 128` | `1.17e-10` | `1.67e-11` | `0` | `0` |

The declared relative L2 and weak gates are `1e-8`; mass is limited at
`5e-13`; normal boundary flux is limited at `1e-14`. All pass.

### Real golden finite-time identity

The golden weights were independently recalibrated at `t-epsilon`, `t`, and
`t+epsilon`. The table tests

```text
dq_h/dt + div_reflected(j_h) = s_h.
```

| Grid | Epsilon | Relative L2 | Maximum weak relative | Correlation |
|:---|---:|---:|---:|---:|
| `64 x 32` | `1e-3` | 5.90706% | 3.03083% | .99996404 |
| `64 x 32` | `5e-4` | 1.52494% | .82924% | .99999771 |
| `64 x 32` | `2e-4` | .24720% | .13472% | .99999994 |
| `128 x 64` | `1e-3` | 5.91001% | 3.02800% | .99996393 |
| `128 x 64` | `5e-4` | 1.52572% | .82847% | .99999770 |
| `128 x 64` | `2e-4` | .24733% | .13460% | .99999994 |
| `256 x 128` | `1e-3` | 5.91075% | 3.02632% | .99996390 |
| `256 x 128` | `5e-4` | 1.52592% | .82801% | .99999769 |
| `256 x 128` | `2e-4` | .24737% | .13452% | .99999994 |

At `epsilon=2e-4`, both error metrics are below `.5%`, correlation exceeds
`.999`, and the relative-L2 range across grids is `1.6384e-6` as a fraction,
or `.00016384` percentage points, versus the `.05`-point limit. Error strictly
decreases as epsilon decreases on every grid. The particle weak moment identity
has maximum absolute error `1.3034e-13`. All normal boundary fluxes are exactly
zero in float64.

These thresholds were selected after the mechanism audit and before any new V2
selection or validation outcome. Grid monotonicity is deliberately not a gate
because the smallest-epsilon residual is limited by independent finite-time
recalibration error.

## 11. Independent reflected Poisson verification

| Case | Grid | Production action | Independent action | Relative action error | Production residual | Independent residual | Bilinear/edge error |
|:---|:---|---:|---:|---:|---:|---:|---:|
| golden Full 4% | `256 x 128` | 28.495213301 | 28.495213301 | `2.79e-14` | `5.31e-12` | `5.04e-13` | `0` |
| golden Law | `128 x 64` | 1.133717026 | 1.133717026 | `1.68e-14` | `3.62e-12` | `2.87e-13` | `0` |
| golden Law | `256 x 128` | 1.134880766 | 1.134880766 | `3.35e-14` | `1.42e-11` | `1.13e-12` | `1.96e-16` |
| golden Full 2%/3% | `128 x 64` | 3.162374263 | 3.162374263 | `2.11e-15` | `1.38e-12` | `9.81e-14` | `0` |
| golden Full 2%/3% | `256 x 128` | 3.166570542 | 3.166570542 | `2.15e-14` | `4.31e-12` | `3.89e-13` | `0` |

The action-discrepancy gate is `2e-6` and the physical-residual gate is
`2e-7`. Flipping `s_h` flips the potential exactly in these recorded solves and
changes every scalar action by zero.

## 12. Reflected common-raster decomposition

These are integrated actions on the reflected fine-grid Hilbert space. The
Tangent action is the minimum-norm raster moment-feasible field; it is not a
reused particle-Tangent optimization value.

| Case | `A_full` | `A_tan` | `A_hid` | Full moment residual | Tangent residual | Hidden residual | `|<tan,hid>|` | `|Pythagorean|` | Max raw `A_tan-A_full` |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| golden Full 4% | 2.591376 | .463490 | 2.127886 | `1.42e-13` | `3.23e-15` | `1.41e-13` | `4.13e-14` | `8.17e-14` | -.100727 |
| golden Law | 1.316394 | .296857 | 1.019537 | `4.44e-14` | `5.59e-16` | `4.44e-14` | `6.62e-15` | `1.33e-14` | -.131951 |
| golden Full 2%/3% | 1.286288 | .314127 | .972161 | `8.88e-14` | `3.13e-15` | `8.86e-14` | `1.72e-14` | `3.42e-14` | -.083550 |
| known Law tail | 2.988056 | .487460 | 2.500596 | `5.11e-14` | `1.78e-15` | `5.09e-14` | `1.99e-14` | `4.26e-14` | -.129595 |
| known Full 1% tail | 1.601339 | .344479 | 1.256859 | `5.80e-14` | `3.78e-15` | `5.82e-14` | `1.36e-14` | `2.69e-14` | -.174402 |
| ordinary Law 1 | 1.384982 | .299329 | 1.085653 | `3.47e-14` | `5.07e-15` | `3.46e-14` | `7.52e-15` | `1.51e-14` | -.099509 |
| ordinary Law 2 | 1.519758 | .299992 | 1.219766 | `1.00e-13` | `8.53e-16` | `1.00e-13` | `3.08e-14` | `6.22e-14` | -.131159 |

The absolute tolerance is `1e-6`. All Gram matrices have rank four. The
maximum Full-energy match error against the production Poisson action is
`4.94e-13`. Every raw hierarchy value is negative, so hierarchy passes with
real slack and no clipping.

## 13. Regression tests

The focused suite reports `16 passed` and covers:

- immutable V1 config and experiment hashes;
- reference-only bandwidth and future median-of-three support;
- reflected scalar-column conservation and strict positivity;
- identical scalar smoothing of mass and signed source;
- exact zero normal reflected face flux;
- reflected scalar/odd-flux derivative identity;
- four-image-pair saturation against five pairs;
- manufactured reflected continuity;
- `K psi=-s`, potential sign flip, scalar-action sign invariance, and energy
  identities;
- physical bandwidth independence from grid resolution;
- rectangular axis orientation;
- independent Poisson agreement;
- float64;
- config fingerprint sensitivity to boundary rule and image-pair count; and
- a fail-fast regression proving V2 production does not call the old
  source-column-normalized raster.

## 14. Amended future protocol

[`VORTICES_V2_VALIDATION_PROTOCOL_DRAFT.md`](VORTICES_V2_VALIDATION_PROTOCOL_DRAFT.md)
is now version 2, dated 2026-08-30. It was amended before any new V2 reference,
selection, or validation bank.

It freezes the hard projection, direct even-reflection scalar raster, matched
odd-normal flux, `delta=-grad psi`, `-div(q grad psi)=-s`, zero density floor,
`256 x 128` grid, 21 time nodes, float64, and the common median-of-three
reference-only bandwidth policy. It replaces the invalid column-kernel
continuity gate with the manufactured and real finite-time reflected gates in
Section 10.

The prohibitions on V1 validation reuse, soft fibers, a `lambda_dot`-only
ridge, trimming, winsorization, post-hoc action caps, and outcome-dependent
stopping remain in force. V1 geometries remain proposal seeds only;
Population, Law, Tangent, and Full selection must be rerun from scratch.

## 15. Implementation map

- `src/mfsi/raster.py` contains the shared cell-integrated reflected scalar
  kernel, odd face kernel, direct density/source raster, matched particle flux,
  and finite-volume divergence.
- `core.py` makes reflection the V2 scientific raster/action, evaluates
  trajectories sequentially through cached JIT kernels, provides the
  reference-only common-bandwidth helper, and implements the real reflected
  continuity check.
- `diagnose_vortices_v2_convergence.py` runs the seven-case grids, bandwidth
  sensitivity, nested particle recalibrations, manufactured and golden
  continuity, independent Poisson checks, and reflected common-raster
  decompositions.
- `continuity_commutator.py` now delegates reflection to the shared primitive.
  Its column-normalized functions remain isolated historical audit replay code.
- `test_v2.py` contains the V2-specific regression suite.

No V1 output path is a V2 write destination.

## 16. Artifacts and figures

| Path | Contents |
|:---|:---|
| `outputs/vortices_v2_reflection_prefreeze/convergence_summary.json` | Full-precision result, gates, provenance, all time actions, continuity and decomposition rows |
| `outputs/vortices_v2_reflection_prefreeze/convergence_summary.csv` | Flattened grid, particle, continuity, solver, manufactured, and decomposition rows |
| `outputs/vortices_v2_reflection_prefreeze/golden_v2_fields.npz` | Fine-grid reflected golden density, source, potential, and energy density |
| `outputs/vortices_v2_reflection_prefreeze/action_vs_grid_resolution.png` | Seven-case default-bandwidth grid convergence |
| `outputs/vortices_v2_reflection_prefreeze/action_vs_particle_count.png` | Three nested recalibrated particle studies |
| `outputs/vortices_v2_reflection_prefreeze/bandwidth_sensitivity.png` | Predeclared reference-only bandwidth sensitivity |
| `outputs/vortices_v2_reflection_prefreeze/golden_q_source_energy.png` | Golden reflected density/source/potential/energy fields |
| `outputs/vortices_v2_reflection_prefreeze/golden_q.png` | Golden reflected log density |
| `outputs/vortices_v2_reflection_prefreeze/golden_source.png` | Golden reflected positive defect |
| `outputs/vortices_v2_reflection_prefreeze/golden_energy_density.png` | Golden reflected correction-energy concentration |

The `outputs/` tree remains ignored because of artifact size. The compact
authority is deterministic from the documented inputs and command.

## 17. Reproduction

From the repository root:

```bash
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/src:$PWD/experiments/vortices_percentage_v2"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

.venv/bin/python -m pytest \
  experiments/vortices_percentage_v2/test_v2.py \
  tests/test_positive_raster.py -q

.venv/bin/python \
  experiments/vortices_percentage_v2/diagnose_vortices_v2_convergence.py
```

The harness writes only to
`experiments/vortices_percentage_v2/outputs/vortices_v2_reflection_prefreeze/`
by default. `--bandwidth` remains diagnostic-only and changes the recorded
object. `--include-512` is optional and was not used. `--case-limit` is a smoke
option and was not used for the authoritative replay.

Recorded provenance:

| Item | Value |
|:---|:---|
| Completion date | 2026-08-30 |
| Repository HEAD | `1e65587684857a84e98fda181fe472d62f46f7a8` |
| Working tree | Modified; exact hashes below are authoritative |
| Projection | hard empirical information projection |
| Development namespace | `19892` |
| Reference particles | `32,768`; nested recalibrations at `8,192/16,384/32,768` |
| Raster | direct reflected particle scalar; matched reflected flux |
| Image pairs | `4` |
| Grid | `256 x 128` final development grid |
| Bandwidth | `0.05883961987664522` |
| Density floor | `0` |
| Precision | float64 |

Code, config, protocol, and compact-authority hashes:

| File | SHA-256 |
|:---|:---|
| `config.json` | `0962e8834f100a293985f24ee60e94b9bd1318673472e98697bb70ce4b5fb1fa` |
| `core.py` | `0c8343aae751cd92fa5411e5e2ba9610dba6b8693165aca8f5c25ca3be5a3fd8` |
| `diagnose_vortices_v2_convergence.py` | `d241c89ab80c9ec5075b5d6f9fde299f9656523738ec0ae398e4315d586a392b` |
| `test_v2.py` | `553a36066e0a86ec74c6c725fdb8742faf4a6313c55860a4e31f2696d8b3ebd6` |
| `src/mfsi/raster.py` | `4208ebe18d9f7f788a117c1bff74dce585a601f2c5bc6e89b1ff1389d6267c28` |
| `VORTICES_V2_VALIDATION_PROTOCOL_DRAFT.md` | `dcf4b268317c923e611e69cbeaee86560dbf784453934f568a2b17c943ce5458` |
| `convergence_summary.json` | `911ba023d1ec44bb049bc020c750373d2927ee64737c3ca4aeb4c5506ebca572` |
| `convergence_summary.csv` | `8d5a4807e9917ccada8efacd23783cb499edb2895db7623b9e308b8b0a0299d3` |

## 18. Remaining limitations

- The replay uses one historical reference and seven historical development
  cases. It does not establish a V2 action-tail distribution.
- The three prospective reference models have not been trained or qualified;
  therefore their common median Scott bandwidth is not yet known.
- Population, Law, Tangent, and Full selection has not begun. No V1 winner is
  a V2 winner.
- No V2 selection or validation bank exists.
- The direct reflected particle raster is more computationally expensive than
  the former CIC raster; performance changes must preserve the exact reflected
  scalar/flux object and pass the same tests.
- A controlled 65,536-particle extension and optional `512 x 256` study were
  not required by the declared gates and were not run.
- The hard fiber still has weak-covariance amplification in the underlying
  projection. The present numerical evidence does not justify a soft fiber.
- Bandwidth sensitivity materially changes conditioning and action even though
  every fixed bandwidth converges. The prospective common reference-only rule
  must be followed without performance tuning.

## 19. Exact final pre-freeze decision table

A. Legacy V1 unchanged?
   **PASS**

B. Reflection/Neumann scientific raster implemented?
   **PASS**

C. Mathematical no-flux continuity identity verified by manufactured tests?
   **PASS**

D. Seven-case reflected action grid convergence?
   **PASS**

E. Reflected particle-count convergence?
   **PASS**

F. Independent reflected Poisson solve agrees?
   **PASS**

G. Reflected common-raster Full/Tangent/Hidden decomposition passes?
   **PASS**

H. Real golden finite-time continuity satisfies the prospectively amended gate?
   **PASS**

I. Soft fiber required?
   **NO**

J. V2 numerical method ready to freeze before reference training and new
   selection?
   **YES**

## Final scientific statement

> Candidate B has been promoted prospectively into the Vortices V2 production
> development path because its reflected scalar and matched flux kernels
> provide the coherent no-flux identity `partial_t q_h+div(j_h)=s_h`, not
> because they produce a favorable action. The fixed seven-case, nested
> particle, manufactured continuity, real finite-time continuity, independent
> Poisson, and reflected Full/Tangent/Hidden suites all pass their unchanged or
> prospectively amended gates. The numerical definition is ready to freeze.
> The next scientific experiment may begin only after this freeze; no new
> reference, selection, or validation result is contained in this record.
