# Production Galerkin evaluation

This is the authoritative record for the isolated fixed-feature Galerkin Full
solver. All scientific calculations used the complete frozen production banks;
no bank was regenerated, no smoke artifact was substituted, and no Pareto sweep
was run.

## 1. Initial repository status

Before this milestone, `git status --short` was:

```text
?? ETA_OPTIMIZATION_AUDIT.md
?? experiments/skyrmions_deep_ritz_full/
```

The whole isolated experiment and the unrelated audit file were already
untracked. The unrelated file was preserved. All task-created changes are below
`experiments/skyrmions_deep_ritz_full/`.

## 2. Artifact discovery result

A complete frozen production set was found at the read-only source:

```text
/home/zanot/projects/tesseract2026_original/experiments/skyrmions_deep_ritz/outputs/pareto_authoritative/risk_3pct
```

It contains the reference checkpoint, endpoint ensemble, design and validation
truth banks, all five reference banks, incumbent Ritz checkpoints, risk metadata,
time nodes, quadrature weights, and deterministic metadata needed to reconstruct
detector noise and whitening. It was hash-verified and copied without modifying
the source to `outputs/production_galerkin/artifacts/`.

Bank sizes are the intended production sizes: endpoints 12,000; design truth
6,000; validation truth 5,000; projection 8,192; Ritz train 8,192; Ritz audit
4,096; validation fit and audit 16,384 each. Every bank has 13 time nodes and 16
particles. The preflight result is `outputs/production_galerkin/preflight/result.json`.

## 3. Artifact hashes and metadata

| file | bytes | SHA-256 |
|---|---:|---|
| `reference.npz` | 138,075 | `f0aa333a38cbd7f99748c83e4a13335e40b81e85385f333dd81b597dfcfad3a9` |
| `truth_banks.npz` | 40,289,022 | `bda32e02692059e966185c0ac8b5a6afd9c32da4d5fd9e69e0caf981b4601bcd` |
| `reference_bank_projection.npz` | 51,306,532 | `d47619a86e116aa32f370c6f7d3ea81e0c20cc2d8d29c28bb1ca9f88b414efab` |
| `reference_bank_ritz_train.npz` | 51,299,368 | `93d8ee2952b71c6aea407340af80d9ff6d47674e2cc7001b849a45cbdeab2574` |
| `reference_bank_ritz_audit.npz` | 25,659,152 | `8eea98164efe2a720ab33c41cfb0036acfdc9bd5512a32cb05db37c80ede8f08` |
| `reference_bank_validation_fit.npz` | 102,648,330 | `b4e49eb65cd85f87923796e8052ce5f78041ddf96c2c1d2c1e9f025c71822c3e` |
| `reference_bank_validation_audit.npz` | 102,449,649 | `670e34e2952e2307aae20418a37e274911572afdc0bfd993b1930cdf3a398694` |
| `ritz_full.npz` | 44,278 | `4cc31315a16b1e5467e37c0e6dad73fc692e256d5a2e207f4c8890c8bcfa260f` |

The isolated manifest records all 15 files, complete hashes, byte counts, source
mtimes, array shapes, production config hash, and source/destination paths.
Existing identical copies are reused; a mismatching destination is never
overwritten.

## 4. Production reproduction availability

Complete reproduction was possible. The backend was the production
`tesseract_cpp` information-projection path from `tesseract_jax 0.2.3`, with JAX
float64 on CPU. The environment logs an unavailable-CUDA plugin warning before
correctly falling back to CPU; this did not alter or invalidate results.

The computation used 13 scientific time nodes, acquisition indices
`[0,2,4,6,8,10,12]`, the first 512 design-truth configurations, exact endpoint
means, the frozen detector-noise convention, and unchanged production spline,
projection, covariance, forcing, and quadrature settings.

## 5. Eta0 reproduction table

| quantity | reproduced | reference | absolute discrepancy |
|---|---:|---:|---:|
| law risk | 5.186549474478024 | 5.186549474478041 | 1.69e-14 |
| 3% risk ceiling | 5.342145958712365 | 5.342145958712383 | 1.78e-14 |
| selected risk | 5.340106050965989 | 5.340106050966004 | 1.51e-14 |
| Deep Ritz held-out action | 0.20345379368395114 | 0.20345379368395117 | 2.78e-17 |
| train projection residual | 8.02795139e-11 | 8.02794256e-11 | 8.83e-17 |
| audit minimum ESS | 0.0691626590244804 | 0.0691626590244806 | 1.67e-16 |
| train forcing mean | 9.75847770e-9 | 9.75849490e-9 | 1.72e-14 |
| weak residual | 0.0877618299430823 | 0.0877618299430822 | 1.53e-16 |
| energy residual | 0.0613163650400986 | 0.0613163650400969 | 1.67e-15 |
| moment-rate residual | 0.0244288645190935 | 0.0244288645190935 | 4.51e-17 |

Reconstruction repeated bit-for-bit (`max_abs=0`). Across the 13 nodes, `c_eta`
entries range from 0.0259233 to 0.0637047 with row norms 0.0925320--0.103674;
`cdot_eta` entries range from -0.0299725 to 0.0644007 with row norms
0.0170919--0.0737905. Full trajectories are in the reproduction JSON.

## 6. Discrepancy from production diagnostics

All scalar reference discrepancies are at float64 rounding level except the
already tiny projection/forcing diagnostics, whose absolute deviations remain
many orders below their hard thresholds. The reproduced held-out gauge residual
is `8.28e-16` versus the published approximately `1.38e-16`; both are effectively
zero relative to the unchanged `1e-9` gate. There is no scientifically material
reproduction discrepancy.

## 7. Fixed projected-law and forcing audit

| bank | max projection | min ESS | max covariance condition | pre-center mean | post-center mean | valid |
|---|---:|---:|---:|---:|---:|:---:|
| projection | 3.36023e-11 | 0.0715943 | 4.00028 | 6.23646e-9 | 9.03e-16 | yes |
| Ritz train | 8.02795e-11 | 0.0710170 | 3.90846 | 9.75848e-9 | 3.96e-16 | yes |
| Ritz audit | 1.46746e-11 | 0.0691627 | 3.77097 | 2.49347e-9 | 7.75e-16 | yes |

Gate A limits were `2e-6`, `0.05`, `1e10`, and `2e-7`, respectively. On the
three banks, lambda row norms span 1.272--87.583 and lambda-dot row norms span
166.818--1099.372. Full `lambda`, `lambda_dot`, `c_eta`, `cdot_eta`, and per-time
diagnostics are retained in `reproduction/result.json`. Gate A passed.

## 8. Exact feature dictionary definition

The fixed analytic dictionary has 160 coordinates and is independent of eta:

- 64 deterministic low-frequency reciprocal vectors for the `[0,2)x[0,1)` box,
  each yielding the particle mean of cosine and sine (128 coordinates);
- 32 permutation-invariant Gaussian functions of smooth periodic chord distance,
  averaged over all unordered particle pairs;
- hybrid order: two Fourier vectors (cosine and sine for each) followed by one
  radial coordinate, repeated 32 times.

The Fourier frequencies are `(2*pi*nx/2, 2*pi*ny/1)` ordered by squared norm and
deterministic integer tie-breakers. Radial centers are uniformly spaced from
0.03 through 0.7117625434, with fixed width 0.03848659519. State derivatives are
analytic JAX expressions over all 32 state coordinates. Unit checks confirm
periodicity, particle-permutation invariance, finite derivatives, and agreement
with `jax.jacrev`.

## 9. Basis dimensions

The actual nested ladder was `K = 20, 40, 60, 80, 100, 120, 140, 160`. No
unsupported dimension was reported. The maximum contains 128 Fourier and 32
pairwise radial coordinates.

## 10. Normalization method

For every time and coordinate, a weighted base-reference mean and a diagonal
Dirichlet-energy scale were computed once on the eta-independent frozen Ritz
train bank and then frozen. Values use `(phi-base_mean)/scale`; state gradients
use `grad(phi)/scale`. There is no eta-specific whitening or refit. The fitted
arrays and dictionary parameters are stored in
`convergence/features/hybrid_dictionary.npz`, with readable metadata beside it.

## 11. Nestedness proof and checks

The dictionary array is constructed once. Every space uses the first `K`
coordinates of that same array and the same frozen per-coordinate transform, so
`V_20 subset V_40 subset ... subset V_160` exactly. There is no K-dependent QR,
SVD, retraining, or reordering. Prefix equality and monotone controlled-problem
tests pass.

## 12. Galerkin convergence table

`A_train` is the quadrature action used for the solve; `A_audit` evaluates the
same coefficients on the independent production Ritz audit bank.

| K | A_train | A_audit | min rank fraction | weak | energy | moment | certificate |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 20 | 0.1220628551 | 0.1222807392 | 1.0000 | 0.125758 | 0.023647 | 0.064304 | no |
| 40 | 0.1657621361 | 0.1653946999 | 1.0000 | 0.097926 | 0.045002 | 0.036325 | yes |
| 60 | 0.1972374097 | 0.1977583936 | 1.0000 | 0.080083 | 0.045835 | 0.020409 | yes |
| 80 | 0.2217061019 | 0.2228561849 | 1.0000 | 0.075367 | 0.043607 | 0.011146 | yes |
| 100 | 0.2342254118 | 0.2351754543 | 1.0000 | 0.073691 | 0.044762 | 0.008445 | yes |
| 120 | 0.2500275561 | 0.2504521902 | 0.9917 | 0.072117 | 0.042495 | 0.007275 | yes |
| 140 | 0.2567410994 | 0.2572606146 | 0.9929 | 0.072495 | 0.049743 | 0.007568 | yes |
| 160 | 0.2645771969 | 0.2650532808 | 0.9938 | 0.074013 | 0.050369 | 0.007939 | yes |

## 13. Monotonicity table

| transition | absolute increase | relative increase | pass |
|---|---:|---:|:---:|
| 20 -> 40 | 0.0436992810 | 0.263626 | yes |
| 40 -> 60 | 0.0314752736 | 0.159581 | yes |
| 60 -> 80 | 0.0244686922 | 0.110365 | yes |
| 80 -> 100 | 0.0125193099 | 0.053450 | yes |
| 100 -> 120 | 0.0158021443 | 0.063202 | yes |
| 120 -> 140 | 0.0067135433 | 0.026149 | yes |
| 140 -> 160 | 0.0078360975 | 0.029617 | yes |

The sequence is nondecreasing. Its final two increments are both at or below the
required 3% stabilization threshold. The preferred 1% final increment was not
reached and is recorded as a limitation rather than silently promoted.

## 14. Eigenvalue, rank, and conditioning summary

Full eigenspectra, retained masks, and coefficients are stored for every K. The
rank fraction is 1 through K=100, then the worst per-time ranks are 119/120,
139/140, and 159/160. The K=160 worst retained condition number is
`3.72055e11`, below the declared `1e12` cap. A diagnostic run at rank tolerance
`1e-10` is preserved separately; the accepted solve uses `1e-12`, not a ridge,
so supported near-null directions are retained while the condition remains
reportable.

## 15. A = -2J residual

The aggregate identity relative errors from K=20 through K=160 are
`2.27e-16, 1.34e-15, 3.98e-14, 1.30e-13, 1.88e-12, 6.79e-12,
1.22e-11, 1.99e-11`. The largest is far below the `1e-8` gate.

## 16. Coefficient stationarity and range compatibility

At K=160 the worst relative coefficient-stationarity residual is `2.51152e-9`
and the worst range residual is `2.51145e-9`, both below the unchanged `1e-8`
acceptance gate. Raw Gram symmetry is at floating-point accumulation precision.
The nonzero high-K residual reflects the explicitly discarded unsupported
near-null direction, not a ridge-regularized scientific solve.

## 17. Held-out weak residual

The independent audit-bank weak residual is `0.0740127` at K=160, below `0.12`.
K=20 alone fails (`0.125758`); every K from 40 onward passes.

## 18. Held-out energy residual

The K=160 audit-bank Ritz-energy residual is `0.0503688`, below `0.08`.

## 19. Gauge residual

The K=160 audit-bank gauge residual is `7.37e-18`, below `1e-9`.

## 20. Moment-rate residual

The K=160 audit-bank moment-rate residual is `0.00793896`, below `0.10`.

## 21. Comparison with matching production Deep Ritz

| method | action | weak | energy | gauge | moment | max projection | min ESS | max forcing mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| production Deep Ritz | 0.203453794 | 0.087762 | 0.061316 | 8.28e-16 | 0.024429 | 1.47e-11 | 0.069163 | 2.49e-9 |
| Galerkin K=40 | 0.165394700 | 0.097926 | 0.045002 | 1.21e-17 | 0.036325 | 1.47e-11 | 0.069163 | 2.49e-9 |
| Galerkin K=80 | 0.222856185 | 0.075367 | 0.043607 | 1.07e-17 | 0.011146 | 1.47e-11 | 0.069163 | 2.49e-9 |
| Galerkin K=120 | 0.250452190 | 0.072117 | 0.042495 | 7.70e-18 | 0.007275 | 1.47e-11 | 0.069163 | 2.49e-9 |
| Galerkin K=160 | 0.265053281 | 0.074013 | 0.050369 | 7.37e-18 | 0.007939 | 1.47e-11 | 0.069163 | 2.49e-9 |

Actions in this comparison are held-out audit-bank actions. The certified
Galerkin sequence converges toward a value materially above the matching Deep
Ritz action; agreement was not forced. This is a scientific finding warranting
future approximation-space and nonlinear-stationarity investigation, not a
reason to rewrite either result. Gate B nevertheless passes its specified
convergence and physical criteria.

## 22. Eta gradient

Using the accepted K=160 coefficients fixed inside the differentiated closure,
the production Galerkin gradient at eta0 is:

```text
[ 0.5287493464801821, -0.4048849971572496,
 -2.2851994287612090, -0.7677045029553380,
  0.4682640756341179, -1.0563594994470600,
 -0.3391727117030091, -0.8530128538714885]
```

Only explicit eta dependence is differentiated. The eigendecomposition,
pseudoinverse, and coefficient solve are outside the JAX closure. Repeated
evaluations pass the `1e-12` determinism check and the gradient is finite with
shape `(8,)`.

## 23. Five-direction AD/FD validation

Every plus/minus point passed geometry, production forcing, Galerkin algebra,
and constant-rank checks. Each table gives `(epsilon, centered FD, relative
discrepancy)`.

Direction 0, `v=[-0.167761,-0.171420,0.261077,-0.369160,-0.473449,0.596764,-0.291231,-0.270064]`, AD `-0.8554557162`:

| eps | FD | rel. discrepancy |
|---:|---:|---:|
| 1e-2 | -0.850015258 | 0.006360 |
| 5e-3 | -0.853983567 | 0.001721 |
| 3e-3 | -0.854775605 | 0.000795 |
| 1e-3 | -0.855173811 | 0.000330 |
| 5e-4 | -0.855316985 | 0.000162 |
| 3e-4 | -0.855214709 | 0.000282 |
| 1e-4 | -0.855057993 | 0.000465 |
| 3e-5 | -0.855392796 | 0.0000736 |

Direction 1, `v=[0.339071,0.247302,0.666630,0.085793,-0.098288,-0.307725,-0.475065,-0.205118]`, AD `-0.8949515543`:

| eps | FD | rel. discrepancy |
|---:|---:|---:|
| 1e-2 | -0.882102938 | 0.014357 |
| 5e-3 | -0.891785492 | 0.003538 |
| 3e-3 | -0.893893394 | 0.001182 |
| 1e-3 | -0.894861317 | 0.000101 |
| 5e-4 | -0.895044862 | 0.000104 |
| 3e-4 | -0.895093994 | 0.000159 |
| 1e-4 | -0.895023784 | 0.0000807 |
| 3e-5 | -0.895034456 | 0.0000926 |

Direction 2, `v=[-0.317189,0.623426,0.087234,-0.010705,-0.253697,-0.287407,0.529539,0.275010]`, AD `-0.8406428648`:

| eps | FD | rel. discrepancy |
|---:|---:|---:|
| 1e-2 | -0.835563672 | 0.006042 |
| 5e-3 | -0.839694927 | 0.001128 |
| 3e-3 | -0.840541255 | 0.000121 |
| 1e-3 | -0.840968939 | 0.000388 |
| 5e-4 | -0.840973493 | 0.000393 |
| 3e-4 | -0.841055244 | 0.000490 |
| 1e-4 | -0.841024902 | 0.000454 |
| 3e-5 | -0.840968921 | 0.000388 |

Direction 3, `v=[-0.307436,0.603603,-0.388662,0.179150,-0.460796,0.264979,-0.179993,-0.207482]`, AD `0.08603574246`:

| eps | FD | rel. discrepancy |
|---:|---:|---:|
| 1e-2 | 0.088885389 | 0.032060 |
| 5e-3 | 0.086360417 | 0.003760 |
| 3e-3 | 0.085859107 | 0.002053 |
| 1e-3 | 0.085720591 | 0.003663 |
| 5e-4 | 0.085817065 | 0.002542 |
| 3e-4 | 0.085768940 | 0.003101 |
| 1e-4 | 0.085765905 | 0.003136 |
| 3e-5 | 0.085758252 | 0.003225 |

Direction 4, `v=[-0.731861,-0.268706,-0.021720,-0.054338,-0.264023,0.459985,0.238570,-0.224815]`, AD `-0.6855155008`:

| eps | FD | rel. discrepancy |
|---:|---:|---:|
| 1e-2 | -0.684746221 | 0.001122 |
| 5e-3 | -0.685194836 | 0.000468 |
| 3e-3 | -0.685332366 | 0.000267 |
| 1e-3 | -0.685418478 | 0.000142 |
| 5e-4 | -0.685299615 | 0.000315 |
| 3e-4 | -0.685304690 | 0.000308 |
| 1e-4 | -0.685345057 | 0.000249 |
| 3e-5 | -0.685451189 | 0.0000938 |

All five directions pass the three-consecutive-sign, two-consecutive-2%,
preferred-0.5%, and decreasing-error requirements. Overall result: 5/5 passed.

## 24. Tiny refinement

Exactly three steps were taken from eta0, with no multistart or sweep. All three
were accepted after exact risk, geometry, rank, forcing, and algebra gates.

```text
eta1 = [0.8953839921146673, 0.2059503590747114,
        1.3345144773868762, 0.8654744150451203,
        0.7508077339024882, 0.5179727362721115,
        1.6423936578820195, 0.5884106107337586]
```

| step | Galerkin action | exact risk |
|---:|---:|---:|
| start | 0.2645771969 | 5.3401060510 |
| 1 | 0.2641557004 | 5.3417389299 |
| 2 | 0.2641030961 | 5.3419433065 |
| 3 | 0.2640767875 | 5.3420455203 |

The reduction is `0.0005004094`; the exact risk remains below
`5.3421459587`. Minimum periodic sensor separation is `0.343889893`. The fresh
eta1 Galerkin audit passes: weak `0.0738286`, energy `0.0500417`, gauge
`1.84e-17`, and moment `0.00791614`.

## 25. Authoritative fixed-design cross-check

The eligible eta1 triggered independent high-accuracy Deep Ritz solves for eta0
and eta1, each warm-started only from the same frozen production checkpoint.
Both fresh solutions and all unchanged hard certificates passed:

| design | exact risk | action | weak | energy | gauge | moment | valid |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| eta0 | 5.3401060510 | 0.2785663836 | 0.0766024 | 0.0707009 | 2.73e-15 | 0.0145266 | yes |
| eta1 | 5.3420455203 | 0.2741148392 | 0.0790777 | 0.0668743 | 2.31e-15 | 0.0131395 | yes |

Eta1 is below the exact risk ceiling `5.3421459587`. Its train/audit projection
residuals are `7.32e-11 / 1.34e-11`, minimum ESS values are
`0.071348 / 0.069496`, forcing residuals are `8.77e-9 / 2.25e-9`, and maximum
covariance conditions are `3.9082 / 3.7711`; every hard forcing gate passes.

The authoritative eta1-minus-eta0 action difference is `-0.0044515444`, below
the declared replacement comparison by more than its `1e-6` tolerance.
Accordingly, this fixed-design cross-check records
`authoritative_improvement=true`. The workflow intentionally records
`incumbent_replaced=false`: it neither changes the production experiment nor
silently promotes the candidate.

Neither fresh L-BFGS solve declared optimizer convergence, although both were
finite, their train energy-identity errors were `0.00253` and `0.01566`, and all
selection/certificate gates passed. The held-out action standard errors are
`0.001506` and `0.001488`; the observed reduction is about 2.10 combined standard
errors. Thus the result meets the requested declared-tolerance rule, while a
further independent rerun would strengthen the replacement evidence.

## 26. Limitations and tests

- The final Galerkin increment is 2.96%, satisfying the required 3% criterion
  but not the preferred 1% target.
- The certified K=160 Galerkin action is materially different from the matching
  Deep Ritz action; the report preserves rather than hides that discrepancy.
- High-K conditioning reaches `3.72e11`; it is below the declared `1e12` cap but
  merits monitoring for still larger dictionaries.
- The nonlinear fixed-theta Deep Ritz envelope gradient remains unvalidated and
  is scientifically distinct from the validated Galerkin envelope gradient.
- The authoritative actions improve by the declared replacement tolerance and
  pass all hard certificates, but neither L-BFGS solve declared convergence and
  the observed difference is about 2.10 combined held-out standard errors.
- No independent-validation sensor claim and no Pareto sweep was attempted.

The new production suite contains the 27 required checks and passes. The
unchanged prior suites also pass: 27 production tests plus 17 Galerkin tests plus
12 continuous-gradient tests, 56 total. JAX logs the unavailable-CUDA plugin
warning and executes successfully on CPU.

## 27. Final repository status and isolation audit

Final `git status --short` was:

```text
?? ETA_OPTIMIZATION_AUDIT.md
?? experiments/skyrmions_deep_ritz_full/
```

This matches the initial top-level status: the unrelated pre-existing audit file
was untouched and every task-created file remains within the already-untracked
isolated experiment. Static source inspection found
no import from `experiments.skyrmions_deep_ritz`, no hard-coded write target into
its outputs, and all new write guards resolve beneath
`experiments/skyrmions_deep_ritz_full/outputs/production_galerkin/`.

A. PRODUCTION GALERKIN SOLVER AND ETA GRADIENT VALIDATED
