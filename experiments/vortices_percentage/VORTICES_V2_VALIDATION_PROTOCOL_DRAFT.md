# Vortices V2 selection and validation protocol draft — version 2

**Version date:** 2026-08-30

## Status

**DRAFT V2 — NUMERICAL METHOD FREEZE-READY; DO NOT RUN SELECTION OR
VALIDATION YET.**

This version was amended after the development-only continuity-commutator audit
and complete reflected pre-freeze replay, but before any new V2 reference,
selection, or validation bank was generated. The reflection/Neumann numerical
method passes every pre-freeze gate and is eligible to be frozen before the
three prospectively seeded reference models are trained. Any later amendment
must be versioned and dated before the affected new scientific outcomes are
inspected.

The V1 64-, 256-, and 2,048-trial banks are excluded from V2 confirmation.
They may be used only as labeled development/mechanism data.

## 1. Scientific identity to freeze

| Item | Proposed V2 definition |
|:---|:---|
| Projection | Hard empirical information projection; no soft fiber |
| Defect | `s = partial_t q + div(q u)` |
| Correction | `delta = -grad psi` |
| PDE | `-div(q grad psi) = -s` |
| Action | normalized trapezoidal integral of `int q |grad psi|^2` over 21 nodes |
| Scalar raster | direct particle, cell-integrated even-reflection/Neumann Gaussian on `[0,2] x [0,1]` |
| Signed source | identical reflected scalar kernel applied to particle signed defect |
| Reference flux | `j_h=S_reflect(q u)`, odd reflection in its normal coordinate and even reflection tangentially |
| Regularized velocity | `u_h=j_h/q_h`; never `u_original` sampled on the grid |
| Boundary rule | homogeneous no-flux; `j_h dot n=0` exactly at domain faces |
| Reflected images | four translated image pairs on either side of the central image |
| Source-column normalization | none |
| Density floor | exactly zero in the scientific operator |
| Raster | `256 x 128` |
| Bandwidth rule | median weighted two-dimensional Scott bandwidth from the frozen reference rollout, without a grid floor |
| Reference particles | `32,768` per reference replicate |
| Time nodes | 21 equally spaced normalized times |
| Precision | float64 throughout |

For each of the three qualified frozen reference rollouts, compute one weighted
two-dimensional Scott bandwidth by the same reference-only rule. Before any
sensor optimization, freeze the median of those three values as one common
physical bandwidth. Use that same value for every reference replicate,
geometry, allowance, observation trial, selection stage, and validation
stage. It may not be selected or changed using action, risk, or Full-vs-Law
performance. The current development replay retains its previously fixed
single-reference value `0.05883961987664522`; it is not the as-yet-uncomputed
three-reference value.

## 2. Required pre-freeze numerical gate

Before creating selection banks, repeat the development harness after resolving
the boundary-kernel/deposition continuity issue. The method is eligible to
freeze only if all conditions pass:

These thresholds are pre-freeze development gates selected after the
mechanism audit and before any new V2 reference, selection, or validation
outcome was generated or inspected.

1. mass error `<= 5e-13` and absolute integrated source `<= 5e-12`;
2. strictly positive `q_h` and exactly one conductive component;
3. physical-Poisson relative residual `<= 2e-7`;
4. independent action relative error `<= 2e-6`;
5. every curated `128 x 64 -> 256 x 128` action change `<= 5%` and smaller
   than its `64 x 32 -> 128 x 64` change;
6. every recalibrated `16,384 -> 32,768` particle action change `<= 10%`;
7. particle weak moment identity maximum absolute error `<= 1e-8`;
8. reflected manufactured continuity for constant interior motion,
   near-boundary motion, and a tangent-to-all-boundaries field has relative L2
   and maximum weak relative error `<= 1e-8`, scalar mass error `<= 5e-13`,
   and zero normal reflected flux to absolute tolerance `1e-14`;
9. for the real golden finite-time identity at `epsilon=2e-4`, every grid has
   relative L2 and maximum weak relative error `<= .005`, the relative-L2 range
   across `64 x 32`, `128 x 64`, and `256 x 128` is `<= .0005`, correlation
   between the two sides is `>= .999`, and error strictly decreases from
   `1e-3` through `5e-4` to `2e-4` on every grid;
10. reflected common-raster Full/Tangent/Hidden checks pass at absolute
    tolerance `1e-6`: Full and Tangent moment feasibility, hidden nullspace,
    Tangent-Hidden orthogonality, Pythagorean identity, and raw
    `A_tan<=A_full`; and
11. every focused V2 test passes under float64.

Failure of any gate blocks selection. A bandwidth multiplier may not be chosen
to repair a failure after comparing scientific action reductions.

## 3. Reference-model replication

Train three endpoint-only reference models from scratch with reference-training
seeds:

```text
20260815, 20260816, 20260817.
```

Each reference uses the same architecture, endpoint data policy, training
steps, optimizer, bridge, box-logit transform, rollout integrator, and
32,768-particle bank rule. Each checkpoint and bank receives an independent
hash and qualification receipt. A failed reference qualification is reported;
it is not silently replaced after scientific outcomes are inspected.

Observation-trial uncertainty and reference-training uncertainty remain
separate. Results are reported per reference seed. No observation-level SE may
be presented as if it included reference-seed variation.

## 4. Population, risk, and sensor optimization

The complete V2 experiment reruns Population, Law, Tangent, and Full. V1
geometries may be proposal seeds only if declared before selection, but they
cannot be frozen winners or validation targets without winning the new V2
selection objectively.

The scientific risk definitions remain those of the Vortices percentage
experiment:

- population Law loss uses multiscale Gaussian MMD bandwidths `.05`, `.10`,
  `.20`, and `.40`;
- absolute population slack is `epsilon_L=.025`;
- allowances are `.5%`, `1%`, `2%`, `3%`, `4%`, and `5%` relative increases
  from the newly selected V2 Law risk anchor;
- sensor centers remain in `[.24,1.76] x [.24,.76]` with minimum separation
  `.24`, four Gaussian sensors, and width `.12`;
- each observation trial retains 2,000 particles, nine acquisition nodes,
  detector noise standard deviation `.005`, exact endpoints, and the declared
  bounded endpoint-anchored cubic reconstruction.

Selection uses paired observation realizations across methods and references.
The proposed sample sizes are fixed for planning, not chosen from the favorable
variance of the seven development cases:

| Stage | Trials per reference | Purpose |
|:---|---:|:---|
| Law finite-risk selection | 64 | establish the new Law anchor and risk caps |
| Tangent/Full action prescreen | 32 | discard clearly noncompetitive numerical candidates only |
| Tangent/Full final selection | 128 | choose one geometry per method/allowance across all three references |
| Independent validation | 1,024 | estimate each frozen geometry on each reference |

Thus final validation contains 3,072 observation trials per geometry, but the
three 1,024-trial reference strata remain identifiable. There is no
outcome-dependent stopping. If resource review changes these counts, the
protocol must be amended before any corresponding bank is generated.

Proposed namespaces are reserved as follows:

| Reference seed | Selection namespace | Validation namespace |
|---:|---:|---:|
| 20260815 | 29890 | 29891 |
| 20260816 | 29892 | 29893 |
| 20260817 | 29894 | 29895 |

## 5. Selection rule

For each allowance, a candidate must pass exact population loss, finite-risk,
calibration, ESS, support, continuity, component, Poisson, sign/moment, and
decomposition gates on every reference replicate. The final action objective is
the equal-weight arithmetic mean of the three reference-specific arithmetic
means over the 128 paired selection trials.

The allowance sweep is nested. The previous tighter winner is a mandatory
incumbent. A winner is replaced only by a fully feasible candidate whose
declared V2 selection action is lower by more than `1e-6`. Selection freezes
all geometries before any validation action is computed.

## 6. Validation estimand and inference

For reference seed `r` and allowance `p`, the primary effect is

```text
D_r,p = 1 - mean(A_full,r,p) / mean(A_law,r).
```

This is a ratio of arithmetic means. No trimming, winsorization, censoring,
median substitution, log contrast, or post-hoc action cap is permitted. All
finite numerically valid actions are retained; invalid evaluations trigger the
numerical gate rather than deletion.

Within each reference stratum, use a paired shared-index bootstrap with
100,000 resamples and seed `821775`. The same resampled observation indices are
used for Law and every Full allowance. The confirmatory family contains all 18
reference-by-allowance reductions. A maximum-absolute-deviation simultaneous
95% interval is constructed across those 18 effects.

Reference-seed replication is a separate gate: every one of the three
reference-specific reductions must have a strictly positive simultaneous lower
bound at every allowance. Also report the equal-reference average and the full
between-reference range, but do not treat three reference seeds as thousands
of independent observation trials.

## 7. Statistical and numerical pass conditions

The final claim passes only if all conditions hold:

1. exactly three qualified reference checkpoints and their frozen banks are
   present;
2. exactly 1,024 validation trials per reference and geometry are present;
3. every numerical evaluation passes calibration, ESS, support, mass/source,
   continuity, component, Poisson, moment, decomposition, and float64 gates;
4. all 18 simultaneous 95% lower bounds for `D_r,p` are strictly positive;
5. the largest simultaneous interval half-width is `<= .05`;
6. every Law and Full arithmetic mean has relative SE `<= .10` within its
   reference stratum; and
7. no result relies on V1 trials, V1 winners, outcome-dependent stopping, or a
   post-inspection change of estimand.

Maximum/median ratios, top-1% action shares, leave-one-out influence, action
quantiles, and bank-to-bank ranges are mandatory diagnostics but are not a
license to delete valid high actions.

## 8. Frozen artifact and provenance requirements

Every selected and validation result records:

- Git commit plus dirty-worktree status;
- code and config hashes;
- reference checkpoint/training seed and bank hashes;
- truth and observation bank hashes and namespaces;
- exact sensor geometry and canonical geometry key;
- projection type;
- reflected scalar and matched-flux definitions, image-pair count, grid,
  bandwidth rule and value;
- density regularization (`0` unless prospectively amended);
- particle count, time grid, and float precision;
- all numerical residuals and validity decisions; and
- per-trial and per-time action receipts.

No output may be written beneath the V1 `outputs/pareto/` tree. Development,
selection, and validation destinations must be distinct and fail closed on a
configuration or input mismatch.

## 9. Soft-fiber contingency

Soft moment fibers are not part of this draft. They may be proposed only after
the hard-fiber raster-continuity issue is resolved and a prospectively fixed
development study shows numerically convergent catastrophic events caused by
hard noisy calibration. Such an amendment requires its own derivation,
reconstruction uncertainty covariance, tests, selection rerun, and validation
protocol. A covariance ridge applied only to `lambda_dot` is forbidden.
