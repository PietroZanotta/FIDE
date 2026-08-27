# Fixed K=280 quadrature qualification: mandatory-ladder early-stop report

## Scope

This report records the completed mandatory part of the frozen empirical-
quadrature qualification in
`GALERKIN_K280_QUADRATURE_EXTENSION_PROTOCOL.md`. It is a selection-development
study only. It did not access validation data, optimize eta, run a Pareto sweep,
change K, alter the dictionary or normalization, retune rank handling, or relax
any scientific threshold.

The fixed numerical problem was:

- `K = 280`;
- dictionary SHA-256
  `37e9b60fcb92c4e5a0ee7ec1651fb7f8889f7ac6bdb02d3bd314e9ef40833326`;
- relative rank tolerance `1e-12`;
- held-out Ritz-energy threshold `0.08`;
- minimum ESS fraction `0.05`;
- the same six frozen geometries used by the preceding resolution study.

All 24 mandatory geometry/support evaluations completed and were written
atomically below `outputs/galerkin_k280_quadrature_extension/support/`. The
maximum deterministic train and audit banks, their hashes, the old-certificate
audit, and the frozen protocol are also preserved below that output root.

## Why the run was stopped

The frozen protocol conditionally requested one more `131072 / 65536` row for
each geometry when the mandatory ladder did not qualify. The first optional row
attempt exhausted GPU memory while JAX tried to allocate another `1.62 GiB`.
No optional row was completed or written.

More importantly, completing that train-only extension could not change the
scientific decision. Its audit support remains fixed at 65,536, while three
geometries already fail the unchanged ESS gate on that exact audit prefix.
Increasing only the train support cannot repair those audit failures. At the
user's direction, no lower-value restart or allocator workaround was attempted.

This is therefore not represented as a fully completed execution of the frozen
optional protocol. It is a complete mandatory-ladder result with a decisive
physical failure and a separately documented optional-level memory blocker.

## Mandatory support ladder

Every geometry was evaluated at the four predeclared nested support pairs:

```text
32768 / 16384
32768 / 32768
65536 / 32768
65536 / 65536
```

The table below gives the largest mandatory row. `A_T` and `A_A` are train and
audit Galerkin actions. All weak, energy, gauge, moment-rate, projection,
forcing-mean, covariance, geometry, rank, range, stationarity, symmetry, and
restricted-identity gates passed at this row. A failed physical certificate is
therefore an ESS failure only.

| geometry | A_T | A_A | gradient norm | ESS T/A | weak | energy | physical certificate |
|---|---:|---:|---:|---:|---:|---:|---|
| Law | 0.3604449280 | 0.3590242264 | 2.87260579 | 0.0511975 / **0.0438389** | 0.0298666 | 0.0154001 | **fail: audit ESS** |
| historical 0.5% | 0.3350985232 | 0.3337679921 | 2.68141778 | 0.0572933 / **0.0477785** | 0.0330584 | 0.0200575 | **fail: audit ESS** |
| historical 1% | 0.3116898172 | 0.3109110823 | 2.72843073 | 0.0555850 / 0.0518067 | 0.0368689 | 0.0382632 | pass |
| historical 2% | 0.2927236423 | 0.2925116188 | 2.60522239 | **0.0458781 / 0.0453484** | 0.0410982 | 0.0617274 | **fail: train and audit ESS** |
| eta0 3% | 0.2737859925 | 0.2733167298 | 2.58063849 | 0.0603463 / 0.0614512 | 0.0290130 | 0.0430113 | pass |
| eta-grad 3% | 0.2732118308 | 0.2727401832 | 2.57945137 | 0.0608275 / 0.0620028 | 0.0288261 | 0.0425338 | pass |

Thus three of six geometries fail the mandatory physical requirement. The ESS
threshold is exactly `0.05`; it was not rounded, reinterpreted, or weakened.

## Mandatory convergence gates

The frozen mandatory analysis compares `32768/32768 -> 65536/32768` for the
train increase and `65536/32768 -> 65536/65536` for the audit increase. The
reported action columns are the train/audit relative changes for the train
increase, followed by the audit-action change for the audit increase. The
gradient comparison belongs to the train increase; changing audit support does
not mathematically change the train-bank eta gradient, and the observed repeat
differences were only floating-point noise.

| geometry | train-step dA T/A | audit-step dA_A | gradient cosine | relative gradient change | action gate | direction gate | magnitude gate |
|---|---:|---:|---:|---:|---|---|---|
| Law | 0.4994% / 0.3472% | 0.05960% | 0.999316 | 4.1194% | pass | pass | pass |
| historical 0.5% | 0.7660% / 0.6389% | 0.02493% | 0.999467 | 3.3573% | pass | pass | pass |
| historical 1% | 1.3197% / 1.1973% | 0.02982% | 0.999493 | 3.8976% | pass | pass | pass |
| historical 2% | 1.4159% / 1.2447% | 0.17022% | 0.999041 | **5.5798%** | pass | pass | **fail** |
| eta0 3% | 1.0119% / 0.9524% | 0.10611% | 0.999160 | 4.8984% | pass | pass | pass |
| eta-grad 3% | 1.0138% / 0.9556% | 0.10616% | 0.999173 | 4.8785% | pass | pass | pass |

All six geometries satisfy the required 2% action limit and 0.995 gradient-
cosine limit. Five satisfy the 5% relative-gradient limit. Historical 2% is the
sole magnitude failure at `0.0557982`.

The aggregate mandatory result is therefore:

```text
physical_valid             = false
action_stable              = true
gradient_direction_stable  = true
gradient_magnitude_stable  = false
```

## Continuous 3% result on the new mandatory banks

The continuously refined 3% geometry remains better than eta0 on the largest
mandatory banks:

| view | eta0 action | eta-grad action | eta-grad minus eta0 | improvement |
|---|---:|---:|---:|---:|
| train | 0.273785992486 | 0.273211830844 | -0.000574161642 | 0.209712% |
| audit | 0.273316729814 | 0.272740183208 | -0.000576546605 | 0.210944% |

Both 3% geometries pass every physical and numerical certificate at
`65536 / 65536`. This preserves evidence for the local eta-gradient improvement,
but it does not override the protocol's all-six-geometry qualification rule.

## Conditional AD/FD stage

The new limited AD/FD audit was not run. Its frozen prerequisite required final
physical validity and basic action stability. Action stability passed, but
physical validity failed. The earlier independent five-direction K=280 AD/FD
result remains preserved in `FINAL_3PCT_GALERKIN_CROSSCHECK.md`; it is not
relabelled as a substitute for this gated stage.

## Consequences and next step

K=280 must not yet be frozen for a new official Pareto or production eta
optimization under this quadrature protocol. The failure is not evidence that
gradient-based eta optimization is conceptually invalid: action ordering,
gradient direction, and the eta-grad improvement were stable on the mandatory
banks. The blocker is the all-geometry physical qualification, specifically ESS,
plus one marginal gradient-magnitude failure.

A scientifically useful follow-up would need a separately frozen protocol that
increases or redesigns the audit quadrature support, rather than merely
increasing train support. Any sampling or estimator change would require a new
qualification audit. The completed banks, 24 rows, hashes, and diagnostics here
remain reusable as immutable comparison data; they do not need to be discarded
or recomputed unchanged.

No validation quantity entered this decision, and no eta candidate was changed
or selected during this study.

C. K280 ACTION/PHYSICAL QUADRATURE NOT YET CONVERGED
