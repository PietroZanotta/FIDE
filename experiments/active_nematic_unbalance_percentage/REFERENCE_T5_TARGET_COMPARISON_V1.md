# Active-nematic reference target comparison: t=10, t=15, and t=25

## Status

THREE-TARGET DEVELOPMENT COMPARISON COMPLETE

All three studies start from physical t=5 and use the same 96 deterministic physical
runs, explicit 64/16/16 roles, defect extraction, three learned-reference base
seeds, plus/minus factorization, architecture, optimizer, endpoint sampling,
KDE quadrature, and endpoint-only protocol. The target endpoints are t=10,
t=15, and t=25.

None of the studies accessed Pareto validation data or replaced an official
reference.

## Clarifying the visual diagnosis

The t=5 to t=25 figure does not show a globally bad final distribution:

- initial spatial overlap is 0.59 for plus and 0.58 for minus;
- t=25 spatial overlap is 0.65 to 0.69, so the final spatial fit is better;
- initial beta overlap is 0.77 for plus and 0.91 for minus;
- t=25 beta overlap falls to 0.65--0.67 for plus and 0.82--0.83 for minus.

The weakness is therefore the terminal orientation marginal, especially for
plus defects, rather than the complete final law. The production periodic 3-D
MMD-squared combines position and beta and places the t=25 target error on the
same order as the initial KDE representation error.

## Endpoint support

| target | training mass at t=5 | training mass at target | training defects at target | nonempty training runs at target |
|---:|---:|---:|---:|---:|
| 10 | 2.78125 | 7.78125 | 498 per species | 64 / 64 |
| 15 | 2.78125 | 10.265625 | 657 per species | 64 / 64 |
| 25 | 2.78125 | 11.65625 | 746 per species | 64 / 64 |

Defect mass grows rapidly between t=5 and t=15. The t=10 population has 75.8%
of the t=15 training defects, while the t=15 population has 88.1% of the t=25
training defects.

## Production endpoint MMD-squared

| seed | species | target MMD2 t=10 | target MMD2 t=15 | target MMD2 t=25 | reduction vs no transport t=10 | reduction t=15 | reduction t=25 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20260818 | plus | 0.00215560 | 0.00175112 | 0.00195127 | 48.09% | 56.61% | 42.55% |
| 20260818 | minus | 0.00227468 | 0.00188044 | 0.00200703 | 44.90% | 46.79% | 39.45% |
| 20260819 | plus | 0.00219224 | 0.00168470 | 0.00167474 | 47.21% | 58.26% | 50.69% |
| 20260819 | minus | 0.00214012 | 0.00182209 | 0.00205826 | 48.16% | 48.44% | 37.90% |
| 20260820 | plus | 0.00223426 | 0.00170000 | 0.00171643 | 46.20% | 57.88% | 49.46% |
| 20260820 | minus | 0.00207471 | 0.00182999 | 0.00192259 | 49.75% | 48.22% | 42.00% |

| target | mean target MMD2 | target MMD2 range | mean reduction vs no transport |
|---:|---:|---:|---:|
| 10 | 0.00217860 | 0.00207471--0.00227468 | 47.39% |
| 15 | 0.00177806 | 0.00168470--0.00188044 | 52.70% |
| 25 | 0.00188839 | 0.00167474--0.00205826 | 43.68% |

The t=15 target is best on both aggregate criteria. Its mean target MMD-squared
is 18.4% below t=10 and 5.8% below t=25. It also has the largest mean
improvement over its target-specific no-transport baseline. The shortest
interval is therefore not the easiest endpoint-law problem under this frozen
training setup.

Absolute MMD values compare fits to different physical target distributions;
they are informative because the metric, initial endpoint, seeds, and fitting
protocol are shared, but they are not a same-target optimizer ablation.

## Spatial and beta marginal comparison

| marginal | t=10 target range | t=15 target range | t=25 target range | best overall |
|---|---:|---:|---:|---|
| plus spatial overlap | 0.664--0.667 | 0.672--0.681 | 0.682--0.687 | t=25 |
| minus spatial overlap | 0.655--0.658 | 0.668--0.672 | 0.652--0.670 | t=15 |
| plus beta overlap | 0.632--0.649 | 0.668--0.683 | 0.652--0.666 | t=15 |
| minus beta overlap | 0.819--0.845 | 0.826--0.839 | 0.816--0.834 | mixed, t=10/t=15 |

Moving the target from t=25 to t=15 modestly improves orientation fit. Moving
it further back to t=10 reverses that improvement for plus beta and both
spatial marginals. None of the targets removes the plus-orientation gap: even
the best t=15 plus-beta overlap remains below the shared t=5 initial overlap of
0.766.

These smoothed-histogram overlaps are visualization diagnostics. The periodic
3-D MMD-squared table is the declared quantitative endpoint audit.

## Interpretation

Using t=15 as the target is preferable when endpoint-reference fidelity is the
priority. It has the best mean production MMD-squared, the largest mean gain
over no transport, and the best plus-beta overlap. The t=10 result establishes
that reference difficulty is not monotone in interval length.

The likely explanation is that target-distribution structure and rapid defect
birth matter alongside horizon length; the current comparison does not isolate
those mechanisms causally. Eliminating the remaining orientation gap would
require a separate prospective training ablation, such as additional
orientation-sensitive model capacity or loss weighting. Either would change
the reference-training setup and should not be introduced retrospectively.

## Artifacts

t=5 to t=10:

- `outputs/reference_t5_t10_v1/reference_endpoint_audit.json`
- `outputs/reference_t5_t10_v1/reference_manifest.json`
- `outputs/reference_t5_t10_v1/reference_seed_*/{plus,minus}_reference.npz`
- `figures/active_nematic_reference_endpoint_fit_t5_t10_v1.png`
- `figures/active_nematic_reference_endpoint_fit_t5_t10_v1.pdf`

t=5 to t=15:

- `outputs/reference_t5_t15_v1/reference_endpoint_audit.json`
- `outputs/reference_t5_t15_v1/reference_manifest.json`
- `outputs/reference_t5_t15_v1/reference_seed_*/{plus,minus}_reference.npz`
- `figures/active_nematic_reference_endpoint_fit_t5_t15_v1.png`
- `figures/active_nematic_reference_endpoint_fit_t5_t15_v1.pdf`

t=5 to t=25:

- `outputs/reference_t5_t25_v1/reference_endpoint_audit.json`
- `outputs/reference_t5_t25_v1/reference_manifest.json`
- `figures/active_nematic_reference_endpoint_fit_t5_t25_v1.png`
- `figures/active_nematic_reference_endpoint_fit_t5_t25_v1.pdf`

NO intermediate-truth training

NO Pareto validation access

NO official reference replacement
