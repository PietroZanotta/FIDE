# Exact Tangent/Full action-decomposition audit

**Overall status: PASS.**

This is a post-selection audit of the saved final Law, Tangent, and Full candidates. Each unique geometry was evaluated by the experiment's existing authoritative exact evaluator on the frozen action-selection bank. Tangent and Full were computed in the same evaluator call from the same reconstructed targets and projected weights.

The hierarchy tolerance is `1e-06`. Raw violations are defined as `A_tan - A_full` and are never clipped; negative values indicate hierarchy slack.

## Concise summary

- Maximum raw violation over aggregate, trial, and time/trial levels: `-0.138640981045`.
- Aggregate violations: `0`.
- Trial-level violations: `0`.
- Time/trial-level violations: `0`.
- Invalid exact-evaluator trials: `0`.
- Maximum absolute audited-minus-reported Tangent action: `5.82639492208e-12`.
- Maximum absolute audited-minus-reported Full action: `7.91620546892e-10`.
- Every saved final candidate passes: `true`.
- Frozen selection bank SHA-256: `0ae52680ba66f07e36e02a0d85d25847fc11dc2554fcb63f95cb4e7aa0636ef9`.

## Candidate table

| Allow. | Design | A_tan | A_full | A_hid | Gamma | max raw violation | trial viol. | time/trial viol. | Pass |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.5% | Law | 0.4477605247 | 3.458634259 | 3.010873734 | 0.8705383423 | -1.7269e-01 | 0 | 0 | yes |
| 0.5% | Tangent | 0.440839066 | 3.857565659 | 3.416726593 | 0.885720917 | -1.7383e-01 | 0 | 0 | yes |
| 0.5% | Full | 0.4434423251 | 3.176462048 | 2.733019723 | 0.8603974112 | -1.8365e-01 | 0 | 0 | yes |
| 1% | Law | 0.4477605247 | 3.458634259 | 3.010873734 | 0.8705383423 | -1.7269e-01 | 0 | 0 | yes |
| 1% | Tangent | 0.4307643098 | 3.579249434 | 3.148485124 | 0.8796495417 | -1.5443e-01 | 0 | 0 | yes |
| 1% | Full | 0.4374615264 | 2.825818445 | 2.388356919 | 0.8451912128 | -1.7527e-01 | 0 | 0 | yes |
| 2% | Law | 0.4477605247 | 3.458634259 | 3.010873734 | 0.8705383423 | -1.7269e-01 | 0 | 0 | yes |
| 2% | Tangent | 0.4307643098 | 3.579249434 | 3.148485124 | 0.8796495417 | -1.5443e-01 | 0 | 0 | yes |
| 2% | Full | 0.4374615264 | 2.825818445 | 2.388356919 | 0.8451912128 | -1.7527e-01 | 0 | 0 | yes |
| 3% | Law | 0.4477605247 | 3.458634259 | 3.010873734 | 0.8705383423 | -1.7269e-01 | 0 | 0 | yes |
| 3% | Tangent | 0.4239900533 | 3.061006837 | 2.637016783 | 0.8614867343 | -1.3864e-01 | 0 | 0 | yes |
| 3% | Full | 0.4374615264 | 2.825818445 | 2.388356919 | 0.8451912128 | -1.7527e-01 | 0 | 0 | yes |
| 4% | Law | 0.4477605247 | 3.458634259 | 3.010873734 | 0.8705383423 | -1.7269e-01 | 0 | 0 | yes |
| 4% | Tangent | 0.3883788541 | 2.471982635 | 2.083603781 | 0.8428877094 | -1.6461e-01 | 0 | 0 | yes |
| 4% | Full | 0.3883788541 | 2.471982635 | 2.083603781 | 0.8428877094 | -1.6461e-01 | 0 | 0 | yes |
| 5% | Law | 0.4477605247 | 3.458634259 | 3.010873734 | 0.8705383423 | -1.7269e-01 | 0 | 0 | yes |
| 5% | Tangent | 0.3877498844 | 2.440305918 | 2.052556033 | 0.8411060345 | -1.5029e-01 | 0 | 0 | yes |
| 5% | Full | 0.3878727661 | 2.43016534 | 2.042292574 | 0.8403924376 | -1.4569e-01 | 0 | 0 | yes |

## Definitions and checks

```text
A_hid = A_full - A_tan
Gamma = 1 - A_tan / A_full
raw hierarchy violation = A_tan - A_full
violation iff raw hierarchy violation > configured tolerance
```

The CSV retains full-precision aggregates, raw signed violations at every reported level, reported-vs-audited action deltas, geometry, and source-result paths. The evaluation JSON retains the per-trial and per-time values returned by the exact evaluator for reproducibility.
