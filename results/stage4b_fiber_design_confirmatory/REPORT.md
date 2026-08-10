# MFSI Stage 4B — Confirmatory Fiber Design

## Confirmatory decisions

- `full_grad - hand` **confirmed**: mean paired difference `-0.39903208`, paired SD `0.21005153`, 95% paired t interval `[-0.54929389, -0.24877027]`, favorable seeds `9/10`.
- `full_grad - stop_grad` **did not confirm**: mean paired difference `-0.0064996792`, paired SD `0.29863689`, 95% paired t interval `[-0.22013164, 0.20713228]`, favorable seeds `6/10`.
- Stage 5 is **not scientifically justified by the strong-success rule**. Stage 5 was not implemented.

The decision uses only the ten predeclared new seeds 426–435. No Stage-4 pilot seed was pooled into either interval.

## Untouched evaluation-bank metrics

| fiber | objective | correction energy | forcing power | minimum ESS | median ESS | max calibration residual | endpoint residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| hand | 0.79511099 | 0.26493856 | 25.718351 | 0.13651606 | 0.26276994 | 1.841e-16 | 5.905e-18 |
| stop_grad | 0.40257859 | 0.11870557 | 14.081189 | 0.33746134 | 0.55089769 | 1.700e-02 | 4.889e-18 |
| full_grad | 0.39607891 | 0.14956563 | 12.139762 | 0.49124998 | 0.64034926 | 2.789e-16 | 5.296e-18 |

## Seed-level objective differences

| seed | full - hand | full - stop |
|---:|---:|---:|
| 426 | -0.38295053 | 0.082125205 |
| 427 | -0.81442053 | -0.37965146 |
| 428 | -0.31265111 | -0.20069643 |
| 429 | -0.38159664 | -0.18337271 |
| 430 | -0.4463216 | -0.26711218 |
| 431 | -0.59446529 | 0.25075324 |
| 432 | -0.29744907 | -0.06434984 |
| 433 | 0 | 0.21705229 |
| 434 | -0.43426999 | -0.13235923 |
| 435 | -0.32619604 | 0.61261433 |

## Numerical and protocol checks

Forward equivalence passed at four deterministic fibers; maximum absolute difference was `0.000e+00`.

The full-gradient directional check had relative error `7.540e-09`. The full-versus-stop gradient discrepancy norm was `25.697737` (relative discrepancy `1.2204035`).

Across all selected fibers, maximum row-orthonormality, endpoint-equivalence, and nullspace residuals were `1.000e-12`, `5.905e-18`, and `5.905e-18`.

q4 and angular descriptors were excluded from the dictionary, objective, optimization, and checkpoint selection. q4 was computed only after every checkpoint choice had been frozen and is retained only as an evaluation diagnostic.

`D_proj` is not emitted because it is not available in the frozen Stage-4 construction code. The optional downstream test was not run.

## Interpretation

Fiber-design success only: optimization beat the hand fiber, but this experiment did not establish that the full implicit gradient is essential.
