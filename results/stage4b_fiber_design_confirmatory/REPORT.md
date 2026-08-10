# MFSI Stage 4B — Confirmatory Fiber Design

## Confirmatory decisions

- `full_grad - hand` **confirmed**: mean paired difference `-0.50039786`, paired SD `0.33106403`, 95% paired t interval `[-0.73722681, -0.26356892]`, favorable seeds `9/10`.
- `full_grad - stop_grad` **did not confirm**: mean paired difference `-0.16098554`, paired SD `0.24970648`, 95% paired t interval `[-0.33961479, 0.017643712]`, favorable seeds `7/10`.
- Stage 5 is **not scientifically justified by the strong-success rule**. Stage 5 was not implemented.

The decision uses only the ten predeclared new seeds 426–435. No Stage-4 pilot seed was pooled into either interval.

## Untouched evaluation-bank metrics

| fiber | objective | correction energy | forcing power | minimum ESS | median ESS | max calibration residual | endpoint residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| hand | 0.79511099 | 0.26493856 | 25.718351 | 0.13651606 | 0.26276994 | 2.077e-16 | 7.792e-18 |
| stop_grad | 0.45569867 | 0.1498083 | 15.276212 | 0.36191332 | 0.55583612 | 2.836e-03 | 4.730e-18 |
| full_grad | 0.29471313 | 0.11046128 | 9.1653937 | 0.52628154 | 0.66841762 | 2.290e-16 | 5.953e-18 |

## Seed-level objective differences

| seed | full - hand | full - stop |
|---:|---:|---:|
| 426 | -0.38295061 | 0.082125126 |
| 427 | -0.72094707 | -0.61105188 |
| 428 | -0.31265115 | -0.20069647 |
| 429 | -0.38159671 | -0.18337278 |
| 430 | -0.44371261 | -0.39383798 |
| 431 | -1.1907725 | -0.37435709 |
| 432 | -0.29744909 | -0.06434986 |
| 433 | 0 | 0.21705229 |
| 434 | -0.45539652 | -0.13150001 |
| 435 | -0.81850234 | 0.050133253 |

## Numerical and protocol checks

Forward equivalence passed at four deterministic fibers; maximum absolute difference was `0.000e+00`.

The full-gradient directional check had relative error `1.082e-09`. The full-versus-stop gradient discrepancy norm was `25.697737` (relative discrepancy `1.2204035`).

Across all selected fibers, maximum row-orthonormality, endpoint-equivalence, and nullspace residuals were `1.000e-12`, `7.792e-18`, and `7.792e-18`.

q4 and angular descriptors were excluded from the dictionary, objective, optimization, and checkpoint selection. q4 was computed only after every checkpoint choice had been frozen and is retained only as an evaluation diagnostic.

`D_proj` is not emitted because it is not available in the frozen Stage-4 construction code. The optional downstream test was not run.

## Interpretation

Fiber-design success only: optimization beat the hand fiber, but this experiment did not establish that the full implicit gradient is essential.
