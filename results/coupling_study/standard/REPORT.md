# MFSI Stage 2: fiber-adapted coupling (coupling only)

## Method

The paper endpoint-population construction and 5 previously selected per-bank schedule(s) were reused. Only the endpoint transport plan changed. Independent pairing, conventional geometric Sinkhorn, and a fiber-aware Sinkhorn relaxation were evaluated on matched banks. The fiber plan used a geometric log-kernel plus nine measured-observable interaction parameters and was optimized for integrated exact finite-bank correction energy plus an ESS-floor penalty.

The fiber parameters were trained on a coupling-optimization bank, selected by the same objective on a validation bank, and then applied to an untouched evaluation bank. Final MMD² and hidden q4 were absent from training and selection. Soft plans were passed downstream by IID categorical pair sampling; plan marginal residuals and finite-sample marginal deviations are reported separately.

The microscopic cost centers each configuration, uses periodic minimum-image displacements, and solves the particle-exchange assignment with the Hungarian algorithm. It contains no moment-fiber, ESS, correction-energy, MMD, or q4 term.

## Frozen quantities

Schedule parameters, endpoint source populations, target moments, observables, invariant neural architecture, random-continuous-time training, optimizer settings, gate procedure, Heun step count, and MMD feature map were identical across coupling methods within each bank.

## Validation

The reduced implicit-gradient check had relative error `2.167e-05`. The largest endpoint-plan marginal L-infinity residual was `4.469e-14`; the largest calibrated finite-endpoint moment residual was `2.483e-16`.

## Aggregate metrics

| method | E_corr | min ESS | D_proj | MMD² | max moment error | generated Δq4 | microscopic cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| independent | 0.163469 | 0.116759 | 0.46964 | 0.017453 | 0.0209266 | 0.612257 | 1.03228 |
| geometric OT | 0.167745 | 0.144271 | 0.401206 | 0.0138903 | 0.0174466 | 0.604484 | 0.925113 |
| fiber-aware | 0.175826 | 0.157872 | 0.413505 | 0.0175201 | 0.018389 | 0.614591 | 0.937516 |

## Primary paired contrast

Fiber-aware minus geometric OT correction energy was `0.00808065` (95% interval `-0.0192329` to `0.0353942`, n=5).

On the separate selection banks, the selected fiber parameterization changed its own objective by `-0.0102417` relative to the geometric initialization (95% interval `-0.0211719` to `0.000688513`).

The trained-objective result and the independent-path claim are intentionally separate. The five-bank evaluation does not establish lower fiber-aware correction burden than geometric OT.

For transfer metrics, fiber-aware minus geometric OT minimum ESS was `0.0136013` (95% interval `-0.0246334` to `0.0518359`), projection distortion was `0.0122994` (`-0.0296066` to `0.0542054`), and projected-law MMD² was `0.00362982` (`0.00054818` to `0.00671145`).

The calibrated source endpoint q4 gap was `0.698954` on average. Generated q4 changes from the first interior evaluation time to t=1 were independent `0.612257`, geometric OT `0.604484`, fiber-aware `0.614591`; q4 remained held out from coupling and network objectives.

## Limitations

The transport plans live on finite resampled endpoint banks, and downstream paired banks are IID samples from soft plans, so their realized empirical marginals fluctuate even though the underlying plans satisfy the marginals numerically. Neural end-to-end limitations remain visible; no method was tuned using final MMD². This is Stage 2 only: schedule parameters were not updated and no joint schedule-coupling optimization was implemented.
