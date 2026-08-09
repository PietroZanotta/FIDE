# MFSI Stage 2: fiber-adapted coupling (coupling only)

## Method

The paper endpoint-population construction and 1 previously selected per-bank schedule(s) were reused. Only the endpoint transport plan changed. Independent pairing, conventional geometric Sinkhorn, and a fiber-aware Sinkhorn relaxation were evaluated on matched banks. The fiber plan used a geometric log-kernel plus nine measured-observable interaction parameters and was optimized for integrated exact finite-bank correction energy plus an ESS-floor penalty.

The fiber parameters were trained on a coupling-optimization bank, selected by the same objective on a validation bank, and then applied to an untouched evaluation bank. Final MMD² and hidden q4 were absent from training and selection. Soft plans were passed downstream by IID categorical pair sampling; plan marginal residuals and finite-sample marginal deviations are reported separately.

The microscopic cost centers each configuration, uses periodic minimum-image displacements, and solves the particle-exchange assignment with the Hungarian algorithm. It contains no moment-fiber, ESS, correction-energy, MMD, or q4 term.

## Frozen quantities

Schedule parameters, endpoint source populations, target moments, observables, invariant neural architecture, random-continuous-time training, optimizer settings, gate procedure, Heun step count, and MMD feature map were identical across coupling methods within each bank.

## Validation

The reduced implicit-gradient check had relative error `2.167e-05`. The largest endpoint-plan marginal L-infinity residual was `5.551e-17`; the largest calibrated finite-endpoint moment residual was `1.841e-16`.

## Aggregate metrics

| method | E_corr | min ESS | D_proj | MMD² | max moment error | generated Δq4 | microscopic cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| independent | 0.313231 | 0.0792145 | 0.757397 | 0.0461879 | 0.0332386 | 0.648151 | 1.01405 |
| geometric OT | 0.200624 | 0.103262 | 0.641745 | 0.0227421 | 0.0266192 | 0.628795 | 0.911378 |
| fiber-aware | 0.187134 | 0.138261 | 0.589882 | 0.00949309 | 0.0166512 | 0.593635 | 0.911807 |

## Primary paired contrast

Fiber-aware minus geometric OT correction energy was `-0.0134905` (95% interval `-0.0134905` to `-0.0134905`, n=1).

On the separate selection banks, the selected fiber parameterization changed its own objective by `-0.0228333` relative to the geometric initialization (95% interval `-0.0228333` to `-0.0228333`).

The trained-objective result and the independent-path claim are intentionally separate. The one-bank quick result is directional only and cannot establish a replicated coupling effect.

For transfer metrics, fiber-aware minus geometric OT minimum ESS was `0.0349993` (95% interval `0.0349993` to `0.0349993`), projection distortion was `-0.0518624` (`-0.0518624` to `-0.0518624`), and projected-law MMD² was `-0.013249` (`-0.013249` to `-0.013249`).

The calibrated source endpoint q4 gap was `0.664887` on average. Generated q4 changes from the first interior evaluation time to t=1 were independent `0.648151`, geometric OT `0.628795`, fiber-aware `0.593635`; q4 remained held out from coupling and network objectives.

## Limitations

The transport plans live on finite resampled endpoint banks, and downstream paired banks are IID samples from soft plans, so their realized empirical marginals fluctuate even though the underlying plans satisfy the marginals numerically. Neural end-to-end limitations remain visible; no method was tuned using final MMD². This is Stage 2 only: schedule parameters were not updated and no joint schedule-coupling optimization was implemented.
