# MFSI Stage 3: rollout-aware differentiable correction adaptation

## Isolated intervention

The completed random-continuous-time invariant MLP, its weights, the selected schedule, endpoint construction, reference velocity, 24-step Heun solver, evaluation times, and final radial-plus-q4 MMD were frozen. The only learned object was a three-parameter bounded harmonic modulation of the frozen conservative correction. Alpha=0 exactly reproduces the frozen scalar Ritz gate.

Adaptation used interior MMD² on the three measured Phi observables, with the established weighted median-RBF kernel. q4 and the final radial-plus-q4 law metric were untouched until evaluation. Forty Adam steps at learning rate 0.04 were run on an adaptation bank; candidates at steps 0,5,...,40 were selected once on an independent selection bank and evaluated on a third untouched bank.

## Held-out evaluation

| method | interior law MMD² | max moment error | interior Phi MMD² | q4 change |
|---|---:|---:|---:|---:|
| raw SI | 0.019143568 | 0.033885017 | 0.035800368 | 0.62379346 |
| tangent | 0.012607731 | 0.0021493784 | 0.010818921 | 0.57539274 |
| frozen neural | 0.028044937 | 0.039574919 | 0.047871879 | 0.62755404 |
| rollout-adapted | 0.020449396 | 0.034001298 | 0.034918825 | 0.62223162 |

## Primary paired effects

Adapted minus frozen interior law MMD²: `-0.0075955408` (95% interval `-0.0075955408` to `-0.0075955408`).

Adapted minus tangent interior law MMD²: `0.0078416656` (95% interval `0.0078416656` to `0.0078416656`).

## Validation

Directional gradient relative error: `1.604e-09`. Functional-Heun maximum parity error: `4.441e-16`. MMD static-shape parity error: `0.000e+00`. Neural parameter hashes were unchanged and all bank-role fingerprints were distinct.

## Interpretation

The paired interval supports a held-out improvement over the frozen neural correction.

It does not establish an improvement over tangent.
