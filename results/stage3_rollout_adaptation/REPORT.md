# MFSI Stage 3: rollout-aware differentiable correction adaptation

## Isolated intervention

The completed random-continuous-time invariant MLP, its weights, the selected schedule, endpoint construction, reference velocity, 24-step Heun solver, evaluation times, and final radial-plus-q4 MMD were frozen. The only learned object was a three-parameter bounded harmonic modulation of the frozen conservative correction. Alpha=0 exactly reproduces the frozen scalar Ritz gate.

Adaptation used interior MMD² on the three measured Phi observables, with the established weighted median-RBF kernel. q4 and the final radial-plus-q4 law metric were untouched until evaluation. Forty Adam steps at learning rate 0.04 were run on an adaptation bank; candidates at steps 0,5,...,40 were selected once on an independent selection bank and evaluated on a third untouched bank.

## Held-out evaluation

| method | interior law MMD² | max moment error | interior Phi MMD² | q4 change |
|---|---:|---:|---:|---:|
| raw SI | 0.019149051 | 0.026830976 | 0.033508285 | 0.60280942 |
| tangent | 0.0098037629 | 0.0016737901 | 0.0098740167 | 0.56233342 |
| frozen neural | 0.019680657 | 0.024931225 | 0.03155668 | 0.6058774 |
| rollout-adapted | 0.016027606 | 0.023246404 | 0.024769247 | 0.60567336 |

## Primary paired effects

Adapted minus frozen interior law MMD²: `-0.0036530507` (95% interval `-0.0073473137` to `4.1212258e-05`).

Adapted minus tangent interior law MMD²: `0.0062238436` (95% interval `-0.001275662` to `0.013723349`).

## Validation

Directional gradient relative error: `1.782e-09`. Functional-Heun maximum parity error: `8.882e-16`. MMD static-shape parity error: `0.000e+00`. Neural parameter hashes were unchanged and all bank-role fingerprints were distinct.

## Interpretation

The experiment does not establish a held-out improvement over the frozen neural correction.

It does not establish an improvement over tangent.
