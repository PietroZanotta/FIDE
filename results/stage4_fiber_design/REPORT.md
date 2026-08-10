# MFSI Stage 4: differentiable moment-fiber design

## Controlled intervention

The endpoint configurations and weights, hidden endpoint q4 gap, angular-sort endpoint coupling, selected reference schedule, six construction times, converged exponential-calibration equations, and eight-function Ritz realization dictionary were fixed. The only optimized object was the rank-three subspace defining the measured observables.

The candidate dictionary contains eleven radial RBF measurements and nests the three hand observables exactly. It contains no angular descriptor or q4. An endpoint-nullspace constraint makes the two fixed weighted endpoint laws exactly equivalent under every candidate, while row-orthonormal coefficients prevent scale or rank collapse. Adaptation, checkpoint selection, and evaluation used disjoint bridge banks.

## Untouched evaluation banks

| fiber | construction objective | correction energy | forcing power | minimum ESS |
|---|---:|---:|---:|---:|
| hand | 0.64081952 | 0.22280602 | 20.630112 | 0.16401613 |
| designed | 0.3235857 | 0.12097731 | 10.13042 | 0.36284271 |

## Primary paired effect

Designed minus hand construction objective: `-0.31723383` (95% interval `-0.65258195` to `0.018114296`).

The construction objective is the established integrated correction energy plus 0.02 times forcing power and the unchanged ESS-floor penalty.

## Interpretation

The experiment does not establish a held-out improvement over the hand-designed fiber.

q4 was accessed only after selection and is reported strictly as a hidden evaluation diagnostic; no rollout, schedule optimization, coupling change, or neural training was performed.
