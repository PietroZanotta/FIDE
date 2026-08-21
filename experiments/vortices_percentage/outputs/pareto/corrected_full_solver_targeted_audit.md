# Corrected physical-q Full solver: targeted audit

**Overall: PASS.**

Operator: `physical q_h` in both the equation and action. Stabilization: conductive-component restriction + one componentwise pin + sparse SuperLU direct solve with equation-preserving PCG fallback; the density floor is preconditioning-only and never enters the operator.

Authoritative equation: `-div(q_h grad psi_h) = q_h h_h`; correction: `delta_h^* = -grad psi_h`.
Declared tolerances: physical Poisson residual `2.000e-07`; moment and energy residuals `1.000e-06`. All residuals and hierarchy gaps are raw and unclipped.

| Design | max Poisson rel. | max component incompat. | incompatible time/trials | unconverged time/trials | max Full moment | max Tangent moment | max hidden null | max orth. | max Pyth. | max raw hierarchy | First failure | Status |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|:---:|
| Law | 7.867e-09 | 3.026e-15 | 0 | 0 | 1.800e-10 | 9.177e-15 | 1.800e-10 | 3.365e-11 | 6.727e-11 | -1.757e-01 | None | PASS |
| Tangent | 2.415e-09 | 2.682e-15 | 0 | 0 | 1.412e-10 | 7.194e-15 | 1.412e-10 | 1.154e-11 | 2.308e-11 | -1.415e-01 | None | PASS |
| Full | 2.684e-09 | 3.896e-15 | 0 | 0 | 1.576e-10 | 1.275e-14 | 1.576e-10 | 1.943e-11 | 3.886e-11 | -1.786e-01 | None | PASS |

Second-stage full rescore authorized: **False**.
Experiment-local eligibility: **True**.
Paired two-experiment gate: **False**; blocking experiments: `['toy_example_percentage']`.
The all-allowance rescore and its conditional outputs must not be produced when this paired gate is false.

## Interpretation

The targeted common-discretization decomposition is numerically resolved, so `Gamma_h = A_hid,h / A_full,h` has a genuine discrete geometric interpretation for these targeted candidates.

All targeted physical sources are compatible on every conductive component.

No saved candidate geometry or frozen bank changed: **True**.
Because the paired gate failed, maximum relative Full-action changes and candidate-ranking changes were not computed, and no optimization stage or optimization output was modified.
