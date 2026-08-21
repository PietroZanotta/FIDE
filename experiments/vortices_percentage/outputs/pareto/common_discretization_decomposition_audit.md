# Common-discretization decomposition audit

**Overall: FAIL.** Absolute tolerance: `1.000e-06`.

The Tangent correction is the minimum-norm solution in the same physical-density raster Hilbert space used by the authoritative Full action. The particle Tangent metric is not used.

## Summary

- Maximum Full moment-rate residual: `0.0163264953989`
- Maximum Tangent moment-rate residual: `2.45427568961e-14`
- Maximum hidden-nullspace residual: `0.0163264953989`
- Maximum absolute orthogonality residual: `0.0199222346691`
- Maximum absolute Pythagorean residual: `0.0398444693381`
- Maximum raw hierarchy violation (`A_tan,h - A_full,h`): `-0.138856148563`
- Maximum Full residual after subtracting floor/gauge stabilization: `3.32801017212e-08`
- Violations (aggregate / trial / time-trial): `18 / 432 / 9072`
- First failing condition: `full_moment`
- Genuine hidden-fraction interpretation supported: `False`
- Saved geometries unchanged: `True`
- Frozen banks unchanged: `True`

The first failed implication is Full moment-rate feasibility. The authoritative linear solve uses `q_h + q_floor` for stability (`operator_floor_rel=2e-05`), while its scientific action—and this requested Hilbert-space audit—uses physical `q_h`. Full action reproduction is at roundoff and the raster Tangent constraint solve is at machine precision, so the downstream nullspace, orthogonality, and Pythagorean failures originate before the projection, at physical-`q_h` Full feasibility.

## Candidate table

| Allowance | Design | A_tan,h | A_full,h | A_hid,h | A_hid,h/A_full,h | max Full feas. | max Tan feas. | max null | max orth. | max Pyth. | max raw hierarchy | Status |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.5% | Law | 0.44589018 | 3.4586343 | 3.0145474 | 0.87160051 | 1.633e-02 | 9.684e-15 | 1.633e-02 | 1.536e-02 | 3.072e-02 | -1.730e-01 | FAIL |
| 0.5% | Tangent | 0.43893766 | 3.8575657 | 3.4203703 | 0.88666547 | 7.521e-03 | 7.296e-15 | 7.521e-03 | 1.843e-02 | 3.686e-02 | -1.741e-01 | FAIL |
| 0.5% | Full | 0.44131066 | 3.176462 | 2.7369109 | 0.8616224 | 6.743e-03 | 1.216e-14 | 6.743e-03 | 1.353e-02 | 2.707e-02 | -1.840e-01 | FAIL |
| 1% | Law | 0.44589018 | 3.4586343 | 3.0145474 | 0.87160051 | 1.633e-02 | 9.684e-15 | 1.633e-02 | 1.536e-02 | 3.072e-02 | -1.730e-01 | FAIL |
| 1% | Tangent | 0.42901444 | 3.5792494 | 3.1519571 | 0.88061958 | 7.101e-03 | 2.454e-14 | 7.101e-03 | 1.992e-02 | 3.984e-02 | -1.548e-01 | FAIL |
| 1% | Full | 0.43577341 | 2.8258184 | 2.3916896 | 0.84637057 | 4.708e-03 | 1.268e-14 | 4.708e-03 | 1.137e-02 | 2.274e-02 | -1.757e-01 | FAIL |
| 2% | Law | 0.44589018 | 3.4586343 | 3.0145474 | 0.87160051 | 1.633e-02 | 9.684e-15 | 1.633e-02 | 1.536e-02 | 3.072e-02 | -1.730e-01 | FAIL |
| 2% | Tangent | 0.42901444 | 3.5792494 | 3.1519571 | 0.88061958 | 7.101e-03 | 2.454e-14 | 7.101e-03 | 1.992e-02 | 3.984e-02 | -1.548e-01 | FAIL |
| 2% | Full | 0.43577341 | 2.8258184 | 2.3916896 | 0.84637057 | 4.708e-03 | 1.268e-14 | 4.708e-03 | 1.137e-02 | 2.274e-02 | -1.757e-01 | FAIL |
| 3% | Law | 0.44589018 | 3.4586343 | 3.0145474 | 0.87160051 | 1.633e-02 | 9.684e-15 | 1.633e-02 | 1.536e-02 | 3.072e-02 | -1.730e-01 | FAIL |
| 3% | Tangent | 0.42215859 | 3.0610068 | 2.6404617 | 0.86261215 | 6.441e-03 | 7.427e-15 | 6.441e-03 | 1.965e-02 | 3.929e-02 | -1.389e-01 | FAIL |
| 3% | Full | 0.43577341 | 2.8258184 | 2.3916896 | 0.84637057 | 4.708e-03 | 1.268e-14 | 4.708e-03 | 1.137e-02 | 2.274e-02 | -1.757e-01 | FAIL |
| 4% | Law | 0.44589018 | 3.4586343 | 3.0145474 | 0.87160051 | 1.633e-02 | 9.684e-15 | 1.633e-02 | 1.536e-02 | 3.072e-02 | -1.730e-01 | FAIL |
| 4% | Tangent | 0.38754841 | 2.4719826 | 2.0857187 | 0.84374325 | 4.133e-03 | 8.258e-15 | 4.133e-03 | 8.202e-03 | 1.640e-02 | -1.649e-01 | FAIL |
| 4% | Full | 0.38754841 | 2.4719826 | 2.0857187 | 0.84374325 | 4.133e-03 | 8.258e-15 | 4.133e-03 | 8.202e-03 | 1.640e-02 | -1.649e-01 | FAIL |
| 5% | Law | 0.44589018 | 3.4586343 | 3.0145474 | 0.87160051 | 1.633e-02 | 9.684e-15 | 1.633e-02 | 1.536e-02 | 3.072e-02 | -1.730e-01 | FAIL |
| 5% | Tangent | 0.38696444 | 2.4403059 | 2.0545424 | 0.84192 | 3.366e-03 | 1.236e-14 | 3.366e-03 | 4.603e-03 | 9.207e-03 | -1.506e-01 | FAIL |
| 5% | Full | 0.38710338 | 2.4301653 | 2.0442527 | 0.84119901 | 3.435e-03 | 1.061e-14 | 3.435e-03 | 4.865e-03 | 9.731e-03 | -1.460e-01 | FAIL |

## Definitions

`r_h = -sum(phi * q_h * h_h) dx^2`; `L_h(-grad z)_j = -<grad phi_j, grad z>_{q_h}`. The Full action, Gram matrix, projection, hidden action, and all inner products use the physical `q_h` edge weights and cell quadrature from the authoritative Full evaluator. Residuals and hierarchy gaps are raw and are never clipped.
