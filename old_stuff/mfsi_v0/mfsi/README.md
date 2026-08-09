# MFSI Example A — two-component JAX prototype with proper MGD benchmark

This is the deliberately small **pre-Tesseract** implementation of

```text
bridge parameters xi
      |
      v
+----------------------+      +---------------------------+
|  ReferenceTransport  | ---> |  MomentFiberRealizer      |
|  future Tesseract 1  |      |  future Tesseract 2       |
|  ordinary JAX AD     |      |  implicit solver VJP      |
+----------------------+      +---------------------------+
      |                             |
      | q_ref, log q_ref, u         | q_fib, lambda, h,
      |                             | delta v, v*, ESS, E_corr
      +-----------------------------+
                    |
                    v
           outer JAX objective
                    |
              d L / d xi
```

There is **no Tesseract dependency yet**. JAX remains the numerical backend so the component contracts and derivative semantics can be validated before adding Tesseract wrappers.

## Code map

### `mfsi_components.py`
The two future Tesseract boundaries and the MFSI mathematics.

- `reference_transport(...)` is future **Tesseract 1**.
- `moment_fiber_realizer(...)` is future **Tesseract 2**.
- Calibration supports `unrolled`, `implicit`, and `stop` differentiation modes.
- The `implicit` mode uses the moment-covariance solve for the custom VJP rather than differentiating through Newton iterations.

### `validate_pipeline.py`
Oracle validation of Example A: projected moments, `lambda_dot`, forcing, continuity equation, moment-tangent ablation, hidden fourth moment, and a small Deep-Ritz check.

```bash
python validate_pipeline.py
```

### `ablate_and_benchmark.py`
Tests whether end-to-end differentiation is useful for upstream bridge design and benchmarks unrolled Newton vs implicit VJP vs stop-gradient.

```bash
python ablate_and_benchmark.py
```

### `mgd.py`
A **real interacting-particle implementation of Moment Guided Diffusion**, not the analytic stationary-Gaussian substitution used in the previous benchmark.

It contains two implementations:

1. `simulate_mgd_predictor_corrector(...)`: the primary baseline, following MGD Sec. 3.2, Eqs. (18)-(21). It computes empirical Gram matrices, solves for the moment-transport coefficient, injects Brownian noise, then applies the paper's linearized moment corrector.
2. `simulate_mgd_theorem_euler(...)`: independent Euler-Maruyama discretization of Theorem 3.1 using `G theta = E[Delta phi]`. This exists only as a cross-check.

The Gram solve follows the normalization/ridge procedure described in MGD Appendix D.1: normalize the Gram diagonal and add a `1e-7` ridge.

**Important implementation detail.** The arXiv v1 equations are internally clear: Eq. (19) gives

```text
X_{t+h} = Y - h sigma^2 theta^T grad phi(Y)
```

and Eq. (21) solves

```text
h sigma^2 G' theta = E[phi(Y)] - m_{t+h}.
```

The code follows those equations. They are also consistent with Theorem 3.1 and reduce to the expected OU drift in the quadratic case.

### `validate_mgd.py`
Validates the actual MGD simulator independently of MFSI.

For Example A, `phi=(x,x^2)`, the MGD base is `N(0,1)`, and the target law also has mean zero and variance one. With MGD's variance-preserving stochastic interpolant, the prescribed moment path is therefore exactly

```text
m_t = (0, 1).
```

MGD Appendix E provides an unusually strong oracle: for Gaussian base and quadratic/linear moments, the exact MGD law is Gaussian with the prescribed mean/covariance for **every sigma**. Here that means `N(0,1)` for all time. The simulator is compared against this oracle rather than assuming it.

```bash
python validate_mgd.py
```

Reference predictor/corrector checks with 8192 particles:

| sigma^2 | steps | max W1 to own Gaussian oracle | max second-moment error |
|---:|---:|---:|---:|
| 0.25 | 1000 | `1.06e-2` | `7.54e-7` |
| 1.0 | 1000 | `1.52e-2` | `2.57e-6` |
| 2.5 | 1500 | `1.19e-2` | `1.27e-5` |

The linearized corrector reduces the predictor moment error by a median factor of roughly `1e3`-`4e3` in these runs. The direct Theorem-3.1 Euler cross-check remains close in distribution to the Gaussian oracle but has larger finite-replica empirical moment fluctuations, as expected because it does not project the particle moments at each time step.

### `benchmark_methods.py`
The full-law comparison now uses the **proper MGD predictor/corrector**.

```bash
python benchmark_methods.py
```

Methods:

1. `raw_si`: uncorrected reference probability flow;
2. `moment_tangent`: self-consistent interacting moment-rate correction;
3. `mgd`: actual stochastic interacting-particle MGD;
4. `mfsi`: I-projected law with exact 1D weighted-Poisson realization.

All methods start from the same 8192-point Gaussian quantile ensemble. Raw SI, moment tangent and MFSI use the common 240-step Heun integration. MGD uses its faithful stochastic predictor/corrector with `sigma=1` and 1000 steps. We do **not** force MGD onto the deterministic 240-step budget, because that would make the implementation less faithful. Step counts and timings are reported separately.

At `t=0,.1,...,1` every generated law is compared against the exact MFSI projected marginal with W1/W2/KS and the held-out fourth moment. MGD is additionally compared against its own Appendix-E Gaussian oracle; this cleanly separates implementation correctness from task mismatch.

Current reference result:

| method | mean interior W1 to projected target | max second-moment error | fourth-moment RMSE | endpoint W1 |
|---|---:|---:|---:|---:|
| raw SI | `1.762e-1` | `5.000e-1` | `1.551` | `4.19e-5` |
| moment tangent | `1.041e-2` | `1.00e-8` | `1.089e-1` | `8.59e-3` |
| proper MGD | `4.841e-2` | `2.89e-6` | `5.385e-1` | `9.98e-2` |
| MFSI | `4.687e-5` | `3.92e-5` | `2.985e-3` | `4.16e-5` |

The primary MGD run has maximum W1 `1.16e-2` to its **own** exact Gaussian oracle, so the roughly `4.8e-2` error to the MFSI path is not caused by replacing MGD with an analytic shortcut. It is the actual behavior of MGD on this quadratic moment specification.

The strongest comparison remains moment tangent vs MFSI: the tangent method preserves the measured second moment to about `1e-8` but remains about `1e-2` away from the prescribed projected law, whereas MFSI tracks that law at approximately the deterministic integration/cubature floor (`~5e-5`).

Do **not** interpret this as a blanket claim that MFSI "beats MGD." MGD is designed to sample the maximum-entropy law determined by its selected moments. In this Example A, those moments identify the standard Gaussian as the maximum-entropy model, so the proper MGD implementation is doing what its theory says it should. The benchmark establishes that **moment/MaxEnt generation and within-fiber full-law transport are different tasks**.

## End-to-end differentiation result

The differentiability ablation remains unchanged by adding MGD:

- implicit and unrolled gradients for correction energy agree to about `1e-11` relative error;
- the implicit gradient agrees with finite differences to about `3e-9`;
- stopping the gradient through `lambda*` changes the correction-energy gradient by about `20.8%`;
- optimizing the reference bridge with the correct end-to-end gradient reaches correction energy about `1.83e-4`, versus about `1.81e-3` with stop-gradient, while minimum ESS fraction improves from roughly `0.649` to `0.9999`.

For projection distortion alone, stop-gradient and full differentiation agree, as required by the envelope theorem. This is the negative control showing that solver-aware differentiation matters for the right downstream objective rather than automatically helping every objective.

## Install

```bash
pip install "jax[cpu]" numpy matplotlib
```

64-bit JAX is enabled because this package is an oracle/quadrature validation.

## Future Tesseract conversion

Preserve these coarse interfaces:

- **Tesseract 1 — `ReferenceTransport`:** JAX-native differentiation.
- **Tesseract 2 — `MomentFiberRealizer`:** solver-aware implicit VJP for calibration; later a nontrivial Deep-Ritz/adjoint realization may live here.
- **Outer JAX/Tesseract-JAX program:** optimize bridge parameters using correction energy plus an ESS/overlap penalty.

Do not split the tiny `2x2` calibration into its own Tesseract. Example A validates interfaces and cross-component gradient semantics; a many-body experiment is where two deployed services should be benchmarked for actual runtime value.

## Outputs

Main outputs are written to `results/`:

- `validation.png`, `validation_metrics.json`
- `ablation_benchmark.png`, `ablation_benchmark_metrics.json`
- `mgd_validation.png`, `mgd_validation_metrics.json`
- `method_benchmark.png`, `method_benchmark_metrics.json`, `method_benchmark_summary.csv`
