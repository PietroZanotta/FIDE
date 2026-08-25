# Fixed-feature Galerkin evaluation

## Outcome

**C. GALERKIN BASIS NOT YET PHYSICALLY ADEQUATE**

The deterministic quadratic implementation is numerically sound on the smoke/local frozen artifacts: all tested systems are full numerical rank, coefficient stationarity and range compatibility are near floating-point precision, and the restricted identity `A_G = -2 J_G` holds to between `1.3e-15` and `2.0e-12`. The tested fixed feature spaces are not yet scientifically adequate. The action does not stabilize over the available width, the unchanged production held-out thresholds fail, and the trained/random feature-family actions differ by 45.47%. The required precondition for scientific eta differentiation therefore failed. In accordance with the requested gate, no scientific eta gradient, five-direction artifact check, sensor refinement, or authoritative cross-check was run.

This report uses “Galerkin Full action,” “finite-dimensional Ritz approximation,” and “restricted Full correction.” Exactness of the quadratic solve inside a finite feature space is not evidence of the exact infinite-dimensional Full action.

## Repository isolation audit

The recorded initial `git status --short`, before this milestone was edited, was:

```text
?? ETA_OPTIMIZATION_AUDIT.md
?? experiments/skyrmions_deep_ritz_full/
```

The entire isolated experiment was already untracked, as was the unrelated root audit file. The latter was preserved unchanged. All task-created code, reports, tests, checkpoints, JSON, and NPZ files are under `experiments/skyrmions_deep_ritz_full/`. The Galerkin commands use the existing `require_output_path` guard and a new `outputs/galerkin/` subtree. Existing Deep Ritz code and historical output subtrees were retained. No production experiment, `src/`, or `native/` file was changed; no Pareto sweep was run.

The final `git status --short` isolation check was identical:

```text
?? ETA_OPTIMIZATION_AUDIT.md
?? experiments/skyrmions_deep_ritz_full/
```

Because the isolated directory was already wholly untracked, Git cannot provide per-file deltas within it; the explicit write guards, static forbidden-import/output scan, and recorded task-created file list below are the applicable isolation evidence. `git diff --check -- experiments/skyrmions_deep_ritz_full` reported no whitespace errors.

Files created for this milestone:

- `galerkin.py`
- `galerkin_workflow.py`
- `test_galerkin.py`
- `GALERKIN_EVALUATION.md`
- machine-readable artifacts below `outputs/galerkin/`

Files modified inside the isolated experiment:

- `config.json`
- `run.py`
- `README.md`

## Existing DeepSets architecture and basis source

The experiment-local Ritz architecture implements particle permutation invariance as follows:

1. Each particle contributes four periodic position coordinates, `[sin(2 pi x/L), sin(2 pi y/L), cos(2 pi x/L), cos(2 pi y/L)]`.
2. Five time coordinates, `[t, sin(pi t), cos(pi t), sin(2 pi t), cos(2 pi t)]`, are appended to every particle before the embedding MLP.
3. The particlewise SiLU embeddings are mean-pooled over particles.
4. The same five time coordinates are appended again after pooling.
5. The first scalar-head layer maps that invariant vector to a hidden SiLU representation; the final layer maps it to one scalar.

Thus the architecture naturally exposes the vector just before its scalar output layer. The local smoke checkpoint `outputs/gradient_checks/smoke/theta_center.npz` has width 12. It was generated at the precise incumbent eta0. Its SHA-256 is:

```text
cb827975d35ceb36e16d4c4d988198fc7510da506721f60a69981d1465ccf0dd
```

The primary basis is the frozen 12-coordinate pre-output representation from that checkpoint. The basis ladder uses deterministic prefixes `K=4,8,12`. The feature space was chosen using a fixed pre-existing potential representation and then frozen before continuous eta optimization; it is never retrained per eta. A deterministic frozen-random DeepSets trunk with the same width and layer structure, seed `20261103`, is the control family.

On fixed float64 inputs, the primary maximum particle-permutation discrepancy was `1.1102230246251565e-16`; the random-family discrepancy was `1.0408340855860843e-16`. Repeated values and state Jacobians were bit-identical in the unit test, the workflow value-only determinism discrepancy was zero, and all state gradients were finite. The basis evaluator has no eta input. Eta enters only through reconstruction, information-projection weights, forcing, and weighted centering.

## Discretization and rank/gauge handling

For every scientific time node, the implementation evaluates fixed basis values `Phi[n,k]` and exact JAX state Jacobians `D_X Phi[n,k,p,d]`. With projected weights `w` and centered forcing `h`, it constructs in float64

```text
mean_phi[k] = sum_n w[n] Phi[n,k]
Phi_tilde[n,k] = Phi[n,k] - mean_phi[k]
K[j,k] = sum_n w[n] <D_X Phi[n,j], D_X Phi[n,k]>
f[j] = sum_n w[n] h[n] Phi_tilde[n,j].
```

Raw Gram asymmetry is measured before applying the machine-noise symmetrization `(K + K.T)/2`. Each system is decomposed with `jax.linalg.eigh`. Eigenvalues larger than `1e-10 * lambda_max` are retained, and the minimum-norm unregularized solution is

```text
a = -K^dagger f.
```

No ridge is used for the scientific value. The code records the supported rank, retained-range condition number, and

```text
||(I - K K^dagger) f|| / max(||f||, eps)
||K a + f|| / max(||f||, eps).
```

Potential centering handles the additive gauge without changing state gradients. At every time node it independently evaluates `A_t=a^T K a`, `J_t=0.5 a^T K a+f^T a`, and `|A_t+2J_t|/max(|A_t|,eps)`, followed by time-weighted aggregate values.

## Fixed-eta and basis convergence results

The evaluation used the existing deterministic smoke/local banks and the exact geometry

```text
[0.8954153767761239, 0.20592631632470587,
 1.3343788098383822, 0.8654288352917223,
 0.7508355365766083, 0.5179100329264751,
 1.6423735249784726, 0.5883599695898114]
```

The local banks have three scientific time nodes and 128 train plus 128 independent audit configurations per time. The configured certificate limits were deliberately fixed at the production values: weak `0.12`, energy `0.08`, gauge `1e-9`, and moment rate `0.10`; the relaxed generic smoke certificate values were not used.

### Frozen incumbent-trained latent family

| K | A_G | J_G | identity | stationarity mean/max | worst range | ranks by time | worst cond. | weak | energy | gauge | moment | runtime (s) |
|---:|---:|---:|---:|:---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| 4 | 6.818757054 | -3.409378527 | 1.30e-15 | 8.30e-16 / 1.86e-15 | 7.57e-16 | 4/4/4 | 2.69e3 | 0.542160 | 1.000000 | 2.49e-15 | 2.440179 | 7.62 |
| 8 | 13.014130740 | -6.507065370 | 2.38e-14 | 3.14e-15 / 4.14e-15 | 6.71e-16 | 8/8/8 | 2.66e4 | 0.575187 | 0.999670 | 2.67e-15 | 4.014006 | 2.20 |
| 12 | 15.556546220 | -7.778273110 | 3.48e-14 | 7.31e-15 / 1.15e-14 | 6.79e-16 | 12/12/12 | 2.21e5 | 0.621703 | 1.000000 | 5.14e-15 | 3.702727 | 2.19 |

For `K=12`, the time-node actions are `[0.0085844467, 2.7211037569, 56.7753929200]`; time-node identity residuals are `[2.42e-14, 1.32e-13, 2.55e-14]`. Raw symmetry residuals are `[1.88e-16, 2.81e-16, 2.37e-16]`. The last-step action change is 16.34%, above the configured 10% stability threshold. Weak residuals do not improve and all largest-basis weak, energy, and moment-rate diagnostics exceed production limits.

### Deterministic random latent control

| K | A_G | J_G | identity | stationarity mean/max | worst range | ranks by time | worst cond. | weak | energy | gauge | moment | runtime (s) |
|---:|---:|---:|---:|:---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| 4 | 1.540091299 | -0.770045650 | 2.16e-15 | 6.44e-15 / 1.37e-14 | 6.17e-16 | 4/4/4 | 1.91e2 | 0.276788 | 0.924883 | 5.78e-16 | 0.949634 | 1.14 |
| 8 | 3.434453709 | -1.717226854 | 1.41e-13 | 1.46e-14 / 3.01e-14 | 1.23e-15 | 8/8/8 | 1.19e4 | 0.266917 | 0.892072 | 2.91e-15 | 0.608885 | 1.13 |
| 12 | 8.483167439 | -4.241583719 | 2.00e-12 | 1.01e-13 / 2.11e-13 | 8.50e-16 | 12/12/12 | 2.53e5 | 0.580347 | 0.997809 | 9.06e-14 | 9.751575 | 1.11 |

The random-family last-step action change is 59.51%, and its weak and moment-rate diagnostics deteriorate sharply at `K=12`. At maximum width, its action differs from the trained-family value by 45.47%. This cross-check reinforces, rather than resolves, the lack of basis convergence.

## Projection, ESS, forcing, and held-out physical audits

The train-bank information-projection/forcing audit at eta0 reports maximum projection residual `0.044329897`, minimum ESS fraction `0.015679997`, maximum pre-centering forcing mean `9289.294647`, and maximum covariance condition `6430.739053`. The independent audit bank reports `0.082164509`, `0.038869129`, `28188.038948`, and `35845.348097`, respectively. Both hard forcing audits fail the unchanged configured smoke problem tolerances. These failures are properties of the matching local reconstruction/projection artifacts and occur for every basis size; they are not hidden by the Galerkin centering step.

The primary `K=12` potential evaluated on the held-out bank has action `87.590447458`, maximum weak residual `0.621702790`, maximum Ritz-energy residual `1.0`, maximum gauge residual `5.14e-15`, and maximum moment-rate residual `3.702726663`. Only the gauge diagnostic passes. Exact train-bank quadratic stationarity therefore does not generalize to an adequate held-out weak solution.

## Matching nonlinear Deep Ritz diagnostic

The matching historical smoke/local file is `outputs/gradient_checks/smoke/result.json`. It uses the same local artifact convention and eta0. Its nonlinear Deep Ritz center reports kinetic action `16.119864376`, Ritz objective `-6.827354359`, `-2J=13.654708718`, and energy-identity relative error `0.152926576`; L-BFGS did not converge. The primary `K=12` Galerkin value `15.556546220` is similar in magnitude but has an exact restricted identity and coefficient stationarity. That algebraic improvement does not repair the Galerkin held-out failures. This is a **smoke/local diagnostic comparison only** and is not commensurate with the published production action `0.203454`.

## Eta envelope implementation and gated validation

The implementation solves the coefficients outside the differentiated closure. It then treats `a_eta` as fixed, rebuilds `K_eta` and `f_eta` through reconstruction, implicit information projection, projected weights, forcing, and centering, and evaluates

```text
-2 * sum_t omega_t [0.5 a_t^T K_eta,t a_t + f_eta,t^T a_t].
```

`jax.value_and_grad` is applied only to eta. There is no eigendecomposition or pseudoinverse call inside that closure. Unit tests verify a finite shape-`(8,)` derivative and centered-FD agreement on a deterministic quadratic surrogate. Rank-change detection is also tested.

The scientific eta derivative at eta0 was **NOT RUN** because the required basis-convergence and held-out-physics prerequisite failed. Accordingly, the requested full scientific eta-gradient vector is:

```text
NOT ESTABLISHED — intentionally gated before differentiation
```

The five-direction artifact table is therefore:

| direction | AD derivative | epsilon ladder / FD values | rank center/plus/minus | result |
|---:|:---:|:---:|:---:|:---|
| 0 | NOT RUN | NOT RUN | NOT RUN | prerequisite failed |
| 1 | NOT RUN | NOT RUN | NOT RUN | prerequisite failed |
| 2 | NOT RUN | NOT RUN | NOT RUN | prerequisite failed |
| 3 | NOT RUN | NOT RUN | NOT RUN | prerequisite failed |
| 4 | NOT RUN | NOT RUN | NOT RUN | prerequisite failed |

The command wrote `outputs/galerkin/gradient_checks/smoke/result.json` with `skipped=true`, an empty direction list, and the C classification. This is not a derivative-test failure: the derivative phase was not scientifically admissible. The implemented checker, when admitted, requires constant center/plus/minus ranks, all hard gates, three consecutive sign-agreeing epsilons, two consecutive relative errors at most 0.02, a decreasing-error regime, and at least four of five passing directions. It separately reports whether 0.005 accuracy is observed; it has no “best lucky epsilon” acceptance rule.

## Refinement and authoritative cross-check

Tiny eta refinement: **NOT RUN**. The command is implemented with the strict convergence/gradient prerequisite, three maximum steps, risk hinge, smooth separation penalty, torus wrapping, exact geometry checks, action-decrease backtracking, and per-step rank-aware resolves. The prerequisite failed before any outer update.

Authoritative cross-check: **NOT ESTABLISHED**. It was optional and prohibited by the failed Galerkin gradient prerequisite. No incumbent was replaced and no improvement is claimed.

## Commands and machine-readable evidence

Implemented commands:

```bash
python -m experiments.skyrmions_deep_ritz_full.run --mode galerkin-fixed --smoke-profile
python -m experiments.skyrmions_deep_ritz_full.run --mode galerkin-convergence --smoke-profile
python -m experiments.skyrmions_deep_ritz_full.run --mode galerkin-gradient-check --smoke-profile
python -m experiments.skyrmions_deep_ritz_full.run --mode galerkin-refine --allowance 3 --smoke-profile
```

Each accepts `--frozen-source PATH`. Outputs are separated into `outputs/galerkin/fixed_eta/`, `convergence/`, `gradient_checks/`, `refinement/`, and `artifacts/`. Basis checkpoints and each solve's coefficients, eigenvalues, retained mask, feature values, and state gradients are stored as NPZ alongside JSON diagnostics.

The dedicated suite contains 17 tests covering all 16 requested categories. It passes together with the 12 pre-existing Full-gradient tests. The tests use JAX float64 for scientific arrays; NumPy is confined to checkpoint/JSON host boundaries already used by the experiment.

## Limitations and recommended next step

This evaluation is limited to the deterministic smoke/local artifact set because a complete production frozen artifact set was not supplied. Width 12 provides only three nontrivial prefix points and is too narrow to establish an asymptotic basis regime. The existing local reconstruction/projection forcing gates fail independently of the feature solve. A trained trunk chosen at eta0 can also bias the restricted space toward the incumbent, while the random control's disagreement shows that the current finite space is feature-sensitive.

The next development step is to establish a larger fixed, regularized-but-unridged scientific feature dictionary on a complete frozen artifact set, without fitting it separately per eta. A useful candidate is a deterministic invariant dictionary that combines multiple frozen trunk widths or analytically controlled low-frequency symmetric features, followed by fixed whitening/orthogonalization on an eta-independent base bank. First repair or resolve the local reconstruction/projection forcing audit; then repeat a ladder extending well beyond 12 and require simultaneous action stabilization and held-out weak/moment-rate improvement. Only after those gates pass should the implemented five-direction Galerkin eta-gradient checker be run.
