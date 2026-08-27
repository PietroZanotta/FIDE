# Continuous Full skyrmion sensor optimization

This is an isolated experimental variant of `skyrmions_deep_ritz`. For all new
continuous selection and validation work, its Full solver is the fixed-feature
Galerkin discretization of the weighted-Poisson weak problem. It imports shared
numerical primitives only from `src/mfsi`; it does not import or write to the
production skyrmion experiment.

The Galerkin method solves the same weighted-Poisson weak problem directly in a
finite, fixed, eta-independent, permutation-invariant trial space. The
historical nonlinear fixed-theta Deep Ritz envelope route is not validated and
is no longer used for continuous selection. Historical fixed-design Deep Ritz
code, checkpoints, results, and reports remain available only for comparison;
they do not rank candidates, certify ordering, select a winner, or validate a
winner in the Galerkin-only study.

The completed GPU-first Galerkin-only 3% study is documented in
`GALERKIN_ONLY_3PCT_EVALUATION.md`. It selected K=280 as a practical finite
discretization, froze a certified selection winner with a `0.2587%` lower
Galerkin action, and then performed one sealed validation. Validation Galerkin
action was also lower, but the frozen winner failed the unchanged validation
risk ceiling. The final classification is **B. GALERKIN-ONLY 3% SELECTION
IMPROVED, VALIDATION DID NOT**. No Deep Ritz result entered that decision.

Run the bounded Galerkin-only workflow in order:

```bash
python -m experiments.skyrmions_deep_ritz_full.galerkin_only_run --mode benchmark
python -m experiments.skyrmions_deep_ritz_full.galerkin_only_run --mode convergence
python -m experiments.skyrmions_deep_ritz_full.galerkin_only_run --mode profile-selected-K
python -m experiments.skyrmions_deep_ritz_full.galerkin_only_run --mode optimize-3pct
python -m experiments.skyrmions_deep_ritz_full.galerkin_only_run --mode validate-3pct \
  --selection-result experiments/skyrmions_deep_ritz_full/outputs/galerkin_only_3pct/selection/result.json
python -m unittest experiments.skyrmions_deep_ritz_full.test_galerkin_only -v
```

These modes prefer CUDA with JAX float64, fall back cleanly to CPU, and write
only below `outputs/galerkin_only_3pct/`.

It also contains two separate deterministic fixed-feature Galerkin studies. The
historical smoke study used frozen DeepSets coordinates and is documented in
`GALERKIN_EVALUATION.md`. The validated production study uses a fixed analytic
hybrid Fourier/pairwise dictionary and is documented in
`PRODUCTION_GALERKIN_EVALUATION.md`. Both keep the nonlinear Deep Ritz route and
historical outputs intact.

The accelerated 3% continuation is documented in
`FAST_PRODUCTION_3PCT_EVALUATION.md`. Its cached K=160 value+gradient path is
numerically equivalent and 4.06x faster in steady CPU measurements. Gradient
convergence and nearby-point audits passed. Four-start refinement found no
further authoritative improvement beyond the prior tiny update; that frozen
winner then failed disjoint validation risk and energy gates. The validation
reversal is retained and was not used to retune selection.

The follow-up authoritative GPU checkpoint is documented in
`AUTHORITATIVE_GPU_ACCELERATION.md`. A fresh fixed-design solve was `2.315x`
faster on the available NVIDIA GPU, and fixed CPU/GPU checkpoint evaluations
agreed to about `1e-15`. However, paired CPU and GPU optimization reversed the
eta0/tiny-update ordering, while no solve declared L-BFGS convergence. GPU
evaluation is therefore numerically validated, but a single optimized rescore
is not a stable candidate-ordering oracle. The new resumable common-restart gate
caches each initialization separately and fails closed on mixed ordering.

That gate has now run and is documented in
`AUTHORITATIVE_STABILITY_EVALUATION.md`. The hash-reused warm pair was valid and
favored eta0 by `0.00337637`. A new common-seed pair also had eta0's raw action
lower, but both members failed the unchanged held-out energy certificate and
neither optimizer converged. The decision is `indeterminate`; further eta
refinement is blocked pending a stationary, restart-robust inner-solver study.

Run its gates in order:

```bash
python -m experiments.skyrmions_deep_ritz_full.run --mode benchmark-production-galerkin --frozen-source PATH
python -m experiments.skyrmions_deep_ritz_full.run --mode production-gradient-convergence --frozen-source PATH
python -m experiments.skyrmions_deep_ritz_full.run --mode production-local-gradient-audit --frozen-source PATH
python -m experiments.skyrmions_deep_ritz_full.run --mode production-refine-3pct --frozen-source PATH
python -m experiments.skyrmions_deep_ritz_full.run --mode production-multistart-3pct --frozen-source PATH
python -m experiments.skyrmions_deep_ritz_full.run --mode production-authoritative-3pct --frozen-source PATH
python -m experiments.skyrmions_deep_ritz_full.run --mode production-validate-3pct --frozen-source PATH --input-result experiments/skyrmions_deep_ritz_full/outputs/fast_production_3pct/selection/result.json
python -m experiments.skyrmions_deep_ritz_full.run --mode benchmark-authoritative-solver --frozen-source PATH
python -m experiments.skyrmions_deep_ritz_full.run --mode production-authoritative-stability --restart-count 3 --frozen-source PATH
```

The last two commands require a JAX environment with GPU access to reproduce
the reported acceleration. The stability command reuses compatible completed
restarts by hashes and writes a partial `result.json` after every pair.

## Status

**ENVELOPE GRADIENT NOT YET VALIDATED.**

That warning applies to the nonlinear fixed-theta Deep Ritz derivative described
below.  The independent production fixed-feature route has now passed its two
scientific gates and its eta derivative is validated: the real frozen 3% artifact
set was reproduced, the nested `K=20,...,160` hybrid Galerkin ladder passed the
production held-out certificates, and all five production AD/FD directions
passed.  Its authoritative record is `PRODUCTION_GALERKIN_EVALUATION.md`; the
classification is **A. PRODUCTION GALERKIN SOLVER AND ETA GRADIENT VALIDATED**.
This does not validate the nonlinear derivative. The subsequent fresh
fixed-design cross-check passed all hard certificates and found the tiny update's
Deep Ritz action lower by `0.00445154` at the declared `1e-6` comparison
tolerance; no production incumbent was modified or replaced.

Methodological status from this point forward:

- historical nonlinear Deep Ritz envelope: not validated and retired from
  continuous selection;
- production fixed-feature Galerkin route: validated and the primary Full
  solver for this isolated study;
- historical fixed-design Deep Ritz results: retained for history only and not
  an optimization or validation oracle.

The earlier permissive smoke result is retained below as historical evidence only. It did not show an epsilon-convergence regime and is not an acceptance test. The new rigorous command uses a mandatory stationary-center gate, deterministic full-bank theta polishing, optimized-value finite differences, consecutive-epsilon criteria, five fixed directions, and a conditional kinetic-action stage.

The rigorous run stopped at its mandatory center prerequisite:

- projection, ESS, forcing compatibility, geometry, and repeated fixed-input determinism passed;
- the stored-warm-start and fresh-initialization solves both failed parameter stationarity and `A ~= -2J`;
- stronger theta minimization showed runaway/overfitting behavior on the empirical smoke Ritz bank;
- therefore local-continuity, directional Stage 1, kinetic-action Stage 2, and all sensor updates were not run;
- production optimization, production validation, incumbent replacement, and a Pareto sweep remain prohibited.

The Galerkin smoke/local evaluation is classified **C. GALERKIN BASIS NOT YET PHYSICALLY ADEQUATE**. Its finite-dimensional linear algebra is accurate to roughly `1e-12` or better, but the action has not stabilized over `K=4,8,12`, the unchanged production held-out certificate thresholds fail, and the frozen-random family materially disagrees with the trained family. Consequently the scientific Galerkin eta gradient and refinement were intentionally not run.

## Galerkin commands

```bash
python -m experiments.skyrmions_deep_ritz_full.run --mode galerkin-fixed --smoke-profile
python -m experiments.skyrmions_deep_ritz_full.run --mode galerkin-convergence --smoke-profile
python -m experiments.skyrmions_deep_ritz_full.run --mode galerkin-gradient-check --smoke-profile
python -m experiments.skyrmions_deep_ritz_full.run --mode galerkin-refine --allowance 3 --smoke-profile
python -m unittest experiments.skyrmions_deep_ritz_full.test_galerkin -v
```

The last two commands enforce prerequisites. A failed physical/convergence gate writes a machine-readable result and exits nonzero without evaluating the scientific eta derivative or taking an outer step. Complete frozen artifacts may be supplied with `--frozen-source PATH`. All Galerkin artifacts are confined to `outputs/galerkin/`; no command launches a Pareto sweep.

For the real frozen production problem, run the explicitly gated commands in
order (the cross-check is warranted only when refinement reports an eligible
decrease):

```bash
python -m experiments.skyrmions_deep_ritz_full.run --mode production-preflight --frozen-source PATH
python -m experiments.skyrmions_deep_ritz_full.run --mode reproduce-production --frozen-source PATH
python -m experiments.skyrmions_deep_ritz_full.run --mode galerkin-production-convergence --frozen-source PATH
python -m experiments.skyrmions_deep_ritz_full.run --mode galerkin-production-gradient-check --frozen-source PATH
python -m experiments.skyrmions_deep_ritz_full.run --mode galerkin-production-refine --allowance 3 --frozen-source PATH
python -m experiments.skyrmions_deep_ritz_full.run --mode production-authoritative-crosscheck --allowance 3 --frozen-source PATH
python -m unittest experiments.skyrmions_deep_ritz_full.test_production_galerkin -v
```

These modes write only below `outputs/production_galerkin/` and never launch a
Pareto sweep.  They require a complete frozen artifact source; missing artifacts
are a hard stop rather than permission to generate substitutes.

## Implemented computational graph

```text
eta (8 sensor coordinates; only differentiated input)
  -> periodic sensor centers
  -> Phi_eta(X_truth), fixed CRN acquisition prefix, fixed noise draw
  -> anchored JAX spline coefficients c_eta(t), cdot_eta(t)
  -> Phi_eta(X_ref), D_X Phi_eta(X_ref) u_ref
  -> implicit I-projection lambda_eta(t) (custom VJP; Newton is not unrolled)
  -> projected weights, moments, covariance, ESS
  -> lambda_dot_eta(t), centered continuity forcing h_eta
  -> centered J(theta_fixed, eta)
  -> -2 J(theta_fixed, eta)
  -> jax.value_and_grad with respect to eta only
```

Reference configurations, velocities, quadrature weights, truth/common-random-number indices, detector-noise draws, and `theta_fixed` are closed-over constants during one eta derivative. Ritz training is never differentiated. The exact implementation is `envelope_full_value_and_grad` in `full_gradient.py`; it evaluates

```text
-2 * partial_eta J(theta_fixed, eta).
```

`ritz_objective_eta` and `full_energy` separately expose the centered Ritz functional and kinetic Full action. `envelope_diagnostics` reports

```text
abs(A + 2 J) / max(abs(A), 1e-12)
```

along with projection residual, relative ESS, forcing mean, and covariance conditioning.

## Inner Ritz tracking and outer refinement

The new config has three experiment-local inner schedules:

- `full`: the copied high-accuracy schedule, used for final selection and validation solves;
- `track`: warm-started deterministic full-bank Adam plus shorter L-BFGS tracking;
- `smoke`: a small deterministic schedule for executable checks.

Every outer iteration reoptimizes theta at the current eta. A poor energy identity triggers an additional `full` solve. Configurable periodic polishing uses `polish_every`. Eta is updated with deterministic Adam, wrapped onto the torus, and backtracked until the exact minimum-separation check passes.

The differentiable objective is `-2 J` plus a normalized squared risk-hinge penalty and a smooth separation penalty. The exact risk ceiling, exact geometry, projection/ESS/forcing checks, and unchanged Deep Ritz certificate checks remain separate hard gates. The law risk is re-evaluated at the fixed law eta on the same bank as each candidate; the number in `config.json` remains a record of the production anchor.

The outer optimizer cannot run unless the reoptimized directional gradient check in that same invocation passes.

## Commands

Run from the repository root in an environment containing the project dependencies:

```bash
python -m experiments.skyrmions_deep_ritz_full.run --mode smoke --allowance 3
python -m experiments.skyrmions_deep_ritz_full.run --mode gradient-check --smoke-profile
python -m experiments.skyrmions_deep_ritz_full.run --mode rigorous-gradient-check --smoke-profile
python -m experiments.skyrmions_deep_ritz_full.run --mode gradient-check
python -m experiments.skyrmions_deep_ritz_full.run --mode optimize --allowance 3
python -m experiments.skyrmions_deep_ritz_full.run --mode certify --allowance 3 --input-result experiments/skyrmions_deep_ritz_full/outputs/selection/result.json
python -m unittest experiments.skyrmions_deep_ritz_full.test_full_gradient -v
```

`--mode smoke` includes the gradient gate, two outer steps, fresh selection audit, and independent validation audit. A nonzero exit after writing `result.json` means the hard scientific gate rejected the result; it is not treated as a successful certified design.

For the current milestone, use `--mode rigorous-gradient-check`. It never calls the eta optimizer. The `optimize` command remains implemented but should not be run until the rigorous Stage-1 check passes.

To reuse a complete frozen artifact set, pass `--frozen-source PATH`. The source is read-only and copied into this experiment's local artifact directory before use. All `--output-dir` values are resolved and rejected unless they are below `experiments/skyrmions_deep_ritz_full/outputs/`.

## Rigorous fixed-eta result

The rigorous result is in `outputs/gradient_checks/rigorous_smoke/summary.json`, with complete center details in `center.json`. It used the same eta shown in the historical section below.

### Center inner solve

The stored warm-start branch was less nonstationary than the fresh branch and was retained only for diagnostic reporting:

| metric | warm branch | fresh branch | required |
|---|---:|---:|---:|
| J | -26.914524204 | -5169.613120427 | — |
| V = -2J | 53.829048408 | 10339.226240854 | — |
| kinetic A | 48.914028372 | 15117.381057188 | — |
| abs(A + 2J) | 4.915020036 | 4778.154816334 | — |
| relative identity error | 0.100482831 | 0.316070277 | <= 0.01 |
| raw parameter-gradient norm | 628.299036 | 1623966.776135 | materially small |
| RMS parameter gradient | 33.632076 | 86928.948525 | <= 0.002 |
| max absolute parameter gradient | 364.072903 | 1069389.316516 | <= 0.02 |
| full-bank L-BFGS iterations | 2000 | 2000 | convergence required |
| optimizer converged | no | no | yes |

For the selected warm branch, the held-out weak residual was `0.359196`, Ritz-energy residual `1.0`, gauge residual `4.79e-14`, and moment-rate residual `31.140598`. These are reported diagnostics, not a passed certificate.

The train forcing audit passed with projection residual `6.69e-13`, ESS fraction `0.357204`, forcing mean `1.71e-10`, and covariance condition `19.0634`. The independent audit-bank counterparts were `6.10e-11`, `0.273102`, `3.97e-9`, and `15.4833`. Thus projection/forcing failure is not the cause of the center stationarity failure.

The validation-only SciPy L-BFGS boundary receives theta vectors through NumPy, but every float64 objective and gradient is evaluated by JAX on the complete frozen bank. It is outside the eta graph and does not differentiate through optimization.

### Envelope AD and determinism

The fixed-theta gradient evaluated at the least nonstationary warm branch was:

```text
[-1318.046723164419, -2133.736177391800,
 -1595.558916093833,  -856.312637657993,
   749.113174427436,  1597.660165968458,
    80.558254352522,  -365.441386418217]
```

This vector is finite but is **not validated**, because theta is not stationary. Two identical evaluations had exactly zero objective, envelope-value, and maximum gradient difference at the reported precision; the configured determinism tolerance was `1e-12`.

### Local continuity and directional checks

No `Vplus/Vminus` continuity table was produced and no directions were evaluated. This is intentional: the center failed before `eta0 +/- epsilon*v` solves were scientifically interpretable. The command records `direction_count=0` and the exact prerequisite failure in `summary.json`.

Had the center passed, the configured ladder would have been `1e-2, 5e-3, 3e-3, 1e-3, 5e-4, 3e-4, 1e-4` in five deterministic admissible directions. The strict gate requires, per direction, three consecutive FD signs agreeing with AD, two consecutive relative errors at or below `0.05`, stationary plus/minus solutions, valid forcing/geometry diagnostics, and optimized values approaching `V0`. Kinetic-action finite differences are disabled unless all Stage-1 rules pass.

### Basin sensitivity and conclusion

The two deterministic center branches were materially different: `Vwarm=53.829048408`, `Vfresh=10339.226240854`, relative difference `0.994794`. Neither was stationary. More aggressive minimization lowered empirical J while parameter gradients and train-versus-audit physical behavior became extreme, consistent with numerical non-coercivity/finite-bank overfitting rather than convergence to a local stationary Ritz branch.

**ENVELOPE GRADIENT NOT YET VALIDATED.** Do not run eta optimization. The next step is a fixed-eta inner-problem study: increase and cross-check frozen Ritz quadrature support and establish a well-resolved stationary branch without changing the envelope derivative or weakening its gates. Only then should the same five-direction optimized-value test be rerun.

## Historical permissive smoke gradient-check result

The checked eta was the precise existing 3% Full geometry:

```text
[0.8954153767761239, 0.20592631632470587,
 1.3343788098383822, 0.8654288352917223,
 0.7508355365766083, 0.5179100329264751,
 1.6423735249784726, 0.5883599695898114]
```

The deterministic normalized direction was:

```text
[ 0.0245126620957966, -0.814283728846668,
 -0.0988113773660549, -0.324687591913467,
  0.162170684250407,   0.278634855979501,
 -0.278425273191138,   0.199243748068737]
```

The fixed-theta AD directional derivative was `-834.9745079251195`.

| epsilon | reoptimized FD | relative discrepancy | A plus | A minus | plus/minus final L-BFGS gradient norm |
|---:|---:|---:|---:|---:|---:|
| 1e-2 | -305.212641272 | 0.634465 | 26.069084301 | 32.173337127 | 10.7663 / 97.8032 |
| 3e-3 | -1056.418579328 | **0.209618** | 30.004409144 | 36.342920620 | 22.2756 / 46.3850 |
| 1e-3 | -1689.460928493 | 0.505775 | 29.671892700 | 33.050814557 | 9.65056 / 32.6500 |
| 3e-4 | 1312.271145965 | 1.636282 | 29.325762732 | 28.538400044 | 13.8532 / 24.0879 |

The isolated best discrepancy, at `epsilon=3e-3`, was below the old smoke tolerance `0.5`. That old rule is now explicitly rejected: the center solve had `A=16.119864376`, `J=-6.827354359`, `abs(A+2J)=2.465155658`, and energy-identity relative error `0.152927`; L-BFGS did not converge. This result establishes only that the code path executes.

The complete machine-readable table is in `outputs/gradient_checks/smoke/result.json`; `outputs/smoke/gradient_check.json` contains the check rerun immediately before refinement.

## Two-step smoke refinement result

Starting eta was the geometry above. The two-step endpoint was:

```text
[0.8914256884240461, 0.2019296180049923,
 1.3383772473387590, 0.8694132839397082,
 0.7548356175140781, 0.5139084871977986,
 1.6463751987052373, 0.5923626846943868]
```

Selection risk changed from `7.314460070` to `7.858039431`; both are below the smoke selection ceiling `17.342061883`. Minimum periodic separation at the endpoint was `0.340569614`, above `0.18`, and the nearest logged periodic branch distance was `0.108460419`.

Fresh selection actions were `316.978478385` for the incumbent and `880.672211048` for the refined candidate. These numbers are not a valid improvement comparison because both candidates failed hard forcing/support audits; the refined candidate also failed its Ritz certificate. Independent validation of the retained incumbent failed the validation risk ceiling (`5.791271494 > 4.438543932`) and hard forcing audit. Consequently `accepted=false`, the incumbent is retained, and no authoritative improvement is claimed.

## Tests

`test_full_gradient.py` covers import isolation, output-path enforcement, reference closure/fixed theta, finite eta gradient shape, implicit projection VJP, forcing VJP, Ritz identity, reoptimized directional finite differences, periodic wrapping, smooth versus exact separation, differentiable risk versus the exact risk gate, JSON-safe diagnostics, rejection of a single lucky epsilon, and acceptance only for a consecutive convergence window. Current result: `12 tests`, all passing.

`test_production_galerkin.py` implements the 27 requested production-workflow
checks, including read-only discovery, path isolation, both invariant feature
families, nested normalization, rank-aware algebra, held-out residual formulas,
and the fixed-coefficient eta derivative contract. `test_fast_production.py`
adds six acceleration, cache, periodic-geometry, numerical-equivalence, and
restart-consensus checks. Together with the unchanged 17 Galerkin tests and 12
continuous-gradient tests, all 62 tests pass.

## Files in this experiment

- `__init__.py`
- `config.json`
- `deep_ritz.py`
- `domain.py`
- `eval.py`
- `experiment.py`
- `forcing.py`
- `full_gradient.py`
- `gradient_check.py`
- `measurements.py`
- `production_artifacts.py`
- `authoritative_platform.py`
- `authoritative_stability.py`
- `fast_production.py`
- `fast_workflow.py`
- `production_authoritative.py`
- `production_basis.py`
- `production_galerkin.py`
- `production_gradient.py`
- `production_refinement.py`
- `production_workflow.py`
- `reference.py`
- `risk.py`
- `rigorous_gradient_check.py`
- `run.py`
- `selection.py`
- `test_full_gradient.py`
- `test_fast_production.py`
- `test_galerkin.py`
- `test_production_galerkin.py`
- `workflow.py`
- `README.md`
- `GALERKIN_EVALUATION.md`
- `PRODUCTION_GALERKIN_EVALUATION.md`
- `FAST_PRODUCTION_3PCT_EVALUATION.md`
- `AUTHORITATIVE_GPU_ACCELERATION.md`
- `AUTHORITATIVE_STABILITY_EVALUATION.md`

The copied files are local implementations and use relative imports. No production experiment module is imported.

## Isolation and limitations

Every write entry point in the new workflow is guarded to remain below this experiment's `outputs/`; the localized legacy runner is guarded as well. Generated truth/reference banks, checkpoints, JSON results, and logs are local. Static search found no import or hard-coded output path targeting the old experiment.

No files were modified outside `experiments/skyrmions_deep_ritz_full/` by this implementation. `ETA_OPTIMIZATION_AUDIT.md` was already untracked before this task and was preserved unchanged.

Not established for the historical nonlinear fixed-theta envelope route:

- production/full-bank gradient agreement;
- production inner Ritz stationarity;

Established separately for the fixed-feature production Galerkin route:

- production/full-bank Galerkin gradient agreement in five AD/FD directions;
- a certified fixed-design authoritative action improvement for the prior tiny
  update, recorded only in this isolated experiment.

For the Galerkin-only continuous 3% program, independent validation is complete
and records a risk-gate reversal. A full Pareto sweep was intentionally not run.

Files modified for the rigorous follow-up were `config.json`, `run.py`, `workflow.py`, `test_full_gradient.py`, and this README; `rigorous_gradient_check.py` was added. Every one is inside this isolated directory.

The old checked-in outputs in this repository copy do not contain the complete
production banks, so smoke runs generated deterministic local artifacts. The
validated production Galerkin work instead used the complete frozen set found at
the recorded read-only source and hash-materialized it into
`outputs/production_galerkin/artifacts/`; it did not label regenerated data as
production.
