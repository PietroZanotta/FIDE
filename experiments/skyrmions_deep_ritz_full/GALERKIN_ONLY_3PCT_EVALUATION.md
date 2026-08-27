# Galerkin-only 3% production evaluation

## Methodological decision

From this checkpoint forward, the Full solver in this isolated experiment is
the fixed-feature Galerkin discretization of the weighted-Poisson weak problem.
It works in the finite permutation-invariant space
`V_K = span(phi_1, ..., phi_K)`, uses the rank-aware solution
`a_t = -K_t^dagger f_t`, evaluates `A_G,K = sum_t omega_t a_t^T K_t a_t`,
and differentiates the fixed-coefficient envelope without differentiating
through the eigendecomposition.

The historical nonlinear Deep Ritz envelope is not validated and is no longer
used for continuous selection. Historical fixed-design Deep Ritz results remain
untouched for comparison only. No new Galerkin-only mode may train Deep Ritz or
use a Deep Ritz action to rank, select, certify, or validate a candidate.

## Repository isolation and initial state

The initial repository status for this task was:

```text
 M experiments/skyrmions_deep_ritz_full/README.md
 M experiments/skyrmions_deep_ritz_full/deep_ritz.py
 M experiments/skyrmions_deep_ritz_full/run.py
 M experiments/skyrmions_deep_ritz_full/workflow.py
?? experiments/skyrmions_deep_ritz_full/AUTHORITATIVE_GPU_ACCELERATION.md
?? experiments/skyrmions_deep_ritz_full/AUTHORITATIVE_STABILITY_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/FAST_PRODUCTION_3PCT_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/authoritative_platform.py
?? experiments/skyrmions_deep_ritz_full/authoritative_stability.py
?? experiments/skyrmions_deep_ritz_full/test_fast_production.py
```

All of those changes predate this Galerkin-only continuation and are preserved.
New code, reports, caches, and outputs are confined to
`experiments/skyrmions_deep_ritz_full/`; new numerical outputs go only below
`outputs/galerkin_only_3pct/`. Production experiments, historical outputs,
`src/`, and `native/` are read only.

## Notification helper

The helper is `scripts/notify.py`. Its inspected interface is:

```bash
python scripts/notify.py "MESSAGE" --title "TITLE" --timeout-ms 7000
```

It sends a silent visual freedesktop D-Bus notification and falls back to a
timed `xmessage`. It is invoked after each required major phase and once after
the complete task; it is never modified by this experiment.

## Outcome summary

The bounded GPU-first workflow is complete. K=280 was selected as the practical
finite discretization, six bounded Galerkin-only trajectories produced six
certified feasible endpoints, and the minimum-action endpoint was frozen before
validation. It improved both selection and validation Galerkin action. The one
sealed validation nevertheless failed the predeclared validation-risk ceiling,
so the scientific outcome is a validation reversal and the selection was not
reopened.

## GPU-first K=160 benchmark

JAX selected `cuda:0`, an NVIDIA GeForce RTX 5090 Laptop GPU, with float64
enabled. The same command falls back to CPU when CUDA discovery fails and
records the actual platform. The validated K=160 train cache was reused by
artifact, dictionary, and cache signatures; no basis tensor was recomputed.

| stage | first / total seconds | steady median seconds |
|---|---:|---:|
| selection-only data load | 6.4246 | — |
| context initialization | 3.0925 | — |
| cached basis device load | 3.0885 | — |
| projection and forcing | 0.0334 | 0.0360 |
| K/f assembly | 0.7199 | 0.7114 |
| rank-aware solve | 0.0243 | 0.0221 |
| action aggregation | 0.0010 | 0.0007 |
| fixed-coefficient value+gradient | 0.0291 | 0.0273 |
| complete value-only evaluation | 10.6964 | 0.8017 |
| complete value+gradient evaluation | 4.2878 | 0.8239 |
| held-out selection certification | 19.7252 | — |

The first complete-call asymmetry reflects compilation order; all timings use
`block_until_ready()`. The cache holds basis values and state gradients only:
approximately `4.189 GiB` at K=160. A rejected per-sample `K x K` Gram cache
would require `20.313 GiB` and is not constructed.

GPU K=160 action was `0.2645771969178057`; the validated CPU reference was
`0.2645771969193504`, a relative difference `5.84e-12`. The eta-gradient
relative difference was `4.46e-9`. Both pass the declared `1e-10` action and
`1e-8` gradient tolerances.

The independent selection-audit action was `0.2650532804800`; weak, energy,
gauge, and moment-rate residuals were `0.074013`, `0.050369`, `8.67e-18`, and
`0.007939`. All certificates and algebra gates passed. Worst retained condition
was `3.719e11`, with range and stationarity residuals about `2.51e-9`.

## Galerkin-only call graph

The new entry point is `galerkin_only_run.py`. Its selection loader reads only
the `design` truth key and selection projection/train/audit banks. It does not
load validation truth, validation-fit, or validation-audit arrays. A separate
validation loader exists but is called only after a frozen selection result is
verified.

Static scanning of the new data, solver, workflow, and command modules found no
neural training, nonlinear audit, restart, or historical fixed-design rescore
call. Candidate judgment uses exact risk and geometry, projection/ESS/forcing,
rank/range/stationarity and restricted identity, plus held-out Galerkin weak,
energy, gauge, and moment-rate certificates.

## Extended dictionary and bounded convergence

The first 160 coordinates, including their order, parameters, fixed means, and
Dirichlet scales, are byte-for-byte unchanged. The extension appends the next
globally ordered periodic wavevectors not already present, with cosine then sine
for each vector. K=200 adds 20 wavevectors, K=240 adds 40 total, and K=280 adds
60 total. No radial coordinate was changed or duplicated, no eta-dependent fit
was performed, and new diagonal normalization used only the eta-independent
selection train base weights.

K=280 was conditionally run because the K=200→240 train-action increment was
`2.398%`, above `1.5%`, and its estimated resident train cache was only
`7.332 GiB`. No K above 280 was evaluated.

| K | train action | selection-audit action | action increment | weak | energy | gauge | moment rate | min rank fraction | worst condition | range residual | stationarity residual | gradient norm | certified |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 160 | 0.264577197 | 0.265053280 | — | 0.074013 | 0.050369 | 1.13e-17 | 0.007939 | 0.99375 | 3.719e11 | 2.51e-9 | 2.51e-9 | 2.90394 | yes |
| 200 | 0.279589966 | 0.281308617 | 5.370% | 0.071466 | 0.069929 | 1.94e-17 | 0.008815 | 0.99500 | 3.767e11 | 1.81e-9 | 1.81e-9 | 3.12991 | yes |
| 240 | 0.286457888 | 0.289068593 | 2.398% | 0.069904 | 0.076748 | 1.44e-17 | 0.009819 | 0.99583 | 3.828e11 | 1.71e-9 | 1.71e-9 | 3.25847 | yes |
| 280 | 0.293500059 | 0.296692769 | 2.399% | 0.069147 | 0.079867 | 1.11e-17 | 0.010403 | 0.99643 | 4.045e11 | 1.43e-9 | 1.43e-9 | 3.35233 | yes |

Every row passed rank/range/stationarity, restricted `A=-2J`, conditioning,
forcing, and held-out physical gates. The K=280 energy residual passes but is
close to the fixed `0.08` threshold; this is retained as a limitation.

| neighboring spaces | gradient cosine | relative gradient difference | relative train-action increment |
|---|---:|---:|---:|
| 160→200 | 0.999963 | 0.072666 | 0.053696 |
| 200→240 | 0.999950 | 0.040660 | 0.023975 |
| 240→280 | 0.999981 | 0.028659 | 0.023994 |

No tested extension met the preferred `<=2%` action-increment rule. K=280 is
therefore selected under the declared fallback: it is the largest scientifically
valid tested space, its neighboring gradient cosine exceeds `0.995`, and its
relative gradient change is below `0.05`. Action convergence is explicitly
incomplete; K=280 is a declared practical finite discretization, not evidence
of infinite-dimensional convergence.

## Selected-K cache and hot path

The K=280 cache signature includes the frozen artifact manifest, complete basis
definition, K, fixed-normalization hash, float64 dtype, production configuration,
and train-bank shape. It reuses the exact old K=160 prefix and the K=240 cache
when extending to 280. Selection keeps only the `7.332 GiB` train values and
state gradients resident; it never creates the rejected `62.207 GiB`
per-sample Gram tensor. Audit arrays are streamed during certification.

On GPU, selection-only data loading took `6.174 s` and cache-hit context
initialization/device loading took `4.904 s`. Complete K=280 value-only timing
was `11.904 s` first and `1.534 s` steady; value+gradient was `4.931 s` first
and `1.553 s` steady. Full independent selection certification took `19.912 s`.
The repeated eta0 result matched the convergence row and remained certified.

## Commands

The new modes use a clean entry point that automatically prefers CUDA and falls
back to CPU. `--force-cpu` is available only as an equivalence diagnostic.

```bash
python -m experiments.skyrmions_deep_ritz_full.galerkin_only_run --mode benchmark
python -m experiments.skyrmions_deep_ritz_full.galerkin_only_run --mode convergence
python -m experiments.skyrmions_deep_ritz_full.galerkin_only_run --mode profile-selected-K
python -m experiments.skyrmions_deep_ritz_full.galerkin_only_run --mode optimize-3pct
python -m experiments.skyrmions_deep_ritz_full.galerkin_only_run --mode validate-3pct \
  --selection-result experiments/skyrmions_deep_ritz_full/outputs/galerkin_only_3pct/selection/result.json
```

All writes are guarded below `outputs/galerkin_only_3pct/`. Convergence rows and
optimization trajectories have complete hash signatures and resume without
recomputation when their artifact, dictionary, configuration, K, and dtype
match. The sealed validation result is hash-locked and cannot be overwritten by
a different selection.

## Bounded 3% optimization

The objective was the K=280 train-bank Galerkin action. The exact risk ceiling
was `1.03 * 5.186549474478041 = 5.342145958712383`. Each trajectory used a
periodic trust radius of `2e-4`, initial step `5e-5`, at most 8 accepted-step
attempts, and at most 10 backtracks per step. Actual Galerkin decrease had to
exceed `1e-10`; rank had to remain unchanged. Exact risk, geometry,
projection/ESS/forcing, conditioning, rank/range/stationarity, and restricted
identity gates were enforced. Full independent selection-audit certification
ran at every start, every fourth accepted step, and each endpoint.

The six deterministic starts, listed before any validation access, were:

| id | starting eta |
|---|---|
| eta0 | `[0.895415376776124, 0.205926316324706, 1.33437880983838, 0.865428835291722, 0.750835536576608, 0.517910032926475, 1.64237352497847, 0.588359969589811]` |
| previous Galerkin tiny | `[0.895383992114667, 0.205950359074711, 1.33451447738688, 0.865474415045120, 0.750807733902488, 0.517972736272111, 1.64239365788202, 0.588410610733759]` |
| prior endpoint 1 | `[0.895382387811834, 0.205951601806098, 1.33452144773279, 0.865476760527674, 0.750806301688284, 0.517975943700434, 1.64239468814311, 0.588413211942954]` |
| prior endpoint 2 | `[0.895382429347652, 0.205951559473630, 1.33452124107265, 0.865476688273251, 0.750806346921625, 0.517975859037171, 1.64239466063877, 0.588413135299205]` |
| prior endpoint 3 | `[0.895415849208097, 0.205928253793949, 1.33452511741145, 0.865477733532902, 0.750798956674559, 0.517958305745967, 1.64241131912525, 0.588426877638217]` |
| prior endpoint 4 | `[0.895346331504072, 0.205931373065342, 1.33452050075795, 0.865417322717141, 0.750798114222458, 0.517959014589482, 1.64235144393868, 0.588458018115851]` |

All six starts and all six endpoints passed the exact gates. Each trajectory
accepted three steps. The search used 24 Galerkin value/gradient evaluations
and 12 full selection certifications.

| finalist source | selection action | exact risk | risk increase | selection-audit action | weak | energy | certified |
|---|---:|---:|---:|---:|---:|---:|---|
| prior endpoint 1 | **0.292740724037** | 5.342099811291 | 2.999110% | 0.295913390008 | 0.069030 | 0.079468 | yes |
| prior endpoint 2 | 0.292741588468 | 5.342096872223 | 2.999054% | 0.295914265172 | 0.069030 | 0.079468 | yes |
| prior endpoint 3 | 0.292760883507 | 5.342096803128 | 2.999052% | 0.295935529483 | 0.069033 | 0.079478 | yes |
| previous Galerkin tiny | 0.292770275627 | 5.341999971496 | 2.997185% | 0.295943333078 | 0.069038 | 0.079488 | yes |
| prior endpoint 4 | 0.292773409357 | 5.342083800698 | 2.998802% | 0.295944129900 | 0.069024 | 0.079431 | yes |
| eta0 trajectory | 0.293346164082 | 5.340060691671 | 2.959795% | 0.296526918654 | 0.069202 | 0.079872 | yes |

The frozen winner is:

```text
[0.895371148114089, 0.205982940238786,
 1.334525121515147, 0.865464965382237,
 0.750749623351011, 0.518133188490931,
 1.642405611981796, 0.588309862016330]
```

Against unrefined eta0, its selection action changed from
`0.293500059196` to `0.292740724037`, a decrease of `0.000759335158`
(`0.258717%`). Its exact selection risk was `5.342099811291`, below the
ceiling by `4.61474e-5`.

The winner's train/audit projection residuals were `7.54e-11` / `1.33e-11`,
ESS fractions `0.071481` / `0.069644`, pre-centering forcing means `9.13e-9` /
`2.24e-9`, and covariance conditions `3.906` / `3.769`. Its restricted
identity residual was `4.34e-11`; minimum rank fraction `0.996429`; worst
condition `4.045e11`; range/stationarity residuals `1.44e-9`; and held-out
weak, energy, gauge, and moment-rate residuals `0.069030`, `0.079468`,
`1.63e-17`, and `0.010377`. Every selection gate passed.

The selection JSON explicitly records `selection_frozen=true`,
`validation_accessed=false`, and the dictionary hash. Neither Deep Ritz nor a
validation quantity was used to rank or arbitrate these finalists.

## Single sealed validation

Only after the visual winner-freeze notification did the validation command
open the validation truth, fit bank, and audit bank. It started in a separate
process, loaded no selection basis cache, rebuilt every design-dependent
quantity, solved the same rank-aware K=280 equations with the identical frozen
dictionary/normalization, and streamed fit and audit arrays sequentially. No
training or Deep Ritz evaluation was invoked. The frozen winner was not changed
after the result.

The predeclared validation convention reevaluates the fixed law eta on the
validation-fit bank. It gave law risk `5.357974522122316` and the 3% ceiling
`5.518713757785985`.

| validation metric | eta0 | frozen winner |
|---|---:|---:|
| scientific risk | 5.548626547535 | 5.550507348678 |
| risk passes | no | no |
| validation-fit Galerkin action | 0.239936893725 | 0.239336294320 |
| validation-audit Galerkin action | 0.238797637727 | **0.238200776201** |
| audit action standard error | 0.000655647082 | 0.000652899770 |
| weak residual | 0.029148 | 0.029060 |
| energy residual | 0.066467 | 0.066399 |
| gauge residual | 2.40e-17 | 2.53e-17 |
| moment-rate residual | 0.005786 | 0.005764 |
| minimum rank fraction | 0.996429 | 0.996429 |
| worst retained condition | 4.476e11 | 4.476e11 |
| range residual | 3.52e-9 | 3.52e-9 |
| stationarity residual | 3.52e-9 | 3.52e-9 |
| Galerkin/forcing/algebra certificates | pass | pass |
| complete validation gate | fail | fail |

The winner-minus-eta0 validation-audit action difference was
`-0.000596861526` (`-0.249944%`), so the Galerkin action ordering transferred.
The uncertainty entries use the production weighted empirical audit-sample
standard-error convention; the bank has no declared independent block/trial
structure, so no pseudo-replicates were introduced and no post-hoc significance
gate was invented.

Fit/audit projection residuals for eta0 were `3.90e-11` / `3.09e-11`, ESS
fractions `0.089653` / `0.104392`, and forcing means `3.17e-8` / `4.03e-9`.
For the winner they were `3.91e-11` / `3.10e-11`, `0.090248` / `0.105001`, and
`3.18e-8` / `4.05e-9`. All were valid. The failure is solely the unchanged
validation-risk gate: the winner exceeded it by `0.031793590892`. This is the
required validation reversal; selection was not reopened to choose another
finalist.

## Verification and limitations

The 20 new Galerkin-only tests pass, including live CPU/GPU float64 equivalence,
exact K=160 preservation, K=240/K=280 nestedness, eta independence,
permutation/periodic invariance, state gradients, signature coverage, streamed
K/f consistency, rank-aware algebra, exact 3% gating, selection sealing, no
validation access during selection, no Deep Ritz invocation, output isolation,
and the notification interface. The unchanged 12 continuous-gradient, 17
Galerkin, 27 production-Galerkin, and 6 fast-production tests also pass: 82
tests total.

Limitations are explicit:

- the K=240→280 action increment remains `2.399%`, so infinite-dimensional
  action convergence is not established;
- the K=280 selection energy residual is close to its fixed `0.08` limit;
- the selection winner sits close to the 3% selection-risk boundary;
- the deterministic validation action decreased, but both eta0 and the winner
  fail the independently reevaluated validation-risk ceiling;
- no alternative winner was chosen after validation and no Pareto sweep ran.

All new machine-readable artifacts are below
`outputs/galerkin_only_3pct/`. Historical outputs and the original production
incumbent were not modified.

## Final repository status

The final `git status --short` is:

```text
 M experiments/skyrmions_deep_ritz_full/README.md
 M experiments/skyrmions_deep_ritz_full/deep_ritz.py
 M experiments/skyrmions_deep_ritz_full/run.py
 M experiments/skyrmions_deep_ritz_full/workflow.py
?? experiments/skyrmions_deep_ritz_full/AUTHORITATIVE_GPU_ACCELERATION.md
?? experiments/skyrmions_deep_ritz_full/AUTHORITATIVE_STABILITY_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/FAST_PRODUCTION_3PCT_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/GALERKIN_ONLY_3PCT_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/authoritative_platform.py
?? experiments/skyrmions_deep_ritz_full/authoritative_stability.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only_data.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only_run.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only_workflow.py
?? experiments/skyrmions_deep_ritz_full/test_fast_production.py
?? experiments/skyrmions_deep_ritz_full/test_galerkin_only.py
```

The Deep Ritz, runner, workflow, acceleration/stability, and fast-production
entries were present in the initial state and were preserved. This task added
only the Galerkin-only report, four `galerkin_only*.py` modules, its isolated
test module, README additions, and ignored numerical outputs. Every listed path
is inside `experiments/skyrmions_deep_ritz_full/`; `git diff --check` passes.

B. GALERKIN-ONLY 3% SELECTION IMPROVED, VALIDATION DID NOT
