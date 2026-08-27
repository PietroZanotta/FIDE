# Official K=280 Galerkin Full Pareto evaluation

## Outcome

**OFFICIAL SWEEP BLOCKED AT THE 0.5% SELECTION PREREQUISITE.**

The fixed-feature K=280 Galerkin finite-dimensional approximation was frozen as
the official skyrmion Full discretization, the selection context reproduced,
and the complete deterministic start manifest frozen. The sequential sweep then
stopped fail-closed before its first gradient step because none of the four
predeclared exact-risk-feasible 0.5% starts passed the unchanged independent
selection-audit Ritz-energy certificate.

No 0.5% winner exists under this frozen protocol. Consequently allowances
1–5% were not run, finalist derivative audits were not run, no selection
manifest was frozen, and the predeclared fresh-validation bank was neither
generated nor opened. No scientific validation classification is assigned to
any allowance because the required selection prerequisite was not reached.

## Methodological statement

For this official attempt, the Full solver is the fixed-feature K=280 Galerkin
approximation of the weighted-Poisson weak problem. The dictionary, ordering,
fixed selection-train normalization, relative rank tolerance `1e-12`, and all
production algebra and physical thresholds were frozen. This is a
finite-dimensional discretization; it is not claimed to be the converged
infinite-dimensional Full solution.

The nonlinear Deep Ritz implementation did not train, rank, select, certify,
validate, or arbitrate any candidate.

## Repository isolation and initial state

The initial `git status --short`, captured before task changes, was:

```text
 M experiments/skyrmions_deep_ritz_full/README.md
 M experiments/skyrmions_deep_ritz_full/deep_ritz.py
 M experiments/skyrmions_deep_ritz_full/run.py
 M experiments/skyrmions_deep_ritz_full/workflow.py
?? experiments/skyrmions_deep_ritz_full/AUTHORITATIVE_GPU_ACCELERATION.md
?? experiments/skyrmions_deep_ritz_full/AUTHORITATIVE_STABILITY_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/FAST_PRODUCTION_3PCT_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/FINAL_3PCT_GALERKIN_CROSSCHECK.md
?? experiments/skyrmions_deep_ritz_full/GALERKIN_ONLY_3PCT_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/authoritative_platform.py
?? experiments/skyrmions_deep_ritz_full/authoritative_stability.py
?? experiments/skyrmions_deep_ritz_full/final_crosscheck.py
?? experiments/skyrmions_deep_ritz_full/final_crosscheck_run.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only_data.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only_run.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only_workflow.py
?? experiments/skyrmions_deep_ritz_full/test_fast_production.py
?? experiments/skyrmions_deep_ritz_full/test_final_crosscheck.py
?? experiments/skyrmions_deep_ritz_full/test_galerkin_only.py
```

Task-created source and report files are confined to
`experiments/skyrmions_deep_ritz_full/`:

- `OFFICIAL_GALERKIN_PARETO_PROTOCOL.md`;
- `OFFICIAL_GALERKIN_PARETO_EVALUATION.md`;
- `official_pareto_common.py`;
- `official_pareto_selection.py`;
- `official_pareto_validation.py`;
- `official_pareto_report.py`;
- `official_pareto_run.py`;
- `test_official_pareto.py`.

Machine-readable records exist only below
`outputs/official_galerkin_pareto/`. No original experiment, shared `src/`,
`native/`, historical output, prior 3% Galerkin seal, or paper artifact was
modified or overwritten.

## Frozen protocol and chronology

The protocol payload SHA-256 is:

```text
9560b50761c59727dfad838db55159b5f423e5b6da3da9eede4f636fe742b665
```

The serialized `protocol.json` SHA-256 is
`913bfa67c9d53e033b36d293cee714817bcfbcbe13c3324e412d282b147f5c9f`.
The validated K=280 dictionary hash is
`37e9b60fcb92c4e5a0ee7ec1651fb7f8889f7ac6bdb02d3bd314e9ef40833326`.

The protocol was written and hash-sealed before reproduction, start-pool
evaluation, or selection. It freezes:

- ordered allowances `[0.5, 1, 2, 3, 4, 5]%`;
- selection risk `R_sel <= (1+p/100) R_Law,sel`;
- validation risk `R_val <= (1+p/100+0.05) R_Law,val`;
- strict validation `p%` reporting as transparency only;
- trust radius `2e-4`, initial step `5e-5`, eight step attempts, ten
  backtracks, `1e-10` replacement tolerance, and certification frequency;
- exact geometry, rank, algebra, forcing, covariance, and held-out certificate
  thresholds;
- deterministic historical/local/global start construction;
- fresh-validation sizes, seed derivation, and reporting conventions.

The complete start-manifest SHA-256 is
`7ff8414d637c8bc92d4553a42e514beab45397d2c5262f20a3bed02ef0c85624`.
It was frozen before the first selection attempt. The historical Pareto JSON
was byte-hashed for provenance, but its old validation fields were not parsed by
the selection process; the historical geometries were copied as fixed
initialization constants before execution.

## Predeclared fresh-validation seeds

These seed records were frozen before selection. No corresponding data were
generated because selection did not freeze.

| label | derivation SHA-256 | integer seed |
|---|---|---:|
| truth | `bbc6944943bf3b124305f856d90b18b097db64dbd590047ba376cd52597d4013` | 994861991 |
| reference_fit | `992084aa37fec4fef86492c352bba2f0e320eb96b89338dbae73506e94bd046c` | 1782566484 |
| reference_audit | `0d7b6671a2a0d2003cfcb5d6fbeb907724e0d63a34e584b47840f5dc14aa1159` | 1033346787 |
| measurement_noise | `f3b275566a600ae7b9804d49540511b143bd77bad5f6cf7a8ff0670f65aef9e3` | 1371862423 |

## K=280 reproduction

The GPU reproduction completed successfully. Its machine record SHA-256 is
`23b5ece9ba6aa8df592672660821b7c78a532e2d17e99f0c0e93318db6bf772c`.

| design | selection risk | train action | gradient norm | train/audit forcing | algebra | held-out energy | complete certificate |
|---|---:|---:|---:|---|---|---:|---|
| Law | 5.186549474478 | 0.374832445766 | 3.003369 | pass/pass | pass | 0.108442 | fail |
| eta0 | 5.340106050966 | 0.293500059196 | 3.352330 | pass/pass | pass | 0.079867 | pass |
| eta_grad | 5.342099811291 | 0.292740724037 | 3.345818 | pass/pass | pass | 0.079468 | pass |

Actions and gradients were deterministic to the `1e-12` repetition gate.
Eta0's gradient matched the previously validated K=280 vector to the frozen
`1e-8` relative tolerance. Law's known selection-audit energy failure was
reproduced rather than treated as a new discrepancy.

The exact Law geometry was

```text
[0.890286510596537, 0.227289528868506,
 1.310368832144490, 0.859163192162967,
 0.797588822714243, 0.535723001316333,
 1.610343150447571, 0.583219225445585]
```

and the recomputed selection Law risk was `5.186549474478042`. Thus the exact
0.5% selection ceiling was `5.212482221850432`.

## Deterministic start generation

The fixed global pool had 48 periodic, separation-valid geometries. None fell
within even the 5% exact selection-risk ceiling; this outcome was recorded and
the pool was not regenerated. Exact-feasible historical and deterministic local
starts remained available.

The four 0.5% starts, frozen before certification, were:

| id | provenance | exact risk | eta |
|---|---|---:|---|
| Law | frozen Law anchor | 5.186549474478 | `[0.890286510596537, 0.227289528868506, 1.310368832144490, 0.859163192162967, 0.797588822714243, 0.535723001316333, 1.610343150447571, 0.583219225445585]` |
| historical_0p5pct | frozen historical selection geometry | 5.203174625200 | `[0.888224002114442, 0.226590028285788, 1.308928302996689, 0.862825514790280, 0.786665206117643, 0.541803221343441, 1.616175859255502, 0.584353406982718]` |
| local_01 | predeclared deterministic local perturbation | 5.203146950267 | `[0.888139999146553, 0.226614168617762, 1.308881247729711, 0.863157743470128, 0.786410344994725, 0.541736539945873, 1.615927508942260, 0.584206507005511]` |
| local_00 | predeclared deterministic local perturbation | 5.203206842546 | `[0.888263004373339, 0.226661487369465, 1.308893582586036, 0.862867899588194, 0.786658158958704, 0.541843426464795, 1.616248144933753, 0.584314009843485]` |

All four risks are strictly below the 0.5% ceiling. All four geometries passed
periodicity and the exact minimum-separation rule.

## Fail-closed 0.5% certification result

Every start passed train and audit projection, ESS, forcing-compatibility, and
covariance gates. Every K=280 solve passed numerical-rank, range,
stationarity, symmetry, conditioning, and restricted `A=-2J` identity gates.
Every held-out weak, gauge, and moment-rate residual passed. The sole hard
failure was the held-out Ritz-energy residual:

| start | risk | train action | weak | energy | threshold | gauge | moment rate | certified |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Law | 5.186549474478 | 0.374832445766 | 0.070193 | 0.108442 | 0.08 | 3.25e-17 | 0.013487 | no |
| historical_0p5pct | 5.203174625200 | 0.350418194530 | 0.064408 | 0.107287 | 0.08 | 1.04e-17 | 0.012097 | no |
| local_01 | 5.203146950267 | 0.350290534674 | 0.064264 | 0.107493 | 0.08 | 1.20e-17 | 0.012139 | no |
| local_00 | 5.203206842546 | 0.350365500001 | 0.064396 | 0.107339 | 0.08 | 1.07e-17 | 0.012096 | no |

The failing values are not marginal numerical noise: they exceed the frozen
threshold by `0.02729–0.02844`. The protocol requires an exactly feasible,
fully certified start and forbids loosening thresholds after outcomes. The
trajectory function therefore performed no gradient step for any start and the
sequential sweep raised:

```text
RuntimeError: no certified feasible candidate at allowance 0.5
```

This is a genuine protocol/scientific blocker, not a GPU, JAX, projection,
rank, geometry, or implementation crash.

## Phases intentionally not run

Because Phase 4 did not produce a 0.5% winner:

- no allowance result or six-point selection draft was written;
- 1%, 2%, 3%, 4%, and 5% optimization did not run;
- no incumbent chain or monotonicity claim was constructed;
- no official Pareto selection table can be populated;
- no finalist gradient audit ran;
- `pareto_selection.json` and `manifest.json` do not exist;
- no selection-frozen visual notification was sent;
- no fresh validation truth, noise, fit, or audit array was generated or read;
- no validation risk, action, standard error, or classification was computed;
- no geometry was substituted and no old validation result entered selection.

The machine-readable blocker is `outputs/official_galerkin_pareto/final_summary.json`.

## Tests

All relevant isolated tests completed: **111 tests ran and all 111 passed**.
The blocker summary retains six explicit allowance-status rows, so the final
machine-table consistency contract is testable even without winners. The suite
comprises 12 continuous-gradient, 17 base Galerkin,
27 production-Galerkin, 6 fast-production, 20 Galerkin-only, 9 final-crosscheck,
and 20 official-Pareto tests. Live CPU/GPU float64 equivalence tests passed.

The new official tests cover K/dictionary fixation, selection and validation
risk arithmetic, deterministic seed derivation, validation construction gating,
selection-path isolation from old validation, nested feasible-set arithmetic,
incumbent retention, action monotonicity, exact risk rejection, write-once
winners, common-solver reductions, disjointness helpers, immutable validation,
Deep Ritz exclusion, output isolation, protocol hashing, and resumable
trajectory signatures.

## Final repository audit

`git diff --check` passed. The final repository status preserves every
pre-existing modification and adds only the official protocol, implementation,
tests, blocker evaluation, and ignored official output subtree inside the
isolated experiment. No historical output was overwritten.

The prior sealed 3% selection and validation SHA-256 values remain
`32ef8fa0f76a5c0cdacc8381fb1345ad1315d3ec5ece8d400c225eff78d829c2`
and
`fe143132293aebdd379f958a2a6b350440c439193db9d790a0b3c83fbf7f9206`.

Final `git status --short`:

```text
 M experiments/skyrmions_deep_ritz_full/README.md
 M experiments/skyrmions_deep_ritz_full/deep_ritz.py
 M experiments/skyrmions_deep_ritz_full/run.py
 M experiments/skyrmions_deep_ritz_full/workflow.py
?? experiments/skyrmions_deep_ritz_full/AUTHORITATIVE_GPU_ACCELERATION.md
?? experiments/skyrmions_deep_ritz_full/AUTHORITATIVE_STABILITY_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/FAST_PRODUCTION_3PCT_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/FINAL_3PCT_GALERKIN_CROSSCHECK.md
?? experiments/skyrmions_deep_ritz_full/GALERKIN_ONLY_3PCT_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/OFFICIAL_GALERKIN_PARETO_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/OFFICIAL_GALERKIN_PARETO_PROTOCOL.md
?? experiments/skyrmions_deep_ritz_full/authoritative_platform.py
?? experiments/skyrmions_deep_ritz_full/authoritative_stability.py
?? experiments/skyrmions_deep_ritz_full/final_crosscheck.py
?? experiments/skyrmions_deep_ritz_full/final_crosscheck_run.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only_data.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only_run.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only_workflow.py
?? experiments/skyrmions_deep_ritz_full/official_pareto_common.py
?? experiments/skyrmions_deep_ritz_full/official_pareto_report.py
?? experiments/skyrmions_deep_ritz_full/official_pareto_run.py
?? experiments/skyrmions_deep_ritz_full/official_pareto_selection.py
?? experiments/skyrmions_deep_ritz_full/official_pareto_validation.py
?? experiments/skyrmions_deep_ritz_full/test_fast_production.py
?? experiments/skyrmions_deep_ritz_full/test_final_crosscheck.py
?? experiments/skyrmions_deep_ritz_full/test_galerkin_only.py
?? experiments/skyrmions_deep_ritz_full/test_official_pareto.py
```

The frozen validation seeds predate selection, but no fresh validation was
generated because all six winners were not frozen. Deep Ritz did not
participate. The old validation bank remained development-only and did not enter
the selection process.

## Scientific interpretation and next permissible work

The current official protocol cannot define a complete 0.5–5% Pareto sweep:
the K=280 selection-audit energy certificate is incompatible with every
predeclared 0.5% start found by the frozen search. Under the stated fail-closed
rules it would be scientifically incorrect to relax `0.08`, discard 0.5%,
substitute a post-hoc start, continue from an uncertified start, or generate
fresh validation.

A future attempt requires a new, explicitly versioned protocol frozen before
new optimization. Its pre-selection methodological study would need to resolve
whether low-risk K=280 designs can satisfy the held-out energy certificate—for
example by diagnosing basis adequacy at the Law/0.5% region or by predeclaring a
search that is allowed to traverse uncertified initial points while certifying
every accepted official endpoint. None of those changes belongs to this frozen
official attempt.

**BLOCKED: NO CERTIFIED K=280 START AT 0.5%; FRESH VALIDATION NOT OPENED.**
