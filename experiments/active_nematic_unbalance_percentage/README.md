# Robust percentage-risk active-nematic MFSI

This directory contains the two-species unbalanced active-nematic experiment
and its robust percentage-risk Pareto workflow. It does not modify shared
`src/mfsi` or the experiment-local native screened-Poisson backend.

The physical model, signed-defect representation, endpoint-only reference,
finite-measure risk, tangent action, screened Full action, and numerical gates
are unchanged. The authoritative workflow adds robust physical views, one
common Law anchor, a nested allowance sweep, frozen inputs, and fail-closed
finalization.

## Status

`outputs/run` and `outputs/run_strengthened_seed18` are historical diagnostics,
not authoritative Pareto results. The strengthened diagnostic showed a large
selection improvement that reversed on held-out physical runs. It also predates
the end-to-end incumbent repair described below.

The production robust Pareto run is complete under `outputs/pareto_robust/`.
Independent reconstruction and recomputation passed with no selection or
validation failures. The finalized run includes `authoritative_pareto.md`,
`.csv`, `.json`, `.png`, and `authoritative_certification_diagnostic.json`.

## Percentage-risk budget

The Law geometry is optimized and exactly audited once, then frozen across the
`0.5%, 1%, 2%, 3%, 4%, 5%` sweep. For physical/reference view `v`, its exact
risk defines `R_star[v]`. Tangent and Full must satisfy

```text
R(eta, v) <= R_star[v] + (p/100) * abs(R_star[v])
```

for every frozen selection view. A good aggregate score cannot hide a failed
physical fold or learned-reference seed.

## Robust selection protocol

The 16 design physical runs are deterministically partitioned into four
leave-one-fold-out views. Every view is crossed with all three learned-reference
seeds. The configured robust objective is the maximum action over this complete
view set. Validation uses the same view construction on the disjoint 16-run
validation split.

Differentiable candidate generation uses one frozen design-fold proxy crossed
with all reference seeds. Every generated candidate is then audited on all
four folds crossed with all references, so this speed choice cannot weaken the
exact risk or action gate.

Each allowance has three direct stages:

1. Law: one modest multistart Adam search, followed by exact complete-bank
   auditing.
2. Tangent: one constrained Adam search, followed by exact risk and action
   certification.
3. Full: one low-resolution constrained Adam search, an exact Full pre-screen,
   and complete-bank rescoring of proxy finalists plus the exact Law, Tangent,
   and previous-Pareto incumbents.

Optimizer outputs are endpoints rather than their initial seeds. Incumbents are
therefore reinserted explicitly after optimization; being used as an optimizer
start is not treated as retention. Certification is derived from the selected
exact audits and is never assigned unconditionally.

Production uses 32 observation trials per physical/reference view for exact
selection and 32 independent trials per held-out view for validation. These
trials measure finite-particle and detector noise; uncertainty is also reported
across physical/reference views. Validation begins only after every allowance
winner has been frozen.

## Reproduction

Build a reusable source bank from the repository root:

```bash
.venv/bin/python experiments/active_nematic_unbalance_percentage/run.py physical-bank \
  --output-dir experiments/active_nematic_unbalance_percentage/outputs/source
.venv/bin/python experiments/active_nematic_unbalance_percentage/run.py defects \
  --output-dir experiments/active_nematic_unbalance_percentage/outputs/source
.venv/bin/python experiments/active_nematic_unbalance_percentage/run.py defect-audit \
  --output-dir experiments/active_nematic_unbalance_percentage/outputs/source
.venv/bin/python experiments/active_nematic_unbalance_percentage/run.py reference \
  --output-dir experiments/active_nematic_unbalance_percentage/outputs/source
```

Run and finalize the robust sweep:

```bash
.venv/bin/python experiments/active_nematic_unbalance_percentage/run_pareto.py \
  --input-dir experiments/active_nematic_unbalance_percentage/outputs/source \
  --output experiments/active_nematic_unbalance_percentage/outputs/pareto_robust

.venv/bin/python experiments/active_nematic_unbalance_percentage/finalize_authoritative_pareto.py \
  --pareto-dir experiments/active_nematic_unbalance_percentage/outputs/pareto_robust
```

Add `--smoke --reference-seeds 20260818 --percent 0.5 1` to the source stages
and Pareto runner for a small end-to-end check. Smoke numbers are structural
tests and have no scientific interpretation.

## Production result

The authoritative held-out evaluation is:

| Allowance | Selection Full action | Validation Full action +/- view SE | Full vs Law |
|---:|---:|---:|---:|
| 0.5% | 29.377607 | 21.761715 +/- 5.361181 | 0.00% |
| 1% | 23.037196 | 20.869286 +/- 4.080682 | 4.10% |
| 2% | 22.529704 | 22.228235 +/- 8.968947 | -2.14% |
| 3% | 21.552532 | 18.157088 +/- 2.117093 | 16.56% |
| 4% | 21.552532 | 18.157088 +/- 2.117093 | 16.56% |
| 5% | 21.552532 | 18.157088 +/- 2.117093 | 16.56% |

The 3% allowance is the smallest point attaining the selected Full-action
plateau, making it the natural operating point in this sweep. Its held-out
Full action is 16.56% below Law. The 2% selection/validation reversal is kept
in the table: the robust protocol exposes it rather than treating selection
improvement as a held-out result. Reported uncertainty is the jackknife
standard error across physical/reference views and should not be interpreted
as a formal significance claim.

The complete table is in `outputs/pareto_robust/authoritative_pareto.md`; the
fail-closed receipt is
`outputs/pareto_robust/authoritative_certification_diagnostic.json`.

## Historical diagnostic

The earlier fast 5% run completed with 32/32 valid validation trials for every
design and seed:

| Reference seed | `R_star` | `R_max` | Validation action: Law | Validation action: Full | Full reduction |
|---:|---:|---:|---:|---:|---:|
| 20260818 | 1.614866 | 1.695609 | 18.1897 | 15.5570 | 14.47% |
| 20260819 | 1.607028 | 1.687380 | 18.1388 | 15.1462 | 16.50% |
| 20260820 | 1.615106 | 1.695861 | 17.5836 | 15.2642 | 13.19% |

The across-reference Full-action reduction is **14.72% +/- 0.96% SE**. Move
action falls by 12.84% +/- 0.22% SE and reaction action by 15.06% +/- 1.11%
SE. Validation finite-measure risk decreases slightly for Full on all three
seeds; the 5% budget is an allowed upper bound, not a target that must be used.

These values remain useful as a diagnostic baseline but are superseded as an
evaluation protocol. The strengthened seed-20260818 diagnostic is analyzed in
`strengthened_optimization_diagnosis.md`: it improved exact selection Full
action by 29.39% but was 5.73% worse than Law on the held-out physical split.
That reversal motivated robust physical-view selection.

The historical aggregate statistics and figure remain under
`outputs/run/evaluation/`. They should not be substituted for the new
authoritative Pareto artifacts.

## Authoritative gates and artifacts

The finalizer fails closed unless all selected Law, Tangent, and Full designs
have matching exact audits, every view-specific risk ceiling passes, the Full
selection curve is nested, all validation trials pass calibration/ESS and
physical screened-PDE residual gates, and the weighted plus/minus
move/reaction decomposition equals total action. By default it reconstructs
the experiments from frozen inputs and independently recomputes every unique
selected geometry on selection and validation before hashing every frozen
input. The `--skip-recompute` option is reserved for quick smoke checks.

| Path | Purpose |
|:---|:---|
| `frozen_inputs/` | defect/physical banks, references, CRN banks, view indices, effective configuration, hashes |
| `risk_*pct/result.json` | exact selection receipts and post-freeze validation rows for one allowance |
| `pareto.csv`, `pareto.json` | sweep checkpoints |
| `authoritative_certification_diagnostic.json` | fail-closed gate results |
| `authoritative_pareto.*` | final table, machine-readable results, and figure |

Core implementation files are `percentage_selection.py` (exact candidate
selection), `robust_selection.py` (physical/reference aggregation),
`run_pareto.py` (nested sweep and validation ordering), and
`finalize_authoritative_pareto.py` (publication gate).

## Read-only saved-result evaluation

From the repository root:

```bash
.venv/bin/python experiments/active_nematic_unbalance_percentage/eval.py
.venv/bin/python experiments/active_nematic_unbalance_percentage/eval_pareto.py
```

These commands only read `published_results.json` and print the saved result.
They do not generate banks, train references, optimize designs, validate new
geometries, write figures, or modify outputs. The compact snapshot transcribes
the authoritative tracked tables because the original ignored production tree
is no longer present in this checkout. Both commands use the repository-wide
saved-evaluator table style and include Law, Tangent, Full, and the available
SD/SE uncertainty. Normal output contains no gate-status labels.
