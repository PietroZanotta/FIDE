# Percentage-budget unbalanced active-nematic MFSI

This directory is an isolated, faster selection variant of
`experiments/active_nematic_unbalanced`. It does not modify the balanced
active-nematic experiment, the earlier unbalanced experiment, shared
`src/mfsi`, or any native Tesseract.

The physical model, signed-defect representation, endpoint-only reference,
finite-measure risk, tangent action, screened Full action, validity gates, and
native screened-Poisson backend are unchanged. Validated dealiased ETD2 banks
and learned references may be supplied read-only with `--input-dir`; all new
design results are written below this directory.

## Percentage risk budget

The Law stage is optimized and audited first on all declared selection trials.
Its exact result defines `R_star`. Tangent and Full are allowed a relative risk
increase rather than the old fixed additive allowance:

```text
epsilon_R = 0.05 * abs(R_star)
R_max     = R_star + epsilon_R
```

Thus every learned-reference seed receives the same 5% relative allowance,
even when its absolute risk scale differs. Only the Law stage defines
`R_star`; action candidates cannot silently replace it.

## Simplified optimizer

Selection has three direct stages:

1. Law: one modest multistart Adam search, followed by exact complete-bank
   auditing.
2. Tangent: one constrained Adam search, followed by exact risk and action
   certification.
3. Full: one low-resolution constrained Adam search, an exact Full pre-screen,
   and complete-bank rescoring of the proxy finalists plus the retained Law
   and Tangent incumbents.

There are no SciPy/L-BFGS refinement loops, projected boundary searches, or
recursive Law/action restarts in the active percentage path. The retained Law
design is included in every later candidate pool, so a certified feasible
incumbent always remains available.

Production uses 32 trials for exact selection audits and 32 independent trials
for validation. Smaller frozen prefixes are used only inside differentiable
optimization.

## Commands

The following paired runs reuse the already audited physical and reference
artifacts without changing them:

```bash
.venv/bin/python experiments/active_nematic_unbalance_percentage/run.py design \
  --smoke --reference-seeds 20260818 \
  --input-dir experiments/active_nematic_unbalanced/outputs/smoke_dealiased_etd2 \
  --output-dir experiments/active_nematic_unbalance_percentage/outputs/smoke

.venv/bin/python experiments/active_nematic_unbalance_percentage/run.py design \
  --input-dir experiments/active_nematic_unbalanced/outputs/run_dealiased_etd2 \
  --output-dir experiments/active_nematic_unbalance_percentage/outputs/run
```

To build every upstream artifact independently instead, use the same staged
`physical-bank`, `defects`, `defect-audit`, and `reference` commands accepted
by `run.py` before `design`.

## Completed production run

The three-reference production run completed with 32/32 valid independent
validation trials for every design and seed:

| Reference seed | `R_star` | `R_max` | Validation action: Law | Validation action: Full | Full reduction |
|---:|---:|---:|---:|---:|---:|
| 20260818 | 1.614866 | 1.695609 | 18.1897 | 15.5570 | 14.47% |
| 20260819 | 1.607028 | 1.687380 | 18.1388 | 15.1462 | 16.50% |
| 20260820 | 1.615106 | 1.695861 | 17.5836 | 15.2642 | 13.19% |

The across-reference Full-action reduction is **14.72% +/- 0.96% SE**. Move
action falls by 12.84% +/- 0.22% SE and reaction action by 15.06% +/- 1.11%
SE. Validation finite-measure risk decreases slightly for Full on all three
seeds; the 5% budget is an allowed upper bound, not a target that must be used.

The table above records the initial fast profile and remains available as a
baseline. The active configuration now uses stronger Law, Tangent, and Full
budgets with normalized objectives/constraints and local incumbent clouds.
The first strengthened production diagnostic is analyzed in
`strengthened_optimization_diagnosis.md`: it uses 3.10% of the allowed risk,
improves exact selection Full action by 29.39%, and exposes a separate held-out
physical-split generalization problem.

Aggregate statistics and the four-panel evaluation figure are generated at
`outputs/run/evaluation/evaluation_stats.json` and
`outputs/run/evaluation/evaluation.png`. New strengthened runs always rescore
the Law and Tangent incumbents with the complete-bank Full action, preventing a
proxy-ranked finalist set from discarding a better exact baseline.
