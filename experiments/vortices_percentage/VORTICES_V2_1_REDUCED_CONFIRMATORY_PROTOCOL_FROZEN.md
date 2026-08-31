# Vortices V2.1 reduced 0.5--2% confirmatory protocol

Status: `FROZEN_BEFORE_CONFIRMATORY_BANK_GENERATION`

Experiment identifier: `V2.1-C3`

## Decision history and claim

The completed namespace-17 DEVELOPMENT study motivated a prospective reduction
of the scientific scope to the already-selected `0.5%`, `1%`, and `2%` points.
Those development outcomes are not reused as confirmation.  This new experiment
tests the frozen common Law geometry against the three frozen Full geometries
on a fresh bank.  Its claim is therefore limited to this partial Pareto front;
it makes no claim about `3%`, `4%`, or `5%`.

The physical model, truth population, three qualified references, bandwidth,
moment reconstruction, Full action, grid, numerical gates, selected geometries,
and arithmetic-mean estimand are unchanged from V2.1.  The original V2.1
selection state remains paused after 2%; this experiment must not create or
modify any 3--5% selection artifact or use namespace 12 or 13.

## Frozen inputs

- Common Law winner from
  `selection/law_refinement_1_feasible_anchor/law/current_result.json`.
- Atomic `PASS` Full winners at `0.5%`, `1%`, and `2%` from the corresponding
  allowance receipts.
- Exactly three frozen qualified reference flows with historical training seeds
  `310000101`, `310000102`, and `310000103`. These identifiers are retained for
  provenance and are not newly assigned randomness.
- The V1 frozen truth population and the V2.1 exact reflected physical solver.

Every input and runner is hash-bound in an execution receipt written before the
confirmatory bank.  The bank and cell receipts are write-once and atomic.

## Fresh two-digit randomness

- observation generation seed: `19`
- observation namespace: `20`
- paired common-index bootstrap seed: `21`

Generate exactly one shared 1,024-trial bank using
`numpy.default_rng(SeedSequence([19, 20]))`. Each trial retains the unchanged
2,000 truth particles, nine acquisition nodes on the 21-node grid, four
observables, and detector-noise construction. Ordered trial IDs `0:1024` are
shared across all references and all four designs.

## Exact evaluation and numerical gate

Evaluate Law and the three Full geometries for all three references using the
unchanged exact V2 Full action on the `256 x 128` grid. Retain all ordered trial
actions, action-by-time values, and diagnostics. Every one of the 12,288
reference/design/trial evaluations must be numerically valid under the frozen
V2.1 gates. No trimming, winsorization, censoring, deletion, replacement,
post-hoc cap, extra trial, or outcome-dependent rerun is permitted.

Finite Law risk is a prespecified secondary cross-evaluation on the same shared
bank. It is descriptive confirmation of where the frozen points lie; it does
not replace the selection-bank risk certificates.

## Primary inference

For each reference `r` and included allowance `p`, compute

```text
D_r,p = 1 - mean_i A_full(r,p,i) / mean_i A_law(r,i).
```

Use exactly 100,000 paired bootstrap resamples with seed `21`. Each replicate
draws one common 1,024-index vector and applies it to all three references, Law,
and all three Full designs. The primary 95% simultaneous family contains all
nine reference-by-allowance effects. Its critical value is the 95th percentile
of the maximum absolute unstudentized deviation from the nine observed effects.

Report arithmetic means, standard errors, relative standard errors, pointwise
intervals, simultaneous intervals, equal-reference descriptive summaries, and
between-reference ranges.

## Prespecified PASS rule

`V2.1-C3` passes if and only if:

1. exactly three qualified references and 1,024 shared trials are present;
2. all 12,288 exact action evaluations are numerically valid;
3. all nine simultaneous lower bounds are strictly positive;
4. the common simultaneous half-width is at most `0.05`;
5. every within-reference Law and Full arithmetic-mean relative SE is at most
   `0.10`; and
6. the bank, estimand, family, gates, designs, trial count, and stopping rule
   were not changed after any confirmatory outcome was generated or inspected.

A failure is reported without opening another bank.  Results must be labeled
as confirmation of the truncated `0.5--2%` Pareto experiment, not of the
original six-point V2.1 experiment.

## Destinations

All scientific outputs are confined to
`outputs/prospective_v2_1_confirmatory_0p5_to_2pct/`. Publication figures are
written under `experiments/vortices_percentage_v2/plots/`. The paused 3--5%
tree and original namespace-13 destination remain untouched.
