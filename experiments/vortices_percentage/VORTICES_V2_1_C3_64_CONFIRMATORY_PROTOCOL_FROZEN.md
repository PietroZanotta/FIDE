# Vortices V2.1-C3-64 reduced confirmatory protocol

Status: `FROZEN_BEFORE_FRESH_64_TRIAL_BANK_GENERATION`

## Design classification and history

This is a **development-adaptive, prospectively confirmed independent-holdout
design**. The V2.1 sensor geometries and scientific method were selected before
the namespace-17 development bank. Development results then motivated two
decisions: truncate the claim to the completed `0.5%`, `1%`, and `2%` points,
and use 64 fresh trials per reference. Those development outcomes are not
reused as confirmation.

An earlier conservative 1,024-trial C3 bank (seed 19, namespace 20) was frozen
and generated, and its evaluator entered outcome computation, but it was
terminated for a time-only sample-size amendment before any action cell,
evaluation receipt, statistic, or outcome was written or inspected. That bank
is retired permanently and cannot be analyzed or resumed. C3-64 uses fresh
randomness.

## Frozen claim and inputs

The confirmatory claim is limited to the truncated 0.5--2% Pareto front. It
makes no claim about 3%, 4%, or 5%, whose original V2.1 selection remains
paused. Inputs are the common frozen Law winner and atomic `PASS` Full winners
at 0.5%, 1%, and 2%; the three qualified references; frozen truth population;
unchanged bandwidth, reconstruction, exact reflected Full action, `256 x 128`
grid, and V2.1 numerical gates.

## Prospective sample-size justification

The namespace-17 development family had simultaneous half-width
`0.0134219` at 256 trials, a smallest observed effect near `0.0767`, and maximum
relative SE near `0.0112`. Square-root scaling prospectively projects at 64
trials a simultaneous half-width near `0.02684`, a smallest lower bound near
`0.0499`, and maximum relative SE near `0.0224`. These remain well inside the
unchanged positive-lower-bound, `.05` half-width, and `.10` relative-SE gates.
This calculation fixes 64 before the fresh bank. No confirmatory top-up is
permitted regardless of the result.

## Fresh two-digit randomness

- observation generation seed: `22`
- observation namespace: `23`
- paired bootstrap seed: `24`

Generate one shared 64-trial bank with
`numpy.default_rng(SeedSequence([22, 23]))`. Every ordered trial uses the
unchanged 2,000 truth particles, nine acquisition nodes on the 21-node grid,
four observables, and detector-noise construction. Trial IDs `0:64` are shared
across all references and four designs.

## Exact evaluation

Evaluate Law and the three Full geometries for all three references using the
unchanged exact action. Retain all 768 reference/design/trial actions,
action-by-time values, and diagnostics. Every evaluation must satisfy all
frozen V2.1 numerical gates. No trimming, winsorization, censoring, deletion,
replacement, post-hoc cap, extra trial, or outcome-dependent rerun is allowed.

The already-qualified ordered four-worker parallel exact solver is allowed.
The development fused-cell benchmark is explicitly rejected because it was
faster but not bitwise equal. No fused-cell execution is allowed.

Finite Law risk on this bank is a prespecified secondary cross-evaluation and
does not replace the selection risk certificates.

## Primary inference and PASS rule

For each reference `r` and allowance `p`, compute
`D_r,p = 1 - mean(A_full_r,p) / mean(A_law_r)`. Use exactly 100,000 paired
bootstrap resamples with seed 24. Each resample applies one common 64-index
vector to all three references and all four designs. The primary 95% family is
the nine reference-by-allowance effects, using the maximum absolute
unstudentized deviation.

C3-64 passes if and only if exactly three qualified references and 64 shared
trials are present; all 768 evaluations are numerically valid; all nine
simultaneous lower bounds are positive; the common half-width is at most `.05`;
all within-reference arithmetic-mean relative SEs are at most `.10`; and no
confirmatory outcome-dependent amendment occurred. A failure is final; no
additional holdout or trial top-up may be opened.

Scientific outputs are confined to
`outputs/prospective_v2_1_c3_64/`. Plots are written under
`experiments/vortices_percentage_v2/plots/`.
