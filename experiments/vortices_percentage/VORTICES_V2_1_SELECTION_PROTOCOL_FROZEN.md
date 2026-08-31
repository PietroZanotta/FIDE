# Vortices V2.1 feasibility-first prospective protocol

Status: **FROZEN BEFORE ANY V2.1 SELECTION BANK**

This amendment defines a new prospective selection experiment. It does not
continue or overwrite the failed V2 version-1 experiment. It is bound by the
V2.1 manifest written and verified before generation of the namespace-11
selection bank.

## Authority and data roles

- The failed V2 namespace-`410000101` bank is DEVELOPMENT data only.
- The new namespace-`11` bank is SELECTION data only.
- The namespace-`12` bank is a DEVELOPMENT/STRESS-TEST bank only.
- The namespace-`13` bank is the single FINAL VALIDATION bank.
- Historical V1, failed V2, frozen references, and numerical-development
  artifacts remain immutable.

The three frozen training references (`310000101`, `310000102`, `310000103`),
their rollout seeds (`310003102`, `310003103`, `310003104`), their qualification
receipts, and `h_common = 0.058816544123815116` are reused exactly and are not
retrained or recomputed.

## Frozen numerical method

V2.1 retains the V2 reflected/Neumann hard empirical information projection
without alteration: `s = partial_t(q) + div(q u)`, identical four-pair even
reflection for `q` and signed `s`, matched reflected flux `j`, correction
`delta = -grad(psi)`, and `K(q_h) psi = -s_h` with homogeneous Neumann boundary
conditions. Exact scientific evaluations use float64, zero density floor,
`256 x 128`, and all 21 time nodes. The Full proxy remains `64 x 32` with the
same physical bandwidth and all 21 nodes.

## Two-digit randomness

All newly chosen V2.1 seeds and namespaces are integers from 10 through 99.
The complete schedule and outcome-blind allocation rule are frozen in
`VORTICES_V2_1_RANDOMNESS_PROVENANCE.md` and the V2.1 config. Immutable
historical seeds are retained verbatim. No alternate schedule may be tested.

## Selection bank

Generate exactly one shared 128-trial bank with generation seed `10` and
namespace `11`. It uses 2,000 sampled truth particles, the nine acquisition
nodes `[0,2,5,8,10,12,15,18,20]`, four Gaussian sensors, detector-noise standard
deviation `.005`, and exact endpoints. All methods and references share trial
identities. Hash the bank immediately; never regenerate it.

## Population, Law, and Tangent

Rerun Population, Law, and all six Tangent allowances from scratch with the
unchanged V2 candidate budgets, objectives, gates, prefixes, and nesting rules.
Freeze `eta_population`, `L_star`, `L_max = L_star + .025`, `eta_law`, `R_star`,
and `R_max(p) = R_star + p/100 abs(R_star)` for `p=.5,1,2,3,4,5`.

## The sole scientific selection change: feasibility first

For every unique Full candidate at each allowance, compute exact equal-reference
Population score `L` and exact 64-trial finite Law risk `R` before any Full
proxy pruning. A candidate is feasible only if all geometry/numerical gates
pass, `L <= L_max`, and `R <= R_max(p)`. Compute and rank the reflected Full
proxy only for feasible candidates.

At `.5%`, the new Law and current Tangent geometries are mandatory exact Full
finalists. At later allowances, the previous tighter Full incumbent and current
Tangent geometry are mandatory. Deduplicate and fill remaining positions by
feasible proxy rank, up to eight unique finalists; if fewer than eight feasible
candidates exist, evaluate all. Exact finalists use all 128 trials, all three
references, `256 x 128`, all 21 nodes, and every frozen numerical gate.

Select the minimum equal-reference mean exact Full action subject to the frozen
`1e-6` incumbent replacement tolerance. At `.5%`, selected Full must not exceed
the mandatory Law action within tolerance. Later Full winners must be nested
non-increasingly. Any violation is a numerical/implementation failure, not a
license to alter the protocol.

## Selection stop and winner freeze

A lack of a valid Population, Law, Tangent, or Full winner under these rules is
a prospective selection FAIL for V2.1. Preserve it and do not generate final
validation. Scientific definitions, candidate counts, optimization steps,
allowances, proxy, and finalist budget may not change after the namespace-11
bank exists.

If selection passes, atomically write and hash `frozen_winners.json` with full
coordinates, objectives, caps, gates, candidate provenance, finalist and
feasibility audits, incumbent decisions, and every input hash. An independent
verifier must reconstruct the decisions from raw receipts before later stages.

## Required development stress test

Before final validation, generate exactly one shared 256-trial namespace-`12`
STRESS-TEST bank and evaluate selected Law and six Full designs exactly across
all references. Report means, SEs, quantiles, maxima, max/median, top-1% share,
reference differences, instantaneous spikes, and numerical validity. This bank
is development data and cannot confirm the method.

If stress testing exposes a mechanism requiring a scientific-method change,
V2.1 stops, the stress bank becomes development evidence, and the repair must
be independently verified, incremented to V2.2 (at most V2.3), frozen, and
given fresh affected selection/stress data. At most two stress-repair cycles
are permitted. No final bank may be opened until the final method and sample
size are frozen.

## Final validation and inference

If every prior gate passes without a method change, write and hash a validation
execution receipt before data generation. Then generate the namespace-`13`
shared 1,024-trial FINAL VALIDATION bank exactly once. Evaluate common Law and
the six frozen Full geometries for all three references using only the exact
V2 action.

For each reference and allowance compute
`D = 1 - mean(A_full) / mean(A_law)` using arithmetic means. Use 100,000 paired
bootstrap resamples with two-digit seed `15`; every resample uses one common
1,024-index vector for all references, Law, and allowances. Use the frozen
max-absolute-deviation simultaneous 95% family across all 18 effects.

PASS requires exactly three qualified references, exactly 1,024 trials, every
evaluation numerically valid, all 18 simultaneous lower bounds positive,
maximum simultaneous half-width at most `.05`, every within-reference Law and
Full relative SE at most `.10`, and no outcome-dependent amendment. Statistical
failure is reported without censoring, trimming, winsorization, extra trials,
or another final holdout.

## Failure policy

Infrastructure repairs require an issue record, a minimal reproduction, and
deterministic evidence of scientific equivalence. Numerical or scientific
changes require a new method version and fresh affected data. Failed versions
and banks are never overwritten. The Toy benchmark and V1 are out of scope and
must remain hash-identical to their frozen authorities.
