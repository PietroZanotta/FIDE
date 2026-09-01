# Official B1 Galerkin Pareto V2 Single-Seed Protocol

Status before scientific execution: **PROSPECTIVE**  
Authority namespace: `outputs/official_b1_galerkin_pareto_v2_single_seed/`

## Purpose and estimand

This new authority repairs the V1 Law-anchor consistency contract without
changing V1. The Full estimand is exactly the frozen, permutation-invariant,
configuration-space **K=280 Galerkin correction action**. This protocol makes
no continuum or infinite-basis convergence claim.

## Randomness freeze

The declared seed range is `20261001` through `20261099`. Repository and Git
history searches found `20261001` and `20261002` in history; `20261003` was the
first value absent from both the working repository and relevant Git history.
It is the one root experiment seed. Every stochastic role is derived by
`jax.random.fold_in(PRNGKey(20261003), role_id)`. Derived keys are roles of one
replicate, not additional scientific seeds. No alternate root seed may be
tested.

## Frozen scientific inputs

- Reference: reuse the byte-identical accepted B1 particle-matched checkpoint
  from V1; no reference is retrained.
- Basis: reuse the byte-identical frozen K=280 hybrid invariant dictionary.
- Precision: `JAX_ENABLE_X64=1`; all scientific action is float64.
- Solver: JAX assembly and JAX symmetric rank-aware eigensolve only, with
  relative eigenvalue threshold `1e-12`. Native/Tesseract Galerkin assembly is
  forbidden as primary path, fallback, audit path, proxy, or helper.
- Projection backend: JAX.
- Allowances: exactly `0.5%`, `1%`, and `2%`.
- Law consistency tolerance: `1e-4` absolute risk.
- Complete anchor-refinement restarts: at most two.
- Replacement tolerance: `1e-10` action.

## Candidate universe

Before any new scientific data or outcome is generated, freeze and persist a
canonical, symmetry-aware, deduplicated universe of exactly 5,645 geometries:

| family | count |
|---|---:|
| historical V1 Law/Tangent/Full geometries, proposal status only | 13 |
| Law-family local cloud | 1,024 |
| Law-family scrambled Sobol | 512 |
| broad local cloud | 1,434 |
| prospectively declared tangent-direction cloud | 1,024 |
| periodic paths | 819 |
| broad scrambled Sobol | 819 |

The historical `candidate_00318` identifier is not privileged or injected.
Historical selected geometries are proposal seeds only. All rows receive the
same new exact risk and support treatment.

## Data and exact feasibility

The design truth contains 6,000 samples. Selection roles contain 32,768 exact
risk-anchor, 8,192 support-screen, 16,384 independent support-audit, 32,768
Full-search train, 16,384 Full-search audit, and independent 65,536/65,536
authoritative train/audit reference samples.

Every pool row receives exact risk on the risk-anchor bank and projection,
rESS, forcing-mean, covariance, and geometry gates on risk-anchor plus both
support banks. Full action is never evaluated before these receipts define the
jointly supported feasible set.

## Law and consistency restart

The initial Law is the unique deterministic minimum-risk jointly valid row in
the complete frozen universe. Its exact float64 risk is `R_star`; ceilings are
`R_star + (p/100)*abs(R_star)`.

After every complete Tangent/Full pass, all exact-valid downstream proposals
are added to the consistency registry. If their union with the original pool
contains risk below `R_star - 1e-4`, the unique deterministic minimum becomes
Law, all ceilings are recomputed, and all three allowances restart from
scratch. At most two complete restarts are permitted; another material
improvement after restart two is selection failure. Intermediate passes remain
immutable.

## Selection and incumbency

Starts are selected from the exact feasible set using mandatory baselines,
lowest risk, strongest robust rESS, and symmetry-aware max-min diversity—never
Full action. Tangent uses six starts and Full four. Each start receives at most
one accepted local step, three backtracks, initial step `5e-5`, and trust radius
`2e-4`. Every generated proposal receives exact risk/support before action
ranking.

Law and the current Tangent winner are mandatory Full baselines at 0.5%. At 1%
and 2%, the previous tighter Full winner and current Tangent winner are
mandatory; Law remains a fallback. Mandatory rows survive both start and
authoritative-finalist caps. An incumbent is replaced only by an improvement
larger than `1e-10`; otherwise its byte-identical receipt is retained.
Selection action must be nonincreasing within `1e-10`.

## Authoritative certification and validation

All unique selected Law/Tangent/Full geometries receive JAX-only K=280 action
and physical-residual certificates on independent 65,536 train and 65,536
audit banks. One held-out validation bank is generated only after winner
geometries are sealed: truth 5,000, reference fit 16,384, reference audit
16,384. Validation cannot modify winners.

Before validation is opened, an independent deterministic verifier
reconstructs every Full winner from the authoritative finalist receipts,
mandatory-baseline identities, and replacement rule. Validation generation is
fail-closed until that receipt passes.

## Performance and persistence

Risk candidates are processed in static batches of 8, the largest preflight-safe
shape on the execution GPU. Galerkin sufficient
statistics use fused 512-sample JAX chunks; full `[N,time,K,state]` gradients
are never materialized. Candidate batches and authoritative evaluations are
atomically checkpointed with full scientific cache keys. Required outputs
include complete candidate/feasibility receipts, every restart, selected and
certified rows, machine-readable summaries, the performance report, and the
final decision table.
