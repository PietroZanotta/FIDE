# Authoritative Deep Ritz restart-stability evaluation

This checkpoint follows `AUTHORITATIVE_GPU_ACCELERATION.md`. It asks whether
the isolated production fixed-design Deep Ritz rescorer can reliably order the
old 3% eta and the previous tiny-update eta. It does not differentiate through
Deep Ritz training, alter the validated Galerkin gradient, use validation data,
or modify a production incumbent.

## Decision

The strict paired-restart gate failed. One common-initialization pair passed all
hard certificates and favored eta0. A second common-seed pair produced finite
low-action solutions for both designs, but both failed the unchanged held-out
energy certificate and neither L-BFGS solve converged. The comparison is
therefore `indeterminate`, and no further eta refinement is authorized.

This reinforces the earlier platform result: fixed checkpoints evaluate
identically on CPU and GPU, but optimized checkpoints are basin-sensitive. The
current authoritative solver is a valid fixed-network evaluator and can produce
certified solutions, but one optimized run is not a stable oracle for action
differences on the order of `1e-3`.

## Protocol

- artifacts: complete hash-materialized frozen 3% production set;
- data: selection-side banks only;
- designs: production eta0 and the previously proposed tiny update;
- device: `cuda:0`, NVIDIA GeForce RTX 5090 Laptop GPU;
- precision: float64;
- scientific schedule: unchanged 1,800 full-bank Adam steps and 160 full-bank
  L-BFGS iterations, chunk size 512;
- hard gates: exact risk, geometry, projection, ESS, forcing, weak residual,
  energy residual, gauge, and moment-rate residual;
- decision rule: every pair must be valid, and every paired action difference
  must have the same sign outside the `1e-6` comparison tolerance;
- caching: eta, artifact manifest, solver/config, initialization, checkpoint,
  and implementation hashes are part of each restart signature.

The already completed GPU warm-start pair was reused only after its two original
signatures and initial-checkpoint SHA-256 were recomputed and matched. The new
seeded pair was checkpointed member-by-member. The run made two new
authoritative calls and reused two compatible cached calls.

## Results

The compared designs were:

```text
eta0 =
[0.8954153767761239, 0.20592631632470587,
 1.3343788098383822, 0.8654288352917223,
 0.7508355365766083, 0.5179100329264751,
 1.6423735249784726, 0.5883599695898114]

tiny update =
[0.8953839921146673, 0.20595035907471138,
 1.3345144773868762, 0.8654744150451203,
 0.7508077339024882, 0.5179727362721115,
 1.6423936578820195, 0.5884106107337586]
```

The exact risk ceiling was `5.34214595871238`.

| initialization | design | risk | action | weak | energy | gauge | moment rate | valid | L-BFGS converged |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| frozen production checkpoint | eta0 | 5.3401060510 | 0.2763324858 | 0.0783735 | 0.0692169 | 7.64e-16 | 0.0137655 | yes | no |
| frozen production checkpoint | tiny | 5.3420455203 | 0.2797088593 | 0.0773846 | 0.0695188 | 3.26e-16 | 0.0139225 | yes | no |
| fresh seed 20270829 | eta0 | 5.3401060510 | 0.2638770575 | 0.0581416 | **0.0882257** | 5.12e-16 | 0.0139039 | **no** | no |
| fresh seed 20270829 | tiny | 5.3420455203 | 0.2644522685 | 0.0534472 | **0.0931218** | 2.20e-16 | 0.0139720 | **no** | no |

The certificate limits were weak `0.12`, energy `0.08`, gauge `1e-9`, and
moment-rate `0.10`. Projection, ESS, forcing compatibility, geometry, and risk
passed for all four rows. The fresh pair fails solely at the held-out energy
gate.

## Paired action differences

| initialization | tiny minus eta0 action | pair admissible? | interpretation |
|---|---:|---|---|
| frozen checkpoint | +0.0033763735 | yes | eta0 lower on GPU |
| fresh seed 20270829 | +0.0005752111 | no | raw ordering agrees, but both certificates fail |

The one admissible GPU pair disagrees with the earlier CPU warm-start pair,
which gave `-0.0044515444` and favored the tiny update. Since CPU and GPU fixed
checkpoint evaluation agrees near `1e-15`, this sign reversal is attributed to
the optimized endpoints, not the action/certificate evaluation graph.

## Stationarity diagnostics

The fresh eta0 and tiny solves ended with train Ritz objectives
`-0.1321428206` and `-0.1322124340`. Their last recorded L-BFGS gradient norms
were `0.0339858` and `0.0272742`, far above the configured `2e-7` convergence
tolerance. Their train energy-identity relative errors were small (`4.43e-4`
and `1.88e-3`), showing that a good train identity is not sufficient for the
held-out physical certificate.

The fresh calls took `1815.680 s` and `1795.684 s`. The hash-reused warm calls
had taken `1749.166 s` and `1793.907 s`. Total new GPU time was approximately
`3611.364 s`; caching avoided approximately `3543.073 s` of duplicate GPU work.

## Consequence for eta optimization

`eligible_for_further_eta_refinement=false`. No new trust-region step,
multistart expansion, Pareto sweep, incumbent replacement, or validation rerun
was launched. The already sealed validation reversal remains historical data
and was not used to choose this outcome.

The next production-facing milestone is an inner-solver study, still inside the
isolated experiment:

1. establish a stationary or explicitly plateaued full-bank solution under a
   deterministic common-initialization family;
2. keep audit banks post hoc and fail closed—never optimize against them;
3. compare longer/alternative exact optimization schedules using the same
   objective and frozen artifacts;
4. require restart-consistent valid action ordering before promoting another
   eta;
5. only then resume the already validated fast Galerkin trust-region loop.

Machine-readable results are in
`outputs/fast_production_3pct/authoritative_stability/result.json`, with the new
network checkpoints and metadata below `restart_001/`.

The final isolated regression run covered 62 tests and all passed: 12 nonlinear
continuous-gradient checks, 17 Galerkin checks, 27 production-Galerkin checks,
and 6 accelerated/restart checks.

## Isolation

All code, reports, and output writes remain below
`experiments/skyrmions_deep_ritz_full/`. Static search found no import from the
old skyrmion experiment. `src/`, `native/`, original frozen sources, historical
production outputs, and the production incumbent were not modified.

**Checkpoint decision: AUTHORITATIVE ACTION ORDERING INDETERMINATE; FURTHER ETA
REFINEMENT BLOCKED PENDING INNER-SOLVER STABILITY.**
