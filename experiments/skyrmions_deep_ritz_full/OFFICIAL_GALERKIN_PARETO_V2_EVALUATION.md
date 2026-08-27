# Official Fast Skyrmion Galerkin Pareto v2 Evaluation

## Outcome

**INCOMPLETE — genuine fail-closed blocker.** The prospective run stopped during
Tangent selection at the first allowance, 0.5%, because none of the four
declared starts passed the independent N=16384 periodic audit. The scientific
classification is **NO CERTIFIED SELECTION WINNER**. All four candidates passed
exact selection risk, geometry, search projection, search forcing, covariance,
and search rESS, but their independent-audit minimum relative ESS values were
below the frozen 0.05 threshold.

No threshold or methodological setting was changed after observing this result.
Full selection did not start, selection was not frozen, and fresh validation was
neither generated nor accessed. Consequently, a complete six-point
Law/Tangent/Full sweep and validation table cannot truthfully be reported.

Machine-readable failure record:
`outputs/official_galerkin_pareto_v2/failure.json`, SHA-256
`e73d4e3ac89f9562bc6038c8ea4c3d0d9bb040b78dfffa477a982585329acf40`.

## Repository isolation and provenance

The official protocol, banks, screening results, trajectories, and failure
record are isolated under
`outputs/official_galerkin_pareto_v2/`. Historical reports and output trees were
not overwritten. Earlier pre-selection attempts remain preserved in their
versioned failure/superseded directories and were not used as scientific
results.

At the user's later explicit request, native acceleration work was separated
outside the experiment directory. Every tracked file in
`native/iprojection_tesseract/` was restored to, and verified identical with,
`origin/main`. Candidate-specific `[C,T,N,M]` projection is isolated in
`native/candidate_iprojection_tesseract/`; Galerkin K/f assembly is isolated in
`native/galerkin_tesseract/`. The shared Python wrappers contain additive entry
points for those components. This later authorization superseded the initial
read-only constraint for only those requested native/shared additions.

Deep Ritz did not enter screening, Tangent selection, or any official decision.

## Frozen methodology and protocol

The Full objective was prospectively fixed as the finite-dimensional,
fixed-feature K=280 Galerkin approximation of the weighted-Poisson weak problem.
It must not be interpreted as a converged infinite-dimensional Full solution.
The dictionary SHA-256 is
`37e9b60fcb92c4e5a0ee7ec1651fb7f8889f7ac6bdb02d3bd314e9ef40833326`.
The relative rank tolerance is `1e-12`, minimum relative ESS is `0.05`, and
maximum held-out Ritz-energy residual is `0.08`; all other established
projection, forcing, covariance, geometry, algebra, weak, gauge, and moment-rate
thresholds remained unchanged.

The frozen machine protocol is
`outputs/official_galerkin_pareto_v2/protocol.json`:

- inner protocol seal:
  `22a33ce47b2a3cc17ff063d100b878ac32c3ef6cc1a2b3e10a6eb8cd076488f1`;
- protocol-file SHA-256:
  `8360afb812b4036cbadaa0e2ca4f12d92c10ffa6900c1a22c5f231eea18dbf3b`;
- version: `skyrmion_official_galerkin_pareto_v2_rerun3`;
- allowances: 0.5, 1, 2, 3, 4, and 5 percent;
- Galerkin assembly flag: `production_galerkin.assembly_backend = "jax"`;
- available opt-in assembly backend: `tesseract_cpp`;
- official Full K/f path: device-resident JAX/XLA;
- candidate information projection: independent C++/OpenMP Tesseract;
- optimizer: four deterministic starts, four accepted-step attempts,
  `5e-5` initial step, `2e-4` trust radius, ten halvings, and `1e-10`
  replacement tolerance;
- authoritative shortlisting: at most three finalists with independent
  N=65536/N=65536 recomputation;
- selection risk ceiling: `R <= (1+p/100) R_Law`;
- validation ceiling, had selection frozen: `R <= (1+p/100+0.05) R_Law`.

The Law geometry was frozen solely through the existing scientific-risk anchor:

`[0.890286510596537, 0.227289528868506, 1.31036883214449,
0.859163192162967, 0.797588822714243, 0.535723001316333,
1.61034315044757, 0.583219225445585]`.

Its selection risk was `5.186549474478042`; the 0.5% ceiling was
`5.212482221850432`.

## Official selection banks

The bank manifest passed its pairwise role-disjointness test and records
`validation_accessed: false`. Manifest SHA-256:
`cc81d24b01ebfcf25eaa91604ba75dc40856299697af1d47e0738667029dc6d3`.

| role | N | integer seed | seed SHA-256 | artifact SHA-256 | generation s |
| --- | ---: | ---: | --- | --- | ---: |
| screen | 8,192 | 462821422 | `93d46b71…60e93` | `05ec181f…34ea0` | 12.124 |
| search train | 32,768 | 1269066990 | `f9ae39ac…f03ed` | `cc8be9fc…3f04` | 38.608 |
| periodic audit | 16,384 | 1839249659 | `059aac7e…775b` | `e0fcbb16…42915` | 19.778 |
| authoritative train | 65,536 | 1331973801 | `20936738…6bad7` | `59987834…0effa6` | 70.227 |
| authoritative audit | 65,536 | 1100690834 | `62f0bf58…f4f7` | `3dd2f3a8…e2b294` | 69.308 |

Total recorded bank-generation time was approximately 210.0 seconds. The
reference network was frozen and not retrained.

## Predeclared validation firewall

Validation seeds were committed inside the protocol before selection:

| role | integer seed | seed SHA-256 |
| --- | ---: | --- |
| truth | 152182844 | `3b0a6859…9922e1` |
| reference fit | 1758326610 | `587ce7e3…91bdbf` |
| reference audit | 1878700624 | `f42c29be…de6a0` |
| measurement noise | 1016358654 | `c35bc328…73ebe9` |

No validation artifact exists beneath the active output root. The run stopped
before `freeze-selection`, so the validation generator's prerequisite could not
be satisfied. This proves that validation was not opened during selection.

## Stage-A risk/rESS screening

The complete deterministic pool contained 337 candidates and constructed zero
Galerkin K/f systems. It used the N=8192 screening bank and the isolated
candidate-batched native projection endpoint. Counts passing exact risk,
geometry, projection, and rESS >= 0.05 were:

| allowance | risk ceiling | feasible screened candidates |
| ---: | ---: | ---: |
| 0.5% | 5.212482222 | 193 |
| 1% | 5.238414969 | 216 |
| 2% | 5.290280464 | 242 |
| 3% | 5.342145959 | 274 |
| 4% | 5.394011453 | 292 |
| 5% | 5.445876948 | 299 |

Thus the blocker was not absence of screening-side feasible starts.
`screening/candidate_pool.json` has SHA-256
`f3ca87f8c251c9bb5aec13e5bc77885cc643a98a9dd4554f4a86c800879d18ab`.

The frozen start pools were:

| allowance | four deterministic starts |
| ---: | --- |
| 0.5% | law `candidate_000` (R=5.186549474, rESS=0.062396); historically strong `candidate_001` (5.203174625, 0.064494); best-rESS `candidate_054` (5.207900752, 0.067184); max-min diverse `candidate_095` (5.196816357, 0.054017) |
| 1% | law `candidate_000` (5.186549474, 0.062396); historically strong `candidate_002` (5.225761943, 0.066221); best-rESS `candidate_057` (5.232208311, 0.068839); max-min diverse `candidate_095` (5.196816357, 0.054017) |
| 2% | law `candidate_000` (5.186549474, 0.062396); historically strong `candidate_003` (5.284504645, 0.058587); best-rESS `candidate_089` (5.256004558, 0.074863); max-min diverse `candidate_111` (5.179213363, 0.056308) |
| 3% | law `candidate_000` (5.186549474, 0.062396); historically strong `candidate_004` (5.340106051, 0.074214); best-rESS `candidate_093` (5.320687423, 0.078310); max-min diverse `candidate_111` (5.179213363, 0.056308) |
| 4% | law `candidate_000` (5.186549474, 0.062396); historically strong `candidate_004` (5.340106051, 0.074214); best-rESS `candidate_168` (5.374692329, 0.081321); max-min diverse `candidate_107` (5.233609462, 0.064176) |
| 5% | law `candidate_000` (5.186549474, 0.062396); historically strong `candidate_004` (5.340106051, 0.074214); best-rESS `candidate_080` (5.437070198, 0.093234); max-min diverse `candidate_112` (5.408501642, 0.079046) |

## Tangent result and blocker diagnostics

At 0.5%, every start passed the search context but failed the independent
periodic audit's rESS gate before any optimization step could be taken. All
projection residuals, Gram conditions, and moment-rate residuals passed their
thresholds; only minimum rESS failed.

| role | risk | search rESS | audit rESS | audit projection residual | Tangent action | status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Law | 5.186549474 | 0.057604 | 0.044666 | 6.97e-11 | 0.111385097 | FAIL rESS |
| historically strong | 5.203174625 | 0.059398 | 0.047230 | 9.29e-11 | 0.104301147 | FAIL rESS |
| best-rESS | 5.207900752 | 0.061478 | 0.048376 | 1.97e-11 | 0.101857676 | FAIL rESS |
| max-min diverse | 5.196816357 | 0.053740 | 0.042700 | 5.32e-11 | 0.103505891 | FAIL rESS |

The audit threshold was exactly 0.05. The best observed audit rESS was
0.0483757149, still below it. No endpoint was eligible for the at-most-three
finalist list, hence there was no independently audited Tangent incumbent and
the command raised `RuntimeError: no certified Tangent winner at 0.5%`.

Trajectory hashes, respectively Law, historical, best-rESS, and diverse, are
`7acb5ffb…830a`, `9b5c1cba…8beb`, `f9fe76c6…a3c8`, and
`5eb3ed11…9963`.

## Stages not reached

The following results do not exist and are intentionally not inferred:

- authoritative Tangent finalists or a complete six-point Tangent sweep;
- Full search trajectories, periodic Full audits, or authoritative Full
  finalists;
- nested-incumbent decisions;
- selection Law/Tangent/Full cross-evaluation;
- a frozen selection manifest/hash;
- fresh-validation artifacts, risk ratios, Tangent or K280 Full validation
  actions, strict-p or p+5pp statuses;
- Full-versus-Law reductions or Full-selected versus Tangent-selected common
  K280 comparisons.

For 0.5% Tangent the classification is **NO CERTIFIED SELECTION WINNER**.
Larger allowances, all Full allowances, and validation are **not evaluated**;
assigning PASS or reversal classifications to them would be scientifically
invalid.

## Native backend flag and equivalence evidence

`production_galerkin.assembly_backend` now accepts `jax` and `tesseract_cpp`.
The official seal chose `jax`, so all eventual Full K/f assembly would have
remained on GPU. The Tesseract path changes assembly execution only; the
rank-aware coefficient solve, action identity, thresholds, and certificates are
shared.

The combined current suite reported 44 passed tests, including direct
JAX-versus-Tesseract K/f statistics, candidate projection value/VJP/JVP
equivalence, deterministic native results, invalid-backend fail-closed behavior,
and all 35 Pareto-v2 unit contracts. The broader historical
Galerkin/ESS/resolution/quadrature regression run reported 82 passed and 4
skipped. The original I-projection folder has no diff against `origin/main`.

## Performance and further optimization

Yes, further computational optimization is possible, but it cannot repair this
scientific audit failure without a new prospective protocol.

| priority | optimization | evidence / impact | semantics |
| --- | --- | --- | --- |
| **HIGH VALUE / LOW RISK** | Independent candidate-batched I-projection | Implemented and tested. At representative `[8,13,8192,4]`, measured 6.42x versus scalar native calls, with zero multiplier discrepancy and maximum residual about 4.1e-12. | unchanged |
| **HIGH VALUE / LOW RISK** | Reuse fixed-shape JITs and exact-eta/hash caches | Implemented in the v2 architecture; avoids recompilation and duplicate authoritative work. | unchanged |
| **MEDIUM VALUE** | Cache/time-shard fixed basis values and gradients where memory permits | Expected to reduce repeated basis evaluation in a completed Full sweep, at substantial host/device memory cost for K=280 and N up to 65536. | unchanged if hashes include bank/dictionary/dtype |
| **MEDIUM VALUE** | Reuse converged nearby-eta multiplier trajectories only as Newton initial guesses | Could reduce iteration count, but requires deterministic convergence/equivalence tests and cache-key discipline. | unchanged only after convergence proof |
| **NOT RECOMMENDED for GPU hot path** | Tesseract CPU/OpenBLAS Galerkin assembly | Implemented and equivalent, but transfer-inclusive measurements were only 0.137x to 0.280x the direct RTX 5090 JAX speed (about 3.6–7.3x slower). Keep as CPU fallback. | unchanged |
| **NOT RECOMMENDED** | Concurrent CPU projection/GPU K/f pipeline before profiling contention | Complexity and determinism risk are disproportionate while host callbacks force transfers. | potentially risky |

In the completed portion, deterministic reference-bank generation dominated
recorded runtime (about 210 seconds). Had Full selection been reached, prior
profiling indicates streamed K/f assembly and independent Full certification
would dominate. This latter statement is an inference from the qualification
benchmarks, not a timing from this aborted sweep.

## Scientific interpretation and limitations

The result does not show that the 0.05 rESS threshold is intrinsically
infeasible across the Pareto range. It shows that under the frozen v2 bank
realizations and declared four-start selection rule, all 0.5% Tangent starts
that passed search-side gates failed the independent periodic rESS audit. The
gap was small for the best candidate (about 0.0016243 absolute), but the
protocol explicitly forbids threshold tuning after results are observed.

A future attempt would require a separately versioned, prospectively justified
protocol—possibly one that selects starts using the already-declared independent
audit or predeclares more starts. This report makes no recommendation to do so
within v2, and no v2 result may be reused as unseen evidence in such a protocol.

The authoritative conclusion is therefore: **Pareto v2 did not complete because
there was no certified 0.5% Tangent selection winner. Fresh validation remained
sealed and no post-result substitution was performed.**
