# Fresh-Bank Robustness Study for the Frozen Skyrmion Candidate Pool

Date completed: 2026-08-25

Status: **complete, development-only**

Interpretation: **`MODERATE_FRESH_BANK_SUPPORT`**

This document reports the prospectively frozen fresh-bank robustness experiment for the fixed-feature `K=280` skyrmion Galerkin project. It is a development diagnostic, not an official Pareto continuation, official selection, certification, validation result, Tangent optimization, Full optimization, eigensolve, or Deep Ritz run.

## Executive conclusion

The low-risk feasible region found by the candidate-coverage study is real enough to recur on independent banks, but it is not uniformly stable at the tightest allowances.

At the 0.5% risk allowance:

- 1,243 of 4,433 candidates pass at least one fresh replicate pair.
- 1,145 candidates pass at least 16/32 pairs.
- 55 candidates pass at least 24/32 pairs.
- No candidate passes at least 28/32 pairs.
- The maximum is 26/32, achieved by three candidates.
- Four of the 32 replicate pairs have zero complete 0.5% survivors.
- The median replicate nevertheless has 1,116 complete 0.5% survivors.
- The six old-bank 0.5% witnesses pass 24--26 of the 32 fresh pairs.

This is much stronger than an old-bank-only artifact: viable candidates recur, the old witnesses mostly generalize, and a symmetry-aware shortlist contains candidates from local-cloud, risk-tangent, and periodic-path generation sources. However, the four zero-survivor replicates, the absence of any 28/32 candidate at 0.5%, the best candidate's performed-pair p10 robust rESS of 0.04786, and substantial screen-to-audit disagreement prevent a strong-support interpretation.

The evidence is sufficient to **consider** a separately named, prospectively frozen official partial-curve Galerkin protocol. It is not permission to launch one automatically.

## Scientific question

The previous development-only candidate-coverage study expanded the original 337-candidate v2 pool by 4,096 deterministic targeted geometries. On the old v2 development banks, the frozen combined pool had the following complete dual-bank survivors:

| Allowance | Old-bank survivors |
|---:|---:|
| 0.5% | 6 |
| 1% | 157 |
| 2% | 774 |
| 3% | 1,140 |
| 4% | 1,532 |
| 5% | 1,759 |

That study was classified `COVERAGE_LIMITED`, because targeted coverage found feasible candidates that the original 337-candidate pool missed. Since the generator had been informed by diagnostics on those old banks, the present study asks whether those regions persist when the candidate pool is frozen first and then tested on completely fresh, independent screen/audit bank pairs.

## Prospective freeze and provenance

The combined pool was reconstructed as exactly 4,433 unique canonical geometries:

- 337 original v2 candidates;
- 4,096 candidate-coverage geometries;
- zero duplicate canonical geometries;
- no new generation, perturbation, interpolation, Sobol sampling, path refinement, or adaptation.

The immutable candidate-coverage inputs matched their expected SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| `generator_spec.json` | `e889aae23c7649f579c8108088441eb68318abb65d8bfe1d49557c2f9aed9600` |
| `candidate_pool.json` | `da5b07e16c9c44d1e44d7831c6badb3a8a5218e6198d1cb1a7c0963a995db5e9` |
| coverage `summary.json` | `2278c34d366a71f37c70a0c8ec30376a7788a24526dce042502b5c108fcc744e` |
| coverage `inventory.json` | `fe5559527652c0b99f20d61cf9de9f29ca0e09256d08e17cf1c3f93f0da3808b` |

The fresh study seals are:

| Seal | SHA-256 |
|---|---|
| `candidate_freeze.json` | `3fae7f1cc7479d0d5413f89838aba9b0ccd8d24374dd27de699d780e5a3e1f4d` |
| canonical candidate rows | `34ffbe831fbdf2e1dd2ccffc0913dfb108af70ee5a5d0787468dfd5d4bac5eb1` |
| `bank_manifest.json` | `9e2d30fa15ed29c27e415b032ce9bd4a7b4c673bc9dda1891cc8c8f7201845d3` |
| `bank_inventory.json` | `e5dd4f14e84b1cbc8e74af1a477a415490b98258bc0d7e4332464737ac02338d` |

All 64 bank seeds were serialized and sealed before candidate evaluation. The 32 screen seeds and 32 audit seeds are unique, each screen/audit pair is distinct, all initial-state hashes are pairwise distinct, and the namespace is development-only. No replicate was dropped, replaced, or added after observing results.

## Fixed scientific contract

The study retained the existing values and semantics:

| Quantity | Frozen value |
|---|---:|
| Law selection risk | `5.186549474478042` |
| Allowances | `0.5, 1, 2, 3, 4, 5%` |
| Minimum rESS | `0.05` exactly |
| Projection residual tolerance | `2e-6` |
| Forcing mean tolerance | `2e-7` |
| Maximum covariance condition | `1e10` |
| Screen size | `N=8192` |
| Audit size | `N=16384` |
| Replicate pairs | `B=32` |
| Arithmetic | `float64` |

Scientific risk uses the authoritative fixed selection projection bank, matching the established v2/v3 selection semantics. The fresh screen/audit banks recompute projection, support, forcing, covariance, and rESS quantities; they do not redefine the fixed Law risk anchor. Thus fresh-bank rESS is independent while risk arithmetic remains exactly comparable to the historical allowance convention.

Every candidate was screened on every replicate. Candidates inside the exact 5% risk ceiling and passing all screen support gates proceeded to that replicate's independent audit. Complete eligibility at allowance `p` requires the exact risk ceiling plus geometry, projection, rESS, forcing, and covariance gates on both banks.

## Main allowance-level result

| Allowance | Ever pass | >=16/32 | >=24/32 | >=28/32 | >=30/32 | 32/32 | Best | Best candidate | Median robust rESS | p10 robust rESS | One-sided 95% Wilson lower bound |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 0.5% | 1,243 | 1,145 | 55 | 0 | 0 | 0 | 26/32 | `coverage_0638` | 0.057393 | 0.047863 | 0.676470 |
| 1% | 1,627 | 1,498 | 310 | 12 | 0 | 0 | 29/32 | `coverage_1913` | 0.060057 | 0.050508 | 0.787251 |
| 2% | 2,381 | 2,209 | 957 | 199 | 79 | 0 | 31/32 | `coverage_0429` | 0.065513 | 0.053004 | 0.871418 |
| 3% | 2,809 | 2,612 | 1,314 | 398 | 187 | 0 | 31/32 | `coverage_0620` | 0.071288 | 0.062417 | 0.871418 |
| 4% | 3,234 | 3,017 | 1,663 | 666 | 247 | 11 | 32/32 | `coverage_0782` | 0.082301 | 0.068541 | 0.922043 |
| 5% | 3,475 | 3,252 | 1,886 | 878 | 340 | 57 | 32/32 | `coverage_0436` | 0.089208 | 0.072274 | 0.922043 |

The Wilson bounds are descriptive only and are not a new gate. Three candidates tie at 26/32 for the best 0.5% pass count, and three tie at 29/32 for the best 1% count.

The allowance dependence is clear. Universal 32/32 candidates first appear at 4%, while no candidate is universal at 3% or below. At 0.5%, a substantial recurrent set exists, but even its best members fail six replicate pairs.

## Replicate-to-replicate variability

| Allowance | Minimum | p10 | Median | p90 | Maximum | Zero replicas | Replicas >=1 | Replicas >=5 | Replicas >=10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5% | 0 | 0.4 | 1,116.0 | 1,227.6 | 1,237 | 4 | 28 | 27 | 26 |
| 1% | 0 | 71.5 | 1,469.5 | 1,602.9 | 1,615 | 2 | 30 | 30 | 29 |
| 2% | 0 | 316.0 | 2,165.0 | 2,335.9 | 2,360 | 1 | 31 | 31 | 31 |
| 3% | 105 | 446.1 | 2,536.0 | 2,749.1 | 2,780 | 0 | 32 | 32 | 32 |

The 0.5% survivor distribution is strongly bimodal-looking at the replicate level: many pairs admit roughly 1,100--1,230 candidates, while several admit very few or none. This is not an implementation failure; all such banks were prospectively frozen and retained.

### Per-replicate screen/audit outcome counts

| Replicate | Candidates audited | 0.5% eligible | 1% eligible |
|---:|---:|---:|---:|
| 00 | 3,415 | 1,228 | 1,604 |
| 01 | 2,265 | 180 | 492 |
| 02 | 3,518 | 1,237 | 1,615 |
| 03 | 2,393 | 433 | 748 |
| 04 | 3,480 | 1,229 | 1,608 |
| 05 | 3,308 | 1,143 | 1,506 |
| 06 | 3,158 | 1,216 | 1,574 |
| 07 | 3,392 | 0 | 70 |
| 08 | 3,315 | 336 | 586 |
| 09 | 273 | 0 | 0 |
| 10 | 3,401 | 0 | 5 |
| 11 | 3,420 | 1,202 | 1,570 |
| 12 | 2,534 | 378 | 716 |
| 13 | 3,473 | 117 | 394 |
| 14 | 3,299 | 1,167 | 1,524 |
| 15 | 569 | 4 | 85 |
| 16 | 3,474 | 1,183 | 1,537 |
| 17 | 3,424 | 1,233 | 1,608 |
| 18 | 3,395 | 1,218 | 1,591 |
| 19 | 3,318 | 7 | 125 |
| 20 | 3,378 | 1,089 | 1,433 |
| 21 | 1,590 | 14 | 148 |
| 22 | 3,440 | 1,206 | 1,552 |
| 23 | 3,364 | 1,207 | 1,572 |
| 24 | 2,343 | 225 | 553 |
| 25 | 3,384 | 1,192 | 1,555 |
| 26 | 3,444 | 311 | 655 |
| 27 | 3,074 | 1,172 | 1,510 |
| 28 | 2,579 | 0 | 0 |
| 29 | 3,410 | 1,224 | 1,593 |
| 30 | 3,421 | 1,209 | 1,579 |
| 31 | 1,099 | 107 | 255 |

## The six old-development 0.5% witnesses

| Candidate | Old risk | Old screen rESS | Old audit rESS | Fresh passes | Fraction | Fresh performed-pair minimum | p10 | Median | p90 | Maximum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `coverage_2895` | 5.210932 | 0.064873 | 0.053806 | 26/32 | 0.81250 | 0.030516 | 0.049427 | 0.056426 | 0.061621 | 0.067709 |
| `coverage_3062` | 5.209786 | 0.063475 | 0.051918 | 24/32 | 0.75000 | 0.029697 | 0.047189 | 0.055635 | 0.061117 | 0.064704 |
| `coverage_0638` | 5.209268 | 0.067690 | 0.051660 | 26/32 | 0.81250 | 0.026946 | 0.047863 | 0.057393 | 0.061622 | 0.064683 |
| `coverage_2893` | 5.199692 | 0.064316 | 0.051410 | 26/32 | 0.81250 | 0.029021 | 0.047477 | 0.055033 | 0.060485 | 0.065827 |
| `coverage_0771` | 5.209564 | 0.068332 | 0.050444 | 25/32 | 0.78125 | 0.029151 | 0.046215 | 0.057637 | 0.061998 | 0.064660 |
| `coverage_1958` | 5.212301 | 0.068338 | 0.050150 | 24/32 | 0.75000 | 0.029137 | 0.047626 | 0.057516 | 0.062440 | 0.065327 |

The numerical fresh robust-rESS distributions use performed audits only; a screen failure remains a categorical replicate failure and is not numerically imputed.

For these six witnesses, every failure is an rESS-threshold failure:

| Candidate | Complete pass | Screen rESS <0.05 | Audit rESS <0.05 | Other gate or risk failures |
|---|---:|---:|---:|---:|
| `coverage_2895` | 26 | 2 | 4 | 0 |
| `coverage_3062` | 24 | 4 | 4 | 0 |
| `coverage_0638` | 26 | 2 | 4 | 0 |
| `coverage_2893` | 26 | 2 | 4 | 0 |
| `coverage_0771` | 25 | 3 | 4 | 0 |
| `coverage_1958` | 24 | 4 | 4 | 0 |

There were no witness failures from the fixed 0.5% risk ceiling, geometry, projection/support other than rESS, forcing, or covariance. The limiting mechanism is reference-bank rESS variability.

## Screen versus audit instability

| Allowance | Inside-risk candidate-pairs | Screen>=0.05, audit>=0.05 | Screen>=0.05, audit<0.05 | Screen<0.05 | Mean audit-screen | Median | p10 | p90 | Pearson | Spearman |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5% | 41,216 | 22,467 | 7,365 | 11,384 | -0.004222 | -0.004197 | -0.015738 | 0.005475 | 0.146 | 0.093 |
| 1% | 53,696 | 31,363 | 9,053 | 13,280 | -0.003935 | -0.002929 | -0.016042 | 0.006486 | 0.186 | 0.140 |
| 2% | 78,464 | 50,522 | 11,329 | 16,613 | -0.003524 | -0.002222 | -0.016762 | 0.007226 | 0.289 | 0.263 |
| 3% | 92,544 | 61,701 | 12,395 | 18,448 | -0.003345 | -0.002225 | -0.016853 | 0.007932 | 0.360 | 0.340 |
| 4% | 106,208 | 72,908 | 13,221 | 20,079 | -0.003294 | -0.002407 | -0.017062 | 0.008535 | 0.445 | 0.413 |
| 5% | 114,240 | 79,811 | 13,539 | 20,890 | -0.003192 | -0.002367 | -0.017135 | 0.009006 | 0.531 | 0.482 |

At 0.5%, 7,365 candidate-replicate pairs clear screen rESS but fail audit rESS. The audit minus screen difference has a negative mean of about 0.0042, but its p90 is positive; neither bank mathematically dominates the other. The low 0.5% Pearson and Spearman associations show that near-threshold ordering is quite unstable across independent banks.

## Controlling physical time node

There are 13 physical time nodes. The prospective rESS argmin diagnostics show a strong concentration at indices 7 and 8.

| Population | Time 6 | Time 7 | Time 8 | Other times |
|---|---:|---:|---:|---:|
| All 141,856 screen trajectories | 9,511 | 102,461 | 25,705 | 4,179 |
| 0.5%-inside-risk screen trajectories | 1,418 | 32,183 | 7,615 | 0 |
| Screen trajectories for >=24/32 0.5% candidates | 56 | 1,459 | 245 | 0 |
| All 93,350 performed audit trajectories | 496 | 80,640 | 12,204 | 10 |
| 0.5%-inside-risk performed audits | 281 | 25,612 | 3,939 | 0 |
| Performed audits for >=24/32 0.5% candidates | 20 | 1,423 | 126 | 0 |

Time index 7 controls roughly 72% of all screen trajectories and 86% of all performed audits. The concentration becomes even stronger among the high-pass 0.5% candidates, so the limiting rESS geometry is predominantly associated with one late-middle physical time node rather than uniformly distributed over the trajectory.

## Law diagnostics on every bank pair

The fixed Law geometry was screened on all 32 screen banks. The generic candidate-audit subset initially omitted Law from nine audits when Law failed the screen rESS gate. Because the scientific contract separately required Law diagnostics on every bank, a diagnostic-only completion pass evaluated Law on exactly those nine already frozen audit banks. It did not recompute candidate eligibility or modify any sealed candidate artifact.

| Law statistic | Screen | Audit |
|---|---:|---:|
| rESS threshold passes | 23/32 | 24/32 |
| Minimum rESS | 0.036046 | 0.023456 |
| p10 | 0.045953 | 0.043210 |
| Median | 0.054609 | 0.054733 |
| p90 | 0.062215 | 0.062574 |
| Maximum | 0.066981 | 0.065089 |
| Controlling index 7 | 25/32 | 28/32 |
| Controlling index 8 | 6/32 | 4/32 |

Law clears both independent rESS gates on 17/32 pairs. This is consistent with the broader conclusion that the exact 0.05 support threshold is materially sensitive to the finite reference bank even near the Law geometry.

The supplemental seal is:

- `law_all_bank_diagnostics.json`: `09a166e54c09fb5e28eefad83aa3a068f16c5935882f1db5d0d6e776de950f0b`
- supplemental audit replicate IDs: `01, 03, 09, 12, 15, 21, 24, 28, 31`
- cache-hit rerun: successful

## Symmetry-aware geometric diversity

The shortlists use the existing periodic/permutation-aware max-min metric, never flattened Euclidean distance. Each list is deterministic and capped at ten representatives.

### 0.5%, candidates passing at least 24/32

| Candidate | Passes | Median robust rESS | Source |
|---|---:|---:|---|
| `coverage_0638` | 26 | 0.057393 | local cloud |
| `coverage_0102` | 24 | 0.055805 | local cloud |
| `coverage_0213` | 24 | 0.054276 | local cloud |
| `coverage_3335` | 24 | 0.054154 | risk tangent |
| `coverage_2895` | 26 | 0.056426 | risk tangent |
| `coverage_0771` | 25 | 0.057637 | local cloud |
| `coverage_2385` | 24 | 0.056506 | periodic path |
| `coverage_2925` | 24 | 0.054226 | risk tangent |
| `coverage_1856` | 24 | 0.055265 | periodic path |
| `coverage_2859` | 24 | 0.054094 | risk tangent |

There are no 0.5% candidates at either the >=28/32 or >=30/32 level.

### 1%, candidates passing at least 28/32

`coverage_1913`, `coverage_0767`, `coverage_2897`, `coverage_0121`, `coverage_2403`, `coverage_0627`, `coverage_0759`, `coverage_0610`, `coverage_2408`, and `coverage_2395`.

This shortlist spans periodic-path, risk-tangent, and local-cloud sources. There are no 1% candidates at >=30/32.

### 2%, candidates passing at least 30/32

`coverage_0429`, `coverage_2259`, `coverage_0642`, `coverage_0635`, `coverage_2434`, `coverage_0640`, `coverage_0632`, `coverage_0181`, `coverage_0550`, and `coverage_0622`.

### 3%, candidates passing at least 30/32

`coverage_0620`, `coverage_0295`, `coverage_0218`, `coverage_0304`, `coverage_0142`, `coverage_3306`, `coverage_1413`, `coverage_0296`, `coverage_0845`, and `coverage_0133`.

The 0.5% shortlist demonstrates more than one symmetry-aware separated representative and multiple generation sources, but the candidates still occupy the deliberately targeted low-risk neighborhood. It is therefore safer to say the feasible region has nonzero geometric breadth than to assert that it contains several well-separated physical basins.

Full eta vectors, p10 robust rESS values, risks, risk increases, and generation sources for every shortlist level are stored in `allowance_summary.json`.

## Performance and resumability

| Work | Recorded time |
|---|---:|
| Generate and seal 64 banks | 747.8 s (12.5 min) |
| 32 complete screen evaluations | 4,019.9 s (67.0 min) |
| 32 variable-size audit evaluations | 2,769.0 s (46.1 min) |
| Aggregate summary | 52.5 s |
| Total recorded core work | 7,590.1 s (126.5 min) |

The run evaluated 141,856 candidate-screen trajectories and 93,350 performed candidate-audit trajectories, for 235,206 total candidate-bank trajectories. The isolated output tree occupies approximately 5.0 GiB.

Every screen and audit has its own NPZ, summary, SHA-256, and replicate inventory. A full post-completion workflow rerun verified all 32 screen and all 32 audit stages as cache hits and reproduced the aggregate result exactly.

## Firewalls and negative work

The top-level result records and tests the following:

- `development_only = true`
- `candidate_pool_frozen = true`
- `candidate_count = 4433`
- `replicate_count = 32`
- `validation_accessed = false`
- `tangent_optimization_run = false`
- `full_kf_constructed = false`
- `eigensolve_run = false`
- `deep_ritz_run = false`
- `official_protocol_created = false`
- `selection_frozen = false`

No official bank, protocol tree, selection hash, or validation artifact was created. `K=280` and the frozen feature dictionary remain unchanged, but no Full `K/f` system was assembled.

## Verification

The completed run passed:

- 24/24 dedicated fresh-bank robustness tests;
- 5/5 supplemental all-bank Law diagnostic tests;
- 107/107 combined Pareto-v2, Pareto-v3, candidate-coverage, and fresh-bank regression tests;
- Python bytecode compilation for all new modules;
- deterministic cache-hit reruns;
- `git diff --check` before report creation.

JAX emitted a CUDA plugin-discovery warning in the sandboxed CPU test process, but both affected commands exited successfully and every test passed. The GPU scientific evaluations themselves completed successfully.

## Artifact map

All scientific results are isolated under:

`experiments/skyrmions_deep_ritz_full/outputs/skyrmion_galerkin_dev_fresh_bank_robustness_v1/`

Principal artifacts and final hashes:

| Artifact | SHA-256 |
|---|---|
| `candidate_robustness_summary.json` | `99121aed7b18d70128cee7cdcc9d4d61dfd64e98d68d25ffb9131164c1a0db77` |
| `allowance_summary.json` | `e8f21ec7f78a4d1996505c6bed9e85e5b2f7b2ef57752b9af401388d63231a0f` |
| `failure_mode_summary.json` | `91281495a59de92c7a5cd3bb7a1ba8b540860ddda5fe0537b5ffc88595b7205e` |
| `time_node_summary.json` | `32ac42f574c054d07ae198c21c2fdd7af7a16364487e6ab900fe8b1708e18d53` |
| `summary.json` | `998a59b5bdb195e15379be085b60d31c5d5c6edd8edf63f78a64ea351f1e8740` |
| `inventory.json` | `3d837c2ea4108283749bdfa1d661e0a90ebd5ad39693ea942641c23f94df466e` |
| `law_all_bank_diagnostics.json` | `09a166e54c09fb5e28eefad83aa3a068f16c5935882f1db5d0d6e776de950f0b` |

The candidate-level summary contains all 4,433 rows, integer pass counts, pass fractions, performed-audit-only rESS distributions, audit rESS distributions, and one-sided 95% Wilson lower bounds. Row-level screen and audit results remain in their sealed per-replicate compressed NPZ files.

## Development interpretation and next step

The appropriate label is **`MODERATE_FRESH_BANK_SUPPORT`**.

Evidence supporting that label:

- viable low-risk candidates recur across independent fresh bank pairs;
- 55 candidates reach at least 24/32 at 0.5%;
- all six old 0.5% witnesses reach 24--26/32;
- the 0.5% robust shortlist has nonzero symmetry-aware breadth and multiple generation sources;
- the median fresh pair has a large 0.5% feasible set.

Evidence against `STRONG_FRESH_BANK_SUPPORT`:

- four fresh pairs have no 0.5% survivor;
- no 0.5% candidate reaches 28/32;
- the best 0.5% candidate's performed-pair p10 robust rESS is below 0.05;
- screen/audit rESS disagreement is substantial near the threshold;
- even Law passes both independent rESS gates on only 17/32 pairs.

The next scientific step may be to consider a **separately named and prospectively frozen official partial-curve protocol** with:

1. fresh official banks;
2. independent allowance failure/success;
3. a frozen improved candidate-generation strategy;
4. Tangent and Full as separate branches;
5. `K=280` Full only after cheap eligibility gates;
6. selection freeze before any validation access.

That official protocol was not created or launched here.
