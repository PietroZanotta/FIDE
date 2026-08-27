# Replicate-Gate Preflight v2 Report

Date completed: 2026-08-25

Status: **complete, development-only, no official launch authorized**

Recommendation: **`NO_REPLICATE_GATE_ARCHITECTURE_READY`**

## Decision

The efficient replicate-gate preflight completed successfully, but its scientific result did not support freezing any tested architecture into the next official protocol. Consequently, the conditional authoritative run was not launched.

This is not a computational failure. The sealed Boolean outcomes show that none of the prospectively frozen primary architectures combines adequate 0.5% empty-set stability, agreement with the full-32 development class, and reliable diverse-start availability.

## Verified source

The analysis consumed the already-sealed 4,433-candidate, 32-pair fresh-bank study and performed zero new scientific evaluations. All required source hashes matched exactly, including:

| Artifact | SHA-256 |
|---|---|
| `candidate_robustness_summary.json` | `99121aed7b18d70128cee7cdcc9d4d61dfd64e98d68d25ffb9131164c1a0db77` |
| `allowance_summary.json` | `e8f21ec7f78a4d1996505c6bed9e85e5b2f7b2ef57752b9af401388d63231a0f` |
| `failure_mode_summary.json` | `91281495a59de92c7a5cd3bb7a1ba8b540860ddda5fe0537b5ffc88595b7205e` |
| `time_node_summary.json` | `32ac42f574c054d07ae198c21c2fdd7af7a16364487e6ab900fe8b1708e18d53` |
| fresh-bank `summary.json` | `998a59b5bdb195e15379be085b60d31c5d5c6edd8edf63f78a64ea351f1e8740` |
| fresh-bank `inventory.json` | `3d837c2ea4108283749bdfa1d661e0a90ebd5ad39693ea942641c23f94df466e` |

All 32 replicate inventories and their sealed screen/audit result artifacts were also verified. The reconstructed eligibility matrix exactly reproduced:

- best pass counts `[26, 29, 31, 31, 32, 32]`;
- >=24/32 counts `[55, 310, 957, 1314, 1663, 1886]`.

## Frozen design

The following architecture grid was frozen before scoring:

`1/1`, `2/2`, `3/4`, `4/4`, `6/8`, `7/8`, `8/8`, `12/16`, `14/16`, and `16/16`.

The subset schedule was also frozen first:

| M | Subsets | Method |
|---:|---:|---|
| 1 | 32 | complete enumeration |
| 2 | 496 | complete enumeration |
| 4 | 35,960 | complete enumeration |
| 8 | 5,000 | unique deterministic uniform samples |
| 16 | 2,500 | unique deterministic uniform samples |

Candidate outcomes and subsets were encoded as `uint32` masks. Subset scoring used vectorized popcount over mask histograms. Candidate-level architecture probabilities used exact hypergeometric tails, not Monte Carlo estimates. The complete analysis took about three seconds and produced only 1.1 MiB of output.

## Central 75% comparison

| Metric | 3/4 | 6/8 | 12/16 |
|---|---:|---:|---:|
| 0.5% empty-set rate | 8.3815% | 10.2600% | 7.7200% |
| 0.5% p10 survivors | 2 | 0 | 3 |
| 0.5% median survivors | 331 | 164.5 | 105 |
| 1% empty-set rate | 1.3654% | 0.2200% | 0% |
| 1% p10 survivors | 84 | 84 | 126.9 |
| 2% empty-set rate | 0% | 0% | 0% |
| 0.5% median Jaccard to >=24/32 | 0.0500 | 0.1662 | 0.2588 |
| 1% median Jaccard | 0.3024 | 0.4778 | 0.6055 |
| 2% median Jaccard | 0.5490 | 0.7051 | 0.8057 |
| 3% median Jaccard | 0.6512 | 0.7557 | 0.8383 |
| 0.5% exact expected survivors | 521.4 | 311.9 | 140.0 |
| 1% exact expected survivors | 792.5 | 563.8 | 382.6 |
| 0.5% sampled subsets with >=10 diverse starts | 96.0% | 77.5% | 79.0% |
| Relative pair cost | 4x | 8x | 16x |

Moving from 3/4 to 6/8 improves agreement with the full-32 >=24 class but does not improve low-allowance set stability. Its 0.5% empty rate is higher, its p10 survivor count falls to zero, and its >=10-start availability falls from 96% to 77.5%.

Moving from 6/8 to 12/16 recovers some 0.5% empty-set stability and improves Jaccard, but still leaves 7.72% empty subsets and only 79% >=10-start availability at twice the cost. It does not deliver the decisive stabilization needed to justify 16 fresh official bank pairs.

## Exact expected classification behavior

The expected precision-like and recall-like ratios below compare each 75% architecture with the corresponding full-32 >=24/32 development class. These are expected-count descriptions, not independent predictive-performance estimates.

| Allowance | 3/4 precision / recall | 6/8 precision / recall | 12/16 precision / recall |
|---:|---:|---:|---:|
| 0.5% | 0.080 / 0.755 | 0.125 / 0.706 | 0.268 / 0.682 |
| 1% | 0.318 / 0.812 | 0.435 / 0.792 | 0.653 / 0.806 |
| 2% | 0.587 / 0.861 | 0.705 / 0.858 | 0.852 / 0.887 |
| 3% | 0.655 / 0.877 | 0.761 / 0.877 | 0.883 / 0.906 |

At 0.5%, the full-32 >=24 class contains only 55 candidates. Small-M 75% rules admit many candidates outside that reference class; increasing M improves precision-like agreement but reduces the number of attemptable starts and does not eliminate empty subsets.

## Strictness comparison

| Architecture | 0.5% empty rate | 0.5% median survivors | 1% empty rate | 1% median survivors |
|---|---:|---:|---:|---:|
| 6/8 | 10.26% | 164.5 | 0.22% | 466 |
| 7/8 | 40.94% | 5 | 8.08% | 122 |
| 12/16 | 7.72% | 105 | 0% | 373.5 |
| 14/16 | 64.28% | 0 | 5.12% | 65 |

The 87.5% architectures make the 0.5% curve structurally unlikely. This is consistent with the full-32 study having zero candidates at >=28/32 for 0.5%. Neither 7/8 nor 14/16 is defensible as a common gate for all allowances if retaining an attemptable 0.5% curve is a priority.

## Hard-bank sensitivity

The four zero-survivor individual 0.5% banks are replicate IDs `7`, `9`, `10`, and `28`.

For 3/4:

- subsets with zero hard banks are almost never empty: 0.044%;
- subsets with one hard bank are empty 4.76% of the time;
- every subset with at least two hard banks is empty.

For 6/8:

- subsets with zero hard banks are never empty;
- one-hard-bank subsets are empty 1.71% of the time;
- two-hard-bank subsets are empty 22.98% of the time;
- every subset with at least three hard banks is empty.

For 12/16, subsets with zero, one, or two hard banks are nonempty, but subsets with at least three hard banks are empty 25.39% of the time. The residual instability therefore has a clear structural relationship to the difficult finite-bank realizations.

## Frozen decision rule and recommendation

The prospectively frozen multi-metric readiness rule required a primary architecture to have:

- 0.5% empty-set rate no greater than 5%;
- >=10 diverse-start availability at least 90%;
- median Jaccard to the corresponding full-32 class at least 0.25.

No primary architecture met all three:

- 3/4 misses empty-set stability and Jaccard;
- 6/8 misses empty-set stability, diverse-start availability, and Jaccard;
- 12/16 misses empty-set stability and diverse-start availability.

The sealed recommendation is therefore:

```text
NO_REPLICATE_GATE_ARCHITECTURE_READY
```

No new untested architecture was introduced after observing the result.

## Why the authoritative run was not launched

The user's authorization was conditional: run the authoritative phase if the preflight works scientifically. Although the analysis executed correctly, its sealed recommendation explicitly says no replicate-gate architecture is ready. Creating an official protocol anyway would contradict both the preflight decision and the required protocol firewall.

Accordingly:

- no official protocol was created;
- no official banks were generated;
- no candidate generation occurred;
- no Tangent or Full branch ran;
- no selection was frozen;
- no validation was generated or accessed.

The next scientific action needs a separately specified development investigation into why the four hard banks destroy low-allowance support, or a prospectively defined alternative architecture grid. The present result does not authorize choosing such an alternative retrospectively.

## Verification

- 43/43 focused preflight and fresh-bank regression tests passed.
- Eligibility reconstruction reproduced the frozen full-32 summary exactly.
- Exact hypergeometric calculations matched complete small-case enumeration.
- Vectorized bitmask scoring matched direct Boolean scoring.
- Repeated analysis reused sealed artifacts and reproduced the console result.
- `git diff --check` passed.

## Artifacts

Output root:

`experiments/skyrmions_deep_ritz_full/outputs/skyrmion_galerkin_dev_replicate_gate_preflight_v2/`

Principal seals:

| Artifact | SHA-256 |
|---|---|
| `source_seal.json` | `30ecbde6594d197084d6d32ab888ae8edf5ff964ef7320edb2802614c1aed075` |
| `architecture_grid.json` | `a20d1849e03ef3197186a2d092f0515fc19df63b9d1337520d4f4acd773b8c4c` |
| `subset_manifest.json` | `330c2e89a266253ddb89bcfbc098af7598433532ec607335accd5926925fbe1d` |
| `exact_hypergeometric_results.json` | `0a0f68214f454446b602eaac36670b1946c79e0a15c5f6cfa6c95c38b3d7f9fe` |
| `resampling_results.json` | `8a139d513b237583fb4c22043463a3775e8b23a8b0fa53f7a534a6691b91183c` |
| `diversity_diagnostics.json` | `d48b16377a976675670778a13bb2b66e646bb347b937a0228273334b4c8334ca` |
| `hard_bank_diagnostics.json` | `b9f7f9a6a0d2d867ea12136469c537bc3f3a428c36ae13a39f48a56016206424` |
| `recommended_official_gate.json` | `b6fbd51c01ef2097e9f5997b1d8ec9ef86220e2ec1116df85ab52d48a02f4cb0` |
| `summary.json` | `87c56fe469ec90b7c13b2207d02a09b678d02d935c062e5762800e28f8da4cc6` |
| `inventory.json` | `e9a34ace9a8ff1f67df3322510446185eee95378daf025de37c89e9779d30868` |
