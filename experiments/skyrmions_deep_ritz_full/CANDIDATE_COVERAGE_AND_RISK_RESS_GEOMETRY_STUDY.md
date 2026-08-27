# Skyrmion Candidate-Coverage and Risk–rESS Geometry Study

Date: 2026-08-25

Status: **completed development-only experiment**. This is not an official
Pareto-v3 continuation, a Pareto-v4 protocol, a selection, a validation result,
or a certification result.

## Question

The frozen 337-candidate v2 pool contained no complete dual-bank survivor at
0.5% Law-relative scientific-risk allowance and only one at 1%. This study
asked whether that scarcity was primarily caused by inadequate coverage of the
low-risk sensor-design manifold or whether robust reference overlap remained
rare under much denser targeted sampling.

All comparisons use the unchanged frozen v2 `N=8192` screen bank, independent
`N=16384` periodic-audit bank, exact scientific-risk function, geometry gates,
forcing/covariance gates, and minimum relative ESS threshold `0.05`.

## Frozen development generator

The generator was informed explicitly by the earlier development diagnostics
and was serialized before any new periodic-audit evaluation. It produced 4,096
new unique feasible geometries:

| Component | Final rows | Fraction |
|:---|---:|---:|
| Multi-scale local clouds | 1,640 | 40.04% |
| Periodic Law/witness and witness/witness paths | 1,024 | 25.00% |
| Risk-tangent feasible-manifold exploration | 816 | 19.92% |
| Scrambled deterministic Sobol global coverage | 616 | 15.04% |
| **Total** | **4,096** | **100%** |

Generation began with 4,409 raw proposals. The unchanged minimum-separation
constraint rejected 313; there were zero duplicates within the final new pool
and zero canonical-identity collisions with the original 337 candidates.

The local component used 23 deduplicated anchors: Law, the best robust
witnesses, and symmetry-aware max-min representatives from the earlier 1–5%
survivor sets. Its fixed perturbation scales were:

```text
[0.00025, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
```

The path component used 30 permutation-matched, periodic minimum-image paths.
The risk-tangent component used deterministic directions projected orthogonal
to the exact frozen selection-risk gradient at Law; it did not run a Tangent
optimizer. Every geometry was periodically wrapped and ordered by exhaustive
sensor-permutation matching to the frozen Law geometry.

## Primary original-versus-expanded result

| Allowance | Exact risk ceiling | Original screen feasible | Original dual-bank | New inside ceiling | New screen feasible | New dual-bank | Combined screen feasible | Combined dual-bank | Best combined audit min-rESS | Best candidate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 0.5% | 5.212482221850432 | 193 | 0 | 1,094 | 1,057 | 6 | 1,250 | 6 | 0.0538055287766885 | `coverage_2895` |
| 1% | 5.238414969222823 | 216 | 1 | 1,461 | 1,420 | 156 | 1,636 | 157 | 0.0569050966308325 | `coverage_2897` |
| 2% | 5.290280463967603 | 242 | 12 | 2,208 | 2,147 | 762 | 2,389 | 774 | 0.0628754443107112 | `coverage_0635` |
| 3% | 5.342145958712384 | 274 | 35 | 2,616 | 2,543 | 1,105 | 2,817 | 1,140 | 0.0705167401834042 | `coverage_0296` |
| 4% | 5.394011453457164 | 292 | 53 | 3,025 | 2,948 | 1,479 | 3,240 | 1,532 | 0.0796745423474696 | `coverage_2504` |
| 5% | 5.445876948201945 | 299 | 59 | 3,269 | 3,182 | 1,700 | 3,481 | 1,759 | 0.0868695920194309 | `coverage_1425` |

The expanded pool therefore found complete dual-bank witnesses inside the 0.5%
ceiling, while preserving the previous result that the original frozen pool
contained none. This does not retroactively alter v2 or v3.

## New 0.5% witnesses

Six new candidates pass the exact 0.5% risk ceiling and every unchanged screen
and audit support gate:

| Candidate | Source | Anchor | Risk | Screen min-rESS | Audit min-rESS / robust rESS |
|:---|:---|:---|---:|---:|---:|
| `coverage_2895` | risk tangent | Law | 5.210932017008545 | 0.0648730331584005 | 0.0538055287766885 |
| `coverage_3062` | risk tangent | Law | 5.209785677081920 | 0.0634749191102360 | 0.0519180707961784 |
| `coverage_0638` | local cloud | `candidate_089` | 5.209267744814202 | 0.0676897299777396 | 0.0516599858992127 |
| `coverage_2893` | risk tangent | Law | 5.199691735161848 | 0.0643158800956392 | 0.0514100399412997 |
| `coverage_0771` | local cloud | `candidate_105` | 5.209563906551200 | 0.0683318460706594 | 0.0504435849708377 |
| `coverage_1958` | periodic path | `candidate_105` | 5.212301073067264 | 0.0683377845062438 | 0.0501497987527475 |

The 0.5% witnesses arise from three different targeted generator components:
three risk-tangent rows, two local-cloud rows, and one periodic-path row. No
Sobol-global row survived at this allowance.

## `candidate_078` neighborhood

The sole original 1% witness is not an isolated finite-pool knife edge under
small periodic perturbations. Of 51 local-cloud candidates around
`candidate_078` that remained inside the exact 1% ceiling, 28 passed both
banks.

| Scale | Generated | Inside 1% | Audited inside 1% | 1% dual-bank survivors | Median robust rESS inside 1% | Best robust rESS inside 1% |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00025 | 11 | 11 | 11 | 10 | 0.05049705 | 0.05107452 |
| 0.00050 | 11 | 11 | 11 | 7 | 0.05004507 | 0.05103316 |
| 0.00100 | 11 | 11 | 11 | 8 | 0.05092813 | 0.05261685 |
| 0.00200 | 11 | 5 | 5 | 2 | 0.04782767 | 0.05408666 |
| 0.00500 | 11 | 8 | 8 | 1 | 0.04480972 | 0.05097358 |
| 0.01000 | 11 | 3 | 3 | 0 | 0.03863710 | 0.04130460 |
| 0.02000 | 11 | 2 | 0 | 0 | unavailable | unavailable |

The smallest three scales yield 25 survivors among 33 inside-1% perturbations.
Survival becomes uneven as the scale increases, and larger perturbations more
often leave the 1% risk band or lose audit overlap. The evidence is therefore
consistent with a real local basin around `candidate_078`, not merely one
isolated row, while also showing a finite basin width.

## Generator-source contribution

The new survivor counts by generator component are nested with allowance:

| Allowance | Local cloud | Periodic path | Risk tangent | Sobol global | Total new |
|---:|---:|---:|---:|---:|---:|
| 0.5% | 2 | 1 | 3 | 0 | 6 |
| 1% | 48 | 98 | 10 | 0 | 156 |
| 2% | 402 | 329 | 31 | 0 | 762 |
| 3% | 619 | 444 | 42 | 0 | 1,105 |
| 4% | 870 | 554 | 55 | 0 | 1,479 |
| 5% | 1,067 | 574 | 59 | 0 | 1,700 |

At 1%, 98 path rows across 20 distinct paths survived. By 5%, all 30 paths
contained at least one survivor. The absence of Sobol-global survivors does not
establish global absence; it only shows that this 616-row global component did
not find a qualifying row on the tested banks.

## Empirical risk–rESS association

Spearman rank correlations are descriptive associations only:

| Pool | n, screen | rho(risk increase, screen rESS) | n, audit | rho(risk increase, audit rESS) | rho(risk increase, robust rESS) |
|:---|---:|---:|---:|---:|---:|
| Original | 337 | 0.4735493051 | 337 | 0.4970222941 | 0.4969470441 |
| New | 4,096 | 0.4766897279 | 3,182 | 0.8233885738 | 0.8233703482 |
| Combined | 4,433 | 0.4844789293 | 3,519 | 0.8005434554 | 0.8005197424 |

The positive association is strong in the audited targeted pool, but this does
not show that scientific risk causes rESS or that rESS must increase with risk.
The generator is deliberately nonuniform and targeted, so these correlations
describe this development sample only.

The complete risk-bin distributions for screen, audit, and robust rESS are in
`summary.json`; the row-level values are in `risk_ress_rows.csv`.

## Development interpretation

The prespecified classification rule returns:

```text
COVERAGE_LIMITED
```

Evidence:

- six new complete dual-bank witnesses were found at 0.5%, where the original
  pool had zero;
- the new pool contributed 156 witnesses at 1%, compared with one original
  witness;
- 28 of 51 local candidates around `candidate_078` that stayed inside 1%
  survived both banks;
- qualifying 1% path rows appeared on 20 distinct deterministic periodic paths.

On these frozen development banks, the earlier low-risk scarcity was therefore
materially influenced by candidate coverage. This is not a claim that every
low-risk neighborhood is broad, that the search has global coverage, or that
any candidate is certified. A future official experiment would require a new,
prospectively frozen protocol and fresh official banks.

## Performance and firewalls

| Stage | Seconds |
|:---|---:|
| Generator | 14.5710 |
| Exact risk evaluation | 32.2729 |
| `N=8192` screen evaluation | 108.5993 |
| `N=16384` audit evaluation | 91.0862 |
| Summary/path/diversity statistics | 24.9317 |

Only the 3,182 new rows inside the exact 5% ceiling and passing complete screen
support received periodic-audit evaluation.

The run accessed no validation data, ran no Tangent optimizer, constructed no
Full K/f system, ran no eigensolve or Deep Ritz solve, created no official
protocol or bank, and froze no selection. The official-v3 firewall was closed
before and after the study.

## Machine-readable artifacts and seals

Artifacts are isolated under:

```text
outputs/skyrmion_galerkin_dev_candidate_coverage_v1/
```

| Artifact | SHA-256 |
|:---|:---|
| `generator_spec.json` | `e889aae23c7649f579c8108088441eb68318abb65d8bfe1d49557c2f9aed9600` |
| `candidate_pool.json` | `da5b07e16c9c44d1e44d7831c6badb3a8a5218e6198d1cb1a7c0963a995db5e9` |
| `screen_results.json` | `a2e80acf9889f6124cc572f2245138186b4d53355e58c06921feebc048fdc23b` |
| `audit_results.json` | `92d8832f92fbbb7b053c051ae25bf67eaaaaec63bf000df4e8fde22e6d1f8660` |
| `path_diagnostics.json` | `394b1c0a6b21f677968a75c1a697e3fb1adbe09f483a75ee4c85a88f11484115` |
| `risk_ress_rows.csv` | `93e8c81e60a4243c6365d7816afe99d8219736cc7466058716bd91670d6da75e` |
| `summary.json` | `2278c34d366a71f37c70a0c8ec30376a7788a24526dce042502b5c108fcc744e` |
| `inventory.json` | `fe5559527652c0b99f20d61cf9de9f29ca0e09256d08e17cf1c3f93f0da3808b` |

The completed study was invoked again in `run` mode. The rerun verified the
frozen source hashes and every artifact digest before returning the cached
result.

## Verification

- 18/18 focused candidate-coverage tests passed.
- 83/83 combined coverage, Pareto-v2, and Pareto-v3 tests passed.
- Python compilation checks passed.
- `git diff --check` passed.
- The original dual-bank counts reproduced exactly as
  `[0, 1, 12, 35, 53, 59]`.
- The original v2, Phase-1, and all-allowance artifact hashes remained
  unchanged.
