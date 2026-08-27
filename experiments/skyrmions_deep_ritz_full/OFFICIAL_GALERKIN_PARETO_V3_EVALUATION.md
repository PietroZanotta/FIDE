# Official Galerkin Pareto v3 Evaluation

## Outcome

**STOPPED PROSPECTIVELY AFTER PHASE 1.** The required development-only v2-bank
diagnostic returned:

> **C. NO ROBUST 0.5% START FOUND ON v2 BANKS**

All 193 candidates that passed the v2 N=8192 screen-side 0.5% risk, geometry,
projection, and `rESS >= 0.05` gates were recomputed on the independent frozen
v2 N=16384 periodic-audit bank. All 193 passed audit projection, but none
reached audit `rESS >= 0.05`. The best audit rESS was
`0.0483757148952091`.

The prospective decision rule, declared before this computation, classified
10 or more dual-bank candidates as “clearly exist,” 1–9 as “rare,” and zero as
classification C. The task explicitly requires a stop before v3 selection for
classification C. Therefore no Pareto-v3 protocol was frozen, no v3 official
selection bank was generated, no Tangent or Full branch ran, no selection was
frozen, and no validation data was generated or opened. No threshold or solver
methodology was changed.

## Initial repository state and isolation

The initial `git status --short` was recorded before task edits. It already
contained the ongoing skyrmion development tree, additive shared projection
wrappers, and native-backend tests as modified/untracked work. Those entries
were preserved. This task added only experiment-local Pareto-v3 source, tests,
diagnostic artifacts, and this report beneath
`experiments/skyrmions_deep_ritz_full/`. It made no change to `src/`, `native/`,
the original production experiment, Pareto v1, or Pareto v2.

The only v3 numerical artifacts are:

- `outputs/official_galerkin_pareto_v3/diagnostic_v2_audit_map/v2_inventory.json`;
- `outputs/official_galerkin_pareto_v3/diagnostic_v2_audit_map/summary.json`.

They are explicitly marked development-only and not official v3 results.

## Proof Pareto v2 remained frozen

Before the diagnostic, the complete v2 tree was inventoried and its key hashes
were compared with fixed expected values. The same verification is exercised
by the v3 test suite.

| v2 artifact | verified SHA-256 |
| --- | --- |
| `OFFICIAL_GALERKIN_PARETO_V2_PROTOCOL.md` | `f7353f821e194ea86a3dc1f891633fcecb77b0909d6c17229942de2032c2f0e6` |
| `OFFICIAL_GALERKIN_PARETO_V2_EVALUATION.md` | `00965dbf9bf78763f0a32ae1a184010dce33042d290b385f60560fe6841487df` |
| `outputs/official_galerkin_pareto_v2/protocol.json` | `8360afb812b4036cbadaa0e2ca4f12d92c10ffa6900c1a22c5f231eea18dbf3b` |
| `outputs/official_galerkin_pareto_v2/failure.json` | `e73d4e3ac89f9562bc6038c8ea4c3d0d9bb040b78dfffa477a982585329acf40` |

The v2 inner protocol seal remains
`22a33ce47b2a3cc17ff063d100b878ac32c3ef6cc1a2b3e10a6eb8cd076488f1`.
The canonical inventory of every file in the active v2 output tree has digest
`e69e58bb0cd02967315b83634551ff66773740c2524f5a110542d8e71f95b723`.
The v2 tree still contains neither a selection hash nor a fresh-validation
directory. Its status remains **FAILED PROTOCOL**, not a successful sweep.

## Why v2 failed

At 0.5%, v2 promoted only four N=8192-screened starts. They passed search-side
risk, geometry, projection, forcing, covariance, and rESS, but their independent
N=16384 audit rESS values were approximately `0.04467`, `0.04723`, `0.04838`,
and `0.04270`. With no eligible Tangent finalist, v2 stopped before Full and
before validation. The v3 Phase-1 diagnostic tested whether this was merely a
four-start selection failure.

## Frozen scientific quantities used in the diagnostic

The diagnostic changed no scientific quantity:

- fixed-feature finite-dimensional Galerkin basis size: `K = 280`;
- dictionary SHA-256:
  `37e9b60fcb92c4e5a0ee7ec1651fb7f8889f7ac6bdb02d3bd314e9ef40833326`;
- relative rank tolerance: `1e-12`;
- minimum rESS: `0.05`;
- held-out energy threshold: `0.08`;
- allowances intended for a possible v3: `[0.5, 1, 2, 3, 4, 5]%`;
- Law risk: `5.186549474478042`;
- exact 0.5% selection ceiling: `5.212482221850432`.

Deep Ritz was not used. The diagnostic did not form a Tangent optimization,
Full K/f system, coefficient eigensolve, or physical Full certificate.

## Complete v2-bank audit-aware diagnostic

Machine-readable summary SHA-256:
`fd856bf004932e467a7abf87e5f158864899d1afe7b592eef2bc01bae35d3d33`.
V2 inventory SHA-256:
`20f7cf4c6c9db8efea82b7e4f2c84b144f9d5959f2d69506681dcfa6b0323078`.

| quantity | result |
| --- | ---: |
| v2 candidates passing N=8192 0.5% screen gates | 193 |
| audit projection valid | 193 |
| audit rESS >= 0.05 | **0** |
| audit projection+rESS+forcing+covariance valid | **0** |
| maximum audit projection residual | `9.8570e-11` |
| maximum audit forcing mean | `2.6120e-08` |
| maximum audit covariance condition | `4.57759` |
| elapsed batched audit-map time | `11.198 s` |

The projection tolerance was `2e-6`, forcing-mean tolerance was `2e-7`, and
covariance-condition limit was `1e10`. Thus the population-wide failure was
specifically the independent-bank rESS gate, not projection, forcing, or
covariance.

Audit-rESS distribution across all 193 screen-feasible candidates:

| statistic | audit rESS |
| --- | ---: |
| minimum | 0.0358957832 |
| p05 | 0.0436266194 |
| p25 | 0.0446095515 |
| median | 0.0446964976 |
| p75 | 0.0450957830 |
| p95 | 0.0472283739 |
| maximum | **0.0483757149** |

No robust candidate exists under the declared dual-bank definition, so a
geometry-diversity statistic for the robust set is undefined. For descriptive
context only, the top 20 near-threshold candidates have periodic pairwise
distances with minimum `0.000106152`, median `0.0143410`, and maximum
`0.0278522`; this does not convert them into eligible starts.

## Top 20 by declared robust score

Here `robust_rESS = min(rESS_screen, rESS_audit)`. None passes 0.05. Exact
eight-coordinate geometries and all audit diagnostics are retained in
`summary.json`.

| rank | candidate | risk | screen rESS | audit rESS | robust rESS |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | candidate_054 | 5.207900752 | 0.067183582 | 0.048375715 | 0.048375715 |
| 2 | candidate_085 | 5.212220462 | 0.065376993 | 0.048311416 | 0.048311416 |
| 3 | candidate_053 | 5.201734445 | 0.066491908 | 0.047766111 | 0.047766111 |
| 4 | candidate_031 | 5.209314805 | 0.065787139 | 0.047760814 | 0.047760814 |
| 5 | candidate_083 | 5.206476414 | 0.064884923 | 0.047607171 | 0.047607171 |
| 6 | candidate_030 | 5.205876322 | 0.065657613 | 0.047590738 | 0.047590738 |
| 7 | candidate_029 | 5.202710466 | 0.065512122 | 0.047409700 | 0.047409700 |
| 8 | candidate_084 | 5.202388076 | 0.064765428 | 0.047404041 | 0.047404041 |
| 9 | candidate_082 | 5.203732951 | 0.064381603 | 0.047283679 | 0.047283679 |
| 10 | candidate_001 | 5.203174625 | 0.064494210 | 0.047229777 | 0.047229777 |
| 11 | candidate_081 | 5.203198773 | 0.064494261 | 0.047227439 | 0.047227439 |
| 12 | candidate_028 | 5.199819241 | 0.065346778 | 0.047215629 | 0.047215629 |
| 13 | candidate_052 | 5.196527543 | 0.065686315 | 0.047095749 | 0.047095749 |
| 14 | candidate_019 | 5.201318884 | 0.064344513 | 0.047039335 | 0.047039335 |
| 15 | candidate_027 | 5.197204764 | 0.065157724 | 0.047006660 | 0.047006660 |
| 16 | candidate_018 | 5.199571103 | 0.064197205 | 0.046852920 | 0.046852920 |
| 17 | candidate_072 | 5.205073868 | 0.064643742 | 0.046827610 | 0.046827610 |
| 18 | candidate_026 | 5.194869338 | 0.064941199 | 0.046781172 | 0.046781172 |
| 19 | candidate_071 | 5.203151678 | 0.064724666 | 0.046754305 | 0.046754305 |
| 20 | candidate_087 | 5.198176910 | 0.064546725 | 0.046735240 | 0.046735240 |

## Phase-1 decision and stages not entered

The result satisfies classification C exactly. In compliance with the
prospective handoff:

- `OFFICIAL_GALERKIN_PARETO_V3_PROTOCOL.md` was **not created**;
- `outputs/official_galerkin_pareto_v3/protocol.json` and its sidecar do not
  exist;
- no official v3 seed was frozen and no v3 selection bank was generated;
- audit-aware Tangent and Full start lists were not selected;
- neither independent optimization branch ran at any allowance;
- no authoritative 65536 finalist recomputation ran;
- no Law/Tangent/Full cross-evaluation or selection freeze exists;
- predeclared validation seeds were never turned into arrays;
- no fresh-validation directory exists and no eta was validated or changed.

Accordingly, there are no official v3 method/allowance classifications to
fabricate. `PASS`, `NO CERTIFIED SELECTION WINNER`, validation reversal, and
validation numerical-failure labels apply only after an official v3 protocol
exists and its relevant branch runs. This attempt stopped earlier at its
prospective development gate.

## Tests and reproducibility

The experiment-local v3 suite adds 20 tests covering v2 immutability, the new
seed namespace, unchanged K/dictionary/rank/rESS/energy constants, exact risk
arithmetic, dual-bank eligibility, the robust-rESS formula, the Phase-1 C gate,
absence of Tangent/Full work, absence of a v3 protocol, and the validation
firewall.

The complete relevant regression command reported:

```text
177 passed, 4 skipped in 23.43s
```

This includes historical ESS, production, final-crosscheck, Galerkin-only,
K280-quadrature, official-Pareto, resolution, Pareto-v2, both native Tesseract
backends, and Pareto-v3 Phase-0/1 tests.

## Performance and further optimization

The candidate-batched native projection endpoint is already the clear
**HIGH VALUE / LOW RISK** optimization for this diagnostic: all 193 audit
projections and forcing diagnostics completed in about 11.2 seconds with fixed
float64 semantics. Larger batching could be profiled, but it cannot change the
observed audit rESS values and therefore cannot change classification C.

Other prospective opportunities, if a future independently justified protocol
is ever designed, remain:

| priority | opportunity | assessment |
| --- | --- | --- |
| HIGH VALUE / LOW RISK | exact candidate/hash reuse across Tangent and Full | avoids duplicate projection only on identical bank/config/method signatures |
| MEDIUM VALUE | nearby-eta Newton warm starts | safe only after convergence-equivalence and determinism tests |
| MEDIUM VALUE | fixed-basis time-shard cache or mmap | trades substantial memory for reduced repeated basis evaluation |
| MEDIUM VALUE | JAX K/f chunk-size tuning | profile prospectively with action/gradient equivalence gates |
| NOT RECOMMENDED for GPU hot path | CPU Tesseract K/f assembly | prior transfer-inclusive tests were 3.6–7.3 times slower than device-resident JAX |

Answer: further computational optimization is possible without changing
scientific semantics, but no optimization can repair the current Phase-1
scientific gate. The completed diagnostic was dominated by candidate feature,
projection, and forcing evaluation; no Full runtime was incurred.

## Limitations and scientific interpretation

This result is conditional on the frozen v2 screen and periodic-audit bank
realizations and the existing 337-candidate feasible-manifold pool. It does not
prove that no 0.5% design can satisfy rESS >= 0.05 on every possible bank, and
it does not invalidate the 0.05 gate. The earlier ESS qualification found
N=32768 witnesses on other independent development banks. It does establish
that the proposed v3 repair—promoting more of the same v2 pool after checking
both specific v2 selection banks—finds zero eligible 0.5% starts.

The strongest honest conclusion is therefore narrower than the motivating
hypothesis: the v2 failure was not merely caused by choosing the wrong four
starts from its N=8192 screen. On its N=16384 periodic-audit bank, every one of
the 193 screen-feasible 0.5% candidates misses the unchanged rESS gate. A
different future protocol would require a separately justified sampling or
candidate-design study; this task does not infer, tune, or launch one.
