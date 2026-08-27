# ESS qualification and K=280 performance audit

## Context, scope, and methodological status

The nonlinear empirical Deep Ritz route remains retired. The intended Full solver is the fixed-feature, rank-aware K=280 Galerkin approximation, with coefficients solved by pseudoinverse and eta derivatives taken by the fixed-coefficient envelope rule. This study changes none of that methodology: it is a selection-development-only diagnostic, constructs no Galerkin system for the candidate pool, performs no eta optimization or Pareto sweep, and accesses no old or fresh validation quantity.

The frozen protocol SHA-256 is `3f2648216de0597b1ce666b1d7d2d0a5b211e80262a3c9259ef11f1b2708bd23`. K is 280, the dictionary SHA-256 is `37e9b60fcb92c4e5a0ee7ec1651fb7f8889f7ac6bdb02d3bd314e9ef40833326`, relative rank tolerance is `1e-12`, rESS threshold is `0.05`, and Ritz-energy threshold is `0.08`.

## Exact ESS implementation audit

The shared implementation at `src/mfsi/projection.py:244-254` normalizes the supplied base weights at every physical time and maps zero base mass to `-inf`. At lines 281-292 it forms projected weights with a float64 softmax and computes `ESS_projected = 1/max(sum_i w_i^2, 1e-300)`, `ESS_base = 1/max(sum_i b_i^2, 1e-300)`, and `rESS = ESS_projected/ESS_base`. There is no clipping of weights before ESS. The Newton solver clips only lambda proposals to ±1000 (`src/mfsi/projection.py:107-124`); logits are normalized by stable softmax/log-sum-exp. The `1e-300` denominator floors are the only ESS stabilization.

The development-bank constructor uses exactly uniform `1/N` weights (`resolution_study.py:288-300`), so `ESS_base=N` and the reported rESS is exactly `ESS/N`. It is not divided by the 16 microscopic particles. Both train and audit use the same `EmpiricalIProjector`. Downstream forcing takes `min` over the physical-time vector (`forcing.py:107-116`), hence the reported minimum is over the 13 time nodes.

For a uniform base and unnormalized exponential tilt `a_i`, `w_i=a_i/sum_j a_j`, so `ESS/N=(sum a)^2/(N sum a^2)=(E_N[a])^2/E_N[a^2]`. Every anchor record checks this identity directly; the largest discrepancy is reported below.

## Six-anchor independent-bank convergence

Each entry is mean ± independent-replicate standard deviation of the minimum over time; brackets are the predeclared 95% Student-t interval. Absolute ESS is the mean minimum absolute ESS.

| anchor | N | reps | min rESS mean ± SD | 95% interval | min absolute ESS | controlling nodes |
|---|---:|---:|---:|---:|---:|---|
| law | 8192 | 4 | 0.047266 ± 0.015394 | [0.022771, 0.071761] | 387.2 | [8, 8, 7, 8] |
| law | 16384 | 4 | 0.057411 ± 0.003248 | [0.052244, 0.062579] | 940.6 | [7, 8, 7, 7] |
| law | 32768 | 3 | 0.054549 ± 0.006543 | [0.038296, 0.070803] | 1787.5 | [8, 7, 7] |
| **law classification** |  |  | **BORDERLINE** | r∞(1/N)=0.059622; r∞(1/√N)=0.064773 |  |  |
| historical_0p5 | 8192 | 4 | 0.052248 ± 0.016330 | [0.026262, 0.078233] | 428.0 | [8, 8, 7, 8] |
| historical_0p5 | 16384 | 4 | 0.061370 ± 0.003206 | [0.056268, 0.066471] | 1005.5 | [7, 8, 7, 7] |
| historical_0p5 | 32768 | 3 | 0.057145 ± 0.007077 | [0.039565, 0.074724] | 1872.5 | [8, 7, 7] |
| **historical_0p5 classification** |  |  | **UNRESOLVED** | r∞(1/N)=0.061706; r∞(1/√N)=0.065168 |  |  |
| historical_1 | 8192 | 4 | 0.048666 ± 0.013690 | [0.026882, 0.070450] | 398.7 | [8, 8, 8, 8] |
| historical_1 | 16384 | 4 | 0.060088 ± 0.002779 | [0.055666, 0.064509] | 984.5 | [7, 8, 7, 7] |
| historical_1 | 32768 | 3 | 0.060205 ± 0.004449 | [0.049153, 0.071257] | 1972.8 | [7, 7, 7] |
| **historical_1 classification** |  |  | **UNRESOLVED** | r∞(1/N)=0.065916; r∞(1/√N)=0.074075 |  |  |
| historical_2 | 8192 | 4 | 0.046198 ± 0.006858 | [0.035285, 0.057111] | 378.5 | [8, 7, 8, 8] |
| historical_2 | 16384 | 4 | 0.053973 ± 0.007896 | [0.041409, 0.066536] | 884.3 | [7, 7, 7, 7] |
| historical_2 | 32768 | 3 | 0.051507 ± 0.005466 | [0.037928, 0.065087] | 1687.8 | [7, 7, 7] |
| **historical_2 classification** |  |  | **BORDERLINE** | r∞(1/N)=0.055395; r∞(1/√N)=0.059149 |  |  |
| eta0_3pct | 8192 | 4 | 0.062031 ± 0.010918 | [0.044658, 0.079405] | 508.2 | [8, 7, 8, 8] |
| eta0_3pct | 16384 | 4 | 0.068736 ± 0.004935 | [0.060883, 0.076590] | 1126.2 | [7, 8, 7, 7] |
| eta0_3pct | 32768 | 3 | 0.068997 ± 0.006560 | [0.052702, 0.085292] | 2260.9 | [7, 7, 7] |
| **eta0_3pct classification** |  |  | **CLEARLY ABOVE 0.05** | r∞(1/N)=0.072350; r∞(1/√N)=0.077276 |  |  |
| eta_grad_3pct | 8192 | 4 | 0.062454 ± 0.011016 | [0.044926, 0.079983] | 511.6 | [8, 7, 8, 8] |
| eta_grad_3pct | 16384 | 4 | 0.069220 ± 0.005000 | [0.061264, 0.077175] | 1134.1 | [7, 8, 7, 7] |
| eta_grad_3pct | 32768 | 3 | 0.069437 ± 0.006591 | [0.053065, 0.085810] | 2275.3 | [7, 7, 7] |
| **eta_grad_3pct classification** |  |  | **CLEARLY ABOVE 0.05** | r∞(1/N)=0.072820; r∞(1/√N)=0.077758 |  |  |

Maximum direct `(E[a])²/E[a²]` versus implementation rESS discrepancy: `4.441e-16`. The extrapolations are descriptive; independent-bank distributions and intervals control the classifications.

Anchor scientific selection risks are `historical_0p5=5.203174625200`, `eta_grad_3pct=5.342099811291`, `law=5.186549474478`, `eta0_3pct=5.340106050966`, `historical_1=5.225761943282`, `historical_2=5.284504645220`. Across all anchor/N/replicate evaluations, maximum projection residual was `9.983e-11`, maximum observable-covariance condition `4.541`, maximum absolute lambda component `78.836`, and maximum pre-centering forcing mean `7.702e-15`. Full timewise arrays are retained in the per-replicate JSON records.

## Risk-feasible design-region map (Stage A, N=8192)

| allowance | total | risk feasible | risk+rESS | risk+projection | risk+projection+rESS | rESS min / p05 / p25 / median / p75 / p95 / max |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.5% | 337 | 208 | 166 | 208 | 166 | 0.029856 / 0.044326 / 0.051120 / 0.051814 / 0.052067 / 0.057815 / 0.060074 |
| 1% | 337 | 233 | 170 | 233 | 170 | 0.029856 / 0.043635 / 0.049872 / 0.051776 / 0.052028 / 0.058009 / 0.064110 |
| 2% | 337 | 260 | 182 | 260 | 182 | 0.029856 / 0.041998 / 0.049544 / 0.051758 / 0.052014 / 0.058028 / 0.065582 |
| 3% | 337 | 286 | 203 | 286 | 203 | 0.029856 / 0.042483 / 0.049638 / 0.051780 / 0.052243 / 0.059239 / 0.065582 |
| 4% | 337 | 301 | 218 | 301 | 218 | 0.029856 / 0.042501 / 0.049793 / 0.051811 / 0.052597 / 0.059832 / 0.071159 |
| 5% | 337 | 304 | 221 | 304 | 221 | 0.029856 / 0.042506 / 0.049801 / 0.051814 / 0.052956 / 0.060035 / 0.071159 |

All 337 Stage-A projections are valid. Their maximum projection residual is `6.909e-03`, maximum covariance condition `516.951`, and maximum absolute lambda component `1000.000`.

Across the full deterministic pool, Pearson correlation between Law-relative risk increase and minimum rESS is `0.253`; rank correlation is `0.174`. Positive values indicate that accepting more scientific risk tends to improve overlap; negative values indicate the converse. The allowance-specific quantiles show directly whether the 0.5% region is concentrated below the gate.

## Progressive rescoring and feasibility

Stage B rescored `297` deduplicated candidates at N=16384. Stage C rescored `286` at N=32768. No Full K/f system was constructed.

| allowance | answer | N32768 evaluated | witnesses rESS≥0.05 | best rESS | absolute ESS | risk increase | controlling node | geometry |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0.5% | **YES** | 201 | 30 | 0.056784 | 1860.7 | 0.4389% | 8 | `candidate_032` [0.89131 0.21876 1.32172 0.86127 0.77992 0.52983 1.62269 0.57894] |
| 1% | **YES** | 226 | 54 | 0.060198 | 1972.6 | 0.9466% | 7 | `candidate_121` [0.89085 0.2177  1.32519 0.86186 0.7764  0.53062 1.62564 0.57748] |
| 2% | **YES** | 243 | 71 | 0.063900 | 2093.9 | 1.7949% | 7 | `candidate_077` [0.8941  0.21131 1.32849 0.86389 0.76246 0.52253 1.63439 0.58704] |
| 3% | **YES** | 268 | 96 | 0.067118 | 2199.3 | 2.9306% | 7 | `candidate_165` [0.8953  0.20455 1.33381 0.86462 0.75043 0.51966 1.64311 0.58891] |
| 4% | **YES** | 283 | 111 | 0.077097 | 2526.3 | 3.8601% | 7 | `candidate_170` [0.89482 0.20841 1.33802 0.86513 0.75981 0.51777 1.64483 0.59028] |
| 5% | **YES** | 286 | 114 | 0.080871 | 2650.0 | 4.4075% | 8 | `candidate_157` [0.88782 0.22033 1.33703 0.86464 0.75895 0.51859 1.64397 0.58921] |

These are ESS-feasibility witnesses, not Full winners. A staged absence is labeled UNRESOLVED because unevaluated candidates could move across the threshold at N=32768.

## Absolute ESS and empirical error versus ESS

Relative ESS diagnoses overlap and does not generally rise with N; absolute ESS diagnoses the effective Monte Carlo count. For example, rESS 0.044 at N=32768 is absolute ESS about 1442. The tables above report both for every anchor and every allowance witness.

The optional five-point diagnostic compares N=8192 and N=32768 without Full action:

| candidate | target band | rESS N8192 | rESS N32768 | absolute change | lambda-norm relative change |
|---|---:|---:|---:|---:|---:|
| candidate_139 | 0.03 | 0.029856 | 0.031121 | 0.001265 | 0.071563 |
| candidate_111 | 0.04 | 0.039996 | 0.031505 | 0.008491 | 0.018544 |
| candidate_058 | 0.05 | 0.049992 | 0.058494 | 0.008502 | 0.020952 |
| candidate_173 | 0.07 | 0.070755 | 0.074542 | 0.003787 | 0.021902 |
| candidate_313 | 0.10 | 0.105293 | 0.104967 | 0.000326 | 0.005815 |

This modest subset does not reveal or define a new numerical-error transition and cannot justify changing the 0.05 gate.

## Performance profile and safe optimization

Device: `NVIDIA GeForce RTX 5090 Laptop GPU` via `gpu`, float64 enabled. Candidate-batched reconstruction/observable preprocessing is the only new execution path used by bulk ESS screening; the benchmark determines whether it is actually an optimization.

| operation | first call (s) | steady median (s) | note |
|---|---:|---:|---|
| 8-candidate scalar preprocessing | 1.197 | 0.009 | baseline |
| 8-candidate batched preprocessing | 0.810 | 0.009 | 0.99×; max tensor discrepancy 5.551e-17; max downstream rESS discrepancy 7.327e-15 |
| N8192 information projection | 0.024 | 0.022 | native time-warm-start solve |
| N8192 forcing construction | 1.347 | 0.044 | projection plus lambda-dot/forcing |
| K/f assembly | 3.210 | 2.255 | fixed K=280 |
| coefficient eigensolve | 1.463 | 0.044 | fixed K=280 |
| fixed-coefficient value+gradient | 4.002 | 0.038 | fixed K=280 |
| complete K280 value+gradient | 2.689 | 1.623 | fixed K=280 |
| held-out K280 audit | 17.004 | 11.989 | N=4096; energy residual 0.079867 |
| 8-candidate native projection: scalar calls | — | 0.0686 | actual N8192 skyrmion shapes |
| 8-candidate native projection: OpenMP batch | — | 0.0095 | 7.19× projection-only speedup; lambda discrepancy 0.0e+00 |

The batched path increases the resident feature temporary from `3.2` MiB to `26.0` MiB for batch 8. Its measured steady speedup is only `0.99×`, so it is equivalence-qualified infrastructure, not a material optimization. It does not touch action or gradient code, so action/gradient discrepancy is not applicable; downstream rESS agrees to `7.327e-15`. The pre-existing cached Full optimization retained action discrepancy `8.644e-14` and gradient discrepancy `2.869e-10`, with a measured historical K160 speedup of `4.06×`. The K280 basis cache is `7.33` GiB at N=8192; a per-sample K×K Gram cache remains intentionally prohibited.

The measured held-out Full audit reproduces action `0.296692769241` and maximum energy residual `0.079866824611` on the fixed eta0 geometry; it performs no optimization.

### Additive native candidate-projection extension

After the ESS protocol and scientific runs were frozen, the user explicitly authorized a separate native optimization track. It did not enter or change any reported ESS value. The additive API accepts candidate-specific `phi[C,T,N,M]` and `targets[C,T,M]` with shared base weights, parallelizes candidates with deterministic OpenMP float64, retains separate time-warm-start trajectories, and exposes native forward, implicit VJP, and JVP operations through tesseract. The pre-existing scalar API is unchanged.

On actual frozen N=8192 skyrmion shapes (C=8, T=13, M=4), it reduced eight scalar native calls from `0.0686` s to `0.0095` s (`7.19×`) with zero lambda discrepancy, maximum calibration residual `5.399e-11`, and all solves converged. Focused native tests pass 5/5, including bitwise repeatability, JIT/VJP smoke tests, and direct derivative errors around 1e-11.

CUDA is not required for this implementation. The installed driver is sufficient, while local `nvcc` 12.0 targets only through compute capability 9.0 and cannot compile a native RTX 5090 (`sm_120`) kernel. A future GPU kernel would require a Blackwell-capable toolkit, but the measured end-to-end priority is fused K/f and held-out audit accumulation rather than porting the already-fast small projection solve merely for language symmetry.

### Is there further computational optimization possible beyond the current multi-fidelity design?

Yes.

HIGH VALUE / LOW RISK

- Shortlist aggressively: screen all designs with risk/ESS, optimize on N=32768, use N=16384 periodic audits, and reserve N=65536 train/audit for 3–5 finalists per allowance. This removes the dominant K=280 basis/assembly work from rejected starts with no scientific-semantic change.
- Reuse the fixed K=280 dictionary and eta-independent basis value/gradient cache per bank. Stream time shards or memory-map them; never cache per-sample K×K tensors. This trades disk/host bandwidth for avoided basis differentiation, with only float64 ordering-level numerical risk already covered by equivalence tests.
- Stabilize JIT shapes (candidate batch 8 and fixed chunk sizes) and pad final batches. This avoids recompilation, costs at most seven duplicate preprocessing rows, and changes no result.

MEDIUM VALUE

- Tune/fuse chunked K/f assembly on the target GPU. Expected benefit is moderate because K assembly is bandwidth-heavy; peak memory can fall or rise with chunk size. Require before/after action and gradient checks.
- Expose outer-design lambda-trajectory warm starts in the native projection API. Physical-time warm starts already exist. A nearby-eta warm start should alter iteration count only, but implementation complexity and branch/convergence testing are medium.
- Use the new additive many-candidate native projection API when candidates share a bank. On actual 8×13×8192×4 skyrmion inputs, OpenMP batching is 7.19× faster than eight scalar native calls with zero lambda discrepancy. Projection remains a minority of Full K280 cost, so the end-to-end gain is bounded. CUDA is unnecessary for this implementation.
- Pipeline CPU-native projection with GPU observable batches. It may hide transfers, but needs bounded queues and deterministic result ordering.
- Schedule independent starts sequentially or in small batches on one GPU. Separate concurrent processes are unlikely to fit alongside the multi-GiB K280 cache and risk allocator contention.

NOT RECOMMENDED

- Per-sample K×K Gram tensors, full N=32768 K280 gradients retained on GPU without a memory plan, or unconstrained multiprocess GPU concurrency: excessive memory with no scientific benefit.
- Treating current candidate preprocessing batching as a speedup: it measured about 0.99× steady while using 8× feature-temporary memory. Retain it only as fixed-shape infrastructure or when combined with a genuinely batched projection backend.
- Differentiating through the eigensolve, changing float precision, lowering K/rank tolerance/gates, or replacing independent audits with search-bank reuse: these alter numerical or scientific semantics.

## Cheapest scientifically sound Pareto-v2 plan

Use one deterministic N=8192 risk/ESS screen over the full start pool, then K=280 optimization on N=32768 train support. Run an independent N=16384 audit at initialization, every four accepted steps, and endpoints. Use 4–6 deduplicated starts per allowance (mandatory incumbent, Law/historical anchors, and the best ESS-screened diverse starts), retain at most 3–5 finalists per allowance, and give only those finalists independent N=65536 train and N=65536 audit certification. Freeze Law/Tangent/Full winners only after selection certificates, then open one fresh validation bank once. Relative cost is roughly proportional to the number of candidates reaching K280 assembly: the ESS screen is cheap; aggressive 3–5 finalist selection avoids applying 65536 Full work to dozens or hundreds of designs.

The eventual official comparison must keep three objectives distinct: Law minimizes scientific risk; Tangent minimizes Tangent action under the exact risk ceiling; Full minimizes fixed K=280 Galerkin action under that ceiling. Every frozen geometry should be cross-evaluated on risk, Tangent action, and Full action, with the common Full comparison `A_Full(eta_Full)` versus `A_Full(eta_Tangent)` versus `A_Full(eta_Law)`. Tangent and Full actions must never be compared as though they were the same quantity.

## Limitations and exact next step

The anchor ladder has only three or four independent banks per N; extrapolations are descriptive. Staged rescoring proves existence when it finds a witness but cannot prove nonexistence. The candidate map uses selection-development data and one designated bank per stage. The error-versus-ESS subset is small. Performance varies with GPU load, cached files, and compilation state. No validation claim is made.

The exact next step is to freeze a separate Pareto-v2 protocol using the recommended multi-fidelity sizes and ESS-feasible start shortlist, then run selection only. Do not generate fresh validation until all Law/Tangent/Full selection winners and certificates are frozen.

## Repository and regression audit

The scientific ESS path accessed selection-development quantities only, ran no eta Full optimization, no Pareto sweep, no Deep Ritz solve, and no old or fresh validation evaluation. K=280, dictionary ordering/hash, rank tolerance, rESS 0.05, and energy 0.08 remained fixed. Historical report/output hashes sealed in the protocol still match. All scientific numerical output is isolated below `outputs/ess_qualification/`.

The later native/source changes are a deliberate exception to the original experiment-only write boundary, explicitly authorized by the user after the ESS protocol was frozen. They are additive performance infrastructure and did not enter the scientific records. The focused native suite passed 5/5; the ESS plus prior Galerkin, Galerkin-only, final-crosscheck, K280-quadrature, official-Pareto, and resolution suites passed 128 tests with two skips.

Final `git diff --check` passes. The final status preserves every initial dirty-worktree entry; no initial user change was reset, cleaned, checked out, or overwritten.

## Final decisions

**A. ORIGINAL 5% rESS GATE IS FEASIBLE ACROSS THE PARETO RANGE**

**READY TO DESIGN FAST PARETO V2**
