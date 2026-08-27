# Fast production 3% evaluation

## Scope and repository safety

This checkpoint accelerates and tests only the validated fixed-feature production Galerkin route in `experiments/skyrmions_deep_ritz_full/`. It does not validate the historical nonlinear fixed-theta envelope derivative, modify the production incumbent, run a Pareto sweep, or write outside the isolated experiment.

Initial status was `?? ETA_OPTIMIZATION_AUDIT.md` and `?? experiments/skyrmions_deep_ritz_full/`. The unrelated root audit was preserved. Production artifacts and prior validated outputs were read-only. New numerical outputs are confined to `outputs/fast_production_3pct/` and ignored by Git.

## Profile and bottlenecks

All timings are synchronized wall-clock measurements on the real frozen artifacts, float64 JAX, CPU, K=160. The old path rebuilt the eta-independent basis/Jacobians for both K/f and potential/kinetic rows. That dominated design evaluation. The rank-aware eigensolve was only about 0.025 s and was retained.

The nonlinear solver is a separate, larger bottleneck. One selection rescore took 3,833.40 s and validation took 6,942.07 s. The complete 21-stage record is `outputs/fast_production_3pct/profiling/result.json`.

## Eta-independent cache and hot path

Eta-independent quantities are the frozen banks/velocities, base and quadrature weights, time nodes, dictionary, normalization, basis values, state Jacobians, artifact metadata, and test definitions. Eta-dependent quantities remain sensor features/JVPs, c/cdot, information projection, weights, lambda/lambda_dot, h, K/f, coefficients, action, gradient, risk, geometry, and diagnostics.

Implemented changes:

- `FastProductionContext` loads arrays once and retains stable-shape JAX functions.
- Float64 K=160 train basis values/Jacobians are sharded by time and keyed by artifact, dictionary, shape, dtype, normalization, and config hashes.
- K, f, potential, and kinetic rows use compiled cached-basis contractions.
- Fixed-coefficient envelope and risk closures compile once per context.
- Search omits held-out certification; full certification runs only at promotion points.
- Rank-aware solves, rank/range/conditioning checks, and all scientific tolerances are unchanged.
- Physical-time projection warm starts remain active. Outer-design starts were not emulated because the fixed native API does not expose a compatible trajectory initializer.
- Authoritative results are keyed by exact eta, artifact hash, solver settings, validation flag, and certificate settings.

## Memory design

| Bank | values | gradients | total | cached |
|---|---:|---:|---:|---|
| train, 8,192/time | 0.127 GiB | 4.062 GiB | 4.189 GiB | disk-sharded and resident |
| audit, 4,096/time | 0.063 GiB | 2.031 GiB | 2.095 GiB | no; finalist only |
| validation fit, 16,384/time | 0.254 GiB | 8.125 GiB | 8.379 GiB | no |
| validation audit, 16,384/time | 0.254 GiB | 8.125 GiB | 8.379 GiB | no |

A 20.313 GiB per-sample train Gram tensor was rejected. Approximate steady host storage is 4.19 GiB plus banks/JAX workspaces; peak RSS was not separately captured. Validation feature caches are intentionally not resident.

## Numerical equivalence

| Quantity | discrepancy | result |
|---|---:|---:|
| risk relative | 4.99e-16 | pass |
| c / cdot max absolute | 0 / 0 | pass |
| lambda / weights max absolute | 0 / 0 | pass |
| lambda_dot / h max absolute | 0 / 0 | pass |
| K relative | 1.85e-16 | <= 1e-10 |
| f relative | 2.67e-15 | <= 1e-10 |
| coefficients relative | 1.84e-6 | <= 1e-5 |
| action relative | 8.64e-14 | <= 1e-10 |
| eta gradient relative | 2.87e-10 | <= 1e-8 |
| rank | identical | pass |

Coefficient sensitivity is confined to ill-conditioned retained eigendirections; action and gradient meet much tighter invariance gates.

## Performance

| Stage | Before | After | Speedup |
|---|---:|---:|---:|
| artifact/context initialization | 10.638 s | 8.079 s | 1.32x |
| repeated basis/potential construction | 4.635 s | cached | eliminated |
| coupled basis + K/f assembly | 5.596 s | 1.972 s | 2.84x |
| rank eigensolve | 0.025 s | 0.025 s | unchanged |
| value-only | 5.596 s | 2.415 s | 2.32x |
| value+gradient first call | 23.295 s | 4.199 s | 5.55x |
| value+gradient steady median | 10.771 s | 2.653 s | **4.06x** |
| held-out certification | 10.726 s | 10.726 s | scheduled, not weakened |
| authoritative selection rescore | — | 3,833.403 s | one new call |
| authoritative validation | — | 6,942.066 s | one new call |

Twenty-three Galerkin evaluations promoted one new selection candidate; two exact prior candidate results were reused. This avoided 22 new selection-side authoritative calls, about 23.4 CPU-hours at the measured finalist cost.

## Basis-gradient convergence

| K | action | gradient norm |
|---:|---:|---:|
| 100 | 0.234225412 | 2.501276745 |
| 120 | 0.250027556 | 2.716352750 |
| 140 | 0.256741099 | 2.818511684 |
| 160 | 0.264577197 | 2.903936994 |

| Pair | cosine | relative gradient difference | relative action difference |
|---|---:|---:|---:|
| 100/120 | 0.999609 | 0.08360 | 0.06320 |
| 120/140 | 0.999878 | 0.03935 | 0.02615 |
| 140/160 | **0.999970** | **0.03040** | 0.02962 |

The declared K=140/160 gates (cosine >= 0.995, relative gradient difference <= 0.05) passed. K=160 remains necessary because action changes remain non-negligible.

## Nearby-point derivative audit

Four deterministic feasible points had K=140/160 cosine 0.99996963–0.99996968 and relative gradient differences 0.03039–0.03044. Two points received two directions and four epsilons each. Best relative AD/optimized-FD discrepancies were 2.04e-4, 4.34e-4, 3.46e-5, and 2.71e-4. All gates passed.

## Trust region and multistart

The periodic minimum-image trust radius was 2e-4. Exact geometry, exact risk, forcing validity, and rank stability gated every step. Four accepted steps reached:

```text
[0.8953823878118340, 0.2059516018060983,
 1.3345214477327900, 0.8654767605276741,
 0.7508063016882840, 0.5179759437004340,
 1.6423946881431137, 0.5884132119429537]
```

K=160 action fell from 0.264577197 to 0.264051106; risk was 5.342145370 versus ceiling 5.342145959. Four starts ended at actions 0.264051106, 0.264051858, 0.264066974, and 0.264078478. The two best passed held-out Galerkin certificates.

## Scientific selection

| Candidate | selection risk | K=160 action | authoritative action | projection | ESS | forcing | weak | energy | gauge | moment | certified |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| old incumbent | 5.340106051 | 0.264577197 | 0.278566384 | 1.47e-11 | 0.06916 | 2.49e-9 | 0.07660 | 0.07070 | 2.73e-15 | 0.01453 | yes |
| previous tiny update | 5.342045520 | 0.264076787 | **0.274114839** | 1.34e-11 | 0.06950 | 2.25e-9 | 0.07908 | 0.06687 | 2.31e-15 | 0.01314 | yes |
| best new continuous | 5.342145370 | **0.264051106** | 0.278070013 | 1.33e-11 | 0.06951 | 2.24e-9 | 0.07759 | 0.06980 | 1.40e-15 | 0.01273 | yes |
| second finalist | 5.342142430 | 0.264051858 | not run | — | — | — | 0.07382 | 0.05003 | 1.56e-17 | 0.00792 | Galerkin only |

The new point improves the old incumbent by 0.00049637 but is worse than the prior tiny update by 0.00395517. The frozen winner is therefore the previous tiny update:

```text
[0.8953839921146673, 0.20595035907471138,
 1.3345144773868762, 0.8654744150451203,
 0.7508077339024882, 0.5179727362721115,
 1.6423936578820195, 0.5884106107337586]
```

No production incumbent was modified.

## Sealed independent validation

Validation began only after selection was frozen. The old incumbent reused its matching frozen production validation; the winner received one new solve initialized from the validation checkpoint.

| Candidate | validation risk | ceiling | action ± SE | weak | energy | gauge | moment | valid |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| old incumbent | 5.548626548 | historical record | 0.230969866 ± 0.000615731 | 0.03189 | 0.04912 | 3.03e-16 | 0.00720 | yes historically |
| frozen winner | 5.550521383 | 5.518713758 | 0.280304513 ± 0.000910206 | 0.02769 | **0.10899** | 1.02e-15 | 0.00697 | **no** |

This is a validation reversal. The winner fails the independently recomputed validation risk ceiling and 0.08 energy threshold, and its action is higher than the incumbent. Projection, ESS, forcing, weak, gauge, and moment-rate checks pass. Selection was not revisited.

## GPU feasibility and next checkpoint

The host exposes an RTX 5090 Laptop GPU with 24,463 MiB VRAM. Sandboxed JAX cannot initialize CUDA, but an approved external check reports JAX 0.8.3 and `CudaDevice(id=0)`. Production eta0 risk was 5.340106050966007 on CUDA versus 5.34010605096599 on CPU (~3e-15 relative). This proves graph compatibility, not optimizer/certificate equivalence.

Next: validate fixed-network objective/gradient, one-step Adam, compiled full-bank accumulation, certificates, and restart-level caching on CPU/GPU before using GPU-trained actions scientifically.

## Commands and resume

```bash
python -m experiments.skyrmions_deep_ritz_full.run --mode benchmark-production-galerkin --frozen-source PATH
python -m experiments.skyrmions_deep_ritz_full.run --mode production-gradient-convergence --frozen-source PATH
python -m experiments.skyrmions_deep_ritz_full.run --mode production-local-gradient-audit --frozen-source PATH
python -m experiments.skyrmions_deep_ritz_full.run --mode production-refine-3pct --frozen-source PATH
python -m experiments.skyrmions_deep_ritz_full.run --mode production-multistart-3pct --frozen-source PATH
python -m experiments.skyrmions_deep_ritz_full.run --mode production-authoritative-3pct --frozen-source PATH
python -m experiments.skyrmions_deep_ritz_full.run --mode production-validate-3pct --frozen-source PATH --input-result experiments/skyrmions_deep_ritz_full/outputs/fast_production_3pct/selection/result.json
```

Caches validate hashes/settings. Missing artifacts remain a hard stop. No command launches a sweep.

## Tests, limitations, and final status

The fast suite now has six checks: the original path isolation, cache memory
accounting, periodic trust displacement, and gradient metrics tests, plus exact
reference/compiled full-bank solver equivalence and fail-closed paired-ordering
consensus. Together with the unchanged scientific suites, all 62 tests pass.

Limitations: independent validation reverses the improvement; K=140/160 gradient magnitude changes 3.04%; coefficients are conditioning-sensitive; outer-design projection starts are unavailable; authoritative CPU solves remain hour-scale without phase checkpointing; fixed-checkpoint GPU evaluation is equivalent but optimized action ordering is not restart/platform stable, as documented in `AUTHORITATIVE_GPU_ACCELERATION.md` and `AUTHORITATIVE_STABILITY_EVALUATION.md`; no Pareto sweep ran.

All source/report changes are under `experiments/skyrmions_deep_ritz_full/`; generated data are under its ignored fast-output subtree.

B. FAST CONTINUOUS 3% REFINEMENT VALIDATED, NO FURTHER AUTHORITATIVE IMPROVEMENT
