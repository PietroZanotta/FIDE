# Active-Nematic More-Training v2: Progress Report

## Purpose

This prospective development study tests whether doubling the independent
reference-training population improves the two-species active-nematic result.
The reference-training split was increased from 32 to 64 physical realizations,
while the original 16 design and 16 validation realizations retain their exact
roles and are not used for reference training.

At the endpoint times, the raw training data increased from 342 to 707 defects
per species at physical time 21 and from 395 to 788 defects per species at
physical time 31. The configured 50,000 endpoint draws remain resamples from
these raw defects; the study increases genuinely independent physical data, not
merely the number of resampled draws.

## What was changed

- Added `config_more_training_v2.json`, which freezes 64 train, 16 design, and
  16 validation runs with explicit, disjoint indices.
- Added optional explicit split-index support to `domain.py` and `run.py`.
- Added `--base-physical-bank` to `run.py`. It accepts only a compatible,
  deterministic prefix bank and verifies the physics, saved times, and seeds
  before extending it.
- Restored two read-only evaluator helper functions required by the focused
  experiment tests. All 14 focused tests passed.
- Built the experiment-local native unbalanced screened-Poisson extension and
  verified it with its production-grid benchmark and a complete smoke workflow.

No physical equation, time step, spatial resolution, defect extractor,
reference architecture, optimizer budget, validation rule, or numerical gate
was relaxed or changed.

## How source generation was made faster

The simulator itself was not mathematically accelerated. The speedup came from
avoiding redundant work:

1. The existing 64-run physical bank was verified as the exact deterministic
   prefix required by v2.
2. Only the 32 missing independent realizations were generated.
3. Those 32 runs were evaluated in two batches using the configured 16 process
   workers.
4. The original 64 trajectories were verified byte-for-byte after extension.

Measured source-stage wall times were:

| stage | wall time |
|---|---:|
| 32 additional physical simulations | 4m 34s |
| defect extraction over all 96 runs | 1m 23s |
| defect audit | <1s |
| six reference flows and rollout banks | 3m 27s |

This produced the 96-run bank without recomputing the already available 64
realizations. It changes computational reuse, not the scientific simulation.

## Completed 2% pilot

The isolated production-resolution 2% pilot completed after the native backend
was built and smoke-tested. Its measured successful runtime was 2h 23m 01s.
Every method passed all 384 held-out trials.

| method | selection Full action | validation action | validation jackknife SE |
|---|---:|---:|---:|
| Law | 22.492744 | 19.820291 | 6.157922 |
| Tangent | 21.732715 | 19.885980 | 6.423416 |
| Full | 20.596595 | 19.307252 | 1.780391 |

The held-out Full-action reduction relative to Law was 2.588%. Tangent was
0.331% higher than Law on held-out action. The pilot is a development result;
the separately nested Pareto sweep remains the primary pending calculation.

## Pause and resume state

The nested Pareto sweep was stopped on request during the 1% exact Full screen.
The 0.5% Law/Tangent/Full selection point had already passed exact
certification and is safely checkpointed. No Pareto validation had begun.

The sweep was subsequently resumed and stopped again on request while the 1%
Tangent optimizer had completed 3 of 15 starts. The runner reused the frozen
0.5% result correctly, but the interrupted 1% point still has not reached its
atomic `result.json` checkpoint. Therefore 0.5% remains reusable and 1% must
restart under whichever execution implementation is selected.

## Multistart-acceleration engineering study

Profiling showed that the Pareto process used approximately one CPU core even
though the machine exposes 24 cores. The active-nematic selection code had
deliberately called the shared multistart optimizer with
`vectorize_starts=False`, causing independent Adam starts to run one after
another. The optimization and audit stages were kept scientifically unchanged;
only alternative execution mechanisms were tested.

Both experimental implementations remain available for inspection alongside
the original serial path:

- `src/mfsi/design.py` retains the original serial single-start loop, a bounded
  fixed-shape JAX-vmap implementation, and a threaded implementation that calls
  the original compiled single-start executable.
- `run_pareto.py --multistart-backend serial` is the default and reproduces the
  frozen methodology.
- `run_pareto.py --multistart-backend threaded --multistart-workers 2` exposes
  the engineered threaded implementation without changing the scientific
  configuration hash.
- `benchmark_multistart_acceleration.py` runs isolated serial and threaded smoke
  workflows and writes a machine-readable comparison.

Twenty-seven focused tests pass, including exact equality of serial and threaded
execution on a deterministic unit objective and a tight numerical comparison
of serial and bounded-vmap execution. Real active-nematic objectives nevertheless
showed that unit-level equivalence was not sufficient.

### Rejected bounded-vmap attempt

The first attempt evaluated two starts per fixed-shape JAX batch. All frozen
configs, observation banks, defect banks, reference checkpoints, and reference
rollout banks were SHA-256 identical between the serial and batched arms. Even
so, the exact smoke Law anchor changed from 154.909889 to 154.423950. The Full
stage then failed because its native screened-Poisson `pure_callback` has no JAX
vmap rule. This implementation is not valid for the authoritative workflow.

### Serial versus two-worker threaded benchmark

The second attempt overlapped calls to the unchanged, compiled single-start
executable in two threads. It avoided the Full callback failure. A fresh 2%
smoke Pareto workflow, including selection, exact audits, and validation, gave:

| execution | end-to-end wall time | relative speed |
|---|---:|---:|
| serial | 282.763 s | 1.000x |
| threaded, 2 workers | 266.265 s | 1.062x |

The wall-time reduction was 16.498 s, or 5.8%. Exact audits and the guarded
scalar validation fallback remain serial, which limits the attainable
end-to-end speedup even when optimizer starts overlap.

The threaded result was not numerically equivalent:

| quantity | serial | threaded | absolute difference |
|---|---:|---:|---:|
| exact Law anchor | 154.909889 | 154.164980 | 0.744908 |
| selected Tangent action | 2,745,174.115 | 2,648,834.663 | 96,339.452 |
| selected Full action | 569,646.178 | 647,477.459 | 77,831.281 |
| validation Law action | 646,868.035 | 654,247.230 | 7,379.194 |
| validation Full action | 629,766.551 | 654,247.230 | 24,480.679 |

The maximum coordinate difference was 0.008 for Law/Tangent and 0.012 for
Full. Both arms happened to certify, but they did not select the same designs
or report the same scientific values. The machine-readable receipt is
`outputs/more_training_v2_multistart_benchmark_v2/benchmark.json`.

### Acceleration decision point

The current evidence does **not** support using either engineered backend for
the frozen authoritative sweep. The two-worker backend is only 1.062x faster
on the complete smoke workload and changes the result. Retaining serial
execution is the conservative recommendation.

A future acceleration attempt should use process isolation rather than vmap or
threads, so every worker owns independent JAX/native state while executing the
unchanged single-start path. It should first add sub-stage checkpoints, then
prove identical optimized candidates, selected geometries, exact audit values,
and certification on a realistic small workflow before production use. Exact
candidate/view audits could also be considered for process-isolated
parallelism, because they dominate the non-optimizer portion of runtime.

Checkpoint reuse is independent of this benchmark choice only when the
scientific configuration and exact result semantics remain unchanged. The
certified 0.5% result can be reused. There is no completed 1% result to reuse,
so 1% must be recomputed. If a future acceleration changes scientific settings
(seeds, views, trials, starts, grids, or tolerances), neither checkpoint should
be reused under the new configuration.

## Exact-evaluation reuse engineering study

A second speedup study retained the original serial optimizer and native scalar
authority path. It removes only repeated evaluations whose inputs have already
been evaluated exactly:

1. Complete-bank selection audits are memoized by metric name, observation-bank
   object, and the byte representation of the canonical float64 geometry. No
   tolerance or nearest-geometry lookup is permitted.
2. Cached certified Pareto receipts hydrate this in-memory audit cache on
   resume. The existing 0.5% Law/Tangent/Full receipts can therefore seed the
   restarted 1% calculation.
3. Validation results are reused when two method labels or Pareto allowances
   contain the exact same canonical geometry. The saved rows, summaries,
   standard errors, validity flags, and action decomposition are copied intact.
4. Once a validation view's batch/scalar guard has failed, that exact view
   remembers that scalar evaluation is authoritative and skips subsequent
   known-failing batch probes. It still evaluates every requested scalar trial.

The feature is execution-only and opt-in:

```bash
.venv/bin/python experiments/active_nematic_unbalance_percentage/run_pareto.py \
  --config experiments/active_nematic_unbalance_percentage/config_more_training_v2.json \
  --input-dir experiments/active_nematic_unbalance_percentage/outputs/more_training_v2_source \
  --output experiments/active_nematic_unbalance_percentage/outputs/more_training_v2_pareto \
  --multistart-backend serial \
  --reuse-exact-evaluations
```

It does not change the scientific configuration hash. It does not change
seeds, physical views, trials, candidate sets, optimization steps, grids,
solvers, tolerances, or certification ceilings.

### End-to-end smoke timing

A fresh 2% smoke Pareto workflow was run once with baseline serial evaluation
and once with serial evaluation plus exact-work reuse:

| execution | wall time | relative speed |
|---|---:|---:|
| serial baseline | 275.625 s | 1.000x |
| serial plus exact-work reuse | 216.989 s | 1.270x |

The optimized arm saved 58.636 s, or 21.3% of baseline wall time. It recorded
2 exact-audit cache hits and 10 misses during selection, then 1 validation hit
and 2 validation misses. The whole scientific payloads were not directly
comparable because the two independent serial processes landed on alternate
Law optimizer endpoints before exact caching was invoked. Repeated serial runs
showed the same two endpoints, exposing pre-existing process-level optimizer
numerical nondeterminism; this is not evidence of a cache-induced candidate
change.

### Fixed-selection validation replay

To remove optimizer drift, both arms were seeded with the same frozen selection
receipt containing exactly the same Law, Tangent, and Full geometries. This
isolated validation and its scalar authority path:

| execution | wall time | relative speed |
|---|---:|---:|
| validation baseline | 126.757 s | 1.000x |
| validation with reuse | 69.575 s | 1.822x |

The optimized path saved 57.181 s, or 45.1%. The baseline emitted six failed
batch probes (two views times three method labels). The optimized path emitted
two, reused the identical Law/Tangent geometry, and evaluated the other unique
geometry directly with scalar authority.

The selection geometries and all validity flags were identical. Across 690
numeric validation values, baseline and reuse agreed at `rtol=1e-8` and
`atol=1e-7`. The largest absolute difference was 0.000128 in a reaction-action
trial whose actions are of order 10^5--10^6. The largest relative difference
was 8.8% in a screened-PDE residual, but its absolute difference was only
1.98e-8 and both values remained far below the 1e-5 validity ceiling. These
small differences also occur when the legacy path reevaluates the same geometry
twice; they arise from repeated native iterative solves. Reuse returns the first
certified scalar receipt exactly instead of invoking another numerically noisy
duplicate solve.

Machine-readable receipts are:

- `outputs/more_training_v2_evaluation_reuse_benchmark_v1/benchmark.json`
- `outputs/more_training_v2_fixed_validation_reuse_benchmark_v1/benchmark.json`

### Recommendation

Exact-work reuse is worth adopting with the original serial optimizer. Unlike
threaded/vmap optimization, it cannot introduce a different candidate: a hit
requires the exact canonical geometry and bank, and returns an audit that was
already computed by the unchanged authority path. The measured complete-smoke
speedup was 1.270x and fixed-validation speedup was 1.822x.

Production benefit will depend on how often Tangent/Full incumbents repeat. A
conservative expectation is roughly 15--25% total wall-time reduction, with a
larger reduction in the final validation phase. This would put the remaining
serial sweep near roughly 10--14 hours rather than the earlier 12--16-hour
estimate, but it is still an estimate until the 1% production point completes.

The native screened-Poisson batch solver already uses static OpenMP
parallelization. No additional safe thread toggle was found. Persistent JAX
compilation caching would mainly reduce restart overhead, not the uninterrupted
multi-point production run. Neither was changed.

Frozen inputs are normally hard-linked into the Pareto output. Resume-time
verification now detects inode identity before hashing; this reduced the local
264 MB physical-bank check from 0.485 s to effectively constant time. Copied
artifacts still use the original SHA-256 comparison. This is a small restart
improvement, not a production evaluation speedup.

## Further engineering-speedup evaluation

Four additional areas were implemented or measured after exact-work reuse.

### Stable observation-bank prefixes: adopt

Law, Tangent, Full-gradient, and Full-prescreen stages repeatedly take exact
prefixes of the same frozen observation bank. A retained prefix cache now gives
each root-bank/trial-count pair one stable object identity across allowances.
This allows the exact-audit cache to recognize a repeated Full incumbent on the
8-trial prescreen bank. Prefixes from different root banks or with different
trial counts cannot alias.

A production-resolution robust Full prescreen of the certified 0.5% incumbent
was timed on the 8-trial prefix across all 12 physical/reference views:

| operation | wall time |
|---|---:|
| first authoritative prescreen | 36.7862 s |
| exact repeated-prefix cache hit | 0.0018 s |

The saved time was 36.7843 s per repeated prescreen. Cached point receipts now
hydrate both complete-bank audits and certified prescreen values, so the 0.5%
incumbent can seed the restarted 1% point. At least one incumbent is carried to
each subsequent allowance, implying a minimum saving of roughly three minutes
over 1--5%, with additional savings when Law/Tangent/Full geometries recur.

Enable it with `--reuse-prefix-banks`. It does not alter bank contents or trial
counts.

### Scalar geometry reuse: do not adopt

An optional path computes sensor features and feature gradients once per
species/view/geometry and reuses them across authoritative scalar trials. Both
arms used the same frozen selection, exact-result reuse, stable prefixes, and
four validation trials per view:

| execution | wall time | relative speed |
|---|---:|---:|
| without geometry reuse | 158.853 s | 1.000x |
| with geometry reuse | 159.175 s | 0.998x |

The change was neutral within timing noise and made validation 0.323 s slower
in this run. All 1,266 numeric values agreed at `rtol=1e-8` and `atol=1e-7`,
and all nonnumeric values and designs agreed. The implementation remains behind
`--reuse-scalar-geometry` for reproducibility, but should remain disabled.
The receipt is
`outputs/more_training_v2_scalar_geometry_benchmark_v1/benchmark.json`.

### Full-only Tangent omission: not worth enabling

Full-only objectives historically computed the complete Tangent action even
though the caller discarded it. An optional execution path omits that unused
branch. On the production Full proxy objective/gradient (three reference views,
four trials, 24x24x12 grid), three warm evaluations gave:

| execution | median objective/gradient time | relative speed |
|---|---:|---:|
| legacy Full plus discarded Tangent | 1.9231 s | 1.000x |
| Full without discarded Tangent | 1.8967 s | 1.014x |

The objective was exactly equal and the maximum gradient difference was
1.11e-16. The gain is only 1.4% of this sub-operation and well below 1% of the
complete workflow, so `--skip-unused-tangent-for-full` should remain disabled.

### Experiment reconstruction/JAX reuse: not worth restructuring

Loading references and constructing the initial 12 production views took
5.222 s once. Rebuilding all 12 views for a new allowance took only 0.454 s,
or less than three seconds over the remaining five points. Progress timings
also showed no large first-start compilation premium relative to subsequent
starts. A lightweight-clone or parameterized-JIT refactor would add meaningful
implementation risk for negligible measured wall-time benefit, so it was not
implemented.

### Updated execution recommendation

Use the original serial optimizer with exact-work and stable-prefix reuse:

```bash
--multistart-backend serial \
--reuse-exact-evaluations \
--reuse-prefix-banks
```

Leave `--reuse-scalar-geometry`, `--skip-unused-tangent-for-full`, and the
threaded optimizer disabled. The additional prefix saving is real but modest,
so the overall remaining-runtime estimate stays approximately 10--14 hours.

## Still missing

- Finish exact selection at 1%, 2%, 3%, 4%, and 5%.
- Run held-out validation for Law, Tangent, and Full only after every allowance
  winner is frozen.
- Verify nested Full action, all per-view risk ceilings, all validation trials,
  and the move/reaction action decomposition.
- Run the independent Pareto finalizer and produce the finalized table/report.
- Compare the completed v2 curve with the original 32-training-run study and
  determine whether the larger physical training pool materially improves
  reference robustness and held-out action.
