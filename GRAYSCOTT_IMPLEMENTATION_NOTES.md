# Gray–Scott Experiment C implementation notes

## Repository inspection

The repository has no current Experiment C implementation or entry point in the
working tree or reachable Git history. Experiment B remains at `example_b.py`,
with its artifacts under `results/example_b/`; neither is modified by this
work. The implementation brief suggests `expB_grayscott_*` names, but the user
requested a distinct Experiment C. New files therefore use `expC_grayscott_*`
and write only below `results/grayscott/`.

Relevant validated implementation:

- `mfsi_components.py`: empirical exponential-tilt calibration
  (`calibrate_empirical_implicit`, `empirical_tilt_from_lambda`), stable
  covariance solves (`_stable_cov_solve`), multiplier differentiation and
  forcing (`empirical_fiber_state`), tangent/safety moment geometry,
  weighted-RBF MMD, reference flow matching, Deep-Ritz training, and Heun
  rollout.
- `example_b.py`: Experiment B's stochastic bridge, reference-velocity fit,
  projected-bank construction, tangent baseline, learned MFSI, MFSI+safety,
  MGD-style baseline, weighted MMD, fixed-seed comparisons, and report/plot
  conventions.
- `level2_paper_study.py`: independently weighted target MMD, schedule/path
  diagnostics, held-out banks, seed-level paired intervals, and method-blind
  readiness criteria.
- `stage3_rollout_adaptation.py` and `stage3b_confirmatory.py`: disjoint
  adaptation/selection/evaluation roles, differentiable rollout adaptation,
  stopped-state controls, and confirmatory seed aggregation.
- `tesseracts/moment_fiber_realizer/tesseract_api.py`: componentized form of
  empirical projection, forcing, and Ritz correction.
- `mgd.py`: trustworthy MGD implementation only for the existing low-dimensional
  polynomial-moment setting.

## Reuse and generalization

The empirical I-projection solver accepts arbitrary `[bank, R]` feature arrays
and is reused unchanged for Gray–Scott endpoint calibration. Its stable
covariance diagnostics and normalized weights are also reused. The forcing
algebra, implicit multiplier derivative, safety/tangent equations, weighted
MMD definition, Heun integration pattern, bank separation, and crossed-seed
aggregation are mathematically reusable.

The Experiment B samplers, analytic observation Jacobian, MLP reference field,
MLP scalar potential, and rollout tensors assume states shaped `[B, 2]` (or the
older scalar `[B]` case). Experiment C therefore needs field-aware observables,
JVPs, a periodic CNN reference velocity, a translation-invariant CNN energy,
and field-preserving rollout plumbing for `[B, 1, H, W]`. These will be added
alongside, not by changing, Experiment B.

The existing `ensemble_safety_velocity` contracts a scalar-state Jacobian with
the velocity elementwise and cannot be called unchanged on image fields. Its
equation and `_stable_cov_solve` can be reused once the contraction is made
over channel and spatial axes. `weighted_mmd_rbf` is scalar-state-specific;
the four-weight version in `level2_paper_study.py` is the better starting point
for flattened/downsampled fields. Existing reference and Ritz networks are
MLPs and do not meet periodic translation-equivariance requirements.

## Mathematical and numerical mismatches

- `mfsi_components.empirical_fiber_state` defaults to the scalar polynomial
  `phi/jphi`; Gray–Scott must pass field features and `J_Phi u` explicitly.
- Existing empirical calibration uses 20 damped Newton iterations from zero.
  The design code selects the common target with a separate symmetric
  centrality solve, then calls the validated repository calibration for each
  endpoint and reports any failure rather than changing the target.
- No existing morphology/topology implementation handles a periodic image
  domain. Experiment C supplies periodic connected components, cubical Euler
  characteristic, interface length, anisotropy, and spectral metrics.
- No trustworthy field-valued MGD implementation exists. Per the brief, MGD is
  deferred rather than independently reinvented.
- The brief contains duplicated headings/lines and uses Experiment-B filenames
  for the proposed Gray–Scott replacement. This implementation treats those as
  naming examples, not instructions to overwrite Experiment B.

This first implementation stage intentionally stops after the design scan and
endpoint calibration gates. It does not train or compare learned MFSI and
tangent models before a benchmark is selected and frozen.

## Phase-2 continuation and field infrastructure (v6)

`experiments/grayscott/feasibility.py` now supplies:

- exact common empirical hull feasibility through SciPy/HiGHS;
- a maximum-minimum-weight interior LP;
- maximum-total-entropy central targets through a convex low-dimensional dual;
- fixed-target empirical hull membership LPs;
- a residual-driven, damped Newton/Armijo I-projection wrapper with full traces;
- independent final weight/moment verification through the repository
  `empirical_tilt_from_lambda` implementation.

The wrapper has no fixed 20-step convergence claim. It records the initial and
final dual objective, every residual/gradient norm, line-search steps, damping,
multiplier norm, covariance spectrum/rank/condition, ESS, entropy, and maximum
weight. It also records both `weights @ Phi - c` and the repository-reported
residual and requires them to agree numerically.

`experiments/grayscott/phase2_continuation.py` writes only to the versioned
`results/grayscott/phase2_feasibility_v6/` directory. It hashes and rechecks all
earlier Phase-2 failure artifacts. The 256-seed dense local scan evaluates the
nested Phi-2/Phi-3/Phi-4 families with a single pooled center/scale per physical
feature; target selection and calibration slice exactly the same transformation.

`experiments/grayscott/field_transport.py` adds the Phase 3–4 field interfaces:

- `[B,1,H,W]` linear interpolant state and derivative plumbing;
- an exact-marginal maximal same-initial-condition endpoint coupling;
- an untrained periodic, translation-equivariant, time-conditioned CNN
  reference velocity with dilated receptive field;
- `J_Phi @ u` through JAX JVP and a small-grid explicit-Jacobian parity test;
- weighted field tangent contractions across channel/spatial dimensions;
- smooth TV, structure-tensor anisotropy, soft area/perimeter, and held-out
  high-frequency power;
- a generic oracle-at-target tangent blind-spot residual and `B_tan` routine.

Separate endpoint tilts generally give different weights to the same paired IC
indices, so a literal all-mass same-index coupling cannot preserve both calibrated
marginals. The implemented coupling puts the maximum mathematically possible
mass on the diagonal and couples residual mass independently. This discrepancy
from the preferred unweighted same-IC wording is explicit in the Phase-3 output.

The CNN architecture and blind-spot routine are implemented and numerically
tested but not trained/evaluated. The user prohibited reference training in this
continuation, and the linear path independently fails the interior projection
gate, so reference FM training would be premature. No Deep-Ritz field potential
or learned-method comparison was added.

The Gray–Scott test suite now has 19 tests, including exact LP feasibility,
positive-target I-projection recovery, seed separation, JVP/Jacobian parity,
weighted tangent-rate cancellation, coupling marginals, CNN translation
equivariance, and finite smooth-hidden gradients.

## Phase-3 reference implementation v7

`field_transport.py` now also contains independent exact-marginal coupling,
pairwise field-L2 costs, an exact weighted transportation LP, centered unit-RMS
field noise, and the finite-derivative `A*sin(pi*t)` bridge. The LP uses a sparse
full-row-rank transportation constraint matrix and HiGHS. Tests require exact
marginals and verify its cost is no worse than independent coupling.

`phase3_reference_design.py` implements the versioned design workflow:

- hash-preserve the v6 directory;
- retain and screen all fourteen frozen Phase-2 targets;
- serialize the analytic/empirical second-moment identity for ordinary and
  smoothstep parameterizations;
- evaluate maximal-same-IC, geometric OT, and independent couplings;
- record raw physical/standardized Phi summaries, fixed-target hull LPs,
  calibration traces, ESS, KL, multipliers, covariance spectra/rank/condition,
  hidden shifts, and unclamped field ranges;
- screen the fixed scalar noise-amplitude grid only after all linear paths fail;
- confirm the selected path with a doubled, independently sampled bank.

The calibration wrapper targets `1e-10`, while the frozen Phase-3A gate is
`1e-5`. An initial confirmation label incorrectly required the optional solver
flag in addition to the gate. `reassess_saved_schedule_gates` corrects that
logic without recomputing or changing any result: all solver flags remain
reported, but selection uses exact hull feasibility and the declared residual
gate. The 8192-bank confirmation has three solver warnings yet a worst residual
of only `1.78e-8`, so it legitimately passes Phase 3A.

`phase3_reference_training.py` generates new simulator endpoint banks with the
unchanged IC/simulation protocol, calibrates the frozen target, trains the
periodic CNN with Adam, evaluates a disjoint held-out interpolant bank, and
integrates a 64-step Heun rollout. A float32 initializer regression was found
before successful optimization: a NumPy scale scalar had promoted convolution
weights to float64. `_init_conv` now casts the scale explicitly, and a test
requires float32 weights/output when float32 is requested.

The single 4000-step training run completed and wrote its checkpoint before a
post-evaluation JSON serialization error on `numpy.bool_`. Evaluation was then
reproduced from that saved checkpoint; training was not rerun. The result
records the missing wall-clock duration explicitly rather than reconstructing
or fabricating it. Both reference quality gates fail, so the code stops before
the implemented tangent blind-spot routine. No Phase-4 result, Deep-Ritz
potential, benchmark selection, or learned comparison exists.

The Gray–Scott suite now contains 23 passing tests. New v7 coverage includes
geometric/independent marginal preservation, OT cost ordering, fixed-endpoint
noise normalization, the bridge second-moment identity under reparameterization,
and requested CNN dtype preservation.

## Phase-3B reference-quality implementation v8

`experiments/grayscott/phase3_reference_quality.py` and
`scripts/run_grayscott_phase3_quality_v8.py` implement the v8-only diagnostic
workflow. The frozen v7 endpoint banks, target, standardization, coupling, and
bridge are consumed read-only. Source manifests cover Phase-2 v6, Phase-3 v7,
and Experiment B; checkpoints and all v8 outputs have SHA-256 records in the
new result directory.

The raw-reference audit uses a four-weight median-RBF MMD generalized from the
validated Experiment B convention. It compares learned Heun rollouts with
independent direct samples from the exact stochastic interpolant, both at raw
64x64 resolution and after deterministic 4x block averaging. A separate
empirical I-projection is solved at each time to verify the fixed target. Raw
SI-versus-target and projected-versus-target diagnostics are never conflated
with learned-versus-direct reference fidelity.

FM instrumentation now reports global and fixed-time MSE, normalized MSE,
predicted/target velocity RMS, cosine alignment, and radial low/mid/high
frequency error. Tests cover the exact analytic bridge derivative against a
shared-`Z` finite difference, empirical endpoint marginals, the semantic
separation of raw SI and fiber projection, a known-vector-field Heun rollout,
an oracle bridge integration, validation-bank replacement logic, residual-CNN
translation equivariance, and generic NumPy-scalar JSON serialization.
`phase2_continuation._json_default` accepts all `np.generic` scalars, including
`np.bool_`, via `.item()`.

The validation-bank builder follows a deterministic append-only rule: start
at seed 61001, simulate contiguous 1024-seed endpoint chunks with the unchanged
Gray–Scott/IC protocol, calibrate the frozen target, and append only if either
endpoint ESS is below `0.20`. The first chunk passed with minimum ESS
`0.51489`; the saved bank records seeds 61001--62024 and target equality.

The controlled-sweep protocol was serialized before v8 training. Variant A
loads the exact v7 checkpoint; B resets AdamW state and continues it for 8,000
steps with cosine decay; C initializes a 28-channel periodic residual CNN with
dilations `1,2,4,8,4` and trains for 12,000 steps. All share the same frozen
training roles, fixed healthy validation draws, continuous-time objective,
batch size 64, global clip 5, and reference-only selection rule. Checkpoints,
training curves, aggregate tables, and per-time rollout CSVs are retained.

The residual CNN remains periodic and translation-equivariant. It has 37,717
parameters versus 17,593 for v7, raw time plus three Fourier frequencies, and
an approximately 43-pixel receptive field. Its improved local FM ratio
(`0.19496`) did not override the failed rollout gate. The 64/128/256 Heun audit
uses a common 1/16 time grid and classifies the residual discrepancy as model
flow accumulation, not numerical integration error.

Reproducibility metadata in `run_metadata.json` records the git commit and
dirty status, Python/JAX/NumPy versions, JAX device and x64 setting, config and
protocol hashes, seeds, optimizer, time sampling, ODE steps, checkpoint hashes,
model kind, and parameter counts. `preserved_source_manifests.json` and
`v8_artifact_sha256.json` provide source/result hashes. The final Gray–Scott
suite has 31 passing tests.

Phase 3B is false in `phase3b_final_decision.json`. The code therefore does not
enter Phase 4, compute `B_tan`, train Deep-Ritz MFSI, create benchmark selection,
or run any learned-method comparison.

## Final global spectral reference implementation v9

`experiments/grayscott/field_transport.py` now provides
`init_spectral_reference_model`, `spectral_conv2d`, and
`spectral_reference_model`. The spectral layer uses normalized JAX `rfft2` and
`irfft2`, separately learned positive/negative vertical low-mode complex
multipliers represented by float32 real/imaginary arrays, and a real-valued
inverse. Every block adds a learned physical-space pointwise path before SiLU
and a normalized residual connection. Spatially constant raw/Fourier time
channels preserve translation equivariance. The output has a learned direct
physical-space skip and is not clipped.

`experiments/grayscott/phase3_reference_global.py` implements four explicit
stages:

1. `prepare` creates the append-only v9 directory, validates exact target and
   standardization identity, records v6/v7/v8 manifests, and freezes the full
   protocol before held-out rollout evaluation;
2. `train` performs the one 18,000-step raw-SI FM run and saves the best
   healthy-validation checkpoint and full optimization trace;
3. `evaluate` applies the exact same fresh draws to the unmodified v8 control
   and v9 spectral model, writing per-time FM bands, short-horizon diagnostics,
   rollout/Phi/MMD/hidden/radial-power trajectories, summaries, and plots;
4. `finalize` evaluates the predeclared adaptation trigger, performs paired
   64/128/256 Heun audits, applies the hard stopping rule, verifies source
   manifests, and writes reproducibility metadata and artifact hashes.

The v9 config fixes 12 modes, width 32, four blocks, 2,364,899 parameters,
float32, batch 32, AdamW with cosine `8e-4 -> 1e-5`, weight decay `1e-6`, clip
5, seed-separated training/validation/paired evaluation roles, and the exact
v8 threshold file. The smaller batch relative to v8 is predeclared and limits
reverse-mode FFT memory. The sole checkpoint is selected by healthy-validation
normalized FM only.

The v8 evaluation helpers now dispatch `spectral_global` checkpoints and add
low/middle/high radial powers to each learned/direct rollout law record. This
does not rewrite v8 outputs. `paired_fm_by_time.csv`,
`paired_rollout_by_time.csv`, `paired_short_horizon.csv`, and
`paired_ode_resolution.csv` are the compact numerical audit tables.
`paired_frequency_and_rollout_diagnostics.png` and
`paired_fm_radial_bands_by_time.png` visualize the primary frequency and flow
failure.

Six new architecture tests require periodic translation equivariance, a real
FFT round trip, finite spectral gradients, batching invariance, float32
parameters/output, and nonzero distant response to a localized perturbation.
The algebraic equivariance/batching checks run on the CPU FFT backend to avoid
cuFFT plan-dependent float32 reduction-order noise; GPU observations differed
only at roundoff scale and are not hidden. All earlier tests remain unchanged,
for a total of 37 passing Gray–Scott tests.

The optional adaptation branch was not executed because its frozen trigger was
false. Final state and negative controls are serialized in
`phase3b_v9_final_decision.json`: Phase 4 is unauthorized, no tangent or
`B_tan` computation occurred, and no MFSI/Deep-Ritz/final comparison or
benchmark selection was created. `run_metadata.json` records git/dirty state,
config hash, architecture, modes, parameters, optimizer, seeds, bank roles,
software/device/dtype, checkpoint hash, and Heun resolutions;
`v9_artifact_sha256.json` hashes the complete v9 output set.
