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
