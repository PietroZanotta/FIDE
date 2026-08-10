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
