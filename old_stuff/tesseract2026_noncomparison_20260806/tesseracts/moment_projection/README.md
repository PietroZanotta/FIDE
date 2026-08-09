# Ensemble moment-projection Tesseract

This Tesseract executes one complete ensemble-level projection solve. It applies
ridge-regularized SQP steps for the nearest-point objective under linearized whitened smooth
pair-moment constraints. The baseline derivative is reverse-mode differentiation through a fixed number of solver
iterations.

Implemented safeguards:

- diagonal coefficient whitening through `moment_scales`;
- positive ridge regularization in moment space;
- effective-rank and singular-value diagnostics;
- explicit basis pruning through `basis_mask`;
- per-step and total correction clipping;
- fixed-loop exact-norm merit backtracking;
- explicit convergence and failure counters.

The current coordinate metric is the identity. This is the first concrete `W_x = I` implementation
of the methodology's constrained projection and leaves the API open to structured coordinate weights.

Build and check:

```bash
./scripts/build_moment_projection_tesseract.sh

tesseract run manybody-moment-projection check

tesseract run manybody-moment-projection apply \
  @tesseracts/moment_projection/examples/s2_payload.json
```

Check gradients with respect to both solver inputs:

```bash
tesseract run \
  --runtime-args "--input-paths coordinates --input-paths target_moments --output-paths projected_coordinates --eps 1e-5" \
  manybody-moment-projection check-gradients \
  @tesseracts/moment_projection/examples/s2_payload.json
```

The returned `rank_deficient` flag must be inspected before trusting gradients. A zero in
`basis_mask` prunes a known redundant coefficient while preserving fixed array shapes.
