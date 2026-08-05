# Physical-relaxation Tesseract

This component wraps one complete proximal relaxation solve. The forward path uses fixed-iteration,
backtracking gradient descent on the smooth repulsive energy plus a periodic proximity penalty. The
baseline VJP is reverse-mode differentiation through the unrolled solver iterations.

## Build and inspect

From the repository root:

```bash
tesseract build tesseracts/physical_relaxation

tesseract run manybody-physical-relaxation check

tesseract run manybody-physical-relaxation apply \
  @tesseracts/physical_relaxation/examples/s1_payload.json
```

Run the runtime gradient checker on the differentiable coordinate output:

```bash
tesseract run \
  --runtime-args "--input-paths coordinates --output-paths relaxed_coordinates --eps 1e-5" \
  manybody-physical-relaxation check-gradients \
  @tesseracts/physical_relaxation/examples/s1_gradient_payload.json
```

The gradient payload uses a fixed 20-step trajectory with early stopping disabled. The converged
forward payload intentionally reaches stopping and backtracking boundaries, where infinitesimal
perturbations can select different discrete solver paths and a finite-difference check is not valid.

The container uses `jax==0.8.1` on CPU for the first correctness milestone. The host training
process can remain on `jax[cuda]==0.8.1`. After correctness and call granularity are established,
the Tesseract image can be moved to a CUDA JAX base and served with GPU access.
