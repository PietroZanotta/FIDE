# Native Batched Weighted-Poisson Tesseract

This directory contains the C++17/OpenMP backend used by the differentiable Full-action search proxy in the [analytical Gaussian-mixture experiment](../../experiments/toy_example_percentage/README.md). It batches the independent weighted-Poisson systems produced for the selected trials and time nodes into one in-process Tesseract call.

## Role in the Analytical Experiment

After information projection and rasterization, each proxy system supplies a density `q` and centered forcing `h`. The adapter forms a stabilized five-point diffusion operator from `q`, the right-hand side `-(q * h)`, and a normalized weighted gauge. The native extension solves the complete `[B,H,W]` batch with matrix-free preconditioned conjugate gradients and an IC(0) preconditioner; independent systems are distributed across OpenMP threads.

For the configured analytical proxy, four common-random-number trials and seven selected time nodes produce 28 independent `41 x 41` systems per Full objective evaluation. The native solve replaces the repeated linear-system work without changing the sensor geometry, projected law, rasterization, forcing, integration weights, or Full-action definition.

The differentiable endpoint has the following contract:

- `q_operator`: stabilized diffusion coefficients with shape `[B,H,W]`;
- `rhs`: linear-system right-hand sides with shape `[B,H,W]`;
- `gauge`: normalized gauge vectors with shape `[B,H,W]`; and
- `psi`: solved potentials with shape `[B,H,W]`.

All arrays are C-contiguous IEEE 754 `float64` at the native boundary. `dx`, `gauge_strength`, `cg_tol`, and `cg_maxiter` are static solver parameters. The extension uses a no-flux five-point stencil and no CUDA or external sparse/PDE library.

## Differentiation and Caching

The VJP solves one adjoint system and evaluates analytical derivatives with respect to `q_operator`, `rhs`, and `gauge`. The JVP constructs the linearized right-hand side and solves one tangent system. Neither rule differentiates through PCG iterations.

The in-process endpoint retains a thread-safe one-entry cache for the immediately following derivative of an identical primal call. It also retains same-shape primal, adjoint, and tangent solutions as algorithmic initial guesses. Cache misses and warm-start misses recompute the required solve and do not change the equations or requested tolerance.

Every Tesseract forward, adjoint, and tangent solve is fail-closed: if any batch member misses the requested residual tolerance, the endpoint raises an error containing the failed indices, iteration counts, and residuals. The lower-level `_poisson_native.solve_batch` function instead returns `psi`, `iterations`, `relative_residual`, and `converged` so tests and diagnostics can inspect every system directly.

## Authoritative-Evaluation Boundary

This backend accelerates `optimization.full_gradient_poisson_backend`, the differentiable lower-resolution search proxy. It does **not** produce the corrected authoritative Full-action values reported in the paper and project README. Those values use the positive-support physical-`q` raster and the host-side batched solver `solve_weighted_poisson_source_physical_direct_batch`, with component compatibility, physical residual, positivity, mass, source, and moment-rate gates applied explicitly.

The configuration still parses and records `optimization.full_exact_poisson_backend` for compatibility with the earlier workflow, but that key does not reroute the current corrected authoritative evaluator through this Tesseract. This distinction prevents proxy acceleration from being mistaken for publication-level numerical evidence.

## Install and Build

Run the following commands from the repository root. The same virtual environment must provide Python, CMake, pybind11, Tesseract Core, and Tesseract JAX:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[analytical,tesseract-cpp]' 'pytest>=8'

.venv/bin/cmake \
  -S native/poisson_tesseract \
  -B native/poisson_tesseract/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python"
.venv/bin/cmake --build native/poisson_tesseract/build --parallel "$(nproc)"
```

The build requires a C++17 compiler and OpenMP. GCC and Clang builds use `-O3`, `-march=native`, `-Wall`, `-Wextra`, and `-Wpedantic`; `-ffast-math` is deliberately not enabled. The extension is CPU-only.

## Select the Backend

The active analytical setting is nested under `optimization`:

```json
{
  "optimization": {
    "full_gradient_poisson_backend": "tesseract_cpp"
  }
}
```

Use `"jax"` for the pure-JAX search solver. The code default is `"jax"` when the key is absent. An explicit `"tesseract_cpp"` request raises a clear availability error if Tesseract Core, Tesseract JAX, or the compiled extension is missing; it does not silently fall back per system.

For repeatable CPU execution, choose an OpenMP thread count no larger than the number of available physical cores and do not enable nested OpenMP:

```bash
export OMP_NUM_THREADS=8
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

Parallelism is bounded by the number of systems in the batch, which is 28 for the default analytical proxy.

## Verification and Benchmarking

The focused tests compare the native stencil and diagonal with JAX, exercise cold and warm PCG solves, check convergence on high-contrast densities, verify analytical VJPs against finite differences, and compare the Tesseract-JAX gradient with the JAX implicit-CG reference:

```bash
OMP_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_poisson_tesseract.py
```

The end-to-end smoke test also requires the information-projection Tesseract:

```bash
OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores \
  .venv/bin/python experiments/toy_example_percentage/run.py \
  --smoke \
  --output-dir experiments/toy_example_percentage/outputs/reproduction/native-smoke
```

The realistic benchmark requires the frozen reference and selection-bank artifacts under `experiments/toy_example_percentage/outputs/run`:

```bash
OMP_NUM_THREADS=24 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores \
  .venv/bin/python experiments/toy_example_percentage/benchmark_poisson_backends.py \
  --repetitions 5
```

The benchmark performs warm-up and synchronization, separates system construction from the linear solve, checks forward and gradient agreement, records native PCG diagnostics, and writes `experiments/toy_example_percentage/outputs/poisson_backend_benchmark.json`. `--include-exact-audit` times the shared corrected authoritative path for diagnostic comparison; it does not turn that path into a Tesseract solve. The reported project-level interpretation is in [Where Does Tesseract Enter the Picture?](../../README.md#where-does-tesseract-enter-the-picture).

## Files

- `src/poisson_solver.cpp` and `src/poisson_solver.hpp` implement the stencil, IC(0)-PCG solver, and derivative kernels;
- `src/bindings.cpp` validates arrays and exposes `_poisson_native` through pybind11;
- `tesseract_api.py` defines the in-process differentiable Tesseract contract;
- `CMakeLists.txt` builds `_poisson_native`; and
- `tesseract_config.yaml` and `tesseract_requirements.txt` describe the standalone Tesseract build context.
