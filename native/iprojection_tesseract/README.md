# Native Batched Information-Projection Tesseract

This directory contains the C++17/OpenMP backend used to calibrate complete empirical information-projection trajectories for the [analytical Gaussian-mixture experiment](../../experiments/toy_example_percentage/README.md).

## Role in the Analytical Experiment

At each time node, the experiment exponentially tilts a frozen weighted reference bank so that the resulting law matches the reconstructed sensor moments. For reference features `phi`, base weights `b`, target moments `c`, and multipliers `lambda`, the native solver finds the root of `E_q[phi] - c = 0`, where `q` is proportional to `b * exp(lambda @ phi)`.

One call solves a complete batch of trial trajectories. Multipliers are warm-started from the preceding time node within each trial, while independent trials are distributed across OpenMP threads. This preserves the sequential continuation in physical time without paying for one Python or Tesseract call per time node.

The hard-projection endpoint has the following contract:

- `phi`: shared reference features with shape `[T,N,M]`;
- `log_base_weights`: shared reference log weights with shape `[T,N]`;
- `targets`: trial-specific target moments with shape `[B,T,M]`; and
- `lambda_values`: calibrated multipliers with shape `[B,T,M]`.

Here `B` is the number of independent trials, `T` the number of time nodes, `N` the number of weighted reference particles, and `M` the number of measured moments. The native extension requires C-contiguous IEEE 754 `float64` arrays; the Python endpoint converts incoming arrays to that representation at the in-process boundary.

The Tesseract endpoint returns the multipliers. [`EmpiricalIProjector`](../../src/mfsi/projection.py) reconstructs the projected weights, moments, covariances, residuals, and effective-sample-size diagnostics in JAX so downstream scientific checks retain the same interface as the pure-JAX implementation.

## Differentiation

The JVP and VJP are obtained by implicitly differentiating the converged moment equation. They solve systems involving the projected moment covariance and do not differentiate through the individual Newton or line-search iterations. Derivatives are available with respect to `phi`, `log_base_weights`, and `targets`; the numerical solver settings are static.

The in-process endpoint keeps a thread-safe one-entry cache of the immediately preceding primal call so its derivative callback can reuse the calibrated multipliers. A cache miss recomputes the forward trajectory and does not change the mathematical result.

## Search and Authoritative Evaluation

The differentiable Full-search proxy uses this backend through [`src/mfsi/projection_tesseract.py`](../../src/mfsi/projection_tesseract.py). The exact law audits use the same native Newton kernel directly with the full iteration budget, independently recompute every calibration residual, and send any missed root to the robust SciPy repair path before accepting it. Tesseract therefore accelerates calibration but does not weaken the authoritative feasibility, residual, or effective-sample-size gates.

The extension also exposes a non-differentiable `solve_soft_batch` endpoint for penalized moment stationarity, `E_q[phi] - c + P lambda = 0`. That additive endpoint is not used by the current analytical experiment and does not alter the hard-projection API.

## Install and Build

Run the following commands from the repository root. The same virtual environment must provide Python, CMake, pybind11, Tesseract Core, and Tesseract JAX:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[analytical,tesseract-cpp]' 'pytest>=8'

.venv/bin/cmake \
  -S native/iprojection_tesseract \
  -B native/iprojection_tesseract/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python"
.venv/bin/cmake --build native/iprojection_tesseract/build --parallel "$(nproc)"
```

The build requires a C++17 compiler and OpenMP. GCC and Clang builds use `-O3`, `-march=native`, `-Wall`, `-Wextra`, and `-Wpedantic`; `-ffast-math` is deliberately not enabled. The extension is CPU-only.

## Select the Backend

The analytical configuration selects the backend under `projection`:

```json
{
  "projection": {
    "trajectory_backend": "tesseract_cpp"
  }
}
```

Use `"jax"` for the pure-JAX implementation. An explicit `"tesseract_cpp"` request raises a clear availability error if Tesseract Core, Tesseract JAX, or the compiled extension is missing; it does not silently switch implementations.

For repeatable CPU execution, choose an OpenMP thread count no larger than the number of available physical cores and disable dynamic thread resizing:

```bash
export OMP_NUM_THREADS=8
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

## Verification and Benchmarking

The focused tests compare native and JAX values, projected weights, JVPs, VJPs, direct-forward diagnostics, and the auxiliary soft endpoint. Candidate-batched tests belong to the separate `candidate_iprojection_tesseract` extension and are excluded from this backend's standalone verification command:

```bash
OMP_NUM_THREADS=4 .venv/bin/python -m pytest -q \
  tests/test_projection_tesseract.py -k 'not candidate'
```

The realistic analytical benchmark requires the frozen reference and selection-bank artifacts under `experiments/toy_example_percentage/outputs/run` as well as the Poisson Tesseract used by the complete proxy comparison:

```bash
OMP_NUM_THREADS=24 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores \
  .venv/bin/python experiments/toy_example_percentage/benchmark_iprojection_backends.py \
  --repetitions 5
```

The benchmark warms and synchronizes both implementations, checks numerical agreement, and writes its receipt to `experiments/toy_example_percentage/outputs/iprojection_backend_benchmark.json`. The reported project-level interpretation is in [Where Does Tesseract Enter the Picture?](../../README.md#where-does-tesseract-enter-the-picture).

## Files

- `src/bindings.cpp` implements the Newton solves and implicit derivatives;
- `tesseract_api.py` defines the in-process differentiable Tesseract contract;
- `CMakeLists.txt` builds `_iprojection_native`; and
- `tesseract_config.yaml` and `tesseract_requirements.txt` describe the standalone Tesseract build context.
