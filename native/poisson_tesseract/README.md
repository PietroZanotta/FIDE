# Native stage-4 weighted-Poisson backend

This optional module accelerates both the differentiable toy and vortices
stage-4 search proxies and their authoritative exact action evaluations. The proxy batches every
CRN-trial/selected-time system into one in-process Tesseract-JAX call. Each
exact trial similarly packs all full-resolution time systems into one call,
instead of launching 21 eager JAX solves. The independent systems use a
C++17/OpenMP matrix-free PCG implementation with an IC(0) preconditioner for
the five-point diffusion stencil. The differentiable reverse pass
uses an analytical implicit VJP and another batched PCG solve; it never
differentiates through iterations.

Authoritative scientific rescoring keeps the configured full grid, all time
nodes, the complete action bank, and unchanged equations/validity rules. Only
the independent linear solves cross the batched Tesseract boundary.

## Install and build

From the repository root:

```bash
uv pip install --python .venv/bin/python -e '.[tesseract-cpp]'

.venv/bin/cmake \
  -S native/poisson_tesseract \
  -B native/poisson_tesseract/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python"
.venv/bin/cmake --build native/poisson_tesseract/build -j "$(nproc)"
```

The build uses double precision, `-O3`, `-march=native`, C++17, and OpenMP. It
does not use `-ffast-math`, CUDA, or an external sparse/PDE library.

## Select the backend

Set the toy configuration key:

```json
{
  "full_gradient_poisson_backend": "tesseract_cpp",
  "full_exact_poisson_backend": "tesseract_cpp"
}
```

Use `"jax"` to disable the accelerator. The code default is `"jax"` when the
key is absent. An explicit `tesseract_cpp` request fails clearly if the optional
packages or extension are missing; there is no hidden per-system fallback.

The current implementation uses `Tesseract.from_tesseract_api`, so callbacks are
in-process. Endpoint code allocates only NumPy/native arrays, never JAX arrays.

## Tests and benchmark

```bash
OMP_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_poisson_tesseract.py
.venv/bin/python -m pytest -q
.venv/bin/python experiments/toy_example_percentage/run.py --smoke
.venv/bin/python experiments/toy_example_percentage/run_gradient_smoke.py

OMP_NUM_THREADS=24 OMP_PROC_BIND=close OMP_PLACES=cores \
  .venv/bin/python experiments/toy_example_percentage/benchmark_poisson_backends.py \
  --repetitions 5 --include-exact-audit
```

The realistic benchmark requires the saved full-run reference and trial-bank
artifacts under `experiments/toy_example_percentage/outputs/run`. It reports synchronized,
warmed-up medians for JAX and native forward/gradient paths, end-to-end proxy
timing, callback overhead, system construction, PCG iterations/residuals, and
numerical errors.

## Run stage 4

```bash
export OMP_NUM_THREADS=24       # choose a reasonable physical-core count
export OMP_PROC_BIND=close
export OMP_PLACES=cores
.venv/bin/python experiments/toy_example_percentage/run.py
```

Do not enable nested OpenMP. Parallelism is naturally limited by the batch size
(28 in the default stage-4 proxy).

## Known limitations

- Input arrays are float64, rank-3, and copied to C-contiguous NumPy arrays at
  the in-process endpoint boundary when needed.
- The in-process endpoint retains a thread-safe one-entry forward cache for the
  immediately following VJP/JVP and same-shape warm guesses for iterative solves;
  cache misses always recompute the required state.
- A native solve that misses its requested residual tolerance raises with batch
  indices, iterations, and residuals. It never silently returns an invalid
  potential or changes equations.
- This backend is intentionally scoped to experiment stage 4. It does not replace
  `mfsi.poisson.solve_weighted_poisson` globally.
