# Native stage-4 weighted-Poisson backend

This optional module accelerates both the differentiable toy stage-4 search
proxy and its authoritative exact action evaluations. The proxy batches every
CRN-trial/selected-time system into one in-process Tesseract-JAX call. Each
exact trial similarly packs all full-resolution time systems into one call,
instead of launching 21 eager JAX solves. The independent systems use a
C++17/OpenMP matrix-free PCG implementation. The differentiable reverse pass
uses an analytical implicit VJP and another batched PCG solve; it never
differentiates through iterations.

Authoritative scientific rescoring keeps the full 51x51 grid, all 21 time
nodes, the complete action bank, and unchanged equations/tolerances. Its exact
I-projection, forcing, rasterization, physical-action formula, validity rules,
and candidate auditing remain in JAX/Python; only the independent linear solves
cross the batched Tesseract boundary.

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
.venv/bin/python experiments/toy_example/run.py --smoke
.venv/bin/python experiments/toy_example/run_gradient_smoke.py

OMP_NUM_THREADS=24 OMP_PROC_BIND=close OMP_PLACES=cores \
  .venv/bin/python experiments/toy_example/benchmark_poisson_backends.py \
  --repetitions 5 --include-exact-audit
```

The realistic benchmark requires the saved full-run reference and trial-bank
artifacts under `experiments/toy_example/outputs/run`. It reports synchronized,
warmed-up medians for JAX and native forward/gradient paths, end-to-end proxy
timing, callback overhead, system construction, PCG iterations/residuals, and
numerical errors.

## Run stage 4

```bash
export OMP_NUM_THREADS=24       # choose a reasonable physical-core count
export OMP_PROC_BIND=close
export OMP_PLACES=cores
.venv/bin/python experiments/toy_example/run.py
```

Do not enable nested OpenMP. Parallelism is naturally limited by the batch size
(28 in the default stage-4 proxy).

## Known limitations

- Input arrays are float64, rank-3, and copied to C-contiguous NumPy arrays at
  the in-process endpoint boundary when needed.
- Forward state is recomputed in the VJP. This keeps the implementation simple
  and avoids unsafe cross-callback caching.
- A native solve that misses its requested residual tolerance raises with batch
  indices, iterations, and residuals. It never silently returns an invalid
  potential or changes equations.
- This backend is intentionally scoped to toy stage 4. It does not replace
  `mfsi.poisson.solve_weighted_poisson` globally.
