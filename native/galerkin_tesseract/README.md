# Native fixed-feature Galerkin assembler

This additive Tesseract is independent of `native/iprojection_tesseract/`. It
assembles the additive raw statistics for one float64 `[N,K,P,D]` basis chunk:
`K = E_w[grad phi grad phi^T]`, raw load `E_w[h phi]`, basis mean, and forcing
mean. The caller applies the global centering term after accumulating chunks.
It never forms per-sample `K x K` tensors and is forward-only because the
qualified eta envelope derivative does not differentiate through K/f or the
rank-aware coefficient solve.

```bash
.venv/bin/cmake -S native/galerkin_tesseract -B native/galerkin_tesseract/build \
  -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PWD/.venv/bin/python"
.venv/bin/cmake --build native/galerkin_tesseract/build -j "$(nproc)"
```

The Python API is `mfsi.galerkin_tesseract`. The native implementation uses
OpenMP preprocessing and SciPy OpenBLAS when available, with system BLAS as a
portable fallback. Benchmark before enabling it: host/device transfer means the
JAX GPU contraction can remain faster even when the native CPU kernel is fast.

