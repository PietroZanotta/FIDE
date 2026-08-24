# Native batched I-projection backend

This optional Tesseract executes a complete `[trial,time]` empirical
I-projection trajectory in one in-process C++/OpenMP call. Time nodes retain the
same multiplier warm starts as the JAX solver. Its VJP and JVP use the implicit
moment-covariance solve and never differentiate through Newton iterations.

```bash
.venv/bin/cmake \
  -S native/iprojection_tesseract \
  -B native/iprojection_tesseract/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python"
.venv/bin/cmake --build native/iprojection_tesseract/build -j "$(nproc)"
```

The implementation is float64 and deliberately does not use `-ffast-math`.

An additive `solve_soft_batch` endpoint supports the ocean-drifter finite-sample
law. It solves `E_q[phi] - target + penalty @ lambda = 0` with Hessian
`Cov_q(phi) + penalty`. The original `solve_batch`, JVP, and VJP endpoints retain
their hard-moment behavior and API, so toy and vortices callers are unaffected.
