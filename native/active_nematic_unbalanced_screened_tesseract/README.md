# Active-nematic unbalanced screened-Poisson Tesseract

This experiment-local backend solves, in batches,

`[-div(q_op grad) + q_op / kappa] psi = rhs`

on the periodic `(x, y, r_beta beta)` grid. It is separate from the balanced
active-nematic native project and does not modify any shared MFSI solver.

The extension is float64-only. It combines matrix-free periodic finite-volume
stencils, an IC(0)-style SPD preconditioner, OpenMP across independent time
slices, true-residual refreshes, optional warm starts, and implicit JVP/VJP
rules. `-ffast-math` is deliberately not enabled.

Build from the repository root:

```bash
.venv/bin/cmake \
  -S native/active_nematic_unbalanced_screened_tesseract \
  -B native/active_nematic_unbalanced_screened_tesseract/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python"
.venv/bin/cmake --build \
  native/active_nematic_unbalanced_screened_tesseract/build -j "$(nproc)"
```

