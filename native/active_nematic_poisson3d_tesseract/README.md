# Active-nematic 3D periodic Poisson Tesseract

This experiment-local backend solves the anisotropic periodic weighted-Poisson
systems used by the active-nematic polarity-state full action. It does not
replace or modify the shared MFSI Poisson implementation.

The discretization is the symmetric seven-point finite-volume operator on
`(x, y, r_theta * theta)`, with arithmetic edge weights and a rank-one gauge.
Independent time systems are batched across OpenMP threads. Each system uses a
matrix-free PCG solve with a local IC(0)-style preconditioner; the periodic wrap
edges and dense gauge remain in every operator application. Implicit VJP/JVP
rules solve the differentiated equation rather than differentiating iterations.

Build from the repository root:

```bash
.venv/bin/cmake \
  -S native/active_nematic_poisson3d_tesseract \
  -B native/active_nematic_poisson3d_tesseract/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python"
.venv/bin/cmake --build \
  native/active_nematic_poisson3d_tesseract/build -j "$(nproc)"
```

The extension is float64-only and requires rank-4 C-contiguous arrays shaped
`[batch, nx, ny, ntheta]`, with at least three cells on every periodic axis.
No `-ffast-math` is used.
