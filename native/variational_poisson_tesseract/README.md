# Variational weighted-Poisson Tesseract

This is an isolated weak-form backend for the weighted-Poisson problem. It does
not replace or modify `native/poisson_tesseract`, which remains the backend used
by the toy and vortices experiments.

The trial space is a tensor-product Neumann cosine basis with the constant mode
removed. Every basis function is centered under the supplied quadrature law, so
the reconstructed potential satisfies `E_q[psi] = 0`. The native solver assembles
only

```text
K_ij = E_q[grad(phi_i) . grad(phi_j)]
f_i  = E_q[h phi_i]
```

and minimizes `0.5 c^T K c + f^T c`. It never assembles a strong-form grid PDE
matrix. The density is supplied as finite `log_q_mass`; normalization uses
long-double shifted exponentials without a density or operator floor.

The dense Galerkin system is assembled, diagonally equilibrated, and eigensolved
in extended-precision `long double` arithmetic with a cyclic symmetric Jacobi
algorithm. This is necessary because concentrated ocean laws have positive weak
Gram eigenvalues below float64 precision. Numerical rank truncation applies only
to redundant trial-space directions and is reported explicitly; it does not alter
`q` or add an operator floor. Acceptance is based on algebraic optimality in the
explicitly retained eigenspace. The complete-space residual and discarded-load
fraction remain separate audit diagnostics. The endpoint also reports the weighted
gauge and its relative form, compatibility, action, objective, energy/load
identity, retained rank, and condition proxy.

Build from the repository root:

```bash
.venv/bin/cmake \
  -S native/variational_poisson_tesseract \
  -B native/variational_poisson_tesseract/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python"
.venv/bin/cmake --build native/variational_poisson_tesseract/build -j "$(nproc)"
```

The shared wrapper is `mfsi.poisson_variational_tesseract`. It exposes a direct
native diagnostic call and an in-process Tesseract call with the same output
schema. Run the focused contract with:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 \
  .venv/bin/python -m pytest tests/test_variational_poisson_tesseract.py -q
```

The ocean experiment now reaches this endpoint through its local
`experiments.ocean_drifters.poisson_backend` adapter and preserves the common
Poisson result fields. This is still a validation pilot, not an authorized ocean
production sweep. The endpoint is intentionally non-differentiable at this stage,
and no toy/vortices backend selector refers to it; their existing differentiable
JAX/native paths remain unchanged. Concentrated ocean cases that fail the original
structured coarse/fine comparison are reprojected and audited on nested local
quadrature through the ocean adapter; that adaptive path does not modify this
shared native endpoint.
