from __future__ import annotations

from collections.abc import Callable

import jax
from jax.scipy.sparse.linalg import cg

Array = jax.Array
LinearMap = Callable[[Array], Array]


def implicit_cg(
    matvec: LinearMap,
    rhs: Array,
    *,
    tol: float = 1.0e-8,
    maxiter: int = 500,
    preconditioner: LinearMap | None = None,
) -> Array:
    """Solve a symmetric positive-definite linear system with implicit gradients.

    JAX's CG rule differentiates the equation K x = rhs implicitly using another
    linear solve. The reverse pass therefore propagates through both rhs and the
    closed-over parameters of ``matvec`` (the operator K), not through CG iterations.
    """
    solution, _ = cg(
        matvec,
        rhs,
        tol=tol,
        atol=0.0,
        maxiter=maxiter,
        M=preconditioner,
    )
    return solution
