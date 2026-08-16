"""Optional Tesseract-JAX wrapper for batched stage-4 weighted-Poisson solves."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from .poisson import PoissonConfig

Array = jax.Array
NATIVE_SOLVER_REVISION = "cpp-openmp-v1"


class TesseractPoissonUnavailable(RuntimeError):
    """Raised when the explicitly requested optional backend is unavailable."""


def _native_root() -> Path:
    return Path(__file__).resolve().parents[2] / "native" / "poisson_tesseract"


def is_tesseract_poisson_available() -> bool:
    try:
        import tesseract_core  # noqa: F401
        import tesseract_jax  # noqa: F401
    except ImportError:
        return False
    root = _native_root()
    return (root / "tesseract_api.py").is_file() and any(
        (root / "build").glob("_poisson_native*.so")
    )


@lru_cache(maxsize=1)
def _client() -> Any:
    try:
        from tesseract_core import Tesseract
        import tesseract_jax  # noqa: F401
    except ImportError as exc:
        raise TesseractPoissonUnavailable(
            "The tesseract_cpp backend requires tesseract-core and tesseract-jax."
        ) from exc

    api_path = _native_root() / "tesseract_api.py"
    if not api_path.is_file():
        raise TesseractPoissonUnavailable(
            f"Native Tesseract API not found at {api_path}."
        )
    try:
        return Tesseract.from_tesseract_api(api_path)
    except (ImportError, RuntimeError) as exc:
        raise TesseractPoissonUnavailable(
            "The tesseract_cpp backend was requested but its C++ extension is not built. "
            "See native/poisson_tesseract/README.md."
        ) from exc


def solve_linear_system_batch_tesseract(
    q_operator: Array,
    rhs: Array,
    gauge: Array,
    *,
    dx: float,
    gauge_strength: float,
    cg_tol: float,
    cg_maxiter: int,
) -> Array:
    """Make one in-process Tesseract call for an explicit ``[B,H,W]`` batch."""
    try:
        from tesseract_jax import apply_tesseract
    except ImportError as exc:
        raise TesseractPoissonUnavailable(
            "The tesseract_cpp backend requires the optional tesseract-jax package."
        ) from exc

    inputs = {
        "q_operator": jnp.asarray(q_operator, dtype=jnp.float64),
        "rhs": jnp.asarray(rhs, dtype=jnp.float64),
        "gauge": jnp.asarray(gauge, dtype=jnp.float64),
        # Built-in scalars remain static in the Tesseract-JAX PyTree.
        "dx": float(dx),
        "gauge_strength": float(gauge_strength),
        "cg_tol": float(cg_tol),
        "cg_maxiter": int(cg_maxiter),
    }
    return apply_tesseract(_client(), inputs)["psi"]


def solve_weighted_poisson_batch_tesseract(
    q: Array,
    h: Array,
    cfg: PoissonConfig,
) -> Array:
    """Construct differentiable solver inputs in JAX and return batched potentials."""
    q = jnp.asarray(q, dtype=jnp.float64)
    h = jnp.asarray(h, dtype=jnp.float64)
    if q.ndim != 3 or h.shape != q.shape:
        raise ValueError("q and h must have the same [B,H,W] shape")

    q_floor = cfg.operator_floor_rel * jnp.max(q, axis=(-2, -1), keepdims=True)
    q_operator = q + q_floor
    rhs = -(q * h)
    flat_q = q.reshape((q.shape[0], -1))
    gauge = flat_q / jnp.maximum(
        jnp.linalg.norm(flat_q, axis=-1, keepdims=True), 1.0e-300
    )
    gauge = gauge.reshape(q.shape)
    return solve_linear_system_batch_tesseract(
        q_operator,
        rhs,
        gauge,
        dx=cfg.dx,
        gauge_strength=cfg.gauge_strength,
        cg_tol=cfg.cg_tol,
        cg_maxiter=cfg.cg_maxiter,
    )
