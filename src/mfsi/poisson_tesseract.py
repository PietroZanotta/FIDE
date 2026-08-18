"""Optional Tesseract-JAX wrapper for batched stage-4 weighted-Poisson solves."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from .poisson import PoissonConfig

Array = jax.Array
NATIVE_SOLVER_REVISION = "cpp-openmp-ic0-v2"


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


def solve_weighted_poisson_batch_tesseract_diagnostics(
    q: Any,
    h: Any,
    cfg: PoissonConfig,
) -> dict[str, Any]:
    """Solve a batch and expose native convergence/PDE diagnostics.

    This non-differentiable audit path uses the same C++ finite-volume operator,
    no-flux boundary, weighted gauge, and IC(0)-PCG implementation as the
    Tesseract stage-4 endpoint.  Each physical density is scaled by its maximum
    only while solving; the reported Dirichlet action uses the unscaled density.
    The scaling leaves the compatible Poisson equation unchanged and avoids a
    meaningless dependence of the gauge penalty on density units.
    """
    import numpy as np

    q_array = np.asarray(q, dtype=np.float64, order="C")
    h_array = np.asarray(h, dtype=np.float64, order="C")
    if q_array.ndim != 3 or h_array.shape != q_array.shape:
        raise ValueError("q and h must have the same [B,H,W] shape")
    if np.any(q_array < 0.0) or not np.isfinite(q_array).all():
        raise ValueError("q must be finite and nonnegative")
    if not np.isfinite(h_array).all():
        raise ValueError("h must be finite")

    q_max = np.max(q_array, axis=(-2, -1), keepdims=True)
    if np.any(q_max <= 0.0):
        raise ValueError("every q batch member must contain positive mass")
    q_solve = np.ascontiguousarray(q_array / q_max)
    floor_relative = float(cfg.operator_floor_rel)
    q_operator = np.ascontiguousarray(q_solve + floor_relative)
    rhs = np.ascontiguousarray(-(q_solve * h_array))
    flat_q = q_solve.reshape((q_solve.shape[0], -1))
    gauge = flat_q / np.maximum(
        np.linalg.norm(flat_q, axis=-1, keepdims=True), 1.0e-300
    )
    gauge = np.ascontiguousarray(gauge.reshape(q_solve.shape))

    build = _native_root() / "build"
    if str(build) not in sys.path:
        sys.path.insert(0, str(build))
    try:
        import _poisson_native as native
    except ImportError as exc:
        raise TesseractPoissonUnavailable(
            "The tesseract_cpp backend was requested but its C++ extension is not built. "
            "See native/poisson_tesseract/README.md."
        ) from exc

    result = native.solve_batch(
        q_operator,
        rhs,
        gauge,
        float(cfg.dx),
        float(cfg.gauge_strength),
        float(cfg.cg_tol),
        int(cfg.cg_maxiter),
        None,
    )
    psi = np.asarray(result["psi"], dtype=np.float64, order="C")
    physical_operator = np.asarray(
        native.weighted_laplacian_batch(psi, q_solve, float(cfg.dx)),
        dtype=np.float64,
    )
    stabilized_operator = np.asarray(
        native.weighted_laplacian_batch(psi, q_operator, float(cfg.dx)),
        dtype=np.float64,
    )
    gauge_dot = np.sum(gauge * psi, axis=(-2, -1))
    stabilized_residual = stabilized_operator + (
        float(cfg.gauge_strength) * gauge * gauge_dot[:, None, None]
    ) - rhs
    physical_residual = physical_operator - rhs
    rhs_norm = np.linalg.norm(rhs.reshape((len(rhs), -1)), axis=1)
    scale = np.maximum(rhs_norm, 1.0e-14)
    cell_area = float(cfg.dx) ** 2
    physical_density_operator = physical_operator * q_max
    action = cell_area * np.sum(psi * physical_density_operator, axis=(-2, -1))
    q_mass = cell_area * np.sum(q_array, axis=(-2, -1))
    weighted_mean = (
        cell_area * np.sum(q_array * psi, axis=(-2, -1))
        / np.maximum(q_mass, 1.0e-300)
    )
    return {
        "action": action,
        "potential": psi,
        "iterations": np.asarray(result["iterations"], dtype=np.int32),
        "converged": np.asarray(result["converged"], dtype=bool),
        "native_relative_residual": np.asarray(result["relative_residual"], dtype=np.float64),
        "stabilized_relative_residual": np.linalg.norm(
            stabilized_residual.reshape((len(rhs), -1)), axis=1
        ) / scale,
        "physical_relative_residual": np.linalg.norm(
            physical_residual.reshape((len(rhs), -1)), axis=1
        ) / scale,
        "physical_absolute_residual": np.linalg.norm(
            physical_residual.reshape((len(rhs), -1)), axis=1
        ),
        "weighted_mean_potential": weighted_mean,
        "operator_floor": floor_relative * q_max.reshape(-1),
        "coefficient_condition_proxy": (
            np.max(q_operator, axis=(-2, -1))
            / np.maximum(np.min(q_operator, axis=(-2, -1)), np.finfo(np.float64).tiny)
        ),
    }
