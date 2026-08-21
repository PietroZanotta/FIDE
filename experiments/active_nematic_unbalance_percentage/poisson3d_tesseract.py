"""Experiment-local Tesseract-JAX wrapper for the polarity Poisson backend."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys
from typing import Any

import jax
import jax.numpy as jnp

try:
    from .periodic_numerics import (
        PeriodicGrid3D,
        PeriodicPoissonBatchResult,
        PeriodicPoissonConfig,
        periodic_weighted_laplacian3d,
        prepare_periodic_poisson3d_batch,
        stable_relative_residual,
    )
except ImportError:  # pragma: no cover - direct experiment-script convention
    from periodic_numerics import (
        PeriodicGrid3D,
        PeriodicPoissonBatchResult,
        PeriodicPoissonConfig,
        periodic_weighted_laplacian3d,
        prepare_periodic_poisson3d_batch,
        stable_relative_residual,
    )

Array = jax.Array
NATIVE_SOLVER_REVISION = "periodic-3d-cpp-openmp-ic0-reliable-residual-v2"


class ActiveNematicPoisson3DUnavailable(RuntimeError):
    """Raised when an explicitly requested native backend cannot be loaded."""


def _native_root() -> Path:
    return Path(__file__).resolve().parents[2] / "native" / "active_nematic_poisson3d_tesseract"


def is_active_nematic_poisson3d_available() -> bool:
    try:
        import tesseract_core  # noqa: F401
        import tesseract_jax  # noqa: F401
    except ImportError:
        return False
    root = _native_root()
    return (root / "tesseract_api.py").is_file() and any(
        (root / "build").glob("_active_nematic_poisson3d_native*.so")
    )


@lru_cache(maxsize=1)
def _client() -> Any:
    try:
        from tesseract_core import Tesseract
        import tesseract_jax  # noqa: F401
    except ImportError as exc:
        raise ActiveNematicPoisson3DUnavailable(
            "The tesseract_cpp backend requires tesseract-core and tesseract-jax."
        ) from exc
    api_path = _native_root() / "tesseract_api.py"
    try:
        return Tesseract.from_tesseract_api(api_path)
    except (ImportError, RuntimeError) as exc:
        raise ActiveNematicPoisson3DUnavailable(
            "The active-nematic 3D Tesseract backend is unavailable; build it using "
            "native/active_nematic_poisson3d_tesseract/README.md."
        ) from exc


def solve_linear_system_batch_tesseract(
    q_operator: Array,
    rhs: Array,
    gauge: Array,
    *,
    grid: PeriodicGrid3D,
    cfg: PeriodicPoissonConfig,
) -> Array:
    """Make one differentiable in-process call for a rank-4 system batch."""
    try:
        from tesseract_jax import apply_tesseract
    except ImportError as exc:
        raise ActiveNematicPoisson3DUnavailable(
            "The tesseract_cpp backend requires tesseract-jax."
        ) from exc
    inputs = {
        "q_operator": jnp.asarray(q_operator, dtype=jnp.float64),
        "rhs": jnp.asarray(rhs, dtype=jnp.float64),
        "gauge": jnp.asarray(gauge, dtype=jnp.float64),
        "dx": float(grid.dx),
        "dy": float(grid.dy),
        "dtheta_metric": float(grid.dtheta_metric),
        "gauge_strength": float(cfg.gauge_strength),
        "cg_tol": float(cfg.cg_tol),
        "cg_maxiter": int(cfg.cg_maxiter),
    }
    return apply_tesseract(_client(), inputs)["potential"]


def solve_periodic_weighted_poisson3d_batch_tesseract(
    q: Array,
    h: Array,
    grid: PeriodicGrid3D,
    cfg: PeriodicPoissonConfig = PeriodicPoissonConfig(),
) -> PeriodicPoissonBatchResult:
    """Solve all time slices together and compute scientific diagnostics in JAX."""
    q = jnp.asarray(q, dtype=jnp.float64)
    h = jnp.asarray(h, dtype=jnp.float64)
    if q.ndim != 4 or h.shape != q.shape or q.shape[1:] != grid.shape:
        raise ValueError(
            f"q and h must have shape [B,{grid.shape[0]},{grid.shape[1]},{grid.shape[2]}]"
        )
    q_operator, rhs, gauge, floor = prepare_periodic_poisson3d_batch(q, h, cfg)
    potential = solve_linear_system_batch_tesseract(
        q_operator, rhs, gauge, grid=grid, cfg=cfg
    )
    spacings = grid.spacings
    stabilized = periodic_weighted_laplacian3d(potential, q_operator, spacings)
    gauge_dot = jnp.sum(gauge * potential, axis=(-3, -2, -1), keepdims=True)
    residual = stabilized + cfg.gauge_strength * gauge * gauge_dot - rhs
    relative = stable_relative_residual(
        residual, rhs, axes=(-3, -2, -1)
    )
    physical = periodic_weighted_laplacian3d(potential, q, spacings)
    action = grid.cell_volume * jnp.sum(
        potential * physical, axis=(-3, -2, -1)
    )
    weighted_mean = grid.cell_volume * jnp.sum(
        q * potential, axis=(-3, -2, -1)
    )
    return PeriodicPoissonBatchResult(
        action, potential, relative, weighted_mean, floor
    )


def native_diagnostics(
    q: Any,
    h: Any,
    grid: PeriodicGrid3D,
    cfg: PeriodicPoissonConfig = PeriodicPoissonConfig(),
) -> dict[str, Any]:
    """Non-differentiable native audit including PCG iterations and residuals."""
    import numpy as np

    q_array = np.ascontiguousarray(q, dtype=np.float64)
    h_array = np.ascontiguousarray(h, dtype=np.float64)
    if q_array.ndim != 4 or h_array.shape != q_array.shape or q_array.shape[1:] != grid.shape:
        raise ValueError("q and h have an invalid batch/grid shape")
    floor = cfg.operator_floor_rel * np.max(q_array, axis=(-3, -2, -1), keepdims=True)
    q_operator = np.ascontiguousarray(q_array + floor)
    rhs = np.ascontiguousarray(-(q_array * h_array))
    flat_q = q_array.reshape((len(q_array), -1))
    gauge = flat_q / np.maximum(np.linalg.norm(flat_q, axis=-1, keepdims=True), 1.0e-300)
    gauge = np.ascontiguousarray(gauge.reshape(q_array.shape))
    build = _native_root() / "build"
    if str(build) not in sys.path:
        sys.path.insert(0, str(build))
    try:
        import _active_nematic_poisson3d_native as native
    except ImportError as exc:
        raise ActiveNematicPoisson3DUnavailable("Native extension is not built") from exc
    result = native.solve_batch(
        q_operator,
        rhs,
        gauge,
        float(grid.dx),
        float(grid.dy),
        float(grid.dtheta_metric),
        float(cfg.gauge_strength),
        float(cfg.cg_tol),
        int(cfg.cg_maxiter),
        None,
    )
    potential = np.asarray(result["potential"], dtype=np.float64)
    physical = native.weighted_laplacian_batch(
        np.ascontiguousarray(potential),
        q_array,
        float(grid.dx),
        float(grid.dy),
        float(grid.dtheta_metric),
    )
    action = grid.cell_volume * np.sum(
        potential * physical, axis=(-3, -2, -1)
    )
    return {
        "action": action,
        "potential": potential,
        "iterations": np.asarray(result["iterations"], dtype=np.int32),
        "converged": np.asarray(result["converged"], dtype=bool),
        "relative_residual": np.asarray(result["relative_residual"], dtype=np.float64),
    }
