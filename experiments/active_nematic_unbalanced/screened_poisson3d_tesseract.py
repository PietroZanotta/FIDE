"""Experiment-local JAX/Tesseract wrapper for unbalanced screened Poisson."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys
import threading
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

try:
    from .periodic_numerics import (
        PeriodicGrid3D,
        periodic_weighted_laplacian3d,
        stable_relative_residual,
    )
    from .unbalanced_correction import (
        UnbalancedCorrectionConfig,
        UnbalancedCorrectionResult,
    )
except ImportError:  # pragma: no cover
    from periodic_numerics import (
        PeriodicGrid3D,
        periodic_weighted_laplacian3d,
        stable_relative_residual,
    )
    from unbalanced_correction import (
        UnbalancedCorrectionConfig,
        UnbalancedCorrectionResult,
    )


Array = jax.Array
NATIVE_SOLVER_REVISION = "unbalanced-screened3d-cpp-openmp-ic0-v1"
_CLIENT_CALL_LOCK = threading.Lock()


class UnbalancedScreenedPoissonUnavailable(RuntimeError):
    pass


def _native_root() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "native"
        / "active_nematic_unbalanced_screened_tesseract"
    )


def is_unbalanced_screened_poisson_available() -> bool:
    try:
        import tesseract_core  # noqa: F401
        import tesseract_jax  # noqa: F401
    except ImportError:
        return False
    root = _native_root()
    return (root / "tesseract_api.py").is_file() and any(
        (root / "build").glob("_active_nematic_unbalanced_screened_native*.so")
    )


@lru_cache(maxsize=1)
def _client() -> Any:
    try:
        from tesseract_core import Tesseract
        import tesseract_jax  # noqa: F401
    except ImportError as exc:
        raise UnbalancedScreenedPoissonUnavailable(
            "The native backend requires tesseract-core and tesseract-jax."
        ) from exc
    try:
        return Tesseract.from_tesseract_api(_native_root() / "tesseract_api.py")
    except (ImportError, RuntimeError) as exc:
        raise UnbalancedScreenedPoissonUnavailable(
            "Build native/active_nematic_unbalanced_screened_tesseract first."
        ) from exc


def solve_linear_system_batch_tesseract(
    q_operator: Array,
    rhs: Array,
    *,
    grid: PeriodicGrid3D,
    config: UnbalancedCorrectionConfig,
) -> Array:
    """Call the Tesseract endpoints through a JIT-safe implicit VJP bridge.

    The installed ``tesseract-jax`` primitive works in eager reverse mode, but
    its Python-callback lowering terminates this environment's process during a
    jitted reverse pass.  This local bridge still calls the same Tesseract
    ``apply`` and ``vector_jacobian_product`` endpoints; ``pure_callback`` only
    replaces the faulty package lowering.  No PCG iterations are differentiated
    through and the native C++ operator remains authoritative.
    """
    q_operator = jnp.asarray(q_operator, dtype=jnp.float64)
    rhs = jnp.asarray(rhs, dtype=jnp.float64)
    output_spec = jax.ShapeDtypeStruct(q_operator.shape, q_operator.dtype)
    # Construct once on the tracing thread.  Plus/minus host callbacks can run
    # concurrently, and racing two dynamic imports of one Tesseract API can
    # otherwise observe a partially initialized Pydantic schema.
    client = _client()
    static_inputs = {
        "dx": float(grid.dx),
        "dy": float(grid.dy),
        "dtheta_metric": float(grid.dtheta_metric),
        "kappa": float(config.reaction_kappa),
        "cg_tol": float(config.cg_tol),
        "cg_maxiter": int(config.cg_maxiter),
    }

    def inputs(q_value, rhs_value):
        return {
            "q_operator": np.ascontiguousarray(q_value, dtype=np.float64),
            "rhs": np.ascontiguousarray(rhs_value, dtype=np.float64),
            **static_inputs,
        }

    def apply_callback(q_value, rhs_value):
        with _CLIENT_CALL_LOCK:
            result = client.apply(inputs(q_value, rhs_value))
        return np.asarray(result["potential"], dtype=np.float64, order="C")

    def primal(q_value, rhs_value):
        return jax.pure_callback(
            apply_callback, output_spec, q_value, rhs_value
        )

    @jax.custom_vjp
    def implicit_solve(q_value, rhs_value):
        return primal(q_value, rhs_value)

    def forward(q_value, rhs_value):
        potential = primal(q_value, rhs_value)
        return potential, (q_value, rhs_value)

    def backward(saved, potential_bar):
        q_value, rhs_value = saved

        def vjp_callback(q_host, rhs_host, bar_host):
            with _CLIENT_CALL_LOCK:
                result = client.vector_jacobian_product(
                    inputs=inputs(q_host, rhs_host),
                    vjp_inputs={"q_operator", "rhs"},
                    vjp_outputs={"potential"},
                    cotangent_vector={
                        "potential": np.ascontiguousarray(
                            bar_host, dtype=np.float64
                        )
                    },
                )
            return (
                np.asarray(result["q_operator"], dtype=np.float64, order="C"),
                np.asarray(result["rhs"], dtype=np.float64, order="C"),
            )

        return jax.pure_callback(
            vjp_callback,
            (output_spec, output_spec),
            q_value,
            rhs_value,
            potential_bar,
        )

    implicit_solve.defvjp(forward, backward)
    return implicit_solve(q_operator, rhs)


def solve_unbalanced_screened_poisson3d_batch_tesseract(
    q: Array,
    h_ub: Array,
    *,
    mass: Array,
    grid: PeriodicGrid3D,
    config: UnbalancedCorrectionConfig = UnbalancedCorrectionConfig(),
) -> UnbalancedCorrectionResult:
    """Solve a full trajectory in one differentiable native batch call."""
    q = jnp.asarray(q, dtype=jnp.float64)
    h_ub = jnp.asarray(h_ub, dtype=jnp.float64)
    mass = jnp.asarray(mass, dtype=jnp.float64)
    if q.ndim != 4 or q.shape[1:] != grid.shape or h_ub.shape != q.shape:
        raise ValueError(f"q and h_ub must have shape [B,{','.join(map(str, grid.shape))}]")
    if mass.shape != (q.shape[0],):
        raise ValueError("mass must have one scalar per batch system")

    axes = (-3, -2, -1)
    floor = float(config.operator_floor_rel) * jnp.max(q, axis=axes, keepdims=True)
    q_operator = q + floor
    rhs = q * h_ub
    potential = solve_linear_system_batch_tesseract(
        q_operator, rhs, grid=grid, config=config
    )
    inverse_kappa = 1.0 / float(config.reaction_kappa)
    physical_laplacian = periodic_weighted_laplacian3d(
        potential, q, grid.spacings
    )
    stabilized_laplacian = periodic_weighted_laplacian3d(
        potential, q_operator, grid.spacings
    )
    scale = mass * float(grid.cell_volume)
    move = scale * jnp.sum(potential * physical_laplacian, axis=axes)
    reaction = scale * inverse_kappa * jnp.sum(q * potential**2, axis=axes)
    total = move + reaction
    physical_residual = physical_laplacian + inverse_kappa * q * potential - rhs
    stabilized_residual = (
        stabilized_laplacian + inverse_kappa * q_operator * potential - rhs
    )
    return UnbalancedCorrectionResult(
        total_action=total,
        move_action=move,
        reaction_action=reaction,
        reaction_fraction=jnp.where(total > 0.0, reaction / total, 0.0),
        potential=potential,
        source_correction=inverse_kappa * potential,
        relative_residual=stable_relative_residual(
            physical_residual, rhs, axes=axes
        ),
        stabilized_relative_residual=stable_relative_residual(
            stabilized_residual, rhs, axes=axes
        ),
        operator_floor=floor.reshape((q.shape[0],)),
    )


def native_diagnostics(
    q: Any,
    h_ub: Any,
    *,
    grid: PeriodicGrid3D,
    config: UnbalancedCorrectionConfig = UnbalancedCorrectionConfig(),
) -> dict[str, Any]:
    """Return non-differentiable native iteration diagnostics for audits."""
    import numpy as np

    q_array = np.ascontiguousarray(q, dtype=np.float64)
    h_array = np.ascontiguousarray(h_ub, dtype=np.float64)
    if q_array.ndim != 4 or h_array.shape != q_array.shape or q_array.shape[1:] != grid.shape:
        raise ValueError("q and h_ub have an invalid batch/grid shape")
    floor = config.operator_floor_rel * np.max(
        q_array, axis=(-3, -2, -1), keepdims=True
    )
    q_operator = np.ascontiguousarray(q_array + floor)
    rhs = np.ascontiguousarray(q_array * h_array)
    build = _native_root() / "build"
    if str(build) not in sys.path:
        sys.path.insert(0, str(build))
    try:
        import _active_nematic_unbalanced_screened_native as native
    except ImportError as exc:
        raise UnbalancedScreenedPoissonUnavailable("Native extension is not built") from exc
    result = native.solve_batch(
        q_operator, rhs, float(grid.dx), float(grid.dy),
        float(grid.dtheta_metric), float(config.reaction_kappa),
        float(config.cg_tol), int(config.cg_maxiter), None,
    )
    return {
        "potential": np.asarray(result["potential"], dtype=np.float64),
        "iterations": np.asarray(result["iterations"], dtype=np.int32),
        "converged": np.asarray(result["converged"], dtype=bool),
        "stabilized_relative_residual": np.asarray(
            result["relative_residual"], dtype=np.float64
        ),
    }
