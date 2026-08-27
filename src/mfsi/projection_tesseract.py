"""Optional Tesseract-JAX wrapper for batched I-projection trajectories."""

from __future__ import annotations

from functools import lru_cache
import importlib
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from .projection import IProjectionConfig

Array = jax.Array


class TesseractIProjectionUnavailable(RuntimeError):
    pass


def _native_root() -> Path:
    return Path(__file__).resolve().parents[2] / "native" / "iprojection_tesseract"


def _candidate_native_root() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "native"
        / "candidate_iprojection_tesseract"
    )


def is_tesseract_iprojection_available() -> bool:
    try:
        import tesseract_core  # noqa: F401
        import tesseract_jax  # noqa: F401
    except ImportError:
        return False
    return (_native_root() / "tesseract_api.py").is_file() and any(
        (_native_root() / "build").glob("_iprojection_native*.so")
    )


def is_tesseract_candidate_iprojection_available() -> bool:
    try:
        import tesseract_core  # noqa: F401
        import tesseract_jax  # noqa: F401
    except ImportError:
        return False
    return (_candidate_native_root() / "tesseract_api.py").is_file() and any(
        (_candidate_native_root() / "build").glob(
            "_candidate_iprojection_native*.so"
        )
    )


@lru_cache(maxsize=1)
def _native_module() -> Any:
    """Load the forward C++ kernel without the differentiable JAX wrapper."""
    build_dir = _native_root() / "build"
    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))
    try:
        return importlib.import_module("_iprojection_native")
    except ImportError as exc:
        raise TesseractIProjectionUnavailable(
            "The native I-projection extension is unavailable; see its README."
        ) from exc


@lru_cache(maxsize=1)
def _candidate_native_module() -> Any:
    """Load the independent candidate-batching C++ kernel."""
    build_dir = _candidate_native_root() / "build"
    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))
    try:
        return importlib.import_module("_candidate_iprojection_native")
    except ImportError as exc:
        raise TesseractIProjectionUnavailable(
            "The native candidate I-projection extension is unavailable; "
            "see its README."
        ) from exc


def solve_i_projection_trajectory_tesseract_forward(
    phi: np.ndarray,
    log_base_weights: np.ndarray,
    targets: np.ndarray,
    cfg: IProjectionConfig,
) -> dict[str, np.ndarray]:
    """Run the native forward solver directly for non-differentiated audits.

    Exact audits never request a derivative, so routing them through a JAX custom
    primitive only adds device transfers and compilation.  The same C++ Newton
    kernel is called here, with contiguous float64 arrays and its convergence
    diagnostics preserved for explicit post-checking by the caller.
    """
    result = _native_module().solve_batch(
        np.ascontiguousarray(phi, dtype=np.float64),
        np.ascontiguousarray(log_base_weights, dtype=np.float64),
        np.ascontiguousarray(targets, dtype=np.float64),
        int(cfg.max_steps),
        float(cfg.residual_tol),
        float(cfg.newton_ridge),
        float(cfg.step_cap),
        float(cfg.lambda_clip),
        int(cfg.line_search_steps),
        float(cfg.implicit_ridge),
    )
    return {name: np.asarray(value) for name, value in result.items()}


def solve_i_projection_candidate_trajectories_tesseract_forward(
    phi: np.ndarray,
    log_base_weights: np.ndarray,
    targets: np.ndarray,
    cfg: IProjectionConfig,
) -> dict[str, np.ndarray]:
    """Run candidate-specific trajectories in one native OpenMP call.

    ``phi`` is ``[C,T,N,M]`` and ``targets`` is ``[C,T,M]``.  The common
    reference bank has log weights ``[T,N]``.  Each candidate keeps its own
    multiplier warm start across time.
    """
    result = _candidate_native_module().solve_candidate_batch(
        np.ascontiguousarray(phi, dtype=np.float64),
        np.ascontiguousarray(log_base_weights, dtype=np.float64),
        np.ascontiguousarray(targets, dtype=np.float64),
        int(cfg.max_steps),
        float(cfg.residual_tol),
        float(cfg.newton_ridge),
        float(cfg.step_cap),
        float(cfg.lambda_clip),
        int(cfg.line_search_steps),
        float(cfg.implicit_ridge),
    )
    return {name: np.asarray(value) for name, value in result.items()}


def solve_soft_i_projection_trajectory_tesseract_forward(
    phi: np.ndarray,
    log_base_weights: np.ndarray,
    targets: np.ndarray,
    penalties: np.ndarray,
    cfg: IProjectionConfig,
) -> dict[str, np.ndarray]:
    """Solve an uncertainty-penalized projection without changing hard callers.

    The stationary equation is
    ``E_q[phi] - target + penalty @ lambda = 0``.  This additive endpoint is
    used by the ocean experiment to represent finite-sample moment uncertainty;
    the existing exact I-projection endpoint remains unchanged.
    """
    result = _native_module().solve_soft_batch(
        np.ascontiguousarray(phi, dtype=np.float64),
        np.ascontiguousarray(log_base_weights, dtype=np.float64),
        np.ascontiguousarray(targets, dtype=np.float64),
        np.ascontiguousarray(penalties, dtype=np.float64),
        int(cfg.max_steps),
        float(cfg.residual_tol),
        float(cfg.newton_ridge),
        float(cfg.step_cap),
        float(cfg.lambda_clip),
        int(cfg.line_search_steps),
    )
    return {name: np.asarray(value) for name, value in result.items()}


@lru_cache(maxsize=1)
def _client() -> Any:
    try:
        from tesseract_core import Tesseract
        import tesseract_jax  # noqa: F401
    except ImportError as exc:
        raise TesseractIProjectionUnavailable(
            "The tesseract_cpp I-projection backend requires tesseract-core and tesseract-jax."
        ) from exc
    try:
        return Tesseract.from_tesseract_api(_native_root() / "tesseract_api.py")
    except (ImportError, RuntimeError) as exc:
        raise TesseractIProjectionUnavailable(
            "The native I-projection extension is unavailable; see its README."
        ) from exc


@lru_cache(maxsize=1)
def _candidate_client() -> Any:
    try:
        from tesseract_core import Tesseract
        import tesseract_jax  # noqa: F401
    except ImportError as exc:
        raise TesseractIProjectionUnavailable(
            "The candidate I-projection backend requires tesseract-core and tesseract-jax."
        ) from exc
    try:
        return Tesseract.from_tesseract_api(
            _candidate_native_root() / "tesseract_api.py"
        )
    except (ImportError, RuntimeError) as exc:
        raise TesseractIProjectionUnavailable(
            "The native candidate I-projection extension is unavailable; see its README."
        ) from exc


def solve_i_projection_trajectory_tesseract(
    phi: Array,
    log_base_weights: Array,
    targets: Array,
    cfg: IProjectionConfig,
) -> Array:
    try:
        from tesseract_jax import apply_tesseract
    except ImportError as exc:
        raise TesseractIProjectionUnavailable(
            "The tesseract_cpp I-projection backend requires tesseract-jax."
        ) from exc
    inputs = {
        "phi": jnp.asarray(phi, dtype=jnp.float64),
        "log_base_weights": jnp.asarray(log_base_weights, dtype=jnp.float64),
        "targets": jnp.asarray(targets, dtype=jnp.float64),
        "max_steps": int(cfg.max_steps),
        "residual_tol": float(cfg.residual_tol),
        "newton_ridge": float(cfg.newton_ridge),
        "step_cap": float(cfg.step_cap),
        "lambda_clip": float(cfg.lambda_clip),
        "line_search_steps": int(cfg.line_search_steps),
        "implicit_ridge": float(cfg.implicit_ridge),
    }
    return apply_tesseract(_client(), inputs)["lambda_values"]


def solve_i_projection_candidate_trajectories_tesseract(
    phi: Array,
    log_base_weights: Array,
    targets: Array,
    cfg: IProjectionConfig,
) -> Array:
    """Differentiable candidate-specific native trajectory solve."""
    try:
        from tesseract_jax import apply_tesseract
    except ImportError as exc:
        raise TesseractIProjectionUnavailable(
            "The candidate I-projection backend requires tesseract-jax."
        ) from exc
    inputs = {
        "phi": jnp.asarray(phi, dtype=jnp.float64),
        "log_base_weights": jnp.asarray(log_base_weights, dtype=jnp.float64),
        "targets": jnp.asarray(targets, dtype=jnp.float64),
        "max_steps": int(cfg.max_steps),
        "residual_tol": float(cfg.residual_tol),
        "newton_ridge": float(cfg.newton_ridge),
        "step_cap": float(cfg.step_cap),
        "lambda_clip": float(cfg.lambda_clip),
        "line_search_steps": int(cfg.line_search_steps),
        "implicit_ridge": float(cfg.implicit_ridge),
    }
    return apply_tesseract(_candidate_client(), inputs)["lambda_values"]
