"""Optional Tesseract wrapper for the weak weighted-Poisson Ritz backend."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys
from typing import Any

import numpy as np


VARIATIONAL_SOLVER_REVISION = "cpp-openmp-neumann-cosine-ritz-v1"


class TesseractVariationalPoissonUnavailable(RuntimeError):
    """Raised when the explicitly requested variational backend is unavailable."""


@dataclass(frozen=True)
class VariationalPoissonConfig:
    dx: float
    maximum_mode: int = 5
    rank_relative_tolerance: float = 1.0e-12
    weak_relative_tolerance: float = 1.0e-9
    eigensolver_tolerance: float = 1.0e-13
    maximum_eigensolver_sweeps: int = 80


def _native_root() -> Path:
    return Path(__file__).resolve().parents[2] / "native" / "variational_poisson_tesseract"


def is_tesseract_variational_poisson_available() -> bool:
    try:
        import tesseract_core  # noqa: F401
        import tesseract_jax  # noqa: F401
    except ImportError:
        return False
    root = _native_root()
    return (root / "tesseract_api.py").is_file() and any(
        (root / "build").glob("_variational_poisson_native*.so")
    )


def _validated_arrays(log_q_mass: Any, forcing: Any) -> tuple[np.ndarray, np.ndarray]:
    log_q = np.asarray(log_q_mass, dtype=np.float64, order="C")
    h = np.asarray(forcing, dtype=np.float64, order="C")
    if log_q.ndim != 3 or h.shape != log_q.shape:
        raise ValueError("log_q_mass and forcing must have the same [B,H,W] shape")
    if min(log_q.shape[-2:]) < 3:
        raise ValueError("variational grids must be at least 3x3")
    if not np.isfinite(log_q).all() or not np.isfinite(h).all():
        raise ValueError("log_q_mass and forcing must contain only finite values")
    return np.ascontiguousarray(log_q), np.ascontiguousarray(h)


def _native_module() -> Any:
    build = _native_root() / "build"
    if str(build) not in sys.path:
        sys.path.insert(0, str(build))
    try:
        import _variational_poisson_native as native
    except ImportError as exc:
        raise TesseractVariationalPoissonUnavailable(
            "The variational weighted-Poisson extension is not built. See "
            "native/variational_poisson_tesseract/README.md."
        ) from exc
    return native


def solve_variational_poisson_batch_native(
    log_q_mass: Any,
    forcing: Any,
    cfg: VariationalPoissonConfig,
) -> dict[str, np.ndarray]:
    """Run the native weak solve and return its complete audit diagnostics."""
    log_q, h = _validated_arrays(log_q_mass, forcing)
    result = _native_module().solve_batch(
        log_q,
        h,
        float(cfg.dx),
        int(cfg.maximum_mode),
        float(cfg.rank_relative_tolerance),
        float(cfg.weak_relative_tolerance),
        float(cfg.eigensolver_tolerance),
        int(cfg.maximum_eigensolver_sweeps),
    )
    return {name: np.asarray(value) for name, value in result.items()}


@lru_cache(maxsize=1)
def _client() -> Any:
    try:
        from tesseract_core import Tesseract
        import tesseract_jax  # noqa: F401
    except ImportError as exc:
        raise TesseractVariationalPoissonUnavailable(
            "The variational backend requires tesseract-core and tesseract-jax."
        ) from exc
    try:
        return Tesseract.from_tesseract_api(_native_root() / "tesseract_api.py")
    except (ImportError, RuntimeError) as exc:
        raise TesseractVariationalPoissonUnavailable(
            "The variational Tesseract endpoint could not be loaded. See "
            "native/variational_poisson_tesseract/README.md."
        ) from exc


def solve_variational_poisson_batch_tesseract(
    log_q_mass: Any,
    forcing: Any,
    cfg: VariationalPoissonConfig,
) -> dict[str, Any]:
    """Run the weak solve through its isolated in-process Tesseract endpoint."""
    try:
        import jax.numpy as jnp
        from tesseract_jax import apply_tesseract
    except ImportError as exc:
        raise TesseractVariationalPoissonUnavailable(
            "The variational backend requires the optional tesseract-jax package."
        ) from exc
    log_q, h = _validated_arrays(log_q_mass, forcing)
    inputs = {
        "log_q_mass": jnp.asarray(log_q),
        "forcing": jnp.asarray(h),
        "dx": float(cfg.dx),
        "maximum_mode": int(cfg.maximum_mode),
        "rank_relative_tolerance": float(cfg.rank_relative_tolerance),
        "weak_relative_tolerance": float(cfg.weak_relative_tolerance),
        "eigensolver_tolerance": float(cfg.eigensolver_tolerance),
        "maximum_eigensolver_sweeps": int(cfg.maximum_eigensolver_sweeps),
    }
    return apply_tesseract(_client(), inputs)
