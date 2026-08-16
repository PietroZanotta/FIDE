"""Optional Tesseract-JAX wrapper for batched I-projection trajectories."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from .projection import IProjectionConfig

Array = jax.Array


class TesseractIProjectionUnavailable(RuntimeError):
    pass


def _native_root() -> Path:
    return Path(__file__).resolve().parents[2] / "native" / "iprojection_tesseract"


def is_tesseract_iprojection_available() -> bool:
    try:
        import tesseract_core  # noqa: F401
        import tesseract_jax  # noqa: F401
    except ImportError:
        return False
    return (_native_root() / "tesseract_api.py").is_file() and any(
        (_native_root() / "build").glob("_iprojection_native*.so")
    )


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
