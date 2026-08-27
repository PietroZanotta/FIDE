"""Optional Tesseract wrapper for fixed-feature Galerkin K/f chunk assembly."""

from __future__ import annotations

from functools import lru_cache
import importlib
from pathlib import Path
import sys
from typing import Any

import jax.numpy as jnp
import numpy as np


class TesseractGalerkinUnavailable(RuntimeError):
    pass


def _native_root() -> Path:
    return Path(__file__).resolve().parents[2] / "native" / "galerkin_tesseract"


@lru_cache(maxsize=1)
def _native_module() -> Any:
    build = _native_root() / "build"
    if str(build) not in sys.path:
        sys.path.insert(0, str(build))
    try:
        return importlib.import_module("_galerkin_native")
    except ImportError as exc:
        raise TesseractGalerkinUnavailable("native Galerkin Tesseract is not built") from exc


def is_tesseract_galerkin_available() -> bool:
    try:
        _native_module()
        import tesseract_core  # noqa: F401
        import tesseract_jax  # noqa: F401
    except (ImportError, TesseractGalerkinUnavailable):
        return False
    return (_native_root() / "tesseract_api.py").is_file()


def assemble_galerkin_chunk_tesseract_forward(
    values: np.ndarray, gradients: np.ndarray, weights: np.ndarray, forcing: np.ndarray,
) -> dict[str, np.ndarray]:
    """Assemble raw additive statistics for one `[N,K,P,D]` chunk."""
    result = _native_module().assemble_chunk(
        np.ascontiguousarray(values, dtype=np.float64),
        np.ascontiguousarray(gradients, dtype=np.float64),
        np.ascontiguousarray(weights, dtype=np.float64),
        np.ascontiguousarray(forcing, dtype=np.float64),
    )
    return {name: np.asarray(value) for name, value in result.items()}


@lru_cache(maxsize=1)
def _client() -> Any:
    try:
        from tesseract_core import Tesseract
        import tesseract_jax  # noqa: F401
        return Tesseract.from_tesseract_api(_native_root() / "tesseract_api.py")
    except (ImportError, RuntimeError) as exc:
        raise TesseractGalerkinUnavailable("Galerkin Tesseract client is unavailable") from exc


def assemble_galerkin_chunk_tesseract(values, gradients, weights, forcing):
    """JAX-callable forward-only Tesseract assembly endpoint."""
    try:
        from tesseract_jax import apply_tesseract
    except ImportError as exc:
        raise TesseractGalerkinUnavailable("tesseract-jax is unavailable") from exc
    return apply_tesseract(_client(), {
        "values": jnp.asarray(values, dtype=jnp.float64),
        "gradients": jnp.asarray(gradients, dtype=jnp.float64),
        "weights": jnp.asarray(weights, dtype=jnp.float64),
        "forcing": jnp.asarray(forcing, dtype=jnp.float64),
    })


def finalize_galerkin_statistics(gram, raw_load, basis_mean, forcing_sum):
    """Apply the global centering term after additive chunk accumulation."""
    gram = 0.5 * (jnp.asarray(gram) + jnp.asarray(gram).T)
    return gram, jnp.asarray(raw_load) - jnp.asarray(forcing_sum).reshape(()) * jnp.asarray(basis_mean)


__all__ = [
    "TesseractGalerkinUnavailable", "assemble_galerkin_chunk_tesseract",
    "assemble_galerkin_chunk_tesseract_forward", "finalize_galerkin_statistics",
    "is_tesseract_galerkin_available",
]
