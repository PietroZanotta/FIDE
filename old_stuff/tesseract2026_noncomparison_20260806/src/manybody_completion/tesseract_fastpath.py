"""Coarse-grained persistent Tesseract execution helpers.

The fastest pattern is:

1. keep each Tesseract server alive for the whole fine-tuning phase;
2. pass a complete minibatch with a leading batch axis in one request;
3. keep all inner solver iterations inside the Tesseract ``apply`` call;
4. call through ``tesseract_jax.apply_tesseract`` inside the compiled loss.

Do not construct containers or clients inside an optimizer step.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Callable

import jax

try:
    from tesseract_core import Tesseract
    from tesseract_jax import apply_tesseract
except ImportError as error:  # pragma: no cover - optional dependency
    Tesseract = None  # type: ignore[assignment]
    apply_tesseract = None  # type: ignore[assignment]
    _IMPORT_ERROR = error
else:
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class TesseractEndpoints:
    relaxation_url: str
    projection_url: str


class PersistentTesseractPair:
    """Open two already-running Tesseract services once and reuse them."""

    def __init__(self, endpoints: TesseractEndpoints):
        if Tesseract is None:
            raise ImportError(
                "tesseract-core and tesseract-jax are required"
            ) from _IMPORT_ERROR
        self.endpoints = endpoints
        self._stack = ExitStack()
        self.relaxation: Any | None = None
        self.projection: Any | None = None

    def __enter__(self) -> "PersistentTesseractPair":
        self.relaxation = self._stack.enter_context(
            Tesseract.from_url(self.endpoints.relaxation_url)
        )
        self.projection = self._stack.enter_context(
            Tesseract.from_url(self.endpoints.projection_url)
        )
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._stack.close()


def make_batched_solver_pipeline(
    clients: PersistentTesseractPair,
    relaxation_inputs: Callable[[Any], Any],
    projection_inputs: Callable[[Any, Any], Any],
    relaxation_output: Callable[[Any], Any],
    projection_output: Callable[[Any], Any],
) -> Callable[[Any, Any], tuple[Any, dict[str, Any]]]:
    """Create a jittable two-Tesseract pipeline using one call per stage.

    The Tesseract APIs themselves must accept arrays with a leading minibatch
    axis.  This avoids ``vmap_method='sequential'``, which would issue one RPC
    per sample.  If you must wrap a scalar API with ``jax.vmap``, use
    ``vmap_method='broadcast_all'`` only after updating the schemas and apply
    functions to accept that leading batch axis.
    """
    if apply_tesseract is None:
        raise ImportError("tesseract-jax is required") from _IMPORT_ERROR
    if clients.relaxation is None or clients.projection is None:
        raise RuntimeError("PersistentTesseractPair must be entered first")

    relaxation_client = clients.relaxation
    projection_client = clients.projection

    def pipeline(coordinates: Any, target_moments: Any) -> tuple[Any, dict[str, Any]]:
        relaxation_result = apply_tesseract(
            relaxation_client,
            relaxation_inputs(coordinates),
        )
        relaxed = relaxation_output(relaxation_result)
        projection_result = apply_tesseract(
            projection_client,
            projection_inputs(relaxed, target_moments),
        )
        projected = projection_output(projection_result)
        return projected, {
            "relaxation": relaxation_result,
            "projection": projection_result,
        }

    return jax.jit(pipeline)


def enable_runtime_vjp_cache(size: int = 1) -> bool:
    """Enable the optional in-process JAX Tesseract VJP cache when available.

    Call this inside each JAX-backed ``tesseract_api.py`` at import time.  A
    size of one covers the usual apply-then-VJP pattern.  Benchmark it: on very
    small solvers, cache bookkeeping can cost more than the saved forward pass.
    """
    try:
        from tesseract_core.runtime.experimental import set_jax_vjp_cache_size
    except ImportError:
        return False
    set_jax_vjp_cache_size(size)
    return True
