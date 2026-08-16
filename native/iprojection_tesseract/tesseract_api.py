"""In-process differentiable API for batched empirical I-projection trajectories."""

from __future__ import annotations

from pathlib import Path
import sys
import threading
from typing import Any

import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64

_BUILD_DIR = Path(__file__).resolve().parent / "build"
if str(_BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(_BUILD_DIR))

try:
    import _iprojection_native as native
except ImportError as exc:  # pragma: no cover - optional build
    raise ImportError(
        "The native I-projection extension is not built; see "
        "native/iprojection_tesseract/README.md."
    ) from exc


Tensor3 = Array[(None, None, None), Float64]
Tensor2 = Array[(None, None), Float64]


class InputSchema(BaseModel):
    phi: Differentiable[Tensor3]
    log_base_weights: Differentiable[Tensor2]
    targets: Differentiable[Tensor3]
    max_steps: int
    residual_tol: float
    newton_ridge: float
    step_cap: float
    lambda_clip: float
    line_search_steps: int
    implicit_ridge: float


class OutputSchema(BaseModel):
    lambda_values: Differentiable[Tensor3]


_CACHE_LOCK = threading.Lock()
_FORWARD_CACHE: tuple[Any, ...] | None = None


def _arrays(inputs: InputSchema):
    return (
        np.ascontiguousarray(inputs.phi, dtype=np.float64),
        np.ascontiguousarray(inputs.log_base_weights, dtype=np.float64),
        np.ascontiguousarray(inputs.targets, dtype=np.float64),
    )


def _args(inputs: InputSchema):
    return (
        inputs.max_steps,
        inputs.residual_tol,
        inputs.newton_ridge,
        inputs.step_cap,
        inputs.lambda_clip,
        inputs.line_search_steps,
        inputs.implicit_ridge,
    )


def _solve(inputs: InputSchema):
    global _FORWARD_CACHE
    arrays = _arrays(inputs)
    args = _args(inputs)
    result = native.solve_batch(*arrays, *args)
    with _CACHE_LOCK:
        # The in-process reverse callback immediately follows this primal call.
        # Retaining references avoids another Newton trajectory in the VJP while
        # exact equality checks below preserve correctness under interleaving.
        _FORWARD_CACHE = (*arrays, args, np.asarray(result["lambda_values"]))
    return result


def _cached_lambda(arrays, args) -> np.ndarray:
    with _CACHE_LOCK:
        cached = _FORWARD_CACHE
        if (
            cached is not None
            and cached[3] == args
            and all(
                old.shape == new.shape and np.array_equal(old, new)
                for old, new in zip(cached[:3], arrays, strict=True)
            )
        ):
            return np.ascontiguousarray(cached[4], dtype=np.float64)
    # Correct fallback for an independently invoked JVP/VJP or interleaved call.
    result = native.solve_batch(*arrays, *args)
    return np.ascontiguousarray(result["lambda_values"], dtype=np.float64)


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(lambda_values=_solve(inputs)["lambda_values"])


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, np.ndarray]:
    if "lambda_values" not in vjp_outputs or "lambda_values" not in cotangent_vector:
        return {}
    phi, log_base, targets = _arrays(inputs)
    args = _args(inputs)
    lambda_values = _cached_lambda((phi, log_base, targets), args)
    result = native.vjp_batch(
        phi,
        log_base,
        targets,
        lambda_values,
        np.ascontiguousarray(cotangent_vector["lambda_values"], dtype=np.float64),
        *args,
    )
    return {name: np.asarray(result[name], dtype=np.float64, order="C")
            for name in ("phi", "log_base_weights", "targets") if name in vjp_inputs}


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
) -> dict[str, np.ndarray]:
    if "lambda_values" not in jvp_outputs:
        return {}
    phi, log_base, targets = _arrays(inputs)
    args = _args(inputs)
    lambda_values = _cached_lambda((phi, log_base, targets), args)
    result = native.jvp_batch(
        phi,
        log_base,
        targets,
        lambda_values,
        np.ascontiguousarray(tangent_vector.get("phi", np.zeros_like(phi)), dtype=np.float64),
        np.ascontiguousarray(
            tangent_vector.get("log_base_weights", np.zeros_like(log_base)), dtype=np.float64
        ),
        np.ascontiguousarray(
            tangent_vector.get("targets", np.zeros_like(targets)), dtype=np.float64
        ),
        *args,
    )
    return {"lambda_values": np.asarray(result, dtype=np.float64, order="C")}


def abstract_eval(abstract_inputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    targets = (
        abstract_inputs["targets"]
        if isinstance(abstract_inputs, dict)
        else abstract_inputs.targets
    )
    shape = tuple(targets["shape"] if isinstance(targets, dict) else targets.shape)
    if len(shape) != 3:
        raise ValueError("targets must have abstract shape [B,T,M]")
    return {"lambda_values": {"shape": shape, "dtype": "float64"}}
