"""In-process Tesseract API for the native batched weighted-Poisson solve.

Endpoint code intentionally uses NumPy and the C++ extension only.  In
particular, no JAX operation is performed from a JAX FFI callback.
"""

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
    import _poisson_native as native
except ImportError as exc:  # pragma: no cover - exercised by optional-import test
    raise ImportError(
        "The native weighted-Poisson extension is not built. See "
        "native/poisson_tesseract/README.md for the CMake commands."
    ) from exc


BatchGrid64 = Array[(None, None, None), Float64]


class InputSchema(BaseModel):
    q_operator: Differentiable[BatchGrid64]
    rhs: Differentiable[BatchGrid64]
    gauge: Differentiable[BatchGrid64]
    dx: float
    gauge_strength: float
    cg_tol: float
    cg_maxiter: int


class OutputSchema(BaseModel):
    psi: Differentiable[BatchGrid64]


_CACHE_LOCK = threading.Lock()
_FORWARD_CACHE: tuple[Any, ...] | None = None
_ADJOINT_GUESS: tuple[Any, ...] | None = None
_TANGENT_GUESS: tuple[Any, ...] | None = None


def _arrays(inputs: InputSchema) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.ascontiguousarray(inputs.q_operator, dtype=np.float64),
        np.ascontiguousarray(inputs.rhs, dtype=np.float64),
        np.ascontiguousarray(inputs.gauge, dtype=np.float64),
    )


def _args(inputs: InputSchema) -> tuple[float, float, float, int]:
    return (
        inputs.dx,
        inputs.gauge_strength,
        inputs.cg_tol,
        inputs.cg_maxiter,
    )


def _solve(
    q_operator: np.ndarray,
    rhs: np.ndarray,
    gauge: np.ndarray,
    inputs: InputSchema,
    *,
    label: str,
    initial_guess: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    result = native.solve_batch(
        q_operator,
        rhs,
        gauge,
        inputs.dx,
        inputs.gauge_strength,
        inputs.cg_tol,
        inputs.cg_maxiter,
        initial_guess,
    )
    converged = np.asarray(result["converged"], dtype=bool)
    if not np.all(converged):
        failed = np.flatnonzero(~converged).tolist()
        residuals = np.asarray(result["relative_residual"])[~converged].tolist()
        iterations = np.asarray(result["iterations"])[~converged].tolist()
        raise RuntimeError(
            f"Native {label} PCG did not converge for batch indices {failed}; "
            f"iterations={iterations}, relative_residuals={residuals}"
        )
    psi = np.asarray(result["psi"], dtype=np.float64, order="C")
    if not np.all(np.isfinite(psi)):
        raise FloatingPointError(f"Native {label} PCG returned non-finite values")
    return result


def _warm_start(arrays: tuple[np.ndarray, np.ndarray, np.ndarray], inputs: InputSchema):
    """Reuse the previous same-resolution primal as an algorithmic initial guess."""
    with _CACHE_LOCK:
        cached = _FORWARD_CACHE
        if (
            cached is not None
            and cached[3] == _args(inputs)
            and cached[4].shape == arrays[0].shape
        ):
            return np.ascontiguousarray(cached[4], dtype=np.float64)
    return None


def _solver_guess(cache: tuple[Any, ...] | None, shape, inputs: InputSchema):
    with _CACHE_LOCK:
        if cache is not None and cache[0] == _args(inputs) and cache[1].shape == shape:
            return np.ascontiguousarray(cache[1], dtype=np.float64)
    return None


def _cached_forward(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    inputs: InputSchema,
) -> np.ndarray:
    args = _args(inputs)
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
    return np.asarray(
        _solve(*arrays, inputs, label="forward-cache-miss")["psi"],
        dtype=np.float64,
        order="C",
    )


def apply(inputs: InputSchema) -> OutputSchema:
    global _FORWARD_CACHE
    arrays = _arrays(inputs)
    q_operator, rhs, gauge = arrays
    result = _solve(
        q_operator,
        rhs,
        gauge,
        inputs,
        label="forward",
        initial_guess=_warm_start(arrays, inputs),
    )
    psi = np.asarray(result["psi"], dtype=np.float64, order="C")
    with _CACHE_LOCK:
        _FORWARD_CACHE = (
            q_operator,
            rhs,
            gauge,
            _args(inputs),
            psi,
        )
    return OutputSchema(psi=result["psi"])


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, np.ndarray]:
    global _ADJOINT_GUESS
    if "psi" not in vjp_outputs or "psi" not in cotangent_vector:
        return {}

    q_operator, rhs, gauge = _arrays(inputs)
    psi = _cached_forward((q_operator, rhs, gauge), inputs)
    psi_bar = np.ascontiguousarray(cotangent_vector["psi"], dtype=np.float64)
    adjoint_result = _solve(
        q_operator,
        psi_bar,
        gauge,
        inputs,
        label="adjoint",
        initial_guess=_solver_guess(_ADJOINT_GUESS, q_operator.shape, inputs),
    )
    lambda_ = np.asarray(
        adjoint_result["psi"],
        dtype=np.float64,
        order="C",
    )
    with _CACHE_LOCK:
        _ADJOINT_GUESS = (_args(inputs), lambda_)

    result: dict[str, np.ndarray] = {}
    if "q_operator" in vjp_inputs:
        result["q_operator"] = native.weighted_operator_vjp(
            psi, lambda_, inputs.dx
        )
    if "rhs" in vjp_inputs:
        result["rhs"] = lambda_
    if "gauge" in vjp_inputs:
        result["gauge"] = native.gauge_vjp(
            psi, lambda_, gauge, inputs.gauge_strength
        )
    return result


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Implicit forward derivative required by Tesseract-JAX's primitive rule."""
    global _TANGENT_GUESS
    if "psi" not in jvp_outputs:
        return {}
    q_operator, rhs, gauge = _arrays(inputs)
    psi = _cached_forward((q_operator, rhs, gauge), inputs)
    zero = np.zeros_like(psi)
    q_dot = np.ascontiguousarray(tangent_vector.get("q_operator", zero), dtype=np.float64)
    rhs_dot = np.ascontiguousarray(tangent_vector.get("rhs", zero), dtype=np.float64)
    gauge_dot = np.ascontiguousarray(tangent_vector.get("gauge", zero), dtype=np.float64)
    effective_rhs = native.linearized_rhs(
        psi,
        q_dot,
        rhs_dot,
        gauge,
        gauge_dot,
        inputs.dx,
        inputs.gauge_strength,
    )
    tangent_result = _solve(
        q_operator,
        np.ascontiguousarray(effective_rhs),
        gauge,
        inputs,
        label="tangent",
        initial_guess=_solver_guess(_TANGENT_GUESS, q_operator.shape, inputs),
    )
    psi_dot = np.asarray(tangent_result["psi"], dtype=np.float64, order="C")
    with _CACHE_LOCK:
        _TANGENT_GUESS = (_args(inputs), psi_dot)
    return {"psi": psi_dot}


def abstract_eval(abstract_inputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    q = (
        abstract_inputs["q_operator"]
        if isinstance(abstract_inputs, dict)
        else abstract_inputs.q_operator
    )
    shape = tuple(q["shape"] if isinstance(q, dict) else q.shape)
    if len(shape) != 3:
        raise ValueError("q_operator must have abstract shape [B,H,W]")
    return {"psi": {"shape": shape, "dtype": "float64"}}
