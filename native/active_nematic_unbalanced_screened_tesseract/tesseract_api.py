"""Differentiable in-process Tesseract API for screened unbalanced MFSI.

The forward, tangent, and adjoint systems use the same symmetric positive
definite native operator. Derivatives are implicit; PCG iterations are never
unrolled by JAX.
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
    import _active_nematic_unbalanced_screened_native as native
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Build native/active_nematic_unbalanced_screened_tesseract first."
    ) from exc


BatchGrid64 = Array[(None, None, None, None), Float64]


class InputSchema(BaseModel):
    q_operator: Differentiable[BatchGrid64]
    rhs: Differentiable[BatchGrid64]
    dx: float
    dy: float
    dtheta_metric: float
    kappa: float
    cg_tol: float
    cg_maxiter: int


class OutputSchema(BaseModel):
    potential: Differentiable[BatchGrid64]


_LOCK = threading.Lock()
_FORWARD_CACHE: tuple[Any, ...] | None = None
_ADJOINT_GUESS: tuple[Any, ...] | None = None
_TANGENT_GUESS: tuple[Any, ...] | None = None


def _arrays(inputs: InputSchema) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.ascontiguousarray(inputs.q_operator, dtype=np.float64),
        np.ascontiguousarray(inputs.rhs, dtype=np.float64),
    )


def _args(inputs: InputSchema) -> tuple[float, float, float, float, float, int]:
    return (
        inputs.dx, inputs.dy, inputs.dtheta_metric, inputs.kappa,
        inputs.cg_tol, inputs.cg_maxiter,
    )


def _solve(
    q_operator: np.ndarray,
    rhs: np.ndarray,
    inputs: InputSchema,
    *,
    label: str,
    initial_guess: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    result = native.solve_batch(
        q_operator, rhs, inputs.dx, inputs.dy, inputs.dtheta_metric,
        inputs.kappa, inputs.cg_tol, inputs.cg_maxiter, initial_guess,
    )
    converged = np.asarray(result["converged"], dtype=bool)
    if not np.all(converged):
        failed = np.flatnonzero(~converged).tolist()
        residuals = np.asarray(result["relative_residual"])[~converged].tolist()
        iterations = np.asarray(result["iterations"])[~converged].tolist()
        raise RuntimeError(
            f"Native screened {label} PCG failed at batches {failed}; "
            f"iterations={iterations}, residuals={residuals}"
        )
    potential = np.asarray(result["potential"], dtype=np.float64, order="C")
    if not np.all(np.isfinite(potential)):
        raise FloatingPointError(f"Native screened {label} returned non-finite values")
    return result


def _guess(cache: tuple[Any, ...] | None, shape, inputs: InputSchema):
    with _LOCK:
        if cache is not None and cache[0] == _args(inputs) and cache[1].shape == shape:
            return np.ascontiguousarray(cache[1], dtype=np.float64)
    return None


def _forward(arrays: tuple[np.ndarray, np.ndarray], inputs: InputSchema) -> np.ndarray:
    args = _args(inputs)
    with _LOCK:
        cached = _FORWARD_CACHE
        if (
            cached is not None and cached[2] == args
            and all(
                old.shape == new.shape and np.array_equal(old, new)
                for old, new in zip(cached[:2], arrays, strict=True)
            )
        ):
            return np.ascontiguousarray(cached[3], dtype=np.float64)
    return np.asarray(
        _solve(*arrays, inputs, label="forward-cache-miss")["potential"],
        dtype=np.float64, order="C",
    )


def apply(inputs: InputSchema) -> OutputSchema:
    global _FORWARD_CACHE
    arrays = _arrays(inputs)
    with _LOCK:
        cached = _FORWARD_CACHE
        guess = (
            np.ascontiguousarray(cached[3], dtype=np.float64)
            if cached is not None and cached[2] == _args(inputs)
            and cached[3].shape == arrays[0].shape else None
        )
    result = _solve(*arrays, inputs, label="forward", initial_guess=guess)
    potential = np.asarray(result["potential"], dtype=np.float64, order="C")
    with _LOCK:
        _FORWARD_CACHE = (*arrays, _args(inputs), potential)
    return OutputSchema(potential=potential)


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, np.ndarray]:
    global _ADJOINT_GUESS
    if "potential" not in vjp_outputs or "potential" not in cotangent_vector:
        return {}
    q_operator, rhs = _arrays(inputs)
    potential = _forward((q_operator, rhs), inputs)
    potential_bar = np.ascontiguousarray(cotangent_vector["potential"], dtype=np.float64)
    result = _solve(
        q_operator, potential_bar, inputs, label="adjoint",
        initial_guess=_guess(_ADJOINT_GUESS, q_operator.shape, inputs),
    )
    adjoint = np.asarray(result["potential"], dtype=np.float64, order="C")
    with _LOCK:
        _ADJOINT_GUESS = (_args(inputs), adjoint)
    output: dict[str, np.ndarray] = {}
    if "q_operator" in vjp_inputs:
        output["q_operator"] = native.operator_q_vjp(
            potential, adjoint, inputs.dx, inputs.dy,
            inputs.dtheta_metric, inputs.kappa,
        )
    if "rhs" in vjp_inputs:
        output["rhs"] = adjoint
    return output


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
) -> dict[str, np.ndarray]:
    global _TANGENT_GUESS
    del jvp_inputs
    if "potential" not in jvp_outputs:
        return {}
    q_operator, rhs = _arrays(inputs)
    potential = _forward((q_operator, rhs), inputs)
    zero = np.zeros_like(potential)
    q_dot = np.ascontiguousarray(tangent_vector.get("q_operator", zero), dtype=np.float64)
    rhs_dot = np.ascontiguousarray(tangent_vector.get("rhs", zero), dtype=np.float64)
    effective_rhs = native.linearized_rhs(
        potential, q_dot, rhs_dot, inputs.dx, inputs.dy,
        inputs.dtheta_metric, inputs.kappa,
    )
    result = _solve(
        q_operator, np.ascontiguousarray(effective_rhs), inputs, label="tangent",
        initial_guess=_guess(_TANGENT_GUESS, q_operator.shape, inputs),
    )
    potential_dot = np.asarray(result["potential"], dtype=np.float64, order="C")
    with _LOCK:
        _TANGENT_GUESS = (_args(inputs), potential_dot)
    return {"potential": potential_dot}


def abstract_eval(abstract_inputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    q = abstract_inputs["q_operator"] if isinstance(abstract_inputs, dict) else abstract_inputs.q_operator
    shape = tuple(q["shape"] if isinstance(q, dict) else q.shape)
    if len(shape) != 4:
        raise ValueError("q_operator must have abstract shape [B,Nx,Ny,Ntheta]")
    return {"potential": {"shape": shape, "dtype": "float64"}}

