"""In-process Tesseract API for the weak weighted-Poisson Ritz solve."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Float64

_BUILD_DIR = Path(__file__).resolve().parent / "build"
if str(_BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(_BUILD_DIR))

try:
    import _variational_poisson_native as native
except ImportError as exc:  # pragma: no cover - exercised by optional-import tests
    raise ImportError(
        "The variational weighted-Poisson extension is not built. See "
        "native/variational_poisson_tesseract/README.md."
    ) from exc


BatchGrid64 = Array[(None, None, None), Float64]
BatchVector64 = Array[(None,), Float64]


class InputSchema(BaseModel):
    log_q_mass: BatchGrid64
    forcing: BatchGrid64
    dx: float
    maximum_mode: int
    rank_relative_tolerance: float
    weak_relative_tolerance: float
    eigensolver_tolerance: float
    maximum_eigensolver_sweeps: int


class OutputSchema(BaseModel):
    potential: BatchGrid64
    action: BatchVector64
    objective: BatchVector64
    weak_relative_residual: BatchVector64
    scaled_weak_relative_residual: BatchVector64
    gauge_residual: BatchVector64
    compatibility_residual: BatchVector64
    compatibility_relative_residual: BatchVector64
    energy_load_identity_relative_error: BatchVector64
    condition_proxy: BatchVector64
    retained_rank: BatchVector64
    basis_size: BatchVector64
    eigensolver_sweeps: BatchVector64
    quadrature_underflow_count: BatchVector64
    converged: BatchVector64


def _solve(inputs: InputSchema) -> dict[str, np.ndarray]:
    return native.solve_batch(
        np.ascontiguousarray(inputs.log_q_mass, dtype=np.float64),
        np.ascontiguousarray(inputs.forcing, dtype=np.float64),
        inputs.dx,
        inputs.maximum_mode,
        inputs.rank_relative_tolerance,
        inputs.weak_relative_tolerance,
        inputs.eigensolver_tolerance,
        inputs.maximum_eigensolver_sweeps,
    )


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(**_solve(inputs))


def abstract_eval(abstract_inputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    log_q = (
        abstract_inputs["log_q_mass"]
        if isinstance(abstract_inputs, dict)
        else abstract_inputs.log_q_mass
    )
    shape = tuple(log_q["shape"] if isinstance(log_q, dict) else log_q.shape)
    if len(shape) != 3:
        raise ValueError("log_q_mass must have abstract shape [B,H,W]")
    batch_shape = (shape[0],)
    output = {"potential": {"shape": shape, "dtype": "float64"}}
    for name in (
        "action",
        "objective",
        "weak_relative_residual",
        "scaled_weak_relative_residual",
        "gauge_residual",
        "compatibility_residual",
        "compatibility_relative_residual",
        "energy_load_identity_relative_error",
        "condition_proxy",
        "retained_rank",
        "basis_size",
        "eigensolver_sweeps",
        "quadrature_underflow_count",
        "converged",
    ):
        output[name] = {"shape": batch_shape, "dtype": "float64"}
    return output
