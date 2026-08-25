"""Forward-only Tesseract API for one fixed-feature Galerkin K/f chunk."""
from __future__ import annotations
from pathlib import Path
import sys
from typing import Any
import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Float64

_BUILD = Path(__file__).resolve().parent / "build"
if str(_BUILD) not in sys.path: sys.path.insert(0, str(_BUILD))
import _galerkin_native as native

Vector = Array[(None,), Float64]; Matrix = Array[(None, None), Float64]
Tensor4 = Array[(None, None, None, None), Float64]
class InputSchema(BaseModel):
    values: Matrix; gradients: Tensor4; weights: Vector; forcing: Vector
class OutputSchema(BaseModel):
    gram: Matrix; raw_load: Vector; basis_mean: Vector; forcing_sum: Vector
def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(**native.assemble_chunk(
        np.ascontiguousarray(inputs.values, dtype=np.float64),
        np.ascontiguousarray(inputs.gradients, dtype=np.float64),
        np.ascontiguousarray(inputs.weights, dtype=np.float64),
        np.ascontiguousarray(inputs.forcing, dtype=np.float64)))
def abstract_eval(inputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values = inputs["values"] if isinstance(inputs, dict) else inputs.values
    shape = tuple(values["shape"] if isinstance(values, dict) else values.shape)
    if len(shape) != 2: raise ValueError("values must have shape [N,K]")
    k = shape[1]
    return {"gram":{"shape":(k,k),"dtype":"float64"},
            "raw_load":{"shape":(k,),"dtype":"float64"},
            "basis_mean":{"shape":(k,),"dtype":"float64"},
            "forcing_sum":{"shape":(1,),"dtype":"float64"}}

