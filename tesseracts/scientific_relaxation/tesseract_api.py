"""Equation-identical Tesseract wrapper for the homometric relaxation solver."""

from __future__ import annotations

from typing import Any

import jax
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, ShapeDType
from tesseract_core.runtime.jax_recipes import jax_apply, jax_jacobian, jax_jvp, jax_vjp

from manybody_completion.energy import PhysicalParameters
from manybody_completion.solvers import RelaxationOptions, relax_ensemble

Numeric = Array[..., None]
Scalar = Array[(), None]
Bool = Array[(), "bool"]


class InputSchema(BaseModel):
    coordinates: Differentiable[Array[(None, None, 2), None]]
    box: Array[(2,), None]
    r0: float
    kappa: float
    num_steps: int = Field(ge=1)
    step_size: float = Field(gt=0.0)
    prox_strength: float = Field(ge=0.0)
    max_particle_step: float = Field(gt=0.0)
    tolerance: float = Field(ge=0.0)


class OutputSchema(BaseModel):
    relaxed_coordinates: Differentiable[Array[(None, None, 2), None]]
    energy_before: Scalar
    energy_after: Scalar
    correction_rms: Scalar
    stationarity_rms: Scalar
    converged: Bool


def apply_jit(inputs: dict[str, Any]) -> dict[str, Any]:
    relaxed, diagnostics = relax_ensemble(
        inputs["coordinates"],
        inputs["box"],
        PhysicalParameters(r0=inputs["r0"], kappa=inputs["kappa"]),
        RelaxationOptions(
            num_steps=inputs["num_steps"],
            step_size=inputs["step_size"],
            prox_strength=inputs["prox_strength"],
            max_particle_step=inputs["max_particle_step"],
            tolerance=inputs["tolerance"],
        ),
    )
    diagnostics = jax.tree_util.tree_map(jax.lax.stop_gradient, diagnostics)
    return {"relaxed_coordinates": relaxed, **diagnostics}


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema.model_validate(jax_apply(apply_jit, inputs))


def abstract_eval(abstract_inputs) -> dict[str, ShapeDType]:
    scalar = ShapeDType(shape=(), dtype=abstract_inputs.coordinates.dtype)
    return {
        "relaxed_coordinates": abstract_inputs.coordinates,
        "energy_before": scalar,
        "energy_after": scalar,
        "correction_rms": scalar,
        "stationarity_rms": scalar,
        "converged": ShapeDType(shape=(), dtype="bool"),
    }


def jacobian(inputs, jac_inputs, jac_outputs):
    return jax_jacobian(apply_jit, inputs, jac_inputs, jac_outputs)


def jacobian_vector_product(inputs, jvp_inputs, jvp_outputs, tangent_vector):
    return jax_jvp(apply_jit, inputs, jvp_inputs, jvp_outputs, tangent_vector)


def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    return jax_vjp(apply_jit, inputs, vjp_inputs, vjp_outputs, cotangent_vector)
