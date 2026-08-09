"""Equation-identical Tesseract wrapper for the homometric projection solver."""

from __future__ import annotations

from typing import Any

import jax
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, ShapeDType
from tesseract_core.runtime.jax_recipes import jax_apply, jax_jacobian, jax_jvp, jax_vjp

from manybody_completion.observables import PairBasis
from manybody_completion.solvers import ProjectionOptions, project_ensemble

Scalar = Array[(), None]
Integer = Array[(), "int32"]
Bool = Array[(), "bool"]


class InputSchema(BaseModel):
    coordinates: Differentiable[Array[(None, None, 2), None]]
    target_moments: Differentiable[Array[(None,), None]]
    box: Array[(2,), None]
    basis_centers: Array[(None,), None]
    basis_widths: Array[(None,), None]
    moment_scales: Array[(None,), None]
    num_steps: int = Field(ge=1)
    ridge: float = Field(gt=0.0)
    max_particle_step: float = Field(gt=0.0)
    tolerance: float = Field(ge=0.0)
    rank_tolerance: float = Field(gt=0.0)
    line_search_steps: int = Field(ge=1)
    line_search_shrink: float = Field(gt=0.0, lt=1.0)
    sufficient_decrease: float = Field(ge=0.0, lt=1.0)


class OutputSchema(BaseModel):
    projected_coordinates: Differentiable[Array[(None, None, 2), None]]
    moments_before: Array[(None,), None]
    moments_after: Array[(None,), None]
    constraint_residual_before: Scalar
    constraint_residual: Scalar
    correction_rms: Scalar
    singular_values: Array[(None,), None]
    effective_rank: Integer
    rank_deficient: Bool
    iterations: Integer
    line_search_failures: Integer
    converged: Bool


def apply_jit(inputs: dict[str, Any]) -> dict[str, Any]:
    projected, diagnostics = project_ensemble(
        inputs["coordinates"],
        inputs["target_moments"],
        inputs["box"],
        PairBasis(inputs["basis_centers"], inputs["basis_widths"]),
        inputs["moment_scales"],
        ProjectionOptions(
            num_steps=inputs["num_steps"],
            ridge=inputs["ridge"],
            max_particle_step=inputs["max_particle_step"],
            tolerance=inputs["tolerance"],
            rank_tolerance=inputs["rank_tolerance"],
            line_search_steps=inputs["line_search_steps"],
            line_search_shrink=inputs["line_search_shrink"],
            sufficient_decrease=inputs["sufficient_decrease"],
        ),
    )
    diagnostics = jax.tree_util.tree_map(jax.lax.stop_gradient, diagnostics)
    return {"projected_coordinates": projected, **diagnostics}


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema.model_validate(jax_apply(apply_jit, inputs))


def abstract_eval(abstract_inputs) -> dict[str, ShapeDType]:
    dtype = abstract_inputs.coordinates.dtype
    scalar = ShapeDType(shape=(), dtype=dtype)
    integer = ShapeDType(shape=(), dtype="int32")
    boolean = ShapeDType(shape=(), dtype="bool")
    return {
        "projected_coordinates": abstract_inputs.coordinates,
        "moments_before": abstract_inputs.target_moments,
        "moments_after": abstract_inputs.target_moments,
        "constraint_residual_before": scalar,
        "constraint_residual": scalar,
        "correction_rms": scalar,
        "singular_values": abstract_inputs.target_moments,
        "effective_rank": integer,
        "rank_deficient": boolean,
        "iterations": integer,
        "line_search_failures": integer,
        "converged": boolean,
    }


def jacobian(inputs, jac_inputs, jac_outputs):
    return jax_jacobian(apply_jit, inputs, jac_inputs, jac_outputs)


def jacobian_vector_product(inputs, jvp_inputs, jvp_outputs, tangent_vector):
    return jax_jvp(apply_jit, inputs, jvp_inputs, jvp_outputs, tangent_vector)


def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    return jax_vjp(apply_jit, inputs, vjp_inputs, vjp_outputs, cotangent_vector)
