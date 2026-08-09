"""Tesseract API for ensemble-level smooth pair-moment projection."""

from __future__ import annotations

from typing import Any

import jax
from pydantic import BaseModel, Field
from tesseract_core.runtime import (
    Array,
    Differentiable,
    Float64,
    Int32,
    ShapeDType,
)
from tesseract_core.runtime.jax_recipes import jax_apply, jax_jacobian, jax_jvp, jax_vjp

from manybody_completion.observables import PairBasis
from manybody_completion.projection import ProjectionOptions, project_ensemble_moments

jax.config.update("jax_enable_x64", True)

Bool = Array[(), "bool"]


class InputSchema(BaseModel):
    """Inputs to one complete ensemble moment-projection solve."""

    coordinates: Differentiable[Array[(None, None, 2), Float64]] = Field(
        description="Relaxed periodic ensemble with shape (M, N, 2)."
    )
    target_moments: Differentiable[Array[(None,), Float64]] = Field(
        description="Raw target ensemble pair moments with shape (R,)."
    )
    box: Array[(2,), Float64] = Field(description="Periodic box side lengths.")
    basis_centers: Array[(None,), Float64] = Field(
        description="Gaussian radial-basis centers with shape (R,)."
    )
    basis_widths: Array[(None,), Float64] = Field(
        description="Positive Gaussian radial-basis widths with shape (R,)."
    )
    moment_scales: Array[(None,), Float64] = Field(
        description="Positive scales used for diagonal coefficient whitening."
    )
    basis_mask: Array[(None,), Float64] = Field(
        description="Nonnegative constraint mask; zero explicitly prunes a basis coefficient."
    )
    num_steps: int = Field(default=16, ge=1, description="Maximum projection iterations.")
    tolerance: float = Field(default=1e-8, ge=0.0, description="Whitened residual tolerance.")
    kkt_tolerance: float = Field(
        default=1e-6, ge=0.0, description="RMS KKT stationarity tolerance."
    )
    ridge: float = Field(default=1e-8, gt=0.0, description="Moment-space ridge regularization.")
    svd_rcond: float = Field(default=1e-7, gt=0.0, description="Relative rank threshold.")
    damping: float = Field(default=1.0, gt=0.0, description="Linearized correction damping.")
    max_step_norm: float = Field(
        default=0.05, gt=0.0, description="Maximum RMS per-particle correction per iteration."
    )
    max_correction_norm: float = Field(
        default=0.25, gt=0.0, description="Maximum total RMS per-particle correction."
    )
    line_search_steps: int = Field(default=10, ge=1, description="Backtracking trials.")
    line_search_shrink: float = Field(
        default=0.5, gt=0.0, lt=1.0, description="Backtracking shrink factor."
    )
    sufficient_decrease: float = Field(
        default=1e-4,
        ge=0.0,
        lt=1.0,
        description="Relative merit decrease coefficient.",
    )
    merit_penalty: float = Field(
        default=1.0, gt=0.0, description="Exact-norm constraint penalty in the SQP merit function."
    )


class OutputSchema(BaseModel):
    """Projected coordinates and complete conditioning/solve diagnostics."""

    projected_coordinates: Differentiable[Array[(None, None, 2), Float64]]
    moments_before: Array[(None,), Float64]
    moments_after: Array[(None,), Float64]
    dual_variables: Array[(None,), Float64]
    singular_values: Array[(None,), Float64]
    correction_norm: Float64
    constraint_residual_before: Float64
    constraint_residual: Float64
    projection_objective: Float64
    merit_before: Float64
    merit_after: Float64
    kkt_stationarity_norm: Float64
    largest_singular_value: Float64
    smallest_singular_value: Float64
    condition_number: Float64
    effective_rank: Int32
    active_constraints: Int32
    rank_deficient: Bool
    iterations: Int32
    line_search_failures: Int32
    correction_clips: Int32
    converged: Bool


def apply_jit(inputs: dict[str, Any]) -> dict[str, Any]:
    """JAX implementation of one coarse ensemble-projection call."""
    basis = PairBasis(
        centers=inputs["basis_centers"],
        widths=inputs["basis_widths"],
    )
    options = ProjectionOptions(
        num_steps=inputs["num_steps"],
        tolerance=inputs["tolerance"],
        kkt_tolerance=inputs["kkt_tolerance"],
        ridge=inputs["ridge"],
        svd_rcond=inputs["svd_rcond"],
        damping=inputs["damping"],
        max_step_norm=inputs["max_step_norm"],
        max_correction_norm=inputs["max_correction_norm"],
        line_search_steps=inputs["line_search_steps"],
        line_search_shrink=inputs["line_search_shrink"],
        sufficient_decrease=inputs["sufficient_decrease"],
        merit_penalty=inputs["merit_penalty"],
    )
    projected, diagnostics = project_ensemble_moments(
        coordinates=inputs["coordinates"],
        target_moments=inputs["target_moments"],
        box=inputs["box"],
        basis=basis,
        moment_scales=inputs["moment_scales"],
        basis_mask=inputs["basis_mask"],
        options=options,
    )
    diagnostics = jax.tree_util.tree_map(jax.lax.stop_gradient, diagnostics)
    return {"projected_coordinates": projected, **diagnostics}


def apply(inputs: InputSchema) -> OutputSchema:
    """Run a complete ensemble moment-projection solve."""
    return OutputSchema.model_validate(jax_apply(apply_jit, inputs))


def abstract_eval(abstract_inputs) -> dict[str, ShapeDType]:
    """Return output shapes and dtypes without executing the projection."""
    scalar_float = ShapeDType(shape=(), dtype="float64")
    scalar_int = ShapeDType(shape=(), dtype="int32")
    scalar_bool = ShapeDType(shape=(), dtype="bool")
    moment_shape = abstract_inputs.target_moments
    return {
        "projected_coordinates": abstract_inputs.coordinates,
        "moments_before": moment_shape,
        "moments_after": moment_shape,
        "dual_variables": moment_shape,
        "singular_values": moment_shape,
        "correction_norm": scalar_float,
        "constraint_residual_before": scalar_float,
        "constraint_residual": scalar_float,
        "projection_objective": scalar_float,
        "merit_before": scalar_float,
        "merit_after": scalar_float,
        "kkt_stationarity_norm": scalar_float,
        "largest_singular_value": scalar_float,
        "smallest_singular_value": scalar_float,
        "condition_number": scalar_float,
        "effective_rank": scalar_int,
        "active_constraints": scalar_int,
        "rank_deficient": scalar_bool,
        "iterations": scalar_int,
        "line_search_failures": scalar_int,
        "correction_clips": scalar_int,
        "converged": scalar_bool,
    }


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    return jax_jacobian(apply_jit, inputs, jac_inputs, jac_outputs)


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    return jax_jvp(apply_jit, inputs, jvp_inputs, jvp_outputs, tangent_vector)


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    return jax_vjp(apply_jit, inputs, vjp_inputs, vjp_outputs, cotangent_vector)
