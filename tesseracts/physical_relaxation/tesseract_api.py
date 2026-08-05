"""Tesseract API for proximal physical relaxation of periodic particle ensembles."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from pydantic import BaseModel, Field
from tesseract_core.runtime import (
    Array,
    Differentiable,
    Float64,
    Int32,
    ShapeDType,
)
from tesseract_core.runtime.jax_recipes import jax_apply, jax_jacobian, jax_jvp, jax_vjp

from manybody_completion.relaxation import RelaxationOptions, relax_proximal

jax.config.update("jax_enable_x64", True)

Bool = Array[(), "bool"]


class InputSchema(BaseModel):
    """Inputs to one complete proximal relaxation solve."""

    coordinates: Differentiable[Array[(None, None, 2), Float64]] = Field(
        description="Initial periodic ensemble with shape (M, N, 2)."
    )
    box: Array[(2,), Float64] = Field(description="Periodic box side lengths.")
    # Numeric Field bounds cannot be used on differentiable scalars because
    # Tesseract's abstract schema replaces them with ShapeDType descriptors.
    r0: Differentiable[Float64] = Field(description="Positive repulsive distance threshold.")
    kappa: Differentiable[Float64] = Field(
        description="Positive softplus sharpness for the repulsive energy."
    )
    prox_strength: Differentiable[Float64] = Field(
        description="Positive proximal strength tau in the relaxation objective."
    )
    num_steps: int = Field(default=128, ge=1, description="Maximum solver iterations.")
    step_size: float = Field(default=2.5e-3, gt=0.0, description="Initial gradient step.")
    tolerance: float = Field(default=1e-7, ge=0.0, description="Stationarity tolerance.")
    max_update_norm: float = Field(
        default=0.04, gt=0.0, description="Maximum per-particle update norm."
    )
    line_search_steps: int = Field(default=12, ge=1, description="Backtracking trials.")
    line_search_shrink: float = Field(
        default=0.5, gt=0.0, lt=1.0, description="Backtracking shrink factor."
    )
    armijo_coefficient: float = Field(
        default=1e-4, gt=0.0, lt=1.0, description="Armijo sufficient-decrease coefficient."
    )


class OutputSchema(BaseModel):
    """Relaxed coordinates and forward-solve diagnostics."""

    relaxed_coordinates: Differentiable[Array[(None, None, 2), Float64]]
    physical_energy_before: Float64
    physical_energy_after: Float64
    proximal_objective_before: Float64
    proximal_objective_after: Float64
    max_force: Float64
    stationarity_norm: Float64
    prox_displacement: Float64
    minimum_pair_distance_before: Float64
    minimum_pair_distance_after: Float64
    iterations: Int32
    line_search_failures: Int32
    converged: Bool


def apply_jit(inputs: dict[str, Any]) -> dict[str, Any]:
    """JAX implementation of one coarse relaxation call."""
    options = RelaxationOptions(
        num_steps=inputs["num_steps"],
        step_size=inputs["step_size"],
        tolerance=inputs["tolerance"],
        max_update_norm=inputs["max_update_norm"],
        line_search_steps=inputs["line_search_steps"],
        line_search_shrink=inputs["line_search_shrink"],
        armijo_coefficient=inputs["armijo_coefficient"],
    )
    relaxed, diagnostics = relax_proximal(
        inputs["coordinates"],
        inputs["box"],
        inputs["r0"],
        inputs["kappa"],
        inputs["prox_strength"],
        options,
    )
    # Diagnostics are intentionally not differentiable API outputs.
    diagnostics = jax.tree_util.tree_map(jax.lax.stop_gradient, diagnostics)
    return {"relaxed_coordinates": relaxed, **diagnostics}


def apply(inputs: InputSchema) -> OutputSchema:
    """Run the complete proximal relaxation solve."""
    return OutputSchema.model_validate(jax_apply(apply_jit, inputs))


def abstract_eval(abstract_inputs) -> dict[str, ShapeDType]:
    """Return output shapes and dtypes without executing the solver."""
    coordinates = abstract_inputs.coordinates
    scalar_float = ShapeDType(shape=(), dtype="float64")
    scalar_int = ShapeDType(shape=(), dtype="int32")
    scalar_bool = ShapeDType(shape=(), dtype="bool")
    return {
        "relaxed_coordinates": coordinates,
        "physical_energy_before": scalar_float,
        "physical_energy_after": scalar_float,
        "proximal_objective_before": scalar_float,
        "proximal_objective_after": scalar_float,
        "max_force": scalar_float,
        "stationarity_norm": scalar_float,
        "prox_displacement": scalar_float,
        "minimum_pair_distance_before": scalar_float,
        "minimum_pair_distance_after": scalar_float,
        "iterations": scalar_int,
        "line_search_failures": scalar_int,
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
