"""Tesseract-native end-to-end optimizer for the Stage-4 fiber objective.

The complete objective, implicit I-projection derivative, Adam updates, and
selection-bank checkpoint choice execute inside this component.  The host sees
only frozen inputs and optimizer outputs; it never differentiates through RPC.
"""
from __future__ import annotations

import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel

import stage4_fiber_design as stage4
import stage4b_fiber_design_confirmatory as stage4b


jax.config.update("jax_enable_x64", True)
MAX_STEPS = 40
CANDIDATE_INTERVAL = 5


def _objective(theta, inputs, bank, stopped):
    raw = inputs["schedule_raw"]
    common = inputs["common_mean"]
    basis = inputs["basis"]

    def full(_):
        return stage4b.objective(raw, bank, common, theta, basis, False)

    def stop(_):
        return stage4b.objective(raw, bank, common, theta, basis, True)

    return jax.lax.cond(stopped, stop, full, operand=None)


def _bank(inputs, prefix):
    return (
        inputs[f"{prefix}_minus"],
        inputs[f"{prefix}_plus"],
        inputs[f"{prefix}_noise"],
    )


@jax.jit
def apply_jax(inputs: dict) -> dict:
    adaptation = _bank(inputs, "adaptation")
    selection = _bank(inputs, "selection")
    stopped = inputs["stopped"] != 0
    active_steps = jnp.clip(inputs["optimizer_steps"], 0, MAX_STEPS)

    train = lambda theta: _objective(theta, inputs, adaptation, stopped)
    select = lambda theta: _objective(theta, inputs, selection, stopped)
    initial_theta = inputs["theta0"]
    initial = (
        initial_theta,
        jnp.zeros_like(initial_theta),
        jnp.zeros_like(initial_theta),
    )

    def step(carry, index):
        count = index + 1
        enabled = count <= active_steps

        def active(carry):
            theta, first, second = carry
            value, gradient = jax.value_and_grad(train)(theta)
            norm = jnp.linalg.norm(gradient)
            clipped = gradient * jnp.minimum(
                1.0, 5.0 / jnp.maximum(norm, 1e-12)
            )
            next_first = 0.9 * first + 0.1 * clipped
            next_second = 0.999 * second + 0.001 * clipped * clipped
            first_hat = next_first / (1.0 - 0.9**count)
            second_hat = next_second / (1.0 - 0.999**count)
            next_theta = theta - stage4.LEARNING_RATE * first_hat / (
                jnp.sqrt(second_hat) + 1e-8
            )
            next_theta = stage4.canonical_rows(next_theta)
            updated = (next_theta, next_first, next_second)
            return updated, (value, norm, next_theta)

        def idle(carry):
            theta, _, _ = carry
            return carry, (jnp.asarray(0.0), jnp.asarray(0.0), theta)

        return jax.lax.cond(enabled, active, idle, carry)

    _, trace = jax.lax.scan(step, initial, jnp.arange(MAX_STEPS))
    checkpoint_indices = jnp.arange(
        CANDIDATE_INTERVAL - 1, MAX_STEPS, CANDIDATE_INTERVAL
    )
    candidates = jnp.concatenate(
        [initial_theta[None, ...], trace[2][checkpoint_indices]], axis=0
    )
    candidate_steps = jnp.arange(
        0, MAX_STEPS + 1, CANDIDATE_INTERVAL, dtype=jnp.int32
    )
    selection_objectives = jax.vmap(select)(candidates)
    eligible = candidate_steps <= active_steps
    eligible_values = jnp.where(
        eligible & jnp.isfinite(selection_objectives),
        selection_objectives,
        jnp.inf,
    )
    selected_index = jnp.argmin(eligible_values)
    selected = candidates[selected_index]
    return {
        "selected_theta": selected,
        "selected_candidate_index": selected_index.astype(jnp.int32),
        "selected_step": candidate_steps[selected_index],
        "candidate_steps": candidate_steps,
        "candidate_thetas": candidates,
        "selection_objectives": selection_objectives,
        "adaptation_objectives": trace[0],
        "gradient_norms": trace[1],
        "initial_adaptation_objective": train(initial_theta),
        "selected_adaptation_objective": train(selected),
        "initial_selection_objective": selection_objectives[0],
        "selected_selection_objective": selection_objectives[selected_index],
    }


def apply_payload(inputs: dict) -> dict:
    converted = {
        key: jnp.asarray(value, dtype=(jnp.int32 if key in {"stopped", "optimizer_steps"} else jnp.float64))
        for key, value in inputs.items()
    }
    return jax.tree.map(np.asarray, apply_jax(converted))


TESSERACT_RUNTIME_AVAILABLE = "TESSERACT_API_PATH" in os.environ
if TESSERACT_RUNTIME_AVAILABLE:
    from tesseract_core.runtime import Array, Differentiable, Float64, Int32
    from tesseract_core.runtime.jax_recipes import (
        jax_abstract_eval, jax_apply, jax_jacobian, jax_jvp, jax_vjp,
    )
else:
    class _AnyAnnotation:
        @classmethod
        def __class_getitem__(cls, item):
            return Any
    Array = Differentiable = _AnyAnnotation
    Float64 = float
    Int32 = int


class InputSchema(BaseModel):
    schedule_raw: Differentiable[Array[(3,), Float64]]
    common_mean: Array[(11,), Float64]
    theta0: Array[(3, 10), Float64]
    basis: Array[(11, 10), Float64]
    adaptation_minus: Array[(None, None, None, 2), Float64]
    adaptation_plus: Array[(None, None, None, 2), Float64]
    adaptation_noise: Array[(None, None, None, 2), Float64]
    selection_minus: Array[(None, None, None, 2), Float64]
    selection_plus: Array[(None, None, None, 2), Float64]
    selection_noise: Array[(None, None, None, 2), Float64]
    stopped: Int32
    optimizer_steps: Int32


class OutputSchema(BaseModel):
    selected_theta: Differentiable[Array[(3, 10), Float64]]
    selected_candidate_index: Int32
    selected_step: Int32
    candidate_steps: Array[(9,), Int32]
    candidate_thetas: Array[(9, 3, 10), Float64]
    selection_objectives: Array[(9,), Float64]
    adaptation_objectives: Array[(40,), Float64]
    gradient_norms: Array[(40,), Float64]
    initial_adaptation_objective: Float64
    selected_adaptation_objective: Float64
    initial_selection_objective: Float64
    selected_selection_objective: Float64


def _require_runtime():
    if not TESSERACT_RUNTIME_AVAILABLE:
        raise RuntimeError("Tesseract endpoints require a built component")


def apply(inputs: InputSchema) -> OutputSchema:
    _require_runtime()
    return OutputSchema(**jax_apply(apply_jax, inputs))


def abstract_eval(abstract_inputs):
    _require_runtime()
    return jax_abstract_eval(apply_jax, abstract_inputs)


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    _require_runtime()
    return jax_jacobian(apply_jax, inputs, jac_inputs, jac_outputs)


def jacobian_vector_product(inputs: InputSchema, jvp_inputs: set[str], jvp_outputs: set[str], tangent_vector):
    _require_runtime()
    return jax_jvp(apply_jax, inputs, jvp_inputs, jvp_outputs, tangent_vector)


def vector_jacobian_product(inputs: InputSchema, vjp_inputs: set[str], vjp_outputs: set[str], cotangent_vector):
    _require_runtime()
    return jax_vjp(apply_jax, inputs, vjp_inputs, vjp_outputs, cotangent_vector)
