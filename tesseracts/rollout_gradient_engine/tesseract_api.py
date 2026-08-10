"""Tesseract-native end-to-end optimizer for Stage-3 rollout gradients.

Control values are 0=full rollout, 1=stopped-state, and 2=scalar.  The entire
Heun rollout, law-valued loss, reverse-mode gradient, Adam loop, and independent
selection-bank checkpoint choice execute inside the Tesseract.
"""
from __future__ import annotations

import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel

import level2_paper_study as paper
import stage3_rollout_adaptation as stage3
import stage3b_confirmatory as stage3b


jax.config.update("jax_enable_x64", True)
MAX_STEPS = 40
CANDIDATE_INTERVAL = 5


def _model(inputs):
    return paper.MLP(
        inputs["model_w1"], inputs["model_b1"],
        inputs["model_w2"], inputs["model_b2"],
        inputs["model_w3"], inputs["model_b3"],
    )


def _generation(inputs, prefix):
    return (
        inputs[f"{prefix}_minus"],
        inputs[f"{prefix}_plus"],
        inputs[f"{prefix}_noise"],
    )


def _oracle(inputs, prefix):
    return inputs[f"{prefix}_oracle_features"], inputs[f"{prefix}_oracle_weights"]


def _loss(parameters, inputs, prefix):
    model = _model(inputs)
    generation = _generation(inputs, prefix)
    oracle = _oracle(inputs, prefix)
    gate = inputs["gate"]
    raw = inputs["schedule_raw"]
    control = inputs["control"]

    def full(_):
        return stage3b.rollout_loss(
            parameters, "full", model, gate, raw, generation, oracle
        )

    def stopped(_):
        return stage3b.rollout_loss(
            parameters, "stopped_state", model, gate, raw, generation, oracle
        )

    def scalar(_):
        masked = jnp.asarray([parameters[0], 0.0, 0.0], dtype=parameters.dtype)
        return stage3b.rollout_loss(
            masked, "full", model, gate, raw, generation, oracle
        )

    return jax.lax.switch(control, (full, stopped, scalar), operand=None)


@jax.jit
def apply_jax(inputs: dict) -> dict:
    active_steps = jnp.clip(inputs["optimizer_steps"], 0, MAX_STEPS)
    train = lambda parameters: _loss(parameters, inputs, "adaptation")
    select = lambda parameters: _loss(parameters, inputs, "selection")
    initial_parameters = jnp.zeros(3, dtype=jnp.float64)
    initial = (
        initial_parameters,
        jnp.zeros_like(initial_parameters),
        jnp.zeros_like(initial_parameters),
    )

    def step(carry, index):
        count = index + 1
        enabled = count <= active_steps

        def active(carry):
            parameters, first, second = carry
            value, gradient = jax.value_and_grad(train)(parameters)
            norm = jnp.linalg.norm(gradient)
            clipped = gradient * jnp.minimum(
                1.0, 5.0 / jnp.maximum(norm, 1e-12)
            )
            next_first = 0.9 * first + 0.1 * clipped
            next_second = 0.999 * second + 0.001 * clipped * clipped
            first_hat = next_first / (1.0 - 0.9**count)
            second_hat = next_second / (1.0 - 0.999**count)
            next_parameters = parameters - stage3.LEARNING_RATE * first_hat / (
                jnp.sqrt(second_hat) + 1e-8
            )
            updated = (next_parameters, next_first, next_second)
            return updated, (value, norm, next_parameters)

        def idle(carry):
            parameters, _, _ = carry
            return carry, (jnp.asarray(0.0), jnp.asarray(0.0), parameters)

        return jax.lax.cond(enabled, active, idle, carry)

    _, trace = jax.lax.scan(step, initial, jnp.arange(MAX_STEPS))
    checkpoint_indices = jnp.arange(
        CANDIDATE_INTERVAL - 1, MAX_STEPS, CANDIDATE_INTERVAL
    )
    candidates = jnp.concatenate(
        [initial_parameters[None, :], trace[2][checkpoint_indices]], axis=0
    )
    candidate_steps = jnp.arange(
        0, MAX_STEPS + 1, CANDIDATE_INTERVAL, dtype=jnp.int32
    )
    selection_losses = jax.vmap(select)(candidates)
    eligible = candidate_steps <= active_steps
    eligible_losses = jnp.where(
        eligible & jnp.isfinite(selection_losses), selection_losses, jnp.inf
    )
    selected_index = jnp.argmin(eligible_losses)
    selected = candidates[selected_index]
    return {
        "selected_parameters": selected,
        "selected_candidate_index": selected_index.astype(jnp.int32),
        "selected_step": candidate_steps[selected_index],
        "candidate_steps": candidate_steps,
        "candidate_parameters": candidates,
        "selection_losses": selection_losses,
        "adaptation_losses": trace[0],
        "gradient_norms": trace[1],
        "parameter_trace": trace[2],
        "initial_adaptation_loss": train(initial_parameters),
        "selected_adaptation_loss": train(selected),
        "initial_selection_loss": selection_losses[0],
        "selected_selection_loss": selection_losses[selected_index],
    }


def apply_payload(inputs: dict) -> dict:
    converted = {
        key: jnp.asarray(value, dtype=(jnp.int32 if key in {"control", "optimizer_steps"} else jnp.float64))
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
    model_w1: Array[(None, None), Float64]
    model_b1: Array[(None,), Float64]
    model_w2: Array[(None, None), Float64]
    model_b2: Array[(None,), Float64]
    model_w3: Array[(None, None), Float64]
    model_b3: Array[(None,), Float64]
    gate: Float64
    schedule_raw: Differentiable[Array[(3,), Float64]]
    adaptation_minus: Array[(1, None, None, 2), Float64]
    adaptation_plus: Array[(1, None, None, 2), Float64]
    adaptation_noise: Array[(1, None, None, 2), Float64]
    adaptation_oracle_features: Array[(4, None, 3), Float64]
    adaptation_oracle_weights: Array[(4, None), Float64]
    selection_minus: Array[(1, None, None, 2), Float64]
    selection_plus: Array[(1, None, None, 2), Float64]
    selection_noise: Array[(1, None, None, 2), Float64]
    selection_oracle_features: Array[(4, None, 3), Float64]
    selection_oracle_weights: Array[(4, None), Float64]
    control: Int32
    optimizer_steps: Int32


class OutputSchema(BaseModel):
    selected_parameters: Differentiable[Array[(3,), Float64]]
    selected_candidate_index: Int32
    selected_step: Int32
    candidate_steps: Array[(9,), Int32]
    candidate_parameters: Array[(9, 3), Float64]
    selection_losses: Array[(9,), Float64]
    adaptation_losses: Array[(40,), Float64]
    gradient_norms: Array[(40,), Float64]
    parameter_trace: Array[(40, 3), Float64]
    initial_adaptation_loss: Float64
    selected_adaptation_loss: Float64
    initial_selection_loss: Float64
    selected_selection_loss: Float64


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
