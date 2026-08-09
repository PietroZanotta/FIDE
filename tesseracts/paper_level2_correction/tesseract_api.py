"""Tesseract recipe for the invariant MLP used in the paper level-2 study."""
from __future__ import annotations

import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel

jax.config.update("jax_enable_x64", True)


def _radial(configuration, centers, width, box_size):
    displacement = configuration[:, None, :] - configuration[None, :, :]
    displacement -= box_size * jnp.round(displacement / box_size)
    rows, cols = jnp.triu_indices(configuration.shape[0], 1)
    distances = jnp.sqrt(jnp.sum(displacement[rows, cols] ** 2, axis=-1) + 1e-10)
    return jnp.mean(
        jnp.exp(-0.5 * ((distances[:, None] - centers[None, :]) / width) ** 2),
        axis=0,
    )


def _potential(configuration, time, inputs):
    descriptors = _radial(
        configuration, inputs["descriptor_centers"], inputs["radial_width"], inputs["box_size"]
    )
    time_features = jnp.asarray(
        [time, jnp.sin(2.0 * jnp.pi * time), jnp.cos(2.0 * jnp.pi * time)]
    )
    features = jnp.concatenate([descriptors, time_features])
    hidden = jax.nn.silu(features @ inputs["w1"] + inputs["b1"])
    hidden = jax.nn.silu(hidden @ inputs["w2"] + inputs["b2"])
    return (hidden @ inputs["w3"] + inputs["b3"])[0]


def _one(configuration, time, inputs):
    potential = _potential(configuration, time, inputs)
    correction = -jax.grad(lambda state: _potential(state, time, inputs))(configuration)
    descriptors = _radial(
        configuration, inputs["descriptor_centers"], inputs["radial_width"], inputs["box_size"]
    )
    observables = _radial(
        configuration, inputs["observable_centers"], inputs["radial_width"], inputs["box_size"]
    )
    centered = configuration - jnp.mean(configuration, axis=0, keepdims=True)
    angles = jnp.arctan2(centered[:, 1], centered[:, 0])
    q4 = jnp.sqrt(
        jnp.mean(jnp.cos(4.0 * angles)) ** 2
        + jnp.mean(jnp.sin(4.0 * angles)) ** 2
        + 1e-16
    )
    return potential, correction, descriptors, observables, q4


@jax.jit
def apply_jax(inputs: dict) -> dict:
    values = jax.vmap(
        lambda time, states: jax.vmap(lambda state: _one(state, time, inputs))(states)
    )(inputs["times"], inputs["states"])
    return {
        "potential": values[0],
        "correction": values[1],
        "descriptors": values[2],
        "observables": values[3],
        "q4": values[4],
        "particle_count": jnp.asarray(inputs["states"].shape[2], dtype=jnp.int32),
        "state_dimension": jnp.asarray(inputs["states"].shape[2] * 2, dtype=jnp.int32),
    }


def apply_payload(inputs: dict) -> dict:
    converted = {key: jnp.asarray(value, dtype=jnp.float64) for key, value in inputs.items()}
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
    times: Differentiable[Array[(None,), Float64]]
    states: Differentiable[Array[(None, None, None, 2), Float64]]
    descriptor_centers: Differentiable[Array[(None,), Float64]]
    observable_centers: Differentiable[Array[(3,), Float64]]
    radial_width: Differentiable[Float64]
    box_size: Differentiable[Float64]
    w1: Differentiable[Array[(None, None), Float64]]
    b1: Differentiable[Array[(None,), Float64]]
    w2: Differentiable[Array[(None, None), Float64]]
    b2: Differentiable[Array[(None,), Float64]]
    w3: Differentiable[Array[(None, 1), Float64]]
    b3: Differentiable[Array[(1,), Float64]]


class OutputSchema(BaseModel):
    potential: Differentiable[Array[(None, None), Float64]]
    correction: Differentiable[Array[(None, None, None, 2), Float64]]
    descriptors: Differentiable[Array[(None, None, None), Float64]]
    observables: Differentiable[Array[(None, None, 3), Float64]]
    q4: Differentiable[Array[(None, None), Float64]]
    particle_count: Int32
    state_dimension: Int32


def _require_runtime():
    if not TESSERACT_RUNTIME_AVAILABLE:
        raise RuntimeError("Tesseract endpoints are available inside a built container")


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
