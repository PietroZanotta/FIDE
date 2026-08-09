"""Tesseract 1: learned reference velocity u_theta(t,x), dimension-generic.

The component accepts states as a rank-2 array ``(n_points, state_dim)``.  This
covers Example A with ``state_dim=1`` and Example B with ``state_dim=2`` while
keeping the same JAX-recipe/Tesseract surface.
"""
from __future__ import annotations

import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, Field

jax.config.update("jax_enable_x64", True)

TIME_FOURIER_FREQUENCIES = 4
REFERENCE_HIDDEN = (64, 64, 64)


def _features(t, x):
    x = jnp.asarray(x)
    batch = x.shape[:-1]
    tt = jnp.broadcast_to(jnp.asarray(t, dtype=x.dtype), batch)
    k = 2.0 ** jnp.arange(TIME_FOURIER_FREQUENCIES, dtype=x.dtype)
    angles = 2.0 * jnp.pi * tt[..., None] * k
    tf = jnp.concatenate([tt[..., None], jnp.sin(angles), jnp.cos(angles)], axis=-1)
    return jnp.concatenate([x, tf], axis=-1)


def _unflatten(flat, state_dim):
    dims = (state_dim + 1 + 2 * TIME_FOURIER_FREQUENCIES, *REFERENCE_HIDDEN, state_dim)
    params, off = [], 0
    for din, dout in zip(dims[:-1], dims[1:]):
        nw = din * dout
        W = flat[off:off + nw].reshape((din, dout)); off += nw
        b = flat[off:off + dout]; off += dout
        params.append((W, b))
    return tuple(params)


def _mlp(params, f):
    z = f
    for W, b in params[:-1]:
        z = jax.nn.silu(z @ W + b)
    W, b = params[-1]
    return z @ W + b


@jax.jit
def apply_jax(inputs: dict) -> dict:
    x = inputs["x"]
    p = _unflatten(inputs["velocity_params"], x.shape[-1])
    return {"velocity": _mlp(p, _features(inputs["t"], x))}


def apply_payload(inputs: dict) -> dict:
    inp = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in inputs.items()}
    return jax.tree.map(np.asarray, apply_jax(inp))


# Tesseract Core validates endpoint objects from the module's top-level AST.
# The runtime itself is only needed in the built container; keeping a small
# host fallback lets the parity tests import ``apply_jax`` without installing
# Tesseract Core's much larger ``runtime`` extra.
TESSERACT_RUNTIME_AVAILABLE = "TESSERACT_API_PATH" in os.environ
if TESSERACT_RUNTIME_AVAILABLE:
    from tesseract_core.runtime import Array, Differentiable, Float64
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


class InputSchema(BaseModel):
    x: Differentiable[Array[(None, None), Float64]] = Field(description="States, shape (n,state_dim).")
    t: Differentiable[Float64]
    velocity_params: Differentiable[Array[(None,), Float64]]


class OutputSchema(BaseModel):
    velocity: Differentiable[Array[(None, None), Float64]]


def _require_runtime():
    if not TESSERACT_RUNTIME_AVAILABLE:
        raise RuntimeError("Tesseract endpoints are available inside a built Tesseract container")


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
