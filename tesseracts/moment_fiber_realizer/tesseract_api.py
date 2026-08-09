"""Tesseract 2: empirical I-projection + fiber forcing + Deep-Ritz correction.

The observable values Phi(x_i) and J_Phi(x_i)u_i remain explicit inputs.  State
arrays are rank-2 ``(n_points,state_dim)``, so the same component covers 1D and
2D examples without hard-coding an observable family.
"""
from __future__ import annotations

import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel

jax.config.update("jax_enable_x64", True)

TIME_FOURIER_FREQUENCIES = 4
RITZ_HIDDEN = (96, 96, 96, 96)
RCOND = 1e-9
DAMPING = 1e-10
NEWTON_ITERS = 20


def _features(t, x):
    x = jnp.asarray(x)
    batch = x.shape[:-1]
    tt = jnp.broadcast_to(jnp.asarray(t, dtype=x.dtype), batch)
    k = 2.0 ** jnp.arange(TIME_FOURIER_FREQUENCIES, dtype=x.dtype)
    angles = 2.0 * jnp.pi * tt[..., None] * k
    tf = jnp.concatenate([tt[..., None], jnp.sin(angles), jnp.cos(angles)], axis=-1)
    return jnp.concatenate([x, tf], axis=-1)


def _unflatten(flat, state_dim):
    dims = (state_dim + 1 + 2 * TIME_FOURIER_FREQUENCIES, *RITZ_HIDDEN, 1)
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


def _potential_single(params, t, x):
    return _mlp(params, _features(t, x[None, :]))[0, 0]


def _potential_grad(params, t, x):
    return jax.grad(lambda xx: jnp.sum(_mlp(params, _features(t, xx))[..., 0]))(x)


def _stable_solve(cov, rhs):
    diag = jnp.maximum(jnp.diag(cov), 1e-30)
    scale = jnp.sqrt(diag)
    cw = cov / (scale[:, None] * scale[None, :])
    cw = 0.5 * (cw + cw.T) + DAMPING * jnp.eye(cov.shape[0], dtype=cov.dtype)
    vals, vecs = jnp.linalg.eigh(cw)
    vmax = jnp.maximum(jnp.max(vals), 1e-30)
    keep = vals > RCOND * vmax
    inv = jnp.where(keep, 1.0 / jnp.maximum(vals, 1e-30), 0.0)
    solw = vecs @ (inv * (vecs.T @ (rhs / scale)))
    sol = solw / scale
    rank = jnp.sum(keep.astype(jnp.int32))
    vmin = jnp.min(jnp.where(keep, vals, jnp.inf))
    cond = jnp.where(rank > 0, vmax / vmin, jnp.inf)
    return sol, rank, cond


def _tilt(lam, log_base_weights, ph):
    w = jax.nn.softmax(log_base_weights + ph @ lam)
    moments = w @ ph
    centered = ph - moments
    cov = (centered.T * w) @ centered
    return w, moments, cov


def _calibrate_primal(log_base_weights, ph, target):
    lam0 = jnp.zeros(target.shape[0], dtype=ph.dtype)
    def body(_, lam):
        _, moments, cov = _tilt(lam, log_base_weights, ph)
        step, _, _ = _stable_solve(cov, moments - target)
        norm = jnp.linalg.norm(step)
        step = step * jnp.minimum(1.0, 2.0 / jnp.maximum(norm, 1e-30))
        return lam - step
    return jax.lax.fori_loop(0, NEWTON_ITERS, body, lam0)


@jax.custom_jvp
def _calibrate(log_base_weights, ph, target):
    return _calibrate_primal(log_base_weights, ph, target)


@_calibrate.defjvp
def _calibrate_jvp(primals, tangents):
    log_base_weights, ph, target = primals
    dlog_base_weights, dph, dtarget = tangents
    lam = _calibrate_primal(log_base_weights, ph, target)
    w, moments, cov = _tilt(lam, log_base_weights, ph)
    centered = ph - moments
    dlogit = dlog_base_weights + jnp.sum(dph * lam[None, :], axis=-1)
    dF = w @ dph + jnp.sum(w[:, None] * centered * dlogit[:, None], axis=0) - dtarget
    dlam, _, _ = _stable_solve(cov, -dF)
    return lam, dlam


@jax.jit
def apply_jax(inputs: dict) -> dict:
    x = inputs["x"]
    u = inputs["velocity"]
    ph = inputs["phi_values"]
    jpu = inputs["jphi_u"]
    target = inputs["target"]
    logbw = inputs["log_base_weights"]
    t = inputs["t"]
    ppsi = _unflatten(inputs["potential_params"], x.shape[-1])

    lam = _calibrate(logbw, ph, target)
    w, moments, cov = _tilt(lam, logbw, ph)
    em = w @ jpu
    scalar = jpu @ lam
    cov_term = jnp.sum(w[:, None] * (ph - target) * scalar[:, None], axis=0)
    lambda_dot, rank, cond = _stable_solve(cov, -em - cov_term)
    h = (ph - target) @ lambda_dot + (jpu - em) @ lam
    h = h - w @ h
    correction = -_potential_grad(ppsi, t, x)
    velocity = u + correction
    return {
        "lambda_value": lam,
        "projected_weights": w,
        "moments": moments,
        "covariance": cov,
        "lambda_dot": lambda_dot,
        "forcing": h,
        "correction": correction,
        "velocity": velocity,
        "ess_fraction": 1.0 / (x.shape[0] * jnp.sum(w * w)),
        "calibration_residual": jnp.linalg.norm(moments - target),
        "covariance_rank": rank,
        "covariance_condition": cond,
    }


def apply_payload(inputs: dict) -> dict:
    inp = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in inputs.items()}
    return jax.tree.map(np.asarray, apply_jax(inp))


# Tesseract Core validates endpoint objects from the module's top-level AST.
# The runtime itself is only needed in the built container; keeping a small
# host fallback lets the parity tests import ``apply_jax`` without installing
# Tesseract Core's much larger ``runtime`` extra.
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
    x: Differentiable[Array[(None, None), Float64]]
    t: Differentiable[Float64]
    velocity: Differentiable[Array[(None, None), Float64]]
    phi_values: Differentiable[Array[(None, None), Float64]]
    jphi_u: Differentiable[Array[(None, None), Float64]]
    target: Differentiable[Array[(None,), Float64]]
    log_base_weights: Differentiable[Array[(None,), Float64]]
    potential_params: Differentiable[Array[(None,), Float64]]


class OutputSchema(BaseModel):
    lambda_value: Differentiable[Array[(None,), Float64]]
    projected_weights: Differentiable[Array[(None,), Float64]]
    moments: Differentiable[Array[(None,), Float64]]
    covariance: Differentiable[Array[(None, None), Float64]]
    lambda_dot: Differentiable[Array[(None,), Float64]]
    forcing: Differentiable[Array[(None,), Float64]]
    correction: Differentiable[Array[(None, None), Float64]]
    velocity: Differentiable[Array[(None, None), Float64]]
    ess_fraction: Differentiable[Float64]
    calibration_residual: Differentiable[Float64]
    covariance_rank: Int32
    covariance_condition: Differentiable[Float64]


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
