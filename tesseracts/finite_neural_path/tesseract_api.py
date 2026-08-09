"""Finite-bank, multi-schedule level-2 MFSI experiment.

The correction potential is a one-hidden-layer neural random-feature model.
Its output weights are learned by the empirical Deep-Ritz normal equations;
hidden weights are fixed before schedule optimization.  Schedule gradients pass
through both this neural solve and an implicit custom JVP for I-projection.
"""
from __future__ import annotations

import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel

jax.config.update("jax_enable_x64", True)

NEWTON_ITERS = 24
OPTIMIZATION_STEPS = 84
CALIBRATION_RIDGE = 1e-10
RITZ_RIDGE = 1e-2


def _tilt(lam, observables):
    weights = jax.nn.softmax(observables @ lam)
    moments = weights @ observables
    centered = observables - moments
    covariance = (centered.T * weights) @ centered
    return weights, moments, covariance


def _solve(matrix, rhs, ridge):
    matrix = 0.5 * (matrix + matrix.T)
    return jnp.linalg.solve(
        matrix + ridge * jnp.eye(matrix.shape[0], dtype=matrix.dtype), rhs
    )


def _calibrate_primal(observables, target):
    initial = jnp.zeros(target.shape[0], dtype=observables.dtype)

    def body(_, lam):
        _, moments, covariance = _tilt(lam, observables)
        step = _solve(covariance, moments - target, CALIBRATION_RIDGE)
        norm = jnp.linalg.norm(step)
        return lam - step * jnp.minimum(1.0, 2.0 / jnp.maximum(norm, 1e-30))

    return jax.lax.fori_loop(0, NEWTON_ITERS, body, initial)


@jax.custom_jvp
def _calibrate(observables, target):
    return _calibrate_primal(observables, target)


@_calibrate.defjvp
def _calibrate_jvp(primals, tangents):
    observables, target = primals
    dobservables, dtarget = tangents
    lam = _calibrate_primal(observables, target)
    weights, moments, covariance = _tilt(lam, observables)
    centered = observables - moments
    dlogit = jnp.sum(dobservables * lam[None, :], axis=-1)
    fixed_lambda_dF = (
        weights @ dobservables
        + jnp.sum(weights[:, None] * centered * dlogit[:, None], axis=0)
        - dtarget
    )
    dlam = _solve(covariance, -fixed_lambda_dF, CALIBRATION_RIDGE)
    return lam, dlam


def _schedule(raw, t):
    features = jnp.asarray([1.0, jnp.cos(2.0 * jnp.pi * t), jnp.sin(2.0 * jnp.pi * t)])
    return jax.nn.softplus(raw @ features)


def _gamma(raw, t):
    return jnp.sqrt(jnp.maximum(2.0 * t * (1.0 - t), 1e-12)) * _schedule(raw, t)


def _bridge(raw, t, x_minus, x_plus, noise):
    gamma = _gamma(raw, t)
    gamma_dot = jax.grad(lambda time: _gamma(raw, time))(t)
    state = (1.0 - t) * x_minus + t * x_plus + gamma * noise
    velocity = x_plus - x_minus + gamma_dot * noise
    return state, velocity


def _observables(state):
    x, y = state[:, 0], state[:, 1]
    return jnp.stack([x, y, x * x, x * y, y * y], axis=-1)


def _jphi_velocity(state, velocity):
    x, y = state[:, 0], state[:, 1]
    ux, uy = velocity[:, 0], velocity[:, 1]
    return jnp.stack(
        [ux, uy, 2.0 * x * ux, y * ux + x * uy, 2.0 * y * uy], axis=-1
    )


def _fiber_state(raw, t, x_minus, x_plus, noise, target):
    state, velocity = _bridge(raw, t, x_minus, x_plus, noise)
    observables = _observables(state)
    lam = _calibrate(observables, target)
    weights, moments, covariance = _tilt(lam, observables)
    jphi_u = _jphi_velocity(state, velocity)
    expected_jphi_u = weights @ jphi_u
    scalar = jphi_u @ lam
    covariance_term = jnp.sum(
        weights[:, None] * (observables - target) * scalar[:, None], axis=0
    )
    lambda_dot = _solve(
        covariance, -expected_jphi_u - covariance_term, CALIBRATION_RIDGE
    )
    forcing = (
        (observables - target) @ lambda_dot
        + (jphi_u - expected_jphi_u) @ lam
    )
    forcing = forcing - weights @ forcing
    ess = 1.0 / (state.shape[0] * jnp.sum(weights * weights))
    residual = jnp.linalg.norm(moments - target)
    angles = jnp.arctan2(state[:, 1], state[:, 0])
    angular4 = jnp.sqrt(
        (weights @ jnp.cos(4.0 * angles)) ** 2
        + (weights @ jnp.sin(4.0 * angles)) ** 2
        + 1e-16
    )
    return state, weights, moments, lam, forcing, ess, residual, angular4


def _neural_features(state, t, feature_weight, feature_bias, feature_time):
    preactivation = state @ feature_weight.T + feature_bias + t * feature_time
    features = jnp.tanh(preactivation)
    derivative = 1.0 - features * features
    gradients = derivative[:, :, None] * feature_weight[None, :, :]
    return features, gradients


def _fit_time(raw, t, x_minus, x_plus, noise, target, inputs):
    state, weights, moments, lam, forcing, ess, residual, angular4 = _fiber_state(
        raw, t, x_minus, x_plus, noise, target
    )
    features, gradients = _neural_features(
        state, t, inputs["feature_weight"], inputs["feature_bias"], inputs["feature_time"]
    )
    gram = jnp.einsum("m,mkd,mld->kl", weights, gradients, gradients)
    rhs = jnp.einsum("m,mk,m->k", weights, features, forcing)
    coefficients = _solve(gram, rhs, RITZ_RIDGE)
    correction = -jnp.einsum("mkd,k->md", gradients, coefficients)
    energy = jnp.sum(weights * jnp.sum(correction * correction, axis=-1))
    train_residual = jnp.linalg.norm(gram @ coefficients - rhs) / jnp.maximum(
        jnp.linalg.norm(rhs), 1e-10
    )
    return (
        coefficients, state, weights, moments, lam, forcing, correction,
        energy, ess, residual, angular4, train_residual,
    )


def _validate_time(raw, coefficients, t, x_minus, x_plus, noise, target, inputs):
    state, weights, moments, lam, forcing, ess, residual, angular4 = _fiber_state(
        raw, t, x_minus, x_plus, noise, target
    )
    features, gradients = _neural_features(
        state, t, inputs["feature_weight"], inputs["feature_bias"], inputs["feature_time"]
    )
    correction = -jnp.einsum("mkd,k->md", gradients, coefficients)
    energy = jnp.sum(weights * jnp.sum(correction * correction, axis=-1))
    potential = features @ coefficients
    ritz_loss = 0.5 * energy - jnp.sum(weights * potential * forcing)
    ritz_gain_over_zero = -ritz_loss

    test_features, test_gradients = _neural_features(
        state,
        t,
        inputs["test_feature_weight"],
        inputs["test_feature_bias"],
        inputs["test_feature_time"],
    )
    lhs = -jnp.einsum("m,mjd,md->j", weights, test_gradients, correction)
    rhs = jnp.einsum("m,mj,m->j", weights, test_features, forcing)
    weak_residual = jnp.linalg.norm(lhs - rhs) / jnp.maximum(jnp.linalg.norm(rhs), 1e-10)
    zero_residual = jnp.linalg.norm(rhs) / jnp.maximum(jnp.linalg.norm(rhs), 1e-10)
    forcing_power = jnp.sum(weights * forcing * forcing)
    return (
        state, weights, moments, lam, correction, energy, ess, residual,
        angular4, weak_residual, zero_residual, ritz_gain_over_zero, forcing_power,
    )


def _fit_path(raw, inputs):
    return jax.vmap(
        lambda t, xm, xp, z: _fit_time(raw, t, xm, xp, z, inputs["target"], inputs)
    )(
        inputs["times"], inputs["train_minus"], inputs["train_plus"], inputs["train_noise"]
    )


def _validate_path(raw, coefficients, inputs):
    return jax.vmap(
        lambda c, t, xm, xp, z: _validate_time(
            raw, c, t, xm, xp, z, inputs["target"], inputs
        )
    )(
        coefficients,
        inputs["times"],
        inputs["validation_minus"],
        inputs["validation_plus"],
        inputs["validation_noise"],
    )


def _objective(raw, inputs):
    fitted = _fit_path(raw, inputs)
    energy, ess = fitted[7], fitted[8]
    energy_integral = jnp.trapezoid(energy, inputs["times"])
    violation = jax.nn.relu(inputs["ess_floor"] - ess)
    penalty = inputs["ess_penalty"] * jnp.trapezoid(violation**2, inputs["times"])
    schedule_curvature = jnp.mean((raw[1:]) ** 2) * inputs["schedule_regularization"]
    return energy_integral + penalty + schedule_curvature


def _optimize(initial_raw, inputs):
    initial = (
        initial_raw,
        jnp.zeros_like(initial_raw),
        jnp.zeros_like(initial_raw),
        jnp.asarray(0, dtype=jnp.int32),
    )

    def step(carry, _):
        raw, first, second, count = carry
        value, gradient = jax.value_and_grad(_objective)(raw, inputs)
        norm = jnp.linalg.norm(gradient)
        gradient = gradient * jnp.minimum(1.0, 5.0 / jnp.maximum(norm, 1e-12))
        count = count + 1
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient * gradient
        mhat = first / (1.0 - 0.9**count)
        vhat = second / (1.0 - 0.999**count)
        raw = raw - inputs["learning_rate"] * mhat / (jnp.sqrt(vhat) + 1e-8)
        return (raw, first, second, count), jnp.concatenate(
            [jnp.asarray([value, norm]), raw]
        )

    final, trace = jax.lax.scan(step, initial, xs=None, length=OPTIMIZATION_STEPS)
    return final[0], trace


@jax.jit
def apply_jax(inputs: dict) -> dict:
    initial_raw = inputs["initial_raw"]
    optimized_raw, trace = _optimize(initial_raw, inputs)
    initial_fit = _fit_path(initial_raw, inputs)
    optimized_fit = _fit_path(optimized_raw, inputs)
    initial_validation = _validate_path(initial_raw, initial_fit[0], inputs)
    optimized_validation = _validate_path(optimized_raw, optimized_fit[0], inputs)

    initial_gradient = jax.grad(_objective)(initial_raw, inputs)
    direction = inputs["gradient_check_direction"]
    direction = direction / jnp.linalg.norm(direction)
    epsilon = inputs["finite_difference_epsilon"]
    implicit_directional = initial_gradient @ direction
    finite_directional = (
        _objective(initial_raw + epsilon * direction, inputs)
        - _objective(initial_raw - epsilon * direction, inputs)
    ) / (2.0 * epsilon)
    gradient_error = jnp.abs(implicit_directional - finite_directional) / jnp.maximum(
        jnp.abs(finite_directional), 1e-10
    )

    return {
        "optimization_steps": jnp.asarray(OPTIMIZATION_STEPS, dtype=jnp.int32),
        "times": inputs["times"],
        "initial_raw": initial_raw,
        "optimized_raw": optimized_raw,
        "initial_beta": jax.vmap(lambda t: _schedule(initial_raw, t))(inputs["times"]),
        "optimized_beta": jax.vmap(lambda t: _schedule(optimized_raw, t))(inputs["times"]),
        "optimization_objective": trace[:, 0],
        "optimization_gradient_norm": trace[:, 1],
        "optimization_raw": trace[:, 2:],
        "initial_objective": _objective(initial_raw, inputs),
        "optimized_objective": _objective(optimized_raw, inputs),
        "implicit_directional_gradient": implicit_directional,
        "finite_difference_directional_gradient": finite_directional,
        "gradient_relative_error": gradient_error,
        "initial_train_energy": initial_fit[7],
        "optimized_train_energy": optimized_fit[7],
        "initial_train_ess": initial_fit[8],
        "optimized_train_ess": optimized_fit[8],
        "initial_train_calibration_residual": initial_fit[9],
        "optimized_train_calibration_residual": optimized_fit[9],
        "initial_train_weak_residual": initial_fit[11],
        "optimized_train_weak_residual": optimized_fit[11],
        "initial_validation_state": initial_validation[0],
        "optimized_validation_state": optimized_validation[0],
        "initial_validation_weights": initial_validation[1],
        "optimized_validation_weights": optimized_validation[1],
        "initial_validation_moments": initial_validation[2],
        "optimized_validation_moments": optimized_validation[2],
        "initial_validation_lambda": initial_validation[3],
        "optimized_validation_lambda": optimized_validation[3],
        "initial_validation_energy": initial_validation[5],
        "optimized_validation_energy": optimized_validation[5],
        "initial_validation_ess": initial_validation[6],
        "optimized_validation_ess": optimized_validation[6],
        "initial_validation_calibration_residual": initial_validation[7],
        "optimized_validation_calibration_residual": optimized_validation[7],
        "initial_validation_angular4": initial_validation[8],
        "optimized_validation_angular4": optimized_validation[8],
        "initial_validation_weak_residual": initial_validation[9],
        "optimized_validation_weak_residual": optimized_validation[9],
        "zero_validation_weak_residual": optimized_validation[10],
        "initial_validation_ritz_gain": initial_validation[11],
        "optimized_validation_ritz_gain": optimized_validation[11],
        "initial_validation_forcing_power": initial_validation[12],
        "optimized_validation_forcing_power": optimized_validation[12],
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
    train_minus: Differentiable[Array[(None, None, 2), Float64]]
    train_plus: Differentiable[Array[(None, None, 2), Float64]]
    train_noise: Differentiable[Array[(None, None, 2), Float64]]
    validation_minus: Differentiable[Array[(None, None, 2), Float64]]
    validation_plus: Differentiable[Array[(None, None, 2), Float64]]
    validation_noise: Differentiable[Array[(None, None, 2), Float64]]
    target: Differentiable[Array[(5,), Float64]]
    feature_weight: Differentiable[Array[(None, 2), Float64]]
    feature_bias: Differentiable[Array[(None,), Float64]]
    feature_time: Differentiable[Array[(None,), Float64]]
    test_feature_weight: Differentiable[Array[(None, 2), Float64]]
    test_feature_bias: Differentiable[Array[(None,), Float64]]
    test_feature_time: Differentiable[Array[(None,), Float64]]
    initial_raw: Differentiable[Array[(3,), Float64]]
    ess_floor: Differentiable[Float64]
    ess_penalty: Differentiable[Float64]
    schedule_regularization: Differentiable[Float64]
    learning_rate: Differentiable[Float64]
    gradient_check_direction: Differentiable[Array[(3,), Float64]]
    finite_difference_epsilon: Differentiable[Float64]


class OutputSchema(BaseModel):
    optimization_steps: Int32
    times: Differentiable[Array[(None,), Float64]]
    initial_raw: Differentiable[Array[(3,), Float64]]
    optimized_raw: Differentiable[Array[(3,), Float64]]
    initial_beta: Differentiable[Array[(None,), Float64]]
    optimized_beta: Differentiable[Array[(None,), Float64]]
    optimization_objective: Differentiable[Array[(None,), Float64]]
    optimization_gradient_norm: Differentiable[Array[(None,), Float64]]
    optimization_raw: Differentiable[Array[(None, 3), Float64]]
    initial_objective: Differentiable[Float64]
    optimized_objective: Differentiable[Float64]
    implicit_directional_gradient: Differentiable[Float64]
    finite_difference_directional_gradient: Differentiable[Float64]
    gradient_relative_error: Differentiable[Float64]
    initial_train_energy: Differentiable[Array[(None,), Float64]]
    optimized_train_energy: Differentiable[Array[(None,), Float64]]
    initial_train_ess: Differentiable[Array[(None,), Float64]]
    optimized_train_ess: Differentiable[Array[(None,), Float64]]
    initial_train_calibration_residual: Differentiable[Array[(None,), Float64]]
    optimized_train_calibration_residual: Differentiable[Array[(None,), Float64]]
    initial_train_weak_residual: Differentiable[Array[(None,), Float64]]
    optimized_train_weak_residual: Differentiable[Array[(None,), Float64]]
    initial_validation_state: Differentiable[Array[(None, None, 2), Float64]]
    optimized_validation_state: Differentiable[Array[(None, None, 2), Float64]]
    initial_validation_weights: Differentiable[Array[(None, None), Float64]]
    optimized_validation_weights: Differentiable[Array[(None, None), Float64]]
    initial_validation_moments: Differentiable[Array[(None, 5), Float64]]
    optimized_validation_moments: Differentiable[Array[(None, 5), Float64]]
    initial_validation_lambda: Differentiable[Array[(None, 5), Float64]]
    optimized_validation_lambda: Differentiable[Array[(None, 5), Float64]]
    initial_validation_energy: Differentiable[Array[(None,), Float64]]
    optimized_validation_energy: Differentiable[Array[(None,), Float64]]
    initial_validation_ess: Differentiable[Array[(None,), Float64]]
    optimized_validation_ess: Differentiable[Array[(None,), Float64]]
    initial_validation_calibration_residual: Differentiable[Array[(None,), Float64]]
    optimized_validation_calibration_residual: Differentiable[Array[(None,), Float64]]
    initial_validation_angular4: Differentiable[Array[(None,), Float64]]
    optimized_validation_angular4: Differentiable[Array[(None,), Float64]]
    initial_validation_weak_residual: Differentiable[Array[(None,), Float64]]
    optimized_validation_weak_residual: Differentiable[Array[(None,), Float64]]
    zero_validation_weak_residual: Differentiable[Array[(None,), Float64]]
    initial_validation_ritz_gain: Differentiable[Array[(None,), Float64]]
    optimized_validation_ritz_gain: Differentiable[Array[(None,), Float64]]
    initial_validation_forcing_power: Differentiable[Array[(None,), Float64]]
    optimized_validation_forcing_power: Differentiable[Array[(None,), Float64]]


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
