"""Finite-bank level-2 MFSI for a periodic many-particle configuration space.

States have shape ``(n_particles, 2)`` and are treated as one microscopic
configuration.  The neural potential consumes permutation/translation/rotation
invariant radial pair descriptors; autodiff therefore produces an equivariant
32-dimensional correction for the default 16-particle experiment.
"""
from __future__ import annotations

import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel

jax.config.update("jax_enable_x64", True)

NEWTON_ITERS = 60
OPTIMIZATION_STEPS = 60
CALIBRATION_RIDGE = 2e-9
RITZ_RIDGE = 5e-5


def _solve(matrix, rhs, ridge):
    matrix = 0.5 * (matrix + matrix.T)
    return jnp.linalg.solve(
        matrix + ridge * jnp.eye(matrix.shape[0], dtype=matrix.dtype), rhs
    )


def _tilt(lam, observables):
    weights = jax.nn.softmax(observables @ lam)
    moments = weights @ observables
    centered = observables - moments
    covariance = (centered.T * weights) @ centered
    return weights, moments, covariance


def _calibrate_primal(observables, target):
    initial = jnp.zeros(target.shape[0], dtype=observables.dtype)

    def body(_, lam):
        _, moments, covariance = _tilt(lam, observables)
        step = _solve(covariance, moments - target, CALIBRATION_RIDGE)
        norm = jnp.linalg.norm(step)
        return lam - step * jnp.minimum(1.0, 3.0 / jnp.maximum(norm, 1e-30))

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
    basis = jnp.asarray([1.0, jnp.cos(2.0 * jnp.pi * t), jnp.sin(2.0 * jnp.pi * t)])
    return jax.nn.softplus(raw @ basis)


def _gamma(raw, t):
    return jnp.sqrt(jnp.maximum(2.0 * t * (1.0 - t), 1e-12)) * _schedule(raw, t)


def _bridge(raw, t, x_minus, x_plus, noise):
    gamma = _gamma(raw, t)
    gamma_dot = jax.grad(lambda time: _gamma(raw, time))(t)
    state = (1.0 - t) * x_minus + t * x_plus + gamma * noise
    velocity = x_plus - x_minus + gamma_dot * noise
    return state, velocity


def _observables(configurations):
    x = configurations[:, :, 0]
    y = configurations[:, :, 1]
    return jnp.stack(
        [
            jnp.mean(x * x + y * y, axis=1),
            jnp.mean(x * x - y * y, axis=1),
            jnp.mean(2.0 * x * y, axis=1),
        ],
        axis=-1,
    )


def _jphi_velocity(configurations, velocity):
    x, y = configurations[:, :, 0], configurations[:, :, 1]
    ux, uy = velocity[:, :, 0], velocity[:, :, 1]
    return jnp.stack(
        [
            jnp.mean(2.0 * (x * ux + y * uy), axis=1),
            jnp.mean(2.0 * (x * ux - y * uy), axis=1),
            jnp.mean(2.0 * (y * ux + x * uy), axis=1),
        ],
        axis=-1,
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
    calibration_residual = jnp.linalg.norm(moments - target)

    angles = jnp.arctan2(state[:, :, 1], state[:, :, 0])
    q4_per_configuration = jnp.sqrt(
        jnp.mean(jnp.cos(4.0 * angles), axis=1) ** 2
        + jnp.mean(jnp.sin(4.0 * angles), axis=1) ** 2
        + 1e-16
    )
    q4 = weights @ q4_per_configuration
    return state, weights, moments, lam, forcing, ess, calibration_residual, q4


def _pair_descriptors_single(configuration, centers, width, box_size):
    displacement = configuration[:, None, :] - configuration[None, :, :]
    displacement = displacement - box_size * jnp.round(displacement / box_size)
    rows, cols = jnp.triu_indices(configuration.shape[0], 1)
    distances = jnp.sqrt(jnp.sum(displacement[rows, cols] ** 2, axis=-1) + 1e-10)
    radial = jnp.exp(-0.5 * ((distances[:, None] - centers[None, :]) / width) ** 2)
    return jnp.mean(radial, axis=0)


def _features_single(configuration, t, inputs):
    descriptors = _pair_descriptors_single(
        configuration, inputs["radial_centers"], inputs["radial_width"], inputs["box_size"]
    )
    preactivation = (
        inputs["feature_weight"] @ descriptors
        + inputs["feature_bias"]
        + t * inputs["feature_time"]
    )
    return jnp.tanh(preactivation)


def _test_features_single(configuration, t, inputs):
    descriptors = _pair_descriptors_single(
        configuration, inputs["radial_centers"], inputs["radial_width"], inputs["box_size"]
    )
    preactivation = (
        inputs["test_feature_weight"] @ descriptors
        + inputs["test_feature_bias"]
        + t * inputs["test_feature_time"]
    )
    return jnp.tanh(preactivation)


def _feature_batch(state, t, inputs, test=False):
    function = _test_features_single if test else _features_single
    values = jax.vmap(lambda configuration: function(configuration, t, inputs))(state)
    gradients = jax.vmap(
        jax.jacrev(lambda configuration: function(configuration, t, inputs))
    )(state)
    return values, gradients


def _fit_time(raw, t, x_minus, x_plus, noise, target, inputs):
    state, weights, moments, lam, forcing, ess, residual, q4 = _fiber_state(
        raw, t, x_minus, x_plus, noise, target
    )
    features, gradients = _feature_batch(state, t, inputs)
    gram = jnp.einsum("m,mknd,mlnd->kl", weights, gradients, gradients)
    rhs = jnp.einsum("m,mk,m->k", weights, features, forcing)
    coefficients = _solve(gram, rhs, RITZ_RIDGE)
    correction = -jnp.einsum("mknd,k->mnd", gradients, coefficients)
    energy = jnp.sum(weights * jnp.sum(correction * correction, axis=(1, 2)))
    return coefficients, state, weights, moments, lam, forcing, correction, energy, ess, residual, q4


def _validate_time(raw, coefficients, t, x_minus, x_plus, noise, target, inputs):
    state, weights, moments, lam, forcing, ess, residual, q4 = _fiber_state(
        raw, t, x_minus, x_plus, noise, target
    )
    features, gradients = _feature_batch(state, t, inputs)
    correction = -jnp.einsum("mknd,k->mnd", gradients, coefficients)
    energy = jnp.sum(weights * jnp.sum(correction * correction, axis=(1, 2)))
    potential = features @ coefficients
    ritz_gain = jnp.sum(weights * potential * forcing) - 0.5 * energy

    test_features, test_gradients = _feature_batch(state, t, inputs, test=True)
    lhs = -jnp.einsum("m,mjnd,mnd->j", weights, test_gradients, correction)
    rhs = jnp.einsum("m,mj,m->j", weights, test_features, forcing)
    weak_residual = jnp.linalg.norm(lhs - rhs) / jnp.maximum(jnp.linalg.norm(rhs), 1e-10)
    forcing_power = jnp.sum(weights * forcing * forcing)
    return state, weights, moments, lam, energy, ess, residual, q4, weak_residual, ritz_gain, forcing_power


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
    integrated_energy = jnp.trapezoid(energy, inputs["times"])
    violation = jax.nn.relu(inputs["ess_floor"] - ess)
    ess_penalty = inputs["ess_penalty"] * jnp.trapezoid(violation**2, inputs["times"])
    regularization = inputs["schedule_regularization"] * jnp.mean(raw[1:] ** 2)
    return integrated_energy + ess_penalty + regularization


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
        gradient = gradient * jnp.minimum(1.0, 4.0 / jnp.maximum(norm, 1e-12))
        count = count + 1
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient * gradient
        mhat = first / (1.0 - 0.9**count)
        vhat = second / (1.0 - 0.999**count)
        raw = raw - inputs["learning_rate"] * mhat / (jnp.sqrt(vhat) + 1e-8)
        trace = jnp.concatenate([jnp.asarray([value, norm]), raw])
        return (raw, first, second, count), trace

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

    gradient = jax.grad(_objective)(initial_raw, inputs)
    direction = inputs["gradient_check_direction"]
    direction = direction / jnp.linalg.norm(direction)
    epsilon = inputs["finite_difference_epsilon"]
    implicit_directional = gradient @ direction
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
        "state_dimension": jnp.asarray(inputs["train_minus"].shape[2] * 2, dtype=jnp.int32),
        "particle_count": jnp.asarray(inputs["train_minus"].shape[2], dtype=jnp.int32),
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
        "initial_validation_state": initial_validation[0],
        "optimized_validation_state": optimized_validation[0],
        "initial_validation_weights": initial_validation[1],
        "optimized_validation_weights": optimized_validation[1],
        "initial_validation_moments": initial_validation[2],
        "optimized_validation_moments": optimized_validation[2],
        "initial_validation_lambda": initial_validation[3],
        "optimized_validation_lambda": optimized_validation[3],
        "initial_validation_energy": initial_validation[4],
        "optimized_validation_energy": optimized_validation[4],
        "initial_validation_ess": initial_validation[5],
        "optimized_validation_ess": optimized_validation[5],
        "initial_validation_calibration_residual": initial_validation[6],
        "optimized_validation_calibration_residual": optimized_validation[6],
        "initial_validation_q4": initial_validation[7],
        "optimized_validation_q4": optimized_validation[7],
        "initial_validation_weak_residual": initial_validation[8],
        "optimized_validation_weak_residual": optimized_validation[8],
        "initial_validation_ritz_gain": initial_validation[9],
        "optimized_validation_ritz_gain": optimized_validation[9],
        "initial_validation_forcing_power": initial_validation[10],
        "optimized_validation_forcing_power": optimized_validation[10],
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
    train_minus: Differentiable[Array[(None, None, None, 2), Float64]]
    train_plus: Differentiable[Array[(None, None, None, 2), Float64]]
    train_noise: Differentiable[Array[(None, None, None, 2), Float64]]
    validation_minus: Differentiable[Array[(None, None, None, 2), Float64]]
    validation_plus: Differentiable[Array[(None, None, None, 2), Float64]]
    validation_noise: Differentiable[Array[(None, None, None, 2), Float64]]
    target: Differentiable[Array[(3,), Float64]]
    radial_centers: Differentiable[Array[(None,), Float64]]
    radial_width: Differentiable[Float64]
    box_size: Differentiable[Float64]
    feature_weight: Differentiable[Array[(None, None), Float64]]
    feature_bias: Differentiable[Array[(None,), Float64]]
    feature_time: Differentiable[Array[(None,), Float64]]
    test_feature_weight: Differentiable[Array[(None, None), Float64]]
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
    state_dimension: Int32
    particle_count: Int32
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
    initial_validation_state: Differentiable[Array[(None, None, None, 2), Float64]]
    optimized_validation_state: Differentiable[Array[(None, None, None, 2), Float64]]
    initial_validation_weights: Differentiable[Array[(None, None), Float64]]
    optimized_validation_weights: Differentiable[Array[(None, None), Float64]]
    initial_validation_moments: Differentiable[Array[(None, 3), Float64]]
    optimized_validation_moments: Differentiable[Array[(None, 3), Float64]]
    initial_validation_lambda: Differentiable[Array[(None, 3), Float64]]
    optimized_validation_lambda: Differentiable[Array[(None, 3), Float64]]
    initial_validation_energy: Differentiable[Array[(None,), Float64]]
    optimized_validation_energy: Differentiable[Array[(None,), Float64]]
    initial_validation_ess: Differentiable[Array[(None,), Float64]]
    optimized_validation_ess: Differentiable[Array[(None,), Float64]]
    initial_validation_calibration_residual: Differentiable[Array[(None,), Float64]]
    optimized_validation_calibration_residual: Differentiable[Array[(None,), Float64]]
    initial_validation_q4: Differentiable[Array[(None,), Float64]]
    optimized_validation_q4: Differentiable[Array[(None,), Float64]]
    initial_validation_weak_residual: Differentiable[Array[(None,), Float64]]
    optimized_validation_weak_residual: Differentiable[Array[(None,), Float64]]
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
