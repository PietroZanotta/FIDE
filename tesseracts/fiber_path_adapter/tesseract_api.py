"""Level-2 MFSI fiber-adapted reference-path experiment.

This Tesseract owns one small, differentiable scientific map: choose the
constant noise amplitude of the one-dimensional stochastic interpolant so as
to minimize integrated exact Poisson-correction energy, with an overlap/ESS
penalty.  ``apply_jax`` is also the direct in-process JAX implementation used
by the experiment runner.

The empirical multiplier uses an implicit custom JVP.  Consequently the
reported objective gradient differentiates the converged calibration equation
instead of backpropagating through its Newton iterations.
"""
from __future__ import annotations

import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, Field

jax.config.update("jax_enable_x64", True)

NEWTON_ITERS = 24
OPTIMIZATION_STEPS = 128
DAMPING = 1e-11


def _stable_solve(cov, rhs):
    # This controlled two-observable experiment is full rank.  A damped direct
    # solve avoids undefined eigenvector derivatives when symmetry makes the
    # whitened covariance exactly proportional to the identity.
    regularized = 0.5 * (cov + cov.T)
    regularized = regularized + DAMPING * jnp.eye(cov.shape[0], dtype=cov.dtype)
    return jnp.linalg.solve(regularized, rhs)


def _tilt(lam, log_base_mass, observables):
    mass = jax.nn.softmax(log_base_mass + observables @ lam)
    moments = mass @ observables
    centered = observables - moments
    covariance = (centered.T * mass) @ centered
    return mass, moments, covariance


def _calibrate_primal(log_base_mass, observables, target):
    lam0 = jnp.zeros(target.shape[0], dtype=observables.dtype)

    def body(_, lam):
        _, moments, covariance = _tilt(lam, log_base_mass, observables)
        step = _stable_solve(covariance, moments - target)
        step_norm = jnp.linalg.norm(step)
        step = step * jnp.minimum(1.0, 2.0 / jnp.maximum(step_norm, 1e-30))
        return lam - step

    return jax.lax.fori_loop(0, NEWTON_ITERS, body, lam0)


@jax.custom_jvp
def _calibrate(log_base_mass, observables, target):
    return _calibrate_primal(log_base_mass, observables, target)


@_calibrate.defjvp
def _calibrate_jvp(primals, tangents):
    log_base_mass, observables, target = primals
    dlog_base_mass, dobservables, dtarget = tangents
    lam = _calibrate_primal(log_base_mass, observables, target)
    mass, moments, covariance = _tilt(lam, log_base_mass, observables)
    centered = observables - moments
    dlogit = dlog_base_mass + jnp.sum(dobservables * lam[None, :], axis=-1)
    fixed_lambda_dF = (
        mass @ dobservables
        + jnp.sum(mass[:, None] * centered * dlogit[:, None], axis=0)
        - dtarget
    )
    dlam = _stable_solve(covariance, -fixed_lambda_dF)
    return lam, dlam


def _normal_logpdf(x, mean, variance):
    return -0.5 * (jnp.log(2.0 * jnp.pi * variance) + (x - mean) ** 2 / variance)


def _reference(raw_schedule, t, x, quadrature_weights, amplitude):
    """Exact SI marginal and velocity for Experiment A.

    gamma(t) = sqrt(2 t (1-t)) * softplus(raw_schedule).  The special value
    beta=1 keeps the raw bridge variance exactly one for all times because both
    endpoints have unit variance.
    """
    beta = jax.nn.softplus(raw_schedule)
    endpoint_component_variance = 1.0 - amplitude * amplitude
    variance = (
        (1.0 - t) ** 2
        + t * t * endpoint_component_variance
        + 2.0 * t * (1.0 - t) * beta * beta
    )
    variance_dt = (
        -2.0 * (1.0 - t)
        + 2.0 * t * endpoint_component_variance
        + 2.0 * (1.0 - 2.0 * t) * beta * beta
    )
    mean = t * amplitude
    log_plus = _normal_logpdf(x, mean, variance)
    log_minus = _normal_logpdf(x, -mean, variance)
    log_density = jax.scipy.special.logsumexp(
        jnp.stack([log_plus, log_minus]), axis=0
    ) - jnp.log(2.0)
    log_normalizer = jax.scipy.special.logsumexp(
        log_density + jnp.log(quadrature_weights)
    )
    log_density = log_density - log_normalizer
    density = jnp.exp(log_density)

    linear_rate = 0.5 * variance_dt / variance
    posterior_sign = jnp.tanh((t * amplitude * x) / variance)
    velocity = (
        linear_rate * x
        + amplitude * (1.0 - linear_rate * t) * posterior_sign
    )
    return log_density, density, velocity


def _cumulative_trapezoid(values, x):
    increments = 0.5 * (values[1:] + values[:-1]) * (x[1:] - x[:-1])
    return jnp.concatenate([jnp.zeros((1,), dtype=values.dtype), jnp.cumsum(increments)])


def _one_time(raw_schedule, t, x, quadrature_weights, amplitude, target):
    log_reference, reference_density, reference_velocity = _reference(
        raw_schedule, t, x, quadrature_weights, amplitude
    )
    observables = jnp.stack([x, x * x], axis=-1)
    log_base_mass = log_reference + jnp.log(quadrature_weights)
    lam = _calibrate(log_base_mass, observables, target)
    projected_mass, moments, covariance = _tilt(lam, log_base_mass, observables)
    projected_density = projected_mass / quadrature_weights

    jphi_u = jnp.stack(
        [reference_velocity, 2.0 * x * reference_velocity], axis=-1
    )
    expected_jphi_u = projected_mass @ jphi_u
    scalar = jphi_u @ lam
    covariance_term = jnp.sum(
        projected_mass[:, None]
        * (observables - target)
        * scalar[:, None],
        axis=0,
    )
    lambda_dot = _stable_solve(covariance, -expected_jphi_u - covariance_term)
    forcing = (
        (observables - target) @ lambda_dot
        + (jphi_u - expected_jphi_u) @ lam
    )
    forcing = forcing - projected_mass @ forcing

    flux = _cumulative_trapezoid(projected_density * forcing, x)
    cdf = _cumulative_trapezoid(projected_density, x)
    flux = flux - flux[-1] * cdf
    correction = -flux / jnp.maximum(projected_density, 1e-12)
    correction = jnp.where(projected_density > 1e-12, correction, 0.0)
    correction_energy = jnp.sum(projected_mass * correction * correction)

    reference_mass = quadrature_weights * reference_density
    ess_fraction = 1.0 / jnp.sum(
        projected_mass * projected_mass / jnp.maximum(reference_mass, 1e-300)
    )
    calibration_residual = jnp.linalg.norm(moments - target)
    return (
        correction_energy,
        ess_fraction,
        calibration_residual,
        moments,
        lam,
        reference_density,
        projected_density,
        correction,
    )


def _evaluate(raw_schedule, inputs):
    per_time = jax.vmap(
        lambda t: _one_time(
            raw_schedule,
            t,
            inputs["grid_x"],
            inputs["grid_weights"],
            inputs["amplitude"],
            inputs["target"],
        )
    )(inputs["times"])
    energy, ess = per_time[0], per_time[1]
    overlap_violation = jax.nn.relu(inputs["ess_floor"] - ess)
    integrated_energy = jnp.trapezoid(energy, inputs["times"])
    ess_penalty = inputs["ess_penalty"] * jnp.trapezoid(
        overlap_violation**2, inputs["times"]
    )
    objective = integrated_energy + ess_penalty
    return objective, ess_penalty, per_time


def _objective(raw_schedule, inputs):
    return _evaluate(raw_schedule, inputs)[0]


def _optimize(initial_raw, inputs):
    """Fixed-budget scalar Adam optimization, kept inside the JAX recipe."""
    initial = (
        initial_raw,
        jnp.zeros_like(initial_raw),
        jnp.zeros_like(initial_raw),
        jnp.asarray(0, dtype=jnp.int32),
    )

    def step(carry, _):
        raw, first_moment, second_moment, count = carry
        value, gradient = jax.value_and_grad(_objective)(raw, inputs)
        gradient = jnp.clip(gradient, -10.0, 10.0)
        count = count + 1
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient * gradient
        mhat = first_moment / (1.0 - 0.9**count)
        vhat = second_moment / (1.0 - 0.999**count)
        raw = raw - inputs["learning_rate"] * mhat / (jnp.sqrt(vhat) + 1e-8)
        trace = jnp.stack([value, jax.nn.softplus(raw), gradient])
        return (raw, first_moment, second_moment, count), trace

    final, trace = jax.lax.scan(step, initial, xs=None, length=OPTIMIZATION_STEPS)
    return final[0], trace


@jax.jit
def apply_jax(inputs: dict) -> dict:
    initial_raw = inputs["schedule_raw"]
    final_raw, optimization_trace = _optimize(initial_raw, inputs)
    initial_objective, initial_penalty, initial = _evaluate(initial_raw, inputs)
    final_objective, final_penalty, final = _evaluate(final_raw, inputs)
    initial_gradient = jax.grad(_objective)(initial_raw, inputs)
    final_gradient = jax.grad(_objective)(final_raw, inputs)
    epsilon = inputs["finite_difference_epsilon"]
    finite_difference_gradient = (
        _objective(initial_raw + epsilon, inputs)
        - _objective(initial_raw - epsilon, inputs)
    ) / (2.0 * epsilon)
    gradient_relative_error = jnp.abs(initial_gradient - finite_difference_gradient) / jnp.maximum(
        jnp.abs(finite_difference_gradient), 1e-12
    )

    landscape = jax.vmap(lambda raw: _evaluate(raw, inputs))(inputs["landscape_raw"])
    landscape_objective = landscape[0]
    landscape_per_time = landscape[2]

    return {
        "optimization_steps": jnp.asarray(OPTIMIZATION_STEPS, dtype=jnp.int32),
        "initial_raw": initial_raw,
        "optimized_raw": final_raw,
        "initial_beta": jax.nn.softplus(initial_raw),
        "optimized_beta": jax.nn.softplus(final_raw),
        "initial_objective": initial_objective,
        "optimized_objective": final_objective,
        "initial_ess_penalty": initial_penalty,
        "optimized_ess_penalty": final_penalty,
        "initial_gradient": initial_gradient,
        "optimized_gradient": final_gradient,
        "finite_difference_gradient": finite_difference_gradient,
        "gradient_relative_error": gradient_relative_error,
        "optimization_objective": optimization_trace[:, 0],
        "optimization_beta": optimization_trace[:, 1],
        "optimization_gradient": optimization_trace[:, 2],
        "times": inputs["times"],
        "grid_x": inputs["grid_x"],
        "initial_correction_energy": initial[0],
        "optimized_correction_energy": final[0],
        "initial_ess_fraction": initial[1],
        "optimized_ess_fraction": final[1],
        "initial_calibration_residual": initial[2],
        "optimized_calibration_residual": final[2],
        "initial_moments": initial[3],
        "optimized_moments": final[3],
        "initial_lambda": initial[4],
        "optimized_lambda": final[4],
        "initial_reference_density": initial[5],
        "optimized_reference_density": final[5],
        "initial_projected_density": initial[6],
        "optimized_projected_density": final[6],
        "initial_correction": initial[7],
        "optimized_correction": final[7],
        "landscape_beta": jax.nn.softplus(inputs["landscape_raw"]),
        "landscape_objective": landscape_objective,
        "landscape_integrated_correction_energy": jax.vmap(
            lambda energy: jnp.trapezoid(energy, inputs["times"])
        )(landscape_per_time[0]),
        "landscape_min_ess_fraction": jnp.min(landscape_per_time[1], axis=1),
    }


def apply_payload(inputs: dict) -> dict:
    converted = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in inputs.items()}
    return jax.tree.map(np.asarray, apply_jax(converted))


TESSERACT_RUNTIME_AVAILABLE = "TESSERACT_API_PATH" in os.environ
if TESSERACT_RUNTIME_AVAILABLE:
    from tesseract_core.runtime import Array, Differentiable, Float64, Int32
    from tesseract_core.runtime.jax_recipes import (
        jax_abstract_eval,
        jax_apply,
        jax_jacobian,
        jax_jvp,
        jax_vjp,
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
    grid_x: Differentiable[Array[(None,), Float64]]
    grid_weights: Differentiable[Array[(None,), Float64]]
    times: Differentiable[Array[(None,), Float64]]
    landscape_raw: Differentiable[Array[(None,), Float64]]
    amplitude: Differentiable[Float64] = Field(description="Experiment-A mixture displacement.")
    target: Differentiable[Array[(2,), Float64]]
    schedule_raw: Differentiable[Float64]
    ess_floor: Differentiable[Float64]
    ess_penalty: Differentiable[Float64]
    learning_rate: Differentiable[Float64]
    finite_difference_epsilon: Differentiable[Float64]


class OutputSchema(BaseModel):
    optimization_steps: Int32
    initial_raw: Differentiable[Float64]
    optimized_raw: Differentiable[Float64]
    initial_beta: Differentiable[Float64]
    optimized_beta: Differentiable[Float64]
    initial_objective: Differentiable[Float64]
    optimized_objective: Differentiable[Float64]
    initial_ess_penalty: Differentiable[Float64]
    optimized_ess_penalty: Differentiable[Float64]
    initial_gradient: Differentiable[Float64]
    optimized_gradient: Differentiable[Float64]
    finite_difference_gradient: Differentiable[Float64]
    gradient_relative_error: Differentiable[Float64]
    optimization_objective: Differentiable[Array[(None,), Float64]]
    optimization_beta: Differentiable[Array[(None,), Float64]]
    optimization_gradient: Differentiable[Array[(None,), Float64]]
    times: Differentiable[Array[(None,), Float64]]
    grid_x: Differentiable[Array[(None,), Float64]]
    initial_correction_energy: Differentiable[Array[(None,), Float64]]
    optimized_correction_energy: Differentiable[Array[(None,), Float64]]
    initial_ess_fraction: Differentiable[Array[(None,), Float64]]
    optimized_ess_fraction: Differentiable[Array[(None,), Float64]]
    initial_calibration_residual: Differentiable[Array[(None,), Float64]]
    optimized_calibration_residual: Differentiable[Array[(None,), Float64]]
    initial_moments: Differentiable[Array[(None, 2), Float64]]
    optimized_moments: Differentiable[Array[(None, 2), Float64]]
    initial_lambda: Differentiable[Array[(None, 2), Float64]]
    optimized_lambda: Differentiable[Array[(None, 2), Float64]]
    initial_reference_density: Differentiable[Array[(None, None), Float64]]
    optimized_reference_density: Differentiable[Array[(None, None), Float64]]
    initial_projected_density: Differentiable[Array[(None, None), Float64]]
    optimized_projected_density: Differentiable[Array[(None, None), Float64]]
    initial_correction: Differentiable[Array[(None, None), Float64]]
    optimized_correction: Differentiable[Array[(None, None), Float64]]
    landscape_beta: Differentiable[Array[(None,), Float64]]
    landscape_objective: Differentiable[Array[(None,), Float64]]
    landscape_integrated_correction_energy: Differentiable[Array[(None,), Float64]]
    landscape_min_ess_fraction: Differentiable[Array[(None,), Float64]]


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
