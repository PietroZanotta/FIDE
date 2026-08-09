#!/usr/bin/env python3
"""Stage-2 MFSI experiment: fiber-adapted endpoint coupling only.

The temporal schedule is loaded from the completed paper-facing schedule study
and is never optimized here.  Independent, geometric Sinkhorn, and fiber-aware
plans share endpoint banks and every downstream setting.  The fiber-aware plan
is selected without consulting the final evaluation bank or projected-law MMD.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

import level2_paper_study as paper

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "coupling_study"
FROZEN_SCHEDULE_ROOT = ROOT / "results" / "level2_paper_study" / "jax"
METHODS = ("independent", "geometric_sinkhorn", "fiber_aware")
METHOD_LABELS = {
    "independent": "independent",
    "geometric_sinkhorn": "geometric OT",
    "fiber_aware": "fiber-aware",
}
SINKHORN_EPSILON = 0.15
SINKHORN_ITERATIONS = 100
ESS_FLOOR = 0.18
ESS_PENALTY = 15.0
COUPLING_LEARNING_RATE = 0.04


class EndpointBank(NamedTuple):
    minus: np.ndarray
    plus: np.ndarray
    minus_weights: np.ndarray
    plus_weights: np.ndarray
    minus_indices: np.ndarray
    plus_indices: np.ndarray


class FiberStatistics(NamedTuple):
    observables: jax.Array
    observable_velocity: jax.Array
    descriptors: jax.Array
    descriptor_gram: jax.Array
    q4: jax.Array


def _canonicalize(configurations: np.ndarray) -> np.ndarray:
    """Apply the pre-existing polar-angle particle convention."""
    result = np.empty_like(configurations)
    for index, configuration in enumerate(configurations):
        centered = configuration - configuration.mean(axis=0, keepdims=True)
        order = np.argsort(np.arctan2(centered[:, 1], centered[:, 0]))
        result[index] = centered[order]
    return result


def make_endpoint_bank(populations, rng: np.random.Generator, count: int) -> EndpointBank:
    minus_indices = rng.choice(
        len(populations["minus"]), size=count, replace=False,
        p=populations["minus_weights"],
    )
    plus_indices = rng.choice(
        len(populations["plus"]), size=count, replace=False,
        p=populations["plus_weights"],
    )
    minus = _canonicalize(populations["minus"][minus_indices])
    plus = _canonicalize(populations["plus"][plus_indices])
    minus_weights, minus_residual = paper.calibrate_numpy(
        np.asarray(paper.v_observables(jnp.asarray(minus))), populations["target"]
    )
    plus_weights, plus_residual = paper.calibrate_numpy(
        np.asarray(paper.v_observables(jnp.asarray(plus))), populations["target"]
    )
    if max(minus_residual, plus_residual) > 1e-7:
        raise RuntimeError(
            "finite endpoint bank does not contain the fixed target moment; "
            f"calibration residuals are {minus_residual:.3e}, {plus_residual:.3e}"
        )
    return EndpointBank(minus, plus, minus_weights, plus_weights,
                        minus_indices, plus_indices)


def microscopic_cost(bank: EndpointBank) -> np.ndarray:
    """Permutation/translation-aware periodic many-particle squared distance."""
    count = len(bank.minus)
    costs = np.empty((count, count), dtype=np.float64)
    for i, minus in enumerate(bank.minus):
        minus = minus - minus.mean(axis=0, keepdims=True)
        for j, plus in enumerate(bank.plus):
            plus = plus - plus.mean(axis=0, keepdims=True)
            displacement = minus[:, None, :] - plus[None, :, :]
            displacement -= paper.BOX_SIZE * np.round(displacement / paper.BOX_SIZE)
            particle_cost = np.sum(displacement * displacement, axis=-1)
            rows, cols = linear_sum_assignment(particle_cost)
            costs[i, j] = float(np.mean(particle_cost[rows, cols]))
    scale = max(float(np.median(costs)), 1e-8)
    return costs / scale


def coupling_features(bank: EndpointBank) -> np.ndarray:
    """Nine measured-observable interactions; hidden q4 is deliberately absent."""
    minus = np.asarray(paper.v_observables(jnp.asarray(bank.minus)))
    plus = np.asarray(paper.v_observables(jnp.asarray(bank.plus)))
    pooled = np.concatenate([minus, plus], axis=0)
    mean = pooled.mean(axis=0)
    scale = np.maximum(pooled.std(axis=0), 1e-6)
    minus = (minus - mean) / scale
    plus = (plus - mean) / scale
    interactions = np.einsum("ir,js->ijrs", minus, plus)
    return interactions.reshape(len(minus), len(plus), -1)


def _log_sinkhorn(logits: jax.Array, row_mass: jax.Array,
                  column_mass: jax.Array,
                  iterations: int = SINKHORN_ITERATIONS) -> jax.Array:
    """Fixed-marginal log-domain Sinkhorn plan."""
    row_target = jnp.log(jnp.maximum(row_mass, 1e-300))
    column_target = jnp.log(jnp.maximum(column_mass, 1e-300))
    row_potential = jnp.zeros_like(row_mass)
    col_potential = jnp.zeros_like(column_mass)

    def body(_, values):
        row, col = values
        row = row_target - jax.scipy.special.logsumexp(
            logits + col[None, :], axis=1
        )
        col = column_target - jax.scipy.special.logsumexp(
            logits + row[:, None], axis=0
        )
        return row, col

    row_potential, col_potential = jax.lax.fori_loop(
        0, iterations, body, (row_potential, col_potential)
    )
    return jnp.exp(logits + row_potential[:, None] + col_potential[None, :])


def build_plan(method: str, cost: jax.Array, features: jax.Array,
               row_mass: jax.Array, column_mass: jax.Array,
               parameters: jax.Array | None = None) -> jax.Array:
    if method == "independent":
        return row_mass[:, None] * column_mass[None, :]
    logits = -cost / SINKHORN_EPSILON
    if method == "fiber_aware":
        if parameters is None:
            raise ValueError("fiber-aware coupling requires parameters")
        logits = logits + jnp.einsum("ijk,k->ij", features, parameters)
    elif method != "geometric_sinkhorn":
        raise ValueError(f"unknown coupling method: {method}")
    return _log_sinkhorn(logits, row_mass, column_mass)


def plan_diagnostics(plan: np.ndarray, cost: np.ndarray,
                     row_mass: np.ndarray, column_mass: np.ndarray) -> dict:
    safe = np.maximum(plan, 1e-300)
    return {
        "row_marginal_linf": float(np.max(np.abs(plan.sum(axis=1) - row_mass))),
        "column_marginal_linf": float(np.max(np.abs(plan.sum(axis=0) - column_mass))),
        "total_mass_error": float(abs(plan.sum() - 1.0)),
        "microscopic_cost": float(np.sum(plan * cost)),
        "entropy": float(-np.sum(plan * np.log(safe))),
        "sinkhorn_epsilon": SINKHORN_EPSILON,
        "sinkhorn_iterations": SINKHORN_ITERATIONS,
    }


def make_noise(seed: int, count: int, replicates: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (count, count, replicates, paper.N_PARTICLES, 2)
    noise = rng.normal(size=shape)
    noise -= noise.mean(axis=3, keepdims=True)
    radius = np.sqrt(np.mean(np.sum(noise * noise, axis=-1), axis=3))
    return noise / np.maximum(radius[..., None, None], 1e-12)


def precompute_statistics(raw: jax.Array, bank: EndpointBank, times: jax.Array,
                          noise: np.ndarray) -> FiberStatistics:
    minus = jnp.asarray(bank.minus)[:, None, None, :, :]
    plus = jnp.asarray(bank.plus)[None, :, None, :, :]
    noise_array = jnp.asarray(noise)
    pair_count = noise.shape[0] * noise.shape[1] * noise.shape[2]

    @jax.jit
    def one_time(t):
        gamma = paper.gamma_schedule(raw, t)
        gamma_dot = jax.grad(lambda value: paper.gamma_schedule(raw, value))(t)
        state = ((1.0 - t) * minus + t * plus + gamma * noise_array).reshape(
            pair_count, paper.N_PARTICLES, 2
        )
        velocity = (plus - minus + gamma_dot * noise_array).reshape(
            pair_count, paper.N_PARTICLES, 2
        )
        observables = paper.v_observables(state)
        observable_jacobian = paper.v_jphi(state)
        observable_velocity = jnp.einsum(
            "mrnd,mnd->mr", observable_jacobian, velocity
        )
        descriptors = paper.v_descriptors(state)
        descriptor_jacobian = paper.v_jdesc(state)
        descriptor_gram = jnp.einsum(
            "mknd,mlnd->mkl", descriptor_jacobian, descriptor_jacobian
        )
        return observables, observable_velocity, descriptors, descriptor_gram, paper.v_q4(state)

    rows = [one_time(t) for t in times]
    result = FiberStatistics(*(jnp.stack([row[i] for row in rows]) for i in range(5)))
    jax.block_until_ready(result.descriptor_gram)
    return result


def _weighted_tilt(log_base: jax.Array, lam: jax.Array, observables: jax.Array):
    weights = jax.nn.softmax(log_base + observables @ lam)
    moments = weights @ observables
    centered = observables - moments
    covariance = (centered.T * weights) @ centered
    return weights, moments, covariance


def _calibrate_weighted_primal(log_base: jax.Array, observables: jax.Array,
                               target: jax.Array) -> jax.Array:
    initial = jnp.zeros(target.shape[0], dtype=observables.dtype)

    def body(_, lam):
        _, moments, covariance = _weighted_tilt(log_base, lam, observables)
        step = paper._solve(covariance, moments - target, paper.CALIBRATION_RIDGE)
        norm = jnp.linalg.norm(step)
        return lam - step * jnp.minimum(1.0, 2.0 / jnp.maximum(norm, 1e-12))

    # The frozen many-body schedule can require multiplier norms above 100 on
    # small finite banks.  With the established trust cap of two per Newton
    # step, 40 iterations stop before the constraint root and invalidate an
    # implicit derivative.  A fixed 160 iterations reaches the same root for
    # every coupling without changing the objective or target.
    return jax.lax.fori_loop(0, 160, body, initial)


@jax.custom_jvp
def calibrate_weighted_implicit(log_base: jax.Array, observables: jax.Array,
                                target: jax.Array) -> jax.Array:
    return _calibrate_weighted_primal(log_base, observables, target)


@calibrate_weighted_implicit.defjvp
def _calibrate_weighted_jvp(primals, tangents):
    log_base, observables, target = primals
    dlog_base, dobservables, dtarget = tangents
    lam = _calibrate_weighted_primal(log_base, observables, target)
    weights, moments, covariance = _weighted_tilt(log_base, lam, observables)
    centered = observables - moments
    dlogit = dlog_base + jnp.sum(dobservables * lam[None, :], axis=-1)
    dconstraint = (
        weights @ dobservables
        + jnp.sum(weights[:, None] * centered * dlogit[:, None], axis=0)
        - dtarget
    )
    tangent = paper._solve(covariance, -dconstraint, paper.CALIBRATION_RIDGE)
    return lam, tangent


def plan_path_metrics(plan: jax.Array, statistics: FiberStatistics,
                      times: jax.Array, target: jax.Array) -> dict[str, jax.Array]:
    replicates = statistics.observables.shape[1] // plan.size
    base = jnp.repeat(plan.reshape(-1), replicates) / replicates
    log_base = jnp.log(jnp.maximum(base, 1e-300))

    def one(obs, obs_velocity, descriptors, descriptor_gram, q4):
        lam = calibrate_weighted_implicit(log_base, obs, target)
        weights, moments, covariance = _weighted_tilt(log_base, lam, obs)
        expected_velocity = weights @ obs_velocity
        scalar = obs_velocity @ lam
        covariance_term = jnp.sum(
            weights[:, None] * (obs - target) * scalar[:, None], axis=0
        )
        lambda_dot = paper._solve(
            covariance, -expected_velocity - covariance_term,
            paper.CALIBRATION_RIDGE,
        )
        forcing = ((obs - target) @ lambda_dot
                   + (obs_velocity - expected_velocity) @ lam)
        forcing = forcing - weights @ forcing
        gram = jnp.einsum("m,mkl->kl", weights, descriptor_gram)
        rhs = jnp.einsum("m,mk,m->k", weights, descriptors, forcing)
        coefficients = paper._solve(gram, rhs, paper.RITZ_RIDGE)
        energy = coefficients @ gram @ coefficients
        forcing_power = jnp.sum(weights * forcing * forcing)
        ess = 1.0 / jnp.sum(weights * weights / jnp.maximum(base, 1e-300))
        distortion = jnp.sum(weights * (jnp.log(jnp.maximum(weights, 1e-300)) - log_base))
        return energy, forcing_power, ess, distortion, jnp.linalg.norm(moments - target), weights @ q4

    energy, forcing, ess, distortion, moment_error, q4 = jax.vmap(one)(*statistics)
    return {
        "correction_energy": energy,
        "forcing_power": forcing,
        "ess": ess,
        "projection_distortion": distortion,
        "moment_error": moment_error,
        "q4": q4,
        "integrated_correction_energy": jnp.trapezoid(energy, times),
        "integrated_forcing_power": jnp.trapezoid(forcing, times),
        "integrated_projection_distortion": jnp.trapezoid(distortion, times),
        "minimum_ess": jnp.min(ess),
        "median_ess": jnp.median(ess),
        "maximum_moment_error": jnp.max(moment_error),
    }


def coupling_objective(parameters: jax.Array, cost: jax.Array, features: jax.Array,
                       statistics: FiberStatistics, times: jax.Array,
                       target: jax.Array, row_mass: jax.Array,
                       column_mass: jax.Array) -> jax.Array:
    plan = build_plan(
        "fiber_aware", cost, features, row_mass, column_mass, parameters,
    )
    metrics = plan_path_metrics(plan, statistics, times, target)
    ess_penalty = ESS_PENALTY * jnp.trapezoid(
        jax.nn.relu(ESS_FLOOR - metrics["ess"]) ** 2, times
    )
    return metrics["integrated_correction_energy"] + ess_penalty


def optimize_coupling(cost, features, statistics, validation_cost,
                      validation_features, validation_statistics, times, target,
                      row_mass, column_mass, validation_row_mass,
                      validation_column_mass,
                      steps: int):
    objective = jax.jit(lambda parameters: coupling_objective(
        parameters, cost, features, statistics, times, target,
        row_mass, column_mass,
    ))
    validation_objective = jax.jit(lambda parameters: coupling_objective(
        parameters, validation_cost, validation_features, validation_statistics,
        times, target, validation_row_mass, validation_column_mass,
    ))
    value_gradient = jax.jit(jax.value_and_grad(objective))
    parameters = jnp.zeros(features.shape[-1], dtype=jnp.float64)
    first = jnp.zeros_like(parameters)
    second = jnp.zeros_like(parameters)
    candidates = []
    trace = []
    start = time.perf_counter()
    for iteration in range(1, steps + 1):
        value, gradient = value_gradient(parameters)
        if iteration == 1 or iteration % 5 == 0 or iteration == steps:
            candidates.append(np.asarray(parameters))
            trace.append({"iteration": iteration - 1, "training_objective": float(value)})
        norm = jnp.linalg.norm(gradient)
        gradient = gradient * jnp.minimum(1.0, 10.0 / jnp.maximum(norm, 1e-12))
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient * gradient
        parameters = parameters - COUPLING_LEARNING_RATE * (
            first / (1.0 - 0.9**iteration)
        ) / (jnp.sqrt(second / (1.0 - 0.999**iteration)) + 1e-8)
    candidates.append(np.asarray(parameters))
    validation_values = [float(validation_objective(jnp.asarray(value))) for value in candidates]
    selected = int(np.argmin(validation_values))
    for row, validation_value in zip(trace, validation_values[:len(trace)]):
        row["validation_objective"] = validation_value
    jax.block_until_ready(parameters)
    return candidates[selected], trace, validation_values, selected, time.perf_counter() - start


def gradient_check(raw, seed: int) -> dict:
    """Reduced implicit/autodiff versus central finite-difference check."""
    populations = paper.build_physical_populations(seed + 70000, True)
    rng = np.random.default_rng(seed + 71000)
    minus_indices = rng.choice(len(populations["minus"]), size=7, replace=False)
    plus_indices = rng.choice(len(populations["plus"]), size=7, replace=False)
    uniform = np.full(7, 1.0 / 7.0)
    bank = EndpointBank(
        _canonicalize(populations["minus"][minus_indices]),
        _canonicalize(populations["plus"][plus_indices]),
        uniform, uniform, minus_indices, plus_indices,
    )
    times = jnp.asarray([0.49, 0.50, 0.51])
    cost = jnp.asarray(microscopic_cost(bank))
    features = jnp.asarray(coupling_features(bank))
    noise = make_noise(seed + 72000, len(bank.minus), 1)
    statistics = precompute_statistics(raw, bank, times, noise)
    geometric = build_plan(
        "geometric_sinkhorn", cost, features,
        jnp.asarray(bank.minus_weights), jnp.asarray(bank.plus_weights),
    )
    target = geometric.reshape(-1) @ statistics.observables[1]
    objective = jax.jit(lambda parameters: coupling_objective(
        parameters, cost, features, statistics, times, target,
        jnp.asarray(bank.minus_weights), jnp.asarray(bank.plus_weights),
    ))
    parameters = jnp.linspace(-0.025, 0.025, features.shape[-1])
    direction = jnp.cos(jnp.arange(features.shape[-1], dtype=jnp.float64) + 0.3)
    direction /= jnp.linalg.norm(direction)
    autodiff = float(jax.grad(objective)(parameters) @ direction)
    step = 1e-4
    finite = float((objective(parameters + step * direction)
                    - objective(parameters - step * direction)) / (2.0 * step))
    relative = abs(autodiff - finite) / max(abs(autodiff), abs(finite), 1e-10)
    return {
        "problem_endpoint_count": 7,
        "times": np.asarray(times).tolist(),
        "finite_difference_step": step,
        "autodiff_directional_derivative": autodiff,
        "central_finite_difference_directional_derivative": finite,
        "relative_error": relative,
        "lambda_differentiation": "custom implicit JVP; no stop-gradient",
    }


def sample_bridge_bank(bank: EndpointBank, plan: np.ndarray, seed: int,
                       times: np.ndarray, count: int):
    """IID pair sampling from a soft plan; the plan itself retains exact marginals."""
    rng = np.random.default_rng(seed)
    flat = rng.choice(plan.size, size=(len(times), count), p=plan.reshape(-1) / plan.sum())
    minus_index, plus_index = np.unravel_index(flat, plan.shape)
    minus = bank.minus[minus_index]
    plus = bank.plus[plus_index]
    noise = rng.normal(size=minus.shape)
    noise -= noise.mean(axis=2, keepdims=True)
    radius = np.sqrt(np.mean(np.sum(noise * noise, axis=-1), axis=2))
    noise /= np.maximum(radius[:, :, None, None], 1e-12)
    minus_frequency = np.bincount(minus_index.reshape(-1), minlength=len(bank.minus)) / flat.size
    plus_frequency = np.bincount(plus_index.reshape(-1), minlength=len(bank.plus)) / flat.size
    diagnostics = {
        "random_seed": int(seed),
        "sample_count": int(flat.size),
        "sampled_minus_total_variation": float(0.5 * np.sum(np.abs(minus_frequency - bank.minus_weights))),
        "sampled_plus_total_variation": float(0.5 * np.sum(np.abs(plus_frequency - bank.plus_weights))),
        "sampling": "iid categorical pairs from the full soft transport plan",
    }
    return tuple(jnp.asarray(value) for value in (minus, plus, noise)), diagnostics


def plan_for_method(method: str, bank: EndpointBank, parameters: np.ndarray | None):
    cost = microscopic_cost(bank)
    features = coupling_features(bank)
    plan = np.asarray(build_plan(
        method, jnp.asarray(cost), jnp.asarray(features),
        jnp.asarray(bank.minus_weights), jnp.asarray(bank.plus_weights),
        None if parameters is None else jnp.asarray(parameters),
    ))
    return plan, cost, features


def metric_payload(metrics: dict[str, jax.Array], times: jax.Array) -> dict:
    curve_names = (
        "correction_energy", "forcing_power", "ess", "projection_distortion",
        "moment_error", "q4",
    )
    return {
        "integrated_correction_energy": float(metrics["integrated_correction_energy"]),
        "integrated_forcing_power": float(metrics["integrated_forcing_power"]),
        "integrated_projection_distortion": float(metrics["integrated_projection_distortion"]),
        "minimum_ess": float(metrics["minimum_ess"]),
        "median_ess": float(metrics["median_ess"]),
        "maximum_moment_error": float(metrics["maximum_moment_error"]),
        "curves": [
            {"t": float(t), **{name: float(metrics[name][i]) for name in curve_names}}
            for i, t in enumerate(np.asarray(times))
        ],
    }


def _frozen_schedule(seed: int) -> tuple[np.ndarray, str]:
    path = FROZEN_SCHEDULE_ROOT / f"seed_{seed}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"frozen schedule for seed {seed} is missing: {path}; this study will not re-optimize it"
        )
    payload = json.loads(path.read_text())
    raw = np.asarray(payload["schedules"]["optimized_multi"]["raw"], dtype=np.float64)
    return raw, str(path.relative_to(ROOT))


def _bank_record(bank: EndpointBank, target: jax.Array) -> dict:
    minus_moments = bank.minus_weights @ np.asarray(
        paper.v_observables(jnp.asarray(bank.minus))
    )
    plus_moments = bank.plus_weights @ np.asarray(
        paper.v_observables(jnp.asarray(bank.plus))
    )
    return {
        "minus_source_indices": bank.minus_indices.tolist(),
        "plus_source_indices": bank.plus_indices.tolist(),
        "minus_weights": bank.minus_weights.tolist(),
        "plus_weights": bank.plus_weights.tolist(),
        "minus_moment_residual": float(np.linalg.norm(minus_moments - np.asarray(target))),
        "plus_moment_residual": float(np.linalg.norm(plus_moments - np.asarray(target))),
        "calibration_shared_across_methods": True,
        "endpoint_count": len(bank.minus),
    }


def run_seed(seed: int, quick: bool) -> dict:
    started = time.perf_counter()
    populations = paper.build_physical_populations(seed + 10000, quick)
    raw_np, schedule_source = _frozen_schedule(seed)
    raw = jnp.asarray(raw_np)
    target = jnp.asarray(populations["target"])
    times = jnp.asarray(np.linspace(0.12, 0.88, 4 if quick else 6))
    endpoint_count = 20 if quick else 48
    evaluation_count = 28 if quick else 64
    replicates = 2
    rng = np.random.default_rng(seed + 20000)
    optimization_bank = make_endpoint_bank(populations, rng, endpoint_count)
    validation_bank = make_endpoint_bank(populations, rng, endpoint_count)
    evaluation_bank = make_endpoint_bank(populations, rng, evaluation_count)

    print(f"[coupling] seed {seed}: precomputing optimization/validation fibers", flush=True)
    optimization_cost = jnp.asarray(microscopic_cost(optimization_bank))
    optimization_features = jnp.asarray(coupling_features(optimization_bank))
    optimization_stats = precompute_statistics(
        raw, optimization_bank, times,
        make_noise(seed + 21000, endpoint_count, replicates),
    )
    validation_cost = jnp.asarray(microscopic_cost(validation_bank))
    validation_features = jnp.asarray(coupling_features(validation_bank))
    validation_stats = precompute_statistics(
        raw, validation_bank, times,
        make_noise(seed + 22000, endpoint_count, replicates),
    )
    optimization_steps = 20 if quick else 60
    parameters, trace, validation_values, selected, optimization_seconds = optimize_coupling(
        optimization_cost, optimization_features, optimization_stats,
        validation_cost, validation_features, validation_stats, times, target,
        jnp.asarray(optimization_bank.minus_weights),
        jnp.asarray(optimization_bank.plus_weights),
        jnp.asarray(validation_bank.minus_weights),
        jnp.asarray(validation_bank.plus_weights),
        optimization_steps,
    )

    print(f"[coupling] seed {seed}: independent evaluation", flush=True)
    evaluation_cost = jnp.asarray(microscopic_cost(evaluation_bank))
    evaluation_features = jnp.asarray(coupling_features(evaluation_bank))
    evaluation_stats = precompute_statistics(
        raw, evaluation_bank, times,
        make_noise(seed + 23000, evaluation_count, replicates),
    )
    methods = {}
    for method in METHODS:
        plan = build_plan(
            method, evaluation_cost, evaluation_features,
            jnp.asarray(evaluation_bank.minus_weights),
            jnp.asarray(evaluation_bank.plus_weights),
            jnp.asarray(parameters) if method == "fiber_aware" else None,
        )
        metrics = plan_path_metrics(plan, evaluation_stats, times, target)
        jax.block_until_ready(metrics["integrated_correction_energy"])
        plan_np = np.asarray(plan)
        methods[method] = {
            **metric_payload(metrics, times),
            "plan": plan_diagnostics(
                plan_np, np.asarray(evaluation_cost),
                evaluation_bank.minus_weights, evaluation_bank.plus_weights,
            ),
        }

    # Established random-continuous-time neural training, with all settings
    # copied from the quick/standard paper protocol and shared across methods.
    neural_endpoint_count = 24 if quick else 48
    neural_rng = np.random.default_rng(seed + 30000)
    neural_bank = make_endpoint_bank(populations, neural_rng, neural_endpoint_count)
    gate_bank = make_endpoint_bank(populations, neural_rng, neural_endpoint_count)
    generation_bank = make_endpoint_bank(populations, neural_rng, 24 if quick else 64)
    oracle_bank = make_endpoint_bank(populations, neural_rng, 28 if quick else 64)
    continuous_time_count = 8 if quick else 18
    continuous_rng = np.random.default_rng(seed + 31000)
    strata = np.arange(continuous_time_count) + continuous_rng.uniform(size=continuous_time_count)
    continuous_times = jnp.asarray(0.12 + 0.76 * strata / continuous_time_count)
    particles_per_time = 48 if quick else 64
    correction_steps = 180 if quick else 420
    evaluation_times = jnp.asarray([0.25, 0.50, 0.75, 1.0])
    integration_steps = 12 if quick else 24

    for method in METHODS:
        print(f"[coupling] seed {seed}: neural downstream {method}", flush=True)
        method_parameters = parameters if method == "fiber_aware" else None
        train_plan, train_cost, _ = plan_for_method(method, neural_bank, method_parameters)
        train_samples, train_sampling = sample_bridge_bank(
            neural_bank, train_plan, seed + 32000, np.asarray(continuous_times),
            particles_per_time,
        )
        model, training_trace, training_seconds = paper.train_neural_correction(
            jax.random.PRNGKey(seed), raw, train_samples, continuous_times, target,
            correction_steps,
        )
        gate_plan, _, _ = plan_for_method(method, gate_bank, method_parameters)
        gate_samples, gate_sampling = sample_bridge_bank(
            gate_bank, gate_plan, seed + 33000, np.asarray(times),
            96 if quick else 384,
        )
        gate, gate_gain, gate_se = paper.select_gate(
            model, raw, gate_samples, times, target
        )
        generation_plan, _, _ = plan_for_method(method, generation_bank, method_parameters)
        generation_samples, generation_sampling = sample_bridge_bank(
            generation_bank, generation_plan, seed + 34000,
            np.asarray([0.5]), 32 if quick else 64,
        )
        oracle_plan, _, _ = plan_for_method(method, oracle_bank, method_parameters)
        oracle_samples, oracle_sampling = sample_bridge_bank(
            oracle_bank, oracle_plan, seed + 35000,
            np.asarray(evaluation_times), 128 if quick else 256,
        )
        generated, integration_seconds, nfe = paper.integrate_method(
            "neural", model, gate, raw,
            generation_samples[0][0], generation_samples[1][0],
            generation_samples[2][0], target, integration_steps, evaluation_times,
        )
        rows = paper.evaluate_generated(
            generated, oracle_samples, raw, evaluation_times, target
        )
        methods[method]["neural_downstream"] = {
            "training_protocol": "stratified random continuous time",
            "architecture": "two-hidden-layer width-18 invariant radial-descriptor MLP",
            "optimizer": "Adam, learning rate 0.005, gradient-norm cap 5",
            "training_times": np.asarray(continuous_times).tolist(),
            "particles_per_time": particles_per_time,
            "optimizer_steps": correction_steps,
            "training_initial_loss": training_trace[0],
            "training_final_loss": training_trace[-1],
            "training_seconds": training_seconds,
            "gate": gate,
            "gate_gain": gate_gain,
            "gate_standard_error": gate_se,
            "integration_steps": integration_steps,
            "ode_solver": "fixed-step Heun",
            "nfe": nfe,
            "integration_seconds": integration_seconds,
            "projected_law_mmd2": paper.interior_mmd2(rows),
            "maximum_moment_error": float(max(row["moment_error"] for row in rows)),
            "rows": rows,
            "sampling_diagnostics": {
                "training": train_sampling,
                "gate": gate_sampling,
                "generation": generation_sampling,
                "oracle": oracle_sampling,
            },
            "train_plan": plan_diagnostics(
                train_plan, train_cost, neural_bank.minus_weights,
                neural_bank.plus_weights,
            ),
        }

    return {
        "seed": seed,
        "wall_seconds": time.perf_counter() - started,
        "fixed_schedule": {
            "raw": raw_np.tolist(),
            "source": schedule_source,
            "reoptimized": False,
        },
        "endpoint": {
            "target": np.asarray(target).tolist(),
            "minus_calibration_residual": populations["minus_residual"],
            "plus_calibration_residual": populations["plus_residual"],
            "minus_q4": populations["minus_q4"],
            "plus_q4": populations["plus_q4"],
        },
        "banks": {
            "coupling_optimization": _bank_record(optimization_bank, target),
            "coupling_validation": _bank_record(validation_bank, target),
            "final_evaluation": _bank_record(evaluation_bank, target),
            "neural_training": _bank_record(neural_bank, target),
            "neural_gate": _bank_record(gate_bank, target),
            "neural_generation": _bank_record(generation_bank, target),
            "neural_oracle": _bank_record(oracle_bank, target),
        },
        "fiber_optimization": {
            "parameters": np.asarray(parameters).tolist(),
            "parameterization": "geometric log-kernel plus 3x3 measured-observable interactions",
            "hidden_q4_used": False,
            "final_test_mmd_used": False,
            "steps": optimization_steps,
            "learning_rate": COUPLING_LEARNING_RATE,
            "ess_penalty": ESS_PENALTY,
            "ess_floor": ESS_FLOOR,
            "trace": trace,
            "validation_objectives": validation_values,
            "selected_candidate_index": selected,
            "initial_validation_objective": validation_values[0],
            "selected_validation_objective": validation_values[selected],
            "wall_seconds": optimization_seconds,
        },
        "methods": methods,
    }


def mean_ci(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if len(values) < 2:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean, "n": len(values)}
    critical = {5: 2.776}.get(len(values), 1.96)
    half = critical * float(np.std(values, ddof=1)) / math.sqrt(len(values))
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half, "n": len(values)}


def aggregate(seed_reports: list[dict]) -> dict:
    metric_paths = {
        "integrated_correction_energy": lambda row, method: row["methods"][method]["integrated_correction_energy"],
        "minimum_ess": lambda row, method: row["methods"][method]["minimum_ess"],
        "integrated_projection_distortion": lambda row, method: row["methods"][method]["integrated_projection_distortion"],
        "projected_law_mmd2": lambda row, method: row["methods"][method]["neural_downstream"]["projected_law_mmd2"],
        "maximum_moment_error": lambda row, method: row["methods"][method]["neural_downstream"]["maximum_moment_error"],
        "generated_q4_change": lambda row, method: (
            row["methods"][method]["neural_downstream"]["rows"][-1]["q4"]
            - row["methods"][method]["neural_downstream"]["rows"][0]["q4"]
        ),
        "oracle_q4_change": lambda row, method: (
            row["methods"][method]["neural_downstream"]["rows"][-1]["oracle_q4"]
            - row["methods"][method]["neural_downstream"]["rows"][0]["oracle_q4"]
        ),
        "microscopic_cost": lambda row, method: row["methods"][method]["plan"]["microscopic_cost"],
    }
    methods = {
        method: {
            metric: mean_ci([getter(row, method) for row in seed_reports])
            for metric, getter in metric_paths.items()
        }
        for method in METHODS
    }
    contrasts = {}
    pairs = (
        ("fiber_minus_independent", "fiber_aware", "independent"),
        ("geometric_minus_independent", "geometric_sinkhorn", "independent"),
        ("fiber_minus_geometric", "fiber_aware", "geometric_sinkhorn"),
    )
    for contrast, left, right in pairs:
        contrasts[contrast] = {
            metric: mean_ci([
                getter(row, left) - getter(row, right) for row in seed_reports
            ])
            for metric, getter in metric_paths.items()
        }
    maximum_marginal_error = max(
        max(method["plan"]["row_marginal_linf"], method["plan"]["column_marginal_linf"])
        for row in seed_reports for method in row["methods"].values()
    )
    maximum_endpoint_moment_residual = max(
        max(bank[side] for side in ("minus_moment_residual", "plus_moment_residual"))
        for row in seed_reports for bank in row["banks"].values()
    )
    selection_objective_change = mean_ci([
        row["fiber_optimization"]["validation_objectives"][
            row["fiber_optimization"]["selected_candidate_index"]
        ] - row["fiber_optimization"]["validation_objectives"][0]
        for row in seed_reports
    ])
    fiber_geo = contrasts["fiber_minus_geometric"]["integrated_correction_energy"]
    independent_path_supported = (
        fiber_geo["ci95_high"] < 0.0 if len(seed_reports) > 1 else fiber_geo["mean"] < 0.0
    )
    return {
        "methods": methods,
        "paired_contrasts": contrasts,
        "endpoint_hidden_q4_gap": mean_ci([
            row["endpoint"]["plus_q4"] - row["endpoint"]["minus_q4"]
            for row in seed_reports
        ]),
        "checks": {
            "maximum_plan_marginal_linf": maximum_marginal_error,
            "endpoint_marginals_preserved": maximum_marginal_error < 5e-8,
            "maximum_finite_endpoint_moment_residual": maximum_endpoint_moment_residual,
            "finite_endpoint_moments_calibrated": maximum_endpoint_moment_residual < 1e-7,
            "same_frozen_schedule_within_each_seed": True,
            "test_mmd_used_for_training_or_selection": False,
        },
        "interpretation": {
            "fiber_selection_objective_change": selection_objective_change,
            "fiber_aware_beats_geometric_in_correction_burden": independent_path_supported,
            "criterion": "paired 95% interval below zero (single quick seed: observed effect below zero)",
        },
    }


def write_csv(summary: dict, output: Path) -> None:
    rows = []
    for seed in summary["seed_reports"]:
        for method in METHODS:
            row = seed["methods"][method]
            rows.append({
                "seed": seed["seed"], "method": method,
                "integrated_correction_energy": row["integrated_correction_energy"],
                "minimum_ess": row["minimum_ess"],
                "median_ess": row["median_ess"],
                "integrated_projection_distortion": row["integrated_projection_distortion"],
                "projected_law_mmd2": row["neural_downstream"]["projected_law_mmd2"],
                "maximum_moment_error": row["neural_downstream"]["maximum_moment_error"],
                "generated_q4_change": (
                    row["neural_downstream"]["rows"][-1]["q4"]
                    - row["neural_downstream"]["rows"][0]["q4"]
                ),
                "oracle_q4_change": (
                    row["neural_downstream"]["rows"][-1]["oracle_q4"]
                    - row["neural_downstream"]["rows"][0]["oracle_q4"]
                ),
                "microscopic_cost": row["plan"]["microscopic_cost"],
            })
    with (output / "coupling_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(summary: dict, output: Path) -> None:
    plt.rcParams.update({
        "figure.facecolor": "#f4f1ea", "axes.facecolor": "#fffdf8",
        "axes.grid": True, "grid.alpha": 0.2,
    })
    colors = ["#457b9d", "#e9c46a", "#2a9d8f"]
    metrics = (
        "integrated_correction_energy", "minimum_ess",
        "integrated_projection_distortion", "projected_law_mmd2",
    )
    titles = ("Correction burden", "Minimum ESS", "Projection distortion", "Projected-law MMD²")
    figure, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    for ax, metric, title in zip(axes, metrics, titles):
        stats = [summary["aggregate"]["methods"][method][metric] for method in METHODS]
        means = [value["mean"] for value in stats]
        errors = [
            [mean - value["ci95_low"] for mean, value in zip(means, stats)],
            [value["ci95_high"] - mean for mean, value in zip(means, stats)],
        ]
        ax.bar([METHOD_LABELS[m] for m in METHODS], means, color=colors, yerr=errors, capsize=4)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=18)
    figure.suptitle("Frozen-schedule endpoint coupling comparison", fontweight="bold")
    figure.savefig(output / "coupling_summary.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    representative = summary["seed_reports"][0]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for method, color in zip(METHODS, colors):
        curves = representative["methods"][method]["curves"]
        times = [row["t"] for row in curves]
        axes[0].plot(times, [row["correction_energy"] for row in curves], "o-", color=color, label=METHOD_LABELS[method])
        axes[1].plot(times, [row["ess"] for row in curves], "o-", color=color, label=METHOD_LABELS[method])
        axes[2].plot(times, [row["projection_distortion"] for row in curves], "o-", color=color, label=METHOD_LABELS[method])
    axes[0].set(title="Correction energy", xlabel="time")
    axes[1].set(title="Relative ESS", xlabel="time", ylim=(0, 1.04))
    axes[1].axhline(ESS_FLOOR, color="black", linestyle="--", linewidth=1)
    axes[2].set(title="Projection distortion", xlabel="time")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    figure.savefig(output / "coupling_time_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    contrast = summary["aggregate"]["paired_contrasts"]["fiber_minus_geometric"]
    figure, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    contrast_metrics = list(metrics)
    values = [contrast[metric]["mean"] for metric in contrast_metrics]
    errors = [[
        value - contrast[metric]["ci95_low"] for value, metric in zip(values, contrast_metrics)
    ], [
        contrast[metric]["ci95_high"] - value for value, metric in zip(values, contrast_metrics)
    ]]
    ax.errorbar(values, np.arange(len(values)), xerr=errors, fmt="o", color="#2a9d8f", capsize=4)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_yticks(np.arange(len(values)), ["E_corr", "ESS_min", "D_proj", "MMD²"])
    ax.set(title="Paired effects: fiber-aware minus geometric OT", xlabel="paired difference")
    figure.savefig(output / "coupling_paired_effects.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    # Plans are small enough to show without implying particle trajectories.
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    for ax, method in zip(axes, METHODS):
        # Reconstructing exact matrices in JSON would be bulky; show the
        # method-level entropy/cost point as a compact plan diagnostic.
        rows = [seed["methods"][method]["plan"] for seed in summary["seed_reports"]]
        ax.scatter([row["microscopic_cost"] for row in rows], [row["entropy"] for row in rows], color=colors[METHODS.index(method)], s=50)
        ax.set(title=METHOD_LABELS[method], xlabel="geometric cost", ylabel="plan entropy")
    figure.savefig(output / "coupling_plan_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_report(summary: dict, output: Path) -> None:
    aggregate = summary["aggregate"]
    fiber_geo = aggregate["paired_contrasts"]["fiber_minus_geometric"]
    lines = [
        "# MFSI Stage 2: fiber-adapted coupling (coupling only)", "",
        "## Method", "",
        f"The paper endpoint-population construction and {len(summary['seed_reports'])} previously selected per-bank schedule(s) were reused. "
        "Only the endpoint transport plan changed. Independent pairing, conventional geometric Sinkhorn, "
        "and a fiber-aware Sinkhorn relaxation were evaluated on matched banks. The fiber plan used a "
        "geometric log-kernel plus nine measured-observable interaction parameters and was optimized for "
        "integrated exact finite-bank correction energy plus an ESS-floor penalty.", "",
        "The fiber parameters were trained on a coupling-optimization bank, selected by the same objective "
        "on a validation bank, and then applied to an untouched evaluation bank. Final MMD² and hidden q4 "
        "were absent from training and selection. Soft plans were passed downstream by IID categorical pair "
        "sampling; plan marginal residuals and finite-sample marginal deviations are reported separately.", "",
        "The microscopic cost centers each configuration, uses periodic minimum-image displacements, and "
        "solves the particle-exchange assignment with the Hungarian algorithm. It contains no moment-fiber, "
        "ESS, correction-energy, MMD, or q4 term.", "",
        "## Frozen quantities", "",
        "Schedule parameters, endpoint source populations, target moments, observables, invariant neural "
        "architecture, random-continuous-time training, optimizer settings, gate procedure, Heun step count, "
        "and MMD feature map were identical across coupling methods within each bank.", "",
        "## Validation", "",
        f"The reduced implicit-gradient check had relative error `{summary['gradient_validation']['relative_error']:.3e}`. "
        f"The largest endpoint-plan marginal L-infinity residual was "
        f"`{aggregate['checks']['maximum_plan_marginal_linf']:.3e}`; the largest calibrated finite-endpoint "
        f"moment residual was `{aggregate['checks']['maximum_finite_endpoint_moment_residual']:.3e}`.", "",
        "## Aggregate metrics", "",
        "| method | E_corr | min ESS | D_proj | MMD² | max moment error | generated Δq4 | microscopic cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = aggregate["methods"][method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {row['integrated_correction_energy']['mean']:.6g} | "
            f"{row['minimum_ess']['mean']:.6g} | {row['integrated_projection_distortion']['mean']:.6g} | "
            f"{row['projected_law_mmd2']['mean']:.6g} | {row['maximum_moment_error']['mean']:.6g} | "
            f"{row['generated_q4_change']['mean']:.6g} | {row['microscopic_cost']['mean']:.6g} |"
        )
    effect = fiber_geo["integrated_correction_energy"]
    selection = aggregate["interpretation"]["fiber_selection_objective_change"]
    lines.extend([
        "", "## Primary paired contrast", "",
        f"Fiber-aware minus geometric OT correction energy was `{effect['mean']:.6g}` "
        f"(95% interval `{effect['ci95_low']:.6g}` to `{effect['ci95_high']:.6g}`, n={effect['n']}).",
        "",
        f"On the separate selection banks, the selected fiber parameterization changed its own objective "
        f"by `{selection['mean']:.6g}` relative to the geometric initialization "
        f"(95% interval `{selection['ci95_low']:.6g}` to `{selection['ci95_high']:.6g}`).",
        "",
        "The trained-objective result and the independent-path claim are intentionally separate. "
        + (
            "The five-bank paired interval supports lower fiber-aware correction burden than geometric OT."
            if len(summary["seed_reports"]) > 1 and aggregate["interpretation"]["fiber_aware_beats_geometric_in_correction_burden"]
            else "The one-bank quick result is directional only and cannot establish a replicated coupling effect."
            if len(summary["seed_reports"]) == 1
            else "The five-bank evaluation does not establish lower fiber-aware correction burden than geometric OT."
        ), "",
        f"For transfer metrics, fiber-aware minus geometric OT minimum ESS was "
        f"`{fiber_geo['minimum_ess']['mean']:.6g}` (95% interval "
        f"`{fiber_geo['minimum_ess']['ci95_low']:.6g}` to `{fiber_geo['minimum_ess']['ci95_high']:.6g}`), "
        f"projection distortion was `{fiber_geo['integrated_projection_distortion']['mean']:.6g}` "
        f"(`{fiber_geo['integrated_projection_distortion']['ci95_low']:.6g}` to "
        f"`{fiber_geo['integrated_projection_distortion']['ci95_high']:.6g}`), and projected-law MMD² was "
        f"`{fiber_geo['projected_law_mmd2']['mean']:.6g}` "
        f"(`{fiber_geo['projected_law_mmd2']['ci95_low']:.6g}` to "
        f"`{fiber_geo['projected_law_mmd2']['ci95_high']:.6g}`).",
        "",
        f"The calibrated source endpoint q4 gap was "
        f"`{aggregate['endpoint_hidden_q4_gap']['mean']:.6g}` on average. Generated q4 changes from the "
        f"first interior evaluation time to t=1 were "
        + ", ".join(
            f"{METHOD_LABELS[method]} `{aggregate['methods'][method]['generated_q4_change']['mean']:.6g}`"
            for method in METHODS
        ) + "; q4 remained held out from coupling and network objectives.",
        "", "## Limitations", "",
        "The transport plans live on finite resampled endpoint banks, and downstream paired banks are IID "
        "samples from soft plans, so their realized empirical marginals fluctuate even though the underlying "
        "plans satisfy the marginals numerically. Neural end-to-end limitations remain visible; no method was "
        "tuned using final MMD². This is Stage 2 only: schedule parameters were not updated and no joint "
        "schedule-coupling optimization was implemented.", "",
    ])
    (output / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="one-seed plumbing/evaluation run")
    parser.add_argument("--seeds", default=None, help="space-separated seed list")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--aggregate-existing", action="store_true")
    args = parser.parse_args()
    seeds = ([int(value) for value in args.seeds.split()] if args.seeds
             else ([401] if args.quick else [401, 402, 403, 404, 405]))
    output = args.output_dir / ("quick" if args.quick else "standard")
    output.mkdir(parents=True, exist_ok=True)
    seed_reports = []
    started = time.perf_counter()
    for seed in seeds:
        path = output / f"seed_{seed}.json"
        if args.aggregate_existing:
            seed_report = json.loads(path.read_text())
            optimization = seed_report["fiber_optimization"]
            optimization.setdefault(
                "initial_validation_objective",
                optimization["validation_objectives"][0],
            )
            optimization.setdefault(
                "selected_validation_objective",
                optimization["validation_objectives"][
                    optimization["selected_candidate_index"]
                ],
            )
        else:
            seed_report = run_seed(seed, args.quick)
        path.write_text(json.dumps(seed_report, indent=2) + "\n")
        seed_reports.append(seed_report)
    raw, _ = _frozen_schedule(seeds[0])
    gradient_validation = gradient_check(jnp.asarray(raw), seeds[0])
    summary = {
        "experiment": "fiber-adapted-coupling-coupling-only",
        "stage": 2,
        "joint_schedule_coupling_optimization": False,
        "mode": "quick" if args.quick else "standard",
        "seeds": seeds,
        "wall_seconds": time.perf_counter() - started,
        "configuration": {
            "sinkhorn_epsilon": SINKHORN_EPSILON,
            "sinkhorn_iterations": SINKHORN_ITERATIONS,
            "coupling_learning_rate": COUPLING_LEARNING_RATE,
            "ess_penalty": ESS_PENALTY,
            "ess_floor": ESS_FLOOR,
            "hyperparameter_policy": "predeclared from numerical stability and the established ESS floor; no final-MMD selection",
        },
        "gradient_validation": gradient_validation,
        "seed_reports": seed_reports,
        "aggregate": aggregate(seed_reports),
    }
    summary["aggregate"]["checks"]["gradient_validation_passed"] = (
        gradient_validation["relative_error"] < 5e-5
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(summary, output)
    write_report(summary, output)
    if not args.no_plots:
        make_plots(summary, output)
    print(json.dumps({
        "gradient_relative_error": gradient_validation["relative_error"],
        "checks": summary["aggregate"]["checks"],
        "fiber_minus_geometric_Ecorr": summary["aggregate"]["paired_contrasts"]["fiber_minus_geometric"]["integrated_correction_energy"],
    }, indent=2))
    print(f"outputs: {output}")


if __name__ == "__main__":
    main()
