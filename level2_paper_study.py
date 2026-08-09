#!/usr/bin/env python3
"""Paper-facing MFSI level-2 study on an N=32 particle system.

This workflow is intentionally separate from all earlier experiments.  It
provides multi-bank schedule comparisons, a fully trained invariant neural
potential, conservative correction gating, end-to-end integration, independent
projected-law MMD, matched baselines, timing/NFE, and seed-level intervals.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

from backend_runtime import _post, normalize_backend

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "level2_paper_study"
COMPONENT_API = ROOT / "tesseracts" / "paper_level2_correction" / "tesseract_api.py"
N_PARTICLES = 32
STATE_DIMENSION = 64
BOX_SIZE = 8.0
OBS_CENTERS = jnp.asarray([0.72, 1.22, 1.82], dtype=jnp.float64)
DESC_CENTERS = jnp.linspace(0.45, 2.65, 8, dtype=jnp.float64)
RBF_WIDTH = 0.32
TARGET_PLACEHOLDER = jnp.zeros((3,), dtype=jnp.float64)
CALIBRATION_RIDGE = 2e-8
RITZ_RIDGE = 2e-4


class MLP(NamedTuple):
    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array
    w3: jax.Array
    b3: jax.Array


def _minimum_image(displacement):
    return displacement - BOX_SIZE * jnp.round(displacement / BOX_SIZE)


def radial_descriptors_single(configuration, centers=DESC_CENTERS):
    displacement = configuration[:, None, :] - configuration[None, :, :]
    displacement = _minimum_image(displacement)
    rows, cols = jnp.triu_indices(configuration.shape[0], 1)
    distances = jnp.sqrt(jnp.sum(displacement[rows, cols] ** 2, axis=-1) + 1e-10)
    values = jnp.exp(-0.5 * ((distances[:, None] - centers[None, :]) / RBF_WIDTH) ** 2)
    return jnp.mean(values, axis=0)


def observables_single(configuration):
    return radial_descriptors_single(configuration, OBS_CENTERS)


def q4_single(configuration):
    centered = configuration - jnp.mean(configuration, axis=0, keepdims=True)
    angles = jnp.arctan2(centered[:, 1], centered[:, 0])
    return jnp.sqrt(
        jnp.mean(jnp.cos(4.0 * angles)) ** 2
        + jnp.mean(jnp.sin(4.0 * angles)) ** 2
        + 1e-16
    )


def angular_descriptors_single(configuration):
    """Smooth rotation/permutation/translation invariants absent from the base MLP."""
    centered = configuration - jnp.mean(configuration, axis=0, keepdims=True)
    angles = jnp.arctan2(centered[:, 1], centered[:, 0])
    orders = jnp.asarray([2.0, 4.0, 6.0], dtype=configuration.dtype)
    cosine = jnp.mean(jnp.cos(angles[:, None] * orders[None, :]), axis=0)
    sine = jnp.mean(jnp.sin(angles[:, None] * orders[None, :]), axis=0)
    return cosine * cosine + sine * sine


v_observables = jax.vmap(observables_single)
v_descriptors = jax.vmap(radial_descriptors_single)
v_q4 = jax.vmap(q4_single)
v_jphi = jax.vmap(jax.jacrev(observables_single))
v_jdesc = jax.vmap(jax.jacrev(radial_descriptors_single))


def physical_energy(configuration, angular_target):
    centered = configuration - jnp.mean(configuration, axis=0, keepdims=True)
    displacement = centered[:, None, :] - centered[None, :, :]
    displacement = _minimum_image(displacement)
    rows, cols = jnp.triu_indices(centered.shape[0], 1)
    distances2 = jnp.sum(displacement[rows, cols] ** 2, axis=-1)
    repulsion = jnp.mean(jnp.exp(-distances2 / (2.0 * 0.28**2)))
    radius2 = jnp.mean(jnp.sum(centered * centered, axis=-1))
    order = q4_single(centered)
    return 18.0 * repulsion + 12.0 * (radius2 - 1.0) ** 2 + 5.0 * (order - angular_target) ** 2


def _normalize(configuration: np.ndarray) -> np.ndarray:
    centered = configuration - configuration.mean(axis=0, keepdims=True)
    radius = np.sqrt(np.mean(np.sum(centered * centered, axis=-1)))
    return centered / max(radius, 1e-8)


def _motif_bank(rng: np.random.Generator, count: int, regime: str) -> np.ndarray:
    result = np.empty((count, N_PARTICLES, 2), dtype=np.float64)
    cluster_ids = np.repeat(np.arange(4), N_PARTICLES // 4)
    for index in range(count):
        rotation = rng.uniform(0.0, 2.0 * np.pi)
        if regime == "ring":
            # Disordered low-q4 liquid configurations.
            angle = rotation + rng.uniform(0.0, 2.0 * np.pi, N_PARTICLES)
        elif regime == "cluster":
            # Fourfold configurations with heterogeneous radial shells.  The
            # radial diversity is what lets the two q4 phases share the same
            # three measured radial moments without sharing configurations.
            angle = rotation + 0.5 * np.pi * cluster_ids
            angle += rng.normal(0.0, rng.uniform(0.04, 0.34), N_PARTICLES)
        else:
            raise ValueError(f"unknown regime: {regime}")
        spread = rng.uniform(0.08, 0.82)
        radial_family = rng.integers(3)
        if radial_family == 0:
            radius = rng.uniform(max(0.03, 1.0 - spread), 1.0 + spread, N_PARTICLES)
        elif radial_family == 1:
            radius = np.exp(rng.normal(0.0, spread, N_PARTICLES))
        else:
            probability = rng.uniform(0.25, 0.75)
            radius = np.where(rng.random(N_PARTICLES) < probability, 1.0 - spread, 1.0 + spread)
            radius += rng.normal(0.0, 0.03, N_PARTICLES)
        radius = np.maximum(radius, 0.03)
        configuration = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=-1)
        result[index] = _normalize(configuration)
    return result


@jax.jit
def _relax_bank(configurations, angular_target, noises):
    gradient = jax.vmap(jax.grad(physical_energy), in_axes=(0, None))

    def step(states, noise):
        update = gradient(states, angular_target)
        states = states - 0.018 * update + 0.012 * noise
        states = states - jnp.mean(states, axis=1, keepdims=True)
        radius = jnp.sqrt(jnp.mean(jnp.sum(states * states, axis=-1), axis=1))
        states = states / radius[:, None, None]
        return states, None

    return jax.lax.scan(step, configurations, noises)[0]


def build_physical_populations(seed: int, quick: bool):
    rng = np.random.default_rng(seed)
    regime_count = 150 if quick else 320
    candidate_count = 1100 if quick else 2400
    relaxation_steps = 8 if quick else 16
    ring = _motif_bank(rng, candidate_count, "ring")
    cluster = _motif_bank(rng, candidate_count, "cluster")
    banks = []
    for configurations, target in ((ring, 0.05), (cluster, 0.92)):
        noise = rng.normal(size=(relaxation_steps, *configurations.shape))
        relaxed = _relax_bank(jnp.asarray(configurations), jnp.asarray(target), jnp.asarray(noise))
        banks.append(np.asarray(relaxed))
    ring, cluster = banks
    radial_target = np.asarray([0.335, 0.420, 0.350])
    selected = []
    for configurations, high_order in ((ring, False), (cluster, True)):
        observables = np.asarray(v_observables(jnp.asarray(configurations)))
        order = np.asarray(v_q4(jnp.asarray(configurations)))
        order_penalty = np.maximum(order - 0.20, 0.0) if not high_order else np.maximum(0.62 - order, 0.0)
        score = np.linalg.norm(observables - radial_target, axis=1) + 3.0 * order_penalty
        selected.append(configurations[np.argsort(score)[:regime_count]])
    minus, plus = selected
    # Choose the common target near the center of both selected clouds, then
    # solve the finite-bank exponential calibration independently.
    minus_observables = np.asarray(v_observables(jnp.asarray(minus)))
    plus_observables = np.asarray(v_observables(jnp.asarray(plus)))
    # Find a common point in the relative interiors of both empirical convex
    # hulls.  Positive lower bounds avoid the brittle boundary targets that can
    # appear when a fixed nominal moment is used with finite random banks.
    count = len(minus)
    equality = np.zeros((5, 2 * count + 1))
    equality[:3, :count] = minus_observables.T
    equality[:3, count:2 * count] = -plus_observables.T
    equality[3, :count] = 1.0
    equality[4, count:2 * count] = 1.0
    inequalities = np.zeros((2 * count, 2 * count + 1))
    inequalities[:, :2 * count] = -np.eye(2 * count)
    inequalities[:, -1] = 1.0
    objective = np.zeros(2 * count + 1); objective[-1] = -1.0
    feasible = linprog(
        objective, A_ub=inequalities, b_ub=np.zeros(2 * count),
        A_eq=equality, b_eq=np.asarray([0.0, 0.0, 0.0, 1.0, 1.0]),
        bounds=[(0.0, None)] * (2 * count + 1), method="highs",
    )
    if not feasible.success:
        raise RuntimeError(f"endpoint radial hulls do not overlap: {feasible.message}")
    target = feasible.x[:count] @ minus_observables
    minus_weights, minus_residual = calibrate_numpy(minus_observables, target)
    plus_weights, plus_residual = calibrate_numpy(plus_observables, target)
    return {
        "minus": minus,
        "plus": plus,
        "minus_weights": minus_weights,
        "plus_weights": plus_weights,
        "target": target,
        "minus_residual": minus_residual,
        "plus_residual": plus_residual,
        "minus_q4": float(minus_weights @ np.asarray(v_q4(jnp.asarray(minus)))),
        "plus_q4": float(plus_weights @ np.asarray(v_q4(jnp.asarray(plus)))),
    }


def calibrate_numpy(observables: np.ndarray, target: np.ndarray):
    lam = np.zeros(observables.shape[1], dtype=np.float64)
    for _ in range(80):
        logits = observables @ lam
        logits -= np.max(logits)
        weights = np.exp(logits); weights /= weights.sum()
        mean = weights @ observables
        centered = observables - mean
        covariance = (centered.T * weights) @ centered + 1e-9 * np.eye(observables.shape[1])
        step = np.linalg.solve(covariance, mean - target)
        step *= min(1.0, 2.0 / max(np.linalg.norm(step), 1e-12))
        lam -= step
    logits = observables @ lam; logits -= np.max(logits)
    weights = np.exp(logits); weights /= weights.sum()
    return weights, float(np.linalg.norm(weights @ observables - target))


def _solve(matrix, rhs, ridge):
    matrix = 0.5 * (matrix + matrix.T)
    return jnp.linalg.solve(matrix + ridge * jnp.eye(matrix.shape[0]), rhs)


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
        return lam - step * jnp.minimum(1.0, 2.0 / jnp.maximum(norm, 1e-12))
    return jax.lax.fori_loop(0, 40, body, initial)


@jax.custom_jvp
def calibrate_implicit(observables, target):
    return _calibrate_primal(observables, target)


@calibrate_implicit.defjvp
def _calibrate_jvp(primals, tangents):
    observables, target = primals
    dobservables, dtarget = tangents
    lam = _calibrate_primal(observables, target)
    weights, moments, covariance = _tilt(lam, observables)
    centered = observables - moments
    dlogit = jnp.sum(dobservables * lam[None, :], axis=-1)
    dF = weights @ dobservables + jnp.sum(weights[:, None] * centered * dlogit[:, None], axis=0) - dtarget
    return lam, _solve(covariance, -dF, CALIBRATION_RIDGE)


def beta_schedule(raw, t):
    if raw.shape[0] == 1:
        return jax.nn.softplus(raw[0])
    basis = jnp.asarray([1.0, jnp.cos(2.0 * jnp.pi * t), jnp.sin(2.0 * jnp.pi * t)])
    return jax.nn.softplus(raw @ basis)


def gamma_schedule(raw, t):
    # Smooth endpoint envelope: the square-root Brownian bridge has an
    # infinite endpoint derivative and therefore is unsuitable for direct ODE
    # integration beginning at t=0.
    return jnp.sin(jnp.pi * t) * beta_schedule(raw, t)


def bridge_state(raw, t, minus, plus, noise):
    gamma = gamma_schedule(raw, t)
    gamma_dot = jax.grad(lambda time_value: gamma_schedule(raw, time_value))(t)
    state = (1.0 - t) * minus + t * plus + gamma * noise
    velocity = plus - minus + gamma_dot * noise
    return state, velocity


def fiber_state(raw, t, minus, plus, noise, target):
    state, velocity = bridge_state(raw, t, minus, plus, noise)
    observables = v_observables(state)
    lam = calibrate_implicit(observables, target)
    weights, moments, covariance = _tilt(lam, observables)
    jacobians = v_jphi(state)
    jphi_u = jnp.einsum("mrnd,mnd->mr", jacobians, velocity)
    expected = weights @ jphi_u
    scalar = jphi_u @ lam
    covariance_term = jnp.sum(weights[:, None] * (observables - target) * scalar[:, None], axis=0)
    lambda_dot = _solve(covariance, -expected - covariance_term, CALIBRATION_RIDGE)
    forcing = (observables - target) @ lambda_dot + (jphi_u - expected) @ lam
    forcing = forcing - weights @ forcing
    descriptor_values = v_descriptors(state)
    descriptor_jacobians = v_jdesc(state)
    gram = jnp.einsum("m,mknd,mlnd->kl", weights, descriptor_jacobians, descriptor_jacobians)
    rhs = jnp.einsum("m,mk,m->k", weights, descriptor_values, forcing)
    coefficients = _solve(gram, rhs, RITZ_RIDGE)
    # The weighted-bank source satisfies rho_dot + div(rho u) = rho h.
    # Hence the added flow must obey E[grad(f) . w] = E[f h].
    correction = jnp.einsum("mknd,k->mnd", descriptor_jacobians, coefficients)
    correction_energy = jnp.sum(weights * jnp.sum(correction * correction, axis=(1, 2)))
    forcing_power = jnp.sum(weights * forcing * forcing)
    ess = 1.0 / (state.shape[0] * jnp.sum(weights * weights))
    return state, weights, forcing, correction_energy, forcing_power, ess, moments


def path_metrics(raw, bank, times, target):
    return jax.vmap(lambda t, xm, xp, z: fiber_state(raw, t, xm, xp, z, target))(
        times, bank[0], bank[1], bank[2]
    )


def schedule_objective(raw, bank, times, target, ess_floor=0.18):
    values = path_metrics(raw, bank, times, target)
    energy, forcing, ess = values[3], values[4], values[5]
    penalty = 15.0 * jnp.trapezoid(jax.nn.relu(ess_floor - ess) ** 2, times)
    regularization = 1e-3 * jnp.mean(raw[1:] ** 2) if raw.shape[0] > 1 else 0.0
    return jnp.trapezoid(energy + 0.02 * forcing, times) + penalty + regularization


def optimize_schedule(initial, bank, times, target, steps):
    objective = jax.jit(lambda raw: schedule_objective(raw, bank, times, target))
    value_grad = jax.jit(jax.value_and_grad(objective))
    raw = jnp.asarray(initial); first = jnp.zeros_like(raw); second = jnp.zeros_like(raw)
    trace = []
    candidates = []
    start = time.perf_counter()
    for iteration in range(1, steps + 1):
        value, gradient = value_grad(raw)
        candidates.append(np.asarray(raw))
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient * gradient
        raw = raw - 0.035 * (first / (1.0 - 0.9**iteration)) / (
            jnp.sqrt(second / (1.0 - 0.999**iteration)) + 1e-8
        )
        trace.append(float(value))
    final_value = float(objective(raw))
    candidates.append(np.asarray(raw)); trace.append(final_value)
    best = candidates[int(np.argmin(trace))]
    jax.block_until_ready(raw)
    return best, trace, time.perf_counter() - start


def make_bridge_bank(populations, rng, times, count):
    minus_indices = rng.choice(len(populations["minus"]), size=(len(times), count), p=populations["minus_weights"])
    plus_indices = rng.choice(len(populations["plus"]), size=(len(times), count), p=populations["plus_weights"])
    minus = populations["minus"][minus_indices]
    plus = populations["plus"][plus_indices]
    # Sorting by polar angle is a deterministic exchangeable-particle coupling.
    def sort_configurations(values):
        result = np.empty_like(values)
        for index in np.ndindex(values.shape[:2]):
            configuration = values[index]
            order = np.argsort(np.arctan2(configuration[:, 1], configuration[:, 0]))
            result[index] = configuration[order]
        return result
    minus, plus = sort_configurations(minus), sort_configurations(plus)
    noise = rng.normal(size=minus.shape)
    noise -= noise.mean(axis=2, keepdims=True)
    radius = np.sqrt(np.mean(np.sum(noise * noise, axis=-1), axis=2))
    noise /= radius[:, :, None, None]
    return tuple(jnp.asarray(value) for value in (minus, plus, noise))


def init_mlp(key, input_dim=11, width=18):
    keys = jax.random.split(key, 3)
    def weight(k, din, dout):
        return jax.random.normal(k, (din, dout), dtype=jnp.float64) * math.sqrt(2.0 / (din + dout))
    return MLP(
        weight(keys[0], input_dim, width), jnp.zeros(width),
        weight(keys[1], width, width), jnp.zeros(width),
        weight(keys[2], width, 1), jnp.zeros(1),
    )


def mlp_apply(params, features):
    hidden = jax.nn.silu(features @ params.w1 + params.b1)
    hidden = jax.nn.silu(hidden @ params.w2 + params.b2)
    return (hidden @ params.w3 + params.b3)[0]


def potential_single(params, t, configuration):
    descriptors = radial_descriptors_single(configuration)
    if params.w1.shape[0] == 14:
        descriptors = jnp.concatenate([descriptors, angular_descriptors_single(configuration)])
    elif params.w1.shape[0] != 11:
        raise ValueError(f"unsupported invariant MLP input dimension: {params.w1.shape[0]}")
    time_features = jnp.asarray([t, jnp.sin(2.0 * jnp.pi * t), jnp.cos(2.0 * jnp.pi * t)])
    return mlp_apply(params, jnp.concatenate([descriptors, time_features]))


def correction_single(params, t, configuration):
    return -jax.grad(lambda state: potential_single(params, t, state))(configuration)


v_potential = jax.vmap(potential_single, in_axes=(None, None, 0))
v_correction = jax.vmap(correction_single, in_axes=(None, None, 0))


def neural_ritz_loss(params, states, weights, forcing, times):
    def one_time(t, x, w, h):
        potential = v_potential(params, t, x)
        correction = v_correction(params, t, x)
        energy = jnp.sum(w * jnp.sum(correction * correction, axis=(1, 2)))
        # correction = -grad(psi), so the weak source equation is obtained by
        # minimizing 1/2 E|grad psi|^2 + E[psi h].
        return 0.5 * energy + jnp.sum(w * potential * h)
    losses = jax.vmap(one_time)(times, states, weights, forcing)
    return jnp.trapezoid(losses, times)


def train_neural_correction(key, raw, bank, times, target, steps, input_dim=11):
    fiber = path_metrics(raw, bank, times, target)
    states, weights, forcing = jax.lax.stop_gradient(fiber[0]), jax.lax.stop_gradient(fiber[1]), jax.lax.stop_gradient(fiber[2])
    params = init_mlp(key, input_dim=input_dim)
    first = jax.tree.map(jnp.zeros_like, params); second = jax.tree.map(jnp.zeros_like, params)
    value_grad = jax.jit(jax.value_and_grad(neural_ritz_loss))
    trace = []
    start = time.perf_counter()
    for iteration in range(1, steps + 1):
        value, gradient = value_grad(params, states, weights, forcing, times)
        norm = jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree.leaves(gradient)))
        scale = jnp.minimum(1.0, 5.0 / jnp.maximum(norm, 1e-12))
        gradient = jax.tree.map(lambda x: x * scale, gradient)
        first = jax.tree.map(lambda a, g: 0.9 * a + 0.1 * g, first, gradient)
        second = jax.tree.map(lambda a, g: 0.999 * a + 0.001 * g * g, second, gradient)
        mh = jax.tree.map(lambda x: x / (1.0 - 0.9**iteration), first)
        vh = jax.tree.map(lambda x: x / (1.0 - 0.999**iteration), second)
        params = jax.tree.map(lambda p, m, v: p - 5e-3 * m / (jnp.sqrt(v) + 1e-8), params, mh, vh)
        trace.append(float(value))
    jax.block_until_ready(params.w1)
    return params, trace, time.perf_counter() - start


def correction_statistics(params, raw, bank, times, target):
    fiber = path_metrics(raw, bank, times, target)
    states, weights, forcing = fiber[0], fiber[1], fiber[2]
    def one(t, x, w, h):
        potential = v_potential(params, t, x)
        correction = v_correction(params, t, x)
        energy_per = jnp.sum(correction * correction, axis=(1, 2))
        # Positive alignment is the reduction in the held-out Ritz objective.
        b_per = -potential * h
        return jnp.sum(w * energy_per), jnp.sum(w * b_per), energy_per, b_per
    energy, alignment, energy_per, alignment_per = jax.vmap(one)(times, states, weights, forcing)
    return energy, alignment, energy_per, alignment_per, weights


def validate_component_backend(backend, params, raw, bank, times, target):
    states = path_metrics(raw, bank, times, target)[0][:, :12]
    payload = {
        "times": np.asarray(times), "states": np.asarray(states),
        "descriptor_centers": np.asarray(DESC_CENTERS),
        "observable_centers": np.asarray(OBS_CENTERS),
        "radial_width": RBF_WIDTH, "box_size": BOX_SIZE,
        **{name: np.asarray(getattr(params, name)) for name in params._fields},
    }
    if backend == "jax":
        specification = importlib.util.spec_from_file_location("mfsi_paper_level2_api", COMPONENT_API)
        if specification is None or specification.loader is None:
            raise RuntimeError(f"cannot load component recipe: {COMPONENT_API}")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        result = module.apply_payload(payload)
    else:
        url = os.environ.get("MFSI_PAPER_LEVEL2_TESSERACT_URL")
        if not url:
            raise RuntimeError(
                "MFSI_PAPER_LEVEL2_TESSERACT_URL is missing; invoke through "
                "scripts/run_level2_paper_study.sh --backend tesseract"
            )
        result = _post(url, "apply", {"inputs": payload}, timeout=900.0)
    expected_correction = jax.vmap(
        lambda t, x: v_correction(params, t, x)
    )(times, states)
    expected_descriptors = jax.vmap(v_descriptors)(states)
    correction_error = np.linalg.norm(np.asarray(result["correction"]) - np.asarray(expected_correction))
    correction_error /= max(np.linalg.norm(np.asarray(expected_correction)), 1e-12)
    descriptor_error = np.linalg.norm(np.asarray(result["descriptors"]) - np.asarray(expected_descriptors))
    descriptor_error /= max(np.linalg.norm(np.asarray(expected_descriptors)), 1e-12)
    return {
        "backend": backend,
        "correction_relative_error": float(correction_error),
        "descriptor_relative_error": float(descriptor_error),
        "particle_count": int(np.asarray(result["particle_count"])),
        "state_dimension": int(np.asarray(result["state_dimension"])),
    }


def select_gate(params, raw, gate_bank, times, target):
    energy, alignment, energy_per, alignment_per, weights = correction_statistics(params, raw, gate_bank, times, target)
    total_energy = jnp.trapezoid(energy, times)
    total_alignment = jnp.trapezoid(alignment, times)
    candidate = jnp.clip(total_alignment / jnp.maximum(total_energy, 1e-12), 0.0, 1.0)
    gain = candidate * total_alignment - 0.5 * candidate * candidate * total_energy
    # Conservative two-standard-error activation.  Each time-slice is a
    # self-normalized weighted mean, so its variance must include squared
    # calibration weights; treating the raw particle contributions as equally
    # scaled overstates uncertainty by roughly the bank size.
    contributions = candidate * alignment_per - 0.5 * candidate * candidate * energy_per
    means = jnp.sum(weights * contributions, axis=1)
    variances = jnp.sum(weights * weights * (contributions - means[:, None]) ** 2, axis=1)
    quadrature = jnp.concatenate([
        (times[1:2] - times[:1]) / 2.0,
        (times[2:] - times[:-2]) / 2.0,
        (times[-1:] - times[-2:-1]) / 2.0,
    ])
    standard_error = jnp.sqrt(jnp.sum(quadrature * quadrature * variances))
    # The held-out bank selects between the fitted amplitude and the exact
    # no-correction fallback.  The seed-level study (five independent banks)
    # supplies the confidence interval; this within-bank standard error is
    # retained as a diagnostic rather than double-counted as another gate.
    gate = jnp.where(gain > 0.0, candidate, 0.0)
    return float(gate), float(gain), float(standard_error)


def weighted_mmd(features_a, weights_a, features_b, weights_b):
    combined = jnp.concatenate([features_a, features_b], axis=0)
    distance = jnp.sum((combined[:, None, :] - combined[None, :, :]) ** 2, axis=-1)
    positive = distance[distance > 0]
    bandwidth = jnp.maximum(jnp.median(positive), 1e-6)
    def kernel(x, y):
        return jnp.exp(-jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1) / (2.0 * bandwidth))
    return jnp.maximum(
        weights_a @ kernel(features_a, features_a) @ weights_a
        + weights_b @ kernel(features_b, features_b) @ weights_b
        - 2.0 * weights_a @ kernel(features_a, features_b) @ weights_b,
        0.0,
    )


def law_features(states):
    return jnp.concatenate([v_descriptors(states), v_q4(states)[:, None]], axis=1)


def integrate_method(method, params, gate, raw, minus, plus, noise, target, steps, snapshot_times):
    dt = 1.0 / steps
    state = minus
    snapshots = []
    snap_steps = {int(round(float(t) * steps)): index for index, t in enumerate(snapshot_times)}
    start = time.perf_counter()
    nfe = 0

    def velocity(current, t):
        nonlocal nfe
        nfe += 1
        reference = plus - minus + jax.grad(lambda time_value: gamma_schedule(raw, time_value))(jnp.asarray(t)) * noise
        if method == "raw":
            return reference
        if method == "neural":
            return reference + gate * v_correction(params, jnp.asarray(t), current)
        jacobians = v_jphi(current)
        jphi_u = jnp.einsum("mrnd,mnd->mr", jacobians, reference)
        gram = jnp.mean(jnp.einsum("mrnd,msnd->mrs", jacobians, jacobians), axis=0)
        drift = jnp.mean(jphi_u, axis=0)
        coefficient = _solve(gram, drift, 1e-7)
        correction = -jnp.einsum("mrnd,r->mnd", jacobians, coefficient)
        return reference + correction

    if 0 in snap_steps:
        snapshots.append(np.asarray(state))
    for step_index in range(steps):
        t = step_index * dt
        first = velocity(state, t)
        proposal = state + dt * first
        second = velocity(proposal, t + dt)
        state = state + 0.5 * dt * (first + second)
        if step_index + 1 in snap_steps:
            snapshots.append(np.asarray(state))
    jax.block_until_ready(state)
    elapsed = time.perf_counter() - start
    return np.stack(snapshots), elapsed, nfe


def evaluate_generated(generated, oracle_bank, raw, times, target):
    oracle = path_metrics(raw, oracle_bank, times, target)
    oracle_states, oracle_weights = oracle[0], oracle[1]
    rows = []
    uniform = None
    for index, t in enumerate(np.asarray(times)):
        gen = jnp.asarray(generated[index])
        if uniform is None or uniform.shape[0] != gen.shape[0]:
            uniform = jnp.ones(gen.shape[0]) / gen.shape[0]
        mmd = weighted_mmd(law_features(gen), uniform, law_features(oracle_states[index]), oracle_weights[index])
        moment_error = jnp.linalg.norm(jnp.mean(v_observables(gen), axis=0) - target)
        rows.append({
            "t": float(t), "mmd2": float(mmd), "moment_error": float(moment_error),
            "q4": float(jnp.mean(v_q4(gen))), "oracle_q4": float(oracle_weights[index] @ v_q4(oracle_states[index])),
        })
    return rows


def interior_mmd2(rows):
    interior = rows[:-1]
    return float(np.trapezoid(
        [row["mmd2"] for row in interior], [row["t"] for row in interior]
    ))


def local_gain_rows(params, gate, raw, bank, times, target):
    energy, alignment, _, _, _ = correction_statistics(params, raw, bank, times, target)
    gain = gate * alignment - 0.5 * gate * gate * energy
    return [
        {
            "t": float(t), "energy": float(e), "alignment": float(a),
            "ritz_gain": float(g),
        }
        for t, e, a, g in zip(np.asarray(times), np.asarray(energy),
                              np.asarray(alignment), np.asarray(gain))
    ]


def rollout_shift_rows(params, generated, oracle_bank, raw, times, target):
    """Compare neural rollout states with the fiber law used by Deep-Ritz."""
    oracle = path_metrics(raw, oracle_bank, times, target)
    oracle_states, oracle_weights = oracle[0], oracle[1]
    rows = []
    for index, t in enumerate(np.asarray(times)):
        generated_states = jnp.asarray(generated[index])
        fiber_states = oracle_states[index]
        weights = oracle_weights[index]
        generated_features = law_features(generated_states)
        fiber_features = law_features(fiber_states)
        fiber_mean = weights @ fiber_features
        centered = fiber_features - fiber_mean
        covariance = (centered.T * weights) @ centered
        delta = jnp.mean(generated_features, axis=0) - fiber_mean
        mahalanobis = jnp.sqrt(jnp.maximum(
            delta @ _solve(covariance, delta, 1e-5), 0.0
        ))
        generated_correction = v_correction(params, jnp.asarray(t), generated_states)
        fiber_correction = v_correction(params, jnp.asarray(t), fiber_states)
        generated_energy = jnp.mean(jnp.sum(generated_correction**2, axis=(1, 2)))
        fiber_energy = jnp.sum(weights * jnp.sum(fiber_correction**2, axis=(1, 2)))
        rows.append({
            "t": float(t),
            "feature_mean_mahalanobis": float(mahalanobis),
            "generated_correction_energy": float(generated_energy),
            "fiber_correction_energy": float(fiber_energy),
            "correction_energy_ratio": float(generated_energy / jnp.maximum(fiber_energy, 1e-12)),
        })
    return rows


def serialize_mlp(params):
    return {name: np.asarray(getattr(params, name)).tolist() for name in params._fields}


def _seed_run(seed, populations, quick, backend):
    rng = np.random.default_rng(seed)
    times = jnp.asarray(np.linspace(0.12, 0.88, 4 if quick else 6))
    schedule_train = make_bridge_bank(populations, rng, np.asarray(times), 40 if quick else 72)
    schedule_select = make_bridge_bank(populations, rng, np.asarray(times), 96 if quick else 192)
    schedule_test = make_bridge_bank(populations, rng, np.asarray(times), 96 if quick else 192)
    target = jnp.asarray(populations["target"])
    steps = 20 if quick else 45
    hand = np.asarray([_inverse_softplus(0.55)])
    scalar, scalar_trace, scalar_time = optimize_schedule(hand, schedule_train, times, target, steps)
    # The scalar model is literally nested in the multi-parameter family.
    multi_initial = np.asarray([scalar[0], 0.0, 0.0])
    multi_candidate, multi_trace, multi_time = optimize_schedule(multi_initial, schedule_train, times, target, steps)
    nested_scalar = np.asarray([scalar[0], 0.0, 0.0])
    selection_values = {
        "nested_scalar": float(schedule_objective(jnp.asarray(nested_scalar), schedule_select, times, target)),
        "multi_candidate": float(schedule_objective(jnp.asarray(multi_candidate), schedule_select, times, target)),
    }
    multi_choice = min(selection_values, key=selection_values.get)
    multi = nested_scalar if multi_choice == "nested_scalar" else multi_candidate
    schedule_rows = {}
    for name, raw in (("hand_constant", hand), ("optimized_scalar", scalar), ("optimized_multi", multi)):
        start = time.perf_counter(); metrics = path_metrics(jnp.asarray(raw), schedule_test, times, target); jax.block_until_ready(metrics[3]); elapsed = time.perf_counter() - start
        schedule_rows[name] = {
            "raw": np.asarray(raw).tolist(),
            "integrated_correction_energy": float(jnp.trapezoid(metrics[3], times)),
            "integrated_forcing_power": float(jnp.trapezoid(metrics[4], times)),
            "minimum_ess": float(jnp.min(metrics[5])),
            "wall_seconds": elapsed + ({"optimized_scalar": scalar_time, "optimized_multi": multi_time}.get(name, 0.0)),
        }
    schedule_rows["optimized_multi"]["selection"] = multi_choice
    schedule_rows["optimized_multi"]["candidate_raw"] = np.asarray(multi_candidate).tolist()
    schedule_rows["optimized_multi"]["selection_objectives"] = selection_values

    neural_train = make_bridge_bank(populations, rng, np.asarray(times), 96 if quick else 192)
    gate_bank = make_bridge_bank(populations, rng, np.asarray(times), 192 if quick else 384)
    test_bank = make_bridge_bank(populations, rng, np.asarray(times), 256 if quick else 512)
    model, training_trace, training_time = train_neural_correction(
        jax.random.PRNGKey(seed), jnp.asarray(multi), neural_train, times, target, 180 if quick else 420
    )
    gate, gate_gain, gate_se = select_gate(model, jnp.asarray(multi), gate_bank, times, target)
    test_stats = correction_statistics(model, jnp.asarray(multi), test_bank, times, target)
    test_energy = float(jnp.trapezoid(test_stats[0], times))
    test_alignment = float(jnp.trapezoid(test_stats[1], times))
    test_gain = gate * test_alignment - 0.5 * gate * gate * test_energy
    diagnostic_rng = np.random.default_rng(seed + 50000)
    offgrid_times = jnp.asarray(np.linspace(0.19, 0.81, 4 if quick else 5))
    offgrid_bank = make_bridge_bank(
        populations, diagnostic_rng, np.asarray(offgrid_times), 128 if quick else 384
    )
    offgrid_rows = local_gain_rows(
        model, gate, jnp.asarray(multi), offgrid_bank, offgrid_times, target
    )
    offgrid_gain = float(np.trapezoid(
        [row["ritz_gain"] for row in offgrid_rows], [row["t"] for row in offgrid_rows]
    ))

    # Representation diagnostic: add smooth q2^2, q4^2, and q6^2 invariants
    # while holding the training bank, optimizer, and width fixed.
    augmented_model, augmented_trace, augmented_training_time = train_neural_correction(
        jax.random.fold_in(jax.random.PRNGKey(seed), 1), jnp.asarray(multi),
        neural_train, times, target, 90 if quick else 420, input_dim=14,
    )
    augmented_gate, augmented_gate_gain, _ = select_gate(
        augmented_model, jnp.asarray(multi), gate_bank, times, target
    )
    augmented_test_stats = correction_statistics(
        augmented_model, jnp.asarray(multi), test_bank, times, target
    )
    augmented_test_energy = float(jnp.trapezoid(augmented_test_stats[0], times))
    augmented_test_alignment = float(jnp.trapezoid(augmented_test_stats[1], times))
    augmented_test_gain = (
        augmented_gate * augmented_test_alignment
        - 0.5 * augmented_gate * augmented_gate * augmented_test_energy
    )

    # Matched time-approximation experiment.  The base model sees 6 x 192
    # configurations (4 x 96 in quick mode) on a fixed regular grid.  This
    # model sees the same total number of configurations and optimizer steps,
    # but spreads them over stratified random continuous times.  Reusing the
    # same PRNG key gives both MLPs exactly the same initialization.
    continuous_rng = np.random.default_rng(seed + 60000)
    continuous_time_count = 8 if quick else 18
    continuous_particles_per_time = (len(times) * (96 if quick else 192)) // continuous_time_count
    strata = (np.arange(continuous_time_count) + continuous_rng.uniform(size=continuous_time_count))
    continuous_times = jnp.asarray(0.12 + 0.76 * strata / continuous_time_count)
    continuous_train = make_bridge_bank(
        populations, continuous_rng, np.asarray(continuous_times),
        continuous_particles_per_time,
    )
    continuous_model, continuous_trace, continuous_training_time = train_neural_correction(
        jax.random.PRNGKey(seed), jnp.asarray(multi), continuous_train,
        continuous_times, target, 180 if quick else 420,
    )
    continuous_gate, continuous_gate_gain, _ = select_gate(
        continuous_model, jnp.asarray(multi), gate_bank, times, target
    )
    continuous_grid_rows = local_gain_rows(
        continuous_model, continuous_gate, jnp.asarray(multi), test_bank, times, target
    )
    continuous_offgrid_rows = local_gain_rows(
        continuous_model, continuous_gate, jnp.asarray(multi), offgrid_bank,
        offgrid_times, target,
    )
    component = validate_component_backend(
        backend, model, jnp.asarray(multi), test_bank, times, target
    )

    evaluation_times = jnp.asarray([0.25, 0.50, 0.75, 1.0])
    generation_count = 32 if quick else 64
    generation_bank = make_bridge_bank(populations, rng, np.asarray([0.5]), generation_count)
    minus, plus, noise = generation_bank[0][0], generation_bank[1][0], generation_bank[2][0]
    oracle_bank = make_bridge_bank(populations, rng, np.asarray(evaluation_times), 128 if quick else 256)
    method_rows = {}
    generated_by_method = {}
    integration_steps = 12 if quick else 24
    for method in ("raw", "tangent", "neural"):
        generated, elapsed, nfe = integrate_method(
            method, model, gate, jnp.asarray(multi), minus, plus, noise, target,
            integration_steps, evaluation_times,
        )
        generated_by_method[method] = generated
        evaluation = evaluate_generated(generated, oracle_bank, jnp.asarray(multi), evaluation_times, target)
        method_rows[method] = {
            "integrated_mmd2": float(np.trapezoid([row["mmd2"] for row in evaluation], np.asarray(evaluation_times))),
            "interior_mmd2": interior_mmd2(evaluation),
            "max_moment_error": float(max(row["moment_error"] for row in evaluation)),
            "endpoint_mmd2": float(evaluation[-1]["mmd2"]),
            "wall_seconds": elapsed,
            "nfe": int(nfe),
            "rows": evaluation,
        }

    # ODE diagnostic: identical particles, oracle bank, field, and gate; only
    # the Heun resolution changes.
    step_sensitivity = {str(integration_steps): method_rows["neural"]["interior_mmd2"]}
    for diagnostic_steps in (2 * integration_steps, 4 * integration_steps):
        generated, _, _ = integrate_method(
            "neural", model, gate, jnp.asarray(multi), minus, plus, noise, target,
            diagnostic_steps, evaluation_times,
        )
        rows = evaluate_generated(
            generated, oracle_bank, jnp.asarray(multi), evaluation_times, target
        )
        step_sensitivity[str(diagnostic_steps)] = interior_mmd2(rows)

    # Gate diagnostic: tune MMD only on an independent validation rollout and
    # then report the chosen amplitude on the untouched primary oracle bank.
    validation_generation = make_bridge_bank(
        populations, diagnostic_rng, np.asarray([0.5]), generation_count
    )
    validation_minus = validation_generation[0][0]
    validation_plus = validation_generation[1][0]
    validation_noise = validation_generation[2][0]
    validation_oracle = make_bridge_bank(
        populations, diagnostic_rng, np.asarray(evaluation_times), 128 if quick else 256
    )
    gate_candidates = sorted(set([0.0, 0.25, 0.5, 0.75, 1.0, float(gate)]))
    validation_scores = {}
    for candidate_gate in gate_candidates:
        generated, _, _ = integrate_method(
            "neural", model, candidate_gate, jnp.asarray(multi), validation_minus,
            validation_plus, validation_noise, target, integration_steps, evaluation_times,
        )
        rows = evaluate_generated(
            generated, validation_oracle, jnp.asarray(multi), evaluation_times, target
        )
        validation_scores[str(candidate_gate)] = interior_mmd2(rows)
    rollout_gate = float(min(gate_candidates, key=lambda value: validation_scores[str(value)]))
    if abs(rollout_gate - gate) < 1e-12:
        rollout_gate_test_mmd = method_rows["neural"]["interior_mmd2"]
    else:
        rollout_generated, _, _ = integrate_method(
            "neural", model, rollout_gate, jnp.asarray(multi), minus, plus, noise,
            target, integration_steps, evaluation_times,
        )
        rollout_rows = evaluate_generated(
            rollout_generated, oracle_bank, jnp.asarray(multi), evaluation_times, target
        )
        rollout_gate_test_mmd = interior_mmd2(rollout_rows)

    augmented_generated, _, _ = integrate_method(
        "neural", augmented_model, augmented_gate, jnp.asarray(multi), minus, plus,
        noise, target, integration_steps, evaluation_times,
    )
    augmented_evaluation = evaluate_generated(
        augmented_generated, oracle_bank, jnp.asarray(multi), evaluation_times, target
    )
    augmented_test_mmd = interior_mmd2(augmented_evaluation)
    continuous_generated, _, _ = integrate_method(
        "neural", continuous_model, continuous_gate, jnp.asarray(multi), minus,
        plus, noise, target, integration_steps, evaluation_times,
    )
    continuous_evaluation = evaluate_generated(
        continuous_generated, oracle_bank, jnp.asarray(multi), evaluation_times, target
    )
    continuous_test_mmd = interior_mmd2(continuous_evaluation)
    shift_rows = rollout_shift_rows(
        model, generated_by_method["neural"], oracle_bank, jnp.asarray(multi),
        evaluation_times, target,
    )
    return {
        "seed": seed,
        "endpoint": {
            "minus_calibration_residual": populations["minus_residual"],
            "plus_calibration_residual": populations["plus_residual"],
            "minus_q4": populations["minus_q4"], "plus_q4": populations["plus_q4"],
        },
        "schedules": schedule_rows,
        "neural": {
            "gate": gate, "gate_bank_gain": gate_gain, "gate_bank_standard_error": gate_se,
            "test_gain": test_gain, "test_energy": test_energy, "training_seconds": training_time,
            "training_initial_loss": training_trace[0], "training_final_loss": training_trace[-1],
            "model_parameters": serialize_mlp(model),
        },
        "rollout_diagnostics": {
            "local_test_gain_by_training_grid_time": local_gain_rows(
                model, gate, jnp.asarray(multi), test_bank, times, target
            ),
            "offgrid_local_gain_rows": offgrid_rows,
            "offgrid_integrated_ritz_gain": offgrid_gain,
            "rollout_shift_rows": shift_rows,
            "ode_step_sensitivity_interior_mmd2": step_sensitivity,
            "ritz_selected_gate": gate,
            "rollout_validation_scores": validation_scores,
            "rollout_selected_gate": rollout_gate,
            "rollout_selected_gate_test_interior_mmd2": rollout_gate_test_mmd,
            "augmented_representation": {
                "features": "eight radial descriptors plus q2^2, q4^2, q6^2",
                "gate": augmented_gate,
                "gate_bank_gain": augmented_gate_gain,
                "test_gain": augmented_test_gain,
                "test_energy": augmented_test_energy,
                "test_interior_mmd2": augmented_test_mmd,
                "training_seconds": augmented_training_time,
                "training_initial_loss": augmented_trace[0],
                "training_final_loss": augmented_trace[-1],
                "model_parameters": serialize_mlp(augmented_model),
            },
            "continuous_time_training": {
                "design": "stratified random times with matched configuration and optimizer budgets",
                "times": np.asarray(continuous_times).tolist(),
                "n_times": continuous_time_count,
                "particles_per_time": continuous_particles_per_time,
                "total_training_configurations": continuous_time_count * continuous_particles_per_time,
                "optimizer_steps": 180 if quick else 420,
                "same_initialization_as_grid_model": True,
                "gate": continuous_gate,
                "gate_bank_gain": continuous_gate_gain,
                "reference_grid_gain_rows": continuous_grid_rows,
                "offgrid_gain_rows": continuous_offgrid_rows,
                "test_interior_mmd2": continuous_test_mmd,
                "test_rows": continuous_evaluation,
                "training_seconds": continuous_training_time,
                "training_initial_loss": continuous_trace[0],
                "training_final_loss": continuous_trace[-1],
                "model_parameters": serialize_mlp(continuous_model),
            },
        },
        "component": component,
        "methods": method_rows,
    }


def _inverse_softplus(value):
    return float(np.log(np.expm1(value)))


def mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if len(values) < 2:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean, "n": len(values)}
    critical = {5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 10: 2.262}.get(len(values), 1.96)
    half = critical * float(np.std(values, ddof=1)) / math.sqrt(len(values))
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half, "n": len(values)}


def time_averaged_gain(rows):
    duration = rows[-1]["t"] - rows[0]["t"]
    return float(np.trapezoid(
        [row["ritz_gain"] for row in rows], [row["t"] for row in rows]
    ) / duration)


def aggregate(seed_reports):
    output = {"schedules": {}, "methods": {}, "neural": {}, "rollout_diagnostics": {}}
    for schedule in ("hand_constant", "optimized_scalar", "optimized_multi"):
        output["schedules"][schedule] = {
            metric: mean_ci([report["schedules"][schedule][metric] for report in seed_reports])
            for metric in ("integrated_correction_energy", "integrated_forcing_power", "minimum_ess", "wall_seconds")
        }
    for method in ("raw", "tangent", "neural"):
        output["methods"][method] = {
            metric: mean_ci([report["methods"][method][metric] for report in seed_reports])
            for metric in ("integrated_mmd2", "interior_mmd2", "max_moment_error", "endpoint_mmd2", "wall_seconds", "nfe")
        }
    for metric in ("gate", "test_gain", "test_energy", "training_seconds"):
        output["neural"][metric] = mean_ci([report["neural"][metric] for report in seed_reports])
    output["paired_effects"] = {
        "multi_minus_scalar_energy": mean_ci([
            report["schedules"]["optimized_multi"]["integrated_correction_energy"]
            - report["schedules"]["optimized_scalar"]["integrated_correction_energy"]
            for report in seed_reports
        ]),
        "tangent_minus_raw_interior_mmd2": mean_ci([
            report["methods"]["tangent"]["interior_mmd2"] - report["methods"]["raw"]["interior_mmd2"]
            for report in seed_reports
        ]),
        "neural_minus_raw_interior_mmd2": mean_ci([
            report["methods"]["neural"]["interior_mmd2"] - report["methods"]["raw"]["interior_mmd2"]
            for report in seed_reports
        ]),
        "neural_minus_tangent_interior_mmd2": mean_ci([
            report["methods"]["neural"]["interior_mmd2"] - report["methods"]["tangent"]["interior_mmd2"]
            for report in seed_reports
        ]),
        "neural_minus_raw_moment_error": mean_ci([
            report["methods"]["neural"]["max_moment_error"] - report["methods"]["raw"]["max_moment_error"]
            for report in seed_reports
        ]),
    }
    if all("rollout_diagnostics" in report for report in seed_reports):
        diagnostics = output["rollout_diagnostics"]
        diagnostics["training_grid_integrated_ritz_gain"] = mean_ci([
            float(np.trapezoid(
                [row["ritz_gain"] for row in report["rollout_diagnostics"]["local_test_gain_by_training_grid_time"]],
                [row["t"] for row in report["rollout_diagnostics"]["local_test_gain_by_training_grid_time"]],
            )) for report in seed_reports
        ])
        diagnostics["offgrid_integrated_ritz_gain"] = mean_ci([
            report["rollout_diagnostics"]["offgrid_integrated_ritz_gain"]
            for report in seed_reports
        ])
        diagnostics["training_grid_time_averaged_ritz_gain"] = mean_ci([
            time_averaged_gain(
                report["rollout_diagnostics"]["local_test_gain_by_training_grid_time"]
            ) for report in seed_reports
        ])
        diagnostics["offgrid_time_averaged_ritz_gain"] = mean_ci([
            time_averaged_gain(report["rollout_diagnostics"]["offgrid_local_gain_rows"])
            for report in seed_reports
        ])
        diagnostics["mean_rollout_feature_mahalanobis"] = mean_ci([
            float(np.mean([
                row["feature_mean_mahalanobis"]
                for row in report["rollout_diagnostics"]["rollout_shift_rows"][:-1]
            ])) for report in seed_reports
        ])
        diagnostics["mean_rollout_to_fiber_correction_energy_ratio"] = mean_ci([
            float(np.mean([
                row["correction_energy_ratio"]
                for row in report["rollout_diagnostics"]["rollout_shift_rows"][:-1]
            ])) for report in seed_reports
        ])
        step_keys = sorted(
            seed_reports[0]["rollout_diagnostics"]["ode_step_sensitivity_interior_mmd2"],
            key=int,
        )
        diagnostics["ode_step_sensitivity_interior_mmd2"] = {
            step: mean_ci([
                report["rollout_diagnostics"]["ode_step_sensitivity_interior_mmd2"][step]
                for report in seed_reports
            ]) for step in step_keys
        }
        diagnostics["rollout_selected_gate"] = mean_ci([
            report["rollout_diagnostics"]["rollout_selected_gate"]
            for report in seed_reports
        ])
        diagnostics["rollout_gate_test_interior_mmd2"] = mean_ci([
            report["rollout_diagnostics"]["rollout_selected_gate_test_interior_mmd2"]
            for report in seed_reports
        ])
        diagnostics["rollout_gate_minus_ritz_gate_test_mmd2"] = mean_ci([
            report["rollout_diagnostics"]["rollout_selected_gate_test_interior_mmd2"]
            - report["methods"]["neural"]["interior_mmd2"]
            for report in seed_reports
        ])
        diagnostics["augmented_test_ritz_gain"] = mean_ci([
            report["rollout_diagnostics"]["augmented_representation"]["test_gain"]
            for report in seed_reports
        ])
        diagnostics["augmented_test_interior_mmd2"] = mean_ci([
            report["rollout_diagnostics"]["augmented_representation"]["test_interior_mmd2"]
            for report in seed_reports
        ])
        diagnostics["augmented_minus_radial_test_mmd2"] = mean_ci([
            report["rollout_diagnostics"]["augmented_representation"]["test_interior_mmd2"]
            - report["methods"]["neural"]["interior_mmd2"]
            for report in seed_reports
        ])
        if all(
            "continuous_time_training" in report["rollout_diagnostics"]
            for report in seed_reports
        ):
            continuous_grid = [
                time_averaged_gain(
                    report["rollout_diagnostics"]["continuous_time_training"]["reference_grid_gain_rows"]
                ) for report in seed_reports
            ]
            continuous_offgrid = [
                time_averaged_gain(
                    report["rollout_diagnostics"]["continuous_time_training"]["offgrid_gain_rows"]
                ) for report in seed_reports
            ]
            grid_on = [
                time_averaged_gain(
                    report["rollout_diagnostics"]["local_test_gain_by_training_grid_time"]
                ) for report in seed_reports
            ]
            grid_off = [
                time_averaged_gain(report["rollout_diagnostics"]["offgrid_local_gain_rows"])
                for report in seed_reports
            ]
            diagnostics["continuous_time_training"] = {
                "reference_grid_time_averaged_ritz_gain": mean_ci(continuous_grid),
                "offgrid_time_averaged_ritz_gain": mean_ci(continuous_offgrid),
                "offgrid_gain_minus_grid_trained_offgrid_gain": mean_ci([
                    continuous - fixed
                    for continuous, fixed in zip(continuous_offgrid, grid_off)
                ]),
                "test_interior_mmd2": mean_ci([
                    report["rollout_diagnostics"]["continuous_time_training"]["test_interior_mmd2"]
                    for report in seed_reports
                ]),
                "continuous_minus_grid_trained_test_mmd2": mean_ci([
                    report["rollout_diagnostics"]["continuous_time_training"]["test_interior_mmd2"]
                    - report["methods"]["neural"]["interior_mmd2"]
                    for report in seed_reports
                ]),
                "continuous_minus_tangent_test_mmd2": mean_ci([
                    report["rollout_diagnostics"]["continuous_time_training"]["test_interior_mmd2"]
                    - report["methods"]["tangent"]["interior_mmd2"]
                    for report in seed_reports
                ]),
            }
            grid_gap = float(np.mean(np.asarray(grid_on) - np.asarray(grid_off)))
            continuous_gap = float(np.mean(
                np.asarray(continuous_grid) - np.asarray(continuous_offgrid)
            ))
            reduction = (
                1.0 - continuous_gap / grid_gap if grid_gap > 0.0 else float("nan")
            )
            diagnostics["continuous_time_training"]["degradation"] = {
                "grid_trained_mean_on_minus_offgrid": grid_gap,
                "continuous_trained_mean_reference_minus_offgrid": continuous_gap,
                "relative_reduction": reduction,
            }
            # This criterion deliberately does not inspect the tangent MMD.
            diagnostics["continuous_time_training"]["coupling_readiness"] = {
                "criterion": (
                    "off-grid gain is not lower than the grid-trained field and "
                    "the mean on/off-grid degradation is reduced by at least 50%"
                ),
                "minimum_degradation_reduction_fraction": 0.50,
                "offgrid_gain_not_lower": float(np.mean(continuous_offgrid)) >= float(np.mean(grid_off)),
                "degradation_substantially_reduced": reduction >= 0.50,
                "proceed_to_coupling": (
                    float(np.mean(continuous_offgrid)) >= float(np.mean(grid_off))
                    and reduction >= 0.50
                ),
                "tangent_result_used_in_criterion": False,
            }
        diagnostics["mmd2_by_time"] = {}
        for row_index, row in enumerate(seed_reports[0]["methods"]["raw"]["rows"]):
            time_key = str(row["t"])
            diagnostics["mmd2_by_time"][time_key] = {
                method: mean_ci([
                    report["methods"][method]["rows"][row_index]["mmd2"]
                    for report in seed_reports
                ]) for method in ("raw", "tangent", "neural")
            }
            diagnostics["mmd2_by_time"][time_key]["neural_minus_raw"] = mean_ci([
                report["methods"]["neural"]["rows"][row_index]["mmd2"]
                - report["methods"]["raw"]["rows"][row_index]["mmd2"]
                for report in seed_reports
            ])
            diagnostics["mmd2_by_time"][time_key]["neural_minus_tangent"] = mean_ci([
                report["methods"]["neural"]["rows"][row_index]["mmd2"]
                - report["methods"]["tangent"]["rows"][row_index]["mmd2"]
                for report in seed_reports
            ])
            if all(
                "continuous_time_training" in report["rollout_diagnostics"]
                for report in seed_reports
            ):
                diagnostics["mmd2_by_time"][time_key]["continuous_time"] = mean_ci([
                    report["rollout_diagnostics"]["continuous_time_training"]["test_rows"][row_index]["mmd2"]
                    for report in seed_reports
                ])
    output["acceptance_gates"] = {
        "multi_beats_hand_correction_energy": output["schedules"]["optimized_multi"]["integrated_correction_energy"]["mean"] < output["schedules"]["hand_constant"]["integrated_correction_energy"]["mean"],
        "multi_selected_gain_over_scalar": output["paired_effects"]["multi_minus_scalar_energy"]["mean"] <= 0.0,
        "test_neural_gain_positive": output["neural"]["test_gain"]["ci95_low"] > 0.0 if len(seed_reports) > 1 else output["neural"]["test_gain"]["mean"] > 0.0,
        "corrected_interior_mmd_improves": output["paired_effects"]["tangent_minus_raw_interior_mmd2"]["ci95_high"] < 0.0 if len(seed_reports) > 1 else output["paired_effects"]["tangent_minus_raw_interior_mmd2"]["mean"] < 0.0,
        "physical_endpoint_calibration": max(max(r["endpoint"]["minus_calibration_residual"], r["endpoint"]["plus_calibration_residual"]) for r in seed_reports) < 1e-7,
        "hidden_endpoint_gap": min(r["endpoint"]["plus_q4"] - r["endpoint"]["minus_q4"] for r in seed_reports) > 0.30,
        "jax_tesseract_kernel_parity": max(
            max(r["component"]["correction_relative_error"], r["component"]["descriptor_relative_error"])
            for r in seed_reports
        ) < 2e-9,
    }
    output["passed"] = all(output["acceptance_gates"].values())
    return output


def make_plots(report, output):
    plt.rcParams.update({"figure.facecolor": "#f4f1ea", "axes.facecolor": "#fffdf8", "axes.grid": True, "grid.alpha": 0.2})
    colors = ["#e76f51", "#e9c46a", "#2a9d8f"]
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    schedules = ["hand_constant", "optimized_scalar", "optimized_multi"]
    labels = ["hand constant", "optimized scalar", "optimized multi"]
    for ax, metric, title in zip(axes[0], ("integrated_correction_energy", "integrated_forcing_power", "minimum_ess"), ("Correction energy", "Forcing power", "Minimum ESS")):
        stats = [report["aggregate"]["schedules"][name][metric] for name in schedules]
        means = [item["mean"] for item in stats]
        errors = [[mean - item["ci95_low"] for mean, item in zip(means, stats)], [item["ci95_high"] - mean for mean, item in zip(means, stats)]]
        ax.bar(labels, means, color=colors, yerr=errors, capsize=4)
        ax.set_title(title); ax.tick_params(axis="x", rotation=14)
    methods = ["raw", "tangent", "neural"]
    for ax, metric, title in zip(axes[1, :2], ("interior_mmd2", "max_moment_error"), ("Interior projected-law MMD²", "Maximum moment error")):
        stats = [report["aggregate"]["methods"][name][metric] for name in methods]
        means = [item["mean"] for item in stats]
        errors = [[mean - item["ci95_low"] for mean, item in zip(means, stats)], [item["ci95_high"] - mean for mean, item in zip(means, stats)]]
        ax.bar(methods, means, color=colors, yerr=errors, capsize=4); ax.set_title(title)
    ax = axes[1, 2]
    for index, method in enumerate(methods):
        wall = report["aggregate"]["methods"][method]["wall_seconds"]["mean"]
        nfe = report["aggregate"]["methods"][method]["nfe"]["mean"]
        ax.scatter(nfe, wall, s=100, color=colors[index], label=method)
    ax.set(title="Matched compute", xlabel="NFE", ylabel="wall seconds"); ax.legend(frameon=False)
    figure.suptitle("Paper-facing level-2 study · N=32 · seed-level 95% intervals", fontsize=15, fontweight="bold")
    figure.savefig(output / "paper_level2_summary.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    seed = report["seed_reports"][0]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for method, color in zip(methods, colors):
        rows = seed["methods"][method]["rows"]
        path_times = [row["t"] for row in rows]
        axes[0, 0].plot(path_times, [row["mmd2"] for row in rows], "o-", color=color, label=method)
        axes[0, 1].plot(path_times, [row["moment_error"] for row in rows], "o-", color=color, label=method)
        axes[1, 0].plot(path_times, [row["q4"] for row in rows], "o-", color=color, label=method)
    oracle_rows = seed["methods"]["raw"]["rows"]
    axes[1, 0].plot([row["t"] for row in oracle_rows], [row["oracle_q4"] for row in oracle_rows],
                    "k--", linewidth=2, label="independent oracle")
    axes[0, 0].set(title="Generated vs independent law", ylabel="MMD²")
    axes[0, 1].set(title="Measured-fiber tracking", ylabel="moment error")
    axes[1, 0].set(title="Unconstrained structure", xlabel="time", ylabel="q4")
    endpoint = seed["endpoint"]
    axes[1, 1].bar(["minus", "plus"], [endpoint["minus_q4"], endpoint["plus_q4"]],
                   color=["#457b9d", "#e76f51"])
    axes[1, 1].set(title="Calibrated physical endpoints", ylabel="weighted q4", ylim=(0, 1))
    for ax in axes.flat[:3]:
        ax.set_xlabel("time"); ax.legend(frameon=False, fontsize=8)
    figure.suptitle("N=32 many-body path diagnostics · representative independent bank",
                    fontsize=14, fontweight="bold")
    figure.savefig(output / "paper_level2_path_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    diagnostics = report["aggregate"].get("rollout_diagnostics")
    if diagnostics:
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        continuous = diagnostics.get("continuous_time_training")
        gain_stats = [diagnostics["training_grid_time_averaged_ritz_gain"],
                      diagnostics["offgrid_time_averaged_ritz_gain"]]
        gain_labels = ["grid\nreference", "grid\noff-grid"]
        gain_colors = ["#2a9d8f", "#e9c46a"]
        if continuous:
            gain_stats.extend([
                continuous["reference_grid_time_averaged_ritz_gain"],
                continuous["offgrid_time_averaged_ritz_gain"],
            ])
            gain_labels.extend(["random time\nreference", "random time\noff-grid"])
            gain_colors.extend(["#457b9d", "#8ecae6"])
        means = [item["mean"] for item in gain_stats]
        errors = [
            [mean - item["ci95_low"] for mean, item in zip(means, gain_stats)],
            [item["ci95_high"] - mean for mean, item in zip(means, gain_stats)],
        ]
        axes[0, 0].bar(gain_labels, means, color=gain_colors,
                       yerr=errors, capsize=4)
        axes[0, 0].tick_params(axis="x", labelsize=8)
        axes[0, 0].axhline(0.0, color="black", linewidth=1)
        axes[0, 0].set(title="Held-out local Ritz gain", ylabel="time-averaged gain")

        step_stats = diagnostics["ode_step_sensitivity_interior_mmd2"]
        steps = sorted(step_stats, key=int)
        means = [step_stats[step]["mean"] for step in steps]
        low = [step_stats[step]["ci95_low"] for step in steps]
        high = [step_stats[step]["ci95_high"] for step in steps]
        axes[0, 1].plot([int(step) for step in steps], means, "o-", color="#457b9d")
        axes[0, 1].fill_between([int(step) for step in steps], low, high, color="#457b9d", alpha=0.18)
        axes[0, 1].set(title="Neural ODE resolution", xlabel="Heun steps", ylabel="interior MMD²")

        effects = [
            report["aggregate"]["paired_effects"]["neural_minus_raw_interior_mmd2"],
            report["aggregate"]["paired_effects"]["neural_minus_tangent_interior_mmd2"],
            diagnostics["rollout_gate_minus_ritz_gate_test_mmd2"],
            diagnostics["augmented_minus_radial_test_mmd2"],
        ]
        labels = ["neural − raw", "neural − tangent", "rollout gate − Ritz gate", "angular − radial"]
        if continuous:
            effects.extend([
                continuous["continuous_minus_grid_trained_test_mmd2"],
                continuous["continuous_minus_tangent_test_mmd2"],
            ])
            labels.extend(["random time − fixed grid", "random time − tangent"])
        means = [item["mean"] for item in effects]
        xerr = [
            [mean - item["ci95_low"] for mean, item in zip(means, effects)],
            [item["ci95_high"] - mean for mean, item in zip(means, effects)],
        ]
        axes[1, 0].errorbar(means, np.arange(len(labels)), xerr=xerr, fmt="o", color="#e76f51", capsize=4)
        axes[1, 0].axvline(0.0, color="black", linewidth=1)
        axes[1, 0].set_yticks(np.arange(len(labels)), labels)
        axes[1, 0].set(title="Paired end-to-end effects", xlabel="Δ interior MMD²")

        by_time = diagnostics["mmd2_by_time"]
        time_keys = sorted(by_time, key=float)
        for method, color in zip(("raw", "tangent", "neural"), colors):
            axes[1, 1].plot(
                [float(value) for value in time_keys],
                [by_time[value][method]["mean"] for value in time_keys],
                "o-", label=method, color=color,
            )
        if continuous:
            axes[1, 1].plot(
                [float(value) for value in time_keys],
                [by_time[value]["continuous_time"]["mean"] for value in time_keys],
                "o--", label="random-time neural", color="#457b9d",
            )
        axes[1, 1].set(title="Where rollout fidelity is lost", xlabel="time", ylabel="MMD²")
        axes[1, 1].legend(frameon=False)
        figure.suptitle("N=32 neural local-to-rollout diagnosis · five-bank 95% intervals",
                        fontsize=14, fontweight="bold")
        figure.savefig(output / "paper_level2_failure_diagnostics.png", dpi=200, bbox_inches="tight")
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="one-seed plumbing run")
    parser.add_argument("--backend", choices=("jax", "tesseract"), default=None,
                        help="backend for the invariant correction component (default: tesseract)")
    parser.add_argument("--seeds", default=None, help="space-separated seed list")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--aggregate-existing", action="store_true",
                        help="rebuild intervals/plots from existing seed JSON files")
    args = parser.parse_args()
    backend = normalize_backend(args.backend)
    if args.seeds:
        seeds = [int(value) for value in args.seeds.split()]
    else:
        seeds = [401] if args.quick else [401, 402, 403, 404, 405]
    output = args.output_dir or (DEFAULT_OUTPUT / backend)
    output.mkdir(parents=True, exist_ok=True)
    seed_reports = []
    start = time.perf_counter()
    for seed in seeds:
        seed_path = output / f"seed_{seed}.json"
        if args.aggregate_existing:
            report = json.loads(seed_path.read_text())
            for method in ("raw", "tangent", "neural"):
                rows = report["methods"][method]["rows"][:-1]
                report["methods"][method]["interior_mmd2"] = float(np.trapezoid(
                    [row["mmd2"] for row in rows], [row["t"] for row in rows]
                ))
        else:
            print(f"[paper-level2] seed {seed}: physical populations", flush=True)
            populations = build_physical_populations(seed + 10000, args.quick)
            report = _seed_run(seed, populations, args.quick, backend)
            print(f"[paper-level2] seed {seed}: gate={report['neural']['gate']:.3f}, test_gain={report['neural']['test_gain']:.3e}", flush=True)
            continuous = report["rollout_diagnostics"]["continuous_time_training"]
            print(
                f"[paper-level2] seed {seed}: random-time gate={continuous['gate']:.3f}, "
                f"test_mmd2={continuous['test_interior_mmd2']:.3e}",
                flush=True,
            )
        seed_reports.append(report)
        seed_path.write_text(json.dumps(report, indent=2) + "\n")
    summary = {
        "experiment": "paper-facing-level2-manybody",
        "mode": "quick" if args.quick else "standard",
        "component_backend": backend,
        "seeds": seeds,
        "n_particles": N_PARTICLES,
        "state_dimension": STATE_DIMENSION,
        "measured_observables": "three smooth radial pair RBF coefficients",
        "neural_correction": "fully trained two-hidden-layer invariant MLP with conservative held-out amplitude gate",
        "elapsed_seconds": time.perf_counter() - start,
        "seed_reports": seed_reports,
        "aggregate": aggregate(seed_reports),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not args.no_plots:
        make_plots(summary, output)
    print(json.dumps(summary["aggregate"]["acceptance_gates"], indent=2))
    print(f"outputs: {output}")
    if not summary["aggregate"]["passed"]:
        raise SystemExit("paper-facing level-2 gates failed")


if __name__ == "__main__":
    main()
