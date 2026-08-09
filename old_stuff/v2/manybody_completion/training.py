"""Exact differentiable training for the finite-support DiffPOP example."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np

from .energy import conditioned_probabilities, prior_probabilities, sample_indices, solve_exact_dual
from .homometric import PopulationSupport
from .network import PriorParameters

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class ConditionalTask:
    target_moment: float
    sample_indices: np.ndarray
    generating_dual: float


def _support_jax(support: PopulationSupport) -> tuple[jax.Array, jax.Array, jax.Array]:
    return (
        jnp.asarray(support.labels),
        jnp.asarray(support.pair),
        jnp.asarray(support.triplet),
    )


def _log_prior_jax(
    params: jax.Array, labels: jax.Array, pair: jax.Array, triplet: jax.Array
) -> jax.Array:
    mode_logit, regime_strength, pair_center, log_pair_penalty = params
    log_plus = jax.nn.log_sigmoid(mode_logit)
    log_minus = jax.nn.log_sigmoid(-mode_logit)
    log_mode = jnp.where(labels > 0, log_plus, log_minus)
    shared = -jnp.exp(log_pair_penalty) * jnp.square(pair - pair_center)
    hidden = regime_strength * labels * triplet
    logits = log_mode + shared + hidden
    return logits - jax.scipy.special.logsumexp(logits)


def _calibrate_exact_jax(
    params: jax.Array,
    target: jax.Array,
    labels: jax.Array,
    pair: jax.Array,
    triplet: jax.Array,
    iterations: int = 18,
) -> jax.Array:
    logp = _log_prior_jax(params, labels, pair, triplet)
    dual = jnp.asarray(0.0, dtype=pair.dtype)
    for _ in range(iterations):
        logq = logp + dual * pair
        q = jax.nn.softmax(logq)
        mean = jnp.sum(q * pair)
        covariance = jnp.sum(q * jnp.square(pair - mean))
        step = jnp.clip((mean - target) / (covariance + 1e-8), -3.0, 3.0)
        dual = dual - step
    return dual


def _clip_params(params: jax.Array) -> jax.Array:
    return jnp.asarray(
        [
            jnp.clip(params[0], -6.0, 6.0),
            jnp.clip(params[1], 0.0, 6.0),
            jnp.clip(params[2], -1.0, 1.0),
            jnp.clip(params[3], -5.0, 5.0),
        ]
    )


def _adam_optimize(
    loss_fn,
    initial: np.ndarray,
    steps: int,
    learning_rate: float,
) -> tuple[np.ndarray, list[float]]:
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    params = jnp.asarray(initial, dtype=jnp.float64)
    m = jnp.zeros_like(params)
    v = jnp.zeros_like(params)
    trace: list[float] = []
    for step in range(int(steps)):
        value, grad = value_and_grad(params)
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * jnp.square(grad)
        m_hat = m / (1.0 - 0.9 ** (step + 1))
        v_hat = v / (1.0 - 0.999 ** (step + 1))
        params = _clip_params(params - learning_rate * m_hat / (jnp.sqrt(v_hat) + 1e-8))
        trace.append(float(value))
    return np.asarray(params, dtype=np.float64), trace


def fit_prior_mle(
    initial: PriorParameters,
    support: PopulationSupport,
    sample_ids: np.ndarray,
    *,
    steps: int,
    learning_rate: float,
) -> tuple[PriorParameters, list[float]]:
    labels, pair, triplet = _support_jax(support)
    ids = jnp.asarray(sample_ids, dtype=jnp.int32)

    def loss_fn(params: jax.Array) -> jax.Array:
        logp = _log_prior_jax(params, labels, pair, triplet)
        return -jnp.mean(logp[ids])

    values, trace = _adam_optimize(loss_fn, initial.as_array(), steps, learning_rate)
    return PriorParameters.from_array(values), trace


def make_conditional_tasks(
    true_params: PriorParameters,
    support: PopulationSupport,
    tilts: Iterable[float],
    samples_per_task: int,
    rng: np.random.Generator,
) -> list[ConditionalTask]:
    tasks: list[ConditionalTask] = []
    for dual in tilts:
        probabilities = conditioned_probabilities(true_params, support, float(dual))
        target = float(np.sum(probabilities * support.pair))
        ids = sample_indices(probabilities, samples_per_task, rng)
        tasks.append(ConditionalTask(target, ids, float(dual)))
    return tasks


def fine_tune_conditional(
    initial: PriorParameters,
    support: PopulationSupport,
    tasks: list[ConditionalTask],
    *,
    steps: int,
    learning_rate: float,
    ess_weight: float,
    differentiate_dual: bool,
) -> tuple[PriorParameters, list[float]]:
    labels, pair, triplet = _support_jax(support)
    task_targets = [jnp.asarray(task.target_moment) for task in tasks]
    task_ids = [jnp.asarray(task.sample_indices, dtype=jnp.int32) for task in tasks]

    def loss_fn(params: jax.Array) -> jax.Array:
        logp = _log_prior_jax(params, labels, pair, triplet)
        losses = []
        for target, ids in zip(task_targets, task_ids, strict=True):
            dual = _calibrate_exact_jax(params, target, labels, pair, triplet)
            if not differentiate_dual:
                dual = jax.lax.stop_gradient(dual)
            logq = logp + dual * pair
            logq = logq - jax.scipy.special.logsumexp(logq)
            nll = -jnp.mean(logq[ids])
            p = jnp.exp(logp)
            q = jnp.exp(logq)
            ess_fraction = 1.0 / jnp.sum(jnp.square(q) / jnp.maximum(p, 1e-15))
            losses.append(nll - ess_weight * jnp.log(jnp.maximum(ess_fraction, 1e-12)))
        return jnp.mean(jnp.stack(losses))

    values, trace = _adam_optimize(loss_fn, initial.as_array(), steps, learning_rate)
    return PriorParameters.from_array(values), trace


def generate_training_samples(
    true_params: PriorParameters,
    support: PopulationSupport,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return sample_indices(prior_probabilities(true_params, support), size, rng)


def composed_objective_and_gradient(
    params: PriorParameters,
    support: PopulationSupport,
    task: ConditionalTask,
    *,
    differentiate_dual: bool,
    ess_weight: float = 0.05,
) -> tuple[float, np.ndarray]:
    labels, pair, triplet = _support_jax(support)
    ids = jnp.asarray(task.sample_indices, dtype=jnp.int32)
    target = jnp.asarray(task.target_moment)

    def objective(values: jax.Array) -> jax.Array:
        logp = _log_prior_jax(values, labels, pair, triplet)
        dual = _calibrate_exact_jax(values, target, labels, pair, triplet)
        if not differentiate_dual:
            dual = jax.lax.stop_gradient(dual)
        logq = logp + dual * pair
        logq = logq - jax.scipy.special.logsumexp(logq)
        p = jnp.exp(logp)
        q = jnp.exp(logq)
        ess_fraction = 1.0 / jnp.sum(jnp.square(q) / jnp.maximum(p, 1e-15))
        return -jnp.mean(logq[ids]) - ess_weight * jnp.log(jnp.maximum(ess_fraction, 1e-12))

    value, grad = jax.value_and_grad(objective)(jnp.asarray(params.as_array()))
    return float(value), np.asarray(grad, dtype=np.float64)


def finite_difference_gradient(
    params: PriorParameters,
    support: PopulationSupport,
    task: ConditionalTask,
    *,
    epsilon: float = 1e-5,
    ess_weight: float = 0.05,
) -> np.ndarray:
    base = params.as_array()
    gradient = np.zeros_like(base)
    for i in range(base.size):
        plus = base.copy()
        minus = base.copy()
        plus[i] += epsilon
        minus[i] -= epsilon
        plus_value, _ = composed_objective_and_gradient(
            PriorParameters.from_array(plus),
            support,
            task,
            differentiate_dual=True,
            ess_weight=ess_weight,
        )
        minus_value, _ = composed_objective_and_gradient(
            PriorParameters.from_array(minus),
            support,
            task,
            differentiate_dual=True,
            ess_weight=ess_weight,
        )
        gradient[i] = (plus_value - minus_value) / (2.0 * epsilon)
    return gradient
