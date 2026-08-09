"""Exact joint population law and exponential moment conditioning."""

from __future__ import annotations

import math
import numpy as np

from .homometric import PopulationSupport
from .network import PriorParameters


def _logsumexp_np(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + math.log(float(np.sum(np.exp(x - m))))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def log_unnormalized_prior(
    params: PriorParameters, support: PopulationSupport
) -> np.ndarray:
    pi = min(max(sigmoid(params.mode_logit), 1e-12), 1.0 - 1e-12)
    log_mode = np.where(support.labels > 0, math.log(pi), math.log1p(-pi))
    pair_penalty = math.exp(params.log_pair_penalty)
    shared = -pair_penalty * np.square(support.pair - params.pair_center)
    hidden = params.regime_strength * support.labels * support.triplet
    return log_mode + shared + hidden


def prior_probabilities(params: PriorParameters, support: PopulationSupport) -> np.ndarray:
    logits = log_unnormalized_prior(params, support)
    return np.exp(logits - _logsumexp_np(logits))


def conditioned_probabilities(
    params: PriorParameters,
    support: PopulationSupport,
    dual: float,
) -> np.ndarray:
    logits = log_unnormalized_prior(params, support) + float(dual) * support.pair
    return np.exp(logits - _logsumexp_np(logits))


def distribution_summaries(probabilities: np.ndarray, support: PopulationSupport) -> dict[str, float]:
    p = np.asarray(probabilities, dtype=np.float64)
    p = p / p.sum()
    pair_mean = float(np.sum(p * support.pair))
    pair_var = float(np.sum(p * np.square(support.pair - pair_mean)))
    triplet_mean = float(np.sum(p * support.triplet))
    triplet_var = float(np.sum(p * np.square(support.triplet - triplet_mean)))
    mode_plus = float(np.sum(p[support.labels > 0]))
    return {
        "pair_mean": pair_mean,
        "pair_variance": pair_var,
        "triplet_mean": triplet_mean,
        "triplet_variance": triplet_var,
        "mode_plus_probability": mode_plus,
    }


def solve_exact_dual(
    params: PriorParameters,
    support: PopulationSupport,
    target_moment: float,
    *,
    initial_dual: float = 0.0,
    max_iterations: int = 80,
    tolerance: float = 1e-12,
    ridge: float = 1e-12,
    max_step: float = 4.0,
) -> tuple[float, list[dict[str, float]]]:
    target = float(target_moment)
    if target < float(support.pair.min()) - 1e-12 or target > float(support.pair.max()) + 1e-12:
        raise ValueError("target moment lies outside the finite support interval")
    dual = float(initial_dual)
    trace: list[dict[str, float]] = []
    for iteration in range(max_iterations):
        probs = conditioned_probabilities(params, support, dual)
        summaries = distribution_summaries(probs, support)
        residual = summaries["pair_mean"] - target
        covariance = summaries["pair_variance"]
        trace.append(
            {
                "iteration": float(iteration),
                "dual": dual,
                "moment": summaries["pair_mean"],
                "residual": residual,
                "covariance": covariance,
            }
        )
        if abs(residual) <= tolerance:
            break
        step = residual / (covariance + ridge)
        step = float(np.clip(step, -max_step, max_step))
        dual -= step
    return dual, trace


def conditioned_from_reference(
    reference_probabilities: np.ndarray,
    support: PopulationSupport,
    target_moment: float,
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, float]:
    ref = np.asarray(reference_probabilities, dtype=np.float64)
    ref = ref / ref.sum()
    log_ref = np.log(np.maximum(ref, 1e-300))
    dual = 0.0
    for _ in range(max_iterations):
        logits = log_ref + dual * support.pair
        probs = np.exp(logits - _logsumexp_np(logits))
        mean = float(np.sum(probs * support.pair))
        var = float(np.sum(probs * np.square(support.pair - mean)))
        residual = mean - target_moment
        if abs(residual) <= tolerance:
            return probs, dual
        dual -= float(np.clip(residual / (var + 1e-12), -4.0, 4.0))
    return probs, dual


def sample_indices(probabilities: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    probs = probs / probs.sum()
    return rng.choice(probs.size, size=int(size), replace=True, p=probs).astype(np.int64)
