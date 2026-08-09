"""Observation-only and finite-reweighting baselines."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .energy import conditioned_from_reference, prior_probabilities
from .homometric import PopulationSupport
from .network import PriorParameters


@dataclass
class BaselineDistribution:
    probabilities: np.ndarray
    dual: float
    effective_sample_size: float
    name: str


def maxent_uniform(support: PopulationSupport, target_moment: float) -> BaselineDistribution:
    reference = np.full(support.size, 1.0 / support.size, dtype=np.float64)
    probabilities, dual = conditioned_from_reference(reference, support, target_moment)
    return BaselineDistribution(
        probabilities=probabilities,
        dual=dual,
        effective_sample_size=float(support.size),
        name="MaxEnt-Uniform",
    )


def prior_only(params: PriorParameters, support: PopulationSupport) -> BaselineDistribution:
    probabilities = prior_probabilities(params, support)
    return BaselineDistribution(
        probabilities=probabilities,
        dual=0.0,
        effective_sample_size=float(support.size),
        name="Prior-Only",
    )


def one_shot_reweight(
    params: PriorParameters,
    support: PopulationSupport,
    target_moment: float,
    *,
    particles: int,
    seed: int,
    max_iterations: int = 80,
) -> tuple[BaselineDistribution, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    prior = prior_probabilities(params, support)
    indices = rng.choice(support.size, size=particles, replace=True, p=prior).astype(np.int64)
    empirical_pair = support.pair[indices]
    dual = 0.0
    weights = np.full(particles, 1.0 / particles, dtype=np.float64)
    for _ in range(max_iterations):
        logits = dual * empirical_pair
        logits -= np.max(logits)
        weights = np.exp(logits)
        weights /= weights.sum()
        mean = float(np.sum(weights * empirical_pair))
        variance = float(np.sum(weights * np.square(empirical_pair - mean)))
        residual = mean - target_moment
        if abs(residual) < 1e-10:
            break
        dual -= float(np.clip(residual / (variance + 1e-10), -4.0, 4.0))
    atom_probabilities = np.bincount(
        indices, weights=weights, minlength=support.size
    ).astype(np.float64)
    atom_probabilities /= atom_probabilities.sum()
    ess = float(1.0 / np.sum(np.square(weights)))
    return (
        BaselineDistribution(atom_probabilities, dual, ess, "One-Shot-Reweight"),
        indices,
        weights,
    )
