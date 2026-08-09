"""Sampling helpers retained under the historical module name."""

from __future__ import annotations

import numpy as np

from .energy import conditioned_probabilities, prior_probabilities, sample_indices
from .homometric import PopulationSupport
from .network import PriorParameters


def sample_prior(
    params: PriorParameters,
    support: PopulationSupport,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return sample_indices(prior_probabilities(params, support), size, rng)


def sample_conditioned_exact(
    params: PriorParameters,
    support: PopulationSupport,
    dual: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return sample_indices(conditioned_probabilities(params, support, dual), size, rng)
