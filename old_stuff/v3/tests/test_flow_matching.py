from __future__ import annotations

import numpy as np

from manybody_completion.energy import conditioned_probabilities, distribution_summaries
from manybody_completion.flow import (
    FlowArchitecture,
    flow_gradient_directional_check,
    initialize_flow_model,
    sample_flow_distribution,
)
from manybody_completion.homometric import build_population_support
from manybody_completion.network import PriorParameters


def test_flow_distribution_is_normalized_and_finite() -> None:
    support = build_population_support(6)
    model = initialize_flow_model(
        FlowArchitecture(state_dim=7, hidden_width=12, hidden_layers=1), seed=3
    )
    distribution = sample_flow_distribution(
        model,
        support,
        sample_count=32,
        seed=4,
        sampling_steps=3,
        assignment_temperature=0.5,
    )
    assert distribution.probabilities.shape == (support.size,)
    assert np.all(np.isfinite(distribution.probabilities))
    assert abs(distribution.probabilities.sum() - 1.0) < 1e-12
    assert distribution.quantization_rmse > 0.0


def test_flow_diffpop_gradient_matches_directional_finite_difference() -> None:
    support = build_population_support(5)
    true = PriorParameters(-0.3, 1.8, -0.05, np.log(2.0))
    reference = conditioned_probabilities(true, support, 0.7)
    target = distribution_summaries(reference, support)["pair_mean"]
    rng = np.random.default_rng(9)
    ids = rng.choice(support.size, size=24, p=reference)
    base = rng.normal(size=(20, 6))
    model = initialize_flow_model(
        FlowArchitecture(state_dim=6, hidden_width=10, hidden_layers=1), seed=8
    )
    check = flow_gradient_directional_check(
        model,
        support,
        target_moment=target,
        sample_indices_for_score=ids,
        base_samples=base,
        sampling_steps=2,
        assignment_temperature=0.5,
        dual_iterations=8,
        epsilon=2e-5,
        seed=10,
    )
    assert check["relative_error"] < 2e-3
