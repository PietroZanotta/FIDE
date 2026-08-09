from __future__ import annotations

import numpy as np

from manybody_completion.adaptive_components import (
    ProposalArchitecture,
    WarmStartArchitecture,
    exact_tilt_probabilities_jax,
    importance_ess_fraction,
    initialize_proposal_model,
    initialize_warm_start_model,
    proposal_probabilities,
    warm_start_dual,
)
from manybody_completion.energy import prior_probabilities
from manybody_completion.homometric import build_population_support
from manybody_completion.network import PriorParameters
from manybody_completion.solvers import tilted_ensemble_from_probabilities
from manybody_completion.synergy_training import pretrain_adaptive_components


def _prior():
    support = build_population_support(6)
    params = PriorParameters(-0.45, 2.1, -0.05, np.log(2.4))
    return support, prior_probabilities(params, support)


def test_defensive_proposal_is_positive_and_normalized() -> None:
    support, prior = _prior()
    model = initialize_proposal_model(
        ProposalArchitecture(hidden_width=10, hidden_layers=1),
        seed=2,
        defensive_mixture=0.12,
    )
    proposal = proposal_probabilities(model, prior, support, 1.7)
    assert np.all(proposal > 0.0)
    assert abs(proposal.sum() - 1.0) < 1e-12
    assert np.all(proposal >= 0.12 * prior - 1e-14)


def test_component_pretraining_improves_proposal_ess_and_warm_start() -> None:
    support, prior = _prior()
    proposal = initialize_proposal_model(
        ProposalArchitecture(hidden_width=12, hidden_layers=1), seed=3
    )
    warm = initialize_warm_start_model(
        WarmStartArchitecture(hidden_width=12, hidden_layers=1), seed=4
    )
    dual = 2.0
    exact = np.asarray(exact_tilt_probabilities_jax(prior, support.pair, dual))
    target = float(np.sum(exact * support.pair))
    initial_ess = importance_ess_fraction(
        exact, proposal_probabilities(proposal, prior, support, dual)
    )
    initial_warm_error = abs(warm_start_dual(warm, prior, support, target) - dual)
    result = pretrain_adaptive_components(
        proposal,
        warm,
        support,
        prior,
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        steps=60,
        learning_rate=0.003,
        proposal_ess_weight=0.2,
        warm_start_weight=0.3,
        gradient_clip=5.0,
    )
    final_ess = importance_ess_fraction(
        exact,
        proposal_probabilities(result.proposal_model, prior, support, dual),
    )
    final_warm_error = abs(
        warm_start_dual(result.warm_start_model, prior, support, target) - dual
    )
    assert final_ess > initial_ess + 0.05
    assert final_warm_error < initial_warm_error


def test_learned_proposal_sampler_remains_target_correct() -> None:
    support, prior = _prior()
    proposal = initialize_proposal_model(
        ProposalArchitecture(hidden_width=12, hidden_layers=1), seed=5
    )
    warm = initialize_warm_start_model(
        WarmStartArchitecture(hidden_width=12, hidden_layers=1), seed=6
    )
    trained = pretrain_adaptive_components(
        proposal,
        warm,
        support,
        prior,
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        steps=60,
        learning_rate=0.003,
        proposal_ess_weight=0.2,
        warm_start_weight=0.3,
        gradient_clip=5.0,
    )
    dual = 1.8
    exact = np.asarray(exact_tilt_probabilities_jax(prior, support.pair, dual))
    exact_moment = float(np.sum(exact * support.pair))
    sampled = tilted_ensemble_from_probabilities(
        prior,
        support,
        dual,
        particles=4000,
        tempering_steps=8,
        rejuvenation_steps=2,
        resample_threshold=0.5,
        seed=11,
        proposal_model=trained.proposal_model,
    )
    assert sampled.proposal_used
    assert sampled.proposal_expected_ess_fraction > 0.8
    assert abs(sampled.moment_mean - exact_moment) < 0.035
