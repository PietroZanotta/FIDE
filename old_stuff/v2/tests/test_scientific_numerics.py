from __future__ import annotations

import numpy as np

from manybody_completion.energy import (
    conditioned_probabilities,
    distribution_summaries,
    prior_probabilities,
    solve_exact_dual,
)
from manybody_completion.homometric import build_population_support, certify_pair_ambiguity
from manybody_completion.network import PriorParameters
from manybody_completion.solvers import calibrate_dual, tilted_ensemble
from manybody_completion.training import (
    composed_objective_and_gradient,
    finite_difference_gradient,
    make_conditional_tasks,
)


def params() -> PriorParameters:
    return PriorParameters(-0.4, 2.2, -0.05, np.log(2.5))


def test_latent_regimes_are_pair_ambiguous_but_triplet_separated() -> None:
    support = build_population_support(8)
    certificate = certify_pair_ambiguity(support, prior_probabilities(params(), support))
    assert certificate["pair_mean_gap"] < 1e-12
    assert certificate["triplet_mean_gap"] > 0.3


def test_exact_dual_recovers_generating_moment() -> None:
    support = build_population_support(8)
    true_dual = 1.25
    target_probs = conditioned_probabilities(params(), support, true_dual)
    target = distribution_summaries(target_probs, support)["pair_mean"]
    recovered, _ = solve_exact_dual(params(), support, target)
    assert abs(recovered - true_dual) < 1e-8


def test_covariance_is_moment_map_jacobian() -> None:
    support = build_population_support(8)
    dual = 0.7
    epsilon = 1e-5
    center = distribution_summaries(conditioned_probabilities(params(), support, dual), support)
    plus = distribution_summaries(
        conditioned_probabilities(params(), support, dual + epsilon), support
    )["pair_mean"]
    minus = distribution_summaries(
        conditioned_probabilities(params(), support, dual - epsilon), support
    )["pair_mean"]
    finite_difference = (plus - minus) / (2 * epsilon)
    assert abs(finite_difference - center["pair_variance"]) < 1e-7


def test_particle_tilt_matches_exact_reference() -> None:
    support = build_population_support(8)
    dual = 1.0
    exact = distribution_summaries(conditioned_probabilities(params(), support, dual), support)
    particle = tilted_ensemble(
        params(),
        support,
        dual,
        particles=5000,
        tempering_steps=12,
        rejuvenation_steps=3,
        resample_threshold=0.5,
        seed=7,
    )
    assert abs(particle.moment_mean - exact["pair_mean"]) < 0.035
    assert abs(particle.mode_plus_probability - exact["mode_plus_probability"]) < 0.04


def test_particle_dual_calibration_uses_fresh_sample() -> None:
    support = build_population_support(8)
    true_dual = 0.8
    target = distribution_summaries(
        conditioned_probabilities(params(), support, true_dual), support
    )["pair_mean"]
    result = calibrate_dual(
        params(),
        support,
        target,
        sampler_options={
            "particles": 3000,
            "tempering_steps": 10,
            "rejuvenation_steps": 3,
            "resample_threshold": 0.5,
        },
        calibration_options={
            "max_iterations": 16,
            "tolerance": 0.02,
            "ridge": 1e-4,
            "damping": 0.85,
            "max_step": 2.0,
            "weak_support_covariance": 1e-4,
            "max_dual_norm": 20.0,
        },
        seed=13,
    )
    assert result.sampler_calls >= 2
    assert abs(result.residual) < 0.06
    assert result.final_ensemble.effective_sample_size > 0


def test_full_composed_gradient_matches_finite_difference() -> None:
    support = build_population_support(6)
    rng = np.random.default_rng(19)
    task = make_conditional_tasks(params(), support, [0.9], 300, rng)[0]
    _, gradient = composed_objective_and_gradient(
        params(), support, task, differentiate_dual=True
    )
    finite = finite_difference_gradient(params(), support, task, epsilon=2e-5)
    relative = np.linalg.norm(gradient - finite) / max(np.linalg.norm(finite), 1e-12)
    assert relative < 2e-3
