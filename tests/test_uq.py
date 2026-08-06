import numpy as np

from manybody_completion.uq import (
    aggregate_seed_higher_order_uq,
    higher_order_conditional_uq,
    multivariate_energy_score,
)


def test_higher_order_uq_reports_predictive_bands_and_mode_uncertainty():
    rng = np.random.default_rng(1)
    mode_a = rng.normal(loc=-1.0, scale=0.05, size=(6, 2, 3))
    mode_b = rng.normal(loc=1.0, scale=0.05, size=(6, 2, 3))
    predicted = np.concatenate((mode_a, mode_b), axis=0)
    labels = np.concatenate(
        (np.zeros((6, 2), dtype=np.int32), np.ones((6, 2), dtype=np.int32)),
        axis=0,
    )
    reference = np.concatenate(
        (rng.normal(-1.0, 0.05, size=(20, 3)), rng.normal(1.0, 0.05, size=(20, 3)))
    )
    result = higher_order_conditional_uq(
        predicted,
        reference,
        labels,
        interval_levels=(0.8, 0.9),
        seed=2,
        num_resamples=200,
    )
    assert result["mode_probability_total_variation"] == 0.0
    assert result["normalized_mode_entropy"] == 1.0
    assert result["predictive_intervals"]["0.90"]["reference_coverage_mean"] > 0.7
    assert np.isfinite(result["multivariate_energy_score"])


def test_energy_score_prefers_matching_distribution():
    rng = np.random.default_rng(3)
    reference = rng.normal(size=(30, 2))
    matching = reference.reshape((10, 3, 2))
    shifted = matching + 4.0
    assert multivariate_energy_score(matching, reference) < multivariate_energy_score(
        shifted, reference
    )


def test_mode_entropy_handles_an_empty_mode_without_log_warnings():
    rng = np.random.default_rng(4)
    predicted = rng.normal(size=(4, 2, 3))
    reference = rng.normal(size=(10, 3))
    labels = np.zeros((4, 2), dtype=np.int32)
    with np.errstate(divide="raise", invalid="raise"):
        result = higher_order_conditional_uq(
            predicted,
            reference,
            labels,
            seed=5,
            num_resamples=100,
        )
    assert result["normalized_mode_entropy"] == 0.0


def test_seed_aggregation_separates_aleatoric_and_epistemic_variance():
    summaries = [
        {
            "predictive_mean": [0.0, 1.0],
            "predictive_variance": [2.0, 3.0],
        },
        {
            "predictive_mean": [2.0, 1.0],
            "predictive_variance": [2.0, 5.0],
        },
    ]
    result = aggregate_seed_higher_order_uq(summaries)
    assert np.allclose(result["aleatoric_variance"], [2.0, 4.0])
    assert np.allclose(result["epistemic_variance"], [2.0, 0.0])
    assert np.allclose(result["total_predictive_variance"], [4.0, 4.0])
