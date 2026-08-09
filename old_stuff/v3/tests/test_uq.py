from __future__ import annotations

import numpy as np

from manybody_completion.uq import aggregate_seed_higher_order_uq, summarize_higher_order


def test_higher_order_summary_is_normalized() -> None:
    values = np.asarray([-1.0, 0.0, 1.0])
    labels = np.asarray([-1.0, 1.0, 1.0])
    probabilities = np.asarray([0.2, 0.3, 0.5])
    summary = summarize_higher_order(values, labels, probabilities)
    assert abs(summary["triplet_mean"] - 0.3) < 1e-12
    assert abs(summary["mode_plus_probability"] - 0.8) < 1e-12
    assert summary["predictive_intervals"]["90"]["lower"] <= summary["triplet_mean"]
    assert summary["predictive_intervals"]["90"]["upper"] >= summary["triplet_mean"]


def test_seed_variance_decomposition() -> None:
    summaries = [
        {"triplet_mean": -0.2, "triplet_variance": 0.4},
        {"triplet_mean": 0.2, "triplet_variance": 0.6},
    ]
    result = aggregate_seed_higher_order_uq(summaries)
    assert abs(result["mean_within_seed_variance"] - 0.5) < 1e-12
    assert result["between_seed_mean_variance"] > 0
    assert result["total_predictive_variance"] > 0.5


def test_higher_order_uq_scores_reference_coverage() -> None:
    values = np.asarray([-1.0, 0.0, 1.0])
    labels = np.asarray([-1.0, 1.0, 1.0])
    predictive = np.asarray([0.2, 0.3, 0.5])
    reference = np.asarray([0.4, 0.2, 0.4])
    summary = summarize_higher_order(
        values,
        labels,
        predictive,
        reference_probabilities=reference,
        effective_sample_size=200.0,
    )
    interval = summary["predictive_intervals"]["90"]
    assert 0.0 <= interval["reference_coverage"] <= 1.0
    assert interval["expected_interval_score"] >= 0.0
    assert summary["mode_probability_standard_error"] > 0.0
    assert summary["expected_mode_log_score"] > 0.0
