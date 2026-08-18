from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

VORTEX_DIR = Path(__file__).parents[1] / "experiments" / "vortices"
if str(VORTEX_DIR) not in sys.path:
    sys.path.insert(0, str(VORTEX_DIR))

from experiment import (
    _mean_se,
    _empirical_coordinate_support_gaps,
    _smooth_bound_moment_curve,
)
from selection import _configured_stage_seeds


def _support_gaps(features: np.ndarray, weights: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return _empirical_coordinate_support_gaps(features, weights, targets)


def test_empirical_coordinate_support_rejects_negative_nonnegative_moment() -> None:
    features = np.array(
        [
            [[0.0, 0.2], [0.5, 0.4], [1.0, 0.8]],
            [[0.1, 0.3], [0.6, 0.5], [0.9, 0.7]],
        ]
    )
    weights = np.full((2, 3), 1.0 / 3.0)
    targets = np.array(
        [
            [[0.4, 0.3], [0.5, 0.6]],
            [[0.4, 0.3], [0.5, -1.25e-4]],
        ]
    )

    gaps = _support_gaps(features, weights, targets)

    np.testing.assert_allclose(gaps, [0.1, -0.300125])
    assert gaps[0] > 0.0
    assert gaps[1] < 0.0


def test_empirical_coordinate_support_ignores_zero_weight_particles() -> None:
    features = np.array([[[0.0], [0.5], [1.0]]])
    weights = np.array([[0.5, 0.5, 0.0]])

    gap = _support_gaps(features, weights, np.array([[0.75]]))[0]

    np.testing.assert_allclose(gap, -0.25)


def test_smooth_bounded_moment_curve_preserves_interior_and_saturates() -> None:
    values = np.array([[-0.2, 0.0015, 0.25, 0.9985, 1.3]])
    derivatives = np.array([[4.0, 3.0, 2.0, -3.0, -4.0]])

    bounded, bounded_dot = _smooth_bound_moment_curve(
        values,
        derivatives,
        0.001,
        0.999,
        0.001,
    )

    np.testing.assert_allclose(bounded, [[0.001, 0.00134375, 0.25, 0.99865625, 0.999]])
    np.testing.assert_allclose(bounded_dot, [[0.0, 4.3125, 2.0, -4.3125, 0.0]])


def test_smooth_bounded_moment_derivative_matches_finite_difference() -> None:
    def curve(t: float) -> tuple[np.ndarray, np.ndarray]:
        raw = np.array([[0.001 + 0.0008 * t]])
        raw_dot = np.array([[0.0008]])
        return _smooth_bound_moment_curve(raw, raw_dot, 0.001, 0.999, 0.001)

    h = 1.0e-6
    left, _ = curve(0.4 - h)
    center, center_dot = curve(0.4)
    right, _ = curve(0.4 + h)
    finite_difference = (right - left) / (2.0 * h)

    assert 0.001 < float(center[0, 0]) < 0.002
    np.testing.assert_allclose(center_dot, finite_difference, rtol=1.0e-8, atol=1.0e-10)


def test_validation_summary_records_action_tail_statistics() -> None:
    summary = _mean_se([1.0, 2.0, 3.0, 20.0])

    assert summary["n"] == 4
    assert summary["median"] == 2.5
    assert summary["max"] == 20.0
    np.testing.assert_allclose(summary["p95"], 17.45)


def test_configured_stage_seeds_require_complete_finite_layouts() -> None:
    seeds = _configured_stage_seeds(
        {"full_seed_etas": [[0.1, 0.2, 0.3, 0.4]]},
        "full_seed_etas",
        parameter_count=4,
    )
    np.testing.assert_allclose(seeds, [[0.1, 0.2, 0.3, 0.4]])

    with pytest.raises(ValueError, match=r"shape \[n, 4\]"):
        _configured_stage_seeds(
            {"full_seed_etas": [[0.1, 0.2]]},
            "full_seed_etas",
            parameter_count=4,
        )
    with pytest.raises(ValueError, match="finite"):
        _configured_stage_seeds(
            {"full_seed_etas": [[0.1, 0.2, 0.3, np.nan]]},
            "full_seed_etas",
            parameter_count=4,
        )
