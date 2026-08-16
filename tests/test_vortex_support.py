from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

VORTEX_DIR = Path(__file__).parents[1] / "experiments" / "vortices"
if str(VORTEX_DIR) not in sys.path:
    sys.path.insert(0, str(VORTEX_DIR))

from experiment import _empirical_coordinate_support_gaps


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
