from __future__ import annotations

import numpy as np

try:
    from .eval import paired_action_arrays, paired_action_by_time, paired_bootstrap_reduction
except ImportError:  # pragma: no cover - direct pytest collection convention.
    from eval import paired_action_arrays, paired_action_by_time, paired_bootstrap_reduction


def test_paired_action_arrays_intersect_valid_trial_ids() -> None:
    result = {
        "validation": {
            "law": {
                "trials": [
                    {"trial": 0, "valid": True, "full_action": 10.0},
                    {"trial": 1, "valid": False, "full_action": 20.0},
                    {"trial": 2, "valid": True, "full_action": 30.0},
                ]
            },
            "full": {
                "trials": [
                    {"trial": 0, "valid": True, "full_action": 4.0},
                    {"trial": 1, "valid": True, "full_action": 8.0},
                    {"trial": 2, "valid": True, "full_action": float("nan")},
                ]
            },
        }
    }
    trial_ids, law, full = paired_action_arrays(result)
    np.testing.assert_array_equal(trial_ids, np.asarray([0]))
    np.testing.assert_allclose(law, np.asarray([10.0]))
    np.testing.assert_allclose(full, np.asarray([4.0]))


def test_paired_bootstrap_reduction_is_reproducible() -> None:
    law = np.asarray([8.0, 10.0, 12.0, 14.0])
    full = 0.6 * law
    first = paired_bootstrap_reduction(law, full, reps=500, seed=17)
    second = paired_bootstrap_reduction(law, full, reps=500, seed=17)
    assert first == second
    assert np.isclose(first["estimate"], 0.4)
    assert np.isclose(first["lower"], 0.4)
    assert np.isclose(first["upper"], 0.4)


def test_paired_action_by_time_skips_legacy_rows() -> None:
    result = {
        "validation": {
            "law": {
                "trials": [
                    {
                        "trial": 0,
                        "valid": True,
                        "full_action": 3.0,
                        "full_action_by_time": [2.0, 4.0],
                    },
                    {"trial": 1, "valid": True, "full_action": 5.0},
                ]
            },
            "full": {
                "trials": [
                    {
                        "trial": 0,
                        "valid": True,
                        "full_action": 1.5,
                        "full_action_by_time": [1.0, 2.0],
                    },
                    {"trial": 1, "valid": True, "full_action": 2.5},
                ]
            },
        }
    }
    trial_ids, law, full = paired_action_by_time(result)
    np.testing.assert_array_equal(trial_ids, np.asarray([0]))
    np.testing.assert_allclose(law, np.asarray([[2.0, 4.0]]))
    np.testing.assert_allclose(full, np.asarray([[1.0, 2.0]]))
