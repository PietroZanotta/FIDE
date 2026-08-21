from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

EXPERIMENTS_DIR = Path(__file__).parents[1] / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from action_decomposition_audit import audit_candidates


def _candidate() -> dict[str, object]:
    return {
        "allowance_percent": 1.0,
        "method": "full",
        "geometry": [0.1, 0.2],
        "result_path": "/saved/result.json",
        "reported_A_tan": None,
        "reported_A_full": None,
    }


def test_audit_preserves_negative_raw_hierarchy_slack() -> None:
    def evaluate(_geometry: object, _key: str) -> list[dict[str, object]]:
        return [
            {
                "valid": True,
                "tangent_action": 1.0,
                "full_action": 2.0,
                "tangent_action_by_time": [0.8, 1.2],
                "full_action_by_time": [1.8, 2.2],
            },
            {
                "valid": True,
                "tangent_action": 1.4,
                "full_action": 2.4,
                "tangent_action_by_time": [1.0, 1.8],
                "full_action_by_time": [2.0, 2.8],
            },
        ]

    rows, _ = audit_candidates(
        [_candidate()], evaluate=evaluate, tolerance=1.0e-6, time_grid=np.array([0.0, 1.0])
    )
    row = rows[0]
    assert row["A_tan"] == pytest.approx(1.2)
    assert row["A_full"] == pytest.approx(2.2)
    assert row["A_hid"] == pytest.approx(1.0)
    assert row["Gamma"] == pytest.approx(1.0 - 1.2 / 2.2)
    assert row["maximum_raw_violation_all_levels"] == pytest.approx(-1.0)
    assert row["trial_violation_count"] == 0
    assert row["time_trial_violation_count"] == 0
    assert row["all_hierarchy_checks_pass"] is True


def test_audit_counts_unclipped_time_trial_and_aggregate_violations() -> None:
    def evaluate(_geometry: object, _key: str) -> list[dict[str, object]]:
        return [
            {
                "valid": True,
                "tangent_action": 2.001,
                "full_action": 2.0,
                "tangent_action_by_time": [1.0, 2.003],
                "full_action_by_time": [1.0, 2.0],
            }
        ]

    rows, _ = audit_candidates(
        [_candidate()], evaluate=evaluate, tolerance=1.0e-6, time_grid=np.array([0.0, 1.0])
    )
    row = rows[0]
    assert row["aggregate_raw_violation_A_tan_minus_A_full"] == pytest.approx(0.001)
    assert row["trial_max_raw_violation"] == pytest.approx(0.001)
    assert row["time_trial_max_raw_violation"] == pytest.approx(0.003)
    assert row["maximum_raw_violation_all_levels"] == pytest.approx(0.003)
    assert row["aggregate_violation_count"] == 1
    assert row["trial_violation_count"] == 1
    assert row["time_trial_violation_count"] == 1
    assert row["all_hierarchy_checks_pass"] is False
