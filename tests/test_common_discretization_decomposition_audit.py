from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

EXPERIMENTS_DIR = Path(__file__).parents[1] / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from common_discretization_decomposition_audit import audit_common_discretization


def _candidate() -> dict[str, object]:
    return {
        "allowance_percent": 1.0,
        "method": "full",
        "geometry": [0.2, 0.8],
        "result_path": "/saved/result.json",
    }


def _evaluation(full_residual: float = 0.0) -> list[dict[str, object]]:
    return [
        {
            "trial": 0,
            "valid": True,
            "full_action_by_time": [3.0, 3.0],
            "common_discretization_decomposition_by_time": {
                "moment_rate_residual": [[0.4, -0.2], [0.3, -0.1]],
                "coefficients": [[1.0, 2.0], [1.0, 2.0]],
                "gram": [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[1.0, 0.0], [0.0, 1.0]],
                ],
                "gram_rank": [2, 2],
                "full_moment_residual": [
                    [full_residual, 0.0],
                    [0.0, 0.0],
                ],
                "tangent_moment_residual": [[0.0, 0.0], [0.0, 0.0]],
                "hidden_moment_residual": [
                    [full_residual, 0.0],
                    [0.0, 0.0],
                ],
                "solver_stabilization_moment_shift": [
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                "full_moment_residual_after_stabilization": [
                    [full_residual, 0.0],
                    [0.0, 0.0],
                ],
                "full_energy": [3.0, 3.0],
                "tangent_energy": [1.0, 1.0],
                "hidden_energy": [2.0, 2.0],
                "tangent_hidden_inner_product": [0.0, 0.0],
                "pythagorean_residual": [0.0, 0.0],
                "hierarchy_raw_violation": [-2.0, -2.0],
            },
        }
    ]


def test_common_audit_passes_exact_raster_decomposition() -> None:
    rows, detail, summary = audit_common_discretization(
        [_candidate()],
        evaluate=lambda _geometry, _key: _evaluation(),
        time_grid=np.asarray([0.0, 1.0]),
        time_weights=np.asarray([0.5, 0.5]),
        tolerance=1.0e-6,
    )
    assert rows[0]["passes"] is True
    assert rows[0]["A_hid_h"] == pytest.approx(2.0)
    assert rows[0]["hidden_fraction_A_hid_over_A_full"] == pytest.approx(2.0 / 3.0)
    assert summary["every_final_candidate_passes"] is True
    assert summary["hidden_fraction_supported"] is True
    assert summary["maximum_raw_hierarchy_violation"] == pytest.approx(-2.0)
    assert len(detail["candidate_0"]["time_trials"]) == 2


def test_common_audit_isolates_full_feasibility_first() -> None:
    rows, _detail, summary = audit_common_discretization(
        [_candidate()],
        evaluate=lambda _geometry, _key: _evaluation(full_residual=2.0e-3),
        time_grid=np.asarray([0.0, 1.0]),
        time_weights=np.asarray([0.5, 0.5]),
        tolerance=1.0e-6,
    )
    assert rows[0]["passes"] is False
    assert rows[0]["first_failing_condition"] == "full_moment"
    assert summary["first_failing_condition"] == "full_moment"
    assert summary["aggregate_violation_count"] == 1
    assert summary["trial_violation_count"] == 1
    assert summary["time_trial_violation_count"] == 1
