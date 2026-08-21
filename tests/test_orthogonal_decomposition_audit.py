from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

EXPERIMENTS_DIR = Path(__file__).parents[1] / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from orthogonal_decomposition_audit import audit_field_decompositions


def test_field_audit_uses_direct_hidden_energy_and_detects_residuals() -> None:
    candidate = {
        "allowance_percent": 4.0,
        "method": "tangent",
        "geometry": [0.1, 0.2],
        "result_path": "/saved/result.json",
    }

    def evaluate(_geometry: object, _key: str):
        return [
            {
                "trial": 0,
                "valid": True,
                "tangent_action_by_time": [1.0, 1.0],
                "full_action_by_time": [3.0, 3.0],
                "decomposition_by_time": {
                    "direct_full_field_energy": [3.0, 3.0],
                    "direct_tangent_field_energy": [1.0, 1.0],
                    # Deliberately differs from scalar subtraction (=2.0).
                    "direct_hidden_field_energy": [1.8, 1.9],
                    "direct_tangent_hidden_inner_product": [0.1, 0.05],
                    "reported_identity_residual": [0.2, 0.1],
                    "discrete_polarization_residual": [0.0, 0.0],
                },
            }
        ]

    rows, detail, summary = audit_field_decompositions(
        [candidate],
        evaluate=evaluate,
        time_grid=np.array([0.0, 1.0]),
        time_weights=np.array([0.5, 0.5]),
        tolerance=1.0e-6,
    )
    row = rows[0]
    assert row["direct_A_hid"] == pytest.approx(1.85)
    assert row["aggregate_decomposition_residual"] == pytest.approx(0.15)
    assert row["maximum_absolute_decomposition_residual"] == pytest.approx(0.2)
    assert row["maximum_absolute_orthogonality_residual"] == pytest.approx(0.1)
    assert row["passes"] is False
    assert len(next(iter(detail.values()))) == 2
    assert summary["every_final_candidate_passes"] is False
