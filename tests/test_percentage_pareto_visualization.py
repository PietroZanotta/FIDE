from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

EXPERIMENTS_DIR = Path(__file__).parents[1] / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from percentage_pareto_visualization import (
    method_records,
    save_method_figure,
    save_method_tables,
)


def _metric(mean: float, se: float) -> dict[str, float]:
    return {"mean": mean, "se": se}


def test_method_outputs_include_law_tangent_and_full(tmp_path) -> None:
    point = tmp_path / "risk_2pct"
    point.mkdir()
    result_path = point / "result.json"
    data = {
        "law_screens": {"R_star": 10.0, "R_max": 10.2},
        "selection": {
            "law_optimum": [0.1, 0.2],
            "tangent_optimum": [0.2, 0.3],
            "full_optimum": [0.3, 0.4],
        },
        "selection_certificates": {
            "law": {"R_selection": 10.0, "L_selection": 1.0, "full_action_selection": 8.0, "certified": True},
            "tangent": {"R_selection": 10.1, "L_selection": 1.1, "full_action_selection": 7.0, "certified": True},
            "full": {"R_selection": 10.2, "L_selection": 1.2, "full_action_selection": 6.0, "certified": True},
        },
        "validation": {
            "law": {"law_risk": _metric(10.0, 0.1), "full_action": _metric(8.0, 0.2), "valid_fraction": 1.0},
            "tangent": {"law_risk": _metric(10.1, 0.1), "full_action": _metric(7.0, 0.2), "valid_fraction": 0.9},
            "full": {"law_risk": _metric(10.2, 0.1), "full_action": _metric(6.0, 0.2), "valid_fraction": 0.8},
        },
    }
    result_path.write_text(json.dumps(data), encoding="utf-8")
    rows = [
        {
            "risk_allowance_percent": 2.0,
            "R_star": 10.0,
            "full_R_excess_selection": 0.2,
            "full_A_selection": 6.0,
            "full_certified": True,
            "result": str(result_path),
        }
    ]

    records = method_records(rows, tmp_path)
    assert [record["method"] for record in records] == ["law", "tangent", "full"]
    assert records[2]["selection_R_increase_percent"] == pytest.approx(2.0)
    assert records[2]["validation_full_action_reduction_vs_law_percent"] == 25.0

    table_paths = save_method_tables(records, tmp_path)
    assert all(path.is_file() for path in table_paths)
    assert "Tangent" in (tmp_path / "pareto_methods_tables.md").read_text(encoding="utf-8")

    figure_path = save_method_figure(
        records, tmp_path / "pareto_methods.png", experiment_label="Test", dpi=50
    )
    assert figure_path.is_file()
