from __future__ import annotations

import json

from experiments.active_nematic_unbalanced.eval import evaluate


def test_saved_unbalanced_result_structural_check(tmp_path) -> None:
    metric_names = (
        "law_risk_total", "law_risk_plus", "law_risk_minus",
        "shape_mmd_plus", "shape_mmd_minus", "mass_error_plus", "mass_error_minus",
        "full_unbalanced_action_total", "full_unbalanced_action_plus", "full_unbalanced_action_minus",
        "move_action_plus", "reaction_action_plus", "move_action_minus", "reaction_action_minus",
        "reaction_fraction_plus", "reaction_fraction_minus", "reaction_fraction_total",
        "max_calibration_residual", "min_ess_fraction", "max_screened_pde_relative_residual",
    )
    values = {name: 0.0 for name in metric_names}
    values.update({
        "full_unbalanced_action_total": 10.0,
        "full_unbalanced_action_plus": 4.0,
        "full_unbalanced_action_minus": 6.0,
        "move_action_plus": 3.0,
        "reaction_action_plus": 1.0,
        "move_action_minus": 4.0,
        "reaction_action_minus": 2.0,
        "min_ess_fraction": 0.5,
    })
    payload = {
        "reference_seed": 1,
        "plus_reference_seed": 1,
        "minus_reference_seed": 10001,
        "physical_interval": [21.0, 31.0],
        "reaction_kappa": 1.0,
        "species_weight_plus": 1.0,
        "species_weight_minus": 1.0,
        "charge_balance_diagnostics": {"passed": True, "maximum_violation": 0.0},
        "validation": {
            "valid_trials": 2,
            "trials": 2,
            "metrics": {name: {"mean": value, "se": 0.0} for name, value in values.items()},
        },
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload))
    assert evaluate(path) == 0
