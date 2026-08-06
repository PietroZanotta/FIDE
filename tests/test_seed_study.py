import json

import numpy as np

from manybody_completion.seed_study import aggregate_comparison_seed_reports


def _report(effect, shift):
    uq = {
        "predictive_mean": [shift, 1.0],
        "predictive_variance": [2.0, 3.0],
    }
    return {
        "methods": {
            "soft_cefm": {
                "results": {
                    "raw": {"higher_order_conditional_uq": uq},
                    "repaired": {"higher_order_conditional_uq": uq},
                }
            },
            "full_e2e_cefm": {
                "results": {
                    "raw": {"higher_order_conditional_uq": uq},
                    "repaired": {"higher_order_conditional_uq": uq},
                }
            },
        },
        "primary_learned_method_comparison": {
            "effect": {"estimate": effect, "lower": effect - 1, "upper": effect + 1}
        },
    }


def test_seed_report_aggregation(tmp_path):
    paths = []
    for index, (effect, shift) in enumerate(((-1.0, 0.0), (-2.0, 2.0))):
        path = tmp_path / f"report_{index}.json"
        path.write_text(json.dumps(_report(effect, shift)), encoding="utf-8")
        paths.append(path)
    aggregate = aggregate_comparison_seed_reports(
        paths, seed=4, num_resamples=200
    )
    assert aggregate["num_training_seeds"] == 2
    interval = aggregate["primary_effects_across_training_seeds"]["effect"]
    assert np.isclose(interval["estimate"], -1.5)
    decomposition = aggregate["higher_order_uncertainty_decomposition"][
        "soft_cefm"
    ]["raw"]
    assert np.allclose(decomposition["epistemic_variance"], [2.0, 0.0])
