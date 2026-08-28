from __future__ import annotations

from experiments.active_nematic_unbalance_percentage.run_pareto import (
    _evaluate_validation,
    _full_action_row,
    _physical_view_summary,
)


def _view(runs, value):
    return {
        "run_indices": runs,
        "summary": {
            "metrics": {"full_unbalanced_action_total": {"mean": value}}
        },
    }


def test_physical_uncertainty_is_leave_fold_out_jackknife() -> None:
    summary = _physical_view_summary([
        _view([0, 1], 1.0),
        _view([0, 1], 3.0),
        _view([2, 3], 5.0),
        _view([2, 3], 7.0),
    ])
    assert summary["mean"] == 4.0
    assert summary["se_across_views"] == 2.0
    assert summary["physical_folds"] == 2


def test_law_action_is_read_from_mandatory_full_audit() -> None:
    result = {
        "designs": {"law": [1.0, 2.0]},
        "selection_candidates": {
            "law": [{"eta": [1.0, 2.0], "audit": {"value": 0.25}}],
            "full": [{"eta": [1.0, 2.0], "audit": {"value": 9.5}}],
        },
    }
    assert _full_action_row(result, "law")["audit"]["value"] == 9.5


def test_validation_cache_reuses_only_identical_canonical_designs() -> None:
    class Sensors:
        @staticmethod
        def canonicalize(eta):
            return eta

    class Experiment:
        sensors = Sensors()

        def __init__(self):
            self.calls = 0

        def certified_trial_rows(self, eta, bank):
            self.calls += 1
            value = float(eta[0])
            return [{
                "trial": 0,
                "valid": True,
                "full_unbalanced_action_total": value,
                "sensor_geometry": [[value, 0.0]],
            }]

    exp = Experiment()
    view = type("View", (), {
        "label": "view0",
        "reference_seed": 1,
        "run_indices": (0, 1),
        "experiment": exp,
    })()
    cache = {}
    stats = {"hits": 0, "misses": 0}
    result = _evaluate_validation(
        [view],
        object(),
        {"law": [1.0], "tangent": [1.0], "unbalanced_full": [2.0]},
        cache=cache,
        cache_stats=stats,
    )
    assert exp.calls == 2
    assert stats == {"hits": 1, "misses": 2}
    assert result["law"] == result["tangent"]
    assert result["law"] != result["unbalanced_full"]
