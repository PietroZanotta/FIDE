from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from experiments.active_nematic_unbalance_percentage.robust_selection import (
    PhysicalView,
    RobustPhysicalViewExperiment,
    leave_one_fold_out_views,
)


def test_leave_one_fold_out_views_are_deterministic_and_in_split() -> None:
    runs = np.arange(8)
    first = leave_one_fold_out_views(runs, count=4, seed=17)
    second = leave_one_fold_out_views(runs, count=4, seed=17)
    assert first == second
    assert len(first) == 4
    assert all(len(view) == 6 for view in first)
    assert all(set(view) <= set(runs) for view in first)


def test_robust_experiment_uses_worst_view_and_preserves_receipts() -> None:
    class Sensors:
        @staticmethod
        def canonicalize(eta):
            return eta

    class Experiment:
        sensors = Sensors()
        times = jnp.asarray([0.0, 1.0])

        def __init__(self, value):
            self.value = value

        def mean_metric(self, eta, bank, name):
            return jnp.asarray(self.value) + eta[0]

        def audit_metric(self, eta, bank, name):
            return {
                "value": float(self.value + eta[0]),
                "valid": True,
                "trials": 2,
                "rows": [],
            }

    cfg = {"robust_selection": {"objective": "max"}}
    robust = RobustPhysicalViewExperiment(
        cfg,
        [
            PhysicalView("a", 1, (0, 1), Experiment(2.0)),
            PhysicalView("b", 2, (2, 3), Experiment(5.0)),
        ],
    )
    assert float(robust.mean_metric(jnp.asarray([1.0]), None, "law_risk")) == 6.0
    audit = robust.audit_metric(jnp.asarray([1.0]), None, "law_risk")
    assert audit["value"] == 6.0
    assert audit["view_values"] == [3.0, 6.0]
    assert [row["label"] for row in audit["views"]] == ["a", "b"]


def test_gradient_proxy_subset_does_not_reduce_exact_audit_views() -> None:
    class Sensors:
        @staticmethod
        def canonicalize(eta):
            return eta

    class Experiment:
        sensors = Sensors()
        times = jnp.asarray([0.0, 1.0])

        def __init__(self, value):
            self.value = value

        def mean_metric(self, eta, bank, name):
            return jnp.asarray(self.value)

        def audit_metric(self, eta, bank, name):
            return {"value": self.value, "valid": True, "trials": 1, "rows": []}

    robust = RobustPhysicalViewExperiment(
        {"robust_selection": {"objective": "max", "gradient_physical_views": 1}},
        [
            PhysicalView("fold0-ref1", 1, (0, 1), Experiment(2.0)),
            PhysicalView("fold0-ref2", 2, (0, 1), Experiment(3.0)),
            PhysicalView("fold1-ref1", 1, (2, 3), Experiment(7.0)),
            PhysicalView("fold1-ref2", 2, (2, 3), Experiment(8.0)),
        ],
    )
    assert float(robust.mean_metric(jnp.asarray([0.0]), None, "full_action")) == 3.0
    assert robust.audit_metric(jnp.asarray([0.0]), None, "full_action")["value"] == 8.0
    assert [row["used_for_gradient_proxy"] for row in robust.view_manifest()] == [
        True, True, False, False
    ]
