from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import pytest

from mfsi.design import OptimizeResult

from experiments.active_nematic_unbalance_percentage.eval import (
    _mean_se,
    _selected_full_audit,
)
from experiments.active_nematic_unbalance_percentage.percentage_selection import (
    _audit_full,
    percentage_risk_budget,
)


ROOT = Path(__file__).resolve().parent


def test_config_declares_only_five_percent_relative_budget() -> None:
    cfg = json.loads((ROOT / "config.json").read_text())
    assert cfg["law"] == {
        "mmd_bandwidths": [0.5, 1.0, 2.0, 4.0],
        "grid_shape": [48, 48, 24],
        "max_relative_risk_violation": 0.05,
    }
    assert "epsilon_r" not in cfg["law"]
    optimization = cfg["optimization"]
    assert (
        optimization["law_steps"],
        optimization["tangent_steps"],
        optimization["full_steps"],
    ) == (60, 50, 40)
    assert optimization["tangent_local_starts"] == 8
    assert optimization["full_local_starts"] == 10
    assert optimization["full_exact_rescore_candidates"] == 4


def test_percentage_budget_is_anchored_to_each_risk_star() -> None:
    epsilon, ceiling = percentage_risk_budget(1.6, 0.05)
    assert epsilon == pytest.approx(0.08)
    assert ceiling == pytest.approx(1.68)
    assert ceiling / 1.6 == pytest.approx(1.05)


def test_fast_eval_marks_omitted_selection_baseline_unavailable() -> None:
    result = {
        "designs": {"law": [1.0, 2.0]},
        "selection_candidates": {"full": []},
    }
    assert _selected_full_audit(result, "law") is None
    assert _mean_se([]) is None


def test_full_audit_always_rescores_mandatory_incumbent() -> None:
    class Bank(NamedTuple):
        plus_sample_indices: jnp.ndarray

    class Sensors:
        @staticmethod
        def canonicalize(eta):
            return eta

    class Experiment:
        sensors = Sensors()

        @staticmethod
        def audit_metric(eta, bank, name):
            assert name == "full_action"
            return {"value": float(eta[0]), "valid": True}

    bank = Bank(jnp.arange(4))
    candidates = [
        OptimizeResult(jnp.asarray([value]), value, True, ())
        for value in (0.0, 1.0, 2.0)
    ]
    screened = [(candidate, {"value": 1.0, "valid": True}) for candidate in candidates]
    best, rows = _audit_full(
        Experiment(), bank, screened,
        prescreen_trials=1,
        finalists=1,
        mandatory=[jnp.asarray([2.0])],
    )
    assert float(best[0].eta[0]) == 0.0
    assert {float(row[0].eta[0]) for row in rows} == {0.0, 2.0}


def test_production_manifest_has_exact_five_percent_ceilings_if_present() -> None:
    manifest = ROOT / "outputs" / "run" / "manifest.json"
    if not manifest.is_file():
        pytest.skip("production results are not present")
    for entry in json.loads(manifest.read_text())["runs"]:
        result = json.loads(Path(entry["result"]).read_text())
        assert result["risk_max"] / result["risk_star"] == pytest.approx(1.05)
        for design in result["validation_designs"].values():
            summary = design["summary"]
            assert summary["valid_trials"] == summary["trials"] == 32
