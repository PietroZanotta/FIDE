from __future__ import annotations

import ast
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parent.parent / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from frozen_diagnostic_core import curve_error_metrics, paired_statistics


def test_paired_statistics_preserve_trial_pairing():
    law = np.asarray([1.0, 2.0, 4.0, 8.0])
    full = np.asarray([0.5, 2.5, 3.0, 9.0])
    result = paired_statistics(law, full, bootstrap_seed=3)
    difference = full - law
    assert result["valid_pair_count"] == 4
    assert np.isclose(result["difference_full_minus_law"]["mean"], np.mean(difference))
    assert np.isclose(result["relative_reduction_trialwise"]["mean"], np.mean((law - full) / law))
    assert result["fraction_full_lower"] == 0.5
    assert result["fraction_full_higher"] == 0.5


def test_curve_errors_are_time_and_channel_resolved():
    times = np.linspace(0.0, 1.0, 5)
    oracle = np.stack([times, times**2], axis=-1)
    predicted = oracle + np.asarray([0.01, -0.02])
    result = curve_error_metrics(predicted, oracle, times)
    assert len(result["channels"]) == 2
    assert np.isclose(result["channels"][0]["rmse"], 0.01)
    assert np.isclose(result["channels"][1]["rmse"], 0.02)
    assert len(result["channels"][0]["error_by_time"]) == len(times)


def test_diagnostic_script_has_no_selection_or_optimizer_import():
    tree = ast.parse((HERE / "diagnose_frozen_selection.py").read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "select" not in imported
    assert "selection" not in imported
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not any("optim" in name.lower() for name in calls)
