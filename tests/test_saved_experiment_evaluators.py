"""Regression tests for the read-only saved-result command-line evaluators."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("skyrmions_deep_ritz", "eval.py", ("Tangent", "0.230970", "30.44%")),
    ("skyrmions_deep_ritz", "eval_pareto.py", ("Tangent", "0.068152", "32.24%")),
    ("skyrmions_galerkin", "eval.py", ("Tangent", "0.028388", "21.82%")),
    ("skyrmions_galerkin", "eval_pareto.py", ("Tangent", "0.038079", "-5.13%")),
    ("toy_example_percentage", "eval.py", ("Tangent", "95.498884", "18.55%")),
    ("toy_example_percentage", "eval_pareto.py", ("Tangent", "26.618646", "34.47%")),
    ("vortices_percentage", "eval.py", ("Tangent", "0.293596", "25.94%")),
    ("vortices_percentage", "eval_pareto.py", ("Tangent", "107.999874", "70.18%")),
)


@pytest.mark.parametrize(("experiment", "script", "expected"), CASES)
def test_saved_evaluator_output(
    experiment: str,
    script: str,
    expected: tuple[str, ...],
) -> None:
    evaluator = REPOSITORY_ROOT / "experiments" / experiment / script
    completed = subprocess.run(
        [sys.executable, str(evaluator)],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "uncertainty:" in completed.stdout
    assert "selection" in completed.stdout
    assert "validation" in completed.stdout
    assert "PASS" not in completed.stdout
    assert "FAIL" not in completed.stdout
    for text in expected:
        assert text in completed.stdout
