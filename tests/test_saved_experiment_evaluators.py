"""Regression tests for the read-only saved-result command-line evaluators."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("toy_example_percentage", "eval.py", ("Tangent", "95.498884", "18.55%")),
    ("toy_example_percentage", "eval_pareto.py", ("Tangent", "26.618646", "34.47%")),
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
