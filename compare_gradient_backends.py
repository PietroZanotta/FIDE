#!/usr/bin/env python3
"""Compare JAX and served-Tesseract end-to-end gradient smoke outputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "results" / "backend_smoke"


def main():
    reports = {
        name: json.loads((BASE / f"{name}.json").read_text())
        for name in ("jax", "tesseract")
    }
    checks = {}
    for engine, fields in {
        "rollout": (
            "selected_parameters", "candidate_parameters", "selection_losses",
            "adaptation_losses", "gradient_norms", "parameter_trace",
        ),
        "fiber": (
            "selected_theta", "candidate_thetas", "selection_objectives",
            "adaptation_objectives", "gradient_norms",
        ),
    }.items():
        left = reports["jax"]["engines"][engine]["output"]
        right = reports["tesseract"]["engines"][engine]["output"]
        checks[engine] = {
            field: float(np.max(np.abs(
                np.asarray(left[field], dtype=float)
                - np.asarray(right[field], dtype=float)
            ))) for field in fields
        }
        checks[engine]["selected_step_equal"] = (
            left["selected_step"] == right["selected_step"]
        )
    maximum = max(
        value
        for engine in checks.values()
        for key, value in engine.items()
        if key != "selected_step_equal"
    )
    result = {
        "passed": maximum <= 2e-9 and all(
            engine["selected_step_equal"] for engine in checks.values()
        ),
        "tolerance": 2e-9,
        "maximum_absolute_difference": maximum,
        "checks": checks,
    }
    (BASE / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("gradient backend parity failed")


if __name__ == "__main__":
    main()
