#!/usr/bin/env python3
"""Validate comparison artifacts without trusting object arrays or NaNs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _check_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _check_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_finite(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {path}: {value}")


def _check_higher_order_uq(value: dict[str, Any], path: str) -> None:
    required = {
        "condition",
        "descriptor_dimension",
        "num_predictive_ensembles",
        "replicas_per_ensemble",
        "num_reference_configurations",
        "predictive_intervals",
        "multivariate_energy_score",
        "mode_probability_intervals",
        "mode_probability_estimate",
        "mode_probability_target",
        "mode_probability_total_variation",
        "normalized_mode_entropy",
        "epistemic_status",
    }
    missing = required.difference(value)
    if missing:
        raise ValueError(f"missing UQ fields at {path}: {sorted(missing)}")
    if value["condition"] != "shared exact pair-statistic vector c":
        raise ValueError(f"unexpected conditioning declaration at {path}")
    descriptor_dimension = int(value["descriptor_dimension"])
    if descriptor_dimension < 1:
        raise ValueError(f"invalid descriptor dimension at {path}")
    for count_name in (
        "num_predictive_ensembles",
        "replicas_per_ensemble",
        "num_reference_configurations",
    ):
        if int(value[count_name]) < 1:
            raise ValueError(f"invalid {count_name} at {path}")

    probabilities = value["mode_probability_estimate"]
    probability_values = np.asarray(
        [probabilities[name] for name in ("A", "B", "far")],
        dtype=np.float64,
    )
    if np.any((probability_values < 0.0) | (probability_values > 1.0)):
        raise ValueError(f"invalid mode probabilities at {path}")
    if not np.isclose(np.sum(probability_values), 1.0, atol=1e-10):
        raise ValueError(f"mode probabilities do not sum to one at {path}")
    total_variation = float(value["mode_probability_total_variation"])
    entropy = float(value["normalized_mode_entropy"])
    if not 0.0 <= total_variation <= 1.0:
        raise ValueError(f"invalid mode total variation at {path}")
    if not -1e-12 <= entropy <= 1.0 + 1e-12:
        raise ValueError(f"invalid normalized mode entropy at {path}")

    intervals = value["predictive_intervals"]
    if not intervals:
        raise ValueError(f"no predictive intervals at {path}")
    for level, interval in intervals.items():
        level_value = float(level)
        if not 0.0 < level_value < 1.0:
            raise ValueError(f"invalid interval level at {path}: {level}")
        lower = np.asarray(interval["lower"]["estimate"], dtype=np.float64)
        upper = np.asarray(interval["upper"]["estimate"], dtype=np.float64)
        width = np.asarray(interval["width"], dtype=np.float64)
        if lower.shape != (descriptor_dimension,) or upper.shape != lower.shape:
            raise ValueError(f"invalid interval endpoint shape at {path}/{level}")
        if width.shape != lower.shape or np.any(width < -1e-12):
            raise ValueError(f"invalid interval width at {path}/{level}")
        if not np.allclose(width, upper - lower, atol=1e-10, rtol=1e-10):
            raise ValueError(f"inconsistent interval width at {path}/{level}")
        coverage = float(interval["reference_coverage_mean"])
        if not 0.0 <= coverage <= 1.0:
            raise ValueError(f"invalid reference coverage at {path}/{level}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.directory / "scientific_comparison_report.json"
    arrays_path = args.directory / "scientific_comparison_arrays.npz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _check_finite(report)
    expected_methods = {
        "reverse_monte_carlo",
        "iterative_boltzmann_inversion",
        "soft_cefm",
        "full_e2e_cefm",
    }
    if set(report["methods"]) != expected_methods:
        raise ValueError("comparison report has an unexpected method set")
    for method in expected_methods:
        for stage in ("raw", "repaired"):
            result = report["methods"][method]["results"][stage]
            if "higher_order_conditional_uq" not in result:
                raise ValueError(f"missing UQ for {method}/{stage}")
            _check_higher_order_uq(
                result["higher_order_conditional_uq"],
                f"methods.{method}.results.{stage}.higher_order_conditional_uq",
            )
    with np.load(arrays_path, allow_pickle=False) as archive:
        if not archive.files:
            raise ValueError("comparison array archive is empty")
        for name in archive.files:
            array = archive[name]
            if array.dtype == object:
                raise ValueError(f"object array is forbidden: {name}")
            if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
                raise ValueError(f"non-finite array values: {name}")
    print(json.dumps({"status": "passed", "directory": str(args.directory)}, indent=2))


if __name__ == "__main__":
    main()
