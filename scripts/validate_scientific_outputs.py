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
