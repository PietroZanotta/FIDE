#!/usr/bin/env python3
"""Compare two scientific-comparison runs after removing wall-clock fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _normalize_report(
    value: Any,
    *,
    allow_backend_difference: bool,
) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalize_report(
                item,
                allow_backend_difference=allow_backend_difference,
            )
            for key, item in value.items()
            if key
            not in {
                "runtime_seconds",
                "training_seconds",
                "sampling_seconds",
                "flow_ablation_report",
            }
            and not (allow_backend_difference and key == "solver_backend")
        }
    if isinstance(value, list):
        return [
            _normalize_report(
                item,
                allow_backend_difference=allow_backend_difference,
            )
            for item in value
        ]
    return value


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-backend-difference",
        action="store_true",
        help=(
            "Ignore the declared solver-backend class while requiring all "
            "numerical report values and arrays to match."
        ),
    )
    args = parser.parse_args()
    left_report = _normalize_report(
        json.loads((args.left / "scientific_comparison_report.json").read_text()),
        allow_backend_difference=args.allow_backend_difference,
    )
    right_report = _normalize_report(
        json.loads((args.right / "scientific_comparison_report.json").read_text()),
        allow_backend_difference=args.allow_backend_difference,
    )
    report_equal = left_report == right_report
    left_arrays = _load_arrays(args.left / "scientific_comparison_arrays.npz")
    right_arrays = _load_arrays(args.right / "scientific_comparison_arrays.npz")
    arrays_equal = set(left_arrays) == set(right_arrays) and all(
        np.array_equal(left_arrays[name], right_arrays[name], equal_nan=True)
        for name in left_arrays
    )
    result = {
        "status": "passed" if report_equal and arrays_equal else "failed",
        "report_equal_excluding_runtime": report_equal,
        "arrays_exactly_equal": arrays_equal,
        "allowed_backend_difference": args.allow_backend_difference,
        "left": str(args.left),
        "right": str(args.right),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
