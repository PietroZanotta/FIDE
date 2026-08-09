#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    directory = Path(args.directory)
    required = [
        directory / "scientific_comparison_report.json",
        directory / "scientific_comparison_summary.csv",
        directory / "scientific_comparison_arrays.npz",
        directory / "SCIENTIFIC_COMPARISON_SUMMARY.md",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing artifacts: {missing}")
    report = json.loads(required[0].read_text(encoding="utf-8"))
    for name, method in report["methods"].items():
        for key in (
            "moment_error",
            "ess_fraction",
            "mode_probability_error",
            "hidden_energy_score",
        ):
            value = float(method[key])
            if not math.isfinite(value):
                raise SystemExit(f"{name}.{key} is non-finite")
        if method["moment_error"] < 0 or method["mode_probability_error"] < 0:
            raise SystemExit(f"{name} has a negative error")
    with required[1].open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(report["methods"]):
        raise SystemExit("CSV method count does not match JSON report")
    with np.load(required[2]) as arrays:
        if not arrays.files:
            raise SystemExit("NPZ contains no arrays")
        for key in arrays.files:
            if not np.all(np.isfinite(arrays[key])):
                raise SystemExit(f"array {key} contains non-finite values")
    print(f"validated {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
