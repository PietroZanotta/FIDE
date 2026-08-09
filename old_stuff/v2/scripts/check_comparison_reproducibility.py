#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def _canonical_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    report.get("metadata", {}).pop("created_utc", None)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    left = Path(args.left)
    right = Path(args.right)
    report_equal = _canonical_report(left / "scientific_comparison_report.json") == _canonical_report(
        right / "scientific_comparison_report.json"
    )
    arrays_equal = True
    with np.load(left / "scientific_comparison_arrays.npz") as a, np.load(
        right / "scientific_comparison_arrays.npz"
    ) as b:
        arrays_equal = set(a.files) == set(b.files) and all(
            np.array_equal(a[key], b[key]) for key in a.files
        )
    payload = {"report_equal": report_equal, "arrays_equal": arrays_equal, "passed": report_equal and arrays_equal}
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
