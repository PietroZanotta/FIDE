"""Validate a generated dataset by recomputing statistics and invariances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .validation import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_dataset(args.dataset, args.metadata)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote validation report: {args.output}")
    if not report["numerical_validation_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
