"""Generate and validate the exact periodic homometric benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from .config import load_yaml
from .datasets import save_dataset
from .homometric import (
    HomometricDatasetConfig,
    generate_homometric_dataset,
    validate_homometric_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    raw = load_yaml(args.config)
    config = HomometricDatasetConfig.from_mapping(raw)
    jax.config.update("jax_enable_x64", config.dtype == "float64")
    arrays, metadata = generate_homometric_dataset(config)
    archive_path, metadata_path = save_dataset(arrays, metadata, args.output)
    report = validate_homometric_dataset(arrays, metadata)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "archive": str(archive_path),
                "metadata": str(metadata_path),
                "report": str(args.report),
                **report,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not report["passed"]:
        raise SystemExit("homometric benchmark validation failed")


if __name__ == "__main__":
    main()
