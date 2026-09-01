from __future__ import annotations

"""Orchestrate v6 with explicit arm/reference/hidden freeze boundaries."""

import argparse
import json
from pathlib import Path
import time

from common import load_config, write_json_atomic
from v6_reference_ensemble import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    prepare_common_inputs,
    train_reference_split,
    v6_paths,
)
from v6_select import combine_freeze, select_arm, select_shared
from v6_validate import validate_v6


def run(cfg, output_dir, stage):
    paths = v6_paths(output_dir)
    started = time.perf_counter()
    result = None
    if stage in {"prepare", "all"}:
        result = prepare_common_inputs(cfg, output_dir)
    if stage in {"design-references", "all"}:
        result = train_reference_split(cfg, output_dir, "design")
    if stage in {"shared", "all"}:
        result = select_shared(cfg, output_dir)
    if stage in {"v6a", "all"}:
        result = select_arm(cfg, output_dir, Path(__file__).resolve().parent / "configs" / "production_v6a.json")
    if stage in {"v6b", "all"}:
        result = select_arm(cfg, output_dir, Path(__file__).resolve().parent / "configs" / "production_v6b.json")
    if stage in {"combine", "all"}:
        result = combine_freeze(cfg, output_dir)
    if stage in {"evaluation-references", "all"}:
        result = train_reference_split(cfg, output_dir, "evaluation")
    if stage in {"validate", "all"}:
        result = validate_v6(cfg, output_dir)
    paths["results"].mkdir(parents=True, exist_ok=True)
    write_json_atomic(paths["results"] / f"last_{stage}_runtime.json", {
        "stage": stage, "elapsed_seconds": time.perf_counter() - started
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stage",
        choices=("prepare", "design-references", "shared", "v6a", "v6b", "combine", "evaluation-references", "validate", "all"),
        required=True,
    )
    args = parser.parse_args()
    result = run(load_config(args.config), args.output_dir, args.stage)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

