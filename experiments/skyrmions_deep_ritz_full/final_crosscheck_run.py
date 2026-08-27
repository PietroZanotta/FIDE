"""CLI for retrospective frozen-design Galerkin cross-checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from mfsi.config import load_config

from .final_crosscheck import (
    OUTPUT_ROOT, run_gradient_check, run_protocol, run_selection_ladder,
    run_summary, run_validation_ladder,
)
from .galerkin_only import execution_device
from .production_artifacts import PRODUCTION_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=True,
        choices=(
            "protocol", "gradient-K280", "selection-ladder",
            "validation-ladder", "summarize",
        ),
    )
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    artifact_dir = PRODUCTION_ROOT / "artifacts"
    if not (artifact_dir / "isolated_artifact_manifest.json").is_file():
        raise SystemExit("isolated production artifacts are not materialized")
    device = jax.devices("cpu")[0] if args.force_cpu else execution_device()
    routes = {
        "protocol": (run_protocol, OUTPUT_ROOT / "protocol"),
        "gradient-K280": (run_gradient_check, OUTPUT_ROOT / "gradient"),
        "selection-ladder": (run_selection_ladder, OUTPUT_ROOT / "selection_ladder"),
        "validation-ladder": (run_validation_ladder, OUTPUT_ROOT / "validation_ladder"),
        "summarize": (run_summary, OUTPUT_ROOT / "summary"),
    }
    function, output_dir = routes[args.mode]
    with jax.default_device(device):
        result = function(cfg, artifact_dir, output_dir)
    print(f"result={output_dir / 'result.json'}")
    print(f"passed={result.get('passed', False)}")


if __name__ == "__main__":
    main()
