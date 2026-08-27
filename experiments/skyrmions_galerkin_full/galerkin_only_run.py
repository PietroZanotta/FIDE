"""Clean command-line entry point for Galerkin-only production modes."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from mfsi.config import load_config

from .galerkin_only import GALERKIN_ONLY_ROOT, execution_device
from .galerkin_only_workflow import (
    run_galerkin_only_benchmark, run_galerkin_only_convergence,
    run_galerkin_only_optimization, run_galerkin_only_validation,
    run_selected_K_profile,
)
from .production_artifacts import PRODUCTION_ROOT


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=True,
        choices=(
            "benchmark", "convergence", "profile-selected-K",
            "optimize-3pct", "validate-3pct",
        ),
    )
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument("--frozen-source", type=Path)
    parser.add_argument("--selection-result", type=Path)
    parser.add_argument(
        "--force-cpu", action="store_true",
        help="diagnostic equivalence override; normal execution prefers GPU",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    cfg = load_config(args.config)
    artifact_dir = PRODUCTION_ROOT / "artifacts"
    if not (artifact_dir / "isolated_artifact_manifest.json").is_file():
        raise SystemExit("isolated production artifacts are not materialized")
    device = jax.devices("cpu")[0] if args.force_cpu else execution_device()
    leaves = {
        "benchmark": "benchmark",
        "convergence": "convergence",
        "profile-selected-K": "profile_selected_K",
        "optimize-3pct": "selection",
        "validate-3pct": "validation",
    }
    output_dir = GALERKIN_ONLY_ROOT / leaves[args.mode]
    with jax.default_device(device):
        if args.mode == "benchmark":
            result = run_galerkin_only_benchmark(cfg, artifact_dir, output_dir)
        elif args.mode == "convergence":
            result = run_galerkin_only_convergence(cfg, artifact_dir, output_dir)
        elif args.mode == "profile-selected-K":
            result = run_selected_K_profile(cfg, artifact_dir, output_dir)
        elif args.mode == "optimize-3pct":
            result = run_galerkin_only_optimization(cfg, artifact_dir, output_dir)
        else:
            selection_result = args.selection_result or (
                GALERKIN_ONLY_ROOT / "selection" / "result.json"
            )
            result = run_galerkin_only_validation(
                cfg, artifact_dir, output_dir,
                selection_result=selection_result,
            )
    print(f"result={output_dir / 'result.json'}")
    print(f"passed={result.get('passed', False)}")
    if not result.get("passed", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
