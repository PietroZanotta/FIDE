"""CLI for the selection-only Galerkin resolution study; no optimizer mode exists."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from mfsi.config import load_config

from .galerkin_only import execution_device
from .resolution_study import (
    OUTPUT_ROOT, analyze_quadrature, freeze_protocol, generate_banks,
    run_basis_rank, run_quadrature, run_start_generator_diagnostic,
)
from .resolution_study_report import run_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=(
        "freeze-protocol", "generate-banks", "quadrature", "analyze-quadrature",
        "basis-rank", "start-generator-diagnostic", "report",
    ))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    routes = {
        "freeze-protocol": freeze_protocol, "generate-banks": generate_banks,
        "quadrature": run_quadrature, "analyze-quadrature": analyze_quadrature,
        "basis-rank": run_basis_rank,
        "start-generator-diagnostic": run_start_generator_diagnostic,
        "report": run_report,
    }
    device = jax.devices("cpu")[0] if args.force_cpu else execution_device()
    with jax.default_device(device):
        result = routes[args.mode](cfg)
    print(f"mode={args.mode}")
    print(f"output_root={OUTPUT_ROOT}")
    print(f"passed={result.get('passed', True)}")
    if result.get("passed") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
