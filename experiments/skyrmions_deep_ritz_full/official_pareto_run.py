"""CLI for the official fixed-feature K=280 Galerkin Pareto sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from mfsi.config import load_config

from .galerkin_only import execution_device
from .official_pareto_common import OUTPUT_ROOT, freeze_protocol
from .official_pareto_report import run_report
from .official_pareto_selection import (
    freeze_selection, prepare_start_manifest, run_finalist_audits,
    run_reproduction, run_selection_sweep,
)
from .official_pareto_validation import generate_fresh_validation, run_fresh_validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=(
        "freeze-protocol", "reproduce", "prepare-starts", "select",
        "audit-finalists", "freeze-selection", "generate-fresh-validation",
        "validate", "report",
    ))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    routes = {
        "freeze-protocol": freeze_protocol,
        "reproduce": run_reproduction,
        "prepare-starts": prepare_start_manifest,
        "select": run_selection_sweep,
        "audit-finalists": run_finalist_audits,
        "freeze-selection": freeze_selection,
        "generate-fresh-validation": generate_fresh_validation,
        "validate": run_fresh_validation,
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
