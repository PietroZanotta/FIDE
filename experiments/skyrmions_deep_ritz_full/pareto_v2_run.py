"""CLI for the isolated official fast skyrmion Pareto-v2 workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

from mfsi.config import load_config

from .galerkin_only import execution_device
from .pareto_v2_common import OUTPUT_ROOT, freeze_protocol
from .pareto_v2_report import run_report
from .pareto_v2_selection import (
    cross_evaluate, freeze_selection, generate_selection_banks, performance_audit,
    screen_starts, select_full, select_tangent,
)
from .pareto_v2_validation import generate_fresh_validation, validate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=(
        "freeze-protocol", "generate-selection-banks", "screen-starts", "select-tangent",
        "select-full", "cross-evaluate", "freeze-selection", "generate-fresh-validation",
        "validate", "report", "performance-audit", "all"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args(); cfg = load_config(args.config)
    routes = {
        "freeze-protocol": freeze_protocol, "generate-selection-banks": generate_selection_banks,
        "screen-starts": screen_starts, "select-tangent": select_tangent, "select-full": select_full,
        "cross-evaluate": cross_evaluate, "freeze-selection": freeze_selection,
        "generate-fresh-validation": generate_fresh_validation, "validate": validate,
        "performance-audit": performance_audit, "report": run_report,
    }
    order = ("freeze-protocol", "generate-selection-banks", "screen-starts", "select-tangent",
             "select-full", "cross-evaluate", "freeze-selection", "generate-fresh-validation",
             "validate", "performance-audit", "report")
    device = jax.devices("cpu")[0] if args.force_cpu else execution_device()
    with jax.default_device(device):
        if args.mode == "all":
            result = None
            for mode in order:
                print(f"starting={mode}", flush=True); result = routes[mode](cfg)
                print(f"completed={mode} passed={result.get('passed', True)}", flush=True)
        else:
            result = routes[args.mode](cfg)
    print(f"mode={args.mode}\noutput_root={OUTPUT_ROOT}\npassed={result.get('passed', True)}")
    if result.get("passed") is False: raise SystemExit(2)


if __name__ == "__main__": main()

