"""CLI for the selection-only ESS qualification; no Full optimizer exists."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

from mfsi.config import load_config
from .ess_study import (OUTPUT_ROOT, freeze_protocol, run_anchors, run_candidate_screen,
                        run_performance_audit, run_staged_rescore)
from .ess_study_report import run_report
from .galerkin_only import execution_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=(
        "freeze-protocol", "anchors", "candidate-screen", "staged-rescore",
        "performance-audit", "report",
    ))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    routes = {"freeze-protocol": freeze_protocol, "anchors": run_anchors,
              "candidate-screen": run_candidate_screen, "staged-rescore": run_staged_rescore,
              "performance-audit": run_performance_audit, "report": run_report}
    device = jax.devices("cpu")[0] if args.force_cpu else execution_device()
    with jax.default_device(device):
        result = routes[args.mode](cfg)
    print(f"mode={args.mode}")
    print(f"output_root={OUTPUT_ROOT}")
    print(f"passed={result.get('passed', True)}")


if __name__ == "__main__":
    main()
