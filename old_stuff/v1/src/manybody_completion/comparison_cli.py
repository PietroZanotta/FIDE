"""Command-line entry point for the complete scientific comparison."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .scientific_comparison import run_scientific_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rerun-flow", action="store_true")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    report = run_scientific_comparison(
        args.config,
        args.output,
        rerun_flow=args.rerun_flow,
        seed_override=args.seed,
    )
    concise = {
        "status": report["status"],
        "decision_summary": report["decision_summary"],
        "report": str(args.output / "scientific_comparison_report.json"),
    }
    print(json.dumps(concise, indent=2, sort_keys=True))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
