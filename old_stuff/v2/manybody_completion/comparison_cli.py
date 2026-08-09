"""Command-line entrypoint for one DiffPOP comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config, with_seed
from .experiment import write_run_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/diffpop_micro.yaml")
    parser.add_argument("--output", default="artifacts/diffpop_micro")
    parser.add_argument("--seed", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.seed is not None:
        config = with_seed(config, args.seed)
    report = write_run_artifacts(config, Path(args.output))
    decision = report["decision_summary"]
    print(f"wrote DiffPOP comparison to {args.output}")
    print(f"Full-E2E calibration gate: {decision['calibration_gate']}")
    print(f"Full-E2E mode gate: {decision['mode_gate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
