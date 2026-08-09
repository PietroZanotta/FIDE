"""Command-line entrypoint for one flow-matching and DiffPOP comparison."""

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
    print(f"wrote flow-matching + DiffPOP comparison to {args.output}")
    print(
        "DiffPOP calibration non-inferior to direct flow: "
        f"{decision['diffpop_full_calibration_noninferior_to_direct']}"
    )
    print(
        "Post-hoc DiffPOP supported in this run: "
        f"{decision['diffpop_posthoc_supported_in_this_run']}"
    )
    print(f"Full-E2E DiffPOP supported in this run: {decision['diffpop_full_supported_in_this_run']}")
    print(
        "Synergy DiffPOP supported in this run: "
        f"{decision['diffpop_synergy_supported_in_this_run']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
