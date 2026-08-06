"""Run independently trained comparison seeds and aggregate their UQ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from manybody_completion.config import load_yaml
from manybody_completion.scientific_comparison import run_scientific_comparison
from manybody_completion.seed_study import aggregate_comparison_seed_reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Reuse completed per-seed reports and only rebuild the aggregate.",
    )
    args = parser.parse_args()
    configuration = load_yaml(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    report_paths: list[Path] = []
    for seed in configuration["seeds"]:
        seed_output = args.output / f"seed_{int(seed)}"
        report_path = seed_output / "scientific_comparison_report.json"
        if not args.aggregate_only:
            run_scientific_comparison(
                configuration["comparison_config"],
                seed_output,
                rerun_flow=True,
                seed_override=int(seed),
            )
        if not report_path.is_file():
            raise FileNotFoundError(f"missing completed seed report: {report_path}")
        report_paths.append(report_path)
    aggregate = aggregate_comparison_seed_reports(
        report_paths,
        seed=int(configuration["aggregate_seed"]),
        num_resamples=int(configuration["bootstrap_resamples"]),
    )
    destination = args.output / "multi_seed_scientific_comparison.json"
    destination.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(destination)


if __name__ == "__main__":
    main()
