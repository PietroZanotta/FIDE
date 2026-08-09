#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml

from manybody_completion.config import load_config, with_seed
from manybody_completion.experiment import write_run_artifacts
from manybody_completion.seed_study import aggregate_reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scientific_seed_sweep.yaml")
    parser.add_argument("--output")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    sweep = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = load_config(sweep["base_config"])
    output_root = Path(args.output or sweep["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    reports = []
    paths = []
    for seed in sweep["seeds"]:
        run_dir = output_root / f"seed_{seed}"
        report_path = run_dir / "scientific_comparison_report.json"
        if args.aggregate_only:
            if not report_path.is_file():
                raise SystemExit(f"missing {report_path}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = write_run_artifacts(with_seed(base, int(seed)), run_dir)
        reports.append(report)
        paths.append(str(report_path))
    aggregate = aggregate_reports(reports, paths)
    aggregate_path = output_root / "multi_seed_scientific_comparison.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
