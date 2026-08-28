"""Benchmark scalar geometry reuse with a fixed multi-trial validation receipt."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
for path in (REPO_ROOT / "src", REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mfsi.cache import fingerprint
from mfsi.config import load_config

from benchmark_fixed_validation_reuse import _numerical_comparison


DEFAULT_CONFIG = SCRIPT_DIR / "config_more_training_v2.json"
METHODOLOGY_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--selection-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--percent", type=float, default=2.0)
    parser.add_argument("--reference-seed", type=int, default=20260818)
    parser.add_argument("--validation-trials", type=int, default=4)
    return parser.parse_args()


def _benchmark_config(args: argparse.Namespace) -> Path:
    raw = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    raw.setdefault("smoke", {}).setdefault("randomness", {})[
        "validation_trials"
    ] = int(args.validation_trials)
    path = args.output_root / "benchmark_config.json"
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return path


def _point_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config.resolve(), smoke=True)
    cfg["reference_training"]["seeds"] = [int(args.reference_seed)]
    point_cfg = copy.deepcopy(cfg)
    point_cfg["law"]["max_relative_risk_violation"] = float(args.percent) / 100.0
    point_cfg["optimization"]["pareto_methodology_version"] = METHODOLOGY_VERSION
    return point_cfg


def _seed_selection(args: argparse.Namespace, output: Path) -> Path:
    tag = f"risk_{f'{args.percent:g}'.replace('.', 'p')}pct"
    result_path = output / tag / "result.json"
    result_path.parent.mkdir(parents=True)
    result = json.loads(args.selection_result.read_text(encoding="utf-8"))
    point_cfg = _point_config(args)
    result["config"] = point_cfg
    result["config_hash"] = fingerprint(point_cfg)
    result["validation_designs"] = None
    result.pop("validation_execution", None)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result_path


def _run(
    args: argparse.Namespace,
    label: str,
    *,
    reuse_geometry: bool,
) -> tuple[float, Path]:
    output = args.output_root / label
    if output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark output: {output}")
    result_path = _seed_selection(args, output)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "run_pareto.py"),
        "--config", str(args.config.resolve()),
        "--input-dir", str(args.input_dir.resolve()),
        "--output", str(output.resolve()),
        "--smoke",
        "--percent", str(args.percent),
        "--reference-seeds", str(args.reference_seed),
        "--multistart-backend", "serial",
        "--reuse-exact-evaluations",
        "--reuse-prefix-banks",
    ]
    if reuse_geometry:
        command.append("--reuse-scalar-geometry")
    print(
        f"scalar-geometry benchmark reuse_geometry={reuse_geometry}", flush=True
    )
    started = time.perf_counter()
    subprocess.run(command, check=True)
    return time.perf_counter() - started, result_path


def main() -> None:
    args = parse_args()
    if args.validation_trials < 2:
        raise ValueError("--validation-trials must be >= 2")
    if args.output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite benchmark root: {args.output_root}"
        )
    args.output_root.mkdir(parents=True)
    args.config = _benchmark_config(args)

    baseline_seconds, baseline_path = _run(
        args, "without_geometry_reuse", reuse_geometry=False
    )
    reuse_seconds, reuse_path = _run(
        args, "with_geometry_reuse", reuse_geometry=True
    )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    reuse = json.loads(reuse_path.read_text(encoding="utf-8"))
    baseline_validation = baseline["validation_designs"]
    reuse_validation = reuse["validation_designs"]
    report = {
        "schema_version": 1,
        "scope": "fixed-selection multi-trial scalar validation",
        "percent": float(args.percent),
        "reference_seed": int(args.reference_seed),
        "validation_trials_per_view": int(args.validation_trials),
        "without_geometry_reuse_wall_seconds": baseline_seconds,
        "with_geometry_reuse_wall_seconds": reuse_seconds,
        "speedup": baseline_seconds / reuse_seconds,
        "wall_seconds_saved": baseline_seconds - reuse_seconds,
        "selection_designs_exactly_equal": baseline["designs"] == reuse["designs"],
        "validation_payload_exactly_equal": baseline_validation == reuse_validation,
        "validation_numerical_comparison": _numerical_comparison(
            baseline_validation, reuse_validation
        ),
        "without_geometry_reuse_result": str(baseline_path.resolve()),
        "with_geometry_reuse_result": str(reuse_path.resolve()),
        "order_note": (
            "without-geometry-reuse ran first; both arms used exact-result/prefix "
            "reuse and the same frozen selection receipt"
        ),
    }
    report_path = args.output_root / "benchmark.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"benchmark complete: {report_path}", flush=True)


if __name__ == "__main__":
    main()
