"""Benchmark validation reuse from one identical frozen selection receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config_more_training_v2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--selection-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--percent", type=float, default=2.0)
    parser.add_argument("--reference-seed", type=int, default=20260818)
    return parser.parse_args()


def _seed_selection(args: argparse.Namespace, output: Path) -> Path:
    tag = f"risk_{f'{args.percent:g}'.replace('.', 'p')}pct"
    result_path = output / tag / "result.json"
    result_path.parent.mkdir(parents=True)
    result = json.loads(args.selection_result.read_text(encoding="utf-8"))
    if float(result["allowance_percent"]) != float(args.percent):
        raise ValueError("selection receipt allowance does not match --percent")
    result["validation_designs"] = None
    result.pop("validation_execution", None)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result_path


def _run(args: argparse.Namespace, label: str, *, reuse: bool) -> tuple[float, Path]:
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
    ]
    if reuse:
        command.append("--reuse-exact-evaluations")
    print(f"fixed-validation benchmark evaluation_reuse={reuse}", flush=True)
    started = time.perf_counter()
    subprocess.run(command, check=True)
    return time.perf_counter() - started, result_path


def _numerical_comparison(
    baseline: Any,
    reuse: Any,
    *,
    rtol: float = 1.0e-8,
    atol: float = 1.0e-7,
) -> dict[str, Any]:
    numeric_count = 0
    max_absolute = (0.0, "")
    max_relative = (0.0, "")
    allclose = True
    nonnumeric_equal = True

    def walk(old: Any, new: Any, path: str) -> None:
        nonlocal numeric_count, max_absolute, max_relative, allclose, nonnumeric_equal
        if isinstance(old, dict) and isinstance(new, dict):
            if set(old) != set(new):
                nonnumeric_equal = False
            for key in set(old) & set(new):
                walk(old[key], new[key], f"{path}.{key}")
            return
        if isinstance(old, list) and isinstance(new, list):
            if len(old) != len(new):
                nonnumeric_equal = False
            for index, (left, right) in enumerate(zip(old, new, strict=False)):
                walk(left, right, f"{path}[{index}]")
            return
        numeric = (
            isinstance(old, (int, float))
            and not isinstance(old, bool)
            and isinstance(new, (int, float))
            and not isinstance(new, bool)
        )
        if numeric:
            numeric_count += 1
            absolute = abs(float(old) - float(new))
            relative = absolute / max(abs(float(old)), abs(float(new)), 1.0e-300)
            if absolute > max_absolute[0]:
                max_absolute = (absolute, path)
            if relative > max_relative[0]:
                max_relative = (relative, path)
            if absolute > atol + rtol * abs(float(new)):
                allclose = False
            return
        if old != new:
            nonnumeric_equal = False

    walk(baseline, reuse, "validation")
    return {
        "rtol": rtol,
        "atol": atol,
        "numeric_values_compared": numeric_count,
        "numeric_allclose": allclose,
        "nonnumeric_values_equal": nonnumeric_equal,
        "max_absolute_difference": max_absolute[0],
        "max_absolute_difference_path": max_absolute[1],
        "max_relative_difference": max_relative[0],
        "max_relative_difference_path": max_relative[1],
    }


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite benchmark root: {args.output_root}"
        )
    args.output_root.mkdir(parents=True)
    baseline_seconds, baseline_path = _run(args, "baseline", reuse=False)
    reuse_seconds, reuse_path = _run(args, "reuse", reuse=True)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    reuse = json.loads(reuse_path.read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "scope": "validation-only replay from an identical frozen selection receipt",
        "percent": float(args.percent),
        "reference_seed": int(args.reference_seed),
        "baseline_wall_seconds": baseline_seconds,
        "reuse_wall_seconds": reuse_seconds,
        "speedup_baseline_over_reuse": baseline_seconds / reuse_seconds,
        "wall_seconds_saved": baseline_seconds - reuse_seconds,
        "validation_payload_exactly_equal": (
            baseline["validation_designs"] == reuse["validation_designs"]
        ),
        "validation_numerical_comparison": _numerical_comparison(
            baseline["validation_designs"], reuse["validation_designs"]
        ),
        "selection_designs_exactly_equal": baseline["designs"] == reuse["designs"],
        "reuse_validation_execution": reuse.get("validation_execution"),
        "baseline_result": str(baseline_path.resolve()),
        "reuse_result": str(reuse_path.resolve()),
        "order_note": "baseline ran first, then reuse; both replayed the same selection JSON",
    }
    report_path = args.output_root / "benchmark.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"benchmark complete: {report_path}", flush=True)


if __name__ == "__main__":
    main()
