"""Benchmark serial and threaded Pareto multistart execution.

The harness runs two isolated smoke Pareto workflows with identical scientific
inputs.  It changes only ``--multistart-backend``, records end-to-end wall time,
and compares the selected designs and reported selection/validation metrics.
Existing benchmark directories are never overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config_more_training_v2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--percent", type=float, default=2.0)
    parser.add_argument("--reference-seed", type=int, default=20260818)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def _run(args: argparse.Namespace, backend: str) -> tuple[float, Path]:
    output = args.output_root / backend
    if output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark output: {output}")
    command = [
        sys.executable,
        str(SCRIPT_DIR / "run_pareto.py"),
        "--config", str(args.config.resolve()),
        "--input-dir", str(args.input_dir.resolve()),
        "--output", str(output.resolve()),
        "--smoke",
        "--percent", str(args.percent),
        "--reference-seeds", str(args.reference_seed),
        "--multistart-backend", backend,
        "--multistart-workers", str(args.workers),
    ]
    print(f"benchmark backend={backend}", flush=True)
    started = time.perf_counter()
    subprocess.run(command, check=True)
    wall_seconds = time.perf_counter() - started
    tag = f"risk_{f'{args.percent:g}'.replace('.', 'p')}pct"
    return wall_seconds, output / tag / "result.json"


def _selected_row(result: dict[str, Any], design: str) -> dict[str, Any]:
    eta = np.asarray(result["designs"][design], dtype=np.float64)
    stage = "law" if design == "law" else ("tangent" if design == "tangent" else "full")
    rows = [
        row for row in result["selection_candidates"][stage]
        if np.allclose(np.asarray(row["eta"]), eta, rtol=0.0, atol=1.0e-10)
    ]
    if not rows:
        raise RuntimeError(f"selected {design} geometry has no audit row")
    return min(rows, key=lambda row: float(row["audit"]["value"]))


def _comparison(serial: dict[str, Any], threaded: dict[str, Any]) -> dict[str, Any]:
    designs = {}
    metrics = {}
    for design in ("law", "tangent", "unbalanced_full"):
        old_eta = np.asarray(serial["designs"][design], dtype=np.float64)
        new_eta = np.asarray(threaded["designs"][design], dtype=np.float64)
        designs[design] = {
            "serial": old_eta.tolist(),
            "threaded": new_eta.tolist(),
            "max_abs_delta": float(np.max(np.abs(old_eta - new_eta))),
            "allclose_1e-10": bool(np.allclose(old_eta, new_eta, rtol=0.0, atol=1.0e-10)),
        }
        old_audit = _selected_row(serial, design)["audit"]
        new_audit = _selected_row(threaded, design)["audit"]
        metrics[design] = {
            "serial_selection_value": float(old_audit["value"]),
            "threaded_selection_value": float(new_audit["value"]),
            "absolute_delta": abs(float(old_audit["value"]) - float(new_audit["value"])),
        }
    validation = {}
    for design in ("law", "tangent", "unbalanced_full"):
        old = serial["validation_designs"][design]["physical_view_action"]
        new = threaded["validation_designs"][design]["physical_view_action"]
        validation[design] = {
            "serial_mean": float(old["mean"]),
            "threaded_mean": float(new["mean"]),
            "absolute_delta": abs(float(old["mean"]) - float(new["mean"])),
        }
    return {
        "certification_equal": bool(
            serial["selection_certified"] == threaded["selection_certified"]
        ),
        "risk_star_absolute_delta": abs(float(serial["risk_star"]) - float(threaded["risk_star"])),
        "risk_max_absolute_delta": abs(float(serial["risk_max"]) - float(threaded["risk_max"])),
        "designs": designs,
        "selection_metrics": metrics,
        "validation_metrics": validation,
    }


def main() -> None:
    args = parse_args()
    if args.workers < 2:
        raise ValueError("--workers must be >= 2")
    if args.output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite benchmark root: {args.output_root}"
        )
    args.output_root.mkdir(parents=True)

    serial_seconds, serial_path = _run(args, "serial")
    threaded_seconds, threaded_path = _run(args, "threaded")
    serial = json.loads(serial_path.read_text(encoding="utf-8"))
    threaded = json.loads(threaded_path.read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "scope": "end-to-end smoke Pareto workflow",
        "percent": float(args.percent),
        "reference_seed": int(args.reference_seed),
        "workers": int(args.workers),
        "serial_wall_seconds": serial_seconds,
        "threaded_wall_seconds": threaded_seconds,
        "speedup_serial_over_threaded": serial_seconds / threaded_seconds,
        "comparison": _comparison(serial, threaded),
        "serial_result": str(serial_path.resolve()),
        "threaded_result": str(threaded_path.resolve()),
        "order_note": "serial ran first, then threaded; both used fresh processes and output trees",
    }
    report_path = args.output_root / "benchmark.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"benchmark complete: {report_path}", flush=True)


if __name__ == "__main__":
    main()
