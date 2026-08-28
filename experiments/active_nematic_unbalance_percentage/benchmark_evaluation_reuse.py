"""Benchmark exact-evaluation reuse against unchanged serial execution.

Both arms run fresh isolated smoke Pareto workflows with the original serial
multistart backend.  The optimized arm adds only byte-exact audit/validation
memoization and remembered scalar-authority fallback decisions.
"""

from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--percent", type=float, default=2.0)
    parser.add_argument("--reference-seed", type=int, default=20260818)
    return parser.parse_args()


def _run(args: argparse.Namespace, label: str, *, reuse: bool) -> tuple[float, Path]:
    output = args.output_root / label
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
        "--multistart-backend", "serial",
    ]
    if reuse:
        command.append("--reuse-exact-evaluations")
    print(f"benchmark evaluation_reuse={reuse}", flush=True)
    started = time.perf_counter()
    subprocess.run(command, check=True)
    wall_seconds = time.perf_counter() - started
    tag = f"risk_{f'{args.percent:g}'.replace('.', 'p')}pct"
    return wall_seconds, output / tag / "result.json"


def _scientific_payload(result: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    payload.pop("selection_execution", None)
    payload.pop("validation_execution", None)
    return payload


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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
    baseline_scientific = _scientific_payload(baseline)
    reuse_scientific = _scientific_payload(reuse)
    report = {
        "schema_version": 1,
        "scope": "end-to-end smoke Pareto workflow",
        "percent": float(args.percent),
        "reference_seed": int(args.reference_seed),
        "baseline_wall_seconds": baseline_seconds,
        "reuse_wall_seconds": reuse_seconds,
        "speedup_baseline_over_reuse": baseline_seconds / reuse_seconds,
        "wall_seconds_saved": baseline_seconds - reuse_seconds,
        "scientific_payload_exactly_equal": baseline_scientific == reuse_scientific,
        "baseline_scientific_sha256": _digest(baseline_scientific),
        "reuse_scientific_sha256": _digest(reuse_scientific),
        "reuse_selection_execution": reuse.get("selection_execution"),
        "reuse_validation_execution": reuse.get("validation_execution"),
        "baseline_result": str(baseline_path.resolve()),
        "reuse_result": str(reuse_path.resolve()),
        "order_note": "baseline ran first, then reuse; both used fresh processes/output trees",
    }
    report_path = args.output_root / "benchmark.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"benchmark complete: {report_path}", flush=True)


if __name__ == "__main__":
    main()
