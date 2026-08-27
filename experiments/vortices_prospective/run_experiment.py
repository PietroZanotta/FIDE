from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import time

from build_prospective_data import build
from common import SCRIPT_DIR, load_config, write_json_atomic
from train_reference import train_and_rollout
from validate import validate


def _load_selection_stage():
    # `select` is also a Python stdlib extension module and may already be in
    # sys.modules through subprocess.  Load the experiment stage by file identity
    # so orchestration cannot accidentally call stdlib `select.select`.
    path = SCRIPT_DIR / "select.py"
    spec = importlib.util.spec_from_file_location("vortices_prospective_selection_stage", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load selection stage from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.select


def run(config_path: str | Path, output_dir: str | Path):
    cfg = load_config(config_path)
    output_dir = Path(output_dir)
    started = time.perf_counter()
    timings = {}
    stage_results = {}
    selection_stage = _load_selection_stage()
    for name, function in (
        ("build_prospective_data", build),
        ("train_reference", train_and_rollout),
        ("select_and_freeze", selection_stage),
        ("hidden_validation", validate),
    ):
        stage_started = time.perf_counter()
        print(f"[pipeline] {name}", flush=True)
        stage_results[name] = function(cfg, output_dir)
        timings[name] = time.perf_counter() - stage_started
    timings["total"] = time.perf_counter() - started
    authoritative_timings = {
        "build_prospective_data": float(
            stage_results["build_prospective_data"]["elapsed_seconds"]
        ),
        "train_reference": float(
            stage_results["train_reference"]["elapsed_seconds"]
        ),
        "select_and_freeze": float(
            stage_results["select_and_freeze"]["selection_elapsed_seconds"]
        ),
        "hidden_validation": float(
            stage_results["hidden_validation"]["runtime_seconds"]
        ),
    }
    authoritative_timings["total"] = sum(authoritative_timings.values())
    write_json_atomic(
        output_dir / "results" / "runtime_breakdown.json", authoritative_timings
    )
    write_json_atomic(
        output_dir / "results" / "last_invocation_runtime.json", timings
    )
    print(f"[pipeline] complete: {output_dir / 'results' / 'report.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=SCRIPT_DIR / "configs" / "production.json",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = args.output_dir or SCRIPT_DIR / "outputs" / str(cfg["mode"])
    run(args.config, output)


if __name__ == "__main__":
    main()
