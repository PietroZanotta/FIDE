from __future__ import annotations

"""Orchestrate the preregistered v4 pipeline with an explicit freeze boundary."""

import argparse
from pathlib import Path
import time

from common import SCRIPT_DIR, load_config, write_json_atomic
from v4_protocol import prepare_v4_inputs
from v4_select import select_and_freeze_v4
from v4_validate import validate_v4


def run(config_path: str | Path, output_dir: str | Path, stage: str = "all"):
    cfg = load_config(config_path)
    output_dir = Path(output_dir)
    timings = {}
    started = time.perf_counter()
    if stage in {"all", "prepare", "select"}:
        t0 = time.perf_counter()
        print("[v4-pipeline] prepare immutable prospective inputs", flush=True)
        prepare_v4_inputs(cfg, output_dir)
        timings["prepare_inputs"] = time.perf_counter() - t0
    if stage in {"all", "select"}:
        t0 = time.perf_counter()
        print("[v4-pipeline] gradient selection and freeze", flush=True)
        select_and_freeze_v4(cfg, output_dir)
        timings["gradient_selection_and_freeze"] = time.perf_counter() - t0
    if stage in {"all", "validate"}:
        t0 = time.perf_counter()
        print("[v4-pipeline] fresh post-freeze hidden validation", flush=True)
        validate_v4(cfg, output_dir)
        timings["fresh_hidden_validation"] = time.perf_counter() - t0
    timings["invocation_total"] = time.perf_counter() - started
    results = output_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    write_json_atomic(results / f"v4_last_{stage}_runtime.json", timings)
    print(f"[v4-pipeline] {stage} complete", flush=True)
    return timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=SCRIPT_DIR / "configs" / "production_v4.json",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--stage", choices=("all", "prepare", "select", "validate"), default="all"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = args.output_dir or SCRIPT_DIR / "outputs" / str(cfg["name"])
    run(args.config, output, args.stage)


if __name__ == "__main__":
    main()
