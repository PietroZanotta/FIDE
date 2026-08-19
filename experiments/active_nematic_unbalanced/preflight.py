"""Production readiness and cost preflight for the active-nematic experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
for path in (REPO_ROOT / "src", REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mfsi.config import load_config  # noqa: E402

from domain import (  # noqa: E402
    DefectPopulationBank,
    PopulationStateConfig,
    generate_physical_bank,
    make_run_split,
)
from poisson3d_tesseract import (  # noqa: E402
    NATIVE_SOLVER_REVISION,
    is_active_nematic_poisson3d_available,
)
from run import _physics, _split, _state_suffix  # noqa: E402


def _population_summary(bank: DefectPopulationBank, split) -> dict:
    output = {
        "runs": int(bank.counts.shape[0]),
        "times": int(bank.counts.shape[1]),
        "total_states": int(len(bank.states)),
        "minimum_count": int(np.min(bank.counts)),
        "maximum_count": int(np.max(bank.counts)),
        "zero_count_cells": int(np.sum(bank.counts == 0)),
    }
    for name in ("train", "design", "validation"):
        indices = np.asarray(getattr(split, name))
        counts = bank.counts[indices]
        pooled = np.sum(counts, axis=0)
        output[name] = {
            "runs": int(len(indices)),
            "mean_count_by_time": np.mean(counts, axis=0).tolist(),
            "minimum_pooled_count_by_time": int(np.min(pooled)),
            "all_times_nonempty": bool(np.all(pooled > 0)),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.json")
    parser.add_argument("--benchmark-steps", type=int, default=20)
    parser.add_argument("--benchmark-workers", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "audits" / "production_preflight.json",
    )
    args = parser.parse_args()
    if args.benchmark_steps < 1:
        raise ValueError("benchmark-steps must be positive")
    cfg = load_config(args.config, smoke=False)
    params = _physics(cfg)
    split_cfg = _split(cfg)
    split = make_run_split(split_cfg)
    output_dir = SCRIPT_DIR / "outputs" / "run"
    physical_path = output_dir / "physical_bank.npz"
    defect_path = output_dir / f"positive_defect_bank_{_state_suffix(cfg)}.npz"

    save_times = np.asarray(cfg["physical_bank"]["save_times"], dtype=np.float64)
    population_times = np.asarray(
        cfg["physical_bank"].get("population_times", save_times), dtype=np.float64
    )
    steps = np.rint(save_times / params.dt).astype(np.int64)
    aligned = bool(np.allclose(steps * params.dt, save_times, atol=1.0e-12, rtol=0.0))
    run_count = split_cfg.total_runs
    raw_bank_bytes = run_count * len(save_times) * params.n**2 * 2 * 8
    disk = shutil.disk_usage(output_dir.parent)
    configured_workers = int(cfg["physical_bank"].get("workers", 1))
    benchmark_workers = min(
        int(args.benchmark_workers or configured_workers), run_count
    )

    benchmark_seeds = (
        int(cfg["seed"])
        + int(cfg["physical_bank"].get("seed_offset", 1001))
        + np.arange(benchmark_workers, dtype=np.int64)
    )
    benchmark_times = np.asarray([0.0, args.benchmark_steps * params.dt])
    start = time.perf_counter()
    benchmark = generate_physical_bank(
        params,
        seeds=benchmark_seeds,
        times=benchmark_times,
        workers=benchmark_workers,
    )
    elapsed = time.perf_counter() - start
    if not (np.isfinite(benchmark.q1).all() and np.isfinite(benchmark.q2).all()):
        raise FloatingPointError("production-resolution physical benchmark is non-finite")
    batches = math.ceil(run_count / benchmark_workers)
    estimated_seconds = (
        elapsed * batches * int(steps[-1]) / int(args.benchmark_steps)
    )

    population = None
    if defect_path.is_file():
        bank = DefectPopulationBank.load(defect_path)
        population = _population_summary(bank, split)
    readiness = {
        "physical_bank": physical_path.is_file(),
        "defect_bank": defect_path.is_file(),
        "tesseract_poisson3d": is_active_nematic_poisson3d_available(),
        "save_times_aligned_to_dt": aligned,
        "disk_has_three_times_raw_bank": disk.free >= 3 * raw_bank_bytes,
        "population_all_splits_nonempty": (
            population is not None
            and all(population[name]["all_times_nonempty"] for name in ("train", "design", "validation"))
        ),
        "defect_bank_matches_population_times": (
            population is not None
            and defect_path.is_file()
            and np.array_equal(
                DefectPopulationBank.load(defect_path).times, population_times
            )
        ),
    }
    payload = {
        "schema_version": 1,
        "config": str(args.config),
        "physics": asdict(params),
        "state": asdict(PopulationStateConfig(**cfg["state"])),
        "run_count": run_count,
        "save_times": save_times.tolist(),
        "population_times": population_times.tolist(),
        "maximum_steps_per_run": int(steps[-1]),
        "total_physical_steps": int(run_count * steps[-1]),
        "configured_workers": configured_workers,
        "estimated_raw_bank_bytes": int(raw_bank_bytes),
        "free_disk_bytes": int(disk.free),
        "benchmark": {
            "workers": benchmark_workers,
            "runs": benchmark_workers,
            "steps_per_run": args.benchmark_steps,
            "elapsed_seconds": elapsed,
            "estimated_full_bank_seconds_at_same_throughput": estimated_seconds,
            "final_order_parameter_min": float(
                2.0 * np.min(np.hypot(benchmark.q1[:, -1], benchmark.q2[:, -1]))
            ),
            "final_order_parameter_max": float(
                2.0 * np.max(np.hypot(benchmark.q1[:, -1], benchmark.q2[:, -1]))
            ),
        },
        "paths": {
            "physical_bank": str(physical_path),
            "defect_bank": str(defect_path),
        },
        "population": population,
        "poisson3d_revision": NATIVE_SOLVER_REVISION,
        "readiness": readiness,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(readiness, sort_keys=True))
    print(
        f"estimated physical-bank wall time: {estimated_seconds / 60.0:.1f} minutes",
        flush=True,
    )
    print(args.output)


if __name__ == "__main__":
    main()
