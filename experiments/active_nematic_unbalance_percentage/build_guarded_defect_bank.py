"""Build a boundary-guarded defect bank from the saved production physics.

The production bank already contains integer times through ``t=30``.  This
utility reuses its saved ``t=21..30`` fields and advances each ``t=30`` state
to ``t=31``.  Only the compact extracted-defect bank is written.  The original
physical and defect banks are never modified.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

try:
    from .active_nematic_solver import ActiveNematic2D, ActiveNematicParams
    from .domain import PhysicalBank, PopulationStateConfig, extract_population_bank
except ImportError:  # pragma: no cover - direct script convention.
    from active_nematic_solver import ActiveNematic2D, ActiveNematicParams
    from domain import PhysicalBank, PopulationStateConfig, extract_population_bank


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PHYSICAL = SCRIPT_DIR / "outputs" / "run" / "physical_bank.npz"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "run_guarded"
DEFAULT_CONFIG = SCRIPT_DIR / "config.json"


def _advance(task):
    params, seed, start_time, final_time, q1, q2 = task
    simulation = ActiveNematic2D(params, seed=int(seed))
    simulation.load_state_dict({"t": start_time, "q1": q1, "q2": q2})
    steps = int(round((final_time - start_time) / params.dt))
    if not np.isclose(start_time + steps * params.dt, final_time, atol=1.0e-12):
        raise ValueError("guard time must lie on the physical time-step grid")
    simulation.step(steps)
    return simulation.q1, simulation.q2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a t=21..31 defect bank for boundary-stable evaluation."
    )
    parser.add_argument("--physical", type=Path, default=DEFAULT_PHYSICAL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--support-start", type=float, default=21.0)
    parser.add_argument("--support-end", type=float, default=31.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    config = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    state_config = PopulationStateConfig(**config["state"])
    suffix = "position" if state_config.mode == "position" else "position_polarity"
    output_dir = args.output_dir.expanduser().resolve()
    output = output_dir / f"positive_defect_bank_{suffix}.npz"
    manifest_path = output_dir / "guarded_defect_bank_manifest.json"
    if (output.exists() or manifest_path.exists()) and not args.force:
        raise SystemExit(
            f"Guarded output already exists under {output_dir}; pass --force to replace it."
        )

    physical = PhysicalBank.load(args.physical.expanduser().resolve())
    existing = np.flatnonzero(
        (physical.times >= args.support_start - 1.0e-12)
        & (physical.times < args.support_end - 1.0e-12)
    )
    if len(existing) < 2 or not np.isclose(
        physical.times[existing[0]], args.support_start, atol=1.0e-12
    ):
        raise SystemExit("the saved physical bank does not contain the requested support start")
    base_index = int(existing[-1])
    base_time = float(physical.times[base_index])
    if args.support_end <= base_time:
        raise SystemExit("--support-end must be later than the last selected saved time")

    params = physical.params
    seeds = np.asarray(physical.seeds, dtype=np.int64).copy()
    times = np.concatenate(
        [np.asarray(physical.times[existing], dtype=np.float64), [float(args.support_end)]]
    )
    q1 = np.empty((len(seeds), len(times), params.n, params.n), dtype=np.float64)
    q2 = np.empty_like(q1)
    q1[:, :-1] = physical.q1[:, existing]
    q2[:, :-1] = physical.q2[:, existing]
    tasks = [
        (
            params,
            int(seeds[run]),
            base_time,
            float(args.support_end),
            physical.q1[run, base_index].copy(),
            physical.q2[run, base_index].copy(),
        )
        for run in range(len(seeds))
    ]
    del physical

    print(
        f"guarded_bank phase=advance runs={len(tasks)} "
        f"from_t={base_time:g} to_t={args.support_end:g}",
        flush=True,
    )
    if args.workers == 1:
        rows = map(_advance, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=min(args.workers, len(tasks)))
        rows = executor.map(_advance, tasks)
    try:
        for run, (run_q1, run_q2) in enumerate(rows):
            q1[run, -1] = run_q1
            q2[run, -1] = run_q2
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    print(f"guarded_bank phase=extract frames={q1.shape[0] * q1.shape[1]}", flush=True)
    population = extract_population_bank(
        PhysicalBank(times, q1, q2, seeds, params), state_config
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    population.save(output)
    manifest = {
        "schema_version": 1,
        "source_physical_bank": str(args.physical.expanduser().resolve()),
        "support_times": times.tolist(),
        "core_action_times": times[1:-1].tolist(),
        "time_guard_points": 1,
        "state": config["state"],
        "defect_bank": str(output),
        "physical_extension_saved": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    print(manifest_path)


if __name__ == "__main__":
    main()
