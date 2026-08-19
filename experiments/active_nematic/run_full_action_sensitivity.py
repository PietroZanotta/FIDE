"""Re-audit frozen selected designs under 3-D full-action fidelity variants."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
for path in (REPO_ROOT / "src", REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

jax.config.update("jax_enable_x64", True)

from mfsi.config import load_config  # noqa: E402

from domain import DefectPopulationBank, make_run_split  # noqa: E402
from experiment import ActiveNematicExperiment, make_observation_bank  # noqa: E402
from run import _split, _state_suffix  # noqa: E402


def _parse_shape(value: str) -> tuple[int, int, int]:
    shape = tuple(int(item) for item in value.lower().split("x"))
    if len(shape) != 3 or any(item < 3 for item in shape):
        raise argparse.ArgumentTypeError("shape must be Nx x Ny x Ntheta with axes >= 3")
    return shape


def _variants(
    baseline: dict[str, Any],
    shapes: list[tuple[int, int, int]],
    radii: list[float],
    bandwidths: list[float],
) -> list[tuple[str, dict[str, Any]]]:
    action = baseline["full_action"]
    baseline_shape = tuple(action["grid_shape_polarity"])
    baseline_radius = float(action["polarity_metric_radius"])
    baseline_bandwidth = float(action["raster_bandwidth"])
    rows = [("baseline", copy.deepcopy(baseline))]
    for shape in shapes:
        if shape != baseline_shape:
            cfg = copy.deepcopy(baseline)
            cfg["full_action"]["grid_shape_polarity"] = list(shape)
            rows.append((f"grid_{shape[0]}x{shape[1]}x{shape[2]}", cfg))
    for radius in radii:
        if float(radius) != baseline_radius:
            cfg = copy.deepcopy(baseline)
            cfg["full_action"]["polarity_metric_radius"] = float(radius)
            rows.append((f"radius_{float(radius):g}", cfg))
    for bandwidth in bandwidths:
        if float(bandwidth) != baseline_bandwidth:
            cfg = copy.deepcopy(baseline)
            cfg["full_action"]["raster_bandwidth"] = float(bandwidth)
            rows.append((f"bandwidth_{float(bandwidth):g}", cfg))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.json")
    parser.add_argument("--reference-seed", type=int)
    parser.add_argument("--bank", choices=("selection", "validation"), default="selection")
    parser.add_argument(
        "--designs",
        nargs="+",
        choices=("law", "tangent", "full", "reference_easy"),
        default=None,
    )
    parser.add_argument(
        "--grid-shapes",
        nargs="+",
        type=_parse_shape,
        default=[(32, 32, 16), (48, 48, 24)],
    )
    parser.add_argument("--radii", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--bandwidths", nargs="+", type=float, default=[0.6, 0.8, 1.0])
    parser.add_argument("--base-bandwidth", type=float)
    parser.add_argument("--cg-maxiter", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config, smoke=args.smoke)
    if args.base_bandwidth is not None:
        if args.base_bandwidth < 0.0:
            raise ValueError("--base-bandwidth must be nonnegative")
        cfg["full_action"]["raster_bandwidth"] = float(args.base_bandwidth)
    if args.cg_maxiter is not None:
        if args.cg_maxiter < 1:
            raise ValueError("--cg-maxiter must be positive")
        cfg["full_action"]["cg_maxiter"] = int(args.cg_maxiter)
    if cfg["state"]["mode"] != "position_polarity":
        raise ValueError("full-action sensitivity requires state.mode='position_polarity'")
    output_dir = SCRIPT_DIR / "outputs" / ("smoke" if args.smoke else "run")
    state_suffix = _state_suffix(cfg)
    population_path = output_dir / f"positive_defect_bank_{state_suffix}.npz"
    if not population_path.is_file():
        raise FileNotFoundError(f"missing frozen defect bank: {population_path}")
    population = DefectPopulationBank.load(population_path)
    split = make_run_split(_split(cfg))
    random_cfg = cfg["randomness"]
    truth_n = int(random_cfg.get("truth_particles", 2048))
    if args.bank == "selection":
        run_indices = split.design
        truth_seed = int(cfg["seed"]) + 4001
        namespace = int(random_cfg.get("selection_namespace", 9890))
        trial_count = int(random_cfg.get("selection_trials", 8))
    else:
        run_indices = split.validation
        truth_seed = int(cfg["seed"]) + 4002
        namespace = int(random_cfg.get("validation_namespace", 9891))
        trial_count = int(random_cfg.get("validation_trials", 32))
    truth = population.resample_trajectory(
        run_indices=run_indices, n=truth_n, seed=truth_seed
    )
    bank = make_observation_bank(
        seed=int(cfg["seed"]),
        namespace=namespace,
        trials=trial_count,
        acquisition_count=int(cfg["measurement"]["acquisition_k"]),
        finite_n=int(cfg["measurement"]["finite_n"]),
        truth_particle_count=truth_n,
        n_observables=int(cfg["measurement"]["n_sensors"])
        * len(cfg["measurement"]["channels"]),
    )

    reference_seed = int(
        args.reference_seed
        if args.reference_seed is not None
        else cfg["reference_training"].get("seeds", [cfg["seed"]])[0]
    )
    seed_dir = output_dir / f"{state_suffix}_reference_seed_{reference_seed}"
    result_path = seed_dir / "result.json"
    reference_path = seed_dir / "reference_bank.npz"
    if not result_path.is_file() or not reference_path.is_file():
        raise FileNotFoundError(
            f"missing frozen result/reference bank under {seed_dir}; run the MFSI stage first"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    with np.load(reference_path, allow_pickle=False) as data:
        times = jnp.asarray(data["times"])
        nodes = jnp.asarray(data["nodes"])
        velocity = jnp.asarray(data["velocity"])
        weights = jnp.asarray(data["weights"])
    designs = {
        name: jnp.asarray(value, dtype=jnp.float64)
        for name, value in result["designs"].items()
        if value is not None and (args.designs is None or name in args.designs)
    }
    if not designs:
        raise ValueError("none of the requested designs is available in the result")
    rows = []
    for label, variant_cfg in _variants(
        cfg, args.grid_shapes, args.radii, args.bandwidths
    ):
        experiment = ActiveNematicExperiment(
            variant_cfg,
            times=times,
            truth_particles=jnp.asarray(truth),
            reference_nodes=nodes,
            reference_velocity=velocity,
            reference_weights=weights,
        )
        for design_name, eta in designs.items():
            audit = experiment.audit_metric(eta, bank, "full_action")
            rows.append(
                {
                    "variant": label,
                    "design": design_name,
                    "eta": np.asarray(eta, dtype=np.float64).tolist(),
                    "full_action": audit["value"],
                    "valid": audit["valid"],
                    "max_calibration_residual": audit["max_calibration_residual"],
                    "min_ess_fraction": audit["min_ess_fraction"],
                    "max_poisson_relative_residual": audit[
                        "max_poisson_relative_residual"
                    ],
                    "poisson_errors": audit.get("poisson_errors", []),
                    "solver": experiment.full_action_provenance(),
                }
            )
        print(f"completed {label}", flush=True)

    baseline = {
        row["design"]: row["full_action"]
        for row in rows
        if row["variant"] == "baseline"
    }
    for row in rows:
        base = baseline[row["design"]]
        row["relative_action_change_from_baseline"] = (
            row["full_action"] / base - 1.0 if base != 0.0 else None
        )
    payload = {
        "schema_version": 1,
        "source_result": str(result_path),
        "source_result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "bank": args.bank,
        "reference_seed": reference_seed,
        "rows": rows,
    }
    target = args.output or (
        output_dir / "audits" / f"full_action_sensitivity_{args.bank}.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(target)


if __name__ == "__main__":
    main()
