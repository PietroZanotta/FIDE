"""Audit frozen empirical I-projection support before design optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

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
from experiment import (  # noqa: E402
    ActiveNematicExperiment,
    _empirical_coordinate_support_gap,
    make_observation_bank,
)
from measurements import random_periodic_sensor_starts  # noqa: E402
from run import _split, _state_suffix  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.json")
    parser.add_argument("--starts", type=int)
    parser.add_argument("--reference-seed", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config, smoke=args.smoke)
    output_dir = SCRIPT_DIR / "outputs" / ("smoke" if args.smoke else "run")
    state_suffix = _state_suffix(cfg)
    population = DefectPopulationBank.load(
        output_dir / f"positive_defect_bank_{state_suffix}.npz"
    )
    split = make_run_split(_split(cfg))
    truth_count = int(cfg["randomness"].get("truth_particles", 2048))
    truth = population.resample_trajectory(
        run_indices=split.design,
        n=truth_count,
        seed=int(cfg["seed"]) + 4001,
    )
    reference_seed = int(
        args.reference_seed
        if args.reference_seed is not None
        else cfg["reference_training"].get("seeds", [cfg["seed"]])[0]
    )
    reference_path = (
        output_dir
        / f"{state_suffix}_reference_seed_{reference_seed}"
        / "reference_bank.npz"
    )
    with np.load(reference_path, allow_pickle=False) as data:
        times = jnp.asarray(data["times"])
        nodes = jnp.asarray(data["nodes"])
        velocity = jnp.asarray(data["velocity"])
        weights = jnp.asarray(data["weights"])

    experiment = ActiveNematicExperiment(
        cfg,
        times=times,
        truth_particles=jnp.asarray(truth),
        reference_nodes=nodes,
        reference_velocity=velocity,
        reference_weights=weights,
    )
    random_cfg = cfg["randomness"]
    bank = make_observation_bank(
        seed=int(cfg["seed"]),
        namespace=int(random_cfg.get("selection_namespace", 9890)),
        trials=int(random_cfg.get("selection_trials", 8)),
        acquisition_count=int(cfg["measurement"]["acquisition_k"]),
        finite_n=int(cfg["measurement"]["finite_n"]),
        truth_particle_count=truth_count,
        n_observables=experiment.family.n_observables,
    )
    optimization = cfg.get("optimization", {})
    start_count = int(
        args.starts
        if args.starts is not None
        else optimization.get("law_start_count", optimization.get("start_count", 16))
    )
    starts = random_periodic_sensor_starts(
        jax.random.PRNGKey(int(cfg["seed"]) + 17),
        start_count,
        n_sensors=experiment.family.n_sensors,
        box_size=experiment.family.box_size,
        min_separation=float(cfg["measurement"].get("min_sep", 0.0)),
        oversample=int(optimization.get("start_oversample", 64)),
    )

    rows = []
    base_weights = np.asarray(weights, dtype=np.float64)
    for index, eta in enumerate(starts):
        phi_truth, phi_reference, _ = experiment._geometry(eta)
        coordinate_gaps = []
        for trial in range(int(bank.sample_indices.shape[0])):
            reconstruction = experiment._reconstruct(phi_truth, bank, trial)
            coordinate_gaps.append(
                _empirical_coordinate_support_gap(
                    np.asarray(phi_reference),
                    base_weights,
                    np.asarray(reconstruction.c),
                )
            )
        started = time.perf_counter()
        audit = experiment.audit_metric(eta, bank, "law_risk")
        rows.append(
            {
                "start": index,
                "eta": np.asarray(eta).tolist(),
                "minimum_coordinate_support_gap": float(min(coordinate_gaps)),
                "exact_audit_seconds": time.perf_counter() - started,
                **audit,
            }
        )
        print(
            f"start={index} valid={audit['valid']} "
            f"residual={audit['max_calibration_residual']:.3e}",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "config": str(args.config),
        "smoke": args.smoke,
        "measurement": cfg["measurement"],
        "reference_bank": str(reference_path),
        "trial_count": int(bank.sample_indices.shape[0]),
        "start_count": start_count,
        "valid_start_count": sum(bool(row["valid"]) for row in rows),
        "rows": rows,
    }
    target = args.output or (
        output_dir / "audits" / f"projection_feasibility_seed_{reference_seed}.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(target)


if __name__ == "__main__":
    main()
