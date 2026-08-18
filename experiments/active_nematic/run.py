"""CLI entry points following the staged vortices experiment convention."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
for path in (SRC_DIR, REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

jax.config.update("jax_enable_x64", True)

from mfsi.config import load_config
from mfsi.flow_matching import FlowMatchingConfig

from active_nematic_solver import ActiveNematicParams
from domain import (
    DefectPopulationBank,
    EmpiricalEndpointSource,
    PhysicalBank,
    PopulationStateConfig,
    SplitConfig,
    extract_population_bank,
    generate_physical_bank,
    make_run_split,
)
from experiment import ActiveNematicExperiment, make_observation_bank
from periodic_reference import PeriodicReferenceFlow, train_periodic_reference_flow

CONFIG_PATH = SCRIPT_DIR / "config.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Active-nematic MFSI experiment")
    parser.add_argument("stage", choices=("physical-bank", "defects", "mfsi"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--position-only",
        action="store_true",
        help="explicit full-action ablation: use state (x,y) and occupancy channels",
    )
    return parser.parse_args()


def _physics(cfg: dict[str, Any]) -> ActiveNematicParams:
    return ActiveNematicParams(**cfg["physics"])


def _state(cfg: dict[str, Any]) -> PopulationStateConfig:
    return PopulationStateConfig(**cfg["state"])


def _split(cfg: dict[str, Any]) -> SplitConfig:
    block = cfg["splits"]
    return SplitConfig(
        train_runs=int(block["train_runs"]),
        design_runs=int(block["design_runs"]),
        validation_runs=int(block["validation_runs"]),
        seed=int(cfg["seed"]) + int(block.get("seed_offset", 1101)),
    )


def _flow_config(cfg: dict[str, Any], seed: int) -> FlowMatchingConfig:
    block = cfg["reference_training"]
    return FlowMatchingConfig(
        seed=int(seed),
        hidden_width=int(block.get("hidden_width", 128)),
        hidden_layers=int(block.get("hidden_layers", 4)),
        train_steps=int(block.get("train_steps", 12000)),
        batch_size=int(block.get("batch_size", 2048)),
        learning_rate=float(block.get("learning_rate", 1.0e-3)),
        min_learning_rate_ratio=float(block.get("min_learning_rate_ratio", 0.05)),
        grad_clip_norm=float(block.get("grad_clip_norm", 10.0)),
        bridge_schedule=str(block.get("bridge_schedule", "linear")),
        bridge_noise_std=float(block.get("bridge_noise_std", 0.15)),
        log_every=int(block.get("log_every", 500)),
    )


def _output_dir(cfg: dict[str, Any], smoke: bool) -> Path:
    return SCRIPT_DIR / "outputs" / ("smoke" if smoke else "run")


def _state_suffix(cfg: dict[str, Any]) -> str:
    return "position" if cfg["state"]["mode"] == "position" else "position_polarity"


def build_physical_bank(cfg: dict[str, Any], output_dir: Path) -> Path:
    split = _split(cfg)
    offset = int(cfg["physical_bank"].get("seed_offset", 1001))
    seeds = int(cfg["seed"]) + offset + np.arange(split.total_runs, dtype=np.int64)
    bank = generate_physical_bank(
        _physics(cfg), seeds=seeds, times=np.asarray(cfg["physical_bank"]["save_times"])
    )
    path = output_dir / "physical_bank.npz"
    bank.save(path)
    return path


def build_defect_bank(cfg: dict[str, Any], output_dir: Path) -> Path:
    physical = PhysicalBank.load(output_dir / "physical_bank.npz")
    population = extract_population_bank(physical, _state(cfg))
    path = output_dir / f"positive_defect_bank_{_state_suffix(cfg)}.npz"
    population.save(path)
    return path


def _endpoint_source(
    population: DefectPopulationBank,
    train_runs: np.ndarray,
    *,
    n: int,
    seed: int,
) -> EmpiricalEndpointSource:
    rng = np.random.default_rng(seed)
    first = population.samples(0, train_runs)
    last = population.samples(len(population.times) - 1, train_runs)
    x0 = first[rng.integers(0, len(first), size=n)]
    x1 = last[rng.integers(0, len(last), size=n)]
    return EmpiricalEndpointSource(jnp.asarray(x0), jnp.asarray(x1))


def _result_metrics(exp, bank, designs):
    output = {}
    for name, eta in designs.items():
        if eta is None:
            output[name] = {"available": False}
            continue
        rows = [
            asdict(exp.trial_metrics(eta, bank, trial, full=exp.full_action_supported))
            for trial in range(bank.sample_indices.shape[0])
        ]
        output[name] = {"available": True, "trials": rows}
    return output


def run_mfsi(cfg: dict[str, Any], output_dir: Path) -> Path:
    state_suffix = _state_suffix(cfg)
    population = DefectPopulationBank.load(output_dir / f"positive_defect_bank_{state_suffix}.npz")
    if population.state_config != _state(cfg):
        raise ValueError("saved defect bank state policy does not match the current config")
    split = make_run_split(_split(cfg))
    normalized_times = (population.times - population.times[0]) / (population.times[-1] - population.times[0])
    random_cfg = cfg["randomness"]
    truth_n = int(random_cfg.get("truth_particles", 2048))
    design_truth = population.resample_trajectory(
        run_indices=split.design, n=truth_n, seed=int(cfg["seed"]) + 4001
    )
    validation_truth = population.resample_trajectory(
        run_indices=split.validation, n=truth_n, seed=int(cfg["seed"]) + 4002
    )
    periods = np.asarray(
        [population.box_size, population.box_size]
        + ([2.0 * np.pi] if population.state_config.state_dim == 3 else []),
        dtype=np.float64,
    )

    summaries = []
    for reference_seed in cfg["reference_training"].get("seeds", [cfg["seed"]]):
        seed_dir = output_dir / f"{state_suffix}_reference_seed_{int(reference_seed)}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = seed_dir / "reference.npz"
        if checkpoint.is_file():
            reference = PeriodicReferenceFlow.from_npz(
                checkpoint,
                substeps_per_interval=int(cfg["reference"].get("rk4_substeps_per_time_interval", 16)),
            )
            history = []
        else:
            source = _endpoint_source(
                population,
                split.train,
                n=int(cfg["reference"].get("endpoint_particles", 50000)),
                seed=int(cfg["seed"]) + int(cfg["reference"].get("endpoint_seed_offset", 2001)),
            )
            reference, history = train_periodic_reference_flow(
                source,
                _flow_config(cfg, int(reference_seed)),
                periods=jnp.asarray(periods),
                substeps_per_interval=int(cfg["reference"].get("rk4_substeps_per_time_interval", 16)),
            )
            reference.save(checkpoint)

        rng = np.random.default_rng(int(reference_seed) + int(cfg["reference"].get("bank_seed_offset", 3001)))
        initial_pool = population.samples(0, split.train)
        particle_n = int(cfg["reference"].get("particles", 8192))
        x0 = jnp.asarray(initial_pool[rng.integers(0, len(initial_pool), size=particle_n)])
        nodes = reference.rollout(x0, jnp.asarray(normalized_times))
        velocity = jax.vmap(lambda time, x: reference.velocity(x, time))(jnp.asarray(normalized_times), nodes)
        weights = jnp.full(nodes.shape[:2], 1.0 / particle_n, dtype=jnp.float64)
        np.savez_compressed(seed_dir / "reference_bank.npz", times=normalized_times, nodes=nodes, velocity=velocity, weights=weights)

        selection_bank = make_observation_bank(
            seed=int(cfg["seed"]), namespace=int(random_cfg.get("selection_namespace", 9890)),
            trials=int(random_cfg.get("selection_trials", 8)), acquisition_count=int(cfg["measurement"]["acquisition_k"]),
            finite_n=int(cfg["measurement"]["finite_n"]), truth_particle_count=truth_n,
            n_observables=int(cfg["measurement"]["n_sensors"]) * len(cfg["measurement"]["channels"]),
        )
        experiment = ActiveNematicExperiment(
            cfg, times=jnp.asarray(normalized_times), truth_particles=jnp.asarray(design_truth),
            reference_nodes=nodes, reference_velocity=velocity, reference_weights=weights,
        )
        comparison = experiment.optimize_designs(selection_bank)
        validation_bank = make_observation_bank(
            seed=int(cfg["seed"]), namespace=int(random_cfg.get("validation_namespace", 9891)),
            trials=int(random_cfg.get("validation_trials", 32)), acquisition_count=int(cfg["measurement"]["acquisition_k"]),
            finite_n=int(cfg["measurement"]["finite_n"]), truth_particle_count=truth_n,
            n_observables=experiment.family.n_observables,
        )
        validation_experiment = ActiveNematicExperiment(
            cfg, times=jnp.asarray(normalized_times), truth_particles=jnp.asarray(validation_truth),
            reference_nodes=nodes, reference_velocity=velocity, reference_weights=weights,
        )
        designs = {"law": comparison.law_eta, "tangent": comparison.tangent_eta, "full": comparison.full_eta}
        payload = {
            "schema_version": 1,
            "experiment": cfg["name"],
            "reference_seed": int(reference_seed),
            "state": cfg["state"],
            "normalized_extant_defect_law": True,
            "count_is_auxiliary": True,
            "full_action_supported": experiment.full_action_supported,
            "full_action_blocker": None if experiment.full_action_supported else "3-D periodic raster and weighted Poisson solver not implemented",
            "risk_star": comparison.risk_star,
            "risk_max": comparison.risk_max,
            "designs": {key: None if value is None else np.asarray(value).tolist() for key, value in designs.items()},
            "selection_candidates": comparison.candidates,
            "validation": _result_metrics(validation_experiment, validation_bank, designs),
            "positive_count": {
                "design_mean": population.mean_count(split.design).tolist(),
                "validation_mean": population.mean_count(split.validation).tolist(),
            },
            "reference_training_history": history,
        }
        (seed_dir / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summaries.append({"reference_seed": int(reference_seed), "result": str(seed_dir / "result.json")})

    manifest = output_dir / f"manifest_{state_suffix}.json"
    manifest.write_text(json.dumps({"schema_version": 1, "runs": summaries}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    if args.position_only:
        cfg["state"]["mode"] = "position"
        cfg["measurement"]["channels"] = ["occupancy"]
    output_dir = _output_dir(cfg, args.smoke)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "physical-bank":
        result = build_physical_bank(cfg, output_dir)
    elif args.stage == "defects":
        result = build_defect_bank(cfg, output_dir)
    else:
        result = run_mfsi(cfg, output_dir)
    print(f"active_nematic stage={args.stage} output={result}", flush=True)


if __name__ == "__main__":
    main()
