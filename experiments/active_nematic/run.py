"""CLI entry points following the staged vortices experiment convention."""

from __future__ import annotations

import argparse
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
    parser.add_argument(
        "stage",
        choices=("physical-bank", "defects", "reference", "mfsi", "validation"),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--position-only",
        action="store_true",
        help="explicit full-action ablation: use state (x,y) and occupancy channels",
    )
    parser.add_argument(
        "--reference-seeds",
        nargs="+",
        type=int,
        help="run only these reference seeds and merge them into the manifest",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="explicit output directory for an isolated experiment variant",
    )
    parser.add_argument(
        "--time-guard-points",
        type=int,
        help="exclude this many saved support times at each boundary from action/risk quadrature",
    )
    parser.add_argument(
        "--reference-particles",
        type=int,
        help="override the particle count used to discretize each learned reference flow",
    )
    parser.add_argument(
        "--selection-trials",
        type=int,
        help="override the number of common-random-number selection trials",
    )
    parser.add_argument(
        "--validation-trials",
        type=int,
        help="override the number of independent held-out validation trials",
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


def _write_merged_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    merged: dict[int, dict[str, Any]] = {}
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for row in existing.get("runs", []):
            merged[int(row["reference_seed"])] = row
    for row in rows:
        merged[int(row["reference_seed"])] = row
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runs": [merged[key] for key in sorted(merged)],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def build_physical_bank(cfg: dict[str, Any], output_dir: Path) -> Path:
    split = _split(cfg)
    offset = int(cfg["physical_bank"].get("seed_offset", 1001))
    seeds = int(cfg["seed"]) + offset + np.arange(split.total_runs, dtype=np.int64)
    bank = generate_physical_bank(
        _physics(cfg),
        seeds=seeds,
        times=np.asarray(cfg["physical_bank"]["save_times"]),
        workers=int(cfg["physical_bank"].get("workers", 1)),
    )
    path = output_dir / "physical_bank.npz"
    bank.save(path)
    return path


def build_defect_bank(cfg: dict[str, Any], output_dir: Path) -> Path:
    physical = PhysicalBank.load(output_dir / "physical_bank.npz")
    population_times = cfg["physical_bank"].get("population_times")
    if population_times is not None:
        physical = physical.select_times(np.asarray(population_times, dtype=np.float64))
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
        rows = exp.exact_trial_rows(
            eta, bank, full=exp.full_action_supported
        )
        valid_rows = [row for row in rows if row["valid"]]
        summary = {"trials": len(rows), "valid_trials": len(valid_rows)}
        for metric in ("law_risk", "tangent_action", "full_action"):
            values = np.asarray(
                [row[metric] for row in valid_rows if np.isfinite(row[metric])], dtype=np.float64
            )
            summary[metric] = {
                "mean": float(np.mean(values)) if len(values) else float("nan"),
                "se": (
                    float(np.std(values, ddof=1) / np.sqrt(len(values)))
                    if len(values) > 1 else (0.0 if len(values) == 1 else float("nan"))
                ),
                "n": int(len(values)),
            }
        output[name] = {"available": True, "summary": summary, "trials": rows}
    return output


def rerun_validation(cfg: dict[str, Any], output_dir: Path) -> Path:
    """Re-evaluate saved designs without repeating selection optimization."""
    state_suffix = _state_suffix(cfg)
    population = DefectPopulationBank.load(
        output_dir / f"positive_defect_bank_{state_suffix}.npz"
    )
    if population.state_config != _state(cfg):
        raise ValueError("saved defect bank state policy does not match the current config")
    split = make_run_split(_split(cfg))
    normalized_times = (
        population.times - population.times[0]
    ) / (population.times[-1] - population.times[0])
    random_cfg = cfg["randomness"]
    truth_n = int(random_cfg.get("truth_particles", 2048))
    validation_truth = population.resample_trajectory(
        run_indices=split.validation,
        n=truth_n,
        seed=int(cfg["seed"]) + 4002,
    )
    summaries = []
    for reference_seed in cfg["reference_training"].get("seeds", [cfg["seed"]]):
        seed_dir = output_dir / f"{state_suffix}_reference_seed_{int(reference_seed)}"
        result_path = seed_dir / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(
                f"missing selected-design result for seed {reference_seed}: {result_path}"
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        with np.load(seed_dir / "reference_bank.npz", allow_pickle=False) as data:
            nodes = jnp.asarray(data["nodes"])
            velocity = jnp.asarray(data["velocity"])
            weights = jnp.asarray(data["weights"])
        experiment = ActiveNematicExperiment(
            cfg,
            times=jnp.asarray(normalized_times),
            truth_particles=jnp.asarray(validation_truth),
            reference_nodes=nodes,
            reference_velocity=velocity,
            reference_weights=weights,
        )
        validation_bank = make_observation_bank(
            seed=int(cfg["seed"]),
            namespace=int(random_cfg.get("validation_namespace", 9891)),
            trials=int(random_cfg.get("validation_trials", 32)),
            acquisition_count=int(cfg["measurement"]["acquisition_k"]),
            finite_n=int(cfg["measurement"]["finite_n"]),
            truth_particle_count=truth_n,
            n_observables=experiment.family.n_observables,
        )
        designs = {
            name: None if eta is None else jnp.asarray(eta, dtype=jnp.float64)
            for name, eta in result["designs"].items()
        }
        print(
            f"active_nematic reference_seed={int(reference_seed)} "
            f"phase=validation_only trials={int(validation_bank.sample_indices.shape[0])}",
            flush=True,
        )
        result["validation"] = _result_metrics(
            experiment, validation_bank, designs
        )
        result["evaluation_times"] = {
            "physical": population.times.tolist(),
            "normalized": normalized_times.tolist(),
            "action_weights": np.asarray(experiment.time_weights).tolist(),
        }
        result["validation_full_action_solver"] = (
            experiment.full_action_provenance()
        )
        result.setdefault("observation_banks", {})["validation"] = {
            "trials": int(validation_bank.sample_indices.shape[0]),
            "namespace": int(random_cfg.get("validation_namespace", 9891)),
        }
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summaries.append(
            {"reference_seed": int(reference_seed), "result": str(result_path)}
        )
    manifest = output_dir / f"manifest_{state_suffix}.json"
    _write_merged_manifest(manifest, summaries)
    return manifest


def _build_or_load_reference_bank(
    cfg: dict[str, Any],
    population: DefectPopulationBank,
    split,
    normalized_times: np.ndarray,
    reference_seed: int,
    seed_dir: Path,
):
    seed_dir.mkdir(parents=True, exist_ok=True)
    periods = np.asarray(
        [population.box_size, population.box_size]
        + ([2.0 * np.pi] if population.state_config.state_dim == 3 else []),
        dtype=np.float64,
    )
    checkpoint = seed_dir / "reference.npz"
    history_path = seed_dir / "reference_training_history.json"
    if checkpoint.is_file():
        reference = PeriodicReferenceFlow.from_npz(
            checkpoint,
            substeps_per_interval=int(
                cfg["reference"].get("rk4_substeps_per_time_interval", 16)
            ),
        )
        history = (
            json.loads(history_path.read_text(encoding="utf-8"))
            if history_path.is_file()
            else []
        )
    else:
        source = _endpoint_source(
            population,
            split.train,
            n=int(cfg["reference"].get("endpoint_particles", 50000)),
            seed=int(cfg["seed"])
            + int(cfg["reference"].get("endpoint_seed_offset", 2001)),
        )
        reference, history = train_periodic_reference_flow(
            source,
            _flow_config(cfg, int(reference_seed)),
            periods=jnp.asarray(periods),
            substeps_per_interval=int(
                cfg["reference"].get("rk4_substeps_per_time_interval", 16)
            ),
        )
        reference.save(checkpoint)
        history_path.write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    bank_path = seed_dir / "reference_bank.npz"
    if bank_path.is_file():
        with np.load(bank_path, allow_pickle=False) as data:
            saved_times = np.asarray(data["times"])
            nodes = jnp.asarray(data["nodes"])
            velocity = jnp.asarray(data["velocity"])
            weights = jnp.asarray(data["weights"])
        if not np.array_equal(saved_times, normalized_times):
            raise ValueError("saved reference bank times do not match the defect bank")
        expected_particles = int(cfg["reference"].get("particles", 8192))
        if nodes.shape[1] != expected_particles:
            raise ValueError(
                "saved reference bank particle count does not match the current config: "
                f"found {nodes.shape[1]}, expected {expected_particles}"
            )
    else:
        rng = np.random.default_rng(
            int(reference_seed)
            + int(cfg["reference"].get("bank_seed_offset", 3001))
        )
        initial_pool = population.samples(0, split.train)
        particle_n = int(cfg["reference"].get("particles", 8192))
        x0 = jnp.asarray(
            initial_pool[rng.integers(0, len(initial_pool), size=particle_n)]
        )
        nodes = reference.rollout(x0, jnp.asarray(normalized_times))
        velocity = jax.vmap(lambda time, x: reference.velocity(x, time))(
            jnp.asarray(normalized_times), nodes
        )
        weights = jnp.full(
            nodes.shape[:2], 1.0 / particle_n, dtype=jnp.float64
        )
        np.savez_compressed(
            bank_path,
            times=normalized_times,
            nodes=nodes,
            velocity=velocity,
            weights=weights,
        )
    return nodes, velocity, weights, history


def build_reference_banks(cfg: dict[str, Any], output_dir: Path) -> Path:
    state_suffix = _state_suffix(cfg)
    population = DefectPopulationBank.load(
        output_dir / f"positive_defect_bank_{state_suffix}.npz"
    )
    if population.state_config != _state(cfg):
        raise ValueError("saved defect bank state policy does not match the current config")
    split = make_run_split(_split(cfg))
    normalized_times = np.asarray(
        (population.times - population.times[0])
        / (population.times[-1] - population.times[0]),
        dtype=np.float64,
    )
    rows = []
    for reference_seed in cfg["reference_training"].get("seeds", [cfg["seed"]]):
        seed_dir = output_dir / f"{state_suffix}_reference_seed_{int(reference_seed)}"
        _build_or_load_reference_bank(
            cfg,
            population,
            split,
            normalized_times,
            int(reference_seed),
            seed_dir,
        )
        rows.append(
            {
                "reference_seed": int(reference_seed),
                "checkpoint": str(seed_dir / "reference.npz"),
                "reference_bank": str(seed_dir / "reference_bank.npz"),
            }
        )
    manifest = output_dir / f"manifest_{state_suffix}_reference.json"
    _write_merged_manifest(manifest, rows)
    return manifest


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
    summaries = []
    for reference_seed in cfg["reference_training"].get("seeds", [cfg["seed"]]):
        seed_dir = output_dir / f"{state_suffix}_reference_seed_{int(reference_seed)}"
        nodes, velocity, weights, history = _build_or_load_reference_bank(
            cfg,
            population,
            split,
            np.asarray(normalized_times),
            int(reference_seed),
            seed_dir,
        )

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
        print(
            f"active_nematic reference_seed={int(reference_seed)} phase=selection "
            f"trials={int(selection_bank.sample_indices.shape[0])}",
            flush=True,
        )
        comparison = experiment.optimize_designs(selection_bank)
        print(
            f"active_nematic reference_seed={int(reference_seed)} "
            f"phase=selection_complete certified={comparison.certified}",
            flush=True,
        )
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
        print(
            f"active_nematic reference_seed={int(reference_seed)} phase=validation "
            f"trials={int(validation_bank.sample_indices.shape[0])}",
            flush=True,
        )
        designs = {"law": comparison.law_eta, "tangent": comparison.tangent_eta, "full": comparison.full_eta}
        easy_eta = cfg.get("controls", {}).get("reference_easy_eta")
        if easy_eta is not None:
            easy_eta = jnp.asarray(easy_eta, dtype=jnp.float64)
            if easy_eta.shape != (2 * experiment.family.n_sensors,):
                raise ValueError("controls.reference_easy_eta must contain two coordinates per sensor")
            designs["reference_easy"] = experiment.family.canonicalize(easy_eta)
        payload = {
            "schema_version": 1,
            "experiment": cfg["name"],
            "reference_seed": int(reference_seed),
            "state": cfg["state"],
            "evaluation_times": {
                "physical": population.times.tolist(),
                "normalized": normalized_times.tolist(),
                "action_weights": np.asarray(experiment.time_weights).tolist(),
            },
            "normalized_extant_defect_law": True,
            "count_is_auxiliary": True,
            "full_action_supported": experiment.full_action_supported,
            "full_action_blocker": None,
            "full_action_backend_3d": experiment.poisson3d_backend,
            "polarity_metric_radius": experiment.polarity_metric_radius,
            "full_action_solver": experiment.full_action_provenance(),
            "observation_banks": {
                "selection": {
                    "trials": int(selection_bank.sample_indices.shape[0]),
                    "namespace": int(random_cfg.get("selection_namespace", 9890)),
                },
                "validation": {
                    "trials": int(validation_bank.sample_indices.shape[0]),
                    "namespace": int(random_cfg.get("validation_namespace", 9891)),
                },
                "finite_n": int(cfg["measurement"]["finite_n"]),
                "truth_particles": truth_n,
            },
            "reference_bank": {
                "particles": int(nodes.shape[1]),
            },
            "risk_star": comparison.risk_star,
            "risk_max": comparison.risk_max,
            "selection_certified": comparison.certified,
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
        print(
            f"active_nematic reference_seed={int(reference_seed)} "
            f"phase=validation_complete result={seed_dir / 'result.json'}",
            flush=True,
        )
        summaries.append({"reference_seed": int(reference_seed), "result": str(seed_dir / "result.json")})

    manifest = output_dir / f"manifest_{state_suffix}.json"
    _write_merged_manifest(manifest, summaries)
    return manifest


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    if args.reference_seeds is not None:
        cfg["reference_training"]["seeds"] = list(args.reference_seeds)
    if args.time_guard_points is not None:
        if args.time_guard_points < 0:
            raise ValueError("--time-guard-points must be nonnegative")
        cfg.setdefault("evaluation", {})["time_guard_points"] = int(
            args.time_guard_points
        )
    if args.reference_particles is not None:
        if args.reference_particles <= 0:
            raise ValueError("--reference-particles must be positive")
        cfg.setdefault("reference", {})["particles"] = int(
            args.reference_particles
        )
    if args.selection_trials is not None:
        if args.selection_trials <= 0:
            raise ValueError("--selection-trials must be positive")
        cfg.setdefault("randomness", {})["selection_trials"] = int(
            args.selection_trials
        )
    if args.validation_trials is not None:
        if args.validation_trials <= 0:
            raise ValueError("--validation-trials must be positive")
        cfg.setdefault("randomness", {})["validation_trials"] = int(
            args.validation_trials
        )
    if args.position_only:
        cfg["state"]["mode"] = "position"
        cfg["measurement"]["channels"] = ["occupancy"]
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else _output_dir(cfg, args.smoke)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "physical-bank":
        result = build_physical_bank(cfg, output_dir)
    elif args.stage == "defects":
        result = build_defect_bank(cfg, output_dir)
    elif args.stage == "reference":
        result = build_reference_banks(cfg, output_dir)
    elif args.stage == "mfsi":
        result = run_mfsi(cfg, output_dir)
    else:
        result = rerun_validation(cfg, output_dir)
    print(f"active_nematic stage={args.stage} output={result}", flush=True)


if __name__ == "__main__":
    main()
