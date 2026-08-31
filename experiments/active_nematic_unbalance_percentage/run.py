"""Staged CLI for the isolated two-species unbalanced experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
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
from mfsi.io import jsonable, write_json

from active_nematic_solver import ActiveNematicParams
from domain import (
    EmpiricalEndpointSource,
    PhysicalBank,
    SplitConfig,
    generate_physical_bank,
    make_run_split,
)
from periodic_reference import PeriodicReferenceFlow, train_periodic_reference_flow
from risk import histogram_mass, periodic_grid_mmd2
from unbalanced_experiment import (
    SpeciesExperimentData,
    UnbalancedActiveNematicExperiment,
    make_unbalanced_observation_bank,
)
from unbalanced_reference import (
    endpoint_pair_mass_schedule,
    endpoint_source_for_species,
    sample_periodic_kde_bank,
)
from unbalanced_state import (
    TwoSpeciesDefectBank,
    UnbalancedStateConfig,
    extract_two_species_bank,
    reconstruct_coupled_mass_trajectory,
)

CONFIG_PATH = SCRIPT_DIR / "config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-species unbalanced active-nematic MFSI")
    parser.add_argument("stage", choices=(
        "physical-bank", "defects", "defect-audit", "reference",
        "fixed-design", "reference-audit", "design",
    ))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--base-physical-bank",
        type=Path,
        help="extend a compatible prefix bank instead of regenerating its runs",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="read physical/defect banks from another isolated output directory",
    )
    parser.add_argument("--reference-seeds", nargs="+", type=int)
    parser.add_argument("--eta", nargs="+", type=float, help="fixed design x1 y1 ...")
    return parser.parse_args()


def physics_config(cfg: dict[str, Any]) -> ActiveNematicParams:
    return ActiveNematicParams(**cfg["physics"])


def state_config(cfg: dict[str, Any]) -> UnbalancedStateConfig:
    block = cfg["state"]
    return UnbalancedStateConfig(
        orientation_coherence_min_plus=float(block["orientation_coherence_min_plus"]),
        orientation_coherence_min_minus=float(block["orientation_coherence_min_minus"]),
        maximum_core_residual=block.get("maximum_core_residual"),
        fit_rmin_cells=float(block.get("fit_rmin_cells", 2.0)),
        fit_rmax_cells=float(block.get("fit_rmax_cells", 6.0)),
    )


def split_config(cfg: dict[str, Any]) -> SplitConfig:
    block = cfg["splits"]
    explicit = {
        key: (tuple(int(value) for value in block[key]) if key in block else None)
        for key in ("train_indices", "design_indices", "validation_indices")
    }
    return SplitConfig(
        train_runs=int(block["train_runs"]),
        design_runs=int(block["design_runs"]),
        validation_runs=int(block["validation_runs"]),
        seed=int(cfg["seed"]) + int(block.get("seed_offset", 1101)),
        **explicit,
    )


def flow_config(cfg: dict[str, Any], seed: int) -> FlowMatchingConfig:
    block = cfg["reference_training"]
    return FlowMatchingConfig(
        seed=int(seed), hidden_width=int(block.get("hidden_width", 128)),
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


def normalized_times(bank: TwoSpeciesDefectBank) -> np.ndarray:
    times = np.asarray(bank.times, dtype=np.float64)
    interval = float(times[-1] - times[0])
    if not np.isfinite(times).all() or interval <= 0.0:
        raise ValueError("reference physical times must define a finite positive interval")
    return np.asarray((times - times[0]) / interval, dtype=np.float64)


def output_dir(args: argparse.Namespace) -> Path:
    return (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else SCRIPT_DIR / "outputs" / ("smoke" if args.smoke else "run")
    )


def charge_diagnostics(cfg, bank, runs=None):
    block = cfg["unbalanced"]
    return bank.charge_balance(
        run_indices=runs,
        tolerance=float(block["charge_balance_tolerance"]),
        expected_imbalance=float(block.get("expected_charge_imbalance", 0.0)),
        enforce=bool(block.get("enforce_charge_balance", True)),
    )


def build_physical_bank(cfg, target: Path, base_path: Path | None = None) -> Path:
    split = split_config(cfg)
    seeds = int(cfg["seed"]) + int(cfg["physical_bank"].get("seed_offset", 1001)) + np.arange(split.total_runs)
    params = physics_config(cfg)
    times = np.asarray(cfg["physical_bank"]["save_times"], dtype=np.float64)
    if base_path is None:
        bank = generate_physical_bank(
            params, seeds=seeds, times=times,
            workers=int(cfg["physical_bank"].get("workers", 1)),
        )
    else:
        base = PhysicalBank.load(base_path.expanduser().resolve())
        if base.params != params:
            raise ValueError("base physical-bank parameters do not match the requested configuration")
        if not np.array_equal(base.times, times):
            raise ValueError("base physical-bank times do not match the requested configuration")
        if len(base.seeds) > len(seeds) or not np.array_equal(base.seeds, seeds[:len(base.seeds)]):
            raise ValueError("base physical-bank seeds are not the required deterministic prefix")
        if len(base.seeds) == len(seeds):
            bank = base
        else:
            extension = generate_physical_bank(
                params, seeds=seeds[len(base.seeds):], times=times,
                workers=int(cfg["physical_bank"].get("workers", 1)),
            )
            bank = PhysicalBank(
                times=times,
                q1=np.concatenate((base.q1, extension.q1), axis=0),
                q2=np.concatenate((base.q2, extension.q2), axis=0),
                seeds=np.concatenate((base.seeds, extension.seeds)),
                params=params,
            )
    path = target / "physical_bank.npz"
    bank.save(path)
    return path


def build_defect_bank(cfg, target: Path) -> Path:
    physical = PhysicalBank.load(target / "physical_bank.npz").select_times(
        np.asarray(cfg["physical_bank"]["population_times"], dtype=np.float64)
    )
    bank = extract_two_species_bank(physical, state_config(cfg))
    diagnostics = charge_diagnostics(cfg, bank)
    path = target / "two_species_defect_bank.npz"
    bank.save(path)
    write_json(target / "charge_balance.json", diagnostics.to_dict())
    return path


def audit_defect_bank(cfg, target: Path) -> Path:
    bank = TwoSpeciesDefectBank.load(target / "two_species_defect_bank.npz")
    split = make_run_split(split_config(cfg))
    payload = {
        "schema_version": 1,
        "experiment": cfg["name"],
        "state": "(x,y,beta); beta_minus is triatic phase",
        "weight_per_defect": "1 / selected physical realization count",
        "all": charge_diagnostics(cfg, bank).to_dict(),
        "train": charge_diagnostics(cfg, bank, split.train).to_dict(),
        "design": charge_diagnostics(cfg, bank, split.design).to_dict(),
        "validation": charge_diagnostics(cfg, bank, split.validation).to_dict(),
    }
    return write_json(target / "defect_bank_audit.json", payload)


def reference_seeds(cfg) -> list[int]:
    return [int(seed) for seed in cfg["reference_training"].get("seeds", [cfg["seed"]])]


def species_reference_seed(cfg, base_seed: int, species: str) -> int:
    default = 0 if species == "plus" else 10_000
    return int(base_seed) + int(cfg["reference_training"].get(f"{species}_seed_offset", default))


def reference_training_source(cfg, bank, train_runs, species: str, periods):
    """Return the frozen endpoint source under the declared initial-law semantics."""
    minimum_mass = float(cfg["unbalanced"]["minimum_mass"])
    endpoint_seed = (
        int(cfg["seed"])
        + int(cfg["reference"].get("endpoint_seed_offset", 2001))
        + (species == "minus")
    )
    source = endpoint_source_for_species(
        bank,
        species,
        run_indices=train_runs,
        sample_count=int(cfg["reference"]["endpoint_particles"]),
        seed=endpoint_seed,
        minimum_mass=minimum_mass,
    )
    density_model = cfg["reference_training"].get(
        "initial_endpoint_density_model", "empirical"
    )
    if density_model == "empirical":
        return source
    if density_model != "periodic_kde":
        raise ValueError(
            "reference_training.initial_endpoint_density_model must be "
            "'empirical' or 'periodic_kde'"
        )

    measure = bank.measure(species, 0, train_runs)
    probabilities = measure.normalized_probabilities(minimum_mass=minimum_mass)
    x0 = sample_periodic_kde_bank(
        measure.states,
        probabilities,
        sample_count=int(cfg["reference"]["endpoint_particles"]),
        seed=endpoint_seed,
        periods=np.asarray(periods),
        position_std=float(
            cfg["reference"].get("bank_position_jitter_std", 0.0)
        ),
        beta_std=float(cfg["reference"].get("bank_beta_jitter_std", 0.0)),
    )
    # Preserve the exact target sample bank used by the empirical-source
    # baseline.  This isolates the initial-density semantics alone.
    return EmpiricalEndpointSource(jnp.asarray(x0), source.x1)


def ensure_reference(cfg, bank, train_runs, base_seed: int, seed_dir: Path):
    """Build/load two normalized endpoint flows and one analytic mass schedule."""
    if cfg["unbalanced"].get("mass_interpolation", "fisher_rao") != "fisher_rao":
        raise ValueError("this implementation requires mass_interpolation='fisher_rao'")
    seed_dir.mkdir(parents=True, exist_ok=True)
    tau = normalized_times(bank)
    periods = jnp.asarray([bank.box_size, bank.box_size, 2.0 * np.pi])
    minimum_mass = float(cfg["unbalanced"]["minimum_mass"])
    training_density_model = cfg["reference_training"].get(
        "initial_endpoint_density_model", "empirical"
    )
    schedule = endpoint_pair_mass_schedule(bank, run_indices=train_runs, minimum_mass=minimum_mass)
    result = {}
    for species in ("plus", "minus"):
        seed = species_reference_seed(cfg, base_seed, species)
        checkpoint = seed_dir / f"{species}_reference.npz"
        history_path = seed_dir / f"{species}_reference_training_history.json"
        if checkpoint.is_file():
            flow = PeriodicReferenceFlow.from_npz(
                checkpoint,
                substeps_per_interval=int(cfg["reference"].get("rk4_substeps_per_time_interval", 16)),
            )
            saved_density_model = (flow.metadata or {}).get(
                "training_initial_density_model", "empirical"
            )
            if saved_density_model != training_density_model:
                raise RuntimeError(
                    f"{checkpoint} was trained with initial density model "
                    f"{saved_density_model!r}, expected {training_density_model!r}"
                )
            history = json.loads(history_path.read_text()).get("history", []) if history_path.is_file() else []
        else:
            source = reference_training_source(
                cfg, bank, train_runs, species, periods
            )
            flow, history = train_periodic_reference_flow(
                source, flow_config(cfg, seed), periods=periods,
                substeps_per_interval=int(cfg["reference"].get("rk4_substeps_per_time_interval", 16)),
            )
            metadata = dict(flow.metadata or {})
            metadata.update({
                "experiment": cfg["name"], "species": species,
                "physical_interval": [float(bank.times[0]), float(bank.times[-1])],
                "endpoint_only": True,
                "intermediate_marginals_used_for_training": False,
                "training_initial_density_model": training_density_model,
                "training_initial_position_jitter_std": (
                    float(cfg["reference"].get("bank_position_jitter_std", 0.0))
                    if training_density_model == "periodic_kde"
                    else 0.0
                ),
                "training_initial_beta_jitter_std": (
                    float(cfg["reference"].get("bank_beta_jitter_std", 0.0))
                    if training_density_model == "periodic_kde"
                    else 0.0
                ),
                "rollout_initial_density_model": "periodic_kde",
            })
            flow = PeriodicReferenceFlow(flow.params, flow.periods, flow.substeps_per_interval, metadata)
            flow.save(checkpoint)
            write_json(history_path, {"history": history})

        bank_path = seed_dir / f"{species}_reference_bank.npz"
        position_jitter = float(
            cfg["reference"].get("bank_position_jitter_std", 0.0)
        )
        beta_jitter = float(cfg["reference"].get("bank_beta_jitter_std", 0.0))
        rebuild_bank = True
        if bank_path.is_file():
            with np.load(bank_path, allow_pickle=False) as saved:
                saved_tau = np.asarray(saved["times"])
                saved_position_jitter = (
                    float(saved["bank_position_jitter_std"])
                    if "bank_position_jitter_std" in saved.files
                    else 0.0
                )
                saved_beta_jitter = (
                    float(saved["bank_beta_jitter_std"])
                    if "bank_beta_jitter_std" in saved.files
                    else 0.0
                )
                rebuild_bank = not (
                    np.array_equal(saved_tau, tau)
                    and int(saved["nodes"].shape[1])
                    == int(cfg["reference"]["particles"])
                    and np.isclose(saved_position_jitter, position_jitter)
                    and np.isclose(saved_beta_jitter, beta_jitter)
                )
                if not rebuild_bank:
                    nodes, velocity, weights = map(
                        jnp.asarray,
                        (saved["nodes"], saved["velocity"], saved["weights"]),
                    )
        if rebuild_bank:
            measure = bank.measure(species, 0, train_runs)
            probabilities = measure.normalized_probabilities(minimum_mass=minimum_mass)
            # Hold reference-bank particles fixed across learned-reference seeds;
            # only network initialization/minibatches vary in the seed audit.
            particle_n = int(cfg["reference"]["particles"])
            x0 = jnp.asarray(
                sample_periodic_kde_bank(
                    measure.states,
                    probabilities,
                    sample_count=particle_n,
                    seed=int(cfg["seed"])
                    + int(cfg["reference"].get("bank_seed_offset", 3001))
                    + (0 if species == "plus" else 1),
                    periods=np.asarray(periods),
                    position_std=position_jitter,
                    beta_std=beta_jitter,
                )
            )
            nodes = flow.rollout(x0, jnp.asarray(tau))
            velocity = jax.vmap(lambda time, row: flow.velocity(row, time))(jnp.asarray(tau), nodes)
            weights = jnp.full(nodes.shape[:2], 1.0 / particle_n)
            np.savez_compressed(
                bank_path,
                times=tau,
                nodes=nodes,
                velocity=velocity,
                weights=weights,
                bank_position_jitter_std=position_jitter,
                bank_beta_jitter_std=beta_jitter,
                bank_density_model="periodic_kde",
            )
        result[species] = {
            "flow": flow, "seed": seed, "history": history, "nodes": nodes,
            "velocity": velocity, "weights": weights, "checkpoint": checkpoint,
            "bank": bank_path,
        }
    result["schedule"] = schedule
    write_json(seed_dir / "reference_mass_schedule.json", schedule.to_dict(tau))
    return result


def build_references(cfg, target: Path, bank_dir: Path) -> Path:
    bank = TwoSpeciesDefectBank.load(bank_dir / "two_species_defect_bank.npz")
    split = make_run_split(split_config(cfg))
    charge_diagnostics(cfg, bank, split.train)
    rows = []
    for base_seed in reference_seeds(cfg):
        seed_dir = target / f"reference_seed_{base_seed}"
        reference = ensure_reference(cfg, bank, split.train, base_seed, seed_dir)
        rows.append({
            "reference_seed": base_seed,
            "plus_reference_seed": reference["plus"]["seed"],
            "minus_reference_seed": reference["minus"]["seed"],
            "plus_checkpoint": str(reference["plus"]["checkpoint"]),
            "minus_checkpoint": str(reference["minus"]["checkpoint"]),
        })
    return write_json(target / "reference_manifest.json", {"schema_version": 1, "runs": rows})


def mass_trajectory(cfg, bank, runs, tau):
    count = int(cfg["measurement"]["acquisition_k"])
    acq = np.unique(np.rint(np.linspace(0, len(tau) - 1, count)).astype(int))
    return reconstruct_coupled_mass_trajectory(
        tau[acq], bank.mean_mass("plus", runs)[acq], bank.mean_mass("minus", runs)[acq], tau,
        minimum_mass=float(cfg["unbalanced"]["minimum_mass"]),
        smoothing=float(cfg["unbalanced"].get("mass_smoothing", 1.0e-4)),
        internal_knots=int(cfg["moment_reconstruction"].get("internal_knots", 3)),
    )


def make_experiment(cfg, bank, runs, truth_seed: int, reference):
    tau = normalized_times(bank)
    minimum_mass = float(cfg["unbalanced"]["minimum_mass"])
    truth_n = int(cfg["randomness"]["truth_particles"])
    masses = mass_trajectory(cfg, bank, runs, tau)
    schedule = reference["schedule"]
    values = {}
    for species in ("plus", "minus"):
        truth = bank.resample_normalized_trajectory(
            species, run_indices=runs, n=truth_n,
            seed=truth_seed + (species == "minus"), minimum_mass=minimum_mass,
        )
        ref = reference[species]
        values[species] = SpeciesExperimentData(
            truth_particles=jnp.asarray(truth), truth_mass=jnp.asarray(bank.mean_mass(species, runs)),
            reference_nodes=ref["nodes"], reference_velocity=ref["velocity"], reference_weights=ref["weights"],
            reference_mass=schedule.species_mass(species, jnp.asarray(tau)),
            reference_source_rate=schedule.species_source_rate(species, jnp.asarray(tau)),
            target_mass=jnp.asarray(getattr(masses, f"mass_{species}")),
            target_mass_dot=jnp.asarray(getattr(masses, f"mass_dot_{species}")),
            target_relative_mass_rate=jnp.asarray(getattr(masses, f"relative_rate_{species}")),
        )
    return UnbalancedActiveNematicExperiment(
        cfg, times=jnp.asarray(tau), plus=values["plus"], minus=values["minus"]
    ), masses


def observation_bank(cfg, experiment, namespace, trials):
    truth_n = int(cfg["randomness"]["truth_particles"])
    return make_unbalanced_observation_bank(
        seed=int(cfg["seed"]), namespace=int(namespace), trials=int(trials),
        acquisition_count=int(cfg["measurement"]["acquisition_k"]),
        finite_n=int(cfg["measurement"]["finite_n"]),
        plus_truth_particle_count=truth_n, minus_truth_particle_count=truth_n,
        plus_observables=experiment.sensors.plus_observables,
        minus_observables=experiment.sensors.minus_observables,
    )


def summarize_rows(rows):
    numeric = [key for key in rows[0] if key not in ("trial", "valid", "sensor_geometry")]
    return {
        "valid_trials": sum(row["valid"] for row in rows), "trials": len(rows),
        "metrics": {
            key: {
                "mean": float(np.mean([row[key] for row in rows])),
                "se": float(np.std([row[key] for row in rows], ddof=1) / np.sqrt(len(rows))) if len(rows) > 1 else 0.0,
            } for key in numeric
        },
    }


def base_payload(cfg, bank, base_seed, reference, masses, eta, rows, diagnostics):
    tau = normalized_times(bank)
    schedule = reference["schedule"]
    validation = summarize_rows(rows)
    metric_mean = {
        name: values["mean"] for name, values in validation["metrics"].items()
    }
    return {
        "schema_version": 1, "experiment": cfg["name"], "config": cfg,
        "reference_seed": base_seed,
        "plus_reference_seed": reference["plus"]["seed"], "minus_reference_seed": reference["minus"]["seed"],
        "state_representation": {
            "plus": "(x,y,beta_plus): vector/comet polarity",
            "minus": "(x,y,beta_minus): triatic phase; arm=beta_minus/3 mod 2pi/3",
        },
        "physical_interval": [float(bank.times[0]), float(bank.times[-1])],
        "normalized_times": tau.tolist(),
        "reference_training_uses_endpoints_only": True,
        "reference_checkpoints": {
            "plus": str(reference["plus"]["checkpoint"]),
            "minus": str(reference["minus"]["checkpoint"]),
        },
        "reference_training_history": {
            "plus": reference["plus"]["history"],
            "minus": reference["minus"]["history"],
        },
        "mass_plus": np.asarray(masses.mass_plus).tolist(), "mass_minus": np.asarray(masses.mass_minus).tolist(),
        "mass_rate_plus": np.asarray(masses.mass_dot_plus).tolist(), "mass_rate_minus": np.asarray(masses.mass_dot_minus).tolist(),
        "reference_mass_plus": np.asarray(schedule.species_mass("plus", jnp.asarray(tau))).tolist(),
        "reference_mass_minus": np.asarray(schedule.species_mass("minus", jnp.asarray(tau))).tolist(),
        "reference_source_plus": np.asarray(schedule.species_source_rate("plus", jnp.asarray(tau))).tolist(),
        "reference_source_minus": np.asarray(schedule.species_source_rate("minus", jnp.asarray(tau))).tolist(),
        "reaction_kappa": float(cfg["unbalanced"]["reaction_kappa"]),
        "species_weight_plus": float(cfg["unbalanced"].get("species_weight_plus", 1.0)),
        "species_weight_minus": float(cfg["unbalanced"].get("species_weight_minus", 1.0)),
        "risk_weight_plus": float(cfg["unbalanced"].get("risk_weight_plus", 1.0)),
        "risk_weight_minus": float(cfg["unbalanced"].get("risk_weight_minus", 1.0)),
        "sensor_geometry": np.asarray(eta).reshape((-1, 2)).tolist(),
        "charge_balance_diagnostics": diagnostics.to_dict(),
        "law_risk_total": metric_mean["law_risk_total"],
        "law_risk_plus": metric_mean["law_risk_plus"],
        "law_risk_minus": metric_mean["law_risk_minus"],
        "full_unbalanced_action_total": metric_mean["full_unbalanced_action_total"],
        "full_unbalanced_action_plus": metric_mean["full_unbalanced_action_plus"],
        "full_unbalanced_action_minus": metric_mean["full_unbalanced_action_minus"],
        "move_action_plus": metric_mean["move_action_plus"],
        "reaction_action_plus": metric_mean["reaction_action_plus"],
        "move_action_minus": metric_mean["move_action_minus"],
        "reaction_action_minus": metric_mean["reaction_action_minus"],
        "validity_flags": [bool(row["valid"]) for row in rows],
        "pde_residuals": [
            float(row["max_screened_pde_relative_residual"]) for row in rows
        ],
        "validation": validation,
        "validation_rows": rows,
    }


def fixed_eta(cfg, values) -> jax.Array:
    values = values if values is not None else cfg.get("controls", {}).get("fixed_design_eta")
    if values is None:
        raise ValueError("fixed-design requires --eta or controls.fixed_design_eta")
    eta = jnp.asarray(values, dtype=jnp.float64)
    expected = 2 * int(cfg["measurement"]["n_sensors"])
    if eta.shape != (expected,):
        raise ValueError(f"fixed design requires {expected} coordinates")
    return eta


def run_fixed_design(cfg, target, bank_dir, eta):
    bank = TwoSpeciesDefectBank.load(bank_dir / "two_species_defect_bank.npz")
    split = make_run_split(split_config(cfg))
    charge_diagnostics(cfg, bank, split.train)
    diagnostics = charge_diagnostics(cfg, bank, split.validation)
    results = []
    for base_seed in reference_seeds(cfg):
        seed_dir = target / f"reference_seed_{base_seed}"
        reference = ensure_reference(cfg, bank, split.train, base_seed, seed_dir)
        experiment, masses = make_experiment(cfg, bank, split.validation, int(cfg["seed"]) + 4002, reference)
        trials = observation_bank(cfg, experiment, cfg["randomness"]["validation_namespace"], cfg["randomness"]["validation_trials"])
        eta_canonical = experiment.sensors.canonicalize(eta)
        rows = experiment.certified_trial_rows(eta_canonical, trials)
        path = seed_dir / "fixed_design_result.json"
        write_json(path, jsonable(base_payload(cfg, bank, base_seed, reference, masses, eta_canonical, rows, diagnostics)))
        results.append({"reference_seed": base_seed, "result": str(path)})
    manifest_path = target / "fixed_design_manifest.json"
    merged = {}
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        merged.update(
            {int(row["reference_seed"]): row for row in previous.get("runs", [])}
        )
    merged.update({int(row["reference_seed"]): row for row in results})
    return write_json(
        manifest_path,
        {"schema_version": 1, "runs": [merged[key] for key in sorted(merged)]},
    )


def _materialize_reference_dir(target: Path, bank_dir: Path, base_seed: int) -> Path:
    """Copy reusable reference artifacts into this experiment's output tree."""
    source = bank_dir / f"reference_seed_{base_seed}"
    destination = target / f"reference_seed_{base_seed}"
    destination.mkdir(parents=True, exist_ok=True)
    required = (
        source / "plus_reference.npz",
        source / "minus_reference.npz",
        source / "plus_reference_bank.npz",
        source / "minus_reference_bank.npz",
    )
    if all(path.is_file() for path in required):
        for source_path in source.glob("*reference*.npz"):
            destination_path = destination / source_path.name
            if not destination_path.is_file():
                shutil.copy2(source_path, destination_path)
        for source_path in source.glob("*reference_training_history.json"):
            destination_path = destination / source_path.name
            if not destination_path.is_file():
                shutil.copy2(source_path, destination_path)
    return destination


def run_design(cfg, target, bank_dir):
    bank = TwoSpeciesDefectBank.load(bank_dir / "two_species_defect_bank.npz")
    split = make_run_split(split_config(cfg))
    charge_diagnostics(cfg, bank, split.train)
    design_diag = charge_diagnostics(cfg, bank, split.design)
    validation_diag = charge_diagnostics(cfg, bank, split.validation)
    results = []
    for base_seed in reference_seeds(cfg):
        result_dir = target / f"reference_seed_{base_seed}"
        result_dir.mkdir(parents=True, exist_ok=True)
        reference_dir = _materialize_reference_dir(target, bank_dir, base_seed)
        reference = ensure_reference(cfg, bank, split.train, base_seed, reference_dir)
        experiment, _ = make_experiment(cfg, bank, split.design, int(cfg["seed"]) + 4001, reference)
        selection = observation_bank(cfg, experiment, cfg["randomness"]["selection_namespace"], cfg["randomness"]["selection_trials"])
        comparison = experiment.optimize_designs(selection)
        validation_experiment, masses = make_experiment(cfg, bank, split.validation, int(cfg["seed"]) + 4002, reference)
        validation_bank = observation_bank(cfg, validation_experiment, cfg["randomness"]["validation_namespace"], cfg["randomness"]["validation_trials"])
        designs = {"law": comparison.law_eta, "tangent": comparison.tangent_eta, "unbalanced_full": comparison.full_eta}
        validation = {}
        for name, eta in designs.items():
            rows = validation_experiment.certified_trial_rows(
                eta, validation_bank
            )
            validation[name] = {"summary": summarize_rows(rows), "rows": rows}
        payload = base_payload(cfg, bank, base_seed, reference, masses, comparison.full_eta, validation["unbalanced_full"]["rows"], validation_diag)
        payload.update({
            "risk_star": comparison.risk_star, "risk_max": comparison.risk_max,
            "selection_certified": comparison.certified,
            "selection_charge_balance": design_diag.to_dict(),
            "designs": {name: np.asarray(eta).tolist() for name, eta in designs.items()},
            "selection_candidates": comparison.candidates, "validation_designs": validation,
        })
        path = result_dir / "result.json"
        write_json(path, jsonable(payload))
        results.append({"reference_seed": base_seed, "result": str(path)})
    manifest_path = target / "manifest.json"
    merged = {}
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        merged.update(
            {int(row["reference_seed"]): row for row in previous.get("runs", [])}
        )
    merged.update({int(row["reference_seed"]): row for row in results})
    return write_json(
        manifest_path,
        {"schema_version": 1, "runs": [merged[key] for key in sorted(merged)]},
    )


def audit_references(cfg, target: Path, bank_dir: Path) -> Path:
    """Cross-evaluate every saved Full candidate under every learned reference."""
    manifest_path = target / ("manifest.json" if (target / "manifest.json").is_file() else "fixed_design_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    candidates = []
    for entry in manifest["runs"]:
        result = json.loads(Path(entry["result"]).read_text())
        eta = (
            result["designs"]["unbalanced_full"]
            if "designs" in result
            else np.asarray(result["sensor_geometry"]).reshape(-1).tolist()
        )
        candidates.append({
            "source_reference_seed": int(result["reference_seed"]),
            "eta": eta,
        })
    bank = TwoSpeciesDefectBank.load(bank_dir / "two_species_defect_bank.npz")
    split = make_run_split(split_config(cfg))
    charge_diagnostics(cfg, bank, split.train)
    cross_rows = []
    references = {}
    for evaluation_seed in reference_seeds(cfg):
        seed_dir = target / f"reference_seed_{evaluation_seed}"
        reference = ensure_reference(cfg, bank, split.train, evaluation_seed, seed_dir)
        references[evaluation_seed] = reference
        experiment, _ = make_experiment(
            cfg, bank, split.validation, int(cfg["seed"]) + 4002, reference
        )
        endpoint_fit = {}
        for species in ("plus", "minus"):
            endpoint_fit[species] = {}
            for label, time_index in (("initial", 0), ("terminal", len(bank.times) - 1)):
                physical = bank.measure(species, time_index, split.train)
                physical_probabilities = physical.normalized_probabilities(
                    minimum_mass=float(cfg["unbalanced"]["minimum_mass"])
                )
                physical_histogram = histogram_mass(
                    jnp.asarray(physical.states),
                    jnp.asarray(physical_probabilities),
                    experiment.law_grid,
                )
                reference_histogram = histogram_mass(
                    reference[species]["nodes"][time_index],
                    reference[species]["weights"][time_index],
                    experiment.law_grid,
                )
                endpoint_fit[species][f"{label}_shape_mmd"] = float(
                    periodic_grid_mmd2(
                        reference_histogram,
                        physical_histogram,
                        experiment.law_kernel_fft,
                    )
                )
        reference["endpoint_fit"] = endpoint_fit
        trials = observation_bank(
            cfg, experiment, cfg["randomness"]["validation_namespace"],
            cfg["randomness"]["validation_trials"],
        )
        summary_cache = {}
        for candidate in candidates:
            eta = experiment.sensors.canonicalize(jnp.asarray(candidate["eta"]))
            eta_key = tuple(np.round(np.asarray(eta), decimals=12).tolist())
            if eta_key not in summary_cache:
                rows = experiment.certified_trial_rows(eta, trials)
                summary_cache[eta_key] = summarize_rows(rows)
            summary = summary_cache[eta_key]
            metrics = summary["metrics"]
            cross_rows.append({
                "source_reference_seed": candidate["source_reference_seed"],
                "evaluation_reference_seed": evaluation_seed,
                "sensor_geometry": np.asarray(eta).reshape((-1, 2)).tolist(),
                "law_risk_total": metrics["law_risk_total"]["mean"],
                "full_unbalanced_action_total": metrics["full_unbalanced_action_total"]["mean"],
                "move_action_plus": metrics["move_action_plus"]["mean"],
                "reaction_action_plus": metrics["reaction_action_plus"]["mean"],
                "move_action_minus": metrics["move_action_minus"]["mean"],
                "reaction_action_minus": metrics["reaction_action_minus"]["mean"],
                "reaction_fraction_total": metrics["reaction_fraction_total"]["mean"],
                "valid_trials": summary["valid_trials"],
                "trials": summary["trials"],
            })
    pairwise_paths = []
    for index, seed_a in enumerate(sorted(references)):
        for seed_b in sorted(references)[index + 1 :]:
            row = {"seed_a": seed_a, "seed_b": seed_b}
            for species in ("plus", "minus"):
                a = references[seed_a][species]
                b = references[seed_b][species]
                nodes_a, nodes_b = np.asarray(a["nodes"]), np.asarray(b["nodes"])
                velocity_a, velocity_b = np.asarray(a["velocity"]), np.asarray(b["velocity"])
                periods = np.asarray([bank.box_size, bank.box_size, 2.0 * np.pi])
                node_delta = np.mod(nodes_a - nodes_b + 0.5 * periods, periods) - 0.5 * periods
                metric_radius = float(
                    cfg["full_action"].get("polarity_metric_radius", 1.0)
                )
                metric_node_delta = node_delta.copy()
                metric_node_delta[..., 2] *= metric_radius
                metric_velocity_a = velocity_a.copy()
                metric_velocity_b = velocity_b.copy()
                metric_velocity_a[..., 2] *= metric_radius
                metric_velocity_b[..., 2] *= metric_radius
                scale = np.sqrt(
                    0.5
                    * (
                        np.mean(metric_velocity_a**2)
                        + np.mean(metric_velocity_b**2)
                    )
                )
                row[f"{species}_position_periodic_rms"] = float(
                    np.sqrt(np.mean(node_delta[..., :2] ** 2))
                )
                row[f"{species}_phase_circular_rms"] = float(
                    np.sqrt(np.mean(node_delta[..., 2] ** 2))
                )
                row[f"{species}_trajectory_product_metric_rms"] = float(
                    np.sqrt(np.mean(metric_node_delta**2))
                )
                row[f"{species}_velocity_normalized_rmse"] = float(
                    np.sqrt(
                        np.mean((metric_velocity_a - metric_velocity_b) ** 2)
                    )
                    / max(scale, 1.0e-300)
                )
            pairwise_paths.append(row)
    training = []
    for seed, reference in sorted(references.items()):
        training.append({
            "reference_seed": seed,
            "plus_reference_seed": reference["plus"]["seed"],
            "minus_reference_seed": reference["minus"]["seed"],
            "plus_last_training_loss": reference["plus"]["history"][-1]["conditional_fm_loss"] if reference["plus"]["history"] else None,
            "minus_last_training_loss": reference["minus"]["history"][-1]["conditional_fm_loss"] if reference["minus"]["history"] else None,
            "endpoint_fit": reference["endpoint_fit"],
        })
    return write_json(target / "reference_seed_audit.json", {
        "schema_version": 2,
        "reference_training": training,
        "pairwise_normalized_shape_reference_path_disagreement": pairwise_paths,
        "cross_reference_candidates": cross_rows,
        "common_conditional_flow_loss": None,
        "common_conditional_flow_loss_note": "not implemented; compare final training losses and path disagreement",
    })


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    if args.reference_seeds is not None:
        cfg["reference_training"]["seeds"] = list(args.reference_seeds)
    target = output_dir(args)
    target.mkdir(parents=True, exist_ok=True)
    bank_dir = (
        args.input_dir.expanduser().resolve()
        if args.input_dir is not None
        else target
    )
    if args.stage == "physical-bank":
        result = build_physical_bank(cfg, target, args.base_physical_bank)
    elif args.stage == "defects":
        result = build_defect_bank(cfg, target)
    elif args.stage == "defect-audit":
        result = audit_defect_bank(cfg, target)
    elif args.stage == "reference":
        result = build_references(cfg, target, bank_dir)
    elif args.stage == "fixed-design":
        result = run_fixed_design(cfg, target, bank_dir, fixed_eta(cfg, args.eta))
    elif args.stage == "reference-audit":
        result = audit_references(cfg, target, bank_dir)
    else:
        result = run_design(cfg, target, bank_dir)
    print(
        f"active_nematic_unbalance_percentage stage={args.stage} output={result}",
        flush=True,
    )


if __name__ == "__main__":
    main()
