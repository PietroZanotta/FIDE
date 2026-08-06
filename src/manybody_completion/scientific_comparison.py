"""Main scientific comparison with classical baselines and higher-order UQ."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .classical_baselines import (
    IBIOptions,
    RMCOptions,
    run_iterative_boltzmann_inversion,
    run_reverse_monte_carlo,
)
from .config import load_yaml
from .energy import PhysicalParameters
from .experiment import run_experiment
from .homometric import build_homometric_dataset
from .metrics import correction_arrays, stage_arrays, summarize_stage
from .observables import PairBasis, angular_cosine_moments
from .routing import evaluate_all_stages
from .solvers import LocalJaxBackend, ProjectionOptions, RelaxationOptions
from .statistics import paired_bootstrap_difference
from .uq import higher_order_conditional_uq


def _to_python(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_python(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_python(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if hasattr(value, "shape"):
        array = np.asarray(jax.device_get(value))
        return array.item() if array.shape == () else array.tolist()
    return value


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _build_problem(flow_config: dict[str, Any]) -> dict[str, Any]:
    dtype = jnp.float64 if flow_config["dtype"] == "float64" else jnp.float32
    jax.config.update("jax_enable_x64", dtype == jnp.float64)
    box = jnp.asarray(flow_config["box"], dtype=dtype)
    pair_config = flow_config["pair_basis"]
    basis = PairBasis.uniform(
        int(pair_config["num_basis"]),
        float(pair_config["r_min"]),
        float(pair_config["r_max"]),
        float(pair_config["width"]),
        dtype=dtype,
    )
    angular_orders = jnp.asarray(flow_config["angular"]["orders"], dtype=dtype)
    angular_neighbor_scale = float(flow_config["angular"]["neighbor_scale"])
    dataset = build_homometric_dataset(
        seed=int(flow_config["seed"]),
        samples_per_mode=int(flow_config["dataset"]["samples_per_mode"]),
        num_replicas=int(flow_config["dataset"]["num_replicas"]),
        box=box,
        basis=basis,
        angular_orders=angular_orders,
        angular_neighbor_scale=angular_neighbor_scale,
    )
    backend = LocalJaxBackend(
        box=box,
        basis=basis,
        moment_scales=jnp.ones_like(dataset["common_pair_moments"]),
        physical=PhysicalParameters(**flow_config["physical"]),
        relaxation_options=RelaxationOptions(**flow_config["relaxation"]),
        projection_options=ProjectionOptions(**flow_config["projection"]),
    )
    return {
        "dtype": dtype,
        "box": box,
        "basis": basis,
        "angular_orders": angular_orders,
        "angular_neighbor_scale": angular_neighbor_scale,
        "dataset": dataset,
        "backend": backend,
    }


def _reference_descriptors(
    problem: dict[str, Any],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> dict[str, Array]:
    dataset = problem["dataset"]
    coordinates = dataset["coordinates"]
    num_replicas = coordinates.shape[1]
    num_particles = coordinates.shape[2]
    train_coordinates = coordinates[jnp.asarray(train_indices)]
    validation_coordinates = coordinates[jnp.asarray(validation_indices)]
    train_flat = train_coordinates.reshape((-1, num_particles, 2))
    validation_flat = validation_coordinates.reshape((-1, num_particles, 2))
    train_descriptors = angular_cosine_moments(
        train_flat,
        problem["box"],
        problem["angular_orders"],
        problem["angular_neighbor_scale"],
    )
    validation_descriptors = angular_cosine_moments(
        validation_flat,
        problem["box"],
        problem["angular_orders"],
        problem["angular_neighbor_scale"],
    )
    labels = dataset["labels"]
    train_labels = jnp.repeat(labels[jnp.asarray(train_indices)], num_replicas)
    reference_a = jnp.mean(train_descriptors[train_labels == 0], axis=0)
    reference_b = jnp.mean(train_descriptors[train_labels == 1], axis=0)
    scale_floor = float(problem["flow_config"]["angular"]["scale_floor"])
    angular_scale = jnp.maximum(
        jnp.std(train_descriptors, axis=0),
        jnp.asarray(scale_floor, dtype=train_descriptors.dtype),
    )
    return {
        "train": train_descriptors,
        "validation": validation_descriptors,
        "reference_a": reference_a,
        "reference_b": reference_b,
        "scale": angular_scale,
    }


def _evaluate_coordinates(
    raw_coordinates: Array,
    target_moments: Array,
    problem: dict[str, Any],
) -> dict[str, Any]:
    stages = evaluate_all_stages(
        jnp.asarray(raw_coordinates, dtype=problem["dtype"]),
        target_moments,
        problem["backend"],
    )
    stages["projected"].block_until_ready()
    return stages


def _summarize_method(
    raw: Array,
    repaired: Array,
    target_moments: Array,
    problem: dict[str, Any],
    references: dict[str, Array],
    *,
    seed: int,
    num_bootstrap: int,
    uq_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    outputs: dict[str, Any] = {}
    saved: dict[str, np.ndarray] = {}
    for stage_offset, (stage_name, coordinates) in enumerate(
        (("raw", raw), ("repaired", repaired))
    ):
        values = stage_arrays(
            jnp.asarray(coordinates),
            target_moments,
            problem["box"],
            problem["basis"],
            problem["backend"].moment_scales,
            problem["backend"].physical,
            float(problem["flow_config"]["evaluation"]["overlap_threshold"]),
            problem["angular_orders"],
            problem["angular_neighbor_scale"],
            references["reference_a"],
            references["reference_b"],
            references["scale"],
            float(problem["flow_config"]["angular"]["far_threshold"]),
        )
        summary = summarize_stage(
            values,
            references["validation"],
            references["scale"],
            bootstrap_seed=seed + 100 * stage_offset,
            num_bootstrap=num_bootstrap,
        )
        uq = higher_order_conditional_uq(
            np.asarray(values["angular_descriptors"]),
            np.asarray(references["validation"]),
            np.asarray(values["labels"]),
            interval_levels=uq_config["interval_levels"],
            seed=seed + 1000 + 100 * stage_offset,
            num_resamples=num_bootstrap,
            confidence=float(uq_config["confidence"]),
        )
        outputs[stage_name] = {
            "metrics": summary,
            "higher_order_conditional_uq": uq,
        }
        saved[f"{stage_name}.coordinates"] = np.asarray(coordinates)
        saved[f"{stage_name}.labels"] = np.asarray(values["labels"])
        saved[f"{stage_name}.angular_descriptors"] = np.asarray(
            values["angular_descriptors"]
        )
    correction = correction_arrays(
        jnp.asarray(repaired),
        jnp.asarray(raw),
        problem["box"],
    )
    outputs["repair_correction"] = {
        **problem["bootstrap_function"](
            np.asarray(correction),
            seed=seed + 3000,
            num_resamples=num_bootstrap,
        ).as_dict(),
        "per_ensemble": np.asarray(correction),
    }
    return outputs, saved


def _write_summary_csv(path: Path, report: dict[str, Any]) -> None:
    fields = [
        "method",
        "information_budget",
        "stage",
        "pair_error",
        "energy",
        "overlap_fraction",
        "mode_a_fraction",
        "mode_b_fraction",
        "far_fraction",
        "angular_mmd2",
        "higher_order_energy_score",
        "mode_probability_tv",
        "repair_correction_rms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, result in report["methods"].items():
            correction = result["results"]["repair_correction"]["estimate"]
            for stage in ("raw", "repaired"):
                metrics = result["results"][stage]["metrics"]
                uq = result["results"][stage]["higher_order_conditional_uq"]
                writer.writerow(
                    {
                        "method": method,
                        "information_budget": result["information_budget"],
                        "stage": stage,
                        "pair_error": metrics["pair_error"]["estimate"],
                        "energy": metrics["energy"]["estimate"],
                        "overlap_fraction": metrics["overlap_fraction"]["estimate"],
                        "mode_a_fraction": metrics["mode_a_fraction"]["estimate"],
                        "mode_b_fraction": metrics["mode_b_fraction"]["estimate"],
                        "far_fraction": metrics["far_fraction"]["estimate"],
                        "angular_mmd2": metrics["angular_mmd2"],
                        "higher_order_energy_score": uq["multivariate_energy_score"],
                        "mode_probability_tv": uq[
                            "mode_probability_total_variation"
                        ],
                        "repair_correction_rms": correction,
                    }
                )


def run_scientific_comparison(
    config_path: str | Path,
    output_directory: str | Path,
    *,
    rerun_flow: bool = False,
    seed_override: int | None = None,
) -> dict[str, Any]:
    """Run the classical/learned comparison and higher-order conditional UQ."""
    config_path = Path(config_path)
    config = load_yaml(config_path)
    root = config_path.resolve().parents[1]
    flow_config_path = root / config["flow_ablation_config"]
    flow_config = load_yaml(flow_config_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    if seed_override is not None:
        flow_config = {**flow_config, "seed": int(seed_override)}
        flow_config_path = output / "resolved_flow_config.yaml"
        import yaml

        flow_config_path.write_text(
            yaml.safe_dump(flow_config, sort_keys=False), encoding="utf-8"
        )
    flow_output = output / "flow_ablation"
    flow_report_path = flow_output / "homometric_ablation_report.json"
    flow_arrays_path = flow_output / "homometric_ablation_arrays.npz"
    reuse_path = config.get("reuse_flow_artifacts")
    if seed_override is not None:
        reuse_path = None
    if not rerun_flow and reuse_path:
        candidate = root / reuse_path
        if (candidate / "homometric_ablation_report.json").is_file():
            flow_output = candidate
            flow_report_path = candidate / "homometric_ablation_report.json"
            flow_arrays_path = candidate / "homometric_ablation_arrays.npz"
    if rerun_flow or not flow_report_path.is_file() or not flow_arrays_path.is_file():
        run_experiment(flow_config_path, flow_output)

    flow_report = json.loads(flow_report_path.read_text(encoding="utf-8"))
    flow_arrays = _load_npz(flow_arrays_path)
    problem = _build_problem(flow_config)
    problem["flow_config"] = flow_config
    from .statistics import bootstrap_mean_interval

    problem["bootstrap_function"] = bootstrap_mean_interval
    train_indices = np.asarray(flow_arrays["train_indices"], dtype=np.int32)
    validation_indices = np.asarray(flow_arrays["validation_indices"], dtype=np.int32)
    references = _reference_descriptors(problem, train_indices, validation_indices)

    source = np.asarray(flow_arrays["evaluation_source"], dtype=np.float64)
    num_ensembles = source.shape[0]
    target = np.broadcast_to(
        np.asarray(problem["dataset"]["common_pair_moments"]),
        (num_ensembles, problem["basis"].centers.shape[0]),
    )
    target_jax = jnp.asarray(target, dtype=problem["dtype"])
    train_reference = np.asarray(
        problem["dataset"]["coordinates"][jnp.asarray(train_indices)]
    )
    num_bootstrap = int(config["uq"]["bootstrap_resamples"])
    seed = int(flow_config["seed"])

    methods_coordinates: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "soft_cefm": (
            flow_arrays["base.initial.coordinates"],
            flow_arrays["post_hoc.projected.coordinates"],
        ),
        "full_e2e_cefm": (
            flow_arrays["full_e2e.initial.coordinates"],
            flow_arrays["full_e2e.projected.coordinates"],
        ),
    }
    diagnostics: dict[str, Any] = {
        "soft_cefm": {
            "source": "flow ablation Base/Post-hoc",
        },
        "full_e2e_cefm": {
            "source": "flow ablation Full-E2E",
        },
    }
    information_budgets = {
        "soft_cefm": (
            "microscopic A/B training configurations and shared pair condition; "
            "soft pair/physical penalties; no solver differentiation"
        ),
        "full_e2e_cefm": (
            "same microscopic training configurations and condition as soft CEFM; "
            "differentiates through relaxation and projection"
        ),
    }

    start = time.perf_counter()
    rmc_coordinates, rmc_diagnostics = run_reverse_monte_carlo(
        source,
        target,
        np.ones(target.shape[-1]),
        np.asarray(problem["box"]),
        np.asarray(problem["basis"].centers),
        np.asarray(problem["basis"].widths),
        RMCOptions(**config["classical_baselines"]["rmc"]),
        seed=seed + 200,
    )
    rmc_runtime = time.perf_counter() - start
    rmc_stages = _evaluate_coordinates(rmc_coordinates, target_jax, problem)
    methods_coordinates["reverse_monte_carlo"] = (
        rmc_coordinates,
        np.asarray(rmc_stages["projected"]),
    )
    diagnostics["reverse_monte_carlo"] = {
        **rmc_diagnostics,
        "runtime_seconds": rmc_runtime,
    }
    information_budgets["reverse_monte_carlo"] = rmc_diagnostics[
        "information_budget"
    ]

    start = time.perf_counter()
    ibi_coordinates, ibi_diagnostics = run_iterative_boltzmann_inversion(
        source,
        train_reference,
        np.asarray(problem["box"]),
        IBIOptions(**config["classical_baselines"]["ibi"]),
        seed=seed + 300,
    )
    ibi_runtime = time.perf_counter() - start
    ibi_stages = _evaluate_coordinates(ibi_coordinates, target_jax, problem)
    methods_coordinates["iterative_boltzmann_inversion"] = (
        ibi_coordinates,
        np.asarray(ibi_stages["projected"]),
    )
    diagnostics["iterative_boltzmann_inversion"] = {
        **ibi_diagnostics,
        "runtime_seconds": ibi_runtime,
    }
    information_budgets["iterative_boltzmann_inversion"] = ibi_diagnostics[
        "information_budget"
    ]

    methods: dict[str, Any] = {}
    output_arrays: dict[str, np.ndarray] = {
        "evaluation_source": source,
        "target_moments": target,
        "reference_angular_descriptors": np.asarray(references["validation"]),
    }
    per_method_correction: dict[str, np.ndarray] = {}
    per_method_pair: dict[str, np.ndarray] = {}
    per_method_mode_tv: dict[str, np.ndarray] = {}
    per_method_reference_distance: dict[str, np.ndarray] = {}
    for method_index, (method, (raw, repaired)) in enumerate(
        methods_coordinates.items()
    ):
        results, arrays = _summarize_method(
            jnp.asarray(raw, dtype=problem["dtype"]),
            jnp.asarray(repaired, dtype=problem["dtype"]),
            target_jax,
            problem,
            references,
            seed=seed + 10_000 * method_index,
            num_bootstrap=num_bootstrap,
            uq_config=config["uq"],
        )
        methods[method] = {
            "information_budget": information_budgets[method],
            "results": results,
            "diagnostics": diagnostics[method],
        }
        for name, array in arrays.items():
            output_arrays[f"{method}.{name}"] = array
        per_method_correction[method] = np.asarray(
            results["repair_correction"].pop("per_ensemble")
        )
        repaired_arrays = stage_arrays(
            jnp.asarray(repaired, dtype=problem["dtype"]),
            target_jax,
            problem["box"],
            problem["basis"],
            problem["backend"].moment_scales,
            problem["backend"].physical,
            float(flow_config["evaluation"]["overlap_threshold"]),
            problem["angular_orders"],
            problem["angular_neighbor_scale"],
            references["reference_a"],
            references["reference_b"],
            references["scale"],
            float(flow_config["angular"]["far_threshold"]),
        )
        per_method_pair[method] = np.asarray(repaired_arrays["pair_error"])
        per_method_mode_tv[method] = np.abs(
            np.asarray(repaired_arrays["mode_b_fraction"]) - 0.5
        )
        per_method_reference_distance[method] = np.asarray(
            repaired_arrays["mean_reference_distance"]
        )

    full = "full_e2e_cefm"
    soft = "soft_cefm"
    primary = {
        "full_minus_soft_repair_correction": paired_bootstrap_difference(
            per_method_correction[full],
            per_method_correction[soft],
            seed=seed + 50_000,
            num_resamples=num_bootstrap,
        ).as_dict(),
        "full_minus_soft_repaired_pair_error": paired_bootstrap_difference(
            per_method_pair[full],
            per_method_pair[soft],
            seed=seed + 50_001,
            num_resamples=num_bootstrap,
        ).as_dict(),
        "full_minus_soft_mode_tv": paired_bootstrap_difference(
            per_method_mode_tv[full],
            per_method_mode_tv[soft],
            seed=seed + 50_002,
            num_resamples=num_bootstrap,
        ).as_dict(),
        "full_minus_soft_reference_distance": paired_bootstrap_difference(
            per_method_reference_distance[full],
            per_method_reference_distance[soft],
            seed=seed + 50_003,
            num_resamples=num_bootstrap,
        ).as_dict(),
    }
    rules = config["decision_rules"]
    decision = {
        "repair_superiority": primary[
            "full_minus_soft_repair_correction"
        ]["upper"] < 0.0,
        "pair_noninferiority": primary[
            "full_minus_soft_repaired_pair_error"
        ]["upper"] <= float(rules["pair_error_noninferiority_margin"]),
        "mode_noninferiority": primary["full_minus_soft_mode_tv"]["upper"]
        <= float(rules["mode_tv_noninferiority_margin"]),
        "higher_order_noninferiority": primary[
            "full_minus_soft_reference_distance"
        ]["upper"] <= float(rules["reference_distance_noninferiority_margin"]),
        "claim_status": "smoke test only; use independently trained seed sweep",
    }
    decision["all_gates_pass"] = all(
        value for key, value in decision.items() if key != "claim_status"
    )

    report = {
        "schema_version": 2,
        "experiment": "homometric-classical-and-population-informed-comparison",
        "status": "single-seed smoke comparison",
        "higher_order_conditional_uq_included": True,
        "configuration": config,
        "flow_ablation_report": str(flow_report_path),
        "information_budget_warning": (
            "RMC uses the reduced RBF pair condition. IBI uses a richer radial "
            "pair histogram. Learned methods additionally use microscopic "
            "training configurations, so observation-only and population-informed "
            "methods are separate scientific comparison tracks."
        ),
        "methods": methods,
        "primary_learned_method_comparison": primary,
        "decision_summary": decision,
        "ablation_reference": {
            "report": flow_report,
            "role": "mechanistic solver-gradient ablation, not the entire baseline set",
        },
    }
    report_path = output / "scientific_comparison_report.json"
    report_path.write_text(
        json.dumps(_to_python(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_summary_csv(output / "scientific_comparison_summary.csv", _to_python(report))
    np.savez_compressed(output / "scientific_comparison_arrays.npz", **output_arrays)
    return _to_python(report)
