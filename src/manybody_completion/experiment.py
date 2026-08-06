"""End-to-end homometric stochastic ablation and scientific report builder."""

from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
import platform
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .config import load_yaml
from .energy import PhysicalParameters
from .flow import SamplingOptions, sample_conditional_flow, sample_uniform_torus
from .geometry import periodic_rms_displacement
from .homometric import (
    build_homometric_dataset,
    motif_coordinates,
    validate_homometric_pair,
)
from .metrics import (
    correction_arrays,
    reference_angular_descriptors,
    stage_arrays,
    summarize_stage,
    transition_matrix,
)
from .network import FlowNetworkConfig, initialize_flow_network
from .observables import PairBasis, angular_cosine_moments
from .routing import AblationMode, ROUTES, evaluate_all_stages
from .solvers import (
    LocalJaxBackend,
    ProjectionOptions,
    RelaxationOptions,
)
from .statistics import bootstrap_mean_interval, paired_bootstrap_difference
from .training import (
    AdamOptions,
    FineTuneWeights,
    fine_tune_route,
    pretrain_flow,
    route_objective,
    tree_l2_norm,
)


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


def _stratified_split(
    labels: np.ndarray,
    validation_per_mode: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    for label in sorted(np.unique(labels)):
        members = np.flatnonzero(labels == label)
        if members.size <= validation_per_mode:
            raise ValueError("each mode needs more samples than validation_per_mode")
        shuffled = rng.permutation(members)
        validation_parts.append(np.sort(shuffled[:validation_per_mode]))
        train_parts.append(np.sort(shuffled[validation_per_mode:]))
    return (
        np.sort(np.concatenate(train_parts)),
        np.sort(np.concatenate(validation_parts)),
    )


def _tree_copy(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda value: jnp.array(value, copy=True), tree)


def _tree_distance(left: Any, right: Any) -> float:
    difference = jax.tree_util.tree_map(lambda a, b: a - b, left, right)
    squared = sum(
        jnp.sum(value * value) for value in jax.tree_util.tree_leaves(difference)
    )
    return float(jnp.sqrt(squared))


def _directional_gradient_check(
    parameters: Any,
    loss_function: Any,
    key: Array,
    epsilons: list[float],
) -> dict[str, Any]:
    leaves, structure = jax.tree_util.tree_flatten(parameters)
    keys = jax.random.split(key, len(leaves))
    direction_leaves = [
        jax.random.normal(key_value, leaf.shape, leaf.dtype)
        for key_value, leaf in zip(keys, leaves)
    ]
    norm = jnp.sqrt(sum(jnp.sum(value * value) for value in direction_leaves) + 1e-24)
    direction = jax.tree_util.tree_unflatten(
        structure, [value / norm for value in direction_leaves]
    )
    _, autodiff = jax.jvp(
        loss_function,
        (parameters,),
        (direction,),
    )

    def shifted(scale: float) -> Any:
        return jax.tree_util.tree_map(
            lambda parameter, delta: parameter + scale * delta,
            parameters,
            direction,
        )

    finite = []
    relative = []
    for epsilon in epsilons:
        estimate = (loss_function(shifted(epsilon)) - loss_function(shifted(-epsilon))) / (
            2.0 * epsilon
        )
        finite.append(float(estimate))
        relative.append(float(jnp.abs(estimate - autodiff) / jnp.maximum(jnp.abs(autodiff), 1e-12)))
    return {
        "autodiff": float(autodiff),
        "epsilons": epsilons,
        "finite_differences": finite,
        "relative_errors": relative,
        "best_relative_error": float(min(relative)),
    }


def _interval(values: Array, seed: int, num_bootstrap: int) -> dict[str, float]:
    return bootstrap_mean_interval(
        np.asarray(jax.device_get(values)),
        seed=seed,
        num_resamples=num_bootstrap,
    ).as_dict()


def _write_trace(path: Path, histories: dict[str, dict[str, np.ndarray]]) -> None:
    metric_names = sorted({name for history in histories.values() for name in history})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "mode", "step", *metric_names])
        writer.writeheader()
        for mode, history in histories.items():
            phase = "pretrain" if mode == "shared_pretrain" else "fine_tune"
            length = max(len(values) for values in history.values())
            for step in range(length):
                row: dict[str, Any] = {"phase": phase, "mode": mode, "step": step}
                for name, values in history.items():
                    row[name] = values[step]
                writer.writerow(row)


def _write_summary_csv(path: Path, report: dict[str, Any]) -> None:
    fields = [
        "mode",
        "serving_stage",
        "pair_error",
        "energy",
        "overlap_fraction",
        "total_correction_rms",
        "mode_a_fraction",
        "mode_b_fraction",
        "far_fraction",
        "mode_entropy",
        "angular_mmd2",
        "training_seconds",
        "sampling_seconds",
        "solver_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode, result in report["modes"].items():
            serving = result["serving_stage"]
            stage = result["stages"][serving]
            writer.writerow(
                {
                    "mode": mode,
                    "serving_stage": serving,
                    "pair_error": stage["pair_error"]["estimate"],
                    "energy": stage["energy"]["estimate"],
                    "overlap_fraction": stage["overlap_fraction"]["estimate"],
                    "total_correction_rms": result["corrections"]["total"]["estimate"],
                    "mode_a_fraction": stage["mode_a_fraction"]["estimate"],
                    "mode_b_fraction": stage["mode_b_fraction"]["estimate"],
                    "far_fraction": stage["far_fraction"]["estimate"],
                    "mode_entropy": stage["mode_entropy"]["estimate"],
                    "angular_mmd2": stage["angular_mmd2"],
                    "training_seconds": result["runtime_seconds"]["training"],
                    "sampling_seconds": result["runtime_seconds"]["sampling"],
                    "solver_seconds": result["runtime_seconds"]["solvers"],
                }
            )


def run_experiment(config_path: str | Path, output_directory: str | Path) -> dict[str, Any]:
    """Run one comparison-ready homometric stochastic ablation."""
    print("[experiment] loading configuration", flush=True)
    config = load_yaml(config_path)

    # Neural-network and solver training dtype.
    training_dtype = (
        jnp.float64 if config["dtype"] == "float64" else jnp.float32
    )

    # Enable float64 support for the tiny exact benchmark validation.
    # This does not force training arrays to become float64.
    jax.config.update("jax_enable_x64", True)

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])

    pair_config = config["pair_basis"]

    # Validate the exact homometric construction in float64.
    validation_dtype = jnp.float64
    validation_box = jnp.asarray(config["box"], dtype=validation_dtype)
    validation_basis = PairBasis.uniform(
        int(pair_config["num_basis"]),
        float(pair_config["r_min"]),
        float(pair_config["r_max"]),
        float(pair_config["width"]),
        dtype=validation_dtype,
    )
    validation_angular_orders = jnp.asarray(
        config["angular"]["orders"],
        dtype=validation_dtype,
    )
    angular_scale_parameter = float(
        config["angular"]["neighbor_scale"]
    )

    benchmark_check = validate_homometric_pair(
        validation_box,
        validation_basis,
        validation_angular_orders,
        angular_scale_parameter,
    )

    if (
        not benchmark_check["non_congruent"]
        or benchmark_check["pair_max_abs_error"] > 1e-10
    ):
        raise RuntimeError(
            f"homometric benchmark validation failed: {benchmark_check}"
        )

    # Construct all training objects explicitly in the requested training dtype.
    dtype = training_dtype
    box = jnp.asarray(config["box"], dtype=dtype)
    basis = PairBasis.uniform(
        int(pair_config["num_basis"]),
        float(pair_config["r_min"]),
        float(pair_config["r_max"]),
        float(pair_config["width"]),
        dtype=dtype,
    )
    angular_orders = jnp.asarray(
        config["angular"]["orders"],
        dtype=dtype,
    )

    # ``benchmark_check`` was already computed in float64 above. Do not repeat
    # exact homometric validation in the float32 training dtype: discrepancies
    # around 1e-7 are expected roundoff, not a benchmark failure.
    print("[experiment] building homometric dataset", flush=True)
    dataset = build_homometric_dataset(
        seed=seed,
        samples_per_mode=int(config["dataset"]["samples_per_mode"]),
        num_replicas=int(config["dataset"]["num_replicas"]),
        box=box,
        basis=basis,
        angular_orders=angular_orders,
        angular_neighbor_scale=angular_scale_parameter,
    )
    labels_np = np.asarray(dataset["labels"])
    train_indices, validation_indices = _stratified_split(
        labels_np,
        int(config["dataset"]["validation_per_mode"]),
        seed + 1,
    )
    train_targets = dataset["coordinates"][jnp.asarray(train_indices)]
    train_conditions = dataset["conditions"][jnp.asarray(train_indices)]
    train_moments = dataset["pair_moments"][jnp.asarray(train_indices)]

    network_config = FlowNetworkConfig(**config["network"])
    parameters = initialize_flow_network(
        jax.random.PRNGKey(seed + 2),
        condition_dim=basis.centers.shape[0],
        config=network_config,
        dtype=dtype,
    )
    pretrain_options = AdamOptions(**config["pretraining"])
    print("[experiment] shared flow pretraining", flush=True)
    start = time.perf_counter()
    pretrained = pretrain_flow(
        parameters,
        train_targets,
        train_conditions,
        box,
        network_config,
        pretrain_options,
        schedule_seed=seed + 3,
        random_seed=seed + 4,
    )
    pretraining_seconds = time.perf_counter() - start

    physical = PhysicalParameters(**config["physical"])
    backend = LocalJaxBackend(
        box=box,
        basis=basis,
        moment_scales=jnp.ones_like(dataset["common_pair_moments"]),
        physical=physical,
        relaxation_options=RelaxationOptions(**config["relaxation"]),
        projection_options=ProjectionOptions(**config["projection"]),
    )
    training_sampling = SamplingOptions(**config["training_sampling"])
    evaluation_sampling = SamplingOptions(**config["evaluation"]["sampling"])
    fine_options = AdamOptions(**config["fine_tuning"])
    weights = FineTuneWeights(**config["objective_weights"])

    trained_parameters: dict[AblationMode, Any] = {}
    histories: dict[str, dict[str, np.ndarray]] = {
        "shared_pretrain": pretrained.history
    }
    training_runtime: dict[str, float] = {}
    for mode in (AblationMode.BASE, AblationMode.RELAX_E2E, AblationMode.FULL_E2E):
        print(f"[experiment] fine-tuning {mode.value}", flush=True)
        start = time.perf_counter()
        result = fine_tune_route(
            _tree_copy(pretrained.parameters),
            mode,
            train_targets,
            train_conditions,
            train_moments,
            backend,
            network_config,
            training_sampling,
            fine_options,
            weights,
            schedule_seed=seed + 5,
            random_seed=seed + 6,
        )
        training_runtime[mode.value] = time.perf_counter() - start
        trained_parameters[mode] = result.parameters
        histories[mode.value] = result.history
    trained_parameters[AblationMode.POST_HOC] = _tree_copy(
        trained_parameters[AblationMode.BASE]
    )
    histories[AblationMode.POST_HOC.value] = histories[AblationMode.BASE.value]
    training_runtime[AblationMode.POST_HOC.value] = training_runtime[AblationMode.BASE.value]

    evaluation_config = config["evaluation"]
    num_ensembles = int(evaluation_config["num_ensembles"])
    num_replicas = int(config["dataset"]["num_replicas"])
    num_particles = dataset["coordinates"].shape[-2]
    evaluation_source = sample_uniform_torus(
        jax.random.PRNGKey(seed + 7),
        (num_ensembles, num_replicas, num_particles, 2),
        box,
        dtype,
    )
    evaluation_conditions = jnp.zeros(
        (num_ensembles, basis.centers.shape[0]), dtype=dtype
    )
    evaluation_moments = jnp.broadcast_to(
        dataset["common_pair_moments"],
        (num_ensembles, basis.centers.shape[0]),
    )

    train_coordinates = dataset["coordinates"][jnp.asarray(train_indices)]
    reference_flat = train_coordinates.reshape((-1, num_particles, 2))
    reference_descriptors = angular_cosine_moments(
        reference_flat, box, angular_orders, angular_scale_parameter
    )
    reference_labels = jnp.repeat(
        dataset["labels"][jnp.asarray(train_indices)], num_replicas
    )
    reference_a = jnp.mean(reference_descriptors[reference_labels == 0], axis=0)
    reference_b = jnp.mean(reference_descriptors[reference_labels == 1], axis=0)
    angular_scale = jnp.maximum(
        jnp.std(reference_descriptors, axis=0),
        jnp.asarray(float(config["angular"]["scale_floor"]), dtype),
    )
    balanced_reference = reference_descriptors
    far_threshold = float(config["angular"]["far_threshold"])
    num_bootstrap = int(evaluation_config["bootstrap_resamples"])

    mode_reports: dict[str, Any] = {}
    arrays_to_save: dict[str, np.ndarray] = {
        "evaluation_source": np.asarray(evaluation_source),
        "train_indices": train_indices,
        "validation_indices": validation_indices,
    }
    stage_arrays_by_mode: dict[str, dict[str, dict[str, Array]]] = {}
    correction_by_mode: dict[str, dict[str, Array]] = {}
    for mode_index, mode in enumerate(AblationMode):
        print(f"[experiment] evaluating {mode.value}", flush=True)
        parameters_for_mode = trained_parameters[mode]
        start = time.perf_counter()
        generated = sample_conditional_flow(
            parameters_for_mode,
            evaluation_source,
            evaluation_conditions,
            box,
            network_config,
            evaluation_sampling,
        )
        generated.block_until_ready()
        sampling_seconds = time.perf_counter() - start
        start = time.perf_counter()
        stages = evaluate_all_stages(generated, evaluation_moments, backend)
        stages["projected"].block_until_ready()
        solver_seconds = time.perf_counter() - start

        mode_stage_arrays: dict[str, dict[str, Array]] = {}
        stage_summaries: dict[str, Any] = {}
        for stage_index, stage_name in enumerate(("initial", "relaxed", "projected")):
            coordinates = stages[stage_name]
            values = stage_arrays(
                coordinates,
                evaluation_moments,
                box,
                basis,
                backend.moment_scales,
                physical,
                float(config["evaluation"]["overlap_threshold"]),
                angular_orders,
                angular_scale_parameter,
                reference_a,
                reference_b,
                angular_scale,
                far_threshold,
            )
            mode_stage_arrays[stage_name] = values
            stage_summaries[stage_name] = summarize_stage(
                values,
                balanced_reference,
                angular_scale,
                bootstrap_seed=seed + 1000 * mode_index + 50 * stage_index,
                num_bootstrap=num_bootstrap,
            )
            arrays_to_save[f"{mode.value}.{stage_name}.coordinates"] = np.asarray(coordinates)
            arrays_to_save[f"{mode.value}.{stage_name}.labels"] = np.asarray(values["labels"])

        relaxation_correction = correction_arrays(
            stages["relaxed"], stages["initial"], box
        )
        projection_correction = correction_arrays(
            stages["projected"], stages["relaxed"], box
        )
        total_correction = correction_arrays(
            stages["projected"], stages["initial"], box
        )
        correction_by_mode[mode.value] = {
            "relaxation": relaxation_correction,
            "projection": projection_correction,
            "total": total_correction,
        }
        correction_summary = {
            "relaxation": _interval(
                relaxation_correction, seed + 5000 + mode_index, num_bootstrap
            ),
            "projection": _interval(
                projection_correction, seed + 5100 + mode_index, num_bootstrap
            ),
            "total": _interval(
                total_correction, seed + 5200 + mode_index, num_bootstrap
            ),
        }
        transitions = {
            "initial_to_relaxed": np.asarray(
                transition_matrix(
                    mode_stage_arrays["initial"]["labels"],
                    mode_stage_arrays["relaxed"]["labels"],
                )
            ).tolist(),
            "relaxed_to_projected": np.asarray(
                transition_matrix(
                    mode_stage_arrays["relaxed"]["labels"],
                    mode_stage_arrays["projected"]["labels"],
                )
            ).tolist(),
            "initial_to_projected": np.asarray(
                transition_matrix(
                    mode_stage_arrays["initial"]["labels"],
                    mode_stage_arrays["projected"]["labels"],
                )
            ).tolist(),
        }
        relaxation_diagnostics = stages["relaxation"]
        projection_diagnostics = stages["projection"]
        mode_reports[mode.value] = {
            "training_stage": ROUTES[mode].training_stage,
            "serving_stage": ROUTES[mode].serving_stage,
            "stages": stage_summaries,
            "corrections": correction_summary,
            "mode_transitions": transitions,
            "solver_diagnostics": {
                "relaxation_convergence_rate": float(
                    jnp.mean(relaxation_diagnostics["converged"].astype(dtype))
                ),
                "projection_convergence_rate": float(
                    jnp.mean(projection_diagnostics["converged"].astype(dtype))
                ),
                "projection_rank_deficient_rate": float(
                    jnp.mean(projection_diagnostics["rank_deficient"].astype(dtype))
                ),
                "mean_projection_residual": float(
                    jnp.mean(projection_diagnostics["constraint_residual"])
                ),
            },
            "runtime_seconds": {
                "training": training_runtime[mode.value],
                "sampling": sampling_seconds,
                "solvers": solver_seconds,
            },
            "parameter_distance_from_shared_pretrain": _tree_distance(
                parameters_for_mode, pretrained.parameters
            ),
        }
        stage_arrays_by_mode[mode.value] = mode_stage_arrays

    post_total = correction_by_mode[AblationMode.POST_HOC.value]["total"]
    full_total = correction_by_mode[AblationMode.FULL_E2E.value]["total"]
    post_pair = stage_arrays_by_mode[AblationMode.POST_HOC.value]["projected"]["pair_error"]
    full_pair = stage_arrays_by_mode[AblationMode.FULL_E2E.value]["projected"]["pair_error"]
    post_mode_tv = jnp.abs(
        stage_arrays_by_mode[AblationMode.POST_HOC.value]["projected"]["mode_b_fraction"]
        - 0.5
    )
    full_mode_tv = jnp.abs(
        stage_arrays_by_mode[AblationMode.FULL_E2E.value]["projected"]["mode_b_fraction"]
        - 0.5
    )
    paired = {
        "full_minus_post_hoc_total_correction": paired_bootstrap_difference(
            np.asarray(full_total),
            np.asarray(post_total),
            seed=seed + 9000,
            num_resamples=num_bootstrap,
        ).as_dict(),
        "full_minus_post_hoc_projected_pair_error": paired_bootstrap_difference(
            np.asarray(full_pair),
            np.asarray(post_pair),
            seed=seed + 9001,
            num_resamples=num_bootstrap,
        ).as_dict(),
        "full_minus_post_hoc_mode_tv": paired_bootstrap_difference(
            np.asarray(full_mode_tv),
            np.asarray(post_mode_tv),
            seed=seed + 9002,
            num_resamples=num_bootstrap,
        ).as_dict(),
    }
    decision = config["decision_rules"]
    correction_superior = paired["full_minus_post_hoc_total_correction"]["upper"] < 0.0
    pair_noninferior = (
        paired["full_minus_post_hoc_projected_pair_error"]["upper"]
        <= float(decision["pair_error_noninferiority_margin"])
    )
    mode_noninferior = (
        paired["full_minus_post_hoc_mode_tv"]["upper"]
        <= float(decision["mode_tv_noninferiority_margin"])
    )

    gradient_config = config["gradient_check"]
    if bool(gradient_config.get("enabled", False)):
        print("[experiment] Full-E2E directional gradient check", flush=True)
        check_count = int(gradient_config["batch_size"])
        check_key = jax.random.PRNGKey(seed + 10)
        check_backend = replace(
            backend,
            relaxation_options=replace(
                backend.relaxation_options,
                num_steps=int(gradient_config.get("relaxation_steps", 1)),
            ),
            projection_options=replace(
                backend.projection_options,
                num_steps=int(gradient_config.get("projection_steps", 1)),
            ),
        )
        check_loss = lambda model: route_objective(
            model,
            AblationMode.FULL_E2E,
            train_targets[:check_count],
            train_conditions[:check_count],
            train_moments[:check_count],
            check_key,
            check_backend,
            network_config,
            SamplingOptions(**gradient_config["sampling"]),
            weights,
        )[0]
        gradient_check = {
            **_directional_gradient_check(
                trained_parameters[AblationMode.FULL_E2E],
                check_loss,
                jax.random.PRNGKey(seed + 11),
                [float(value) for value in gradient_config["epsilons"]],
            ),
            "status": "completed",
            "scope": "reduced-iteration probe of the same relaxation and projection algorithms",
            "relaxation_steps": check_backend.relaxation_options.num_steps,
            "projection_steps": check_backend.projection_options.num_steps,
        }
    else:
        gradient_check = {
            "status": "separate_acceptance_job",
            "reason": (
                "Disabled in the comparison worker to isolate high-memory "
                "derivative compilation from completed scientific results."
            ),
        }

    report = {
        "schema_version": 1,
        "experiment": "conditional-equivariant-flow-homometric-ablation",
        "status": "single-seed exploratory comparison",
        "configuration": config,
        "software": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "platform": platform.platform(),
        },
        "benchmark_validation": benchmark_check,
        "data": {
            "shape": list(dataset["coordinates"].shape),
            "train_indices": train_indices,
            "validation_indices": validation_indices,
            "pair_numerical_deviation": dataset["pair_numerical_deviation"],
            "unique_condition_rows": 1,
        },
        "runtime_seconds": {"shared_pretraining": pretraining_seconds},
        "modes": mode_reports,
        "paired_comparisons": paired,
        "gradient_check": gradient_check,
        "decision_summary": {
            "correction_superiority": correction_superior,
            "pair_error_noninferiority": pair_noninferior,
            "mode_balance_noninferiority": mode_noninferior,
            "all_numerical_endpoints_pass": correction_superior
            and pair_noninferior
            and mode_noninferior,
            "scientific_claim_ready": False,
            "reason": (
                "A single bounded smoke seed cannot establish a scientific claim; "
                "use the registered multi-seed protocol."
            ),
        },
        "parameter_checks": {
            "base_post_hoc_distance": _tree_distance(
                trained_parameters[AblationMode.BASE],
                trained_parameters[AblationMode.POST_HOC],
            )
        },
    }

    print("[experiment] writing artifacts", flush=True)
    report_path = output / "homometric_ablation_report.json"
    report_path.write_text(
        json.dumps(_to_python(report), indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_trace(output / "homometric_ablation_trace.csv", histories)
    _write_summary_csv(output / "homometric_ablation_summary.csv", _to_python(report))
    np.savez(output / "homometric_ablation_arrays.npz", **arrays_to_save)
    parameter_payload: dict[str, np.ndarray] = {}
    for mode, tree in trained_parameters.items():
        for index, leaf in enumerate(jax.tree_util.tree_leaves(tree)):
            parameter_payload[f"{mode.value}.leaf_{index:03d}"] = np.asarray(leaf)
    np.savez(
        output / "homometric_ablation_parameters.npz", **parameter_payload
    )
    return _to_python(report)