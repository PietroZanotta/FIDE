"""Experiment C Phase-2 feasibility/calibration continuation (version 6)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .benchmark_design import ROOT, _classification
from .feasibility import (
    calibrate_iprojection_instrumented,
    solve_common_hull_lp,
    solve_maximum_entropy_common_target,
    solve_maximum_minimum_weight_lp,
)
from .morphology_metrics import metric_rows, summarize_rows, weighted_metric_mean
from .observables import ShellDefinition, field_observables, fit_standardization
from .simulator import generate_initial_conditions, simulate


DEFAULT_CONFIG = ROOT / "configs" / "expC_grayscott_phase2_v6.yaml"
MORPHOLOGY_KEYS = (
    "minority_component_count", "euler_characteristic", "interface_length",
    "anisotropy", "heldout_spectrum_1",
)


def _json_default(value):
    if isinstance(value, (np.ndarray, jax.Array)):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path):
    return json.loads(path.read_text())


def _load_config(path: Path):
    return _read_json(path.resolve())


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preserve_source_manifest(config: dict, output: Path) -> dict:
    source = ROOT / config["source_results"]
    files = sorted(path for path in source.rglob("*") if path.is_file() and output not in path.parents)
    manifest = {str(path.relative_to(ROOT)): _file_hash(path) for path in files}
    _write_json(output / "preserved_phase2_failed_sha256.json", manifest)
    return manifest


def verify_source_manifest(manifest: dict) -> bool:
    return all((ROOT / relative).exists() and _file_hash(ROOT / relative) == digest
               for relative, digest in manifest.items())


def _strip_details(result: dict | None) -> dict:
    if result is None:
        return {}
    keys = (
        "success", "status", "message", "maximum_equality_residual",
        "endpoint_target_disagreement", "maximum_minimum_weight",
        "maximum_minimum_weight_fraction_of_uniform", "converged",
        "convergence_reason", "iterations", "initial_dual_objective",
        "final_dual_objective", "maximum_absolute_standardized_residual",
        "lambda_norm", "covariance_rank", "covariance_condition",
        "ess_fraction", "maximum_weight", "minimum_weight", "entropy_fraction",
        "residual_identity_maximum_difference", "numpy_repository_weight_maximum_difference",
    )
    return {key: result.get(key) for key in keys if key in result}


def diagnose_pair(
    minus: np.ndarray,
    plus: np.ndarray,
    *,
    center: np.ndarray,
    scale: np.ndarray,
    dimension: int,
    tolerance: float,
    max_iterations: int,
    minus_morphology: list[dict] | None = None,
    plus_morphology: list[dict] | None = None,
    morphology_scale: np.ndarray | None = None,
) -> tuple[dict, dict]:
    physical_minus = np.asarray(minus[:, :dimension], dtype=np.float64)
    physical_plus = np.asarray(plus[:, :dimension], dtype=np.float64)
    used_center, used_scale = center[:dimension], scale[:dimension]
    standardized_minus = (physical_minus - used_center) / used_scale
    standardized_plus = (physical_plus - used_center) / used_scale
    affine_roundtrip = max(
        float(np.max(np.abs(standardized_minus * used_scale + used_center - physical_minus))),
        float(np.max(np.abs(standardized_plus * used_scale + used_center - physical_plus))),
    )
    feasible = solve_common_hull_lp(standardized_minus, standardized_plus)
    central_lp = solve_maximum_minimum_weight_lp(standardized_minus, standardized_plus) if feasible["success"] else None
    entropy = solve_maximum_entropy_common_target(standardized_minus, standardized_plus) if feasible["success"] else None
    target, target_source = None, None
    minus_warm, plus_warm = None, None
    if entropy is not None and entropy["converged"]:
        target, target_source = np.asarray(entropy["target"]), "maximum_total_entropy"
        minus_warm, plus_warm = np.asarray(entropy["lambda"]), -np.asarray(entropy["lambda"])
    elif central_lp is not None and central_lp["success"]:
        target, target_source = np.asarray(central_lp["target"]), "maximum_minimum_weight_lp"
    minus_calibration = plus_calibration = None
    if target is not None:
        minus_calibration = calibrate_iprojection_instrumented(
            standardized_minus, target, initial_lambda=minus_warm,
            tolerance=tolerance, max_iterations=max_iterations,
        )
        plus_calibration = calibrate_iprojection_instrumented(
            standardized_plus, target, initial_lambda=plus_warm,
            tolerance=tolerance, max_iterations=max_iterations,
        )
    effect = None
    hidden_minus = hidden_plus = None
    if minus_calibration and plus_calibration and minus_morphology is not None:
        hidden_minus = weighted_metric_mean(minus_morphology, minus_calibration["weights"])
        hidden_plus = weighted_metric_mean(plus_morphology, plus_calibration["weights"])
        difference = np.asarray([hidden_plus[key] - hidden_minus[key] for key in MORPHOLOGY_KEYS])
        effect = float(np.linalg.norm(difference / morphology_scale) / np.sqrt(len(MORPHOLOGY_KEYS)))
    max_residual = (
        max(minus_calibration["maximum_absolute_standardized_residual"],
            plus_calibration["maximum_absolute_standardized_residual"])
        if minus_calibration else None
    )
    minimum_ess = (
        min(minus_calibration["ess_fraction"], plus_calibration["ess_fraction"])
        if minus_calibration else None
    )
    summary = {
        "observation_dimension": dimension,
        "lp_feasible": feasible["success"],
        "lp_maximum_equality_residual": feasible.get("maximum_equality_residual"),
        "lp_status": feasible["message"],
        "central_lp_success": central_lp["success"] if central_lp else False,
        "central_maximum_minimum_weight": central_lp.get("maximum_minimum_weight") if central_lp else None,
        "central_minimum_weight_uniform_fraction": central_lp.get("maximum_minimum_weight_fraction_of_uniform") if central_lp else None,
        "entropy_central_converged": entropy["converged"] if entropy else False,
        "entropy_equality_residual": entropy.get("maximum_equality_residual") if entropy else None,
        "target_source": target_source,
        "target_standardized": target,
        "target_physical": target * used_scale + used_center if target is not None else None,
        "standardization_affine_roundtrip_error": affine_roundtrip,
        "minus_calibration_converged": minus_calibration["converged"] if minus_calibration else False,
        "plus_calibration_converged": plus_calibration["converged"] if plus_calibration else False,
        "maximum_standardized_calibration_residual": max_residual,
        "minimum_ess_fraction": minimum_ess,
        "hidden_morphology_effect": effect,
        "maximum_lambda_norm": (
            max(minus_calibration["lambda_norm"], plus_calibration["lambda_norm"])
            if minus_calibration else None
        ),
    }
    details = {
        "standardization": {"center": used_center, "scale": used_scale, "affine_roundtrip_error": affine_roundtrip},
        "feasibility_lp": feasible, "centrality_lp": central_lp,
        "maximum_entropy_centrality": entropy, "target_source": target_source,
        "target_standardized": target,
        "target_physical": target * used_scale + used_center if target is not None else None,
        "minus_calibration": minus_calibration, "plus_calibration": plus_calibration,
        "minus_hidden_mean": hidden_minus, "plus_hidden_mean": hidden_plus,
        "hidden_morphology_effect": effect,
    }
    return summary, details


def _old_morphology_rows(source: Path, regime_ids: list[str]):
    grouped = {regime_id: [] for regime_id in regime_ids}
    all_rows = []
    with (source / "metrics" / "design_morphology_metrics.csv").open() as handle:
        for row in csv.DictReader(handle):
            converted = {key: float(row[key]) for key in MORPHOLOGY_KEYS}
            grouped[row["regime_id"]].append((int(row["sample_index"]), converted))
            all_rows.append(converted)
    ordered = {key: [row for _, row in sorted(values)] for key, values in grouped.items()}
    values = np.asarray([[row[key] for key in MORPHOLOGY_KEYS] for row in all_rows])
    scale = np.maximum(values.std(axis=0, ddof=1), 1e-12)
    return ordered, scale


def run_existing_bank(config_path: Path = DEFAULT_CONFIG) -> dict:
    config = _load_config(config_path)
    source = ROOT / config["source_results"]
    output = ROOT / config["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output / "phase2_v6_config.yaml")
    manifest = preserve_source_manifest(config, output)
    bank = np.load(source / "design_banks.npz")
    ids = list(map(str, bank["regime_ids"]))
    features = np.asarray(bank["features"], dtype=np.float64)
    center, scale = fit_standardization(features.reshape(-1, features.shape[-1]))
    old_rows = list(csv.DictReader((source / "endpoint_calibration_candidates.csv").open()))
    named = list(config["named_pairs"])
    eligible = [
        row for row in old_rows
        if float(row["calibrated_morphology_effect_size"]) >= 1.0 and row["pair_id"] not in named
    ]
    eligible.sort(key=lambda row: (
        float(row["calibration_residual_standardized"]), -float(row["minimum_ess_fraction"])
    ))
    selected_pairs = named + [row["pair_id"] for row in eligible[:int(config["additional_pair_count"])]]
    morphology, morphology_scale = _old_morphology_rows(source, ids)
    rows, details = [], {}
    calibration = config["calibration"]
    for pair_id in selected_pairs:
        minus_id, plus_id = pair_id.split("__")
        summary, detail = diagnose_pair(
            features[ids.index(minus_id)], features[ids.index(plus_id)], center=center, scale=scale,
            dimension=4, tolerance=float(calibration["residual_tolerance"]),
            max_iterations=int(calibration["maximum_iterations"]),
            minus_morphology=morphology[minus_id], plus_morphology=morphology[plus_id],
            morphology_scale=morphology_scale,
        )
        summary.update({"pair_id": pair_id, "minus_regime": minus_id, "plus_regime": plus_id})
        rows.append(summary)
        details[pair_id] = detail
    _write_csv(output / "existing_bank_phi4_feasibility.csv", rows)
    _write_json(output / "existing_bank_phi4_diagnostics.json", {
        "source_bank": str((source / "design_banks.npz").relative_to(ROOT)),
        "bank_size_per_endpoint": int(features.shape[1]), "selected_pair_rule": config["additional_pair_rule"],
        "selected_pairs": selected_pairs, "feature_center": center, "feature_scale": scale,
        "rows": rows, "details": details,
        "source_artifacts_unchanged_after_run": verify_source_manifest(manifest),
    })
    return {"rows": rows, "any_feasible": any(row["lp_feasible"] for row in rows)}


def _large_regimes(config: dict) -> list[dict]:
    regimes = []
    for spec in config["large_bank"]["feeds_and_kill_ranges"]:
        count = int(round((spec["kill_stop"] - spec["kill_start"]) / spec["kill_step"])) + 1
        for kill in np.linspace(spec["kill_start"], spec["kill_stop"], count):
            regimes.append({
                "id": f"v6_F{int(round(10000 * spec['feed'])):04d}_k{int(round(100000 * kill)):05d}",
                "feed": float(spec["feed"]), "kill": float(kill),
            })
    return regimes


def run_large_bank(config_path: Path = DEFAULT_CONFIG) -> dict:
    config = _load_config(config_path)
    design_config = _read_json(ROOT / config["source_design_config"])
    source = ROOT / config["source_results"]
    output = ROOT / config["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(output / "preserved_phase2_failed_sha256.json") if (output / "preserved_phase2_failed_sha256.json").exists() else preserve_source_manifest(config, output)
    large = config["large_bank"]
    count = int(large["initial_condition_count"])
    seeds = np.arange(int(large["initial_condition_seed_start"]), int(large["initial_condition_seed_start"]) + count)
    old_roles = _read_json(source / "run_metadata.json")["seed_roles"]
    used_old = set(old_roles["design_initial_conditions"]) | set(old_roles["training_model"]) | set(old_roles["final_evaluation_bank"])
    if used_old & set(map(int, seeds)):
        raise ValueError("large-bank design seeds overlap an existing seed role")
    grid, ic, simulator = design_config["grid"], design_config["initial_conditions"], design_config["simulator"]
    initial_u, initial_v, _ = generate_initial_conditions(
        seeds, height=int(grid["height"]), width=int(grid["width"]),
        blob_count=tuple(ic["blob_count"]), radius_range=tuple(ic["radius_range"]),
        u_depletion_range=tuple(ic["u_depletion_range"]), v_amplitude_range=tuple(ic["v_amplitude_range"]),
        noise_std=float(ic["noise_std"]),
    )
    regimes = _large_regimes(config)
    regime_count = len(regimes)
    tiled_u = np.tile(initial_u[None], (regime_count, 1, 1, 1, 1)).reshape((-1,) + initial_u.shape[1:])
    tiled_v = np.tile(initial_v[None], (regime_count, 1, 1, 1, 1)).reshape((-1,) + initial_v.shape[1:])
    _, final_v = simulate(
        tiled_u, tiled_v, feed=np.repeat([row["feed"] for row in regimes], count),
        kill=np.repeat([row["kill"] for row in regimes], count),
        diffusion_u=float(simulator["diffusion_u"]), diffusion_v=float(simulator["diffusion_v"]),
        dt=float(simulator["dt"]), physical_time=float(simulator["physical_time"]),
        spacing=float(grid["spacing"]),
    )
    final_v = np.asarray(final_v).reshape((regime_count, count) + initial_v.shape[1:])
    threshold = float(_read_json(source / "design_scan_summary.json")["global_threshold"])
    obs = design_config["observables"]
    shells = ShellDefinition(tuple(obs["shell_centers_cycles_per_pixel"]), tuple(obs["shell_widths_cycles_per_pixel"]))
    features = np.asarray(field_observables(jnp.asarray(final_v), shells, tuple(obs["components"])), dtype=np.float64)
    morphology, summaries, all_morphology = [], [], []
    for index, regime in enumerate(regimes):
        rows = metric_rows(final_v[index], threshold)
        morphology.append(rows)
        all_morphology.extend(rows)
        summary = {**regime, **summarize_rows(rows)}
        summary["within_regime_diversity"] = float(np.mean(np.std(final_v[index, :, 0], axis=0)))
        summary["pattern_presence_fraction"] = float(np.mean(np.std(final_v[index, :, 0], axis=(-2, -1)) >= 0.02))
        summaries.append(summary)
    convergence_count = int(simulator["convergence_samples"])
    conv_u = np.tile(initial_u[:convergence_count][None], (regime_count, 1, 1, 1, 1)).reshape((-1,) + initial_u.shape[1:])
    conv_v = np.tile(initial_v[:convergence_count][None], (regime_count, 1, 1, 1, 1)).reshape((-1,) + initial_v.shape[1:])
    _, fine_v = simulate(
        conv_u, conv_v, feed=np.repeat([row["feed"] for row in regimes], convergence_count),
        kill=np.repeat([row["kill"] for row in regimes], convergence_count),
        diffusion_u=float(simulator["diffusion_u"]), diffusion_v=float(simulator["diffusion_v"]),
        dt=float(simulator["convergence_dt"]), physical_time=float(simulator["physical_time"]),
        spacing=float(grid["spacing"]),
    )
    fine_v = np.asarray(fine_v).reshape((regime_count, convergence_count) + initial_v.shape[1:])
    coarse_v = final_v[:, :convergence_count]
    classification = design_config["classification"]
    for index, summary in enumerate(summaries):
        rmse = np.sqrt(np.mean((coarse_v[index] - fine_v[index]) ** 2, axis=(1, 2, 3)))
        rms = np.maximum(np.sqrt(np.mean(fine_v[index] ** 2, axis=(1, 2, 3))), 1e-12)
        relative = rmse / rms
        summary["mean_timestep_relative_rmse"] = float(np.mean(relative))
        summary["worst_timestep_relative_rmse"] = float(np.max(relative))
        failures = []
        if summary["pattern_presence_fraction"] < float(classification["minimum_pattern_presence_fraction"]):
            failures.append("pattern_presence")
        if summary["mean_timestep_relative_rmse"] > float(classification["maximum_mean_timestep_relative_rmse"]):
            failures.append("mean_timestep_convergence")
        if summary["worst_timestep_relative_rmse"] > float(classification["maximum_worst_timestep_relative_rmse"]):
            failures.append("worst_timestep_convergence")
        summary["design_regime_gate_pass"] = not failures
        summary["design_regime_failure_reason"] = ";".join(failures)
    _classification(summaries, design_config)
    center, scale = fit_standardization(features.reshape(-1, features.shape[-1]))
    morphology_values = np.asarray([[row[key] for key in MORPHOLOGY_KEYS] for row in all_morphology])
    morphology_scale = np.maximum(morphology_values.std(axis=0, ddof=1), 1e-12)
    spots = [i for i, row in enumerate(summaries) if row["empirical_class"] == "spot_like"]
    labyrinths = [i for i, row in enumerate(summaries) if row["empirical_class"] == "labyrinth_like"]
    gates, calibration = config["gates"], config["calibration"]
    result_rows, passing, feasible_counts = [], [], {2: 0, 3: 0, 4: 0}
    trace_directory = output / "large_bank_calibration_traces"
    trace_directory.mkdir(exist_ok=True)
    for dimension in map(int, large["observation_dimensions"]):
        for minus_index in spots:
            for plus_index in labyrinths:
                pair_id = f"{regimes[minus_index]['id']}__{regimes[plus_index]['id']}"
                summary, detail = diagnose_pair(
                    features[minus_index], features[plus_index], center=center, scale=scale,
                    dimension=dimension, tolerance=float(calibration["residual_tolerance"]),
                    max_iterations=int(calibration["maximum_iterations"]),
                    minus_morphology=morphology[minus_index], plus_morphology=morphology[plus_index],
                    morphology_scale=morphology_scale,
                )
                summary.update({
                    "pair_id": pair_id, "spot_regime": regimes[minus_index]["id"],
                    "labyrinth_regime": regimes[plus_index]["id"],
                    "spot_feed": regimes[minus_index]["feed"], "spot_kill": regimes[minus_index]["kill"],
                    "labyrinth_feed": regimes[plus_index]["feed"], "labyrinth_kill": regimes[plus_index]["kill"],
                })
                if summary["lp_feasible"]:
                    feasible_counts[dimension] += 1
                    _write_json(trace_directory / f"phi{dimension}_{pair_id}.json", detail)
                reasons = []
                if not summary["lp_feasible"]:
                    reasons.append("convex_hulls_disjoint")
                elif not summary["minus_calibration_converged"] or not summary["plus_calibration_converged"]:
                    reasons.append("calibration_not_converged")
                if summary["maximum_standardized_calibration_residual"] is None or summary["maximum_standardized_calibration_residual"] > float(gates["maximum_standardized_residual"]):
                    reasons.append("calibration_residual")
                if summary["minimum_ess_fraction"] is None or summary["minimum_ess_fraction"] < float(gates["minimum_ess_fraction"]):
                    reasons.append("endpoint_ess")
                if summary["hidden_morphology_effect"] is None or summary["hidden_morphology_effect"] < float(gates["minimum_hidden_morphology_effect"]):
                    reasons.append("hidden_morphology_effect")
                summary["endpoint_gate_pass"] = not reasons
                summary["rejection_reasons"] = ";".join(reasons)
                if summary["endpoint_gate_pass"]:
                    summary["method_blind_passing_score"] = (
                        summary["hidden_morphology_effect"] + 0.5 * summary["minimum_ess_fraction"]
                        - 0.05 * summary["maximum_lambda_norm"]
                    )
                    passing.append(summary)
                result_rows.append(summary)
    passing.sort(key=lambda row: -row["method_blind_passing_score"])
    np.savez_compressed(
        output / "large_design_banks.npz", endpoint_v=final_v, features=features,
        regime_ids=np.asarray([row["id"] for row in regimes]), seeds=seeds,
        feature_center=center, feature_scale=scale, threshold=np.asarray(threshold),
    )
    _write_csv(output / "large_bank_regimes.csv", summaries)
    _write_csv(output / "large_bank_nested_phi_results.csv", result_rows)
    result = {
        "status": "phase_2_pass" if passing else "phase_2_failed",
        "is_final_benchmark_selection": False,
        "bank_size_per_regime": count, "regime_count": regime_count,
        "evaluated_pair_dimension_cells": len(result_rows),
        "feasible_pair_counts": feasible_counts, "passing_candidate_count": len(passing),
        "passing_candidates_ranked": passing,
        "provisional_phase2_candidate": passing[0] if passing else None,
        "standardization": {"center": center, "scale": scale, "same_for_target_and_calibration": True},
        "fixed_threshold": threshold, "source_artifacts_unchanged_after_run": verify_source_manifest(manifest),
    }
    _write_json(output / "large_bank_phase2_summary.json", result)
    return result


def run_zero_start_check(config_path: Path = DEFAULT_CONFIG) -> dict:
    """Recalibrate the ranked Phase-2 target from zero, without entropy warm starts."""
    config = _load_config(config_path)
    output = ROOT / config["output_directory"]
    phase2 = _read_json(output / "large_bank_phase2_summary.json")
    candidate = phase2["provisional_phase2_candidate"]
    bank = np.load(output / "large_design_banks.npz")
    ids = list(map(str, bank["regime_ids"]))
    features = np.asarray(bank["features"], dtype=np.float64)
    dimension = int(candidate["observation_dimension"])
    center = np.asarray(phase2["standardization"]["center"][:dimension])
    scale = np.asarray(phase2["standardization"]["scale"][:dimension])
    target = np.asarray(candidate["target_standardized"])
    minus = (features[ids.index(candidate["spot_regime"]), :, :dimension] - center) / scale
    plus = (features[ids.index(candidate["labyrinth_regime"]), :, :dimension] - center) / scale
    calibration = config["calibration"]
    result = {
        "candidate": candidate,
        "initial_lambda": "zero",
        "minus": calibrate_iprojection_instrumented(
            minus, target, tolerance=float(calibration["residual_tolerance"]),
            max_iterations=int(calibration["maximum_iterations"]),
        ),
        "plus": calibrate_iprojection_instrumented(
            plus, target, tolerance=float(calibration["residual_tolerance"]),
            max_iterations=int(calibration["maximum_iterations"]),
        ),
    }
    _write_json(output / "selected_candidate_zero_start_calibration.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("existing", "large-bank", "zero-start", "all"), nargs="?", default="all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if args.command in ("existing", "all"):
        print(json.dumps({"existing": run_existing_bank(args.config)}, indent=2, default=_json_default))
    if args.command in ("large-bank", "all"):
        print(json.dumps({"large_bank": run_large_bank(args.config)}, indent=2, default=_json_default))
    if args.command in ("zero-start", "all"):
        print(json.dumps({"zero_start": run_zero_start_check(args.config)}, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
