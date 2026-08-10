"""Final global spectral reference-velocity attempt for Gray--Scott (v9)."""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .benchmark_design import ROOT
from .field_transport import (
    init_spectral_reference_model,
    maximal_same_index_coupling,
    spectral_reference_model,
)
from .observables import ShellDefinition, field_observables
from .phase2_continuation import _write_csv, _write_json
from .phase3_reference_design import _read_json
from .phase3_reference_quality import (
    _fm_summary,
    _stratified_times,
    _train_reference_variant,
    directory_manifest,
    evaluate_fm_bank,
    evaluate_rollout_against_direct_si,
    sample_frozen_bridge,
    weighted_mmd2_four_weight,
)


DEFAULT_CONFIG = ROOT / "configs" / "expC_grayscott_phase3_global_v9.json"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_sources(config: dict):
    v7 = ROOT / config["source_phase3_v7_directory"]
    v8 = ROOT / config["source_phase3_v8_directory"]
    with (ROOT / config["control_checkpoint"]).open("rb") as handle:
        control = pickle.load(handle)
    training = np.load(v7 / "reference_training_endpoint_banks.npz")
    validation = np.load(v8 / "healthy_validation_endpoint_bank.npz")
    return control, training, validation


def _runtime_v8_config(config: dict, rollout_seed: int) -> dict:
    evaluation = config["paired_confirmatory_evaluation"]
    gates = config["reference_quality_gates"]
    return {
        "source_phase2_directory": config["source_phase2_directory"],
        "source_design_config": config["source_design_config"],
        "frozen_phase3a": config["frozen_phase3a"],
        "fm_diagnostics": {"time_grid": evaluation["time_grid"]},
        "rollout_diagnostics": {
            "diagnostic_seed": int(rollout_seed),
            "particle_count": int(evaluation["rollout_particle_count"]),
            "mmd_particle_count_raw": int(evaluation["raw_mmd_particle_count"]),
            "mmd_particle_count_downsampled": int(evaluation["downsampled_mmd_particle_count"]),
        },
        "reference_quality_gates": {
            "serious_field_minimum": gates["serious_field_minimum"],
            "serious_field_maximum": gates["serious_field_maximum"],
            "maximum_serious_range_fraction": gates["maximum_serious_range_fraction"],
        },
    }


def prepare_v9(config_path: Path = DEFAULT_CONFIG) -> dict:
    config_path = config_path.resolve(); config = _read_json(config_path)
    output = ROOT / config["output_directory"]
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("v9 output directory is not empty; append-only prepare refuses overwrite")
    output.mkdir(parents=True, exist_ok=True)
    control, training, validation = _load_sources(config)
    architecture_config = config["architecture"]
    parameters, architecture = init_spectral_reference_model(
        jax.random.PRNGKey(int(config["training"]["initialization_seed"])),
        width=int(architecture_config["width"]), blocks=int(architecture_config["blocks"]),
        modes=int(architecture_config["modes"]),
        time_frequencies=int(architecture_config["time_frequencies"]), dtype=jnp.float32,
    )
    target_equal = bool(np.array_equal(np.asarray(control["target"]), np.asarray(validation["target"])))
    center_equal = bool(np.array_equal(np.asarray(control["center"]), np.asarray(validation["center"])))
    scale_equal = bool(np.array_equal(np.asarray(control["scale"]), np.asarray(validation["scale"])))
    protocol = {
        "config": config,
        "config_sha256": _hash_file(config_path),
        "serialized_before_v9_heldout_rollout_evaluation": True,
        "architecture_runtime": architecture,
        "initial_parameter_count": int(sum(
            np.asarray(value).size for value in jax.tree_util.tree_leaves(parameters)
        )),
        "frozen_geometry_checks": {
            "target_exactly_equal": target_equal,
            "center_exactly_equal": center_equal,
            "scale_exactly_equal": scale_equal,
            "training_bank_size_per_endpoint": int(len(training["training_minus"])),
            "validation_bank_size_per_endpoint": int(len(validation["minus"])),
            "validation_minimum_ess_fraction": float(min(
                1.0 / (len(validation["minus_weights"]) * np.sum(validation["minus_weights"] ** 2)),
                1.0 / (len(validation["plus_weights"]) * np.sum(validation["plus_weights"] ** 2)),
            )),
        },
    }
    if not all((target_equal, center_equal, scale_equal)):
        raise RuntimeError("v9 source target/standardization differs from the frozen v8 control")
    manifests = {
        "phase2_v6": directory_manifest(ROOT / config["source_phase2_directory"]),
        "phase3_v7": directory_manifest(ROOT / config["source_phase3_v7_directory"]),
        "phase3_v8": directory_manifest(ROOT / config["source_phase3_v8_directory"]),
    }
    _write_json(output / "v9_predeclared_protocol.json", protocol)
    _write_json(output / "preserved_source_manifests_before_v9.json", manifests)
    return {"status": "v9_protocol_frozen", **protocol["frozen_geometry_checks"],
            "parameter_count": protocol["initial_parameter_count"]}


def train_spectral(config_path: Path = DEFAULT_CONFIG) -> dict:
    config = _read_json(config_path.resolve()); output = ROOT / config["output_directory"]
    if not (output / "v9_predeclared_protocol.json").exists():
        raise RuntimeError("run prepare before training")
    control, training_bank, validation_bank = _load_sources(config)
    train_minus = np.asarray(training_bank["training_minus"])
    train_plus = np.asarray(training_bank["training_plus"])
    train_coupling = maximal_same_index_coupling(
        np.asarray(training_bank["training_minus_weights"]),
        np.asarray(training_bank["training_plus_weights"]),
    )
    validation_minus = np.asarray(validation_bank["minus"])
    validation_plus = np.asarray(validation_bank["plus"])
    validation_coupling = maximal_same_index_coupling(
        np.asarray(validation_bank["minus_weights"]), np.asarray(validation_bank["plus_weights"])
    )
    training = config["training"]; architecture_config = config["architecture"]
    initial, architecture = init_spectral_reference_model(
        jax.random.PRNGKey(int(training["initialization_seed"])),
        width=int(architecture_config["width"]), blocks=int(architecture_config["blocks"]),
        modes=int(architecture_config["modes"]),
        time_frequencies=int(architecture_config["time_frequencies"]), dtype=jnp.float32,
    )
    rng = np.random.default_rng(int(training["validation_interpolant_seed"]))
    validation_count = int(training["fixed_validation_interpolant_count"])
    validation_times = _stratified_times(rng, validation_count)
    validation_states, validation_targets, _ = sample_frozen_bridge(
        rng, validation_minus, validation_plus, validation_coupling, validation_times,
        float(config["frozen_phase3a"]["schedule_amplitude"]),
    )
    training_protocol = {
        "batch_size": int(training["batch_size"]),
        "gradient_clip": float(training["gradient_clip"]),
        "weight_decay": float(training["weight_decay"]),
        "evaluation_interval": int(training["evaluation_interval"]),
        "fixed_training_interpolant_seed": int(training["training_interpolant_seed"]),
    }
    variant = {
        "id": "D_global_spectral_fm", "training_steps": int(training["maximum_steps"]),
        "learning_rate_start": float(training["learning_rate_start"]),
        "learning_rate_end": float(training["learning_rate_end"]),
    }
    best, trace, seconds = _train_reference_variant(
        variant=variant, initial_trainable=initial, model_kind="spectral_global",
        architecture=architecture, train_minus=train_minus, train_plus=train_plus,
        train_coupling=train_coupling, validation_times=validation_times,
        validation_states=validation_states, validation_targets=validation_targets,
        amplitude=float(config["frozen_phase3a"]["schedule_amplitude"]),
        protocol=training_protocol, output=output,
    )
    checkpoint = {
        "trainable": best, "architecture": architecture, "model_kind": "spectral_global",
        "variant_id": "D_global_spectral_fm", "training_protocol": training,
        "candidate": control["candidate"], "selected_path": control["selected_path"],
        "center": control["center"], "scale": control["scale"], "target": control["target"],
    }
    checkpoint_path = output / "checkpoint_D_global_spectral_fm.pkl"
    with checkpoint_path.open("wb") as handle:
        pickle.dump(checkpoint, handle)
    _write_csv(output / "training_trace_D_global_spectral_fm.csv", trace)
    best_row = min(trace, key=lambda row: row["validation_normalized_fm_mse"])
    result = {
        "status": "spectral_fm_training_complete", "training_seconds": seconds,
        "maximum_steps": int(training["maximum_steps"]),
        "best_checkpoint_step": int(best_row["step"]),
        "best_validation_normalized_fm_mse": best_row["validation_normalized_fm_mse"],
        "best_validation_mse_per_pixel": best_row["validation_mse_per_pixel"],
        "parameter_count": int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(best))),
        "checkpoint_sha256": _hash_file(checkpoint_path),
    }
    _write_json(output / "spectral_training_summary.json", result)
    return result


def _band_power(fields: np.ndarray) -> np.ndarray:
    values = np.asarray(fields)[:, 0]
    spectrum = np.abs(np.fft.fft2(values, axes=(-2, -1), norm="ortho")) ** 2
    height, width = values.shape[-2:]
    fy, fx = np.fft.fftfreq(height), np.fft.fftfreq(width)
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    bands = ((0.0, 0.125), (0.125, 0.25), (0.25, np.inf))
    return np.stack([spectrum[:, (radius >= low) & (radius < high)].mean(axis=1)
                     for low, high in bands], axis=1)


def _heun_interval(apply, initial: np.ndarray, start: float, stop: float, steps_per_unit: int):
    count = max(1, int(round((stop - start) * steps_per_unit)))
    dt = (stop - start) / count
    state = jnp.asarray(initial)
    for index in range(count):
        time_value = start + index * dt
        first = apply(time_value, state)
        proposal = state + dt * first
        second = apply(time_value + dt, proposal)
        state = state + 0.5 * dt * (first + second)
    return np.asarray(state)


def short_horizon_diagnostics(checkpoint: dict, validation, config: dict) -> dict:
    from .phase3_reference_quality import _checkpoint_apply
    evaluation = config["paired_confirmatory_evaluation"]
    rng = np.random.default_rng(int(evaluation["short_horizon_seed"]))
    minus, plus = np.asarray(validation["minus"]), np.asarray(validation["plus"])
    coupling = maximal_same_index_coupling(
        np.asarray(validation["minus_weights"]), np.asarray(validation["plus_weights"])
    )
    count = int(evaluation["short_horizon_particles"])
    center, scale = np.asarray(checkpoint["center"]), np.asarray(checkpoint["scale"])
    design = _read_json(ROOT / config["source_design_config"])["observables"]
    shells = ShellDefinition(tuple(design["shell_centers_cycles_per_pixel"]),
                             tuple(design["shell_widths_cycles_per_pixel"]))
    apply = _checkpoint_apply(checkpoint)
    rows = []
    for start in evaluation["short_horizon_starts"]:
        for horizon in evaluation["short_horizons"]:
            stop = float(start + horizon)
            if stop > 1.0 + 1e-12:
                continue
            initial, _, _ = sample_frozen_bridge(
                rng, minus, plus, coupling, np.full(count, start, np.float32),
                float(config["frozen_phase3a"]["schedule_amplitude"]),
            )
            direct, _, _ = sample_frozen_bridge(
                rng, minus, plus, coupling, np.full(count, stop, np.float32),
                float(config["frozen_phase3a"]["schedule_amplitude"]),
            )
            learned = _heun_interval(
                apply, initial, float(start), stop, int(evaluation["short_horizon_steps_per_unit"])
            )
            learned_phi = np.asarray(field_observables(
                jnp.asarray(learned), shells, ("mean", "second_moment")
            )).mean(0)
            direct_phi = np.asarray(field_observables(
                jnp.asarray(direct), shells, ("mean", "second_moment")
            )).mean(0)
            learned_power, direct_power = _band_power(learned), _band_power(direct)
            uniform = np.full(count, 1.0 / count)
            rows.append({
                "start": float(start), "horizon": float(horizon), "stop": stop,
                "maximum_standardized_phi_error": float(np.max(np.abs(
                    (learned_phi - direct_phi) / scale
                ))),
                "low_frequency_power_relative_mean_error": float(abs(
                    learned_power[:, 0].mean() - direct_power[:, 0].mean()
                ) / max(direct_power[:, 0].mean(), 1e-30)),
                "low_frequency_power_mmd2": weighted_mmd2_four_weight(
                    learned_power[:, :1], uniform, direct_power[:, :1], uniform
                ),
                "raw_field_mmd2": weighted_mmd2_four_weight(
                    learned.reshape(count, -1), uniform, direct.reshape(count, -1), uniform
                ),
                "learned_field_minimum": float(learned.min()),
                "learned_field_maximum": float(learned.max()),
            })
    return {"rows": rows, "maximum_standardized_phi_error": max(
        row["maximum_standardized_phi_error"] for row in rows
    ), "mean_low_frequency_power_mmd2": float(np.mean([
        row["low_frequency_power_mmd2"] for row in rows
    ]))}


def _flatten_fm(model_id: str, result: dict) -> list[dict]:
    rows = []
    for row in result["per_time"]:
        output = {"model": model_id, "t": row["t"]}
        for key in ("fm_mse_per_pixel", "zero_predictor_mse_per_pixel", "normalized_fm_mse",
                    "target_velocity_rms", "predicted_velocity_rms", "cosine_alignment"):
            output[key] = row[key]
        for index, name in enumerate(("low", "middle", "high")):
            output[f"{name}_frequency_error_energy"] = row["spatial_frequency_diagnostics"][index]["error_energy"]
            output[f"{name}_frequency_target_energy"] = row["spatial_frequency_diagnostics"][index]["target_energy"]
            output[f"{name}_frequency_error_fraction"] = row["spatial_frequency_diagnostics"][index]["error_fraction_of_target"]
        rows.append(output)
    return rows


def _flatten_rollout(model_id: str, rollout: dict) -> list[dict]:
    rows = []
    for row in rollout["rows"]:
        rows.append({
            "model": model_id, "t": row["t"],
            "learned_mean": row["learned"]["phi_physical"][0],
            "direct_mean": row["direct_si"]["phi_physical"][0],
            "learned_m2": row["learned"]["phi_physical"][1],
            "direct_m2": row["direct_si"]["phi_physical"][1],
            "maximum_standardized_phi_error": row["maximum_learned_minus_direct_standardized_phi"],
            "phi_threshold": row["thresholds"]["maximum_standardized_phi_error"],
            "raw_field_mmd2": row["raw_field_mmd2"],
            "raw_mmd_threshold": row["thresholds"]["raw_field_mmd2"],
            "downsampled_field_mmd2": row["downsampled_field_mmd2"],
            "downsampled_mmd_threshold": row["thresholds"]["downsampled_field_mmd2"],
            "phi_gate_pass": row["phi_fidelity_gate_pass"],
            "raw_mmd_gate_pass": row["raw_mmd_gate_pass"],
            "downsampled_mmd_gate_pass": row["downsampled_mmd_gate_pass"],
            "learned_smooth_tv": row["learned"]["smooth_tv"],
            "direct_smooth_tv": row["direct_si"]["smooth_tv"],
            "learned_anisotropy": row["learned"]["anisotropy"],
            "direct_anisotropy": row["direct_si"]["anisotropy"],
            "learned_soft_area": row["learned"]["soft_area"],
            "direct_soft_area": row["direct_si"]["soft_area"],
            "learned_soft_perimeter": row["learned"]["soft_perimeter"],
            "direct_soft_perimeter": row["direct_si"]["soft_perimeter"],
            "learned_radial_power_low": row["learned"]["radial_power_low"],
            "direct_radial_power_low": row["direct_si"]["radial_power_low"],
            "learned_radial_power_middle": row["learned"]["radial_power_middle"],
            "direct_radial_power_middle": row["direct_si"]["radial_power_middle"],
            "learned_radial_power_high": row["learned"]["radial_power_high"],
            "direct_radial_power_high": row["direct_si"]["radial_power_high"],
            "projected_phi_residual_max": float(np.max(np.abs(row["projected_phi_minus_c"]))),
            "projected_ess_fraction": row["projected_ess_fraction"],
            "learned_field_minimum": row["learned"]["field_minimum"],
            "learned_field_maximum": row["learned"]["field_maximum"],
        })
    return rows


def _plot_comparison(output: Path, fm_rows: list[dict], rollout_rows: list[dict]):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for model in sorted({row["model"] for row in fm_rows}):
        subset = [row for row in fm_rows if row["model"] == model]
        axes[0].plot([r["t"] for r in subset], [r["low_frequency_error_fraction"] for r in subset], label=model)
        axes[1].plot([r["t"] for r in subset], [r["normalized_fm_mse"] for r in subset], label=model)
    for model in sorted({row["model"] for row in rollout_rows}):
        subset = [row for row in rollout_rows if row["model"] == model]
        axes[2].plot([r["t"] for r in subset], [r["maximum_standardized_phi_error"] for r in subset], label=model)
        axes[2].plot([r["t"] for r in subset], [r["phi_threshold"] for r in subset], color="black", alpha=0.15)
    axes[0].set_title("Low-frequency FM error / target"); axes[1].set_title("Normalized FM error")
    axes[2].set_title("Rollout Phi error and frozen gate")
    for axis in axes:
        axis.set_xlabel("t"); axis.grid(alpha=0.25)
    axes[0].legend(); fig.tight_layout()
    fig.savefig(output / "paired_frequency_and_rollout_diagnostics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    for axis, band in zip(axes, ("low", "middle", "high")):
        for model in sorted({row["model"] for row in fm_rows}):
            subset = [row for row in fm_rows if row["model"] == model]
            axis.plot([r["t"] for r in subset],
                      [r[f"{band}_frequency_error_fraction"] for r in subset], label=model)
        axis.set_title(f"{band.capitalize()} radial band")
        axis.set_xlabel("t"); axis.set_ylabel("FM error / target energy"); axis.grid(alpha=0.25)
    axes[0].legend(); fig.tight_layout()
    fig.savefig(output / "paired_fm_radial_bands_by_time.png", dpi=180)
    plt.close(fig)


def evaluate_paired(config_path: Path = DEFAULT_CONFIG) -> dict:
    config = _read_json(config_path.resolve()); output = ROOT / config["output_directory"]
    with (ROOT / config["control_checkpoint"]).open("rb") as handle:
        control = pickle.load(handle)
    with (output / "checkpoint_D_global_spectral_fm.pkl").open("rb") as handle:
        spectral = pickle.load(handle)
    _, _, validation = _load_sources(config)
    minus, plus = np.asarray(validation["minus"]), np.asarray(validation["plus"])
    mw, pw = np.asarray(validation["minus_weights"]), np.asarray(validation["plus_weights"])
    coupling = maximal_same_index_coupling(mw, pw)
    evaluation = config["paired_confirmatory_evaluation"]
    thresholds = _read_json(ROOT / config["threshold_source"])["reference_split_thresholds"]
    runtime = _runtime_v8_config(config, int(evaluation["rollout_seed"]))
    results = {}
    fm_rows, rollout_rows = [], []
    for model_id, checkpoint in (("v8_control_C", control), ("v9_global_spectral", spectral)):
        fm = evaluate_fm_bank(
            checkpoint, minus, plus, coupling,
            sample_count=int(evaluation["fm_sample_count"]),
            fixed_time_count=int(evaluation["fm_fixed_time_count"]),
            time_grid=evaluation["time_grid"], seed=int(evaluation["fm_seed"]),
            amplitude=float(config["frozen_phase3a"]["schedule_amplitude"]),
        )
        rollout = evaluate_rollout_against_direct_si(
            checkpoint, minus, plus, mw, pw, runtime,
            steps=int(evaluation["primary_heun_steps"]), thresholds=thresholds,
        )
        short = short_horizon_diagnostics(checkpoint, validation, config)
        failed = [row["t"] for row in rollout["rows"] if not (
            row["phi_fidelity_gate_pass"] and row["raw_mmd_gate_pass"]
            and row["downsampled_mmd_gate_pass"]
        )]
        results[model_id] = {"fm": fm, "rollout": rollout, "short_horizon": short,
                             "failed_time_points": failed}
        fm_rows.extend(_flatten_fm(model_id, fm)); rollout_rows.extend(_flatten_rollout(model_id, rollout))
    _write_csv(output / "paired_fm_by_time.csv", fm_rows)
    _write_csv(output / "paired_rollout_by_time.csv", rollout_rows)
    _write_csv(output / "paired_short_horizon.csv", [
        {"model": model, **row} for model, result in results.items()
        for row in result["short_horizon"]["rows"]
    ])
    _plot_comparison(output, fm_rows, rollout_rows)
    summary_rows = []
    for model, result in results.items():
        fm, rollout = result["fm"]["overall"], result["rollout"]
        summary_rows.append({
            "model": model,
            "parameter_count": int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(
                control["trainable"] if model == "v8_control_C" else spectral["trainable"]
            ))),
            "normalized_fm_mse": fm["normalized_fm_mse"], "fm_mse_per_pixel": fm["fm_mse_per_pixel"],
            "cosine_alignment": fm["cosine_alignment"],
            "predicted_velocity_rms": fm["predicted_velocity_rms"], "target_velocity_rms": fm["target_velocity_rms"],
            "low_frequency_error_fraction": fm["spatial_frequency_diagnostics"][0]["error_fraction_of_target"],
            "middle_frequency_error_fraction": fm["spatial_frequency_diagnostics"][1]["error_fraction_of_target"],
            "high_frequency_error_fraction": fm["spatial_frequency_diagnostics"][2]["error_fraction_of_target"],
            "maximum_rollout_standardized_phi_error": rollout["maximum_learned_minus_direct_standardized_phi"],
            "endpoint_rollout_standardized_phi_error": rollout["endpoint_maximum_standardized_phi_error"],
            "integrated_raw_field_mmd2": rollout["integrated_raw_field_mmd2"],
            "integrated_downsampled_field_mmd2": rollout["integrated_downsampled_field_mmd2"],
            "failed_time_point_count": len(result["failed_time_points"]),
            "field_pathology_fraction": rollout["serious_field_range_fraction"],
            "local_gate_pass": fm["normalized_fm_mse"] <= config["reference_quality_gates"]["local_normalized_fm_mse"],
            "rollout_gate_pass": rollout["rollout_fidelity_gate_pass"],
        })
    _write_csv(output / "paired_model_summary.csv", summary_rows)
    result = {"status": "paired_v9_evaluation_complete", "models": results,
              "summary": summary_rows, "paired_draws": True,
              "thresholds_reused_exactly_from_v8": True}
    _write_json(output / "paired_evaluation.json", result)
    return {"status": result["status"], "summary": summary_rows,
            "failed_times": {key: value["failed_time_points"] for key, value in results.items()}}


def finalize_v9(config_path: Path = DEFAULT_CONFIG) -> dict:
    config_path = config_path.resolve(); config = _read_json(config_path)
    output = ROOT / config["output_directory"]
    paired = _read_json(output / "paired_evaluation.json")
    by_name = {row["model"]: row for row in paired["summary"]}
    control_row, spectral_row = by_name["v8_control_C"], by_name["v9_global_spectral"]
    spectral_detail = paired["models"]["v9_global_spectral"]
    rescue = config["optional_single_adaptation"]["permitted_only_if_all_conditions_hold"]
    rollout_rows = spectral_detail["rollout"]["rows"]
    maximum_exceedance = max(
        row["maximum_learned_minus_direct_standardized_phi"]
        / max(row["thresholds"]["maximum_standardized_phi_error"], 1e-30)
        for row in rollout_rows
    )
    all_mmd_pass = bool(all(
        row["raw_mmd_gate_pass"] and row["downsampled_mmd_gate_pass"]
        for row in rollout_rows
    ))
    relative_phi_improvement = (
        control_row["maximum_rollout_standardized_phi_error"]
        - spectral_row["maximum_rollout_standardized_phi_error"]
    ) / control_row["maximum_rollout_standardized_phi_error"]
    relative_down_improvement = (
        control_row["integrated_downsampled_field_mmd2"]
        - spectral_row["integrated_downsampled_field_mmd2"]
    ) / control_row["integrated_downsampled_field_mmd2"]
    conditions = {
        "local_fm_comfortably_passes": spectral_row["normalized_fm_mse"] <= rescue["local_normalized_fm_mse_maximum"],
        "failed_time_points_are_near_miss": spectral_row["failed_time_point_count"] <= rescue["maximum_failed_time_points"],
        "phi_exceedance_is_near_miss": maximum_exceedance <= rescue["maximum_phi_threshold_exceedance_ratio"],
        "all_mmd_gates_pass": all_mmd_pass,
        "max_phi_clearly_improves_over_v8": relative_phi_improvement >= rescue["minimum_relative_improvement_over_v8_in_max_phi"],
        "downsampled_mmd_clearly_improves_over_v8": relative_down_improvement >= rescue["minimum_relative_improvement_over_v8_in_integrated_downsampled_mmd2"],
        "field_pathology_gate_pass": spectral_row["field_pathology_fraction"] <= config["reference_quality_gates"]["maximum_serious_range_fraction"],
    }
    adaptation_permitted = bool(all(conditions.values()))
    adaptation = {
        "permitted": adaptation_permitted, "performed": False,
        "conditions": conditions, "maximum_phi_threshold_exceedance_ratio": maximum_exceedance,
        "relative_max_phi_improvement_over_v8": relative_phi_improvement,
        "relative_integrated_downsampled_mmd_improvement_over_v8": relative_down_improvement,
        "reason": (
            "all predeclared narrow-failure conditions hold"
            if adaptation_permitted else
            "standard spectral FM is not a genuine improving near miss under the predeclared trigger"
        ),
    }
    _write_json(output / "optional_adaptation_decision.json", adaptation)
    if adaptation_permitted:
        raise RuntimeError("predeclared rescue trigger unexpectedly passed; adaptation implementation is required")

    with (ROOT / config["control_checkpoint"]).open("rb") as handle:
        control = pickle.load(handle)
    with (output / "checkpoint_D_global_spectral_fm.pkl").open("rb") as handle:
        spectral = pickle.load(handle)
    _, _, validation = _load_sources(config)
    minus, plus = np.asarray(validation["minus"]), np.asarray(validation["plus"])
    mw, pw = np.asarray(validation["minus_weights"]), np.asarray(validation["plus_weights"])
    evaluation = config["paired_confirmatory_evaluation"]
    common_times = np.linspace(0.0, 1.0, 17).tolist()
    ode = {}
    for model_name, checkpoint in (("v8_control_C", control), ("v9_global_spectral", spectral)):
        model_results = {}
        for steps in evaluation["ode_resolutions"]:
            runtime = _runtime_v8_config(config, int(evaluation["rollout_seed"]) + 1000)
            model_results[str(steps)] = evaluate_rollout_against_direct_si(
                checkpoint, minus, plus, mw, pw, runtime, steps=int(steps),
                thresholds=None, time_grid=common_times,
            )
        m128 = model_results["128"]["integrated_downsampled_field_mmd2"]
        m256 = model_results["256"]["integrated_downsampled_field_mmd2"]
        ode[model_name] = {
            "resolutions": model_results,
            "relative_downsampled_mmd_improvement_128_to_256": (m128 - m256) / max(m128, 1e-30),
        }
    _write_json(output / "paired_ode_resolution.json", ode)
    _write_csv(output / "paired_ode_resolution.csv", [
        {
            "model": model, "heun_steps": int(steps),
            "integrated_raw_field_mmd2": value["integrated_raw_field_mmd2"],
            "integrated_downsampled_field_mmd2": value["integrated_downsampled_field_mmd2"],
            "maximum_standardized_phi_error": value["maximum_learned_minus_direct_standardized_phi"],
            "endpoint_standardized_phi_error": value["endpoint_maximum_standardized_phi_error"],
            "endpoint_raw_field_mmd2": value["endpoint_raw_field_mmd2"],
        }
        for model, result in ode.items() for steps, value in result["resolutions"].items()
    ])

    standard_pass = bool(
        spectral_row["local_gate_pass"] and spectral_row["rollout_gate_pass"]
        and spectral_row["field_pathology_fraction"] <= config["reference_quality_gates"]["maximum_serious_range_fraction"]
    )
    phase3b_pass = standard_pass
    final_state = config["hard_stop"]["pass_state"] if phase3b_pass else config["hard_stop"]["failure_state"]
    decision = {
        "status": final_state, "phase3b_pass": phase3b_pass,
        "standard_spectral_model_pass": standard_pass,
        "optional_adaptation_permitted": adaptation_permitted,
        "optional_adaptation_performed": False,
        "gray_scott_reference_failed_after_v9": not phase3b_pass,
        "gray_scott_headline_benchmark_parked": not phase3b_pass,
        "phase4_authorized": phase3b_pass, "phase4_performed": False,
        "benchmark_selection_created": False,
        "deep_ritz_training_performed": False, "mfsi_training_performed": False,
        "tangent_training_or_comparison_performed": False,
        "final_learned_method_comparison_performed": False,
        "scientific_obstacle": (
            None if phase3b_pass else
            "reference-flow realization: local/global conditional-velocity FM improvements do not control accumulated raw-SI rollout error"
        ),
        "endpoint_fiber_feasibility_remains_passed": True,
        "intermediate_iprojection_overlap_remains_passed": True,
        "paired_summary": paired["summary"], "adaptation_decision": adaptation,
        "ode_convergence": {
            model: value["relative_downsampled_mmd_improvement_128_to_256"]
            for model, value in ode.items()
        },
    }
    _write_json(output / "phase3b_v9_final_decision.json", decision)

    before = _read_json(output / "preserved_source_manifests_before_v9.json")
    after = {
        "phase2_v6": directory_manifest(ROOT / config["source_phase2_directory"]),
        "phase3_v7": directory_manifest(ROOT / config["source_phase3_v7_directory"]),
        "phase3_v8": directory_manifest(ROOT / config["source_phase3_v8_directory"]),
    }
    preservation = {name: before[name] == after[name] for name in before}
    if not all(preservation.values()):
        raise RuntimeError(f"source artifact preservation failure: {preservation}")
    _write_json(output / "source_preservation_after_v9.json", preservation)

    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                capture_output=True, text=True).stdout.strip()
    git_status = subprocess.run(["git", "status", "--short"], cwd=ROOT, check=True,
                                capture_output=True, text=True).stdout.splitlines()
    metadata = {
        "experiment": "C", "version": config["version"], "final_state": final_state,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit, "git_status": git_status,
        "python": sys.version, "platform": platform.platform(),
        "jax_version": jax.__version__, "numpy_version": np.__version__,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "devices": [str(device) for device in jax.devices()],
        "config_path": str(config_path.relative_to(ROOT)), "config_sha256": _hash_file(config_path),
        "architecture": config["architecture"], "training": config["training"],
        "evaluation": config["paired_confirmatory_evaluation"],
        "threshold_source": config["threshold_source"],
        "checkpoint_sha256": _hash_file(output / "checkpoint_D_global_spectral_fm.pkl"),
        "parameter_count": spectral_row["parameter_count"],
        "source_preservation": preservation,
    }
    _write_json(output / "run_metadata.json", metadata)
    artifact_hashes = {
        str(path.relative_to(ROOT)): _hash_file(path) for path in sorted(output.iterdir())
        if path.is_file() and path.name != "v9_artifact_sha256.json"
    }
    _write_json(output / "v9_artifact_sha256.json", artifact_hashes)
    return {"status": final_state, "phase3b_pass": phase3b_pass,
            "adaptation_performed": False, "source_preservation": preservation,
            "ode_convergence": decision["ode_convergence"]}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "train", "evaluate", "finalize"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if args.command == "prepare": result = prepare_v9(args.config)
    elif args.command == "train": result = train_spectral(args.config)
    elif args.command == "evaluate": result = evaluate_paired(args.config)
    else: result = finalize_v9(args.config)
    print(json.dumps(result, indent=2, default=lambda value: np.asarray(value).tolist()))


if __name__ == "__main__":
    main()
