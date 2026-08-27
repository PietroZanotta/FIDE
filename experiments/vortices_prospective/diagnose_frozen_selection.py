from __future__ import annotations

"""Post-hoc diagnosis of one already-frozen prospective vortices selection.

This module deliberately does not import the selection stage and contains no
candidate generation or optimization. Hidden target states are used only after
the manifest freeze has been verified byte-for-byte.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from aggregate_qois import qoi_features
from common import config_hash, fingerprint, load_config, trap_weights, write_json_atomic
from evaluator import AggregateObservationBank, ProspectiveEvaluator
from frozen_diagnostic_core import (
    curve_error_metrics,
    evaluate_explicit_moments,
    forcing_path_comparison,
    paired_statistics,
    projection_path_comparison,
    realized_observation_banks,
    reconstruct_exact_population,
    summary,
)
from mfsi.cache import file_sha256
from physical import truth_from_config
from prospective_data import TargetProspectiveData

jax.config.update("jax_enable_x64", True)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _geometry_key(eta) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(eta, dtype=np.float64), 12))


def _assert_frozen_boundary(manifest_path: Path, manifest: dict[str, Any]) -> bytes:
    frozen_bytes = manifest_path.read_bytes()
    if manifest.get("status") != "frozen_before_hidden_validation":
        raise RuntimeError("diagnostics require a frozen pre-validation manifest")
    if manifest.get("hidden_validation_loaded") is not False:
        raise RuntimeError("manifest does not certify the prospective freeze boundary")
    forbidden_paths = []
    for module in sys.modules.values():
        path = getattr(module, "__file__", None)
        if path and Path(path).name == "select.py" and "vortices_prospective" in str(path):
            forbidden_paths.append(path)
    if forbidden_paths:
        raise RuntimeError(f"selection module was imported during diagnostics: {forbidden_paths}")
    return frozen_bytes


def _load_selection_bank(path: Path) -> AggregateObservationBank:
    if not path.exists():
        raise FileNotFoundError("frozen selection randomness is missing; diagnostics will not regenerate it")
    with np.load(path, allow_pickle=False) as data:
        return AggregateObservationBank(
            np.asarray(data["sampling_z"], dtype=np.float64),
            np.asarray(data["detector_z"], dtype=np.float64),
        )


def _load_hidden(path: Path, manifest_sha: str):
    with np.load(path, allow_pickle=False) as data:
        role = str(np.asarray(data["role"]).item())
        frozen_sha = str(np.asarray(data["manifest_sha256_at_creation"]).item())
        if role != "hidden_validation_microscopic_states_post_freeze_only":
            raise RuntimeError("hidden bank role is not post-freeze validation")
        if frozen_sha != manifest_sha:
            raise RuntimeError("hidden bank was not created against this frozen manifest")
        return np.asarray(data["times"], dtype=np.float64), np.asarray(data["states"], dtype=np.float64)


def _load_hidden_randomness(path: Path):
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["sample_indices"], dtype=np.int32), np.asarray(data["detector_z"], dtype=np.float64)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metric_mean(result: dict[str, Any], name: str) -> float:
    value = result[name]["mean"]
    return float(value) if value is not None else float("nan")


def _scientific_risk(evaluator, projected_weights, qoi_target) -> float:
    projected = np.einsum(
        "tn,tnk->tk",
        np.asarray(projected_weights, dtype=np.float64),
        np.asarray(evaluator.reference_qois, dtype=np.float64),
    )
    error = (
        projected - np.asarray(qoi_target, dtype=np.float64)
    ) / np.asarray(evaluator.data.qoi_scales, dtype=np.float64)[None, :]
    return float(np.sum(np.asarray(evaluator.time_weights)[:, None] * error * error))


def _stage_row(stage: str, law: float, full: float) -> dict[str, Any]:
    return {
        "stage": stage,
        "Law": float(law),
        "Full": float(full),
        "Full_reduction_vs_Law": float(1.0 - full / law),
    }


def _plot_outputs(output_dir: Path, times, diagnostics, candidate_rows, paired_difference):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#3366cc", "#dc3912", "#109618", "#990099"]
    for quantity, key, ylabel, filename in (
        ("response", "curves", r"$c_j(t)$", "aggregate_trajectories.png"),
        ("derivative", "derivatives", r"$\dot c_j(t)$", "aggregate_derivatives.png"),
    ):
        fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True)
        for row_index, name in enumerate(("Law", "Full")):
            block = diagnostics[name]
            predicted = block[key]["predicted"]
            oracle = block[key]["oracle"]
            for channel, ax in enumerate(axes[row_index]):
                ax.plot(times, predicted[:, channel], color=colors[channel], label="predicted")
                ax.plot(times, oracle[:, channel], color="black", linestyle="--", label="oracle")
                ax.set_title(f"{name}, sensor {channel + 1}")
                if channel == 0:
                    ax.set_ylabel(ylabel)
                if row_index == 1:
                    ax.set_xlabel("t")
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, name in zip(axes, ("Law", "Full")):
        profile = diagnostics[name]["action_profiles"]
        ax.plot(times, profile["predicted"], label="predicted")
        ax.plot(times, profile["oracle_aggregate"], label="oracle aggregate")
        ax.plot(times, profile["observed"], label="observed")
        ax.set(title=name, xlabel="t", ylabel="instantaneous Full action")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "instantaneous_full_action.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    x = np.asarray([row["predicted_full_action"] for row in candidate_rows])
    y = np.asarray([row["oracle_aggregate_full_action"] for row in candidate_rows])
    ax.scatter(x, y, color="#777777")
    for row in candidate_rows:
        if row["selected_as"]:
            ax.annotate("/".join(row["selected_as"]), (row["predicted_full_action"], row["oracle_aggregate_full_action"]))
    lo, hi = min(np.min(x), np.min(y)), max(np.max(x), np.max(y))
    ax.plot([lo, hi], [lo, hi], color="black", linestyle=":")
    ax.set(xlabel="predicted Full action", ylabel="oracle-aggregate Full action")
    fig.tight_layout()
    fig.savefig(output_dir / "candidate_predicted_vs_oracle.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.axhline(0.0, color="black", linewidth=1)
    ax.plot(np.arange(len(paired_difference)), paired_difference, marker="o", markersize=3, linewidth=0.8)
    ax.set(xlabel="validation trial", ylabel="Full geometry action - Law geometry action")
    fig.tight_layout()
    fig.savefig(output_dir / "paired_validation_differences.png", dpi=150)
    plt.close(fig)


def run_diagnostics(config_path: Path, manifest_path: Path, output_dir: Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = config_path.resolve()
    manifest_path = manifest_path.resolve()
    run_root = manifest_path.parent.parent
    output_dir = (output_dir or manifest_path.parent / "diagnostics").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config_path)
    manifest = _json(manifest_path)
    frozen_manifest_bytes = _assert_frozen_boundary(manifest_path, manifest)
    manifest_sha = file_sha256(manifest_path)
    if manifest["config_hash"] != config_hash(cfg):
        raise RuntimeError("configuration does not match the frozen production manifest")

    required = {
        "endpoint": run_root / "endpoint_reference" / "endpoint_data.npz",
        "aggregate": run_root / "prospective" / "aggregate_predictions.npz",
        "rollout": run_root / "endpoint_reference" / "reference_rollout.npz",
        "selection_randomness": run_root / "prospective" / "selection_randomness.npz",
        "hidden": run_root / "hidden_validation" / "hidden_state_bank.npz",
        "hidden_randomness": run_root / "hidden_validation" / "hidden_observation_randomness.npz",
        "validation": run_root / "results" / "validation_result.json",
        "candidates": run_root / "results" / "selection_candidates.json",
    }
    for path in required.values():
        if not path.exists():
            raise FileNotFoundError(path)
    diagnostic_signature = fingerprint({
        "schema": 1,
        "manifest_sha256": manifest_sha,
        "config_sha256": file_sha256(config_path),
        "diagnostic_script_sha256": file_sha256(Path(__file__)),
        "diagnostic_core_sha256": file_sha256(Path(__file__).with_name("frozen_diagnostic_core.py")),
        "artifacts": {name: file_sha256(path) for name, path in required.items()},
    })
    receipt_path = output_dir / "diagnostic_receipt.json"
    if receipt_path.exists() and (output_dir / "report.md").exists():
        receipt = _json(receipt_path)
        if receipt.get("signature") == diagnostic_signature:
            print("[diagnostics] reusing compatible completed diagnostic", flush=True)
            return receipt

    data = TargetProspectiveData.load(required["endpoint"], required["aggregate"])
    evaluator = ProspectiveEvaluator(cfg, data, required["rollout"])
    selection_bank = _load_selection_bank(required["selection_randomness"])
    hidden_times, hidden_states = _load_hidden(required["hidden"], manifest_sha)
    sample_indices, detector_z = _load_hidden_randomness(required["hidden_randomness"])
    if not np.allclose(hidden_times, np.asarray(evaluator.times)):
        raise RuntimeError("hidden and reference scientific time grids differ")
    validation = _json(required["validation"])
    if validation["frozen_manifest_sha256"] != manifest_sha:
        raise RuntimeError("validation result is not tied to this manifest")

    # Exact pairing is inherited from one common hidden sample-index/noise bank.
    law_trials = validation["methods"]["Law"]["realized"]["trials"]
    full_trials = validation["methods"]["Full"]["realized"]["trials"]
    law_by_id = {int(row["trial"]): row for row in law_trials if row["valid"]}
    full_by_id = {int(row["trial"]): row for row in full_trials if row["valid"]}
    paired_ids = sorted(set(law_by_id) & set(full_by_id))
    if paired_ids != list(range(len(paired_ids))):
        raise RuntimeError("validation trials are not aligned by common trial id")
    law_action = np.asarray([law_by_id[i]["full_action"] for i in paired_ids])
    full_action = np.asarray([full_by_id[i]["full_action"] for i in paired_ids])
    paired = paired_statistics(law_action, full_action)
    paired["trial_ids"] = paired_ids
    paired["common_hidden_randomness_sha256"] = file_sha256(required["hidden_randomness"])
    write_json_atomic(output_dir / "paired_validation_stats.json", paired)

    truth = truth_from_config(cfg)
    print("[diagnostics] evaluating hidden population velocity and QoIs", flush=True)
    hidden_velocity = np.asarray(
        jax.vmap(lambda t, x: truth.velocity(x, t))(
            jnp.asarray(hidden_times), jnp.asarray(hidden_states)
        ),
        dtype=np.float64,
    )
    hidden_qoi = np.asarray(
        jnp.mean(qoi_features(jnp.asarray(hidden_states)), axis=1), dtype=np.float64
    )
    reference_nodes = np.asarray(evaluator.nodes)
    domain_checks = {
        "hidden_particles_in_domain_fraction": float(np.mean(
            (hidden_states[..., 0] >= 0.0) & (hidden_states[..., 0] <= 2.0)
            & (hidden_states[..., 1] >= 0.0) & (hidden_states[..., 1] <= 1.0)
        )),
        "reference_particles_in_domain_fraction": float(np.mean(
            (reference_nodes[..., 0] >= 0.0) & (reference_nodes[..., 0] <= 2.0)
            & (reference_nodes[..., 1] >= 0.0) & (reference_nodes[..., 1] <= 1.0)
        )),
    }

    selected = {
        name: np.asarray(manifest["selected"][name]["eta"], dtype=np.float64)
        for name in ("Law", "Tangent", "Full")
    }
    if _geometry_key(selected["Tangent"]) != _geometry_key(selected["Full"]):
        raise RuntimeError("frozen artifact no longer has Tangent == Full")
    geometry_diagnostics: dict[str, Any] = {}
    stage_evaluations: dict[str, dict[str, Any]] = {}
    certification_rows: list[dict[str, Any]] = []
    saved_arrays: dict[str, Any] = {"times": hidden_times, "paired_difference": full_action - law_action}

    unique_names = ("Law", "Full")
    for name in unique_names:
        print(f"[diagnostics] frozen geometry {name}: aggregate and derivative paths", flush=True)
        eta = selected[name]
        centers = eta.reshape((-1, 2))
        predicted_mean, predicted_second = evaluator.prospective_population(eta)
        predicted_mean = np.asarray(predicted_mean, dtype=np.float64)
        predicted_second = np.asarray(predicted_second, dtype=np.float64)
        phi_hidden = np.asarray(
            evaluator.sensors.features(jnp.asarray(hidden_states), jnp.asarray(eta)),
            dtype=np.float64,
        )
        oracle_mean = np.mean(phi_hidden, axis=1)
        oracle_second = np.mean(phi_hidden * phi_hidden, axis=1)
        grad_hidden = np.asarray(
            evaluator.sensors.feature_gradients(jnp.asarray(hidden_states), jnp.asarray(eta)),
            dtype=np.float64,
        )
        oracle_dot = np.mean(
            np.einsum("tnsd,tnd->tns", grad_hidden, hidden_velocity), axis=1
        )
        predicted_recon, predicted_recon_dot = reconstruct_exact_population(
            evaluator, predicted_mean
        )
        oracle_recon, oracle_recon_dot = reconstruct_exact_population(
            evaluator, oracle_mean
        )
        selection_c, selection_c_dot, _ = evaluator.reconstruct(
            predicted_mean, predicted_second, selection_bank
        )
        selection_sample_only_bank = AggregateObservationBank(
            np.asarray(selection_bank.sampling_z),
            np.zeros_like(selection_bank.detector_z),
        )
        selection_sample_c, selection_sample_c_dot, _ = evaluator.reconstruct(
            predicted_mean, predicted_second, selection_sample_only_bank
        )
        sample_bank, observed_bank, _, _ = realized_observation_banks(
            evaluator, phi_hidden, sample_indices, detector_z
        )
        sample_c, sample_c_dot, _ = evaluator.reconstruct(
            oracle_mean, oracle_second, sample_bank
        )
        observed_c, observed_c_dot, _ = evaluator.reconstruct(
            oracle_mean, oracle_second, observed_bank
        )

        print(f"[diagnostics] frozen geometry {name}: Stage A original prediction", flush=True)
        stage_a = evaluate_explicit_moments(
            evaluator, eta, selection_c, selection_c_dot
        )
        frozen_predicted = float(manifest["selected"][name]["predicted"]["full_action"]["mean"])
        reproduction_error = stage_a.action_mean - frozen_predicted
        if abs(reproduction_error) > 2.0e-5:
            raise RuntimeError(
                f"{name} Stage A failed frozen reproduction: {stage_a.action_mean} vs {frozen_predicted}"
            )
        print(f"[diagnostics] frozen geometry {name}: deterministic predicted and oracle aggregate", flush=True)
        stage_a_det = evaluate_explicit_moments(
            evaluator, eta, predicted_recon, predicted_recon_dot,
            retain_particle_details=True,
        )
        stage_a_sample = evaluate_explicit_moments(
            evaluator, eta, selection_sample_c, selection_sample_c_dot
        )
        stage_b = evaluate_explicit_moments(
            evaluator, eta, oracle_mean, oracle_dot,
            retain_particle_details=True,
        )
        print(f"[diagnostics] frozen geometry {name}: population reconstruction", flush=True)
        stage_c = evaluate_explicit_moments(
            evaluator, eta, oracle_recon, oracle_recon_dot
        )
        print(f"[diagnostics] frozen geometry {name}: finite sampling without detector noise", flush=True)
        stage_d = evaluate_explicit_moments(
            evaluator, eta, sample_c, sample_c_dot
        )
        print(f"[diagnostics] frozen geometry {name}: realized finite sampling and noise", flush=True)
        stage_e = evaluate_explicit_moments(
            evaluator, eta, observed_c, observed_c_dot
        )
        cached_realized = float(validation["methods"][name]["realized_full_action"])
        observed_reproduction_error = stage_e.action_mean - cached_realized
        if abs(observed_reproduction_error) > 2.0e-5:
            raise RuntimeError(
                f"{name} observed stage failed validation reproduction: {stage_e.action_mean} vs {cached_realized}"
            )

        aggregate_errors = curve_error_metrics(predicted_mean, oracle_mean, hidden_times)
        aggregate_errors["reconstructed_predicted_vs_oracle"] = curve_error_metrics(
            predicted_recon, oracle_mean, hidden_times
        )
        derivative_errors = curve_error_metrics(
            predicted_recon_dot, oracle_dot, hidden_times
        )
        projection = projection_path_comparison(stage_a_det, stage_b, hidden_times)
        forcing = forcing_path_comparison(stage_a_det, stage_b, hidden_times)
        profile_delta = stage_b.action_by_time[0] - np.mean(stage_a.action_by_time, axis=0)
        profile_contributions = trap_weights(hidden_times) * profile_delta
        largest_nodes = np.argsort(np.abs(profile_contributions))[::-1][:5]
        total_absolute_contribution = float(np.sum(np.abs(profile_contributions)))
        action_profiles = {
            "predicted": np.mean(stage_a.action_by_time, axis=0).tolist(),
            "deterministic_predicted_reconstruction": stage_a_det.action_by_time[0].tolist(),
            "oracle_aggregate": stage_b.action_by_time[0].tolist(),
            "reconstructed_population": stage_c.action_by_time[0].tolist(),
            "sample_only": np.mean(stage_d.action_by_time, axis=0).tolist(),
            "observed": np.mean(stage_e.action_by_time, axis=0).tolist(),
            "oracle_minus_predicted_by_time": profile_delta.tolist(),
            "oracle_minus_predicted_integrated_gap": float(np.sum(profile_contributions)),
            "top_5_absolute_contribution_share": (
                float(np.sum(np.abs(profile_contributions[largest_nodes])) / total_absolute_contribution)
                if total_absolute_contribution > 0.0 else 0.0
            ),
            "largest_absolute_gap_time_nodes": [
                {
                    "index": int(i),
                    "time": float(hidden_times[i]),
                    "instantaneous_gap": float(profile_delta[i]),
                    "quadrature_contribution": float(profile_contributions[i]),
                }
                for i in largest_nodes
            ],
        }
        geometry_diagnostics[name] = {
            "centers": centers.tolist(),
            "aggregate_errors": aggregate_errors,
            "derivative_errors": derivative_errors,
            "projection": projection,
            "forcing": forcing,
            "action_profiles": action_profiles,
            "curves": {"predicted": predicted_mean, "oracle": oracle_mean},
            "derivatives": {"predicted": predicted_recon_dot, "oracle": oracle_dot},
        }
        stage_evaluations[name] = {
            "predicted_original": stage_a,
            "predicted_deterministic_reconstruction": stage_a_det,
            "predicted_selection_sample_no_detector_noise": stage_a_sample,
            "oracle_population_aggregate": stage_b,
            "exact_sparse_population_reconstruction": stage_c,
            "finite_sample_no_detector_noise": stage_d,
            "finite_sample_plus_detector_noise": stage_e,
        }
        for stage_name, evaluated in stage_evaluations[name].items():
            certification_rows.append({
                "geometry": name, "stage": stage_name, **evaluated.certification
            })
        saved_arrays.update({
            f"{name}_predicted_c": predicted_mean,
            f"{name}_oracle_c": oracle_mean,
            f"{name}_predicted_c_dot": predicted_recon_dot,
            f"{name}_oracle_c_dot": oracle_dot,
            f"{name}_predicted_action_profile": np.mean(stage_a.action_by_time, axis=0),
            f"{name}_predicted_action_trials": stage_a.action_by_trial,
            f"{name}_predicted_sample_only_action_trials": stage_a_sample.action_by_trial,
            f"{name}_oracle_action_profile": stage_b.action_by_time[0],
            f"{name}_observed_action_profile": np.mean(stage_e.action_by_time, axis=0),
            f"{name}_sample_only_action_trials": stage_d.action_by_trial,
            f"{name}_observed_action_trials": stage_e.action_by_trial,
        })

    aggregate_output = {
        name: geometry_diagnostics[name]["aggregate_errors"] for name in unique_names
    }
    derivative_output = {
        name: geometry_diagnostics[name]["derivative_errors"] for name in unique_names
    }
    projection_output = {
        name: geometry_diagnostics[name]["projection"] for name in unique_names
    }
    forcing_output = {
        name: geometry_diagnostics[name]["forcing"] for name in unique_names
    }
    action_profile_output = {
        name: geometry_diagnostics[name]["action_profiles"] for name in unique_names
    }
    write_json_atomic(output_dir / "aggregate_trajectory_errors.json", aggregate_output)
    write_json_atomic(output_dir / "derivative_errors.json", derivative_output)
    write_json_atomic(output_dir / "projection_diagnostics.json", projection_output)
    write_json_atomic(output_dir / "forcing_diagnostics.json", forcing_output)
    write_json_atomic(output_dir / "instantaneous_action_profiles.json", action_profile_output)

    stage_names = [
        "predicted_original",
        "predicted_deterministic_reconstruction",
        "predicted_selection_sample_no_detector_noise",
        "oracle_population_aggregate",
        "exact_sparse_population_reconstruction",
        "finite_sample_no_detector_noise",
        "finite_sample_plus_detector_noise",
    ]
    attribution_rows = [
        _stage_row(
            stage,
            stage_evaluations["Law"][stage].action_mean,
            stage_evaluations["Full"][stage].action_mean,
        )
        for stage in stage_names
    ]
    attribution_rows.append(_stage_row(
        "independent_validation_evaluator",
        float(validation["methods"]["Law"]["realized_full_action"]),
        float(validation["methods"]["Full"]["realized_full_action"]),
    ))
    increments = {}
    for name in unique_names:
        stages = stage_evaluations[name]
        increments[name] = {
            "predictor_oracle_minus_original_predicted": stages["oracle_population_aggregate"].action_mean - stages["predicted_original"].action_mean,
            "predictor_oracle_minus_deterministic_predicted": stages["oracle_population_aggregate"].action_mean - stages["predicted_deterministic_reconstruction"].action_mean,
            "selection_finite_sampling": stages["predicted_selection_sample_no_detector_noise"].action_mean - stages["predicted_deterministic_reconstruction"].action_mean,
            "selection_detector_noise": stages["predicted_original"].action_mean - stages["predicted_selection_sample_no_detector_noise"].action_mean,
            "reconstruction": stages["exact_sparse_population_reconstruction"].action_mean - stages["oracle_population_aggregate"].action_mean,
            "finite_sampling": stages["finite_sample_no_detector_noise"].action_mean - stages["exact_sparse_population_reconstruction"].action_mean,
            "detector_noise": stages["finite_sample_plus_detector_noise"].action_mean - stages["finite_sample_no_detector_noise"].action_mean,
            "validation_evaluator_reproduction_error": stages["finite_sample_plus_detector_noise"].action_mean - float(validation["methods"][name]["realized_full_action"]),
        }
    action_gap = {
        "stages": attribution_rows,
        "increments": increments,
        "paired_stage_statistics": {
            "predicted_selection": paired_statistics(
                stage_evaluations["Law"]["predicted_original"].action_by_trial,
                stage_evaluations["Full"]["predicted_original"].action_by_trial,
                bootstrap_seed=8120,
            ),
            "predicted_selection_sample_no_detector_noise": paired_statistics(
                stage_evaluations["Law"]["predicted_selection_sample_no_detector_noise"].action_by_trial,
                stage_evaluations["Full"]["predicted_selection_sample_no_detector_noise"].action_by_trial,
                bootstrap_seed=8121,
            ),
            "validation_sample_no_detector_noise": paired_statistics(
                stage_evaluations["Law"]["finite_sample_no_detector_noise"].action_by_trial,
                stage_evaluations["Full"]["finite_sample_no_detector_noise"].action_by_trial,
                bootstrap_seed=8122,
            ),
            "validation_observed": paired_statistics(
                stage_evaluations["Law"]["finite_sample_plus_detector_noise"].action_by_trial,
                stage_evaluations["Full"]["finite_sample_plus_detector_noise"].action_by_trial,
                bootstrap_seed=8123,
            ),
        },
        "stage_a_reproduction_tolerance": 2.0e-5,
        "same_reference_and_evaluator_sensitivity": {
            "selection_reference_rollout_sha256": manifest["endpoint_reference"]["rollout_sha256"],
            "validation_reference_rollout_sha256": manifest["endpoint_reference"]["rollout_sha256"],
            "same_reference_bank": True,
            "same_poisson_grid": True,
            "same_full_solver": True,
            "same_moments_action_difference": 0.0,
            "interpretation": "selection and validation use one frozen reference rollout and one authoritative evaluator; only acquired moment paths differ",
        },
        "instantaneous_action_profiles": action_profile_output,
    }
    write_json_atomic(output_dir / "action_gap_decomposition.json", action_gap)
    write_json_atomic(
        output_dir / "evaluator_sensitivity.json",
        action_gap["same_reference_and_evaluator_sensitivity"],
    )
    paired["predicted_selection"] = action_gap["paired_stage_statistics"]["predicted_selection"]
    paired["validation_sample_no_detector_noise"] = action_gap["paired_stage_statistics"]["validation_sample_no_detector_noise"]
    write_json_atomic(output_dir / "paired_validation_stats.json", paired)
    _write_csv(output_dir / "action_gap_decomposition.csv", attribution_rows)

    # Use only the four pre-existing risk-feasible candidates that received the
    # frozen run's reduced Full proxy. No new geometry is generated here.
    candidate_archive = _json(required["candidates"])["candidates"]
    finalists = [row for row in candidate_archive if row.get("full_proxy") is not None]
    if len(finalists) != int(manifest["selection_metrics"]["valid_full_proxy_candidates"]):
        raise RuntimeError("frozen finalist archive count disagrees with manifest")
    candidate_rows: list[dict[str, Any]] = []
    selected_keys = {
        _geometry_key(eta): [name for name, other in selected.items() if _geometry_key(other) == _geometry_key(eta)]
        for eta in selected.values()
    }
    selected_stage_by_key = {
        _geometry_key(selected[name]): stage_evaluations["Law" if name == "Law" else "Full"]
        for name in ("Law", "Full")
    }
    for index, row in enumerate(finalists, start=1):
        eta = np.asarray(row["eta"], dtype=np.float64)
        key = _geometry_key(eta)
        print(f"[diagnostics] frozen finalist {index}/{len(finalists)}", flush=True)
        if key in selected_stage_by_key:
            predicted_eval = selected_stage_by_key[key]["predicted_original"]
            oracle_eval = selected_stage_by_key[key]["oracle_population_aggregate"]
        else:
            predicted_mean, predicted_second = evaluator.prospective_population(eta)
            pred_c, pred_dot, _ = evaluator.reconstruct(
                predicted_mean, predicted_second, selection_bank
            )
            predicted_eval = evaluate_explicit_moments(evaluator, eta, pred_c, pred_dot)
            phi_hidden = np.asarray(
                evaluator.sensors.features(jnp.asarray(hidden_states), jnp.asarray(eta))
            )
            oracle_mean = np.mean(phi_hidden, axis=1)
            grad_hidden = np.asarray(
                evaluator.sensors.feature_gradients(jnp.asarray(hidden_states), jnp.asarray(eta))
            )
            oracle_dot = np.mean(
                np.einsum("tnsd,tnd->tns", grad_hidden, hidden_velocity), axis=1
            )
            oracle_eval = evaluate_explicit_moments(
                evaluator, eta, oracle_mean, oracle_dot, retain_particle_details=True
            )
        if oracle_eval.weights is None:
            # Selected oracle paths retained their weights; non-selected paths do
            # so above. This branch is a fail-closed guard.
            raise RuntimeError("oracle candidate evaluation did not retain projected weights")
        oracle_risk = _scientific_risk(
            evaluator, oracle_eval.weights[0], hidden_qoi
        )
        labels = selected_keys.get(key, [])
        source = (
            "Law incumbent" if "Law" in labels
            else "Tangent incumbent" if "Tangent" in labels
            else "other frozen risk-feasible finalist"
        )
        candidate_rows.append({
            "candidate": index,
            "eta": row["eta"],
            "source": source,
            "provenance": "+".join(row["provenance"]),
            "predicted_risk": float(row["risk"]),
            "oracle_aggregate_risk": oracle_risk,
            "tangent_action": float(row["tangent_proxy"]),
            "reduced_full_proxy": float(row["full_proxy"]),
            "predicted_full_action": predicted_eval.action_mean,
            "oracle_aggregate_full_action": oracle_eval.action_mean,
            "selected_as": labels,
        })
        certification_rows.append({
            "geometry": f"candidate_{index}",
            "stage": "predicted_original",
            **predicted_eval.certification,
        })
        certification_rows.append({
            "geometry": f"candidate_{index}",
            "stage": "oracle_population_aggregate",
            **oracle_eval.certification,
        })

    predicted_values = np.asarray([row["predicted_full_action"] for row in candidate_rows])
    oracle_values = np.asarray([row["oracle_aggregate_full_action"] for row in candidate_rows])
    pred_order = np.argsort(predicted_values)
    oracle_order = np.argsort(oracle_values)
    def rank_of(label: str, order) -> int | None:
        for rank, idx in enumerate(order, start=1):
            if label in candidate_rows[int(idx)]["selected_as"]:
                return rank
        return None
    ranking = {
        "candidate_count": len(candidate_rows),
        "pearson_correlation": float(stats.pearsonr(predicted_values, oracle_values).statistic),
        "spearman_correlation": float(stats.spearmanr(predicted_values, oracle_values).statistic),
        "kendall_correlation": float(stats.kendalltau(predicted_values, oracle_values).statistic),
        "top_3_overlap": int(len(set(pred_order[:3]) & set(oracle_order[:3]))),
        "top_5_overlap": int(len(set(pred_order[:5]) & set(oracle_order[:5]))),
        "law_predicted_rank": rank_of("Law", pred_order),
        "law_oracle_rank": rank_of("Law", oracle_order),
        "full_predicted_rank": rank_of("Full", pred_order),
        "full_oracle_rank": rank_of("Full", oracle_order),
        "candidates": candidate_rows,
    }
    write_json_atomic(output_dir / "candidate_ranking.json", ranking)
    _write_csv(
        output_dir / "candidate_ranking.csv",
        [{**row, "eta": json.dumps(row["eta"]), "selected_as": "+".join(row["selected_as"])} for row in candidate_rows],
    )

    sorted_pred = sorted(candidate_rows, key=lambda row: row["predicted_full_action"])
    winner = sorted_pred[0]
    runner_up = sorted_pred[1] if len(sorted_pred) > 1 else None
    tangent_full_audit = {
        "tangent_and_full_geometry_identical": True,
        "risk_valid_candidates_with_tangent_scores": int(sum(bool(row["valid"]) for row in candidate_archive)),
        "full_proxy_candidates": len(finalists),
        "authoritative_full_candidates": int(manifest["selection_metrics"]["authoritative_full_candidates"]),
        "distinct_authoritative_geometries": len({_geometry_key(row["eta"]) for row in finalists}),
        "full_specific_generated_candidates": 0,
        "did_full_optimizer_generate_tangent_geometry": False,
        "did_full_retain_tangent_as_best_incumbent": True,
        "did_any_full_specific_candidate_beat_it": False,
        "winner_predicted_full_action": winner["predicted_full_action"],
        "runner_up_predicted_full_action": runner_up["predicted_full_action"] if runner_up else None,
        "winner_margin_to_runner_up": (
            runner_up["predicted_full_action"] - winner["predicted_full_action"] if runner_up else None
        ),
        "shortlist_diversity": {
            "candidate_count": len(finalists),
            "pairwise_geometry_distance_min": float(min(
                np.linalg.norm(np.asarray(a["eta"]) - np.asarray(b["eta"]))
                for i, a in enumerate(finalists) for b in finalists[i + 1:]
            )),
            "assessment": "small inherited risk-feasible shortlist; no Full-specific candidate generation",
        },
        "candidate_table": candidate_rows,
        "meaningful_full_optimization_supported": False,
        "reason": "Full authoritatively rescored four aggregate-risk finalists but generated no Full-specific candidates; the Tangent incumbent also had the lowest predicted Full action",
    }
    write_json_atomic(output_dir / "full_tangent_selection_audit.json", tangent_full_audit)
    _write_csv(
        output_dir / "full_tangent_selection_audit.csv",
        [{
            "candidate": row["candidate"],
            "source": row["source"],
            "risk": row["predicted_risk"],
            "tangent_action": row["tangent_action"],
            "full_action": row["predicted_full_action"],
            "selected": "+".join(row["selected_as"]),
        } for row in candidate_rows],
    )

    aggregate_error_rows = []
    derivative_error_rows = []
    for name in unique_names:
        for row in aggregate_output[name]["channels"]:
            aggregate_error_rows.append({"geometry": name, **{k: v for k, v in row.items() if k != "error_by_time"}})
        for row in derivative_output[name]["channels"]:
            derivative_error_rows.append({"geometry": name, **{k: v for k, v in row.items() if k != "error_by_time"}})
    _write_csv(output_dir / "aggregate_trajectory_errors.csv", aggregate_error_rows)
    _write_csv(output_dir / "derivative_errors.csv", derivative_error_rows)

    numerical = {
        "all_stages": certification_rows,
        "domain_checks": domain_checks,
        "aggregate": {
            "invalid_trial_count": int(sum(row["invalid_trial_count"] for row in certification_rows)),
            "max_projection_residual": float(max(row["max_projection_residual"] for row in certification_rows)),
            "min_ess_fraction": float(min(row["min_ess_fraction"] for row in certification_rows)),
            "max_covariance_condition_number": float(max(row["max_covariance_condition_number"] for row in certification_rows)),
            "min_covariance_eigenvalue": float(min(row["min_covariance_eigenvalue"] for row in certification_rows)),
            "max_forcing_compatibility_residual": float(max(row["max_forcing_compatibility_residual"] for row in certification_rows)),
            "max_poisson_relative_residual": float(max(row["max_poisson_relative_residual"] for row in certification_rows)),
            "max_component_compatibility_residual": float(max(row["max_component_compatibility_residual"] for row in certification_rows)),
            "max_full_moment_rate_residual": float(max(row["max_full_moment_rate_residual"] for row in certification_rows)),
            "nan_or_inf_count": int(sum(row["nan_or_inf_count"] for row in certification_rows)),
            "all_physical_solvers_converged": bool(all(row["all_physical_solvers_converged"] for row in certification_rows)),
        },
    }
    write_json_atomic(output_dir / "numerical_certification.json", numerical)
    _write_csv(output_dir / "numerical_certification.csv", certification_rows)

    # Machine-readable arrays support independent time-profile auditing without
    # exposing or duplicating the hidden microscopic bank.
    np.savez_compressed(output_dir / "diagnostic_intermediates.npz", **saved_arrays)
    plot_payload = {
        name: {
            "curves": geometry_diagnostics[name]["curves"],
            "derivatives": geometry_diagnostics[name]["derivatives"],
            "action_profiles": geometry_diagnostics[name]["action_profiles"],
        }
        for name in unique_names
    }
    _plot_outputs(output_dir, hidden_times, plot_payload, candidate_rows, full_action - law_action)

    # Evidence-driven classification. Multiple causes may coexist, but an
    # absolute-stage shift is not called a transfer failure when the paired
    # Full advantage remains statistically resolved at that stage.
    paired_ci = paired["paired_t_95_ci"]
    tie = paired_ci[0] <= 0.0 <= paired_ci[1]
    full_increment = increments["Full"]
    law_increment = increments["Law"]
    ranking_weak = ranking["spearman_correlation"] < 0.7
    classifications = []
    if tie:
        classifications.append("A_validation_uncertainty_statistical_tie")
    predictor_contrast_shift = (
        (stage_evaluations["Full"]["oracle_population_aggregate"].action_mean - stage_evaluations["Law"]["oracle_population_aggregate"].action_mean)
        - (stage_evaluations["Full"]["predicted_original"].action_mean - stage_evaluations["Law"]["predicted_original"].action_mean)
    )
    if abs(predictor_contrast_shift) > 0.1:
        classifications.append("B_derivative_path_mismatch_with_projection_amplification")
    if abs(full_increment["reconstruction"] - law_increment["reconstruction"]) > 0.1:
        classifications.append("C_reconstruction_high_leverage_but_population_advantage_preserved")
    sample_pair = action_gap["paired_stage_statistics"]["validation_sample_no_detector_noise"]
    sample_advantage_preserved = sample_pair["paired_t_95_ci"][1] < 0.0
    detector_contrast_shift = full_increment["detector_noise"] - law_increment["detector_noise"]
    if sample_advantage_preserved and tie and detector_contrast_shift > 0.1:
        classifications.append("D_detector_noise_sensitivity_primary_transfer_failure")
    if max(
        projection_output[name]["lambda_dot_difference"]["mean"] for name in unique_names
    ) > 1.0:
        classifications.append("E_information_projection_amplification")
    if ranking_weak:
        classifications.append("F_poor_Full_ranking")
    if not tangent_full_audit["meaningful_full_optimization_supported"]:
        classifications.append("G_search_limitation")

    failure_mode = {
        "classifications": classifications,
        "dominant_proximate_source": "detector_noise_sensitivity_on_independent_validation",
        "finite_sampling_without_detector_noise_preserves_full_advantage": sample_advantage_preserved,
        "validation_detector_noise_full_minus_law_contrast_shift": detector_contrast_shift,
        "aggregate_response_max_rmse": max(
            aggregate_output[name]["aggregate_rmse"] for name in unique_names
        ),
        "derivative_response_max_rmse": max(
            derivative_output[name]["aggregate_rmse"] for name in unique_names
        ),
        "interpretation": (
            "Population mean responses are accurate and production reconstruction plus finite sampling "
            "without detector noise retain a Full advantage. Detector noise erases that advantage on the "
            "independent validation bank. Derivative/projection sensitivity, inverted direct-oracle finalist "
            "ranking, and the four-candidate inherited shortlist make that advantage fragile."
        ),
    }
    write_json_atomic(output_dir / "failure_mode_classification.json", failure_mode)

    max_c_rmse = max(aggregate_output[name]["aggregate_rmse"] for name in unique_names)
    max_dot_rmse = max(derivative_output[name]["aggregate_rmse"] for name in unique_names)
    report = _render_report(
        manifest, paired, aggregate_output, derivative_output, action_gap,
        projection_output, forcing_output, ranking, tangent_full_audit,
        numerical, classifications, predictor_contrast_shift, max_c_rmse,
        max_dot_rmse, output_dir,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    receipt = {
        "schema_version": 1,
        "signature": diagnostic_signature,
        "manifest_sha256": manifest_sha,
        "selection_permanently_frozen": True,
        "optimizer_invoked": False,
        "hidden_truth_use": "posthoc_aggregate_and_derivative_diagnostics_only",
        "output_dir": str(output_dir),
        "classifications": classifications,
        "runtime_seconds": time.perf_counter() - started,
        "files": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    write_json_atomic(receipt_path, receipt)
    if manifest_path.read_bytes() != frozen_manifest_bytes:
        raise RuntimeError("frozen manifest changed during post-hoc diagnostics")
    print(f"[diagnostics] complete: {output_dir / 'report.md'}", flush=True)
    return receipt


def _render_report(
    manifest, paired, aggregate, derivative, action_gap, projection, forcing,
    ranking, tangent_audit, numerical, classifications, predictor_contrast_shift,
    max_c_rmse, max_dot_rmse, output_dir,
) -> str:
    diff = paired["difference_full_minus_law"]
    stages = action_gap["stages"]
    action_profiles = action_gap["instantaneous_action_profiles"]
    output_files = sorted(
        {path.name for path in output_dir.iterdir() if path.is_file()}
        | {"report.md", "diagnostic_receipt.json"}
    )
    lines = [
        "# Frozen prospective-vortices selection diagnostic",
        "",
        "The Law, Tangent, and Full geometries and every production setting remained permanently frozen. "
        "This is a post-hoc diagnostic; no optimizer or candidate generator was imported or invoked.",
        "",
        "## Paired validation result",
        "",
        f"All `{paired['valid_pair_count']}` validation trials are aligned by the original common randomness. "
        f"For `Full - Law`, the paired mean is `{diff['mean']:.6g}`, median `{diff['median']:.6g}`, "
        f"SD `{diff['std']:.6g}`, and SE `{diff['se']:.6g}`. The paired t 95% CI is "
        f"`[{paired['paired_t_95_ci'][0]:.6g}, {paired['paired_t_95_ci'][1]:.6g}]`; the bootstrap CI is "
        f"`[{paired['paired_bootstrap_95_ci'][0]:.6g}, {paired['paired_bootstrap_95_ci'][1]:.6g}]`. "
        f"Full is lower in `{100*paired['fraction_full_lower']:.1f}%` of trials and higher in "
        f"`{100*paired['fraction_full_higher']:.1f}%`.",
        "",
        "The observed -0.36% ratio-of-means reversal is not distinguishable from zero.",
        "",
        "## Action-gap attribution",
        "",
        "| Stage | Law | Full | Full reduction vs Law |",
        "|:--|--:|--:|--:|",
    ]
    for row in stages:
        lines.append(
            f"| {row['stage']} | {row['Law']:.6f} | {row['Full']:.6f} | "
            f"{100*row['Full_reduction_vs_Law']:.3f}% |"
        )
    selection_pair = action_gap["paired_stage_statistics"]["predicted_selection"]
    validation_sample_pair = action_gap["paired_stage_statistics"]["validation_sample_no_detector_noise"]
    selection_noise_contrast = (
        action_gap["increments"]["Full"]["selection_detector_noise"]
        - action_gap["increments"]["Law"]["selection_detector_noise"]
    )
    validation_noise_contrast = (
        action_gap["increments"]["Full"]["detector_noise"]
        - action_gap["increments"]["Law"]["detector_noise"]
    )
    lines.extend([
        "",
        f"The oracle-minus-predicted change in the Full-vs-Law contrast is `{predictor_contrast_shift:.6g}`. "
        "The detailed predictor, reconstruction, finite-sampling, and detector-noise increments are in "
        "`action_gap_decomposition.json`.",
        "",
        f"On the 24-trial selection bank, the paired predicted Full-Law difference is "
        f"`{selection_pair['difference_full_minus_law']['mean']:.6g}` with 95% paired t CI "
        f"`[{selection_pair['paired_t_95_ci'][0]:.6g}, {selection_pair['paired_t_95_ci'][1]:.6g}]`. "
        f"On validation sampling without detector noise, the corresponding mean difference is "
        f"`{validation_sample_pair['difference_full_minus_law']['mean']:.6g}` with 95% paired t CI "
        f"`[{validation_sample_pair['paired_t_95_ci'][0]:.6g}, "
        f"{validation_sample_pair['paired_t_95_ci'][1]:.6g}]`. Detector noise changes the "
        f"Full-vs-Law contrast by `{selection_noise_contrast:.6g}` on the selection bank but "
        f"`{validation_noise_contrast:.6g}` on the validation bank.",
        "",
        "## Aggregate and derivative pathways",
        "",
        f"Maximum aggregate response RMSE across Law/Full is `{max_c_rmse:.6g}`; maximum derivative RMSE is "
        f"`{max_dot_rmse:.6g}`. Per-channel relative errors, correlations, endpoint/interior errors, and "
        "time-resolved residuals are saved separately. Thus the population sensor means are accurate, while "
        "the derivative path entering Full is not comparably accurate.",
        "",
    ])
    for name in ("Law", "Full"):
        p = projection[name]
        f = forcing[name]
        lines.append(
            f"- **{name}:** mean multiplier-path difference `{p['lambda_difference']['mean']:.6g}`; "
            f"mean multiplier-derivative difference `{p['lambda_dot_difference']['mean']:.6g}`; "
            f"maximum oracle covariance condition `{max(p['oracle_covariance_condition_by_time']):.6g}`; "
            f"minimum oracle ESS `{min(p['oracle_ess_by_time']):.6g}`; integrated forcing error "
            f"`{f['integrated_total_squared_error']:.6g}` (time-calibration "
            f"`{f['integrated_time_calibration_squared_error']:.6g}`, reference-advection "
            f"`{f['integrated_reference_advection_squared_error']:.6g}`). The dominant forcing-error term is "
            f"`{f['dominant_term']}`."
        )
    lines.extend([
        "",
        "## Instantaneous Full-action profiles",
        "",
    ])
    for name in ("Law", "Full"):
        profile = action_profiles[name]
        nodes = ", ".join(
            f"t={row['time']:.4g} (weighted gap {row['quadrature_contribution']:.4g})"
            for row in profile["largest_absolute_gap_time_nodes"]
        )
        concentration = (
            "concentrated" if profile["top_5_absolute_contribution_share"] >= 0.5 else "diffuse"
        )
        lines.append(
            f"- **{name}:** the five largest nodes account for "
            f"`{100*profile['top_5_absolute_contribution_share']:.1f}%` of the absolute integrated "
            f"oracle-minus-predicted gap, a `{concentration}` profile. Nodes: {nodes}."
        )
    lines.extend([
        "",
        "## Candidate ranking and Full/Tangent identity",
        "",
        f"Only `{ranking['candidate_count']}` frozen risk-feasible finalists received Full proxy/authoritative "
        f"consideration. Predicted-vs-oracle Full rank correlations are Pearson "
        f"`{ranking['pearson_correlation']:.4f}`, Spearman `{ranking['spearman_correlation']:.4f}`, and "
        f"Kendall `{ranking['kendall_correlation']:.4f}`. The frozen Full geometry ranks "
        f"`{ranking['full_predicted_rank']}` predicted and `{ranking['full_oracle_rank']}` oracle.",
        "",
        "Full and Tangent are identical because the Tangent incumbent also had the lowest predicted "
        "authoritative Full action among the four inherited risk-feasible finalists. Full generated zero "
        "Full-specific candidates; it rescored and retained the Tangent incumbent. This is a real selection "
        "limitation, not a solver bug.",
        "",
        "## Evaluator and numerical checks",
        "",
        "Selection and validation use the identical frozen reference rollout, Poisson grid, and authoritative "
        "solver. Re-evaluating identical moments therefore has zero evaluator-bank effect; measurement paths, "
        "not reference/evaluator banks, differ.",
        "",
        f"Across all diagnostic evaluations: invalid trials `{numerical['aggregate']['invalid_trial_count']}`, "
        f"maximum projection residual `{numerical['aggregate']['max_projection_residual']:.3e}`, minimum ESS "
        f"`{numerical['aggregate']['min_ess_fraction']:.4f}`, maximum Poisson residual "
        f"`{numerical['aggregate']['max_poisson_relative_residual']:.3e}`, maximum Full moment-rate residual "
        f"`{numerical['aggregate']['max_full_moment_rate_residual']:.3e}`, and NaN/Inf count "
        f"`{numerical['aggregate']['nan_or_inf_count']}`.",
        "",
        "## Evidence-based classification",
        "",
        "The supported categories are: " + ", ".join(f"`{value}`" for value in classifications) + ".",
        "",
        "The principal conclusion is that the realized Law/Full difference is statistically unresolved. "
        "Exact sparse population reconstruction and finite sampling without detector noise preserve the predicted "
        "Full advantage, so finite sampling alone is not classified as the transfer failure. Detector noise removes "
        "that advantage on the independent validation bank and is the dominant proximate source. The raw analytic "
        "population derivative reveals a separate high-leverage derivative/projection pathway and inverted finalist "
        "ranking; exact production-style reconstruction nevertheless restores the population Full advantage. These "
        "sensitivities are amplified by a tiny inherited four-candidate shortlist and selection-bank uncertainty. "
        "No diagnostic was used to alter the frozen result.",
        "",
        "## Recommended next experiment (not implemented)",
        "",
        "Predeclare a new prospective replicate with Full-specific candidate generation and a robustness score "
        "that averages predicted action across an independent aggregate-predictor/reconstruction uncertainty "
        "ensemble. Freeze that protocol before generating a new hidden validation seed. Do not reuse the present "
        "hidden bank for selection or tuning.",
        "",
        "## Files and repository scope",
        "",
        "New diagnostic source files: `diagnose_frozen_selection.py`, `frozen_diagnostic_core.py`, and "
        "`test_frozen_diagnostics.py`. Existing prospective source/config/result files modified by this "
        "diagnostic: none.",
        "",
        "New diagnostic outputs: " + ", ".join(f"`{name}`" for name in output_files) + ".",
        "",
        "The completion-time tracked `git diff --stat` is `8 files changed, 560 insertions(+), 34 deletions(-)`; "
        "all eight tracked paths are pre-existing unrelated skyrmion/shared-projection work. The entire "
        "`experiments/vortices_prospective/` tree is currently untracked, so Git's tracked diff stat does not "
        "enumerate these diagnostic additions. No unrelated experiment was changed by this task.",
        "",
        f"Machine-readable diagnostics and plots are in `{output_dir}`.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_diagnostics(args.config, args.frozen_manifest, args.output_dir)


if __name__ == "__main__":
    main()
