"""Versioned method-blind Gray--Scott Phase-3A reference-path design."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .benchmark_design import ROOT
from .feasibility import calibrate_iprojection_instrumented, solve_target_hull_lp
from .field_transport import (
    field_l2_cost,
    geometric_l2_transport_coupling,
    independent_coupling,
    maximal_same_index_coupling,
    noisy_field_interpolant,
    smooth_hidden_observables,
    standardized_noise_bank,
)
from .observables import ShellDefinition, field_observables
from .phase2_continuation import _json_default, _write_csv, _write_json


DEFAULT_CONFIG = ROOT / "configs" / "expC_grayscott_phase3_v7.yaml"


def _read_json(path: Path):
    return json.loads(path.read_text())


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): _hash(path)
        for path in sorted(directory.rglob("*")) if path.is_file()
    }


def _systematic_coupling_sample(coupling: np.ndarray, count: int, seed: int):
    """Low-variance equal-weight sampling from a fixed discrete coupling."""
    flat = np.asarray(coupling, dtype=np.float64).ravel()
    flat /= flat.sum()
    rng = np.random.default_rng(seed)
    positions = (rng.random() + np.arange(count)) / count
    selected = np.searchsorted(np.cumsum(flat), positions, side="right")
    return np.unravel_index(np.minimum(selected, len(flat) - 1), coupling.shape)


def _calibration_diagnostics(result: dict | None) -> dict | None:
    if result is None:
        return None
    keys = (
        "converged", "convergence_reason", "iterations", "initial_dual_objective",
        "final_dual_objective", "maximum_absolute_standardized_residual", "lambda_norm",
        "covariance_eigenvalues", "covariance_rank", "covariance_condition", "ess_fraction",
        "maximum_weight", "minimum_weight", "entropy_fraction",
        "residual_identity_maximum_difference", "direct_weighted_mean_residual",
        "reported_residual", "trace", "lambda",
    )
    return {key: result[key] for key in keys}


def _coupling_for_name(name, minus_fields, plus_fields, minus_weights, plus_weights):
    if name == "maximal_same_ic":
        coupling = maximal_same_index_coupling(minus_weights, plus_weights)
        diagnostics = {"construction": "maximum same-index mass with independent residual"}
    elif name == "geometric_l2_ot":
        coupling, diagnostics = geometric_l2_transport_coupling(
            minus_fields, plus_fields, minus_weights, plus_weights
        )
    elif name == "independent":
        coupling = independent_coupling(minus_weights, plus_weights)
        diagnostics = {"construction": "outer product of calibrated endpoint marginals"}
    else:
        raise ValueError(f"unknown coupling {name}")
    cost = field_l2_cost(minus_fields, plus_fields)
    diagnostics.update({
        "transport_cost_mean_squared_per_pixel": float(np.sum(coupling * cost)),
        "transport_displacement_rms": float(np.sqrt(np.sum(coupling * cost))),
        "maximum_marginal_residual": float(max(
            np.max(np.abs(coupling.sum(axis=1) - minus_weights)),
            np.max(np.abs(coupling.sum(axis=0) - plus_weights)),
        )),
        "same_initial_condition_mass": float(np.trace(coupling)),
        "positive_edge_count": int(np.count_nonzero(coupling > 1e-14)),
    })
    return coupling, diagnostics


def _second_moment_identity(
    minus_fields, plus_fields, coupling, target_second_moment, times
):
    minus_flat = np.asarray(minus_fields, dtype=np.float64).reshape((len(minus_fields), -1))
    plus_flat = np.asarray(plus_fields, dtype=np.float64).reshape((len(plus_fields), -1))
    minus_second = np.mean(minus_flat * minus_flat, axis=1)
    plus_second = np.mean(plus_flat * plus_flat, axis=1)
    cost = field_l2_cost(minus_fields, plus_fields)
    cross = 0.5 * (minus_second[:, None] + plus_second[None, :] - cost)
    displacement = float(np.sum(coupling * cost))
    rows = []
    for map_name, time_map in (
        ("identity", lambda value: value),
        ("smoothstep_reparameterization", lambda value: value * value * (3.0 - 2.0 * value)),
    ):
        for time in times:
            s = float(time_map(float(time)))
            empirical_lhs = float(np.sum(coupling * (
                (1.0 - s) ** 2 * minus_second[:, None]
                + s * s * plus_second[None, :]
                + 2.0 * s * (1.0 - s) * cross
            )))
            rhs = float(target_second_moment - s * (1.0 - s) * displacement)
            rows.append({
                "time_parameterization": map_name, "t": float(time), "s": s,
                "empirical_lhs_second_moment": empirical_lhs,
                "analytic_rhs_second_moment": rhs,
                "absolute_identity_error": abs(empirical_lhs - rhs),
                "second_moment_deficit": float(s * (1.0 - s) * displacement),
                "mean_squared_endpoint_displacement": displacement,
            })
    return rows


def _hidden_scales(all_fields: np.ndarray, threshold: float):
    values = []
    for start in range(0, len(all_fields), 1024):
        values.append(np.asarray(smooth_hidden_observables(
            jnp.asarray(all_fields[start:start + 1024]), threshold=threshold
        )))
    values = np.concatenate(values)
    return np.maximum(values.std(axis=0, ddof=1), 1e-12)


def _feature_summary(values: np.ndarray) -> dict:
    return {
        "mean": values.mean(axis=0), "minimum": values.min(axis=0),
        "maximum": values.max(axis=0),
        "quantile_01": np.quantile(values, 0.01, axis=0),
        "quantile_50": np.quantile(values, 0.50, axis=0),
        "quantile_99": np.quantile(values, 0.99, axis=0),
    }


def _evaluate_path(
    *, candidate, candidate_rank, coupling_name, coupling, coupling_diagnostics,
    minus_fields, plus_fields, center, scale, target, shells, threshold, hidden_scales,
    times, count, seed, amplitude, gates, calibration_config, capture_weights=False,
):
    minus_indices, plus_indices = _systematic_coupling_sample(coupling, count, seed)
    sampled_minus = minus_fields[minus_indices]
    sampled_plus = plus_fields[plus_indices]
    noise = standardized_noise_bank(count, sampled_minus.shape[1:], seed + 100000)
    components = ("mean", "second_moment", "shell_1")[:len(target)]
    time_rows, details = [], []
    for time_index, time in enumerate(times):
        states, _ = noisy_field_interpolant(
            sampled_minus, sampled_plus, noise, float(time), float(amplitude)
        )
        states = np.asarray(states)
        physical = np.asarray(field_observables(jnp.asarray(states), shells, components), dtype=np.float64)
        standardized = (physical - center) / scale
        hull = solve_target_hull_lp(standardized, target)
        calibrated = None
        if hull["success"]:
            calibrated = calibrate_iprojection_instrumented(
                standardized, target,
                tolerance=float(calibration_config["residual_tolerance"]),
                max_iterations=int(calibration_config["maximum_iterations"]),
            )
        hidden = np.asarray(smooth_hidden_observables(jnp.asarray(states), threshold=threshold))
        uniform_hidden = hidden.mean(axis=0)
        projected_hidden = calibrated["weights"] @ hidden if calibrated is not None else None
        hidden_shift = (
            float(np.linalg.norm((projected_hidden - uniform_hidden) / hidden_scales) / np.sqrt(len(hidden_scales)))
            if projected_hidden is not None else None
        )
        state_minimum, state_maximum = float(states.min()), float(states.max())
        violation_fraction = float(np.mean(
            (states < float(gates["hard_field_minimum"]))
            | (states > float(gates["hard_field_maximum"]))
        ))
        kl = None
        if calibrated is not None:
            weights = calibrated["weights"]
            kl = float(np.sum(weights * np.log(np.maximum(weights * len(weights), 1e-300))))
        row = {
            "candidate_rank": candidate_rank, "pair_id": candidate["pair_id"],
            "observation_dimension": len(target), "coupling": coupling_name,
            "schedule_amplitude": float(amplitude), "t": float(time),
            "bank_count": count, "bank_seed": seed,
            "target_hull_feasible": bool(hull["success"]),
            "target_hull_lp_residual": hull.get("maximum_equality_residual"),
            "calibration_converged": calibrated["converged"] if calibrated else False,
            "maximum_standardized_residual": calibrated["maximum_absolute_standardized_residual"] if calibrated else None,
            "ess_fraction": calibrated["ess_fraction"] if calibrated else None,
            "projection_kl_distortion": kl,
            "lambda_norm": calibrated["lambda_norm"] if calibrated else None,
            "maximum_weight": calibrated["maximum_weight"] if calibrated else None,
            "covariance_rank": calibrated["covariance_rank"] if calibrated else None,
            "covariance_condition": calibrated["covariance_condition"] if calibrated else None,
            "hidden_shift_standardized_rms": hidden_shift,
            "hidden_shift_nontrivial": bool(hidden_shift is not None and hidden_shift >= float(gates["nontrivial_hidden_shift_standardized_rms"])),
            "field_minimum": state_minimum, "field_maximum": state_maximum,
            "hard_range_violation_fraction": violation_fraction,
            "physical_range_gate_pass": violation_fraction <= float(gates["maximum_hard_range_violation_fraction"]),
        }
        time_rows.append(row)
        detail = {
            "row": row, "raw_phi_physical": _feature_summary(physical),
            "raw_phi_standardized": _feature_summary(standardized),
            "hull": {key: value for key, value in hull.items() if key != "weights"},
            "calibration": _calibration_diagnostics(calibrated),
            "uniform_hidden_mean": uniform_hidden, "projected_hidden_mean": projected_hidden,
        }
        if capture_weights and calibrated is not None:
            detail["calibrated_weights"] = calibrated["weights"]
        details.append(detail)
    feasible_all = all(row["target_hull_feasible"] for row in time_rows)
    converged_all = all(row["calibration_converged"] for row in time_rows)
    ess = [row["ess_fraction"] for row in time_rows if row["ess_fraction"] is not None]
    residual = [row["maximum_standardized_residual"] for row in time_rows if row["maximum_standardized_residual"] is not None]
    kl_values = [row["projection_kl_distortion"] for row in time_rows if row["projection_kl_distortion"] is not None]
    lambda_values = [row["lambda_norm"] for row in time_rows if row["lambda_norm"] is not None]
    hidden_count = sum(row["hidden_shift_nontrivial"] for row in time_rows)
    physical_all = all(row["physical_range_gate_pass"] for row in time_rows)
    minimum_ess = min(ess) if ess else None
    maximum_residual = max(residual) if residual else None
    residual_gate_all = bool(
        len(residual) == len(time_rows) and maximum_residual is not None
        and maximum_residual <= float(gates["maximum_standardized_residual"])
    )
    passes = bool(
        feasible_all and residual_gate_all and minimum_ess is not None
        and minimum_ess >= float(gates["minimum_interior_ess_fraction"])
        and hidden_count >= int(gates["minimum_nontrivial_hidden_shift_time_count"])
        and physical_all
    )
    reasons = []
    if not feasible_all: reasons.append("target_not_in_empirical_hull_at_all_times")
    if minimum_ess is None or minimum_ess < float(gates["minimum_interior_ess_fraction"]): reasons.append("minimum_interior_ess")
    if not residual_gate_all: reasons.append("calibration_residual_gate")
    if hidden_count < int(gates["minimum_nontrivial_hidden_shift_time_count"]): reasons.append("insufficient_nontrivial_hidden_projection_shift")
    if not physical_all: reasons.append("hard_field_range_pathology")
    summary = {
        "candidate_rank": candidate_rank, "pair_id": candidate["pair_id"],
        "observation_dimension": len(target), "coupling": coupling_name,
        "schedule_amplitude": float(amplitude),
        "mean_squared_endpoint_displacement": coupling_diagnostics["transport_cost_mean_squared_per_pixel"],
        "endpoint_displacement_rms": coupling_diagnostics["transport_displacement_rms"],
        "maximum_coupling_marginal_residual": coupling_diagnostics["maximum_marginal_residual"],
        "all_time_target_hull_feasible": feasible_all,
        "feasible_time_count": sum(row["target_hull_feasible"] for row in time_rows),
        "all_time_calibrations_converged": converged_all,
        "calibration_solver_warning_time_count": sum(not row["calibration_converged"] for row in time_rows),
        "all_time_calibration_residual_gate_pass": residual_gate_all,
        "minimum_interior_ess_fraction": minimum_ess,
        "maximum_interior_standardized_residual": maximum_residual,
        "mean_projection_kl_distortion": float(np.mean(kl_values)) if kl_values else None,
        "maximum_lambda_norm": max(lambda_values) if lambda_values else None,
        "nontrivial_hidden_shift_time_count": hidden_count,
        "all_time_physical_range_gate_pass": physical_all,
        "phase3a_gate_pass": passes, "rejection_reasons": ";".join(reasons),
    }
    return summary, time_rows, details


def run_linear_screen(config_path: Path = DEFAULT_CONFIG) -> dict:
    config = _read_json(config_path.resolve())
    source = ROOT / config["source_phase2_directory"]
    output = ROOT / config["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    source_manifest = _manifest(source)
    _write_json(output / "preserved_phase2_v6_sha256.json", source_manifest)
    (output / "phase3_v7_config.json").write_text(config_path.read_text())
    phase2 = _read_json(source / "large_bank_phase2_summary.json")
    candidates = phase2["passing_candidates_ranked"]
    bank = np.load(source / "large_design_banks.npz")
    regime_ids = list(map(str, bank["regime_ids"]))
    endpoint_fields = np.asarray(bank["endpoint_v"])
    design_config = _read_json(ROOT / config["source_design_config"])
    obs = design_config["observables"]
    shells = ShellDefinition(tuple(obs["shell_centers_cycles_per_pixel"]), tuple(obs["shell_widths_cycles_per_pixel"]))
    threshold = float(phase2["fixed_threshold"])
    hidden_scales = _hidden_scales(endpoint_fields.reshape((-1,) + endpoint_fields.shape[2:]), threshold)
    times = list(map(float, config["design_times"]))
    all_summaries, all_time_rows, identity_rows, coupling_rows = [], [], [], []
    detail_directory = output / "linear_details"
    detail_directory.mkdir(exist_ok=True)
    for candidate_index, candidate in enumerate(candidates):
        rank = candidate_index + 1
        dimension = int(candidate["observation_dimension"])
        trace = _read_json(source / "large_bank_calibration_traces" / f"phi{dimension}_{candidate['pair_id']}.json")
        minus_weights = np.asarray(trace["minus_calibration"]["weights"], dtype=np.float64)
        plus_weights = np.asarray(trace["plus_calibration"]["weights"], dtype=np.float64)
        minus_fields = endpoint_fields[regime_ids.index(candidate["spot_regime"])]
        plus_fields = endpoint_fields[regime_ids.index(candidate["labyrinth_regime"])]
        center = np.asarray(phase2["standardization"]["center"][:dimension])
        scale = np.asarray(phase2["standardization"]["scale"][:dimension])
        target = np.asarray(candidate["target_standardized"])
        for coupling_index, coupling_name in enumerate(config["couplings"]):
            coupling, coupling_diagnostics = _coupling_for_name(
                coupling_name, minus_fields, plus_fields, minus_weights, plus_weights
            )
            coupling_rows.append({
                "candidate_rank": rank, "pair_id": candidate["pair_id"],
                "observation_dimension": dimension, "coupling": coupling_name,
                **coupling_diagnostics,
            })
            identity = _second_moment_identity(
                minus_fields, plus_fields, coupling, float(candidate["target_physical"][1]), times
            )
            for row in identity:
                row.update({"candidate_rank": rank, "pair_id": candidate["pair_id"], "coupling": coupling_name})
            identity_rows.extend(identity)
            seed = int(config["interior_bank_seed_start"]) + 100 * candidate_index + 10 * coupling_index
            summary, time_rows, details = _evaluate_path(
                candidate=candidate, candidate_rank=rank, coupling_name=coupling_name,
                coupling=coupling, coupling_diagnostics=coupling_diagnostics,
                minus_fields=minus_fields, plus_fields=plus_fields,
                center=center, scale=scale, target=target, shells=shells,
                threshold=threshold, hidden_scales=hidden_scales, times=times,
                count=int(config["interior_bank_count"]), seed=seed, amplitude=0.0,
                gates=config["phase3a_gates"], calibration_config=config["calibration"],
            )
            all_summaries.append(summary)
            all_time_rows.extend(time_rows)
            _write_json(detail_directory / f"rank{rank:02d}_{coupling_name}.json", {
                "candidate": candidate, "coupling_diagnostics": coupling_diagnostics,
                "second_moment_identity": identity, "time_details": details,
            })
    passing = [row for row in all_summaries if row["phase3a_gate_pass"]]
    # Candidate rank was fixed in Phase 2, before any learned method existed.
    passing.sort(key=lambda row: (row["candidate_rank"], config["couplings"].index(row["coupling"])))
    result = {
        "status": "phase3a_linear_pass" if passing else "phase3a_linear_failed",
        "candidate_count": len(candidates), "coupling_count": len(config["couplings"]),
        "interior_bank_count": int(config["interior_bank_count"]), "design_times": times,
        "fixed_phase2_targets_retained": True, "passing_path_count": len(passing),
        "passing_paths_phase2_rank_order": passing,
        "provisional_reference_path": passing[0] if passing else None,
        "schedule_screen_required": not bool(passing),
        "reference_velocity_training_performed": False,
        "deep_ritz_training_performed": False, "method_comparison_performed": False,
        "hidden_observable_scales": hidden_scales,
        "source_phase2_artifacts_unchanged_after_run": _manifest(source) == source_manifest,
    }
    _write_csv(output / "linear_coupling_diagnostics.csv", coupling_rows)
    _write_csv(output / "linear_second_moment_identity.csv", identity_rows)
    _write_csv(output / "linear_time_diagnostics.csv", all_time_rows)
    _write_csv(output / "linear_path_summary.csv", all_summaries)
    _write_json(output / "linear_phase3a_summary.json", result)
    return result


def run_schedule_screen(config_path: Path = DEFAULT_CONFIG) -> dict:
    """Screen the frozen one-scalar endpoint-zero noise schedule after linear failure."""
    config = _read_json(config_path.resolve())
    source = ROOT / config["source_phase2_directory"]
    output = ROOT / config["output_directory"]
    linear_summary_path = output / "linear_phase3a_summary.json"
    if not linear_summary_path.exists():
        raise RuntimeError("run the complete linear coupling screen first")
    linear_result = _read_json(linear_summary_path)
    if linear_result["passing_path_count"]:
        raise RuntimeError("a linear path already passes; schedule modification is not authorized")
    source_manifest = _read_json(output / "preserved_phase2_v6_sha256.json")
    if _manifest(source) != source_manifest:
        raise RuntimeError("Phase-2 v6 artifacts changed after the linear screen")
    phase2 = _read_json(source / "large_bank_phase2_summary.json")
    candidates = phase2["passing_candidates_ranked"]
    bank = np.load(source / "large_design_banks.npz")
    regime_ids = list(map(str, bank["regime_ids"]))
    endpoint_fields = np.asarray(bank["endpoint_v"])
    design_config = _read_json(ROOT / config["source_design_config"])
    obs = design_config["observables"]
    shells = ShellDefinition(tuple(obs["shell_centers_cycles_per_pixel"]), tuple(obs["shell_widths_cycles_per_pixel"]))
    threshold = float(phase2["fixed_threshold"])
    hidden_scales = np.asarray(linear_result["hidden_observable_scales"], dtype=np.float64)
    times = list(map(float, config["design_times"]))
    amplitudes = list(map(float, config["schedule_if_linear_fails"]["amplitude_grid"]))
    all_summaries, all_time_rows = [], []
    detail_directory = output / "schedule_details"
    detail_directory.mkdir(exist_ok=True)
    for candidate_index, candidate in enumerate(candidates):
        rank = candidate_index + 1
        dimension = int(candidate["observation_dimension"])
        trace = _read_json(source / "large_bank_calibration_traces" / f"phi{dimension}_{candidate['pair_id']}.json")
        minus_weights = np.asarray(trace["minus_calibration"]["weights"], dtype=np.float64)
        plus_weights = np.asarray(trace["plus_calibration"]["weights"], dtype=np.float64)
        minus_fields = endpoint_fields[regime_ids.index(candidate["spot_regime"])]
        plus_fields = endpoint_fields[regime_ids.index(candidate["labyrinth_regime"])]
        center = np.asarray(phase2["standardization"]["center"][:dimension])
        scale = np.asarray(phase2["standardization"]["scale"][:dimension])
        target = np.asarray(candidate["target_standardized"])
        candidate_path_count = 0
        for coupling_index, coupling_name in enumerate(config["couplings"]):
            coupling, coupling_diagnostics = _coupling_for_name(
                coupling_name, minus_fields, plus_fields, minus_weights, plus_weights
            )
            seed = int(config["interior_bank_seed_start"]) + 100 * candidate_index + 10 * coupling_index
            for amplitude in amplitudes:
                summary, time_rows, details = _evaluate_path(
                    candidate=candidate, candidate_rank=rank, coupling_name=coupling_name,
                    coupling=coupling, coupling_diagnostics=coupling_diagnostics,
                    minus_fields=minus_fields, plus_fields=plus_fields,
                    center=center, scale=scale, target=target, shells=shells,
                    threshold=threshold, hidden_scales=hidden_scales, times=times,
                    count=int(config["interior_bank_count"]), seed=seed, amplitude=amplitude,
                    gates=config["phase3a_gates"], calibration_config=config["calibration"],
                )
                all_summaries.append(summary)
                all_time_rows.extend(time_rows)
                candidate_path_count += int(summary["phase3a_gate_pass"])
                _write_json(
                    detail_directory / f"rank{rank:02d}_{coupling_name}_a{amplitude:.3f}.json",
                    {"candidate": candidate, "coupling_diagnostics": coupling_diagnostics,
                     "schedule": {"amplitude": amplitude, "envelope": "sin(pi*t)",
                                  "noise": config["schedule_if_linear_fails"]["noise_convention"]},
                     "time_details": details},
                )
        print(f"schedule screen candidate {rank:02d}/14: {candidate_path_count} passing paths", flush=True)
    passing = [row for row in all_summaries if row["phase3a_gate_pass"]]
    # Apply the frozen schedule objective within each Phase-2 candidate. A
    # larger worst-time ESS is primary after feasibility, followed by lower
    # distortion, multiplier/conditioning burden, and finally amplitude.
    best_by_candidate = []
    for rank in range(1, len(candidates) + 1):
        available = [row for row in passing if row["candidate_rank"] == rank]
        if available:
            available.sort(key=lambda row: (
                -row["minimum_interior_ess_fraction"], row["mean_projection_kl_distortion"],
                row["maximum_lambda_norm"], row["schedule_amplitude"],
                config["couplings"].index(row["coupling"]),
            ))
            best_by_candidate.append(available[0])
    # Cross-candidate ordering remains the predeclared method-blind Phase-2 rank.
    best_by_candidate.sort(key=lambda row: row["candidate_rank"])
    result = {
        "status": "phase3a_pass" if best_by_candidate else "phase3a_failed",
        "linear_path_passing_count": 0, "schedule_path_count": len(all_summaries),
        "schedule_passing_path_count": len(passing),
        "phase2_candidates_with_passing_schedule": len(best_by_candidate),
        "best_passing_schedule_by_phase2_candidate": best_by_candidate,
        "provisional_reference_path": best_by_candidate[0] if best_by_candidate else None,
        "fixed_phase2_targets_retained": True,
        "schedule_family": config["schedule_if_linear_fails"],
        "reference_velocity_training_performed": False,
        "deep_ritz_training_performed": False, "method_comparison_performed": False,
        "source_phase2_artifacts_unchanged_after_run": _manifest(source) == source_manifest,
    }
    _write_csv(output / "schedule_time_diagnostics.csv", all_time_rows)
    _write_csv(output / "schedule_path_summary.csv", all_summaries)
    _write_json(output / "schedule_phase3a_summary.json", result)
    return result


def run_selected_confirmation(config_path: Path = DEFAULT_CONFIG) -> dict:
    """Independent, doubled-bank confirmation of the selected Phase-3A path."""
    config = _read_json(config_path.resolve())
    source = ROOT / config["source_phase2_directory"]
    output = ROOT / config["output_directory"]
    schedule_result = _read_json(output / "schedule_phase3a_summary.json")
    selected = schedule_result["provisional_reference_path"]
    if selected is None:
        raise RuntimeError("there is no Phase-3A path to confirm")
    source_manifest = _read_json(output / "preserved_phase2_v6_sha256.json")
    if _manifest(source) != source_manifest:
        raise RuntimeError("Phase-2 v6 artifacts changed before confirmation")
    phase2 = _read_json(source / "large_bank_phase2_summary.json")
    candidate = phase2["passing_candidates_ranked"][int(selected["candidate_rank"]) - 1]
    bank = np.load(source / "large_design_banks.npz")
    regime_ids = list(map(str, bank["regime_ids"]))
    endpoint_fields = np.asarray(bank["endpoint_v"])
    dimension = int(candidate["observation_dimension"])
    trace = _read_json(source / "large_bank_calibration_traces" / f"phi{dimension}_{candidate['pair_id']}.json")
    minus_weights = np.asarray(trace["minus_calibration"]["weights"], dtype=np.float64)
    plus_weights = np.asarray(trace["plus_calibration"]["weights"], dtype=np.float64)
    minus_fields = endpoint_fields[regime_ids.index(candidate["spot_regime"])]
    plus_fields = endpoint_fields[regime_ids.index(candidate["labyrinth_regime"])]
    coupling, coupling_diagnostics = _coupling_for_name(
        selected["coupling"], minus_fields, plus_fields, minus_weights, plus_weights
    )
    design_config = _read_json(ROOT / config["source_design_config"])
    obs = design_config["observables"]
    shells = ShellDefinition(tuple(obs["shell_centers_cycles_per_pixel"]), tuple(obs["shell_widths_cycles_per_pixel"]))
    center = np.asarray(phase2["standardization"]["center"][:dimension])
    scale = np.asarray(phase2["standardization"]["scale"][:dimension])
    target = np.asarray(candidate["target_standardized"])
    count = 2 * int(config["interior_bank_count"])
    seed = int(config["interior_bank_seed_start"]) + 200000 + int(selected["candidate_rank"])
    times = list(map(float, config["design_times"]))
    summary, time_rows, details = _evaluate_path(
        candidate=candidate, candidate_rank=int(selected["candidate_rank"]),
        coupling_name=selected["coupling"], coupling=coupling,
        coupling_diagnostics=coupling_diagnostics, minus_fields=minus_fields,
        plus_fields=plus_fields, center=center, scale=scale, target=target,
        shells=shells, threshold=float(phase2["fixed_threshold"]),
        hidden_scales=np.asarray(_read_json(output / "linear_phase3a_summary.json")["hidden_observable_scales"]),
        times=times, count=count, seed=seed, amplitude=float(selected["schedule_amplitude"]),
        gates=config["phase3a_gates"], calibration_config=config["calibration"],
        capture_weights=True,
    )
    minus_indices, plus_indices = _systematic_coupling_sample(coupling, count, seed)
    projected_weights = np.stack([np.asarray(detail.pop("calibrated_weights")) for detail in details])
    np.savez_compressed(
        output / "selected_phase3a_confirmation_bank.npz", times=np.asarray(times),
        minus_indices=minus_indices, plus_indices=plus_indices,
        projected_weights=projected_weights, target=target, center=center, scale=scale,
        noise_seed=np.asarray(seed + 100000), sampling_seed=np.asarray(seed),
        schedule_amplitude=np.asarray(float(selected["schedule_amplitude"])),
    )
    result = {
        "status": "phase3a_pass_confirmed" if summary["phase3a_gate_pass"] else "phase3a_confirmation_failed",
        "screen_selected_path": selected, "confirmation_path": summary,
        "confirmation_bank_count": count, "confirmation_sampling_seed": seed,
        "independent_from_screen_sampling_seed": True,
        "phase3a_pass_confirmed": bool(summary["phase3a_gate_pass"]),
        "reference_velocity_training_performed": False,
        "source_phase2_artifacts_unchanged_after_run": _manifest(source) == source_manifest,
    }
    _write_csv(output / "selected_confirmation_time_diagnostics.csv", time_rows)
    _write_json(output / "selected_confirmation_details.json", {**result, "time_details": details})
    _write_json(output / "selected_confirmation_summary.json", result)
    return result


def _parse_scalar(value):
    if value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def reassess_saved_schedule_gates(config_path: Path = DEFAULT_CONFIG) -> dict:
    """Correctly apply the frozen 1e-5 gate to already-computed diagnostics."""
    config = _read_json(config_path.resolve())
    output = ROOT / config["output_directory"]
    with (output / "schedule_path_summary.csv").open() as handle:
        summaries = [{key: _parse_scalar(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    with (output / "schedule_time_diagnostics.csv").open() as handle:
        time_rows = [{key: _parse_scalar(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    grouped = {}
    for row in time_rows:
        key = (row["candidate_rank"], row["coupling"], row["schedule_amplitude"])
        grouped.setdefault(key, []).append(row)
    gates = config["phase3a_gates"]
    for summary in summaries:
        key = (summary["candidate_rank"], summary["coupling"], summary["schedule_amplitude"])
        rows = grouped[key]
        feasible_all = all(row["target_hull_feasible"] for row in rows)
        solver_all = all(row["calibration_converged"] for row in rows)
        residuals = [row["maximum_standardized_residual"] for row in rows if row["maximum_standardized_residual"] is not None]
        ess = [row["ess_fraction"] for row in rows if row["ess_fraction"] is not None]
        residual_gate = bool(len(residuals) == len(rows) and max(residuals) <= float(gates["maximum_standardized_residual"]))
        minimum_ess = min(ess) if ess else None
        hidden_count = sum(row["hidden_shift_nontrivial"] for row in rows)
        physical_all = all(row["physical_range_gate_pass"] for row in rows)
        passed = bool(
            feasible_all and residual_gate and minimum_ess is not None
            and minimum_ess >= float(gates["minimum_interior_ess_fraction"])
            and hidden_count >= int(gates["minimum_nontrivial_hidden_shift_time_count"])
            and physical_all
        )
        reasons = []
        if not feasible_all: reasons.append("target_not_in_empirical_hull_at_all_times")
        if minimum_ess is None or minimum_ess < float(gates["minimum_interior_ess_fraction"]): reasons.append("minimum_interior_ess")
        if not residual_gate: reasons.append("calibration_residual_gate")
        if hidden_count < int(gates["minimum_nontrivial_hidden_shift_time_count"]): reasons.append("insufficient_nontrivial_hidden_projection_shift")
        if not physical_all: reasons.append("hard_field_range_pathology")
        summary.update({
            "all_time_calibrations_converged": solver_all,
            "calibration_solver_warning_time_count": sum(not row["calibration_converged"] for row in rows),
            "all_time_calibration_residual_gate_pass": residual_gate,
            "phase3a_gate_pass": passed, "rejection_reasons": ";".join(reasons),
        })
    passing = [row for row in summaries if row["phase3a_gate_pass"]]
    best_by_candidate = []
    for rank in range(1, 15):
        available = [row for row in passing if row["candidate_rank"] == rank]
        if available:
            available.sort(key=lambda row: (
                -row["minimum_interior_ess_fraction"], row["mean_projection_kl_distortion"],
                row["maximum_lambda_norm"], row["schedule_amplitude"],
                config["couplings"].index(row["coupling"]),
            ))
            best_by_candidate.append(available[0])
    best_by_candidate.sort(key=lambda row: row["candidate_rank"])
    prior = _read_json(output / "schedule_phase3a_summary.json")
    result = {
        **prior, "status": "phase3a_pass" if best_by_candidate else "phase3a_failed",
        "schedule_passing_path_count": len(passing),
        "phase2_candidates_with_passing_schedule": len(best_by_candidate),
        "best_passing_schedule_by_phase2_candidate": best_by_candidate,
        "provisional_reference_path": best_by_candidate[0] if best_by_candidate else None,
        "gate_logic_correction": (
            "The predeclared Phase-3A residual gate is 1e-5. The 1e-10 calibration "
            "solver target remains reported separately and is not an additional selection gate."
        ),
    }
    _write_csv(output / "schedule_path_summary.csv", summaries)
    _write_json(output / "schedule_phase3a_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("linear", "schedule", "confirm", "reassess"), nargs="?", default="linear")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if args.command == "linear":
        result = run_linear_screen(args.config)
    elif args.command == "schedule":
        result = run_schedule_screen(args.config)
    elif args.command == "confirm":
        result = run_selected_confirmation(args.config)
    else:
        result = reassess_saved_schedule_gates(args.config)
    print(json.dumps(result, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
