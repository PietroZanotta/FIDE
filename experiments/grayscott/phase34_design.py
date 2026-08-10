"""Phase 3–4 infrastructure checks that do not train a reference or MFSI model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .benchmark_design import ROOT
from .feasibility import calibrate_iprojection_instrumented, solve_target_hull_lp
from .field_transport import (
    maximal_same_index_coupling,
    sample_reference_interpolant,
    smooth_hidden_observables,
)
from .observables import ShellDefinition, field_observables
from .phase2_continuation import DEFAULT_CONFIG, _json_default, _load_config, _write_csv, _write_json


DESIGN_TIMES = np.linspace(0.10, 0.90, 9)


def run_intermediate_projections(config_path: Path = DEFAULT_CONFIG, bank_count: int = 4096) -> dict:
    config = _load_config(config_path)
    output = ROOT / config["output_directory"]
    summary = json.loads((output / "large_bank_phase2_summary.json").read_text())
    candidate = summary["provisional_phase2_candidate"]
    if candidate is None:
        raise RuntimeError("Phase 2 has no passing provisional candidate")
    if int(candidate["observation_dimension"]) != 2:
        raise RuntimeError("the current prespecified Phase-2 ranking is expected to select Phi-2")
    bank = np.load(output / "large_design_banks.npz")
    regime_ids = list(map(str, bank["regime_ids"]))
    endpoint_fields = np.asarray(bank["endpoint_v"])
    spot_index = regime_ids.index(candidate["spot_regime"])
    labyrinth_index = regime_ids.index(candidate["labyrinth_regime"])
    trace_path = output / "large_bank_calibration_traces" / f"phi2_{candidate['pair_id']}.json"
    endpoint_trace = json.loads(trace_path.read_text())
    spot_weights = np.asarray(endpoint_trace["minus_calibration"]["weights"], dtype=np.float64)
    labyrinth_weights = np.asarray(endpoint_trace["plus_calibration"]["weights"], dtype=np.float64)
    coupling = maximal_same_index_coupling(spot_weights, labyrinth_weights)
    same_ic_mass = float(np.trace(coupling))
    center = np.asarray(summary["standardization"]["center"][:2], dtype=np.float64)
    scale = np.asarray(summary["standardization"]["scale"][:2], dtype=np.float64)
    target = np.asarray(candidate["target_standardized"], dtype=np.float64)
    source_design = json.loads((ROOT / config["source_design_config"]).read_text())
    obs = source_design["observables"]
    shells = ShellDefinition(tuple(obs["shell_centers_cycles_per_pixel"]), tuple(obs["shell_widths_cycles_per_pixel"]))
    threshold = float(summary["fixed_threshold"])
    rows, details = [], {}
    projected_states, projected_weights = [], []
    for time_index, time in enumerate(DESIGN_TIMES):
        states, derivatives, minus_indices, plus_indices = sample_reference_interpolant(
            endpoint_fields[spot_index], endpoint_fields[labyrinth_index], coupling,
            float(time), bank_count, 33000 + time_index,
        )
        physical_features = np.asarray(field_observables(
            jnp.asarray(states), shells, ("mean", "second_moment")
        ), dtype=np.float64)
        standardized = (physical_features - center) / scale
        calibrated = calibrate_iprojection_instrumented(
            standardized, target,
            tolerance=float(config["calibration"]["residual_tolerance"]),
            max_iterations=int(config["calibration"]["maximum_iterations"]),
        )
        hidden = np.asarray(smooth_hidden_observables(jnp.asarray(states), threshold=threshold))
        uniform_hidden = hidden.mean(axis=0)
        projected_hidden = calibrated["weights"] @ hidden
        weights = calibrated["weights"]
        divergence = float(np.sum(weights * np.log(np.maximum(weights * len(weights), 1e-300))))
        row = {
            "t": float(time), "bank_seed": 33000 + time_index, "bank_count": bank_count,
            "converged": calibrated["converged"],
            "convergence_reason": calibrated["convergence_reason"],
            "iterations": calibrated["iterations"],
            "maximum_standardized_residual": calibrated["maximum_absolute_standardized_residual"],
            "ess_fraction": calibrated["ess_fraction"],
            "maximum_weight": calibrated["maximum_weight"],
            "lambda_norm": calibrated["lambda_norm"],
            "covariance_rank": calibrated["covariance_rank"],
            "covariance_condition": calibrated["covariance_condition"],
            "projection_kl_distortion": divergence,
            "hidden_shift_l2": float(np.linalg.norm(projected_hidden - uniform_hidden)),
            "standardization_roundtrip_error": float(np.max(np.abs(standardized * scale + center - physical_features))),
        }
        rows.append(row)
        details[str(float(time))] = {
            "row": row, "calibration": calibrated,
            "reference_hidden_mean": uniform_hidden,
            "projected_hidden_mean": projected_hidden,
            "projected_minus_indices": minus_indices,
            "projected_plus_indices": plus_indices,
        }
        projected_states.append(states)
        projected_weights.append(weights)
    minimum_ess = min(row["ess_fraction"] for row in rows)
    maximum_residual = max(row["maximum_standardized_residual"] for row in rows)
    overlap_gate = minimum_ess >= 0.15 and maximum_residual <= 1e-5
    result = {
        "candidate": candidate,
        "reference_coupling": "maximum same-initial-condition mass with exact calibrated endpoint marginals",
        "same_initial_condition_coupling_mass": same_ic_mass,
        "design_times": DESIGN_TIMES,
        "bank_count_per_time": bank_count,
        "rows": rows,
        "minimum_interior_ess_fraction": minimum_ess,
        "maximum_interior_standardized_residual": maximum_residual,
        "interior_projection_overlap_gate_pass": overlap_gate,
        "reference_velocity_training_performed": False,
        "phase3_complete": False,
        "phase3_incomplete_reason": "reference CNN architecture is implemented, but reference flow-matching training is prohibited in this continuation",
        "phase4_complete": False,
        "phase4_incomplete_reason": "tangent blind-spot code is implemented but requires a validated trained reference velocity",
    }
    _write_csv(output / "intermediate_projection_phi2.csv", rows)
    _write_json(output / "intermediate_projection_phi2.json", {**result, "details": details})
    np.savez_compressed(
        output / "intermediate_projection_phi2_banks.npz",
        times=DESIGN_TIMES, states=np.asarray(projected_states),
        weights=np.asarray(projected_weights), target=target, center=center, scale=scale,
    )
    return result


def screen_all_phase2_candidates(config_path: Path = DEFAULT_CONFIG, bank_count: int = 2048) -> dict:
    config = _load_config(config_path)
    output = ROOT / config["output_directory"]
    phase2 = json.loads((output / "large_bank_phase2_summary.json").read_text())
    candidates = phase2["passing_candidates_ranked"]
    bank = np.load(output / "large_design_banks.npz")
    regime_ids = list(map(str, bank["regime_ids"]))
    endpoint_fields = np.asarray(bank["endpoint_v"])
    source_design = json.loads((ROOT / config["source_design_config"]).read_text())
    obs = source_design["observables"]
    shells = ShellDefinition(tuple(obs["shell_centers_cycles_per_pixel"]), tuple(obs["shell_widths_cycles_per_pixel"]))
    all_rows, candidate_rows = [], []
    for candidate_index, candidate in enumerate(candidates):
        dimension = int(candidate["observation_dimension"])
        pair_id = candidate["pair_id"]
        trace = json.loads((
            output / "large_bank_calibration_traces" / f"phi{dimension}_{pair_id}.json"
        ).read_text())
        minus_weights = np.asarray(trace["minus_calibration"]["weights"])
        plus_weights = np.asarray(trace["plus_calibration"]["weights"])
        coupling = maximal_same_index_coupling(minus_weights, plus_weights)
        minus_fields = endpoint_fields[regime_ids.index(candidate["spot_regime"])]
        plus_fields = endpoint_fields[regime_ids.index(candidate["labyrinth_regime"])]
        center = np.asarray(phase2["standardization"]["center"][:dimension])
        scale = np.asarray(phase2["standardization"]["scale"][:dimension])
        target = np.asarray(candidate["target_standardized"])
        components = ("mean", "second_moment", "shell_1")[:dimension]
        time_rows = []
        for time_index, time in enumerate(DESIGN_TIMES):
            states, _, _, _ = sample_reference_interpolant(
                minus_fields, plus_fields, coupling, float(time), bank_count,
                34000 + 100 * candidate_index + time_index,
            )
            physical = np.asarray(field_observables(jnp.asarray(states), shells, components))
            standardized = (physical - center) / scale
            hull = solve_target_hull_lp(standardized, target)
            calibrated = None
            if hull["success"]:
                calibrated = calibrate_iprojection_instrumented(
                    standardized, target,
                    tolerance=float(config["calibration"]["residual_tolerance"]),
                    max_iterations=int(config["calibration"]["maximum_iterations"]),
                )
            row = {
                "candidate_rank": candidate_index + 1, "pair_id": pair_id,
                "observation_dimension": dimension, "t": float(time),
                "target_hull_feasible": hull["success"],
                "target_hull_lp_residual": hull.get("maximum_equality_residual"),
                "calibration_converged": calibrated["converged"] if calibrated else False,
                "maximum_standardized_residual": (
                    calibrated["maximum_absolute_standardized_residual"] if calibrated else None
                ),
                "ess_fraction": calibrated["ess_fraction"] if calibrated else None,
                "maximum_weight": calibrated["maximum_weight"] if calibrated else None,
            }
            time_rows.append(row)
            all_rows.append(row)
        feasible_all = all(row["target_hull_feasible"] for row in time_rows)
        calibrated_all = all(row["calibration_converged"] for row in time_rows)
        ess_values = [row["ess_fraction"] for row in time_rows if row["ess_fraction"] is not None]
        residual_values = [
            row["maximum_standardized_residual"] for row in time_rows
            if row["maximum_standardized_residual"] is not None
        ]
        minimum_ess = min(ess_values) if ess_values else None
        maximum_residual = max(residual_values) if residual_values else None
        candidate_rows.append({
            "candidate_rank": candidate_index + 1, "pair_id": pair_id,
            "observation_dimension": dimension,
            "same_initial_condition_coupling_mass": float(np.trace(coupling)),
            "all_time_targets_hull_feasible": feasible_all,
            "all_time_calibrations_converged": calibrated_all,
            "feasible_time_count": sum(row["target_hull_feasible"] for row in time_rows),
            "minimum_interior_ess_fraction": minimum_ess,
            "maximum_interior_standardized_residual": maximum_residual,
            "phase3_overlap_gate_pass": bool(
                feasible_all and calibrated_all and minimum_ess is not None
                and minimum_ess >= 0.15 and maximum_residual <= 1e-5
            ),
        })
    passing = [row for row in candidate_rows if row["phase3_overlap_gate_pass"]]
    result = {
        "candidate_count": len(candidates), "bank_count_per_time": bank_count,
        "design_times": DESIGN_TIMES, "candidate_rows": candidate_rows,
        "phase3_overlap_passing_count": len(passing),
        "phase3_overlap_passing_candidates": passing,
        "reference_velocity_training_performed": False,
    }
    _write_csv(output / "phase3_all_candidate_time_rows.csv", all_rows)
    _write_csv(output / "phase3_all_candidate_summary.csv", candidate_rows)
    _write_json(output / "phase3_all_candidate_overlap.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("selected", "screen-all"), nargs="?", default="selected")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--bank-count", type=int, default=4096)
    args = parser.parse_args()
    result = (run_intermediate_projections(args.config, args.bank_count)
              if args.command == "selected" else screen_all_phase2_candidates(args.config, args.bank_count))
    print(json.dumps(result, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
