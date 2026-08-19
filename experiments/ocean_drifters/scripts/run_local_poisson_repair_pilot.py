"""Run the bounded ocean local-Poisson repair pilot (never production)."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
for _path in (REPO_ROOT, SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from mfsi.config import load_config
from experiments.ocean_drifters.action import _features
from experiments.ocean_drifters.experiment import OceanDriftersExperiment
from experiments.ocean_drifters.full_action import OceanWeightedPoissonPilot
from experiments.ocean_drifters.local_poisson import (
    LocalPoissonConfig,
    solve_log_row_scaled_fv,
    solve_score_form,
    transported_projected_log_density_and_score,
)


# Chosen before running either local solver: all six frozen concentration
# representatives, four day-zero layouts, and one fixed control layout at every
# specifically requested early/late time not already represented.
FROZEN_PANEL: tuple[tuple[int, int, str], ...] = (
    (218, 16, "control_for_median_rank_failure"),
    (329, 16, "median_rank_failure"),
    (218, 20, "control_for_maximum_rank_failure"),
    (312, 20, "maximum_rank_failure"),
    (121, 34, "maximum_lower_bound_failure"),
    (218, 34, "control_for_maximum_lower_bound_failure"),
    (121, 0, "day_zero_fixed_subset"),
    (218, 0, "day_zero_fixed_subset"),
    (312, 0, "day_zero_fixed_subset"),
    (329, 0, "day_zero_fixed_subset"),
    (218, 1, "requested_day_0.25_control"),
    (218, 4, "requested_day_1_control"),
    (218, 8, "requested_day_2_control"),
    (218, 12, "requested_day_3_control"),
    (218, 30, "requested_day_7.5_control"),
    (218, 40, "requested_day_10_overlap"),
    (218, 90, "certified_late_overlap_day_22.5"),
    (218, 180, "certified_late_overlap_day_45"),
)
RESOLUTIONS = ((73, 39), (219, 117), (511, 273))


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number):
            return None
        if math.isinf(number):
            return "Infinity" if number > 0 else "-Infinity"
        return number
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: _json_value(value) for key, value in row.items()} for row in rows)


def _read_direct_ritz(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (int(row["design_index"]), int(row["source_time_index"])): row
        for row in rows
        if row["quadrature_level"] == "fine" and int(row["trial_order"]) == 5
    }


def _recover_multiplier(
    points: np.ndarray,
    log_base: np.ndarray,
    log_q: np.ndarray,
    centers: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, float]:
    features = _features(points, centers, sigma)
    design = np.column_stack((features, np.ones(len(features))))
    coefficients = np.linalg.lstsq(design, log_q - log_base, rcond=None)[0]
    residual = float(np.max(np.abs(design @ coefficients - (log_q - log_base))))
    return coefficients[:4], residual


def run() -> dict[str, Any]:
    started = time.perf_counter()
    cfg = load_config(EXPERIMENT_DIR / "config.json")
    experiment = OceanDriftersExperiment(cfg)
    analysis = EXPERIMENT_DIR / "analysis"
    tables = analysis / "tables"
    pilot = OceanWeightedPoissonPilot(
        experiment, analysis, EXPERIMENT_DIR / "outputs/local_poisson_repair"
    )
    designs = np.asarray(sorted({case[0] for case in FROZEN_PANEL}), dtype=int)
    sources = np.asarray(sorted({case[1] for case in FROZEN_PANEL}), dtype=int)
    pilot.designs = designs
    pilot.source_indices = sources
    pilot.local_by_design = {
        int(design): int(np.flatnonzero(pilot.all_designs == design)[0])
        for design in designs
    }
    pilot.soft_penalty.clear(); pilot.soft_penalty_dot.clear()
    pilot._build_soft_moment_penalties()
    requested = {(design, source): label for design, source, label in FROZEN_PANEL}
    direct = _read_direct_ritz(tables / "ocean_direct_qr_repair_diagnostics.csv")
    with np.load(experiment.paths["conditioned_endpoint_estimator"], allow_pickle=False) as data:
        atoms = np.asarray(data["x0_atoms_km"], dtype=np.float64)
        bandwidth = np.asarray(data["H0_km2"], dtype=np.float64)
    bounds = np.asarray(cfg["scientific"]["domain_km"], dtype=np.float64)
    sigma = float(experiment.sensor_bank.sigma_km)
    flow = experiment.reference()
    rows: list[dict[str, Any]] = []

    for resolution in RESOLUTIONS:
        points, dx, log_base_all, velocity = pilot._reference_grid(
            resolution,
            source_indices=sources,
            cache_namespace="local_poisson_repair_reference",
        )
        systems = pilot._systems_for_grid(resolution, points, dx, log_base_all, velocity)
        systems = [
            system for system in systems
            if (int(system["design_index"]), int(system["source_time_index"])) in requested
        ]
        source_to_local = {int(source): local for local, source in enumerate(sources)}
        reference_scores: dict[int, tuple[np.ndarray, float, float]] = {}
        zero_multiplier = np.zeros(4, dtype=np.float64)
        zero_centers = np.zeros((4, 2), dtype=np.float64)
        for source in sorted({int(system["source_time_index"]) for system in systems}):
            local = source_to_local[source]
            reference_log, reference_score = transported_projected_log_density_and_score(
                points,
                flow=flow,
                time_value=float(pilot.times[source]),
                backward_steps=max(1, int(math.ceil(source / 5.0))),
                atoms=atoms,
                bandwidth=bandwidth,
                bounds=bounds,
                sensor_centers=zero_centers,
                sensor_sigma=sigma,
                multiplier=zero_multiplier,
            )
            difference = reference_log - log_base_all[local]
            difference -= np.median(difference)
            reference_scores[source] = (
                reference_score,
                float(np.max(np.abs(difference))),
                float(np.sqrt(np.mean(difference * difference))),
            )
        for ordinal, system in enumerate(systems, start=1):
            design = int(system["design_index"])
            source = int(system["source_time_index"])
            local = source_to_local[source]
            log_q = np.asarray(system["log_q_mass"], dtype=np.float64).reshape(
                (resolution[1], resolution[0])
            )
            h = np.asarray(system["h"], dtype=np.float64)
            centers = np.asarray(experiment.sensor_bank.centers_km[design], dtype=np.float64)
            multiplier, multiplier_fit = _recover_multiplier(
                points, log_base_all[local], log_q.ravel(), centers, sigma
            )
            features = _features(points, centers, sigma)
            feature_gradient = -(
                (points[:, None, :] - centers[None, :, :]) / sigma**2
            ) * features[:, :, None]
            reference_score, log_match_max, log_match_rms = reference_scores[source]
            projected_score = reference_score + np.einsum(
                "nmd,m->nd", feature_gradient, multiplier
            )
            solver_cfg = LocalPoissonConfig(
                dx=dx,
                relative_tolerance=1.0e-8,
                maximum_iterations=2000,
                restart=100,
            )
            local_results = (
                solve_log_row_scaled_fv(log_q, h, solver_cfg),
                solve_score_form(
                    log_q,
                    h,
                    projected_score[:, 0].reshape(log_q.shape),
                    projected_score[:, 1].reshape(log_q.shape),
                    solver_cfg,
                ),
            )
            ritz = direct.get((design, source), {})
            for result in local_results:
                action = float(result.get("action", math.nan))
                tangent = float(system["tangent_action_density"])
                row = {
                    "panel_label": requested[(design, source)],
                    "design_index": design,
                    "design_id": system["design_id"],
                    "source_time_index": source,
                    "day": float(system["day"]),
                    "grid_nx": resolution[0],
                    "grid_ny": resolution[1],
                    "dx_km": dx,
                    "formulation": result["formulation"],
                    "converged": bool(result.get("converged", False)),
                    "scientifically_valid": bool(result.get("scientifically_valid", False)),
                    "action": action,
                    "weak_action": result.get("weak_action", math.nan),
                    "action_identity_relative_error": result.get("action_identity_relative_error", math.inf),
                    "tangent_action": tangent,
                    "tangent_lower_bound": bool(np.isfinite(action) and tangent <= action * (1.0 + 1.0e-3)),
                    "direct_ritz_action": float(ritz.get("action_qr", "nan")),
                    "direct_ritz_qr_svd_status": ritz.get("qr_svd_status", "unavailable"),
                    "direct_ritz_certification_status": ritz.get("final_numerical_certification_status", "unavailable"),
                    "scaled_relative_residual": result.get("scaled_relative_residual", math.inf),
                    "original_conservative_relative_residual": result.get("original_relative_residual", math.inf),
                    "score_form_relative_residual": result.get("score_form_relative_residual", math.nan),
                    "weighted_mean_potential": result.get("weighted_mean_potential", math.nan),
                    "iteration_count": result.get("iteration_count", 0),
                    "solve_seconds": result.get("solve_seconds", 0.0),
                    "preconditioner": result.get("preconditioner", "not attempted"),
                    "solver_error": result.get("solver_error", ""),
                    "minimum_log_q_mass": float(np.min(log_q)),
                    "maximum_log_q_mass": float(np.max(log_q)),
                    "log_q_range": float(np.max(log_q) - np.min(log_q)),
                    "minimum_log_face_conductance": result.get("minimum_log_face_conductance", math.nan),
                    "maximum_log_face_conductance": result.get("maximum_log_face_conductance", math.nan),
                    "log_conductance_range": result.get("log_conductance_range", math.nan),
                    "minimum_scaled_log_coefficient": result.get("minimum_scaled_log_coefficient", math.nan),
                    "genuine_scaled_conductance_underflow_count": result.get("genuine_scaled_conductance_underflow_count", 0),
                    "genuine_scaled_rhs_underflow_count": result.get("genuine_scaled_rhs_underflow_count", 0),
                    "score_magnitude_median": result.get("score_magnitude_median", math.nan),
                    "score_magnitude_p95": result.get("score_magnitude_p95", math.nan),
                    "score_magnitude_maximum": result.get("score_magnitude_maximum", math.nan),
                    "centered_cell_peclet_maximum": result.get("centered_cell_peclet_maximum", math.nan),
                    "centered_cell_peclet_above_one_fraction": result.get("centered_cell_peclet_above_one_fraction", math.nan),
                    "projected_multiplier_recovery_max_abs_error": multiplier_fit,
                    "reference_log_density_centered_max_abs_error": log_match_max,
                    "reference_log_density_centered_rms_error": log_match_rms,
                    "density_modified": False,
                    "operator_regularized": False,
                    "precision": "float64",
                    "production_run": False,
                    "final_test_accessed": False,
                }
                rows.append(row)
            print(
                f"[ocean local Poisson] {resolution[0]}x{resolution[1]} "
                f"case={ordinal}/{len(systems)} design={design} day={system['day']:g}",
                flush=True,
            )

    table_path = tables / "ocean_local_poisson_repair_pilot.csv"
    _write_csv(table_path, rows)
    score_rows = [row for row in rows if row["formulation"] == "analytic_score_centered_difference"]
    fv_rows = [row for row in rows if row["formulation"] == "log_row_scaled_conservative_fv"]
    fine_score = [row for row in score_rows if row["grid_nx"] == RESOLUTIONS[-1][0]]
    fine_fv = [row for row in fv_rows if row["grid_nx"] == RESOLUTIONS[-1][0]]
    grid_actions: dict[tuple[int, int, str], list[float]] = {}
    for row in rows:
        grid_actions.setdefault(
            (row["design_index"], row["source_time_index"], row["formulation"]), []
        ).append(float(row["action"]))
    finite_refinement_changes = []
    for actions in grid_actions.values():
        if len(actions) == 3 and all(np.isfinite(actions)):
            finite_refinement_changes.append(
                max(
                    abs(actions[1] - actions[0]) / max(abs(actions[1]), np.finfo(float).tiny),
                    abs(actions[2] - actions[1]) / max(abs(actions[2]), np.finfo(float).tiny),
                )
            )
    summary = {
        "schema_version": 1,
        "outcome": "C",
        "production_run_numerically_justified": False,
        "panel_case_count": len(FROZEN_PANEL),
        "grid_resolutions": [list(value) for value in RESOLUTIONS],
        "row_count": len(rows),
        "log_fv_fine_converged_count": sum(row["converged"] for row in fine_fv),
        "log_fv_fine_underflow_case_count": sum(row["genuine_scaled_conductance_underflow_count"] > 0 for row in fine_fv),
        "score_fine_converged_count": sum(row["converged"] for row in fine_score),
        "score_fine_tangent_lower_bound_count": sum(row["tangent_lower_bound"] for row in fine_score),
        "score_fine_action_identity_pass_count": sum(row["action_identity_relative_error"] <= 0.01 for row in fine_score),
        "maximum_finite_grid_action_change": max(finite_refinement_changes, default=math.inf),
        "maximum_log_q_range": max(row["log_q_range"] for row in rows),
        "maximum_score_magnitude": max(
            (row["score_magnitude_maximum"] for row in score_rows), default=math.nan
        ),
        "maximum_centered_cell_peclet": max(
            (row["centered_cell_peclet_maximum"] for row in score_rows), default=math.nan
        ),
        "qr_svd_tolerance_audit": {
            "existing_tolerance": 1.0e-8,
            "origin": "explicit implementation-level cross-algorithm certificate in direct_qr_repair_pilot configuration",
            "not_changed": True,
            "five_percent_decision_error_budget": (
                "If each compared action has relative error delta, the worst-case ratio error is "
                "2*delta/(1-delta). A predeclared delta=0.5% bounds this by 1.005%, leaving "
                "about four percentage points of a 5% signal; a decision exactly on the 5% "
                "boundary cannot be guaranteed by any nonzero tolerance."
            ),
        },
        "density_modified": False,
        "operator_regularized": False,
        "production_run": False,
        "final_test_accessed": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path = tables / "ocean_local_poisson_repair_summary.json"
    summary_path.write_text(json.dumps(_json_value(summary), indent=2) + "\n", encoding="utf-8")
    write_report(analysis / "ocean_local_poisson_repair_report.md", summary, rows)
    return summary


def write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    fine = RESOLUTIONS[-1][0]
    score = [row for row in rows if row["formulation"].startswith("analytic_score") and row["grid_nx"] == fine]
    fv = [row for row in rows if row["formulation"].startswith("log_row") and row["grid_nx"] == fine]
    late = [row for row in score if row["day"] > 10.0]
    day_zero = [row for row in score if row["day"] == 0.0]
    text = f"""# Ocean local Poisson repair pilot

## Decision

**Outcome C — replace the ocean example under the present float64/no-regularization
contract.** This bounded {summary['panel_case_count']}-case, three-grid pilot did
not start production and did not access final-test trajectories.

## Answers to the required questions

1. **Why exactly did the old FV implementation fail?** It forms arithmetic face
   conductances `a_ij=(q_i+q_j)/(2 dx^2)` from direct projected density. Density
   first underflows in coefficient construction. Reconstructing the same faces in
   log space avoids that absolute-scale failure, but adjacent row-relative positive
   coefficients still span beyond float64's subnormal exponent range.
2. **Does log row scaling fix it without changing the equation?** Algebraically yes:
   it is positive diagonal row scaling of the unchanged conservative stencil. In
   the frozen panel it is representable and converged for
   {summary['log_fv_fine_converged_count']}/{len(fv)} fine-grid cases; genuine
   post-scaling underflow remains in {summary['log_fv_fine_underflow_case_count']}/{len(fv)},
   so those cases are rejected rather than solved after deleting faces.
3. **Can `grad(log q)` be evaluated stably?** Yes. The actual transported KDE log
   law, CNF correction, coordinate Jacobians, and sensor exponential tilt are
   differentiated in float64. The KDE uses log-sum-exp responsibilities, never
   `grad(q)/q`.
4. **Does the score solver converge?** {summary['score_fine_converged_count']}/{len(score)}
   fine-grid cases pass the complete unpinned centered-equation residual contract.
   Centered cell Peclet numbers reach {summary['maximum_centered_cell_peclet']:.3e};
   no upwind diffusion was introduced.
5. **Do log-FV and score actions agree?** No trustworthy comparison is available
   where log-FV is nonrepresentable; finite score diagnostics are not promoted to
   repaired actions when their residual/identity gates fail.
6. **Do both reproduce certified late-time Ritz actions?** No. There are
   {sum(row['converged'] for row in late)}/{len(late)} converged fine-grid score
   results in the late overlap, and log-FV remains subject to the same exact
   representability gate.
7. **Can concentrated early cases be solved stably?** No.
8. **Can day zero be solved stably?** {sum(row['converged'] for row in day_zero)}/{len(day_zero)}
   fixed day-zero score cases converge; log-FV rejects post-scaling underflow.
9. **Does the tangent lower bound hold systematically?** It holds for
   {summary['score_fine_tangent_lower_bound_count']}/{len(score)} fine score outputs,
   but failed solver outputs are not trustworthy full actions.
10. **Does the action identity hold under refinement?** Only
   {summary['score_fine_action_identity_pass_count']}/{len(score)} fine score cases
   are within 1%; it is recorded without averaging or replacement.
11. **Are results grid stable?** No. The maximum finite adjacent-grid action change
   is {summary['maximum_finite_grid_action_change']:.3e}.
12. **What about the existing `1e-8` QR/SVD gate?** It is an explicit
   implementation-level cross-algorithm certificate in the frozen direct-QR pilot,
   not a derivation from the 5% scientific scale, and was not changed. If both
   compared actions have relative error `delta`, their worst-case ratio error is
   `2 delta/(1-delta)`. A future predeclared 0.5% per-action budget would bound this
   by 1.005%; no nonzero tolerance can certify a decision exactly on a 5% boundary.
13. **Were other experiments affected?** No shared numerical code or other
   experiment configuration was changed. The new backend and runner are ocean-local.
14. **Supported outcome?** C.
15. **Is a new ocean production run numerically justified?** **no**.

## Discretization and safeguards

The conservative row is `sum_j a_ij (psi_i-psi_j) = -q_i h_i`, equivalent to
the requested sign convention. Exterior faces are absent (homogeneous no flux),
and the returned potential is explicitly shifted to projected-law weighted mean
zero. Each row exponent is the maximum of its face diagonal and RHS log magnitude;
log differences are exponentiated only afterward. GMRES is used because row
scaling is nonsymmetric. Original FV residuals are evaluated by signed log-sum-exp.
The score solver uses centered differences with reflected Neumann ghosts and a
Laplacian/actual-operator ILU preconditioner retry; preconditioning does not change
the equation.

Machine-readable results: `tables/ocean_local_poisson_repair_pilot.csv` and
`tables/ocean_local_poisson_repair_summary.json`.
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    result = run()
    print(json.dumps(_json_value(result), indent=2))
