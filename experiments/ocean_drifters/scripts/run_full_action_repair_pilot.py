"""Run the frozen, non-production ocean full-action repair pilot.

The script never ranks layouts, freezes a selection, or accesses final-test
trajectories.  It audits the existing frozen failure/control panel and the
original six-layout Poisson pilot only.
"""

from __future__ import annotations

import argparse
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
for path in (REPO_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mfsi.config import load_config
from mfsi.poisson_log import LogPoissonConfig, solve_log_conductance_poisson
from experiments.ocean_drifters.action import _read_csv, _write_csv
from experiments.ocean_drifters.experiment import OceanDriftersExperiment
from experiments.ocean_drifters.full_action import OceanWeightedPoissonPilot
from experiments.ocean_drifters.full_action_repair import (
    assemble_variational_system,
    assemble_variational_system_longdouble,
    cell_centers,
    decimal_cholesky_action,
    deterministic_coordinate_maps,
    enriched_basis,
    fixed_physical_gram,
    generalized_cutoff_actions,
    normalized_weights,
    old_equilibrated_cutoff_actions,
    solve_full_rank_ritz,
    structurally_reduced_system,
    transformed_system,
)


def _relative_change(left: float, right: float) -> float:
    if not np.isfinite(left) or not np.isfinite(right):
        return math.inf
    return abs(left - right) / max(abs(left), abs(right), np.finfo(float).tiny)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _case_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["design_index"]), int(row["source_time_index"])


def _frozen_panel(
    experiment: OceanDriftersExperiment,
    repair_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    selection = _read_csv(EXPERIMENT_DIR / "analysis/tables/poisson_pilot_selection.csv")
    if len(selection) != 6 or any(row["final_test_accessed"] != "False" for row in selection):
        raise RuntimeError("the original six-layout pilot freeze is invalid")
    production = _read_csv(
        EXPERIMENT_DIR / "analysis/tables/full_action_production_time.csv"
    )
    if len(production) != 68 * 181 or any(
        row["final_test_accessed"] != "False" for row in production
    ):
        raise RuntimeError("the completed production diagnostics are incomplete or unlocked")
    source_by_key = {
        (int(row["design_index"]), int(row["source_time_index"])): row
        for row in production
    }
    panel: dict[tuple[int, int], dict[str, Any]] = {}
    original_sources = np.rint(
        np.asarray(experiment.cfg["action"]["poisson_pilot"]["days"]) * 4.0
    ).astype(int)
    for selected in selection:
        design = int(selected["design_index"])
        for source in original_sources:
            key = (design, int(source))
            source_row = source_by_key[key]
            panel[key] = {
                "case_label": "original_frozen_poisson_pilot",
                "panel_role": "original_pilot",
                "design_index": design,
                "design_id": source_row["design_id"],
                "source_time_index": int(source),
                "day": float(source_row["day"]),
            }
    for frozen in repair_cfg["representative_cases"]:
        key = (int(frozen["design_index"]), int(frozen["source_time_index"]))
        source_row = source_by_key[key]
        label = str(frozen["case_label"])
        if key in panel:
            panel[key]["case_label"] += f";{label}"
            panel[key]["panel_role"] += ";repair_representative"
        else:
            panel[key] = {
                "case_label": label,
                "panel_role": "repair_representative",
                "design_index": key[0],
                "design_id": source_row["design_id"],
                "source_time_index": key[1],
                "day": float(source_row["day"]),
            }
        panel[key]["old_production_action"] = float(source_row["full_action_density"])
        panel[key]["old_maximum_relative_rank_action_change"] = float(
            source_row["maximum_relative_rank_action_change"]
        )
        panel[key]["old_tangent_full_inequality_valid"] = (
            source_row["tangent_full_inequality_valid"] == "True"
        )
    for key, item in panel.items():
        source_row = source_by_key[key]
        item.setdefault("old_production_action", float(source_row["full_action_density"]))
        item.setdefault(
            "old_maximum_relative_rank_action_change",
            float(source_row["maximum_relative_rank_action_change"]),
        )
        item.setdefault(
            "old_tangent_full_inequality_valid",
            source_row["tangent_full_inequality_valid"] == "True",
        )
    return sorted(panel.values(), key=lambda row: (_case_key(row), row["case_label"]))


def _basis_indices(names: tuple[str, ...], order: int) -> np.ndarray:
    selected = []
    for index, name in enumerate(names):
        if name.startswith("gaussian_sensor_"):
            selected.append(index)
            continue
        pieces = name.removeprefix("cosine_x").split("_y")
        if len(pieces) == 2 and max(int(pieces[0]), int(pieces[1])) <= order:
            selected.append(index)
    return np.asarray(selected, dtype=int)


def _invariance_audit(
    stiffness: np.ndarray,
    forcing: np.ndarray,
    physical_gram: np.ndarray,
    baseline,
    repair_cfg: dict[str, Any],
) -> dict[str, Any]:
    action_changes: dict[str, float] = {}
    spectrum_changes: dict[str, float] = {}
    ranks: dict[str, int] = {}
    certifications: dict[str, bool] = {}
    for label, coordinate_map in deterministic_coordinate_maps(len(forcing)).items():
        transformed = solve_full_rank_ritz(
            *transformed_system(stiffness, forcing, physical_gram, coordinate_map),
            structural_relative_tolerance=float(
                repair_cfg["structural_relative_tolerance"]
            ),
            residual_tolerance=float(repair_cfg["linear_residual_tolerance"]),
            backward_error_tolerance=float(repair_cfg["backward_error_tolerance"]),
        )
        action_changes[label] = _relative_change(transformed.action, baseline.action)
        ranks[label] = transformed.structural_rank
        certifications[label] = bool(transformed.certified)
        if len(transformed.generalized_eigenvalues) == len(
            baseline.generalized_eigenvalues
        ):
            spectrum_changes[label] = float(
                np.linalg.norm(
                    transformed.generalized_eigenvalues
                    - baseline.generalized_eigenvalues
                )
                / max(
                    np.linalg.norm(baseline.generalized_eigenvalues),
                    np.finfo(float).tiny,
                )
            )
        else:
            spectrum_changes[label] = math.inf
    tolerance = float(repair_cfg["basis_invariance_relative_tolerance"])
    evaluable = bool(baseline.certified and all(certifications.values()))
    valid = bool(
        evaluable
        and all(value <= tolerance for value in action_changes.values())
        and all(value <= tolerance for value in spectrum_changes.values())
        and all(rank == baseline.structural_rank for rank in ranks.values())
    )
    return {
        "basis_invariance_status": (
            "pass" if valid else ("fail" if evaluable else "unresolved")
        ),
        "basis_invariance_valid": valid,
        "maximum_basis_action_relative_change": max(action_changes.values()),
        "maximum_basis_spectrum_relative_change": max(spectrum_changes.values()),
        "basis_action_relative_changes": action_changes,
        "basis_spectrum_relative_changes": spectrum_changes,
        "basis_transform_certifications": certifications,
    }


def _independent_solver_config(experiment: OceanDriftersExperiment, dx: float) -> LogPoissonConfig:
    source = experiment.cfg["action"]["poisson_solver_repair"]
    return LogPoissonConfig(
        dx=float(dx),
        iterative_relative_tolerance=float(source["iterative_relative_tolerance"]),
        physical_relative_tolerance=float(source["physical_relative_residual_tolerance"]),
        gauge_absolute_tolerance=float(source["gauge_absolute_tolerance"]),
        maximum_iterations=int(source["maximum_iterations"]),
        ilu_drop_tolerance=float(source["ilu_drop_tolerance"]),
        ilu_fill_factor=float(source["ilu_fill_factor"]),
        direct_maximum_cells=int(source["direct_maximum_cells"]),
        iterative_solver=str(source["iterative_solver"]),
    )


def _summary_row(row: dict[str, Any]) -> dict[str, Any]:
    array_fields = {
        "generalized_eigenvalues",
        "generalized_forcing_coefficients",
        "generalized_action_contributions",
        "generalized_cumulative_action",
        "basis_action_relative_changes",
        "basis_spectrum_relative_changes",
        "basis_transform_certifications",
    }
    return {
        key: json.dumps(_json_ready(value), separators=(",", ":"))
        if key in array_fields else _json_ready(value)
        for key, value in row.items()
        if key not in {"stiffness", "forcing_vector", "physical_gram"}
    }


def _detail_row(row: dict[str, Any]) -> dict[str, Any]:
    """Keep diagnostic arrays as JSON arrays in the canonical detail artifact."""
    return _json_ready({
        key: value
        for key, value in row.items()
        if key not in {"stiffness", "forcing_vector", "physical_gram"}
    })


def _classification(
    panel: list[dict[str, Any]], rows: list[dict[str, Any]], repair_cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    difficult = [
        item for item in panel
        if item["case_label"] in {
            "median_rank_failure",
            "maximum_rank_failure",
            "maximum_lower_bound_failure",
        }
    ]
    fine = tuple(int(value) for value in repair_cfg["quadrature_resolutions"][-1])
    maximum_order = max(int(value) for value in repair_cfg["mode_orders"])
    output = []
    for case in difficult:
        target = next(
            row for row in rows
            if _case_key(row) == _case_key(case)
            and (int(row["grid_nx"]), int(row["grid_ny"])) == fine
            and int(row["trial_space_order"]) == maximum_order
        )
        common = bool(
            target["basis_invariance_valid"]
            and target["precision_convergence_status"] == "pass"
            and target["nested_mode_order_status"] == "pass"
            and target["quadrature_convergence_status"] == "pass"
            and target["independent_solver_agreement_status"] == "pass"
        )
        near_null_material = bool(
            np.isfinite(float(target["condition_proxy"]))
            and float(target["condition_proxy"]) >= 1.0e12
            and target["small_generalized_modes_carry_material_action"]
        )
        if common and float(case["old_maximum_relative_rank_action_change"]) > 0.05:
            category = "A_coordinate_rank_selection_artifact"
            reason = (
                "the untruncated result is certified across basis, precision, "
                "quadrature, nested spaces, and the independent solver"
            )
        elif (
            not target["double_precision_certified"]
            and target["high_precision_certified"]
            and target["independent_solver_agreement_status"] == "pass"
        ):
            category = "B_precision_artifact"
            reason = "double precision fails while high precision and the grid solver agree"
        elif (
            target["quadrature_convergence_status"] == "fail"
            and target["basis_invariance_valid"]
        ):
            category = "unresolved_possible_C_quadrature_discretization_artifact"
            reason = "two-grid movement is material; more refinement is needed to establish convergence"
        elif common and near_null_material:
            category = "D_genuine_near_degenerate_scientific_operator"
            reason = "all required cross-checks agree and small generalized modes carry material action"
        else:
            category = "unresolved_not_yet_classifiable_as_A_B_C_or_D"
            reason = (
                "one or more required basis, precision, quadrature, nestedness, "
                "or independent-grid certificates is unavailable"
            )
        output.append({
            "case_label": case["case_label"],
            "design_index": case["design_index"],
            "design_id": case["design_id"],
            "source_time_index": case["source_time_index"],
            "day": case["day"],
            "classification": category,
            "reason": reason,
            "final_test_accessed": False,
        })
    return output


def _report(
    path: Path,
    panel: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
    elapsed: float,
) -> None:
    fine_rows = [
        row for row in rows
        if row["quadrature_level"] == "fine"
        and row["trial_space_order"] == max(item["trial_space_order"] for item in rows)
    ]
    representative_rows = [
        row for row in fine_rows if "repair_representative" in row["panel_role"]
    ]
    category_lines = "\n".join(
        f"- `{item['case_label']}` (`{item['design_id']}`, day {item['day']:g}): "
        f"**{item['classification']}** — {item['reason']}."
        for item in classifications
    )
    production_supported = bool(
        fine_rows
        and all(
            row["double_precision_certified"]
            and row["tangent_full_status"] == "pass"
            and row["basis_invariance_status"] == "pass"
            and row["nested_mode_order_status"] == "pass"
            and row["quadrature_convergence_status"] == "pass"
            and row["precision_convergence_status"] == "pass"
            and row["independent_solver_agreement_status"] == "pass"
            for row in fine_rows
        )
    )
    report = f"""# Ocean full-action invariant Ritz repair pilot

## Decision

This frozen pilot evaluated {len(panel)} layout-time cases without ranking
layouts or accessing final-test trajectories. A repaired production run is
**{'numerically supported' if production_supported else 'not yet numerically supported'}**.
No production run was started.

## Implemented finite-dimensional problem

The enriched scalar space contains every nonconstant Neumann cosine mode
through order `p` and the exact four design-specific Gaussian sensor
observables. The assembly uses the existing sign convention
`K_ij = E_q[grad(phi_i).grad(phi_j)]` and
`f_i = -E_q[h(phi_i-E_q phi_i)]`. The primary result attempts the complete
gauge-fixed SPD solve `Kc=f`; no positive weighted mode is removed. Actions
with relative generalized-eigenvalue cutoffs `1e-10`, `1e-12`, and `1e-14`
are diagnostic-only.

## Verified previous implementation

The previous production backend excluded the constant cosine, centered each
remaining potential under `q`, assembled the same weighted stiffness and
opposite-sign load, and then scaled raw coordinates by
`D_ii=1/sqrt(K_ii)`. It eigendecomposed `D K D`, deleted eigenvalues below
`tau*lambda_max`, solved only in the retained space, and reported the retained
spectral sum. The potential was finally shifted to weighted mean zero. Thus
the old cutoff acted on a coordinate-equilibrated matrix rather than a fixed
physical generalized spectrum.

## Fixed physical norm and gauge

Structural dependence is determined with a q-independent uniform-reference
mean-zero H1 Gram matrix,
`H(u,v)=E_mu[(u-E_mu u)(v-E_mu v)] + (200 km)^2 E_mu[grad u.grad v]`,
on the frozen 129x69 reference grid. This treats scalar potentials modulo
constants and does not confuse a small weighted stiffness eigenvalue with a
redundant trial function. The constant cosine mode remains excluded.

## Frozen difficult-case classifications

{category_lines}

## Aggregate checks at order 7 on the fine grid

- Certified double-precision full solves: {sum(row['double_precision_certified'] for row in fine_rows)}/{len(fine_rows)}.
- Tangent lower bound: {sum(row['tangent_full_status'] == 'pass' for row in fine_rows)} pass, {sum(row['tangent_full_status'] == 'fail' for row in fine_rows)} fail, {sum(row['tangent_full_status'] == 'unresolved' for row in fine_rows)} unresolved.
- Basis invariance: {sum(row['basis_invariance_status'] == 'pass' for row in fine_rows)} pass, {sum(row['basis_invariance_status'] == 'fail' for row in fine_rows)} fail, {sum(row['basis_invariance_status'] == 'unresolved' for row in fine_rows)} unresolved.
- Nested Ritz monotonicity: {sum(row['nested_mode_order_status'] == 'pass' for row in fine_rows)} pass, {sum(row['nested_mode_order_status'] == 'fail' for row in fine_rows)} fail, {sum(row['nested_mode_order_status'] == 'unresolved' for row in fine_rows)} unresolved.
- Precision convergence: {sum(row['precision_convergence_status'] == 'pass' for row in fine_rows)} pass, {sum(row['precision_convergence_status'] == 'high_precision_only' for row in fine_rows)} high-precision-only, {sum(row['precision_convergence_status'] == 'unresolved' for row in fine_rows)} unresolved.
- Two-grid quadrature: {sum(row['quadrature_convergence_status'] == 'pass' for row in fine_rows)} pass, {sum(row['quadrature_convergence_status'] == 'fail' for row in fine_rows)} fail, {sum(row['quadrature_convergence_status'] == 'unresolved' for row in fine_rows)} unresolved.
- Independent FV agreement: {sum(row['independent_solver_agreement_status'] == 'pass' for row in fine_rows)} pass; it was run on {sum(row['independent_solver_agreement_status'] != 'not_run_nonrepresentative' for row in fine_rows)} representatives/controls.

For the six representatives/controls, extended precision normalizes log
weights and accumulates `K` and `f` in `longdouble`, then solves the complete
matrix with 80-digit Decimal Cholesky. This certified {sum(row['high_precision_certified'] for row in representative_rows)}/{len(representative_rows)} order-7 cases. The remaining cases
still developed nonpositive pivots, so the available precision does not
certify their assembled full-rank matrices. This is reported as unresolved,
not as permission to truncate those modes.

The independent reference is the existing unfloored arithmetic-face
finite-volume discretization with homogeneous no-flux boundaries and an
explicit weighted-mean gauge. It has no spectral-rank truncation. A rejected
or nonrepresentable grid solve is recorded as unresolved, never replaced by a
regularized value.

## Scope and locks

- Density flooring/clipping/thresholding: none.
- Operator/coercivity floor or ridge: none.
- Scientific validity gates relaxed: no.
- Final-test access: no.
- Layout ranking or selection: none.
- Production authorization emitted by this pilot: no.
- Elapsed time: {elapsed:.1f} seconds.

Machine-readable details are in
[`full_action_repair_pilot_details.json`](tables/full_action_repair_pilot_details.json)
and the flat audit table is
[`full_action_repair_pilot_summary.csv`](tables/full_action_repair_pilot_summary.csv).
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def run() -> dict[str, Any]:
    started = time.perf_counter()
    cfg = load_config(EXPERIMENT_DIR / "config.json")
    repair_cfg = cfg["action"]["full_action_repair_pilot"]
    if (
        repair_cfg.get("final_test_accessed") is not False
        or repair_cfg.get("production_run_authorized") is not False
        or float(repair_cfg["operator_floor"]) != 0.0
        or repair_cfg["density_floor_or_threshold_allowed"] is not False
    ):
        raise RuntimeError("repair-pilot safety locks changed")
    experiment = OceanDriftersExperiment(cfg)
    panel = _frozen_panel(experiment, repair_cfg)
    runner = OceanWeightedPoissonPilot(
        experiment,
        EXPERIMENT_DIR / "analysis",
        EXPERIMENT_DIR / "outputs/full_action_repair_pilot",
    )
    designs = np.asarray(sorted({int(row["design_index"]) for row in panel}), dtype=int)
    runner.designs = designs
    runner.local_by_design = {
        int(design): int(np.flatnonzero(runner.all_designs == design)[0])
        for design in designs
    }
    runner._build_soft_moment_penalties()

    bounds = np.asarray(cfg["scientific"]["domain_km"], dtype=np.float64)
    sigma = float(experiment.sensor_bank.sigma_km)
    orders = tuple(int(value) for value in repair_cfg["mode_orders"])
    maximum_order = max(orders)
    cutoffs = tuple(float(value) for value in repair_cfg["generalized_cutoff_diagnostics"])
    reference_points = cell_centers(
        bounds,
        tuple(int(value) for value in repair_cfg["physical_norm_reference_resolution"]),
    )
    physical_grams: dict[int, tuple[np.ndarray, tuple[str, ...]]] = {}
    for design in designs:
        reference_basis = enriched_basis(
            reference_points,
            bounds,
            experiment.sensor_bank.centers_km[design],
            sigma,
            maximum_order,
        )
        physical_grams[int(design)] = (
            fixed_physical_gram(
                reference_basis,
                length_scale=float(repair_cfg["physical_norm_length_scale_km"]),
            ),
            reference_basis.names,
        )

    panel_by_source: dict[int, list[dict[str, Any]]] = {}
    for case in panel:
        panel_by_source.setdefault(int(case["source_time_index"]), []).append(case)
    rows: list[dict[str, Any]] = []
    assembled: dict[tuple[int, int, int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    system_lookup: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    independent: dict[tuple[int, int, int, int], dict[str, Any]] = {}

    resolutions = [tuple(int(value) for value in item) for item in repair_cfg["quadrature_resolutions"]]
    for resolution_index, resolution in enumerate(resolutions):
        level = "coarse" if resolution_index == 0 else "fine"
        namespace = (
            "poisson_pilot_reference"
            if resolution_index == 0 else "full_action_production_reference"
        )
        for source, local_cases in sorted(panel_by_source.items()):
            local_designs = np.asarray(
                sorted({int(case["design_index"]) for case in local_cases}), dtype=int
            )
            runner.source_indices = np.asarray([source], dtype=int)
            runner.designs = local_designs
            points, dx, log_base, velocity = runner._reference_grid(
                resolution,
                source_indices=runner.source_indices,
                cache_namespace=namespace,
            )
            systems = runner._systems_for_grid(
                resolution, points, dx, log_base, velocity
            )
            case_by_design = {int(case["design_index"]): case for case in local_cases}
            for system in systems:
                design = int(system["design_index"])
                case = case_by_design[design]
                system_lookup[(resolution[0], resolution[1], design, source)] = system
                basis = enriched_basis(
                    points,
                    bounds,
                    experiment.sensor_bank.centers_km[design],
                    sigma,
                    maximum_order,
                )
                weights = normalized_weights(system["log_q_mass"])
                full_k, full_f = assemble_variational_system(
                    basis, weights, system["h"].ravel()
                )
                extended_system = None
                if level == "fine" and "repair_representative" in case["panel_role"]:
                    extended_system = assemble_variational_system_longdouble(
                        basis,
                        system["log_q_mass"],
                        system["h"].ravel(),
                    )
                reference_h, reference_names = physical_grams[design]
                if basis.names != reference_names:
                    raise RuntimeError("physical and quadrature trial order differ")
                if "repair_representative" in case["panel_role"]:
                    grid_result = solve_log_conductance_poisson(
                        system["log_q_mass"].reshape(system["h"].shape),
                        system["h"],
                        _independent_solver_config(experiment, dx),
                    )
                    independent[(resolution[0], resolution[1], design, source)] = grid_result
                for order in orders:
                    indices = _basis_indices(basis.names, order)
                    stiffness = full_k[np.ix_(indices, indices)]
                    forcing = full_f[indices]
                    physical_gram = reference_h[np.ix_(indices, indices)]
                    solve = solve_full_rank_ritz(
                        stiffness,
                        forcing,
                        physical_gram,
                        structural_relative_tolerance=float(
                            repair_cfg["structural_relative_tolerance"]
                        ),
                        residual_tolerance=float(repair_cfg["linear_residual_tolerance"]),
                        backward_error_tolerance=float(repair_cfg["backward_error_tolerance"]),
                    )
                    invariant_cutoff = generalized_cutoff_actions(solve, cutoffs)
                    old_cutoff = old_equilibrated_cutoff_actions(stiffness, forcing, cutoffs)
                    invariance = _invariance_audit(
                        stiffness, forcing, physical_gram, solve, repair_cfg
                    )
                    tangent = float(system["tangent_action_density"])
                    tangent_gap = solve.action - tangent
                    tangent_scale = max(abs(solve.action), abs(tangent), 1.0)
                    tangent_valid = bool(
                        solve.certified
                        and tangent_gap
                        >= -float(repair_cfg["tangent_full_relative_tolerance"])
                        * tangent_scale
                    )
                    maximum_eigenvalue = max(
                        solve.maximum_generalized_eigenvalue, np.finfo(float).tiny
                    )
                    small = (
                        solve.generalized_eigenvalues / maximum_eigenvalue < 1.0e-10
                    ) & (solve.generalized_eigenvalues > 0.0)
                    small_action = float(np.nansum(solve.generalized_contributions[small]))
                    small_fraction = (
                        small_action / solve.spectral_action
                        if np.isfinite(solve.spectral_action) and solve.spectral_action > 0.0
                        else math.nan
                    )
                    row = {
                        **case,
                        "quadrature_level": level,
                        "quadrature_method": "uniform_cell_center_projected_law",
                        "grid_nx": resolution[0],
                        "grid_ny": resolution[1],
                        "dx_km": float(dx),
                        "trial_space_order": order,
                        "raw_basis_function_count": len(indices),
                        "structural_rank_under_H": solve.structural_rank,
                        "physical_norm": repair_cfg["physical_norm"],
                        "physical_norm_length_scale_km": float(
                            repair_cfg["physical_norm_length_scale_km"]
                        ),
                        "tangent_action": tangent,
                        "full_untruncated_ritz_action": solve.action,
                        "c_transpose_K_c": solve.action_energy,
                        "f_transpose_c": solve.action_load,
                        "ritz_identity_relative_error": solve.ritz_identity_relative_error,
                        "linear_relative_residual": solve.relative_residual,
                        "linear_backward_error": solve.backward_error,
                        "double_precision_certified": solve.certified,
                        "factorization": solve.factorization,
                        "solve_failure_reason": solve.failure_reason,
                        "generalized_eigenvalues": solve.generalized_eigenvalues,
                        "generalized_forcing_coefficients": solve.generalized_forcing,
                        "generalized_action_contributions": solve.generalized_contributions,
                        "generalized_cumulative_action": solve.cumulative_action,
                        "generalized_spectral_action": solve.spectral_action,
                        "generalized_spectral_identity_relative_error": solve.spectral_identity_relative_error,
                        "minimum_generalized_eigenvalue": solve.minimum_generalized_eigenvalue,
                        "maximum_generalized_eigenvalue": solve.maximum_generalized_eigenvalue,
                        "condition_proxy": solve.condition_proxy,
                        "small_generalized_mode_action": small_action,
                        "small_generalized_mode_action_fraction": small_fraction,
                        "small_generalized_modes_carry_material_action": bool(
                            np.isfinite(small_fraction) and small_fraction > 0.05
                        ),
                        **{
                            f"action_truncated_{tolerance:.0e}".replace("e-", "e"): action
                            for tolerance, action in invariant_cutoff.items()
                        },
                        **{
                            f"old_equilibrated_action_truncated_{tolerance:.0e}".replace("e-", "e"): action
                            for tolerance, action in old_cutoff.items()
                        },
                        "tangent_full_absolute_gap": tangent_gap,
                        "tangent_full_relative_gap": tangent_gap / tangent_scale,
                        "tangent_full_ratio": (
                            tangent / solve.action
                            if np.isfinite(solve.action) and solve.action != 0.0 else math.nan
                        ),
                        "tangent_full_status": (
                            "pass" if tangent_valid
                            else ("fail" if solve.certified else "unresolved")
                        ),
                        **invariance,
                        "precision_used": "float64",
                        "high_precision_assembly": "not_checked",
                        "high_precision_certified": False,
                        "high_precision_action": math.nan,
                        "high_precision_relative_residual": math.nan,
                        "high_precision_failure_reason": "",
                        "precision_relative_action_change": math.nan,
                        "precision_convergence_status": "not_checked",
                        "mode_order_monotonicity_status": "not_checked",
                        "nested_mode_order_status": "not_checked",
                        "quadrature_K_relative_change": math.nan,
                        "quadrature_f_relative_change": math.nan,
                        "quadrature_minimum_eigenvalue_relative_change": math.nan,
                        "quadrature_dominant_contribution_relative_change": math.nan,
                        "quadrature_action_relative_change": math.nan,
                        "quadrature_convergence_status": "not_checked",
                        "grid_reference_action": math.nan,
                        "grid_reference_converged": False,
                        "grid_reference_physical_residual_valid": False,
                        "independent_solver_relative_action_change": math.nan,
                        "independent_solver_agreement_status": "not_checked",
                        "density_modified": False,
                        "operator_floor": 0.0,
                        "regularized_estimand": False,
                        "production_run_authorized": False,
                        "final_test_accessed": False,
                    }
                    rows.append(row)
                    assembled[(resolution[0], resolution[1], design, source, order)] = (
                        stiffness, forcing
                    )
                    if level == "fine" and order == maximum_order:
                        if extended_system is not None:
                            extended_k, extended_f = extended_system
                            selected_k = extended_k[np.ix_(indices, indices)]
                            selected_f = extended_f[indices]
                            _, _, structural = structurally_reduced_system(
                                stiffness,
                                forcing,
                                physical_gram,
                                structural_relative_tolerance=float(
                                    repair_cfg["structural_relative_tolerance"]
                                ),
                            )
                            transform = np.asarray(
                                structural.transform, dtype=np.longdouble
                            )
                            reduced_k = transform.T @ selected_k @ transform
                            reduced_k = 0.5 * (reduced_k + reduced_k.T)
                            reduced_f = transform.T @ selected_f
                            assembly_label = (
                                "longdouble log-weight normalization and matrix accumulation; "
                                "float64 trial-function samples"
                            )
                        else:
                            reduced_k, reduced_f, _ = structurally_reduced_system(
                                stiffness,
                                forcing,
                                physical_gram,
                                structural_relative_tolerance=float(
                                    repair_cfg["structural_relative_tolerance"]
                                ),
                            )
                            assembly_label = "float64 assembled matrix"
                        high = decimal_cholesky_action(
                            reduced_k,
                            reduced_f,
                            decimal_digits=int(repair_cfg["high_precision_decimal_digits"]),
                        )
                        row["precision_used"] = (
                            "float64 primary plus Decimal algebra certification"
                        )
                        row["high_precision_assembly"] = assembly_label
                        row["high_precision_certified"] = bool(high["certified"])
                        row["high_precision_action"] = float(high["action"])
                        row["high_precision_relative_residual"] = float(
                            high["relative_residual"]
                        )
                        row["high_precision_failure_reason"] = str(
                            high["failure_reason"]
                        )
                        row["precision_relative_action_change"] = _relative_change(
                            solve.action, float(high["action"])
                        )
                        precision_valid = bool(
                            solve.certified
                            and high["certified"]
                            and row["precision_relative_action_change"]
                            <= float(repair_cfg["precision_relative_tolerance"])
                        )
                        if precision_valid:
                            precision_status = "pass"
                        elif high["certified"] and not solve.certified:
                            precision_status = "high_precision_only"
                        elif high["certified"] and solve.certified:
                            precision_status = "fail"
                        else:
                            precision_status = "unresolved"
                        row["precision_convergence_status"] = precision_status
                print(
                    f"[ocean invariant repair] grid={resolution[0]}x{resolution[1]} "
                    f"design={design} source={source}",
                    flush=True,
                )

    row_by_key = {
        (
            int(row["grid_nx"]), int(row["grid_ny"]),
            int(row["design_index"]), int(row["source_time_index"]),
            int(row["trial_space_order"]),
        ): row
        for row in rows
    }
    coarse, fine = resolutions[0], resolutions[-1]
    for case in panel:
        design, source = _case_key(case)
        for resolution in resolutions:
            local = [
                row_by_key[(resolution[0], resolution[1], design, source, order)]
                for order in orders
            ]
            actions = np.asarray([row["full_untruncated_ritz_action"] for row in local])
            certified = all(row["double_precision_certified"] for row in local)
            tolerance = float(repair_cfg["nested_action_relative_tolerance"])
            monotone = bool(
                certified
                and np.all(
                    np.diff(actions)
                    >= -tolerance * np.maximum(np.abs(actions[:-1]), np.abs(actions[1:]))
                )
            )
            status = "pass" if monotone else ("fail" if certified else "unresolved")
            for row in local:
                row["mode_order_monotonicity_status"] = status
                row["nested_mode_order_status"] = status
        for order in orders:
            coarse_row = row_by_key[(coarse[0], coarse[1], design, source, order)]
            fine_row = row_by_key[(fine[0], fine[1], design, source, order)]
            coarse_k, coarse_f = assembled[(coarse[0], coarse[1], design, source, order)]
            fine_k, fine_f = assembled[(fine[0], fine[1], design, source, order)]
            k_change = float(
                np.linalg.norm(fine_k - coarse_k)
                / max(np.linalg.norm(fine_k), np.finfo(float).tiny)
            )
            f_change = float(
                np.linalg.norm(fine_f - coarse_f)
                / max(np.linalg.norm(fine_f), np.finfo(float).tiny)
            )
            lambda_change = _relative_change(
                coarse_row["minimum_generalized_eigenvalue"],
                fine_row["minimum_generalized_eigenvalue"],
            )
            coarse_dominant = float(np.nanmax(coarse_row["generalized_action_contributions"]))
            fine_dominant = float(np.nanmax(fine_row["generalized_action_contributions"]))
            dominant_change = _relative_change(coarse_dominant, fine_dominant)
            action_change = _relative_change(
                coarse_row["full_untruncated_ritz_action"],
                fine_row["full_untruncated_ritz_action"],
            )
            quadrature_valid = bool(
                coarse_row["double_precision_certified"]
                and fine_row["double_precision_certified"]
                and action_change <= float(repair_cfg["quadrature_relative_tolerance"])
            )
            quadrature_evaluable = bool(
                coarse_row["double_precision_certified"]
                and fine_row["double_precision_certified"]
            )
            status = (
                "pass" if quadrature_valid
                else ("fail" if quadrature_evaluable else "unresolved")
            )
            for row in (coarse_row, fine_row):
                row["quadrature_K_relative_change"] = k_change
                row["quadrature_f_relative_change"] = f_change
                row["quadrature_minimum_eigenvalue_relative_change"] = lambda_change
                row["quadrature_dominant_contribution_relative_change"] = dominant_change
                row["quadrature_action_relative_change"] = action_change
                row["quadrature_convergence_status"] = status
        for resolution in resolutions:
            grid = independent.get((resolution[0], resolution[1], design, source))
            if grid is None:
                continue
            row = row_by_key[(resolution[0], resolution[1], design, source, maximum_order)]
            grid_action = float(grid.get("action", math.nan))
            row["grid_reference_action"] = grid_action
            row["grid_reference_converged"] = bool(grid.get("converged", False))
            row["grid_reference_physical_residual_valid"] = bool(
                grid.get("physical_residual_valid", False)
            )
            agreement = _relative_change(row["full_untruncated_ritz_action"], grid_action)
            row["independent_solver_relative_action_change"] = agreement
            valid = bool(
                row["double_precision_certified"]
                and row["grid_reference_converged"]
                and row["grid_reference_physical_residual_valid"]
                and agreement <= float(repair_cfg["independent_solver_relative_tolerance"])
            )
            grid_evaluable = bool(
                row["double_precision_certified"]
                and row["grid_reference_converged"]
                and row["grid_reference_physical_residual_valid"]
                and np.isfinite(grid_action)
            )
            row["independent_solver_agreement_status"] = (
                "pass" if valid else ("fail" if grid_evaluable else "unresolved")
            )
        fine_row = row_by_key[(fine[0], fine[1], design, source, maximum_order)]
        if "repair_representative" not in case["panel_role"]:
            fine_row["independent_solver_agreement_status"] = "not_run_nonrepresentative"

    classifications = _classification(panel, rows, repair_cfg)
    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": 1,
        "stage": "full_action_repair_pilot",
        "panel_case_count": len(panel),
        "row_count": len(rows),
        "mathematical_formulation": {
            "stiffness": "K_ij=E_q[grad(phi_i).grad(phi_j)]",
            "forcing": "f_i=-E_q[h(phi_i-E_q phi_i)]",
            "primary_solve": "full structurally independent SPD solve Kc=f without positive-mode truncation",
            "action": "c^T K c = f^T c",
            "gauge": "constant cosine excluded; scalar potentials represented modulo constants",
            "physical_norm": repair_cfg["physical_norm"],
        },
        "panel": panel,
        "classifications": classifications,
        "rows": [_detail_row(row) for row in rows],
        "density_modified": False,
        "operator_floor": 0.0,
        "production_run_authorized": False,
        "selection_performed": False,
        "final_test_accessed": False,
        "elapsed_seconds": elapsed,
    }
    details_path = experiment._resolve(repair_cfg["details_json"])
    summary_path = experiment._resolve(repair_cfg["summary_csv"])
    report_path = experiment._resolve(repair_cfg["report"])
    _write_json(details_path, payload)
    _write_csv(summary_path, [_summary_row(row) for row in rows])
    _report(report_path, panel, rows, classifications, elapsed)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen ocean invariant full-action repair pilot."
    )
    parser.add_argument(
        "--acknowledge-no-production",
        action="store_true",
        help="Required acknowledgement that this command cannot run production.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.acknowledge_no_production:
        raise SystemExit("pass --acknowledge-no-production to run the frozen pilot")
    payload = run()
    print(
        f"wrote {payload['row_count']} invariant repair rows for "
        f"{payload['panel_case_count']} frozen cases"
    )


if __name__ == "__main__":
    main()
