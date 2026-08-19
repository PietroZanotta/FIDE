"""Audit the frozen post-dispersion window with explicit regularization.

This is a bounded diagnostic, never a production or layout-ranking run.  The
unregularized direct-QR Ritz action remains the primary reference.  Positive
operator floors solve the explicitly different conductivity-regularized PDE
documented in ``post_dispersion_regularization.py``.
"""

from __future__ import annotations

import os

try:
    _available_cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, _available_cpus[:8])
except (AttributeError, OSError):
    pass
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import argparse
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

from mfsi.cache import write_json_atomic
from mfsi.config import load_config
from mfsi.poisson import PoissonConfig
from mfsi.poisson_tesseract import (
    solve_weighted_poisson_batch_tesseract_diagnostics,
)
from experiments.ocean_drifters.action import _read_csv, _write_csv
from experiments.ocean_drifters.direct_qr_ritz import (
    prepare_direct_ritz_basis,
    solve_prepared_direct_ritz,
)
from experiments.ocean_drifters.experiment import OceanDriftersExperiment
from experiments.ocean_drifters.full_action import OceanWeightedPoissonPilot
from experiments.ocean_drifters.full_action_repair import (
    cell_centers,
    enriched_basis,
    fixed_physical_gram,
    normalized_weights,
)
from experiments.ocean_drifters.post_dispersion_regularization import (
    post_dispersion_source_indices,
    solve_conductivity_regularized_ritz,
    symmetric_relative_difference,
)


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
        number = float(value)
        if math.isnan(number):
            return "NaN"
        if number == math.inf:
            return "Infinity"
        if number == -math.inf:
            return "-Infinity"
        return number
    return value


def _maximum_pairwise_relative(values: list[float]) -> float:
    return max(
        symmetric_relative_difference(left, right)
        for index, left in enumerate(values)
        for right in values[index + 1 :]
    )


class PostDispersionRegularizationAudit:
    """Frozen six-layout, seven-time, two-grid numerical audit."""

    def __init__(self, experiment: OceanDriftersExperiment):
        self.experiment = experiment
        self.cfg = experiment.cfg["action"]["post_dispersion_regularization_audit"]
        self._validate_contract()
        self.analysis = EXPERIMENT_DIR / "analysis"
        self.pilot = OceanWeightedPoissonPilot(
            experiment,
            self.analysis,
            EXPERIMENT_DIR / "outputs/post_dispersion_regularization_audit",
        )
        selection = _read_csv(experiment._resolve(self.cfg["layout_selection_table"]))
        self.designs = np.asarray(
            [int(row["design_index"]) for row in selection], dtype=int
        )
        if len(self.designs) != int(self.cfg["layout_count"]):
            raise RuntimeError("post-dispersion layout freeze has changed")
        if not all(
            row.get("selection_frozen_before_full_action") == "True"
            for row in selection
        ):
            raise RuntimeError("post-dispersion layouts were not previously frozen")
        requested_days = np.asarray(self.cfg["audit_days"], dtype=np.float64)
        self.sources = np.rint(requested_days * 4.0).astype(int)
        actual_days = self.pilot.times[self.sources] * 45.0
        if not np.allclose(requested_days, actual_days, rtol=0.0, atol=1.0e-12):
            raise RuntimeError("audit days do not lie on the frozen ocean time grid")
        window_sources = post_dispersion_source_indices(
            self.pilot.times,
            start_day_inclusive=float(self.cfg["window_start_day_inclusive"]),
            end_day_inclusive=float(self.cfg["window_end_day_inclusive"]),
        )
        if not set(self.sources.tolist()).issubset(set(window_sources.tolist())):
            raise RuntimeError("an audit time lies outside the frozen post-dispersion window")
        if int(self.sources[0]) != int(window_sources[0]):
            raise RuntimeError("the audit must include the first post-dispersion time")
        if int(self.sources[-1]) != int(window_sources[-1]):
            raise RuntimeError("the audit must include the end of the frozen window")
        self.resolutions = tuple(
            tuple(int(value) for value in resolution)
            for resolution in self.cfg["grid_resolutions"]
        )
        self.floors = tuple(
            float(value) for value in self.cfg["operator_floor_relative_values"]
        )
        self.bounds = np.asarray(
            experiment.cfg["scientific"]["domain_km"], dtype=np.float64
        )
        self.sigma = float(experiment.sensor_bank.sigma_km)
        self.pilot.designs = self.designs.copy()
        self.pilot.source_indices = self.sources.copy()
        self.pilot.local_by_design = {
            int(design): int(np.flatnonzero(self.pilot.all_designs == design)[0])
            for design in self.designs
        }
        self.pilot.soft_penalty.clear()
        self.pilot.soft_penalty_dot.clear()
        self.pilot._build_soft_moment_penalties()

    def _validate_contract(self) -> None:
        locks = {
            "diagnostic_only": True,
            "precision": "float64",
            "production_run_authorized": False,
            "scientific_ranking_allowed": False,
            "final_test_accessed": False,
            "window_start_day_inclusive": 12.0,
            "window_end_day_inclusive": 45.0,
            "audit_days": [12.0, 12.5, 15.0, 22.5, 30.0, 37.5, 45.0],
            "previous_rejected_boundary_days": [10.25, 10.5],
            "operator_floor_relative_values": [2.0e-7, 2.0e-6, 2.0e-5],
            "primary_operator_floor_relative": 2.0e-7,
            "maximum_relative_regularization_bias": 0.05,
            "maximum_relative_floor_ladder_change": 0.05,
            "maximum_relative_grid_change": 0.1,
            "maximum_relative_fv_ritz_difference": 0.1,
        }
        changed = {
            key: (self.cfg.get(key), expected)
            for key, expected in locks.items()
            if self.cfg.get(key) != expected
        }
        if changed:
            raise RuntimeError(f"post-dispersion audit freeze changed: {changed}")
        inherited_floors = self.experiment.cfg["action"]["poisson_pilot"][
            "operator_floor_relative_values"
        ][1:]
        if inherited_floors != self.cfg["operator_floor_relative_values"]:
            raise RuntimeError("regularization ladder differs from the earlier pilot")

    def _prepared_bases(self, points: np.ndarray) -> dict[int, Any]:
        order = int(self.cfg["trial_order"])
        physical_points = cell_centers(
            self.bounds,
            tuple(int(value) for value in self.cfg[
                "physical_norm_reference_resolution"
            ]),
        )
        output: dict[int, Any] = {}
        for design in self.designs:
            centers = self.experiment.sensor_bank.centers_km[design]
            basis = enriched_basis(points, self.bounds, centers, self.sigma, order)
            physical_basis = enriched_basis(
                physical_points, self.bounds, centers, self.sigma, order
            )
            physical_gram = fixed_physical_gram(
                physical_basis,
                length_scale=float(self.cfg["physical_norm_length_scale_km"]),
            )
            output[int(design)] = prepare_direct_ritz_basis(
                basis,
                physical_gram,
                structural_relative_tolerance=float(
                    self.cfg["structural_relative_tolerance"]
                ),
            )
        return output

    def _run_grid(self, resolution: tuple[int, int]) -> list[dict[str, Any]]:
        namespace = (
            "full_action_production_reference"
            if resolution == self.resolutions[-1]
            else "direct_qr_repair_reference"
        )
        points, dx, log_base, velocity = self.pilot._reference_grid(
            resolution,
            source_indices=self.sources,
            cache_namespace=namespace,
        )
        systems = self.pilot._systems_for_grid(
            resolution, points, dx, log_base, velocity
        )
        if len(systems) != len(self.designs) * len(self.sources):
            raise RuntimeError("post-dispersion system panel is incomplete")
        if not all(bool(system["compatibility_valid"]) for system in systems):
            raise RuntimeError("post-dispersion forcing compatibility failed")
        prepared = self._prepared_bases(points)

        scientific: dict[tuple[int, int], dict[str, Any]] = {}
        for system in systems:
            design = int(system["design_index"])
            source = int(system["source_time_index"])
            weights = normalized_weights(system["log_q_mass"])
            direct = solve_prepared_direct_ritz(
                prepared[design], weights, np.asarray(system["h"]).ravel()
            ).direct
            tangent = float(system["tangent_action_density"])
            tangent_scale = max(abs(tangent), abs(direct.action_qr), 1.0)
            scientific[(design, source)] = {
                "system": system,
                "weights": weights,
                "direct_action_qr": direct.action_qr,
                "direct_action_svd": direct.action_svd,
                "direct_qr_svd_relative_difference": (
                    direct.qr_svd_relative_discrepancy
                ),
                "direct_condition_number": direct.kappa_c,
                "direct_full_column_rank": direct.lapack_full_column_rank,
                "direct_numerical_success": bool(
                    direct.qr_success
                    and direct.svd_success
                    and direct.lapack_full_column_rank
                    and direct.qr_svd_relative_discrepancy
                    <= float(self.cfg["direct_qr_svd_relative_tolerance"])
                ),
                "tangent_action": tangent,
                "direct_tangent_lower_bound_valid": bool(
                    tangent
                    <= direct.action_qr
                    + float(self.cfg["tangent_full_relative_tolerance"])
                    * tangent_scale
                ),
            }

        rows: list[dict[str, Any]] = []
        for floor in self.floors:
            native = solve_weighted_poisson_batch_tesseract_diagnostics(
                np.stack([system["q"] for system in systems]),
                np.stack([system["h"] for system in systems]),
                PoissonConfig(
                    dx=dx,
                    operator_floor_rel=floor,
                    cg_tol=float(self.cfg["cg_tolerance"]),
                    cg_maxiter=int(self.cfg["cg_maximum_iterations"]),
                    gauge_strength=float(self.cfg["gauge_strength"]),
                ),
            )
            for index, system in enumerate(systems):
                design = int(system["design_index"])
                source = int(system["source_time_index"])
                common = scientific[(design, source)]
                ritz = solve_conductivity_regularized_ritz(
                    prepared[design],
                    common["weights"],
                    np.asarray(system["h"]).ravel(),
                    floor,
                )
                stabilized_residual = float(
                    native["stabilized_relative_residual"][index]
                )
                gauge = abs(float(native["weighted_mean_potential"][index]))
                fv_success = bool(
                    native["converged"][index]
                    and stabilized_residual
                    <= float(self.cfg["maximum_relative_stabilized_pde_residual"])
                    and gauge
                    <= float(self.cfg["maximum_absolute_weighted_mean_potential"])
                )
                underflow = np.asarray(system["q"]).ravel() == 0.0
                rows.append({
                    "design_index": design,
                    "design_id": system["design_id"],
                    "source_time_index": source,
                    "day": float(system["day"]),
                    "grid_nx": resolution[0],
                    "grid_ny": resolution[1],
                    "dx_km": dx,
                    "operator_floor_relative": floor,
                    "regularized_equation": self.cfg["regularized_equation"],
                    "reported_action": self.cfg["reported_action"],
                    "direct_action_qr": common["direct_action_qr"],
                    "direct_action_svd": common["direct_action_svd"],
                    "direct_qr_svd_relative_difference": common[
                        "direct_qr_svd_relative_difference"
                    ],
                    "direct_condition_number": common["direct_condition_number"],
                    "direct_full_column_rank": common["direct_full_column_rank"],
                    "direct_numerical_success": common["direct_numerical_success"],
                    "tangent_action": common["tangent_action"],
                    "direct_tangent_lower_bound_valid": common[
                        "direct_tangent_lower_bound_valid"
                    ],
                    "regularized_ritz_physical_action": ritz.physical_action,
                    "regularized_ritz_regularizer_action": (
                        ritz.regularization_action
                    ),
                    "regularized_ritz_operator_action": ritz.operator_action,
                    "regularized_ritz_identity_relative_error": (
                        ritz.action_identity_relative_error
                    ),
                    "regularized_ritz_cholesky_spectral_relative_difference": (
                        ritz.cholesky_spectral_relative_difference
                    ),
                    "regularized_ritz_coefficient_relative_difference": (
                        ritz.coefficient_relative_difference
                    ),
                    "regularized_ritz_condition_number": ritz.condition_number,
                    "regularized_ritz_success": ritz.success,
                    "fv_physical_action": float(native["action"][index]),
                    "fv_iterations": int(native["iterations"][index]),
                    "fv_converged": bool(native["converged"][index]),
                    "fv_native_relative_residual": float(
                        native["native_relative_residual"][index]
                    ),
                    "fv_stabilized_relative_residual": stabilized_residual,
                    "fv_unregularized_relative_residual_diagnostic": float(
                        native["physical_relative_residual"][index]
                    ),
                    "fv_weighted_mean_potential": float(
                        native["weighted_mean_potential"][index]
                    ),
                    "fv_condition_proxy": float(
                        native["coefficient_condition_proxy"][index]
                    ),
                    "fv_regularized_solve_success": fv_success,
                    "underflow_cell_fraction": float(np.mean(underflow)),
                    "underflow_log10_probability_mass": float(
                        system["underflow_log10_probability_mass"]
                    ),
                    "compatibility_relative_residual": float(
                        system["compatibility_relative_residual"]
                    ),
                    "production_run": False,
                    "scientific_ranking_performed": False,
                    "final_test_accessed": False,
                })
        return rows

    def _cross_checks(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_key = {
            (
                int(row["design_index"]),
                int(row["source_time_index"]),
                int(row["grid_nx"]),
                float(row["operator_floor_relative"]),
            ): row
            for row in rows
        }
        coarse, fine = self.resolutions
        cases: list[dict[str, Any]] = []
        for design in self.designs:
            for source in self.sources:
                local_rows = [
                    by_key[(int(design), int(source), fine[0], floor)]
                    for floor in self.floors
                ]
                ritz_ladder = [
                    float(row["regularized_ritz_physical_action"])
                    for row in local_rows
                ]
                fv_ladder = [float(row["fv_physical_action"]) for row in local_rows]
                ritz_floor_change = _maximum_pairwise_relative(ritz_ladder)
                fv_floor_change = _maximum_pairwise_relative(fv_ladder)
                floor_ladder_valid = bool(
                    max(ritz_floor_change, fv_floor_change)
                    <= float(self.cfg["maximum_relative_floor_ladder_change"])
                )
                all_floor_valid = True
                for floor in self.floors:
                    low = by_key[(int(design), int(source), coarse[0], floor)]
                    high = by_key[(int(design), int(source), fine[0], floor)]
                    direct_grid = symmetric_relative_difference(
                        low["direct_action_qr"], high["direct_action_qr"]
                    )
                    ritz_grid = symmetric_relative_difference(
                        low["regularized_ritz_physical_action"],
                        high["regularized_ritz_physical_action"],
                    )
                    fv_grid = symmetric_relative_difference(
                        low["fv_physical_action"], high["fv_physical_action"]
                    )
                    regularization_bias = symmetric_relative_difference(
                        high["regularized_ritz_physical_action"],
                        high["direct_action_qr"],
                    )
                    fv_ritz = symmetric_relative_difference(
                        high["fv_physical_action"],
                        high["regularized_ritz_physical_action"],
                    )
                    direct_valid = bool(
                        low["direct_numerical_success"]
                        and high["direct_numerical_success"]
                        and high["direct_tangent_lower_bound_valid"]
                        and direct_grid
                        <= float(self.cfg["maximum_relative_grid_change"])
                    )
                    grid_valid = bool(
                        max(ritz_grid, fv_grid)
                        <= float(self.cfg["maximum_relative_grid_change"])
                    )
                    bias_valid = bool(
                        regularization_bias
                        <= float(self.cfg["maximum_relative_regularization_bias"])
                    )
                    agreement_valid = bool(
                        fv_ritz
                        <= float(self.cfg["maximum_relative_fv_ritz_difference"])
                    )
                    floor_valid = bool(
                        direct_valid
                        and low["regularized_ritz_success"]
                        and high["regularized_ritz_success"]
                        and low["fv_regularized_solve_success"]
                        and high["fv_regularized_solve_success"]
                        and grid_valid
                        and bias_valid
                        and agreement_valid
                        and floor_ladder_valid
                    )
                    all_floor_valid = all_floor_valid and floor_valid
                    for row in (low, high):
                        row.update({
                            "direct_grid_relative_action_change": direct_grid,
                            "regularized_ritz_grid_relative_action_change": ritz_grid,
                            "fv_grid_relative_action_change": fv_grid,
                            "regularization_bias_relative_to_unregularized_ritz": (
                                regularization_bias
                            ),
                            "fv_regularized_ritz_relative_action_difference": fv_ritz,
                            "regularized_ritz_floor_ladder_maximum_relative_change": (
                                ritz_floor_change
                            ),
                            "fv_floor_ladder_maximum_relative_change": fv_floor_change,
                            "direct_reference_valid": direct_valid,
                            "regularized_grid_valid": grid_valid,
                            "regularization_bias_valid": bias_valid,
                            "fv_ritz_agreement_valid": agreement_valid,
                            "floor_ladder_valid": floor_ladder_valid,
                            "combined_floor_case_valid": floor_valid,
                        })
                primary = by_key[(
                    int(design), int(source), fine[0],
                    float(self.cfg["primary_operator_floor_relative"]),
                )]
                cases.append({
                    "design_index": int(design),
                    "design_id": primary["design_id"],
                    "source_time_index": int(source),
                    "day": float(primary["day"]),
                    "direct_reference_valid": bool(primary["direct_reference_valid"]),
                    "all_regularization_floors_valid": all_floor_valid,
                    "primary_regularization_bias": float(
                        primary["regularization_bias_relative_to_unregularized_ritz"]
                    ),
                    "primary_fv_ritz_relative_difference": float(
                        primary["fv_regularized_ritz_relative_action_difference"]
                    ),
                    "maximum_regularized_grid_change": max(
                        max(
                            float(by_key[(int(design), int(source), fine[0], floor)][
                                "regularized_ritz_grid_relative_action_change"
                            ]),
                            float(by_key[(int(design), int(source), fine[0], floor)][
                                "fv_grid_relative_action_change"
                            ]),
                        )
                        for floor in self.floors
                    ),
                    "maximum_floor_ladder_change": max(
                        ritz_floor_change, fv_floor_change
                    ),
                })
        return cases

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        rows: list[dict[str, Any]] = []
        for resolution in self.resolutions:
            grid_started = time.perf_counter()
            rows.extend(self._run_grid(resolution))
            print(
                f"[ocean post-dispersion] grid={resolution[0]}x{resolution[1]} "
                f"elapsed={time.perf_counter() - grid_started:.1f}s",
                flush=True,
            )
        cases = self._cross_checks(rows)
        direct_valid = sum(bool(row["direct_reference_valid"]) for row in cases)
        combined_valid = sum(
            bool(row["all_regularization_floors_valid"]) for row in cases
        )
        contract_passed = bool(
            direct_valid == len(cases) and combined_valid == len(cases)
        )
        first_day = float(self.cfg["audit_days"][0])
        later_cases = [row for row in cases if float(row["day"]) > first_day]
        later_valid = sum(
            bool(
                row["direct_reference_valid"]
                and row["all_regularization_floors_valid"]
            )
            for row in later_cases
        )
        continue_development = bool(
            direct_valid == len(cases)
            and later_valid == len(later_cases)
            and all(
                float(row["day"]) == first_day
                for row in cases
                if not row["all_regularization_floors_valid"]
            )
        )
        primary_floor = float(self.cfg["primary_operator_floor_relative"])
        primary_fine = [
            row for row in rows
            if int(row["grid_nx"]) == self.resolutions[-1][0]
            and float(row["operator_floor_relative"]) == primary_floor
        ]
        summary = {
            "schema_version": 1,
            "decision": (
                "current_frozen_contract_passed"
                if contract_passed else (
                    "continue_ocean_but_revise_post_dispersion_boundary"
                    if continue_development else
                    "combined_redesign_not_sufficient"
                )
            ),
            "current_frozen_contract_passed": contract_passed,
            "continue_ocean_example_recommended": continue_development,
            "production_authorized": False,
            "window": {
                "start_day_inclusive": self.cfg["window_start_day_inclusive"],
                "first_audited_day": self.cfg["audit_days"][0],
                "end_day_inclusive": self.cfg["window_end_day_inclusive"],
                "audit_days": self.cfg["audit_days"],
                "previous_rejected_boundary_days": self.cfg[
                    "previous_rejected_boundary_days"
                ],
                "boundary_selection_provenance": self.cfg[
                    "boundary_selection_provenance"
                ],
            },
            "regularized_equation": self.cfg["regularized_equation"],
            "reported_action": self.cfg["reported_action"],
            "layout_count": len(self.designs),
            "case_count": len(cases),
            "detail_row_count": len(rows),
            "direct_reference_valid_count": direct_valid,
            "all_regularization_floors_valid_count": combined_valid,
            "cases_after_first_window_node_count": len(later_cases),
            "cases_after_first_window_node_valid_count": later_valid,
            "regularized_fv_solve_success_count": sum(
                bool(row["fv_regularized_solve_success"]) for row in rows
            ),
            "regularized_fv_solve_evaluated_count": len(rows),
            "regularized_ritz_success_count": sum(
                bool(row["regularized_ritz_success"]) for row in rows
            ),
            "maximum_primary_regularization_bias": max(
                float(row["primary_regularization_bias"]) for row in cases
            ),
            "maximum_primary_fv_ritz_relative_difference": max(
                float(row["primary_fv_ritz_relative_difference"]) for row in cases
            ),
            "maximum_regularized_grid_change": max(
                float(row["maximum_regularized_grid_change"]) for row in cases
            ),
            "maximum_floor_ladder_change": max(
                float(row["maximum_floor_ladder_change"]) for row in cases
            ),
            "maximum_primary_fv_stabilized_relative_residual": max(
                float(row["fv_stabilized_relative_residual"])
                for row in primary_fine
            ),
            "maximum_primary_fv_unregularized_relative_residual_diagnostic": max(
                float(row["fv_unregularized_relative_residual_diagnostic"])
                for row in primary_fine
            ),
            "thresholds": {
                key: self.cfg[key] for key in (
                    "maximum_relative_regularization_bias",
                    "maximum_relative_floor_ladder_change",
                    "maximum_relative_grid_change",
                    "maximum_relative_fv_ritz_difference",
                    "maximum_relative_stabilized_pde_residual",
                    "direct_qr_svd_relative_tolerance",
                )
            },
            "failed_cases": [
                row for row in cases
                if not (
                    row["direct_reference_valid"]
                    and row["all_regularization_floors_valid"]
                )
            ],
            "elapsed_seconds": time.perf_counter() - started,
            "precision": "float64",
            "diagnostic_only": True,
            "scientific_ranking_performed": False,
            "final_test_accessed": False,
        }
        details_path = self.experiment._resolve(self.cfg["details_table"])
        _write_csv(details_path, sorted(rows, key=lambda row: (
            float(row["day"]), int(row["design_index"]), int(row["grid_nx"]),
            float(row["operator_floor_relative"])
        )))
        write_json_atomic(
            self.experiment._resolve(self.cfg["summary_json"]),
            _json_ready(summary),
        )
        self._write_report(summary)
        return summary

    def _write_report(self, summary: dict[str, Any]) -> None:
        failed = summary["failed_cases"]
        failed_text = (
            "- None."
            if not failed
            else "\n".join(
                f"- `{row['design_id']}`, day {row['day']:.2f}: "
                f"direct valid={row['direct_reference_valid']}, all floors "
                f"valid={row['all_regularization_floors_valid']}, primary bias "
                f"{row['primary_regularization_bias']:.3%}, primary FV/Ritz "
                f"difference {row['primary_fv_ritz_relative_difference']:.3%}."
                for row in failed
            )
        )
        contract_passed = bool(summary["current_frozen_contract_passed"])
        continue_development = bool(
            summary["continue_ocean_example_recommended"]
        )
        if contract_passed:
            decision = (
                "**Keep working on the ocean example under the frozen "
                "post-dispersion redesign.**"
            )
        elif continue_development:
            decision = (
                "**Keep working on the ocean example, but reject the current "
                "first post-dispersion node.**"
            )
        else:
            decision = (
                "**The two changes do not provide enough evidence to continue "
                "with this example.**"
            )
        report = f"""# Ocean post-dispersion regularization audit

## Decision

{decision}

This bounded diagnostic used the six layouts selected before the original
Poisson pilot and seven fixed times from day {summary['window']['first_audited_day']:g}
through day {summary['window']['end_day_inclusive']:g}. The boundary provenance
is: {summary['window']['boundary_selection_provenance']}. It did not
rank layouts, access final-test trajectories, or run production.

## Estimand

The unregularized direct-QR Ritz action remains the primary scientific
reference. The regularized diagnostic solves

`{summary['regularized_equation']}`

and reports `{summary['reported_action']}`. Therefore a positive floor is
explicitly a different PDE, not a repaired residual for the unregularized PDE.
The regularizer energy is recorded separately in the detailed table.

## Frozen acceptance results

| Check | Result |
|---|---:|
| Unregularized direct references valid | {summary['direct_reference_valid_count']}/{summary['case_count']} cases |
| All three regularization floors pass the combined contract | {summary['all_regularization_floors_valid_count']}/{summary['case_count']} cases |
| Cases after the first window node that pass | {summary['cases_after_first_window_node_valid_count']}/{summary['cases_after_first_window_node_count']} cases |
| Regularized FV solves pass their own PDE residual | {summary['regularized_fv_solve_success_count']}/{summary['regularized_fv_solve_evaluated_count']} grid/floor cases |
| Regularized Ritz cross-checks succeed | {summary['regularized_ritz_success_count']}/{summary['regularized_fv_solve_evaluated_count']} grid/floor cases |
| Maximum primary-floor regularization bias | {summary['maximum_primary_regularization_bias']:.3%} (limit 5%) |
| Maximum primary-floor FV/Ritz difference | {summary['maximum_primary_fv_ritz_relative_difference']:.3%} (limit 10%) |
| Maximum regularized grid change | {summary['maximum_regularized_grid_change']:.3%} (limit 10%) |
| Maximum positive-floor ladder change | {summary['maximum_floor_ladder_change']:.3%} (limit 5%) |
| Maximum primary-floor stabilized FV residual | {summary['maximum_primary_fv_stabilized_relative_residual']:.3e} |
| Maximum residual against the unregularized PDE (diagnostic only) | {summary['maximum_primary_fv_unregularized_relative_residual_diagnostic']:.3e} |

The last residual is not an acceptance criterion: it demonstrates rather than
hides that regularization changes the equation.

## Failed cases

{failed_text}

## Scope and next decision

The previously tested day-10.25 and day-10.5 boundaries were rejected because
three of six layouts failed at each boundary. The present
`[{summary['window']['first_audited_day']:g}, {summary['window']['end_day_inclusive']:g}]`
boundary is a prospective replacement rather than a silently selected favorable
node. Passing this bounded audit supports continued development, but does not
authorize production or validate the original release-phase/full-horizon ocean
action.
"""
        path = self.experiment._resolve(self.cfg["report"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def main() -> None:
    parse_args()
    experiment = OceanDriftersExperiment(
        load_config(EXPERIMENT_DIR / "config.json")
    )
    summary = PostDispersionRegularizationAudit(experiment).run()
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
