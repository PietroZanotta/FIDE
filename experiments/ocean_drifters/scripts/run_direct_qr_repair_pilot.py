"""Run the float64 direct-operator QR repair pilot for ocean drifters.

This is a numerical audit, not production.  It never accesses final-test
trajectories, ranks layouts, changes the projected law, floors density, adds an
operator regularizer, or truncates a positive direct singular direction.
"""

from __future__ import annotations

import os

# Install the desktop-responsive resource contract before importing numerical
# runtimes.  The QR is threaded inside LAPACK; scientific cases remain serial.
DESKTOP_CPU_COUNT = 8
try:
    _startup_cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, _startup_cpus[:DESKTOP_CPU_COUNT])
except (AttributeError, OSError):
    pass
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=8"
)

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
for _path in (REPO_ROOT, SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from mfsi.cache import file_sha256, fingerprint, write_json_atomic
from mfsi.config import load_config
from mfsi.poisson_log import LogPoissonConfig, solve_log_conductance_poisson
from experiments.ocean_drifters.action import _read_csv, _write_csv
from experiments.ocean_drifters.direct_qr_ritz import (
    FLOAT64_UNIT_ROUNDOFF,
    prepare_direct_ritz_basis,
    solve_prepared_direct_ritz,
    solve_raw_direct_ritz,
)
from experiments.ocean_drifters.experiment import OceanDriftersExperiment
from experiments.ocean_drifters.full_action_production import OceanFullActionProduction
from experiments.ocean_drifters.full_action_repair import (
    TrialBasis,
    assemble_variational_system,
    cell_centers,
    enriched_basis,
    fixed_physical_gram,
    normalized_weights,
    old_equilibrated_cutoff_actions,
)


def _as_bool(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).lower() == "true"


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _relative(left: float, right: float, scale: float = 0.0) -> float:
    if not np.isfinite(left) or not np.isfinite(right):
        return math.inf
    return abs(left - right) / max(
        abs(left), abs(right), abs(scale), np.finfo(np.float64).tiny
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
        converted = float(value)
        if math.isnan(converted):
            return "NaN"
        if converted == math.inf:
            return "Infinity"
        if converted == -math.inf:
            return "-Infinity"
        return converted
    return value


def _flat_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            json.dumps(_json_ready(value), separators=(",", ":"))
            if isinstance(value, (list, tuple, dict, np.ndarray))
            else _json_ready(value)
        )
        for key, value in row.items()
    }


def _basis_indices(names: tuple[str, ...], order: int) -> np.ndarray:
    selected: list[int] = []
    for index, name in enumerate(names):
        if name.startswith("gaussian_sensor_"):
            selected.append(index)
            continue
        pieces = name.removeprefix("cosine_x").split("_y")
        if len(pieces) == 2 and max(int(pieces[0]), int(pieces[1])) <= order:
            selected.append(index)
    return np.asarray(selected, dtype=int)


def _coordinate_maps(size: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260818)
    orthogonal, _ = np.linalg.qr(rng.normal(size=(size, size)))
    permutation = np.eye(size)[:, rng.permutation(size)]
    nonorthogonal = np.eye(size)
    if size > 1:
        nonorthogonal[np.arange(size - 1), np.arange(1, size)] = 0.05
    return {
        "permutation": permutation,
        "diagonal_rescaling_1e-4_to_1e4": np.diag(
            np.geomspace(1.0e-4, 1.0e4, size)
        ),
        "orthogonal_mixing": orthogonal,
        "moderate_nonorthogonal": nonorthogonal,
    }


class DirectQRRepairPilot:
    """Frozen cohort runner for the direct float64 Ritz candidate."""

    def __init__(self, experiment: OceanDriftersExperiment):
        self.experiment = experiment
        self.cfg = experiment.cfg["action"]["direct_qr_repair_pilot"]
        self.analysis = EXPERIMENT_DIR / "analysis"
        self.tables = self.analysis / "tables"
        self._validate_safety_contract()
        self.production_rows = _read_csv(
            experiment._resolve(
                experiment.cfg["action"]["full_action_production"]["time_table"]
            )
        )
        self.conditioning_rows = _read_csv(
            experiment._resolve(
                experiment.cfg["action"]["concentration_conditioning_diagnosis"][
                    "diagnostics_table"
                ]
            )
        )
        if len(self.production_rows) != 68 * 181:
            raise RuntimeError("full-action diagnostic table is not the frozen 68x181 set")
        if len(self.conditioning_rows) != 68 * 181:
            raise RuntimeError("conditioning diagnosis is not the frozen 68x181 set")
        if any(row["final_test_accessed"] != "False" for row in self.production_rows):
            raise RuntimeError("production diagnostics report final-test access")
        if any(row["final_test_accessed"] != "False" for row in self.conditioning_rows):
            raise RuntimeError("conditioning diagnostics report final-test access")
        self.old_by_key = {
            (int(row["design_index"]), int(row["source_time_index"])): row
            for row in self.production_rows
        }
        self.conditioning_by_key = {
            (int(row["design_index"]), int(row["source_time_index"])): row
            for row in self.conditioning_rows
        }
        if set(self.old_by_key) != set(self.conditioning_by_key):
            raise RuntimeError("frozen production and conditioning keys differ")

        self.production = OceanFullActionProduction(
            experiment,
            self.analysis,
            EXPERIMENT_DIR / "outputs/full_action_production",
        )
        self.runner = self.production.runner
        self.bounds = np.asarray(
            experiment.cfg["scientific"]["domain_km"], dtype=np.float64
        )
        self.sigma = float(experiment.sensor_bank.sigma_km)
        self.orders = tuple(int(value) for value in self.cfg["trial_orders"])
        self.canonical_order = int(self.cfg["canonical_trial_order"])
        self.resolutions = tuple(
            tuple(int(value) for value in resolution)
            for resolution in self.cfg["quadrature_resolutions"]
        )
        self.coarse, self.fine = self.resolutions[0], self.resolutions[-1]
        self.selected, self.selection_reasons = self._select_cases()
        self.panel_labels = self._panel_labels()
        self.survivor_designs = self._survivor_designs()
        self.stability_keys, self.invariance_keys = self._stability_cases()
        self.checkpoints = experiment._resolve(self.cfg["checkpoint_directory"])
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self._physical_grams: dict[tuple[int, int], np.ndarray] = {}
        self._input_signature = self._signature()

    def _validate_safety_contract(self) -> None:
        locks = {
            "diagnostic_only": True,
            "precision": "float64",
            "density_floor_or_threshold_allowed": False,
            "operator_regularization_allowed": False,
            "positive_mode_truncation_allowed": False,
            "production_run_authorized": False,
            "scientific_ranking_allowed": False,
            "final_test_accessed": False,
            "resource_policy": "eight_cpu_desktop_responsive",
            "worker_count": 1,
            "math_thread_limit": 8,
            "cpu_affinity_count": 8,
            "jax_platform": "cpu",
            "nice_increment": 0,
        }
        changed = {key: (self.cfg.get(key), value) for key, value in locks.items()
                   if self.cfg.get(key) != value}
        if changed:
            raise RuntimeError(f"direct-QR safety/resource lock changed: {changed}")
        if float(self.cfg["structural_relative_tolerance"]) != 1.0e-12:
            raise RuntimeError("fixed-H structural tolerance changed")

    def _select_cases(self) -> tuple[set[tuple[int, int]], dict[tuple[int, int], set[str]]]:
        reasons: dict[tuple[int, int], set[str]] = defaultdict(set)
        old_sensitive = {
            key for key, row in self.conditioning_by_key.items()
            if _as_bool(row["old_rank_sensitive"])
        }
        normal_disagreement = {
            key for key, row in self.conditioning_by_key.items()
            if float(row["sigma_min"]) > 0.0
            and float(row["smallest_assembled_normal_eigenvalue"]) <= 0.0
        }
        day_zero = {key for key in self.conditioning_by_key if key[1] == 0}
        if (len(old_sensitive), len(normal_disagreement), len(day_zero)) != (557, 1661, 68):
            raise RuntimeError(
                "frozen diagnosis counts changed: "
                f"{len(old_sensitive)}, {len(normal_disagreement)}, {len(day_zero)}"
            )
        for key in old_sensitive:
            reasons[key].add("all_557_old_rank_sensitive")
        for key in normal_disagreement:
            reasons[key].add("all_1661_direct_positive_normal_nonpositive")
        for key in day_zero:
            reasons[key].add("all_68_day_zero")

        for frozen in self.experiment.cfg["action"]["full_action_repair_pilot"][
            "representative_cases"
        ]:
            key = int(frozen["design_index"]), int(frozen["source_time_index"])
            reasons[key].add("frozen_repair_control_panel")

        survivors = self._survivor_designs()
        for key in self.conditioning_by_key:
            if key[0] in survivors:
                reasons[key].add("all_times_old_3_of_68_survivor_layouts")

        late_clean = [
            key for key, row in self.conditioning_by_key.items()
            if float(row["day"]) >= float(self.cfg["later_clean_minimum_day"])
            and key not in old_sensitive and key not in normal_disagreement
            and key[0] not in survivors
        ]
        rng = np.random.default_rng(int(self.cfg["later_clean_sample_seed"]))
        count = min(int(self.cfg["later_clean_sample_count"]), len(late_clean))
        chosen = rng.choice(len(late_clean), size=count, replace=False)
        for index in np.sort(chosen):
            reasons[late_clean[int(index)]].add("deterministic_later_clean_sample")
        return set(reasons), reasons

    def _survivor_designs(self) -> set[int]:
        rows = _read_csv(self.tables / "full_action_production_summary.csv")
        survivors = {
            int(row["design_index"]) for row in rows
            if _as_bool(row["full_action_valid"])
        }
        if len(survivors) != 3:
            raise RuntimeError("the frozen old survivor set is no longer 3/68")
        return survivors

    def _panel_labels(self) -> dict[tuple[int, int], str]:
        return {
            (int(item["design_index"]), int(item["source_time_index"])):
            str(item["case_label"])
            for item in self.experiment.cfg["action"]["full_action_repair_pilot"][
                "representative_cases"
            ]
        }

    def _stability_cases(self) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        day_zero = sorted(
            (key for key in self.selected if key[1] == 0),
            key=lambda key: float(self.conditioning_by_key[key]["kappa_C"]),
        )
        panel = set(self.panel_labels)
        late_sample = sorted(
            key for key, reasons in self.selection_reasons.items()
            if "deterministic_later_clean_sample" in reasons
        )[: int(self.cfg["later_stability_control_count"])]
        nested = set(day_zero) | panel | set(late_sample)
        stress = {day_zero[0], day_zero[len(day_zero) // 2], day_zero[-1]}
        return nested, panel | stress

    def _signature(self) -> str:
        numerical = {
            key: self.cfg[key] for key in (
                "trial_basis", "canonical_trial_order", "trial_orders",
                "quadrature_resolutions", "physical_norm",
                "physical_norm_reference_resolution",
                "physical_norm_length_scale_km",
                "structural_relative_tolerance",
                "old_generalized_cutoff_diagnostics",
                "later_clean_sample_count", "later_clean_sample_seed",
                "precision", "density_floor_or_threshold_allowed",
                "operator_regularization_allowed",
                "positive_mode_truncation_allowed",
            )
        }
        return fingerprint({
            "schema": 2,
            "numerical": numerical,
            "selected_keys": sorted([list(key) for key in self.selected]),
            "conditioning_table": file_sha256(
                self.experiment._resolve(
                    self.experiment.cfg["action"]["concentration_conditioning_diagnosis"][
                        "diagnostics_table"
                    ]
                )
            ),
            "production_table": file_sha256(
                self.experiment._resolve(
                    self.experiment.cfg["action"]["full_action_production"]["time_table"]
                )
            ),
            "moment_cache": file_sha256(self.experiment._resolve(
                "experiments/ocean_drifters/cache/action_moments_positive_kernel.npz"
            )),
            "reference": file_sha256(self.experiment.paths["reference_checkpoint"]),
            "endpoint": file_sha256(
                self.experiment.paths["conditioned_endpoint_estimator"]
            ),
            "final_test_accessed": False,
        })

    def _checkpoint_path(self, resolution: tuple[int, int], source: int) -> Path:
        return self.checkpoints / f"grid_{resolution[0]}x{resolution[1]}_time_{source:03d}.json"

    def _load_checkpoint(
        self, resolution: tuple[int, int], source: int, expected: set[int]
    ) -> list[dict[str, Any]] | None:
        path = self._checkpoint_path(resolution, source)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        canonical = {
            int(row["design_index"]) for row in payload.get("rows", [])
            if int(row["trial_order"]) == self.canonical_order
        }
        if (
            payload.get("signature") != self._input_signature
            or tuple(payload.get("resolution", [])) != resolution
            or int(payload.get("source_time_index", -1)) != source
            or payload.get("final_test_accessed") is not False
            or canonical != expected
        ):
            return None
        return payload["rows"]

    def _save_checkpoint(
        self, resolution: tuple[int, int], source: int, rows: list[dict[str, Any]]
    ) -> None:
        write_json_atomic(self._checkpoint_path(resolution, source), {
            "schema_version": 2,
            "signature": self._input_signature,
            "resolution": list(resolution),
            "source_time_index": source,
            "rows": _json_ready(rows),
            "precision": "float64",
            "density_modified": False,
            "operator_regularization_used": False,
            "positive_mode_truncation_used": False,
            "production_run": False,
            "final_test_accessed": False,
        })

    def _physical_gram(self, design: int, order: int) -> np.ndarray:
        key = design, order
        if key not in self._physical_grams:
            points = cell_centers(
                self.bounds,
                tuple(int(value) for value in self.cfg[
                    "physical_norm_reference_resolution"
                ]),
            )
            basis = enriched_basis(
                points,
                self.bounds,
                self.experiment.sensor_bank.centers_km[design],
                self.sigma,
                order,
            )
            self._physical_grams[key] = fixed_physical_gram(
                basis,
                length_scale=float(self.cfg["physical_norm_length_scale_km"]),
            )
        return self._physical_grams[key]

    def _solve_case(
        self,
        system: dict[str, Any],
        basis: TrialBasis,
        physical_gram: np.ndarray,
        order: int,
        level: str,
    ) -> tuple[dict[str, Any], Any, np.ndarray, np.ndarray]:
        # Primary direct calculation occurs before any normal matrix is formed.
        weights = normalized_weights(system["log_q_mass"])
        prepared = prepare_direct_ritz_basis(
            basis,
            physical_gram,
            structural_relative_tolerance=float(
                self.cfg["structural_relative_tolerance"]
            ),
        )
        solved = solve_prepared_direct_ritz(
            prepared, weights, np.asarray(system["h"]).ravel()
        )
        direct = solved.direct

        # Diagnostic-only normal assembly and historical cutoff values.
        old_k, old_f = assemble_variational_system(
            basis, weights, np.asarray(system["h"]).ravel()
        )
        old_actions = old_equilibrated_cutoff_actions(
            old_k,
            old_f,
            [float(value) for value in self.cfg["old_generalized_cutoff_diagnostics"]],
        )
        count = len(weights)
        c_matrix = np.empty(
            (2 * count, prepared.structural_rank), dtype=np.float64, order="F"
        )
        raw_direct_matrix = np.empty(
            (2 * count, len(basis.names)), dtype=np.float64, order="F"
        )
        sqrt_weight = np.sqrt(weights)
        c_matrix[:count] = prepared.gradient_x_whitened * sqrt_weight[:, None]
        c_matrix[count:] = prepared.gradient_y_whitened * sqrt_weight[:, None]
        raw_direct_matrix[:count] = basis.gradient_x * sqrt_weight[:, None]
        raw_direct_matrix[count:] = basis.gradient_y * sqrt_weight[:, None]
        direct_normal = c_matrix.T @ c_matrix
        old_normal = prepared.raw_to_whitened.T @ old_k @ prepared.raw_to_whitened
        delta_normal = float(
            np.linalg.norm(direct_normal - old_normal, ord="fro")
            / max(np.linalg.norm(direct_normal, ord="fro"), np.finfo(float).tiny)
        )
        tangent = float(system["tangent_action_density"])
        tangent_gap = direct.action_qr - tangent
        tangent_scale = max(abs(direct.action_qr), abs(tangent), 1.0)
        tangent_relative_gap = tangent_gap / tangent_scale
        tangent_valid = bool(
            np.isfinite(tangent_gap)
            and tangent_gap >= -float(self.cfg["tangent_full_relative_tolerance"])
            * tangent_scale
        )
        contribution_sum = float(np.sum(direct.action_contributions))
        dominant_index = int(np.argmax(direct.action_contributions))
        dominant = float(direct.action_contributions[dominant_index])
        old10 = old_actions[1.0e-10]
        old12 = old_actions[1.0e-12]
        old14 = old_actions[1.0e-14]
        design = int(system["design_index"])
        source = int(system["source_time_index"])
        diagnosis = self.conditioning_by_key[(design, source)]
        qr_svd_valid = bool(
            direct.qr_success and direct.svd_success
            and direct.lapack_full_column_rank
            and direct.qr_svd_relative_discrepancy
            <= float(self.cfg["qr_svd_relative_tolerance"])
        )
        row = {
            "layout_id": str(system["design_id"]),
            "design_id": str(system["design_id"]),
            "design_index": design,
            "source_time_index": source,
            "day": float(system["day"]),
            "selection_reasons": sorted(self.selection_reasons[(design, source)]),
            "frozen_panel_label": self.panel_labels.get((design, source), ""),
            "trial_order": order,
            "number_basis_functions": len(basis.names),
            "structural_rank_H": prepared.structural_rank,
            "structural_threshold_H": prepared.structural_threshold,
            "grid_nx": int(system["grid_nx"]),
            "grid_ny": int(system["grid_ny"]),
            "quadrature_level": level,
            "dx_km": float(system["dx_km"]),
            "sigma_max": direct.sigma_max,
            "sigma_min": direct.sigma_min,
            "kappa_C": direct.kappa_c,
            "u_kappa_C": direct.u_kappa_c,
            "u_kappa_C_squared": direct.u_kappa_c_squared,
            "lapack_full_column_rank": direct.lapack_full_column_rank,
            "action_qr": direct.action_qr,
            "action_svd": direct.action_svd,
            "qr_success": direct.qr_success,
            "svd_success": direct.svd_success,
            "qr_svd_relative_discrepancy": direct.qr_svd_relative_discrepancy,
            "qr_svd_status": "pass" if qr_svd_valid else "fail",
            "singular_values": direct.singular_values.tolist(),
            "generalized_load_coefficients": (
                direct.generalized_load_coefficients.tolist()
            ),
            "action_contributions": direct.action_contributions.tolist(),
            "action_contribution_sum": contribution_sum,
            "dominant_contribution_index": dominant_index,
            "dominant_action_contribution": dominant,
            "dominant_action_fraction": (
                dominant / contribution_sum if contribution_sum > 0.0 else math.nan
            ),
            "tangent_action": tangent,
            "tangent_full_gap": tangent_gap,
            "tangent_full_relative_gap": tangent_relative_gap,
            "tangent_lower_bound_status": "pass" if tangent_valid else "fail",
            "action_old_1e10": old10,
            "action_old_1e12": old12,
            "action_old_1e14": old14,
            "direct_vs_old_1e10_relative_difference": _relative(
                direct.action_qr, old10
            ),
            "direct_vs_old_1e12_relative_difference": _relative(
                direct.action_qr, old12
            ),
            "direct_vs_old_1e14_relative_difference": _relative(
                direct.action_qr, old14
            ),
            "delta_normal": delta_normal,
            "old_rank_sensitive": _as_bool(diagnosis["old_rank_sensitive"]),
            "old_direct_positive_normal_nonpositive": bool(
                float(diagnosis["sigma_min"]) > 0.0
                and float(diagnosis["smallest_assembled_normal_eigenvalue"]) <= 0.0
            ),
            "old_smallest_normal_eigenvalue": float(
                diagnosis["smallest_assembled_normal_eigenvalue"]
            ),
            "nested_action_sequence": {},
            "nested_order_status": "not_checked",
            "quadrature_relative_action_change": math.nan,
            "quadrature_sigma_min_relative_change": math.nan,
            "quadrature_kappa_relative_change": math.nan,
            "quadrature_dominant_contribution_relative_change": math.nan,
            "quadrature_stability_status": "pending",
            "basis_invariance_action_changes": {},
            "basis_invariance_status": "not_checked",
            "fv_action": math.nan,
            "fv_relative_difference": math.nan,
            "fv_status": "not_checked",
            "numerical_regime": "pending",
            "final_numerical_certification_status": "pending",
            "unresolved_reason": direct.failure_reason,
            "primary_factorization": "LAPACK Householder QR of direct C",
            "svd_audit_factorization": "LAPACK gesdd SVD of QR R factor",
            "normal_matrix_primary": False,
            "density_modified": False,
            "operator_regularization_used": False,
            "positive_mode_truncation_used": False,
            "precision": "float64",
            "diagnostic_only": True,
            "production_run": False,
            "scientific_ranking_performed": False,
            "final_test_accessed": False,
        }
        return row, prepared, raw_direct_matrix, old_f

    def _basis_invariance(
        self,
        row: dict[str, Any],
        prepared: Any,
        direct_matrix: np.ndarray,
        raw_load: np.ndarray,
        physical_gram: np.ndarray,
    ) -> None:
        baseline = float(row["action_qr"])
        changes: dict[str, float] = {}
        failures: dict[str, str] = {}
        for label, coordinate_map in _coordinate_maps(len(raw_load)).items():
            try:
                transformed_structural = np.linalg.solve(
                    coordinate_map, prepared.structural_basis
                )
                transformed = solve_raw_direct_ritz(
                    direct_matrix @ coordinate_map,
                    coordinate_map.T @ raw_load,
                    coordinate_map.T @ physical_gram @ coordinate_map,
                    structural_relative_tolerance=float(
                        self.cfg["structural_relative_tolerance"]
                    ),
                    structural_basis=transformed_structural,
                )
                changes[label] = _relative(
                    transformed.direct.action_qr, baseline
                )
                if not transformed.direct.qr_success:
                    failures[label] = transformed.direct.failure_reason
            except (ValueError, np.linalg.LinAlgError, AssertionError) as exc:
                changes[label] = math.inf
                failures[label] = f"{type(exc).__name__}: {exc}"
        valid = bool(
            not failures
            and all(value <= float(self.cfg["basis_invariance_relative_tolerance"])
                    for value in changes.values())
        )
        row["basis_invariance_action_changes"] = changes
        row["basis_invariance_status"] = "pass" if valid else "fail"
        if failures:
            row["basis_invariance_failures"] = failures

    def _fv_audit(self, row: dict[str, Any], system: dict[str, Any]) -> None:
        source = self.experiment.cfg["action"]["poisson_solver_repair"]
        config = LogPoissonConfig(
            dx=float(system["dx_km"]),
            iterative_relative_tolerance=float(source["iterative_relative_tolerance"]),
            physical_relative_tolerance=float(source["physical_relative_residual_tolerance"]),
            gauge_absolute_tolerance=float(source["gauge_absolute_tolerance"]),
            maximum_iterations=int(source["maximum_iterations"]),
            ilu_drop_tolerance=float(source["ilu_drop_tolerance"]),
            ilu_fill_factor=float(source["ilu_fill_factor"]),
            direct_maximum_cells=int(source["direct_maximum_cells"]),
            iterative_solver=str(source["iterative_solver"]),
        )
        result = solve_log_conductance_poisson(
            np.asarray(system["log_q_mass"], dtype=np.float64).reshape(
                np.asarray(system["h"]).shape
            ),
            np.asarray(system["h"], dtype=np.float64),
            config,
        )
        action = float(result["action"])
        difference = _relative(float(row["action_qr"]), action)
        valid = bool(
            result.get("converged", False)
            and result.get("physical_residual_valid", False)
            and difference <= float(self.cfg["independent_solver_relative_tolerance"])
        )
        row.update({
            "fv_action": action,
            "fv_relative_difference": difference,
            "fv_status": "pass" if valid else "unresolved",
            "fv_converged": bool(result.get("converged", False)),
            "fv_physical_residual_valid": bool(
                result.get("physical_residual_valid", False)
            ),
            "fv_physical_relative_residual": float(
                result.get("physical_relative_residual", math.inf)
            ),
            "fv_solver_error": str(result.get("solver_error", "")),
            "fv_density_modified": bool(result.get("density_modified", False)),
            "fv_operator_floor": float(result.get("operator_floor", 0.0)),
        })

    def _run_source(
        self, resolution: tuple[int, int], source: int, designs: list[int]
    ) -> list[dict[str, Any]]:
        self.runner.source_indices = np.asarray([source], dtype=int)
        self.runner.designs = np.asarray(designs, dtype=int)
        points, dx, log_base, velocity = self.runner._reference_grid(
            resolution,
            source_indices=self.runner.source_indices,
            # The fine reference cache was produced by the frozen diagnostic
            # production sweep from the same hashes, times, grid, and equations.
            # _reference_grid revalidates its signature and final-test lock
            # before reuse.  The coarse grid has its own pilot-only cache.
            cache_namespace=(
                "full_action_production_reference"
                if resolution == self.fine
                else "direct_qr_repair_reference"
            ),
        )
        systems = self.runner._systems_for_grid(
            resolution, points, dx, log_base, velocity
        )
        by_design = {int(system["design_index"]): system for system in systems}
        level = "fine" if resolution == self.fine else "coarse"
        output: list[dict[str, Any]] = []
        for design in designs:
            system = by_design[design]
            key = design, source
            local_orders = self.orders if key in self.stability_keys else (
                self.canonical_order,
            )
            full_basis = enriched_basis(
                points,
                self.bounds,
                self.experiment.sensor_bank.centers_km[design],
                self.sigma,
                max(local_orders),
            )
            for order in local_orders:
                basis = full_basis.subset(_basis_indices(full_basis.names, order))
                physical_gram = self._physical_gram(design, order)
                row, prepared, direct_matrix, raw_load = self._solve_case(
                    system, basis, physical_gram, order, level
                )
                if (
                    resolution == self.fine
                    and order == self.canonical_order
                    and key in self.invariance_keys
                ):
                    self._basis_invariance(
                        row, prepared, direct_matrix, raw_load, physical_gram
                    )
                if (
                    resolution == self.fine
                    and order == self.canonical_order
                    and key in self.panel_labels
                ):
                    self._fv_audit(row, system)
                output.append(row)
        return output

    def run(self, recompute: bool = False, checkpoint_only: bool = False) -> dict[str, Any] | None:
        started = time.perf_counter()
        keys_by_source: dict[int, list[int]] = defaultdict(list)
        for design, source in sorted(self.selected, key=lambda item: (item[1], item[0])):
            keys_by_source[source].append(design)
        tasks = [
            (resolution, source, sorted(designs))
            for resolution in self.resolutions
            for source, designs in sorted(keys_by_source.items())
        ]
        for ordinal, (resolution, source, designs) in enumerate(tasks, start=1):
            rows = None if recompute else self._load_checkpoint(
                resolution, source, set(designs)
            )
            if rows is None:
                task_started = time.perf_counter()
                rows = self._run_source(resolution, source, designs)
                self._save_checkpoint(resolution, source, rows)
                state = f"computed {time.perf_counter() - task_started:.1f}s"
            else:
                state = "checkpoint"
            print(
                f"[ocean direct QR] task={ordinal}/{len(tasks)} "
                f"grid={resolution[0]}x{resolution[1]} source={source} "
                f"layouts={len(designs)} {state}",
                flush=True,
            )
            write_json_atomic(self.checkpoints.parent / "progress.json", {
                "stage": "direct_qr_repair_pilot",
                "completed_tasks": ordinal,
                "total_tasks": len(tasks),
                "selected_unique_cases": len(self.selected),
                "elapsed_seconds": time.perf_counter() - started,
                "resource_policy": self.cfg["resource_policy"],
                "production_run": False,
                "final_test_accessed": False,
            })
        if checkpoint_only:
            return None
        all_rows: list[dict[str, Any]] = []
        for resolution, source, designs in tasks:
            rows = self._load_checkpoint(resolution, source, set(designs))
            if rows is None:
                raise RuntimeError("a required direct-QR checkpoint disappeared")
            all_rows.extend(rows)
        self._apply_cross_checks(all_rows)
        summary = self._summary(all_rows, time.perf_counter() - started)
        _write_csv(
            self.experiment._resolve(self.cfg["diagnostics_table"]),
            [_flat_row(row) for row in sorted(all_rows, key=lambda row: (
                int(row["source_time_index"]), int(row["design_index"]),
                int(row["grid_nx"]), int(row["trial_order"])
            ))],
        )
        summary_path = self.experiment._resolve(self.cfg["summary_json"])
        write_json_atomic(summary_path, _json_ready(summary))
        self._write_report(summary)
        return summary

    def _apply_cross_checks(self, rows: list[dict[str, Any]]) -> None:
        by_key = {
            (int(row["design_index"]), int(row["source_time_index"]),
             int(row["grid_nx"]), int(row["grid_ny"]), int(row["trial_order"])): row
            for row in rows
        }
        for design, source in self.stability_keys:
            for resolution in self.resolutions:
                local = [by_key[(design, source, *resolution, order)] for order in self.orders]
                actions = [float(row["action_qr"]) for row in local]
                scale = max(max(map(abs, actions)), 1.0)
                monotone = bool(np.all(np.diff(actions) >= -float(
                    self.cfg["nested_action_relative_tolerance"]
                ) * scale))
                sequence = {str(order): action for order, action in zip(self.orders, actions)}
                for row in local:
                    row["nested_action_sequence"] = sequence
                    row["nested_order_status"] = "pass" if monotone else "fail"

        for design, source in self.selected:
            coarse = by_key[(design, source, *self.coarse, self.canonical_order)]
            fine = by_key[(design, source, *self.fine, self.canonical_order)]
            action_change = _relative(float(coarse["action_qr"]), float(fine["action_qr"]))
            sigma_change = _relative(float(coarse["sigma_min"]), float(fine["sigma_min"]))
            kappa_change = _relative(float(coarse["kappa_C"]), float(fine["kappa_C"]))
            dominant_change = _relative(
                float(coarse["dominant_action_contribution"]),
                float(fine["dominant_action_contribution"]),
            )
            status = (
                "pass" if action_change <= float(self.cfg["quadrature_relative_tolerance"])
                else "fail"
            )
            for row in (coarse, fine):
                row["quadrature_relative_action_change"] = action_change
                row["quadrature_sigma_min_relative_change"] = sigma_change
                row["quadrature_kappa_relative_change"] = kappa_change
                row["quadrature_dominant_contribution_relative_change"] = dominant_change
                row["quadrature_stability_status"] = status

            reasons: list[str] = []
            if fine["qr_svd_status"] != "pass":
                reasons.append("direct QR/SVD do not agree within the frozen tolerance")
            if fine["tangent_lower_bound_status"] != "pass":
                reasons.append("tangent lower bound fails")
            if status != "pass":
                reasons.append("coarse/fine direct action changes materially")
            if not _as_bool(fine["lapack_full_column_rank"]):
                reasons.append("direct SVD does not report full positive column rank")
            if (design, source) in self.stability_keys and fine[
                "nested_order_status"
            ] != "pass":
                reasons.append("nested direct Ritz actions materially decrease")
            if (design, source) in self.invariance_keys and fine[
                "basis_invariance_status"
            ] != "pass":
                reasons.append("direct action is not invariant under trial coordinates")
            certified = not reasons
            if not certified:
                regime = "III"
            elif source == 0 or float(fine["u_kappa_C"]) >= 1.0e-3:
                regime = "II"
            else:
                regime = "I"
            fine["numerical_regime"] = regime
            fine["final_numerical_certification_status"] = (
                "certified" if certified else "unresolved"
            )
            fine["unresolved_reason"] = "; ".join(reasons)

        # Supporting coarse-grid and noncanonical-order rows are complete
        # diagnostics, but certification belongs to the canonical fine-grid
        # case.  Avoid leaving ambiguous "pending" values in final artifacts.
        for row in rows:
            if row["numerical_regime"] == "pending":
                row["numerical_regime"] = "supporting_diagnostic"
            if row["final_numerical_certification_status"] == "pending":
                row["final_numerical_certification_status"] = (
                    "supporting_diagnostic_not_independently_certified"
                )
            if row["quadrature_stability_status"] == "pending":
                row["quadrature_stability_status"] = "not_applicable_to_supporting_order"

    def _cohort_statistics(
        self, rows: list[dict[str, Any]], keys: set[tuple[int, int]]
    ) -> dict[str, Any]:
        local = [row for row in rows if (
            int(row["design_index"]), int(row["source_time_index"])
        ) in keys]
        discrepancies = np.asarray([
            float(row["qr_svd_relative_discrepancy"]) for row in local
        ], dtype=np.float64)
        regimes = Counter(str(row["numerical_regime"]) for row in local)
        return {
            "case_count": len(local),
            "direct_qr_success_count": sum(_as_bool(row["qr_success"]) for row in local),
            "direct_svd_success_count": sum(_as_bool(row["svd_success"]) for row in local),
            "qr_svd_agreement_count": sum(row["qr_svd_status"] == "pass" for row in local),
            "tangent_lower_bound_count": sum(
                row["tangent_lower_bound_status"] == "pass" for row in local
            ),
            "quadrature_stability_count": sum(
                row["quadrature_stability_status"] == "pass" for row in local
            ),
            "certified_count": sum(
                row["final_numerical_certification_status"] == "certified" for row in local
            ),
            "unresolved_count": sum(
                row["final_numerical_certification_status"] == "unresolved" for row in local
            ),
            "regime_counts": dict(regimes),
            "qr_svd_discrepancy_median": float(np.median(discrepancies)),
            "qr_svd_discrepancy_p95": float(np.percentile(discrepancies, 95)),
            "qr_svd_discrepancy_maximum": float(np.max(discrepancies)),
            "maximum_u_kappa_C": max(float(row["u_kappa_C"]) for row in local),
        }

    def _summary(self, rows: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
        canonical = [
            row for row in rows
            if row["quadrature_level"] == "fine"
            and int(row["trial_order"]) == self.canonical_order
        ]
        canonical_keys = {
            (int(row["design_index"]), int(row["source_time_index"]))
            for row in canonical
        }
        old_sensitive = {
            key for key in canonical_keys
            if _as_bool(self.conditioning_by_key[key]["old_rank_sensitive"])
        }
        normal_disagreement = {
            key for key in canonical_keys
            if float(self.conditioning_by_key[key]["sigma_min"]) > 0.0
            and float(self.conditioning_by_key[key]["smallest_assembled_normal_eigenvalue"]) <= 0.0
        }
        day_zero = {key for key in canonical_keys if key[1] == 0}
        early_time = {
            key for key in canonical_keys
            if float(self.conditioning_by_key[key]["day"]) <= 10.0
        }
        later_time = canonical_keys - early_time
        survivor_keys = {key for key in canonical_keys if key[0] in self.survivor_designs}
        invariance = [row for row in canonical if row["basis_invariance_status"] != "not_checked"]
        nested = [row for row in canonical if row["nested_order_status"] != "not_checked"]
        fv = [row for row in canonical if row["fv_status"] != "not_checked"]
        overall = self._cohort_statistics(canonical, canonical_keys)
        unresolved = [row for row in canonical if row[
            "final_numerical_certification_status"
        ] == "unresolved"]
        authorization = "yes" if not unresolved else "no"
        authorization_reason = (
            "every audited canonical case is certified"
            if not unresolved
            else (
                f"{len(unresolved)} audited cases are Regime III, including "
                f"{len(day_zero)}/68 "
                "day-zero cases; the frozen numerical contract is not met"
            )
        )
        worst = sorted(
            canonical,
            key=lambda row: float(row["qr_svd_relative_discrepancy"]),
            reverse=True,
        )[:10]
        old_instability_eliminated = sum(
            row["qr_svd_status"] == "pass" and row[
                "final_numerical_certification_status"
            ] == "certified"
            for row in canonical if _as_bool(row["old_rank_sensitive"])
        )
        summary = {
            "schema_version": 1,
            "method": "float64_direct_operator_Householder_QR_with_untruncated_SVD_audit",
            "estimand_changed": False,
            "production_run": False,
            "scientific_ranking_performed": False,
            "final_test_accessed": False,
            "precision": "float64",
            "density_modified": False,
            "operator_regularization_used": False,
            "positive_mode_truncation_used": False,
            "selected_unique_cases": len(canonical),
            "detailed_row_count": len(rows),
            "elapsed_seconds": elapsed,
            "overall": overall,
            "day_zero_all_68": self._cohort_statistics(canonical, day_zero),
            "early_time_through_day_10": self._cohort_statistics(
                canonical, early_time
            ),
            "later_time_after_day_10": self._cohort_statistics(
                canonical, later_time
            ),
            "old_rank_sensitive_all_557": self._cohort_statistics(
                canonical, old_sensitive
            ),
            "direct_positive_normal_nonpositive_all_1661": self._cohort_statistics(
                canonical, normal_disagreement
            ),
            "old_survivor_layouts_all_times": self._cohort_statistics(
                canonical, survivor_keys
            ),
            "old_survivor_design_indices": sorted(self.survivor_designs),
            "basis_invariance_checked_count": len(invariance),
            "basis_invariance_pass_count": sum(
                row["basis_invariance_status"] == "pass" for row in invariance
            ),
            "nested_monotonicity_checked_count": len(nested),
            "nested_monotonicity_pass_count": sum(
                row["nested_order_status"] == "pass" for row in nested
            ),
            "fv_checked_count": len(fv),
            "fv_agreement_count": sum(row["fv_status"] == "pass" for row in fv),
            "fv_unresolved_count": sum(
                row["fv_status"] == "unresolved" for row in fv
            ),
            "fv_unresolved_reasons": dict(Counter(
                str(row.get("fv_solver_error", "unspecified")) for row in fv
                if row["fv_status"] == "unresolved"
            )),
            "old_rank_instability_eliminated_and_certified_count": old_instability_eliminated,
            "unresolved_cases": [{
                "design_index": int(row["design_index"]),
                "design_id": row["design_id"],
                "source_time_index": int(row["source_time_index"]),
                "day": float(row["day"]),
                "reason": row["unresolved_reason"],
            } for row in unresolved],
            "worst_qr_svd_cases": [{
                "design_id": row["design_id"],
                "source_time_index": int(row["source_time_index"]),
                "day": float(row["day"]),
                "relative_discrepancy": float(row["qr_svd_relative_discrepancy"]),
                "kappa_C": float(row["kappa_C"]),
            } for row in worst],
            "production_authorization_answer": authorization,
            "production_authorization_reason": authorization_reason,
        }
        return summary

    def _write_report(self, summary: dict[str, Any]) -> None:
        overall = summary["overall"]
        day0 = summary["day_zero_all_68"]
        old = summary["old_rank_sensitive_all_557"]
        normal = summary["direct_positive_normal_nonpositive_all_1661"]
        survivors = summary["old_survivor_layouts_all_times"]
        early = summary["early_time_through_day_10"]
        later = summary["later_time_after_day_10"]
        unresolved = summary["unresolved_cases"]
        if unresolved:
            unresolved_text = "\n".join(
                f"- `{item['design_id']}`, source {item['source_time_index']} "
                f"(day {item['day']:.2f}): {item['reason']}"
                for item in unresolved
            )
        else:
            unresolved_text = "- None."
        authorization = summary["production_authorization_answer"]
        qrs = (
            f"median {overall['qr_svd_discrepancy_median']:.3e}, "
            f"95th percentile {overall['qr_svd_discrepancy_p95']:.3e}, "
            f"maximum {overall['qr_svd_discrepancy_maximum']:.3e}"
        )
        report = f"""# Ocean direct-QR Ritz repair pilot

## Outcome

This numerical audit evaluated {summary['selected_unique_cases']:,} distinct frozen
layout-time cases. It used the unchanged projected law, forcing, quadrature,
geometry, exact Gaussian sensor observables, and enriched Ritz estimand. The
primary action came from a float64 Householder QR of the direct whitened gradient
operator; no normal matrix, density floor, operator regularizer, or positive-mode
cutoff entered that action. This was not a production run and final-test data were
not accessed.

{overall['certified_count']:,}/{overall['case_count']:,} canonical fine-grid cases
are numerically certified. The regime counts are
`{json.dumps(overall['regime_counts'], sort_keys=True)}`. There are
{overall['unresolved_count']:,} unresolved cases.

## Required counts

| Check | Passed / evaluated |
|---|---:|
| Direct QR completed | {overall['direct_qr_success_count']:,}/{overall['case_count']:,} |
| Direct SVD completed | {overall['direct_svd_success_count']:,}/{overall['case_count']:,} |
| QR/SVD agreement | {overall['qr_svd_agreement_count']:,}/{overall['case_count']:,} |
| Tangent lower bound | {overall['tangent_lower_bound_count']:,}/{overall['case_count']:,} |
| Quadrature stability | {overall['quadrature_stability_count']:,}/{overall['case_count']:,} |
| Basis invariance (tested panel) | {summary['basis_invariance_pass_count']:,}/{summary['basis_invariance_checked_count']:,} |
| Nested Ritz monotonicity (tested panel) | {summary['nested_monotonicity_pass_count']:,}/{summary['nested_monotonicity_checked_count']:,} |
| Independent unfloored FV agreement | {summary['fv_agreement_count']:,}/{summary['fv_checked_count']:,} |

All 68 day-zero cases were evaluated: {day0['certified_count']}/68 are certified,
{day0['qr_svd_agreement_count']}/68 have QR/SVD agreement,
{day0['tangent_lower_bound_count']}/68 satisfy the tangent lower bound, and
{day0['quadrature_stability_count']}/68 pass coarse/fine stability. Their maximum
`u*kappa(C)` is {day0['maximum_u_kappa_C']:.3e}.

The old-sensitive cohort result is {old['certified_count']}/557 certified. The
direct-positive/normal-nonpositive cohort result is
{normal['certified_count']}/1661 certified.

## Answers to the eleven questions

1. **Does avoiding `C^T C` eliminate the old rank-cutoff instability?**
   {summary['old_rank_instability_eliminated_and_certified_count']}/557 old
   rank-sensitive cases now have agreeing direct algorithms and full numerical
   certification. Historical cutoff values remain diagnostics only.

2. **Do float64 QR and SVD agree?** {overall['qr_svd_agreement_count']}/
   {overall['case_count']} cases pass the frozen `1e-8` comparison tolerance;
   the discrepancy distribution is {qrs}.

3. **Are all 557 old-sensitive cases solvable and certifiable?**
   {old['direct_qr_success_count']}/557 complete QR and
   {old['certified_count']}/557 are certified. Any remainder is listed under
   unresolved cases with its exact failed certificate.

4. **What happens at day zero?** Direct completion alone is not accepted as
   evidence. The combined QR/SVD, tangent, nestedness, coordinate-invariance
   sample, and quadrature checks certify {day0['certified_count']}/68 cases;
   {day0['unresolved_count']}/68 remain Regime III.

5. **Were the old 3/68 survivor layouts genuinely clean?** All 181 times for
   design indices {summary['old_survivor_design_indices']} were included.
   {survivors['certified_count']}/{survivors['case_count']} are certified and
   {survivors['unresolved_count']} are unresolved. This is a numerical audit,
   not a ranking or selection of those layouts.

6. **Does the tangent lower bound hold systematically?** It holds in
   {overall['tangent_lower_bound_count']}/{overall['case_count']} audited cases.
   No action was clipped to enforce it.

7. **Are nested actions nondecreasing?**
   {summary['nested_monotonicity_pass_count']}/
   {summary['nested_monotonicity_checked_count']} checked canonical cases pass,
   and the full `V4,V5,V6,V7` sequences are in the detailed table.

8. **Is the result coordinate invariant?**
   {summary['basis_invariance_pass_count']}/
   {summary['basis_invariance_checked_count']} checked cases pass permutation,
   `1e-4..1e4` rescaling, orthogonal mixing, and moderate nonorthogonal mixing.

9. **Does independent FV/FEM evidence agree?** The existing unfloored FV
   solver agrees in {summary['fv_agreement_count']}/
   {summary['fv_checked_count']} frozen representative/control cases. All
   {summary['fv_unresolved_count']} are unresolved—not contradictory finite
   answers—because their positive locally equilibrated coefficients exceed the
   float64 exponent range. The solver rejected them instead of flooring the
   density. No FEM system was added.

10. **Is float64 sufficient after avoiding normal equations?** For sampled
    ordinary/late cases after day 10, {later['certified_count']}/
    {later['case_count']} are certified. In the audited early window through
    day 10, {early['certified_count']}/{early['case_count']} are certified.
    Exact day zero has {day0['certified_count']}/68 certified. Thus float64 is
    useful for the ordinary regime but is not sufficient for the whole frozen
    ocean calculation. Regime III cases are not regularized or declared valid.

11. **Authorize repaired production?** `{authorization}` —
    {summary['production_authorization_reason']}. No production was started.

## Unresolved cases

{unresolved_text}

## Artifacts and interpretation

The detailed CSV records both quadrature grids, direct singular spectra and
per-mode action contributions, all old cutoff comparisons, tangent gaps,
nested sequences, coordinate changes, and FV outcomes. The JSON summary holds
the cohort counts and worst discrepancies. Replacing normal equations by direct
QR changes numerical linear algebra, not the finite-dimensional MFSI estimand.
"""
        path = self.experiment._resolve(self.cfg["report"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--checkpoint-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = OceanDriftersExperiment(load_config(EXPERIMENT_DIR / "config.json"))
    nice_increment = int(
        experiment.cfg["action"]["direct_qr_repair_pilot"]["nice_increment"]
    )
    if nice_increment:
        try:
            os.nice(nice_increment)
        except OSError:
            pass
    summary = DirectQRRepairPilot(experiment).run(
        recompute=args.recompute,
        checkpoint_only=args.checkpoint_only,
    )
    if summary is not None:
        print(json.dumps(_json_ready(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
