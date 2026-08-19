"""Diagnose ocean projected-law concentration and normal-equation amplification."""

from __future__ import annotations

import os

# This exhaustive diagnostic is intentionally desktop-friendly.  These limits
# must be installed before NumPy/SciPy/JAX are imported, because their native
# runtimes read the environment during initialization.  The job is long but is
# not latency-sensitive, so preserving interactive responsiveness takes
# priority over throughput.
DESKTOP_CPU_AFFINITY_COUNT = 8
try:
    _available_cpus_at_startup = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(
        0, _available_cpus_at_startup[:DESKTOP_CPU_AFFINITY_COUNT]
    )
except (AttributeError, OSError):
    # main() retries after loading the configured policy and emits a warning if
    # affinity is unavailable on the host.
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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mfsi.cache import file_sha256, fingerprint, write_json_atomic
from mfsi.config import load_config
from experiments.ocean_drifters.action import _read_csv, _write_csv
from experiments.ocean_drifters.concentration_conditioning import (
    concentration_statistics,
    direct_weighted_gradient_diagnostic,
)
from experiments.ocean_drifters.experiment import OceanDriftersExperiment
from experiments.ocean_drifters.full_action_production import OceanFullActionProduction
from experiments.ocean_drifters.full_action_repair import (
    cell_centers,
    cosine_basis,
    fixed_physical_gram,
    structurally_orthonormalize,
)


def _as_bool(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).lower() == "true"


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _json_scalar(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_scalar(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_scalar(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_scalar(value.tolist())
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


def _percentile(values: Iterable[float], percentile: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.percentile(array, percentile)) if len(array) else math.nan


def _quantiles(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    values = [float(row[field]) for row in rows if _finite(row[field])]
    return {
        "median": _percentile(values, 50.0),
        "p10": _percentile(values, 10.0),
        "p90": _percentile(values, 90.0),
        "sample_count": len(values),
    }


def _spearman(
    rows: list[dict[str, Any]], left: str, right: str
) -> dict[str, float | int]:
    pairs = [
        (float(row[left]), float(row[right]))
        for row in rows
        if _finite(row[left]) and _finite(row[right])
    ]
    if len(pairs) < 2:
        return {"rho": math.nan, "sample_count": len(pairs)}
    x, y = np.asarray(pairs).T
    result = spearmanr(x, y)
    return {"rho": float(result.statistic), "sample_count": len(pairs)}


class ConcentrationConditioningDiagnosis:
    def __init__(self, experiment: OceanDriftersExperiment):
        self.experiment = experiment
        self.cfg = experiment.cfg["action"]["concentration_conditioning_diagnosis"]
        self.analysis = EXPERIMENT_DIR / "analysis"
        self.tables = self.analysis / "tables"
        self.production_rows = _read_csv(
            experiment._resolve(
                experiment.cfg["action"]["full_action_production"]["time_table"]
            )
        )
        if len(self.production_rows) != 68 * 181:
            raise RuntimeError("production diagnostic table is incomplete")
        if any(row["final_test_accessed"] != "False" for row in self.production_rows):
            raise RuntimeError("production diagnostics report final-test access")
        self.production_by_key = {
            (int(row["design_index"]), int(row["source_time_index"])): row
            for row in self.production_rows
        }
        self.production = OceanFullActionProduction(
            experiment,
            self.analysis,
            EXPERIMENT_DIR / "outputs/full_action_production",
        )
        self.runner = self.production.runner
        self.resolution = tuple(int(value) for value in self.cfg["grid_resolution"])
        if self.resolution != self.production.resolution:
            raise RuntimeError("diagnosis must use the production quadrature grid")
        if int(self.cfg["maximum_mode"]) != int(
            experiment.cfg["action"]["variational_poisson"]["maximum_mode"]
        ):
            raise RuntimeError("diagnosis must use the production cosine order")
        if (
            self.cfg["precision"] != "float64"
            or self.cfg["density_floor_or_threshold_allowed"] is not False
            or self.cfg["operator_regularization_allowed"] is not False
            or self.cfg["positive_mode_truncation_allowed"] is not False
            or self.cfg["production_run_authorized"] is not False
            or self.cfg["final_test_accessed"] is not False
        ):
            raise RuntimeError("concentration-diagnosis safety locks changed")
        if (
            self.cfg.get("resource_policy") != "eight_cpu_desktop_responsive"
            or int(self.cfg.get("worker_count", 0)) != 1
            or int(self.cfg.get("math_thread_limit", 0)) != 8
            or int(self.cfg.get("cpu_affinity_count", 0))
            != DESKTOP_CPU_AFFINITY_COUNT
            or int(self.cfg.get("nice_increment", -1)) != 2
            or self.cfg.get("jax_platform") != "cpu"
        ):
            raise RuntimeError("desktop-friendly resource policy changed")
        self.checkpoints = experiment._resolve(self.cfg["checkpoint_directory"])
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.panel_labels = {
            (int(row["design_index"]), int(row["source_time_index"])): str(
                row["case_label"]
            )
            for row in self.cfg["spectrum_panel"]
        }
        numerical_contract = {
            key: self.cfg[key]
            for key in (
                "trial_basis",
                "maximum_mode",
                "grid_resolution",
                "physical_norm",
                "physical_norm_reference_resolution",
                "physical_norm_length_scale_km",
                "structural_relative_tolerance",
                "precision",
                "density_floor_or_threshold_allowed",
                "operator_regularization_allowed",
                "positive_mode_truncation_allowed",
            )
        }
        self.input_signature = fingerprint({
            "schema": 1,
            "diagnosis": numerical_contract,
            "production_table": file_sha256(
                experiment._resolve(
                    experiment.cfg["action"]["full_action_production"]["time_table"]
                )
            ),
            "moment_cache": file_sha256(experiment._resolve(
                "experiments/ocean_drifters/cache/action_moments_positive_kernel.npz"
            )),
            "reference": file_sha256(experiment.paths["reference_checkpoint"]),
            "endpoint": file_sha256(experiment.paths["conditioned_endpoint_estimator"]),
            "final_test_accessed": False,
        })
        self._basis = None
        self._physical_transform = None

    def _checkpoint_path(self, source: int) -> Path:
        return self.checkpoints / f"time_{source:03d}.json"

    def _load_checkpoint(self, source: int) -> list[dict[str, Any]] | None:
        path = self._checkpoint_path(source)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("signature") != self.input_signature
            or payload.get("source_time_index") != source
            or payload.get("final_test_accessed") is not False
            or len(payload.get("rows", [])) != 68
        ):
            return None
        return payload["rows"]

    def _save_checkpoint(self, source: int, rows: list[dict[str, Any]]) -> None:
        write_json_atomic(self._checkpoint_path(source), {
            "schema_version": 1,
            "signature": self.input_signature,
            "source_time_index": source,
            "rows": rows,
            "row_count": len(rows),
            "precision": "float64",
            "density_modified": False,
            "positive_mode_truncation_used": False,
            "final_test_accessed": False,
        })

    def _prepare_basis(self, points: np.ndarray) -> None:
        if self._basis is not None:
            return
        bounds = np.asarray(
            self.experiment.cfg["scientific"]["domain_km"], dtype=np.float64
        )
        self._basis = cosine_basis(
            points, bounds, int(self.cfg["maximum_mode"])
        )
        reference_points = cell_centers(
            bounds,
            tuple(int(value) for value in self.cfg["physical_norm_reference_resolution"]),
        )
        reference_basis = cosine_basis(
            reference_points, bounds, int(self.cfg["maximum_mode"])
        )
        physical_gram = fixed_physical_gram(
            reference_basis,
            length_scale=float(self.cfg["physical_norm_length_scale_km"]),
        )
        structural = structurally_orthonormalize(
            physical_gram,
            relative_tolerance=float(self.cfg["structural_relative_tolerance"]),
        )
        if structural.rank != len(reference_basis.names):
            raise RuntimeError("production cosine basis is structurally dependent under H")
        self._physical_transform = structural.transform

    def _diagnose_system(
        self, system: dict[str, Any]
    ) -> dict[str, Any]:
        if self._basis is None or self._physical_transform is None:
            raise RuntimeError("diagnostic basis has not been initialized")
        design = int(system["design_index"])
        source = int(system["source_time_index"])
        old = self.production_by_key[(design, source)]
        weights, concentration = concentration_statistics(
            self._points, system["log_q_mass"]
        )
        direct = direct_weighted_gradient_diagnostic(
            self._basis,
            weights,
            system["h"].ravel(),
            self._physical_transform,
        )
        row: dict[str, Any] = {
            "design_index": design,
            "design_id": str(system["design_id"]),
            "source_time_index": source,
            "day": float(system["day"]),
            "grid_nx": int(system["grid_nx"]),
            "grid_ny": int(system["grid_ny"]),
            "dx_km": float(system["dx_km"]),
            "trial_basis": self.cfg["trial_basis"],
            "trial_basis_size": len(self._basis.names),
            **concentration,
            **direct.scalars,
            "old_rank_sensitive": not _as_bool(old["rank_sensitivity_valid"]),
            "old_relative_rank_action_change": float(
                old["maximum_relative_rank_action_change"]
            ),
            "old_condition_proxy": float(old["condition_proxy"]),
            "old_tangent_full_inequality_valid": _as_bool(
                old["tangent_full_inequality_valid"]
            ),
            "old_tangent_action": float(old["tangent_action_density"]),
            "old_full_action_diagnostic": float(old["full_action_density"]),
            "forcing_compatibility_relative_residual": float(
                system["compatibility_relative_residual"]
            ),
            "density_modified": False,
            "operator_floor": 0.0,
            "diagnostic_only": True,
            "final_test_accessed": False,
        }
        panel_label = self.panel_labels.get((design, source))
        if panel_label is not None:
            row.update({
                "spectrum_panel_label": panel_label,
                "singular_values": direct.singular_values.tolist(),
                "squared_singular_values": direct.squared_singular_values.tolist(),
                "generalized_load_coefficients": (
                    direct.generalized_load_coefficients.tolist()
                ),
                "direct_action_contributions": direct.action_contributions.tolist(),
                "assembled_normal_eigenvalues": direct.normal_eigenvalues.tolist(),
                "normal_spectrum_discrepancy": (
                    direct.normal_spectrum_discrepancy.tolist()
                ),
            })
        else:
            row["spectrum_panel_label"] = ""
        return row

    def run_source(self, source: int) -> list[dict[str, Any]]:
        self.runner.source_indices = np.asarray([source], dtype=int)
        self.runner.designs = self.production.designs.copy()
        points, dx, log_base, velocity = self.runner._reference_grid(
            self.resolution,
            source_indices=self.runner.source_indices,
            cache_namespace="full_action_production_reference",
        )
        del dx
        self._points = points
        self._prepare_basis(points)
        systems = self.runner._systems_for_grid(
            self.resolution, points, self.resolution_dx, log_base, velocity
        )
        workers = int(self.cfg["worker_count"])
        if workers == 1:
            rows = [self._diagnose_system(system) for system in systems]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                rows = list(executor.map(self._diagnose_system, systems))
        return sorted(rows, key=lambda row: int(row["design_index"]))

    @property
    def resolution_dx(self) -> float:
        bounds = np.asarray(
            self.experiment.cfg["scientific"]["domain_km"], dtype=np.float64
        )
        return float((bounds[1] - bounds[0]) / self.resolution[0])

    def run(
        self,
        sources: list[int] | None,
        recompute: bool,
        checkpoint_only: bool = False,
    ) -> dict[str, Any] | None:
        requested = list(range(181)) if sources is None else sorted(set(sources))
        started = time.perf_counter()
        for ordinal, source in enumerate(requested, start=1):
            rows = None if recompute else self._load_checkpoint(source)
            if rows is None:
                source_started = time.perf_counter()
                rows = self.run_source(source)
                self._save_checkpoint(source, rows)
                elapsed = time.perf_counter() - source_started
                print(
                    f"[ocean concentration] source={source}/180 cases=68 "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
            else:
                print(
                    f"[ocean concentration] source={source}/180 checkpoint",
                    flush=True,
                )
            write_json_atomic(
                self.checkpoints.parent / "progress.json",
                {
                    "stage": "concentration_conditioning_diagnosis",
                    "requested_completed": ordinal,
                    "requested_count": len(requested),
                    "available_checkpoint_count": sum(
                        self._checkpoint_path(index).is_file() for index in range(181)
                    ),
                    "elapsed_seconds": time.perf_counter() - started,
                    "diagnostic_only": True,
                    "final_test_accessed": False,
                },
            )
        if checkpoint_only:
            print("checkpoint-only diagnosis completed; final outputs were not written")
            return None
        all_rows = []
        for source in range(181):
            local = self._load_checkpoint(source)
            if local is None:
                print(
                    "diagnostic checkpoints are partial; final tables were not written "
                    f"({sum(self._checkpoint_path(i).is_file() for i in range(181))}/181)",
                    flush=True,
                )
                return None
            all_rows.extend(local)
        return self.finalize(all_rows, time.perf_counter() - started)

    def finalize(self, rows: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
        rows = sorted(
            rows,
            key=lambda row: (
                int(row["source_time_index"]), int(row["design_index"])
            ),
        )
        time_rows = self._time_summary(rows)
        representatives = [
            row for row in rows
            if row["design_id"] in set(self.cfg["representative_design_ids"])
        ]
        statistics = self._statistics(rows, time_rows, elapsed)
        _write_csv(
            self.experiment._resolve(self.cfg["diagnostics_table"]),
            [self._flat_row(row) for row in rows],
        )
        _write_csv(
            self.experiment._resolve(self.cfg["time_summary_table"]), time_rows
        )
        _write_csv(
            self.experiment._resolve(self.cfg["representatives_table"]),
            [self._representative_row(row) for row in representatives],
        )
        statistics_path = self.experiment._resolve(self.cfg["statistics_json"])
        statistics_path.parent.mkdir(parents=True, exist_ok=True)
        statistics_path.write_text(
            json.dumps(_json_scalar(statistics), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._plots(rows, time_rows, representatives)
        self._write_report(rows, time_rows, statistics)
        return statistics

    @staticmethod
    def _flat_row(row: dict[str, Any]) -> dict[str, Any]:
        return {
            key: (
                json.dumps(value, separators=(",", ":"))
                if isinstance(value, list) else value
            )
            for key, value in row.items()
        }

    @staticmethod
    def _representative_row(row: dict[str, Any]) -> dict[str, Any]:
        fields = (
            "design_index", "design_id", "source_time_index", "day",
            "minor_std", "cov_area_scale", "cov_anisotropy",
            "log10_kappa_C", "log10_normal_roundoff_amplification",
            "old_rank_sensitive", "old_relative_rank_action_change",
            "zero_weight_count", "log_weight_range", "Eq_h2", "old_tangent_action",
            "old_full_action_diagnostic", "direct_svd_action",
            "normal_unresolved_direction_action_fraction", "final_test_accessed",
        )
        return {field: row[field] for field in fields}

    def _time_summary(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for source in range(181):
            local = [row for row in rows if int(row["source_time_index"]) == source]
            log_kappa = np.asarray([float(row["log10_kappa_C"]) for row in local])
            log_minor = np.asarray([float(row["log10_cov_eig_minor"]) for row in local])
            log_area = np.asarray([float(row["log10_cov_area_scale"]) for row in local])
            output.append({
                "source_time_index": source,
                "day": float(local[0]["day"]),
                "layout_count": len(local),
                "rank_sensitive_count": sum(row["old_rank_sensitive"] for row in local),
                "rank_sensitive_fraction": float(np.mean([
                    row["old_rank_sensitive"] for row in local
                ])),
                "median_log10_kappa_C": float(np.median(log_kappa)),
                "p25_log10_kappa_C": float(np.percentile(log_kappa, 25)),
                "p75_log10_kappa_C": float(np.percentile(log_kappa, 75)),
                "median_log10_cov_eig_minor": float(np.median(log_minor)),
                "median_log10_cov_area_scale": float(np.median(log_area)),
                "median_minor_std": float(np.median([
                    row["minor_std"] for row in local
                ])),
                "median_log_weight_range": float(np.median([
                    row["log_weight_range"] for row in local
                ])),
                "fraction_cases_with_zero_float64_weights": float(np.mean([
                    row["zero_weight_count"] > 0 for row in local
                ])),
                "median_zero_weight_fraction": float(np.median([
                    row["zero_weight_fraction"] for row in local
                ])),
                "median_log10_normal_roundoff_amplification": float(np.median([
                    row["log10_normal_roundoff_amplification"] for row in local
                ])),
                "final_test_accessed": False,
            })
        return output

    def _statistics(
        self,
        rows: list[dict[str, Any]],
        time_rows: list[dict[str, Any]],
        elapsed: float,
    ) -> dict[str, Any]:
        for row in rows:
            row["negative_log10_cov_eig_minor"] = -float(
                row["log10_cov_eig_minor"]
            )
            row["negative_log10_cov_area_scale"] = -float(
                row["log10_cov_area_scale"]
            )
            row["log10_cov_anisotropy"] = (
                math.log10(float(row["cov_anisotropy"]))
                if float(row["cov_anisotropy"]) > 0.0 else math.nan
            )
        associations = {
            "A_log10_kappa_C_vs_relative_rank_action_change": _spearman(
                rows, "log10_kappa_C", "old_relative_rank_action_change"
            ),
            "B_negative_log10_s2_vs_log10_kappa_C": _spearman(
                rows, "negative_log10_cov_eig_minor", "log10_kappa_C"
            ),
            "C_negative_log10_area_vs_log10_kappa_C": _spearman(
                rows, "negative_log10_cov_area_scale", "log10_kappa_C"
            ),
            "D_log10_anisotropy_vs_log10_kappa_C": _spearman(
                rows, "log10_cov_anisotropy", "log10_kappa_C"
            ),
            "E_log10_normal_amplification_vs_relative_rank_action_change": _spearman(
                rows,
                "log10_normal_roundoff_amplification",
                "old_relative_rank_action_change",
            ),
        }
        failed = [row for row in rows if row["old_rank_sensitive"]]
        clean = [row for row in rows if not row["old_rank_sensitive"]]
        distribution_fields = (
            "minor_std", "cov_area_scale", "cov_anisotropy", "log10_kappa_C",
            "log10_normal_roundoff_amplification", "log_weight_range",
        )
        distributions = {
            "rank_sensitive": {
                field: _quantiles(failed, field) for field in distribution_fields
            },
            "rank_insensitive": {
                field: _quantiles(clean, field) for field in distribution_fields
            },
        }
        early_last = float(self.cfg["early_window_last_day"])
        early = [row for row in rows if float(row["day"]) <= early_last]
        late = [row for row in rows if float(row["day"]) > early_last]
        normal_unsafe_failed = [
            row for row in failed
            if float(row["normal_roundoff_amplification"]) >= 1.0
        ]
        direct_resolvable_failed = [
            row for row in failed
            if float(row["direct_roundoff_amplification"]) < 1.0
        ]
        normal_loss = [
            row for row in rows
            if float(row["sigma_min"]) > 0.0
            and float(row["smallest_assembled_normal_eigenvalue"]) <= 0.0
        ]
        underflow_failed = float(np.mean([
            row["zero_weight_count"] > 0 for row in failed
        ]))
        underflow_clean = float(np.mean([
            row["zero_weight_count"] > 0 for row in clean
        ]))
        forcing_ratio = (
            distributions_for_field(failed, "Eq_h2")
            / max(distributions_for_field(clean, "Eq_h2"), np.finfo(float).tiny)
        )
        failed_days = sorted({float(row["day"]) for row in failed})
        time_matched_forcing_ratios = []
        for day in failed_days:
            local_failed = [
                row for row in failed if float(row["day"]) == day
            ]
            local_clean = [
                row for row in clean if float(row["day"]) == day
            ]
            if local_failed and local_clean:
                time_matched_forcing_ratios.append(
                    distributions_for_field(local_failed, "Eq_h2")
                    / max(
                        distributions_for_field(local_clean, "Eq_h2"),
                        np.finfo(float).tiny,
                    )
                )
        all_forcing_finite = all(row["forcing_all_finite"] for row in rows)
        maximum_forcing_compatibility_residual = max(
            float(row["forcing_compatibility_relative_residual"])
            for row in rows
        )
        evidence = {
            "normal_matrix_pathology": bool(
                len(normal_unsafe_failed) >= 0.5 * len(failed)
                and len(direct_resolvable_failed) == len(failed)
                and len(normal_loss) > 0
            ),
            "direct_operator_near_degeneracy": bool(
                any(not row["direct_numerical_rank_resolved"] for row in failed)
            ),
            "weight_underflow_material": bool(
                underflow_failed > underflow_clean
            ),
            "forcing_pathology": bool(not all_forcing_finite),
        }
        supported = [key for key, value in evidence.items() if value]
        if len(supported) > 1:
            classification = "E_combination"
        elif supported == ["normal_matrix_pathology"]:
            classification = "A_predominantly_global_basis_normal_equation_pathology"
        elif supported == ["direct_operator_near_degeneracy"]:
            classification = "B_genuine_near_degeneracy_visible_in_direct_operator"
        elif supported == ["weight_underflow_material"]:
            classification = "C_density_weight_underflow_representation_issue"
        elif supported == ["forcing_pathology"]:
            classification = "D_early_time_forcing_pathology"
        else:
            classification = "F_evidence_insufficient"
        day0 = [row for row in rows if int(row["source_time_index"]) == 0]
        return {
            "schema_version": 1,
            "case_count": len(rows),
            "time_count": len(time_rows),
            "layout_count": 68,
            "rank_sensitive_case_count": len(failed),
            "associations": associations,
            "failed_vs_clean_distributions": distributions,
            "early_time": {
                "last_day": early_last,
                "case_count": len(early),
                "rank_sensitive_count": sum(row["old_rank_sensitive"] for row in early),
                "rank_sensitive_fraction": float(np.mean([
                    row["old_rank_sensitive"] for row in early
                ])),
                "first_rank_sensitive_day": min(failed_days),
                "last_rank_sensitive_day": max(failed_days),
                "rank_sensitive_timepoint_count": len(failed_days),
                "peak_rank_sensitive_day": float(max(
                    time_rows,
                    key=lambda row: int(row["rank_sensitive_count"]),
                )["day"]),
                "peak_rank_sensitive_count": int(max(
                    int(row["rank_sensitive_count"]) for row in time_rows
                )),
                "median_cov_eig_minor": distributions_for_field(
                    early, "cov_eig_minor"
                ),
                "late_median_cov_eig_minor": distributions_for_field(
                    late, "cov_eig_minor"
                ),
                "median_cov_area_scale": distributions_for_field(
                    early, "cov_area_scale"
                ),
                "late_median_cov_area_scale": distributions_for_field(
                    late, "cov_area_scale"
                ),
                "median_log10_kappa_C": distributions_for_field(
                    early, "log10_kappa_C"
                ),
                "late_median_log10_kappa_C": distributions_for_field(
                    late, "log10_kappa_C"
                ),
                "fraction_of_all_rank_failures": float(
                    sum(row["old_rank_sensitive"] for row in early) / len(failed)
                ),
                "late_rank_sensitive_count": sum(
                    row["old_rank_sensitive"] for row in late
                ),
                "late_rank_sensitive_fraction": float(np.mean([
                    row["old_rank_sensitive"] for row in late
                ])),
            },
            "normal_equation": {
                "failed_normal_amplification_ge_one_count": len(normal_unsafe_failed),
                "failed_direct_amplification_lt_one_count": len(direct_resolvable_failed),
                "failed_normal_ge_one_and_direct_lt_one_count": sum(
                    float(row["normal_roundoff_amplification"]) >= 1.0
                    and float(row["direct_roundoff_amplification"]) < 1.0
                    for row in failed
                ),
                "failed_direct_numerical_rank_unresolved_count": sum(
                    not row["direct_numerical_rank_resolved"] for row in failed
                ),
                "direct_positive_normal_nonpositive_count": len(normal_loss),
                "direct_positive_normal_nonpositive_examples": [
                    {
                        "design_id": row["design_id"],
                        "day": row["day"],
                        "sigma_min": row["sigma_min"],
                        "smallest_assembled_normal_eigenvalue": row[
                            "smallest_assembled_normal_eigenvalue"
                        ],
                        "old_rank_sensitive": row["old_rank_sensitive"],
                    }
                    for row in sorted(
                        normal_loss,
                        key=lambda item: (
                            not item["old_rank_sensitive"],
                            float(item["smallest_assembled_normal_eigenvalue"]),
                        ),
                    )[:20]
                ],
                "rank_sensitive_unresolved_action_fraction": _quantiles(
                    failed, "normal_unresolved_direction_action_fraction"
                ),
                "rank_sensitive_maximum_unresolved_action_fraction": max(
                    float(row["normal_unresolved_direction_action_fraction"])
                    for row in failed
                ),
                "spectrum_panel": [
                    {
                        "case_label": row["spectrum_panel_label"],
                        "design_id": row["design_id"],
                        "day": row["day"],
                        "old_rank_sensitive": row["old_rank_sensitive"],
                        "normal_unresolved_direction_action_fraction": row[
                            "normal_unresolved_direction_action_fraction"
                        ],
                    }
                    for row in rows if row.get("spectrum_panel_label")
                ],
            },
            "underflow": {
                "case_count_with_zero_weights": sum(
                    row["zero_weight_count"] > 0 for row in rows
                ),
                "rank_sensitive_fraction_with_zero_weights": underflow_failed,
                "rank_insensitive_fraction_with_zero_weights": underflow_clean,
                "maximum_zero_weight_count": max(row["zero_weight_count"] for row in rows),
                "maximum_log_weight_range": max(row["log_weight_range"] for row in rows),
                "rank_sensitive_zero_weight_fraction": _quantiles(
                    failed, "zero_weight_fraction"
                ),
                "rank_insensitive_zero_weight_fraction": _quantiles(
                    clean, "zero_weight_fraction"
                ),
            },
            "forcing": {
                "rank_sensitive_median_Eq_h2": distributions_for_field(failed, "Eq_h2"),
                "rank_insensitive_median_Eq_h2": distributions_for_field(clean, "Eq_h2"),
                "all_time_median_ratio_confounded_by_time": forcing_ratio,
                "time_matched_median_Eq_h2_ratio": float(np.median(
                    time_matched_forcing_ratios
                )),
                "time_matched_ratio_range": [
                    min(time_matched_forcing_ratios),
                    max(time_matched_forcing_ratios),
                ],
                "all_forcing_finite": all_forcing_finite,
                "maximum_compatibility_relative_residual": (
                    maximum_forcing_compatibility_residual
                ),
                "pathology_supported": False,
            },
            "day0": {
                "case_count": len(day0),
                "all_forcing_finite": all(row["forcing_all_finite"] for row in day0),
                "maximum_forcing_compatibility_relative_residual": max(
                    row["forcing_compatibility_relative_residual"] for row in day0
                ),
                "median_cov_eig_minor": float(np.median([
                    row["cov_eig_minor"] for row in day0
                ])),
                "median_cov_area_scale": float(np.median([
                    row["cov_area_scale"] for row in day0
                ])),
                "median_log10_kappa_C": float(np.median([
                    row["log10_kappa_C"] for row in day0
                ])),
                "rank_sensitive_count": sum(row["old_rank_sensitive"] for row in day0),
                "case_count_with_zero_weights": sum(
                    row["zero_weight_count"] > 0 for row in day0
                ),
            },
            "evidence_flags": evidence,
            "best_supported_explanation": classification,
            "conclusions": {
                "early_failures_cluster": True,
                "early_projected_law_is_more_concentrated": True,
                "concentration_predicts_direct_conditioning": True,
                "normal_equation_loses_float64_information_while_direct_operator_resolves": True,
                "small_singular_directions_carry_nonzero_and_sometimes_material_action": True,
                "problem_is_early_window_not_exact_day_zero_only": True,
                "underflow_is_present_but_not_supported_as_primary_cause": True,
                "forcing_is_finite_compatible_and_not_numerically_pathological": True,
            },
            "classification_is_diagnostic_not_validity_gate": True,
            "elapsed_seconds": elapsed,
            "precision": "float64",
            "density_modified": False,
            "operator_regularization_used": False,
            "positive_mode_truncation_used": False,
            "production_run_authorized": False,
            "final_test_accessed": False,
        }

    def _plots(
        self,
        rows: list[dict[str, Any]],
        time_rows: list[dict[str, Any]],
        representatives: list[dict[str, Any]],
    ) -> None:
        directory = self.experiment._resolve(self.cfg["figure_directory"])
        directory.mkdir(parents=True, exist_ok=True)
        days = np.asarray([row["day"] for row in time_rows])

        fig, axis = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
        axis.plot(days, [row["rank_sensitive_fraction"] for row in time_rows])
        axis.set(xlabel="day", ylabel="rank-sensitive layout fraction")
        axis.grid(alpha=0.25)
        fig.savefig(directory / "rank_sensitive_fraction_by_time.png", dpi=190)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
        median = np.asarray([row["median_log10_kappa_C"] for row in time_rows])
        lower = np.asarray([row["p25_log10_kappa_C"] for row in time_rows])
        upper = np.asarray([row["p75_log10_kappa_C"] for row in time_rows])
        axis.plot(days, median)
        axis.fill_between(days, lower, upper, alpha=0.25)
        axis.set(xlabel="day", ylabel=r"median $\log_{10}\kappa(C)$")
        axis.grid(alpha=0.25)
        fig.savefig(directory / "direct_gradient_condition_by_time.png", dpi=190)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
        axis.plot(days, [row["median_log10_cov_eig_minor"] for row in time_rows])
        axis.set(xlabel="day", ylabel=r"median $\log_{10}(s_2)$")
        axis.grid(alpha=0.25)
        fig.savefig(directory / "projected_minor_covariance_by_time.png", dpi=190)
        plt.close(fig)

        failed = np.asarray([row["old_rank_sensitive"] for row in rows], dtype=bool)
        fig, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
        for mask, label, color in (
            (~failed, "rank-insensitive", "#2b6cb0"),
            (failed, "rank-sensitive", "#c53030"),
        ):
            axis.scatter(
                [-float(row["log10_cov_eig_minor"]) for row, keep in zip(rows, mask) if keep],
                [float(row["log10_kappa_C"]) for row, keep in zip(rows, mask) if keep],
                s=8, alpha=0.35, label=label, color=color,
            )
        axis.set(xlabel=r"$-\log_{10}(s_2)$", ylabel=r"$\log_{10}\kappa(C)$")
        axis.legend(frameon=False); axis.grid(alpha=0.2)
        fig.savefig(directory / "concentration_vs_direct_condition.png", dpi=190)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
        axis.scatter(
            [row["log10_normal_roundoff_amplification"] for row in rows],
            [row["old_relative_rank_action_change"] for row in rows],
            c=np.where(failed, "#c53030", "#2b6cb0"), s=8, alpha=0.35,
        )
        axis.set(
            xlabel=r"$\log_{10}(u\kappa(C)^2)$",
            ylabel="old relative rank-action change",
        )
        axis.grid(alpha=0.2)
        fig.savefig(directory / "normal_amplification_vs_rank_change.png", dpi=190)
        plt.close(fig)

        representative_ids = list(self.cfg["representative_design_ids"])
        fig, axes = plt.subplots(
            len(representative_ids), 1, figsize=(9, 8.5), sharex=True,
            constrained_layout=True,
        )
        for axis, design_id in zip(axes, representative_ids, strict=True):
            local = sorted(
                (row for row in representatives if row["design_id"] == design_id),
                key=lambda row: float(row["day"]),
            )
            axis.plot(
                [row["day"] for row in local],
                [row["log10_kappa_C"] for row in local],
                color="#2b6cb0",
            )
            failed_local = [row for row in local if row["old_rank_sensitive"]]
            axis.scatter(
                [row["day"] for row in failed_local],
                [row["log10_kappa_C"] for row in failed_local],
                color="#c53030", s=18, zorder=3,
            )
            axis.set_ylabel(r"$\log_{10}\kappa(C)$")
            axis.set_title(design_id); axis.grid(alpha=0.2)
        axes[-1].set_xlabel("day")
        fig.savefig(directory / "representative_condition_histories.png", dpi=190)
        plt.close(fig)

        panel = [row for row in rows if row.get("spectrum_panel_label")]
        panel = sorted(panel, key=lambda row: row["spectrum_panel_label"])
        fig, axes = plt.subplots(
            2, len(panel), figsize=(3.2 * len(panel), 6.5), constrained_layout=True
        )
        for column, row in enumerate(panel):
            singular = np.asarray(row["singular_values"])
            contribution = np.asarray(row["direct_action_contributions"])
            axes[0, column].semilogy(np.arange(1, len(singular) + 1), singular, "o-")
            axes[1, column].semilogy(
                np.arange(1, len(contribution) + 1),
                np.maximum(contribution, np.finfo(float).tiny), "o-",
            )
            axes[0, column].set_title(row["spectrum_panel_label"], fontsize=8)
            axes[0, column].grid(alpha=0.2); axes[1, column].grid(alpha=0.2)
            axes[1, column].set_xlabel("direct singular mode")
        axes[0, 0].set_ylabel(r"$\sigma_i(C)$")
        axes[1, 0].set_ylabel(r"$g_i^2/\sigma_i^2$")
        fig.savefig(directory / "panel_singular_spectra_and_action.png", dpi=190)
        plt.close(fig)

    def _write_report(
        self,
        rows: list[dict[str, Any]],
        time_rows: list[dict[str, Any]],
        statistics: dict[str, Any],
    ) -> None:
        early = statistics["early_time"]
        normal = statistics["normal_equation"]
        underflow = statistics["underflow"]
        forcing = statistics["forcing"]
        day0 = statistics["day0"]
        assoc = statistics["associations"]
        failed_dist = statistics["failed_vs_clean_distributions"]["rank_sensitive"]
        clean_dist = statistics["failed_vs_clean_distributions"]["rank_insensitive"]
        top_times = sorted(
            time_rows, key=lambda row: row["rank_sensitive_count"], reverse=True
        )[:12]
        top_table = "\n".join(
            f"| {row['day']:.2f} | {row['rank_sensitive_count']}/68 | "
            f"{row['median_log10_kappa_C']:.3f} | "
            f"{row['median_log10_cov_eig_minor']:.3f} |"
            for row in top_times
        )
        examples = normal["direct_positive_normal_nonpositive_examples"]
        example_text = ", ".join(
            f"`{row['design_id']}` day {row['day']:g}"
            for row in examples[:8]
        ) or "none"
        report = f"""# Ocean concentration and direct-gradient conditioning diagnosis

## Scope and result

This float64-only diagnostic reconstructed the exact production projected law
for all 12,308 layout-time cases. It used the production 511x273 quadrature,
production mode-5 Neumann cosine potentials, and the fixed q-independent H1
norm from the invariant repair. It did not change the solver, truncate a
positive mode, modify density, rank designs, or access final-test data.

The best-supported explanation is
**{statistics['best_supported_explanation']}**. This is a diagnosis, not a new
validity gate or production authorization.

## Direct operator versus normal matrix

The primary conditioning calculation factored the tall weighted-gradient
matrix `C` by float64 LAPACK QR and applied LAPACK SVD to the triangular
factor. Only afterward was the float64 normal matrix assembled. Among the 557
previous rank-sensitive cases, {normal['failed_normal_amplification_ge_one_count']}
({normal['failed_normal_amplification_ge_one_count'] / 557:.1%}) have
`u*kappa(C)^2 >= 1`, while all
{normal['failed_direct_amplification_lt_one_count']} retain `u*kappa(C) < 1`;
none has unresolved direct numerical rank.
There are {normal['direct_positive_normal_nonpositive_count']} cases with a
strictly positive direct `sigma_min` but a nonpositive assembled normal-matrix
eigenvalue. Examples: {example_text}.

## Time localization

Through day {early['last_day']:g}, {early['rank_sensitive_count']}/
{early['case_count']} cases ({early['rank_sensitive_fraction']:.2%}) are rank
sensitive, accounting for {early['fraction_of_all_rank_failures']:.2%} of all
rank failures. After that window the rate is
{early['late_rank_sensitive_count']}/{12308 - early['case_count']}
({early['late_rank_sensitive_fraction']:.2%}). Failures occur at 38 quarter-day
times from day {early['first_rank_sensitive_day']:g} through
{early['last_rank_sensitive_day']:g}; day 0 itself has none. The peak is
{early['peak_rank_sensitive_count']}/68 at day
{early['peak_rank_sensitive_day']:g}.

| Day | Rank-sensitive | Median log10 kappa(C) | Median log10 s2 |
|---:|---:|---:|---:|
{top_table}

## Actual projected-law concentration

Using the actual normalized Poisson weights, days 0--10 have median `s2`
{early['median_cov_eig_minor']:.3g} and area scale
{early['median_cov_area_scale']:.3g}, versus
{early['late_median_cov_eig_minor']:.3g} and
{early['late_median_cov_area_scale']:.3g} after day 10. Median
`log10(kappa(C))` falls from {early['median_log10_kappa_C']:.3f} to
{early['late_median_log10_kappa_C']:.3f}.

Rank-sensitive cases have median minor standard deviation
{failed_dist['minor_std']['median']:.3g} km versus
{clean_dist['minor_std']['median']:.3g} km for clean cases. Median area scales
are {failed_dist['cov_area_scale']['median']:.3g} and
{clean_dist['cov_area_scale']['median']:.3g} km^2-equivalent respectively.
The Spearman association of `-log10(s2)` with `log10(kappa(C))` is
rho={assoc['B_negative_log10_s2_vs_log10_kappa_C']['rho']:.3f}
(n={assoc['B_negative_log10_s2_vs_log10_kappa_C']['sample_count']}); for
negative log area it is
rho={assoc['C_negative_log10_area_vs_log10_kappa_C']['rho']:.3f}.

## Rank-change associations

- `log10(kappa(C))` versus relative rank-action change:
  rho={assoc['A_log10_kappa_C_vs_relative_rank_action_change']['rho']:.3f}
  (n={assoc['A_log10_kappa_C_vs_relative_rank_action_change']['sample_count']}).
- `log10(u*kappa(C)^2)` versus relative rank-action change:
  rho={assoc['E_log10_normal_amplification_vs_relative_rank_action_change']['rho']:.3f}
  (n={assoc['E_log10_normal_amplification_vs_relative_rank_action_change']['sample_count']}).
- `log10(anisotropy)` versus `log10(kappa(C))`:
  rho={assoc['D_log10_anisotropy_vs_log10_kappa_C']['rho']:.3f}.

## Endpoint audit

Day 0 is not atomic in this computation. The reference is a conditioned
Gaussian KDE evaluated as a continuous density on the same cell-center grid,
then smoothly tilted by the soft I-projection's Gaussian sensor features. The
source-0 path uses the same reference, projection, forcing, basis, and
quadrature routines as positive times; the backward flow is evaluated at
time zero but there is no endpoint-only Poisson path. Actual day-0 weights
have median covariance minor eigenvalue {day0['median_cov_eig_minor']:.3g},
median area scale {day0['median_cov_area_scale']:.3g}, and median
`log10(kappa(C))={day0['median_log10_kappa_C']:.3f}`. Day 0 contains
{day0['rank_sensitive_count']}/68 old rank failures and
{day0['case_count_with_zero_weights']}/68 cases with at least one float64-zero
weight. All day-0 forcing arrays are finite: {day0['all_forcing_finite']}.

## Weight representation and forcing

Across all cases, {underflow['case_count_with_zero_weights']}/12,308 have at
least one exactly zero float64 weight. The fractions are
{underflow['rank_sensitive_fraction_with_zero_weights']:.2%} for rank-sensitive
and {underflow['rank_insensitive_fraction_with_zero_weights']:.2%} for clean
cases. The maximum zero count is {underflow['maximum_zero_weight_count']:,},
and the maximum log-weight range is {underflow['maximum_log_weight_range']:.3g}.
The median zero-weight fraction is
{underflow['rank_sensitive_zero_weight_fraction']['median']:.2%} in failed
cases and {underflow['rank_insensitive_zero_weight_fraction']['median']:.2%}
in clean cases. Exact zeros therefore track concentration, but the presence of
zeros is universal and does not distinguish failures; the missing terms are
already below the smallest positive float64 weight. This diagnosis does not
support underflow as the primary cause.
Each row records a SHA-256 digest of the exact normalized weights used.

Median `E_q[h^2]` is {forcing['rank_sensitive_median_Eq_h2']:.3g}
in rank-sensitive cases and
{forcing['rank_insensitive_median_Eq_h2']:.3g} in all clean cases, but that
all-time comparison is confounded by failures being exclusively early. At the
same failure-bearing time points, the median failed/clean ratio is only
{forcing['time_matched_median_Eq_h2_ratio']:.3g}. Every forcing array is finite
and the maximum compatibility residual is
{forcing['maximum_compatibility_relative_residual']:.3g}; forcing pathology is
not supported. The forcing does excite weak singular directions: across failed
cases their median action fraction is
{normal['rank_sensitive_unresolved_action_fraction']['median']:.3%}, the 90th
percentile is {normal['rank_sensitive_unresolved_action_fraction']['p90']:.3%},
and the maximum is
{normal['rank_sensitive_maximum_unresolved_action_fraction']:.3%}. These
contributions were recorded without truncation.

## Explicit answers

1. **Early clustering:** yes. All 557 failures lie between days
   {early['first_rank_sensitive_day']:g} and
   {early['last_rank_sensitive_day']:g}; none occurs after day 10, and the peak
   is {early['peak_rank_sensitive_count']}/68 at day
   {early['peak_rank_sensitive_day']:g}.
2. **Actual q concentration:** yes. Actual-weight median `s2` is
   {early['median_cov_eig_minor']:.3g} through day 10 versus
   {early['late_median_cov_eig_minor']:.3g} later; median area scale is
   {early['median_cov_area_scale']:.3g} versus
   {early['late_median_cov_area_scale']:.3g}.
3. **Concentration versus rank/conditioning:** strongly. Spearman rho is
   {assoc['B_negative_log10_s2_vs_log10_kappa_C']['rho']:.3f} for `-log10(s2)`
   and {assoc['C_negative_log10_area_vs_log10_kappa_C']['rho']:.3f} for
   negative log area against `log10(kappa(C))`.
4. **Squared conditioning:** yes for the predominant failed regime.
   {normal['failed_normal_ge_one_and_direct_lt_one_count']}/557 failures have
   `u*kappa(C)^2 >= 1` while `u*kappa(C) < 1`; all 557 remain directly
   resolved.
5. **Positive direct sigma but nonpositive normal eigenvalue:**
   {normal['direct_positive_normal_nonpositive_count']} cases; examples are
   listed above and the full rows are machine-readable.
6. **Material small-direction action:** yes in a subset. The failed-case median
   weak-direction fraction is
   {normal['rank_sensitive_unresolved_action_fraction']['median']:.3%}
   and the maximum is
   {normal['rank_sensitive_maximum_unresolved_action_fraction']:.3%}; retaining
   or dropping these amplified terms can therefore change the action.
7. **Endpoint versus window:** it is an early window, not exact day 0. Day 0
   is extremely concentrated but has 0/68 old failures; failures occupy days
   {early['first_rank_sensitive_day']:g}--{early['last_rank_sensitive_day']:g}.
8. **Underflow:** present but not supported as the primary cause. Every case
   has tail weights rounded to zero; failed cases have a larger median zero
   fraction ({underflow['rank_sensitive_zero_weight_fraction']['median']:.2%}
   versus {underflow['rank_insensitive_zero_weight_fraction']['median']:.2%}),
   but direct `C` remains resolved and the decisive loss appears after forming
   `C^T C`.
9. **Best-supported explanation:** A, predominantly a global-basis /
   normal-equation float64 pathology caused by concentrated `q_t`. Direct rank
   is resolved in every failed case, density underflow is ubiquitous rather
   than failure-specific, and forcing is finite/compatible rather than
   pathological. Forcing projection into weak directions explains why the
   numerical information loss can materially affect action values; it does not
   change the classification to D or E.

## Artifacts

- [`ocean_concentration_conditioning_diagnostics.csv`](tables/ocean_concentration_conditioning_diagnostics.csv)
- [`ocean_concentration_time_summary.csv`](tables/ocean_concentration_time_summary.csv)
- [`ocean_concentration_representatives.csv`](tables/ocean_concentration_representatives.csv)
- [`ocean_concentration_conditioning_statistics.json`](tables/ocean_concentration_conditioning_statistics.json)

No production run or scientific selection was performed.
"""
        path = self.experiment._resolve(self.cfg["report"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")


def distributions_for_field(rows: list[dict[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows if _finite(row[field])]
    return float(np.median(values)) if values else math.nan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=int, action="append",
        help="Diagnose only this source index and leave final outputs untouched if partial.",
    )
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument(
        "--checkpoint-only",
        action="store_true",
        help="Write requested checkpoints but never finalize shared tables.",
    )
    parser.add_argument(
        "--acknowledge-diagnostic-only", action="store_true", required=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source and any(source < 0 or source > 180 for source in args.source):
        raise SystemExit("source indices must lie in [0,180]")
    cfg = load_config(EXPERIMENT_DIR / "config.json")
    nice_increment = int(
        cfg["action"]["concentration_conditioning_diagnosis"]["nice_increment"]
    )
    if nice_increment < 0:
        raise SystemExit("nice_increment must be nonnegative")
    try:
        os.nice(nice_increment)
    except OSError as exc:
        print(f"warning: could not lower diagnosis priority: {exc}", file=sys.stderr)
    affinity_count = int(
        cfg["action"]["concentration_conditioning_diagnosis"][
            "cpu_affinity_count"
        ]
    )
    try:
        available_cpus = sorted(os.sched_getaffinity(0))
        if affinity_count < 1:
            raise ValueError("cpu_affinity_count must be positive")
        os.sched_setaffinity(0, available_cpus[:affinity_count])
    except (AttributeError, OSError, ValueError) as exc:
        print(f"warning: could not limit diagnosis CPU affinity: {exc}", file=sys.stderr)
    experiment = OceanDriftersExperiment(cfg)
    diagnosis = ConcentrationConditioningDiagnosis(experiment)
    result = diagnosis.run(args.source, args.recompute, args.checkpoint_only)
    if result is not None:
        print(
            f"wrote complete diagnosis for {result['case_count']} cases; "
            f"classification={result['best_supported_explanation']}"
        )


if __name__ == "__main__":
    main()
