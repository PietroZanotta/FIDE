#!/usr/bin/env python3
"""Run the development-only ocean Law/Tangent/Full percentage comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

if __name__ == "__main__":
    try:
        available_cpus = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, available_cpus[:8])
    except (AttributeError, OSError):
        pass
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mfsi.cache import file_sha256, fingerprint, write_json_atomic  # noqa: E402
from mfsi.config import load_config  # noqa: E402

from action import _read_csv, _write_csv  # noqa: E402
from direct_qr_ritz import (  # noqa: E402
    prepare_direct_ritz_basis,
    solve_prepared_direct_ritz,
)
from experiments.ocean_drifters.experiment import OceanDriftersExperiment  # noqa: E402
from experiments.ocean_drifters.full_action import OceanWeightedPoissonPilot  # noqa: E402
from experiments.ocean_drifters.full_action_repair import (  # noqa: E402
    cell_centers,
    enriched_basis,
    fixed_physical_gram,
    normalized_weights,
)
from experiments.ocean_drifters.temporal_refinement import (  # noqa: E402
    nested_source_grids,
    summarize_temporal_levels,
)


METHODS = ("law", "tangent", "full")
COLORS = {"law": "#2878B5", "tangent": "#E39D24", "full": "#D1495B"}
MARKERS = {"law": "s", "tangent": "^", "full": "o"}


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _write_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _bool(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).lower() == "true"


def _typed_detail(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    integer_fields = (
        "design_index", "source_time_index", "grid_nx", "grid_ny",
    )
    float_fields = (
        "day", "tangent_action", "direct_action_qr", "direct_action_svd",
        "direct_qr_svd_relative_difference", "direct_condition_number",
        "forcing_compatibility_relative_residual",
        "lambda_dot_solve_relative_residual",
        "tangent_compatibility_relative_residual",
    )
    bool_fields = (
        "direct_full_column_rank", "direct_numerical_success",
        "forcing_compatibility_valid", "multiplier_coordinate_ready",
        "lambda_dot_valid", "tangent_compatibility_valid",
        "tangent_full_lower_bound_valid", "local_valid", "final_test_accessed",
    )
    for field in integer_fields:
        result[field] = int(result[field])
    for field in float_fields:
        result[field] = float(result[field])
    for field in bool_fields:
        result[field] = _bool(result[field])
    return result


def choose_methods(
    risk: np.ndarray,
    actions: Mapping[int, Mapping[str, Any]],
    eligible: np.ndarray,
) -> dict[str, int]:
    """Choose all methods from one certified feasible set."""
    candidates = [
        int(index) for index in np.flatnonzero(eligible)
        if int(index) in actions and bool(actions[int(index)]["certified"])
    ]
    if not candidates:
        raise RuntimeError("the percentage point has no action-certified layout")
    return {
        "law": min(candidates, key=lambda index: (float(risk[index]), index)),
        "tangent": min(
            candidates,
            key=lambda index: (float(actions[index]["tangent_action"]), index),
        ),
        "full": min(
            candidates,
            key=lambda index: (float(actions[index]["full_action"]), index),
        ),
    }


class OceanPercentagePareto:
    def __init__(self, cfg: dict[str, Any], output: Path):
        self.cfg = cfg
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.source_cfg = load_config(_resolve(cfg["source_experiment_config"]))
        if self.source_cfg["scientific"]["final_test_access_allowed"] is not False:
            raise RuntimeError("the percentage experiment requires final-test access disabled")
        if self.source_cfg["final_evaluation"]["authorization_status"] != (
            "one_shot_final_evaluation_consumed"
        ):
            raise RuntimeError("the source final-test lifecycle is not in consumed state")
        if cfg["validation"]["final_test_reuse_allowed"] is not False:
            raise RuntimeError("the percentage experiment may not reuse the final test")
        self.experiment = OceanDriftersExperiment(self.source_cfg)
        self.action_cfg = cfg["action"]
        self.percentages = tuple(float(value) for value in cfg["risk_allowance_percentages"])
        if tuple(sorted(set(self.percentages))) != self.percentages:
            raise ValueError("risk allowances must be unique and increasing")
        self._load_risk()
        self._freeze_candidates()
        self.sources = nested_source_grids(
            int(self.action_cfg["window_source_indices"][0]),
            int(self.action_cfg["window_source_indices"][1]),
            self.action_cfg["source_steps"],
        )[-1]
        if len(self.sources) != int(self.action_cfg["node_counts"][-1]):
            raise RuntimeError("the percentage action node contract changed")
        self.resolution = tuple(int(value) for value in self.action_cfg["grid_resolution"])
        self.contract_hash = self._contract_hash()
        self.pilot = OceanWeightedPoissonPilot(
            self.experiment,
            SCRIPT_DIR,
            self.output / "working",
        )
        self.pilot.designs = self.candidates.copy()
        self.pilot.local_by_design = {
            int(design): int(np.flatnonzero(self.pilot.all_designs == design)[0])
            for design in self.candidates
        }
        self.pilot.soft_penalty.clear()
        self.pilot.soft_penalty_dot.clear()
        self.pilot._build_soft_moment_penalties()

    def _load_risk(self) -> None:
        path = self.experiment.paths["risk_projection_embeddings"]
        with np.load(path, allow_pickle=False) as data:
            if bool(data["final_test_accessed"]):
                raise RuntimeError("risk bank reports final-test access")
            self.design_ids = np.asarray(data["design_id"]).astype(str)
            self.risk = np.asarray(data["risks"], dtype=np.float64)
            self.risk_eligible = np.asarray(data["eligible"], dtype=bool)
            self.bootstrap_risk = np.asarray(data["bootstrap_risk"], dtype=np.float64)
        anchor = self.cfg["law_anchor"]
        self.anchor = int(anchor["design_index"])
        self.r_star = float(anchor["risk"])
        if self.design_ids[self.anchor] != anchor["design_id"]:
            raise RuntimeError("the percentage Law anchor ID changed")
        if not np.isclose(self.risk[self.anchor], self.r_star, rtol=0.0, atol=2e-15):
            raise RuntimeError("the percentage Law anchor risk changed")
        best = int(np.flatnonzero(self.risk_eligible)[np.argmin(self.risk[self.risk_eligible])])
        if best != self.anchor:
            raise RuntimeError("the frozen Law anchor is no longer optimal")

    def _freeze_candidates(self) -> None:
        maximum = max(self.percentages) / 100.0
        mask = self.risk_eligible & (self.risk <= self.r_star * (1.0 + maximum))
        self.candidates = np.flatnonzero(mask).astype(int)
        self.candidates = self.candidates[np.argsort(self.risk[self.candidates])]
        with np.load(
            self.experiment._resolve(
                "experiments/ocean_drifters/cache/action_moments_positive_kernel.npz"
            ),
            allow_pickle=False,
        ) as data:
            if json.loads(str(data["__metadata_json__"].item()))[
                "final_test_accessed"
            ] is not False:
                raise RuntimeError("action moment bank reports final-test access")
            available = set(np.asarray(data["design_indices"], dtype=int).tolist())
        missing = sorted(set(self.candidates.tolist()) - available)
        if missing:
            raise RuntimeError(f"percentage candidates lack frozen action moments: {missing}")

    def _contract_hash(self) -> str:
        return fingerprint({
            "schema": 1,
            "config": self.cfg,
            "risk_sha256": file_sha256(self.experiment.paths["risk_projection_embeddings"]),
            "action_moments_sha256": file_sha256(self.experiment._resolve(
                "experiments/ocean_drifters/cache/action_moments_positive_kernel.npz"
            )),
            "reference_sha256": file_sha256(self.experiment.paths["reference_checkpoint"]),
            "candidate_designs": self.candidates.tolist(),
        })

    def _prepare_basis(self, design: int):
        source = self.source_cfg["action"]["post_dispersion_regularization_audit"]
        bounds = np.asarray(self.source_cfg["scientific"]["domain_km"], dtype=np.float64)
        points = cell_centers(bounds, self.resolution)
        reference_resolution = tuple(
            int(value) for value in source["physical_norm_reference_resolution"]
        )
        physical_points = cell_centers(bounds, reference_resolution)
        centers = self.experiment.sensor_bank.centers_km[int(design)]
        sigma = float(self.experiment.sensor_bank.sigma_km)
        order = int(source["trial_order"])
        basis = enriched_basis(points, bounds, centers, sigma, order)
        physical_basis = enriched_basis(
            physical_points, bounds, centers, sigma, order
        )
        physical_gram = fixed_physical_gram(
            physical_basis,
            length_scale=float(source["physical_norm_length_scale_km"]),
        )
        return prepare_direct_ritz_basis(
            basis,
            physical_gram,
            structural_relative_tolerance=float(source["structural_relative_tolerance"]),
        )

    def _row(self, system: Mapping[str, Any], prepared: Any) -> dict[str, Any]:
        weights = normalized_weights(system["log_q_mass"])
        direct = solve_prepared_direct_ritz(
            prepared, weights, np.asarray(system["h"]).ravel()
        ).direct
        tangent = float(system["tangent_action_density"])
        scale = max(abs(tangent), abs(direct.action_qr), 1.0)
        direct_success = bool(
            direct.qr_success
            and direct.svd_success
            and direct.lapack_full_column_rank
            and direct.qr_svd_relative_discrepancy
            <= float(self.action_cfg["direct_qr_svd_relative_tolerance"])
        )
        lower_bound_valid = bool(
            tangent <= direct.action_qr
            + float(self.action_cfg["tangent_full_relative_tolerance"]) * scale
        )
        soft = self.source_cfg["action"]["soft_moment_projection"]
        lambda_dot_valid = bool(
            np.isfinite(system["lambda_dot_solve_relative_residual"])
            and float(system["lambda_dot_solve_relative_residual"])
            <= float(soft["stationarity_residual_tolerance"])
        )
        tangent_compatibility_valid = bool(
            np.isfinite(system["tangent_compatibility_relative_residual"])
            and float(system["tangent_compatibility_relative_residual"])
            <= float(self.source_cfg["action"]["gram_compatibility_relative_tolerance"])
        )
        local_valid = bool(
            system["compatibility_valid"]
            and system["multiplier_coordinate_ready"]
            and lambda_dot_valid
            and tangent_compatibility_valid
            and direct_success
            and lower_bound_valid
        )
        return {
            "contract_hash": self.contract_hash,
            "design_index": int(system["design_index"]),
            "design_id": str(system["design_id"]),
            "source_time_index": int(system["source_time_index"]),
            "day": float(system["day"]),
            "grid_nx": self.resolution[0],
            "grid_ny": self.resolution[1],
            "tangent_action": tangent,
            "direct_action_qr": float(direct.action_qr),
            "direct_action_svd": float(direct.action_svd),
            "direct_qr_svd_relative_difference": float(
                direct.qr_svd_relative_discrepancy
            ),
            "direct_condition_number": float(direct.kappa_c),
            "direct_full_column_rank": bool(direct.lapack_full_column_rank),
            "direct_numerical_success": direct_success,
            "forcing_compatibility_relative_residual": float(
                system["compatibility_relative_residual"]
            ),
            "forcing_compatibility_valid": bool(system["compatibility_valid"]),
            "multiplier_coordinate_ready": bool(system["multiplier_coordinate_ready"]),
            "lambda_dot_solve_relative_residual": float(
                system["lambda_dot_solve_relative_residual"]
            ),
            "lambda_dot_valid": lambda_dot_valid,
            "tangent_compatibility_relative_residual": float(
                system["tangent_compatibility_relative_residual"]
            ),
            "tangent_compatibility_valid": tangent_compatibility_valid,
            "tangent_full_lower_bound_valid": lower_bound_valid,
            "local_valid": local_valid,
            "final_test_accessed": False,
        }

    def evaluate_design(self, design: int) -> dict[str, Any]:
        detail_dir = self.output / "action_details"
        summary_dir = self.output / "action_summaries"
        details_path = detail_dir / f"design_{design:06d}.csv"
        summary_path = summary_dir / f"design_{design:06d}.json"
        if summary_path.is_file() and details_path.is_file():
            cached = json.loads(summary_path.read_text(encoding="utf-8"))
            if cached.get("contract_hash") == self.contract_hash:
                print(f"[ocean percentage] reuse design={design}", flush=True)
                return cached
        rows: list[dict[str, Any]] = []
        if details_path.is_file():
            candidate = [_typed_detail(row) for row in _read_csv(details_path)]
            if candidate and all(
                row.get("contract_hash") == self.contract_hash for row in candidate
            ):
                rows = candidate
        completed = {int(row["source_time_index"]) for row in rows}
        prepared = self._prepare_basis(int(design))
        original_designs = self.pilot.designs.copy()
        self.pilot.designs = np.asarray([design], dtype=int)
        chunk_size = int(self.action_cfg["source_chunk_size"])
        try:
            for start in range(0, len(self.sources), chunk_size):
                chunk = self.sources[start : start + chunk_size]
                missing = np.asarray(
                    [source for source in chunk if int(source) not in completed], dtype=int
                )
                if len(missing) == 0:
                    continue
                self.pilot.source_indices = missing.copy()
                points, dx, log_base, velocity = self.pilot._reference_grid(
                    self.resolution,
                    source_indices=missing,
                    cache_namespace="full_action_production_reference",
                )
                systems = self.pilot._systems_for_grid(
                    self.resolution, points, dx, log_base, velocity
                )
                if len(systems) != len(missing):
                    raise RuntimeError("percentage action chunk is incomplete")
                rows.extend(self._row(system, prepared) for system in systems)
                rows.sort(key=lambda row: int(row["source_time_index"]))
                _write_csv(details_path, rows)
                print(
                    f"[ocean percentage] design={design} "
                    f"completed={len(rows)}/{len(self.sources)}",
                    flush=True,
                )
        finally:
            self.pilot.designs = original_designs
        rows.sort(key=lambda row: int(row["source_time_index"]))
        temporal_cfg = {
            "window_start_source_index": int(self.action_cfg["window_source_indices"][0]),
            "window_end_source_index": int(self.action_cfg["window_source_indices"][1]),
            "source_steps": self.action_cfg["source_steps"],
            "node_counts": self.action_cfg["node_counts"],
            "source_horizon_days": float(self.action_cfg["source_horizon_days"]),
            "maximum_consecutive_relative_action_change": float(
                self.action_cfg["maximum_consecutive_relative_action_change"]
            ),
            "tangent_full_relative_tolerance": float(
                self.action_cfg["tangent_full_relative_tolerance"]
            ),
            "production_authorized_if_certified": True,
        }
        derived = summarize_temporal_levels(rows, temporal_cfg)
        summary = {
            "schema_version": 1,
            "contract_hash": self.contract_hash,
            "design_index": int(design),
            "design_id": str(self.design_ids[int(design)]),
            "risk": float(self.risk[int(design)]),
            "risk_increase_percent": 100.0 * (float(self.risk[int(design)]) / self.r_star - 1.0),
            "tangent_action": float(derived["levels"][-1]["tangent_action"]),
            "full_action": float(derived["levels"][-1]["full_action"]),
            "certified": bool(derived["temporal_quadrature_refinement_certified"]),
            **derived,
            "final_test_accessed": False,
        }
        write_json_atomic(summary_path, summary)
        return summary

    def _cv_risks(self, design: int) -> list[float]:
        values: list[float] = []
        for repeat in (1, 2):
            path = self.experiment._resolve(
                f"experiments/ocean_drifters/processed/iprojection_cv_repeat_{repeat}.npz"
            )
            with np.load(path, allow_pickle=False) as data:
                if bool(data["final_test_accessed"]):
                    raise RuntimeError("repeated-CV risk reports final-test access")
                values.append(float(data["risks"][int(design)]))
        return values

    def build_outputs(self, summaries: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
        candidate_rows = []
        for design in self.candidates:
            summary = dict(summaries[int(design)])
            later_changes = [
                value
                for level in summary["levels"][-2:]
                for value in (
                    level["tangent_relative_change_from_previous"],
                    level["full_relative_change_from_previous"],
                )
            ]
            summary["maximum_later_two_transition_change"] = max(later_changes)
            summary["later_two_transition_stable"] = bool(
                max(later_changes)
                <= float(self.action_cfg["maximum_consecutive_relative_action_change"])
            )
            summary["strict_failure_reason"] = (
                ""
                if summary["certified"]
                else "coarse_ladder_transition_exceeds_5pct"
            )
            candidate_rows.append(summary)
        _write_rows(self.output / "candidate_actions.csv", candidate_rows)
        point_rows: list[dict[str, Any]] = []
        method_rows: list[dict[str, Any]] = []
        previous_full = math.inf
        for percent in self.percentages:
            ceiling = self.r_star * (1.0 + percent / 100.0)
            eligible = self.risk_eligible & (self.risk <= ceiling)
            chosen = choose_methods(self.risk, summaries, eligible)
            certified_eligible = sum(
                bool(summaries[int(design)]["certified"])
                for design in np.flatnonzero(eligible)
                if int(design) in summaries
            )
            full_action = float(summaries[chosen["full"]]["full_action"])
            if full_action > previous_full + 1e-9 * max(abs(previous_full), 1.0):
                raise RuntimeError("nested percentage full action increased")
            previous_full = min(previous_full, full_action)
            law_action = float(summaries[chosen["law"]]["full_action"])
            point_dir = self.output / f"risk_{str(percent).replace('.', 'p')}pct"
            methods: dict[str, Any] = {}
            for method in METHODS:
                design = chosen[method]
                summary = summaries[design]
                bootstrap = self.bootstrap_risk[design]
                cv = self._cv_risks(design)
                record = {
                    "risk_allowance_percent": percent,
                    "method": method,
                    "design_index": design,
                    "design_id": self.design_ids[design],
                    "R_star": self.r_star,
                    "R_max": ceiling,
                    "selection_R": float(self.risk[design]),
                    "selection_R_increase_percent": 100.0 * (
                        float(self.risk[design]) / self.r_star - 1.0
                    ),
                    "selection_budget_used_percent": 100.0 * (
                        float(self.risk[design]) - self.r_star
                    ) / max(ceiling - self.r_star, np.finfo(float).tiny),
                    "selection_tangent_action": float(summary["tangent_action"]),
                    "selection_full_action": float(summary["full_action"]),
                    "full_action_reduction_vs_law_percent": 100.0 * (
                        law_action - float(summary["full_action"])
                    ) / abs(law_action),
                    "certified": bool(summary["certified"]),
                    "development_bootstrap_R_mean": float(np.mean(bootstrap)),
                    "development_bootstrap_R_se": float(np.std(bootstrap, ddof=1)),
                    "repeated_cv_R_values": cv,
                    "repeated_cv_R_mean": float(np.mean(cv)),
                    "sensor_centers_km": self.experiment.sensor_bank.centers_km[design].tolist(),
                    "independent_action_validation_available": False,
                    "final_test_accessed": False,
                }
                method_rows.append(record)
                methods[method] = record
            full_record = methods["full"]
            point = {
                "risk_allowance_percent": percent,
                "risk_allowance_fraction": percent / 100.0,
                "R_star": self.r_star,
                "R_max": ceiling,
                "eligible_layout_count": int(np.sum(eligible)),
                "certified_eligible_layout_count": int(certified_eligible),
                "law_design_id": methods["law"]["design_id"],
                "tangent_design_id": methods["tangent"]["design_id"],
                "full_design_id": methods["full"]["design_id"],
                "full_R_selection": full_record["selection_R"],
                "full_R_excess_selection": full_record["selection_R"] - self.r_star,
                "full_A_selection": full_record["selection_full_action"],
                "law_A_selection": methods["law"]["selection_full_action"],
                "tangent_A_selection": methods["tangent"]["selection_full_action"],
                "selection_action_reduction": (
                    full_record["full_action_reduction_vs_law_percent"] / 100.0
                ),
                "full_certified": full_record["certified"],
                "final_test_accessed": False,
            }
            point_rows.append(point)
            write_json_atomic(point_dir / "result.json", {
                "schema_version": 1,
                "experiment": self.cfg["name"],
                "analysis_scope": self.cfg["validation"]["scope"],
                "law_screens": {
                    "R_star": self.r_star,
                    "R_max": ceiling,
                    "epsilon_r": ceiling - self.r_star,
                },
                "selection": {f"{method}_optimum": chosen[method] for method in METHODS},
                "methods": methods,
                "independent_action_validation_available": False,
                "final_test_accessed": False,
            })
        _write_rows(self.output / "pareto.csv", point_rows)
        write_json_atomic(self.output / "pareto.json", point_rows)
        _write_rows(self.output / "pareto_methods_selection.csv", method_rows)
        validation_rows = [{
            "risk_allowance_percent": row["risk_allowance_percent"],
            "method": row["method"],
            "design_id": row["design_id"],
            "development_bootstrap_R_mean": row["development_bootstrap_R_mean"],
            "development_bootstrap_R_se": row["development_bootstrap_R_se"],
            "legacy_hard_projection_repeated_cv_R_values": row[
                "repeated_cv_R_values"
            ],
            "independent_full_action_validation": "unavailable",
            "validation_scope": "development_only_not_confirmatory",
            "final_test_accessed": False,
        } for row in method_rows]
        _write_rows(self.output / "pareto_methods_validation.csv", validation_rows)
        self._figures(point_rows, method_rows)
        result = {
            "schema_version": 1,
            "experiment": self.cfg["name"],
            "contract_hash": self.contract_hash,
            "risk_allowance_percentages": list(self.percentages),
            "candidate_design_count": len(self.candidates),
            "candidate_design_ids": self.design_ids[self.candidates].tolist(),
            "certified_candidate_count": sum(
                bool(summaries[int(design)]["certified"]) for design in self.candidates
            ),
            "all_local_case_count": 18 * 133,
            "all_local_valid_count": sum(
                int(summaries[int(design)]["local_valid_count"])
                for design in self.candidates
            ),
            "later_two_transition_stable_candidate_count": sum(
                bool(row["later_two_transition_stable"]) for row in candidate_rows
            ),
            "points": point_rows,
            "selection_common_metric": "certified_full_action",
            "analysis_scope": self.cfg["validation"]["scope"],
            "independent_action_validation_available": False,
            "final_test_accessed": False,
        }
        write_json_atomic(self.output / "result.json", result)
        manifest_files = {}
        for path in sorted(self.output.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                name = str(path.relative_to(self.output))
                manifest_files[name] = {
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
        write_json_atomic(self.output / "manifest.json", {
            "schema_version": 1,
            "experiment": self.cfg["name"],
            "contract_hash": self.contract_hash,
            "files": manifest_files,
            "final_test_accessed": False,
        })
        return result

    def _figures(
        self, point_rows: list[Mapping[str, Any]], method_rows: list[Mapping[str, Any]]
    ) -> None:
        percentages = np.asarray(self.percentages)
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
        for method in METHODS:
            rows = [row for row in method_rows if row["method"] == method]
            axes[0, 0].plot(
                percentages,
                [row["selection_full_action"] for row in rows],
                marker=MARKERS[method], color=COLORS[method], label=method.title(),
            )
            axes[0, 1].plot(
                percentages,
                [row["selection_R_increase_percent"] for row in rows],
                marker=MARKERS[method], color=COLORS[method], label=method.title(),
            )
            axes[1, 0].plot(
                percentages,
                [row["full_action_reduction_vs_law_percent"] for row in rows],
                marker=MARKERS[method], color=COLORS[method], label=method.title(),
            )
        axes[0, 0].set_ylabel("Common full action")
        axes[0, 0].set_title("A  Common-metric action")
        axes[0, 1].plot(percentages, percentages, "--", color="#4B9A73", label="risk limit")
        axes[0, 1].set_ylabel("Selection Law-risk increase (%)")
        axes[0, 1].set_title("B  Use of Law-risk allowance")
        axes[1, 0].set_ylabel("Full-action reduction vs Law (%)")
        axes[1, 0].set_title("C  Benefit relative to Law")
        axes[1, 1].plot(
            percentages,
            [row["eligible_layout_count"] for row in point_rows],
            marker="o", color="#5C6670", label="risk-eligible",
        )
        axes[1, 1].plot(
            percentages,
            [row["certified_eligible_layout_count"] for row in point_rows],
            marker="s", color="#4B9A73", label="strict-certified",
        )
        axes[1, 1].set_ylabel("Eligible layout count")
        axes[1, 1].set_title("D  Discrete feasible-set growth")
        axes[1, 1].legend()
        for axis in axes.ravel():
            axis.set_xlabel("Allowed extra Law risk (%)")
            axis.grid(alpha=0.25)
        axes[0, 0].legend(ncol=3)
        fig.suptitle("Ocean drifters · Law, Tangent, and Full percentage comparison")
        fig.savefig(self.output / "pareto_methods.png", dpi=220)
        fig.savefig(self.output / "pareto.png", dpi=220)
        plt.close(fig)

        selected = sorted({int(row["design_index"]) for row in method_rows})
        columns = min(3, len(selected))
        rows_n = int(math.ceil(len(selected) / columns))
        fig, axes = plt.subplots(rows_n, columns, figsize=(4 * columns, 3.5 * rows_n), squeeze=False)
        bounds = self.source_cfg["scientific"]["domain_km"]
        for axis, design in zip(axes.ravel(), selected):
            centers = self.experiment.sensor_bank.centers_km[design]
            axis.scatter(centers[:, 0], centers[:, 1], s=70, color="#D1495B")
            axis.set_title(f"{self.design_ids[design]}\nΔR={100*(self.risk[design]/self.r_star-1):.2f}%")
            axis.set_xlim(bounds[0], bounds[1]); axis.set_ylim(bounds[2], bounds[3])
            axis.set_aspect("equal")
            axis.grid(alpha=0.2)
        for axis in axes.ravel()[len(selected):]:
            axis.set_visible(False)
        fig.suptitle("Sensor layouts selected anywhere on the percentage sweep")
        fig.savefig(self.output / "pareto_sensor_layouts.png", dpi=220)
        plt.close(fig)

    def run(self) -> dict[str, Any]:
        print(
            f"[ocean percentage] candidates={len(self.candidates)} "
            f"allowances={list(self.percentages)} final_test_accessed=false",
            flush=True,
        )
        summaries: dict[int, dict[str, Any]] = {}
        for position, design in enumerate(self.candidates, start=1):
            print(
                f"[ocean percentage] candidate {position}/{len(self.candidates)} "
                f"{self.design_ids[int(design)]}",
                flush=True,
            )
            summaries[int(design)] = self.evaluate_design(int(design))
        return self.build_outputs(summaries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.json")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    output = _resolve(cfg["output"]) if args.output is None else args.output.resolve()
    result = OceanPercentagePareto(cfg, output).run()
    print(json.dumps({
        "candidate_design_count": result["candidate_design_count"],
        "certified_candidate_count": result["certified_candidate_count"],
        "result": str(output / "result.json"),
        "final_test_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
