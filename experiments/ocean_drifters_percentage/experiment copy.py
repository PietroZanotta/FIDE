"""Canonical experiment API for the frozen NOAA ocean-drifter benchmark.

The historical ``scripts/`` directory remains provenance.  This module is the
production-facing orchestration layer, analogous to ``toy_example/experiment.py``
and ``vortices/experiment.py``.  Ordinary stages consume only the frozen 270-ID
development artifact.  A separately guarded one-shot stage can consume the 69-ID
final-test artifact only after explicit authorization.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable, Mapping

import numpy as np

from mfsi.cache import (
    file_sha256,
    fingerprint,
    load_npz_cache,
    save_npz_cache,
    write_json_atomic,
)
from mfsi.measurements import GaussianPointSensors2D
from mfsi.projection import IProjectionConfig


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


class FrozenArtifactError(RuntimeError):
    """Raised when a frozen input is missing or has changed."""


class StageNotReadyError(RuntimeError):
    """Raised when a requested action stage has not passed its readiness gate."""


@dataclass(frozen=True)
class DevelopmentCohort:
    positions: np.ndarray
    ids: np.ndarray
    split: np.ndarray
    normalized_time: np.ndarray
    relative_days: np.ndarray

    @property
    def inference(self) -> np.ndarray:
        return self.positions[self.split == "inference"]

    @property
    def validation(self) -> np.ndarray:
        return self.positions[self.split == "validation"]


@dataclass(frozen=True)
class SensorBank:
    centers_km: np.ndarray
    design_ids: np.ndarray
    styles: np.ndarray
    sigma_km: float

    def eta(self, design_index: int) -> np.ndarray:
        return np.asarray(self.centers_km[int(design_index)]).reshape(-1)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    rows = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _rff_map(points: np.ndarray, omega: np.ndarray, phase: np.ndarray) -> np.ndarray:
    scale = math.sqrt(2.0 / omega.shape[1])
    # Precision is part of the frozen MMD estimator: projected and validation RFF
    # features are stored/evaluated in float32, then averaged and scored in float64.
    return np.asarray(
        scale * np.cos(np.asarray(points, dtype=np.float64) @ omega + phase),
        dtype=np.float32,
    )


def _cell_centres(bounds: np.ndarray, nx: int, ny: int) -> np.ndarray:
    xmin, xmax, ymin, ymax = (float(value) for value in bounds)
    x = xmin + (np.arange(nx, dtype=np.float64) + 0.5) * (xmax - xmin) / nx
    y = ymin + (np.arange(ny, dtype=np.float64) + 0.5) * (ymax - ymin) / ny
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.stack((xx.ravel(), yy.ravel()), axis=-1)


def _gaussian_features(points: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    delta = np.asarray(points)[..., None, :] - np.asarray(centers)
    return np.exp(-0.5 * np.sum(delta * delta, axis=-1) / float(sigma) ** 2)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty() -> bool | None:
    try:
        return bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


class OceanDriftersExperiment:
    """Frozen NOAA benchmark using the shared MFSI experiment conventions."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = json.loads(json.dumps(cfg))
        self._validate_config()
        self.paths = {
            name: self._resolve(entry["path"] if isinstance(entry, dict) else entry)
            for name, entry in self.cfg["artifacts"].items()
            if name != "reference_density_cache"
        }
        self.reference_density_cache = self._resolve(
            self.cfg["artifacts"]["reference_density_cache"]
        )
        self._validate_frozen_artifacts()
        self.cohort = self._load_development_cohort()
        self.sensor_bank = self._load_sensor_bank()
        self.family = GaussianPointSensors2D(
            width=self.sensor_bank.sigma_km,
            n_sensors=int(self.cfg["scientific"]["sensors_per_layout"]),
        )
        self._reference = None
        self._admissibility_rows = _read_csv(self.paths["numerical_admissibility_table"])
        self._admissible = np.asarray(
            [row["numerically_admissible"] == "True" for row in self._admissibility_rows],
            dtype=bool,
        )
        self._validate_scientific_shapes()

    @staticmethod
    def _resolve(path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else REPO_ROOT / path

    def _validate_config(self) -> None:
        scientific = self.cfg.get("scientific", {})
        final_evaluation = self.cfg.get("final_evaluation", {})
        final_access = scientific.get("final_test_access_allowed")
        authorization = final_evaluation.get("authorization_status")
        valid_authorization_pair = (
            final_access is False
            and authorization == "awaiting_explicit_user_authorization"
        ) or (
            final_access is True
            and authorization == "explicit_user_authorization_recorded"
        ) or (
            final_access is False
            and authorization == "one_shot_final_evaluation_consumed"
        )
        if not valid_authorization_pair:
            raise FrozenArtifactError(
                "final-test access requires a consistent explicit two-key authorization"
            )
        release_hash = final_evaluation.get("release_manifest_expected_sha256", "")
        if len(release_hash) != 64 or any(
            character not in "0123456789abcdef" for character in release_hash
        ):
            raise FrozenArtifactError("the pre-final release manifest hash is not frozen")
        if (
            final_evaluation.get("locked_cohort_path")
            != "experiments/ocean_drifters/processed/cohort_45d.npz"
            or final_evaluation.get("locked_cohort_expected_sha256")
            != "798943884c324ffddd93dd6d16cf6246048b3740bc9a217f806b563f9f2fa338"
            or final_evaluation.get("split_manifest_path")
            != "experiments/ocean_drifters/processed/splits/split_manifest.csv"
            or int(final_evaluation.get("final_test_id_count", -1)) != 69
            or int(final_evaluation.get("selected_design_index", -1)) != 216
            or final_evaluation.get("selected_design_id") != "design_000216"
            or int(final_evaluation.get("evaluation_time_count", -1)) != 19
            or final_evaluation.get("acceptance_rule")
            != "final_rff_mmd2_le_validation_bootstrap_one_sided_95pct_upper"
            or float(final_evaluation.get("validation_bootstrap_quantile", math.nan))
            != 0.95
            or float(final_evaluation.get("validation_bootstrap_upper_bound", math.nan))
            != 0.023351205336727864
            or final_evaluation.get("failure_policy")
            != "report_without_retuning_or_reopening_selection"
            or final_evaluation.get("canonical_output_dir")
            != "experiments/ocean_drifters/outputs/final_evaluation"
        ):
            raise FrozenArtifactError("the predeclared final-evaluation contract changed")
        for entry in self.cfg.get("artifacts", {}).values():
            path = entry.get("path") if isinstance(entry, dict) else entry
            if path and "final_test" in str(path).lower():
                raise FrozenArtifactError("the production API may not configure a final-test artifact")
        if self.cfg.get("projection", {}).get("backend") != "tesseract_cpp":
            raise FrozenArtifactError("the frozen production I-projection backend is tesseract_cpp")
        if float(self.cfg.get("law", {}).get(
            "max_relative_risk_violation", math.nan
        )) != 0.05:
            raise FrozenArtifactError(
                "the ocean extra-risk allowance must remain 5% of Law's risk"
            )
        if "frozen_additive_epsilon" in self.cfg.get("law", {}):
            raise FrozenArtifactError(
                "the ocean law may not use a fixed additive risk allowance"
            )
        poisson_backend = self.cfg.get("action", {}).get("poisson_backend")
        if poisson_backend not in {
            "tesseract_cpp",
            "tesseract_cpp_variational_ritz",
        }:
            raise FrozenArtifactError(
                "the ocean Poisson backend must be tesseract_cpp or "
                "tesseract_cpp_variational_ritz"
            )
        if poisson_backend == "tesseract_cpp_variational_ritz":
            variational = self.cfg["action"].get("variational_poisson", {})
            if float(variational.get("operator_floor", math.nan)) != 0.0:
                raise FrozenArtifactError(
                    "the ocean variational Poisson backend requires operator_floor=0"
                )
            if variational.get("density_floor_or_threshold_allowed") is not False:
                raise FrozenArtifactError(
                    "the ocean variational Poisson backend may not modify the density"
                )
        post = self.cfg.get("action", {}).get(
            "post_dispersion_regularization_audit", {}
        )
        if tuple(float(value) for value in scientific.get(
            "action_window_days", ()
        )) != (12.0, 45.0):
            raise FrozenArtifactError(
                "the ocean action window must remain frozen at [12,45] days"
            )
        if (
            float(post.get("window_start_day_inclusive", math.nan)) != 12.0
            or float(post.get("window_end_day_inclusive", math.nan)) != 45.0
            or float(post.get("source_horizon_days", math.nan)) != 45.0
            or post.get("primary_action") != "unregularized_direct_qr_ritz"
            or post.get("production_run_authorized") is not False
            or post.get("scientific_ranking_allowed") is not False
            or post.get("final_test_accessed") is not False
        ):
            raise FrozenArtifactError(
                "the ocean post-dispersion action contract changed"
            )
        temporal = self.cfg.get("action", {}).get(
            "post_dispersion_temporal_refinement", {}
        )
        if (
            int(temporal.get("design_count", -1)) != 1
            or temporal.get("source_steps") != [12, 6, 3, 1]
            or temporal.get("node_counts") != [12, 23, 45, 133]
            or temporal.get("grid_resolution") != [511, 273]
            or float(temporal.get(
                "maximum_consecutive_relative_action_change", math.nan
            )) != 0.05
            or temporal.get("require_all_consecutive_refinements") is not True
            or temporal.get("production_authorized_if_certified") is not True
            or temporal.get("scientific_ranking_allowed") is not False
            or temporal.get("final_test_accessed") is not False
        ):
            raise FrozenArtifactError(
                "the ocean temporal-refinement contract changed"
            )
        soft = self.cfg.get("action", {}).get("soft_moment_projection", {})
        if bool(soft.get("enabled", False)):
            if soft.get("finite_sample_standard_error_floor_rule") != (
                "inverse_inference_trajectory_count"
            ):
                raise FrozenArtifactError(
                    "the ocean soft projection requires the inference-only 1/n floor rule"
                )
            if not (0.0 < float(soft.get("stationarity_residual_tolerance", math.nan)) <= 1e-8):
                raise FrozenArtifactError(
                    "the ocean soft-projection stationarity tolerance must be in (0, 1e-8]"
                )
            rank_tolerances = tuple(float(value) for value in soft.get(
                "tangent_rank_sensitivity_relative_tolerances", ()
            ))
            if rank_tolerances != (1e-10, 1e-12, 1e-14):
                raise FrozenArtifactError(
                    "the ocean tangent rank-sensitivity cutoffs must be 1e-10, 1e-12, 1e-14"
                )
            if not (0.0 < float(soft.get(
                "maximum_relative_tangent_rank_action_change", math.nan
            )) <= 0.1):
                raise FrozenArtifactError(
                    "the ocean tangent rank-sensitivity tolerance must be in (0, 0.1]"
                )
            if float(soft.get("minimum_valid_layout_fraction", math.nan)) != 0.95:
                raise FrozenArtifactError(
                    "the ocean tangent validity fraction must remain frozen at 0.95"
                )
        production = self.cfg.get("action", {}).get(
            "full_action_production", {}
        )
        if production:
            if (
                int(production.get("layout_count", -1)) != 1
                or int(production.get("time_count", -1)) != 181
            ):
                raise FrozenArtifactError(
                    "the revised ocean full-action production set must remain 1x181"
                )
            if production.get("production_run_authorized") is not False:
                raise FrozenArtifactError(
                    "revised ocean production must remain unauthorized until re-audited"
                )
            pilot_grids = self.cfg["action"]["poisson_pilot"][
                "grid_resolutions"
            ]
            if production.get("grid_resolution") != pilot_grids[-1]:
                raise FrozenArtifactError(
                    "ocean production must retain the pilot-authorized fine grid"
                )
            if float(production.get(
                "minimum_valid_layout_fraction", math.nan
            )) != 0.95:
                raise FrozenArtifactError(
                    "the ocean production validity fraction must remain 0.95"
                )

    def _validate_frozen_artifacts(self) -> None:
        for name, entry in self.cfg["artifacts"].items():
            if not isinstance(entry, dict):
                continue
            path = self._resolve(entry["path"])
            if not path.is_file():
                raise FrozenArtifactError(f"missing frozen artifact {name}: {path}")
            expected = entry.get("sha256")
            if expected and file_sha256(path) != expected:
                raise FrozenArtifactError(
                    f"frozen artifact hash mismatch for {name}: {path}"
                )

    def _load_development_cohort(self) -> DevelopmentCohort:
        with np.load(self.paths["development_cohort"], allow_pickle=False) as data:
            return DevelopmentCohort(
                positions=np.asarray(data["X"], dtype=np.float64),
                ids=np.asarray(data["ids"], dtype=np.int64),
                split=np.asarray(data["split"]).astype(str),
                normalized_time=np.asarray(data["normalized_time"], dtype=np.float64),
                relative_days=np.asarray(data["relative_days"], dtype=np.float64),
            )

    def _load_sensor_bank(self) -> SensorBank:
        with np.load(self.paths["sensor_bank"], allow_pickle=False) as data:
            return SensorBank(
                centers_km=np.asarray(data["centers_km"], dtype=np.float64),
                design_ids=np.asarray(data["design_id"]).astype(str),
                styles=np.asarray(data["style"]).astype(str),
                sigma_km=float(data["sigma_km"]),
            )

    def _validate_scientific_shapes(self) -> None:
        frozen = self.cfg["scientific"]
        counts = frozen["trajectory_counts"]
        got = {
            "inference": int(np.sum(self.cohort.split == "inference")),
            "validation": int(np.sum(self.cohort.split == "validation")),
        }
        if got != {"inference": counts["inference"], "validation": counts["validation"]}:
            raise FrozenArtifactError(f"development split changed: {got}")
        if len(self.cohort.positions) != counts["inference"] + counts["validation"]:
            raise FrozenArtifactError("development artifact contains an unexpected ID count")
        if len(self.sensor_bank.centers_km) != frozen["layout_count"]:
            raise FrozenArtifactError("sensor layout count changed")
        if self.sensor_bank.centers_km.shape[1] != frozen["sensors_per_layout"]:
            raise FrozenArtifactError("sensors-per-layout changed")
        if self.sensor_bank.sigma_km != frozen["sensor_sigma_km"]:
            raise FrozenArtifactError("sensor sigma changed")
        expected_ids = self.sensor_bank.design_ids.tolist()
        table_ids = [row["design_id"] for row in self._admissibility_rows]
        if table_ids != expected_ids or len(self._admissible) != frozen["layout_count"]:
            raise FrozenArtifactError("admissibility manifest does not match the sensor bank")
        if int(np.sum(self._admissible)) != 512:
            raise FrozenArtifactError("the frozen numerical class must contain all 512 layouts")

    def reference(self):
        """Load the frozen shared ``ReferenceFlow`` implementation lazily."""
        if self._reference is None:
            from mfsi.reference import DomainPreservingReferenceFlow

            self._reference = DomainPreservingReferenceFlow.from_npz(
                self.paths["reference_checkpoint"], substeps_per_interval=2
            )
        return self._reference

    def measurements(self) -> GaussianPointSensors2D:
        """Return the shared Gaussian point-sensor family."""
        return self.family

    def numerical_admissibility(self) -> dict[str, Any]:
        summary = json.loads(
            self.paths["numerical_admissibility_summary"].read_text(encoding="utf-8")
        )
        adaptive = _read_csv(self.paths["adaptive_case_summary"])
        classifications = {
            "resolved": sum(row["classification"] == "A_resolved" for row in adaptive),
            "numerically_unresolved_but_mathematically_feasible": sum(
                row["classification"].startswith("B_") for row in adaptive
            ),
            "implementation_error": sum(row["classification"].startswith("C_") for row in adaptive),
        }
        if classifications != {
            "resolved": 27,
            "numerically_unresolved_but_mathematically_feasible": 0,
            "implementation_error": 0,
        }:
            raise FrozenArtifactError(f"adaptive classifications changed: {classifications}")
        return {
            "schema_version": 1,
            "experiment": self.cfg["name"],
            "frozen_before_validation_risk": True,
            "evaluation_time_count": int(self.cfg["scientific"]["risk_evaluation_time_count"]),
            "attempted_concentrated_cases": int(summary["attempted_projection_count"]),
            "classifications": classifications,
            "admissible_layout_count": int(np.sum(self._admissible)),
            "excluded_layout_count": int(np.sum(~self._admissible)),
            "admissible_design_ids": self.sensor_bank.design_ids[self._admissible].tolist(),
            "excluded_design_ids": self.sensor_bank.design_ids[~self._admissible].tolist(),
            "source_table": str(self.paths["numerical_admissibility_table"].relative_to(REPO_ROOT)),
            "source_table_sha256": file_sha256(self.paths["numerical_admissibility_table"]),
            "final_test_accessed": False,
        }

    def _risk_signature(self, *, designs: np.ndarray, time_positions: np.ndarray, replicates: int) -> str:
        return fingerprint({
            "schema": 2,
            "rff_precision_contract": "float32_features_float64_means_and_scores",
            "scientific": self.cfg["scientific"],
            "law": self.cfg["law"],
            "designs": designs.tolist(),
            "time_positions": time_positions.tolist(),
            "bootstrap_replicates": int(replicates),
            "development_sha256": file_sha256(self.paths["development_cohort"]),
            "sensor_bank_sha256": file_sha256(self.paths["sensor_bank"]),
            "admissibility_sha256": file_sha256(self.paths["numerical_admissibility_table"]),
            "projection_embeddings_sha256": file_sha256(self.paths["risk_projection_embeddings"]),
        })

    def scientific_risk(
        self,
        *,
        design_indices: np.ndarray | None = None,
        time_positions: np.ndarray | None = None,
        bootstrap_replicates: int | None = None,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Evaluate frozen validation-ID RFF-MMD risk through the canonical API."""
        with np.load(self.paths["risk_projection_embeddings"], allow_pickle=False) as data:
            if bool(data["final_test_accessed"]):
                raise FrozenArtifactError("risk inputs report final-test access")
            evaluation_indices = np.asarray(data["evaluation_indices"], dtype=int)
            evaluation_days = np.asarray(data["evaluation_days"], dtype=np.float64)
            source_ids = np.asarray(data["design_id"]).astype(str)
            source_eligible = np.asarray(data["eligible"], dtype=bool)
            projected = np.asarray(data["projected_rff_embedding"], dtype=np.float64)
            frozen_risks = np.asarray(data["risks"], dtype=np.float64)
            frozen_bootstrap_risk = np.asarray(data["bootstrap_risk"], dtype=np.float64)
            omega = np.asarray(data["rff_omega"], dtype=np.float64)
            phase = np.asarray(data["rff_phase"], dtype=np.float64)
            bandwidth = float(data["bandwidth_km"])
        if not np.array_equal(source_ids, self.sensor_bank.design_ids):
            raise FrozenArtifactError("risk embeddings do not match the sensor bank")
        if not np.array_equal(source_eligible, self._admissible):
            raise FrozenArtifactError("risk embeddings do not match numerical admissibility")
        if omega.shape != (2, int(self.cfg["law"]["rff_features"])):
            raise FrozenArtifactError("RFF feature configuration changed")
        if not np.isclose(bandwidth, float(self.cfg["law"]["bandwidth_km"]), rtol=0.0, atol=1e-12):
            raise FrozenArtifactError("RFF bandwidth changed")

        designs = (
            np.flatnonzero(self._admissible) if design_indices is None
            else np.asarray(design_indices, dtype=int)
        )
        positions = (
            np.arange(len(evaluation_indices), dtype=int) if time_positions is None
            else np.asarray(time_positions, dtype=int)
        )
        if np.any(~self._admissible[designs]):
            raise FrozenArtifactError("risk requested for a numerically inadmissible design")
        replicates = int(
            self.cfg["law"]["bootstrap_replicates"]
            if bootstrap_replicates is None else bootstrap_replicates
        )
        signature = self._risk_signature(
            designs=designs, time_positions=positions, replicates=replicates
        )
        cache_path = Path(output_dir) / "cache/risk.npz" if output_dir is not None else None
        cached = load_npz_cache(cache_path, signature=signature) if cache_path else None
        if cached is not None:
            arrays, _ = cached
            risk = arrays["risk"]
            risk_by_time = arrays["risk_by_time"]
            bootstrap_risk = arrays["bootstrap_risk"]
        else:
            validation = self.cohort.validation[:, evaluation_indices[positions]]
            validation_features = _rff_map(validation, omega, phase)
            validation_embedding = validation_features.mean(axis=0, dtype=np.float64)
            selected_projected = projected[designs][:, positions]
            difference = selected_projected - validation_embedding[None]
            risk_by_time = np.sum(difference * difference, axis=-1)
            risk = risk_by_time.mean(axis=1)
            rng = np.random.default_rng(int(self.cfg["law"]["bootstrap_seed"]))
            bootstrap_ids = rng.integers(
                0, len(validation), size=(replicates, len(validation))
            )
            bootstrap_embedding = validation_features[bootstrap_ids].mean(axis=1, dtype=np.float64)
            projected_sq = np.sum(selected_projected * selected_projected, axis=-1)
            bootstrap_sq = np.sum(bootstrap_embedding * bootstrap_embedding, axis=-1)
            cross = np.einsum(
                "dtf,btf->dbt", selected_projected, bootstrap_embedding, optimize=True
            )
            bootstrap_risk = np.mean(
                projected_sq[:, None] + bootstrap_sq[None] - 2.0 * cross, axis=-1
            )
            if cache_path:
                save_npz_cache(
                    cache_path,
                    {
                        "design_indices": designs,
                        "time_positions": positions,
                        "risk": risk,
                        "risk_by_time": risk_by_time,
                        "bootstrap_risk": bootstrap_risk,
                    },
                    signature=signature,
                    metadata={"final_test_accessed": False},
                )

        if len(designs) == 512 and len(positions) == 19:
            maximum_difference = float(np.max(np.abs(risk - frozen_risks[designs])))
            if maximum_difference > 2e-12:
                raise FrozenArtifactError(
                    f"canonical risk disagrees with the frozen result by {maximum_difference:.3e}"
                )
            if replicates == frozen_bootstrap_risk.shape[1]:
                bootstrap_difference = float(
                    np.max(np.abs(bootstrap_risk - frozen_bootstrap_risk[designs]))
                )
                if bootstrap_difference > 2e-12:
                    raise FrozenArtifactError(
                        "canonical validation-ID bootstrap disagrees with the frozen result by "
                        f"{bootstrap_difference:.3e}"
                    )
        alpha = 1.0 - float(self.cfg["law"]["confidence_level"])
        best_local = int(np.argmin(risk))
        best_design = int(designs[best_local])
        risk_freeze = json.loads(self.paths["risk_freeze"].read_text(encoding="utf-8"))
        if len(designs) == 512 and best_design != int(risk_freeze["best_design_id"].split("_")[-1]):
            raise FrozenArtifactError("canonical risk changed the frozen best design")
        relative_allowance = float(self.cfg["law"]["max_relative_risk_violation"])
        law_risk_star = float(risk_freeze["best_validation_risk"])
        epsilon = relative_allowance * abs(law_risk_star)
        risk_ceiling = law_risk_star + epsilon
        if not np.isclose(
            epsilon,
            float(risk_freeze["frozen_additive_epsilon"]),
            rtol=0.0,
            atol=1e-15,
        ):
            raise FrozenArtifactError("frozen relative risk allowance changed")
        if float(risk_freeze.get("relative_risk_allowance", math.nan)) != (
            relative_allowance
        ):
            raise FrozenArtifactError("frozen relative risk fraction changed")
        near_local = np.flatnonzero(risk <= risk_ceiling)
        near_design_ids = self.sensor_bank.design_ids[designs[near_local]].tolist()
        if len(designs) == 512 and set(near_design_ids) != set(risk_freeze["near_optimal_design_ids"]):
            raise FrozenArtifactError("canonical risk changed the frozen near-optimal set")
        diagnostics = {
            int(row["design_index"]): row
            for row in _read_csv(self.paths["projection_diagnostics_table"])
        }
        rows: list[dict[str, Any]] = []
        order = np.argsort(risk)
        rank = np.empty(len(designs), dtype=int)
        rank[order] = np.arange(1, len(designs) + 1)
        bootstrap_rank = np.empty_like(bootstrap_risk, dtype=np.int32)
        for replicate in range(bootstrap_risk.shape[1]):
            bootstrap_rank[np.argsort(bootstrap_risk[:, replicate]), replicate] = np.arange(
                1, len(designs) + 1
            )
        for local, design in enumerate(designs):
            lower, upper = np.quantile(
                bootstrap_risk[local], [0.5 * alpha, 1.0 - 0.5 * alpha]
            )
            rank_lower, rank_upper = np.quantile(
                bootstrap_rank[local], [0.5 * alpha, 1.0 - 0.5 * alpha]
            )
            diag = diagnostics.get(int(design), {})
            rows.append({
                "design_index": int(design),
                "design_id": self.sensor_bank.design_ids[design],
                "rank": int(rank[local]),
                "risk": float(risk[local]),
                "bootstrap_standard_error": float(np.std(bootstrap_risk[local], ddof=1)),
                "bootstrap_ci_lower": float(lower),
                "bootstrap_ci_upper": float(upper),
                "bootstrap_rank_median": float(np.median(bootstrap_rank[local])),
                "bootstrap_rank_ci_lower": float(rank_lower),
                "bootstrap_rank_ci_upper": float(rank_upper),
                "projection_valid": True,
                "law_risk_valid": bool(np.isfinite(risk[local])),
                "multiplier_dynamics_valid": False,
                "tangent_action_valid": False,
                "full_action_valid": False,
                "maximum_projection_residual": diag.get("maximum_moment_residual", ""),
                "mean_projection_kl": diag.get("mean_projection_kl", ""),
                "minimum_log10_intrinsic_ess": diag.get("minimum_log10_intrinsic_ess", ""),
                "worst_covariance_condition": diag.get("worst_covariance_condition_regularized", ""),
            })
        return {
            "signature": signature,
            "design_indices": designs,
            "time_positions": positions,
            "evaluation_indices": evaluation_indices[positions],
            "evaluation_days": evaluation_days[positions],
            "risk": risk,
            "risk_by_time": risk_by_time,
            "bootstrap_risk": bootstrap_risk,
            "rows": rows,
            "summary": {
                "design_count": len(designs),
                "evaluation_time_count": len(positions),
                "validation_id_count": len(self.cohort.validation),
                "bootstrap_replicates": replicates,
                "bootstrap_unit": "complete validation drifter ID",
                "best_design_index": best_design,
                "best_design_id": self.sensor_bank.design_ids[best_design],
                "R_star": law_risk_star,
                "relative_risk_allowance": relative_allowance,
                "frozen_additive_epsilon": epsilon,
                "risk_ceiling": risk_ceiling,
                "near_optimal_layout_count": len(near_design_ids),
                "near_optimal_design_ids": near_design_ids,
                "kernel": self.cfg["law"]["metric"],
                "bandwidth_km": bandwidth,
                "rff_features": omega.shape[1],
                "time_weighting": "uniform",
                "final_test_accessed": False,
            },
        }

    def smoke_projection(self) -> dict[str, Any]:
        """Exercise the native projector on a small cached continuous grid."""
        smoke = self.cfg["smoke_run"]
        designs = np.asarray(smoke["design_indices"], dtype=int)
        source_indices = np.asarray(smoke["source_time_indices"], dtype=int)
        nx, ny = (int(value) for value in self.cfg["projection"]["grid_resolution"])
        points = _cell_centres(np.asarray(self.cfg["scientific"]["domain_km"]), nx, ny)
        import jax.numpy as jnp

        reference_velocity = np.asarray(
            self.reference().velocity(
                jnp.asarray(points[[0, len(points) // 2, -1]]),
                jnp.asarray(float(source_indices[0]) / 180.0),
            )
        )
        if not np.isfinite(reference_velocity).all():
            raise FrozenArtifactError("the frozen reference returned a non-finite smoke velocity")
        checkpoint_prefix = file_sha256(self.paths["reference_checkpoint"])[:12]
        log_base = []
        for source in source_indices:
            cache = self.reference_density_cache / (
                f"density_{checkpoint_prefix}_t{int(source):03d}_{nx}x{ny}.npz"
            )
            if not cache.is_file():
                raise FrozenArtifactError(f"smoke density cache is missing: {cache}")
            with np.load(cache, allow_pickle=False) as data:
                values = np.asarray(data["log_base_mass"], dtype=np.float64)
            if not np.isfinite(np.exp(values).sum()) or abs(np.exp(values).sum() - 1.0) > 1e-10:
                raise FrozenArtifactError(f"invalid continuous-density cache: {cache}")
            log_base.append(values)
        log_base_array = np.stack(log_base)
        inference = self.cohort.inference[:, source_indices]
        targets = []
        phi_by_design = []
        for design in designs:
            centres = self.sensor_bank.centers_km[design]
            phi = _gaussian_features(points, centres, self.sensor_bank.sigma_km)
            phi_by_design.append(np.broadcast_to(phi, (len(source_indices), *phi.shape)))
            targets.append(
                _gaussian_features(inference, centres, self.sensor_bank.sigma_km).mean(axis=0)
            )
        from mfsi.projection_tesseract import solve_i_projection_trajectory_tesseract_forward

        p = self.cfg["projection"]
        solver_cfg = IProjectionConfig(
                max_steps=int(p["max_steps"]), residual_tol=float(p["residual_tol"]),
                newton_ridge=float(p["newton_ridge"]), step_cap=float(p["step_cap"]),
                lambda_clip=float(p["lambda_clip"]),
                line_search_steps=int(p["line_search_steps"]), implicit_ridge=0.0,
            )
        # The native kernel shares one feature geometry across its leading target
        # batch.  Smoke normally uses one design; retaining this loop keeps the
        # public smoke schema correct if more objective-blind designs are added.
        residual_blocks = []
        for phi, target in zip(phi_by_design, targets, strict=True):
            native = solve_i_projection_trajectory_tesseract_forward(
                np.asarray(phi, dtype=np.float64),
                log_base_array,
                np.asarray(target, dtype=np.float64)[None],
                solver_cfg,
            )
            residual_blocks.append(np.asarray(native["residual_norm"], dtype=np.float64)[0])
        residual = np.stack(residual_blocks)
        converged = residual <= float(p["acceptance_tol"])
        return {
            "backend": "tesseract_cpp",
            "design_indices": designs.tolist(),
            "source_time_indices": source_indices.tolist(),
            "grid_resolution": [nx, ny],
            "maximum_moment_residual": float(np.max(residual)),
            "all_converged": bool(np.all(converged)),
            "reference_density_normalized": True,
            "reference_velocity_finite": True,
        }

    def post_dispersion_action(self):
        """Return the certified singleton [12,45] production-action API."""
        try:
            from .post_dispersion_regularization import OceanPostDispersionAction
        except ImportError:  # direct script invocation
            from post_dispersion_regularization import OceanPostDispersionAction
        action = OceanPostDispersionAction(
            self.cfg["action"]["post_dispersion_regularization_audit"],
            self.paths["post_dispersion_action_audit"],
            self.sensor_bank.design_ids,
            self.paths["post_dispersion_temporal_details"],
            self.paths["post_dispersion_temporal_summary"],
        )
        freeze = json.loads(
            self.paths["risk_freeze"].read_text(encoding="utf-8")
        )
        risk_designs = tuple(
            int(value.split("_")[-1])
            for value in freeze["near_optimal_design_ids"]
        )
        if action.design_indices != risk_designs:
            raise FrozenArtifactError(
                "post-dispersion audit does not match the relative-risk set"
            )
        return action

    def action_status(self, stage: str) -> dict[str, Any]:
        if stage == "post_dispersion_action":
            payload = self.post_dispersion_action().result_payload()
            return {
                "stage": stage,
                "window_days": payload["window_days"],
                "layout_count": payload["layout_count"],
                "valid_layout_count": payload["valid_layout_count"],
                "all_layouts_valid": payload["all_layouts_valid"],
                "primary_formulation": payload["primary_formulation"],
                "temporal_quadrature_refinement_certified": payload[
                    "temporal_quadrature_refinement_certified"
                ],
                "production_run": payload["production_run"],
            }
        if stage == "tangent_action":
            path = self._resolve(self.cfg["action"]["tangent_readiness_table"])
            if not path.is_file():
                raise StageNotReadyError(
                    "tangent-action readiness has not been completed through the canonical API"
                )
            rows = _read_csv(path)
            return {
                "stage": stage,
                "layout_count": len(rows),
                "valid_count": sum(row.get("valid") == "True" for row in rows),
                "backend": self.cfg["action"]["iprojection_backend"],
            }
        if stage == "full_action":
            path = self._resolve(self.cfg["action"]["poisson_pilot_table"])
            if not path.is_file():
                raise StageNotReadyError(
                    "the weighted-Poisson pilot has not been completed"
                )
            rows = _read_csv(path)
            return {
                "stage": stage,
                "pilot_layout_count": len(rows),
                "pilot_backend_valid_count": sum(
                    row.get("pilot_full_action_valid") == "True" for row in rows
                ),
                "production_sweep_authorized": bool(rows) and all(
                    row.get("pilot_full_action_valid") == "True" for row in rows
                ),
                "backend": self.cfg["action"]["poisson_backend"],
                "full_action_valid": False,
            }
        if stage == "full_action_production":
            path = self._resolve(
                self.cfg["action"]["full_action_production"]["summary_table"]
            )
            if not path.is_file():
                raise StageNotReadyError(
                    "the production full-action sweep has not been completed"
                )
            rows = _read_csv(path)
            return {
                "stage": stage,
                "production_layout_count": len(rows),
                "valid_layout_count": sum(
                    row.get("full_action_valid") == "True" for row in rows
                ),
                "backend": self.cfg["action"]["poisson_backend"],
                "full_action_valid": bool(rows) and (
                    sum(row.get("full_action_valid") == "True" for row in rows)
                    / len(rows)
                    >= float(self.cfg["action"]["full_action_production"][
                        "minimum_valid_layout_fraction"
                    ])
                ),
            }
        raise ValueError(f"unknown action stage {stage!r}")

    def provenance(self) -> dict[str, Any]:
        cohort_manifest = json.loads(self.paths["cohort_manifest"].read_text(encoding="utf-8"))
        return {
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
            "config_hash": fingerprint(self.cfg),
            "seed": int(self.cfg["seed"]),
            "split_manifest_sha256": cohort_manifest["split_manifest_sha256"],
            "artifacts": {
                name: {
                    "path": str(self._resolve(entry["path"]).relative_to(REPO_ROOT)),
                    "sha256": file_sha256(self._resolve(entry["path"])),
                }
                for name, entry in self.cfg["artifacts"].items()
                if isinstance(entry, dict)
            },
            "mmd": dict(self.cfg["law"]),
            "grid_resolution": list(self.cfg["projection"]["grid_resolution"]),
            "solver": dict(self.cfg["projection"]),
            "final_test_accessed": False,
        }


def _write_run_outputs(output_dir: Path, payload: dict[str, Any], experiment: OceanDriftersExperiment) -> None:
    write_json_atomic(output_dir / "result.json", payload)
    files = {}
    for path in sorted(output_dir.glob("result*")):
        if path.is_file():
            files[path.name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    for name in ("numerical_admissibility_manifest.json", "cache/risk.npz"):
        path = output_dir / name
        if path.is_file():
            files[name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    write_json_atomic(output_dir / "manifest.json", {
        "schema_version": 1,
        "experiment": experiment.cfg["name"],
        "config_hash": fingerprint(experiment.cfg),
        "files": files,
        "provenance": experiment.provenance(),
    })


def run_experiment(
    cfg: dict[str, Any], output_dir: Path, *, smoke: bool = False, stage: str = "benchmark"
) -> dict[str, Any]:
    """Run one canonical ocean stage, mirroring the other experiment entrypoint."""
    started = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment = OceanDriftersExperiment(cfg)
    admissibility = experiment.numerical_admissibility()
    write_json_atomic(output_dir / "numerical_admissibility_manifest.json", admissibility)

    if stage in {"final_evaluation_dry_run", "final_evaluation"}:
        try:
            from .final_evaluation import (
                dry_run_final_evaluation,
                run_locked_final_evaluation,
            )
        except ImportError:  # direct script invocation
            from final_evaluation import (  # type: ignore
                dry_run_final_evaluation,
                run_locked_final_evaluation,
            )
        final_result = (
            dry_run_final_evaluation(experiment)
            if stage == "final_evaluation_dry_run"
            else run_locked_final_evaluation(experiment, output_dir)
        )
        payload = {
            "schema_version": 1,
            "experiment": cfg["name"],
            "smoke": False,
            "stage": stage,
            "numerical_admissibility": admissibility,
            stage: final_result,
            "statuses": {
                "projection_valid": admissibility["excluded_layout_count"] == 0,
                "law_risk_valid": True,
                "final_evaluation_ready": stage == "final_evaluation_dry_run",
                "final_evaluation_accepted": (
                    final_result.get("accepted") if stage == "final_evaluation" else None
                ),
            },
            "provenance": experiment.provenance(),
            "elapsed_seconds": time.perf_counter() - started,
            "final_test_accessed": stage == "final_evaluation",
        }
        _write_run_outputs(output_dir, payload, experiment)
        return payload
    if smoke:
        projection = experiment.smoke_projection()
        if not projection["all_converged"]:
            raise RuntimeError("native smoke I-projection did not meet the frozen residual gate")
        smoke_cfg = cfg["smoke_run"]
        source_indices = np.asarray(smoke_cfg["source_time_indices"], dtype=int)
        with np.load(experiment.paths["risk_projection_embeddings"], allow_pickle=False) as data:
            evaluation_indices = np.asarray(data["evaluation_indices"], dtype=int)
        positions = np.asarray([
            int(np.flatnonzero(evaluation_indices == source)[0]) for source in source_indices
        ])
        risk = experiment.scientific_risk(
            design_indices=np.asarray(smoke_cfg["design_indices"], dtype=int),
            time_positions=positions,
            bootstrap_replicates=int(cfg["law"]["bootstrap_replicates"]),
            output_dir=output_dir,
        )
        payload = {
            "schema_version": 1,
            "experiment": cfg["name"],
            "smoke": True,
            "stage": "risk",
            "smoke_design_ids": [
                experiment.sensor_bank.design_ids[index]
                for index in smoke_cfg["design_indices"]
            ],
            "smoke_projection": projection,
            "smoke_metrics": risk["summary"],
            "statuses": {
                "projection_valid": projection["all_converged"],
                "law_risk_valid": True,
                "multiplier_dynamics_valid": False,
                "tangent_action_valid": False,
                "full_action_valid": False,
                "post_dispersion_action_valid": False,
                "full_horizon_action_valid": False,
            },
            "elapsed_seconds": time.perf_counter() - started,
            "final_test_accessed": False,
        }
        _write_run_outputs(output_dir, payload, experiment)
        return payload

    result: dict[str, Any] = {"numerical_admissibility": admissibility}
    if stage in {"risk", "benchmark", "plots"}:
        risk = experiment.scientific_risk(output_dir=output_dir)
        _write_csv(output_dir / "result.layout_risk.csv", risk["rows"])
        time_rows = []
        for local, design in enumerate(risk["design_indices"]):
            for position, day in enumerate(risk["evaluation_days"]):
                time_rows.append({
                    "design_index": int(design),
                    "design_id": experiment.sensor_bank.design_ids[design],
                    "source_time_index": int(risk["evaluation_indices"][position]),
                    "day": float(day),
                    "risk": float(risk["risk_by_time"][local, position]),
                })
        _write_csv(output_dir / "result.risk_by_time.csv", time_rows)
        result["risk"] = risk["summary"]
        if stage == "plots":
            try:
                from .results import generate_law_results
            except ImportError:  # direct script invocation
                from results import generate_law_results

            result["plots"] = generate_law_results(
                experiment, risk, experiment._resolve("experiments/ocean_drifters/analysis")
            )
    if stage == "tangent_action":
        try:
            from .action import run_action_readiness
        except ImportError:  # direct script invocation
            from action import run_action_readiness
        result[stage] = run_action_readiness(
            experiment,
            experiment._resolve("experiments/ocean_drifters/analysis"),
            output_dir,
        )
    if stage == "full_action":
        try:
            from .full_action import run_weighted_poisson_pilot
        except ImportError:  # direct script invocation
            from full_action import run_weighted_poisson_pilot
        result[stage] = run_weighted_poisson_pilot(
            experiment,
            experiment._resolve("experiments/ocean_drifters/analysis"),
            output_dir,
        )
    if stage == "full_action_production":
        try:
            from .full_action_production import run_full_action_production
        except ImportError:  # direct script invocation
            from full_action_production import run_full_action_production
        result[stage] = run_full_action_production(
            experiment,
            experiment._resolve("experiments/ocean_drifters/analysis"),
            output_dir,
        )
    if stage == "solver_repair":
        try:
            from .solver_repair import run_poisson_solver_repair
        except ImportError:  # direct script invocation
            from solver_repair import run_poisson_solver_repair
        result[stage] = run_poisson_solver_repair(
            experiment,
            experiment._resolve("experiments/ocean_drifters/analysis"),
            output_dir,
        )
    if stage == "post_dispersion_action":
        result[stage] = experiment.post_dispersion_action().result_payload()
    if stage not in {"projection", "risk", "benchmark", "plots", "tangent_action", "full_action", "full_action_production", "solver_repair", "post_dispersion_action"}:
        raise ValueError(f"unknown stage {stage!r}")
    payload = {
        "schema_version": 1,
        "experiment": cfg["name"],
        "smoke": False,
        "stage": stage,
        **result,
        "statuses": {
            "projection_valid": admissibility["excluded_layout_count"] == 0,
            "law_risk_valid": "risk" in result or stage in {"plots", "tangent_action", "full_action", "full_action_production", "solver_repair", "post_dispersion_action"},
            "multiplier_dynamics_valid": bool(
                (
                    stage == "tangent_action"
                    and result[stage]["multiplier_dynamics_valid_count"]
                    / max(result[stage]["frozen_layout_count"], 1)
                    >= float(cfg["action"]["soft_moment_projection"][
                        "minimum_valid_layout_fraction"
                    ])
                )
                or (
                    stage == "full_action"
                    and result[stage]["authorized_action_ready_layout_count"]
                    / max(len(json.loads(
                        experiment.paths["risk_freeze"].read_text(encoding="utf-8")
                    )["near_optimal_design_ids"]), 1)
                    >= float(cfg["action"]["soft_moment_projection"][
                        "minimum_valid_layout_fraction"
                    ])
                )
                or stage == "full_action_production"
                or (
                    stage == "post_dispersion_action"
                    and result[stage]["all_layouts_valid"]
                )
            ),
            "tangent_action_valid": bool(
                (
                    stage == "tangent_action"
                    and result[stage]["tangent_action_valid_count"]
                    / max(result[stage]["frozen_layout_count"], 1)
                    >= float(cfg["action"]["soft_moment_projection"][
                        "minimum_valid_layout_fraction"
                    ])
                )
                or (
                    stage == "full_action"
                    and result[stage]["authorized_action_ready_layout_count"]
                    / max(len(json.loads(
                        experiment.paths["risk_freeze"].read_text(encoding="utf-8")
                    )["near_optimal_design_ids"]), 1)
                    >= float(cfg["action"]["soft_moment_projection"][
                        "minimum_valid_layout_fraction"
                    ])
                )
                or stage == "full_action_production"
                or (
                    stage == "post_dispersion_action"
                    and result[stage]["all_layouts_valid"]
                )
            ),
            "full_action_valid": bool(
                (
                    stage == "full_action_production"
                    and result[stage]["full_action_valid"]
                )
                or (
                    stage == "post_dispersion_action"
                    and result[stage]["full_action_production_valid"]
                )
            ),
            "post_dispersion_action_valid": bool(
                stage == "post_dispersion_action"
                and result[stage]["all_layouts_valid"]
            ),
            "full_horizon_action_valid": bool(
                stage == "full_action_production"
                and result[stage]["full_action_valid"]
            ),
        },
        "provenance": experiment.provenance(),
        "elapsed_seconds": time.perf_counter() - started,
        "final_test_accessed": False,
    }
    _write_run_outputs(output_dir, payload, experiment)
    return payload
