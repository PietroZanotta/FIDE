"""Predeclared, one-shot final evaluation for the ocean-drifter experiment.

The dry-run path uses only the frozen development artifact.  The locked cohort
path is resolved, hashed, and opened only after both authorization keys have
been changed explicitly in ``config.json``.
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from mfsi.cache import file_sha256, fingerprint

try:
    from .experiment import FrozenArtifactError, REPO_ROOT, _rff_map
except ImportError:  # direct ``python experiments/ocean_drifters/run.py`` invocation
    from experiment import FrozenArtifactError, REPO_ROOT, _rff_map  # type: ignore

if TYPE_CHECKING:
    from .experiment import OceanDriftersExperiment


AUTHORIZATION_PENDING = "awaiting_explicit_user_authorization"
AUTHORIZATION_RECORDED = "explicit_user_authorization_recorded"
ACCEPTANCE_RULE = "final_rff_mmd2_le_validation_bootstrap_one_sided_95pct_upper"
FAILURE_POLICY = "report_without_retuning_or_reopening_selection"


def protocol_config_hash(cfg: dict[str, Any]) -> str:
    """Hash the frozen protocol while normalizing only its authorization keys."""
    normalized = copy.deepcopy(cfg)
    normalized["scientific"]["final_test_access_allowed"] = False
    normalized["final_evaluation"]["authorization_status"] = AUTHORIZATION_PENDING
    # This field anchors the release manifest, which itself contains the protocol
    # hash. Excluding only the anchor avoids a circular hash dependency.
    normalized["final_evaluation"].pop("release_manifest_expected_sha256", None)
    return fingerprint(normalized)


def _load_risk_contract(experiment: "OceanDriftersExperiment") -> dict[str, Any]:
    final_cfg = experiment.cfg["final_evaluation"]
    design_index = int(final_cfg["selected_design_index"])
    with np.load(experiment.paths["risk_projection_embeddings"], allow_pickle=False) as data:
        if bool(data["final_test_accessed"]):
            raise FrozenArtifactError("frozen risk inputs report final-test access")
        design_ids = np.asarray(data["design_id"]).astype(str)
        evaluation_indices = np.asarray(data["evaluation_indices"], dtype=int)
        evaluation_days = np.asarray(data["evaluation_days"], dtype=np.float64)
        projected = np.asarray(data["projected_rff_embedding"][design_index], dtype=np.float64)
        validation_risk = float(data["risks"][design_index])
        bootstrap = np.asarray(data["bootstrap_risk"][design_index], dtype=np.float64)
        omega = np.asarray(data["rff_omega"], dtype=np.float64)
        phase = np.asarray(data["rff_phase"], dtype=np.float64)
        bandwidth = float(data["bandwidth_km"])
    if design_ids[design_index] != final_cfg["selected_design_id"]:
        raise FrozenArtifactError("the predeclared final design no longer matches the risk bank")
    if len(evaluation_indices) != int(final_cfg["evaluation_time_count"]):
        raise FrozenArtifactError("the predeclared final time grid changed")
    if projected.shape != (len(evaluation_indices), int(experiment.cfg["law"]["rff_features"])):
        raise FrozenArtifactError("the predeclared projected RFF embedding changed shape")
    if not np.isclose(
        bandwidth, float(experiment.cfg["law"]["bandwidth_km"]), rtol=0.0, atol=1e-12
    ):
        raise FrozenArtifactError("the predeclared final RFF bandwidth changed")
    return {
        "design_index": design_index,
        "design_id": str(design_ids[design_index]),
        "evaluation_indices": evaluation_indices,
        "evaluation_days": evaluation_days,
        "projected": projected,
        "validation_risk": validation_risk,
        "bootstrap": bootstrap,
        "omega": omega,
        "phase": phase,
        "bandwidth_km": bandwidth,
    }


def _score_positions(positions: np.ndarray, contract: dict[str, Any]) -> tuple[float, np.ndarray]:
    sample = np.asarray(positions, dtype=np.float64)
    expected_times = int(np.max(contract["evaluation_indices"])) + 1
    if sample.ndim != 3 or sample.shape[2] != 2 or sample.shape[1] < expected_times:
        raise FrozenArtifactError("final-evaluation trajectories have an unexpected shape")
    if len(sample) == 0 or not np.isfinite(sample).all():
        raise FrozenArtifactError("final-evaluation trajectories must be nonempty and finite")
    selected = sample[:, contract["evaluation_indices"]]
    features = _rff_map(selected, contract["omega"], contract["phase"])
    embedding = features.mean(axis=0, dtype=np.float64)
    difference = contract["projected"] - embedding
    risk_by_time = np.sum(difference * difference, axis=-1)
    return float(np.mean(risk_by_time)), np.asarray(risk_by_time, dtype=np.float64)


def validation_gate(experiment: "OceanDriftersExperiment") -> dict[str, Any]:
    """Derive the final threshold solely from the frozen validation bootstrap."""
    contract = _load_risk_contract(experiment)
    final_cfg = experiment.cfg["final_evaluation"]
    quantile = float(final_cfg["validation_bootstrap_quantile"])
    upper = float(np.quantile(contract["bootstrap"], quantile))
    configured_upper = float(final_cfg["validation_bootstrap_upper_bound"])
    if not np.isclose(upper, configured_upper, rtol=0.0, atol=2e-15):
        raise FrozenArtifactError(
            f"validation-only final gate changed: {upper:.17g} != {configured_upper:.17g}"
        )
    return {
        "schema_version": 1,
        "experiment": experiment.cfg["name"],
        "status": "frozen_before_final_test_access",
        "selected_design_index": contract["design_index"],
        "selected_design_id": contract["design_id"],
        "metric": experiment.cfg["law"]["metric"],
        "rff_features": int(experiment.cfg["law"]["rff_features"]),
        "bandwidth_km": contract["bandwidth_km"],
        "evaluation_time_count": len(contract["evaluation_indices"]),
        "evaluation_indices": contract["evaluation_indices"].tolist(),
        "evaluation_days": contract["evaluation_days"].tolist(),
        "validation_risk": contract["validation_risk"],
        "validation_bootstrap_replicates": len(contract["bootstrap"]),
        "validation_bootstrap_quantile": quantile,
        "validation_bootstrap_upper_bound": upper,
        "acceptance_rule": ACCEPTANCE_RULE,
        "failure_policy": FAILURE_POLICY,
        "derived_only_from_development_validation_ids": True,
        "final_test_artifact_opened": False,
        "final_test_accessed": False,
    }


def dry_run_final_evaluation(experiment: "OceanDriftersExperiment") -> dict[str, Any]:
    """Exercise the final scorer on validation positions without opening locked data."""
    contract = _load_risk_contract(experiment)
    gate = validation_gate(experiment)
    reproduced, risk_by_time = _score_positions(experiment.cohort.validation, contract)
    difference = abs(reproduced - contract["validation_risk"])
    if difference > 2e-12:
        raise FrozenArtifactError(
            f"final scorer does not reproduce frozen validation risk ({difference:.3e})"
        )
    cohort_manifest = json.loads(
        experiment.paths["cohort_manifest"].read_text(encoding="utf-8")
    )
    final_cfg = experiment.cfg["final_evaluation"]
    if cohort_manifest["cohort_sha256"] != final_cfg["locked_cohort_expected_sha256"]:
        raise FrozenArtifactError("declared locked-cohort hash disagrees with its safe manifest")
    if int(cohort_manifest["sizes"]["final_test"]) != int(final_cfg["final_test_id_count"]):
        raise FrozenArtifactError("declared final-test count disagrees with its safe manifest")
    return {
        "schema_version": 1,
        "experiment": experiment.cfg["name"],
        "stage": "final_evaluation_dry_run",
        "status": "ready_for_explicit_one_shot_authorization",
        "protocol_config_hash": protocol_config_hash(experiment.cfg),
        "selected_design_index": contract["design_index"],
        "selected_design_id": contract["design_id"],
        "validation_id_count": int(len(experiment.cohort.validation)),
        "evaluation_time_count": len(contract["evaluation_indices"]),
        "reproduced_validation_risk": reproduced,
        "frozen_validation_risk": contract["validation_risk"],
        "absolute_reproduction_error": difference,
        "risk_by_time": risk_by_time.tolist(),
        "acceptance_rule": gate["acceptance_rule"],
        "acceptance_upper_bound": gate["validation_bootstrap_upper_bound"],
        "gate_derived_only_from_development_validation_ids": True,
        "failure_policy": FAILURE_POLICY,
        "locked_cohort_declared_path": final_cfg["locked_cohort_path"],
        "locked_cohort_expected_sha256_from_safe_manifest": cohort_manifest["cohort_sha256"],
        "locked_final_test_id_count_from_safe_manifest": int(
            cohort_manifest["sizes"]["final_test"]
        ),
        "locked_cohort_path_resolved": False,
        "locked_cohort_hashed": False,
        "locked_cohort_opened": False,
        "split_manifest_opened": False,
        "final_test_accessed": False,
    }


def _validate_release_manifest(experiment: "OceanDriftersExperiment") -> dict[str, Any]:
    manifest_path = experiment._resolve(experiment.cfg["final_evaluation"]["release_manifest_path"])
    if not manifest_path.is_file():
        raise FrozenArtifactError("the pre-final release manifest is missing")
    expected_hash = experiment.cfg["final_evaluation"][
        "release_manifest_expected_sha256"
    ]
    if file_sha256(manifest_path) != expected_hash:
        raise FrozenArtifactError("the pre-final release manifest hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_config_hash") != protocol_config_hash(experiment.cfg):
        raise FrozenArtifactError("the frozen pre-final protocol config changed")
    for item in manifest.get("frozen_files", []):
        path = REPO_ROOT / item["path"]
        if not path.is_file() or file_sha256(path) != item["sha256"]:
            raise FrozenArtifactError(f"pre-final frozen file changed: {item['path']}")
    return manifest


def run_locked_final_evaluation(
    experiment: "OceanDriftersExperiment", output_dir: Path
) -> dict[str, Any]:
    """Run the frozen final test once, only after explicit two-key authorization."""
    final_cfg = experiment.cfg["final_evaluation"]
    if (
        experiment.cfg["scientific"].get("final_test_access_allowed") is not True
        or final_cfg.get("authorization_status") != AUTHORIZATION_RECORDED
    ):
        raise PermissionError(
            "final-test trajectories are locked; explicit one-shot authorization is required"
        )
    output_dir = Path(output_dir)
    canonical_output_dir = experiment._resolve(final_cfg["canonical_output_dir"])
    if output_dir.resolve() != canonical_output_dir.resolve():
        raise PermissionError(
            "one-shot final evaluation requires the frozen canonical output directory"
        )
    if (output_dir / "result.json").exists():
        raise PermissionError("one-shot final evaluation refused: result.json already exists")
    release = _validate_release_manifest(experiment)
    if release.get("final_test_accessed") is not False:
        raise FrozenArtifactError("pre-final release manifest reports final-test access")

    # No locked-data path is touched above this line.
    cohort_path = experiment._resolve(final_cfg["locked_cohort_path"])
    split_path = experiment._resolve(final_cfg["split_manifest_path"])
    expected_hash = final_cfg["locked_cohort_expected_sha256"]
    if file_sha256(cohort_path) != expected_hash:
        raise FrozenArtifactError("locked cohort hash changed")
    cohort_manifest = json.loads(
        experiment.paths["cohort_manifest"].read_text(encoding="utf-8")
    )
    if file_sha256(split_path) != cohort_manifest["split_manifest_sha256"]:
        raise FrozenArtifactError("locked split manifest hash changed")
    with split_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    final_ids = {row["drifter_id"] for row in rows if row["split"] == "final_test"}
    if len(final_ids) != int(final_cfg["final_test_id_count"]):
        raise FrozenArtifactError("locked split does not contain the predeclared final-test count")
    with np.load(cohort_path, allow_pickle=False) as data:
        all_positions = np.asarray(data["X"], dtype=np.float64)
        all_ids = np.asarray(data["ids"]).astype(str)
    mask = np.asarray([identifier in final_ids for identifier in all_ids], dtype=bool)
    if int(mask.sum()) != len(final_ids) or set(all_ids[mask]) != final_ids:
        raise FrozenArtifactError("locked cohort and final-test split IDs disagree")
    if final_ids & set(experiment.cohort.ids.astype(str)):
        raise FrozenArtifactError("final-test IDs overlap the development cohort")

    contract = _load_risk_contract(experiment)
    risk, risk_by_time = _score_positions(all_positions[mask], contract)
    upper = float(final_cfg["validation_bootstrap_upper_bound"])
    passed = bool(np.isfinite(risk) and risk <= upper)
    return {
        "schema_version": 1,
        "experiment": experiment.cfg["name"],
        "stage": "final_evaluation",
        "status": "pass" if passed else "fail_report_only",
        "selected_design_index": contract["design_index"],
        "selected_design_id": contract["design_id"],
        "final_test_id_count": int(mask.sum()),
        "evaluation_time_count": len(contract["evaluation_indices"]),
        "final_rff_mmd2": risk,
        "risk_by_time": risk_by_time.tolist(),
        "acceptance_rule": ACCEPTANCE_RULE,
        "acceptance_upper_bound": upper,
        "accepted": passed,
        "failure_policy": FAILURE_POLICY,
        "retuning_allowed": False,
        "selection_reopened": False,
        "locked_cohort_sha256": expected_hash,
        "protocol_config_hash": protocol_config_hash(experiment.cfg),
        "final_test_accessed": True,
    }
