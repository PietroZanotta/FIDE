"""Frozen protocol and safety helpers for the official Galerkin Pareto sweep."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from mfsi.cache import fingerprint

from .production_artifacts import PRODUCTION_ROOT, file_sha256


PACKAGE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PACKAGE_ROOT / "outputs" / "official_galerkin_pareto"
PROTOCOL_PATH = OUTPUT_ROOT / "protocol.json"
PROTOCOL_HASH_PATH = OUTPUT_ROOT / "protocol_hash.txt"
PROTOCOL_DOCUMENT = PACKAGE_ROOT / "OFFICIAL_GALERKIN_PARETO_PROTOCOL.md"
REPORT_PATH = PACKAGE_ROOT / "OFFICIAL_GALERKIN_PARETO_EVALUATION.md"
ARTIFACT_DIR = PRODUCTION_ROOT / "artifacts"
DICTIONARY_PATH = (
    PACKAGE_ROOT / "outputs" / "galerkin_only_3pct" / "cache"
    / "dictionaries" / "dictionary_K280.npz"
)
TRAIN_CACHE = (
    PACKAGE_ROOT / "outputs" / "galerkin_only_3pct" / "cache" / "K280"
)
HISTORICAL_PARETO = (
    PACKAGE_ROOT.parent / "skyrmions_deep_ritz" / "outputs"
    / "pareto_authoritative" / "pareto.json"
)

ALLOWANCES = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
OFFICIAL_K = 280
PROTOCOL_VERSION = "skyrmion_official_galerkin_pareto_v1"
EXPECTED_DICTIONARY_SHA256 = (
    "37e9b60fcb92c4e5a0ee7ec1651fb7f8889f7ac6bdb02d3bd314e9ef40833326"
)
OLD_VALIDATION_FILES = (
    ARTIFACT_DIR / "truth_banks.npz",
    ARTIFACT_DIR / "reference_bank_validation_fit.npz",
    ARTIFACT_DIR / "reference_bank_validation_audit.npz",
    PACKAGE_ROOT / "outputs" / "galerkin_only_3pct" / "validation" / "result.json",
)

INITIAL_GIT_STATUS = """ M experiments/skyrmions_deep_ritz_full/README.md
 M experiments/skyrmions_deep_ritz_full/deep_ritz.py
 M experiments/skyrmions_deep_ritz_full/run.py
 M experiments/skyrmions_deep_ritz_full/workflow.py
?? experiments/skyrmions_deep_ritz_full/AUTHORITATIVE_GPU_ACCELERATION.md
?? experiments/skyrmions_deep_ritz_full/AUTHORITATIVE_STABILITY_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/FAST_PRODUCTION_3PCT_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/FINAL_3PCT_GALERKIN_CROSSCHECK.md
?? experiments/skyrmions_deep_ritz_full/GALERKIN_ONLY_3PCT_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/authoritative_platform.py
?? experiments/skyrmions_deep_ritz_full/authoritative_stability.py
?? experiments/skyrmions_deep_ritz_full/final_crosscheck.py
?? experiments/skyrmions_deep_ritz_full/final_crosscheck_run.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only_data.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only_run.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only_workflow.py
?? experiments/skyrmions_deep_ritz_full/test_fast_production.py
?? experiments/skyrmions_deep_ritz_full/test_final_crosscheck.py
?? experiments/skyrmions_deep_ritz_full/test_galerkin_only.py"""


def require_official_output_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"official Pareto output must be beneath {root}, got {resolved}")
    return resolved


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def write_json(path: Path, payload: dict[str, Any], *, overwrite: bool = True) -> None:
    path = require_official_output_path(path)
    if path.exists() and not overwrite:
        raise RuntimeError(f"refusing to overwrite existing official output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def derived_seed(global_seed: int, label: str) -> dict[str, Any]:
    text = f"{int(global_seed)}:skyrmion:official_galerkin_pareto:v1:{label}"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    # Positive signed-32-bit seeds work identically at every JAX boundary.
    seed = int(digest[:16], 16) % (2**31 - 1)
    return {"label": label, "derivation_text": text, "sha256": digest, "seed": seed}


def selection_ceiling(law_risk: float, allowance_percent: float) -> float:
    return float(law_risk) * (1.0 + float(allowance_percent) / 100.0)


def validation_ceiling(law_risk: float, allowance_percent: float) -> float:
    return float(law_risk) * (1.0 + float(allowance_percent) / 100.0 + 0.05)


def strict_validation_ceiling(law_risk: float, allowance_percent: float) -> float:
    return float(law_risk) * (1.0 + float(allowance_percent) / 100.0)


def risk_feasible(risk: float, law_risk: float, allowance_percent: float) -> bool:
    return float(risk) <= selection_ceiling(law_risk, allowance_percent)


def retain_incumbent(candidate_action: float, incumbent_action: float, tolerance: float) -> bool:
    return float(candidate_action) >= float(incumbent_action) - float(tolerance)


def actions_nonincreasing(actions: Iterable[float], tolerance: float) -> bool:
    values = [float(value) for value in actions]
    return all(right <= left + float(tolerance) for left, right in zip(values[:-1], values[1:]))


def common_solver_reduction(law_action: float, full_action: float) -> float:
    if float(law_action) <= 0.0:
        raise ValueError("Law action must be positive")
    return (float(law_action) - float(full_action)) / float(law_action)


def validation_classification(*, numerical_valid: bool, declared_risk_pass: bool) -> str:
    if not numerical_valid:
        return "VALIDATION NUMERICAL FAILURE"
    if not declared_risk_pass:
        return "VALIDATION RISK REVERSAL"
    return "PASS"


def allowance_slug(value: float) -> str:
    return str(float(value)).replace(".", "p").removesuffix("p0")


def _hash_inventory(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        path = Path(path).resolve()
        if not path.is_file():
            raise RuntimeError(f"required frozen artifact is missing: {path}")
        rows.append({
            "path": str(path.relative_to(PACKAGE_ROOT.parent.parent)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    return rows


def decision_code_hash() -> str:
    names = (
        "config.json",
        "galerkin.py",
        "galerkin_only.py",
        "galerkin_only_data.py",
        "production_basis.py",
        "production_galerkin.py",
        "official_pareto_common.py",
        "official_pareto_selection.py",
        "official_pareto_validation.py",
        "official_pareto_report.py",
        "official_pareto_run.py",
    )
    return payload_sha256({name: file_sha256(PACKAGE_ROOT / name) for name in names})


def protocol_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    if file_sha256(DICTIONARY_PATH) != EXPECTED_DICTIONARY_SHA256:
        raise RuntimeError("official K=280 dictionary hash differs from the validated hash")
    validation_labels = (
        "truth", "reference_fit", "reference_audit", "measurement_noise",
    )
    start_labels = ("global_starts", "local_starts", "finalist_audit_directions")
    frozen_artifacts = (
        ARTIFACT_DIR / "isolated_artifact_manifest.json",
        ARTIFACT_DIR / "reference.npz",
        ARTIFACT_DIR / "truth_banks.npz",
        ARTIFACT_DIR / "reference_bank_projection.npz",
        ARTIFACT_DIR / "reference_bank_ritz_train.npz",
        ARTIFACT_DIR / "reference_bank_ritz_audit.npz",
        DICTIONARY_PATH,
        HISTORICAL_PARETO,
    )
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "methodological_decision": (
            "fixed-feature K=280 Galerkin finite-dimensional approximation "
            "is the official skyrmion Full discretization"
        ),
        "initial_git_status_short": INITIAL_GIT_STATUS.splitlines(),
        "allowances_percent": list(ALLOWANCES),
        "selection_risk_rule": "R_sel <= (1 + p/100) * R_Law_sel",
        "validation_risk_rule": "R_val <= (1 + p/100 + 0.05) * R_Law_val",
        "strict_validation_rule_reported_only": "R_val <= (1 + p/100) * R_Law_val",
        "validation_relative_slack": 0.05,
        "full_discretization": {
            "basis_size": OFFICIAL_K,
            "dictionary_sha256": EXPECTED_DICTIONARY_SHA256,
            "dictionary_path": str(DICTIONARY_PATH.relative_to(PACKAGE_ROOT)),
            "dictionary_ordering": "unchanged validated hybrid Fourier/pairwise ordering",
            "normalization": "unchanged eta-independent selection-train normalization",
            "rank_tolerance": float(cfg["production_galerkin"]["relative_rank_tolerance"]),
            "algebra_thresholds": {
                key: cfg["production_galerkin"][key]
                for key in (
                    "maximum_range_residual", "maximum_stationarity_residual",
                    "maximum_identity_relerr", "maximum_symmetry_residual",
                    "minimum_rank_fraction", "maximum_retained_condition",
                )
            },
            "certificate_thresholds": cfg["production_galerkin"]["certificate_thresholds"],
            "chunk_size": int(cfg["production_galerkin"]["chunk_size"]),
        },
        "optimizer": {
            "algorithm": "periodic projected-gradient bounded trust trajectory with exact backtracking gates",
            "trust_radius": 2.0e-4,
            "initial_step": 5.0e-5,
            "maximum_accepted_step_attempts": 8,
            "maximum_backtracks_per_step": 10,
            "backtrack_factor": 0.5,
            "successful_step_cap": 7.5e-5,
            "periodic_full_certificate_every": 4,
            "replacement_tolerance": 1.0e-10,
            "risk_penalty_weight": 100.0,
            "smooth_separation_penalty_weight": 100.0,
            "penalty_role": (
                "frozen diagnostic/globalization settings; exact risk and exact "
                "minimum-separation gates remain authoritative"
            ),
            "rank_must_equal_previous_step": True,
            "authoritative_objective": "K=280 selection-train Galerkin action",
        },
        "start_generation": {
            "maximum_starts_per_allowance_including_incumbent": 8,
            "mandatory_preceding_incumbent": True,
            "include_law": True,
            "historical_source": str(HISTORICAL_PARETO.relative_to(PACKAGE_ROOT.parent.parent)),
            "historical_rule": "include every exact-selection-feasible frozen historical Pareto geometry, independent of old validation",
            "local_count_per_historical_center": 2,
            "local_scale": 2.0e-4,
            "global_candidate_count": 48,
            "global_oversample": 16,
            "global_selected_count_per_allowance": 2,
            "ranking": "exact selection feasibility then K=280 selection-train action, stable id tie-break",
            "deduplication_euclidean_tolerance": 1.0e-12,
            "seeds": [derived_seed(cfg["seed"], label) for label in start_labels],
        },
        "finalist_gradient_audit": {
            "directions_per_unique_winner": 1,
            "epsilon_window": [3.0e-4, 1.0e-4],
            "maximum_relative_error": 0.02,
            "rank_stability_required": True,
            "forcing_projection_required": True,
        },
        "fresh_validation": {
            "truth_samples": int(cfg["banks"]["truth_validation_samples"]),
            "reference_fit_samples": int(cfg["banks"]["validation_fit_samples"]),
            "reference_audit_samples": int(cfg["banks"]["validation_audit_samples"]),
            "truth_substeps": int(cfg["physics"]["truth_substeps"]),
            "reference_substeps": int(cfg["banks"]["reference_substeps"]),
            "reference_checkpoint_retrained": False,
            "initial_distribution": "fresh independent samples from the frozen truth initial distribution",
            "seeds": [derived_seed(cfg["seed"], label) for label in validation_labels],
        },
        "random_seed_global": int(cfg["seed"]),
        "law_eta": cfg["envelope"]["law_eta"],
        "eta0": cfg["envelope"]["eta0"],
        "geometry": {
            "box": cfg["physics"]["box"],
            "sensor_count": int(cfg["measurement"]["n_sensors"]),
            "minimum_separation": float(cfg["measurement"]["min_separation"]),
            "periodic": True,
        },
        "decision_path": {
            "deep_ritz_allowed": False,
            "validation_allowed_during_selection": False,
            "old_validation_role": "development data; historical comparison only after fresh validation",
            "selection_winners_immutable_after_validation": True,
        },
        "frozen_artifacts": _hash_inventory(frozen_artifacts),
        "old_validation_hashes_recorded_without_array_access": _hash_inventory(OLD_VALIDATION_FILES),
        "decision_code_hash": decision_code_hash(),
        "reporting": {
            "action_comparison": "common K=280 Galerkin solver only",
            "validation_actual_ratio_always_reported": True,
            "strict_p_percent_is_transparency_only": True,
            "no_posthoc_significance_test": True,
            "no_pseudo_replicates": True,
            "classification_labels": [
                "PASS", "VALIDATION RISK REVERSAL", "VALIDATION NUMERICAL FAILURE",
            ],
        },
    }


def freeze_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    payload = protocol_payload(cfg)
    digest = payload_sha256(payload)
    wrapped = {**payload, "protocol_sha256": digest, "protocol_frozen": True}
    if PROTOCOL_PATH.exists():
        previous = read_json(PROTOCOL_PATH)
        if previous != wrapped:
            raise RuntimeError("official protocol already exists with different content")
    else:
        write_json(PROTOCOL_PATH, wrapped, overwrite=False)
        PROTOCOL_HASH_PATH.write_text(digest + "\n", encoding="utf-8")
    if PROTOCOL_HASH_PATH.read_text(encoding="utf-8").strip() != digest:
        raise RuntimeError("official protocol hash sidecar mismatch")
    return wrapped


def require_frozen_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    if not PROTOCOL_PATH.is_file() or not PROTOCOL_HASH_PATH.is_file():
        raise RuntimeError("freeze-protocol must run before this phase")
    frozen = read_json(PROTOCOL_PATH)
    digest = frozen.pop("protocol_sha256", None)
    is_frozen = frozen.pop("protocol_frozen", None)
    expected = payload_sha256(frozen)
    if not is_frozen or digest != expected:
        raise RuntimeError("frozen protocol content/hash mismatch")
    if PROTOCOL_HASH_PATH.read_text(encoding="utf-8").strip() != expected:
        raise RuntimeError("frozen protocol sidecar mismatch")
    current = protocol_payload(cfg)
    if current != frozen:
        raise RuntimeError("current code/config/artifact state differs from frozen protocol")
    return {**frozen, "protocol_sha256": expected, "protocol_frozen": True}


def file_set_hash(paths: Iterable[Path]) -> str:
    return payload_sha256({str(Path(path).resolve()): file_sha256(Path(path)) for path in paths})


__all__ = [
    "ALLOWANCES", "ARTIFACT_DIR", "DICTIONARY_PATH", "EXPECTED_DICTIONARY_SHA256",
    "HISTORICAL_PARETO", "OFFICIAL_K", "OUTPUT_ROOT", "PROTOCOL_DOCUMENT",
    "PROTOCOL_HASH_PATH", "PROTOCOL_PATH", "REPORT_PATH", "TRAIN_CACHE",
    "allowance_slug", "canonical_json", "derived_seed", "file_set_hash",
    "freeze_protocol", "payload_sha256", "read_json", "require_frozen_protocol",
    "require_official_output_path", "selection_ceiling", "strict_validation_ceiling",
    "risk_feasible", "retain_incumbent", "actions_nonincreasing",
    "common_solver_reduction", "validation_classification",
    "validation_ceiling", "write_json",
]
