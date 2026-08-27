"""Frozen constants, seals, and atomic I/O for official skyrmion Pareto v2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from mfsi.cache import fingerprint

from .production_artifacts import PRODUCTION_ROOT, file_sha256


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs" / "official_galerkin_pareto_v2"
PROTOCOL_PATH = OUTPUT_ROOT / "protocol.json"
PROTOCOL_HASH_PATH = OUTPUT_ROOT / "protocol_hash.txt"
PROTOCOL_DOCUMENT = ROOT / "OFFICIAL_GALERKIN_PARETO_V2_PROTOCOL.md"
REPORT_PATH = ROOT / "OFFICIAL_GALERKIN_PARETO_V2_EVALUATION.md"
ARTIFACT_DIR = PRODUCTION_ROOT / "artifacts"
DICTIONARY_PATH = ROOT / "outputs" / "galerkin_only_3pct" / "cache" / "dictionaries" / "dictionary_K280.npz"
EXPECTED_DICTIONARY_SHA256 = "37e9b60fcb92c4e5a0ee7ec1651fb7f8889f7ac6bdb02d3bd314e9ef40833326"
VERSION = "skyrmion_official_galerkin_pareto_v2_rerun3"
FAILED_PRESTART_ROOT = ROOT / "outputs" / "official_galerkin_pareto_v2_failed_prestart_08dd28ef"
SUPERSEDED_ACCEL_ROOT = ROOT / "outputs" / "official_galerkin_pareto_v2_superseded_galerkin_tesseract_3418fd58"
SUPERSEDED_NATIVE_SEPARATION_ROOT = (
    ROOT
    / "outputs"
    / "official_galerkin_pareto_v2_superseded_native_separation_0890c254"
)
ALLOWANCES = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
K = 280
RANK_TOLERANCE = 1.0e-12
MINIMUM_RESS = 0.05
MAXIMUM_ENERGY_RESIDUAL = 0.08
BANK_SIZES = {
    "screen": 8192,
    "search_train": 32768,
    "periodic_audit": 16384,
    "authoritative_train": 65536,
    "authoritative_audit": 65536,
}
VALIDATION_SIZES = {"truth": 5000, "reference_fit": 16384, "reference_audit": 16384}
HISTORICAL_REPORTS = (
    "FINAL_3PCT_GALERKIN_CROSSCHECK.md",
    "GALERKIN_ONLY_3PCT_EVALUATION.md",
    "OFFICIAL_GALERKIN_PARETO_PROTOCOL.md",
    "OFFICIAL_GALERKIN_PARETO_EVALUATION.md",
    "GALERKIN_RESOLUTION_STUDY.md",
    "GALERKIN_K280_QUADRATURE_QUALIFICATION.md",
    "ESS_QUALIFICATION_AND_PERFORMANCE.md",
    "ESS_QUALIFICATION_PROTOCOL.md",
)


def canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical(payload)).hexdigest()


def output_path(path: Path) -> Path:
    resolved, root = Path(path).resolve(), OUTPUT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Pareto-v2 output must be beneath {root}: {resolved}")
    return resolved


def atomic_json(path: Path, payload: Any, *, immutable: bool = False) -> None:
    path = output_path(path)
    if immutable and path.exists():
        raise RuntimeError(f"refusing to overwrite immutable Pareto-v2 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_text(path: Path, text: str, *, immutable: bool = False) -> None:
    path = output_path(path)
    if immutable and path.exists():
        raise RuntimeError(f"refusing to overwrite immutable Pareto-v2 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def derive_seed(global_seed: int, scope: str, label: str) -> dict[str, Any]:
    text = f"{int(global_seed)}:skyrmion:official_pareto_v2:{scope}:{label}"
    digest = hashlib.sha256(text.encode()).hexdigest()
    return {"label": label, "derivation_text": text, "sha256": digest,
            "seed": int(digest[:16], 16) % (2**31 - 1)}


def selection_ceiling(law_risk: float, allowance: float) -> float:
    return float(law_risk) * (1.0 + float(allowance) / 100.0)


def validation_ceiling(law_risk: float, allowance: float) -> float:
    return float(law_risk) * (1.0 + float(allowance) / 100.0 + 0.05)


def slug(value: float) -> str:
    return str(float(value)).replace(".", "p").removesuffix("p0")


def eta_key(eta: Any) -> str:
    return payload_sha256([float(value) for value in eta])[:20]


def signature(protocol: dict[str, Any], kind: str, extra: Any = None) -> str:
    return fingerprint({
        "kind": kind, "protocol_sha256": protocol["protocol_sha256"],
        "dictionary_sha256": file_sha256(DICTIONARY_PATH), "K": K,
        "dtype": "float64", "extra": extra,
    })


def protocol_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    if file_sha256(DICTIONARY_PATH) != EXPECTED_DICTIONARY_SHA256:
        raise RuntimeError("fixed K=280 dictionary hash changed")
    if float(cfg["production_galerkin"]["relative_rank_tolerance"]) != RANK_TOLERANCE:
        raise RuntimeError("rank tolerance changed")
    if float(cfg["forcing"]["minimum_ess_fraction"]) != MINIMUM_RESS:
        raise RuntimeError("rESS threshold changed")
    certificates = cfg["production_galerkin"]["certificate_thresholds"]
    if float(certificates["maximum_energy_residual"]) != MAXIMUM_ENERGY_RESIDUAL:
        raise RuntimeError("energy threshold changed")
    galerkin_backend = str(cfg["production_galerkin"].get("assembly_backend", "jax"))
    if galerkin_backend not in {"jax", "tesseract_cpp"}:
        raise RuntimeError("invalid production Galerkin assembly backend")
    selection_seeds = [derive_seed(cfg["seed"], "selection", label)
                       for label in (*BANK_SIZES, "candidate_pool")]
    validation_seeds = [derive_seed(cfg["seed"], "validation", label) for label in (
        "truth", "reference_fit", "reference_audit", "measurement_noise")]
    source_names = ("pareto_v2_common.py", "pareto_v2_selection.py", "pareto_v2_validation.py",
                    "pareto_v2_report.py", "pareto_v2_run.py", "config.json")
    return {
        "schema_version": 2, "version": VERSION,
        "allowances_percent": list(ALLOWANCES),
        "full_method": "fixed-feature K=280 Galerkin finite-dimensional approximation",
        "execution_backends": {
            "information_projection": "candidate/shared-trajectory Tesseract C++ OpenMP",
            "galerkin_Kf_selected": galerkin_backend,
            "galerkin_Kf_default": "jax",
            "galerkin_Kf_available": ["jax", "tesseract_cpp"],
            "jax": "JAX/XLA GPU (device resident)",
            "tesseract_cpp": "independent Tesseract CPU/OpenBLAS assembly (host callback and transfers)",
        },
        "constants": {
            "K": K, "dictionary_sha256": EXPECTED_DICTIONARY_SHA256,
            "relative_rank_tolerance": RANK_TOLERANCE,
            "minimum_relative_ess": MINIMUM_RESS,
            "maximum_heldout_energy_residual": MAXIMUM_ENERGY_RESIDUAL,
            "projection_and_forcing": cfg["forcing"],
            "galerkin_algebra": {key: value for key, value in cfg["production_galerkin"].items()
                                   if key.startswith("maximum_") or key.startswith("minimum_")},
            "physical_certificate": certificates,
            "minimum_sensor_separation": cfg["measurement"]["min_separation"],
        },
        "risk": {
            "selection_rule": "R <= (1+p/100) R_Law",
            "validation_rule": "R <= (1+p/100+0.05) R_Law",
            "strict_nominal_reported": True,
            "anchor": "original frozen selection truth/projection data and config envelope.law_eta",
            "law_eta": cfg["envelope"]["law_eta"],
        },
        "banks": {"sizes": BANK_SIZES, "seed_records": selection_seeds,
                  "new_selection_side": True, "reference_retrained": False},
        "screening": {
            "pool": "law/history interpolation + deterministic local + risk-tangent + global",
            "interpolation_points": 17, "local_per_center": 16, "local_scale": 0.01,
            "risk_tangent_directions": 16, "risk_tangent_radii": [0.0001, 0.0005, 0.001, 0.005],
            "global_count": 32, "global_oversample": 16,
            "full_Kf_solve_permitted": False, "candidate_projection_batch": 8,
        },
        "starts": {
            "count_per_allowance": 4,
            "algorithm": "previous incumbent, Law, closest feasible history, maximum-rESS then max-min diversity",
            "deduplication_tolerance": 1.0e-12,
            "starts_need_complete_physical_certificate": False,
        },
        "optimizer": {
            "algorithm": "periodic projected normalized-gradient trust trajectory with exact backtracking",
            "trust_radius": 2.0e-4, "initial_step": 5.0e-5,
            "maximum_accepted_step_attempts": 4, "maximum_backtracks": 10,
            "backtrack_factor": 0.5, "successful_step_cap": 7.5e-5,
            "periodic_audit_every_accepted_steps": 4,
            "replacement_tolerance": 1.0e-10,
            "rank_must_equal_previous_step": True,
            "tangent": "exact four-dimensional Gram solve; autodifferentiate closed form",
            "full": "K280 fixed-coefficient envelope gradient; never differentiate eigensolve",
        },
        "shortlisting": {"maximum_finalists": 3,
                          "rule": "certified periodic-audit endpoints ordered by search action; incumbent included",
                          "authoritative_replacement_tolerance": 1.0e-10},
        "authoritative": {"recompute_from_scratch": True, "train_samples": 65536,
                          "audit_samples": 65536, "duplicate_eta_reuse_by_hash": True},
        "validation": {"sizes": VALIDATION_SIZES, "seed_records": validation_seeds,
                       "generation_forbidden_before_selection_seal": True,
                       "K": K, "no_optimization": True,
                       "size_justification": "16,384/16,384 is frozen from qualified K280 quadrature evidence; independent held-out certification controls numerical validity"},
        "reporting": {"no_pseudo_replicates": True, "actual_risk_increase": True,
                      "common_metric": "K280 Full action", "deep_ritz_excluded": True,
                      "classification": ["PASS", "VALIDATION RISK REVERSAL",
                                         "VALIDATION NUMERICAL FAILURE", "NO CERTIFIED SELECTION WINNER"]},
        "historical_immutable": {name: file_sha256(ROOT / name) for name in HISTORICAL_REPORTS},
        "protocol_document_sha256": file_sha256(PROTOCOL_DOCUMENT),
        "preserved_failed_prestart_attempt": {
            "reason": "distance helper call-signature error before any start selection or optimization",
            "protocol_sha256": file_sha256(FAILED_PRESTART_ROOT / "protocol.json"),
            "bank_manifest_sha256": file_sha256(FAILED_PRESTART_ROOT / "banks" / "manifest.json"),
            "failure_record_sha256": file_sha256(FAILED_PRESTART_ROOT / "failure.json"),
            "results_used": False,
        },
        "preserved_superseded_acceleration_attempt": {
            "reason": "paused before selection to isolate and benchmark Galerkin Tesseract",
            "protocol_sha256": file_sha256(SUPERSEDED_ACCEL_ROOT / "protocol.json"),
            "record_sha256": file_sha256(SUPERSEDED_ACCEL_ROOT / "superseded_record.json"),
            "results_used": False,
        },
        "preserved_superseded_native_separation_attempt": {
            "reason": "paused before banks/selection to restore original I-projection Tesseract and add an isolated candidate backend plus Galerkin backend flag",
            "protocol_sha256": file_sha256(
                SUPERSEDED_NATIVE_SEPARATION_ROOT / "protocol.json"
            ),
            "record_sha256": file_sha256(
                SUPERSEDED_NATIVE_SEPARATION_ROOT / "superseded_record.json"
            ),
            "results_used": False,
        },
        "native_accelerator_sha256": {
            "candidate_CMakeLists.txt": file_sha256(ROOT.parents[1] / "native" / "candidate_iprojection_tesseract" / "CMakeLists.txt"),
            "candidate_bindings.cpp": file_sha256(ROOT.parents[1] / "native" / "candidate_iprojection_tesseract" / "src" / "bindings.cpp"),
            "candidate_tesseract_api.py": file_sha256(ROOT.parents[1] / "native" / "candidate_iprojection_tesseract" / "tesseract_api.py"),
            "galerkin_CMakeLists.txt": file_sha256(ROOT.parents[1] / "native" / "galerkin_tesseract" / "CMakeLists.txt"),
            "galerkin_bindings.cpp": file_sha256(ROOT.parents[1] / "native" / "galerkin_tesseract" / "src" / "bindings.cpp"),
            "galerkin_tesseract_api.py": file_sha256(ROOT.parents[1] / "native" / "galerkin_tesseract" / "tesseract_api.py"),
            "mfsi_galerkin_tesseract.py": file_sha256(ROOT.parents[1] / "src" / "mfsi" / "galerkin_tesseract.py"),
            "mfsi_projection.py": file_sha256(ROOT.parents[1] / "src" / "mfsi" / "projection.py"),
            "mfsi_projection_tesseract.py": file_sha256(ROOT.parents[1] / "src" / "mfsi" / "projection_tesseract.py"),
        },
        "source_sha256": {name: file_sha256(ROOT / name) for name in source_names},
    }


def freeze_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    body = protocol_payload(cfg)
    digest = payload_sha256(body)
    wrapped = {**body, "protocol_sha256": digest, "protocol_frozen": True,
               "validation_arrays_generated": False}
    if PROTOCOL_PATH.exists():
        if read_json(PROTOCOL_PATH) != wrapped:
            raise RuntimeError("different Pareto-v2 protocol already exists")
    else:
        atomic_json(PROTOCOL_PATH, wrapped, immutable=True)
        atomic_text(PROTOCOL_HASH_PATH, digest + "\n", immutable=True)
    if PROTOCOL_HASH_PATH.read_text().strip() != digest:
        raise RuntimeError("Pareto-v2 protocol sidecar mismatch")
    return wrapped


def require_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    if not PROTOCOL_PATH.is_file() or not PROTOCOL_HASH_PATH.is_file():
        raise RuntimeError("freeze-protocol must complete first")
    saved = read_json(PROTOCOL_PATH)
    body = dict(saved)
    digest = body.pop("protocol_sha256", None)
    frozen = body.pop("protocol_frozen", None)
    generated = body.pop("validation_arrays_generated", None)
    if not frozen or generated is not False or payload_sha256(body) != digest:
        raise RuntimeError("Pareto-v2 protocol seal mismatch")
    if PROTOCOL_HASH_PATH.read_text().strip() != digest or protocol_payload(cfg) != body:
        raise RuntimeError("current code/config differs from frozen Pareto-v2 protocol")
    return saved


def hashes(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [{"path": str(path.relative_to(OUTPUT_ROOT)), "bytes": path.stat().st_size,
             "sha256": file_sha256(path)} for path in paths]
