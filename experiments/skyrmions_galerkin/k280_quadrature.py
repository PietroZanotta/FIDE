"""Fixed-K=280 selection-development quadrature qualification.

This module intentionally exposes no sensor optimizer, Pareto workflow, or
validation loader.  It reuses the already validated streamed Galerkin
implementation while varying only the empirical train/audit support.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import resource
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import numpy as np

from mfsi.cache import fingerprint

from . import galerkin_only, production_galerkin, resolution_study
from .domain import SkyrmionTruth
from .galerkin_only_data import load_selection_galerkin_data
from .production_artifacts import file_sha256
from .production_basis import load_dictionary
from .reference import load_reference


PACKAGE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PACKAGE_ROOT / "outputs" / "galerkin_k280_quadrature_extension"
PROTOCOL_PATH = OUTPUT_ROOT / "protocol.json"
PROTOCOL_HASH_PATH = OUTPUT_ROOT / "protocol_hash.txt"
GATE_AUDIT_PATH = OUTPUT_ROOT / "gate_audit" / "previous_32768_16384.json"
BANK_ROOT = OUTPUT_ROOT / "banks"
TRAIN_BANK_PATH = BANK_ROOT / "selection_development_only_train_131072.npz"
AUDIT_BANK_PATH = BANK_ROOT / "selection_development_only_audit_65536.npz"
BANK_MANIFEST_PATH = OUTPUT_ROOT / "bank_manifest.json"
SUPPORT_RESULT_PATH = OUTPUT_ROOT / "support" / "result.json"
ANALYSIS_PATH = OUTPUT_ROOT / "support" / "analysis.json"
FD_RESULT_PATH = OUTPUT_ROOT / "finite_difference" / "result.json"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
REPORT_PATH = PACKAGE_ROOT / "GALERKIN_K280_QUADRATURE_QUALIFICATION.md"
PROTOCOL_MD_PATH = PACKAGE_ROOT / "GALERKIN_K280_QUADRATURE_EXTENSION_PROTOCOL.md"

ARTIFACT_DIR = resolution_study.ARTIFACT_DIR
DICTIONARY_PATH = resolution_study.DICTIONARY_PATH
EXPECTED_DICTIONARY_SHA256 = (
    "37e9b60fcb92c4e5a0ee7ec1651fb7f8889f7ac6bdb02d3bd314e9ef40833326"
)
VERSION = "skyrmion_k280_quadrature_extension_v1"
K_FIXED = 280
RANK_TOLERANCE = 1.0e-12
ENERGY_THRESHOLD = 0.08
MANDATORY_SUPPORTS = (
    (32768, 16384),
    (32768, 32768),
    (65536, 32768),
    (65536, 65536),
)
OPTIONAL_SUPPORT = (131072, 65536)
MAXIMUM_TRAIN_SAMPLES = 131072
MAXIMUM_AUDIT_SAMPLES = 65536
FD_GEOMETRIES = ("law", "historical_0p5", "eta0_3pct", "eta_grad_3pct")
FD_EPSILONS = (1.0e-3, 3.0e-4, 1.0e-4)
INITIAL_GIT_STATUS = """ M experiments/skyrmions_deep_ritz_full/README.md
 M experiments/skyrmions_deep_ritz_full/deep_ritz.py
 M experiments/skyrmions_deep_ritz_full/run.py
 M experiments/skyrmions_deep_ritz_full/workflow.py
?? experiments/skyrmions_deep_ritz_full/AUTHORITATIVE_GPU_ACCELERATION.md
?? experiments/skyrmions_deep_ritz_full/AUTHORITATIVE_STABILITY_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/FAST_PRODUCTION_3PCT_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/FINAL_3PCT_GALERKIN_CROSSCHECK.md
?? experiments/skyrmions_deep_ritz_full/GALERKIN_ONLY_3PCT_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/GALERKIN_RESOLUTION_STUDY.md
?? experiments/skyrmions_deep_ritz_full/GALERKIN_RESOLUTION_STUDY_PROTOCOL.md
?? experiments/skyrmions_deep_ritz_full/OFFICIAL_GALERKIN_PARETO_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/OFFICIAL_GALERKIN_PARETO_PROTOCOL.md
?? experiments/skyrmions_deep_ritz_full/authoritative_platform.py
?? experiments/skyrmions_deep_ritz_full/authoritative_stability.py
?? experiments/skyrmions_deep_ritz_full/final_crosscheck.py
?? experiments/skyrmions_deep_ritz_full/final_crosscheck_run.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only_data.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only_run.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only_workflow.py
?? experiments/skyrmions_deep_ritz_full/official_pareto_common.py
?? experiments/skyrmions_deep_ritz_full/official_pareto_report.py
?? experiments/skyrmions_deep_ritz_full/official_pareto_run.py
?? experiments/skyrmions_deep_ritz_full/official_pareto_selection.py
?? experiments/skyrmions_deep_ritz_full/official_pareto_validation.py
?? experiments/skyrmions_deep_ritz_full/resolution_study.py
?? experiments/skyrmions_deep_ritz_full/resolution_study_report.py
?? experiments/skyrmions_deep_ritz_full/resolution_study_run.py
?? experiments/skyrmions_deep_ritz_full/test_fast_production.py
?? experiments/skyrmions_deep_ritz_full/test_final_crosscheck.py
?? experiments/skyrmions_deep_ritz_full/test_galerkin_only.py
?? experiments/skyrmions_deep_ritz_full/test_official_pareto.py
?? experiments/skyrmions_deep_ritz_full/test_resolution_study.py"""

HISTORICAL_RECORDS = {
    "OFFICIAL_GALERKIN_PARETO_PROTOCOL.md": PACKAGE_ROOT / "OFFICIAL_GALERKIN_PARETO_PROTOCOL.md",
    "OFFICIAL_GALERKIN_PARETO_EVALUATION.md": PACKAGE_ROOT / "OFFICIAL_GALERKIN_PARETO_EVALUATION.md",
    "GALERKIN_RESOLUTION_STUDY_PROTOCOL.md": PACKAGE_ROOT / "GALERKIN_RESOLUTION_STUDY_PROTOCOL.md",
    "GALERKIN_RESOLUTION_STUDY.md": PACKAGE_ROOT / "GALERKIN_RESOLUTION_STUDY.md",
}
HISTORICAL_OUTPUT_ROOTS = {
    "official_galerkin_pareto": PACKAGE_ROOT / "outputs" / "official_galerkin_pareto",
    "galerkin_resolution_study": PACKAGE_ROOT / "outputs" / "galerkin_resolution_study",
}


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def require_output_path(path: Path) -> Path:
    resolved, root = Path(path).resolve(), OUTPUT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"K280 quadrature output must be beneath {root}, got {resolved}")
    return resolved


def write_json(path: Path, payload: dict[str, Any], *, overwrite: bool = False) -> None:
    path = require_output_path(path)
    if path.exists() and not overwrite:
        raise RuntimeError(f"refusing to overwrite K280 quadrature output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def historical_snapshot() -> dict[str, Any]:
    return {
        "records": {name: file_sha256(path) for name, path in HISTORICAL_RECORDS.items()},
        "output_trees": {name: tree_hashes(root) for name, root in HISTORICAL_OUTPUT_ROOTS.items()},
    }


def derive_seed(global_seed: int, role: str) -> dict[str, Any]:
    text = f"{int(global_seed)}:skyrmion:k280_quad_extension:v1:{role}:max"
    digest = hashlib.sha256(text.encode()).hexdigest()
    return {
        "role": role,
        "text": text,
        "sha256": digest,
        "seed": int(digest[:16], 16) % (2**31 - 1),
    }


def deterministic_directions(global_seed: int, geometry_id: str) -> list[list[float]]:
    directions = []
    for index in range(2):
        label = f"{int(global_seed)}:skyrmion:k280_quad_extension:v1:fd:{geometry_id}:{index}"
        seed = int(hashlib.sha256(label.encode()).hexdigest()[:16], 16) % (2**32)
        vector = np.random.default_rng(seed).normal(size=8)
        vector /= np.linalg.norm(vector)
        directions.append(vector.tolist())
    return directions


def _line_reference(function: Any) -> str:
    _, line = inspect.getsourcelines(function)
    return f"{Path(inspect.getsourcefile(function) or '').name}:{line}"


def _forcing_gates(payload: dict[str, Any], cfg: dict[str, Any], suffix: str) -> dict[str, bool]:
    forcing = cfg["forcing"]
    return {
        f"projection_valid_{suffix}": bool(
            payload["maximum_projection_residual"] <= float(forcing["projection_tolerance"])
        ),
        f"ESS_valid_{suffix}": bool(
            payload["minimum_ess_fraction"] >= float(forcing["minimum_ess_fraction"])
        ),
        f"forcing_valid_{suffix}": bool(
            payload["maximum_forcing_mean"] <= float(forcing["forcing_mean_tolerance"])
        ),
        f"covariance_valid_{suffix}": bool(
            payload["maximum_covariance_condition"] <= float(forcing["max_covariance_condition"])
        ),
    }


def gate_decomposition(row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    settings = cfg["production_galerkin"]
    thresholds = settings["certificate_thresholds"]
    algebra = row["algebra"]
    heldout = row["heldout_certificate"]
    gates: dict[str, Any] = {}
    gates.update(_forcing_gates(row["train_forcing"], cfg, "train"))
    gates.update(_forcing_gates(row["audit_forcing"], cfg, "audit"))
    gates.update({
        "geometry_valid": bool(row["geometry_valid"]),
        "rank_valid": bool(
            algebra["minimum_rank_fraction"] >= float(settings["minimum_rank_fraction"])
            and algebra["worst_retained_condition"] <= float(settings["maximum_retained_condition"])
        ),
        "range_valid": bool(algebra["worst_range_residual"] <= float(settings["maximum_range_residual"])),
        "stationarity_valid": bool(
            algebra["worst_stationarity_residual"] <= float(settings["maximum_stationarity_residual"])
        ),
        "symmetry_valid": bool(
            algebra["worst_symmetry_residual"] <= float(settings["maximum_symmetry_residual"])
        ),
        "restricted_identity_valid": bool(
            algebra["identity_relerr"] <= float(settings["maximum_identity_relerr"])
        ),
        "weak_valid": bool(
            heldout["maximum_weak_residual"] <= float(thresholds["maximum_weak_residual"])
        ),
        "energy_valid": bool(
            heldout["maximum_energy_residual"] <= float(thresholds["maximum_energy_residual"])
        ),
        "gauge_valid": bool(
            heldout["maximum_gauge_residual"] <= float(thresholds["maximum_gauge_residual"])
        ),
        "moment_rate_valid": bool(
            heldout["maximum_moment_rate_residual"] <= float(thresholds["maximum_moment_rate_residual"])
        ),
    })
    train_forcing = all(gates[key] for key in (
        "projection_valid_train", "ESS_valid_train", "forcing_valid_train",
        "covariance_valid_train",
    ))
    audit_forcing = all(gates[key] for key in (
        "projection_valid_audit", "ESS_valid_audit", "forcing_valid_audit",
        "covariance_valid_audit",
    ))
    algebra_valid = all(gates[key] for key in (
        "rank_valid", "range_valid", "stationarity_valid", "symmetry_valid",
        "restricted_identity_valid",
    ))
    heldout_valid = all(gates[key] for key in (
        "weak_valid", "energy_valid", "gauge_valid", "moment_rate_valid",
    ))
    gates.update({
        "train_forcing_aggregate_valid": train_forcing,
        "audit_forcing_aggregate_valid": audit_forcing,
        "algebra_aggregate_valid": algebra_valid,
        "heldout_physical_aggregate_valid": heldout_valid,
        "support_or_feasibility_gate_present": False,
        "support_or_feasibility_valid": None,
        "study_specific_convergence_gate_present_in_complete": False,
        "study_specific_convergence_valid": None,
        "implementation_complete_gate": bool(
            gates["geometry_valid"] and train_forcing and audit_forcing
            and algebra_valid and heldout_valid
        ),
        "final_certified": bool(row["complete_certificate"]),
    })
    gates["failed_booleans"] = [
        key for key, value in gates.items()
        if isinstance(value, bool) and not value
        and key not in (
            "support_or_feasibility_gate_present",
            "study_specific_convergence_gate_present_in_complete",
        )
    ]
    return gates


def audit_old_certificates(cfg: dict[str, Any]) -> dict[str, Any]:
    if GATE_AUDIT_PATH.is_file():
        return {**read_json(GATE_AUDIT_PATH), "cache_hit": True}
    old_root = HISTORICAL_OUTPUT_ROOTS["galerkin_resolution_study"]
    analysis = read_json(old_root / "quadrature" / "analysis.json")
    comparisons = {item["geometry_id"]: item["consecutive_comparisons"][-1]
                   for item in analysis["analyses"]}
    rows = []
    for geometry_id, _, _ in resolution_study.FIXED_GEOMETRIES:
        path = old_root / "quadrature" / "K280" / geometry_id / "train_32768_audit_16384.json"
        row = read_json(path)
        gates = gate_decomposition(row, cfg)
        comparison = comparisons[geometry_id]
        study_checks = {
            "energy": bool(row["heldout_certificate"]["maximum_energy_residual"] <= ENERGY_THRESHOLD),
            "train_action": bool(comparison["train_action_relative_change"] <= 0.03),
            "gradient_direction": bool(comparison["cosine"] >= 0.995),
            "gradient_magnitude": bool(comparison["relative_difference"] <= 0.05),
            "algebra": bool(row["algebra"]["valid"]),
        }
        gates["study_specific_convergence_valid"] = bool(all(study_checks.values()))
        rows.append({
            "geometry_id": geometry_id,
            "source": str(path.relative_to(PACKAGE_ROOT)),
            "source_sha256": file_sha256(path),
            "metrics": {
                "train_minimum_ess_fraction": row["train_forcing"]["minimum_ess_fraction"],
                "audit_minimum_ess_fraction": row["audit_forcing"]["minimum_ess_fraction"],
                "weak": row["heldout_certificate"]["maximum_weak_residual"],
                "energy": row["heldout_certificate"]["maximum_energy_residual"],
                "gauge": row["heldout_certificate"]["maximum_gauge_residual"],
                "moment_rate": row["heldout_certificate"]["maximum_moment_rate_residual"],
            },
            "gates": gates,
            "old_study_qualification_checks": study_checks,
            "physical_numerical_certificate": gates["implementation_complete_gate"],
            "resolution_study_qualification": gates["study_specific_convergence_valid"],
        })
    result = {
        "ran": True,
        "passed": all(row["gates"]["final_certified"] == row["physical_numerical_certificate"] for row in rows),
        "rows": rows,
        "thresholds": {"forcing": cfg["forcing"], "galerkin": cfg["production_galerkin"]},
        "implementation_trace": {
            "forcing_aggregate": _line_reference(galerkin_only._forcing_state_payload),
            "heldout_physical": _line_reference(production_galerkin.audit_hybrid_solutions),
            "complete_certificate": _line_reference(resolution_study.evaluate_case),
            "study_qualification": _line_reference(resolution_study.analyze_quadrature),
            "exact_complete_expression": (
                "geometry_valid and train_forcing.valid and audit_forcing.valid "
                "and algebra.valid and heldout_certificate.valid"
            ),
        },
        "validation_accessed": False,
        "eta_optimization_run": False,
    }
    write_json(GATE_AUDIT_PATH, result)
    return result


def _source_hashes() -> dict[str, str]:
    names = (
        "k280_quadrature.py", "k280_quadrature_run.py", "k280_quadrature_report.py",
        "test_k280_quadrature.py", "GALERKIN_K280_QUADRATURE_EXTENSION_PROTOCOL.md",
        "resolution_study.py", "galerkin_only.py", "production_galerkin.py",
        "production_gradient.py", "production_basis.py", "config.json",
    )
    return {name: file_sha256(PACKAGE_ROOT / name) for name in names}


def protocol_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    if not GATE_AUDIT_PATH.is_file():
        raise RuntimeError("audit-old-certificates must run before protocol freeze")
    if file_sha256(DICTIONARY_PATH) != EXPECTED_DICTIONARY_SHA256:
        raise RuntimeError("fixed K=280 dictionary hash changed")
    if float(cfg["production_galerkin"]["relative_rank_tolerance"]) != RANK_TOLERANCE:
        raise RuntimeError("rank tolerance differs from fixed 1e-12")
    if float(cfg["production_galerkin"]["certificate_thresholds"]["maximum_energy_residual"]) != ENERGY_THRESHOLD:
        raise RuntimeError("energy threshold differs from fixed 0.08")
    seed_rows = [derive_seed(cfg["seed"], role) for role in ("train", "audit")]
    geometries = [
        {"id": name, "provenance": provenance, "eta": eta}
        for name, provenance, eta in resolution_study.FIXED_GEOMETRIES
    ]
    return {
        "schema_version": 1,
        "version": VERSION,
        "initial_git_status_short": INITIAL_GIT_STATUS.splitlines(),
        "purpose": "selection_development_only fixed-K=280 empirical quadrature qualification",
        "validation_access_permitted": False,
        "eta_optimization_permitted": False,
        "K": K_FIXED,
        "dictionary_sha256": EXPECTED_DICTIONARY_SHA256,
        "dictionary_ordering": "unchanged",
        "dictionary_normalization": "unchanged eta-independent frozen normalization",
        "relative_rank_tolerance": RANK_TOLERANCE,
        "forcing_thresholds": cfg["forcing"],
        "algebra_thresholds": {key: cfg["production_galerkin"][key] for key in (
            "maximum_range_residual", "maximum_stationarity_residual",
            "maximum_identity_relerr", "maximum_symmetry_residual",
            "minimum_rank_fraction", "maximum_retained_condition",
        )},
        "certificate_thresholds": cfg["production_galerkin"]["certificate_thresholds"],
        "fixed_geometries": geometries,
        "mandatory_support_ladder": [list(row) for row in MANDATORY_SUPPORTS],
        "optional_support": {
            "pair": list(OPTIONAL_SUPPORT),
            "predeclared": True,
            "run_if": "mandatory 65536-level qualification is not fully resolved",
        },
        "development_banks": {
            "label": "selection_development_only",
            "maximum_train_samples": MAXIMUM_TRAIN_SAMPLES,
            "maximum_audit_samples": MAXIMUM_AUDIT_SAMPLES,
            "exact_prefix_nesting": True,
            "seeds": seed_rows,
            "reference_checkpoint_sha256": file_sha256(ARTIFACT_DIR / "reference.npz"),
            "reference_retrained": False,
            "validation_overlap_check": "forbidden; independent namespace and continuous fresh draws",
        },
        "streaming": {
            "chunk_size": int(cfg["production_galerkin"]["chunk_size"]),
            "per_sample_KxK_gram_prohibited": True,
        },
        "paired_comparisons": {
            "action_relative_change": "abs(high-low)/max(abs(high),1e-12)",
            "gradient_cosine": "dot(high,low)/(norm(high)*norm(low))",
            "gradient_relative_change": "norm(high-low)/max(norm(high),1e-12)",
            "source_attribution_dominance_ratio": 1.5,
            "source_attribution_negligible_energy_change": 0.005,
        },
        "qualification": {
            "physical_final_two": "all six rows pass every physical/numerical gate",
            "train_action_relative_maximum": 0.02,
            "audit_action_relative_maximum": 0.02,
            "preferred_action_relative_maximum": 0.01,
            "gradient_cosine_minimum": 0.995,
            "gradient_relative_maximum": 0.05,
            "direction_stable_scale_unresolved_cosine": 0.999,
            "direction_stable_scale_unresolved_action": 0.01,
            "material_gradient_component": 1.0e-6,
            "mandatory_relevant_comparisons": {
                "train": [[32768, 32768], [65536, 32768]],
                "audit": [[65536, 32768], [65536, 65536]],
            },
            "optional_relevant_comparison": [[65536, 65536], [131072, 65536]],
        },
        "finite_difference": {
            "run_if": "final physical validity and basic action stability pass",
            "geometries": list(FD_GEOMETRIES),
            "directions": {
                geometry_id: deterministic_directions(cfg["seed"], geometry_id)
                for geometry_id in FD_GEOMETRIES
            },
            "epsilons": list(FD_EPSILONS),
            "relative_error_required": 0.01,
            "relative_error_preferred": 0.005,
            "all_perturbations_require_rank_and_physical_numerical_validity": True,
        },
        "old_gate_audit_sha256": file_sha256(GATE_AUDIT_PATH),
        "historical_snapshot": historical_snapshot(),
        "source_hashes": _source_hashes(),
    }


def freeze_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    payload = protocol_payload(cfg)
    digest = payload_sha256(payload)
    result = {**payload, "protocol_sha256": digest, "protocol_frozen": True}
    if PROTOCOL_PATH.exists():
        if read_json(PROTOCOL_PATH) != result:
            raise RuntimeError("different K280 quadrature protocol already exists")
    else:
        write_json(PROTOCOL_PATH, result)
        PROTOCOL_HASH_PATH.write_text(digest + "\n", encoding="utf-8")
    return result


def require_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    if not PROTOCOL_PATH.is_file() or not PROTOCOL_HASH_PATH.is_file():
        raise RuntimeError("freeze-protocol must run first")
    frozen = read_json(PROTOCOL_PATH)
    payload = dict(frozen)
    digest = payload.pop("protocol_sha256", None)
    flag = payload.pop("protocol_frozen", None)
    if not flag or payload_sha256(payload) != digest:
        raise RuntimeError("K280 quadrature protocol hash mismatch")
    if PROTOCOL_HASH_PATH.read_text().strip() != digest:
        raise RuntimeError("K280 quadrature protocol hash file mismatch")
    if protocol_payload(cfg) != payload:
        raise RuntimeError("current state differs from frozen K280 quadrature protocol")
    return frozen


def _save_npz_once(path: Path, **arrays: Any) -> None:
    path = require_output_path(path)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite development bank: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in arrays.items()})


def _row_hashes(rows: np.ndarray) -> set[str]:
    flat = np.ascontiguousarray(rows).reshape((rows.shape[0], -1))
    return {hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in flat}


def _initial_rows(path: Path, key: str = "configurations") -> np.ndarray:
    with np.load(path, allow_pickle=False) as arrays:
        values = np.asarray(arrays[key])
        return values[0] if values.ndim == 4 else values


def generate_banks(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_protocol(cfg)
    if BANK_MANIFEST_PATH.is_file():
        old = read_json(BANK_MANIFEST_PATH)
        if old.get("protocol_sha256") != protocol["protocol_sha256"] or not old.get("passed"):
            raise RuntimeError("incompatible K280 bank manifest exists")
        for artifact in old["artifacts"]:
            if file_sha256(OUTPUT_ROOT / artifact["path"]) != artifact["sha256"]:
                raise RuntimeError("sealed K280 development bank changed")
        return {**old, "cache_hit": True}
    times = jnp.linspace(0.0, 1.0, int(cfg["physics"]["time_nodes"]), dtype=jnp.float64)
    truth = SkyrmionTruth(resolution_study._physics_config(cfg))
    flow = load_reference(ARTIFACT_DIR / "reference.npz")
    seeds = {row["role"]: row for row in protocol["development_banks"]["seeds"]}
    for role, samples, path in (
        ("train", MAXIMUM_TRAIN_SAMPLES, TRAIN_BANK_PATH),
        ("audit", MAXIMUM_AUDIT_SAMPLES, AUDIT_BANK_PATH),
    ):
        configurations, velocity, weights = resolution_study._reference_bank(
            flow, truth, times, seed=seeds[role]["seed"], samples=samples,
            substeps=int(cfg["banks"]["reference_substeps"]), chunk_size=2048,
        )
        _save_npz_once(
            path,
            times=times,
            configurations=configurations,
            velocity=velocity,
            base_weights=weights,
            selection_development_only=np.asarray(True),
            seed_label=np.asarray(seeds[role]["text"]),
        )
    fresh = {
        "train": _row_hashes(_initial_rows(TRAIN_BANK_PATH)),
        "audit": _row_hashes(_initial_rows(AUDIT_BANK_PATH)),
    }
    permitted_paths = {
        f"historical_selection_{name}": ARTIFACT_DIR / f"reference_bank_{name}.npz"
        for name in ("projection", "ritz_train", "ritz_audit")
    }
    permitted_paths.update({
        "prior_resolution_train": resolution_study.TRAIN_BANK_PATH,
        "prior_resolution_audit": resolution_study.AUDIT_BANK_PATH,
    })
    permitted = {name: _row_hashes(_initial_rows(path)) for name, path in permitted_paths.items()}
    checks = [{"left": "train", "right": "audit", "overlap": len(fresh["train"] & fresh["audit"])}]
    for left in ("train", "audit"):
        for right, hashes in permitted.items():
            checks.append({"left": left, "right": right, "overlap": len(fresh[left] & hashes)})
    artifacts = [{
        "path": str(path.relative_to(OUTPUT_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    } for path in (TRAIN_BANK_PATH, AUDIT_BANK_PATH)]
    result = {
        "ran": True,
        "passed": all(check["overlap"] == 0 for check in checks),
        "protocol_sha256": protocol["protocol_sha256"],
        "label": "selection_development_only",
        "exact_sample_counts": {"train": MAXIMUM_TRAIN_SAMPLES, "audit": MAXIMUM_AUDIT_SAMPLES},
        "nested_prefixes": {
            "train": [32768, 65536, 131072],
            "audit": [16384, 32768, 65536],
        },
        "seed_records": list(seeds.values()),
        "exact_overlap_checks": checks,
        "historical_validation_arrays_accessed": False,
        "validation_disjointness_basis": (
            "independent versioned seed namespace and fresh continuous initial-state draws; "
            "forbidden validation arrays were not opened"
        ),
        "artifacts": artifacts,
    }
    write_json(BANK_MANIFEST_PATH, result)
    if not result["passed"]:
        raise RuntimeError("K280 development-bank overlap check failed")
    return result


def load_bank(path: Path, samples: int):
    return resolution_study.load_development_bank(path, samples)


def relative_change(high: float, low: float) -> float:
    return abs(float(high) - float(low)) / max(abs(float(high)), 1.0e-12)


def gradient_comparison(high: Iterable[float], low: Iterable[float]) -> dict[str, Any]:
    high_array, low_array = np.asarray(high, dtype=float), np.asarray(low, dtype=float)
    high_norm, low_norm = np.linalg.norm(high_array), np.linalg.norm(low_array)
    difference = high_array - low_array
    material = np.maximum(np.abs(high_array), np.abs(low_array)) >= 1.0e-6
    sign_stable = np.logical_or(~material, np.signbit(high_array) == np.signbit(low_array))
    return {
        "cosine": float(np.dot(high_array, low_array) / max(high_norm * low_norm, 1.0e-30)),
        "relative_difference": float(np.linalg.norm(difference) / max(high_norm, 1.0e-12)),
        "per_coordinate_difference": difference.tolist(),
        "material_component_sign_stable": bool(np.all(sign_stable)),
    }


def _case_signature(protocol: dict[str, Any], manifest: dict[str, Any], geometry: dict[str, Any],
                    support: tuple[int, int]) -> str:
    return fingerprint({
        "kind": VERSION,
        "protocol": protocol["protocol_sha256"],
        "banks": manifest["artifacts"],
        "geometry": geometry,
        "support": list(support),
        "K": K_FIXED,
        "rank_tolerance": RANK_TOLERANCE,
    })


def _enhance_row(row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    gradient = np.asarray(row["gradient"], dtype=float)
    gates = gate_decomposition(row, cfg)
    return {
        **row,
        "gradient_rms": float(np.sqrt(np.mean(gradient * gradient))),
        "gradient_maximum_absolute_component": float(np.max(np.abs(gradient))),
        "individual_gates": gates,
        "physical_numerical_certificate": bool(gates["implementation_complete_gate"]),
        "support_convergence_qualification": None,
    }


def _memory_snapshot() -> dict[str, Any]:
    device = jax.devices()[0]
    stats = device.memory_stats() or {}
    return {
        "platform": device.platform,
        "device": str(device),
        "host_peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "device_bytes_in_use": stats.get("bytes_in_use"),
        "device_peak_bytes_in_use": stats.get("peak_bytes_in_use"),
        "device_bytes_limit": stats.get("bytes_limit"),
    }


def _evaluate_supports(cfg: dict[str, Any], supports: list[tuple[int, int]]) -> list[dict[str, Any]]:
    protocol, manifest = require_protocol(cfg), read_json(BANK_MANIFEST_PATH)
    selection_data = load_selection_galerkin_data(cfg, ARTIFACT_DIR)
    dictionary = load_dictionary(DICTIONARY_PATH, box=tuple(cfg["physics"]["box"]))
    results = []
    for geometry in protocol["fixed_geometries"]:
        rows = []
        for support in supports:
            train_samples, audit_samples = support
            output = OUTPUT_ROOT / "support" / f"{train_samples}_{audit_samples}" / f"{geometry['id']}.json"
            signature = _case_signature(protocol, manifest, geometry, support)
            if output.is_file():
                row = read_json(output)
                if row.get("signature") != signature:
                    raise RuntimeError(f"support signature mismatch: {output}")
            else:
                train = load_bank(TRAIN_BANK_PATH, train_samples)
                audit = load_bank(AUDIT_BANK_PATH, audit_samples)
                row = resolution_study.evaluate_case(
                    cfg, selection_data, dictionary, train, audit, geometry["eta"],
                    K=K_FIXED, rank_tolerance=RANK_TOLERANCE,
                )
                row = _enhance_row(row, cfg)
                row.update({
                    "signature": signature,
                    "geometry_id": geometry["id"],
                    "selection_development_only": True,
                    "validation_accessed": False,
                    "eta_optimization_run": False,
                })
                write_json(output, row)
            rows.append(row)
        results.append({"geometry_id": geometry["id"], "rows": rows})
    return results


def paired_comparison(low: dict[str, Any], high: dict[str, Any]) -> dict[str, Any]:
    return {
        "low_support": [low["train_samples"], low["audit_samples"]],
        "high_support": [high["train_samples"], high["audit_samples"]],
        "train_action_relative_change": relative_change(high["train_action"], low["train_action"]),
        "audit_action_relative_change": relative_change(high["audit_action"], low["audit_action"]),
        **gradient_comparison(high["gradient"], low["gradient"]),
        "weak_residual_difference": (
            high["heldout_certificate"]["maximum_weak_residual"]
            - low["heldout_certificate"]["maximum_weak_residual"]
        ),
        "energy_residual_difference": (
            high["heldout_certificate"]["maximum_energy_residual"]
            - low["heldout_certificate"]["maximum_energy_residual"]
        ),
        "moment_rate_residual_difference": (
            high["heldout_certificate"]["maximum_moment_rate_residual"]
            - low["heldout_certificate"]["maximum_moment_rate_residual"]
        ),
        "train_support_ratio": high["train_samples"] / low["train_samples"],
        "audit_support_ratio": high["audit_samples"] / low["audit_samples"],
    }


def _source_attribution(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    if len(rows) < 4:
        return {"classification": "unavailable"}
    audit_first = abs(
        rows[1]["heldout_certificate"]["maximum_energy_residual"]
        - rows[0]["heldout_certificate"]["maximum_energy_residual"]
    )
    train = abs(
        rows[2]["heldout_certificate"]["maximum_energy_residual"]
        - rows[1]["heldout_certificate"]["maximum_energy_residual"]
    )
    audit_second = abs(
        rows[3]["heldout_certificate"]["maximum_energy_residual"]
        - rows[2]["heldout_certificate"]["maximum_energy_residual"]
    )
    audit = max(audit_first, audit_second)
    ratio = float(protocol["paired_comparisons"]["source_attribution_dominance_ratio"])
    negligible = float(protocol["paired_comparisons"]["source_attribution_negligible_energy_change"])
    if max(train, audit) < negligible:
        classification = "neither"
    elif train > ratio * max(audit, 1.0e-15):
        classification = "train fitting error"
    elif audit > ratio * max(train, 1.0e-15):
        classification = "audit Monte Carlo error"
    else:
        classification = "both"
    return {
        "audit_only_first_energy_change": audit_first,
        "train_only_energy_change": train,
        "audit_only_second_energy_change": audit_second,
        "classification": classification,
    }


def _qualification(geometries: list[dict[str, Any]], optional_used: bool) -> dict[str, Any]:
    checks = []
    for item in geometries:
        rows = item["rows"]
        if optional_used:
            low, high = rows[-2], rows[-1]
            comparisons = [paired_comparison(low, high)]
        else:
            low_train, high_train = rows[1], rows[2]
            low_audit, high_audit = rows[2], rows[3]
            comparisons = [paired_comparison(low_train, high_train), paired_comparison(low_audit, high_audit)]
            low, high = rows[2], rows[3]
        physical = bool(low["physical_numerical_certificate"] and high["physical_numerical_certificate"])
        action = bool(all(
            comparison["train_action_relative_change"] <= 0.02
            and comparison["audit_action_relative_change"] <= 0.02
            for comparison in comparisons
        ))
        direction = bool(all(comparison["cosine"] >= 0.995 for comparison in comparisons))
        magnitude = bool(all(comparison["relative_difference"] <= 0.05 for comparison in comparisons))
        direction_scale_unresolved = bool(
            direction and not magnitude
            and all(comparison["cosine"] >= 0.999 for comparison in comparisons)
            and all(comparison["material_component_sign_stable"] for comparison in comparisons)
            and all(
                comparison["train_action_relative_change"] <= 0.01
                and comparison["audit_action_relative_change"] <= 0.01
                for comparison in comparisons
            )
        )
        checks.append({
            "geometry_id": item["geometry_id"],
            "physical_final_two": physical,
            "action_stable": action,
            "gradient_direction_stable": direction,
            "gradient_magnitude_stable": magnitude,
            "direction_stable_scale_unresolved": direction_scale_unresolved,
            "relevant_comparisons": comparisons,
        })
    return {
        "geometry_checks": checks,
        "physical_valid": all(row["physical_final_two"] for row in checks),
        "action_stable": all(row["action_stable"] for row in checks),
        "gradient_direction_stable": all(row["gradient_direction_stable"] for row in checks),
        "gradient_magnitude_stable": all(row["gradient_magnitude_stable"] for row in checks),
        "direction_stable_scale_unresolved": all(
            row["gradient_direction_stable"]
            and (row["gradient_magnitude_stable"] or row["direction_stable_scale_unresolved"])
            for row in checks
        ) and not all(row["gradient_magnitude_stable"] for row in checks),
    }


def analyze_supports(cfg: dict[str, Any], geometries: list[dict[str, Any]],
                     optional_used: bool) -> dict[str, Any]:
    protocol = require_protocol(cfg)
    analyses = []
    for item in geometries:
        comparisons = [paired_comparison(low, high) for low, high in zip(item["rows"][:-1], item["rows"][1:])]
        action_changes = [max(row["train_action_relative_change"], row["audit_action_relative_change"])
                          for row in comparisons]
        scaling = []
        for previous, current in zip(action_changes[:-1], action_changes[1:]):
            scaling.append({
                "previous_max_action_change": previous,
                "current_max_action_change": current,
                "decay_ratio": current / max(previous, 1.0e-15),
                "decreased": bool(current < previous),
            })
        analyses.append({
            **item,
            "comparisons": comparisons,
            "empirical_support_scaling": scaling,
            "train_vs_audit_energy_source": _source_attribution(item["rows"], protocol),
        })
    qualification = _qualification(analyses, optional_used)
    return {
        "geometries": analyses,
        "qualification": qualification,
        "optional_support_used": optional_used,
        "finite_difference_unlocked": bool(
            qualification["physical_valid"] and qualification["action_stable"]
        ),
    }


def run_evaluate(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_protocol(cfg)
    if not read_json(BANK_MANIFEST_PATH).get("passed"):
        raise RuntimeError("development banks are not qualified")
    if SUPPORT_RESULT_PATH.is_file() and ANALYSIS_PATH.is_file():
        result = read_json(SUPPORT_RESULT_PATH)
        analysis = read_json(ANALYSIS_PATH)
        if result.get("protocol_sha256") == protocol["protocol_sha256"] and result.get("passed"):
            return {**result, "analysis": analysis, "cache_hit": True}
        raise RuntimeError("incompatible completed support result exists")
    mandatory = _evaluate_supports(cfg, list(MANDATORY_SUPPORTS))
    mandatory_analysis = analyze_supports(cfg, mandatory, optional_used=False)
    optional_required = not (
        mandatory_analysis["qualification"]["physical_valid"]
        and mandatory_analysis["qualification"]["action_stable"]
        and mandatory_analysis["qualification"]["gradient_direction_stable"]
        and mandatory_analysis["qualification"]["gradient_magnitude_stable"]
    )
    if optional_required:
        complete = _evaluate_supports(cfg, list(MANDATORY_SUPPORTS) + [OPTIONAL_SUPPORT])
        final_analysis = analyze_supports(cfg, complete, optional_used=True)
    else:
        complete, final_analysis = mandatory, mandatory_analysis
    result = {
        "ran": True,
        "passed": True,
        "protocol_sha256": protocol["protocol_sha256"],
        "geometries": complete,
        "mandatory_analysis": mandatory_analysis,
        "optional_support_required": optional_required,
        "optional_support_used": optional_required,
        "memory": _memory_snapshot(),
        "validation_accessed": False,
        "eta_optimization_run": False,
        "K": K_FIXED,
        "rank_tolerance": RANK_TOLERANCE,
        "energy_threshold": ENERGY_THRESHOLD,
    }
    analysis = {
        "ran": True,
        "passed": True,
        "protocol_sha256": protocol["protocol_sha256"],
        **final_analysis,
        "validation_accessed": False,
        "eta_optimization_run": False,
    }
    write_json(SUPPORT_RESULT_PATH, result)
    write_json(ANALYSIS_PATH, analysis)
    return {**result, "analysis": analysis}


def centered_fd(plus: float, minus: float, epsilon: float) -> float:
    return (float(plus) - float(minus)) / (2.0 * float(epsilon))


def _fd_case_signature(protocol: dict[str, Any], geometry: dict[str, Any], direction: list[float],
                       epsilon: float, support: tuple[int, int]) -> str:
    return fingerprint({
        "kind": f"{VERSION}_finite_difference",
        "protocol": protocol["protocol_sha256"],
        "geometry": geometry,
        "direction": direction,
        "epsilon": epsilon,
        "support": support,
    })


def run_finite_difference(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_protocol(cfg)
    analysis = read_json(ANALYSIS_PATH)
    if FD_RESULT_PATH.is_file():
        old = read_json(FD_RESULT_PATH)
        if old.get("protocol_sha256") == protocol["protocol_sha256"]:
            return {**old, "cache_hit": True}
        raise RuntimeError("incompatible finite-difference result exists")
    if not analysis["finite_difference_unlocked"]:
        result = {
            "ran": False,
            "passed": True,
            "skipped": True,
            "reason": "physical validity and basic action-stability prerequisite did not pass",
            "protocol_sha256": protocol["protocol_sha256"],
            "validation_accessed": False,
            "eta_optimization_run": False,
        }
        write_json(FD_RESULT_PATH, result)
        return result
    support = tuple(analysis["geometries"][0]["rows"][-1][key]
                    for key in ("train_samples", "audit_samples"))
    manifest = read_json(BANK_MANIFEST_PATH)
    selection_data = load_selection_galerkin_data(cfg, ARTIFACT_DIR)
    dictionary = load_dictionary(DICTIONARY_PATH, box=tuple(cfg["physics"]["box"]))
    train = load_bank(TRAIN_BANK_PATH, support[0])
    audit = load_bank(AUDIT_BANK_PATH, support[1])
    final_rows = {item["geometry_id"]: item["rows"][-1] for item in analysis["geometries"]}
    geometry_map = {item["id"]: item for item in protocol["fixed_geometries"]}
    direction_results = []
    for geometry_id in FD_GEOMETRIES:
        geometry = geometry_map[geometry_id]
        center = final_rows[geometry_id]
        center_rank = center["algebra"]["rank_by_time"]
        for direction_index, direction in enumerate(protocol["finite_difference"]["directions"][geometry_id]):
            direction_array = np.asarray(direction, dtype=float)
            ad = float(np.dot(np.asarray(center["gradient"], dtype=float), direction_array))
            epsilon_rows = []
            for epsilon in FD_EPSILONS:
                signature = _fd_case_signature(protocol, geometry, direction, epsilon, support)
                output = (OUTPUT_ROOT / "finite_difference" / geometry_id
                          / f"direction_{direction_index}_epsilon_{epsilon:.0e}.json")
                if output.is_file():
                    row = read_json(output)
                    if row.get("signature") != signature:
                        raise RuntimeError("finite-difference signature mismatch")
                else:
                    evaluations = {}
                    for sign_name, sign in (("plus", 1.0), ("minus", -1.0)):
                        eta = np.asarray(geometry["eta"], dtype=float) + sign * epsilon * direction_array
                        evaluated = resolution_study.evaluate_case(
                            cfg, selection_data, dictionary, train, audit, eta,
                            K=K_FIXED, rank_tolerance=RANK_TOLERANCE,
                        )
                        evaluations[sign_name] = _enhance_row(evaluated, cfg)
                    fd = centered_fd(
                        evaluations["plus"]["train_action"],
                        evaluations["minus"]["train_action"], epsilon,
                    )
                    relative_error = abs(fd - ad) / max(abs(fd), abs(ad), 1.0e-12)
                    rank_stable = all(
                        evaluations[name]["algebra"]["rank_by_time"] == center_rank
                        for name in ("plus", "minus")
                    )
                    gates_valid = all(
                        evaluations[name]["physical_numerical_certificate"]
                        for name in ("plus", "minus")
                    )
                    sign_valid = bool(abs(ad) <= 1.0e-12 or abs(fd) <= 1.0e-12 or np.signbit(ad) == np.signbit(fd))
                    row = {
                        "signature": signature,
                        "geometry_id": geometry_id,
                        "direction_index": direction_index,
                        "direction": direction,
                        "epsilon": epsilon,
                        "AD": ad,
                        "FD": fd,
                        "relative_error": relative_error,
                        "sign_valid": sign_valid,
                        "rank_stable": rank_stable,
                        "physical_numerical_gates_valid": gates_valid,
                        "plus": evaluations["plus"],
                        "minus": evaluations["minus"],
                        "bank_artifacts": manifest["artifacts"],
                    }
                    write_json(output, row)
                epsilon_rows.append(row)
            passed = bool(
                all(row["rank_stable"] and row["physical_numerical_gates_valid"] and row["sign_valid"]
                    for row in epsilon_rows)
                and any(row["relative_error"] <= 0.01 for row in epsilon_rows)
            )
            direction_results.append({
                "geometry_id": geometry_id,
                "direction_index": direction_index,
                "direction": direction,
                "AD": ad,
                "epsilons": epsilon_rows,
                "passed": passed,
            })
    result = {
        "ran": True,
        "skipped": False,
        "passed": all(row["passed"] for row in direction_results),
        "support": list(support),
        "directions": direction_results,
        "protocol_sha256": protocol["protocol_sha256"],
        "validation_accessed": False,
        "eta_optimization_run": False,
    }
    write_json(FD_RESULT_PATH, result)
    return result


def final_classification(analysis: dict[str, Any], finite_difference: dict[str, Any]) -> str:
    qualification = analysis["qualification"]
    fd_passed = bool(finite_difference.get("passed", False))
    if (
        qualification["physical_valid"]
        and qualification["action_stable"]
        and qualification["gradient_direction_stable"]
        and qualification["gradient_magnitude_stable"]
        and fd_passed
    ):
        return "A. K280 QUADRATURE QUALIFIED — READY TO FREEZE PARETO V2"
    if (
        qualification["physical_valid"]
        and qualification["action_stable"]
        and qualification["gradient_direction_stable"]
        and not qualification["gradient_magnitude_stable"]
        and fd_passed
    ):
        return "B. K280 PHYSICALLY VALID BUT GRADIENT QUADRATURE NOT YET CONVERGED"
    return "C. K280 ACTION/PHYSICAL QUADRATURE NOT YET CONVERGED"
