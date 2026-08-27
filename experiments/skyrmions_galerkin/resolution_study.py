"""Selection-development-only Galerkin resolution qualification study."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import numpy as np

from mfsi.cache import fingerprint

from .forcing import ForcingConfig
from .full_gradient import forcing_state, reconstruct_moments, wrap_periodic
from .galerkin import GalerkinSystem, aggregate_quadratic_values, rank_aware_quadratic_solve
from .galerkin_only import GalerkinCertificateThresholds, _forcing_state_payload, prefix_dictionary
from .galerkin_only_data import GalerkinReferenceBank, load_selection_galerkin_data, selection_risk
from .measurements import local_sensor_designs, random_sensor_designs
from .production_artifacts import PRODUCTION_ROOT, file_sha256
from .production_basis import load_dictionary
from .production_galerkin import (
    _normalized_chunk, assemble_hybrid_system, audit_hybrid_solutions,
    make_basis_evaluators,
)
from .production_gradient import production_hybrid_envelope_value_and_grad
from .reference import load_reference
from .domain import SkyrmionTruth


PACKAGE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PACKAGE_ROOT / "outputs" / "galerkin_resolution_study"
PROTOCOL_PATH = OUTPUT_ROOT / "protocol.json"
PROTOCOL_HASH_PATH = OUTPUT_ROOT / "protocol_hash.txt"
BANK_ROOT = OUTPUT_ROOT / "banks"
TRAIN_BANK_PATH = BANK_ROOT / "selection_development_only_train_32768.npz"
AUDIT_BANK_PATH = BANK_ROOT / "selection_development_only_audit_16384.npz"
BANK_MANIFEST_PATH = OUTPUT_ROOT / "bank_manifest.json"
ARTIFACT_DIR = PRODUCTION_ROOT / "artifacts"
DICTIONARY_PATH = (
    PACKAGE_ROOT / "outputs" / "galerkin_only_3pct" / "cache"
    / "dictionaries" / "dictionary_K280.npz"
)
V1_PROTOCOL_MD = PACKAGE_ROOT / "OFFICIAL_GALERKIN_PARETO_PROTOCOL.md"
V1_EVALUATION_MD = PACKAGE_ROOT / "OFFICIAL_GALERKIN_PARETO_EVALUATION.md"
V1_OUTPUT_ROOT = PACKAGE_ROOT / "outputs" / "official_galerkin_pareto"
REPORT_PATH = PACKAGE_ROOT / "GALERKIN_RESOLUTION_STUDY.md"

VERSION = "skyrmion_galerkin_resolution_v1"
K_PRIMARY = 280
K_LADDER = (120, 160, 200, 240, 280)
RANK_TOLERANCES = (1e-10, 1e-11, 1e-12)
SUPPORT_LADDER = ((8192, 4096), (16384, 8192), (16384, 16384), (32768, 16384))
EXPECTED_DICTIONARY_SHA256 = "37e9b60fcb92c4e5a0ee7ec1651fb7f8889f7ac6bdb02d3bd314e9ef40833326"
V1_IMMUTABLE_HASHES = {
    "OFFICIAL_GALERKIN_PARETO_PROTOCOL.md": "b948eb36e0c6c9d1e3b7045d45b18b9b5c8fe5cb917eb991209c01c16a1742b3",
    "OFFICIAL_GALERKIN_PARETO_EVALUATION.md": "af63788bd37ba547f248bdfa2229f0d295fe69c0640d7633df553419611ed540",
    "outputs/official_galerkin_pareto/protocol.json": "913bfa67c9d53e033b36d293cee714817bcfbcbe13c3324e412d282b147f5c9f",
    "outputs/official_galerkin_pareto/final_summary.json": "39dc4d407de58e74f9d6755630ad9de83e450d2b6239d7edeade03ad510e3dbd",
}

ETA_GRAD = [
    0.895371148114089, 0.205982940238786, 1.334525121515147,
    0.865464965382237, 0.750749623351011, 0.518133188490931,
    1.642405611981796, 0.588309862016330,
]
FIXED_GEOMETRIES = (
    ("law", "config.json envelope.law_eta", [
        0.890286510596537, 0.22728952886850587, 1.3103688321444902,
        0.8591631921629669, 0.7975888227142434, 0.5357230013163333,
        1.6103431504475714, 0.583219225445585,
    ]),
    ("historical_0p5", "original selection Pareto 0.5% frozen geometry", [
        0.8882240021144415, 0.2265900282857875, 1.3089283029966885,
        0.8628255147902797, 0.7866652061176428, 0.5418032213434409,
        1.6161758592555022, 0.584353406982718,
    ]),
    ("historical_1", "original selection Pareto 1% frozen geometry", [
        0.8916497660872147, 0.21592104181723273, 1.3254990498968335,
        0.861978425574543, 0.7740333752184337, 0.5278590825172568,
        1.6268094810638665, 0.5775087460426114,
    ]),
    ("historical_2", "original selection Pareto 2% frozen geometry", [
        0.894577442995983, 0.20411161892557242, 1.3400864770591099,
        0.8635508182176649, 0.76001873964639, 0.5143515626749267,
        1.6376150652013575, 0.5666851609212433,
    ]),
    ("eta0_3pct", "config.json envelope.eta0", [
        0.8954153767761239, 0.20592631632470587, 1.3343788098383822,
        0.8654288352917223, 0.7508355365766083, 0.5179100329264751,
        1.6423735249784726, 0.5883599695898114,
    ]),
    ("eta_grad_3pct", "frozen selection-only continuous 3% Galerkin winner", ETA_GRAD),
)


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def require_output_path(path: Path) -> Path:
    resolved, root = Path(path).resolve(), OUTPUT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"resolution-study output must be beneath {root}, got {resolved}")
    return resolved


def write_json(path: Path, payload: dict[str, Any], *, overwrite: bool = False) -> None:
    path = require_output_path(path)
    if path.exists() and not overwrite:
        raise RuntimeError(f"refusing to overwrite resolution-study output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def derive_seed(global_seed: int, role: str, size: int) -> dict[str, Any]:
    text = f"{int(global_seed)}:skyrmion:galerkin_resolution:v1:{role}:{int(size)}"
    digest = hashlib.sha256(text.encode()).hexdigest()
    return {"role": role, "size": int(size), "text": text, "sha256": digest,
            "seed": int(digest[:16], 16) % (2**31 - 1)}


def _v1_paths() -> dict[str, Path]:
    return {
        "OFFICIAL_GALERKIN_PARETO_PROTOCOL.md": V1_PROTOCOL_MD,
        "OFFICIAL_GALERKIN_PARETO_EVALUATION.md": V1_EVALUATION_MD,
        "outputs/official_galerkin_pareto/protocol.json": V1_OUTPUT_ROOT / "protocol.json",
        "outputs/official_galerkin_pareto/final_summary.json": V1_OUTPUT_ROOT / "final_summary.json",
    }


def verify_v1_immutable() -> dict[str, Any]:
    actual = {name: file_sha256(path) for name, path in _v1_paths().items()}
    return {"passed": actual == V1_IMMUTABLE_HASHES, "expected": V1_IMMUTABLE_HASHES,
            "actual": actual}


def _source_hashes() -> dict[str, str]:
    names = (
        "resolution_study.py", "resolution_study_run.py", "resolution_study_report.py",
        "galerkin.py", "galerkin_only.py", "production_basis.py",
        "production_galerkin.py", "production_gradient.py", "config.json",
    )
    return {name: file_sha256(PACKAGE_ROOT / name) for name in names}


def protocol_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    if file_sha256(DICTIONARY_PATH) != EXPECTED_DICTIONARY_SHA256:
        raise RuntimeError("validated K=280 dictionary hash changed")
    v1 = verify_v1_immutable()
    if not v1["passed"]:
        raise RuntimeError("frozen official Pareto v1 changed")
    original_pareto = PACKAGE_ROOT.parent / "skyrmions_deep_ritz" / "outputs" / "pareto_authoritative" / "pareto.json"
    geometries = [{"id": name, "provenance": provenance, "eta": eta}
                  for name, provenance, eta in FIXED_GEOMETRIES]
    return {
        "schema_version": 1, "version": VERSION,
        "purpose": "selection_development_only Galerkin quadrature and basis/rank qualification",
        "no_eta_optimization": True, "validation_access_permitted": False,
        "v1_immutable": v1,
        "fixed_geometries": geometries,
        "geometry_source_hashes": {
            "config.json": file_sha256(PACKAGE_ROOT / "config.json"),
            "original_pareto.json": file_sha256(original_pareto),
            "prior_selection_result.json": file_sha256(PACKAGE_ROOT / "outputs" / "galerkin_only_3pct" / "selection" / "result.json"),
        },
        "primary": {
            "K": K_PRIMARY, "dictionary_sha256": EXPECTED_DICTIONARY_SHA256,
            "normalization": "unchanged eta-independent original selection-train normalization",
            "rank_tolerance": float(cfg["production_galerkin"]["relative_rank_tolerance"]),
            "certificate_thresholds": cfg["production_galerkin"]["certificate_thresholds"],
            "algebra_thresholds": {key: cfg["production_galerkin"][key] for key in (
                "maximum_range_residual", "maximum_stationarity_residual",
                "maximum_identity_relerr", "maximum_symmetry_residual",
                "minimum_rank_fraction", "maximum_retained_condition",
            )},
            "support_ladder": [list(row) for row in SUPPORT_LADDER],
            "streamed_chunk_size": int(cfg["production_galerkin"]["chunk_size"]),
            "full_basis_cache_prohibited": True,
        },
        "development_banks": {
            "label": "selection_development_only",
            "maximum_train_samples": 32768, "maximum_audit_samples": 16384,
            "nested_exact_prefixes": True,
            "initial_distribution": "fresh independent draws from frozen truth initial distribution",
            "reference_checkpoint_sha256": file_sha256(ARTIFACT_DIR / "reference.npz"),
            "reference_retrained": False,
            "seeds": [derive_seed(cfg["seed"], "train", 32768), derive_seed(cfg["seed"], "audit", 16384)],
            "exact_overlap_checks": "fresh train/audit and permitted historical selection banks only; validation arrays forbidden",
        },
        "comparison_metrics": {
            "action_relative_difference": "abs(A_high-A_low)/max(abs(A_high),1e-12)",
            "gradient_cosine": "dot(g_high,g_low)/(norm(g_high)*norm(g_low))",
            "gradient_relative_difference": "norm(g_high-g_low)/max(norm(g_high),1e-12)",
        },
        "primary_decision": {
            "A": "both low-risk anchors have final energy <=0.08, algebra valid, final action relchange<=0.03, gradient cosine>=0.995 and relchange<=0.05",
            "B": "meaningful full ladder completes but A is false",
            "C": "resource failure prevents meaningful full ladder",
            "safety_margin_energy": 0.075,
        },
        "conditional_basis_rank": {
            "run_only_if_primary_B": True, "K_ladder": list(K_LADDER),
            "rank_tolerances": list(RANK_TOLERANCES),
            "support": [32768, 16384], "energy_threshold_unchanged": 0.08,
            "neighbor_action_relative_tolerance": 0.05,
            "neighbor_gradient_cosine_minimum": 0.995,
            "neighbor_gradient_relative_tolerance": 0.10,
            "rank_tolerance_action_spread": 0.02,
            "rank_tolerance_energy_spread": 0.01,
            "rank_tolerance_gradient_cosine_minimum": 0.995,
            "selection_rule": "all fixed geometries pass hard gates; prefer smaller stable well-conditioned K with neighboring action/gradient stability and rank-tolerance robustness",
        },
        "candidate_v2_initialization": {
            "start_requires": ["exact_risk", "exact_geometry", "projection", "ESS", "forcing", "covariance", "Galerkin_algebra", "rank_range_stationarity"],
            "start_does_not_require": ["heldout_weak", "heldout_energy", "heldout_moment"],
            "official_endpoint_requires_complete_certificate": True,
        },
        "future_start_generator": {
            "interpolation_points_per_segment": 17,
            "local_count_per_center": 16, "local_scale": 0.01,
            "risk_tangent_direction_count": 16,
            "risk_tangent_radii": [0.0001, 0.0005, 0.001, 0.005],
            "global_count": 32, "global_oversample": 16,
            "seed": derive_seed(cfg["seed"], "future_starts", 1),
            "uses_selection_risk_only": True,
        },
        "source_hashes": _source_hashes(),
    }


def freeze_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    payload = protocol_payload(cfg)
    digest = payload_sha256(payload)
    result = {**payload, "protocol_sha256": digest, "protocol_frozen": True}
    if PROTOCOL_PATH.exists():
        if read_json(PROTOCOL_PATH) != result:
            raise RuntimeError("different resolution-study protocol already exists")
    else:
        write_json(PROTOCOL_PATH, result)
        PROTOCOL_HASH_PATH.write_text(digest + "\n", encoding="utf-8")
    return result


def require_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    if not PROTOCOL_PATH.is_file() or not PROTOCOL_HASH_PATH.is_file():
        raise RuntimeError("freeze-protocol must run first")
    result = read_json(PROTOCOL_PATH)
    digest = result.pop("protocol_sha256", None)
    frozen = result.pop("protocol_frozen", None)
    if not frozen or payload_sha256(result) != digest:
        raise RuntimeError("resolution-study protocol hash mismatch")
    if PROTOCOL_HASH_PATH.read_text().strip() != digest or protocol_payload(cfg) != result:
        raise RuntimeError("current state differs from frozen resolution-study protocol")
    return {**result, "protocol_sha256": digest, "protocol_frozen": True}


def _physics_config(cfg: dict[str, Any]):
    from .domain import SkyrmionConfig
    values = dict(cfg["physics"])
    values.pop("time_nodes", None)
    values.pop("truth_substeps", None)
    values["box"] = tuple(values["box"])
    values["pinning_centers"] = tuple(tuple(row) for row in values["pinning_centers"])
    return SkyrmionConfig(**values)


def _save_npz_once(path: Path, **arrays: Any) -> None:
    path = require_output_path(path)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite development bank: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in arrays.items()})


def _reference_bank(flow: Any, truth: SkyrmionTruth, times: jax.Array, *, seed: int,
                    samples: int, substeps: int, chunk_size: int = 2048):
    initial = truth.sample_initial(jax.random.PRNGKey(int(seed)), int(samples))
    configurations, velocities = [], []
    for start in range(0, int(samples), int(chunk_size)):
        stop = min(start + int(chunk_size), int(samples))
        rows = flow.rollout(initial[start:stop], times, substeps_per_interval=int(substeps))
        configurations.append(np.asarray(rows))
        velocities.append(np.asarray(flow.velocity(rows, times)))
    return (
        np.concatenate(configurations, axis=1), np.concatenate(velocities, axis=1),
        np.full((len(times), int(samples)), 1.0 / float(samples), dtype=np.float64),
    )


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
        if old.get("protocol_sha256") == protocol["protocol_sha256"] and old.get("passed"):
            for row in old["artifacts"]:
                if file_sha256(OUTPUT_ROOT / row["path"]) != row["sha256"]:
                    raise RuntimeError("development bank changed after sealing")
            return {**old, "cache_hit": True}
        raise RuntimeError("incompatible development bank manifest exists")
    times = jnp.linspace(0.0, 1.0, int(cfg["physics"]["time_nodes"]), dtype=jnp.float64)
    truth = SkyrmionTruth(_physics_config(cfg))
    flow = load_reference(ARTIFACT_DIR / "reference.npz")
    seed_rows = {row["role"]: row for row in protocol["development_banks"]["seeds"]}
    for role, size, path in (("train", 32768, TRAIN_BANK_PATH), ("audit", 16384, AUDIT_BANK_PATH)):
        configurations, velocity, weights = _reference_bank(
            flow, truth, times, seed=seed_rows[role]["seed"], samples=size,
            substeps=int(cfg["banks"]["reference_substeps"]),
        )
        _save_npz_once(path, times=times, configurations=configurations,
                       velocity=velocity, base_weights=weights,
                       selection_development_only=np.asarray(True))
    fresh = {"train": _row_hashes(_initial_rows(TRAIN_BANK_PATH)),
             "audit": _row_hashes(_initial_rows(AUDIT_BANK_PATH))}
    permitted = {}
    for name in ("projection", "ritz_train", "ritz_audit"):
        permitted[name] = _row_hashes(_initial_rows(ARTIFACT_DIR / f"reference_bank_{name}.npz"))
    checks = [{"left": "train", "right": "audit", "overlap": len(fresh["train"] & fresh["audit"])}]
    for left in ("train", "audit"):
        for right, hashes in permitted.items():
            checks.append({"left": left, "right": f"historical_selection_{right}",
                           "overlap": len(fresh[left] & hashes)})
    passed = all(row["overlap"] == 0 for row in checks)
    artifacts = []
    for path in (TRAIN_BANK_PATH, AUDIT_BANK_PATH):
        artifacts.append({"path": str(path.relative_to(OUTPUT_ROOT)),
                          "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    result = {
        "ran": True, "passed": passed, "cache_hit": False,
        "protocol_sha256": protocol["protocol_sha256"], "label": "selection_development_only",
        "nested_prefixes": {"train": [8192, 16384, 32768], "audit": [4096, 8192, 16384]},
        "seed_records": list(seed_rows.values()), "exact_overlap_checks": checks,
        "historical_validation_arrays_accessed": False,
        "historical_validation_disjointness_basis": "independent versioned seeds and fresh continuous initial-state draws; forbidden arrays were not opened",
        "artifacts": artifacts,
    }
    write_json(BANK_MANIFEST_PATH, result)
    if not passed:
        raise RuntimeError("development bank exact overlap check failed")
    return result


def load_development_bank(path: Path, samples: int) -> GalerkinReferenceBank:
    with np.load(path, allow_pickle=False) as arrays:
        if not bool(np.asarray(arrays["selection_development_only"]).item()):
            raise RuntimeError("bank is not labeled selection_development_only")
        configurations = jnp.asarray(arrays["configurations"][:, :samples], dtype=jnp.float64)
        velocity = jnp.asarray(arrays["velocity"][:, :samples], dtype=jnp.float64)
    weights = jnp.full(configurations.shape[:2], 1.0 / float(samples), dtype=jnp.float64)
    return GalerkinReferenceBank(configurations, velocity, weights)


def _algebra(cfg: dict[str, Any], system: GalerkinSystem, solve: Any,
             aggregate: dict[str, Any], K: int) -> dict[str, Any]:
    settings = cfg["production_galerkin"]
    retained_values = jnp.where(solve.retained, solve.eigenvalues, jnp.inf)
    smallest = jnp.min(retained_values, axis=-1)
    largest = jnp.max(solve.eigenvalues, axis=-1)
    payload = {
        "rank_by_time": np.asarray(solve.numerical_rank).tolist(),
        "minimum_rank_fraction": float(jnp.min(solve.numerical_rank / float(K))),
        "smallest_retained_eigenvalue_by_time": np.asarray(smallest).tolist(),
        "smallest_retained_eigenvalue": float(jnp.min(smallest)),
        "largest_eigenvalue_by_time": np.asarray(largest).tolist(),
        "largest_eigenvalue": float(jnp.max(largest)),
        "worst_retained_condition": float(jnp.max(solve.condition_number)),
        "worst_range_residual": float(jnp.max(solve.range_residual)),
        "worst_stationarity_residual": float(jnp.max(solve.stationarity_residual)),
        "worst_symmetry_residual": float(jnp.max(system.raw_symmetry_residual)),
        "identity_relerr": float(aggregate["identity_relerr"]),
    }
    payload["valid"] = bool(
        payload["minimum_rank_fraction"] >= float(settings["minimum_rank_fraction"])
        and payload["worst_retained_condition"] <= float(settings["maximum_retained_condition"])
        and payload["worst_range_residual"] <= float(settings["maximum_range_residual"])
        and payload["worst_stationarity_residual"] <= float(settings["maximum_stationarity_residual"])
        and payload["worst_symmetry_residual"] <= float(settings["maximum_symmetry_residual"])
        and payload["identity_relerr"] <= float(settings["maximum_identity_relerr"])
    )
    return payload


def _potential_rows(dictionary: Any, coefficients: jax.Array, bank: Any,
                    evaluators: list[Any], chunk_size: int) -> tuple[jax.Array, jax.Array]:
    potentials, kinetic = [], []
    for time_index in range(int(bank.configurations.shape[0])):
        p_chunks, k_chunks = [], []
        for start in range(0, int(bank.configurations.shape[1]), int(chunk_size)):
            stop = min(start + int(chunk_size), int(bank.configurations.shape[1]))
            values, gradients = evaluators[time_index](bank.configurations[time_index, start:stop])
            p_chunks.append(jnp.einsum("k,nk->n", coefficients[time_index], values))
            grad = jnp.einsum("k,nkpd->npd", coefficients[time_index], gradients)
            k_chunks.append(jnp.sum(grad * grad, axis=(-2, -1)))
        potentials.append(jnp.concatenate(p_chunks)); kinetic.append(jnp.concatenate(k_chunks))
    return jnp.stack(potentials), jnp.stack(kinetic)


def evaluate_case(cfg: dict[str, Any], selection_data: Any, dictionary: Any,
                  train: GalerkinReferenceBank, audit: GalerkinReferenceBank,
                  eta: Any, *, K: int, rank_tolerance: float,
                  evaluators: list[Any] | None = None) -> dict[str, Any]:
    problem = selection_data.selection_problem
    eta = wrap_periodic(jnp.asarray(eta, dtype=jnp.float64), problem.family)
    reconstruction = reconstruct_moments(eta, problem)
    train_state = forcing_state(eta, problem, train, reconstruction)
    audit_state = forcing_state(eta, problem, audit, reconstruction)
    prefix = prefix_dictionary(dictionary, K)
    evaluators = evaluators or make_basis_evaluators(prefix, int(train.configurations.shape[0]))
    system = assemble_hybrid_system(
        prefix, train, train_state.projection.weights, train_state.forcing,
        chunk_size=int(cfg["production_galerkin"]["chunk_size"]), evaluators=evaluators,
    )
    solve = rank_aware_quadratic_solve(system.gram, system.load,
                                       relative_rank_tolerance=float(rank_tolerance))
    aggregate = aggregate_quadratic_values(solve, problem.time_weights)
    potential, kinetic = _potential_rows(prefix, solve.coefficients, train, evaluators,
                                         int(cfg["production_galerkin"]["chunk_size"]))
    adapter = SimpleNamespace(selection_problem=problem, ritz_train_bank=train)
    value, gradient = production_hybrid_envelope_value_and_grad(
        eta, solve.coefficients, adapter, potential, kinetic,
    )
    certificate_adapter = SimpleNamespace(selection_problem=problem, ritz_audit_bank=audit)
    certificate = audit_hybrid_solutions(
        prefix, solve.coefficients[None], certificate_adapter, eta, reconstruction,
        audit_state, GalerkinCertificateThresholds(**cfg["production_galerkin"]["certificate_thresholds"]),
        chunk_size=int(cfg["production_galerkin"]["chunk_size"]),
    )[0]
    algebra = _algebra(cfg, system, solve, aggregate, K)
    train_forcing, audit_forcing = _forcing_state_payload(train_state, problem), _forcing_state_payload(audit_state, problem)
    geometry = bool(problem.family.geometry_valid(eta))
    action = float(value)
    audit_action = float(certificate["action"])
    complete = bool(geometry and train_forcing["valid"] and audit_forcing["valid"]
                    and algebra["valid"] and certificate["valid"])
    return {
        "eta": np.asarray(eta).tolist(), "K": int(K), "rank_tolerance": float(rank_tolerance),
        "train_samples": int(train.configurations.shape[1]), "audit_samples": int(audit.configurations.shape[1]),
        "scientific_risk": float(selection_risk(eta, selection_data)),
        "train_action": action, "quadratic_train_action": float(aggregate["action"]),
        "audit_action": audit_action,
        "train_audit_action_relative_discrepancy": abs(audit_action - action) / max(abs(audit_action), 1e-12),
        "gradient": np.asarray(gradient).tolist(), "gradient_norm": float(jnp.linalg.norm(gradient)),
        "gradient_finite": bool(jnp.all(jnp.isfinite(gradient))),
        "train_forcing": train_forcing, "audit_forcing": audit_forcing,
        "geometry_valid": geometry, "algebra": algebra, "heldout_certificate": certificate,
        "complete_certificate": complete,
    }


def _case_signature(protocol: dict[str, Any], geometry: dict[str, Any], train: int,
                    audit: int, K: int, tol: float, bank_manifest: dict[str, Any]) -> str:
    return fingerprint({"kind": "resolution_case_v1", "protocol": protocol["protocol_sha256"],
                        "geometry": geometry, "train": train, "audit": audit, "K": K,
                        "rank_tolerance": tol, "bank_hashes": bank_manifest["artifacts"]})


def run_quadrature(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol, banks = require_protocol(cfg), read_json(BANK_MANIFEST_PATH)
    if not banks.get("passed"):
        raise RuntimeError("development banks are not qualified")
    final_path = OUTPUT_ROOT / "quadrature" / "K280" / "result.json"
    if final_path.is_file():
        old = read_json(final_path)
        if old.get("protocol_sha256") == protocol["protocol_sha256"] and old.get("passed"):
            return {**old, "cache_hit": True}
        raise RuntimeError("incompatible completed quadrature result exists")
    selection_data = load_selection_galerkin_data(cfg, ARTIFACT_DIR)
    dictionary = load_dictionary(DICTIONARY_PATH, box=tuple(cfg["physics"]["box"]))
    results = []
    for geometry in protocol["fixed_geometries"]:
        geometry_rows = []
        for train_size, audit_size in SUPPORT_LADDER:
            output = OUTPUT_ROOT / "quadrature" / "K280" / geometry["id"] / f"train_{train_size}_audit_{audit_size}.json"
            signature = _case_signature(protocol, geometry, train_size, audit_size, K_PRIMARY,
                                        protocol["primary"]["rank_tolerance"], banks)
            if output.is_file():
                row = read_json(output)
                if row.get("signature") != signature:
                    raise RuntimeError("quadrature case signature mismatch")
            else:
                train = load_development_bank(TRAIN_BANK_PATH, train_size)
                audit = load_development_bank(AUDIT_BANK_PATH, audit_size)
                row = evaluate_case(cfg, selection_data, dictionary, train, audit, geometry["eta"],
                                    K=K_PRIMARY, rank_tolerance=protocol["primary"]["rank_tolerance"])
                row.update({"signature": signature, "geometry_id": geometry["id"],
                            "selection_development_only": True})
                write_json(output, row)
            geometry_rows.append(row)
        results.append({"geometry_id": geometry["id"], "rows": geometry_rows})
    result = {"ran": True, "passed": True, "protocol_sha256": protocol["protocol_sha256"],
              "geometries": results, "validation_accessed": False, "eta_optimization_run": False}
    write_json(final_path, result)
    return result


def relative_change(high: float, low: float) -> float:
    return abs(float(high) - float(low)) / max(abs(float(high)), 1e-12)


def gradient_comparison(high: Iterable[float], low: Iterable[float]) -> dict[str, float]:
    high, low = np.asarray(high, dtype=float), np.asarray(low, dtype=float)
    hn, ln = np.linalg.norm(high), np.linalg.norm(low)
    return {"cosine": float(np.dot(high, low) / max(hn * ln, 1e-30)),
            "relative_difference": float(np.linalg.norm(high - low) / max(hn, 1e-12))}


def analyze_quadrature(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_protocol(cfg)
    output_path = OUTPUT_ROOT / "quadrature" / "analysis.json"
    if output_path.is_file():
        old = read_json(output_path)
        if old.get("protocol_sha256") == protocol["protocol_sha256"] and old.get("passed"):
            return {**old, "cache_hit": True}
        raise RuntimeError("incompatible quadrature analysis exists")
    raw = read_json(OUTPUT_ROOT / "quadrature" / "K280" / "result.json")
    analyses = []
    for geometry in raw["geometries"]:
        rows, comparisons = geometry["rows"], []
        for low, high in zip(rows[:-1], rows[1:]):
            comparisons.append({
                "low_support": [low["train_samples"], low["audit_samples"]],
                "high_support": [high["train_samples"], high["audit_samples"]],
                "train_action_relative_change": relative_change(high["train_action"], low["train_action"]),
                "audit_action_relative_change": relative_change(high["audit_action"], low["audit_action"]),
                **gradient_comparison(high["gradient"], low["gradient"]),
                "energy_change": high["heldout_certificate"]["maximum_energy_residual"] - low["heldout_certificate"]["maximum_energy_residual"],
            })
        analyses.append({"geometry_id": geometry["geometry_id"], "rows": rows,
                         "consecutive_comparisons": comparisons})
    by_id = {row["geometry_id"]: row for row in analyses}
    low = [by_id["law"], by_id["historical_0p5"]]
    final_checks = []
    for item in low:
        final, comparison = item["rows"][-1], item["consecutive_comparisons"][-1]
        final_checks.append(bool(
            final["heldout_certificate"]["maximum_energy_residual"] <= 0.08
            and final["algebra"]["valid"] and comparison["train_action_relative_change"] <= 0.03
            and comparison["cosine"] >= 0.995 and comparison["relative_difference"] <= 0.05
        ))
    classification = (
        "A. K280 QUADRATURE SUPPORT WAS THE MAIN PROBLEM"
        if all(final_checks) else "B. K280 REMAINS PHYSICALLY INADEQUATE AT LOW RISK"
    )
    result = {"ran": True, "passed": True, "classification": classification,
              "conditional_basis_rank_required": classification.startswith("B."),
              "analyses": analyses, "energy_threshold": 0.08,
              "protocol_sha256": protocol["protocol_sha256"],
              "validation_accessed": False, "eta_optimization_run": False}
    write_json(output_path, result)
    return result


def _prefix_system(system: GalerkinSystem, K: int) -> GalerkinSystem:
    gram = system.gram[:, :K, :K]
    symmetry = jax.vmap(lambda matrix: jnp.linalg.norm(matrix - matrix.T) /
                        jnp.maximum(jnp.linalg.norm(matrix), 1e-30))(gram)
    empty = jnp.zeros((0,), dtype=jnp.float64)
    return GalerkinSystem(
        gram=gram, load=system.load[:, :K], basis_means=system.basis_means[:, :K],
        centered_basis=empty, weights=empty, forcing=empty,
        raw_symmetry_residual=symmetry, forcing_mean=system.forcing_mean,
    )


def _batch_potential_rows(dictionary: Any, padded_coefficients: jax.Array, bank: Any,
                          evaluators: list[Any], chunk_size: int) -> tuple[jax.Array, jax.Array]:
    potentials, kinetic = [], []
    for time_index in range(int(bank.configurations.shape[0])):
        p_chunks, k_chunks = [], []
        coefficients = padded_coefficients[:, time_index]
        for start in range(0, int(bank.configurations.shape[1]), int(chunk_size)):
            stop = min(start + int(chunk_size), int(bank.configurations.shape[1]))
            values, gradients = evaluators[time_index](bank.configurations[time_index, start:stop])
            p_chunks.append(jnp.einsum("lk,nk->ln", coefficients, values))
            grad = jnp.einsum("lk,nkpd->lnpd", coefficients, gradients)
            k_chunks.append(jnp.sum(grad * grad, axis=(-2, -1)))
        potentials.append(jnp.concatenate(p_chunks, axis=1))
        kinetic.append(jnp.concatenate(k_chunks, axis=1))
    return jnp.stack(potentials, axis=1), jnp.stack(kinetic, axis=1)


def run_basis_rank(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_protocol(cfg)
    primary = read_json(OUTPUT_ROOT / "quadrature" / "analysis.json")
    if not primary.get("conditional_basis_rank_required"):
        raise RuntimeError("basis-rank phase is permitted only after primary classification B")
    output_path = OUTPUT_ROOT / "basis_rank" / "result.json"
    signature = fingerprint({"kind": "basis_rank_v1", "protocol": protocol["protocol_sha256"],
                             "banks": file_sha256(BANK_MANIFEST_PATH)})
    if output_path.is_file():
        old = read_json(output_path)
        if old.get("signature") == signature:
            return {**old, "cache_hit": True}
        raise RuntimeError("basis-rank signature mismatch")
    selection_data = load_selection_galerkin_data(cfg, ARTIFACT_DIR)
    problem = selection_data.selection_problem
    dictionary = load_dictionary(DICTIONARY_PATH, box=tuple(cfg["physics"]["box"]))
    train = load_development_bank(TRAIN_BANK_PATH, 32768)
    audit = load_development_bank(AUDIT_BANK_PATH, 16384)
    evaluators = make_basis_evaluators(dictionary, int(train.configurations.shape[0]))
    results = []
    for geometry in protocol["fixed_geometries"]:
        eta = wrap_periodic(jnp.asarray(geometry["eta"], dtype=jnp.float64), problem.family)
        reconstruction = reconstruct_moments(eta, problem)
        train_state = forcing_state(eta, problem, train, reconstruction)
        audit_state = forcing_state(eta, problem, audit, reconstruction)
        full_system = assemble_hybrid_system(
            dictionary, train, train_state.projection.weights, train_state.forcing,
            chunk_size=int(cfg["production_galerkin"]["chunk_size"]), evaluators=evaluators,
        )
        solves, metadata, aggregates, algebra_rows, padded = [], [], [], [], []
        for K in K_LADDER:
            system = _prefix_system(full_system, K)
            for tolerance in RANK_TOLERANCES:
                solve = rank_aware_quadratic_solve(system.gram, system.load,
                                                   relative_rank_tolerance=tolerance)
                aggregate = aggregate_quadratic_values(solve, problem.time_weights)
                solves.append(solve); aggregates.append(aggregate)
                metadata.append((K, tolerance, system))
                algebra_rows.append(_algebra(cfg, system, solve, aggregate, K))
                padded.append(jnp.pad(solve.coefficients, ((0, 0), (0, K_PRIMARY-K))))
        padded_array = jnp.stack(padded)
        certificate_adapter = SimpleNamespace(selection_problem=problem, ritz_audit_bank=audit)
        certificates = audit_hybrid_solutions(
            dictionary, padded_array, certificate_adapter, eta, reconstruction, audit_state,
            GalerkinCertificateThresholds(**cfg["production_galerkin"]["certificate_thresholds"]),
            chunk_size=int(cfg["production_galerkin"]["chunk_size"]),
        )
        potentials, kinetic = _batch_potential_rows(
            dictionary, padded_array, train, evaluators,
            int(cfg["production_galerkin"]["chunk_size"]),
        )
        adapter = SimpleNamespace(selection_problem=problem, ritz_train_bank=train)
        train_forcing, audit_forcing = _forcing_state_payload(train_state, problem), _forcing_state_payload(audit_state, problem)
        rows = []
        for index, ((K, tolerance, _), solve, aggregate, algebra, certificate) in enumerate(
            zip(metadata, solves, aggregates, algebra_rows, certificates, strict=True)
        ):
            value, gradient = production_hybrid_envelope_value_and_grad(
                eta, solve.coefficients, adapter, potentials[index], kinetic[index]
            )
            complete = bool(problem.family.geometry_valid(eta) and train_forcing["valid"]
                            and audit_forcing["valid"] and algebra["valid"] and certificate["valid"])
            rows.append({
                "K": K, "rank_tolerance": tolerance, "scientific_risk": float(selection_risk(eta, selection_data)),
                "train_action": float(value), "quadratic_train_action": float(aggregate["action"]),
                "audit_action": float(certificate["action"]),
                "train_audit_action_relative_discrepancy": abs(float(certificate["action"])-float(value))/max(abs(float(certificate["action"])),1e-12),
                "gradient": np.asarray(gradient).tolist(), "gradient_norm": float(jnp.linalg.norm(gradient)),
                "gradient_finite": bool(jnp.all(jnp.isfinite(gradient))),
                "train_forcing": train_forcing, "audit_forcing": audit_forcing,
                "geometry_valid": bool(problem.family.geometry_valid(eta)), "algebra": algebra,
                "heldout_certificate": certificate, "complete_certificate": complete,
            })
        results.append({"geometry_id": geometry["id"], "eta": geometry["eta"], "rows": rows})
        write_json(OUTPUT_ROOT / "basis_rank" / f"{geometry['id']}.json", results[-1])
    qualification = _qualify_basis_rank(protocol, results)
    result = {"ran": True, "passed": True, "cache_hit": False, "signature": signature,
              "support": [32768, 16384], "geometries": results, **qualification,
              "validation_accessed": False, "eta_optimization_run": False}
    write_json(output_path, result)
    return result


def _qualify_basis_rank(protocol: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    settings = protocol["conditional_basis_rank"]
    candidates = []
    for K in K_LADDER:
        by_geometry = []
        for geometry in results:
            tolerance_rows = [row for row in geometry["rows"] if row["K"] == K]
            default = next(row for row in tolerance_rows if row["rank_tolerance"] == 1e-12)
            actions = [row["train_action"] for row in tolerance_rows]
            energies = [row["heldout_certificate"]["maximum_energy_residual"] for row in tolerance_rows]
            cosines = []
            for left, right in zip(tolerance_rows[:-1], tolerance_rows[1:]):
                cosines.append(gradient_comparison(right["gradient"], left["gradient"])["cosine"])
            robust_tol = bool(
                all(row["complete_certificate"] for row in tolerance_rows)
                and (max(actions)-min(actions))/max(abs(default["train_action"]),1e-12) <= settings["rank_tolerance_action_spread"]
                and max(energies)-min(energies) <= settings["rank_tolerance_energy_spread"]
                and min(cosines, default=1.0) >= settings["rank_tolerance_gradient_cosine_minimum"]
            )
            neighbor = None
            if K != K_LADDER[-1]:
                next_K = K_LADDER[K_LADDER.index(K)+1]
                next_row = next(row for row in geometry["rows"] if row["K"] == next_K and row["rank_tolerance"] == 1e-12)
                comp = gradient_comparison(next_row["gradient"], default["gradient"])
                neighbor = {"action_relative_change": relative_change(next_row["train_action"], default["train_action"]), **comp}
            stable_neighbor = bool(neighbor is None or (
                neighbor["action_relative_change"] <= settings["neighbor_action_relative_tolerance"]
                and neighbor["cosine"] >= settings["neighbor_gradient_cosine_minimum"]
                and neighbor["relative_difference"] <= settings["neighbor_gradient_relative_tolerance"]
            ))
            by_geometry.append({"geometry_id": geometry["geometry_id"], "robust_to_rank_tolerance": robust_tol,
                                "neighbor": neighbor, "stable_neighbor": stable_neighbor,
                                "default_complete_certificate": default["complete_certificate"]})
        qualified = all(row["robust_to_rank_tolerance"] and row["stable_neighbor"]
                        and row["default_complete_certificate"] for row in by_geometry)
        candidates.append({"K": K, "qualified": qualified, "geometry_checks": by_geometry})
    chosen = next((row["K"] for row in candidates if row["qualified"]), None)
    return {"qualification_candidates": candidates, "recommended_K": chosen,
            "recommended_rank_tolerance": 1e-12 if chosen is not None else None,
            "future_discretization_qualified": chosen is not None}


def initialization_gate(payload: dict[str, Any]) -> bool:
    return bool(payload["exact_risk_valid"] and payload["geometry_valid"]
                and payload["train_forcing_valid"] and payload["algebra_valid"])


def endpoint_gate(payload: dict[str, Any]) -> bool:
    return bool(initialization_gate(payload) and payload["complete_heldout_certificate"])


def run_start_generator_diagnostic(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_protocol(cfg)
    output_path = OUTPUT_ROOT / "start_generator_diagnostics" / "result.json"
    if output_path.is_file():
        old = read_json(output_path)
        if old.get("protocol_sha256") == protocol["protocol_sha256"]:
            return {**old, "cache_hit": True}
        raise RuntimeError("incompatible start-generator diagnostic exists")
    data = load_selection_galerkin_data(cfg, ARTIFACT_DIR)
    family = data.selection_problem.family
    law = jnp.asarray(protocol["fixed_geometries"][0]["eta"], dtype=jnp.float64)
    historical = jnp.asarray([row["eta"] for row in protocol["fixed_geometries"][1:]], dtype=jnp.float64)
    settings = protocol["future_start_generator"]
    seed = int(settings["seed"]["seed"])
    alphas = jnp.linspace(0.0, 1.0, int(settings["interpolation_points_per_segment"]), dtype=jnp.float64)
    interpolated = jnp.concatenate([
        jax.vmap(lambda alpha: wrap_periodic(law + alpha * (center - law), family))(alphas)
        for center in historical
    ])
    local = local_sensor_designs(
        jax.random.PRNGKey(seed), jnp.concatenate([law[None], historical]),
        count_per_center=int(settings["local_count_per_center"]),
        scale=float(settings["local_scale"]), family=family,
    )
    _, risk_gradient = jax.value_and_grad(lambda eta: selection_risk(eta, data))(law)
    tangent = []
    key = jax.random.PRNGKey(seed + 1)
    for index in range(int(settings["risk_tangent_direction_count"])):
        direction = jax.random.normal(jax.random.fold_in(key, index), law.shape, dtype=jnp.float64)
        direction = direction - jnp.dot(direction, risk_gradient) / jnp.maximum(jnp.dot(risk_gradient, risk_gradient), 1e-30) * risk_gradient
        direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1e-30)
        for radius in settings["risk_tangent_radii"]:
            tangent.extend([wrap_periodic(law + radius * direction, family), wrap_periodic(law - radius * direction, family)])
    global_rows = random_sensor_designs(jax.random.PRNGKey(seed + 2), count=int(settings["global_count"]),
                                        family=family, oversample=int(settings["global_oversample"]))
    pool = np.asarray(jnp.concatenate([law[None], historical, interpolated, local,
                                      jnp.asarray(tangent), global_rows]))
    unique = []
    for eta in pool:
        if bool(family.geometry_valid(jnp.asarray(eta))) and not any(np.linalg.norm(eta-old) <= 1e-12 for old in unique):
            unique.append(eta)
    risks = np.asarray([float(selection_risk(jnp.asarray(eta), data)) for eta in unique])
    law_risk = float(selection_risk(law, data))
    rows = []
    for allowance in (0.5, 1, 2, 3, 4, 5):
        mask = risks <= (1 + allowance / 100.0) * law_risk
        feasible = np.asarray(unique)[mask]
        distances = []
        for left in range(len(feasible)):
            for right in range(left + 1, len(feasible)):
                distances.append(float(np.linalg.norm(feasible[left] - feasible[right])))
        rows.append({"allowance_percent": allowance, "feasible_count": int(np.sum(mask)),
                     "pool_count": len(unique), "risk_ceiling": (1 + allowance/100) * law_risk,
                     "minimum_pairwise_eta_distance": min(distances) if distances else None,
                     "median_pairwise_eta_distance": float(np.median(distances)) if distances else None})
    v1_start_payloads = []
    for item in ("law", "historical_0p5"):
        v1_row = next(row for row in read_json(V1_OUTPUT_ROOT / "selection" / "starts.json")["allowances"]["0p5"]["starts"] if row["id"] in (item, "historical_0p5pct"))
        cert_file = next((V1_OUTPUT_ROOT / "selection" / "allowance_0p5").glob(f"*{v1_row['id']}*.json"))
        cert = read_json(cert_file)["start_certificate"]
        gate_payload = {"exact_risk_valid": True, "geometry_valid": cert["geometry_valid"],
                        "train_forcing_valid": cert["train_forcing_audit"]["valid"] and cert["audit_forcing_audit"]["valid"],
                        "algebra_valid": cert["algebra_valid"],
                        "complete_heldout_certificate": cert["heldout_certificate"]["valid"]}
        v1_start_payloads.append({"id": item, "candidate_v2_start_gate": initialization_gate(gate_payload),
                                  "official_endpoint_gate": endpoint_gate(gate_payload)})
    result = {"ran": True, "passed": all(row["feasible_count"] >= 10 for row in rows),
              "protocol_sha256": protocol["protocol_sha256"],
              "selection_risk_only": True, "validation_accessed": False,
              "eta_optimization_run": False, "feasibility": rows,
              "candidate_v2_gate_assessment": v1_start_payloads,
              "logical_assessment": "safe as an initialization rule because no uncertified point can become an incumbent or endpoint; it permits v1 starts to enter optimization but cannot guarantee a certified endpoint"}
    write_json(output_path, result)
    return result


__all__ = [
    "AUDIT_BANK_PATH", "BANK_MANIFEST_PATH", "DICTIONARY_PATH", "EXPECTED_DICTIONARY_SHA256",
    "FIXED_GEOMETRIES", "K_LADDER", "K_PRIMARY", "OUTPUT_ROOT", "PROTOCOL_PATH",
    "RANK_TOLERANCES", "REPORT_PATH", "SUPPORT_LADDER", "TRAIN_BANK_PATH",
    "V1_IMMUTABLE_HASHES", "analyze_quadrature", "derive_seed", "endpoint_gate",
    "evaluate_case", "freeze_protocol", "generate_banks", "gradient_comparison",
    "initialization_gate", "load_development_bank", "payload_sha256",
    "relative_change", "require_output_path", "require_protocol",
    "run_basis_rank", "run_quadrature", "run_start_generator_diagnostic",
    "verify_v1_immutable",
]
