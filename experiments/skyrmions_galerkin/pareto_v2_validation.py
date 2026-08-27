"""Fresh validation, callable only after the complete Pareto-v2 selection seal."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mfsi.moments import AnchoredCubicSplineConfig, AnchoredCubicSplineReconstructor
from mfsi.projection import IProjectionConfig

from .domain import SkyrmionTruth
from .forcing import ForcingConfig
from .full_gradient import FrozenEtaProblem
from .galerkin_only_data import GalerkinReferenceBank, ValidationGalerkinData
from .measurements import LocalDensitySensors
from .official_pareto_validation import _evaluate, _rollout_reference_bank
from .pareto_v2_common import (
    ALLOWANCES, ARTIFACT_DIR, DICTIONARY_PATH, K, OUTPUT_ROOT, VALIDATION_SIZES,
    atomic_json, eta_key, hashes, payload_sha256, read_json, require_protocol,
    validation_ceiling,
)
from .pareto_v2_selection import _tangent_audit
from .production_artifacts import file_sha256
from .production_basis import load_dictionary
from .risk import many_body_features, whitening_from_truth


FRESH = OUTPUT_ROOT / "fresh_validation"
TRUTH = FRESH / "truth.npz"
FIT = FRESH / "reference_fit.npz"
AUDIT = FRESH / "reference_audit.npz"
NOISE = FRESH / "measurement_noise.npz"
MANIFEST = FRESH / "artifact_manifest.json"


def _selection_seal() -> tuple[dict[str, Any], dict[str, Any]]:
    selection_path = OUTPUT_ROOT / "selection" / "pareto_selection.json"
    manifest_path = OUTPUT_ROOT / "selection" / "selection_manifest.json"
    if not selection_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("fresh validation is forbidden before complete selection freeze")
    selection, manifest = read_json(selection_path), read_json(manifest_path)
    if not selection.get("selection_frozen") or selection.get("validation_accessed"):
        raise RuntimeError("selection is not sealed against validation")
    if file_sha256(selection_path) != manifest.get("pareto_selection_sha256"):
        raise RuntimeError("selection hash mismatch")
    if payload_sha256(selection["winners"]) != manifest.get("winner_geometry_hash"):
        raise RuntimeError("winner geometry seal mismatch")
    return selection, manifest


def _save(path: Path, **arrays: Any) -> None:
    if path.exists(): raise RuntimeError(f"refusing to overwrite fresh validation artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent); os.close(fd)
    try:
        np.savez(temporary, **{key: np.asarray(value) for key, value in arrays.items()})
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _physics_config(cfg: dict[str, Any]):
    from .domain import SkyrmionConfig
    values = dict(cfg["physics"]); values.pop("time_nodes", None); values.pop("truth_substeps", None)
    values["box"] = tuple(values["box"]); values["pinning_centers"] = tuple(tuple(row) for row in values["pinning_centers"])
    return SkyrmionConfig(**values)


def _seed(protocol: dict[str, Any], label: str) -> dict[str, Any]:
    return next(row for row in protocol["validation"]["seed_records"] if row["label"] == label)


def generate_fresh_validation(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_protocol(cfg); selection, seal = _selection_seal()
    if MANIFEST.exists():
        previous = read_json(MANIFEST)
        if previous["selection_sha256"] != seal["pareto_selection_sha256"]: raise RuntimeError("fresh manifest selection mismatch")
        for row in previous["artifacts"]:
            if file_sha256(OUTPUT_ROOT / row["path"]) != row["sha256"]: raise RuntimeError("fresh artifact hash mismatch")
        return {**previous, "cache_hit": True}
    atomic_json(FRESH / "generation_seal.json", {"selection_sha256": seal["pareto_selection_sha256"],
        "winner_geometry_hash": seal["winner_geometry_hash"], "protocol_sha256": protocol["protocol_sha256"],
        "all_selection_winners_frozen_before_generation": True}, immutable=True)
    times = jnp.linspace(0, 1, int(cfg["physics"]["time_nodes"]), dtype=jnp.float64)
    truth_model = SkyrmionTruth(_physics_config(cfg)); truth_seed = _seed(protocol, "truth")
    truth = truth_model.make_bank(seed=truth_seed["seed"], samples=VALIDATION_SIZES["truth"], times=times,
                                  substeps_per_interval=int(cfg["physics"]["truth_substeps"]))
    _save(TRUTH, times=times, configurations=truth.configurations, seed=np.asarray(truth_seed["seed"]))
    family = LocalDensitySensors(int(cfg["measurement"]["n_sensors"]), float(cfg["measurement"]["sensor_width"]),
                                 tuple(cfg["physics"]["box"]), float(cfg["measurement"]["min_separation"]))
    noise_seed = _seed(protocol, "measurement_noise")
    noise = float(cfg["measurement"]["observation_noise_std"]) * jax.random.normal(
        jax.random.PRNGKey(noise_seed["seed"]),
        (int(cfg["measurement"]["acquisition_count"]), family.n_sensors), dtype=jnp.float64)
    _save(NOISE, detector_noise=noise, seed=np.asarray(noise_seed["seed"]))
    from .reference import load_reference
    flow = load_reference(ARTIFACT_DIR / "reference.npz")
    for label, path in (("reference_fit", FIT), ("reference_audit", AUDIT)):
        record = _seed(protocol, label)
        x, v, w = _rollout_reference_bank(flow, truth_model, times, record["seed"], VALIDATION_SIZES[label],
                                           int(cfg["banks"]["reference_substeps"]))
        _save(path, configurations=x, velocity=v, base_weights=w, seed=np.asarray(record["seed"]))
    fresh_hashes = {}
    for label, path in (("truth", TRUTH), ("reference_fit", FIT), ("reference_audit", AUDIT)):
        with np.load(path, allow_pickle=False) as values:
            rows = values["configurations"]
            fresh_hashes[label] = hashlib.sha256(np.ascontiguousarray(rows[0]).tobytes()).hexdigest()
    selection_hashes = read_json(OUTPUT_ROOT / "banks" / "manifest.json")["initial_state_hashes"]
    disjoint = len(set(fresh_hashes.values())) == len(fresh_hashes) and not (set(fresh_hashes.values()) & set(selection_hashes.values()))
    result = {"schema_version": 2, "passed": disjoint, "protocol_sha256": protocol["protocol_sha256"],
        "selection_sha256": seal["pareto_selection_sha256"], "selection_geometry_hash": seal["winner_geometry_hash"],
        "generated_after_selection_freeze": True, "seed_records": protocol["validation"]["seed_records"],
        "fresh_initial_hashes": fresh_hashes, "selection_initial_hashes": selection_hashes,
        "selection_validation_disjoint": disjoint, "artifacts": hashes((TRUTH, FIT, AUDIT, NOISE))}
    atomic_json(MANIFEST, result, immutable=True)
    if not disjoint: raise RuntimeError("fresh validation overlaps selection")
    return result


def _load_bank(path: Path) -> GalerkinReferenceBank:
    with np.load(path, allow_pickle=False) as values:
        return GalerkinReferenceBank(jnp.asarray(values["configurations"]), jnp.asarray(values["velocity"]),
                                     jnp.asarray(values["base_weights"]))


def _time_weights(times):
    delta = jnp.diff(times); values = jnp.concatenate((delta[:1] / 2, (delta[:-1] + delta[1:]) / 2, delta[-1:] / 2))
    return values / jnp.sum(values)


def _data(cfg: dict[str, Any]) -> ValidationGalerkinData:
    with np.load(TRUTH, allow_pickle=False) as values:
        times = jnp.asarray(values["times"]); truth = jnp.asarray(values["configurations"])
    with np.load(NOISE, allow_pickle=False) as values: noise = jnp.asarray(values["detector_noise"])
    acquisition_count = int(cfg["measurement"]["acquisition_count"])
    acquisition = jnp.asarray(tuple(round(i * (len(times) - 1) / (acquisition_count - 1)) for i in range(acquisition_count)), dtype=jnp.int32)
    family = LocalDensitySensors(int(cfg["measurement"]["n_sensors"]), float(cfg["measurement"]["sensor_width"]),
                                 tuple(cfg["physics"]["box"]), float(cfg["measurement"]["min_separation"]))
    projection_values = dict(cfg["projection"]); backend = projection_values.pop("trajectory_backend", "jax")
    allowed = {field.name for field in fields(IProjectionConfig)}
    projection = IProjectionConfig(**{key: value for key, value in projection_values.items() if key in allowed})
    forcing_allowed = {field.name for field in fields(ForcingConfig)}
    forcing = ForcingConfig(**{key: value for key, value in cfg["forcing"].items() if key in forcing_allowed})
    reconstructor = AnchoredCubicSplineReconstructor(np.asarray(times[acquisition]), np.asarray(times),
        AnchoredCubicSplineConfig(**cfg["moment_reconstruction"]))
    problem = FrozenEtaProblem(truth, times, _time_weights(times), acquisition,
        min(int(cfg["measurement"]["finite_configurations"]), int(truth.shape[1])), noise,
        family, reconstructor, projection, forcing, str(backend), tuple(cfg["physics"]["box"]))
    fit, audit = _load_bank(FIT), _load_bank(AUDIT); box = tuple(cfg["physics"]["box"])
    with np.load(ARTIFACT_DIR / "truth_banks.npz", allow_pickle=False) as values:
        selection_truth = jnp.asarray(values["design"])
    return ValidationGalerkinData(problem, fit, audit, many_body_features(fit.configurations, box),
        jnp.mean(many_body_features(truth, box), axis=1), whitening_from_truth(many_body_features(selection_truth, box)))


def validate(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_protocol(cfg); selection, seal = _selection_seal(); manifest = read_json(MANIFEST)
    if not manifest.get("passed") or manifest["selection_sha256"] != seal["pareto_selection_sha256"]:
        raise RuntimeError("fresh validation is not sealed to selection")
    path = FRESH / "results.json"
    if path.exists():
        old = read_json(path)
        if old["selection_sha256"] != seal["pareto_selection_sha256"]: raise RuntimeError("validation result seal mismatch")
        return old
    data = _data(cfg); dictionary = load_dictionary(DICTIONARY_PATH, box=tuple(cfg["physics"]["box"]))
    if dictionary.size != K: raise RuntimeError("validation dictionary is not K=280")
    law = _evaluate(cfg, data, dictionary, cfg["envelope"]["law_eta"])
    adapter = type("TangentData", (), {"selection_problem": data.validation_problem,
        "train_bank": data.fit_bank, "audit_bank": data.audit_bank})()
    cache = {}; rows = []
    for winner in selection["winners"]:
        allowance = float(winner["allowance_percent"])
        for method in ("Law", "Tangent", "Full"):
            eta = winner[method] if method == "Law" else winner[method]["eta"]
            key = eta_key(eta)
            if key not in cache:
                full = _evaluate(cfg, data, dictionary, eta)
                tangent_fit = _tangent_audit(adapter, eta, train=True); tangent_audit = _tangent_audit(adapter, eta)
                cache[key] = {"eta": eta, "full": full, "tangent_fit": tangent_fit, "tangent_audit": tangent_audit}
            item = cache[key]; ratio = item["full"]["risk"] / law["risk"] - 1
            strict_pass = ratio <= allowance / 100
            declared_pass = item["full"]["risk"] <= validation_ceiling(law["risk"], allowance)
            numerical = bool(item["full"]["numerically_certified"] and item["tangent_audit"]["valid"])
            classification = "VALIDATION NUMERICAL FAILURE" if not numerical else ("PASS" if declared_pass else "VALIDATION RISK REVERSAL")
            rows.append({"allowance_percent": allowance, "selected_by": method, "eta": eta,
                "validation_risk": item["full"]["risk"], "validation_risk_increase": ratio,
                "strict_p_validation_pass": strict_pass, "p_plus_5pp_validation_pass": declared_pass,
                "tangent_action": item["tangent_fit"]["action"], "full_fit_action": item["full"]["validation_fit_action"],
                "full_audit_action": item["full"]["validation_audit_action"],
                "full_reduction_vs_law": (law["validation_fit_action"] - item["full"]["validation_fit_action"]) / law["validation_fit_action"],
                "action_standard_error": item["full"]["action_standard_error"], "classification": classification,
                "numerically_certified": numerical, "diagnostics": item})
    unchanged = payload_sha256(selection["winners"]) == seal["winner_geometry_hash"]
    result = {"schema_version": 2, "passed": unchanged, "protocol_sha256": protocol["protocol_sha256"],
        "selection_sha256": seal["pareto_selection_sha256"], "selection_geometry_unchanged": unchanged,
        "optimization_run": False, "deep_ritz_used": False, "law": law, "rows": rows,
        "unique_geometry_count": len(cache), "fresh_manifest_sha256": file_sha256(MANIFEST)}
    atomic_json(path, result, immutable=True)
    if not unchanged: raise RuntimeError("validation changed frozen geometry")
    return result

