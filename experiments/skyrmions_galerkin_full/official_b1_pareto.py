"""Fresh official single-reference B1 Galerkin Pareto production workflow."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

from .domain import SkyrmionTruth
from .galerkin_only_data import GalerkinReferenceBank, SelectionGalerkinData
from .pareto_v3_common import file_sha256, eta_key
from .pareto_v3_diagnostic import _symmetry_aware_distance
from .reference import load_reference
from .reference_seed_robustness import _ReferenceEvaluator, _array_sha256
from .risk import many_body_features, whitening_from_truth
from .single_reference_b1_preflight import (
    ACCEPTED_REFERENCE_PATH, OUTPUT_ROOT as DEV_ROOT, _family, _make_problem,
    _physics_config, canonicalize_eta, minimum_periodic_separation,
)


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
VERSION = "skyrmion_official_b1_galerkin_pareto_v1"
OUTPUT_ROOT = ROOT / "outputs" / "official_b1_galerkin_pareto_v1"
PROTOCOL_DOCUMENT = ROOT / "OFFICIAL_B1_GALERKIN_PARETO_V1_PROTOCOL.md"
PROTOCOL_PATH = OUTPUT_ROOT / "protocol.json"
PROTOCOL_HASH_PATH = OUTPUT_ROOT / "protocol_hash.txt"
CONFIG_PATH = ROOT / "config.json"
CONFIRMATION_ROOT = ROOT / "outputs" / "skyrmion_b1_final_support_confirmation_v1"
CHECKPOINT = DEV_ROOT / "reference_training" / "attempt_A" / "reference.npz"
CHECKPOINT_SHA256 = "1e13e2ea58df122702d4f555f8788a148b3150bbfbfc953cbac9f963c03d539b"
DICTIONARY_PATH = ROOT / "outputs" / "galerkin_only_3pct" / "cache" / "dictionaries" / "dictionary_K280.npz"
DICTIONARY_SHA256 = "37e9b60fcb92c4e5a0ee7ec1651fb7f8889f7ac6bdb02d3bd314e9ef40833326"
DESIGN_PATH = OUTPUT_ROOT / "design_truth" / "design_truth.npz"
DESIGN_RECORD = OUTPUT_ROOT / "design_truth" / "manifest.json"
LAW_PATH = OUTPUT_ROOT / "law" / "official_law.json"
LAW_POOL_PATH = OUTPUT_ROOT / "law" / "search_pool.json"
LAW_RESULTS_PATH = OUTPUT_ROOT / "law" / "search_results.json"
CANDIDATE_SPEC = OUTPUT_ROOT / "candidate_pool" / "generator_spec.json"
CANDIDATE_POOL = OUTPUT_ROOT / "candidate_pool" / "candidate_pool.json"
CANDIDATE_RESULTS = OUTPUT_ROOT / "candidate_pool" / "support_results.npz"
SCREENING_PATH = OUTPUT_ROOT / "screening" / "candidate_pool.json"
ARTIFACT_DIR = OUTPUT_ROOT / "artifacts"
ALLOWANCES = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
BANK_SIZES = {
    "law_search": 32768, "risk_anchor": 32768, "screen": 8192,
    "search_train": 32768, "periodic_audit": 16384,
    "authoritative_train": 65536, "authoritative_audit": 65536,
}
VALIDATION_SIZES = {"truth": 5000, "reference_fit": 16384, "reference_audit": 16384}
GLOBAL_SEED = 20260826
DESIGN_N = 6000
CANDIDATE_COUNT = 4096
K = 280
MINIMUM_RESS = 0.05
BOX = (2.0, 1.0)


def canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical(payload)).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _inside(path: Path) -> Path:
    resolved, root = path.resolve(), OUTPUT_ROOT.resolve()
    if resolved != root and root not in resolved.parents: raise ValueError(f"official output escaped {root}: {resolved}")
    return resolved


def atomic_bytes(path: Path, data: bytes, *, immutable: bool = True) -> None:
    path = _inside(path)
    if path.exists():
        if immutable and path.read_bytes() != data: raise RuntimeError(f"refusing to overwrite sealed artifact: {path}")
        if path.read_bytes() == data: return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle: handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def atomic_json(path: Path, payload: Any, *, immutable: bool = True) -> None:
    atomic_bytes(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n", immutable=immutable)


def atomic_text(path: Path, value: str, *, immutable: bool = True) -> None:
    atomic_bytes(path, value.encode(), immutable=immutable)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path = _inside(path)
    if path.exists(): raise RuntimeError(f"refusing to overwrite sealed artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent); os.close(fd)
    try:
        np.savez_compressed(temporary, **{key: np.asarray(value) for key, value in arrays.items()}); os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def hashes(paths: Any) -> list[dict[str, Any]]:
    return [{"path": str(Path(path).relative_to(OUTPUT_ROOT)), "bytes": Path(path).stat().st_size, "sha256": file_sha256(Path(path))} for path in paths]


def derive_seed(scope: str, label: str) -> dict[str, Any]:
    text = f"{VERSION}|{GLOBAL_SEED}|{scope}|{label}"; digest = hashlib.sha256(text.encode()).hexdigest()
    return {"label": label, "scope": scope, "derivation_text": text, "sha256": digest, "seed": int(digest[:16], 16) % (2**31 - 1)}


def slug(value: float) -> str:
    return str(float(value)).replace(".", "p").removesuffix("p0")


def selection_ceiling(law_risk: float, allowance: float) -> float:
    return float(law_risk) * (1.0 + float(allowance) / 100.0)


def validation_ceiling(law_risk: float, allowance: float) -> float:
    return float(law_risk) * (1.0 + float(allowance) / 100.0 + 0.05)


def signature(protocol: dict[str, Any], kind: str, extra: Any = None) -> str:
    return payload_sha256({"protocol_sha256": protocol["protocol_sha256"], "kind": kind, "extra": extra, "dictionary": DICTIONARY_SHA256})


def _all_seeds(value: Any) -> set[int]:
    if isinstance(value, dict): return ({int(value["seed"])} if isinstance(value.get("seed"), int) else set()) | set().union(*(_all_seeds(v) for v in value.values()), set())
    if isinstance(value, list): return set().union(*(_all_seeds(v) for v in value), set())
    return set()


def protocol_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    confirmation = read_json(CONFIRMATION_ROOT / "summary.json")
    if confirmation["classification"] != "PRODUCTION_LAUNCH_READY" or not all(row["passed"] for row in confirmation["conditions"].values()):
        raise RuntimeError("official production is not authorized by confirmation")
    if file_sha256(CHECKPOINT) != CHECKPOINT_SHA256 or read_json(ACCEPTED_REFERENCE_PATH)["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise RuntimeError("accepted B1 checkpoint changed")
    if file_sha256(DICTIONARY_PATH) != DICTIONARY_SHA256: raise RuntimeError("K280 dictionary changed")
    if float(cfg["forcing"]["minimum_ess_fraction"]) != MINIMUM_RESS: raise RuntimeError("rESS threshold changed")
    selection_labels = ["design_truth", "selection_observation_noise", "law_search_pool", "candidate_local", "candidate_tangent", "candidate_paths", "candidate_sobol", *BANK_SIZES]
    validation_labels = ["truth", "reference_fit", "reference_audit", "measurement_noise"]
    selection_seeds = [derive_seed("selection", label) for label in selection_labels]
    validation_seeds = [derive_seed("validation", label) for label in validation_labels]
    prior = _all_seeds(read_json(DEV_ROOT / "experiment_manifest.json")) | _all_seeds(read_json(CONFIRMATION_ROOT / "confirmation_bank_manifest.json"))
    new = {row["seed"] for row in selection_seeds + validation_seeds}
    if len(new) != len(selection_seeds) + len(validation_seeds) or new & prior: raise RuntimeError("official seed collision")
    source_names = ["official_b1_pareto.py", "official_b1_pareto_run.py", "test_official_b1_pareto.py", "pareto_v2_selection.py", "pareto_v2_validation.py", "config.json", "domain.py", "risk.py", "forcing.py", "single_reference_b1_preflight.py"]
    return {
        "schema_version": 1, "version": VERSION, "new_experiment": True,
        "authorization": {"confirmation": "PRODUCTION_LAUNCH_READY", "summary_sha256": file_sha256(CONFIRMATION_ROOT / "summary.json")},
        "reference": {"checkpoint_sha256": CHECKPOINT_SHA256, "retrained": False, "B1_particle_matching": True, "configuration_OT": False, "provenance_sha256": file_sha256(ACCEPTED_REFERENCE_PATH)},
        "source_hashes": {name: file_sha256(ROOT / name) for name in source_names},
        "protocol_document_sha256": file_sha256(PROTOCOL_DOCUMENT),
        "physical_simulator_sha256": file_sha256(ROOT / "domain.py"), "risk_definition_sha256": file_sha256(ROOT / "risk.py"),
        "whitening_rule": "fresh selection truth via unchanged whitening_from_truth", "time_weights": "normalized trapezoid over 13 nodes",
        "constants": {"dtype": "float64", "minimum_rESS": MINIMUM_RESS, "projection_residual": 2e-6, "forcing_mean": 2e-7, "maximum_covariance_condition": 1e10, "K": K, "dictionary_sha256": DICTIONARY_SHA256, "relative_rank_tolerance": 1e-12, "replacement_tolerance": 1e-10, "galerkin_backend": "jax"},
        "data": {"design_truth_N": DESIGN_N, "bank_sizes": BANK_SIZES, "selection_seed_records": selection_seeds},
        "law": {"algorithm": "frozen multi-component 1536-point scientific-risk pool plus three 128-point local refinements; top-24 independent risk-anchor support choice", "development_Law_mandatory_start": True, "development_R_Law_used_as_anchor": False},
        "candidate_generator": {"count": CANDIDATE_COUNT, "component_targets": {"local": 1434, "risk_tangent": 1024, "periodic_paths": 819, "sobol": 819}, "canonicalization": "periodic wrap plus exhaustive unordered-sensor matching to official Law", "frozen_before_scoring": True},
        "screening": {"screen_N": 8192, "periodic_audit_N": 16384, "dual_bank_required": True},
        "starts": {"Tangent": 6, "Full": 3, "algorithm": "incumbent, robust-rESS, low-risk, symmetry-aware max-min"},
        "optimizer": {"maximum_accepted_step_attempts": 1, "maximum_backtracks": 3, "initial_step": 5e-5, "backtrack_factor": 0.5, "trust_radius": 2e-4, "periodic_audit_every_accepted_steps": 1, "replacement_tolerance": 1e-10, "rank_must_equal_previous_step": True, "tangent": "exact Gram objective", "full": "K280 fixed-coefficient envelope; no eigensolve differentiation"},
        "allowances_percent": list(ALLOWANCES), "risk_rule": "R <= (1+p/100)*R_Law_official exactly", "allowance_failures_independent": True, "nested_incumbent_rule": True,
        "validation": {"sizes": VALIDATION_SIZES, "seed_records": validation_seeds, "generation_forbidden_before_selection_hash": True, "rule": "R_method,val <= (1+p/100+0.05)*R_Law,val", "strict_nominal_reported": True, "optimization_forbidden": True},
        "full_method": "fixed-feature finite-dimensional K=280 Galerkin approximation", "deep_ritz_used": False,
    }


def freeze_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    body = protocol_payload(cfg); digest = payload_sha256(body); wrapped = {**body, "protocol_sha256": digest, "protocol_frozen": True, "validation_arrays_generated": False}
    if PROTOCOL_PATH.exists():
        if read_json(PROTOCOL_PATH) != wrapped or PROTOCOL_HASH_PATH.read_text().strip() != digest: raise RuntimeError("official protocol seal mismatch")
    else:
        atomic_json(PROTOCOL_PATH, wrapped); atomic_text(PROTOCOL_HASH_PATH, digest + "\n")
    return wrapped


def require_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    if not PROTOCOL_PATH.exists() or not PROTOCOL_HASH_PATH.exists(): raise RuntimeError("freeze official protocol first")
    saved = read_json(PROTOCOL_PATH); body = {key: value for key, value in saved.items() if key not in {"protocol_sha256", "protocol_frozen", "validation_arrays_generated"}}
    if payload_sha256(body) != saved["protocol_sha256"] or protocol_payload(cfg) != body or PROTOCOL_HASH_PATH.read_text().strip() != saved["protocol_sha256"]: raise RuntimeError("official protocol differs from code/config")
    return saved


def _seed(protocol: dict[str, Any], label: str, *, validation: bool = False) -> dict[str, Any]:
    rows = protocol["validation"]["seed_records"] if validation else protocol["data"]["selection_seed_records"]
    return next(row for row in rows if row["label"] == label)


def official_config(cfg: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(cfg)
    if LAW_PATH.exists(): updated["envelope"]["law_eta"] = read_json(LAW_PATH)["eta_Law_official"]
    return updated


def generate_design_truth(cfg: dict[str, Any], progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    protocol = require_protocol(cfg)
    if DESIGN_RECORD.exists():
        row = read_json(DESIGN_RECORD)
        if file_sha256(DESIGN_PATH) != row["sha256"]: raise RuntimeError("design truth changed")
        return row
    seed = _seed(protocol, "design_truth")["seed"]; times = jnp.linspace(0, 1, 13, dtype=jnp.float64)
    bank = SkyrmionTruth(_physics_config(cfg)).make_bank(seed=seed, samples=DESIGN_N, times=times, substeps_per_interval=int(cfg["physics"]["truth_substeps"]))
    configurations = np.asarray(bank.configurations); features = many_body_features(jnp.asarray(configurations), BOX)
    truth_means, whitening = np.asarray(jnp.mean(features, axis=1)), np.asarray(whitening_from_truth(features))
    atomic_npz(DESIGN_PATH, times=times, configurations=configurations, truth_means=truth_means, whitening=whitening, seed=np.asarray(seed))
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CHECKPOINT, ARTIFACT_DIR / "reference.npz")
    atomic_npz(ARTIFACT_DIR / "truth_banks.npz", times=times, design=configurations)
    row = {"schema_version": 1, "N": DESIGN_N, "seed": seed, "sha256": file_sha256(DESIGN_PATH), "whitening_sha256": _array_sha256(whitening), "checkpoint_copy_sha256": file_sha256(ARTIFACT_DIR / "reference.npz"), "fresh_official_selection_truth": True, "validation": False}
    atomic_json(DESIGN_RECORD, row)
    if progress: progress(f"official design truth: N={DESIGN_N}")
    return row


def _bank_path(label: str) -> Path:
    return OUTPUT_ROOT / "banks" / f"{label}_N{BANK_SIZES[label]}.npz"


def load_bank(label: str) -> GalerkinReferenceBank:
    with np.load(_bank_path(label), allow_pickle=False) as values: return GalerkinReferenceBank(jnp.asarray(values["configurations"]), jnp.asarray(values["velocity"]), jnp.asarray(values["base_weights"]))


def generate_banks(cfg: dict[str, Any], progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    protocol = require_protocol(cfg); generate_design_truth(cfg, progress); records = []
    times = jnp.linspace(0, 1, 13, dtype=jnp.float64)
    for label, count in BANK_SIZES.items():
        path, record_path = _bank_path(label), _bank_path(label).with_suffix(".json"); seed = _seed(protocol, label)["seed"]
        if record_path.exists():
            row = read_json(record_path)
            if file_sha256(path) != row["sha256"]: raise RuntimeError(f"bank changed: {label}")
            records.append(row); continue
        started = time.perf_counter(); truth = SkyrmionTruth(_physics_config(cfg)); initial = np.asarray(truth.sample_initial(jax.random.PRNGKey(seed), count)); flow = load_reference(CHECKPOINT)
        xs, vs = [], []
        for start in range(0, count, 2048):
            trajectory = flow.rollout(jnp.asarray(initial[start:start + 2048]), times, substeps_per_interval=int(cfg["banks"]["reference_substeps"]))
            xs.append(np.asarray(trajectory)); vs.append(np.asarray(flow.velocity(trajectory, times)))
        configurations, velocity = np.concatenate(xs, axis=1), np.concatenate(vs, axis=1); weights = np.full((13, count), 1.0 / count)
        if not np.array_equal(configurations[0], initial): raise RuntimeError("rollout changed P0")
        atomic_npz(path, configurations=configurations, velocity=velocity, base_weights=weights, role=np.asarray(label), seed=np.asarray(seed))
        row = {"label": label, "N": count, "seed": seed, "initial_P0_sha256": _array_sha256(initial), "checkpoint_sha256": CHECKPOINT_SHA256, "sha256": file_sha256(path), "wall_time_seconds": time.perf_counter() - started}
        atomic_json(record_path, row); records.append(row)
        if progress: progress(f"official bank {label}: N={count}")
        del flow, configurations, velocity, initial; gc.collect()
    manifest = {"schema_version": 1, "passed": True, "protocol_sha256": protocol["protocol_sha256"], "roles": BANK_SIZES, "seed_records": protocol["data"]["selection_seed_records"], "initial_state_hashes": {row["label"]: row["initial_P0_sha256"] for row in records}, "pairwise_role_disjoint": len({row["initial_P0_sha256"] for row in records}) == len(records), "reference_retrained": False, "artifacts": hashes([_bank_path(label) for label in BANK_SIZES])}
    atomic_json(OUTPUT_ROOT / "banks" / "manifest.json", manifest)
    return manifest


def selection_data(cfg: dict[str, Any], train: str, audit: str, *, projection: str = "risk_anchor") -> SelectionGalerkinData:
    generate_banks(cfg)
    with np.load(DESIGN_PATH, allow_pickle=False) as values:
        times, configurations, truth_means, whitening = (jnp.asarray(values[key]) for key in ("times", "configurations", "truth_means", "whitening"))
    protocol = require_protocol(cfg); problem = _make_problem(cfg, configurations, times, _family(cfg), noise_seed=_seed(protocol, "selection_observation_noise")["seed"])
    projection_bank = load_bank(projection)
    return SelectionGalerkinData(problem, projection_bank, load_bank(train), load_bank(audit), many_body_features(projection_bank.configurations, BOX), truth_means, whitening)


def _consider(rows: list[dict[str, Any]], seen: set[str], eta: Any, reference: np.ndarray, cfg: dict[str, Any], component: str, **meta: Any) -> bool:
    value = canonicalize_eta(eta, reference, BOX)
    if minimum_periodic_separation(value, BOX) < float(cfg["measurement"]["min_separation"]): return False
    key = eta_key(value)
    if key in seen: return False
    seen.add(key); rows.append({"candidate_id": f"candidate_{len(rows):05d}", "eta": value.tolist(), "eta_sha256": key, "component": component, **meta}); return True


def _evaluate(etas: np.ndarray, cfg: dict[str, Any], label: str) -> dict[str, np.ndarray]:
    data = selection_data(cfg, label, label, projection=label); evaluator = _ReferenceEvaluator(data.selection_problem, np.asarray(data.truth_means), np.asarray(data.whitening))
    return evaluator.evaluate(etas, load_bank(label), BANK_SIZES[label])


def reconstruct_law(cfg: dict[str, Any], progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    protocol = require_protocol(cfg); generate_banks(cfg, progress)
    if LAW_PATH.exists(): return read_json(LAW_PATH)
    dev = read_json(DEV_ROOT / "law_freeze.json"); reference = np.asarray(dev["eta_Law_B1"]); rows: list[dict[str, Any]] = []; seen: set[str] = set()
    contexts = [reference, np.asarray(json.loads(CONFIG_PATH.read_text())["envelope"]["law_eta"])]
    for value in contexts: _consider(rows, seen, value, reference, cfg, "mandatory_context")
    seed = _seed(protocol, "law_search_pool")["seed"]; rng = np.random.default_rng(seed)
    while sum(row["component"] == "local" for row in rows) < 1022:
        scale = (0.001, 0.002, 0.005, 0.01, 0.02, 0.04)[len(rows) % 6]; _consider(rows, seen, reference + scale * rng.normal(size=8), reference, cfg, "local", scale=scale)
    sobol = qmc.Sobol(d=8, scramble=True, seed=seed + 1)
    for index, point in enumerate(sobol.random_base2(m=13)):
        if sum(row["component"] == "sobol" for row in rows) >= 512: break
        _consider(rows, seen, point * np.tile(np.asarray(BOX), 4), reference, cfg, "sobol", sobol_index=index)
    atomic_json(LAW_POOL_PATH, {"frozen_before_evaluation": True, "count": len(rows), "rows_sha256": payload_sha256(rows), "rows": rows})
    search = _evaluate(np.asarray([row["eta"] for row in rows]), cfg, "law_search")
    scored = [{**row, "search_risk": float(search["scientific_risk"][i]), "search_support_valid": bool(search["support_valid"][i])} for i, row in enumerate(rows)]
    for round_index, scale in enumerate((0.01, 0.005, 0.002)):
        centers = sorted(scored, key=lambda row: (not row["search_support_valid"], row["search_risk"], row["candidate_id"]))[:16]; fresh = []
        for center in centers:
            made = 0
            while made < 8:
                if _consider(fresh, seen, np.asarray(center["eta"]) + scale * rng.normal(size=8), reference, cfg, "refinement", round=round_index + 1, center=center["candidate_id"]): made += 1
        atomic_json(OUTPUT_ROOT / "law" / f"refinement_{round_index + 1}_pool.json", {"frozen_before_evaluation": True, "rows": fresh})
        result = _evaluate(np.asarray([row["eta"] for row in fresh]), cfg, "law_search")
        scored.extend([{**row, "search_risk": float(result["scientific_risk"][i]), "search_support_valid": bool(result["support_valid"][i])} for i, row in enumerate(fresh)])
        if progress: progress(f"official Law refinement {round_index + 1}: {len(fresh)}")
    shortlist = [row for row in sorted(scored, key=lambda row: (not row["search_support_valid"], row["search_risk"], row["candidate_id"])) if row["search_support_valid"]][:24]
    anchor = _evaluate(np.asarray([row["eta"] for row in shortlist]), cfg, "risk_anchor"); valid = np.asarray(anchor["support_valid"])
    if not np.any(valid): raise RuntimeError("official Law has no supported risk-anchor finalist")
    index = min(np.flatnonzero(valid), key=lambda i: (float(anchor["scientific_risk"][i]), shortlist[int(i)]["candidate_id"])); winner = shortlist[int(index)]
    law = {"schema_version": 1, "status": "FROZEN", "eta_Law_official": winner["eta"], "R_Law_official": float(anchor["scientific_risk"][index]), "checkpoint_sha256": CHECKPOINT_SHA256, "design_truth_sha256": file_sha256(DESIGN_PATH), "law_search_bank_sha256": file_sha256(_bank_path("law_search")), "risk_anchor_bank_sha256": file_sha256(_bank_path("risk_anchor")), "search_provenance": winner, "development_R_Law_used": False, "risk_ceilings": {str(p): selection_ceiling(float(anchor["scientific_risk"][index]), p) for p in ALLOWANCES}}
    atomic_json(LAW_RESULTS_PATH, {"initial_and_refined_count": len(scored), "shortlist": [{**row, "anchor_risk": float(anchor["scientific_risk"][i]), "anchor_support_valid": bool(valid[i])} for i, row in enumerate(shortlist)]}); atomic_json(LAW_PATH, law)
    return law


def generate_candidates(cfg: dict[str, Any], progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    protocol = require_protocol(cfg); law = reconstruct_law(cfg, progress)
    if CANDIDATE_POOL.exists(): return read_json(CANDIDATE_POOL)
    spec = {"schema_version": 1, **protocol["candidate_generator"], "seeds": {name: _seed(protocol, f"candidate_{name}") for name in ("local", "tangent", "paths", "sobol")}, "official_law_sha256": file_sha256(LAW_PATH), "frozen_before_scoring": True}
    atomic_json(CANDIDATE_SPEC, spec); reference = np.asarray(law["eta_Law_official"]); rows: list[dict[str, Any]] = []; seen: set[str] = set(); targets = spec["component_targets"]
    contexts = [reference, np.asarray(read_json(DEV_ROOT / "law_freeze.json")["eta_Law_B1"]), np.asarray(json.loads(CONFIG_PATH.read_text())["envelope"]["law_eta"])]
    for value in contexts: _consider(rows, seen, value, reference, cfg, "local", anchor="context")
    rng = np.random.default_rng(spec["seeds"]["local"]["seed"])
    while sum(row["component"] == "local" for row in rows) < targets["local"]:
        index = sum(row["component"] == "local" for row in rows); scale = (0.00025, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.04)[index % 8]
        _consider(rows, seen, contexts[index % len(contexts)] + scale * rng.normal(size=8), reference, cfg, "local", scale=scale)
    epsilon = 1e-4; finite = []
    for coordinate in range(8):
        for sign in (-1, 1):
            eta = reference.copy(); eta[coordinate] += sign * epsilon; finite.append(canonicalize_eta(eta, reference, BOX))
    finite_result = _evaluate(np.asarray(finite), cfg, "risk_anchor"); finite_risk = np.asarray(finite_result["scientific_risk"]); gradient = np.asarray([(finite_risk[2*i+1]-finite_risk[2*i])/(2*epsilon) for i in range(8)]); norm2 = float(gradient @ gradient)
    tangent_rng = np.random.default_rng(spec["seeds"]["tangent"]["seed"]); direction_index = 0; radii = (0.0001,0.0002,0.0004,0.0007,0.001,0.0015,0.0022,0.0032,0.0045,0.0064,0.009,0.0125,0.017,0.023,0.031,0.041,0.055)
    while sum(row["component"] == "risk_tangent" for row in rows) < targets["risk_tangent"]:
        direction = tangent_rng.normal(size=8); direction -= (direction @ gradient) / max(norm2, 1e-30) * gradient; direction /= max(np.linalg.norm(direction), 1e-30)
        for radius in radii:
            for sign in (-1,1):
                if sum(row["component"] == "risk_tangent" for row in rows) >= targets["risk_tangent"]: break
                _consider(rows, seen, reference + sign*radius*direction, reference, cfg, "risk_tangent", direction=direction_index, radius=radius, sign=sign)
        direction_index += 1
    golden = (math.sqrt(5)-1)/2; path_index = 0
    while sum(row["component"] == "periodic_paths" for row in rows) < targets["periodic_paths"]:
        target = contexts[1 + path_index % (len(contexts)-1)]; alpha = ((path_index+1)*golden)%1; delta=(target-reference).reshape(-1,2); delta-=np.asarray(BOX)*np.round(delta/np.asarray(BOX)); value=reference+alpha*delta.reshape(-1)
        _consider(rows, seen, value, reference, cfg, "periodic_paths", alpha=alpha); path_index += 1
    sobol = qmc.Sobol(d=8, scramble=True, seed=spec["seeds"]["sobol"]["seed"])
    for index, point in enumerate(sobol.random_base2(m=15)):
        if sum(row["component"] == "sobol" for row in rows) >= targets["sobol"]: break
        _consider(rows, seen, point*np.tile(np.asarray(BOX),4), reference,cfg,"sobol",sobol_index=index)
    if len(rows) != CANDIDATE_COUNT: raise RuntimeError(f"candidate count {len(rows)} != {CANDIDATE_COUNT}")
    payload = {"schema_version": 1, "count": len(rows), "component_counts": {key: sum(row["component"]==key for row in rows) for key in targets}, "canonical_unique": len(seen)==len(rows), "rows_sha256": payload_sha256(rows), "rows": rows}
    atomic_json(CANDIDATE_POOL, payload)
    if progress: progress(f"official candidate pool frozen: {len(rows)}")
    return payload


def screen_candidates(cfg: dict[str, Any], progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    protocol = require_protocol(cfg); pool = generate_candidates(cfg, progress); law = read_json(LAW_PATH)
    if SCREENING_PATH.exists(): return read_json(SCREENING_PATH)
    etas = np.asarray([row["eta"] for row in pool["rows"]]); risk = _evaluate(etas, cfg, "risk_anchor"); screen = _evaluate(etas, cfg, "screen"); audit = _evaluate(etas, cfg, "periodic_audit")
    atomic_npz(CANDIDATE_RESULTS, risk=np.asarray(risk["scientific_risk"]), screen_support=np.asarray(screen["support_valid"]), audit_support=np.asarray(audit["support_valid"]), screen_minimum_rESS=np.asarray(screen["minimum_ress"]), audit_minimum_rESS=np.asarray(audit["minimum_ress"]))
    rows=[]
    for i, source in enumerate(pool["rows"]):
        rows.append({**source, "scientific_selection_risk": float(risk["scientific_risk"][i]), "screen_valid": bool(screen["support_valid"][i]), "audit_valid": bool(audit["support_valid"][i]), "projection_valid": bool(screen["support_valid"][i] and audit["support_valid"][i]), "minimum_ess_fraction": float(min(screen["minimum_ress"][i], audit["minimum_ress"][i])), "robust_rESS": float(min(screen["minimum_ress"][i], audit["minimum_ress"][i]))})
    starts={}; box=_family(cfg).box
    for allowance in ALLOWANCES:
        eligible=[row for row in rows if row["scientific_selection_risk"]<=selection_ceiling(law["R_Law_official"],allowance) and row["screen_valid"] and row["audit_valid"]]
        selected=[]
        for row, role in ((min(eligible,key=lambda x:(x["scientific_selection_risk"],x["candidate_id"])),"low_risk"),(max(eligible,key=lambda x:(x["robust_rESS"],x["candidate_id"])),"best_ress")):
            if not any(eta_key(row["eta"])==eta_key(old["eta"]) for old in selected): selected.append({**row,"start_role":role})
        while len(selected)<6:
            remaining=[row for row in eligible if not any(eta_key(row["eta"])==eta_key(old["eta"]) for old in selected)]
            if not remaining: break
            row=max(remaining,key=lambda x:(min(_symmetry_aware_distance(x["eta"],old["eta"],BOX) for old in selected),x["candidate_id"])); selected.append({**row,"start_role":"maxmin_diverse"})
        starts[slug(allowance)]=selected
    result={"schema_version":1,"passed":True,"signature":signature(protocol,"official_screening"),"law_risk":law["R_Law_official"],"law_eta":law["eta_Law_official"],"pool_count":len(rows),"rows":rows,"starts":starts,"full_Kf_solve_count":0,"validation_accessed":False,"candidate_pool_sha256":file_sha256(CANDIDATE_POOL)}
    atomic_json(SCREENING_PATH,result)
    if progress: progress(f"official dual-bank screen: {sum(row['projection_valid'] for row in rows)}/{len(rows)}")
    return result


def configure_selection_engine(cfg: dict[str, Any], *, start_cap: int) -> Any:
    from . import pareto_v2_selection as engine
    for name, value in {"ALLOWANCES":ALLOWANCES,"ARTIFACT_DIR":ARTIFACT_DIR,"BANK_SIZES":BANK_SIZES,"DICTIONARY_PATH":DICTIONARY_PATH,"EXPECTED_DICTIONARY_SHA256":DICTIONARY_SHA256,"K":K,"MINIMUM_RESS":MINIMUM_RESS,"OUTPUT_ROOT":OUTPUT_ROOT,"atomic_json":atomic_json,"eta_key":eta_key,"hashes":hashes,"payload_sha256":payload_sha256,"read_json":read_json,"require_protocol":require_protocol,"selection_ceiling":selection_ceiling,"signature":signature,"slug":slug,"load_bank":load_bank,"selection_data":selection_data}.items(): setattr(engine,name,value)
    def method_starts(screening: dict[str,Any], allowance: float, incumbent: dict[str,Any]|None) -> list[dict[str,Any]]:
        rows=list(screening["starts"][slug(allowance)])
        if incumbent is not None: rows.insert(0,{"candidate_id":"mandatory_previous_incumbent","eta":incumbent["eta"],"start_role":"mandatory_previous_incumbent"})
        kept=[]
        for row in rows:
            if not any(eta_key(row["eta"])==eta_key(old["eta"]) for old in kept): kept.append(row)
            if len(kept)==start_cap: break
        return kept
    engine._method_starts=method_starts
    return engine


def select_tangent(cfg: dict[str, Any]) -> dict[str, Any]:
    screen_candidates(cfg); return configure_selection_engine(official_config(cfg),start_cap=6).select_tangent(official_config(cfg))


def select_full(cfg: dict[str, Any]) -> dict[str, Any]:
    screen_candidates(cfg); return configure_selection_engine(official_config(cfg),start_cap=3).select_full(official_config(cfg))


def cross_evaluate(cfg: dict[str, Any]) -> dict[str, Any]:
    return configure_selection_engine(official_config(cfg),start_cap=3).cross_evaluate(official_config(cfg))


def freeze_selection(cfg: dict[str, Any]) -> dict[str, Any]:
    manifest=configure_selection_engine(official_config(cfg),start_cap=3).freeze_selection(official_config(cfg))
    source=OUTPUT_ROOT/"selection"/"selection_hash.txt"; target=OUTPUT_ROOT/"selection"/"selection_hash.json"
    atomic_json(target,{"schema_version":1,"selection_sha256":source.read_text().strip(),"selection_manifest_sha256":file_sha256(OUTPUT_ROOT/"selection"/"selection_manifest.json")})
    return manifest


def configure_validation_engine(cfg: dict[str, Any]) -> Any:
    from . import pareto_v2_validation as engine
    for name,value in {"ALLOWANCES":ALLOWANCES,"ARTIFACT_DIR":ARTIFACT_DIR,"DICTIONARY_PATH":DICTIONARY_PATH,"K":K,"OUTPUT_ROOT":OUTPUT_ROOT,"VALIDATION_SIZES":VALIDATION_SIZES,"atomic_json":atomic_json,"eta_key":eta_key,"hashes":hashes,"payload_sha256":payload_sha256,"read_json":read_json,"require_protocol":require_protocol,"validation_ceiling":validation_ceiling}.items(): setattr(engine,name,value)
    engine.FRESH=OUTPUT_ROOT/"fresh_validation"; engine.TRUTH=engine.FRESH/"truth.npz"; engine.FIT=engine.FRESH/"reference_fit.npz"; engine.AUDIT=engine.FRESH/"reference_audit.npz"; engine.NOISE=engine.FRESH/"measurement_noise.npz"; engine.MANIFEST=engine.FRESH/"artifact_manifest.json"
    return engine


def generate_validation(cfg: dict[str, Any]) -> dict[str, Any]:
    if not (OUTPUT_ROOT/"selection"/"selection_hash.json").exists(): raise RuntimeError("validation forbidden before selection_hash.json")
    return configure_validation_engine(official_config(cfg)).generate_fresh_validation(official_config(cfg))


def validate(cfg: dict[str, Any]) -> dict[str, Any]:
    return configure_validation_engine(official_config(cfg)).validate(official_config(cfg))


def write_report(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol=require_protocol(cfg); law=read_json(LAW_PATH); selection=read_json(OUTPUT_ROOT/"selection"/"pareto_selection.json"); cross=read_json(OUTPUT_ROOT/"selection"/"cross_evaluation.json"); validation=read_json(OUTPUT_ROOT/"fresh_validation"/"results.json")
    lines=["# Official B1 Galerkin Pareto v1","","Status: COMPLETE","","The official Full method is the **fixed-feature finite-dimensional K=280 Galerkin approximation**; it is not an infinite-dimensional converged Full solution. Deep Ritz was not used.","",f"Protocol SHA-256: `{protocol['protocol_sha256']}`",f"Accepted B1 checkpoint SHA-256: `{CHECKPOINT_SHA256}`","","## Official Law","",f"eta: `{law['eta_Law_official']}`",f"R_Law_official: `{law['R_Law_official']}`","","## Selection and fresh validation","","| allowance | method | selection risk | selection Tangent | selection Full K280 | validation risk | validation Full K280 | strict nominal | p+5pp | classification |","|---:|---|---:|---:|---:|---:|---:|---|---|---|"]
    for vrow in validation["rows"]:
        srow=next(row for row in cross["rows"] if row["allowance_percent"]==vrow["allowance_percent"] and row["selected_by"]==vrow["selected_by"])
        lines.append(f"| {vrow['allowance_percent']}% | {vrow['selected_by']} | {srow['risk']:.9g} | {srow['tangent_action']:.9g} | {srow['full_action']:.9g} | {vrow['validation_risk']:.9g} | {vrow['full_fit_action']:.9g} | {vrow['strict_p_validation_pass']} | {vrow['p_plus_5pp_validation_pass']} | {vrow['classification']} |")
    lines += ["",f"Selection SHA-256: `{read_json(OUTPUT_ROOT/'selection'/'selection_manifest.json')['pareto_selection_sha256']}`","","Validation generated after selection freeze: YES","Validation modified selection: NO",""]
    atomic_text(OUTPUT_ROOT/"report.md","\n".join(lines)); summary={"schema_version":1,"status":"COMPLETE","protocol_sha256":protocol["protocol_sha256"],"accepted_checkpoint_sha256":CHECKPOINT_SHA256,"official_law":law,"selection_sha256":read_json(OUTPUT_ROOT/"selection"/"selection_manifest.json")["pareto_selection_sha256"],"validation_sha256":file_sha256(OUTPUT_ROOT/"fresh_validation"/"results.json"),"validation_rows":validation["rows"],"deep_ritz_used":False,"full_method":"fixed-feature finite-dimensional K=280 Galerkin approximation"}; atomic_json(OUTPUT_ROOT/"final_summary.json",summary)
    files=[path for path in sorted(OUTPUT_ROOT.rglob("*")) if path.is_file() and path.name!="inventory.json"]; atomic_json(OUTPUT_ROOT/"inventory.json",{"schema_version":1,"artifact_count":len(files),"files":hashes(files)})
    return summary
