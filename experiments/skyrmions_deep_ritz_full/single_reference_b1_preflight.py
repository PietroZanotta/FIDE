"""Clean-room, development-only, single-reference B1 preflight.

The module owns a new seed namespace and output tree.  It trains from endpoint
states only, accepts the first endpoint-qualified B1 checkpoint, then freezes a
new selection Law and measures candidate support.  It intentionally exposes no
validation, Tangent, Full/Galerkin, eigensolve, or Deep-Ritz entry point.
"""

from __future__ import annotations

from dataclasses import asdict, replace
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
from scipy.optimize import linear_sum_assignment
from scipy.stats import qmc

from .bridge_ablation import particle_assignment
from .candidate_coverage import (
    _risk_rows_batched,
    canonicalize_eta,
    minimum_periodic_separation,
    periodic_interpolate,
)
from .domain import SkyrmionTruth
from .galerkin_only_data import (
    GalerkinReferenceBank,
    SelectionGalerkinData,
    _family,
    _make_problem,
    _physics_config,
)
from .pareto_v2_common import ARTIFACT_DIR
from .pareto_v3_common import ROOT, eta_key, file_sha256
from .pareto_v3_diagnostic import _symmetry_aware_distance
from .reference import (
    EquivariantReferenceFlow,
    ReferenceTrainingConfig,
    _AdamState,
    _tree_norm,
    equivariant_velocity,
    init_equivariant_reference,
    load_reference,
    save_reference,
)
from .reference_seed_robustness import (
    EXPECTED_BASELINE_CHECKPOINT_SHA256,
    EXPECTED_CONFIG_SHA256,
    _ReferenceEvaluator,
    _array_sha256,
)
from .reference_semantics_audit import _mean_phi, _mean_psi
from .risk import many_body_features, whitening_from_truth


VERSION = "skyrmion_galerkin_dev_single_reference_b1_preflight_v1"
SEED_NAMESPACE = VERSION
GLOBAL_SEED = 20260826
OUTPUT_ROOT = ROOT / "outputs" / VERSION
REPO_ROOT = ROOT.parent.parent
CONFIG_PATH = ROOT / "config.json"
BASELINE_CHECKPOINT_PATH = ARTIFACT_DIR / "reference.npz"
BRIDGE_ROOT = ROOT / "outputs" / "skyrmion_galerkin_dev_bridge_ablation_v1"
HISTORICAL_COVERAGE_ROOT = ROOT / "outputs" / "skyrmion_galerkin_dev_candidate_coverage_v1"

SOURCE_SEAL_PATH = OUTPUT_ROOT / "source_seal.json"
MANIFEST_PATH = OUTPUT_ROOT / "experiment_manifest.json"
MANIFEST_HASH_PATH = OUTPUT_ROOT / "experiment_manifest.sha256"
TRAIN_DATA_PATH = OUTPUT_ROOT / "reference_endpoint_train.npz"
TRAIN_DATA_MANIFEST_PATH = OUTPUT_ROOT / "reference_endpoint_train_manifest.json"
QUAL_DATA_PATH = OUTPUT_ROOT / "endpoint_qualification_holdout.npz"
QUAL_DATA_MANIFEST_PATH = OUTPUT_ROOT / "endpoint_qualification_manifest.json"
DESIGN_DATA_PATH = OUTPUT_ROOT / "design_truth_selection.npz"
DESIGN_DATA_MANIFEST_PATH = OUTPUT_ROOT / "design_truth_manifest.json"
COUPLING_PATH = OUTPUT_ROOT / "reference_training" / "b1_coupling_maps.npz"
COUPLING_MANIFEST_PATH = OUTPUT_ROOT / "reference_training" / "b1_coupling_manifest.json"
QUAL_COUPLING_PATH = OUTPUT_ROOT / "reference_training" / "qualification_b1_coupling_maps.npz"
ACCEPTED_REFERENCE_PATH = OUTPUT_ROOT / "accepted_reference.json"
REFERENCE_BANK_MANIFEST_PATH = OUTPUT_ROOT / "reference_bank_manifest.json"
LAW_POOL_PATH = OUTPUT_ROOT / "law_search_pool.json"
LAW_RESULTS_PATH = OUTPUT_ROOT / "law_search_results.json"
LAW_FREEZE_PATH = OUTPUT_ROOT / "law_freeze.json"
CANDIDATE_SPEC_PATH = OUTPUT_ROOT / "candidate_generator_spec.json"
CANDIDATE_POOL_PATH = OUTPUT_ROOT / "candidate_pool.json"
CANDIDATE_RISK_PATH = OUTPUT_ROOT / "candidate_risk_results.json"
ALLOWANCE_PATH = OUTPUT_ROOT / "allowance_support_summary.json"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
INVENTORY_PATH = OUTPUT_ROOT / "inventory.json"
REPORT_PATH = OUTPUT_ROOT / "report.md"

TRAIN_N = 12000
QUAL_N = 4096
DESIGN_N = 6000
TIME_COUNT = 13
COUPLING_MAPS = 4
CFM_QUAL_N = 32768
LAW_SEARCH_N = 32768
RISK_ANCHOR_N = 32768
SUPPORT_SCREEN_N = 8192
SUPPORT_AUDIT_N = 16384
SUPPORT_PAIRS = 4
CANDIDATE_COUNT = 2048
CANDIDATE_BATCH_SIZE = 8
ROLLOUT_BATCH_SIZE = 2048
NODE7 = 7
MINIMUM_RESS = 0.05
HISTORICAL_R_LAW = 5.186549474478042
ALLOWANCES = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
BOX = (2.0, 1.0)
ATTEMPTS = ("A", "B", "C")


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _derive_seed(role: str) -> dict[str, Any]:
    text = f"{SEED_NAMESPACE}|{GLOBAL_SEED}|{role}"
    digest = hashlib.sha256(text.encode()).hexdigest()
    return {"role": role, "derivation_text": text, "sha256": digest, "seed": int(digest[:16], 16) % (2**31 - 1)}


def _inside(path: Path) -> Path:
    resolved, root = path.resolve(), OUTPUT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"output must remain beneath {root}: {resolved}")
    return resolved


def _atomic_bytes(path: Path, data: bytes) -> None:
    path = _inside(path)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"refusing to overwrite sealed artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_bytes(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")


def _atomic_text(path: Path, value: str) -> None:
    _atomic_bytes(path, value.encode())


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path = _inside(path)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite sealed artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **{key: np.asarray(value) for key, value in arrays.items()})
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _distribution(values: Any) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {"minimum": float(np.min(x)), "p10": float(np.quantile(x, 0.10)), "median": float(np.median(x)), "mean": float(np.mean(x)), "p90": float(np.quantile(x, 0.90)), "maximum": float(np.max(x))}


def _historical_anchors(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    spec_path = HISTORICAL_COVERAGE_ROOT / "generator_spec.json"
    spec = _json(spec_path)
    anchors = [{"anchor_id": "historical_Law", "eta": [float(x) for x in cfg["envelope"]["law_eta"]], "source": "immutable config historical Law"}]
    seen = {eta_key(anchors[0]["eta"])}
    for row in spec["anchors"]:
        key = eta_key(row["eta"])
        if key in seen:
            continue
        seen.add(key)
        anchors.append({"anchor_id": row["anchor_id"], "eta": [float(x) for x in row["eta"]], "source": row["source"]})
        if len(anchors) == 16:
            break
    if len(anchors) < 8:
        raise RuntimeError("insufficient sealed historical geometry context")
    return anchors


def verify_and_seal_sources() -> dict[str, Any]:
    if file_sha256(CONFIG_PATH) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("authoritative config changed")
    if file_sha256(BASELINE_CHECKPOINT_PATH) != EXPECTED_BASELINE_CHECKPOINT_SHA256:
        raise RuntimeError("immutable production reference changed")
    bridge_summary = _json(BRIDGE_ROOT / "summary.json")
    if bridge_summary["interpretation"] != "PERMUTATION_MATCHING_SUFFICIENT" or bridge_summary["candidate_family"] != "B1":
        raise RuntimeError("B1 development selection evidence changed")
    fixed = {
        "config": file_sha256(CONFIG_PATH), "physical_simulator": file_sha256(ROOT / "domain.py"),
        "reference_architecture": file_sha256(ROOT / "reference.py"), "scientific_risk": file_sha256(ROOT / "risk.py"),
        "projection": file_sha256(REPO_ROOT / "src" / "mfsi" / "projection.py"), "forcing": file_sha256(ROOT / "forcing.py"),
        "measurement": file_sha256(ROOT / "measurements.py"), "bridge_ablation_summary": file_sha256(BRIDGE_ROOT / "summary.json"),
        "bridge_ablation_report": file_sha256(BRIDGE_ROOT / "report.md"),
        "historical_geometry_spec": file_sha256(HISTORICAL_COVERAGE_ROOT / "generator_spec.json"),
        "production_reference_context_only": file_sha256(BASELINE_CHECKPOINT_PATH),
    }
    sources = [Path(__file__), ROOT / "single_reference_b1_preflight_run.py", ROOT / "test_single_reference_b1_preflight.py"]
    payload = {
        "schema_version": 1, "version": VERSION, "development_only": True,
        "immutable_definition_hashes": fixed,
        "analysis_source_hashes": {str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in sources},
        "physical_benchmark_changed": False, "B1_particle_matching": True, "configuration_OT": False,
        "scientific_definitions_changed": False, "validation_accessed": False,
        "official_protocol_created": False, "production_reference_replaced": False,
    }
    if SOURCE_SEAL_PATH.exists():
        if _json(SOURCE_SEAL_PATH) != payload:
            raise RuntimeError("source seal changed")
        return payload
    _atomic_json(SOURCE_SEAL_PATH, payload)
    return payload


def freeze_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    verify_and_seal_sources()
    if MANIFEST_PATH.exists() or MANIFEST_HASH_PATH.exists():
        if not MANIFEST_PATH.exists() or not MANIFEST_HASH_PATH.exists():
            raise RuntimeError("incomplete experiment manifest seal")
        if file_sha256(MANIFEST_PATH) != MANIFEST_HASH_PATH.read_text().strip():
            raise RuntimeError("experiment manifest changed")
        return _json(MANIFEST_PATH)
    base = ReferenceTrainingConfig(**cfg["reference_training"])
    required_training = {"hidden_width": 64, "hidden_layers": 3, "train_steps": 6000, "batch_size": 512, "learning_rate": 8e-4, "min_learning_rate_ratio": 0.08, "grad_clip_norm": 8.0, "bridge_noise_std": 0.01}
    if any(asdict(base)[key] != value for key, value in required_training.items()):
        raise RuntimeError("reference architecture/training configuration changed")
    roles = [
        "reference_endpoint_training_data", "endpoint_qualification_holdout", "bridge_qualification",
        "reference_training_A", "reference_training_B", "reference_training_C", "design_truth_selection",
        "selection_observation_noise", "law_search_pool", "law_risk_search_bank", "law_risk_anchor_bank",
        "candidate_generator_local", "candidate_generator_tangent", "candidate_generator_paths", "candidate_generator_sobol",
    ]
    roles += [f"support_{kind}_{index}" for index in range(SUPPORT_PAIRS) for kind in ("screen", "audit")]
    seeds = {role: _derive_seed(role) for role in roles}
    if len({row["seed"] for row in seeds.values()}) != len(seeds):
        raise RuntimeError("prospective seed collision")
    anchors = _historical_anchors(cfg)
    payload = {
        "schema_version": 1, "version": VERSION, "development_only": True,
        "seed_namespace": SEED_NAMESPACE, "source_seal_sha256": file_sha256(SOURCE_SEAL_PATH),
        "seeds": seeds, "all_seeds_frozen_before_generation": True,
        "data": {
            "reference_endpoint_training": {"count": TRAIN_N, "persisted_time_slices": [0.0, 1.0]},
            "endpoint_qualification": {"count": QUAL_N, "persisted_time_slices": [0.0, 1.0]},
            "design_truth_selection": {"count": DESIGN_N, "time_nodes": TIME_COUNT, "role": "development selection truth, never reference training"},
            "truth_substeps_per_interval": int(cfg["physics"]["truth_substeps"]),
        },
        "B1_bridge": {
            "family": "PARTICLE_MATCH_ONLY", "configuration_OT": False,
            "configuration_pairing": "four frozen independent random full-bank pairing maps",
            "pairing_map_count": COUPLING_MAPS, "assignment": "exact scipy.optimize.linear_sum_assignment on 16x16 authoritative periodic squared-distance cost",
            "bridge_noise": 0.01, "target_endpoint_only_reindexed": True,
        },
        "reference_training": {
            "attempt_order": list(ATTEMPTS), "first_passing_seed_wins": True, "train_later_attempts_after_acceptance": False,
            "attempts": {attempt: asdict(replace(base, seed=seeds[f"reference_training_{attempt}"]["seed"])) for attempt in ATTEMPTS},
        },
        "endpoint_qualification": {
            "CFM_examples": CFM_QUAL_N, "CFM_loss_maximum": 0.20, "endpoint_Psi_L2_maximum": 0.020,
            "endpoint_whitened_Psi_norm_maximum": 1.50, "endpoint_Law_Phi_L2_maximum": 0.005,
            "Law_Phi_geometry": "historical frozen Law (new Law does not exist at qualification time)",
            "configuration_distribution_metric": None, "decision_inputs": "endpoint/bridge only",
        },
        "whitening": {
            "rule": "rebuild with unchanged whitening_from_truth over fresh selection/design truth because authoritative implementation derives M from selection truth",
            "feature_definition_changed": False, "sealed_before_Law_search": True,
        },
        "reference_banks": {
            "law_search": LAW_SEARCH_N, "risk_anchor": RISK_ANCHOR_N,
            "support_pairs": [{"pair": index, "screen": SUPPORT_SCREEN_N, "audit": SUPPORT_AUDIT_N} for index in range(SUPPORT_PAIRS)],
            "rollout": "deterministic periodic RK4", "substeps_per_scientific_interval": 14,
        },
        "Law_search": {
            "historical_context": anchors,
            "initial_components": {"Sobol_global": 512, "local_historical_Law": 512, "historical_low_risk_local": 512},
            "local_refinement": {"rounds": 3, "centers": 16, "count_per_center": 8, "scales": [0.02, 0.01, 0.005]},
            "shortlist": {"top_by_search_risk": 16, "diverse_near_best_maximum": 8, "near_best_ratio": 1.02},
            "authoritative_choice": "minimum risk-anchor-bank projected scientific risk among numerically support-valid shortlist; deterministic candidate_id tie-break",
            "rESS_optimized": False, "historical_R_Law_used": False,
        },
        "candidate_generator": {
            "count": CANDIDATE_COUNT,
            "component_targets": {"mandatory_and_local": 717, "risk_tangent": 512, "periodic_paths": 409, "Sobol_global": 410},
            "risk_tangent_gradient": "central finite difference of exact risk-anchor-bank B1 scientific risk at new Law; epsilon=1e-4",
            "canonicalization": "periodic wrap plus exhaustive unordered-sensor matching to new B1 Law",
            "minimum_sensor_separation": float(cfg["measurement"]["min_separation"]),
            "diverse_survivor_rule": "deterministic robust-rESS-descending greedy set with symmetry-aware distance >=0.02",
            "frozen_before_support": True,
        },
        "allowances_percent": list(ALLOWANCES), "allowance_rule": "risk <= (1+p/100)*R_Law_B1 exactly",
        "support": {"pairs": SUPPORT_PAIRS, "pair_pass": "both screen and audit pass all unchanged gates", "minimum_rESS": MINIMUM_RESS, "average_screen_audit": False},
        "classification": {
            "READY": "qualified first reference; new Law; Law all 8 banks and minimum>=0.10; >=10 all-pair candidates at 0.5%; >=1 all-pair candidate at every >=1%; no population-wide non-rESS numerical pathology",
            "BORDERLINE": "reference and Law pass ordinary 0.05 but Law near boundary or 0.5% set thin",
            "REFERENCE_QUALIFICATION_FAILED": "all three sequential endpoint-only attempts fail",
            "LAW_RECONSTRUCTION_FAILED": "qualified reference but no stable valid new Law",
            "SUPPORT_PROBLEM_PERSISTS": "qualified reference and Law but widespread fresh support failure",
        },
        "validation_access_permitted": False, "Tangent_permitted": False, "Full_permitted": False,
        "Galerkin_Kf_permitted": False, "Deep_Ritz_permitted": False, "official_protocol_permitted": False,
    }
    _atomic_json(MANIFEST_PATH, payload)
    _atomic_text(MANIFEST_HASH_PATH, file_sha256(MANIFEST_PATH) + "\n")
    return payload


def _load_endpoints(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) - {"endpoint0", "endpoint1", "seed"}:
            raise RuntimeError("endpoint artifact contains a non-endpoint state")
        return np.asarray(arrays["endpoint0"], dtype=np.float64), np.asarray(arrays["endpoint1"], dtype=np.float64)


def generate_clean_data(cfg: dict[str, Any], progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    manifest = freeze_manifest(cfg)
    physics = SkyrmionTruth(_physics_config(cfg))
    outputs = []
    endpoint_specs = (
        ("train", TRAIN_N, "reference_endpoint_training_data", TRAIN_DATA_PATH, TRAIN_DATA_MANIFEST_PATH),
        ("qualification", QUAL_N, "endpoint_qualification_holdout", QUAL_DATA_PATH, QUAL_DATA_MANIFEST_PATH),
    )
    for label, count, role, path, record_path in endpoint_specs:
        if path.exists() or record_path.exists():
            if not path.exists() or not record_path.exists() or file_sha256(path) != _json(record_path)["sha256"]:
                raise RuntimeError(f"{label} endpoint cache seal mismatch")
            outputs.append(_json(record_path))
            continue
        started = time.perf_counter()
        seed = int(manifest["seeds"][role]["seed"])
        bank = physics.make_bank(seed=seed, samples=count, times=jnp.asarray([0.0, 1.0], dtype=jnp.float64), substeps_per_interval=int(cfg["physics"]["truth_substeps"]) * (TIME_COUNT - 1))
        endpoint0, endpoint1 = np.asarray(bank.configurations[0]), np.asarray(bank.configurations[-1])
        del bank
        _atomic_npz(path, endpoint0=endpoint0, endpoint1=endpoint1, seed=np.asarray(seed, dtype=np.int64))
        record = {
            "schema_version": 1, "role": label, "generated": True, "count": count, "seed": seed,
            "retained_time_slices": [0.0, 1.0], "retained_intermediate_truth": False,
            "endpoint0_sha256": _array_sha256(endpoint0), "endpoint1_sha256": _array_sha256(endpoint1),
            "physical_simulator_sha256": file_sha256(ROOT / "domain.py"), "config_sha256": file_sha256(CONFIG_PATH),
            "sha256": file_sha256(path), "wall_time_seconds": time.perf_counter() - started,
        }
        _atomic_json(record_path, record)
        outputs.append(record)
        if progress:
            progress(f"generated {label} endpoints: {count}")
    train0, train1 = _load_endpoints(TRAIN_DATA_PATH)
    qual0, qual1 = _load_endpoints(QUAL_DATA_PATH)
    if len({manifest["seeds"]["reference_endpoint_training_data"]["seed"], manifest["seeds"]["endpoint_qualification_holdout"]["seed"]}) != 2:
        raise RuntimeError("training and qualification endpoint seeds collide")
    if _array_sha256(train0) in {_array_sha256(qual0), _array_sha256(qual1)} or _array_sha256(train1) in {_array_sha256(qual0), _array_sha256(qual1)}:
        raise RuntimeError("endpoint qualification artifact duplicates training data")
    if DESIGN_DATA_PATH.exists() or DESIGN_DATA_MANIFEST_PATH.exists():
        if not DESIGN_DATA_PATH.exists() or not DESIGN_DATA_MANIFEST_PATH.exists() or file_sha256(DESIGN_DATA_PATH) != _json(DESIGN_DATA_MANIFEST_PATH)["sha256"]:
            raise RuntimeError("design truth cache seal mismatch")
    else:
        started = time.perf_counter()
        seed = int(manifest["seeds"]["design_truth_selection"]["seed"])
        times = jnp.linspace(0.0, 1.0, TIME_COUNT, dtype=jnp.float64)
        bank = physics.make_bank(seed=seed, samples=DESIGN_N, times=times, substeps_per_interval=int(cfg["physics"]["truth_substeps"]))
        configurations = np.asarray(bank.configurations, dtype=np.float64)
        del bank
        features = many_body_features(jnp.asarray(configurations), BOX)
        truth_means = np.asarray(jnp.mean(features, axis=1), dtype=np.float64)
        whitening = np.asarray(whitening_from_truth(features), dtype=np.float64)
        _atomic_npz(DESIGN_DATA_PATH, times=np.asarray(times), configurations=configurations, truth_means=truth_means, whitening=whitening, seed=np.asarray(seed, dtype=np.int64))
        record = {
            "schema_version": 1, "role": "development_selection_truth", "generated": True, "count": DESIGN_N,
            "seed": seed, "time_nodes": TIME_COUNT, "contains_intermediate_selection_truth": True,
            "passed_to_reference_training": False, "whitening_rule": "unchanged whitening_from_truth",
            "truth_means_sha256": _array_sha256(truth_means), "whitening_sha256": _array_sha256(whitening),
            "scientific_feature_source_sha256": file_sha256(ROOT / "risk.py"), "sha256": file_sha256(DESIGN_DATA_PATH),
            "wall_time_seconds": time.perf_counter() - started,
        }
        _atomic_json(DESIGN_DATA_MANIFEST_PATH, record)
        if progress:
            progress(f"generated design truth: {DESIGN_N}")
    return {"endpoint_train": _json(TRAIN_DATA_MANIFEST_PATH), "endpoint_qualification": _json(QUAL_DATA_MANIFEST_PATH), "design_truth": _json(DESIGN_DATA_MANIFEST_PATH)}


def _build_pair_maps(path: Path, endpoint0: np.ndarray, endpoint1: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as arrays:
            return {key: np.asarray(arrays[key]) for key in arrays.files}
    count = len(endpoint0)
    source_rows, target_rows, permutation_rows = [], [], []
    for map_index in range(COUPLING_MAPS):
        rng = np.random.default_rng(seed + 104729 * map_index)
        source, target = rng.permutation(count).astype(np.int32), rng.permutation(count).astype(np.int32)
        permutations = np.empty((count, 16), dtype=np.int16)
        for slot in range(count):
            permutations[slot], _ = particle_assignment(endpoint0[source[slot]], endpoint1[target[slot]])
        source_rows.append(source); target_rows.append(target); permutation_rows.append(permutations)
    arrays = {"source_index": np.stack(source_rows), "target_index": np.stack(target_rows), "particle_permutation": np.stack(permutation_rows)}
    _atomic_npz(path, **arrays)
    return arrays


def build_b1_couplings(cfg: dict[str, Any]) -> dict[str, Any]:
    generate_clean_data(cfg)
    manifest = freeze_manifest(cfg)
    train0, train1 = _load_endpoints(TRAIN_DATA_PATH)
    arrays = _build_pair_maps(COUPLING_PATH, train0, train1, int(manifest["seeds"]["reference_endpoint_training_data"]["seed"]) + 17)
    qual0, qual1 = _load_endpoints(QUAL_DATA_PATH)
    _build_pair_maps(QUAL_COUPLING_PATH, qual0, qual1, int(manifest["seeds"]["bridge_qualification"]["seed"]))
    identity = np.arange(16)
    nonidentity = float(np.mean(np.any(arrays["particle_permutation"] != identity, axis=-1)))
    payload = {
        "schema_version": 1, "family": "B1_PARTICLE_MATCH_ONLY", "configuration_OT": False,
        "map_count": COUPLING_MAPS, "training_endpoint_sha256": file_sha256(TRAIN_DATA_PATH),
        "coupling_sha256": file_sha256(COUPLING_PATH), "qualification_coupling_sha256": file_sha256(QUAL_COUPLING_PATH),
        "particle_assignment_rule_sha256": file_sha256(ROOT / "bridge_ablation.py"),
        "nonidentity_pair_fraction": nonidentity, "endpoint_marginals_preserved": True,
    }
    _atomic_json(COUPLING_MANIFEST_PATH, payload)
    return payload


def _attempt_checkpoint(attempt: str) -> Path:
    return OUTPUT_ROOT / "reference_training" / f"attempt_{attempt}" / "reference.npz"


def _attempt_record(attempt: str) -> Path:
    return _attempt_checkpoint(attempt).with_name("qualification.json")


def _attempt_history(attempt: str) -> Path:
    return _attempt_checkpoint(attempt).with_name("training_history.json")


def train_b1_reference(endpoint0: np.ndarray, endpoint1: np.ndarray, pair_maps: dict[str, np.ndarray], cfg: ReferenceTrainingConfig) -> tuple[EquivariantReferenceFlow, list[dict[str, float]]]:
    endpoint0_j, endpoint1_j = jnp.asarray(endpoint0), jnp.asarray(endpoint1)
    source, target, permutations = (jnp.asarray(pair_maps[key]) for key in ("source_index", "target_index", "particle_permutation"))
    key = jax.random.PRNGKey(int(cfg.seed)); key, init_key = jax.random.split(key)
    params = init_equivariant_reference(init_key, hidden_width=cfg.hidden_width, hidden_layers=cfg.hidden_layers)
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    state = _AdamState(zeros, zeros, jnp.asarray(0, dtype=jnp.int32))

    def sample_batch(batch_key):
        kt, km, ks, kz = jax.random.split(batch_key, 4)
        count = int(cfg.batch_size)
        map_index = jax.random.randint(km, (count,), 0, COUPLING_MAPS)
        slot = jax.random.randint(ks, (count,), 0, len(endpoint0))
        x0 = endpoint0_j[source[map_index, slot]]
        x1 = endpoint1_j[target[map_index, slot]]
        x1 = jnp.take_along_axis(x1, permutations[map_index, slot, :, None], axis=1)
        t = jax.random.uniform(kt, (count,), dtype=jnp.float64)
        delta = x1 - x0
        delta -= jnp.asarray(BOX) * jnp.floor(delta / jnp.asarray(BOX) + 0.5)
        noise = jax.random.normal(kz, x0.shape, dtype=jnp.float64)
        gamma = cfg.bridge_noise_std * jnp.sin(jnp.pi * t)[:, None, None]
        gamma_dot = cfg.bridge_noise_std * jnp.pi * jnp.cos(jnp.pi * t)[:, None, None]
        xt = jnp.mod(x0 + t[:, None, None] * delta + gamma * noise, jnp.asarray(BOX))
        return t, xt, delta + gamma_dot * noise

    def loss_fn(p, t, x, target_velocity):
        predicted = equivariant_velocity(p, t, x, box=BOX)
        return jnp.mean(jnp.sum((predicted - target_velocity) ** 2, axis=(-2, -1)))

    @jax.jit
    def step(p, adam, step_key):
        t, x, target_velocity = sample_batch(step_key)
        loss, grads = jax.value_and_grad(loss_fn)(p, t, x, target_velocity)
        norm = _tree_norm(grads)
        scale = jnp.minimum(1.0, cfg.grad_clip_norm / jnp.maximum(norm, 1e-30))
        grads = jax.tree_util.tree_map(lambda g: scale * g, grads)
        count = adam.step + 1
        beta1, beta2 = 0.9, 0.999
        m = jax.tree_util.tree_map(lambda old, g: beta1 * old + (1 - beta1) * g, adam.m, grads)
        v = jax.tree_util.tree_map(lambda old, g: beta2 * old + (1 - beta2) * g * g, adam.v, grads)
        mhat = jax.tree_util.tree_map(lambda z: z / (1 - beta1**count), m)
        vhat = jax.tree_util.tree_map(lambda z: z / (1 - beta2**count), v)
        fraction = jnp.clip(count / max(float(cfg.train_steps), 1.0), 0.0, 1.0)
        cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * fraction))
        lr = cfg.learning_rate * (cfg.min_learning_rate_ratio + (1.0 - cfg.min_learning_rate_ratio) * cosine)
        p = jax.tree_util.tree_map(lambda q, a, b: q - lr * a / (jnp.sqrt(b) + 1e-8), p, mhat, vhat)
        return p, _AdamState(m, v, count), loss, norm, lr

    history = []; started = time.perf_counter()
    for index in range(1, cfg.train_steps + 1):
        key, step_key = jax.random.split(key)
        params, state, loss, norm, lr = step(params, state, step_key)
        if index == 1 or index % cfg.log_every == 0 or index == cfg.train_steps:
            history.append({"step": index, "loss": float(loss), "gradient_norm": float(norm), "learning_rate": float(lr), "elapsed_seconds": time.perf_counter() - started})
    metadata = {
        "kind": "permutation_equivariant_endpoint_cfm_v1", "endpoint_only": True, "box": list(BOX),
        "training": asdict(cfg), "bridge_family": "B1", "configuration_OT": False,
        "coupling_sha256": file_sha256(COUPLING_PATH), "seed_namespace": SEED_NAMESPACE,
    }
    return EquivariantReferenceFlow(params, box=BOX, metadata=metadata), history


def _rollout(flow: EquivariantReferenceFlow, initial: np.ndarray, cfg: dict[str, Any], *, final_only: bool) -> np.ndarray:
    times = jnp.linspace(0.0, 1.0, TIME_COUNT, dtype=jnp.float64)
    pieces = []
    for start in range(0, len(initial), ROLLOUT_BATCH_SIZE):
        trajectory = flow.rollout(jnp.asarray(initial[start:start + ROLLOUT_BATCH_SIZE]), times, substeps_per_interval=int(cfg["banks"]["reference_substeps"]))
        pieces.append(np.asarray(trajectory[-1] if final_only else trajectory, dtype=np.float64))
    return np.concatenate(pieces, axis=0 if final_only else 1)


def _fixed_bridge_cfm_loss(flow: EquivariantReferenceFlow, endpoint0: np.ndarray, endpoint1: np.ndarray, pair_maps: dict[str, np.ndarray], seed: int) -> float:
    rng = np.random.default_rng(seed)
    map_index = rng.integers(0, COUPLING_MAPS, size=CFM_QUAL_N, dtype=np.int32)
    slot = rng.integers(0, len(endpoint0), size=CFM_QUAL_N, dtype=np.int32)
    t_all = rng.random(CFM_QUAL_N); noise_all = rng.normal(size=(CFM_QUAL_N, 16, 2))
    source, target_index, permutations = (pair_maps[key] for key in ("source_index", "target_index", "particle_permutation"))
    total = 0.0
    for start in range(0, CFM_QUAL_N, 512):
        stop = min(start + 512, CFM_QUAL_N); m, s = map_index[start:stop], slot[start:stop]
        x0 = endpoint0[source[m, s]]; x1 = endpoint1[target_index[m, s]]
        x1 = np.take_along_axis(x1, permutations[m, s, :, None], axis=1)
        delta = x1 - x0; delta -= np.asarray(BOX) * np.floor(delta / np.asarray(BOX) + 0.5)
        t, noise = t_all[start:stop], noise_all[start:stop]
        xt = np.mod(x0 + t[:, None, None] * delta + 0.01 * np.sin(np.pi * t)[:, None, None] * noise, np.asarray(BOX))
        target_velocity = delta + 0.01 * np.pi * np.cos(np.pi * t)[:, None, None] * noise
        predicted = np.asarray(flow.velocity(jnp.asarray(xt), jnp.asarray(t)))
        total += float(np.sum(np.sum((predicted - target_velocity) ** 2, axis=(-2, -1))))
    return total / CFM_QUAL_N


def _load_design_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(DESIGN_DATA_PATH, allow_pickle=False) as arrays:
        return tuple(np.asarray(arrays[key], dtype=np.float64) for key in ("times", "configurations", "truth_means", "whitening"))  # type: ignore[return-value]


def _qualification_metrics(flow: EquivariantReferenceFlow, cfg: dict[str, Any]) -> dict[str, Any]:
    manifest = freeze_manifest(cfg)
    endpoint0, endpoint1 = _load_endpoints(QUAL_DATA_PATH)
    with np.load(QUAL_COUPLING_PATH, allow_pickle=False) as arrays:
        pair_maps = {key: np.asarray(arrays[key]) for key in arrays.files}
    _, _, _, whitening = _load_design_arrays()
    family = _family(cfg); historical_law = np.asarray(cfg["envelope"]["law_eta"], dtype=np.float64)
    target_psi, target_phi = _mean_psi(endpoint1, BOX), _mean_phi(endpoint1, family, historical_law)
    final = _rollout(flow, endpoint0, cfg, final_only=True)
    psi, phi = _mean_psi(final, BOX), _mean_phi(final, family, historical_law)
    delta_psi, delta_phi = psi - target_psi, phi - target_phi
    metrics = {
        "CFM_qualification_loss": _fixed_bridge_cfm_loss(flow, endpoint0, endpoint1, pair_maps, int(manifest["seeds"]["bridge_qualification"]["seed"])),
        "endpoint_Psi_errors": delta_psi.tolist(), "endpoint_Psi_L2": float(np.linalg.norm(delta_psi)),
        "endpoint_whitened_Psi_norm": float(np.sqrt(delta_psi @ whitening @ delta_psi)),
        "endpoint_Law_Phi_errors": delta_phi.tolist(), "endpoint_Law_Phi_L2": float(np.linalg.norm(delta_phi)),
        "configuration_distribution_metric": None, "common_holdout_sha256": file_sha256(QUAL_DATA_PATH),
        "final_state_sha256": _array_sha256(final),
    }
    thresholds = manifest["endpoint_qualification"]
    gates = {
        "CFM": metrics["CFM_qualification_loss"] <= thresholds["CFM_loss_maximum"],
        "Psi_L2": metrics["endpoint_Psi_L2"] <= thresholds["endpoint_Psi_L2_maximum"],
        "whitened_Psi": metrics["endpoint_whitened_Psi_norm"] <= thresholds["endpoint_whitened_Psi_norm_maximum"],
        "Law_Phi_L2": metrics["endpoint_Law_Phi_L2"] <= thresholds["endpoint_Law_Phi_L2_maximum"],
    }
    return {**metrics, "gates": gates, "passed": all(gates.values()), "decision_uses_intermediate_truth": False}


def train_and_accept_reference(cfg: dict[str, Any], progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    build_b1_couplings(cfg)
    manifest = freeze_manifest(cfg)
    if ACCEPTED_REFERENCE_PATH.exists():
        accepted = _json(ACCEPTED_REFERENCE_PATH)
        if file_sha256(OUTPUT_ROOT / accepted["checkpoint_path"]) != accepted["checkpoint_sha256"]:
            raise RuntimeError("accepted checkpoint changed")
        later = ATTEMPTS[ATTEMPTS.index(accepted["attempt"]) + 1:]
        if any(_attempt_checkpoint(item).exists() for item in later):
            raise RuntimeError("a later reference was trained after acceptance")
        return accepted
    endpoint0, endpoint1 = _load_endpoints(TRAIN_DATA_PATH)
    with np.load(COUPLING_PATH, allow_pickle=False) as arrays:
        pair_maps = {key: np.asarray(arrays[key]) for key in arrays.files}
    attempted = []
    for attempt in ATTEMPTS:
        checkpoint, record_path = _attempt_checkpoint(attempt), _attempt_record(attempt)
        if checkpoint.exists() or record_path.exists():
            if not checkpoint.exists() or not record_path.exists():
                raise RuntimeError(f"incomplete attempt {attempt}")
            record = _json(record_path)
            if file_sha256(checkpoint) != record["checkpoint_sha256"]:
                raise RuntimeError(f"attempt {attempt} checkpoint changed")
            attempted.append(record)
        else:
            started = time.perf_counter()
            train_cfg = ReferenceTrainingConfig(**manifest["reference_training"]["attempts"][attempt])
            flow, history = train_b1_reference(endpoint0, endpoint1, pair_maps, train_cfg)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_name("reference.temporary.npz")
            save_reference(temporary, flow); os.replace(temporary, checkpoint)
            _atomic_json(_attempt_history(attempt), history)
            metrics = _qualification_metrics(flow, cfg)
            record = {
                "schema_version": 1, "attempt": attempt, "training_seed": train_cfg.seed,
                "checkpoint_path": str(checkpoint.relative_to(OUTPUT_ROOT)), "checkpoint_sha256": file_sha256(checkpoint),
                "endpoint_training_data_sha256": file_sha256(TRAIN_DATA_PATH), "B1_bridge_sha256": file_sha256(COUPLING_PATH),
                "qualification_holdout_sha256": file_sha256(QUAL_DATA_PATH), "training_config_sha256": _payload_sha256(asdict(train_cfg)),
                "training_steps": train_cfg.train_steps, "final_training_loss": history[-1]["loss"],
                "qualification": metrics, "status": "PASS" if metrics["passed"] else "REFERENCE_QUALIFICATION_FAILED",
                "wall_time_seconds": time.perf_counter() - started,
            }
            _atomic_json(record_path, record); attempted.append(record)
            if progress:
                progress(f"reference attempt {attempt}: {record['status']}")
            del flow; gc.collect()
        if record["qualification"]["passed"]:
            accepted = {
                "schema_version": 1, "accepted": True, "attempt": attempt, "training_seed": record["training_seed"],
                "checkpoint_path": record["checkpoint_path"], "checkpoint_sha256": record["checkpoint_sha256"],
                "endpoint_training_data_sha256": record["endpoint_training_data_sha256"], "B1_bridge_sha256": record["B1_bridge_sha256"],
                "qualification_holdout_sha256": record["qualification_holdout_sha256"], "qualification_metrics": record["qualification"],
                "training_config_sha256": record["training_config_sha256"], "first_passing_seed_rule": True,
                "further_reference_training_prohibited": True, "attempts_made": [row["attempt"] for row in attempted],
            }
            _atomic_json(ACCEPTED_REFERENCE_PATH, accepted)
            return accepted
    failure = {"schema_version": 1, "accepted": False, "classification": "B1_REFERENCE_QUALIFICATION_FAILED", "attempts_made": [row["attempt"] for row in attempted]}
    _atomic_json(ACCEPTED_REFERENCE_PATH, failure)
    return failure


def _bank_specs(manifest: dict[str, Any]) -> list[tuple[str, int, int]]:
    specs = [
        ("law_search", LAW_SEARCH_N, int(manifest["seeds"]["law_risk_search_bank"]["seed"])),
        ("risk_anchor", RISK_ANCHOR_N, int(manifest["seeds"]["law_risk_anchor_bank"]["seed"])),
    ]
    for index in range(SUPPORT_PAIRS):
        specs.append((f"screen_{index}", SUPPORT_SCREEN_N, int(manifest["seeds"][f"support_screen_{index}"]["seed"])))
        specs.append((f"audit_{index}", SUPPORT_AUDIT_N, int(manifest["seeds"][f"support_audit_{index}"]["seed"])))
    return specs


def _bank_path(label: str) -> Path:
    return OUTPUT_ROOT / "reference_banks" / f"{label}.npz"


def _load_bank(label: str) -> GalerkinReferenceBank:
    with np.load(_bank_path(label), allow_pickle=False) as arrays:
        return GalerkinReferenceBank(jnp.asarray(arrays["configurations"]), jnp.asarray(arrays["velocity"]), jnp.asarray(arrays["base_weights"]))


def generate_reference_banks(cfg: dict[str, Any], progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    accepted = train_and_accept_reference(cfg)
    if not accepted.get("accepted"):
        return accepted
    manifest = freeze_manifest(cfg); checkpoint = OUTPUT_ROOT / accepted["checkpoint_path"]
    rows = []
    for label, count, seed in _bank_specs(manifest):
        path, record_path = _bank_path(label), _bank_path(label).with_suffix(".json")
        if path.exists() or record_path.exists():
            if not path.exists() or not record_path.exists() or file_sha256(path) != _json(record_path)["sha256"]:
                raise RuntimeError(f"reference bank cache seal mismatch: {label}")
            record = _json(record_path)
            if record["checkpoint_sha256"] != accepted["checkpoint_sha256"]:
                raise RuntimeError("reference bank uses a non-accepted checkpoint")
            rows.append(record); continue
        started = time.perf_counter(); truth = SkyrmionTruth(_physics_config(cfg))
        initial = np.asarray(truth.sample_initial(jax.random.PRNGKey(seed), count), dtype=np.float64)
        flow = load_reference(checkpoint); configurations = _rollout(flow, initial, cfg, final_only=False)
        velocities = []; times = jnp.linspace(0.0, 1.0, TIME_COUNT, dtype=jnp.float64)
        for start in range(0, count, ROLLOUT_BATCH_SIZE):
            velocities.append(np.asarray(flow.velocity(jnp.asarray(configurations[:, start:start + ROLLOUT_BATCH_SIZE]), times), dtype=np.float64))
        velocity = np.concatenate(velocities, axis=1); weights = np.full((TIME_COUNT, count), 1.0 / count, dtype=np.float64)
        if not np.array_equal(configurations[0], initial):
            raise RuntimeError("reference rollout changed initial P0")
        _atomic_npz(path, configurations=configurations, velocity=velocity, base_weights=weights, initial_P0=initial)
        record = {
            "schema_version": 1, "label": label, "N": count, "seed": seed,
            "checkpoint_sha256": accepted["checkpoint_sha256"], "initial_P0_sha256": _array_sha256(initial),
            "rollout": "deterministic periodic RK4", "substeps_per_scientific_interval": 14,
            "sha256": file_sha256(path), "wall_time_seconds": time.perf_counter() - started,
        }
        _atomic_json(record_path, record); rows.append(record)
        if progress:
            progress(f"reference bank {label}: N={count}")
        del flow, configurations, velocity, weights, initial; gc.collect()
    payload = {
        "schema_version": 1, "accepted_checkpoint_sha256": accepted["checkpoint_sha256"],
        "all_banks_use_accepted_checkpoint": len({row["checkpoint_sha256"] for row in rows}) == 1,
        "independent_seeds": len({row["seed"] for row in rows}) == len(rows), "banks": rows,
    }
    _atomic_json(REFERENCE_BANK_MANIFEST_PATH, payload)
    return payload


def _selection_context(cfg: dict[str, Any], bank_label: str) -> tuple[SelectionGalerkinData, np.ndarray, np.ndarray]:
    times, configurations, truth_means, whitening = _load_design_arrays()
    manifest = freeze_manifest(cfg); family = _family(cfg)
    problem = _make_problem(cfg, jnp.asarray(configurations), jnp.asarray(times), family, noise_seed=int(manifest["seeds"]["selection_observation_noise"]["seed"]))
    bank = _load_bank(bank_label)
    reference_features = many_body_features(bank.configurations, BOX)
    data = SelectionGalerkinData(problem, bank, bank, bank, reference_features, jnp.asarray(truth_means), jnp.asarray(whitening))
    return data, truth_means, whitening


def _consider_geometry(rows: list[dict[str, Any]], seen: set[str], eta: Any, reference: np.ndarray, cfg: dict[str, Any], metadata: dict[str, Any]) -> bool:
    canonical = canonicalize_eta(eta, reference, BOX)
    if minimum_periodic_separation(canonical, BOX) < float(cfg["measurement"]["min_separation"]):
        return False
    key = eta_key(canonical)
    if key in seen:
        return False
    seen.add(key); rows.append({"candidate_id": f"law_{len(rows):05d}", "eta": canonical.tolist(), "eta_sha256": key, **metadata})
    return True


def generate_law_search_pool(cfg: dict[str, Any]) -> dict[str, Any]:
    generate_reference_banks(cfg)
    if LAW_POOL_PATH.exists():
        saved = _json(LAW_POOL_PATH)
        if saved["rows_sha256"] != _payload_sha256(saved["rows"]):
            raise RuntimeError("Law pool rows changed")
        return saved
    manifest = freeze_manifest(cfg); anchors = manifest["Law_search"]["historical_context"]
    historical_law = np.asarray(anchors[0]["eta"], dtype=np.float64); rows = []; seen: set[str] = set()
    for anchor in anchors:
        _consider_geometry(rows, seen, anchor["eta"], historical_law, cfg, {"component": "mandatory_historical_context", "anchor_id": anchor["anchor_id"]})
    rng = np.random.default_rng(int(manifest["seeds"]["law_search_pool"]["seed"]))
    local_start = len(rows); local_scales = (0.002, 0.005, 0.01, 0.02, 0.04, 0.08)
    while len(rows) - local_start < 512:
        scale = local_scales[(len(rows) - local_start) % len(local_scales)]
        raw = historical_law + scale * rng.normal(size=historical_law.shape)
        _consider_geometry(rows, seen, raw, historical_law, cfg, {"component": "local_historical_Law", "scale": scale})
    informed_start = len(rows); informed_scales = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02)
    anchor_index = 0
    while len(rows) - informed_start < 512:
        anchor = np.asarray(anchors[anchor_index % len(anchors)]["eta"], dtype=np.float64)
        scale = informed_scales[anchor_index % len(informed_scales)]
        _consider_geometry(rows, seen, anchor + scale * rng.normal(size=anchor.shape), historical_law, cfg, {"component": "historical_low_risk_local", "anchor_id": anchors[anchor_index % len(anchors)]["anchor_id"], "scale": scale})
        anchor_index += 1
    global_start = len(rows); sobol = qmc.Sobol(d=8, scramble=True, seed=int(manifest["seeds"]["law_search_pool"]["seed"]) + 1)
    for index, point in enumerate(sobol.random_base2(m=13)):
        if len(rows) - global_start >= 512:
            break
        _consider_geometry(rows, seen, point * np.tile(np.asarray(BOX), 4), historical_law, cfg, {"component": "Sobol_global", "sobol_index": index})
    if len(rows) - global_start != 512:
        raise RuntimeError("Law Sobol reservoir insufficient")
    payload = {
        "schema_version": 1, "manifest_sha256": file_sha256(MANIFEST_PATH), "count": len(rows),
        "component_counts": {name: sum(row["component"] == name for row in rows) for name in {row["component"] for row in rows}},
        "canonical_unique": len(seen) == len(rows), "minimum_separation": float(cfg["measurement"]["min_separation"]),
        "rows_sha256": _payload_sha256(rows), "frozen_before_risk_evaluation": True, "rows": rows,
    }
    _atomic_json(LAW_POOL_PATH, payload)
    return payload


def _evaluate_risks(data: SelectionGalerkinData, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    risks = _risk_rows_batched(data, rows, CANDIDATE_BATCH_SIZE)
    return [{**row, "scientific_risk": float(risk)} for row, risk in zip(rows, risks, strict=True)]


def _full_eval(data: SelectionGalerkinData, etas: np.ndarray, bank: GalerkinReferenceBank, N: int) -> dict[str, np.ndarray]:
    evaluator = _ReferenceEvaluator(data.selection_problem, np.asarray(data.truth_means), np.asarray(data.whitening))
    return evaluator.evaluate(etas, bank, N)


def reconstruct_and_freeze_law(cfg: dict[str, Any], progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    generate_law_search_pool(cfg)
    if LAW_FREEZE_PATH.exists():
        saved = _json(LAW_FREEZE_PATH)
        if saved["accepted_checkpoint_sha256"] != _json(ACCEPTED_REFERENCE_PATH)["checkpoint_sha256"]:
            raise RuntimeError("Law freeze checkpoint changed")
        return saved
    manifest = freeze_manifest(cfg); data, _, _ = _selection_context(cfg, "law_search")
    pool = _json(LAW_POOL_PATH); evaluated = _evaluate_risks(data, pool["rows"])
    all_rows = list(evaluated); round_records = []
    historical_law = np.asarray(manifest["Law_search"]["historical_context"][0]["eta"])
    for round_index, scale in enumerate(manifest["Law_search"]["local_refinement"]["scales"]):
        ranked = sorted(all_rows, key=lambda row: (row["scientific_risk"], row["candidate_id"]))
        centers = ranked[: int(manifest["Law_search"]["local_refinement"]["centers"])]
        rows = []; seen = {row["eta_sha256"] for row in all_rows}; rng = np.random.default_rng(int(manifest["seeds"]["law_search_pool"]["seed"]) + 1000 + round_index)
        for center in centers:
            produced = 0
            while produced < int(manifest["Law_search"]["local_refinement"]["count_per_center"]):
                raw = np.asarray(center["eta"]) + float(scale) * rng.normal(size=8)
                if _consider_geometry(rows, seen, raw, historical_law, cfg, {"component": "Law_local_refinement", "round": round_index + 1, "center_id": center["candidate_id"], "scale": scale}):
                    produced += 1
        round_path = OUTPUT_ROOT / "law_refinement" / f"round_{round_index + 1}_pool.json"
        round_payload = {"round": round_index + 1, "count": len(rows), "rows_sha256": _payload_sha256(rows), "frozen_before_evaluation": True, "rows": rows}
        _atomic_json(round_path, round_payload)
        scored = _evaluate_risks(data, rows); all_rows.extend(scored)
        round_records.append({"round": round_index + 1, "pool_sha256": file_sha256(round_path), "best_risk": min(row["scientific_risk"] for row in scored)})
        if progress:
            progress(f"Law refinement round {round_index + 1}: {len(rows)} candidates")
    ranked = sorted(all_rows, key=lambda row: (row["scientific_risk"], row["candidate_id"]))
    shortlist = ranked[:16]; near = [row for row in ranked if row["scientific_risk"] <= ranked[0]["scientific_risk"] * 1.02]
    while len(shortlist) < 24:
        remaining = [row for row in near if row["candidate_id"] not in {x["candidate_id"] for x in shortlist}]
        if not remaining:
            break
        chosen = max(remaining, key=lambda row: (min(_symmetry_aware_distance(row["eta"], old["eta"], BOX) for old in shortlist), -row["scientific_risk"], row["candidate_id"]))
        shortlist.append(chosen)
    etas = np.asarray([row["eta"] for row in shortlist], dtype=np.float64)
    search_result = _full_eval(data, etas, _load_bank("law_search"), LAW_SEARCH_N)
    anchor_data, _, _ = _selection_context(cfg, "risk_anchor")
    anchor_result = _full_eval(anchor_data, etas, _load_bank("risk_anchor"), RISK_ANCHOR_N)
    valid = np.asarray(search_result["support_valid"]) & np.asarray(anchor_result["support_valid"])
    if not np.any(valid):
        failure = {"schema_version": 1, "status": "B1_LAW_RECONSTRUCTION_FAILED", "reason": "no shortlist candidate passed normal numerical support on search and anchor banks"}
        _atomic_json(LAW_FREEZE_PATH, failure); return failure
    choices = [index for index in range(len(shortlist)) if valid[index]]
    winner_index = min(choices, key=lambda index: (float(anchor_result["scientific_risk"][index]), shortlist[index]["candidate_id"]))
    winner = shortlist[winner_index]; accepted = _json(ACCEPTED_REFERENCE_PATH)
    freeze = {
        "schema_version": 1, "status": "FROZEN", "eta_Law_B1": winner["eta"], "R_Law_B1": float(anchor_result["scientific_risk"][winner_index]),
        "winner_candidate_id": winner["candidate_id"], "accepted_checkpoint_sha256": accepted["checkpoint_sha256"],
        "design_truth_sha256": file_sha256(DESIGN_DATA_PATH), "law_search_bank_sha256": file_sha256(_bank_path("law_search")),
        "risk_anchor_bank_sha256": file_sha256(_bank_path("risk_anchor")), "whitening_sha256": _json(DESIGN_DATA_MANIFEST_PATH)["whitening_sha256"],
        "search_support_valid": True, "anchor_support_valid": True,
        "search_minimum_rESS": float(search_result["minimum_ress"][winner_index]), "anchor_minimum_rESS": float(anchor_result["minimum_ress"][winner_index]),
        "historical_R_Law": HISTORICAL_R_LAW, "historical_R_Law_status": "NOT USED AS NEW ANCHOR",
        "allowance_ceilings": {str(p): float((1.0 + p / 100.0) * float(anchor_result["scientific_risk"][winner_index])) for p in ALLOWANCES},
        "selection_rule": manifest["Law_search"]["authoritative_choice"],
    }
    results_payload = {
        "schema_version": 1, "initial_pool_sha256": file_sha256(LAW_POOL_PATH), "initial_and_refined_count": len(all_rows),
        "refinement_rounds": round_records, "shortlist": [{**row, "search_support_valid": bool(search_result["support_valid"][i]), "anchor_support_valid": bool(anchor_result["support_valid"][i]), "anchor_risk": float(anchor_result["scientific_risk"][i])} for i, row in enumerate(shortlist)],
    }
    _atomic_json(LAW_RESULTS_PATH, results_payload); _atomic_json(LAW_FREEZE_PATH, freeze)
    return freeze


def freeze_candidate_generator(cfg: dict[str, Any]) -> dict[str, Any]:
    law = reconstruct_and_freeze_law(cfg)
    if law.get("status") != "FROZEN":
        return law
    manifest = freeze_manifest(cfg)
    payload = {
        "schema_version": 1, "manifest_sha256": file_sha256(MANIFEST_PATH), "Law_freeze_sha256": file_sha256(LAW_FREEZE_PATH),
        "accepted_checkpoint_sha256": _json(ACCEPTED_REFERENCE_PATH)["checkpoint_sha256"],
        "count": CANDIDATE_COUNT, "component_targets": manifest["candidate_generator"]["component_targets"],
        "seeds": {key: manifest["seeds"][f"candidate_generator_{key}"] for key in ("local", "tangent", "paths", "sobol")},
        "new_Law": law["eta_Law_B1"], "historical_context": manifest["Law_search"]["historical_context"],
        "local_scales": [0.00025, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.04],
        "tangent_radii": [0.0001, 0.0002, 0.0004, 0.0007, 0.001, 0.0015, 0.0022, 0.0032, 0.0045, 0.0064, 0.009, 0.0125, 0.017, 0.023, 0.031, 0.041, 0.055],
        "tangent_gradient_epsilon": 1e-4, "periodic_path_alpha": "golden-ratio low-discrepancy interior sequence",
        "Sobol_reservoir_power": 14, "canonicalization": manifest["candidate_generator"]["canonicalization"],
        "minimum_sensor_separation": float(cfg["measurement"]["min_separation"]), "frozen_before_pool_and_support": True,
        "validation_accessed": False, "Tangent_optimizer_run": False, "Full_run": False,
    }
    _atomic_json(CANDIDATE_SPEC_PATH, payload)
    return payload


def _candidate_consider(rows: list[dict[str, Any]], seen: set[str], eta: Any, law_eta: np.ndarray, cfg: dict[str, Any], metadata: dict[str, Any]) -> bool:
    canonical = canonicalize_eta(eta, law_eta, BOX)
    if minimum_periodic_separation(canonical, BOX) < float(cfg["measurement"]["min_separation"]):
        return False
    key = eta_key(canonical)
    if key in seen:
        return False
    seen.add(key); rows.append({"candidate_id": f"B1_candidate_{len(rows):05d}", "eta": canonical.tolist(), "eta_sha256": key, **metadata})
    return True


def generate_candidate_pool(cfg: dict[str, Any], progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    spec = freeze_candidate_generator(cfg)
    if spec.get("status") == "B1_LAW_RECONSTRUCTION_FAILED":
        return spec
    if CANDIDATE_POOL_PATH.exists():
        saved = _json(CANDIDATE_POOL_PATH)
        if saved["rows_sha256"] != _payload_sha256(saved["rows"]):
            raise RuntimeError("candidate pool changed")
        return saved
    law_eta = np.asarray(spec["new_Law"], dtype=np.float64); anchors = spec["historical_context"]
    rows = []; seen: set[str] = set(); targets = spec["component_targets"]
    _candidate_consider(rows, seen, law_eta, law_eta, cfg, {"component": "mandatory_and_local", "anchor_id": "new_B1_Law"})
    for anchor in anchors:
        if len(rows) >= min(len(anchors) + 1, int(targets["mandatory_and_local"])):
            break
        _candidate_consider(rows, seen, anchor["eta"], law_eta, cfg, {"component": "mandatory_and_local", "anchor_id": anchor["anchor_id"]})
    rng = np.random.default_rng(int(spec["seeds"]["local"]["seed"])); scales = spec["local_scales"]; anchor_rows = [{"anchor_id": "new_B1_Law", "eta": law_eta.tolist()}] + anchors
    attempt = 0
    while len(rows) < int(targets["mandatory_and_local"]):
        anchor = anchor_rows[attempt % len(anchor_rows)]; scale = float(scales[attempt % len(scales)])
        _candidate_consider(rows, seen, np.asarray(anchor["eta"]) + scale * rng.normal(size=8), law_eta, cfg, {"component": "mandatory_and_local", "anchor_id": anchor["anchor_id"], "scale": scale})
        attempt += 1
        if attempt > 100000:
            raise RuntimeError("could not fill candidate local component")
    data, _, _ = _selection_context(cfg, "risk_anchor"); epsilon = float(spec["tangent_gradient_epsilon"])
    finite_rows = []
    for coordinate in range(8):
        for sign in (-1, 1):
            eta = law_eta.copy(); eta[coordinate] += sign * epsilon
            finite_rows.append({"candidate_id": f"gradient_{coordinate}_{sign}", "eta": canonicalize_eta(eta, law_eta, BOX).tolist()})
    finite_risks = _risk_rows_batched(data, finite_rows, CANDIDATE_BATCH_SIZE)
    gradient = np.asarray([(finite_risks[2 * i + 1] - finite_risks[2 * i]) / (2 * epsilon) for i in range(8)], dtype=np.float64)
    gradient_norm2 = float(np.dot(gradient, gradient)); tangent_rng = np.random.default_rng(int(spec["seeds"]["tangent"]["seed"])); tangent_start = len(rows); direction_index = 0
    while len(rows) - tangent_start < int(targets["risk_tangent"]):
        direction = tangent_rng.normal(size=8)
        if gradient_norm2 > 1e-30:
            direction -= float(np.dot(direction, gradient)) / gradient_norm2 * gradient
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-14:
            continue
        direction /= norm
        for radius in spec["tangent_radii"]:
            for sign in (-1, 1):
                if len(rows) - tangent_start >= int(targets["risk_tangent"]):
                    break
                _candidate_consider(rows, seen, law_eta + sign * float(radius) * direction, law_eta, cfg, {"component": "risk_tangent", "direction_index": direction_index, "radius": radius, "sign": sign})
        direction_index += 1
        if direction_index > 10000:
            raise RuntimeError("could not fill risk-tangent component")
    path_start = len(rows); golden = (math.sqrt(5.0) - 1.0) / 2.0; path_index = 0
    while len(rows) - path_start < int(targets["periodic_paths"]):
        anchor = anchors[path_index % len(anchors)]; alpha = ((path_index + 1) * golden) % 1.0
        eta = periodic_interpolate(law_eta, anchor["eta"], alpha, law_eta, BOX)
        _candidate_consider(rows, seen, eta, law_eta, cfg, {"component": "periodic_paths", "anchor_id": anchor["anchor_id"], "alpha": alpha})
        path_index += 1
        if path_index > 100000:
            raise RuntimeError("could not fill periodic-path component")
    sobol_start = len(rows); sobol = qmc.Sobol(d=8, scramble=True, seed=int(spec["seeds"]["sobol"]["seed"]))
    for index, point in enumerate(sobol.random_base2(m=int(spec["Sobol_reservoir_power"]))):
        if len(rows) - sobol_start >= int(targets["Sobol_global"]):
            break
        _candidate_consider(rows, seen, point * np.tile(np.asarray(BOX), 4), law_eta, cfg, {"component": "Sobol_global", "sobol_index": index})
    if len(rows) != CANDIDATE_COUNT:
        raise RuntimeError(f"candidate pool count {len(rows)} != {CANDIDATE_COUNT}")
    payload = {
        "schema_version": 1, "candidate_generator_spec_sha256": file_sha256(CANDIDATE_SPEC_PATH), "count": len(rows),
        "component_counts": {name: sum(row["component"] == name for row in rows) for name in targets},
        "risk_gradient": gradient.tolist(), "risk_gradient_sha256": _array_sha256(gradient),
        "canonical_unique": len(seen) == len(rows), "minimum_observed_separation": min(minimum_periodic_separation(row["eta"], BOX) for row in rows),
        "rows_sha256": _payload_sha256(rows), "frozen_before_support_evaluation": True, "rows": rows,
    }
    _atomic_json(CANDIDATE_POOL_PATH, payload)
    if progress:
        progress(f"candidate pool frozen: {len(rows)}")
    return payload


def evaluate_candidate_risk(cfg: dict[str, Any], progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    pool = generate_candidate_pool(cfg)
    if pool.get("status") == "B1_LAW_RECONSTRUCTION_FAILED":
        return pool
    if CANDIDATE_RISK_PATH.exists():
        saved = _json(CANDIDATE_RISK_PATH)
        if saved["candidate_pool_sha256"] != file_sha256(CANDIDATE_POOL_PATH) or saved["rows_sha256"] != _payload_sha256(saved["rows"]):
            raise RuntimeError("candidate risk cache changed")
        return saved
    data, _, _ = _selection_context(cfg, "risk_anchor"); risks = _risk_rows_batched(data, pool["rows"], CANDIDATE_BATCH_SIZE)
    law = _json(LAW_FREEZE_PATH); anchor = float(law["R_Law_B1"])
    rows = [{"candidate_id": row["candidate_id"], "eta": row["eta"], "eta_sha256": row["eta_sha256"], "scientific_risk": float(risk), "Law_relative_increase_percent": float(100.0 * (risk / anchor - 1.0)), "allowance_membership": {str(p): bool(risk <= (1.0 + p / 100.0) * anchor) for p in ALLOWANCES}, "geometry_valid": True} for row, risk in zip(pool["rows"], risks, strict=True)]
    payload = {
        "schema_version": 1, "candidate_pool_sha256": file_sha256(CANDIDATE_POOL_PATH), "risk_anchor_bank_sha256": file_sha256(_bank_path("risk_anchor")),
        "R_Law_B1": anchor, "historical_R_Law_used": False, "rows_sha256": _payload_sha256(rows), "rows": rows,
    }
    _atomic_json(CANDIDATE_RISK_PATH, payload)
    if progress:
        progress(f"candidate risk evaluated: {len(rows)}")
    return payload


def _support_result_path(label: str) -> Path:
    return OUTPUT_ROOT / "support_results" / f"{label}.npz"


def _support_record_path(label: str) -> Path:
    return _support_result_path(label).with_suffix(".json")


def evaluate_support(cfg: dict[str, Any], progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    risk = evaluate_candidate_risk(cfg)
    if risk.get("status") == "B1_LAW_RECONSTRUCTION_FAILED":
        return risk
    etas = np.asarray([row["eta"] for row in risk["rows"]], dtype=np.float64)
    manifest = _json(REFERENCE_BANK_MANIFEST_PATH); completed = []
    for label, count, _ in _bank_specs(freeze_manifest(cfg)):
        if label in {"law_search", "risk_anchor"}:
            continue
        result_path, record_path = _support_result_path(label), _support_record_path(label)
        if result_path.exists() or record_path.exists():
            if not result_path.exists() or not record_path.exists() or file_sha256(result_path) != _json(record_path)["result_sha256"]:
                raise RuntimeError(f"support cache seal mismatch: {label}")
            completed.append(_json(record_path)); continue
        started = time.perf_counter(); data, _, _ = _selection_context(cfg, label); bank = _load_bank(label)
        result = _full_eval(data, etas, bank, count)
        _atomic_npz(result_path, candidate_index=np.arange(CANDIDATE_COUNT, dtype=np.int32), **result)
        record = {
            "schema_version": 1, "label": label, "N": count, "candidate_pool_sha256": file_sha256(CANDIDATE_POOL_PATH),
            "checkpoint_sha256": _json(ACCEPTED_REFERENCE_PATH)["checkpoint_sha256"], "bank_sha256": file_sha256(_bank_path(label)),
            "result_sha256": file_sha256(result_path), "support_pass_count": int(np.sum(result["support_valid"])),
            "wall_time_seconds": time.perf_counter() - started,
        }
        _atomic_json(record_path, record); completed.append(record)
        if progress:
            progress(f"support {label}: {record['support_pass_count']}/{CANDIDATE_COUNT}")
        del data, bank, result; gc.collect()
    return {"schema_version": 1, "bank_count": len(completed), "all_use_accepted_checkpoint": len({row["checkpoint_sha256"] for row in completed}) == 1, "banks": completed}


def _load_support_result(label: str) -> dict[str, np.ndarray]:
    path = _support_result_path(label); record = _json(_support_record_path(label))
    if file_sha256(path) != record["result_sha256"]:
        raise RuntimeError(f"support result changed: {label}")
    with np.load(path, allow_pickle=False) as arrays:
        return {key: np.asarray(arrays[key]) for key in arrays.files if key != "candidate_index"}


def _diverse_count(rows: list[dict[str, Any]]) -> int:
    ordered = sorted(rows, key=lambda row: (-row["robust_rESS"], row["candidate_id"]))
    selected = []
    for row in ordered:
        if all(_symmetry_aware_distance(row["eta"], old["eta"], BOX) >= 0.02 for old in selected):
            selected.append(row)
    return len(selected)


def summarize(cfg: dict[str, Any]) -> dict[str, Any]:
    accepted = train_and_accept_reference(cfg)
    if not accepted.get("accepted"):
        summary = {"schema_version": 1, "classification": "B1_REFERENCE_QUALIFICATION_FAILED", "accepted_reference": accepted, "validation_accessed": False}
        _atomic_json(SUMMARY_PATH, summary); _atomic_text(REPORT_PATH, _report(summary)); _write_inventory(); return summary
    law = reconstruct_and_freeze_law(cfg)
    if law.get("status") != "FROZEN":
        summary = {"schema_version": 1, "classification": "B1_LAW_RECONSTRUCTION_FAILED", "accepted_reference": accepted, "law": law, "validation_accessed": False}
        _atomic_json(SUMMARY_PATH, summary); _atomic_text(REPORT_PATH, _report(summary)); _write_inventory(); return summary
    evaluate_support(cfg); risk = _json(CANDIDATE_RISK_PATH); pool = _json(CANDIDATE_POOL_PATH)
    support = {label: _load_support_result(label) for index in range(SUPPORT_PAIRS) for label in (f"screen_{index}", f"audit_{index}")}
    law_index = 0
    if pool["rows"][0]["eta_sha256"] != eta_key(law["eta_Law_B1"]):
        raise RuntimeError("new B1 Law is not the mandatory first candidate")
    law_rows = []
    for label, values in support.items():
        law_rows.append({
            "bank": label, "minimum_rESS": float(values["minimum_ress"][law_index]),
            "controlling_time_node": int(values["controlling_time_index"][law_index]),
            "node7_rESS": float(values["ress_trajectory"][law_index, NODE7]),
            "node7_lambda_norm": float(values["lambda_norm"][law_index, NODE7]),
            "node7_top1pct_mass": float(values["top_1pct_weight_mass"][law_index, NODE7]),
            "support_valid": bool(values["support_valid"][law_index]),
        })
    all_valid = np.stack([support[label]["support_valid"] for label in support], axis=0)
    all_minimum = np.stack([support[label]["minimum_ress"] for label in support], axis=0)
    pair_valid = np.stack([support[f"screen_{index}"]["support_valid"] & support[f"audit_{index}"]["support_valid"] for index in range(SUPPORT_PAIRS)], axis=0)
    all_pair = np.all(pair_valid, axis=0); all_screen = np.all(np.stack([support[f"screen_{i}"]["support_valid"] for i in range(SUPPORT_PAIRS)]), axis=0)
    robust = np.min(all_minimum, axis=0)
    allowance_rows = []
    risk_by_id = {row["candidate_id"]: row for row in risk["rows"]}
    for allowance in ALLOWANCES:
        ceiling = (1.0 + allowance / 100.0) * float(law["R_Law_B1"])
        inside = np.asarray([risk_by_id[row["candidate_id"]]["scientific_risk"] <= ceiling for row in pool["rows"]])
        survivors = np.flatnonzero(inside & all_pair)
        survivor_rows = [{**pool["rows"][index], "robust_rESS": float(robust[index])} for index in survivors]
        robust_values = robust[survivors]
        allowance_rows.append({
            "allowance_percent": allowance, "exact_risk_ceiling": float(ceiling), "inside_exact_risk_count": int(np.sum(inside)),
            "all_screen_feasible_count": int(np.sum(inside & all_screen)),
            "pair_complete_counts": [int(np.sum(inside & pair_valid[index])) for index in range(SUPPORT_PAIRS)],
            "all_four_pair_survivors": int(len(survivors)), "diverse_survivors": _diverse_count(survivor_rows),
            "robust_rESS": None if not len(survivors) else {"best": float(np.max(robust_values)), "median": float(np.median(robust_values)), "p10": float(np.quantile(robust_values, 0.10))},
            "survivor_ids": [pool["rows"][index]["candidate_id"] for index in survivors],
        })
    numerical_pathology_fraction = float(np.mean(np.stack([~support[label]["projection_valid"] | ~support[label]["forcing_valid"] | ~support[label]["covariance_valid"] for label in support])))
    law_minimum = min(row["minimum_rESS"] for row in law_rows); law_all = all(row["support_valid"] for row in law_rows)
    ready = bool(
        law_all and law_minimum >= 0.10
        and allowance_rows[0]["all_four_pair_survivors"] >= 10
        and all(row["all_four_pair_survivors"] >= 1 for row in allowance_rows[1:])
        and numerical_pathology_fraction < 0.50
    )
    if ready:
        classification = "B1_SINGLE_REFERENCE_READY"
        next_step = "NEW_OFFICIAL_B1_GALERKIN_PARETO_PROTOCOL"
    elif law_all and law_minimum >= MINIMUM_RESS:
        classification = "B1_SINGLE_REFERENCE_BORDERLINE"
        next_step = "Do not launch an official protocol; inspect the frozen Law/support thinness without adapting this experiment."
    else:
        classification = "B1_SUPPORT_PROBLEM_PERSISTS"
        next_step = "Stop: the qualified B1 reference did not establish sufficiently stable fresh-bank support."
    payload = {
        "schema_version": 1, "classification": classification, "recommended_next_scientific_step": next_step,
        "accepted_reference": accepted, "law": law,
        "law_support": {"banks": law_rows, "minimum": law_minimum, "median": float(np.median([row["minimum_rESS"] for row in law_rows])), "all_eight_support_valid": law_all},
        "allowances": allowance_rows, "population_numerical_pathology_fraction": numerical_pathology_fraction,
        "candidate_count": CANDIDATE_COUNT,
        "safeguards": {"physical_benchmark_changed": False, "B1_particle_matching": True, "configuration_OT": False, "validation_accessed": False, "intermediate_truth_used_for_reference_training": False, "Tangent": False, "Full": False, "production_reference_replaced": False, "official_protocol_created": False},
    }
    _atomic_json(ALLOWANCE_PATH, {"schema_version": 1, "R_Law_B1": law["R_Law_B1"], "allowances": allowance_rows})
    _atomic_json(SUMMARY_PATH, payload); _atomic_text(REPORT_PATH, _report(payload)); _write_inventory()
    return payload


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# Clean-Room Single-Reference B1 Preflight", "", "CLEAN-ROOM B1 PREFLIGHT", "",
        "physical benchmark changed: NO", "B1 particle matching: YES", "configuration OT: NO", "",
        "validation accessed: NO", "intermediate truth used for reference training: NO", "", "## DATA PROVENANCE", "",
    ]
    if TRAIN_DATA_MANIFEST_PATH.exists():
        train, qual, design = _json(TRAIN_DATA_MANIFEST_PATH), _json(QUAL_DATA_MANIFEST_PATH), _json(DESIGN_DATA_MANIFEST_PATH)
        lines += [f"endpoint train: generated; count {train['count']}; hash `{train['sha256']}`", f"endpoint qualification: generated; count {qual['count']}; hash `{qual['sha256']}`", f"design truth selection: generated; count {design['count']}; hash `{design['sha256']}`", ""]
    lines += ["## REFERENCE QUALIFICATION", "", "| attempt | seed | CFM loss | endpoint Psi L2 | whitened Psi | Law-Phi L2 | status |", "|---|---:|---:|---:|---:|---:|---|"]
    attempts = []
    for attempt in ATTEMPTS:
        if _attempt_record(attempt).exists():
            attempts.append(_json(_attempt_record(attempt)))
    for row in attempts:
        q = row["qualification"]
        lines.append(f"| {row['attempt']} | {row['training_seed']} | {q['CFM_qualification_loss']:.6f} | {q['endpoint_Psi_L2']:.6f} | {q['endpoint_whitened_Psi_norm']:.6f} | {q['endpoint_Law_Phi_L2']:.6f} | {row['status']} |")
    accepted = summary.get("accepted_reference", {})
    lines += ["", "## ACCEPTED SINGLE REFERENCE", "", f"attempt: {accepted.get('attempt', 'NONE')}", f"seed: {accepted.get('training_seed', 'NONE')}", f"checkpoint SHA-256: `{accepted.get('checkpoint_sha256', 'NONE')}`", ""]
    if summary["classification"] == "B1_REFERENCE_QUALIFICATION_FAILED":
        lines += ["## DEVELOPMENT CLASSIFICATION", "", summary["classification"], ""]
    elif summary["classification"] == "B1_LAW_RECONSTRUCTION_FAILED":
        lines += ["## DEVELOPMENT CLASSIFICATION", "", summary["classification"], ""]
    else:
        law = summary["law"]
        lines += ["## NEW B1 LAW", "", f"eta: `{law['eta_Law_B1']}`", f"R_Law_B1: {law['R_Law_B1']:.12g}", "", f"historical R_Law: {HISTORICAL_R_LAW}", "status: NOT USED AS NEW ANCHOR", "", "Exact new risk ceilings:", ""]
        for allowance in summary["allowances"]:
            lines.append(f"- {allowance['allowance_percent']:g}%: {allowance['exact_risk_ceiling']:.12g}")
        lines += ["", "## LAW SUPPORT", "", "| bank | min rESS | controlling node | node7 rESS | lambda norm | top1% mass |", "|---|---:|---:|---:|---:|---:|"]
        for row in summary["law_support"]["banks"]:
            lines.append(f"| {row['bank']} | {row['minimum_rESS']:.6f} | {row['controlling_time_node']} | {row['node7_rESS']:.6f} | {row['node7_lambda_norm']:.3f} | {row['node7_top1pct_mass']:.6f} |")
        lines += ["", f"minimum: {summary['law_support']['minimum']:.6f}", f"median: {summary['law_support']['median']:.6f}", "", "## B1 CANDIDATE SUPPORT", "", "| allowance | inside risk | all-pair survivors | diverse survivors | p10 rESS | median rESS |", "|---:|---:|---:|---:|---:|---:|"]
        for row in summary["allowances"]:
            robust = row["robust_rESS"]
            lines.append(f"| {row['allowance_percent']:g}% | {row['inside_exact_risk_count']} | {row['all_four_pair_survivors']} | {row['diverse_survivors']} | {'—' if robust is None else f'{robust['p10']:.6f}'} | {'—' if robust is None else f'{robust['median']:.6f}'} |")
        lines += ["", "## DEVELOPMENT CLASSIFICATION", "", summary["classification"], "", "RECOMMENDED NEXT SCIENTIFIC STEP:", summary["recommended_next_scientific_step"], ""]
    lines += ["NO Tangent", "NO Full", "NO validation", "NO production reference replacement", "NO official protocol created", ""]
    return "\n".join(lines)


def _write_inventory() -> dict[str, Any]:
    files = []
    for path in sorted(OUTPUT_ROOT.rglob("*")):
        if path.is_file() and path != INVENTORY_PATH:
            files.append({"path": str(path.relative_to(OUTPUT_ROOT)), "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    payload = {"schema_version": 1, "artifact_count": len(files), "files": files}
    _atomic_json(INVENTORY_PATH, payload); return payload


def console_report() -> str:
    return REPORT_PATH.read_text(encoding="utf-8")


def run_all(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    cfg = _json(CONFIG_PATH); freeze_manifest(cfg); generate_clean_data(cfg, progress); build_b1_couplings(cfg)
    accepted = train_and_accept_reference(cfg, progress)
    if accepted.get("accepted"):
        generate_reference_banks(cfg, progress); reconstruct_and_freeze_law(cfg, progress); generate_candidate_pool(cfg, progress)
        evaluate_candidate_risk(cfg, progress); evaluate_support(cfg, progress)
    return summarize(cfg)
