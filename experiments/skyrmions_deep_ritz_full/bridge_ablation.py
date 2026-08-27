"""Development-only endpoint-coupling ablation for the skyrmion reference.

This module is deliberately narrow: it builds endpoint pair maps, trains the
unchanged endpoint CFM architecture, and evaluates the resulting references on
the frozen development panel.  It has no validation loader, candidate
generator, sensor optimizer, Tangent/Full construction, or Ritz entry point.
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
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment

from .domain import SkyrmionTruth, minimum_image
from .galerkin_only_data import GalerkinReferenceBank, _family, _physics_config
from .pareto_v2_common import ARTIFACT_DIR
from .pareto_v3_common import ROOT, file_sha256
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
    EXPECTED_PANEL_SHA256,
    _ReferenceEvaluator,
    _array_sha256,
    _load_design_context,
)
from .reference_semantics_audit import _mean_phi, _mean_psi, _metric_definition
from .risk import many_body_features


VERSION = "skyrmion_galerkin_dev_bridge_ablation_v1"
SEED_NAMESPACE = VERSION
GLOBAL_SEED = 20260826
OUTPUT_ROOT = ROOT / "outputs" / VERSION
REPO_ROOT = ROOT.parent.parent
CONFIG_PATH = ROOT / "config.json"
TRUTH_BANKS_PATH = ARTIFACT_DIR / "truth_banks.npz"
BASELINE_CHECKPOINT_PATH = ARTIFACT_DIR / "reference.npz"
UPSTREAM_ROOT = ROOT / "outputs" / "skyrmion_galerkin_dev_reference_seed_robustness_v1"
PANEL_SOURCE_PATH = UPSTREAM_ROOT / "candidate_panel_reference.json"
PANEL_PATH = OUTPUT_ROOT / "candidate_panel_reference.json"

SOURCE_SEAL_PATH = OUTPUT_ROOT / "source_seal.json"
MANIFEST_PATH = OUTPUT_ROOT / "bridge_ablation_manifest.json"
MANIFEST_HASH_PATH = OUTPUT_ROOT / "bridge_ablation_manifest.sha256"
COUPLING_SPEC_PATH = OUTPUT_ROOT / "coupling_bank_spec.json"
COUPLING_DIAGNOSTICS_PATH = OUTPUT_ROOT / "coupling_diagnostics.json"
HOLDOUT_PATH = OUTPUT_ROOT / "endpoint_holdout.npz"
HOLDOUT_MANIFEST_PATH = OUTPUT_ROOT / "endpoint_holdout_manifest.json"
TRAINING_SUMMARY_PATH = OUTPUT_ROOT / "reference_training_summary.json"
ENDPOINT_QUALIFICATION_PATH = OUTPUT_ROOT / "endpoint_qualification.json"
EVAL_MANIFEST_PATH = OUTPUT_ROOT / "reference_eval_bank_manifest.json"
FAMILY_SUMMARY_PATH = OUTPUT_ROOT / "bridge_family_summary.json"
PAIRED_PATH = OUTPUT_ROOT / "paired_seed_comparison.json"
VARIABILITY_PATH = OUTPUT_ROOT / "seed_variability_summary.json"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
REPORT_PATH = OUTPUT_ROOT / "report.md"
INVENTORY_PATH = OUTPUT_ROOT / "inventory.json"

FAMILIES = ("B0", "B1", "B2", "B3")
FAMILY_NAMES = {
    "B0": "CURRENT_BASELINE",
    "B1": "PARTICLE_MATCH_ONLY",
    "B2": "CONFIG_OT_ONLY",
    "B3": "OT_PLUS_PARTICLE_MATCH",
}
MATCHED_SEEDS = 3
COUPLING_MAPS = 4
OT_BLOCK_SIZE = 32
TRAIN_ENDPOINTS = 12000
HOLDOUT_N = 4096
EVAL_BANKS = 4
EVAL_N = 32768
CFM_EVAL_N = 32768
TIME_COUNT = 13
NODE7 = 7
HIGH_PASS_COUNT = 55
CONTROL_COUNT = 8
MINIMUM_RESS = 0.05
ROLLOUT_BATCH_SIZE = 2048
FEATURE_BATCH_SIZE = 2048
BRIDGE_DIAGNOSTIC_PAIRS = 4096
BOX = (2.0, 1.0)


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _derive_seed(role: str) -> dict[str, Any]:
    encoded = f"{SEED_NAMESPACE}|{GLOBAL_SEED}|{role}".encode()
    digest = hashlib.sha256(encoded).hexdigest()
    return {"role": role, "seed": int(digest[:16], 16) % (2**31 - 1), "sha256": digest}


def _inside(path: Path) -> Path:
    resolved, root = path.resolve(), OUTPUT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"output must be beneath {root}: {resolved}")
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
    return {
        "minimum": float(np.min(x)), "p10": float(np.quantile(x, 0.10)),
        "median": float(np.median(x)), "mean": float(np.mean(x)),
        "p90": float(np.quantile(x, 0.90)), "maximum": float(np.max(x)),
        "sd": float(np.std(x)),
    }


def _training_endpoints() -> tuple[np.ndarray, np.ndarray]:
    with np.load(TRUTH_BANKS_PATH, allow_pickle=False) as arrays:
        x0 = np.asarray(arrays["endpoint0"], dtype=np.float64)
        x1 = np.asarray(arrays["endpoint1"], dtype=np.float64)
    if x0.shape != (TRAIN_ENDPOINTS, 16, 2) or x1.shape != x0.shape:
        raise RuntimeError("authoritative endpoint-bank shape changed")
    return x0, x1


def _permutation_records() -> list[dict[str, Any]]:
    records = []
    for index in range(COUPLING_MAPS):
        seed_record = _derive_seed(f"coupling_map_{index:02d}")
        rng = np.random.default_rng(seed_record["seed"])
        source = rng.permutation(TRAIN_ENDPOINTS).astype(np.int32)
        target = rng.permutation(TRAIN_ENDPOINTS).astype(np.int32)
        records.append({
            **seed_record, "map_index": index,
            "source_permutation_sha256": _array_sha256(source),
            "target_permutation_sha256": _array_sha256(target),
        })
    return records


def verify_and_seal_sources() -> dict[str, Any]:
    upstream_seal = _json(UPSTREAM_ROOT / "source_seal.json")
    fixed = upstream_seal["immutable_source_hashes"]
    checks = {
        "production_checkpoint": file_sha256(BASELINE_CHECKPOINT_PATH),
        "configuration": file_sha256(CONFIG_PATH),
        "candidate_panel": file_sha256(PANEL_SOURCE_PATH),
        "physical_model": file_sha256(ROOT / "domain.py"),
        "scientific_risk": file_sha256(ROOT / "risk.py"),
        "reference_semantics": file_sha256(ROOT / "reference.py"),
    }
    if checks["production_checkpoint"] != EXPECTED_BASELINE_CHECKPOINT_SHA256:
        raise RuntimeError("production reference checkpoint changed")
    if checks["configuration"] != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("authoritative configuration changed")
    if checks["candidate_panel"] != EXPECTED_PANEL_SHA256:
        raise RuntimeError("frozen candidate panel changed")
    upstream_expected = {
        "physical_model": fixed.get("physical_model_source", checks["physical_model"]),
        "scientific_risk": fixed["scientific_risk_source"],
        "reference_semantics": fixed["reference_training_bridge_and_rollout_source"],
    }
    for key, expected in upstream_expected.items():
        if checks[key] != expected:
            raise RuntimeError(f"sealed {key} source changed")
    sources = [Path(__file__), ROOT / "bridge_ablation_run.py", ROOT / "test_bridge_ablation.py"]
    payload = {
        "schema_version": 1, "version": VERSION, "development_only": True,
        "fixed_input_hashes": checks,
        "analysis_source_hashes": {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sources},
        "physical_benchmark_changed": False, "scientific_risk_changed": False,
        "whitening_changed": False, "rESS_threshold_changed": False,
        "intermediate_truth_used_for_training": False, "validation_accessed": False,
        "official_protocol_created": False, "production_reference_replaced": False,
    }
    if SOURCE_SEAL_PATH.exists():
        cached = _json(SOURCE_SEAL_PATH)
        if cached != payload:
            raise RuntimeError("sealed sources changed")
        return cached
    _atomic_json(SOURCE_SEAL_PATH, payload)
    return payload


def freeze_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    verify_and_seal_sources()
    if PANEL_PATH.exists():
        if file_sha256(PANEL_PATH) != EXPECTED_PANEL_SHA256:
            raise RuntimeError("panel copy changed")
    else:
        PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PANEL_SOURCE_PATH, PANEL_PATH)
    if MANIFEST_PATH.exists() or MANIFEST_HASH_PATH.exists():
        if not MANIFEST_PATH.exists() or not MANIFEST_HASH_PATH.exists():
            raise RuntimeError("incomplete manifest seal")
        if file_sha256(MANIFEST_PATH) != MANIFEST_HASH_PATH.read_text().strip():
            raise RuntimeError("bridge manifest changed after freezing")
        return _json(MANIFEST_PATH)
    base = ReferenceTrainingConfig(**cfg["reference_training"])
    expected = {
        "hidden_width": 64, "hidden_layers": 3, "train_steps": 6000,
        "batch_size": 512, "learning_rate": 8e-4,
        "min_learning_rate_ratio": 0.08, "grad_clip_norm": 8.0,
        "bridge_noise_std": 0.01,
    }
    for key, value in expected.items():
        if asdict(base)[key] != value:
            raise RuntimeError(f"training constant changed: {key}")
    training_seeds = [_derive_seed(f"matched_training_seed_{i}") for i in range(MATCHED_SEEDS)]
    eval_seeds = [_derive_seed(f"reference_eval_bank_{i}") for i in range(EVAL_BANKS)]
    holdout_seed = _derive_seed("endpoint_holdout")
    cfm_seed = _derive_seed("fixed_bridge_cfm_evaluation")
    all_seeds = training_seeds + eval_seeds + [holdout_seed, cfm_seed] + _permutation_records()
    if len({row["seed"] for row in all_seeds}) != len(all_seeds):
        raise RuntimeError("seed collision")
    payload = {
        "schema_version": 1, "version": VERSION, "development_only": True,
        "source_seal_sha256": file_sha256(SOURCE_SEAL_PATH),
        "candidate_panel_sha256": file_sha256(PANEL_PATH),
        "bridge_families": FAMILY_NAMES,
        "coupling_maps": {
            "count": COUPLING_MAPS, "endpoint_count": TRAIN_ENDPOINTS,
            "configuration_OT_block_size": OT_BLOCK_SIZE,
            "input_endpoint_permutations": _permutation_records(),
            "B0_B1_share_configuration_pairs": True,
            "B2_B3_share_configuration_pairs": True,
            "hard_uniform_assignment": True, "entropic_regularization": False,
        },
        "particle_assignment": {
            "solver": "scipy.optimize.linear_sum_assignment",
            "cost": "periodic squared Euclidean distance under authoritative minimum image",
            "tie_breaking": "SciPy deterministic row-sorted assignment output",
            "B1_applies_assignment": True, "B2_applies_assignment": False,
            "B3_applies_assignment": True,
        },
        "training": {
            "matched_seeds": training_seeds,
            "configurations": {
                family: [asdict(replace(base, seed=row["seed"])) for row in training_seeds]
                for family in FAMILIES
            },
            "same_network_initialization_and_stochastic_schedule_within_matched_seed": True,
            "train_all_12_without_adaptive_stopping": True,
        },
        "fixed_bridge_CFM_evaluation": {"seed": cfm_seed, "examples": CFM_EVAL_N},
        "endpoint_holdout": {
            "seed": holdout_seed, "samples": HOLDOUT_N,
            "persist_endpoints_only": True, "scientific_intervals": TIME_COUNT - 1,
            "truth_substeps_per_interval": int(cfg["physics"]["truth_substeps"]),
        },
        "reference_evaluation": {
            "seeds": eval_seeds, "N": EVAL_N, "time_nodes": TIME_COUNT,
            "rollout_substeps_per_scientific_interval": int(cfg["banks"]["reference_substeps"]),
            "minimum_rESS": MINIMUM_RESS,
        },
        "readiness_rule": {
            "endpoint_comparable": "family median whitened endpoint norm <= 1.25 times matched B0 median",
            "no_catastrophic_risk": "every seed median projected Law risk < 20 and raw Law risk < 20",
            "rESS_majority": "paired Law node7 rESS delta >= -0.005 in at least 2 of 3 seeds",
            "risk_variability_reduced": "projected Law-risk between-seed SD <= 0.80 times B0 SD",
            "nontrivial_panel_support": "family median candidate-bank rESS pass fraction >= 0.10",
            "not_one_seed": "at least 2 of 3 seeds individually satisfy endpoint, risk, and rESS conditions",
            "all_conditions_required": True,
        },
        "interpretation_rule": [
            "readiness is evaluated first without scalar scoring",
            "if only B1 is ready, PARTICLE_LABEL_COUPLING_WAS_PRIMARY",
            "if only B2 is ready, CONFIGURATION_COUPLING_WAS_PRIMARY",
            "if B3 is ready and clearly improves both B1 and B2, BOTH_COUPLINGS_MATTER",
            "if B1 and B3 are ready and B3 is not clearly better, PERMUTATION_MATCHING_SUFFICIENT",
            "if B2 and B3 are ready and B3 is not clearly better, OT_COUPLING_SUFFICIENT",
            "if no modified family is ready, BRIDGE_CHANGES_DO_NOT_RESOLVE_REFERENCE_TRADEOFF",
            "otherwise MIXED_BRIDGE_EFFECT",
        ],
        "intermediate_truth_training_permitted": False,
        "validation_access_permitted": False, "sensor_generation_permitted": False,
        "tangent_full_eigensolve_deep_ritz_permitted": False,
        "production_installation_permitted": False, "frozen_before_training": True,
    }
    _atomic_json(MANIFEST_PATH, payload)
    _atomic_text(MANIFEST_HASH_PATH, file_sha256(MANIFEST_PATH) + "\n")
    return payload


def particle_assignment(x0: np.ndarray, x1: np.ndarray, box: tuple[float, float] = BOX) -> tuple[np.ndarray, float]:
    delta = x1[None, :, :] - x0[:, None, :]
    delta -= np.asarray(box) * np.floor(delta / np.asarray(box) + 0.5)
    cost = np.sum(delta * delta, axis=-1)
    rows, cols = linear_sum_assignment(cost)
    if not np.array_equal(rows, np.arange(len(x0))):
        raise RuntimeError("assignment rows are not canonical")
    return cols.astype(np.int16), float(np.mean(cost[rows, cols]))


def _pair_file(family: str, map_index: int) -> Path:
    return OUTPUT_ROOT / "endpoint_coupling" / family / f"map_{map_index:02d}.npz"


def _pair_record_file(family: str, map_index: int) -> Path:
    return _pair_file(family, map_index).with_suffix(".json")


def _load_pair_map(family: str, map_index: int) -> dict[str, np.ndarray]:
    path, record_path = _pair_file(family, map_index), _pair_record_file(family, map_index)
    record = _json(record_path)
    if file_sha256(path) != record["sha256"]:
        raise RuntimeError(f"coupling map changed: {family}/{map_index}")
    with np.load(path, allow_pickle=False) as arrays:
        return {key: np.asarray(arrays[key]) for key in arrays.files}


def build_coupling_banks(cfg: dict[str, Any], progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    manifest = freeze_manifest(cfg)
    x0, x1 = _training_endpoints()
    map_rows = []
    identity = np.tile(np.arange(16, dtype=np.int16), (TRAIN_ENDPOINTS, 1))
    for map_record in manifest["coupling_maps"]["input_endpoint_permutations"]:
        index, seed = int(map_record["map_index"]), int(map_record["seed"])
        paths = [_pair_file(f, index) for f in FAMILIES]
        records = [_pair_record_file(f, index) for f in FAMILIES]
        if all(p.exists() for p in paths + records):
            for family in FAMILIES:
                _load_pair_map(family, index)
            map_rows.append({"map_index": index})
            if progress:
                progress(f"coupling map {index}: cache verified")
            continue
        if any(p.exists() for p in paths + records):
            raise RuntimeError(f"incomplete coupling map cache {index}")
        rng = np.random.default_rng(seed)
        source = rng.permutation(TRAIN_ENDPOINTS).astype(np.int32)
        random_target = rng.permutation(TRAIN_ENDPOINTS).astype(np.int32)
        if _array_sha256(source) != map_record["source_permutation_sha256"] or _array_sha256(random_target) != map_record["target_permutation_sha256"]:
            raise RuntimeError("frozen endpoint permutation reproduction failed")
        random_particle = np.empty((TRAIN_ENDPOINTS, 16), dtype=np.int16)
        random_cost = np.empty(TRAIN_ENDPOINTS, dtype=np.float64)
        for slot in range(TRAIN_ENDPOINTS):
            random_particle[slot], random_cost[slot] = particle_assignment(x0[source[slot]], x1[random_target[slot]])
        ot_target = np.empty(TRAIN_ENDPOINTS, dtype=np.int32)
        ot_particle = np.empty((TRAIN_ENDPOINTS, 16), dtype=np.int16)
        ot_cost = np.empty(TRAIN_ENDPOINTS, dtype=np.float64)
        for start in range(0, TRAIN_ENDPOINTS, OT_BLOCK_SIZE):
            stop = start + OT_BLOCK_SIZE
            source_block, target_block = source[start:stop], random_target[start:stop]
            costs = np.empty((OT_BLOCK_SIZE, OT_BLOCK_SIZE), dtype=np.float64)
            permutations = np.empty((OT_BLOCK_SIZE, OT_BLOCK_SIZE, 16), dtype=np.int16)
            for a, source_index in enumerate(source_block):
                for b, target_index in enumerate(target_block):
                    permutations[a, b], costs[a, b] = particle_assignment(x0[source_index], x1[target_index])
            rows, cols = linear_sum_assignment(costs)
            if not np.array_equal(rows, np.arange(OT_BLOCK_SIZE)):
                raise RuntimeError("configuration OT rows are not canonical")
            ot_target[start:stop] = target_block[cols]
            ot_particle[start:stop] = permutations[rows, cols]
            ot_cost[start:stop] = costs[rows, cols]
        if not np.array_equal(np.sort(ot_target), np.arange(TRAIN_ENDPOINTS)):
            raise RuntimeError("OT map failed to preserve target marginal")
        arm_arrays = {
            "B0": dict(source_index=source, target_index=random_target, particle_permutation=identity, configuration_cost=random_cost),
            "B1": dict(source_index=source, target_index=random_target, particle_permutation=random_particle, configuration_cost=random_cost),
            "B2": dict(source_index=source, target_index=ot_target, particle_permutation=identity, configuration_cost=ot_cost),
            "B3": dict(source_index=source, target_index=ot_target, particle_permutation=ot_particle, configuration_cost=ot_cost),
        }
        for family, arrays in arm_arrays.items():
            path = _pair_file(family, index)
            _atomic_npz(path, **arrays)
            record = {
                "schema_version": 1, "family": family, "family_name": FAMILY_NAMES[family],
                "map_index": index, "manifest_sha256": file_sha256(MANIFEST_PATH),
                "endpoint_archive_sha256": file_sha256(TRUTH_BANKS_PATH),
                "source_index_sha256": _array_sha256(arrays["source_index"]),
                "target_index_sha256": _array_sha256(arrays["target_index"]),
                "particle_permutation_sha256": _array_sha256(arrays["particle_permutation"]),
                "sha256": file_sha256(path),
            }
            _atomic_json(_pair_record_file(family, index), record)
        map_rows.append({"map_index": index})
        if progress:
            progress(f"coupling map {index}: built and sealed")
    spec = {
        "schema_version": 1, "manifest_sha256": file_sha256(MANIFEST_PATH),
        "strategy": "four deterministic full-bank pairing maps; map and slot sampled uniformly during CFM",
        "fairness": "all arms use the analogous frozen bank; matched seeds use identical map/slot/t/noise schedules",
        "B0": "random complete-configuration pairing; original target labels retained",
        "B1": "same B0 pairs; exact periodic particle assignment applied",
        "B2": "blockwise hard configuration OT under permutation-invariant cost; original target labels retained",
        "B3": "same B2 pairs; cached exact particle assignment applied",
        "endpoint_samples_synthesized": False, "endpoint_marginals_preserved": True,
        "maps": map_rows,
    }
    _atomic_json(COUPLING_SPEC_PATH, spec)
    return spec


def _bridge_arrays(family: str, map_index: int, slots: np.ndarray, x0: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pair = _load_pair_map(family, map_index)
    start = x0[pair["source_index"][slots]]
    target = x1[pair["target_index"][slots]]
    if family in {"B1", "B3"}:
        target = np.take_along_axis(target, pair["particle_permutation"][slots, :, None], axis=1)
    return start, target


def _particle_displacements(start: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = target - start
    delta -= np.asarray(BOX) * np.floor(delta / np.asarray(BOX) + 0.5)
    return np.linalg.norm(delta, axis=-1)


def _structure(configurations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = configurations[:, :, None, :] - configurations[:, None, :, :]
    delta -= np.asarray(BOX) * np.floor(delta / np.asarray(BOX) + 0.5)
    distance = np.linalg.norm(delta, axis=-1)
    distance[:, np.arange(16), np.arange(16)] = np.inf
    nearest = np.min(distance, axis=-1)
    return np.min(nearest, axis=-1), np.mean(nearest, axis=-1)


def coupling_diagnostics(cfg: dict[str, Any]) -> dict[str, Any]:
    build_coupling_banks(cfg)
    if COUPLING_DIAGNOSTICS_PATH.exists():
        return _json(COUPLING_DIAGNOSTICS_PATH)
    x0, x1 = _training_endpoints()
    rows = {}
    rng = np.random.default_rng(_derive_seed("coupling_diagnostic_noise")["seed"])
    noise = rng.normal(size=(BRIDGE_DIAGNOSTIC_PAIRS, 16, 2))
    slots = np.arange(BRIDGE_DIAGNOSTIC_PAIRS, dtype=np.int32)
    for family in FAMILIES:
        start, target = _bridge_arrays(family, 0, slots, x0, x1)
        displacement = _particle_displacements(start, target)
        pair = _load_pair_map(family, 0)
        bridge = {}
        delta = target - start
        delta -= np.asarray(BOX) * np.floor(delta / np.asarray(BOX) + 0.5)
        for t in (0.25, 0.50, 0.75):
            state = np.mod(start + t * delta + 0.01 * np.sin(np.pi * t) * noise, np.asarray(BOX))
            minimum, nearest = _structure(state)
            bridge[f"t={t:.2f}"] = {"minimum_pair_separation": _distribution(minimum), "mean_nearest_neighbor_distance": _distribution(nearest)}
        nonidentity = np.any(pair["particle_permutation"] != np.arange(16), axis=1)
        rows[family] = {
            "family_name": FAMILY_NAMES[family],
            "mean_periodic_squared_particle_displacement": _distribution(displacement**2),
            "median_particle_displacement_per_pair": _distribution(np.median(displacement, axis=1)),
            "p90_particle_displacement_per_pair": _distribution(np.quantile(displacement, 0.9, axis=1)),
            "maximum_particle_displacement_per_pair": _distribution(np.max(displacement, axis=1)),
            "configuration_transport_cost": _distribution(pair["configuration_cost"]),
            "nonidentity_assignment_fraction": float(np.mean(nonidentity)) if family in {"B1", "B3"} else None,
            "bridge_internal_geometry": bridge,
        }
    rows["B2"]["OT_cost_vs_random_pair_cost_ratio"] = rows["B2"]["configuration_transport_cost"]["mean"] / rows["B0"]["configuration_transport_cost"]["mean"]
    rows["B3"]["OT_cost_vs_random_pair_cost_ratio"] = rows["B3"]["configuration_transport_cost"]["mean"] / rows["B1"]["configuration_transport_cost"]["mean"]
    payload = {"schema_version": 1, "endpoint_only_bridge_internal_diagnostics": True, "hidden_truth_comparison": False, "diagnostic_pair_count": BRIDGE_DIAGNOSTIC_PAIRS, "families": rows}
    _atomic_json(COUPLING_DIAGNOSTICS_PATH, payload)
    return payload


def generate_endpoint_holdout(cfg: dict[str, Any]) -> dict[str, Any]:
    manifest = freeze_manifest(cfg)
    if HOLDOUT_PATH.exists() or HOLDOUT_MANIFEST_PATH.exists():
        if not HOLDOUT_PATH.exists() or not HOLDOUT_MANIFEST_PATH.exists():
            raise RuntimeError("incomplete holdout cache")
        record = _json(HOLDOUT_MANIFEST_PATH)
        if file_sha256(HOLDOUT_PATH) != record["sha256"]:
            raise RuntimeError("endpoint holdout changed")
        return record
    seed = int(manifest["endpoint_holdout"]["seed"]["seed"])
    truth = SkyrmionTruth(_physics_config(cfg))
    started = time.perf_counter()
    bank = truth.make_bank(
        seed=seed, samples=HOLDOUT_N,
        times=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        substeps_per_interval=int(cfg["physics"]["truth_substeps"]) * (TIME_COUNT - 1),
    )
    endpoint0 = np.asarray(bank.configurations[0], dtype=np.float64)
    endpoint1 = np.asarray(bank.configurations[-1], dtype=np.float64)
    del bank
    training0, training1 = _training_endpoints()
    if _array_sha256(endpoint0) in {_array_sha256(training0), _array_sha256(training1)} or _array_sha256(endpoint1) in {_array_sha256(training0), _array_sha256(training1)}:
        raise RuntimeError("holdout duplicates a training endpoint array")
    _atomic_npz(HOLDOUT_PATH, endpoint0=endpoint0, endpoint1=endpoint1, seed=np.asarray(seed, dtype=np.int64))
    record = {
        "schema_version": 1, "development_only": True, "seed": seed, "samples": HOLDOUT_N,
        "shape": list(endpoint0.shape), "retained_time_slices": [0.0, 1.0],
        "retained_intermediate_truth": False, "training_seed_disjoint": True,
        "endpoint0_sha256": _array_sha256(endpoint0), "endpoint1_sha256": _array_sha256(endpoint1),
        "sha256": file_sha256(HOLDOUT_PATH), "wall_time_seconds": time.perf_counter() - started,
    }
    _atomic_json(HOLDOUT_MANIFEST_PATH, record)
    return record


def _model_label(family: str, seed_index: int) -> str:
    return f"{family}_seed{seed_index}"


def _checkpoint_path(family: str, seed_index: int) -> Path:
    return OUTPUT_ROOT / "reference_models" / _model_label(family, seed_index) / "reference.npz"


def _training_record_path(family: str, seed_index: int) -> Path:
    return _checkpoint_path(family, seed_index).with_name("training_record.json")


def _history_path(family: str, seed_index: int) -> Path:
    return _checkpoint_path(family, seed_index).with_name("training_history.json")


def _coupling_arrays(family: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maps = [_load_pair_map(family, i) for i in range(COUPLING_MAPS)]
    return tuple(np.stack([row[key] for row in maps]) for key in ("source_index", "target_index", "particle_permutation"))  # type: ignore[return-value]


def train_coupled_reference(endpoint0: np.ndarray, endpoint1: np.ndarray, family: str, cfg: ReferenceTrainingConfig) -> tuple[EquivariantReferenceFlow, list[dict[str, float]]]:
    source, target, permutations = (jnp.asarray(x) for x in _coupling_arrays(family))
    endpoint0_j, endpoint1_j = jnp.asarray(endpoint0), jnp.asarray(endpoint1)
    key = jax.random.PRNGKey(int(cfg.seed))
    key, init_key = jax.random.split(key)
    params = init_equivariant_reference(init_key, hidden_width=cfg.hidden_width, hidden_layers=cfg.hidden_layers)
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    state = _AdamState(zeros, zeros, jnp.asarray(0, dtype=jnp.int32))

    def sample_batch(batch_key):
        kt, km, ks, kz = jax.random.split(batch_key, 4)
        count = int(cfg.batch_size)
        map_index = jax.random.randint(km, (count,), 0, COUPLING_MAPS)
        slot = jax.random.randint(ks, (count,), 0, TRAIN_ENDPOINTS)
        x0 = endpoint0_j[source[map_index, slot]]
        x1 = endpoint1_j[target[map_index, slot]]
        if family in {"B1", "B3"}:
            x1 = jnp.take_along_axis(x1, permutations[map_index, slot, :, None], axis=1)
        t = jax.random.uniform(kt, (count,), dtype=jnp.float64)
        displacement = minimum_image(x1 - x0, jnp.asarray(BOX))
        noise = jax.random.normal(kz, x0.shape, dtype=jnp.float64)
        gamma = cfg.bridge_noise_std * jnp.sin(jnp.pi * t)[:, None, None]
        gamma_dot = cfg.bridge_noise_std * jnp.pi * jnp.cos(jnp.pi * t)[:, None, None]
        xt = jnp.mod(x0 + t[:, None, None] * displacement + gamma * noise, jnp.asarray(BOX))
        return t, xt, displacement + gamma_dot * noise

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

    history = []
    started = time.perf_counter()
    for index in range(1, cfg.train_steps + 1):
        key, step_key = jax.random.split(key)
        params, state, loss, norm, lr = step(params, state, step_key)
        if index == 1 or index % cfg.log_every == 0 or index == cfg.train_steps:
            history.append({"step": index, "loss": float(loss), "gradient_norm": float(norm), "learning_rate": float(lr), "elapsed_seconds": time.perf_counter() - started})
    metadata = {
        "kind": "permutation_equivariant_endpoint_cfm_v1", "endpoint_only": True,
        "box": list(BOX), "training": asdict(cfg), "bridge_family": family,
        "bridge_family_name": FAMILY_NAMES[family], "coupling_bank_sha256": _coupling_hash(family),
    }
    return EquivariantReferenceFlow(params, box=BOX, metadata=metadata), history


def _coupling_hash(family: str) -> str:
    digest = hashlib.sha256()
    for index in range(COUPLING_MAPS):
        digest.update(bytes.fromhex(file_sha256(_pair_file(family, index))))
    return digest.hexdigest()


def train_models(cfg: dict[str, Any], progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    build_coupling_banks(cfg)
    manifest = freeze_manifest(cfg)
    x0, x1 = _training_endpoints()
    rows = []
    for seed_index, seed_record in enumerate(manifest["training"]["matched_seeds"]):
        train_cfg = replace(ReferenceTrainingConfig(**cfg["reference_training"]), seed=int(seed_record["seed"]))
        for family in FAMILIES:
            checkpoint, record_path = _checkpoint_path(family, seed_index), _training_record_path(family, seed_index)
            if checkpoint.exists() or record_path.exists():
                if not checkpoint.exists() or not record_path.exists():
                    raise RuntimeError(f"incomplete training cache: {family}/{seed_index}")
                record = _json(record_path)
                if record["checkpoint_sha256"] != file_sha256(checkpoint) or record["coupling_bank_sha256"] != _coupling_hash(family):
                    raise RuntimeError(f"training cache seal mismatch: {family}/{seed_index}")
                rows.append(record)
                if progress:
                    progress(f"training {_model_label(family, seed_index)}: cache verified")
                continue
            started = time.perf_counter()
            flow, history = train_coupled_reference(x0, x1, family, train_cfg)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_name("reference.temporary.npz")
            save_reference(temporary, flow)
            os.replace(temporary, checkpoint)
            _atomic_json(_history_path(family, seed_index), history)
            record = {
                "schema_version": 1, "label": _model_label(family, seed_index),
                "family": family, "family_name": FAMILY_NAMES[family], "matched_seed_index": seed_index,
                "training_seed": int(seed_record["seed"]), "endpoint_only": True,
                "endpoint_dataset_sha256": file_sha256(TRUTH_BANKS_PATH),
                "coupling_bank_sha256": _coupling_hash(family),
                "particle_matching_rule_sha256": _payload_sha256(manifest["particle_assignment"]),
                "training_config_sha256": _payload_sha256(asdict(train_cfg)),
                "checkpoint_path": str(checkpoint.relative_to(OUTPUT_ROOT)),
                "checkpoint_sha256": file_sha256(checkpoint), "final_CFM_training_loss": history[-1]["loss"],
                "training_steps": train_cfg.train_steps, "wall_time_seconds": time.perf_counter() - started,
            }
            _atomic_json(record_path, record)
            rows.append(record)
            if progress:
                progress(f"training {_model_label(family, seed_index)}: complete in {record['wall_time_seconds']:.1f}s")
            del flow
            gc.collect()
    payload = {"schema_version": 1, "model_count": len(rows), "models": rows}
    _atomic_json(TRAINING_SUMMARY_PATH, payload)
    return payload


def _fixed_cfm_loss(flow: EquivariantReferenceFlow, family: str, seed: int, endpoint0: np.ndarray, endpoint1: np.ndarray) -> float:
    rng = np.random.default_rng(seed)
    map_index = rng.integers(0, COUPLING_MAPS, size=CFM_EVAL_N, dtype=np.int32)
    slots = rng.integers(0, TRAIN_ENDPOINTS, size=CFM_EVAL_N, dtype=np.int32)
    t_all = rng.random(CFM_EVAL_N)
    noise_all = rng.normal(size=(CFM_EVAL_N, 16, 2))
    maps = [_load_pair_map(family, i) for i in range(COUPLING_MAPS)]
    source = np.stack([row["source_index"] for row in maps])
    target_index = np.stack([row["target_index"] for row in maps])
    perms = np.stack([row["particle_permutation"] for row in maps])
    total = 0.0
    for start_index in range(0, CFM_EVAL_N, 512):
        stop = min(start_index + 512, CFM_EVAL_N)
        m, s = map_index[start_index:stop], slots[start_index:stop]
        x0 = endpoint0[source[m, s]]
        x1 = endpoint1[target_index[m, s]]
        if family in {"B1", "B3"}:
            x1 = np.take_along_axis(x1, perms[m, s, :, None], axis=1)
        t, noise = t_all[start_index:stop], noise_all[start_index:stop]
        delta = _particle_displacement_vectors(x0, x1)
        xt = np.mod(x0 + t[:, None, None] * delta + 0.01 * np.sin(np.pi * t)[:, None, None] * noise, np.asarray(BOX))
        target = delta + 0.01 * np.pi * np.cos(np.pi * t)[:, None, None] * noise
        predicted = np.asarray(flow.velocity(jnp.asarray(xt), jnp.asarray(t)))
        total += float(np.sum(np.sum((predicted - target) ** 2, axis=(-2, -1))))
    return total / CFM_EVAL_N


def _particle_displacement_vectors(start: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = target - start
    return delta - np.asarray(BOX) * np.floor(delta / np.asarray(BOX) + 0.5)


def _rollout(flow: EquivariantReferenceFlow, initial: np.ndarray, cfg: dict[str, Any], final_only: bool = False) -> np.ndarray:
    times = jnp.linspace(0.0, 1.0, TIME_COUNT, dtype=jnp.float64)
    pieces = []
    for start in range(0, len(initial), ROLLOUT_BATCH_SIZE):
        trajectory = flow.rollout(jnp.asarray(initial[start:start + ROLLOUT_BATCH_SIZE]), times, substeps_per_interval=int(cfg["banks"]["reference_substeps"]))
        pieces.append(np.asarray(trajectory[-1] if final_only else trajectory, dtype=np.float64))
    return np.concatenate(pieces, axis=0 if final_only else 1)


def endpoint_qualification(cfg: dict[str, Any], progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    train_models(cfg)
    generate_endpoint_holdout(cfg)
    if ENDPOINT_QUALIFICATION_PATH.exists():
        return _json(ENDPOINT_QUALIFICATION_PATH)
    with np.load(HOLDOUT_PATH, allow_pickle=False) as arrays:
        endpoint0, endpoint1 = np.asarray(arrays["endpoint0"]), np.asarray(arrays["endpoint1"])
    metric = _metric_definition()["metric"]
    family_object, law_eta = _family(cfg), np.asarray([row["eta"] for row in _json(PANEL_PATH)["rows"] if row["panel_role"] == "law"])[0]
    target_psi, target_phi = _mean_psi(endpoint1, BOX), _mean_phi(endpoint1, family_object, law_eta)
    training0, training1 = _training_endpoints()
    cfm_seed = int(freeze_manifest(cfg)["fixed_bridge_CFM_evaluation"]["seed"]["seed"])
    rows = []
    for seed_index in range(MATCHED_SEEDS):
        for family in FAMILIES:
            label, checkpoint = _model_label(family, seed_index), _checkpoint_path(family, seed_index)
            started = time.perf_counter()
            flow = load_reference(checkpoint)
            final = _rollout(flow, endpoint0, cfg, final_only=True)
            psi, phi = _mean_psi(final, BOX), _mean_phi(final, family_object, law_eta)
            delta_psi, delta_phi = psi - target_psi, phi - target_phi
            row = {
                "label": label, "family": family, "matched_seed_index": seed_index,
                "checkpoint_sha256": file_sha256(checkpoint), "holdout_sha256": file_sha256(HOLDOUT_PATH),
                "common_initial_state_sha256": _array_sha256(endpoint0), "final_state_sha256": _array_sha256(final),
                "held_out_bridge_CFM_loss": _fixed_cfm_loss(flow, family, cfm_seed, training0, training1),
                "CFM_evaluation_note": "fixed bridge tuples from training endpoint bank; stochastic tuples held out, endpoint data not independent",
                "endpoint_Psi_delta": delta_psi.tolist(), "endpoint_Psi_euclidean_error": float(np.linalg.norm(delta_psi)),
                "endpoint_scientific_whitened_norm": float(np.sqrt(delta_psi @ metric @ delta_psi)),
                "endpoint_Law_Phi_delta": delta_phi.tolist(), "endpoint_Law_Phi_euclidean_error": float(np.linalg.norm(delta_phi)),
                "endpoint_distribution_metric": None,
                "endpoint_distribution_metric_note": "No authoritative pre-existing skyrmion endpoint distribution metric was found.",
                "rollout": {"integrator": "deterministic periodic RK4", "substeps_per_scientific_interval": 14, "dtype": "float64"},
                "wall_time_seconds": time.perf_counter() - started,
            }
            rows.append(row)
            if progress:
                progress(f"endpoint {label}: complete in {row['wall_time_seconds']:.1f}s")
            del final, flow
            gc.collect()
    payload = {
        "schema_version": 1, "development_only": True, "independent_endpoint_holdout": True,
        "holdout_sha256": file_sha256(HOLDOUT_PATH), "common_initial_state_sha256": _array_sha256(endpoint0),
        "target_P1_Psi_mean": target_psi.tolist(), "target_P1_Law_Phi_mean": target_phi.tolist(), "models": rows,
    }
    _atomic_json(ENDPOINT_QUALIFICATION_PATH, payload)
    return payload


def _initial_states(cfg: dict[str, Any], seed: int) -> np.ndarray:
    truth = SkyrmionTruth(_physics_config(cfg))
    return np.asarray(truth.sample_initial(jax.random.PRNGKey(seed), EVAL_N), dtype=np.float64)


def _result_path(label: str, bank_index: int) -> Path:
    return OUTPUT_ROOT / "reference_eval_results" / label / f"bank_{bank_index:02d}.npz"


def _result_record_path(label: str, bank_index: int) -> Path:
    return _result_path(label, bank_index).with_suffix(".json")


def _raw_risk(bank: GalerkinReferenceBank, truth_means: np.ndarray, whitening: np.ndarray, time_weights: np.ndarray) -> tuple[float, np.ndarray]:
    features = np.asarray(many_body_features(bank.configurations, BOX), dtype=np.float64)
    mean = np.mean(features, axis=1)
    error = mean - truth_means
    by_time = np.einsum("ti,ij,tj->t", error, whitening, error)
    return float(np.sum(time_weights * by_time)), by_time


def evaluate_references(cfg: dict[str, Any], progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    endpoint_qualification(cfg)
    manifest = freeze_manifest(cfg)
    panel = _json(PANEL_PATH)
    etas = np.asarray([row["eta"] for row in panel["rows"]], dtype=np.float64)
    problem, truth_means, whitening = _load_design_context(cfg)
    evaluator = _ReferenceEvaluator(problem, truth_means, whitening)
    bank_rows = []
    for bank_index, seed_record in enumerate(manifest["reference_evaluation"]["seeds"]):
        seed = int(seed_record["seed"])
        initial = _initial_states(cfg, seed)
        initial_hash = _array_sha256(initial)
        model_rows = []
        for seed_index in range(MATCHED_SEEDS):
            for family in FAMILIES:
                label, checkpoint = _model_label(family, seed_index), _checkpoint_path(family, seed_index)
                result_path, record_path = _result_path(label, bank_index), _result_record_path(label, bank_index)
                if result_path.exists() or record_path.exists():
                    if not result_path.exists() or not record_path.exists():
                        raise RuntimeError(f"incomplete evaluation cache: {label}/{bank_index}")
                    record = _json(record_path)
                    checks = (
                        record["result_sha256"] == file_sha256(result_path),
                        record["checkpoint_sha256"] == file_sha256(checkpoint),
                        record["initial_P0_sha256"] == initial_hash,
                        record["candidate_panel_sha256"] == file_sha256(PANEL_PATH),
                    )
                    if not all(checks):
                        raise RuntimeError(f"evaluation cache seal mismatch: {label}/{bank_index}")
                    model_rows.append(record)
                    if progress:
                        progress(f"evaluation {label}/bank{bank_index}: cache verified")
                    continue
                started = time.perf_counter()
                flow = load_reference(checkpoint)
                configurations = _rollout(flow, initial, cfg, final_only=False)
                velocities = []
                times = jnp.linspace(0.0, 1.0, TIME_COUNT, dtype=jnp.float64)
                for start in range(0, EVAL_N, ROLLOUT_BATCH_SIZE):
                    velocities.append(np.asarray(flow.velocity(jnp.asarray(configurations[:, start:start + ROLLOUT_BATCH_SIZE]), times)))
                velocity = np.concatenate(velocities, axis=1)
                weights = np.full(configurations.shape[:2], 1.0 / EVAL_N, dtype=np.float64)
                bank = GalerkinReferenceBank(jnp.asarray(configurations), jnp.asarray(velocity), jnp.asarray(weights))
                result = evaluator.evaluate(etas, bank, EVAL_N)
                raw_risk, raw_by_time = _raw_risk(bank, truth_means, whitening, np.asarray(problem.time_weights))
                result["raw_scientific_risk"] = np.asarray(raw_risk)
                result["raw_risk_by_time"] = raw_by_time
                _atomic_npz(result_path, candidate_index=np.arange(len(etas), dtype=np.int16), **result)
                record = {
                    "schema_version": 1, "label": label, "family": family, "matched_seed_index": seed_index,
                    "bank_index": bank_index, "reference_evaluation_seed": seed, "N": EVAL_N,
                    "checkpoint_sha256": file_sha256(checkpoint), "initial_P0_sha256": initial_hash,
                    "candidate_panel_sha256": file_sha256(PANEL_PATH),
                    "risk_definition_sha256": file_sha256(ROOT / "risk.py"),
                    "W_metric_sha256": _array_sha256(whitening), "rESS_threshold": MINIMUM_RESS,
                    "result_sha256": file_sha256(result_path), "wall_time_seconds": time.perf_counter() - started,
                }
                _atomic_json(record_path, record)
                model_rows.append(record)
                if progress:
                    progress(f"evaluation {label}/bank{bank_index}: complete in {record['wall_time_seconds']:.1f}s")
                del flow, configurations, velocity, weights, bank, result
                gc.collect()
        if len({row["initial_P0_sha256"] for row in model_rows}) != 1:
            raise RuntimeError("common evaluation P0 hashes differ")
        bank_rows.append({"bank_index": bank_index, "seed_record": seed_record, "initial_P0_sha256": initial_hash, "models": model_rows})
        del initial
    payload = {
        "schema_version": 1, "development_only": True, "N": EVAL_N,
        "bank_count": EVAL_BANKS, "common_P0_within_bank": True,
        "candidate_panel_sha256": file_sha256(PANEL_PATH), "banks": bank_rows,
    }
    _atomic_json(EVAL_MANIFEST_PATH, payload)
    return payload


def _load_results(label: str) -> dict[str, np.ndarray]:
    rows = []
    for bank in range(EVAL_BANKS):
        path = _result_path(label, bank)
        record = _json(_result_record_path(label, bank))
        if file_sha256(path) != record["result_sha256"]:
            raise RuntimeError(f"result changed: {label}/{bank}")
        with np.load(path, allow_pickle=False) as arrays:
            rows.append({key: np.asarray(arrays[key]) for key in arrays.files if key != "candidate_index"})
    return {key: np.stack([row[key] for row in rows]) for key in rows[0]}


def _panel_indices() -> tuple[int, np.ndarray, np.ndarray]:
    rows = _json(PANEL_PATH)["rows"]
    law = [i for i, row in enumerate(rows) if row["panel_role"] == "law"]
    high = np.asarray([i for i, row in enumerate(rows) if row["panel_role"] == "high_pass_ge24_of_32"])
    controls = np.asarray([i for i, row in enumerate(rows) if "control" in row["panel_role"]])
    if len(law) != 1 or len(high) != HIGH_PASS_COUNT or len(controls) != CONTROL_COUNT:
        raise RuntimeError("panel roles changed")
    return law[0], high, controls


def summarize(cfg: dict[str, Any]) -> dict[str, Any]:
    evaluate_references(cfg)
    endpoint = endpoint_qualification(cfg)
    law, high, _ = _panel_indices()
    endpoint_by_label = {row["label"]: row for row in endpoint["models"]}
    model_rows, arrays = [], {}
    for seed_index in range(MATCHED_SEEDS):
        for family in FAMILIES:
            label = _model_label(family, seed_index)
            values = _load_results(label)
            arrays[label] = values
            pass_matrix = values["ress_trajectory"][:, high, NODE7] >= MINIMUM_RESS
            candidate_passes = np.sum(pass_matrix, axis=0)
            law_projected = values["scientific_risk"][:, law]
            relative = 100.0 * (values["scientific_risk"][:, high] / law_projected[:, None] - 1.0)
            row = {
                "label": label, "family": family, "matched_seed_index": seed_index,
                "endpoint": endpoint_by_label[label],
                "law": {
                    "raw_scientific_risk": _distribution(values["raw_scientific_risk"]),
                    "projected_scientific_risk": _distribution(law_projected),
                    "minimum_rESS": _distribution(values["minimum_ress"][:, law]),
                    "node7_rESS": _distribution(values["ress_trajectory"][:, law, NODE7]),
                    "node7_lambda_norm": _distribution(values["lambda_norm"][:, law, NODE7]),
                    "node7_top1pct_mass": _distribution(values["top_1pct_weight_mass"][:, law, NODE7]),
                },
                "panel": {
                    "median_node7_rESS": float(np.median(values["ress_trajectory"][:, high, NODE7])),
                    "p10_node7_rESS": float(np.quantile(values["ress_trajectory"][:, high, NODE7], 0.10)),
                    "candidate_bank_rESS_pass_fraction": float(np.mean(pass_matrix)),
                    "candidates_at_least_2_of_4": int(np.sum(candidate_passes >= 2)),
                    "candidates_at_least_3_of_4": int(np.sum(candidate_passes >= 3)),
                    "candidates_4_of_4": int(np.sum(candidate_passes == 4)),
                    "median_projected_scientific_risk": float(np.median(values["scientific_risk"][:, high])),
                    "median_Law_relative_risk_increase_percent": float(np.median(relative)),
                },
            }
            model_rows.append(row)
    family_rows, variability = {}, {}
    for family in FAMILIES:
        rows = [row for row in model_rows if row["family"] == family]
        def vals(path):
            return np.asarray([path(row) for row in rows], dtype=np.float64)
        family_rows[family] = {
            "endpoint_whitened_norm": _distribution(vals(lambda r: r["endpoint"]["endpoint_scientific_whitened_norm"])),
            "raw_Law_risk": _distribution(vals(lambda r: r["law"]["raw_scientific_risk"]["median"])),
            "projected_Law_risk": _distribution(vals(lambda r: r["law"]["projected_scientific_risk"]["median"])),
            "Law_node7_rESS": _distribution(vals(lambda r: r["law"]["node7_rESS"]["median"])),
            "panel_median_node7_rESS": _distribution(vals(lambda r: r["panel"]["median_node7_rESS"])),
            "panel_pass_fraction": _distribution(vals(lambda r: r["panel"]["candidate_bank_rESS_pass_fraction"])),
        }
        variability[family] = {
            "SD_endpoint_Psi_error": float(np.std(vals(lambda r: r["endpoint"]["endpoint_Psi_euclidean_error"]))),
            "SD_raw_Law_risk": float(np.std(vals(lambda r: r["law"]["raw_scientific_risk"]["median"]))),
            "SD_projected_Law_risk": float(np.std(vals(lambda r: r["law"]["projected_scientific_risk"]["median"]))),
            "SD_Law_node7_rESS": float(np.std(vals(lambda r: r["law"]["node7_rESS"]["median"]))),
            "SD_panel_median_node7_rESS": float(np.std(vals(lambda r: r["panel"]["median_node7_rESS"]))),
        }
    paired = {}
    principal = {
        "endpoint_Psi_error": lambda r: r["endpoint"]["endpoint_Psi_euclidean_error"],
        "endpoint_Law_Phi_error": lambda r: r["endpoint"]["endpoint_Law_Phi_euclidean_error"],
        "raw_Law_risk": lambda r: r["law"]["raw_scientific_risk"]["median"],
        "projected_Law_risk": lambda r: r["law"]["projected_scientific_risk"]["median"],
        "Law_node7_rESS": lambda r: r["law"]["node7_rESS"]["median"],
        "Law_lambda_norm": lambda r: r["law"]["node7_lambda_norm"]["median"],
        "Law_top1pct_mass": lambda r: r["law"]["node7_top1pct_mass"]["median"],
        "panel_median_node7_rESS": lambda r: r["panel"]["median_node7_rESS"],
    }
    by_label = {row["label"]: row for row in model_rows}
    for family in ("B1", "B2", "B3"):
        seed_rows = []
        for seed_index in range(MATCHED_SEEDS):
            current, baseline = by_label[_model_label(family, seed_index)], by_label[_model_label("B0", seed_index)]
            seed_rows.append({"matched_seed_index": seed_index, **{name: float(fn(current) - fn(baseline)) for name, fn in principal.items()}})
        paired[family] = {"per_seed": seed_rows, "medians": {name: float(np.median([row[name] for row in seed_rows])) for name in principal}}
    ready = _readiness(model_rows, family_rows, variability, paired)
    interpretation, candidate = _interpret(ready, family_rows, paired)
    family_payload = {"schema_version": 1, "models": model_rows, "families": family_rows, "readiness": ready}
    paired_payload = {"schema_version": 1, "comparisons_vs_matched_B0": paired}
    variability_payload = {"schema_version": 1, "families": variability}
    summary = {
        "schema_version": 1, "development_only": True, "interpretation": interpretation,
        "bridge_family_ready_for_single_reference_preflight": candidate != "NONE", "candidate_family": candidate,
        "models": model_rows, "families": family_rows, "readiness": ready,
        "safeguards": {"physical_benchmark_changed": False, "scientific_risk_changed": False, "rESS_changed": False, "intermediate_truth_used_in_reference_training": False, "validation_accessed": False, "production_reference_replaced": False, "sensor_optimization": False, "Tangent": False, "Full": False, "official_protocol_created": False},
    }
    _atomic_json(FAMILY_SUMMARY_PATH, family_payload)
    _atomic_json(PAIRED_PATH, paired_payload)
    _atomic_json(VARIABILITY_PATH, variability_payload)
    _atomic_json(SUMMARY_PATH, summary)
    _atomic_text(REPORT_PATH, _report(summary, paired, variability))
    _write_inventory()
    return summary


def _readiness(models: list[dict[str, Any]], families: dict[str, Any], variability: dict[str, Any], paired: dict[str, Any]) -> dict[str, Any]:
    baseline_endpoint = families["B0"]["endpoint_whitened_norm"]["median"]
    baseline_sd = variability["B0"]["SD_projected_Law_risk"]
    result = {}
    for family in ("B1", "B2", "B3"):
        rows = [row for row in models if row["family"] == family]
        per_seed_joint = []
        for row in rows:
            delta = paired[family]["per_seed"][row["matched_seed_index"]]["Law_node7_rESS"]
            per_seed_joint.append(
                row["endpoint"]["endpoint_scientific_whitened_norm"] <= 1.25 * baseline_endpoint
                and row["law"]["raw_scientific_risk"]["median"] < 20
                and row["law"]["projected_scientific_risk"]["median"] < 20
                and delta >= -0.005
            )
        conditions = {
            "endpoint_comparable": families[family]["endpoint_whitened_norm"]["median"] <= 1.25 * baseline_endpoint,
            "no_catastrophic_risk": all(row["law"]["raw_scientific_risk"]["median"] < 20 and row["law"]["projected_scientific_risk"]["median"] < 20 for row in rows),
            "rESS_majority": sum(row["Law_node7_rESS"] >= -0.005 for row in paired[family]["per_seed"]) >= 2,
            "risk_variability_reduced": variability[family]["SD_projected_Law_risk"] <= 0.8 * baseline_sd,
            "nontrivial_panel_support": families[family]["panel_pass_fraction"]["median"] >= 0.10,
            "not_one_seed": sum(per_seed_joint) >= 2,
        }
        result[family] = {"ready": all(conditions.values()), "conditions": conditions, "individual_joint_passes": per_seed_joint}
    return result


def _interpret(ready: dict[str, Any], families: dict[str, Any], paired: dict[str, Any]) -> tuple[str, str]:
    ready_set = {family for family in ("B1", "B2", "B3") if ready[family]["ready"]}
    if not ready_set:
        return "BRIDGE_CHANGES_DO_NOT_RESOLVE_REFERENCE_TRADEOFF", "NONE"
    if ready_set == {"B1"}:
        return "PARTICLE_LABEL_COUPLING_WAS_PRIMARY", "B1"
    if ready_set == {"B2"}:
        return "CONFIGURATION_COUPLING_WAS_PRIMARY", "B2"
    if "B3" in ready_set:
        b3_better_b1 = paired["B3"]["medians"]["projected_Law_risk"] < paired["B1"]["medians"]["projected_Law_risk"] and paired["B3"]["medians"]["Law_node7_rESS"] > paired["B1"]["medians"]["Law_node7_rESS"]
        b3_better_b2 = paired["B3"]["medians"]["projected_Law_risk"] < paired["B2"]["medians"]["projected_Law_risk"] and paired["B3"]["medians"]["Law_node7_rESS"] > paired["B2"]["medians"]["Law_node7_rESS"]
        if b3_better_b1 and b3_better_b2:
            return "BOTH_COUPLINGS_MATTER", "B3"
        if "B1" in ready_set:
            return "PERMUTATION_MATCHING_SUFFICIENT", "B1"
        if "B2" in ready_set:
            return "OT_COUPLING_SUFFICIENT", "B2"
        return "BOTH_COUPLINGS_MATTER", "B3"
    return "MIXED_BRIDGE_EFFECT", sorted(ready_set)[0]


def _report(summary: dict[str, Any], paired: dict[str, Any], variability: dict[str, Any]) -> str:
    lines = [
        "# Permutation-Aware + OT Endpoint-Bridge Ablation", "", "SOURCE VERIFIED", "",
        "physical benchmark changed: NO", "scientific risk changed: NO", "rESS changed: NO",
        "intermediate truth used in reference training: NO", "validation accessed: NO", "",
        "## COUPLING GEOMETRY", "",
        "| family | mean displacement² | p90 displacement | mean config cost | nonidentity assignment fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    diagnostics = _json(COUPLING_DIAGNOSTICS_PATH)["families"]
    for family in FAMILIES:
        row = diagnostics[family]
        nonidentity = row["nonidentity_assignment_fraction"]
        lines.append(f"| {family} | {row['mean_periodic_squared_particle_displacement']['mean']:.6f} | {row['p90_particle_displacement_per_pair']['mean']:.6f} | {row['configuration_transport_cost']['mean']:.6f} | {'—' if nonidentity is None else f'{nonidentity:.6f}'} |")
    lines += ["", "## ENDPOINT QUALIFICATION", "", "| family/seed | CFM loss | endpoint Psi error | endpoint whitened Psi | endpoint Law-Phi error | endpoint distribution metric |", "|---|---:|---:|---:|---:|---|"]
    for row in summary["models"]:
        endpoint = row["endpoint"]
        lines.append(f"| {row['label']} | {endpoint['held_out_bridge_CFM_loss']:.6f} | {endpoint['endpoint_Psi_euclidean_error']:.6f} | {endpoint['endpoint_scientific_whitened_norm']:.6f} | {endpoint['endpoint_Law_Phi_euclidean_error']:.6f} | not available |")
    lines += ["", "## REFERENCE SCIENTIFIC QUALITY", "", "| family/seed | raw Law risk | projected Law risk | node7 rESS | node7 lambda | node7 top1% mass |", "|---|---:|---:|---:|---:|---:|"]
    for row in summary["models"]:
        law = row["law"]
        lines.append(f"| {row['label']} | {law['raw_scientific_risk']['median']:.6f} | {law['projected_scientific_risk']['median']:.6f} | {law['node7_rESS']['median']:.6f} | {law['node7_lambda_norm']['median']:.3f} | {law['node7_top1pct_mass']['median']:.6f} |")
    lines += ["", "## LOW-RISK PANEL", "", "| family/seed | median node7 rESS | candidate-bank rESS pass fraction | candidates 4/4 |", "|---|---:|---:|---:|"]
    for row in summary["models"]:
        panel = row["panel"]
        lines.append(f"| {row['label']} | {panel['median_node7_rESS']:.6f} | {panel['candidate_bank_rESS_pass_fraction']:.4f} | {panel['candidates_4_of_4']} |")
    lines += ["", "## SEED STABILITY", "", "| family | SD endpoint error | SD raw risk | SD projected risk | SD node7 rESS |", "|---|---:|---:|---:|---:|"]
    for family in FAMILIES:
        row = variability[family]
        lines.append(f"| {family} | {row['SD_endpoint_Psi_error']:.6f} | {row['SD_raw_Law_risk']:.6f} | {row['SD_projected_Law_risk']:.6f} | {row['SD_Law_node7_rESS']:.6f} |")
    lines += ["", "## PAIRED BRIDGE EFFECTS", "", "Deltas are modified family minus matched B0.", "", "| family/seed | endpoint Psi | Law-Phi | raw risk | projected risk | node7 rESS | lambda | top1% mass | panel node7 rESS |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for family in ("B1", "B2", "B3"):
        for row in paired[family]["per_seed"]:
            lines.append(f"| {family}/seed{row['matched_seed_index']} | {row['endpoint_Psi_error']:+.6f} | {row['endpoint_Law_Phi_error']:+.6f} | {row['raw_Law_risk']:+.6f} | {row['projected_Law_risk']:+.6f} | {row['Law_node7_rESS']:+.6f} | {row['Law_lambda_norm']:+.3f} | {row['Law_top1pct_mass']:+.6f} | {row['panel_median_node7_rESS']:+.6f} |")
    lines += [
        "", "## DEVELOPMENT INTERPRETATION", "", summary["interpretation"], "",
        "BRIDGE FAMILY READY FOR SINGLE-REFERENCE PREFLIGHT:",
        "YES" if summary["bridge_family_ready_for_single_reference_preflight"] else "NO", "",
        f"candidate family: {summary['candidate_family']}", "",
    ]
    if summary["candidate_family"] != "NONE":
        lines += ["Recommended next task: SINGLE_REFERENCE_B3_PREFLIGHT" if summary["candidate_family"] == "B3" else f"Recommended next task: SINGLE_REFERENCE_{summary['candidate_family']}_PREFLIGHT", ""]
    else:
        lines += ["Recommended next task: separately specify a more structured endpoint-only bridge study.", ""]
    lines += ["NO production reference replaced", "NO sensor optimization", "NO Tangent", "NO Full", "NO validation", "NO official protocol created", ""]
    return "\n".join(lines)


def _write_inventory() -> dict[str, Any]:
    files = []
    for path in sorted(OUTPUT_ROOT.rglob("*")):
        if path.is_file() and path != INVENTORY_PATH:
            files.append({"path": str(path.relative_to(OUTPUT_ROOT)), "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    payload = {"schema_version": 1, "artifact_count": len(files), "files": files}
    _atomic_json(INVENTORY_PATH, payload)
    return payload


def console_report() -> str:
    if not REPORT_PATH.exists():
        raise RuntimeError("report has not been generated")
    return REPORT_PATH.read_text(encoding="utf-8")


def run_all(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    cfg = _json(CONFIG_PATH)
    freeze_manifest(cfg)
    build_coupling_banks(cfg, progress=progress)
    coupling_diagnostics(cfg)
    generate_endpoint_holdout(cfg)
    train_models(cfg, progress=progress)
    endpoint_qualification(cfg, progress=progress)
    evaluate_references(cfg, progress=progress)
    return summarize(cfg)
