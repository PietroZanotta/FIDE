"""Resumable three-reference B1 Galerkin Pareto selection study.

The three independently trained B1 particle-matched flows from the frozen
bridge ablation are treated as an equal-weight design ensemble.  Optimization
uses the ensemble-mixture action, while scientific-risk and forcing-support
eligibility must hold separately for every reference flow.
"""

from __future__ import annotations

import copy
import csv
import gc
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from . import official_b1_pareto as single
from . import reference_seed_robustness as reference_screen
from .galerkin_only_data import GalerkinReferenceBank, SelectionGalerkinData
from .pareto_v3_common import eta_key, file_sha256
from .reference import load_reference
from .reference_seed_robustness import _array_sha256
from .risk import many_body_features
from .single_reference_b1_preflight import _family, _make_problem


ROOT = Path(__file__).resolve().parent
VERSION = "skyrmion_b1_galerkin_pareto_3references_v1"
OUTPUT_ROOT = ROOT / "outputs" / VERSION
CONFIG_PATH = ROOT / "config.json"
PROTOCOL_PATH = OUTPUT_ROOT / "protocol.json"
PROTOCOL_HASH_PATH = OUTPUT_ROOT / "protocol_hash.txt"
DESIGN_PATH = OUTPUT_ROOT / "design_truth" / "design_truth.npz"
DESIGN_RECORD = OUTPUT_ROOT / "design_truth" / "manifest.json"
ARTIFACT_DIR = OUTPUT_ROOT / "artifacts"
LAW_PATH = OUTPUT_ROOT / "law" / "official_law.json"
SCREENING_PATH = OUTPUT_ROOT / "screening" / "candidate_pool.json"
FINAL_PARETO = OUTPUT_ROOT / "pareto.json"
RUNTIME_PATCH_RECEIPT = OUTPUT_ROOT / "runtime_implementation_patch.json"
RUNTIME_PATCH_RECEIPT_2 = OUTPUT_ROOT / "runtime_implementation_patch_2.json"
RUNTIME_PATCH_RECEIPT_3 = OUTPUT_ROOT / "runtime_implementation_patch_3.json"
RUNTIME_PATCH_RECEIPT_4 = OUTPUT_ROOT / "runtime_implementation_patch_4.json"
RUNTIME_PATCH_RECEIPT_5 = OUTPUT_ROOT / "runtime_implementation_patch_5.json"
RUNTIME_PATCH_RECEIPT_6 = OUTPUT_ROOT / "runtime_implementation_patch_6.json"
RUNTIME_PATCH_RECEIPT_7 = OUTPUT_ROOT / "runtime_implementation_patch_7.json"
RUNTIME_PATCH_RECEIPT_8 = OUTPUT_ROOT / "runtime_implementation_patch_8.json"
RUNTIME_PATCH_RECEIPT_9 = OUTPUT_ROOT / "runtime_implementation_patch_9.json"
RUNTIME_PATCH_RECEIPT_10 = OUTPUT_ROOT / "runtime_implementation_patch_10.json"
RUNTIME_PATCH_RECEIPT_11 = OUTPUT_ROOT / "runtime_implementation_patch_11.json"
RUNTIME_PATCH_RECEIPT_12 = OUTPUT_ROOT / "runtime_implementation_patch_12.json"
GLOBAL_SEED = 20260830
FLOW_IDS = ("B1_seed0", "B1_seed1", "B1_seed2")
BRIDGE_ROOT = ROOT / "outputs" / "skyrmion_galerkin_dev_bridge_ablation_v1"
FLOW_PATHS = {
    flow_id: BRIDGE_ROOT / "reference_models" / flow_id / "reference.npz"
    for flow_id in FLOW_IDS
}
FLOW_SHA256 = {
    "B1_seed0": "9f62701a3f163cf826c8abb60e79b369b409eee7486e0bc399900f10d575867f",
    "B1_seed1": "ed42abca42f53fd7557644bdb33edff467d29899a0688c554592977f7754fa02",
    "B1_seed2": "62a9a4cb3b1b09e0c8e2b426fc985af084232019f65ca5a10c1190d8a8681c26",
}
# The scientific protocol was frozen before the flow-wise feature-construction
# memory patch.  Keep its prospective source seal immutable and record the
# exact, semantics-preserving runtime patch separately.
FROZEN_IMPLEMENTATION_SHA256 = "49aa77433b1250f3da4e4c5a13a91def89fe074132cc833f0a700e1f2dadb315"
FLOWWISE_FEATURE_PATCH_SHA256 = "30dc97ae60049e3d1c40afe6572b706fed39d6511ab42da11d2564f434c5c46f"
SHARED_PROJECTION_PATCH_SHA256 = "1a771684f1f85343a66c5c7c45fc142ef82bb6e705de0f6e38244d32d74efd43"
LAZY_CERTIFICATION_PATCH_SHA256 = "26f73131648164a3d925c1dab387ea350061a697d345b3a65593c19efeca9634"
PER_FLOW_OBJECTIVE_PATCH_SHA256 = "b37ebbf17da4e24e5d5b64c0df94c0f2353611821ab4db620cb18a5866dd91cd"
PHASED_LOADING_PATCH_SHA256 = "ace7e186b4c43e94aa3c088a684bc3ba1910b26d98ef345780701c825289c30e"
TRAJECTORY_RESUME_PATCH_SHA256 = "196e1b2d6aef01b0125e3d2f7e8890544586077ccf92c8af3501c283c6b0d12f"
ENDPOINT_POOL_PATCH_SHA256 = "eea1c9bdb9c5bbc0d1a49ab1fefe2c8ea98b688f3a9a2ad58f23a84b47f164aa"
FULL_ENDPOINT_POOL_PATCH_SHA256 = "85972b3f600346d1c04ae8106eae3b1e9c3dd4089dca3c22e6b30968ac866215"
_SHARED_SELECTION: dict[str, tuple[Any, ...]] = {}
_FLOW_SELECTION_SHARED: dict[str, tuple[Any, ...]] = {}
ALLOWANCES = single.ALLOWANCES
BANK_SIZES = dict(single.BANK_SIZES)
K = single.K
MINIMUM_RESS = single.MINIMUM_RESS
BOX = single.BOX


def _activate() -> None:
    """Point the reusable single-reference machinery at this isolated run."""
    values = {
        "VERSION": VERSION,
        "OUTPUT_ROOT": OUTPUT_ROOT,
        "PROTOCOL_PATH": PROTOCOL_PATH,
        "PROTOCOL_HASH_PATH": PROTOCOL_HASH_PATH,
        "DESIGN_PATH": DESIGN_PATH,
        "DESIGN_RECORD": DESIGN_RECORD,
        "LAW_PATH": LAW_PATH,
        "LAW_POOL_PATH": OUTPUT_ROOT / "law" / "search_pool.json",
        "LAW_RESULTS_PATH": OUTPUT_ROOT / "law" / "search_results.json",
        "CANDIDATE_SPEC": OUTPUT_ROOT / "candidate_pool" / "generator_spec.json",
        "CANDIDATE_POOL": OUTPUT_ROOT / "candidate_pool" / "candidate_pool.json",
        "CANDIDATE_RESULTS": OUTPUT_ROOT / "candidate_pool" / "support_results.npz",
        "SCREENING_PATH": SCREENING_PATH,
        "ARTIFACT_DIR": ARTIFACT_DIR,
        "GLOBAL_SEED": GLOBAL_SEED,
        "CHECKPOINT": FLOW_PATHS[FLOW_IDS[0]],
        "CHECKPOINT_SHA256": FLOW_SHA256[FLOW_IDS[0]],
    }
    for name, value in values.items():
        setattr(single, name, value)
    single.protocol_payload = protocol_payload
    single.freeze_protocol = freeze_protocol
    single.require_protocol = require_protocol
    single.generate_banks = generate_banks
    single._bank_path = _bank_path
    single.load_bank = load_bank
    single.selection_data = selection_data
    single._evaluate = evaluate_references
    single.screen_candidates = screen_candidates
    single.select_tangent = select_tangent
    single.select_full = select_full


def derive_seed(label: str) -> dict[str, Any]:
    text = f"{GLOBAL_SEED}:{VERSION}:selection:{label}"
    digest = __import__("hashlib").sha256(text.encode()).hexdigest()
    return {"scope": "selection", "label": label, "text": text,
            "sha256": digest, "seed": int(digest[:16], 16) % (2**31 - 1)}


def protocol_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    for flow_id, path in FLOW_PATHS.items():
        if file_sha256(path) != FLOW_SHA256[flow_id]:
            raise RuntimeError(f"frozen B1 checkpoint changed: {flow_id}")
    bridge = single.read_json(BRIDGE_ROOT / "summary.json")
    if (not bridge["bridge_family_ready_for_single_reference_preflight"]
            or bridge["candidate_family"] != "B1"):
        raise RuntimeError("frozen B1 bridge evidence no longer authorizes the ensemble")
    if file_sha256(single.DICTIONARY_PATH) != single.DICTIONARY_SHA256:
        raise RuntimeError("K280 dictionary changed")
    labels = ["design_truth", "selection_observation_noise", "law_search_pool",
              "candidate_local", "candidate_tangent", "candidate_paths",
              "candidate_sobol", *BANK_SIZES]
    seeds = [derive_seed(label) for label in labels]
    return {
        "schema_version": 1,
        "version": VERSION,
        "study_type": "three-reference B1 robust Pareto selection",
        "reference": {
            "flow_ids": list(FLOW_IDS),
            "checkpoint_sha256": FLOW_SHA256,
            "equal_weight": True,
            "matched_initial_configurations_across_flows": True,
            "B1_particle_matching": True,
            "configuration_OT": False,
            "retrained_for_this_study": False,
            "source_bridge_summary_sha256": file_sha256(BRIDGE_ROOT / "summary.json"),
        },
        "source_hashes": {
            "three_reference_pareto.py": FROZEN_IMPLEMENTATION_SHA256,
            "official_b1_pareto.py": file_sha256(ROOT / "official_b1_pareto.py"),
            "pareto_v2_selection.py": file_sha256(ROOT / "pareto_v2_selection.py"),
            "config.json": file_sha256(CONFIG_PATH),
        },
        "constants": {
            "dtype": "float64", "minimum_rESS": MINIMUM_RESS, "K": K,
            "dictionary_sha256": single.DICTIONARY_SHA256,
            "flow_count": len(FLOW_IDS), "objective": "equal-weight mean action",
        },
        "data": {"design_truth_N": single.DESIGN_N, "bank_sizes_per_flow": BANK_SIZES,
                 "selection_seed_records": seeds},
        "law": {
            "algorithm": "same frozen pool/refinement algorithm as official B1 v1",
            "selection_objective": "mean scientific risk across three flows",
            "support_required_for_every_flow": True,
            "development_R_Law_used_as_anchor": False,
        },
        "candidate_generator": {
            "count": single.CANDIDATE_COUNT,
            "component_targets": {"local": 1434, "risk_tangent": 1024,
                                  "periodic_paths": 819, "sobol": 819},
            "canonicalization": "periodic wrap plus unordered-sensor matching",
            "frozen_before_scoring": True,
        },
        "screening": {"screen_N_per_flow": BANK_SIZES["screen"],
                      "audit_N_per_flow": BANK_SIZES["periodic_audit"],
                      "every_flow_required": True},
        "starts": {"Tangent": 6, "Full": 3},
        "optimizer": {
            "maximum_accepted_step_attempts": 1, "maximum_backtracks": 3,
            "initial_step": 5e-5, "backtrack_factor": 0.5,
            "trust_radius": 2e-4, "periodic_audit_every_accepted_steps": 1,
            "replacement_tolerance": 1e-10, "rank_must_equal_previous_step": True,
            "tangent": "exact Gram objective on equal-weight flow mixture",
            "full": "K280 fixed-coefficient envelope on equal-weight flow mixture",
        },
        "allowances_percent": list(ALLOWANCES),
        "risk_rule": "R_j(eta) <= (1+p/100) R_j(Law) separately for every flow j",
        "allowance_failures_independent": True,
        "nested_incumbent_rule": True,
        "finalization": {
            "per_flow_risk_and_support_recertification": True,
            "optimized_endpoint_pool_includes_all_trajectories_and_starts": True,
            "validation_accessed": False,
        },
        "full_method": "equal-weight three-flow fixed-feature K=280 Galerkin approximation",
        "deep_ritz_used": False,
    }


def freeze_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    body = protocol_payload(cfg)
    digest = single.payload_sha256(body)
    wrapped = {**body, "protocol_sha256": digest, "protocol_frozen": True}
    if PROTOCOL_PATH.exists():
        if single.read_json(PROTOCOL_PATH) != wrapped:
            raise RuntimeError("three-reference protocol seal mismatch")
        if PROTOCOL_HASH_PATH.read_text().strip() != digest:
            raise RuntimeError("three-reference protocol hash mismatch")
    else:
        single.atomic_json(PROTOCOL_PATH, wrapped)
        single.atomic_text(PROTOCOL_HASH_PATH, digest + "\n")
    return wrapped


def require_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    if not PROTOCOL_PATH.exists() or not PROTOCOL_HASH_PATH.exists():
        raise RuntimeError("freeze the three-reference protocol first")
    saved = single.read_json(PROTOCOL_PATH)
    body = {key: value for key, value in saved.items()
            if key not in {"protocol_sha256", "protocol_frozen"}}
    if protocol_payload(cfg) != body or single.payload_sha256(body) != saved["protocol_sha256"]:
        raise RuntimeError("three-reference protocol differs from code/config")
    return saved


def freeze_runtime_patch() -> dict[str, Any]:
    first = {
        "schema_version": 1,
        "reason": "avoid a 4.88 GiB temporary pair tensor when constructing mixture risk features",
        "frozen_protocol_source_sha256": FROZEN_IMPLEMENTATION_SHA256,
        "runtime_source_sha256": FLOWWISE_FEATURE_PATCH_SHA256,
        "change": "compute many-body features independently per flow, then concatenate feature channels",
        "mathematical_result_changed": False,
        "completed_artifacts_reused": True,
    }
    if RUNTIME_PATCH_RECEIPT.exists():
        if single.read_json(RUNTIME_PATCH_RECEIPT) != first:
            raise RuntimeError("runtime implementation changed after the OOM patch was sealed")
    else:
        single.atomic_json(RUNTIME_PATCH_RECEIPT, first)
    second = {
        "schema_version": 1,
        "reason": "avoid recomputing immutable projection features for search and authoritative data views",
        "parent_patch_sha256": file_sha256(RUNTIME_PATCH_RECEIPT),
        "runtime_source_sha256": SHARED_PROJECTION_PATCH_SHA256,
        "change": "cache and share the projection bank, risk features, problem, truth means, and whitening",
        "mathematical_result_changed": False,
        "completed_artifacts_reused": True,
    }
    if RUNTIME_PATCH_RECEIPT_2.exists():
        if single.read_json(RUNTIME_PATCH_RECEIPT_2) != second:
            raise RuntimeError("runtime implementation changed after the shared-projection patch was sealed")
    else:
        single.atomic_json(RUNTIME_PATCH_RECEIPT_2, second)
    third = {
        "schema_version": 1,
        "reason": "avoid holding search and authoritative reference mixtures concurrently",
        "parent_patch_sha256": file_sha256(RUNTIME_PATCH_RECEIPT_2),
        "runtime_source_sha256": LAZY_CERTIFICATION_PATCH_SHA256,
        "change": "load, evaluate, and release search and authoritative data sequentially per allowance",
        "mathematical_result_changed": False,
        "completed_artifacts_reused": True,
    }
    if RUNTIME_PATCH_RECEIPT_3.exists():
        if single.read_json(RUNTIME_PATCH_RECEIPT_3) != third:
            raise RuntimeError("runtime implementation changed after the lazy-certification patch was sealed")
    else:
        single.atomic_json(RUNTIME_PATCH_RECEIPT_3, third)
    fourth = {
        "schema_version": 1,
        "reason": "avoid differentiating a concatenated three-flow particle tensor",
        "parent_patch_sha256": file_sha256(RUNTIME_PATCH_RECEIPT_3),
        "runtime_source_sha256": PER_FLOW_OBJECTIVE_PATCH_SHA256,
        "change": "evaluate each flow independently and average the three action values and gradients",
        "mathematical_result_changed": False,
        "clarification": "implements the prospectively declared equal-weight mean-action objective literally",
        "completed_artifacts_reused": True,
    }
    if RUNTIME_PATCH_RECEIPT_4.exists():
        if single.read_json(RUNTIME_PATCH_RECEIPT_4) != fourth:
            raise RuntimeError("runtime implementation changed after the per-flow objective patch was sealed")
    else:
        single.atomic_json(RUNTIME_PATCH_RECEIPT_4, fourth)
    fifth = {
        "schema_version": 1,
        "reason": "avoid feature temporaries while authoritative train/audit banks are resident",
        "parent_patch_sha256": file_sha256(RUNTIME_PATCH_RECEIPT_4),
        "runtime_source_sha256": PHASED_LOADING_PATCH_SHA256,
        "change": "precompute all per-flow risk features before loading any train/audit bank set",
        "mathematical_result_changed": False,
        "completed_artifacts_reused": True,
    }
    if RUNTIME_PATCH_RECEIPT_5.exists():
        if single.read_json(RUNTIME_PATCH_RECEIPT_5) != fifth:
            raise RuntimeError("runtime implementation changed after the phased-loading patch was sealed")
    else:
        single.atomic_json(RUNTIME_PATCH_RECEIPT_5, fifth)
    sixth = {
        "schema_version": 1,
        "reason": "resume completed immutable Tangent trajectories after interrupted certification",
        "parent_patch_sha256": file_sha256(RUNTIME_PATCH_RECEIPT_5),
        "runtime_source_sha256": TRAJECTORY_RESUME_PATCH_SHA256,
        "change": "read an existing trajectory receipt instead of recomputing or overwriting it",
        "mathematical_result_changed": False,
        "completed_artifacts_reused": True,
    }
    if RUNTIME_PATCH_RECEIPT_6.exists():
        if single.read_json(RUNTIME_PATCH_RECEIPT_6) != sixth:
            raise RuntimeError("runtime implementation changed after the trajectory-resume patch was sealed")
    else:
        single.atomic_json(RUNTIME_PATCH_RECEIPT_6, sixth)
    seventh = {
        "schema_version": 1,
        "reason": "apply the frozen finalization rule to every trajectory endpoint",
        "parent_patch_sha256": file_sha256(RUNTIME_PATCH_RECEIPT_6),
        "runtime_source_sha256": ENDPOINT_POOL_PATCH_SHA256,
        "change": "authoritatively recertify all endpoints, including lower-fidelity search failures",
        "protocol_alignment": "optimized_endpoint_pool_includes_all_trajectories_and_starts",
        "completed_artifacts_reused": True,
    }
    if RUNTIME_PATCH_RECEIPT_7.exists():
        if single.read_json(RUNTIME_PATCH_RECEIPT_7) != seventh:
            raise RuntimeError("runtime implementation changed after the endpoint-pool patch was sealed")
    else:
        single.atomic_json(RUNTIME_PATCH_RECEIPT_7, seventh)
    eighth = {
        "schema_version": 1,
        "reason": "apply the same frozen endpoint-pool rule to Full trajectories",
        "parent_patch_sha256": file_sha256(RUNTIME_PATCH_RECEIPT_7),
        "runtime_source_sha256": FULL_ENDPOINT_POOL_PATCH_SHA256,
        "change": "authoritatively recertify every Full endpoint regardless of search-bank eligibility",
        "protocol_alignment": "optimized_endpoint_pool_includes_all_trajectories_and_starts",
        "completed_artifacts_reused": True,
    }
    if RUNTIME_PATCH_RECEIPT_8.exists():
        if single.read_json(RUNTIME_PATCH_RECEIPT_8) != eighth:
            raise RuntimeError("runtime implementation changed after the Full endpoint-pool patch was sealed")
    else:
        single.atomic_json(RUNTIME_PATCH_RECEIPT_8, eighth)
    ninth = {
        "schema_version": 1,
        "reason": "retain strict gates while allowing the remaining Pareto allowances to run",
        "parent_patch_sha256": file_sha256(RUNTIME_PATCH_RECEIPT_8),
        "runtime_source_sha256": "af15d1186f011724a8d9f78a9203820efd4221f33f090199ab2c942c0f44790d",
        "change": "serialize a no-certified-Full-point result instead of aborting the sweep",
        "thresholds_relaxed": False,
        "completed_artifacts_reused": True,
    }
    if RUNTIME_PATCH_RECEIPT_9.exists():
        if single.read_json(RUNTIME_PATCH_RECEIPT_9) != ninth:
            raise RuntimeError("runtime implementation changed after the Pareto-gap patch was sealed")
    else:
        single.atomic_json(RUNTIME_PATCH_RECEIPT_9, ninth)
    tenth = {
        "schema_version": 1,
        "reason": "make frozen-pool forcing-support screening match the production forcing definition",
        "parent_patch_sha256": file_sha256(RUNTIME_PATCH_RECEIPT_9),
        "runtime_source_sha256": "1edbf4b05a9c931173191b15018f9ddceabe62561be4a0757a6f5735deef4383",
        "reference_evaluator_source_sha256": file_sha256(ROOT / "reference_seed_robustness.py"),
        "change": (
            "center the lambda-dot forcing term on target moments rather than achieved "
            "projected moments, and require the same forcing audit for Tangent endpoints"
        ),
        "mathematical_result_changed": True,
        "thresholds_relaxed": False,
        "invalidated_artifacts": ["screening", "tangent", "full", "final Pareto"],
        "completed_artifacts_reused": ["protocol", "banks", "Law", "candidate pool"],
    }
    if RUNTIME_PATCH_RECEIPT_10.exists():
        if single.read_json(RUNTIME_PATCH_RECEIPT_10) != tenth:
            raise RuntimeError("runtime implementation changed after the forcing-screen correction was sealed")
    else:
        single.atomic_json(RUNTIME_PATCH_RECEIPT_10, tenth)
    eleventh = {
        "schema_version": 1,
        "reason": "prevent the optimizer from receiving starts that fail its search-train forcing gate",
        "parent_patch_sha256": file_sha256(RUNTIME_PATCH_RECEIPT_10),
        "runtime_source_sha256": "434e7862e81c1497dd1ae957a7bd27eb418c1bf2109d327f7ac8bb6993fba229",
        "change": (
            "require every candidate start to pass target-centered forcing support "
            "on screen, periodic-audit, and search-train banks for every flow"
        ),
        "mathematical_result_changed": True,
        "thresholds_relaxed": False,
        "validation_accessed": False,
        "invalidated_artifacts": ["screening starts", "tangent", "full", "final Pareto"],
        "completed_artifacts_reused": ["protocol", "banks", "Law", "candidate pool"],
    }
    if RUNTIME_PATCH_RECEIPT_11.exists():
        if single.read_json(RUNTIME_PATCH_RECEIPT_11) != eleventh:
            raise RuntimeError("runtime implementation changed after the search-train screen was sealed")
    else:
        single.atomic_json(RUNTIME_PATCH_RECEIPT_11, eleventh)
    twelfth = {
        "schema_version": 1,
        "reason": "finish every allowance after strict frozen-pool feasibility gaps",
        "parent_patch_sha256": file_sha256(RUNTIME_PATCH_RECEIPT_11),
        "runtime_source_sha256": file_sha256(Path(__file__)),
        "change": (
            "record an independent no-feasible-start gap instead of aborting, and reuse "
            "the corrected screen/audit receipt while adding search-train diagnostics"
        ),
        "mathematical_result_changed": False,
        "thresholds_relaxed": False,
        "validation_accessed": False,
        "completed_artifacts_reused": [
            "protocol", "banks", "Law", "candidate pool", "corrected screen/audit evaluations"
        ],
    }
    if RUNTIME_PATCH_RECEIPT_12.exists():
        if single.read_json(RUNTIME_PATCH_RECEIPT_12) != twelfth:
            raise RuntimeError("runtime implementation changed after feasibility-gap handling was sealed")
    else:
        single.atomic_json(RUNTIME_PATCH_RECEIPT_12, twelfth)
    return twelfth


def _seed(protocol: dict[str, Any], label: str) -> dict[str, Any]:
    return next(row for row in protocol["data"]["selection_seed_records"]
                if row["label"] == label)


def _bank_path(label: str, flow_id: str | None = None) -> Path:
    if flow_id is None:
        flow_id = FLOW_IDS[0]
    return OUTPUT_ROOT / "banks" / flow_id / f"{label}_N{BANK_SIZES[label]}.npz"


def load_flow_bank(label: str, flow_id: str) -> GalerkinReferenceBank:
    with np.load(_bank_path(label, flow_id), allow_pickle=False) as values:
        return GalerkinReferenceBank(jnp.asarray(values["configurations"]),
                                     jnp.asarray(values["velocity"]),
                                     jnp.asarray(values["base_weights"]))


def load_bank(label: str) -> GalerkinReferenceBank:
    banks = [load_flow_bank(label, flow_id) for flow_id in FLOW_IDS]
    count = sum(int(bank.configurations.shape[1]) for bank in banks)
    return GalerkinReferenceBank(
        jnp.concatenate([bank.configurations for bank in banks], axis=1),
        jnp.concatenate([bank.velocity for bank in banks], axis=1),
        jnp.full((13, count), 1.0 / count, dtype=jnp.float64),
    )


def generate_banks(cfg: dict[str, Any], progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    protocol = require_protocol(cfg)
    single.generate_design_truth(cfg, progress)
    records = []
    times = jnp.linspace(0, 1, 13, dtype=jnp.float64)
    from .domain import SkyrmionTruth
    from .single_reference_b1_preflight import _physics_config
    for label, count in BANK_SIZES.items():
        seed = _seed(protocol, label)["seed"]
        truth = SkyrmionTruth(_physics_config(cfg))
        initial = truth.sample_initial(jax.random.PRNGKey(seed), count)
        initial_hash = _array_sha256(np.asarray(initial))
        for flow_id in FLOW_IDS:
            path = _bank_path(label, flow_id)
            record_path = path.with_suffix(".json")
            if record_path.exists():
                row = single.read_json(record_path)
                if file_sha256(path) != row["sha256"]:
                    raise RuntimeError(f"bank changed: {flow_id}/{label}")
                records.append(row)
                continue
            started = time.perf_counter()
            flow = load_reference(FLOW_PATHS[flow_id])
            configurations, velocities = [], []
            for start in range(0, count, 2048):
                rows = flow.rollout(initial[start:start + 2048], times,
                                    substeps_per_interval=int(cfg["banks"]["reference_substeps"]))
                configurations.append(np.asarray(rows))
                velocities.append(np.asarray(flow.velocity(rows, times)))
            x = np.concatenate(configurations, axis=1)
            v = np.concatenate(velocities, axis=1)
            single.atomic_npz(path, configurations=x, velocity=v,
                              base_weights=np.full((13, count), 1.0 / count),
                              role=np.asarray(label), flow_id=np.asarray(flow_id), seed=np.asarray(seed))
            row = {"flow_id": flow_id, "label": label, "N": count, "seed": seed,
                   "initial_P0_sha256": initial_hash, "checkpoint_sha256": FLOW_SHA256[flow_id],
                   "sha256": file_sha256(path), "wall_time_seconds": time.perf_counter() - started}
            single.atomic_json(record_path, row)
            records.append(row)
            if progress:
                progress(f"three-reference bank {flow_id}/{label}: N={count}")
            del flow, x, v
            gc.collect()
    matched = all(len({row["initial_P0_sha256"] for row in records if row["label"] == label}) == 1
                  for label in BANK_SIZES)
    manifest = {"schema_version": 1, "passed": matched, "flow_ids": list(FLOW_IDS),
                "bank_sizes_per_flow": BANK_SIZES, "equal_weight": True,
                "matched_initial_configurations_across_flows": matched,
                "records": records, "validation_accessed": False}
    single.atomic_json(OUTPUT_ROOT / "banks" / "manifest.json", manifest, immutable=False)
    return manifest


def _flow_selection_shared(cfg: dict[str, Any], flow_id: str,
                           projection: str = "risk_anchor") -> tuple[Any, ...]:
    generate_banks(cfg)
    key = f"{projection}:{flow_id}"
    if key not in _FLOW_SELECTION_SHARED:
        with np.load(DESIGN_PATH, allow_pickle=False) as values:
            times, configurations, truth_means, whitening = (
                jnp.asarray(values[name])
                for name in ("times", "configurations", "truth_means", "whitening"))
        protocol = require_protocol(cfg)
        problem = _make_problem(cfg, configurations, times, _family(cfg),
            noise_seed=_seed(protocol, "selection_observation_noise")["seed"])
        projection_bank = load_flow_bank(projection, flow_id)
        reference_features = many_body_features(projection_bank.configurations, BOX)
        _FLOW_SELECTION_SHARED[key] = (
            problem, projection_bank, reference_features, truth_means, whitening)
    return _FLOW_SELECTION_SHARED[key]


def selection_data_for_flow(cfg: dict[str, Any], train: str, audit: str,
                            flow_id: str, *, projection: str = "risk_anchor") -> SelectionGalerkinData:
    problem, projection_bank, reference_features, truth_means, whitening = (
        _flow_selection_shared(cfg, flow_id, projection))
    return SelectionGalerkinData(problem, projection_bank, load_flow_bank(train, flow_id),
        load_flow_bank(audit, flow_id), reference_features, truth_means, whitening)


def selection_data(cfg: dict[str, Any], train: str, audit: str,
                   *, projection: str = "risk_anchor") -> SelectionGalerkinData:
    generate_banks(cfg)
    if projection not in _SHARED_SELECTION:
        with np.load(DESIGN_PATH, allow_pickle=False) as values:
            times, configurations, truth_means, whitening = (
                jnp.asarray(values[key])
                for key in ("times", "configurations", "truth_means", "whitening"))
        protocol = require_protocol(cfg)
        problem = _make_problem(cfg, configurations, times, _family(cfg),
            noise_seed=_seed(protocol, "selection_observation_noise")["seed"])
        projection_bank = load_bank(projection)
        # Computing pair features after concatenating configurations creates a
        # multi-GiB temporary [time, 3N, particle, particle, xy] tensor. Feature
        # extraction is sample-separable, so concatenate the small results.
        reference_features = jnp.concatenate([
            many_body_features(load_flow_bank(projection, flow_id).configurations, BOX)
            for flow_id in FLOW_IDS
        ], axis=1)
        _SHARED_SELECTION[projection] = (
            problem, projection_bank, reference_features, truth_means, whitening)
    problem, projection_bank, reference_features, truth_means, whitening = _SHARED_SELECTION[projection]
    return SelectionGalerkinData(problem, projection_bank, load_bank(train), load_bank(audit),
        reference_features, truth_means, whitening)


class _TargetCenteredReferenceEvaluator(reference_screen._ReferenceEvaluator):
    """Reference evaluator using the same forcing definition as production."""

    def _postprocessor(self, N: int):
        if N in self.postprocessors:
            return self.postprocessors[N]
        top_count = max(1, int(math.ceil(0.01 * N)))
        problem, truth_means, whitening = self.problem, self.truth_means, self.whitening

        @jax.jit
        def postprocess(
            weights,
            lam,
            moments,
            covariance,
            residual,
            ess,
            features,
            advective,
            derivatives,
            reference_features,
        ):
            moment_m = jnp.einsum("btn,btnr->btr", weights, advective)
            scalar_m = jnp.einsum("btnr,btr->btn", advective, lam)
            centered_phi = features - moments[:, :, None, :]
            centered_g = scalar_m - jnp.einsum(
                "btn,btn->bt", weights, scalar_m
            )[:, :, None]
            covariance_phi_g = jnp.einsum(
                "btn,btnr,btn->btr", weights, centered_phi, centered_g
            )
            rhs = derivatives - moment_m - covariance_phi_g
            regularized = (
                covariance
                + float(problem.forcing_config.covariance_ridge)
                * jnp.eye(features.shape[-1])
            )
            lambda_dot = jnp.linalg.solve(regularized, rhs[..., None])[..., 0]
            # projected residual = achieved moments - target moments.  Production
            # continuity_forcing centers this term on the targets, not the achieved
            # moments, so the residual remains visible to the compatibility gate.
            targets = moments - residual
            forcing = (
                jnp.einsum(
                    "btr,btnr->btn",
                    lambda_dot,
                    features - targets[:, :, None, :],
                )
                + jnp.einsum(
                    "btr,btnr->btn", lam, advective - moment_m[:, :, None, :]
                )
            )
            forcing_mean = jnp.abs(jnp.einsum("btn,btn->bt", weights, forcing))
            eigenvalues = jnp.linalg.eigvalsh(regularized)
            condition = eigenvalues[..., -1] / jnp.maximum(
                eigenvalues[..., 0], 1e-300
            )
            top_mass = jnp.sum(jax.lax.top_k(weights, top_count)[0], axis=-1)
            predicted = jnp.einsum("btn,tnf->btf", weights, reference_features)
            error = predicted - truth_means[None, ...]
            risk_by_time = jnp.einsum(
                "bti,ij,btj->bt", error, whitening, error
            )
            risk = jnp.sum(problem.time_weights[None, :] * risk_by_time, axis=1)
            return (
                ess,
                jnp.linalg.norm(lam, axis=-1),
                jnp.max(weights, axis=-1),
                top_mass,
                -jnp.log(jnp.maximum(ess, 1e-300)),
                condition,
                jnp.linalg.norm(residual, axis=-1),
                forcing_mean,
                risk,
            )

        self.postprocessors[N] = postprocess
        return postprocess


def evaluate_references(etas: np.ndarray, cfg: dict[str, Any], label: str) -> dict[str, np.ndarray]:
    per_flow = []
    for flow_id in FLOW_IDS:
        data = selection_data_for_flow(cfg, label, label, flow_id, projection=label)
        evaluator = _TargetCenteredReferenceEvaluator(
            data.selection_problem,
            np.asarray(data.truth_means),
            np.asarray(data.whitening),
        )
        per_flow.append(evaluator.evaluate(etas, load_flow_bank(label, flow_id), BANK_SIZES[label]))
    risks = np.stack([np.asarray(row["scientific_risk"]) for row in per_flow])
    support = np.stack([np.asarray(row["support_valid"]) for row in per_flow])
    ress = np.stack([np.asarray(row["minimum_ress"]) for row in per_flow])
    forcing_mean = np.stack(
        [np.asarray(row["maximum_forcing_mean"]) for row in per_flow]
    )
    result = dict(per_flow[0])
    result["scientific_risk"] = np.mean(risks, axis=0)
    result["support_valid"] = np.all(support, axis=0)
    result["minimum_ress"] = np.min(ress, axis=0)
    result["per_flow_scientific_risk"] = risks
    result["per_flow_support_valid"] = support
    result["per_flow_minimum_ress"] = ress
    result["per_flow_maximum_forcing_mean"] = forcing_mean
    return result


def _law_flow_risks(cfg: dict[str, Any], law_eta: Any) -> np.ndarray:
    result = evaluate_references(np.asarray([law_eta]), cfg, "risk_anchor")
    return np.asarray(result["per_flow_scientific_risk"])[:, 0]


def screen_candidates(cfg: dict[str, Any], progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    protocol = require_protocol(cfg)
    pool = single.generate_candidates(cfg, progress)
    law = single.read_json(LAW_PATH)
    if SCREENING_PATH.exists():
        return single.read_json(SCREENING_PATH)
    reusable_path = (
        OUTPUT_ROOT / "superseded_corrected_screen_without_search_train_v10"
        / "screening" / "candidate_pool.json"
    )
    reusable = single.read_json(reusable_path) if reusable_path.exists() else None
    if reusable is not None and reusable.get("candidate_pool_sha256") != file_sha256(single.CANDIDATE_POOL):
        raise RuntimeError("corrected screen/audit receipt does not match the frozen candidate pool")
    source_rows = pool["rows"] if reusable is None else reusable["rows"]
    etas = np.asarray([row["eta"] for row in source_rows])
    train = evaluate_references(etas, cfg, "search_train")
    if reusable is None:
        risk = evaluate_references(etas, cfg, "risk_anchor")
        screen = evaluate_references(etas, cfg, "screen")
        audit = evaluate_references(etas, cfg, "periodic_audit")
        law_risks = _law_flow_risks(cfg, law["eta_Law_official"])
    else:
        law_risks = np.asarray(
            [reusable["law_risk_by_flow"][flow_id] for flow_id in FLOW_IDS]
        )
    rows = []
    for index, source in enumerate(source_rows):
        train_valid = np.asarray(train["per_flow_support_valid"])[:, index]
        train_forcing_mean = np.asarray(
            train["per_flow_maximum_forcing_mean"]
        )[:, index]
        if reusable is None:
            flow_risks = np.asarray(risk["per_flow_scientific_risk"])[:, index]
            screen_valid = np.asarray(screen["per_flow_support_valid"])[:, index]
            audit_valid = np.asarray(audit["per_flow_support_valid"])[:, index]
            screen_forcing_mean = np.asarray(
                screen["per_flow_maximum_forcing_mean"]
            )[:, index]
            audit_forcing_mean = np.asarray(
                audit["per_flow_maximum_forcing_mean"]
            )[:, index]
            base = {
                **source,
                "scientific_selection_risk": float(np.mean(flow_risks)),
                "per_flow_scientific_risk": dict(zip(FLOW_IDS, map(float, flow_risks))),
                "per_flow_screen_valid": dict(zip(FLOW_IDS, map(bool, screen_valid))),
                "per_flow_audit_valid": dict(zip(FLOW_IDS, map(bool, audit_valid))),
                "per_flow_screen_maximum_forcing_mean": dict(
                    zip(FLOW_IDS, map(float, screen_forcing_mean))
                ),
                "per_flow_audit_maximum_forcing_mean": dict(
                    zip(FLOW_IDS, map(float, audit_forcing_mean))
                ),
                "minimum_ess_fraction": float(np.min(np.minimum(
                    np.asarray(screen["per_flow_minimum_ress"])[:, index],
                    np.asarray(audit["per_flow_minimum_ress"])[:, index],
                ))),
            }
            base_valid = bool(np.all(screen_valid) and np.all(audit_valid))
        else:
            base = dict(source)
            base_valid = bool(source["projection_valid"])
        minimum_ess = min(
            float(base["minimum_ess_fraction"]),
            float(np.min(np.asarray(train["per_flow_minimum_ress"])[:, index])),
        )
        rows.append({**base,
                     "per_flow_search_train_valid": dict(zip(FLOW_IDS, map(bool, train_valid))),
                     "per_flow_search_train_maximum_forcing_mean": dict(
                         zip(FLOW_IDS, map(float, train_forcing_mean))
                     ),
                     "projection_valid": bool(base_valid and np.all(train_valid)),
                     "minimum_ess_fraction": minimum_ess,
                     "robust_rESS": minimum_ess})
    starts, feasible_count_by_allowance = {}, {}
    for allowance in ALLOWANCES:
        ceilings = (1 + allowance / 100) * law_risks
        eligible = [row for row in rows if row["projection_valid"] and
                    np.all(np.asarray(list(row["per_flow_scientific_risk"].values())) <= ceilings)]
        feasible_count_by_allowance[single.slug(allowance)] = len(eligible)
        selected = []
        if eligible:
            for row, role in ((min(eligible, key=lambda x: (x["scientific_selection_risk"], x["candidate_id"])), "low_risk"),
                              (max(eligible, key=lambda x: (x["robust_rESS"], x["candidate_id"])), "best_ress")):
                if not any(eta_key(row["eta"]) == eta_key(old["eta"]) for old in selected):
                    selected.append({**row, "start_role": role})
        while len(selected) < 6:
            remaining = [row for row in eligible if not any(eta_key(row["eta"]) == eta_key(old["eta"]) for old in selected)]
            if not remaining:
                break
            row = max(remaining, key=lambda x: (min(single._symmetry_aware_distance(x["eta"], old["eta"], BOX)
                                                   for old in selected), x["candidate_id"]))
            selected.append({**row, "start_role": "maxmin_diverse"})
        starts[single.slug(allowance)] = selected
    gap_count = sum(count == 0 for count in feasible_count_by_allowance.values())
    result = {"schema_version": 2, "passed": True,
              "status": "COMPLETE" if gap_count == 0 else "COMPLETE_WITH_GAPS",
              "screening_gap_count": gap_count,
              "signature": single.signature(protocol, "three_reference_screening"),
              "law_risk": float(np.mean(law_risks)),
              "law_risk_by_flow": dict(zip(FLOW_IDS, map(float, law_risks))),
              "law_eta": law["eta_Law_official"], "pool_count": len(rows), "rows": rows,
              "starts": starts,
              "feasible_count_by_allowance": feasible_count_by_allowance,
              "search_train_support_required": True,
              "reused_corrected_screen_audit": reusable is not None,
              "validation_accessed": False,
              "candidate_pool_sha256": file_sha256(single.CANDIDATE_POOL)}
    single.atomic_json(SCREENING_PATH, result)
    if progress:
        progress(f"three-reference screen: {sum(row['projection_valid'] for row in rows)}/{len(rows)}")
    return result


def _release_stage_data() -> None:
    gc.collect()
    jax.clear_caches()


def _flow_data(cfg: dict[str, Any], train: str, audit: str) -> list[SelectionGalerkinData]:
    for flow_id in FLOW_IDS:
        _flow_selection_shared(cfg, flow_id)
    return [selection_data_for_flow(cfg, train, audit, flow_id) for flow_id in FLOW_IDS]


def _ensemble_tangent_eval(engine: Any, data: list[SelectionGalerkinData],
                           eta: Any, *, gradient: bool) -> dict[str, Any]:
    rows = [engine._tangent_eval(item, eta, gradient=gradient) for item in data]
    gradients = None if not gradient else np.mean(
        np.asarray([row["gradient"] for row in rows], dtype=np.float64), axis=0).tolist()
    return {
        "eta": rows[0]["eta"],
        "action": float(np.mean([row["action"] for row in rows])),
        "gradient": gradients,
        "risk": float(np.mean([row["risk"] for row in rows])),
        "risk_by_flow": dict(zip(FLOW_IDS, (float(row["risk"]) for row in rows))),
        "forcing_by_flow": dict(zip(FLOW_IDS, (row["forcing"] for row in rows))),
        "geometry_valid": all(row["geometry_valid"] for row in rows),
        "valid": all(row["valid"] for row in rows),
        "per_flow_action": dict(zip(FLOW_IDS, (float(row["action"]) for row in rows))),
    }


def _ensemble_tangent_audit(engine: Any, data: list[SelectionGalerkinData], eta: Any) -> dict[str, Any]:
    rows = []
    for item in data:
        tangent = engine._tangent_audit(item, eta)
        reconstruction = engine.reconstruct_moments(eta, item.selection_problem)
        state = engine.forcing_state(
            eta, item.selection_problem, item.audit_bank, reconstruction
        )
        forcing = engine._forcing_state_payload(state, item.selection_problem)
        rows.append({
            **tangent,
            "forcing_audit": forcing,
            "valid": bool(tangent["valid"] and forcing["valid"]),
        })
    return {"valid": all(row["valid"] for row in rows),
            "per_flow": dict(zip(FLOW_IDS, rows)), "equal_weight": True}


def _ensemble_risk_gradient(engine: Any, data: list[SelectionGalerkinData], eta: Any) -> jax.Array:
    gradients = [jax.value_and_grad(lambda point, item=item:
        engine.selection_risk(point, item))(eta)[1] for item in data]
    return jnp.mean(jnp.stack(gradients), axis=0)


class EnsembleFullContext:
    """Mean of three independently solved K280 Full objectives."""

    def __init__(self, engine: Any, cfg: dict[str, Any], data: list[SelectionGalerkinData],
                 law_risks: np.ndarray):
        self.engine, self.contexts = engine, [engine.FullContext(cfg, item) for item in data]
        self.data = data[0]
        self.law_risks = jnp.asarray(law_risks, dtype=jnp.float64)
        self.mean_law = float(np.mean(law_risks))

        def robust_risk(eta: Any) -> jax.Array:
            risks = jnp.stack([engine.selection_risk(eta, context.data)
                               for context in self.contexts])
            return jnp.max(risks / self.law_risks) * self.mean_law

        self._risk = jax.jit(robust_risk)
        self._risk_grad = jax.jit(jax.value_and_grad(robust_risk))

    def evaluate(self, eta: Any, gradient: bool = True) -> dict[str, Any]:
        rows = [context.evaluate(eta, gradient=gradient) for context in self.contexts]
        result = dict(rows[0])
        result["action"] = float(np.mean([row["action"] for row in rows]))
        result["risk"] = float(self._risk(rows[0]["_eta"]))
        result["risk_by_flow"] = dict(zip(FLOW_IDS,
            (float(row["risk"]) for row in rows)))
        result["per_flow_action"] = dict(zip(FLOW_IDS,
            (float(row["action"]) for row in rows)))
        result["gradient"] = (None if not gradient else np.mean(
            np.asarray([row["gradient"] for row in rows], dtype=np.float64), axis=0).tolist())
        result["gradient_norm"] = (None if result["gradient"] is None else
                                   float(np.linalg.norm(result["gradient"])))
        result["rank_by_time"] = [row["rank_by_time"] for row in rows]
        result["train_forcing_audit"] = {
            "valid": all(row["train_forcing_audit"]["valid"] for row in rows),
            "per_flow": dict(zip(FLOW_IDS, (row["train_forcing_audit"] for row in rows))),
        }
        result["search_valid"] = all(row["search_valid"] for row in rows)
        result["algebra_valid"] = all(row["algebra_valid"] for row in rows)
        result["_per_flow_raw"] = rows
        return result

    def audit(self, evaluation: dict[str, Any], *, require_physical: bool) -> dict[str, Any]:
        rows = [context.audit(raw, require_physical=require_physical)
                for context, raw in zip(self.contexts, evaluation["_per_flow_raw"], strict=True)]
        return {"valid": all(row["valid"] for row in rows),
                "per_flow": dict(zip(FLOW_IDS, rows)),
                "require_physical": require_physical, "equal_weight": True}


def select_tangent(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run the unchanged Tangent algorithm with lazy authoritative loading."""
    from . import pareto_v2_selection as engine
    screen_candidates(cfg)
    engine = single.configure_selection_engine(single.official_config(cfg), start_cap=6)
    protocol = require_protocol(cfg)
    screening = single.read_json(SCREENING_PATH)
    path = OUTPUT_ROOT / "tangent" / "selection.json"
    if path.exists():
        return single.read_json(path)
    law_risk, incumbent, results = screening["law_risk"], None, []
    law_risks = np.asarray([screening["law_risk_by_flow"][flow_id] for flow_id in FLOW_IDS])
    for allowance in ALLOWANCES:
        ceiling = single.selection_ceiling(law_risk, allowance)
        ceilings = (1 + allowance / 100) * law_risks
        method_starts = engine._method_starts(screening, allowance, incumbent)
        if not method_starts and incumbent is None:
            point = {
                "allowance_percent": allowance,
                "ceiling": ceiling,
                "risk_ceiling_by_flow": dict(zip(FLOW_IDS, map(float, ceilings))),
                "trajectories": [],
                "finalists": [],
                "winner": None,
                "status": "NO_FEASIBLE_SCREENED_START",
                "incumbent_retained": False,
            }
            results.append(point)
            single.atomic_json(
                OUTPUT_ROOT / "tangent" / f"allowance_{single.slug(allowance)}"
                / "result.json",
                point,
            )
            continue
        data = _flow_data(single.official_config(cfg), "search_train", "periodic_audit")
        trajectories = []
        for index, start in enumerate(method_starts):
            trajectory_path = (OUTPUT_ROOT / "tangent" / f"allowance_{single.slug(allowance)}"
                               / f"trajectory_{index:02d}.json")
            if trajectory_path.exists():
                trajectories.append(single.read_json(trajectory_path))
                continue
            center = jnp.asarray(start["eta"])
            current = _ensemble_tangent_eval(engine, data, center, gradient=True)
            start_audit = _ensemble_tangent_audit(engine, data, center)
            history = []
            current_risks = np.asarray(list(current["risk_by_flow"].values()))
            if current["valid"] and np.all(current_risks <= ceilings) and start_audit["valid"]:
                eta = center
                for step in range(int(protocol["optimizer"]["maximum_accepted_step_attempts"])):
                    risk_gradient = _ensemble_risk_gradient(engine, data, eta)
                    direction = engine._projected_direction(jnp.asarray(current["gradient"]), risk_gradient)
                    accepted, attempts = None, []
                    for backtrack in range(int(protocol["optimizer"]["maximum_backtracks"])):
                        length = float(protocol["optimizer"]["initial_step"]) * 0.5 ** backtrack
                        proposal = engine.wrap_periodic(eta + length * direction,
                                                        data[0].selection_problem.family)
                        candidate = _ensemble_tangent_eval(engine, data, proposal, gradient=True)
                        candidate_risks = np.asarray(list(candidate["risk_by_flow"].values()))
                        if (candidate["valid"] and np.all(candidate_risks <= ceilings)
                                and candidate["action"] < current["action"] - 1e-10):
                            accepted = candidate
                            attempts.append({"length": length, "accepted": True})
                            break
                        attempts.append({"length": length, "accepted": False})
                    history.append({"step": step + 1, "attempts": attempts,
                                    "accepted": accepted is not None})
                    if accepted is None:
                        break
                    current, eta = accepted, jnp.asarray(accepted["eta"])
            audit = _ensemble_tangent_audit(engine, data, current["eta"])
            endpoint_risks = np.asarray(list(current["risk_by_flow"].values()))
            row = {"start": start, "endpoint": current, "audit": audit, "history": history,
                   "eligible": bool(current["valid"] and np.all(endpoint_risks <= ceilings)
                                    and audit["valid"])}
            trajectories.append(row)
            single.atomic_json(trajectory_path, row)
        finalists = [row["endpoint"] for row in trajectories]
        if incumbent is not None:
            finalists.append(incumbent)
        unique_finalists = {eta_key(row["eta"]): row for row in finalists}
        finalists = sorted(unique_finalists.values(),
                           key=lambda row: (row["action"], eta_key(row["eta"])))
        del data
        _release_stage_data()
        authoritative = _flow_data(single.official_config(cfg), "authoritative_train",
                                   "authoritative_audit")
        certified = []
        for finalist in finalists:
            train = _ensemble_tangent_eval(engine, authoritative, finalist["eta"], gradient=False)
            audit = _ensemble_tangent_audit(engine, authoritative, finalist["eta"])
            train_risks = np.asarray(list(train["risk_by_flow"].values()))
            if np.all(train_risks <= ceilings) and train["valid"] and audit["valid"]:
                certified.append({**train, "authoritative_audit": audit})
        incumbent_auth = (None if incumbent is None else
            next((row for row in certified if eta_key(row["eta"]) == eta_key(incumbent["eta"])), incumbent))
        best = min(certified, key=lambda row: (row["action"], eta_key(row["eta"]))) if certified else None
        winner = (incumbent_auth if incumbent_auth is not None
                  and (best is None or best["action"] >= incumbent_auth["action"] - 1e-10)
                  else best)
        point = {"allowance_percent": allowance, "ceiling": ceiling,
                 "trajectories": trajectories, "finalists": certified, "winner": winner,
                 "status": "CERTIFIED" if winner is not None else "NO_CERTIFIED_TANGENT_POINT",
                 "incumbent_retained": incumbent_auth is not None
                    and winner is not None
                    and eta_key(winner["eta"]) == eta_key(incumbent_auth["eta"])}
        results.append(point)
        incumbent = winner
        single.atomic_json(OUTPUT_ROOT / "tangent" / f"allowance_{single.slug(allowance)}"
                           / "result.json", point)
        del authoritative
        _release_stage_data()
    winners = [row["winner"] for row in results if row["winner"] is not None]
    passed = all(b["action"] <= a["action"] + 1e-10
                 for a, b in zip(winners[:-1], winners[1:]))
    gap_count = sum(row["winner"] is None for row in results)
    result = {"schema_version": 2, "passed": passed,
              "protocol_sha256": protocol["protocol_sha256"], "allowances": results,
              "status": "COMPLETE" if gap_count == 0 else "COMPLETE_WITH_GAPS",
              "certified_point_count": len(winners), "gap_count": gap_count,
              "validation_accessed": False, "lazy_authoritative_loading": True}
    single.atomic_json(path, result)
    return result


def select_full(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run the unchanged Full algorithm with lazy authoritative loading."""
    screen_candidates(cfg)
    from . import pareto_v2_selection as engine
    engine = single.configure_selection_engine(single.official_config(cfg), start_cap=3)
    protocol = require_protocol(cfg)
    screening = single.read_json(SCREENING_PATH)
    path = OUTPUT_ROOT / "full_search" / "selection.json"
    if path.exists():
        return single.read_json(path)
    incumbent, results = None, []
    law_risks = np.asarray([screening["law_risk_by_flow"][flow_id] for flow_id in FLOW_IDS])
    for allowance in ALLOWANCES:
        ceiling = single.selection_ceiling(screening["law_risk"], allowance)
        ceilings = (1.0 + allowance / 100.0) * law_risks
        method_starts = engine._method_starts(screening, allowance, incumbent)
        if not method_starts and incumbent is None:
            point = {
                "allowance_percent": allowance,
                "risk_ceiling": ceiling,
                "normalized_robust_risk_ceiling": ceiling,
                "maximum_risk_ratio": 1.0 + allowance / 100.0,
                "risk_ceiling_by_flow": dict(zip(FLOW_IDS, map(float, ceilings))),
                "trajectories": [],
                "shortlist": [],
                "authoritative_finalists": [],
                "winner": None,
                "status": "NO_FEASIBLE_SCREENED_START",
                "incumbent_retained": False,
            }
            results.append(point)
            single.atomic_json(
                OUTPUT_ROOT / "authoritative" / f"allowance_{single.slug(allowance)}"
                / "result.json",
                point,
            )
            continue
        search = EnsembleFullContext(engine, single.official_config(cfg),
            _flow_data(single.official_config(cfg), "search_train", "periodic_audit"), law_risks)
        trajectories = [engine._trajectory(single.official_config(cfg), protocol, search, start,
            allowance, OUTPUT_ROOT / "full_search" / f"allowance_{single.slug(allowance)}"
            / f"trajectory_{index:02d}.json")
            for index, start in enumerate(method_starts)]
        endpoints = [row["endpoint"] for row in trajectories]
        if incumbent is not None:
            endpoints.append(incumbent)
        unique = {eta_key(row["eta"]): row for row in endpoints}
        shortlist = sorted(unique.values(), key=lambda row: (row["action"], eta_key(row["eta"])))[:3]
        del search
        _release_stage_data()
        authoritative = EnsembleFullContext(engine, single.official_config(cfg),
            _flow_data(single.official_config(cfg), "authoritative_train", "authoritative_audit"), law_risks)
        certified = []
        for finalist in shortlist:
            key = eta_key(finalist["eta"])
            cache = OUTPUT_ROOT / "authoritative" / "cache" / f"{key}.json"
            if cache.exists():
                evaluation = single.read_json(cache)
            else:
                raw = authoritative.evaluate(finalist["eta"], gradient=False)
                audit = authoritative.audit(raw, require_physical=True)
                evaluation = {**engine._public(raw), "authoritative_audit": audit,
                              "certified": bool(raw["risk"] <= ceiling and audit["valid"])}
                single.atomic_json(cache, evaluation)
            if evaluation["risk"] <= ceiling and evaluation["certified"]:
                certified.append(evaluation)
        incumbent_auth = (None if incumbent is None else
            next((row for row in certified if eta_key(row["eta"]) == eta_key(incumbent["eta"])), incumbent))
        best = min(certified, key=lambda row: (row["action"], eta_key(row["eta"]))) if certified else None
        winner = (incumbent_auth if incumbent_auth is not None
                  and (best is None or best["action"] >= incumbent_auth["action"] - 1e-10)
                  else best)
        point = {"allowance_percent": allowance, "risk_ceiling": ceiling,
                 "normalized_robust_risk_ceiling": ceiling,
                 "maximum_risk_ratio": 1.0 + allowance / 100.0,
                 "risk_ceiling_by_flow": dict(zip(FLOW_IDS, map(float, ceilings))),
                 "trajectories": trajectories, "shortlist": shortlist,
                 "authoritative_finalists": certified, "winner": winner,
                 "status": "CERTIFIED" if winner is not None else "NO_CERTIFIED_FULL_POINT",
                 "incumbent_retained": incumbent_auth is not None
                    and winner is not None
                    and eta_key(winner["eta"]) == eta_key(incumbent_auth["eta"])}
        results.append(point)
        incumbent = winner
        single.atomic_json(OUTPUT_ROOT / "authoritative" / f"allowance_{single.slug(allowance)}"
                           / "result.json", point)
        del authoritative
        _release_stage_data()
    winners = [row["winner"] for row in results if row["winner"] is not None]
    monotone = all(b["action"] <= a["action"] + 1e-10
                   for a, b in zip(winners[:-1], winners[1:]))
    result = {"schema_version": 2, "passed": True, "certified_curve_monotone": monotone,
              "protocol_sha256": protocol["protocol_sha256"], "allowances": results,
              "validation_accessed": False, "deep_ritz_used": False,
              "lazy_authoritative_loading": True,
              "certified_point_count": len(winners),
              "status": "COMPLETE" if len(winners) == len(ALLOWANCES) else "COMPLETE_WITH_GAPS"}
    single.atomic_json(path, result)
    return result


def _candidate_geometries(selection: dict[str, Any], allowance_index: int) -> list[Any]:
    point = selection["allowances"][allowance_index]
    rows = [point.get("winner")]
    rows.extend(point.get("finalists", point.get("authoritative_finalists", [])))
    for trajectory in point.get("trajectories", []):
        rows.extend((trajectory.get("endpoint"), trajectory.get("start")))
    output, seen = [], set()
    for row in rows:
        if not row:
            continue
        eta = row["eta"]
        key = eta_key(eta)
        if key not in seen:
            output.append(eta)
            seen.add(key)
    return output


def finalize_robust_pareto(cfg: dict[str, Any]) -> dict[str, Any]:
    if FINAL_PARETO.exists():
        return single.read_json(FINAL_PARETO)
    screening = single.read_json(SCREENING_PATH)
    tangent = single.read_json(OUTPUT_ROOT / "tangent" / "selection.json")
    full = single.read_json(OUTPUT_ROOT / "full_search" / "selection.json")
    law_risks = screening["law_risk_by_flow"]
    pareto_rows = []
    for index, allowance in enumerate(ALLOWANCES):
        ceilings = {flow_id: (1 + allowance / 100) * law_risks[flow_id]
                    for flow_id in FLOW_IDS}
        tangent_winner = tangent["allowances"][index]["winner"]
        full_winner = full["allowances"][index]["winner"]
        point = {
            "allowance_percent": allowance,
            "risk_ceiling_by_flow": ceilings,
            "Law": {"eta": screening["law_eta"], "risk_by_flow": law_risks,
                    "mean_risk": float(np.mean(list(law_risks.values()))),
                    "tangent_action": None, "full_action": None, "status": "REFERENCE"},
            "Tangent": None,
            "Full": None,
        }
        if tangent_winner is not None:
            point["Tangent"] = {"eta": tangent_winner["eta"],
                                "risk_by_flow": tangent_winner["risk_by_flow"],
                                "mean_risk": float(np.mean(list(tangent_winner["risk_by_flow"].values()))),
                                "tangent_action": tangent_winner["action"], "full_action": None,
                                "per_flow_action": tangent_winner["per_flow_action"],
                                "status": "CERTIFIED"}
        if full_winner is not None:
            point["Full"] = {"eta": full_winner["eta"],
                             "risk_by_flow": full_winner["risk_by_flow"],
                             "mean_risk": float(np.mean(list(full_winner["risk_by_flow"].values()))),
                             "tangent_action": None, "full_action": full_winner["action"],
                             "per_flow_action": full_winner["per_flow_action"],
                             "status": "CERTIFIED"}
        pareto_rows.append(point)
    full_gap_count = sum(point["Full"] is None for point in pareto_rows)
    tangent_gap_count = sum(point["Tangent"] is None for point in pareto_rows)
    gap_count = full_gap_count + tangent_gap_count
    result = {"schema_version": 1,
              "status": "COMPLETE" if gap_count == 0 else "COMPLETE_WITH_GAPS",
              "full_gap_count": full_gap_count,
              "tangent_gap_count": tangent_gap_count, "version": VERSION,
              "protocol_sha256": require_protocol(cfg)["protocol_sha256"],
              "runtime_patch_sha256": file_sha256(RUNTIME_PATCH_RECEIPT_12),
              "flow_ids": list(FLOW_IDS), "risk_gate": "separate for every flow",
              "objective": "equal-weight mean action", "allowances": pareto_rows,
              "validation_accessed": False, "deep_ritz_used": False}
    single.atomic_json(FINAL_PARETO, result)
    csv_path = OUTPUT_ROOT / "pareto.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["allowance_percent", "method", "mean_risk", "tangent_action", "full_action",
                         *[f"risk_{flow_id}" for flow_id in FLOW_IDS]])
        for point in pareto_rows:
            for method in ("Law", "Tangent", "Full"):
                row = point[method]
                writer.writerow([point["allowance_percent"], method,
                    None if row is None else row["mean_risk"],
                    None if row is None else row["tangent_action"],
                    None if row is None else row["full_action"],
                    *([None] * len(FLOW_IDS) if row is None else
                      [row["risk_by_flow"][flow_id] for flow_id in FLOW_IDS])])
    lines = ["# Three-reference B1 Galerkin Pareto", "",
             f"Status: {result['status']}", "",
             "Reported winners pass the scientific-risk and numerical gates separately on all three frozen B1 flows. Actions are equal-weight means of independent per-flow actions. Missing Full entries failed the unchanged numerical gates.", "",
             "Each physical risk ceiling is flow-specific: `R_j(candidate) <= (1 + p/100) R_j(Law)`. A candidate risk below its flow's Law risk is an improvement (negative signed budget use), not a negative allowance. The normalized robust scalar used internally for optimization is not a physical per-flow ceiling.", "",
             "| allowance | method | mean risk | Tangent action | Full K280 action |", "|---:|---|---:|---:|---:|"]
    for point in pareto_rows:
        for method in ("Law", "Tangent", "Full"):
            row = point[method]
            if row is None:
                tangent_cell = "NO CERTIFIED POINT" if method == "Tangent" else "—"
                full_cell = "NO CERTIFIED POINT" if method == "Full" else "—"
                lines.append(
                    f"| {point['allowance_percent']}% | {method} | — | "
                    f"{tangent_cell} | {full_cell} |"
                )
            else:
                tangent_action = "—" if row["tangent_action"] is None else f"{row['tangent_action']:.9g}"
                full_action = "—" if row["full_action"] is None else f"{row['full_action']:.9g}"
                lines.append(f"| {point['allowance_percent']}% | {method} | {row['mean_risk']:.9g} | {tangent_action} | {full_action} |")
    single.atomic_text(OUTPUT_ROOT / "report.md", "\n".join(lines) + "\n")
    return result


def run_stage(cfg: dict[str, Any], stage: str, progress: Callable[[str], None] | None = print) -> dict[str, Any]:
    _activate()
    freeze_runtime_patch()
    routes = {
        "protocol": lambda: freeze_protocol(cfg),
        "data": lambda: generate_banks(cfg, progress),
        "law": lambda: single.reconstruct_law(cfg, progress),
        "candidates": lambda: single.generate_candidates(cfg, progress),
        "screen": lambda: screen_candidates(cfg, progress),
        "tangent": lambda: single.select_tangent(cfg),
        "full": lambda: single.select_full(cfg),
        "finalize": lambda: finalize_robust_pareto(cfg),
    }
    order = ("protocol", "data", "law", "candidates", "screen", "tangent", "full", "finalize")
    result: dict[str, Any] = {}
    for name in order if stage == "all" else (stage,):
        if progress:
            progress(f"starting={name}")
        result = routes[name]()
        if progress:
            progress(f"completed={name}")
    return result
