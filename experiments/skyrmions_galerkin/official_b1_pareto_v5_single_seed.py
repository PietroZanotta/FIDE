"""Prospective single-root-seed Skyrmion B1 K=280 JAX authority (V5).

V5 retains the V4 scientific method while adding four pre-seal exact-risk
guards and increasing all certificate-bearing fit/audit roles to N=131,072.
Native Galerkin is not an option on this execution path.
"""

from __future__ import annotations

import ast
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from .galerkin_only_data import SelectionGalerkinData
from .jax_galerkin_v2 import JaxGalerkinContext, K, tangent_audit
from .production_artifacts import file_sha256
from . import official_b1_pareto_v2_single_seed as base
from . import official_b1_pareto_v3_support_robust as v3


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
VERSION = "official_b1_galerkin_pareto_v5_single_seed"
OUTPUT_ROOT = ROOT / "outputs" / VERSION
CONFIG_PATH = ROOT / "config_v5_single_seed.json"
PROTOCOL_DOCUMENT = ROOT / "OFFICIAL_B1_GALERKIN_PARETO_V5_PROTOCOL.md"
RUNNER_PATH = ROOT / "official_b1_pareto_v5_single_seed_run.py"
TEST_PATH = ROOT / "test_official_b1_pareto_v5_single_seed.py"
SOURCE_PATH = Path(__file__)

PROTOCOL_PATH = OUTPUT_ROOT / "protocol_v5.json"
FREEZE_MANIFEST_PATH = OUTPUT_ROOT / "freeze_manifest_v5.json"
RANDOMNESS_PATH = OUTPUT_ROOT / "V5_RANDOMNESS_PROVENANCE.md"
CALL_GRAPH_PATH = OUTPUT_ROOT / "jax_only_call_graph_v5.json"
EFFECTIVE_CONFIG_PATH = OUTPUT_ROOT / "effective_config.json"
CANDIDATE_POOL_PATH = OUTPUT_ROOT / "candidate_pool" / "candidate_pool.json"
SCIENTIFIC_ARRAYS_PATH = OUTPUT_ROOT / "feasibility" / "exact_receipts.npz"
SCIENTIFIC_ROWS_PATH = OUTPUT_ROOT / "feasibility" / "exact_receipts.json"
LAW_PATH = OUTPUT_ROOT / "law" / "initial_law.json"
LAW_GUARD_SUMMARY_PATH = OUTPUT_ROOT / "law" / "guard_screen.json"
SELECTION_SEAL_PATH = OUTPUT_ROOT / "selection" / "selection_seal.json"
SELECTION_VERIFICATION_PATH = OUTPUT_ROOT / "selection" / "independent_verification.json"
FINAL_REPORT_PATH = OUTPUT_ROOT / "OFFICIAL_B1_GALERKIN_PARETO_V5_FINAL_RESULT.md"
FINAL_SUMMARY_PATH = OUTPUT_ROOT / "final_summary.json"
FINAL_CSV_PATH = OUTPUT_ROOT / "final_rows.csv"
INVENTORY_PATH = OUTPUT_ROOT / "terminal_inventory.json"

V1_ROOT = ROOT / "outputs" / "official_b1_galerkin_pareto_v1"
V2_ROOT = ROOT / "outputs" / "old_stuff" / "official_b1_galerkin_pareto_v2_single_seed"
V2_1_ROOT = ROOT / "outputs" / "official_b1_galerkin_pareto_v2_1_single_seed_amended"
V3_ROOT = ROOT / "outputs" / "official_b1_galerkin_pareto_v3_support_robust_single_seed"
V4_ROOT = ROOT / "outputs" / "official_b1_galerkin_pareto_v4_single_seed"
DEVELOPMENT_ROOT = ROOT / "outputs" / "development_v5_terminal_diagnosis"
HISTORY_ROOTS = {
    "V1_authority": V1_ROOT,
    "V2_authority": V2_ROOT,
    "V2_1_authority": V2_1_ROOT,
    "V3_4_authority": V3_ROOT,
    "V4_authority": V4_ROOT,
    "V4_terminal_diagnosis": DEVELOPMENT_ROOT,
}
V3_TERMINAL_PATH = V3_ROOT / "OFFICIAL_B1_GALERKIN_PARETO_V3_4_TERMINAL_RESULT.md"
DEVELOPMENT_REPORT_PATH = DEVELOPMENT_ROOT / "SKYRMION_V4_TERMINAL_FAILURE_DIAGNOSIS.md"

REFERENCE_SOURCE = V1_ROOT / "artifacts" / "reference.npz"
DICTIONARY_SOURCE = (
    ROOT / "outputs" / "galerkin_only_3pct" / "cache" / "dictionaries"
    / "dictionary_K280.npz"
)
REFERENCE_PATH = OUTPUT_ROOT / "artifacts" / "reference.npz"
DICTIONARY_PATH = OUTPUT_ROOT / "artifacts" / "dictionary_K280.npz"

ROOT_SEED = 20261005
GUARD_BLOCK_SIZE = 8
FEATURE_SAMPLE_CHUNK = 8192
RISK_CANDIDATE_BATCH_SIZE = 32
GUARD_COUNTS = {
    "law_guard_screen": 8192,
    "law_guard_search_train": 32768,
    "law_guard_periodic_audit": 16384,
    "law_guard_authoritative_train": 65536,
}
STANDARD_ROLE_IDS = {role: index for index, role in enumerate(base.ROLE_NAMES)}
GUARD_ROLE_IDS = {
    "law_guard_screen": 1010,
    "law_guard_search_train": 1011,
    "law_guard_periodic_audit": 1012,
    "law_guard_authoritative_train": 1013,
}
RISK_GUARD_COUNTS = {
    "risk_guard_1": {"truth": 5000, "reference": 65536},
    "risk_guard_2": {"truth": 5000, "reference": 65536},
    "risk_guard_3": {"truth": 5000, "reference": 65536},
    "risk_guard_4": {"truth": 5000, "reference": 65536},
}
RISK_GUARD_ROLE_IDS = {
    "risk_guard_1_truth": 2001, "risk_guard_1_reference": 2002,
    "risk_guard_1_observation_noise": 2003,
    "risk_guard_2_truth": 2011, "risk_guard_2_reference": 2012,
    "risk_guard_2_observation_noise": 2013,
    "risk_guard_3_truth": 2021, "risk_guard_3_reference": 2022,
    "risk_guard_3_observation_noise": 2023,
    "risk_guard_4_truth": 2031, "risk_guard_4_reference": 2032,
    "risk_guard_4_observation_noise": 2033,
}
ROLE_IDS = {**STANDARD_ROLE_IDS, **GUARD_ROLE_IDS, **RISK_GUARD_ROLE_IDS}
RISK_ROLE_NAMES = ("selection", *RISK_GUARD_COUNTS)
RISK_MATRIX_PATH = OUTPUT_ROOT / "feasibility" / "five_role_risk_matrix.json"
RISK_MATRIX_ARRAYS_PATH = OUTPUT_ROOT / "feasibility" / "five_role_risk_matrix.npz"
RISK_MATRIX_CSV_PATH = OUTPUT_ROOT / "complete_candidate_risk_matrix.csv"
ROBUST_FEASIBLE_PATH = OUTPUT_ROOT / "robust_feasible_sets.json"
BASE_SELECTION_SEAL_PATH = OUTPUT_ROOT / "selection" / "base_selection_seal.json"
BASE_SELECTION_VERIFICATION_PATH = OUTPUT_ROOT / "selection" / "base_independent_verification.json"
PERFORMANCE_REPORT_PATH = OUTPUT_ROOT / "V5_PERFORMANCE_REPORT.md"

ORIGINAL_BASE_GENERATE_DATA = base.generate_data
ORIGINAL_MANY_BODY_FEATURES = base.many_body_features
ORIGINAL_BASE_SELECTION_RUNTIME = base.SelectionRuntime
ORIGINAL_BASE_CERTIFY_SELECTION = base.certify_and_freeze_selection


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return file_sha256(path)


def _tree_receipt(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    total = 0
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        value = bytes.fromhex(_sha256(path))
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(value)
        total += path.stat().st_size
    return {
        "path": str(root.relative_to(REPO_ROOT)),
        "tree_sha256": digest.hexdigest(),
        "file_count": len(files),
        "bytes": total,
    }


def _atomic_bytes(path: Path, data: bytes, *, immutable: bool = True) -> None:
    resolved = path.resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"V5 output escaped {root}: {resolved}")
    if path.exists():
        if path.read_bytes() == data:
            return
        if immutable:
            raise RuntimeError(f"refusing to overwrite sealed V5 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, payload: Any, *, immutable: bool = True) -> None:
    _atomic_bytes(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n",
        immutable=immutable,
    )


def _atomic_text(path: Path, value: str, *, immutable: bool = True) -> None:
    _atomic_bytes(path, value.encode(), immutable=immutable)


def _link_verified(source: Path, destination: Path) -> None:
    if destination.exists():
        if _sha256(source) != _sha256(destination):
            raise RuntimeError(f"deterministic input mismatch: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        os.close(descriptor)
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def _values() -> dict[str, Any]:
    return _read_json(CONFIG_PATH)


def _role_record(role: str) -> dict[str, Any]:
    role_id = ROLE_IDS[role]
    key = base._role_key(ROOT_SEED, role_id)
    words = np.asarray(jax.random.key_data(key), dtype=np.uint32).tolist()
    seed_key = jax.random.fold_in(key, 0x5EED)
    seed = int(jax.random.bits(seed_key, (), dtype=jnp.uint32)) % (2**31 - 1)
    return {
        "role": role,
        "role_id": role_id,
        "jax_key_words_uint32": words,
        "integer_seed_adapter": seed,
        "derivation": f"fold_in(PRNGKey({ROOT_SEED}), {role_id})",
    }


def randomness_records() -> list[dict[str, Any]]:
    return [_role_record(role) for role in ROLE_IDS]


def role_seed(role: str) -> int:
    if role not in ROLE_IDS:
        raise KeyError(f"unfrozen V5 stochastic role: {role}")
    return int(_role_record(role)["integer_seed_adapter"])


def chunked_many_body_features(
    configurations: Any, box: tuple[float, float] = (2.0, 1.0),
) -> jax.Array:
    rows = jnp.asarray(configurations, dtype=jnp.float64)
    samples = int(rows.shape[-3])
    if samples <= FEATURE_SAMPLE_CHUNK:
        return ORIGINAL_MANY_BODY_FEATURES(rows, box)
    pieces = []
    for start in range(0, samples, FEATURE_SAMPLE_CHUNK):
        piece = ORIGINAL_MANY_BODY_FEATURES(
            rows[..., start:start + FEATURE_SAMPLE_CHUNK, :, :], box
        )
        piece.block_until_ready()
        pieces.append(piece)
    return jnp.concatenate(pieces, axis=-2)


def _patch_paths() -> None:
    base.VERSION = VERSION
    base.OUTPUT_ROOT = OUTPUT_ROOT
    mapping = {
        "PROTOCOL_PATH": PROTOCOL_PATH,
        "FREEZE_MANIFEST_PATH": FREEZE_MANIFEST_PATH,
        "RANDOMNESS_PATH": RANDOMNESS_PATH,
        "CALL_GRAPH_PATH": CALL_GRAPH_PATH,
        "EFFECTIVE_CONFIG_PATH": EFFECTIVE_CONFIG_PATH,
        "CANDIDATE_POOL_PATH": CANDIDATE_POOL_PATH,
        "SCIENTIFIC_ARRAYS_PATH": SCIENTIFIC_ARRAYS_PATH,
        "SCIENTIFIC_ROWS_PATH": SCIENTIFIC_ROWS_PATH,
        "LAW_PATH": LAW_PATH,
        "SELECTION_SEAL_PATH": SELECTION_SEAL_PATH,
        "SELECTION_VERIFICATION_PATH": SELECTION_VERIFICATION_PATH,
        "FINAL_SUMMARY_PATH": FINAL_SUMMARY_PATH,
        "FINAL_CSV_PATH": FINAL_CSV_PATH,
        "PERFORMANCE_REPORT_PATH": OUTPUT_ROOT / "V5_PERFORMANCE_REPORT.md",
        "RESULT_REPORT_PATH": FINAL_REPORT_PATH,
        "PREFLIGHT_PATH": OUTPUT_ROOT / "provenance" / "preflight.json",
    }
    for name, value in mapping.items():
        setattr(base, name, value)
    base.PROTOCOL_DOCUMENT = PROTOCOL_DOCUMENT
    base.V2_CONFIG_PATH = CONFIG_PATH
    base.REFERENCE_PATH = REFERENCE_PATH
    base.DICTIONARY_PATH = DICTIONARY_PATH

    v3.OUTPUT_ROOT = OUTPUT_ROOT
    v3.ROOT_SEED = ROOT_SEED
    v3.GUARD_COUNTS = GUARD_COUNTS
    v3.GUARD_BLOCK_SIZE = GUARD_BLOCK_SIZE
    v3.LAW_GUARD_SUMMARY_PATH = LAW_GUARD_SUMMARY_PATH
    v3.V3_PROTOCOL_PATH = PROTOCOL_PATH
    v3.V3_CALL_GRAPH_PATH = CALL_GRAPH_PATH


def activate() -> None:
    _patch_paths()
    base._v2_values = _values
    base.role_seed = role_seed
    base.require_protocol = require_v5
    base.generate_data = generate_data
    base.many_body_features = chunked_many_body_features
    base._select_starts = v3._amended_select_starts
    base.SelectionRuntime = V5SelectionRuntime
    base.run_selection_pass = run_selection_pass_v5
    base.run_selection_with_restarts = run_selection_with_restarts_v5
    base.verify_frozen_selection = verify_selection_v5
    base.certify_and_freeze_selection = certify_selection
    v3.require_v3 = require_v5
    v3.generate_data = generate_data
    v3.role_seed = role_seed
    v3.guard_qualify_rows = isolated_guard_qualify_rows


def _source_hashes() -> dict[str, str]:
    paths = (
        SOURCE_PATH, RUNNER_PATH, TEST_PATH, CONFIG_PATH, PROTOCOL_DOCUMENT,
        ROOT / "official_b1_pareto_v2_single_seed.py",
        ROOT / "official_b1_pareto_v3_support_robust.py",
        ROOT / "jax_galerkin_v2.py",
        ROOT / "production_galerkin.py",
        ROOT / "galerkin.py",
    )
    return {path.name: _sha256(path) for path in paths}


def _static_call_graph() -> dict[str, Any]:
    sources = (
        SOURCE_PATH, RUNNER_PATH, ROOT / "jax_galerkin_v2.py",
        ROOT / "production_galerkin.py", ROOT / "galerkin.py",
    )
    forbidden_imports = {"mfsi.galerkin_tesseract", ".pareto_v2_selection"}
    forbidden_calls = {
        "assemble_galerkin_chunk_tesseract", "evaluate_galerkin_action",
        "solve_native_galerkin", "galerkin_tesseract",
    }
    violations = []
    for source in sources:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden_imports:
                        violations.append({"source": source.name, "import": alias.name})
            elif isinstance(node, ast.ImportFrom):
                module = "." * node.level + (node.module or "")
                if module in forbidden_imports:
                    violations.append({"source": source.name, "import": module})
            elif isinstance(node, ast.Call):
                function = node.func
                called = (
                    function.id if isinstance(function, ast.Name)
                    else function.attr if isinstance(function, ast.Attribute)
                    else ""
                )
                if called in forbidden_calls:
                    violations.append({"source": source.name, "call": called})
    return {
        "schema_version": 1,
        "entrypoint": f"{RUNNER_PATH.name}:main",
        "reachable_scientific_sources": {
            path.name: _sha256(path) for path in sources
        },
        "scientific_edges": [
            ["runner.main", "study.run_selection"],
            ["study.run_selection", "study.run_selection_with_restarts_v5"],
            ["study.run_selection_with_restarts_v5", "study.run_selection_pass_v5"],
            ["study.run_selection_pass_v5", "study.V5SelectionRuntime"],
            ["base.SelectionRuntime", "jax_galerkin_v2.JaxGalerkinContext"],
            ["JaxGalerkinContext.evaluate", "galerkin.rank_aware_quadratic_solve"],
            ["JaxGalerkinContext.audit", "production_galerkin.audit_hybrid_solutions"],
            ["study.validate_heldout", "jax_galerkin_v2.JaxGalerkinContext"],
        ],
        "native_fallback": False,
        "native_galerkin_reachable": bool(violations),
        "violations": violations,
        "passed": not violations,
    }


def _seed_search_evidence() -> dict[str, Any]:
    exclusions = (
        SOURCE_PATH.name, RUNNER_PATH.name, TEST_PATH.name,
        CONFIG_PATH.name, PROTOCOL_DOCUMENT.name,
    )
    matches = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.name in exclusions or "outputs" in path.parts:
            continue
        try:
            if str(ROOT_SEED) in path.read_text(encoding="utf-8"):
                matches.append(str(path.relative_to(REPO_ROOT)))
        except (UnicodeDecodeError, OSError):
            continue
    git = subprocess.run(
        [
            "git", "log", "--all", "-S", str(ROOT_SEED),
            "--format=%H %ad %s", "--date=iso", "--",
            "experiments/skyrmions_deep_ritz_full",
            "experiments/skyrmions_galerkin",
        ],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    return {
        "declared_range": [20261001, 20261099],
        "prior_official_roots": [20261001, 20261002, 20261003, 20261004],
        "selection_rule": "first unused integer after the largest prior official root",
        "selected_root": ROOT_SEED,
        "preexisting_repository_matches_excluding_V5_declarations_and_outputs": matches,
        "relevant_git_history_matches": git,
        "passed": not matches and not git,
    }


def freeze_v5(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    if PROTOCOL_PATH.exists():
        activate()
        return require_v5()
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.rglob("*")):
        raise RuntimeError("V5 output root is not empty before freeze")
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("V5 freeze requires JAX x64")
    if K != 280:
        raise RuntimeError("V5 K changed")
    values = _values()
    if values["root_seed"] != ROOT_SEED or values["allowances_percent"] != [0.5, 1.0, 2.0]:
        raise RuntimeError("V5 seed or allowances changed")
    if values["validation"]["reference_fit_samples"] != 131072 or values["validation"]["reference_audit_samples"] != 131072:
        raise RuntimeError("V5 held-out counts changed")
    if values["bank_sizes"]["authoritative_train"] != 131072 or values["bank_sizes"]["authoritative_audit"] != 131072:
        raise RuntimeError("V5 authoritative counts changed")
    if not all(root.is_dir() for root in HISTORY_ROOTS.values()):
        raise RuntimeError("historical authority root missing")
    history = {name: _tree_receipt(root) for name, root in HISTORY_ROOTS.items()}
    history["V3_4_terminal_result"] = {
        "path": str(V3_TERMINAL_PATH.relative_to(REPO_ROOT)),
        "sha256": _sha256(V3_TERMINAL_PATH),
    }
    history["V4_terminal_development_report"] = {
        "path": str(DEVELOPMENT_REPORT_PATH.relative_to(REPO_ROOT)),
        "sha256": _sha256(DEVELOPMENT_REPORT_PATH),
    }
    seed_evidence = _seed_search_evidence()
    if not seed_evidence["passed"]:
        raise RuntimeError(f"V5 root seed was already used: {seed_evidence}")
    graph = _static_call_graph()
    if not graph["passed"]:
        raise RuntimeError(f"native Galerkin reachable: {graph['violations']}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    _link_verified(REFERENCE_SOURCE, REFERENCE_PATH)
    _link_verified(DICTIONARY_SOURCE, DICTIONARY_PATH)
    activate()
    rows = base._candidate_universe_rows()
    reference_hash = _sha256(REFERENCE_PATH)
    dictionary_hash = _sha256(DICTIONARY_PATH)
    cfg = base.effective_config()
    thresholds = cfg["production_galerkin"]["certificate_thresholds"]
    if thresholds != {
        "maximum_weak_residual": 0.12,
        "maximum_energy_residual": 0.08,
        "maximum_gauge_residual": 1e-09,
        "maximum_moment_rate_residual": 0.1,
    }:
        raise RuntimeError(f"certificate thresholds changed: {thresholds}")
    protocol_body = {
        "schema_version": 4,
        "version": VERSION,
        "status": "FROZEN_BEFORE_V5_SCIENTIFIC_OUTCOMES",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "single_seed": True,
        "single_root_seed": True,
        "root_seed": ROOT_SEED,
        "alternate_root_seeds_tested": [],
        "seed_range_rule": seed_evidence,
        "randomness": randomness_records(),
        "reference_policy": {
            "policy": "reuse accepted deterministic V1 reference byte-for-byte",
            "retrained": False,
            "path": str(REFERENCE_PATH.relative_to(OUTPUT_ROOT)),
            "sha256": reference_hash,
        },
        "estimand": "fixed K=280 permutation-invariant configuration-space Galerkin correction action",
        "continuum_convergence_claim": False,
        "solver": {
            "galerkin_backend": "jax",
            "projection_backend": "jax",
            "K": 280,
            "dictionary_sha256": dictionary_hash,
            "relative_rank_tolerance": 1e-12,
            "rank_rule": "positive eigenvalues > 1e-12 * lambda_max",
            "dtype": "float64",
            "native_fallback": False,
            "native_solver": "prohibited/unreachable",
            "chunk_size": int(values["galerkin_chunk_size"]),
            "feature_sample_chunk": FEATURE_SAMPLE_CHUNK,
        },
        "candidate_universe": {
            "count": len(rows),
            "component_counts": values["candidate_components"],
            "rows_sha256": _payload_sha256(rows),
            "frozen_before_outcomes": True,
            "historical_geometries_only_through_declared_candidate_family": True,
        },
        "allowances_percent": values["allowances_percent"],
        "risk_rule": "R_star + (p/100)*abs(R_star)",
        "robust_risk_rule": {
            "roles": list(RISK_ROLE_NAMES),
            "preseal_guard_count": 4,
            "guard_samples": RISK_GUARD_COUNTS,
            "relative_formula": "R_b(eta) / R_b(Law) - 1",
            "pass_condition": "every role relative risk <= allowance/100",
            "aggregation": "maximum/no averaging/no majority vote/no slack",
            "law_role_relative_risk": 0.0,
            "risk_only_candidate_batch_size": RISK_CANDIDATE_BATCH_SIZE,
        },
        "law_consistency_tolerance": values["law_consistency_tolerance"],
        "maximum_anchor_refinement_restarts": values["maximum_anchor_refinement_restarts"],
        "replacement_tolerance": values["replacement_tolerance"],
        "feasibility_first": True,
        "minimum_rESS": 0.05,
        "bank_sizes": values["bank_sizes"],
        "law_support_rule": {
            "unchanged_from": "V3.4 four-guard support-robust rule",
            "guard_counts": GUARD_COUNTS,
            "candidate_order": "jointly-supported exact risk then eta_sha256",
            "stopping_block_size": GUARD_BLOCK_SIZE,
            "every_guard_must_pass": True,
            "authoritative_action_audit_reserved": True,
        },
        "design_truth_samples": values["design_truth_samples"],
        "starts": values["starts"],
        "optimizer": values["optimizer"],
        "candidate_batch_size": values["candidate_batch_size"],
        "validation": values["validation"],
        "authoritative_certificate_samples": {"fit": 131072, "audit": 131072},
        "certificate_thresholds": thresholds,
        "energy_formula": {
            "numerator": "abs(a_fit^T K_audit a_fit + a_fit^T f_audit)",
            "denominator": "max(a_fit^T K_audit a_fit + abs(a_fit^T f_audit), 1e-12)",
            "statistic": "max over 13 time nodes",
            "threshold": 0.08,
        },
        "memory_and_performance": {
            "guard_roles_in_fresh_processes": True,
            "feature_sample_chunk": FEATURE_SAMPLE_CHUNK,
            "direct_galerkin_chunk": values["galerkin_chunk_size"],
            "unique_geometry_deduplication": True,
            "full_bank_basis_gradient_tensor_materialized": False,
        },
        "cache_requirements": [
            "geometry_hash", "role_bank_hash", "N", "K", "dictionary_hash",
            "reference_hash", "forcing_config_hash", "scientific_config_hash",
        ],
        "output_paths": {
            "root": str(OUTPUT_ROOT.relative_to(REPO_ROOT)),
            "risk_matrix_csv": str(RISK_MATRIX_CSV_PATH.relative_to(OUTPUT_ROOT)),
            "robust_feasible_sets": str(ROBUST_FEASIBLE_PATH.relative_to(OUTPUT_ROOT)),
            "selection_seal": str(SELECTION_SEAL_PATH.relative_to(OUTPUT_ROOT)),
            "selection_verification": str(SELECTION_VERIFICATION_PATH.relative_to(OUTPUT_ROOT)),
            "heldout_results": "heldout_validation/results.json",
            "performance_report": str(PERFORMANCE_REPORT_PATH.relative_to(OUTPUT_ROOT)),
            "final_report": str(FINAL_REPORT_PATH.relative_to(OUTPUT_ROOT)),
            "final_summary": str(FINAL_SUMMARY_PATH.relative_to(OUTPUT_ROOT)),
            "final_rows": str(FINAL_CSV_PATH.relative_to(OUTPUT_ROOT)),
            "terminal_inventory": str(INVENTORY_PATH.relative_to(OUTPUT_ROOT)),
        },
        "history_before_v5": history,
        "source_hashes": _source_hashes(),
        "protocol_document_sha256": _sha256(PROTOCOL_DOCUMENT),
        "final_heldout_access": "only after selection seal and passing independent reconstruction",
        "post_heldout_tuning": False,
    }
    digest = _payload_sha256(protocol_body)
    protocol = {**protocol_body, "protocol_sha256": digest}
    freeze_manifest = {
        "schema_version": 4,
        "status": "FROZEN_BEFORE_V5_SCIENTIFIC_OUTCOMES",
        "frozen_at_utc": protocol_body["frozen_at_utc"],
        "protocol_sha256": digest,
        "freeze_is_pre_outcome": True,
        "scientific_outcomes_present_before_freeze": False,
        "root_seed": ROOT_SEED,
        "candidate_pool_rows_sha256": _payload_sha256(rows),
        "reference_sha256": reference_hash,
        "dictionary_sha256": dictionary_hash,
        "history_before_v5": history,
        "source_hashes": protocol_body["source_hashes"],
        "jax_only_call_graph_sha256": _payload_sha256(graph),
    }
    lines = [
        "# V5 Randomness Provenance", "",
        f"Exactly one root experiment seed is frozen: `{ROOT_SEED}`.",
        "Derived role keys are deterministic descendants, not scientific replicates.",
        "The root was selected as the first unused integer after the prior authority root.",
        "", "| role id | role | JAX key words | integer adapter |", "|---:|---|---|---:|",
    ]
    for row in protocol_body["randomness"]:
        lines.append(
            f"| {row['role_id']} | `{row['role']}` | `{row['jax_key_words_uint32']}` | {row['integer_seed_adapter']} |"
        )
    lines += ["", "No alternate root may be tested. All role banks must be disjoint by initial-state hash.", ""]
    _atomic_json(PROTOCOL_PATH, protocol)
    _atomic_json(FREEZE_MANIFEST_PATH, freeze_manifest)
    _atomic_json(CALL_GRAPH_PATH, graph)
    _atomic_json(EFFECTIVE_CONFIG_PATH, cfg)
    _atomic_json(CANDIDATE_POOL_PATH, {
        "schema_version": 4,
        "protocol_sha256": digest,
        "frozen_before_outcomes": True,
        "count": len(rows),
        "component_counts": values["candidate_components"],
        "rows_sha256": _payload_sha256(rows),
        "rows": rows,
    })
    _atomic_text(RANDOMNESS_PATH, "\n".join(lines))
    if progress:
        progress(f"V5 protocol frozen before outcomes: {digest}")
    return protocol


def require_v5() -> dict[str, Any]:
    required = (
        PROTOCOL_PATH, FREEZE_MANIFEST_PATH, RANDOMNESS_PATH, CALL_GRAPH_PATH,
        EFFECTIVE_CONFIG_PATH, CANDIDATE_POOL_PATH, REFERENCE_PATH, DICTIONARY_PATH,
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError("V5 protocol is not fully frozen")
    protocol = _read_json(PROTOCOL_PATH)
    body = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if _payload_sha256(body) != protocol["protocol_sha256"]:
        raise RuntimeError("V5 protocol hash mismatch")
    if _read_json(FREEZE_MANIFEST_PATH)["protocol_sha256"] != protocol["protocol_sha256"]:
        raise RuntimeError("V5 freeze manifest mismatch")
    if _source_hashes() != protocol["source_hashes"]:
        raise RuntimeError("V5 scientific source changed after freeze")
    graph = _read_json(CALL_GRAPH_PATH)
    if not graph["passed"] or _payload_sha256(graph) != _read_json(FREEZE_MANIFEST_PATH)["jax_only_call_graph_sha256"]:
        raise RuntimeError("V5 JAX-only call graph mismatch")
    if _sha256(REFERENCE_PATH) != protocol["reference_policy"]["sha256"]:
        raise RuntimeError("V5 reference changed")
    if _sha256(DICTIONARY_PATH) != protocol["solver"]["dictionary_sha256"]:
        raise RuntimeError("V5 dictionary changed")
    pool = _read_json(CANDIDATE_POOL_PATH)
    if _payload_sha256(pool["rows"]) != protocol["candidate_universe"]["rows_sha256"]:
        raise RuntimeError("V5 candidate universe changed")
    # V3 helpers expect this alias; it is derived, not part of the sealed body.
    return {**protocol, "v3_protocol_sha256": protocol["protocol_sha256"]}


def _guard_path(label: str) -> Path:
    return OUTPUT_ROOT / "banks" / "guard" / f"{label}_N{GUARD_COUNTS[label]}.npz"


def _load_guard(label: str) -> base.GalerkinReferenceBank:
    with np.load(_guard_path(label), allow_pickle=False) as arrays:
        return base.GalerkinReferenceBank(
            jnp.asarray(arrays["configurations"], dtype=jnp.float64),
            jnp.asarray(arrays["velocity"], dtype=jnp.float64),
            jnp.asarray(arrays["base_weights"], dtype=jnp.float64),
        )


def _risk_guard_root(label: str) -> Path:
    return OUTPUT_ROOT / "banks" / "risk_guard" / label


def _risk_guard_reference_path(label: str) -> Path:
    count = RISK_GUARD_COUNTS[label]["reference"]
    return _risk_guard_root(label) / f"reference_N{count}.npz"


def _load_npz_bank(path: Path) -> base.GalerkinReferenceBank:
    with np.load(path, allow_pickle=False) as arrays:
        return base.GalerkinReferenceBank(
            jnp.asarray(arrays["configurations"], dtype=jnp.float64),
            jnp.asarray(arrays["velocity"], dtype=jnp.float64),
            jnp.asarray(arrays["base_weights"], dtype=jnp.float64),
        )


def _fresh_risk_guard(
    cfg: dict[str, Any], flow: Any, truth_model: Any, times: Any, label: str,
) -> dict[str, Any]:
    root = _risk_guard_root(label)
    record_path = root / "manifest.json"
    counts = RISK_GUARD_COUNTS[label]
    if record_path.exists():
        record = _read_json(record_path)
        for artifact in record["artifacts"]:
            if _sha256(OUTPUT_ROOT / artifact["path"]) != artifact["sha256"]:
                raise RuntimeError(f"V5 risk-guard artifact changed: {artifact['path']}")
        return record
    started = time.perf_counter()
    truth_seed = role_seed(f"{label}_truth")
    truth = truth_model.make_bank(
        seed=truth_seed, samples=counts["truth"], times=times,
        substeps_per_interval=int(cfg["physics"]["truth_substeps"]),
    )
    truth_path = root / "truth.npz"
    base.atomic_npz(
        truth_path, times=times, configurations=truth.configurations,
        derived_role_seed=np.asarray(truth_seed),
    )
    reference_seed = role_seed(f"{label}_reference")
    x, velocity, weights, reference_initial_hash = base._rollout_bank(
        cfg, flow, truth_model, times, seed=reference_seed,
        samples=counts["reference"],
    )
    reference_path = _risk_guard_reference_path(label)
    base.atomic_npz(
        reference_path, configurations=x, velocity=velocity, base_weights=weights,
        derived_role_seed=np.asarray(reference_seed),
    )
    family = base._family(cfg)
    noise_seed = role_seed(f"{label}_observation_noise")
    noise = float(cfg["measurement"]["observation_noise_std"]) * jax.random.normal(
        jax.random.PRNGKey(noise_seed),
        (int(cfg["measurement"]["acquisition_count"]), family.n_sensors),
        dtype=jnp.float64,
    )
    noise_path = root / "observation_noise.npz"
    base.atomic_npz(
        noise_path, detector_noise=noise, derived_role_seed=np.asarray(noise_seed)
    )
    truth_initial_hash = hashlib.sha256(
        np.ascontiguousarray(np.asarray(truth.configurations[0])).tobytes()
    ).hexdigest()
    artifacts = []
    for path in (truth_path, reference_path, noise_path):
        artifacts.append({
            "path": str(path.relative_to(OUTPUT_ROOT)), "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    record = {
        "schema_version": 5, "label": label,
        "truth_samples": counts["truth"], "reference_samples": counts["reference"],
        "truth_seed": truth_seed, "reference_seed": reference_seed,
        "observation_noise_seed": noise_seed,
        "truth_initial_state_sha256": truth_initial_hash,
        "reference_initial_state_sha256": reference_initial_hash,
        "artifacts": artifacts, "wall_time_seconds": time.perf_counter() - started,
    }
    _atomic_json(record_path, record)
    del truth, x, velocity, weights, noise
    gc.collect()
    return record


def _risk_guard_data(label: str) -> SelectionGalerkinData:
    cfg = base.effective_config(); root = _risk_guard_root(label)
    with np.load(root / "truth.npz", allow_pickle=False) as arrays:
        times = jnp.asarray(arrays["times"], dtype=jnp.float64)
        truth = jnp.asarray(arrays["configurations"], dtype=jnp.float64)
    with np.load(root / "observation_noise.npz", allow_pickle=False) as arrays:
        noise = jnp.asarray(arrays["detector_noise"], dtype=jnp.float64)
    bank = _load_npz_bank(_risk_guard_reference_path(label))
    problem = base._problem(
        cfg, truth, times, noise_seed=role_seed(f"{label}_observation_noise")
    )
    problem = problem.__class__(
        problem.truth_configurations, problem.times, problem.time_weights,
        problem.acquisition_indices, problem.finite_configuration_count, noise,
        problem.family, problem.reconstructor, problem.projection_config,
        problem.forcing_config, "jax", problem.box,
    )
    with np.load(OUTPUT_ROOT / "design_truth" / "design_truth.npz", allow_pickle=False) as arrays:
        whitening = jnp.asarray(arrays["whitening"], dtype=jnp.float64)
    return SelectionGalerkinData(
        selection_problem=problem, projection_bank=bank, train_bank=bank,
        audit_bank=bank,
        reference_features=chunked_many_body_features(bank.configurations, base.BOX),
        truth_means=jnp.mean(chunked_many_body_features(truth, base.BOX), axis=1),
        whitening=whitening,
    )


class RiskOnlyEvaluator:
    """Exact paired scientific risk/projection diagnostics without forcing work."""

    def __init__(self, data: SelectionGalerkinData, *, batch_size: int):
        self.data = data; self.batch_size = int(batch_size)
        problem = data.selection_problem
        self.projector = base.EmpiricalIProjector(
            problem.projection_config, trajectory_backend="jax"
        )

        def preprocess_one(eta: Any, configurations: Any):
            reconstruction = base.reconstruct_moments(eta, problem)
            return reconstruction.values, problem.family.features(configurations, eta)

        self.preprocess = jax.jit(jax.vmap(preprocess_one, in_axes=(0, None)))

        @jax.jit
        def finish(weights: Any, residual: Any, ess: Any, reference_features: Any):
            predicted = jnp.einsum("btn,tnf->btf", weights, reference_features)
            error = predicted - data.truth_means[None]
            by_time = jnp.einsum("bti,ij,btj->bt", error, data.whitening, error)
            risk = jnp.sum(problem.time_weights[None] * by_time, axis=1)
            return (
                risk, jnp.min(ess, axis=1),
                jnp.max(jnp.linalg.norm(residual, axis=-1), axis=1),
            )

        self.finish = finish

    def evaluate(self, etas: Any) -> dict[str, np.ndarray]:
        etas = np.asarray(etas, dtype=np.float64).reshape(-1, 8)
        bank = self.data.projection_bank
        normalized = bank.base_weights / jnp.sum(bank.base_weights, axis=1, keepdims=True)
        reference_features = self.data.reference_features
        risks=[]; ress=[]; residuals=[]
        for start in range(0, len(etas), self.batch_size):
            batch=etas[start:start+self.batch_size]; actual=len(batch)
            if actual < self.batch_size:
                batch=np.concatenate((batch,np.repeat(batch[-1:],self.batch_size-actual,axis=0)))
            targets, features = self.preprocess(jnp.asarray(batch), bank.configurations)
            projected = self.projector.project_candidate_trajectories(
                features, normalized, targets
            )
            values = self.finish(
                projected.weights, projected.residual, projected.ess_fraction,
                reference_features,
            )
            numpy=[np.asarray(value)[:actual] for value in values]
            risks.append(numpy[0]); ress.append(numpy[1]); residuals.append(numpy[2])
        risk=np.concatenate(risks); minimum_ress=np.concatenate(ress)
        maximum_projection_residual=np.concatenate(residuals)
        valid=(
            np.isfinite(risk) & np.isfinite(minimum_ress)
            & np.isfinite(maximum_projection_residual)
            & (minimum_ress >= base.MINIMUM_RESS)
            & (maximum_projection_residual
               <= self.data.selection_problem.forcing_config.projection_tolerance)
        )
        return {
            "scientific_risk": risk, "minimum_ress": minimum_ress,
            "maximum_projection_residual": maximum_projection_residual,
            "valid": valid,
        }


def _evaluate_risk_checkpointed(
    evaluator: RiskOnlyEvaluator, etas: np.ndarray, label: str,
    progress: Callable[[str], None] | None = None,
) -> dict[str, np.ndarray]:
    directory=OUTPUT_ROOT/"feasibility"/"risk_batches"/label; parts={}
    checkpoint=256
    for start in range(0,len(etas),checkpoint):
        stop=min(start+checkpoint,len(etas)); path=directory/f"batch_{start:05d}_{stop:05d}.npz"
        if path.exists():
            with np.load(path,allow_pickle=False) as arrays:
                result={name:np.asarray(arrays[name]) for name in arrays.files}
        else:
            result=evaluator.evaluate(etas[start:stop]); base.atomic_npz(path,compressed=True,**result)
        for name,value in result.items():parts.setdefault(name,[]).append(value)
        if progress:progress(f"V5 {label} risk checkpoint {stop}/{len(etas)}")
    return {name:np.concatenate(values) for name,values in parts.items()}


def _fresh_guard_bank(
    cfg: dict[str, Any], flow: Any, truth_model: Any, times: Any,
    label: str, count: int,
) -> dict[str, Any]:
    path = _guard_path(label)
    record_path = path.with_suffix(".json")
    seed = role_seed(label)
    if record_path.exists():
        record = _read_json(record_path)
        if (
            not path.is_file() or _sha256(path) != record["sha256"]
            or record["derived_role_seed"] != seed or record["samples"] != count
        ):
            raise RuntimeError(f"V5 guard checkpoint mismatch: {label}")
        return record
    started = time.perf_counter()
    x, velocity, weights, initial_hash = base._rollout_bank(
        cfg, flow, truth_model, times, seed=seed, samples=count
    )
    base.atomic_npz(
        path,
        configurations=x,
        velocity=velocity,
        base_weights=weights,
        root_seed=np.asarray(ROOT_SEED),
        derived_role_seed=np.asarray(seed),
    )
    record = {
        "schema_version": 4,
        "label": label,
        "path": str(path.relative_to(OUTPUT_ROOT)),
        "sha256": _sha256(path),
        "samples": count,
        "derived_role_seed": seed,
        "initial_state_sha256": initial_hash,
        "wall_time_seconds": time.perf_counter() - started,
    }
    _atomic_json(record_path, record)
    del x, velocity, weights
    gc.collect()
    return record


def generate_data(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Generate all fresh selection and Law-guard banks from the V5 root."""

    protocol = require_v5()
    manifest_path = OUTPUT_ROOT / "banks" / "manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        for row in manifest["artifacts"]:
            if _sha256(OUTPUT_ROOT / row["path"]) != row["sha256"]:
                raise RuntimeError(f"V5 data artifact changed: {row['path']}")
        return manifest

    # The V2 generator is retained only as the fresh standard-role generator;
    # all of its globals have been rebound to the sealed V5 protocol and root.
    standard_manifest = ORIGINAL_BASE_GENERATE_DATA(progress)
    standard_manifest_path = OUTPUT_ROOT / "banks" / "manifest.json"
    provenance_path = OUTPUT_ROOT / "provenance" / "standard_bank_manifest.json"
    if provenance_path.exists():
        if _read_json(provenance_path) != standard_manifest:
            raise RuntimeError("V5 standard-bank provenance mismatch")
        standard_manifest_path.unlink(missing_ok=True)
    else:
        _atomic_json(provenance_path, standard_manifest)
        standard_manifest_path.unlink()

    cfg = base.effective_config()
    with np.load(OUTPUT_ROOT / "design_truth" / "design_truth.npz", allow_pickle=False) as arrays:
        times = jnp.asarray(arrays["times"], dtype=jnp.float64)
    truth_model = base.SkyrmionTruth(base._physics_config(cfg))
    flow = base.load_reference(REFERENCE_PATH)
    guard_records = []
    for label, count in GUARD_COUNTS.items():
        record = _fresh_guard_bank(cfg, flow, truth_model, times, label, count)
        guard_records.append(record)
        if progress:
            progress(f"V5 fresh guard bank {label} N={count}")
    risk_guard_records = []
    for label in RISK_GUARD_COUNTS:
        record = _fresh_risk_guard(cfg, flow, truth_model, times, label)
        risk_guard_records.append(record)
        if progress:
            progress(
                f"V5 fresh risk guard {label} truth N={record['truth_samples']} "
                f"reference N={record['reference_samples']}"
            )

    initial_hashes = dict(standard_manifest["initial_state_hashes"])
    initial_hashes.update({
        row["label"]: row["initial_state_sha256"] for row in guard_records
    })
    for row in risk_guard_records:
        initial_hashes[f"{row['label']}_truth"] = row["truth_initial_state_sha256"]
        initial_hashes[f"{row['label']}_reference"] = row["reference_initial_state_sha256"]
    artifacts = list(standard_manifest["artifacts"])
    artifacts.append({
        "path": str(provenance_path.relative_to(OUTPUT_ROOT)),
        "bytes": provenance_path.stat().st_size,
        "sha256": _sha256(provenance_path),
    })
    for row in guard_records:
        path = OUTPUT_ROOT / row["path"]
        for artifact in (path, path.with_suffix(".json")):
            artifacts.append({
                "path": str(artifact.relative_to(OUTPUT_ROOT)),
                "bytes": artifact.stat().st_size,
                "sha256": _sha256(artifact),
            })
    for row in risk_guard_records:
        for item in row["artifacts"]:
            artifacts.append(item)
        record_path = _risk_guard_root(row["label"]) / "manifest.json"
        artifacts.append({
            "path": str(record_path.relative_to(OUTPUT_ROOT)),
            "bytes": record_path.stat().st_size, "sha256": _sha256(record_path),
        })
    distinct = len(set(initial_hashes.values())) == len(initial_hashes)
    manifest = {
        "schema_version": 4,
        "passed": distinct,
        "protocol_sha256": protocol["protocol_sha256"],
        "v3_protocol_sha256": protocol["protocol_sha256"],
        "root_seed": ROOT_SEED,
        "derived_roles_are_not_replicates": True,
        "inherited_roles": [],
        "fresh_roles": (
            list(protocol["bank_sizes"]) + list(GUARD_COUNTS)
            + list(RISK_GUARD_ROLE_IDS)
        ),
        "initial_state_hashes": initial_hashes,
        "role_disjoint": distinct,
        "reference_retrained": False,
        "reference_sha256": _sha256(REFERENCE_PATH),
        "wall_time_seconds": {
            **standard_manifest["wall_time_seconds"],
            **{row["label"]: row["wall_time_seconds"] for row in guard_records},
            **{row["label"]: row["wall_time_seconds"] for row in risk_guard_records},
        },
        "artifacts": artifacts,
    }
    _atomic_json(manifest_path, manifest)
    if not distinct:
        raise RuntimeError("V5 fresh stochastic roles overlap")
    return manifest


def score_candidate_universe(
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    require_v5()
    generate_data(progress)
    base_receipt = base.score_candidate_universe(progress)
    if RISK_MATRIX_PATH.exists():
        return _read_json(RISK_MATRIX_PATH)
    etas = np.asarray([row["eta"] for row in base_receipt["rows"]], dtype=np.float64)
    selection_risk = np.asarray(
        [row["exact_scientific_risk"] for row in base_receipt["rows"]],
        dtype=np.float64,
    )
    role_results: dict[str, dict[str, np.ndarray]] = {}
    started = time.perf_counter()
    for label in RISK_GUARD_COUNTS:
        role_started = time.perf_counter()
        data = _risk_guard_data(label)
        evaluator = RiskOnlyEvaluator(
            data, batch_size=RISK_CANDIDATE_BATCH_SIZE
        )
        role_results[label] = _evaluate_risk_checkpointed(
            evaluator, etas, f"five_role_{label}", progress=progress,
        )
        role_results[label]["wall_time_seconds"] = np.asarray(
            time.perf_counter() - role_started
        )
        del evaluator, data
        gc.collect()
        if progress:
            progress(f"V5 exact risk guard {label}: {len(etas)} candidates")
    arrays: dict[str, Any] = {"eta": etas, "selection__risk": selection_risk}
    rows = []
    for index, source in enumerate(base_receipt["rows"]):
        risk_by_role = {"selection": float(selection_risk[index])}
        valid_by_role = {"selection": bool(source["risk_anchor_valid"])}
        support_by_role = {}
        for label, result in role_results.items():
            risk_by_role[label] = float(result["scientific_risk"][index])
            valid_by_role[label] = bool(result["valid"][index])
            support_by_role[label] = {
                "valid": bool(result["valid"][index]),
                "minimum_rESS": float(result["minimum_ress"][index]),
                "maximum_projection_residual": float(result["maximum_projection_residual"][index]),
            }
        rows.append({
            **source, "risk_by_role": risk_by_role,
            "risk_numerically_valid_by_role": valid_by_role,
            "risk_guard_support": support_by_role,
            "all_five_risk_roles_valid": all(valid_by_role.values()),
        })
    for label, result in role_results.items():
        for name, value in result.items(): arrays[f"{label}__{name}"] = value
    base.atomic_npz(RISK_MATRIX_ARRAYS_PATH, compressed=True, **arrays)
    receipt = {
        "schema_version": 5, "passed": True,
        "protocol_sha256": require_v5()["protocol_sha256"],
        "candidate_count": len(rows), "risk_roles": list(RISK_ROLE_NAMES),
        "risk_guard_roles": list(RISK_GUARD_COUNTS),
        "exact_before_full_action": base_receipt["exact_before_full_action"],
        "full_action_evaluations_before_receipt": base_receipt["full_action_evaluations_before_receipt"],
        "one_evaluation_per_candidate_role": True,
        "reused_across_allowances": [0.5, 1.0, 2.0],
        "arrays_sha256": _sha256(RISK_MATRIX_ARRAYS_PATH),
        "wall_time_seconds": time.perf_counter() - started, "rows": rows,
    }
    _atomic_json(RISK_MATRIX_PATH, receipt)
    fields = [
        "candidate_id", "eta_sha256", "component", "geometry_valid",
        "jointly_supported", "all_five_risk_roles_valid",
        *[f"risk_{role}" for role in RISK_ROLE_NAMES],
        *[f"valid_{role}" for role in RISK_ROLE_NAMES],
    ]
    with tempfile.NamedTemporaryFile(
        mode="w", prefix=".risk_matrix.", suffix=".csv", dir=OUTPUT_ROOT,
        delete=False, newline="", encoding="utf-8",
    ) as handle:
        temporary = Path(handle.name); writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "candidate_id": row["candidate_id"], "eta_sha256": row["eta_sha256"],
                "component": row["component"], "geometry_valid": row["geometry_valid"],
                "jointly_supported": row["jointly_supported"],
                "all_five_risk_roles_valid": row["all_five_risk_roles_valid"],
                **{f"risk_{role}": row["risk_by_role"][role] for role in RISK_ROLE_NAMES},
                **{f"valid_{role}": row["risk_numerically_valid_by_role"][role] for role in RISK_ROLE_NAMES},
            })
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, RISK_MATRIX_CSV_PATH)
    return receipt


class V5SelectionRuntime(ORIGINAL_BASE_SELECTION_RUNTIME):
    """V4 runtime augmented with cached five-role exact-risk feasibility."""

    def __init__(self) -> None:
        super().__init__()
        matrix = _read_json(RISK_MATRIX_PATH)
        self.v5_exact_cache = {
            row["eta_sha256"]: row for row in matrix["rows"]
        }
        self.risk_guard_evaluators: dict[str, RiskOnlyEvaluator] = {}
        self.constraint_law: dict[str, Any] | None = None
        self.constraint_allowance: float | None = None

    def set_constraint(self, law: dict[str, Any], allowance: float) -> None:
        self.constraint_law = self.raw_exact_receipt(law["eta"])
        self.constraint_allowance = float(allowance)

    def raw_exact_receipt(
        self, eta: Any, *, provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        key = base.eta_key(eta)
        if key in self.v5_exact_cache:
            return self.v5_exact_cache[key]
        original = super().exact_receipt(eta, provenance=provenance)
        risk_by_role = {"selection": original["exact_scientific_risk"]}
        valid_by_role = {"selection": bool(original["risk_anchor_valid"])}
        support = {}
        for label in RISK_GUARD_COUNTS:
            if label not in self.risk_guard_evaluators:
                self.risk_guard_evaluators[label] = RiskOnlyEvaluator(
                    _risk_guard_data(label), batch_size=1
                )
            result = self.risk_guard_evaluators[label].evaluate(
                np.asarray(eta, dtype=np.float64)[None]
            )
            risk_by_role[label] = float(result["scientific_risk"][0])
            valid_by_role[label] = bool(result["valid"][0])
            support[label] = {
                "valid": bool(result["valid"][0]),
                "minimum_rESS": float(result["minimum_ress"][0]),
                "maximum_projection_residual": float(result["maximum_projection_residual"][0]),
            }
        row = {
            **original, "risk_by_role": risk_by_role,
            "risk_numerically_valid_by_role": valid_by_role,
            "risk_guard_support": support,
            "all_five_risk_roles_valid": all(valid_by_role.values()),
        }
        self.v5_exact_cache[key] = row
        self.generated_exact_rows[-1] = row
        return row

    @staticmethod
    def robust_feasible(
        receipt: dict[str, Any], law: dict[str, Any], allowance: float,
    ) -> bool:
        limit = float(allowance) / 100.0
        return bool(
            receipt["jointly_supported"]
            and receipt["all_five_risk_roles_valid"]
            and all(
                receipt["risk_by_role"][role] / law["risk_by_role"][role] - 1.0
                <= limit
                for role in RISK_ROLE_NAMES
            )
        )

    def exact_receipt(
        self, eta: Any, *, provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raw = self.raw_exact_receipt(eta, provenance=provenance)
        if self.constraint_law is None or self.constraint_allowance is None:
            return raw
        feasible = self.robust_feasible(
            raw, self.constraint_law, self.constraint_allowance
        )
        return {**raw, "jointly_supported": feasible, "robust_feasible": feasible}


def _raw_law(law: dict[str, Any], runtime: V5SelectionRuntime) -> dict[str, Any]:
    return runtime.raw_exact_receipt(law["eta"])


def _robust_rows(
    rows: list[dict[str, Any]], law: dict[str, Any], allowance: float,
    runtime: V5SelectionRuntime,
) -> list[dict[str, Any]]:
    raw_law = _raw_law(law, runtime)
    return [
        row for source in rows
        for row in [runtime.raw_exact_receipt(source["eta"])]
        if runtime.robust_feasible(row, raw_law, allowance)
    ]


def run_selection_pass_v5(
    pass_index: int, law: dict[str, Any],
    *, progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    path = OUTPUT_ROOT / f"selection_pass_{pass_index}" / "complete.json"
    if path.exists(): return _read_json(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    protocol=require_v5(); matrix=_read_json(RISK_MATRIX_PATH); runtime=V5SelectionRuntime()
    law=runtime.raw_exact_receipt(law["eta"])
    tangent_results=[]; full_results=[]; tangent_incumbent=None; full_incumbent=None
    robust_sets={}
    for allowance in protocol["allowances_percent"]:
        runtime.set_constraint(law, allowance)
        feasible=_robust_rows(matrix["rows"],law,allowance,runtime)
        for row in (law,tangent_incumbent,full_incumbent):
            if row is not None and runtime.robust_feasible(runtime.raw_exact_receipt(row["eta"]),law,allowance):
                feasible.append(runtime.raw_exact_receipt(row["eta"]))
        feasible=base._unique_rows(feasible)
        robust_sets[str(allowance)] = [base.eta_key(row["eta"]) for row in feasible]
        tangent_starts=base._select_starts(
            feasible,law,tangent_incumbent,
            count=int(protocol["starts"]["tangent_per_allowance"]),
        )
        trajectories=[base._tangent_trajectory(
            runtime,start,ceiling=base.selection_ceiling(law["exact_scientific_risk"],allowance),
            pass_index=pass_index,allowance=allowance,trajectory_index=index,
        ) for index,start in enumerate(tangent_starts)]
        endpoints=[{**row["endpoint_exact_receipt"],"search":row["endpoint"]} for row in trajectories if row["eligible_endpoint"]]
        shortlist=base._mandatory_shortlist(
            endpoints,law,tangent_incumbent,
            cap=int(protocol["starts"]["authoritative_finalist_cap"]),
            action_key=lambda row:float(row.get("search",{}).get("action",float("inf"))),
        )
        certified=[]
        for row in shortlist:
            exact=runtime.exact_receipt(row["eta"])
            authoritative=runtime.tangent_authoritative_evaluate(row["eta"])
            if exact["jointly_supported"] and authoritative["valid"]:
                certified.append({**exact,"authoritative":authoritative,"selection_action":authoritative["train_certificate"]["action"]})
        if not certified: raise RuntimeError(f"no robust Tangent finalist at {allowance}%")
        best=min(certified,key=lambda row:(row["selection_action"],base.eta_key(row["eta"])))
        old=None if tangent_incumbent is None else next((row for row in certified if base.eta_key(row["eta"])==base.eta_key(tangent_incumbent["eta"])),None)
        winner=old if old is not None and best["selection_action"]>=old["selection_action"]-float(protocol["replacement_tolerance"]) else best
        tangent_result={
            "allowance_percent":allowance,"risk_ceiling":base.selection_ceiling(law["exact_scientific_risk"],allowance),
            "robust_five_role_feasible_pool_count":len(feasible),"starts":tangent_starts,
            "trajectories":trajectories,"authoritative_finalists":certified,"winner":winner,
            "incumbent_retained":bool(tangent_incumbent is not None and base.eta_key(winner["eta"])==base.eta_key(tangent_incumbent["eta"])),
        }
        tangent_results.append(tangent_result);tangent_incumbent=winner
        base.atomic_json(OUTPUT_ROOT/f"selection_pass_{pass_index}"/"tangent"/f"allowance_{base.slug(allowance)}"/"result.json",tangent_result)
        if progress:progress(f"V5 pass {pass_index} Tangent {allowance}% complete")

        feasible=base._unique_rows([*feasible,runtime.raw_exact_receipt(winner["eta"])])
        full_starts=base._select_starts(
            feasible,law,full_incumbent,count=int(protocol["starts"]["full_per_allowance"]),
            additional_mandatory=[winner],
        )
        full_trajectories=[base._full_trajectory(
            runtime,start,ceiling=base.selection_ceiling(law["exact_scientific_risk"],allowance),
            pass_index=pass_index,allowance=allowance,trajectory_index=index,
        ) for index,start in enumerate(full_starts)]
        full_endpoints=[{**row["endpoint_exact_receipt"],"search":row["endpoint"]} for row in full_trajectories if row["eligible_endpoint"]]
        full_shortlist=base._mandatory_shortlist(
            full_endpoints,law,full_incumbent,
            cap=int(protocol["starts"]["authoritative_finalist_cap"]),
            action_key=lambda row:float(row.get("search",{}).get("action",float("inf"))),
            additional_mandatory=[winner],
        )
        full_certified=[]
        for row in full_shortlist:
            exact=runtime.exact_receipt(row["eta"]); authoritative=runtime.full_authoritative_evaluate(row["eta"])
            if exact["jointly_supported"] and authoritative["valid"]:
                full_certified.append({**exact,"authoritative":authoritative,"selection_action":authoritative["train_action"]})
        if not full_certified: raise RuntimeError(f"no robust Full finalist at {allowance}%")
        full_best=min(full_certified,key=lambda row:(row["selection_action"],base.eta_key(row["eta"])))
        old=None if full_incumbent is None else next((row for row in full_certified if base.eta_key(row["eta"])==base.eta_key(full_incumbent["eta"])),None)
        full_winner=old if old is not None and full_best["selection_action"]>=old["selection_action"]-float(protocol["replacement_tolerance"]) else full_best
        full_result={
            "allowance_percent":allowance,"risk_ceiling":base.selection_ceiling(law["exact_scientific_risk"],allowance),
            "robust_five_role_feasible_pool_count":len(feasible),"starts":full_starts,
            "trajectories":full_trajectories,"authoritative_finalists":full_certified,"winner":full_winner,
            "law_mandatory":any("mandatory_law" in row["mandatory_roles"] for row in full_starts),
            "previous_incumbent_mandatory":full_incumbent is None or any("mandatory_previous_incumbent" in row["mandatory_roles"] for row in full_starts),
            "current_tangent_mandatory":any("mandatory_current_tangent" in row["mandatory_roles"] for row in full_starts),
            "incumbent_retained":bool(full_incumbent is not None and base.eta_key(full_winner["eta"])==base.eta_key(full_incumbent["eta"])),
        }
        full_results.append(full_result);full_incumbent=full_winner
        base.atomic_json(OUTPUT_ROOT/f"selection_pass_{pass_index}"/"full"/f"allowance_{base.slug(allowance)}"/"result.json",full_result)
        if progress:progress(f"V5 pass {pass_index} Full {allowance}% complete")
    generated=base._unique_rows(runtime.generated_exact_rows)
    result={
        "schema_version":5,"protocol_sha256":protocol["protocol_sha256"],"pass_index":pass_index,
        "law":law,"risk_ceilings":{str(a):base.selection_ceiling(law["exact_scientific_risk"],a) for a in protocol["allowances_percent"]},
        "robust_feasible_sets":robust_sets,"tangent":tangent_results,"full":full_results,
        "generated_exact_receipts":generated,"generated_candidate_count":len(generated),
        "full_action_nonincreasing":all(b["winner"]["selection_action"]<=a["winner"]["selection_action"]+float(protocol["replacement_tolerance"]) for a,b in zip(full_results[:-1],full_results[1:])),
        "complete":True,
    }
    base.atomic_json(path,result);return result


def _light_guard_data(bank: Any) -> SelectionGalerkinData:
    cfg = base.effective_config()
    with np.load(
        OUTPUT_ROOT / "design_truth" / "design_truth.npz", allow_pickle=False
    ) as arrays:
        times = jnp.asarray(arrays["times"], dtype=jnp.float64)
        truth = jnp.asarray(arrays["configurations"], dtype=jnp.float64)
        truth_means = jnp.asarray(arrays["truth_means"], dtype=jnp.float64)
        whitening = jnp.asarray(arrays["whitening"], dtype=jnp.float64)
    problem = base._problem(
        cfg, truth, times, noise_seed=role_seed("selection_observation_noise")
    )
    return SelectionGalerkinData(
        selection_problem=problem,
        projection_bank=bank,
        train_bank=bank,
        audit_bank=bank,
        reference_features=jnp.empty((0,), dtype=jnp.float64),
        truth_means=truth_means,
        whitening=whitening,
    )


def _guard_cache_path(eta: Any) -> Path:
    return OUTPUT_ROOT / "law" / "guard_cache" / f"{base.eta_key(eta)}.json"


def _guard_checkpoint_root(rows: list[dict[str, Any]]) -> Path:
    key = _payload_sha256([base.eta_key(row["eta"]) for row in rows])
    return OUTPUT_ROOT / "law" / "guard_role_checkpoints" / key


def guard_worker(role: str, input_path: Path, output_path: Path) -> None:
    activate()
    require_v5()
    rows = _read_json(input_path)["rows"]
    bank = _load_guard(role)
    data = _light_guard_data(bank)
    evaluator = base.CandidateEvaluator(data, batch_size=1)
    result = evaluator.evaluate(
        np.asarray([row["eta"] for row in rows], dtype=np.float64), bank
    )
    base.atomic_npz(
        output_path,
        support_valid=result["support_valid"],
        minimum_ress=result["minimum_ress"],
        maximum_projection_residual=result["maximum_projection_residual"],
        maximum_forcing_mean=result["maximum_forcing_mean"],
        maximum_covariance_condition=result["maximum_covariance_condition"],
    )


def isolated_guard_qualify_rows(
    rows: list[dict[str, Any]],
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    generate_data(progress)
    cached: dict[str, dict[str, Any]] = {}
    missing = []
    for row in rows:
        path = _guard_cache_path(row["eta"])
        if path.exists():
            cached[base.eta_key(row["eta"])] = _read_json(path)
        else:
            missing.append(row)
    if missing:
        root = _guard_checkpoint_root(missing)
        input_path = root / "input.json"
        _atomic_json(input_path, {"rows": missing})
        by_role: dict[str, dict[str, np.ndarray]] = {}
        for role in GUARD_COUNTS:
            output_path = root / f"{role}.npz"
            if not output_path.exists():
                environment = dict(os.environ)
                environment["JAX_ENABLE_X64"] = "1"
                environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
                environment.setdefault(
                    "JAX_COMPILATION_CACHE_DIR",
                    str(OUTPUT_ROOT / "performance" / "jax_compilation_cache"),
                )
                subprocess.run(
                    [
                        sys.executable, "-m",
                        "experiments.skyrmions_galerkin.official_b1_pareto_v5_single_seed_run",
                        "--worker-role", role,
                        "--worker-input", str(input_path),
                        "--worker-output", str(output_path),
                    ],
                    cwd=REPO_ROOT, check=True, env=environment,
                )
            with np.load(output_path, allow_pickle=False) as arrays:
                by_role[role] = {
                    name: np.asarray(arrays[name]) for name in arrays.files
                }
            if progress:
                progress(f"V5 isolated Law guard {role} sealed")
        protocol = require_v5()
        for index, row in enumerate(missing):
            support = {
                role: {
                    "support_valid": bool(result["support_valid"][index]),
                    "minimum_rESS": float(result["minimum_ress"][index]),
                    "maximum_projection_residual": float(result["maximum_projection_residual"][index]),
                    "maximum_forcing_mean": float(result["maximum_forcing_mean"][index]),
                    "maximum_covariance_condition": float(result["maximum_covariance_condition"][index]),
                }
                for role, result in by_role.items()
            }
            receipt = {
                "schema_version": 4,
                "protocol_sha256": protocol["protocol_sha256"],
                "candidate_id": row.get(
                    "candidate_id", f"downstream_{base.eta_key(row['eta'])}"
                ),
                "eta": row["eta"],
                "eta_sha256": base.eta_key(row["eta"]),
                "exact_scientific_risk": row["exact_scientific_risk"],
                "support_by_fresh_guard_role": support,
                "support_robust": all(item["support_valid"] for item in support.values()),
                "minimum_guard_rESS": min(item["minimum_rESS"] for item in support.values()),
                "threshold": 0.05,
                "authoritative_audit_used": False,
                "evaluation_minibatch_size": 1,
                "feature_sample_chunk": FEATURE_SAMPLE_CHUNK,
            }
            _atomic_json(_guard_cache_path(row["eta"]), receipt)
            cached[receipt["eta_sha256"]] = receipt
        if progress:
            progress(f"V5 guard-qualified {len(missing)} candidate rows")
    return [cached[base.eta_key(row["eta"])] for row in rows]


def refreeze_law(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    activate()
    score_candidate_universe(progress)
    return v3.refreeze_law(progress)


def run_selection(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    activate()
    refreeze_law(progress)
    return run_selection_with_restarts_v5(progress)


def run_selection_with_restarts_v5(
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Apply the frozen support-robust Law restart rule to V5 passes."""

    require_v5()
    refreeze_law(progress)
    final_path = OUTPUT_ROOT / "selection" / "restart_summary.json"
    if final_path.exists():
        return _read_json(final_path)
    matrix = _read_json(RISK_MATRIX_PATH)
    initial = _read_json(LAW_PATH)
    law = next(
        row for row in matrix["rows"]
        if row["eta_sha256"] == initial["eta_sha256"]
    )
    passes: list[dict[str, Any]] = []
    maximum_restarts = int(require_v5()["maximum_anchor_refinement_restarts"])
    tolerance = float(require_v5()["law_consistency_tolerance"])
    status = "PASS"
    for pass_index in range(maximum_restarts + 1):
        result = run_selection_pass_v5(pass_index, law, progress=progress)
        challengers = [
            row for row in result["generated_exact_receipts"]
            if row["jointly_supported"]
            and row["exact_scientific_risk"]
            < law["exact_scientific_risk"] - tolerance
        ]
        guard_receipts = (
            isolated_guard_qualify_rows(challengers, progress)
            if challengers else []
        )
        robust = [
            row for row, receipt in zip(challengers, guard_receipts, strict=True)
            if receipt["support_robust"]
        ]
        best = min(
            [law, *robust],
            key=lambda row: (row["exact_scientific_risk"], base.eta_key(row["eta"])),
        )
        material = bool(
            best["exact_scientific_risk"]
            < law["exact_scientific_risk"] - tolerance
        )
        passes.append({
            "pass_index": pass_index,
            "path": str(
                (OUTPUT_ROOT / f"selection_pass_{pass_index}" / "complete.json")
                .relative_to(OUTPUT_ROOT)
            ),
            "law_eta_sha256": base.eta_key(law["eta"]),
            "R_star": law["exact_scientific_risk"],
            "downstream_material_risk_challengers": len(challengers),
            "downstream_guard_robust_challengers": len(robust),
            "material_law_improvement": material,
            "law_improvement": (
                best["exact_scientific_risk"] - law["exact_scientific_risk"]
            ),
        })
        if not material:
            break
        if pass_index == maximum_restarts:
            status = "FAIL_SUPPORT_ROBUST_ANCHOR_INCONSISTENT_AFTER_MAXIMUM_RESTARTS"
            break
        previous = law
        law = best
        _atomic_json(OUTPUT_ROOT / "law" / f"reanchor_{pass_index + 1}.json", {
            "schema_version": 5,
            "triggering_pass": pass_index,
            "previous_R_star": previous["exact_scientific_risk"],
            "new_R_star": law["exact_scientific_risk"],
            "improvement": law["exact_scientific_risk"] - previous["exact_scientific_risk"],
            "tolerance": tolerance,
            "new_law": law,
            "fresh_guard_receipt_sha256": _sha256(_guard_cache_path(law["eta"])),
            "complete_restart_required": True,
        })
    final_pass_index = passes[-1]["pass_index"]
    final_pass = _read_json(
        OUTPUT_ROOT / f"selection_pass_{final_pass_index}" / "complete.json"
    )
    robust_payload = {
        "schema_version": 5,
        "protocol_sha256": require_v5()["protocol_sha256"],
        "law_eta_sha256": base.eta_key(final_pass["law"]["eta"]),
        "law_risk_by_role": final_pass["law"]["risk_by_role"],
        "sets": final_pass["robust_feasible_sets"],
        "source_matrix_sha256": _sha256(RISK_MATRIX_PATH),
        "all_five_roles_required": True,
    }
    _atomic_json(ROBUST_FEASIBLE_PATH, robust_payload)
    summary = {
        "schema_version": 5,
        "passed": status == "PASS",
        "status": status,
        "protocol_sha256": require_v5()["protocol_sha256"],
        "passes": passes,
        "restart_count": len(passes) - 1,
        "maximum_restart_count": maximum_restarts,
        "final_pass_index": final_pass_index,
        "final_law": final_pass["law"],
        "final_law_consistent": not passes[-1]["material_law_improvement"],
        "final_full_action_nonincreasing": final_pass["full_action_nonincreasing"],
        "guard_robust_reanchor_rule": True,
        "five_role_robust_search_from_start": True,
        "robust_feasible_sets_sha256": _sha256(ROBUST_FEASIBLE_PATH),
    }
    _atomic_json(final_path, summary)
    if not summary["passed"]:
        raise RuntimeError(status)
    return summary


def _selected_rows(final_pass: dict[str, Any]) -> list[dict[str, Any]]:
    selected = [{
        "method": "Law", "allowance_percent": None,
        "row": final_pass["law"],
        "incumbent_status": "mandatory Law baseline",
    }]
    for tangent, full in zip(final_pass["tangent"], final_pass["full"], strict=True):
        selected.extend(({
            "method": "Tangent",
            "allowance_percent": tangent["allowance_percent"],
            "row": tangent["winner"],
            "incumbent_status": (
                "retained previous incumbent" if tangent["incumbent_retained"]
                else "new authoritative winner"
            ),
        }, {
            "method": "Full",
            "allowance_percent": full["allowance_percent"],
            "row": full["winner"],
            "incumbent_status": (
                "retained previous incumbent" if full["incumbent_retained"]
                else "new authoritative winner"
            ),
        }))
    return selected


def verify_selection_v5(
    selection: dict[str, Any] | None = None, *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Independently reconstruct robust sets, winners, and the sealed geometry."""

    if selection is None:
        selection = _read_json(SELECTION_SEAL_PATH)
    seal_sha = _sha256(SELECTION_SEAL_PATH)
    if SELECTION_VERIFICATION_PATH.exists():
        result = _read_json(SELECTION_VERIFICATION_PATH)
        if result["selection_seal_sha256"] != seal_sha or not result["passed"]:
            raise RuntimeError("V5 independent verification mismatch")
        return result
    protocol = require_v5()
    restart = _read_json(OUTPUT_ROOT / "selection" / "restart_summary.json")
    final_path = OUTPUT_ROOT / f"selection_pass_{restart['final_pass_index']}" / "complete.json"
    final_pass = _read_json(final_path)
    matrix = _read_json(RISK_MATRIX_PATH)
    runtime = V5SelectionRuntime()
    law = runtime.raw_exact_receipt(final_pass["law"]["eta"])
    tolerance = float(protocol["replacement_tolerance"])
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    def roles(row: dict[str, Any]) -> set[str]:
        return set(row.get("mandatory_roles", ())) | {row.get("start_role", "")}

    def mandatory(starts: list[dict[str, Any]], role: str, eta: Any) -> bool:
        key = base.eta_key(eta)
        return any(base.eta_key(row["eta"]) == key and role in roles(row) for row in starts)

    def winner(finalists: list[dict[str, Any]], incumbent: dict[str, Any] | None):
        best = min(finalists, key=lambda row: (row["selection_action"], base.eta_key(row["eta"])))
        old = None if incumbent is None else next((
            row for row in finalists
            if base.eta_key(row["eta"]) == base.eta_key(incumbent["eta"])
        ), None)
        return old if old is not None and best["selection_action"] >= old["selection_action"] - tolerance else best

    check("exact_pool_precedes_action", bool(
        matrix["exact_before_full_action"]
        and matrix["full_action_evaluations_before_receipt"] == 0
    ))
    check("all_four_fresh_risk_guards_present", set(RISK_GUARD_COUNTS) == set(
        matrix["risk_guard_roles"]
    ))
    check("sealed_complete_candidate_risks_reconstructed", bool(
        len(selection["candidate_five_role_risks"]) == matrix["candidate_count"]
        and selection["candidate_five_role_risks_sha256"]
        == _payload_sha256(selection["candidate_five_role_risks"])
        and all(
            sealed["risk_by_role"] == source["risk_by_role"]
            for sealed, source in zip(
                selection["candidate_five_role_risks"], matrix["rows"], strict=True
            )
        )
    ))
    for allowance in protocol["allowances_percent"]:
        expected = [base.eta_key(row["eta"]) for row in _robust_rows(
            matrix["rows"], law, allowance, runtime
        )]
        expected = sorted(set(expected))
        observed = sorted(set(final_pass["robust_feasible_sets"][str(allowance)]))
        check(f"robust_set_{allowance:g}_reconstructed", expected == observed, {
            "expected_count": len(expected), "observed_count": len(observed),
        })

    tangent_incumbent = None
    full_incumbent = None
    reconstructed = [{
        "method": "Law", "allowance_percent": None,
        "eta": law["eta"], "eta_sha256": base.eta_key(law["eta"]),
    }]
    for index, (tangent, full) in enumerate(zip(final_pass["tangent"], final_pass["full"], strict=True)):
        allowance = float(tangent["allowance_percent"])
        for method, result in (("tangent", tangent), ("full", full)):
            check(f"{method}_{allowance:g}_all_finalists_robust", all(
                runtime.robust_feasible(runtime.raw_exact_receipt(row["eta"]), law, allowance)
                for row in result["authoritative_finalists"]
            ))
        tangent_expected = winner(tangent["authoritative_finalists"], tangent_incumbent)
        check(f"tangent_{allowance:g}_winner_reconstructed", base.eta_key(tangent_expected["eta"]) == base.eta_key(tangent["winner"]["eta"]))
        tangent_incumbent = tangent_expected
        reconstructed.append({
            "method": "Tangent", "allowance_percent": allowance,
            "eta": tangent_expected["eta"], "eta_sha256": base.eta_key(tangent_expected["eta"]),
        })
        check(f"full_{allowance:g}_law_mandatory", mandatory(full["starts"], "mandatory_law", law["eta"]))
        check(f"full_{allowance:g}_current_tangent_mandatory", mandatory(full["starts"], "mandatory_current_tangent", tangent["winner"]["eta"]))
        if index:
            check(f"full_{allowance:g}_previous_incumbent_mandatory", mandatory(full["starts"], "mandatory_previous_incumbent", full_incumbent["eta"]))
        full_expected = winner(full["authoritative_finalists"], full_incumbent)
        check(f"full_{allowance:g}_winner_reconstructed", base.eta_key(full_expected["eta"]) == base.eta_key(full["winner"]["eta"]))
        full_incumbent = full_expected
        reconstructed.append({
            "method": "Full", "allowance_percent": allowance,
            "eta": full_expected["eta"], "eta_sha256": base.eta_key(full_expected["eta"]),
        })
    sealed = [{key: row[key] for key in ("method", "allowance_percent", "eta", "eta_sha256")} for row in selection["winners"]]
    check("selection_seal_matches_reconstruction", sealed == reconstructed)
    check("all_authoritative_samples_131072", all(
        row["authoritative_fit_samples"] == 131072
        and row["authoritative_audit_samples"] == 131072
        for row in selection["rows"]
    ))
    check("final_law_consistent", restart["final_law_consistent"])
    result = {
        "schema_version": 5,
        "passed": all(row["passed"] for row in checks),
        "validation_accessed": False,
        "protocol_sha256": protocol["protocol_sha256"],
        "selection_seal_sha256": seal_sha,
        "final_pass_sha256": _sha256(final_path),
        "risk_matrix_sha256": _sha256(RISK_MATRIX_PATH),
        "robust_feasible_sets_sha256": _sha256(ROBUST_FEASIBLE_PATH),
        "checks": checks,
        "reconstructed_winners": reconstructed,
    }
    _atomic_json(SELECTION_VERIFICATION_PATH, result)
    if not result["passed"]:
        raise RuntimeError(f"V5 independent verification failed: {[row['name'] for row in checks if not row['passed']]}")
    if progress:
        progress("V5 independent robust-set and winner reconstruction passed")
    return result


def certify_selection(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    activate()
    restart = run_selection_with_restarts_v5(progress)
    if SELECTION_SEAL_PATH.exists():
        seal = _read_json(SELECTION_SEAL_PATH)
        verify_selection_v5(seal, progress=progress)
        return seal
    protocol = require_v5()
    final_pass = _read_json(
        OUTPUT_ROOT / f"selection_pass_{restart['final_pass_index']}" / "complete.json"
    )
    runtime = V5SelectionRuntime()
    law = runtime.raw_exact_receipt(final_pass["law"]["eta"])
    certified_cache: dict[str, dict[str, Any]] = {}
    rows = []
    for chosen in _selected_rows(final_pass):
        eta = chosen["row"]["eta"]
        key = base.eta_key(eta)
        if key not in certified_cache:
            exact = runtime.raw_exact_receipt(eta)
            full = runtime.full_authoritative_evaluate(eta)
            tangent = runtime.tangent_authoritative_evaluate(eta)
            certified_cache[key] = {"exact": exact, "full": full, "tangent": tangent}
        item = certified_cache[key]
        exact, full, tangent = item["exact"], item["full"], item["tangent"]
        full_eval, full_audit = full["evaluation"], full["audit"]
        allowance = chosen["allowance_percent"]
        relative = {
            role: exact["risk_by_role"][role] / law["risk_by_role"][role] - 1.0
            for role in RISK_ROLE_NAMES
        }
        robust = bool(
            allowance is None or runtime.robust_feasible(exact, law, allowance)
        )
        rows.append({
            "method": chosen["method"], "allowance_percent": allowance,
            "eta": eta, "eta_sha256": key,
            "exact_risk": exact["exact_scientific_risk"],
            "relative_risk_increase": exact["exact_scientific_risk"] / law["exact_scientific_risk"] - 1.0,
            "risk_by_role": exact["risk_by_role"],
            "relative_risk_by_role": relative,
            "maximum_relative_risk_increase": max(relative.values()),
            "all_five_role_risk_pass": robust,
            "train_K280_action": full["train_action"],
            "audit_K280_action": full["audit_action"],
            "selection_action": chosen["row"].get("selection_action", full["train_action"]),
            "tangent_train_action": tangent["train_certificate"]["action"],
            "minimum_rESS": exact["minimum_rESS"],
            "maximum_covariance_condition": exact["maximum_covariance_condition"],
            "galerkin_rank_by_time": full_eval["rank_by_time"],
            "minimum_galerkin_rank": min(full_eval["rank_by_time"]),
            "maximum_galerkin_condition": full_eval["worst_retained_condition"],
            "maximum_range_residual": full_eval["worst_range_residual"],
            "maximum_stationarity_residual": full_eval["worst_stationarity_residual"],
            "maximum_weak_residual": full_audit["heldout_certificate"]["maximum_weak_residual"],
            "maximum_energy_residual": full_audit["heldout_certificate"]["maximum_energy_residual"],
            "full_certificate_pass": full["valid"],
            "tangent_certificate_pass": tangent["valid"],
            "incumbent_status": chosen["incumbent_status"],
            "authoritative_fit_samples": 131072,
            "authoritative_audit_samples": 131072,
            "galerkin_backend": "jax", "dtype": "float64",
        })
        if progress:
            progress(f"V5 authoritative certificate {chosen['method']} {allowance if allowance is not None else 'Law'}")
    full_rows = [row for row in rows if row["method"] == "Full"]
    law_row = next(row for row in rows if row["method"] == "Law")
    replacement = float(protocol["replacement_tolerance"])
    law_baseline = full_rows[0]["selection_action"] <= law_row["train_K280_action"] + replacement
    nonincreasing = all(
        current["selection_action"] <= previous["selection_action"] + replacement
        for previous, current in zip(full_rows[:-1], full_rows[1:])
    )
    winners = [{
        "method": row["method"], "allowance_percent": row["allowance_percent"],
        "eta": row["eta"], "eta_sha256": row["eta_sha256"],
    } for row in rows]
    risk_matrix = _read_json(RISK_MATRIX_PATH)
    compact_candidate_risks = [{
        "candidate_id": row["candidate_id"],
        "eta_sha256": row["eta_sha256"],
        "geometry_valid": row["geometry_valid"],
        "support_valid": row["jointly_supported"],
        "risk_by_role": row["risk_by_role"],
        "risk_numerically_valid_by_role": row["risk_numerically_valid_by_role"],
    } for row in risk_matrix["rows"]]
    seal = {
        "schema_version": 5,
        "passed": bool(
            restart["passed"] and restart["final_law_consistent"]
            and all(item["full"]["valid"] for item in certified_cache.values())
            and all(row["all_five_role_risk_pass"] for row in rows)
            and law_baseline and nonincreasing
        ),
        "selection_frozen": True, "validation_accessed": False,
        "protocol_sha256": protocol["protocol_sha256"],
        "restart_summary_sha256": _sha256(OUTPUT_ROOT / "selection" / "restart_summary.json"),
        "final_pass_sha256": _sha256(OUTPUT_ROOT / f"selection_pass_{restart['final_pass_index']}" / "complete.json"),
        "risk_matrix_sha256": _sha256(RISK_MATRIX_PATH),
        "robust_feasible_sets_sha256": _sha256(ROBUST_FEASIBLE_PATH),
        "law_risk_by_role": law["risk_by_role"],
        "law_support_receipt": _read_json(_guard_cache_path(law["eta"])),
        "law_support_receipt_sha256": _sha256(_guard_cache_path(law["eta"])),
        "candidate_five_role_risks": compact_candidate_risks,
        "candidate_five_role_risks_sha256": _payload_sha256(compact_candidate_risks),
        "robust_feasible_sets": _read_json(ROBUST_FEASIBLE_PATH)["sets"],
        "incumbent_lineage": [{
            "allowance_percent": full["allowance_percent"],
            "incumbent_retained": full["incumbent_retained"],
            "winner_eta_sha256": base.eta_key(full["winner"]["eta"]),
        } for full in final_pass["full"]],
        "winner_geometry_hash": _payload_sha256(winners),
        "winners": winners, "rows": rows,
        "unique_geometry_count": len(certified_cache),
        "law_mandatory_baseline_pass": law_baseline,
        "full_action_nonincreasing": nonincreasing,
        "all_authoritative_full_certificates_pass": all(item["full"]["valid"] for item in certified_cache.values()),
        "all_five_role_risk_constraints_pass": all(row["all_five_role_risk_pass"] for row in rows),
        "authoritative_fit_samples": 131072, "authoritative_audit_samples": 131072,
        "all_galerkin_backends": ["jax"], "all_scientific_action_dtypes": ["float64"],
    }
    _atomic_json(SELECTION_SEAL_PATH, seal)
    if not seal["passed"]:
        raise RuntimeError("V5 selection/certification seal failed")
    verify_selection_v5(seal, progress=progress)
    return seal


def generate_heldout(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    activate()
    return base.generate_heldout(progress)


def _heldout_bank_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    hashes = {Path(row["path"]).name: row["sha256"] for row in manifest["artifacts"]}
    protocol = require_v5()
    return {
        "fit": hashes[f"reference_fit_N{protocol['validation']['reference_fit_samples']}.npz"],
        "audit": hashes[f"reference_audit_N{protocol['validation']['reference_audit_samples']}.npz"],
    }


def validate_heldout(
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run every frozen row and preserve either a pass or a terminal failure."""

    activate()
    protocol = require_v5()
    selection = certify_selection(progress)
    manifest = generate_heldout(progress)
    path = OUTPUT_ROOT / "heldout_validation" / "results.json"
    if path.exists():
        return _read_json(path)
    data = base._heldout_data()
    cfg = base.effective_config()
    context = JaxGalerkinContext(
        cfg, data, DICTIONARY_PATH,
        chunk_size=int(protocol["solver"]["chunk_size"]),
    )
    bank_hashes = _heldout_bank_hashes(manifest)
    forcing_hash = _payload_sha256(cfg["forcing"])
    scientific_hash = _payload_sha256({
        "solver": protocol["solver"],
        "thresholds": protocol["certificate_thresholds"],
        "energy_formula": protocol["energy_formula"],
        "time_nodes": int(cfg["physics"]["time_nodes"]),
    })
    unique: dict[str, dict[str, Any]] = {}
    for selected in selection["rows"]:
        geometry_hash = selected["eta_sha256"]
        if geometry_hash in unique:
            continue
        fit_started = time.perf_counter()
        evaluation = context.evaluate(selected["eta"], gradient=False)
        fit_wall = time.perf_counter() - fit_started
        audit_started = time.perf_counter()
        audit, audit_seconds = context.audit(evaluation.payload)
        audit_wall = time.perf_counter() - audit_started
        tangent_started = time.perf_counter()
        tangent_fit = tangent_audit(data, selected["eta"], use_train=True)
        tangent_holdout = tangent_audit(data, selected["eta"])
        tangent_wall = time.perf_counter() - tangent_started
        cache_fingerprint = {
            "geometry_hash": geometry_hash,
            "role_bank_hash": bank_hashes,
            "N": {
                "fit": int(protocol["validation"]["reference_fit_samples"]),
                "audit": int(protocol["validation"]["reference_audit_samples"]),
            },
            "K": 280,
            "dictionary_hash": protocol["solver"]["dictionary_sha256"],
            "reference_hash": protocol["reference_policy"]["sha256"],
            "forcing_config_hash": forcing_hash,
            "scientific_config_hash": scientific_hash,
        }
        unique[geometry_hash] = {
            "cache_key": _payload_sha256(cache_fingerprint),
            "cache_fingerprint": cache_fingerprint,
            "evaluation": base._public_timed(evaluation),
            "audit": audit,
            "tangent_fit": tangent_fit,
            "tangent_audit": tangent_holdout,
            "timings_seconds": {
                "heldout_fit": fit_wall,
                "heldout_audit_and_final_full_certificates": audit_wall,
                "reported_audit_kernel": audit_seconds,
                "tangent_fit_and_audit": tangent_wall,
            },
        }
        if progress:
            progress(f"V5 held-out unique geometry {geometry_hash} complete")

    law_selected = next(row for row in selection["rows"] if row["method"] == "Law")
    law_risk = float(unique[law_selected["eta_sha256"]]["evaluation"]["risk"])
    rows = []
    for selected in selection["rows"]:
        item = unique[selected["eta_sha256"]]
        evaluation = item["evaluation"]
        audit = item["audit"]
        certificate = audit["heldout_certificate"]
        allowance = selected["allowance_percent"]
        risk = float(evaluation["risk"])
        nominal = bool(
            allowance is None
            or risk <= base.selection_ceiling(law_risk, float(allowance))
        )
        tangent_pass = bool(
            item["tangent_fit"]["valid"] and item["tangent_audit"]["valid"]
        )
        forcing_pass = bool(
            evaluation["train_forcing_audit"]["valid"]
            and audit["audit_forcing"]["valid"]
        )
        thresholds = certificate["thresholds"]
        weak_pass = bool(
            certificate["maximum_weak_residual"]
            <= thresholds["maximum_weak_residual"]
        )
        energy_pass = bool(
            certificate["maximum_energy_residual"]
            <= thresholds["maximum_energy_residual"]
        )
        gauge_pass = bool(
            certificate["maximum_gauge_residual"]
            <= thresholds["maximum_gauge_residual"]
        )
        moment_pass = bool(
            certificate["maximum_moment_rate_residual"]
            <= thresholds["maximum_moment_rate_residual"]
        )
        range_limit = float(cfg["production_galerkin"]["maximum_range_residual"])
        stationarity_limit = float(
            cfg["production_galerkin"]["maximum_stationarity_residual"]
        )
        range_pass = bool(evaluation["worst_range_residual"] <= range_limit)
        stationarity_pass = bool(
            evaluation["worst_stationarity_residual"] <= stationarity_limit
        )
        full_pass = bool(
            audit["valid"] and forcing_pass and weak_pass and energy_pass
            and gauge_pass and moment_pass and range_pass and stationarity_pass
        )
        rows.append({
            "method": selected["method"],
            "allowance_percent": allowance,
            "eta": selected["eta"],
            "eta_sha256": selected["eta_sha256"],
            "heldout_scientific_risk": risk,
            "heldout_relative_risk_increase": risk / law_risk - 1.0,
            "strict_nominal_risk_pass": nominal,
            "heldout_fit_K280_action": float(evaluation["action"]),
            "heldout_audit_K280_action": float(certificate["action"]),
            "tangent_fit_certificate": item["tangent_fit"],
            "tangent_audit_certificate": item["tangent_audit"],
            "heldout_tangent_certificate_pass": tangent_pass,
            "full_train_forcing": evaluation["train_forcing_audit"],
            "full_audit_forcing": audit["audit_forcing"],
            "heldout_full_forcing_pass": forcing_pass,
            "maximum_weak_residual": float(certificate["maximum_weak_residual"]),
            "maximum_energy_residual": float(certificate["maximum_energy_residual"]),
            "maximum_gauge_residual": float(certificate["maximum_gauge_residual"]),
            "maximum_moment_rate_residual": float(certificate["maximum_moment_rate_residual"]),
            "weak_residual_pass": weak_pass,
            "energy_residual_pass": energy_pass,
            "gauge_residual_pass": gauge_pass,
            "moment_rate_residual_pass": moment_pass,
            "maximum_range_residual": float(evaluation["worst_range_residual"]),
            "maximum_stationarity_residual": float(
                evaluation["worst_stationarity_residual"]
            ),
            "range_residual_threshold": range_limit,
            "stationarity_residual_threshold": stationarity_limit,
            "range_residual_pass": range_pass,
            "stationarity_residual_pass": stationarity_pass,
            "heldout_full_certificate_pass": full_pass,
            "full_certificate": certificate,
            "cache_key": item["cache_key"],
            "timings_seconds": item["timings_seconds"],
            "galerkin_backend": "jax",
            "dtype": "float64",
            "K": 280,
        })
        if progress:
            progress(
                f"V5 held-out row {selected['method']} "
                f"{allowance if allowance is not None else 'Law'} "
                f"energy={certificate['maximum_energy_residual']:.9g} "
                f"full={'PASS' if full_pass else 'FAIL'}"
            )
    unchanged = base.payload_sha256(selection["winners"]) == selection["winner_geometry_hash"]
    result = {
        "schema_version": 4,
        "status": "PASS" if (
            unchanged
            and all(row["strict_nominal_risk_pass"] for row in rows)
            and all(row["heldout_tangent_certificate_pass"] for row in rows)
            and all(row["heldout_full_certificate_pass"] for row in rows)
        ) else "FAIL_HELDOUT_VALIDATION",
        "passed": bool(
            unchanged
            and all(row["strict_nominal_risk_pass"] for row in rows)
            and all(row["heldout_tangent_certificate_pass"] for row in rows)
            and all(row["heldout_full_certificate_pass"] for row in rows)
        ),
        "protocol_sha256": protocol["protocol_sha256"],
        "selection_geometry_unchanged": unchanged,
        "optimization_run": False,
        "post_heldout_tuning": False,
        "selection_seal_sha256": _sha256(SELECTION_SEAL_PATH),
        "selection_verification_sha256": _sha256(SELECTION_VERIFICATION_PATH),
        "heldout_manifest_sha256": _sha256(
            OUTPUT_ROOT / "heldout_validation" / "manifest.json"
        ),
        "winner_geometry_hash": selection["winner_geometry_hash"],
        "law_heldout_risk": law_risk,
        "reference_fit_samples": int(protocol["validation"]["reference_fit_samples"]),
        "reference_audit_samples": int(protocol["validation"]["reference_audit_samples"]),
        "rows": rows,
        "unique_geometry_count": len(unique),
        "unique_geometry_receipts": unique,
        "all_galerkin_backends": ["jax"],
        "all_dtypes": ["float64"],
        "K": 280,
    }
    _atomic_json(path, result)
    return result


def _history_unchanged(protocol: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    current = {name: _tree_receipt(root) for name, root in HISTORY_ROOTS.items()}
    old = protocol["history_before_v5"]
    passed = all(
        current[name]["tree_sha256"] == old[name]["tree_sha256"]
        for name in HISTORY_ROOTS
    )
    return passed, current


def _stage_receipts() -> list[dict[str, Any]]:
    root = OUTPUT_ROOT / "performance" / "stages"
    return [_read_json(path) for path in sorted(root.glob("*.json"))]


def _write_csv(rows: list[dict[str, Any]]) -> None:
    flattened = []
    for row in rows:
        flattened.append({
            key: json.dumps(value, sort_keys=True, allow_nan=False)
            if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        })
    fields = list(flattened[0])
    with tempfile.NamedTemporaryFile(
        mode="w", prefix=".final_rows.", suffix=".csv",
        dir=OUTPUT_ROOT, delete=False, newline="", encoding="utf-8",
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)
        handle.flush()
        os.fsync(handle.fileno())
    if FINAL_CSV_PATH.exists():
        temporary.unlink()
    else:
        os.replace(temporary, FINAL_CSV_PATH)


def _write_performance_report(
    stages: list[dict[str, Any]], selection: dict[str, Any],
    validation: dict[str, Any], final_pass: dict[str, Any],
) -> None:
    by_mode = {row["mode"]: row for row in stages}
    bank_manifest = _read_json(OUTPUT_ROOT / "banks" / "manifest.json")
    risk_matrix = _read_json(RISK_MATRIX_PATH)
    peak = max([row.get("jax_process_peak_bytes", 0) for row in stages] + [0])
    total = sum(row["wall_time_seconds"] for row in stages)
    unique = int(selection["unique_geometry_count"])
    selected_count = len(selection["rows"])
    dedup_hits = selected_count - unique
    hit_rate = dedup_hits / selected_count if selected_count else 0.0
    lines = [
        "# Skyrmion Galerkin V5 Performance Report", "",
        "All scientific Galerkin work used JAX float64 K=280; native Galerkin was unreachable.", "",
        "## Stage totals", "",
        "| quantity | value |", "|---|---:|",
        f"| Bank generation wall time | {by_mode.get('generate-data', {}).get('wall_time_seconds', 0.0):.6f} s |",
        f"| Four risk-guard scoring wall time | {risk_matrix['wall_time_seconds']:.6f} s |",
        f"| Full robust search wall time | {by_mode.get('selection', {}).get('wall_time_seconds', 0.0):.6f} s |",
        f"| 131k authoritative certification wall time | {by_mode.get('certify', {}).get('wall_time_seconds', 0.0):.6f} s |",
        f"| Heldout generation wall time | {by_mode.get('heldout-generate', {}).get('wall_time_seconds', 0.0):.6f} s |",
        f"| Heldout certification wall time | {by_mode.get('validation', {}).get('wall_time_seconds', 0.0):.6f} s |",
        f"| Recorded total wall time | {total:.6f} s |",
        f"| Peak JAX process allocation | {peak / 2**20:.3f} MiB |",
        f"| Selected-row geometry cache hit rate | {hit_rate:.9f} |",
        "| Instrumented JIT compile count | unavailable (JAX exposes no stable per-process counter) |",
        "", "## Robust feasible pool", "",
        "| allowance | frozen-pool robust candidates |", "|---:|---:|",
    ]
    for allowance in require_v5()["allowances_percent"]:
        lines.append(
            f"| {allowance:.17g}% | {len(final_pass['robust_feasible_sets'][str(allowance)])} |"
        )
    lines += ["", "## Fresh bank generation detail", "",
        "| role | wall seconds |", "|---|---:|",
    ]
    for role, seconds in bank_manifest["wall_time_seconds"].items():
        lines.append(f"| `{role}` | {float(seconds):.6f} |")
    lines += ["", "## Recorded runner stages", "",
        "| stage | wall seconds | peak JAX MiB |", "|---|---:|---:|",
    ]
    for row in stages:
        lines.append(
            f"| `{row['mode']}` | {row['wall_time_seconds']:.6f} | "
            f"{row.get('jax_process_peak_bytes', 0) / 2**20:.3f} |"
        )
    lines += ["", "## Applied optimizations", "",
        "Risk was computed once per candidate/role and reused for all allowances. "
        "Batch-32 vectorized risk kernels remained resident across the four guard roles. "
        "Only five-role robust-feasible candidates reached K=280 ranking. "
        "The 131k Gram/load statistics used fixed 8,192-sample feature chunks, and selected "
        "Tangent/Full duplicate geometries were certificate-deduplicated by geometry hash.", "",
        f"Heldout unique geometries: `{validation['unique_geometry_count']}` of `{len(validation['rows'])}` reported rows.", "",
    ]
    _atomic_text(PERFORMANCE_REPORT_PATH, "\n".join(lines))


def finalize(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    activate()
    if FINAL_SUMMARY_PATH.exists():
        return _read_json(FINAL_SUMMARY_PATH)
    protocol = require_v5()
    selection = certify_selection(progress)
    verification = verify_selection_v5(selection, progress=progress)
    validation = validate_heldout(progress)
    restart = _read_json(OUTPUT_ROOT / "selection" / "restart_summary.json")
    heldout_manifest = _read_json(OUTPUT_ROOT / "heldout_validation" / "manifest.json")
    law_guard = _read_json(_guard_cache_path(restart["final_law"]["eta"]))
    history_pass, history_after = _history_unchanged(protocol)
    graph = _read_json(CALL_GRAPH_PATH)
    heldout_rows = validation["rows"]
    final_pass = _read_json(
        OUTPUT_ROOT / f"selection_pass_{restart['final_pass_index']}" / "complete.json"
    )
    risk_matrix = _read_json(RISK_MATRIX_PATH)
    cfg = base.effective_config()
    decisions = {
        "A_historical_V1_V4_unchanged": history_pass,
        "B_one_fresh_root_frozen_preoutcome": bool(
            _read_json(FREEZE_MANIFEST_PATH)["freeze_is_pre_outcome"]
            and protocol["single_root_seed"] and protocol["root_seed"] == ROOT_SEED
            and protocol["alternate_root_seeds_tested"] == []
        ),
        "C_native_Galerkin_unreachable": bool(graph["passed"]),
        "D_JAX_float64_K280_everywhere": bool(
            selection["all_galerkin_backends"] == ["jax"]
            and selection["all_scientific_action_dtypes"] == ["float64"]
            and validation["all_galerkin_backends"] == ["jax"]
            and validation["all_dtypes"] == ["float64"] and validation["K"] == 280
        ),
        "E_K_rank_rules_unchanged": bool(
            protocol["solver"]["K"] == 280
            and protocol["solver"]["relative_rank_tolerance"] == 1e-12
        ),
        "F_energy_formula_threshold_unchanged": bool(
            protocol["energy_formula"]["numerator"] == "abs(a_fit^T K_audit a_fit + a_fit^T f_audit)"
            and protocol["energy_formula"]["denominator"] == "max(a_fit^T K_audit a_fit + abs(a_fit^T f_audit), 1e-12)"
            and protocol["energy_formula"]["threshold"] == 0.08
        ),
        "G_range_stationarity_thresholds_1e8": bool(
            cfg["production_galerkin"]["maximum_range_residual"] == 1e-8
            and cfg["production_galerkin"]["maximum_stationarity_residual"] == 1e-8
        ),
        "H_exactly_four_preseal_risk_guards": len(RISK_GUARD_COUNTS) == 4,
        "I_risk_guard_truth_N5000": all(row["truth"] == 5000 for row in RISK_GUARD_COUNTS.values()),
        "J_risk_guard_reference_N65536": all(row["reference"] == 65536 for row in RISK_GUARD_COUNTS.values()),
        "K_all_candidate_feasibility_used_five_roles": bool(
            risk_matrix["one_evaluation_per_candidate_role"]
            and risk_matrix["risk_roles"] == list(RISK_ROLE_NAMES)
            and all(set(row["risk_by_role"]) == set(RISK_ROLE_NAMES) for row in risk_matrix["rows"])
        ),
        "L_role_specific_Law_baseline_used": all(
            abs(row["relative_risk_by_role"][role] - (
                row["risk_by_role"][role] / selection["law_risk_by_role"][role] - 1.0
            )) <= 1e-15
            for row in selection["rows"] for role in RISK_ROLE_NAMES
        ),
        "M_Law_support_robust_procedure_passed": bool(law_guard["support_robust"]),
        "N_Law_consistency_reanchor_passed": bool(restart["passed"] and restart["final_law_consistent"]),
        "O_only_allowances_0p5_1_2": protocol["allowances_percent"] == [0.5, 1.0, 2.0],
        "P_Law_mandatory_Full_baseline_0p5": bool(final_pass["full"][0]["law_mandatory"]),
        "Q_previous_Full_incumbent_mandatory_1_2": all(row["previous_incumbent_mandatory"] for row in final_pass["full"][1:]),
        "R_authoritative_fit_N131072": all(row["authoritative_fit_samples"] == 131072 for row in selection["rows"]),
        "S_authoritative_audit_N131072": all(row["authoritative_audit_samples"] == 131072 for row in selection["rows"]),
        "T_sealed_range_le_1e8": all(row["maximum_range_residual"] <= 1e-8 for row in selection["rows"]),
        "U_sealed_stationarity_le_1e8": all(row["maximum_stationarity_residual"] <= 1e-8 for row in selection["rows"]),
        "V_sealed_energy_le_0p08": all(row["maximum_energy_residual"] <= 0.08 for row in selection["rows"]),
        "W_selection_sealed_before_heldout": bool(
            verification["passed"] and heldout_manifest["generated_after_selection_freeze"]
            and heldout_manifest["selection_seal_sha256"] == _sha256(SELECTION_SEAL_PATH)
        ),
        "X_heldout_fit_audit_N131072": bool(
            validation["reference_fit_samples"] == 131072
            and validation["reference_audit_samples"] == 131072
        ),
        "Y_heldout_nominal_risk_all_pass": all(row["strict_nominal_risk_pass"] for row in heldout_rows),
        "Z_heldout_Full_certificates_all_pass": all(row["heldout_full_certificate_pass"] for row in heldout_rows),
        "AA_no_postheldout_tuning": bool(not validation["optimization_run"] and not validation["post_heldout_tuning"]),
    }
    decisions["AB_V5_SINGLE_SEED_K280_ROBUST_PARETO_AUTHORITY"] = all(decisions.values())
    stages = _stage_receipts()
    _write_performance_report(stages, selection, validation, final_pass)
    combined_rows = []
    heldout_by_position = list(heldout_rows)
    for selected, heldout in zip(selection["rows"], heldout_by_position, strict=True):
        combined_rows.append({**selected, **{
            key: value for key, value in heldout.items()
            if key not in {"method", "allowance_percent", "eta", "eta_sha256"}
        }})
    _write_csv(combined_rows)

    status = (
        "PASS" if decisions["AB_V5_SINGLE_SEED_K280_ROBUST_PARETO_AUTHORITY"]
        else validation["status"] if not validation["passed"]
        else "FAIL_FINAL_DECISION_TABLE"
    )
    labels = [
        ("A", "Historical V1--V4 authorities unchanged?", "A_historical_V1_V4_unchanged"),
        ("B", "One fresh V5 root frozen before outcomes?", "B_one_fresh_root_frozen_preoutcome"),
        ("C", "Native Galerkin unreachable?", "C_native_Galerkin_unreachable"),
        ("D", "JAX float64 K=280 everywhere?", "D_JAX_float64_K280_everywhere"),
        ("E", "K/rank rules unchanged?", "E_K_rank_rules_unchanged"),
        ("F", "Energy formula and 0.08 threshold unchanged?", "F_energy_formula_threshold_unchanged"),
        ("G", "Range/stationarity thresholds both 1e-8?", "G_range_stationarity_thresholds_1e8"),
        ("H", "Exactly four pre-seal risk guards?", "H_exactly_four_preseal_risk_guards"),
        ("I", "Risk guard truth N exactly 5,000?", "I_risk_guard_truth_N5000"),
        ("J", "Risk guard reference N exactly 65,536?", "J_risk_guard_reference_N65536"),
        ("K", "Candidate feasibility used selection plus all guards?", "K_all_candidate_feasibility_used_five_roles"),
        ("L", "Role-specific Law baseline used on every guard?", "L_role_specific_Law_baseline_used"),
        ("M", "Law support-robust procedure passed?", "M_Law_support_robust_procedure_passed"),
        ("N", "Law consistency/reanchor gate passed?", "N_Law_consistency_reanchor_passed"),
        ("O", "Only 0.5%, 1%, 2% run?", "O_only_allowances_0p5_1_2"),
        ("P", "Law mandatory Full baseline at 0.5%?", "P_Law_mandatory_Full_baseline_0p5"),
        ("Q", "Previous Full incumbent mandatory at 1%/2%?", "Q_previous_Full_incumbent_mandatory_1_2"),
        ("R", "Authoritative certificate fit N=131,072?", "R_authoritative_fit_N131072"),
        ("S", "Authoritative certificate audit N=131,072?", "S_authoritative_audit_N131072"),
        ("T", "Range certificate <=1e-8 for every sealed row?", "T_sealed_range_le_1e8"),
        ("U", "Stationarity certificate <=1e-8 for every sealed row?", "U_sealed_stationarity_le_1e8"),
        ("V", "Energy certificate <=0.08 for every sealed row?", "V_sealed_energy_le_0p08"),
        ("W", "Selection sealed and verified before heldout?", "W_selection_sealed_before_heldout"),
        ("X", "Heldout fit/audit exactly 131,072/131,072?", "X_heldout_fit_audit_N131072"),
        ("Y", "Heldout nominal risk passed every row?", "Y_heldout_nominal_risk_all_pass"),
        ("Z", "Heldout Full certificates passed every row?", "Z_heldout_Full_certificates_all_pass"),
        ("AA", "No post-heldout tuning?", "AA_no_postheldout_tuning"),
        ("AB", "V5 SINGLE-SEED K280 ROBUST PARETO AUTHORITY", "AB_V5_SINGLE_SEED_K280_ROBUST_PARETO_AUTHORITY"),
    ]
    lines = [
        "# Official B1 Galerkin Pareto V5 Final Result", "",
        f"Overall status: **{status}**", "",
        "V5 is a prospective finite-sample precision repair. It retains the fixed JAX float64 K=280 estimand and makes no continuum-convergence claim.",
        "",
        f"Root seed: `{ROOT_SEED}` (no alternate root tested)",
        f"Protocol SHA-256: `{protocol['protocol_sha256']}`",
        f"Selected Law: `{restart['final_law']['candidate_id']}` / `{base.eta_key(restart['final_law']['eta'])}`",
        f"Law exact risk: `{float(restart['final_law']['exact_scientific_risk']):.17g}`",
        f"Law minimum four-guard rESS: `{float(law_guard['minimum_guard_rESS']):.17g}`",
        "", "## Authoritative selection and held-out results", "",
        "| method | allowance | geometry | exact risk | max five-role rel. risk | selection train | selection audit | held-out risk | held-out fit | held-out audit | range | stationarity | weak | energy | gauge | moment | Tangent | Full |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in combined_rows:
        allowance = "—" if row["allowance_percent"] is None else f"{row['allowance_percent']:.17g}%"
        lines.append(
            f"| {row['method']} | {allowance} | `{row['eta_sha256']}` | "
            f"{row['exact_risk']:.17g} | {row['maximum_relative_risk_increase']:.17g} | "
            f"{row['train_K280_action']:.17g} | {row['audit_K280_action']:.17g} | "
            f"{row['heldout_scientific_risk']:.17g} | {row['heldout_fit_K280_action']:.17g} | "
            f"{row['heldout_audit_K280_action']:.17g} | {row['maximum_range_residual']:.17g} | "
            f"{row['maximum_stationarity_residual']:.17g} | {row['maximum_weak_residual']:.17g} | "
            f"{row['maximum_energy_residual']:.17g} | {row['maximum_gauge_residual']:.17g} | "
            f"{row['maximum_moment_rate_residual']:.17g} | "
            f"{'PASS' if row['heldout_tangent_certificate_pass'] else 'FAIL'} | "
            f"{'PASS' if row['heldout_full_certificate_pass'] else 'FAIL'} |"
        )
    lines += ["", "## Four fresh Law guards", "",
        "| role | N | rESS | projection | forcing mean | covariance condition | result |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for role, item in law_guard["support_by_fresh_guard_role"].items():
        lines.append(
            f"| `{role}` | {GUARD_COUNTS[role]} | {item['minimum_rESS']:.17g} | "
            f"{item['maximum_projection_residual']:.17g} | {item['maximum_forcing_mean']:.17g} | "
            f"{item['maximum_covariance_condition']:.17g} | {'PASS' if item['support_valid'] else 'FAIL'} |"
        )
    lines += ["", "## Performance timings", "",
        "| stage | wall seconds | JAX process peak bytes |", "|---|---:|---:|",
    ]
    for row in stages:
        lines.append(
            f"| `{row['mode']}` | {row['wall_time_seconds']:.17g} | {row.get('jax_process_peak_bytes', 0)} |"
        )
    lines += ["", "Unique held-out geometry timings are retained at full precision in `heldout_validation/results.json`.",
        "", "## Final decision table", "", "| item | decision |", "|---|:---:|",
    ]
    for letter, label, key in labels:
        lines.append(f"| {letter}. {label} | **{'PASS' if decisions[key] else 'FAIL'}** |")
    lines += ["", "No post-heldout tuning, seed switch, bank regeneration, K/N change, or threshold change was performed.", ""]
    _atomic_text(FINAL_REPORT_PATH, "\n".join(lines))

    summary = {
        "schema_version": 4,
        "status": status,
        "passed": decisions["AB_V5_SINGLE_SEED_K280_ROBUST_PARETO_AUTHORITY"],
        "authority": VERSION,
        "root_seed": ROOT_SEED,
        "alternate_root_seeds_tested": [],
        "protocol_sha256": protocol["protocol_sha256"],
        "full_estimand": protocol["estimand"],
        "continuum_convergence_claim": False,
        "restart_summary": restart,
        "selected_law": restart["final_law"],
        "law_guard_receipt": law_guard,
        "selection_seal_sha256": _sha256(SELECTION_SEAL_PATH),
        "selection_verification_sha256": _sha256(SELECTION_VERIFICATION_PATH),
        "heldout_results_sha256": _sha256(
            OUTPUT_ROOT / "heldout_validation" / "results.json"
        ),
        "rows": combined_rows,
        "performance_stages": stages,
        "history_after_v5": history_after,
        "decision_table": decisions,
        "post_heldout_tuning": False,
    }
    _atomic_json(FINAL_SUMMARY_PATH, summary)
    inventory_files = [
        path for path in sorted(OUTPUT_ROOT.rglob("*"))
        if path.is_file() and path != INVENTORY_PATH
        and "jax_compilation_cache" not in path.parts
    ]
    inventory = {
        "schema_version": 4,
        "status": "COMPLETE",
        "protocol_sha256": protocol["protocol_sha256"],
        "self_excluded": str(INVENTORY_PATH.relative_to(OUTPUT_ROOT)),
        "file_count": len(inventory_files),
        "artifacts": [{
            "path": str(path.relative_to(OUTPUT_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        } for path in inventory_files],
    }
    _atomic_json(INVENTORY_PATH, inventory)
    if progress:
        progress(f"V5 terminal result: {status}")
    return summary


def write_terminal_failure(
    stage: str, error: BaseException,
) -> dict[str, Any]:
    """Fail closed for a pre-heldout or operationally terminal V5 stage."""

    activate()
    protocol = require_v5()
    payload = {
        "schema_version": 4,
        "status": f"FAIL_{stage.upper().replace('-', '_')}",
        "passed": False,
        "authority": VERSION,
        "root_seed": ROOT_SEED,
        "alternate_root_seeds_tested": [],
        "protocol_sha256": protocol["protocol_sha256"],
        "failure": {
            "stage": stage,
            "exception_type": type(error).__name__,
            "message": str(error),
        },
        "selection_sealed": SELECTION_SEAL_PATH.exists(),
        "heldout_generated": (
            OUTPUT_ROOT / "heldout_validation" / "manifest.json"
        ).exists(),
        "post_heldout_tuning": False,
    }
    if not FINAL_SUMMARY_PATH.exists():
        _atomic_json(FINAL_SUMMARY_PATH, payload)
    if not FINAL_REPORT_PATH.exists():
        _atomic_text(FINAL_REPORT_PATH, "\n".join((
            "# Official B1 Galerkin Pareto V5 Final Result", "",
            f"Overall status: **{payload['status']}**", "",
            f"The single-root authority failed closed during `{stage}`.",
            f"Failure: `{type(error).__name__}: {error}`",
            f"Selection sealed: `{str(payload['selection_sealed']).lower()}`.",
            f"Held-out generated: `{str(payload['heldout_generated']).lower()}`.",
            "No alternate root or post-result tuning was used.", "",
        )))
    return payload


__all__ = [
    "freeze_v5", "require_v5", "generate_data", "score_candidate_universe",
    "refreeze_law", "run_selection", "certify_selection", "generate_heldout",
    "validate_heldout", "finalize", "write_terminal_failure", "guard_worker",
    "activate", "chunked_many_body_features",
]
