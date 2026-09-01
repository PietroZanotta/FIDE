"""Prospective single-root-seed Skyrmion B1 K=280 JAX authority (V4).

V4 reuses the corrected V3.4 selection and support-robust Law algorithms but
regenerates every stochastic role from a new root.  Native Galerkin is not an
option on this execution path.
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
VERSION = "official_b1_galerkin_pareto_v4_single_seed"
OUTPUT_ROOT = ROOT / "outputs" / VERSION
CONFIG_PATH = ROOT / "config_v4_single_seed.json"
PROTOCOL_DOCUMENT = ROOT / "OFFICIAL_B1_GALERKIN_PARETO_V4_PROTOCOL.md"
RUNNER_PATH = ROOT / "official_b1_pareto_v4_single_seed_run.py"
TEST_PATH = ROOT / "test_official_b1_pareto_v4_single_seed.py"
SOURCE_PATH = Path(__file__)

PROTOCOL_PATH = OUTPUT_ROOT / "protocol_v4.json"
FREEZE_MANIFEST_PATH = OUTPUT_ROOT / "freeze_manifest_v4.json"
RANDOMNESS_PATH = OUTPUT_ROOT / "V4_RANDOMNESS_PROVENANCE.md"
CALL_GRAPH_PATH = OUTPUT_ROOT / "jax_only_call_graph_v4.json"
EFFECTIVE_CONFIG_PATH = OUTPUT_ROOT / "effective_config.json"
CANDIDATE_POOL_PATH = OUTPUT_ROOT / "candidate_pool" / "candidate_pool.json"
SCIENTIFIC_ARRAYS_PATH = OUTPUT_ROOT / "feasibility" / "exact_receipts.npz"
SCIENTIFIC_ROWS_PATH = OUTPUT_ROOT / "feasibility" / "exact_receipts.json"
LAW_PATH = OUTPUT_ROOT / "law" / "initial_law.json"
LAW_GUARD_SUMMARY_PATH = OUTPUT_ROOT / "law" / "guard_screen.json"
SELECTION_SEAL_PATH = OUTPUT_ROOT / "selection" / "selection_seal.json"
SELECTION_VERIFICATION_PATH = OUTPUT_ROOT / "selection" / "independent_verification.json"
FINAL_REPORT_PATH = OUTPUT_ROOT / "OFFICIAL_B1_GALERKIN_PARETO_V4_FINAL_RESULT.md"
FINAL_SUMMARY_PATH = OUTPUT_ROOT / "final_summary.json"
FINAL_CSV_PATH = OUTPUT_ROOT / "final_rows.csv"
INVENTORY_PATH = OUTPUT_ROOT / "terminal_inventory.json"

V1_ROOT = ROOT / "outputs" / "official_b1_galerkin_pareto_v1"
V2_ROOT = ROOT / "outputs" / "old_stuff" / "official_b1_galerkin_pareto_v2_single_seed"
V2_1_ROOT = ROOT / "outputs" / "official_b1_galerkin_pareto_v2_1_single_seed_amended"
V3_ROOT = ROOT / "outputs" / "official_b1_galerkin_pareto_v3_support_robust_single_seed"
DEVELOPMENT_ROOT = ROOT / "outputs" / "development_v4_energy_diagnosis"
HISTORY_ROOTS = {
    "V1_authority": V1_ROOT,
    "V2_authority": V2_ROOT,
    "V2_1_authority": V2_1_ROOT,
    "V3_4_authority": V3_ROOT,
    "V3_4_development_diagnosis": DEVELOPMENT_ROOT,
}
V3_TERMINAL_PATH = V3_ROOT / "OFFICIAL_B1_GALERKIN_PARETO_V3_4_TERMINAL_RESULT.md"
DEVELOPMENT_REPORT_PATH = DEVELOPMENT_ROOT / "SKYRMION_V3_4_ENERGY_FAILURE_DIAGNOSIS.md"

REFERENCE_SOURCE = V1_ROOT / "artifacts" / "reference.npz"
DICTIONARY_SOURCE = (
    ROOT / "outputs" / "galerkin_only_3pct" / "cache" / "dictionaries"
    / "dictionary_K280.npz"
)
REFERENCE_PATH = OUTPUT_ROOT / "artifacts" / "reference.npz"
DICTIONARY_PATH = OUTPUT_ROOT / "artifacts" / "dictionary_K280.npz"

ROOT_SEED = 20261004
GUARD_BLOCK_SIZE = 8
FEATURE_SAMPLE_CHUNK = 8192
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
ROLE_IDS = {**STANDARD_ROLE_IDS, **GUARD_ROLE_IDS}

ORIGINAL_BASE_GENERATE_DATA = base.generate_data
ORIGINAL_MANY_BODY_FEATURES = base.many_body_features


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
        raise ValueError(f"V4 output escaped {root}: {resolved}")
    if path.exists():
        if path.read_bytes() == data:
            return
        if immutable:
            raise RuntimeError(f"refusing to overwrite sealed V4 artifact: {path}")
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
        raise KeyError(f"unfrozen V4 stochastic role: {role}")
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
        "PERFORMANCE_REPORT_PATH": OUTPUT_ROOT / "V4_JAX_PERFORMANCE_REPORT.md",
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
    base.require_protocol = require_v4
    base.generate_data = generate_data
    base.many_body_features = chunked_many_body_features
    base._select_starts = v3._amended_select_starts
    base.run_selection_with_restarts = v3.run_selection_with_restarts
    base.verify_frozen_selection = v3.verify_frozen_selection
    v3.require_v3 = require_v4
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
            ["study.run_selection", "v3.run_selection_with_restarts"],
            ["v3.run_selection_with_restarts", "base.run_selection_pass"],
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
        "prior_official_roots": [20261001, 20261002, 20261003],
        "selection_rule": "first unused integer after the largest prior official root",
        "selected_root": ROOT_SEED,
        "preexisting_repository_matches_excluding_V4_declarations_and_outputs": matches,
        "relevant_git_history_matches": git,
        "passed": not matches and not git,
    }


def freeze_v4(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    if PROTOCOL_PATH.exists():
        activate()
        return require_v4()
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.rglob("*")):
        raise RuntimeError("V4 output root is not empty before freeze")
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("V4 freeze requires JAX x64")
    if K != 280:
        raise RuntimeError("V4 K changed")
    values = _values()
    if values["root_seed"] != ROOT_SEED or values["allowances_percent"] != [0.5, 1.0, 2.0]:
        raise RuntimeError("V4 seed or allowances changed")
    if values["validation"]["reference_fit_samples"] != 65536 or values["validation"]["reference_audit_samples"] != 65536:
        raise RuntimeError("V4 held-out counts changed")
    if not all(root.is_dir() for root in HISTORY_ROOTS.values()):
        raise RuntimeError("historical authority root missing")
    history = {name: _tree_receipt(root) for name, root in HISTORY_ROOTS.items()}
    history["V3_4_terminal_result"] = {
        "path": str(V3_TERMINAL_PATH.relative_to(REPO_ROOT)),
        "sha256": _sha256(V3_TERMINAL_PATH),
    }
    history["V3_4_development_report"] = {
        "path": str(DEVELOPMENT_REPORT_PATH.relative_to(REPO_ROOT)),
        "sha256": _sha256(DEVELOPMENT_REPORT_PATH),
    }
    seed_evidence = _seed_search_evidence()
    if not seed_evidence["passed"]:
        raise RuntimeError(f"V4 root seed was already used: {seed_evidence}")
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
        "status": "FROZEN_BEFORE_V4_SCIENTIFIC_OUTCOMES",
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
        "history_before_v4": history,
        "source_hashes": _source_hashes(),
        "protocol_document_sha256": _sha256(PROTOCOL_DOCUMENT),
        "final_heldout_access": "only after selection seal and passing independent reconstruction",
        "post_heldout_tuning": False,
    }
    digest = _payload_sha256(protocol_body)
    protocol = {**protocol_body, "protocol_sha256": digest}
    freeze_manifest = {
        "schema_version": 4,
        "status": "FROZEN_BEFORE_V4_SCIENTIFIC_OUTCOMES",
        "frozen_at_utc": protocol_body["frozen_at_utc"],
        "protocol_sha256": digest,
        "freeze_is_pre_outcome": True,
        "scientific_outcomes_present_before_freeze": False,
        "root_seed": ROOT_SEED,
        "candidate_pool_rows_sha256": _payload_sha256(rows),
        "reference_sha256": reference_hash,
        "dictionary_sha256": dictionary_hash,
        "history_before_v4": history,
        "source_hashes": protocol_body["source_hashes"],
        "jax_only_call_graph_sha256": _payload_sha256(graph),
    }
    lines = [
        "# V4 Randomness Provenance", "",
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
        progress(f"V4 protocol frozen before outcomes: {digest}")
    return protocol


def require_v4() -> dict[str, Any]:
    required = (
        PROTOCOL_PATH, FREEZE_MANIFEST_PATH, RANDOMNESS_PATH, CALL_GRAPH_PATH,
        EFFECTIVE_CONFIG_PATH, CANDIDATE_POOL_PATH, REFERENCE_PATH, DICTIONARY_PATH,
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError("V4 protocol is not fully frozen")
    protocol = _read_json(PROTOCOL_PATH)
    body = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if _payload_sha256(body) != protocol["protocol_sha256"]:
        raise RuntimeError("V4 protocol hash mismatch")
    if _read_json(FREEZE_MANIFEST_PATH)["protocol_sha256"] != protocol["protocol_sha256"]:
        raise RuntimeError("V4 freeze manifest mismatch")
    if _source_hashes() != protocol["source_hashes"]:
        raise RuntimeError("V4 scientific source changed after freeze")
    graph = _read_json(CALL_GRAPH_PATH)
    if not graph["passed"] or _payload_sha256(graph) != _read_json(FREEZE_MANIFEST_PATH)["jax_only_call_graph_sha256"]:
        raise RuntimeError("V4 JAX-only call graph mismatch")
    if _sha256(REFERENCE_PATH) != protocol["reference_policy"]["sha256"]:
        raise RuntimeError("V4 reference changed")
    if _sha256(DICTIONARY_PATH) != protocol["solver"]["dictionary_sha256"]:
        raise RuntimeError("V4 dictionary changed")
    pool = _read_json(CANDIDATE_POOL_PATH)
    if _payload_sha256(pool["rows"]) != protocol["candidate_universe"]["rows_sha256"]:
        raise RuntimeError("V4 candidate universe changed")
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
            raise RuntimeError(f"V4 guard checkpoint mismatch: {label}")
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
    """Generate all fresh selection and Law-guard banks from the V4 root."""

    protocol = require_v4()
    manifest_path = OUTPUT_ROOT / "banks" / "manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        for row in manifest["artifacts"]:
            if _sha256(OUTPUT_ROOT / row["path"]) != row["sha256"]:
                raise RuntimeError(f"V4 data artifact changed: {row['path']}")
        return manifest

    # The V2 generator is retained only as the fresh standard-role generator;
    # all of its globals have been rebound to the sealed V4 protocol and root.
    standard_manifest = ORIGINAL_BASE_GENERATE_DATA(progress)
    standard_manifest_path = OUTPUT_ROOT / "banks" / "manifest.json"
    provenance_path = OUTPUT_ROOT / "provenance" / "standard_bank_manifest.json"
    if provenance_path.exists():
        if _read_json(provenance_path) != standard_manifest:
            raise RuntimeError("V4 standard-bank provenance mismatch")
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
            progress(f"V4 fresh guard bank {label} N={count}")

    initial_hashes = dict(standard_manifest["initial_state_hashes"])
    initial_hashes.update({
        row["label"]: row["initial_state_sha256"] for row in guard_records
    })
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
    distinct = len(set(initial_hashes.values())) == len(initial_hashes)
    manifest = {
        "schema_version": 4,
        "passed": distinct,
        "protocol_sha256": protocol["protocol_sha256"],
        "v3_protocol_sha256": protocol["protocol_sha256"],
        "root_seed": ROOT_SEED,
        "derived_roles_are_not_replicates": True,
        "inherited_roles": [],
        "fresh_roles": list(protocol["bank_sizes"]) + list(GUARD_COUNTS),
        "initial_state_hashes": initial_hashes,
        "role_disjoint": distinct,
        "reference_retrained": False,
        "reference_sha256": _sha256(REFERENCE_PATH),
        "wall_time_seconds": {
            **standard_manifest["wall_time_seconds"],
            **{row["label"]: row["wall_time_seconds"] for row in guard_records},
        },
        "artifacts": artifacts,
    }
    _atomic_json(manifest_path, manifest)
    if not distinct:
        raise RuntimeError("V4 fresh stochastic roles overlap")
    return manifest


def score_candidate_universe(
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    require_v4()
    generate_data(progress)
    return base.score_candidate_universe(progress)


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
    require_v4()
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
                        "experiments.skyrmions_galerkin.official_b1_pareto_v4_single_seed_run",
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
                progress(f"V4 isolated Law guard {role} sealed")
        protocol = require_v4()
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
            progress(f"V4 guard-qualified {len(missing)} candidate rows")
    return [cached[base.eta_key(row["eta"])] for row in rows]


def refreeze_law(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    activate()
    score_candidate_universe(progress)
    return v3.refreeze_law(progress)


def run_selection(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    activate()
    refreeze_law(progress)
    return v3.run_selection_with_restarts(progress)


def certify_selection(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    activate()
    return base.certify_and_freeze_selection(progress)


def generate_heldout(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    activate()
    return base.generate_heldout(progress)


def _heldout_bank_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    hashes = {Path(row["path"]).name: row["sha256"] for row in manifest["artifacts"]}
    protocol = require_v4()
    return {
        "fit": hashes[f"reference_fit_N{protocol['validation']['reference_fit_samples']}.npz"],
        "audit": hashes[f"reference_audit_N{protocol['validation']['reference_audit_samples']}.npz"],
    }


def validate_heldout(
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run every frozen row and preserve either a pass or a terminal failure."""

    activate()
    protocol = require_v4()
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
            progress(f"V4 held-out unique geometry {geometry_hash} complete")

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
        full_pass = bool(
            audit["valid"] and forcing_pass and weak_pass and energy_pass
            and gauge_pass and moment_pass
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
                f"V4 held-out row {selected['method']} "
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
    old = protocol["history_before_v4"]
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


def finalize(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    activate()
    if FINAL_SUMMARY_PATH.exists():
        return _read_json(FINAL_SUMMARY_PATH)
    protocol = require_v4()
    selection = certify_selection(progress)
    verification = v3.verify_frozen_selection(selection, progress=progress)
    validation = validate_heldout(progress)
    restart = _read_json(OUTPUT_ROOT / "selection" / "restart_summary.json")
    heldout_manifest = _read_json(OUTPUT_ROOT / "heldout_validation" / "manifest.json")
    law_guard = _read_json(_guard_cache_path(restart["final_law"]["eta"]))
    history_pass, history_after = _history_unchanged(protocol)
    graph = _read_json(CALL_GRAPH_PATH)
    heldout_rows = validation["rows"]
    decisions = {
        "A_V3_4_preserved_unchanged": history_pass,
        "B_V4_frozen_before_new_outcomes": bool(
            _read_json(FREEZE_MANIFEST_PATH)["freeze_is_pre_outcome"]
        ),
        "C_exactly_one_V4_root_seed": bool(
            protocol["single_root_seed"] and protocol["root_seed"] == ROOT_SEED
            and protocol["alternate_root_seeds_tested"] == []
        ),
        "D_native_Galerkin_unreachable": bool(graph["passed"]),
        "E_JAX_float64_K280_everywhere": bool(
            selection["all_galerkin_backends"] == ["jax"]
            and selection["all_scientific_action_dtypes"] == ["float64"]
            and validation["all_galerkin_backends"] == ["jax"]
            and validation["all_dtypes"] == ["float64"] and validation["K"] == 280
        ),
        "F_rank_rule_unchanged": bool(
            protocol["solver"]["relative_rank_tolerance"] == 1e-12
        ),
        "G_energy_formula_unchanged": bool(
            protocol["energy_formula"]["numerator"]
            == "abs(a_fit^T K_audit a_fit + a_fit^T f_audit)"
            and protocol["energy_formula"]["denominator"]
            == "max(a_fit^T K_audit a_fit + abs(a_fit^T f_audit), 1e-12)"
        ),
        "H_energy_threshold_0p08": protocol["energy_formula"]["threshold"] == 0.08,
        "I_Law_support_robust_rule_unchanged": bool(
            protocol["law_support_rule"]["unchanged_from"].startswith("V3.4")
            and law_guard["support_robust"]
        ),
        "J_Law_consistency_reanchor_gate_passed": bool(
            restart["passed"] and restart["final_law_consistent"]
        ),
        "K_only_allowances_0p5_1_2": protocol["allowances_percent"] == [0.5, 1.0, 2.0],
        "L_selection_sealed_before_heldout_generation": bool(
            heldout_manifest["generated_after_selection_freeze"]
            and heldout_manifest["selection_seal_sha256"] == _sha256(SELECTION_SEAL_PATH)
        ),
        "M_independent_reconstruction_passed": bool(verification["passed"]),
        "N_heldout_fit_count_65536": validation["reference_fit_samples"] == 65536,
        "O_heldout_audit_count_65536": validation["reference_audit_samples"] == 65536,
        "P_heldout_nominal_risk_all_pass": all(
            row["strict_nominal_risk_pass"] for row in heldout_rows
        ),
        "Q_heldout_Tangent_all_pass": all(
            row["heldout_tangent_certificate_pass"] for row in heldout_rows
        ),
        "R_heldout_Full_weak_gauge_moment_all_pass": all(
            row["weak_residual_pass"] and row["gauge_residual_pass"]
            and row["moment_rate_residual_pass"] for row in heldout_rows
        ),
        "S_heldout_Full_energy_all_le_0p08": all(
            row["energy_residual_pass"] for row in heldout_rows
        ),
        "T_Full_Pareto_action_nested": bool(selection["full_action_nonincreasing"]),
        "U_no_postheldout_tuning": bool(
            not validation["optimization_run"] and not validation["post_heldout_tuning"]
        ),
    }
    decisions["V_V4_SINGLE_SEED_K280_AUTHORITY"] = all(decisions.values())
    stages = _stage_receipts()
    combined_rows = []
    heldout_by_position = list(heldout_rows)
    for selected, heldout in zip(selection["rows"], heldout_by_position, strict=True):
        combined_rows.append({**selected, **{
            key: value for key, value in heldout.items()
            if key not in {"method", "allowance_percent", "eta", "eta_sha256"}
        }})
    _write_csv(combined_rows)

    status = "PASS" if decisions["V_V4_SINGLE_SEED_K280_AUTHORITY"] else validation["status"]
    labels = [
        ("A", "V3.4 preserved unchanged?", "A_V3_4_preserved_unchanged"),
        ("B", "V4 frozen before new outcomes?", "B_V4_frozen_before_new_outcomes"),
        ("C", "Exactly one V4 root seed?", "C_exactly_one_V4_root_seed"),
        ("D", "Native Galerkin unreachable?", "D_native_Galerkin_unreachable"),
        ("E", "JAX float64 K=280 used everywhere?", "E_JAX_float64_K280_everywhere"),
        ("F", "Rank rule unchanged?", "F_rank_rule_unchanged"),
        ("G", "Energy formula unchanged?", "G_energy_formula_unchanged"),
        ("H", "Energy threshold unchanged at 0.08?", "H_energy_threshold_0p08"),
        ("I", "Law support-robust rule unchanged?", "I_Law_support_robust_rule_unchanged"),
        ("J", "Law consistency/reanchor gate passed?", "J_Law_consistency_reanchor_gate_passed"),
        ("K", "Only allowances 0.5%, 1%, 2% run?", "K_only_allowances_0p5_1_2"),
        ("L", "Selection sealed before held-out generation?", "L_selection_sealed_before_heldout_generation"),
        ("M", "Independent reconstruction passed?", "M_independent_reconstruction_passed"),
        ("N", "Held-out fit count exactly 65,536?", "N_heldout_fit_count_65536"),
        ("O", "Held-out audit count exactly 65,536?", "O_heldout_audit_count_65536"),
        ("P", "Held-out nominal risk passed for all rows?", "P_heldout_nominal_risk_all_pass"),
        ("Q", "Held-out Tangent certificates passed?", "Q_heldout_Tangent_all_pass"),
        ("R", "Held-out Full weak/gauge/moment certificates passed?", "R_heldout_Full_weak_gauge_moment_all_pass"),
        ("S", "Held-out Full energy certificate <=0.08 for every row?", "S_heldout_Full_energy_all_le_0p08"),
        ("T", "Full Pareto action nested?", "T_Full_Pareto_action_nested"),
        ("U", "No post-heldout tuning?", "U_no_postheldout_tuning"),
        ("V", "V4 SINGLE-SEED K280 AUTHORITY", "V_V4_SINGLE_SEED_K280_AUTHORITY"),
    ]
    lines = [
        "# Official B1 Galerkin Pareto V4 Final Result", "",
        f"Overall status: **{status}**", "",
        "V4 is a prospective finite-sample precision repair. It retains the fixed JAX float64 K=280 estimand and makes no continuum-convergence claim.",
        "",
        f"Root seed: `{ROOT_SEED}` (no alternate root tested)",
        f"Protocol SHA-256: `{protocol['protocol_sha256']}`",
        f"Selected Law: `{restart['final_law']['candidate_id']}` / `{base.eta_key(restart['final_law']['eta'])}`",
        f"Law exact risk: `{float(restart['final_law']['exact_scientific_risk']):.17g}`",
        f"Law minimum four-guard rESS: `{float(law_guard['minimum_guard_rESS']):.17g}`",
        "", "## Authoritative selection and held-out results", "",
        "| method | allowance | geometry | exact risk | rel. risk | selection train | selection audit | held-out risk | held-out fit | held-out audit | weak | energy | gauge | moment | Tangent | Full |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in combined_rows:
        allowance = "—" if row["allowance_percent"] is None else f"{row['allowance_percent']:.17g}%"
        lines.append(
            f"| {row['method']} | {allowance} | `{row['eta_sha256']}` | "
            f"{row['exact_risk']:.17g} | {row['relative_risk_increase']:.17g} | "
            f"{row['train_K280_action']:.17g} | {row['audit_K280_action']:.17g} | "
            f"{row['heldout_scientific_risk']:.17g} | {row['heldout_fit_K280_action']:.17g} | "
            f"{row['heldout_audit_K280_action']:.17g} | {row['maximum_weak_residual']:.17g} | "
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
        "passed": decisions["V_V4_SINGLE_SEED_K280_AUTHORITY"],
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
        "history_after_v4": history_after,
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
        progress(f"V4 terminal result: {status}")
    return summary


def write_terminal_failure(
    stage: str, error: BaseException,
) -> dict[str, Any]:
    """Fail closed for a pre-heldout or operationally terminal V4 stage."""

    activate()
    protocol = require_v4()
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
            "# Official B1 Galerkin Pareto V4 Final Result", "",
            f"Overall status: **{payload['status']}**", "",
            f"The single-root authority failed closed during `{stage}`.",
            f"Failure: `{type(error).__name__}: {error}`",
            f"Selection sealed: `{str(payload['selection_sealed']).lower()}`.",
            f"Held-out generated: `{str(payload['heldout_generated']).lower()}`.",
            "No alternate root or post-result tuning was used.", "",
        )))
    return payload


__all__ = [
    "freeze_v4", "require_v4", "generate_data", "score_candidate_universe",
    "refreeze_law", "run_selection", "certify_selection", "generate_heldout",
    "validate_heldout", "finalize", "write_terminal_failure", "guard_worker",
    "activate", "chunked_many_body_features",
]
