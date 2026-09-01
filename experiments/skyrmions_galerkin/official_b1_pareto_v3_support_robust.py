"""Prospective one-root-seed support-robust repair of the B1 Pareto study."""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import ast
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from . import official_b1_pareto_v2_single_seed as base


ROOT = base.ROOT
PARENT_ROOT = ROOT / "outputs" / "official_b1_galerkin_pareto_v2_1_single_seed_amended"
OUTPUT_ROOT = ROOT / "outputs" / "official_b1_galerkin_pareto_v3_support_robust_single_seed"
V3_PROTOCOL_PATH = OUTPUT_ROOT / "protocol_v3_support_robust.json"
V3_CALL_GRAPH_PATH = OUTPUT_ROOT / "jax_only_call_graph_v3.json"
LAW_GUARD_SUMMARY_PATH = OUTPUT_ROOT / "law" / "guard_screen.json"
V3_REPORT_PATH = OUTPUT_ROOT / "OFFICIAL_B1_GALERKIN_PARETO_V3_SUPPORT_ROBUST_RESULT.md"
V3_SUMMARY_PATH = OUTPUT_ROOT / "v3_final_summary.json"
RUNNER_PATH = ROOT / "official_b1_pareto_v3_support_robust_run.py"
TEST_PATH = ROOT / "test_official_b1_pareto_v3_support_robust.py"

ROOT_SEED = 20261003
GUARD_BLOCK_SIZE = 8
GUARD_COUNTS = {
    "law_guard_screen": 8192,
    "law_guard_search_train": 32768,
    "law_guard_periodic_audit": 16384,
    "law_guard_authoritative_train": 65536,
}
FRESH_STANDARD_ROLES = {
    "search_train": 32768,
    "search_audit": 16384,
    "authoritative_train": 65536,
    "authoritative_audit": 65536,
}
ROLE_IDS = {
    "search_train": 1001,
    "search_audit": 1002,
    "authoritative_train": 1003,
    "authoritative_audit": 1004,
    "law_guard_screen": 1010,
    "law_guard_search_train": 1011,
    "law_guard_periodic_audit": 1012,
    "law_guard_authoritative_train": 1013,
    "heldout_truth": 1020,
    "heldout_reference_fit": 1021,
    "heldout_reference_audit": 1022,
    "heldout_observation_noise": 1023,
}
INHERITED_BANK_ROLES = ("risk_anchor", "support_screen", "support_audit")

ORIGINAL_OUTPUT_ROOT = base.OUTPUT_ROOT
ORIGINAL_ROLE_SEED = base.role_seed
ORIGINAL_RUN_SELECTION_PASS = base.run_selection_pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _atomic_json(path: Path, payload: Any) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"refusing to overwrite V3 artifact: {path}")
        return
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


def _atomic_text(path: Path, value: str) -> None:
    data = value.encode()
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"refusing to overwrite V3 artifact: {path}")
        return
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


def _link(source: Path, destination: Path) -> None:
    if destination.exists():
        if _sha256(source) != _sha256(destination):
            raise RuntimeError(f"V3 inherited input mismatch: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def _stage_inherited_inputs() -> None:
    for relative in (
        "protocol_v2_single_seed.json",
        "freeze_manifest_v2_single_seed.json",
        "randomness_provenance_v2_single_seed.md",
        "jax_only_call_graph.json",
        "effective_config.json",
    ):
        _link(PARENT_ROOT / relative, OUTPUT_ROOT / relative)
    for relative in (
        "development_preflight",
        "candidate_pool",
        "design_truth",
        "artifacts",
        "feasibility",
    ):
        source_root = PARENT_ROOT / relative
        for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
            _link(source, OUTPUT_ROOT / source.relative_to(PARENT_ROOT))
    for role in INHERITED_BANK_ROLES:
        for source in sorted((PARENT_ROOT / "banks").glob(f"{role}_N*.*")):
            _link(source, OUTPUT_ROOT / "banks" / source.name)
    _link(
        PARENT_ROOT / "banks" / "manifest.json",
        OUTPUT_ROOT / "provenance" / "parent_bank_manifest.json",
    )
    _link(
        PARENT_ROOT / "amendment_final_summary.json",
        OUTPUT_ROOT / "provenance" / "parent_failed_summary.json",
    )


def _patch_base_paths() -> None:
    base.VERSION = "official_b1_galerkin_pareto_v3_support_robust_single_seed"
    base.OUTPUT_ROOT = OUTPUT_ROOT
    mapping = {
        "PROTOCOL_PATH": "protocol_v2_single_seed.json",
        "FREEZE_MANIFEST_PATH": "freeze_manifest_v2_single_seed.json",
        "RANDOMNESS_PATH": "randomness_provenance_v2_single_seed.md",
        "CALL_GRAPH_PATH": "jax_only_call_graph.json",
        "EFFECTIVE_CONFIG_PATH": "effective_config.json",
        "CANDIDATE_POOL_PATH": "candidate_pool/candidate_pool.json",
        "SCIENTIFIC_ARRAYS_PATH": "feasibility/exact_receipts.npz",
        "SCIENTIFIC_ROWS_PATH": "feasibility/exact_receipts.json",
        "LAW_PATH": "law/initial_law.json",
        "SELECTION_SEAL_PATH": "selection/selection_seal.json",
        "SELECTION_VERIFICATION_PATH": "selection/independent_verification.json",
        "FINAL_SUMMARY_PATH": "final_summary.json",
        "FINAL_CSV_PATH": "final_summary.csv",
        "PERFORMANCE_REPORT_PATH": "B1_V3_JAX_PERFORMANCE_REPORT.md",
        "RESULT_REPORT_PATH": "OFFICIAL_B1_GALERKIN_PARETO_V3_ENGINE_RESULT.md",
        "PREFLIGHT_PATH": "development_preflight/historical_equivalence.json",
    }
    for name, relative in mapping.items():
        setattr(base, name, OUTPUT_ROOT / relative)
    # Keep the V3 authority executable from its own staged scientific inputs.
    base.REFERENCE_PATH = OUTPUT_ROOT / "artifacts" / "reference.npz"
    base.DICTIONARY_PATH = OUTPUT_ROOT / "artifacts" / "dictionary_K280.npz"


def _source_hashes() -> dict[str, str]:
    return {
        Path(__file__).name: _sha256(Path(__file__)),
        RUNNER_PATH.name: _sha256(RUNNER_PATH),
        TEST_PATH.name: _sha256(TEST_PATH),
    }


def _v3_call_graph() -> dict[str, Any]:
    sources = (Path(__file__), RUNNER_PATH, base.ROOT / "jax_galerkin_v2.py")
    forbidden_modules = {
        "mfsi.galerkin_tesseract",
        ".pareto_v2_selection",
    }
    forbidden_calls = {
        "assemble_galerkin_chunk_tesseract",
        "evaluate_galerkin_action",
    }
    violations = []
    for source in sources:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden_modules:
                        violations.append({"source": source.name, "forbidden_import": alias.name})
            elif isinstance(node, ast.ImportFrom):
                module = ("." * node.level) + (node.module or "")
                if module in forbidden_modules:
                    violations.append({"source": source.name, "forbidden_import": module})
            elif isinstance(node, ast.Call):
                function = node.func
                called = (
                    function.id if isinstance(function, ast.Name)
                    else function.attr if isinstance(function, ast.Attribute)
                    else ""
                )
                if called in forbidden_calls:
                    violations.append({"source": source.name, "forbidden_call": called})
    inherited = json.loads((OUTPUT_ROOT / "jax_only_call_graph.json").read_text())
    return {
        "schema_version": 1,
        "entrypoint": "official_b1_pareto_v3_support_robust_run.py:main",
        "reachable_sources": {source.name: _sha256(source) for source in sources},
        "scientific_edges": [
            ["runner.main", "study.run_selection_with_restarts"],
            ["study.run_selection_with_restarts", "base.run_selection_pass"],
            ["base.run_selection_pass", "base.JaxGalerkinContext.evaluate"],
            ["base.JaxGalerkinContext.evaluate", "rank_aware_quadratic_solve"],
            ["study.guard_qualify_rows", "base.CandidateEvaluator.evaluate"],
        ],
        "inherited_v2_call_graph_sha256": _sha256(OUTPUT_ROOT / "jax_only_call_graph.json"),
        "inherited_v2_passed": bool(inherited["passed"]),
        "violations": violations,
        "native_galerkin_reachable": bool(violations),
        "passed": bool(inherited["passed"] and not violations),
    }


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


def role_seed(role: str) -> int:
    if role == "selection_observation_noise":
        # The inherited exact-risk estimand includes this already-frozen noise role.
        return ORIGINAL_ROLE_SEED(role)
    if role not in ROLE_IDS:
        return ORIGINAL_ROLE_SEED(role)
    return int(_role_record(role)["integer_seed_adapter"])


def _amended_select_starts(
    feasible: list[dict[str, Any]], law: dict[str, Any],
    incumbent: dict[str, Any] | None, *, count: int,
    additional_mandatory: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not feasible:
        raise RuntimeError("no exactly feasible candidates")
    feasible = base._unique_rows(feasible)
    selected: list[dict[str, Any]] = []

    def add(row: dict[str, Any], role: str) -> None:
        existing = next((
            old for old in selected
            if base.eta_key(row["eta"]) == base.eta_key(old["eta"])
        ), None)
        if existing is None:
            selected.append({
                **row, "start_role": role,
                "mandatory_roles": [role] if role.startswith("mandatory_") else [],
            })
        elif role.startswith("mandatory_") and role not in existing["mandatory_roles"]:
            existing["mandatory_roles"].append(role)

    add(law, "mandatory_law")
    if incumbent is not None:
        add(incumbent, "mandatory_previous_incumbent")
    for row in additional_mandatory or []:
        add(row, "mandatory_current_tangent")
    if len(selected) > count:
        raise RuntimeError("mandatory starts exceed frozen cap")
    add(min(feasible, key=lambda row: (
        row["exact_scientific_risk"], base.eta_key(row["eta"])
    )), "lowest_exact_risk")
    add(max(feasible, key=lambda row: (
        row["minimum_rESS"], base.eta_key(row["eta"])
    )), "strongest_robust_rESS")
    while len(selected) < count:
        remaining = [
            row for row in feasible
            if not any(base.eta_key(row["eta"]) == base.eta_key(old["eta"]) for old in selected)
        ]
        if not remaining:
            break
        row = max(remaining, key=lambda candidate: (
            min(base._symmetry_aware_distance(candidate["eta"], old["eta"], base.BOX)
                for old in selected),
            base.eta_key(candidate["eta"]),
        ))
        add(row, "symmetry_aware_maxmin")
    selected = selected[:count]
    for row in selected:
        row["start_availability"] = {
            "requested_cap": count,
            "available_distinct_feasible": len(feasible),
            "used": len(selected),
            "all_available_used_when_below_cap": len(feasible) >= count or len(selected) == len(feasible),
            "rule": "vortices_v2_1_up_to_cap",
        }
    return selected


def prepare_v3(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    if V3_PROTOCOL_PATH.exists():
        protocol = json.loads(V3_PROTOCOL_PATH.read_text())
        _stage_inherited_inputs()
        _activate()
        return protocol
    _stage_inherited_inputs()
    forbidden_banks = [
        path for path in (OUTPUT_ROOT / "banks").rglob("*")
        if path.is_file() and not any(path.name.startswith(role + "_N") for role in INHERITED_BANK_ROLES)
    ]
    outcome_roots = [
        OUTPUT_ROOT / "law",
        OUTPUT_ROOT / "authoritative",
        OUTPUT_ROOT / "heldout_validation",
        *OUTPUT_ROOT.glob("selection_pass_*"),
    ]
    forbidden_outcomes = [
        path for root in outcome_roots if root.exists()
        for path in root.rglob("*") if path.is_file()
    ]
    if forbidden_banks or forbidden_outcomes:
        raise RuntimeError(
            f"fresh V3 data/outcomes exist before protocol freeze: "
            f"{forbidden_banks + forbidden_outcomes}"
        )
    parent = json.loads((PARENT_ROOT / "amendment_final_summary.json").read_text())
    parent_protocol = json.loads((PARENT_ROOT / "protocol_v2_single_seed.json").read_text())
    body = {
        "schema_version": 1,
        "version": "official_b1_galerkin_pareto_v3_support_robust_single_seed",
        "status": "FROZEN_BEFORE_FRESH_GUARD_OR_ACTION_DATA",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "prospective repair after V2.1 Law authoritative support failure",
        "root_seed": ROOT_SEED,
        "single_root_seed": True,
        "alternate_root_seeds_tested": [],
        "derived_roles_are_not_replicates": True,
        "parent_failed_summary_sha256": _sha256(
            PARENT_ROOT / "amendment_final_summary.json"
        ),
        "parent_status": parent["status"],
        "parent_protocol_sha256": parent_protocol["protocol_sha256"],
        "inherited": {
            "candidate_pool_sha256": _sha256(PARENT_ROOT / "candidate_pool" / "candidate_pool.json"),
            "exact_receipts_sha256": _sha256(PARENT_ROOT / "feasibility" / "exact_receipts.json"),
            "exact_arrays_sha256": _sha256(PARENT_ROOT / "feasibility" / "exact_receipts.npz"),
            "design_truth_sha256": _sha256(PARENT_ROOT / "design_truth" / "design_truth.npz"),
            "reference_sha256": _sha256(PARENT_ROOT / "artifacts" / "reference.npz"),
            "risk_and_support_roles": list(INHERITED_BANK_ROLES),
        },
        "law_refreeze": {
            "candidate_order": "jointly-supported inherited rows sorted by exact risk then eta_sha256",
            "block_size": GUARD_BLOCK_SIZE,
            "fresh_guard_roles": GUARD_COUNTS,
            "unchanged_minimum_rESS": 0.05,
            "every_guard_role_must_pass_all_original_support_gates": True,
            "selection_rule": "first risk-ordered candidate passing every fresh guard role",
            "stop_rule_frozen_before_guard_outcomes": True,
            "no_posthoc_support_margin": True,
        },
        "fresh_action_roles": FRESH_STANDARD_ROLES,
        "fresh_heldout_roles": [role for role in ROLE_IDS if role.startswith("heldout_")],
        "role_records": [_role_record(role) for role in ROLE_IDS],
        "inherited_selection_observation_noise": True,
        "allowances_percent": [0.5, 1.0, 2.0],
        "K": 280,
        "galerkin_backend": "jax",
        "dtype": "float64",
        "law_consistency_tolerance": 1.0e-4,
        "maximum_restarts": 2,
        "downstream_reanchor_requires_all_fresh_guard_roles": True,
        "authoritative_audit_reserved_from_law_selection": True,
        "source_hashes": _source_hashes(),
        "validation_accessed": False,
    }
    call_graph = _v3_call_graph()
    if not call_graph["passed"]:
        raise RuntimeError(f"native Galerkin is reachable in V3: {call_graph['violations']}")
    body["jax_only_call_graph_sha256"] = hashlib.sha256(_canonical(call_graph)).hexdigest()
    protocol = {
        **body,
        "v3_protocol_sha256": hashlib.sha256(_canonical(body)).hexdigest(),
    }
    _atomic_json(V3_CALL_GRAPH_PATH, call_graph)
    _atomic_json(V3_PROTOCOL_PATH, protocol)
    _activate()
    if progress:
        progress(f"V3 support-robust protocol frozen: {protocol['v3_protocol_sha256']}")
    return protocol


def require_v3() -> dict[str, Any]:
    protocol = prepare_v3()
    body = {key: value for key, value in protocol.items() if key != "v3_protocol_sha256"}
    if hashlib.sha256(_canonical(body)).hexdigest() != protocol["v3_protocol_sha256"]:
        raise RuntimeError("V3 protocol digest mismatch")
    observed = _source_hashes()
    if observed != protocol["source_hashes"]:
        raise RuntimeError("V3 source changed after freeze")
    graph = json.loads(V3_CALL_GRAPH_PATH.read_text())
    if hashlib.sha256(_canonical(graph)).hexdigest() != protocol["jax_only_call_graph_sha256"]:
        raise RuntimeError("V3 call-graph seal mismatch")
    if not graph["passed"]:
        raise RuntimeError("V3 JAX-only call graph failed")
    return protocol


def _activate() -> None:
    _patch_base_paths()
    base.role_seed = role_seed
    base.generate_data = generate_data
    base._select_starts = _amended_select_starts
    base.run_selection_with_restarts = run_selection_with_restarts
    base.verify_frozen_selection = verify_frozen_selection


def _guard_path(label: str) -> Path:
    return OUTPUT_ROOT / "banks" / "guard" / f"{label}_N{GUARD_COUNTS[label]}.npz"


def _fresh_bank(
    cfg: dict[str, Any], flow: Any, truth_model: Any, times: Any,
    label: str, count: int, path: Path,
) -> dict[str, Any]:
    record_path = path.with_suffix(".json")
    seed = role_seed(label)
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"V3 bank checkpoint mismatch: {label}")
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
        "schema_version": 1,
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
    require_v3()
    manifest_path = OUTPUT_ROOT / "banks" / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for row in manifest["artifacts"]:
            if _sha256(OUTPUT_ROOT / row["path"]) != row["sha256"]:
                raise RuntimeError(f"V3 bank artifact changed: {row['path']}")
        return manifest
    cfg = base.effective_config()
    with np.load(OUTPUT_ROOT / "design_truth" / "design_truth.npz", allow_pickle=False) as arrays:
        times = jnp.asarray(arrays["times"], dtype=jnp.float64)
    truth_model = base.SkyrmionTruth(base._physics_config(cfg))
    flow = base.load_reference(OUTPUT_ROOT / "artifacts" / "reference.npz")
    records = []
    for label, count in {**FRESH_STANDARD_ROLES, **GUARD_COUNTS}.items():
        path = base._bank_path(label) if label in FRESH_STANDARD_ROLES else _guard_path(label)
        record = _fresh_bank(cfg, flow, truth_model, times, label, count, path)
        records.append(record)
        if progress:
            progress(f"V3 fresh bank {label} N={count}")
    parent_manifest = json.loads(
        (OUTPUT_ROOT / "provenance" / "parent_bank_manifest.json").read_text()
    )
    initial_hashes = {
        role: parent_manifest["initial_state_hashes"][role]
        for role in INHERITED_BANK_ROLES
    }
    initial_hashes.update({row["label"]: row["initial_state_sha256"] for row in records})
    artifact_paths = [
        OUTPUT_ROOT / "design_truth" / "design_truth.npz",
        OUTPUT_ROOT / "artifacts" / "reference.npz",
    ]
    artifact_paths += [base._bank_path(role) for role in INHERITED_BANK_ROLES]
    for row in records:
        path = OUTPUT_ROOT / row["path"]
        artifact_paths.extend((path, path.with_suffix(".json")))
    manifest = {
        "schema_version": 3,
        "passed": len(set(initial_hashes.values())) == len(initial_hashes),
        "v3_protocol_sha256": require_v3()["v3_protocol_sha256"],
        "root_seed": ROOT_SEED,
        "derived_roles_are_not_replicates": True,
        "inherited_roles": list(INHERITED_BANK_ROLES),
        "fresh_roles": list(FRESH_STANDARD_ROLES) + list(GUARD_COUNTS),
        "initial_state_hashes": initial_hashes,
        "role_disjoint": len(set(initial_hashes.values())) == len(initial_hashes),
        "artifacts": [{
            "path": str(path.relative_to(OUTPUT_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        } for path in artifact_paths],
    }
    _atomic_json(manifest_path, manifest)
    if not manifest["passed"]:
        raise RuntimeError("V3 fresh/inherited role collision")
    return manifest


def _load_guard(label: str) -> base.GalerkinReferenceBank:
    with np.load(_guard_path(label), allow_pickle=False) as arrays:
        return base.GalerkinReferenceBank(
            jnp.asarray(arrays["configurations"], dtype=jnp.float64),
            jnp.asarray(arrays["velocity"], dtype=jnp.float64),
            jnp.asarray(arrays["base_weights"], dtype=jnp.float64),
        )


def _guard_cache_path(eta: Any) -> Path:
    return OUTPUT_ROOT / "law" / "guard_cache" / f"{base.eta_key(eta)}.json"


def guard_qualify_rows(
    rows: list[dict[str, Any]],
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    generate_data(progress)
    cached: dict[str, dict[str, Any]] = {}
    missing = []
    for row in rows:
        path = _guard_cache_path(row["eta"])
        if path.exists():
            cached[base.eta_key(row["eta"])] = json.loads(path.read_text())
        else:
            missing.append(row)
    if missing:
        data = base.selection_data("risk_anchor", "risk_anchor", projection="risk_anchor")
        evaluator = base.CandidateEvaluator(data, batch_size=GUARD_BLOCK_SIZE)
        etas = np.asarray([row["eta"] for row in missing], dtype=np.float64)
        by_role = {
            role: evaluator.evaluate(etas, _load_guard(role))
            for role in GUARD_COUNTS
        }
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
                "schema_version": 1,
                "candidate_id": row.get("candidate_id", f"downstream_{base.eta_key(row['eta'])}"),
                "eta": row["eta"],
                "eta_sha256": base.eta_key(row["eta"]),
                "exact_scientific_risk": row["exact_scientific_risk"],
                "support_by_fresh_guard_role": support,
                "support_robust": all(item["support_valid"] for item in support.values()),
                "minimum_guard_rESS": min(item["minimum_rESS"] for item in support.values()),
                "threshold": 0.05,
                "authoritative_audit_used": False,
            }
            _atomic_json(_guard_cache_path(row["eta"]), receipt)
            cached[receipt["eta_sha256"]] = receipt
        if progress:
            progress(f"V3 guard-qualified {len(missing)} candidates")
    return [cached[base.eta_key(row["eta"])] for row in rows]


def refreeze_law(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    require_v3()
    generate_data(progress)
    if base.LAW_PATH.exists():
        return json.loads(base.LAW_PATH.read_text())
    if any(path.is_file() for root in OUTPUT_ROOT.glob("selection_pass_*") for path in root.rglob("*")):
        raise RuntimeError("V3 Law must freeze before action outcomes")
    exact = json.loads(base.SCIENTIFIC_ROWS_PATH.read_text())
    ordered = sorted(
        (row for row in exact["rows"] if row["jointly_supported"]),
        key=lambda row: (row["exact_scientific_risk"], row["eta_sha256"]),
    )
    evaluated: list[dict[str, Any]] = []
    winner = None
    for start in range(0, len(ordered), GUARD_BLOCK_SIZE):
        block = ordered[start:start + GUARD_BLOCK_SIZE]
        receipts = guard_qualify_rows(block, progress)
        evaluated.extend(receipts)
        passing = [
            (row, receipt) for row, receipt in zip(block, receipts, strict=True)
            if receipt["support_robust"]
        ]
        if passing:
            winner = passing[0][0]
            break
    if winner is None:
        raise RuntimeError("no support-robust V3 Law in frozen candidate universe")
    winning_guard = json.loads(_guard_cache_path(winner["eta"]).read_text())
    summary = {
        "schema_version": 1,
        "status": "COMPLETE",
        "v3_protocol_sha256": require_v3()["v3_protocol_sha256"],
        "ordered_supported_count": len(ordered),
        "evaluated_count": len(evaluated),
        "block_size": GUARD_BLOCK_SIZE,
        "winner_candidate_id": winner["candidate_id"],
        "winner_eta_sha256": winner["eta_sha256"],
        "winner_guard_receipt_sha256": _sha256(_guard_cache_path(winner["eta"])),
        "evaluated": evaluated,
        "action_outcomes_accessed": False,
    }
    _atomic_json(LAW_GUARD_SUMMARY_PATH, summary)
    law = {
        "schema_version": 3,
        "status": "FROZEN_SUPPORT_ROBUST_BEFORE_ACTION",
        "protocol_sha256": json.loads(base.PROTOCOL_PATH.read_text())["protocol_sha256"],
        "v3_protocol_sha256": require_v3()["v3_protocol_sha256"],
        "candidate_id": winner["candidate_id"],
        "eta": winner["eta"],
        "eta_sha256": winner["eta_sha256"],
        "R_star": winner["exact_scientific_risk"],
        "minimum_guard_rESS": winning_guard["minimum_guard_rESS"],
        "law_consistency_tolerance": 1.0e-4,
        "risk_ceilings": {
            str(value): base.selection_ceiling(winner["exact_scientific_risk"], value)
            for value in (0.5, 1.0, 2.0)
        },
        "selection_rule": "minimum exact risk in frozen order among candidates passing all four fresh guard roles",
        "tie_break": "eta_sha256",
        "authoritative_audit_used_for_selection": False,
    }
    _atomic_json(base.LAW_PATH, law)
    if progress:
        progress(
            f"V3 Law frozen: {winner['candidate_id']} risk={winner['exact_scientific_risk']:.12g} "
            f"guard_rESS={winning_guard['minimum_guard_rESS']:.6g}"
        )
    return law


def run_selection_with_restarts(
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    require_v3()
    refreeze_law(progress)
    final_path = OUTPUT_ROOT / "selection" / "restart_summary.json"
    if final_path.exists():
        return json.loads(final_path.read_text())
    exact_rows = json.loads(base.SCIENTIFIC_ROWS_PATH.read_text())["rows"]
    initial = json.loads(base.LAW_PATH.read_text())
    law = next(row for row in exact_rows if row["eta_sha256"] == initial["eta_sha256"])
    passes = []
    status = "PASS"
    maximum_restarts = 2
    tolerance = 1.0e-4
    for pass_index in range(maximum_restarts + 1):
        result = ORIGINAL_RUN_SELECTION_PASS(pass_index, law, progress=progress)
        challengers = [
            row for row in result["generated_exact_receipts"]
            if row["jointly_supported"]
            and row["exact_scientific_risk"] < law["exact_scientific_risk"] - tolerance
        ]
        guard_receipts = guard_qualify_rows(challengers, progress) if challengers else []
        robust = [
            row for row, receipt in zip(challengers, guard_receipts, strict=True)
            if receipt["support_robust"]
        ]
        best = min([law, *robust], key=lambda row: (
            row["exact_scientific_risk"], base.eta_key(row["eta"])
        ))
        material = best["exact_scientific_risk"] < law["exact_scientific_risk"] - tolerance
        passes.append({
            "pass_index": pass_index,
            "path": str((OUTPUT_ROOT / f"selection_pass_{pass_index}" / "complete.json").relative_to(OUTPUT_ROOT)),
            "law_eta_sha256": base.eta_key(law["eta"]),
            "R_star": law["exact_scientific_risk"],
            "downstream_material_risk_challengers": len(challengers),
            "downstream_guard_robust_challengers": len(robust),
            "material_law_improvement": material,
            "law_improvement": best["exact_scientific_risk"] - law["exact_scientific_risk"],
        })
        if not material:
            break
        if pass_index == maximum_restarts:
            status = "FAIL_SUPPORT_ROBUST_ANCHOR_INCONSISTENT_AFTER_MAXIMUM_RESTARTS"
            break
        previous = law
        law = best
        _atomic_json(OUTPUT_ROOT / "law" / f"reanchor_{pass_index + 1}.json", {
            "schema_version": 3,
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
    final_pass = json.loads(
        (OUTPUT_ROOT / f"selection_pass_{final_pass_index}" / "complete.json").read_text()
    )
    summary = {
        "schema_version": 3,
        "passed": status == "PASS",
        "status": status,
        "protocol_sha256": json.loads(base.PROTOCOL_PATH.read_text())["protocol_sha256"],
        "v3_protocol_sha256": require_v3()["v3_protocol_sha256"],
        "passes": passes,
        "restart_count": len(passes) - 1,
        "maximum_restart_count": maximum_restarts,
        "final_pass_index": final_pass_index,
        "final_law": final_pass["law"],
        "final_law_consistent": not passes[-1]["material_law_improvement"],
        "final_full_action_nonincreasing": final_pass["full_action_nonincreasing"],
        "guard_robust_reanchor_rule": True,
    }
    _atomic_json(final_path, summary)
    if not summary["passed"]:
        raise RuntimeError(status)
    return summary


def verify_frozen_selection(
    selection: dict[str, Any] | None = None, *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if selection is None:
        selection = json.loads(base.SELECTION_SEAL_PATH.read_text())
    seal_sha = _sha256(base.SELECTION_SEAL_PATH)
    if base.SELECTION_VERIFICATION_PATH.exists():
        verification = json.loads(base.SELECTION_VERIFICATION_PATH.read_text())
        if verification["selection_seal_sha256"] != seal_sha or not verification["passed"]:
            raise RuntimeError("V3 independent verification mismatch")
        return verification
    restart = json.loads((OUTPUT_ROOT / "selection" / "restart_summary.json").read_text())
    final_path = OUTPUT_ROOT / f"selection_pass_{restart['final_pass_index']}" / "complete.json"
    final_pass = json.loads(final_path.read_text())
    tolerance = 1.0e-10
    checks = []

    def check(name: str, passed: bool) -> None:
        checks.append({"name": name, "passed": bool(passed)})

    def roles(row: dict[str, Any]) -> set[str]:
        return set(row.get("mandatory_roles", ())) | {row.get("start_role", "")}

    def mandatory(starts: list[dict[str, Any]], role: str, eta: Any) -> bool:
        key = base.eta_key(eta)
        return any(base.eta_key(row["eta"]) == key and role in roles(row) for row in starts)

    def winner(finalists: list[dict[str, Any]], incumbent: dict[str, Any] | None):
        best = min(finalists, key=lambda row: (
            row["selection_action"], base.eta_key(row["eta"])
        ))
        old = None if incumbent is None else next((
            row for row in finalists if base.eta_key(row["eta"]) == base.eta_key(incumbent["eta"])
        ), None)
        return old if old is not None and best["selection_action"] >= old["selection_action"] - tolerance else best

    exact = json.loads(base.SCIENTIFIC_ROWS_PATH.read_text())
    check("exact_pool_precedes_action", bool(
        exact["exact_before_full_action"] and exact["full_action_evaluations_before_receipt"] == 0
    ))
    manifest = json.loads((OUTPUT_ROOT / "banks" / "manifest.json").read_text())
    check("one_root_seed", manifest["root_seed"] == ROOT_SEED)
    check("fresh_roles_disjoint", bool(manifest["role_disjoint"]))
    check("authoritative_audit_is_fresh_and_reserved", bool(
        "authoritative_audit" in manifest["fresh_roles"]
        and "authoritative_audit" not in GUARD_COUNTS
    ))

    guard_summary = json.loads(LAW_GUARD_SUMMARY_PATH.read_text())
    ordered = sorted(
        (row for row in exact["rows"] if row["jointly_supported"]),
        key=lambda row: (row["exact_scientific_risk"], row["eta_sha256"]),
    )
    evaluated = guard_summary["evaluated"]
    check("initial_law_guard_prefix_reconstructed", [
        row["eta_sha256"] for row in evaluated
    ] == [row["eta_sha256"] for row in ordered[:len(evaluated)]])
    first_guard_robust = next((row for row in evaluated if row["support_robust"]), None)
    check("initial_law_is_first_guard_robust_in_risk_order", bool(
        first_guard_robust is not None
        and first_guard_robust["eta_sha256"] == guard_summary["winner_eta_sha256"]
    ))
    check("law_guards_did_not_open_authoritative_audit", all(
        not row["authoritative_audit_used"] for row in evaluated
    ))

    reanchor_reconstruction = True
    for index, pass_row in enumerate(restart["passes"]):
        complete = json.loads(
            (OUTPUT_ROOT / f"selection_pass_{pass_row['pass_index']}" / "complete.json").read_text()
        )
        current_law = complete["law"]
        challengers = [
            row for row in complete["generated_exact_receipts"]
            if row["jointly_supported"]
            and row["exact_scientific_risk"] < current_law["exact_scientific_risk"] - 1.0e-4
        ]
        robust = [
            row for row in challengers
            if json.loads(_guard_cache_path(row["eta"]).read_text())["support_robust"]
        ]
        best = min([current_law, *robust], key=lambda row: (
            row["exact_scientific_risk"], base.eta_key(row["eta"])
        ))
        material = best["exact_scientific_risk"] < current_law["exact_scientific_risk"] - 1.0e-4
        reanchor_reconstruction &= (
            base.eta_key(current_law["eta"]) == pass_row["law_eta_sha256"]
            and len(challengers) == pass_row["downstream_material_risk_challengers"]
            and len(robust) == pass_row["downstream_guard_robust_challengers"]
            and material == pass_row["material_law_improvement"]
        )
        if material and index + 1 < len(restart["passes"]):
            reanchor_reconstruction &= (
                base.eta_key(best["eta"])
                == restart["passes"][index + 1]["law_eta_sha256"]
            )
    check("support_robust_reanchors_reconstructed", reanchor_reconstruction)
    final_law_guard = json.loads(_guard_cache_path(final_pass["law"]["eta"]).read_text())
    check("final_law_passes_every_fresh_guard", bool(
        final_law_guard["support_robust"]
        and all(row["support_valid"] for row in final_law_guard["support_by_fresh_guard_role"].values())
    ))
    tangent_incumbent = None
    full_incumbent = None
    reconstructed = [{
        "method": "Law", "allowance_percent": None,
        "eta_sha256": base.eta_key(final_pass["law"]["eta"]),
    }]
    for index, (tangent, full) in enumerate(zip(final_pass["tangent"], final_pass["full"], strict=True)):
        allowance = float(tangent["allowance_percent"])
        tangent_expected = winner(tangent["authoritative_finalists"], tangent_incumbent)
        check(f"tangent_{allowance:g}_winner_reconstructed", base.eta_key(tangent_expected["eta"]) == base.eta_key(tangent["winner"]["eta"]))
        check(f"tangent_{allowance:g}_winner_feasible_and_valid", bool(tangent["winner"]["jointly_supported"] and tangent["winner"]["authoritative"]["valid"]))
        tangent_incumbent = tangent_expected
        reconstructed.append({"method": "Tangent", "allowance_percent": allowance, "eta_sha256": base.eta_key(tangent_expected["eta"])})
        check(f"full_{allowance:g}_current_tangent_mandatory", mandatory(full["starts"], "mandatory_current_tangent", tangent["winner"]["eta"]))
        if index == 0:
            check("full_0.5_law_mandatory", mandatory(full["starts"], "mandatory_law", final_pass["law"]["eta"]))
        else:
            check(f"full_{allowance:g}_previous_winner_mandatory", mandatory(full["starts"], "mandatory_previous_incumbent", full_incumbent["eta"]))
        full_expected = winner(full["authoritative_finalists"], full_incumbent)
        check(f"full_{allowance:g}_winner_reconstructed", base.eta_key(full_expected["eta"]) == base.eta_key(full["winner"]["eta"]))
        check(f"full_{allowance:g}_winner_feasible_and_valid", bool(full["winner"]["jointly_supported"] and full["winner"]["authoritative"]["valid"]))
        full_incumbent = full_expected
        reconstructed.append({"method": "Full", "allowance_percent": allowance, "eta_sha256": base.eta_key(full_expected["eta"])})
    sealed = [{
        "method": row["method"], "allowance_percent": row["allowance_percent"],
        "eta_sha256": row["eta_sha256"],
    } for row in selection["winners"]]
    check("selection_seal_matches_reconstruction", sealed == reconstructed)
    check("final_law_consistent", restart["final_law_consistent"])
    verification = {
        "schema_version": 3,
        "passed": all(row["passed"] for row in checks),
        "validation_accessed": False,
        "v3_protocol_sha256": require_v3()["v3_protocol_sha256"],
        "selection_seal_sha256": seal_sha,
        "final_pass_sha256": _sha256(final_path),
        "checks": checks,
        "reconstructed_winners": reconstructed,
    }
    _atomic_json(base.SELECTION_VERIFICATION_PATH, verification)
    if not verification["passed"]:
        raise RuntimeError(f"V3 independent verification failed: {[r['name'] for r in checks if not r['passed']]}")
    if progress:
        progress("V3 independent winner reconstruction passed")
    return verification


def certify(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    require_v3()
    return base.certify_and_freeze_selection(progress)


def validate(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    require_v3()
    return base.validate_heldout(progress)


def write_reports(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    protocol = require_v3()
    summary = base.write_final_reports(progress)
    law = json.loads(base.LAW_PATH.read_text())
    lines = [
        "# Official B1 Galerkin Pareto V3 Support-Robust Result",
        "",
        f"Status: **{summary['status']}**",
        "",
        "This is the prospective one-root-seed repair of the V2.1 Law support failure.",
        "",
        f"- Root seed: `{ROOT_SEED}`; alternate roots tested: `none`",
        f"- V3 protocol: `{protocol['v3_protocol_sha256']}`",
        f"- Support-robust Law: `{law['candidate_id']}`",
        f"- Law risk: `{law['R_star']:.17g}`",
        f"- Minimum fresh-guard rESS: `{law['minimum_guard_rESS']:.9g}`",
        "- Four fresh guard roles used the unchanged 0.05 threshold; authoritative audit was reserved.",
        "- Detailed Pareto and A-T tables are in `OFFICIAL_B1_GALERKIN_PARETO_V3_ENGINE_RESULT.md`.",
    ]
    _atomic_text(V3_REPORT_PATH, "\n".join(lines) + "\n")
    v3_summary = {
        "schema_version": 1,
        "status": summary["status"],
        "passed": summary["passed"],
        "authority": "official_b1_galerkin_pareto_v3_support_robust_single_seed",
        "v3_protocol_sha256": protocol["v3_protocol_sha256"],
        "underlying_final_summary_sha256": _sha256(base.FINAL_SUMMARY_PATH),
        "law": law,
        "rows": summary["rows"],
        "heldout_rows": summary["heldout_rows"],
        "decision_table": summary["decision_table"],
    }
    _atomic_json(V3_SUMMARY_PATH, v3_summary)
    return v3_summary


def write_failure_report(stage: str, error: BaseException) -> dict[str, Any]:
    """Persist a terminal fail-closed result without opening held-out validation."""

    protocol = require_v3()
    heldout_opened = (OUTPUT_ROOT / "heldout_validation" / "manifest.json").exists()
    payload = {
        "schema_version": 1,
        "status": f"FAIL_{stage.upper().replace('-', '_')}",
        "passed": False,
        "authority": "official_b1_galerkin_pareto_v3_support_robust_single_seed",
        "v3_protocol_sha256": protocol["v3_protocol_sha256"],
        "failure": {
            "stage": stage,
            "exception_type": type(error).__name__,
            "message": str(error),
        },
        "heldout_validation_opened": heldout_opened,
        "single_root_seed": ROOT_SEED,
        "alternate_root_seeds_tested": [],
    }
    _atomic_json(V3_SUMMARY_PATH, payload)
    _atomic_text(V3_REPORT_PATH, "\n".join((
        "# Official B1 Galerkin Pareto V3 Support-Robust Result",
        "",
        f"Status: **{payload['status']}**",
        "",
        f"The prospective authority failed closed during `{stage}`.",
        f"Held-out validation opened: `{str(heldout_opened).lower()}`.",
        f"Failure: `{type(error).__name__}: {error}`",
        "",
    )))
    return payload


__all__ = [
    "prepare_v3", "require_v3", "generate_data", "refreeze_law",
    "run_selection_with_restarts", "certify", "validate", "write_reports",
    "write_failure_report",
]
