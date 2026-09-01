"""Pre-action start-availability amendment for the single-seed B1 V2 study.

Strict V2 remains untouched as a failed authority.  This namespace reuses its
byte-identical frozen inputs and exact feasibility receipts, and changes only
the start cardinality rule: a requested count is a cap, and every available
distinct feasible start is used when fewer than the cap exist.  This is the
rule used by the frozen Vortices V2.1 selection protocol.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

from . import official_b1_pareto_v2_single_seed as base


ROOT = base.ROOT
PARENT_ROOT = ROOT / "outputs" / "official_b1_galerkin_pareto_v2_single_seed"
OUTPUT_ROOT = ROOT / "outputs" / "official_b1_galerkin_pareto_v2_1_single_seed_amended"
AMENDMENT_PATH = OUTPUT_ROOT / "amendment_pre_action.json"
STRICT_FAILURE_PATH = PARENT_ROOT / "selection" / "pre_action_failure.json"
AMENDMENT_REPORT_PATH = OUTPUT_ROOT / "OFFICIAL_B1_GALERKIN_PARETO_V2_1_AMENDMENT_RESULT.md"
AMENDMENT_SUMMARY_PATH = OUTPUT_ROOT / "amendment_final_summary.json"
AMENDMENT_INVENTORY_PATH = OUTPUT_ROOT / "amendment_inventory.json"
RUNNER_PATH = ROOT / "official_b1_pareto_v2_1_amendment_run.py"

INPUT_ROOT_FILES = (
    "protocol_v2_single_seed.json",
    "freeze_manifest_v2_single_seed.json",
    "randomness_provenance_v2_single_seed.md",
    "jax_only_call_graph.json",
    "effective_config.json",
)
INPUT_DIRECTORIES = (
    "development_preflight",
    "candidate_pool",
    "design_truth",
    "artifacts",
    "banks",
    "feasibility",
    "law",
    "performance/stages",
)


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
            raise RuntimeError(f"refusing to overwrite amendment artifact: {path}")
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


def _atomic_text(path: Path, text: str) -> None:
    data = text.encode()
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"refusing to overwrite amendment artifact: {path}")
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


def _link_file(source: Path, destination: Path) -> None:
    if destination.exists():
        if _sha256(source) != _sha256(destination):
            raise RuntimeError(f"amended input differs from parent: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def _stage_parent_inputs() -> None:
    for relative in INPUT_ROOT_FILES:
        _link_file(PARENT_ROOT / relative, OUTPUT_ROOT / relative)
    for relative in INPUT_DIRECTORIES:
        source_root = PARENT_ROOT / relative
        for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
            _link_file(source, OUTPUT_ROOT / source.relative_to(PARENT_ROOT))


def _parent_action_files() -> list[str]:
    candidates: list[str] = []
    roots = [
        *PARENT_ROOT.glob("selection_pass_*"),
        PARENT_ROOT / "authoritative",
        PARENT_ROOT / "heldout_validation",
        PARENT_ROOT / "selection",
    ]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path != STRICT_FAILURE_PATH:
                candidates.append(str(path.relative_to(PARENT_ROOT)))
    return sorted(set(candidates))


def _patch_base_paths() -> None:
    base.VERSION = "official_b1_galerkin_pareto_v2_1_single_seed_amended"
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
        "PERFORMANCE_REPORT_PATH": "B1_V2_JAX_PERFORMANCE_REPORT.md",
        "RESULT_REPORT_PATH": "OFFICIAL_B1_GALERKIN_PARETO_V2_SINGLE_SEED_RESULT.md",
        "PREFLIGHT_PATH": "development_preflight/historical_equivalence.json",
    }
    for name, relative in mapping.items():
        setattr(base, name, OUTPUT_ROOT / relative)


def _amended_select_starts(
    feasible: list[dict[str, Any]],
    law: dict[str, Any],
    incumbent: dict[str, Any] | None,
    *,
    count: int,
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
                **row,
                "start_role": role,
                "mandatory_roles": [role] if role.startswith("mandatory_") else [],
            })
        elif role.startswith("mandatory_") and role not in existing["mandatory_roles"]:
            existing["mandatory_roles"].append(role)

    add(law, "mandatory_law")
    if incumbent is not None:
        add(incumbent, "mandatory_previous_incumbent")
    for row in additional_mandatory or []:
        add(row, "mandatory_current_tangent")
    add(min(feasible, key=lambda row: (
        row["exact_scientific_risk"], base.eta_key(row["eta"])
    )), "lowest_exact_risk")
    add(max(feasible, key=lambda row: (
        row["minimum_rESS"], base.eta_key(row["eta"])
    )), "strongest_robust_rESS")
    while len(selected) < count:
        remaining = [
            row for row in feasible
            if not any(
                base.eta_key(row["eta"]) == base.eta_key(old["eta"])
                for old in selected
            )
        ]
        if not remaining:
            break
        row = max(remaining, key=lambda candidate: (
            min(
                base._symmetry_aware_distance(candidate["eta"], old["eta"], base.BOX)
                for old in selected
            ),
            base.eta_key(candidate["eta"]),
        ))
        add(row, "symmetry_aware_maxmin")
    selected = selected[:count]
    for row in selected:
        row["start_availability"] = {
            "requested_cap": int(count),
            "available_distinct_feasible": len(feasible),
            "used": len(selected),
            "all_available_used_when_below_cap": len(feasible) >= count or len(selected) == len(feasible),
            "amendment": "vortices_v2_1_up_to_cap",
        }
    return selected


def prepare_amendment(
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if AMENDMENT_PATH.exists():
        manifest = json.loads(AMENDMENT_PATH.read_text())
        _stage_parent_inputs()
        _patch_base_paths()
        base._select_starts = _amended_select_starts
        base.require_protocol()
        return manifest
    # Verify the strict authority with its original globals before redirecting.
    if base.OUTPUT_ROOT == PARENT_ROOT:
        protocol = base.require_protocol()
    else:
        protocol = json.loads((PARENT_ROOT / "protocol_v2_single_seed.json").read_text())
    exact_path = PARENT_ROOT / "feasibility" / "exact_receipts.json"
    exact = json.loads(exact_path.read_text())
    law = json.loads((PARENT_ROOT / "law" / "initial_law.json").read_text())
    if _parent_action_files():
        raise RuntimeError(f"strict V2 already contains action outcomes: {_parent_action_files()}")
    if not (
        exact["count"] == 5645
        and exact["exact_before_full_action"]
        and exact["full_action_evaluations_before_receipt"] == 0
    ):
        raise RuntimeError("parent exact-feasibility receipt is not amendment-safe")
    ceiling = base.selection_ceiling(law["R_star"], 0.5)
    available = [
        row for row in exact["rows"]
        if row["jointly_supported"] and row["exact_scientific_risk"] <= ceiling
    ]
    if len(available) != 3:
        raise RuntimeError(f"unexpected strict V2 0.5% feasible count: {len(available)}")

    failure = {
        "schema_version": 1,
        "status": "FAILED_BEFORE_ANY_TANGENT_OR_FULL_ACTION",
        "protocol_sha256": protocol["protocol_sha256"],
        "reason": "requested six distinct Tangent starts but only three exact-feasible geometries exist at 0.5%",
        "allowance_percent": 0.5,
        "requested_distinct_starts": 6,
        "available_distinct_exact_feasible": 3,
        "full_action_evaluations": 0,
        "exact_receipts_sha256": _sha256(exact_path),
    }
    _atomic_json(STRICT_FAILURE_PATH, failure)
    _stage_parent_inputs()

    manifest_body = {
        "schema_version": 1,
        "status": "FROZEN_PRE_ACTION_AVAILABILITY_AMENDMENT",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "user_authorized": True,
        "single_root_seed": 20261003,
        "alternate_root_seeds_tested": [],
        "parent_authority": str(PARENT_ROOT.relative_to(ROOT)),
        "parent_protocol_sha256": protocol["protocol_sha256"],
        "strict_failure_sha256": _sha256(STRICT_FAILURE_PATH),
        "candidate_pool_sha256": _sha256(PARENT_ROOT / "candidate_pool" / "candidate_pool.json"),
        "exact_receipts_sha256": _sha256(exact_path),
        "exact_arrays_sha256": _sha256(PARENT_ROOT / "feasibility" / "exact_receipts.npz"),
        "law_sha256": _sha256(PARENT_ROOT / "law" / "initial_law.json"),
        "inputs_reused_byte_identically": True,
        "candidate_universe_changed": False,
        "risk_support_or_law_rule_changed": False,
        "action_outcomes_available_before_amendment": False,
        "full_action_evaluations_before_amendment": 0,
        "trigger": {
            "allowance_percent": 0.5,
            "requested_cap": 6,
            "available_distinct_exact_feasible": 3,
        },
        "sole_change": (
            "requested Tangent/Full start counts are upper caps; when fewer "
            "distinct exact-feasible candidates exist, evaluate every available candidate"
        ),
        "precedent": (
            "experiments/vortices_percentage/"
            "VORTICES_V2_1_SELECTION_PROTOCOL_FROZEN.md lines 64-68"
        ),
        "source_hashes": {
            Path(__file__).name: _sha256(Path(__file__)),
            RUNNER_PATH.name: _sha256(RUNNER_PATH),
        },
    }
    manifest = {
        **manifest_body,
        "amendment_sha256": hashlib.sha256(_canonical(manifest_body)).hexdigest(),
    }
    _atomic_json(AMENDMENT_PATH, manifest)
    _patch_base_paths()
    base._select_starts = _amended_select_starts
    base.require_protocol()
    if progress:
        progress(
            "V2.1 amendment frozen before action: use all 3/6 available starts at 0.5%"
        )
    return manifest


def require_amendment() -> dict[str, Any]:
    manifest = prepare_amendment()
    body = {key: value for key, value in manifest.items() if key != "amendment_sha256"}
    if hashlib.sha256(_canonical(body)).hexdigest() != manifest["amendment_sha256"]:
        raise RuntimeError("V2.1 amendment digest mismatch")
    observed = {
        Path(__file__).name: _sha256(Path(__file__)),
        RUNNER_PATH.name: _sha256(RUNNER_PATH),
    }
    if observed != manifest["source_hashes"]:
        raise RuntimeError("V2.1 amendment source changed after freeze")
    return manifest


def run_selection(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    require_amendment()
    return base.run_selection_with_restarts(progress)


def certify(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    require_amendment()
    return base.certify_and_freeze_selection(progress)


def validate(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    require_amendment()
    return base.validate_heldout(progress)


def write_reports(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    amendment = require_amendment()
    summary = base.write_final_reports(progress)
    lines = [
        "# Official B1 Galerkin Pareto V2.1 Amendment Result",
        "",
        f"Status: **{summary['status']}**",
        "",
        "This authority preserves strict V2 as a pre-action failure and applies one transparent availability amendment before any Tangent/Full action was observed.",
        "",
        f"- Root seed: `{summary['single_root_seed']}` (one seed; no alternates)",
        f"- Parent protocol: `{amendment['parent_protocol_sha256']}`",
        f"- Amendment: `{amendment['amendment_sha256']}`",
        "- Sole change: requested start counts are caps; all distinct exact-feasible starts are used when fewer exist.",
        "- Candidate universe, exact receipts, Law, risk ceilings, JAX K=280 estimand, banks, precision, and restart rule are unchanged.",
        "",
        "The detailed Pareto table and A-T decision table are in `OFFICIAL_B1_GALERKIN_PARETO_V2_SINGLE_SEED_RESULT.md` in this amended namespace.",
    ]
    _atomic_text(AMENDMENT_REPORT_PATH, "\n".join(lines) + "\n")
    amended_summary = {
        "schema_version": 1,
        "status": summary["status"],
        "passed": summary["passed"],
        "authority": "official_b1_galerkin_pareto_v2_1_single_seed_amended",
        "amendment_sha256": amendment["amendment_sha256"],
        "strict_v2_failure_sha256": amendment["strict_failure_sha256"],
        "underlying_final_summary_sha256": _sha256(base.FINAL_SUMMARY_PATH),
        "rows": summary["rows"],
        "decision_table": summary["decision_table"],
    }
    _atomic_json(AMENDMENT_SUMMARY_PATH, amended_summary)
    files = [
        path for path in sorted(OUTPUT_ROOT.rglob("*"))
        if path.is_file() and path != AMENDMENT_INVENTORY_PATH
    ]
    _atomic_json(AMENDMENT_INVENTORY_PATH, {
        "schema_version": 1,
        "artifact_count": len(files),
        "files": [{
            "path": str(path.relative_to(OUTPUT_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        } for path in files],
    })
    return amended_summary


__all__ = [
    "prepare_amendment", "require_amendment", "run_selection", "certify",
    "validate", "write_reports",
]
