"""Finalize the terminal failed V3.4 authority without new scientific work."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "official_b1_galerkin_pareto_v3_support_robust_single_seed"
REPORT = OUTPUT / "OFFICIAL_B1_GALERKIN_PARETO_V3_4_TERMINAL_RESULT.md"
SUMMARY = OUTPUT / "v3_4_terminal_summary.json"
CSV_PATH = OUTPUT / "v3_4_terminal_rows.csv"
INVENTORY = OUTPUT / "v3_4_terminal_inventory.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(relative: str) -> Any:
    return json.loads((OUTPUT / relative).read_text())


def atomic(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"refusing to overwrite terminal artifact: {path}")
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


def atomic_json(path: Path, payload: Any) -> None:
    atomic(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")


def main() -> None:
    protocol = read("protocol_v3_support_robust.json")
    law = read("law/initial_law.json")
    law_guard = read(f"law/guard_cache/{law['eta_sha256']}.json")
    restart = read("selection/restart_summary.json")
    selection = read("selection/selection_seal.json")
    verification = read("selection/independent_verification.json")
    heldout_manifest = read("heldout_validation/manifest.json")
    heldout = read("heldout_validation/results.json")
    amendments = {
        "v3_1_memory_schedule": read("memory_schedule_amendment_v3_1.json")["amendment_sha256"],
        "v3_2_low_memory": read("low_memory_amendment_v3_2.json")["amendment_sha256"],
        "v3_3_isolated_guard": read("isolated_guard_amendment_v3_3.json")["amendment_sha256"],
        "v3_4_chunked_guard": read("chunked_guard_amendment_v3_4.json")["amendment_sha256"],
    }
    heldout_by_key = {
        (row["method"], row["allowance_percent"]): row for row in heldout["rows"]
    }
    rows = []
    for row in selection["rows"]:
        validation = heldout_by_key[(row["method"], row["allowance_percent"])]
        rows.append({
            "method": row["method"],
            "allowance_percent": row["allowance_percent"],
            "eta_sha256": row["eta_sha256"],
            "exact_risk": row["exact_risk"],
            "relative_risk_increase": row["relative_risk_increase"],
            "authoritative_train_K280_action": row["train_K280_action"],
            "authoritative_audit_K280_action": row["audit_K280_action"],
            "authoritative_full_certificate_pass": row["full_certificate_pass"],
            "heldout_risk": validation["heldout_risk"],
            "heldout_relative_risk_increase": validation["heldout_relative_risk_increase"],
            "heldout_nominal_risk_pass": validation["strict_nominal_risk_pass"],
            "heldout_train_K280_action": validation["heldout_train_K280_action"],
            "heldout_audit_K280_action": validation["heldout_audit_K280_action"],
            "heldout_full_certificate_pass": validation["heldout_full_certificate_pass"],
            "heldout_tangent_certificate_pass": validation["heldout_tangent_certificate_pass"],
        })
    decisions = {
        "A_v3_protocol_frozen_before_fresh_data": True,
        "B_exactly_one_root_seed": protocol["root_seed"] == 20261003 and protocol["single_root_seed"],
        "C_jax_only_K280_float64": protocol["K"] == 280 and protocol["galerkin_backend"] == "jax" and protocol["dtype"] == "float64",
        "D_support_robust_law_all_four_guards": law_guard["support_robust"],
        "E_authoritative_audit_reserved_from_law": not law["authoritative_audit_used_for_selection"],
        "F_selection_and_authoritative_certification": selection["passed"],
        "G_independent_reconstruction": verification["passed"],
        "H_no_reanchor_required": restart["restart_count"] == 0 and restart["final_law_consistent"],
        "I_full_action_nonincreasing": selection["full_action_nonincreasing"],
        "J_heldout_generated_after_seal_and_disjoint": heldout_manifest["passed"] and heldout_manifest["generated_after_selection_freeze"],
        "K_all_heldout_nominal_risk_constraints": all(row["heldout_nominal_risk_pass"] for row in rows),
        "L_all_heldout_tangent_certificates": all(row["heldout_tangent_certificate_pass"] for row in rows),
        "M_all_heldout_full_certificates": all(row["heldout_full_certificate_pass"] for row in rows),
        "N_overall_authority": False,
    }
    summary = {
        "schema_version": 1,
        "status": "FAIL_HELDOUT_FULL_CERTIFICATES",
        "passed": False,
        "terminal": True,
        "post_heldout_tuning_or_rerun": False,
        "single_root_seed": protocol["root_seed"],
        "alternate_root_seeds_tested": [],
        "allowances_percent": protocol["allowances_percent"],
        "v3_protocol_sha256": protocol["v3_protocol_sha256"],
        "amendments": amendments,
        "law": {
            "candidate_id": law["candidate_id"],
            "eta_sha256": law["eta_sha256"],
            "exact_risk": law["R_star"],
            "minimum_fresh_guard_rESS": law_guard["minimum_guard_rESS"],
            "fresh_guard_rESS": {
                role: values["minimum_rESS"]
                for role, values in law_guard["support_by_fresh_guard_role"].items()
            },
        },
        "selection": {
            "passed": selection["passed"],
            "restart_count": restart["restart_count"],
            "raw_risk_challengers": restart["passes"][0]["downstream_material_risk_challengers"],
            "guard_robust_challengers": restart["passes"][0]["downstream_guard_robust_challengers"],
            "independent_verification_passed": verification["passed"],
        },
        "heldout": {
            "passed": heldout["passed"],
            "selection_geometry_unchanged": heldout["selection_geometry_unchanged"],
            "all_nominal_risk_pass": decisions["K_all_heldout_nominal_risk_constraints"],
            "all_tangent_certificates_pass": decisions["L_all_heldout_tangent_certificates"],
            "full_certificate_pass_count": sum(row["heldout_full_certificate_pass"] for row in rows),
            "full_certificate_row_count": len(rows),
        },
        "rows": rows,
        "decision_table": decisions,
    }
    atomic_json(SUMMARY, summary)

    lines = [
        "# Official B1 Galerkin Pareto V3.4 Terminal Result",
        "",
        "Status: **FAIL — held-out Full certificates**",
        "",
        "The support-robust Law repair and the sealed authoritative Pareto selection passed. The terminal authority fails because all seven held-out Full K=280 certificate flags failed. No alternate seed, post-heldout tuning, or rerun was performed.",
        "",
        "## Law repair",
        "",
        f"- Law: `{law['candidate_id']}` (`{law['eta_sha256']}`)",
        f"- Exact risk: `{law['R_star']:.17g}`",
        f"- Minimum fresh-guard rESS: `{law_guard['minimum_guard_rESS']:.9g}` (threshold `0.05`)",
        "- All four prospective guard roles passed; the authoritative action audit was reserved.",
        f"- Selection passed with `{restart['restart_count']}` restarts; one raw-risk challenger was found and zero challengers passed all four guards.",
        "",
        "## Pareto and held-out result",
        "",
        "| Method | Allowance | Exact risk | Authoritative train K280 | Authoritative audit K280 | Auth Full | Held-out risk | Held-out nominal risk | Held-out Tangent | Held-out Full |",
        "|---|---:|---:|---:|---:|:---:|---:|:---:|:---:|:---:|",
    ]
    for row in rows:
        allowance = "Law" if row["allowance_percent"] is None else f"{row['allowance_percent']:g}%"
        lines.append(
            f"| {row['method']} | {allowance} | {row['exact_risk']:.12g} | "
            f"{row['authoritative_train_K280_action']:.12g} | "
            f"{row['authoritative_audit_K280_action']:.12g} | "
            f"{'PASS' if row['authoritative_full_certificate_pass'] else 'FAIL'} | "
            f"{row['heldout_risk']:.12g} | "
            f"{'PASS' if row['heldout_nominal_risk_pass'] else 'FAIL'} | "
            f"{'PASS' if row['heldout_tangent_certificate_pass'] else 'FAIL'} | "
            f"{'PASS' if row['heldout_full_certificate_pass'] else 'FAIL'} |"
        )
    lines.extend((
        "",
        "## Decision table",
        "",
        "| Gate | Result |",
        "|---|:---:|",
    ))
    for name, passed in decisions.items():
        lines.append(f"| {name} | **{'PASS' if passed else 'FAIL'}** |")
    lines.extend((
        "",
        "## Operational amendments",
        "",
        "The 65,536 guard exposed GPU-memory limits before a Law or candidate receipt existed. Four sealed scheduling amendments progressively isolated roles and chunked the per-sample many-body feature kernel. V3.4 recorded a maximum development feature discrepancy of `5.55e-17`; scientific inputs and selection rules did not change.",
        "",
        "The two initial OOM reports remain under `provenance/operational_attempt_1` and `provenance/operational_attempt_2`.",
        "",
    ))
    atomic(REPORT, ("\n".join(lines) + "\n").encode())

    fieldnames = list(rows[0])
    descriptor, temporary = tempfile.mkstemp(prefix=f".{CSV_PATH.name}.", dir=CSV_PATH.parent)
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        data = Path(temporary).read_bytes()
        if CSV_PATH.exists() and CSV_PATH.read_bytes() != data:
            raise RuntimeError("refusing to overwrite terminal CSV")
        if not CSV_PATH.exists():
            os.replace(temporary, CSV_PATH)
            temporary = ""
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)

    files = [
        path for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path != INVENTORY
    ]
    atomic_json(INVENTORY, {
        "schema_version": 1,
        "authority_status": summary["status"],
        "artifact_count": len(files),
        "files": [{
            "path": str(path.relative_to(OUTPUT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        } for path in files],
    })
    print(json.dumps({
        "status": summary["status"],
        "rows": len(rows),
        "inventory_artifacts": len(files),
        "report": str(REPORT),
    }, indent=2))


if __name__ == "__main__":
    main()
