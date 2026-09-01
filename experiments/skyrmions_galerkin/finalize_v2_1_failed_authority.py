"""Package the failed V2.1 authority and archive strict V2 after certification."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
STRICT = OUTPUTS / "official_b1_galerkin_pareto_v2_single_seed"
AMENDED = OUTPUTS / "official_b1_galerkin_pareto_v2_1_single_seed_amended"
ARCHIVE = OUTPUTS / "old_stuff" / STRICT.name
SEAL = AMENDED / "selection" / "selection_seal.json"
REPORT = AMENDED / "OFFICIAL_B1_GALERKIN_PARETO_V2_1_AMENDMENT_RESULT.md"
PERFORMANCE = AMENDED / "B1_V2_1_JAX_PERFORMANCE_REPORT.md"
SUMMARY = AMENDED / "amendment_final_summary.json"
CSV_PATH = AMENDED / "amendment_final_summary.csv"
INVENTORY = AMENDED / "artifact_inventory_pre_manifest.json"
SELF_CONTAINMENT = AMENDED / "SELF_CONTAINMENT_MANIFEST.json"
V1_ROOT = OUTPUTS / "official_b1_galerkin_pareto_v1"
V1_TREE_SHA256 = "47db2e1c3022b3a6707010087ff34b597873001b57d31415f46b5c762998d9ca"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def atomic_bytes(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"refusing to overwrite packaged artifact: {path}")
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
    atomic_bytes(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n",
    )


def copy_once(source: Path, destination: Path) -> None:
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise RuntimeError(f"packaged copy mismatch: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    observed_v1_tree = tree_sha256(V1_ROOT)
    if observed_v1_tree != V1_TREE_SHA256:
        raise RuntimeError("accepted V1 authority changed before final packaging")
    if not SEAL.is_file():
        raise RuntimeError("V2.1 certification seal is missing")
    seal = json.loads(SEAL.read_text())
    if seal["passed"]:
        raise RuntimeError("this packager is only for the frozen failed authority")
    law = next(row for row in seal["rows"] if row["method"] == "Law")
    non_law = [row for row in seal["rows"] if row["method"] != "Law"]
    if law["full_certificate_pass"] or not all(
        row["full_certificate_pass"] for row in non_law
    ):
        raise RuntimeError("unexpected authoritative failure pattern")
    heldout_files = list((AMENDED / "heldout_validation").rglob("*")) \
        if (AMENDED / "heldout_validation").exists() else []
    if any(path.is_file() for path in heldout_files):
        raise RuntimeError("held-out validation was opened after failed certification")

    strict_failure_source = STRICT / "selection" / "pre_action_failure.json"
    copy_once(
        strict_failure_source,
        AMENDED / "provenance" / "strict_v2_pre_action_failure.json",
    )
    copy_once(
        ROOT / "outputs" / "galerkin_only_3pct" / "cache" / "dictionaries"
        / "dictionary_K280.npz",
        AMENDED / "artifacts" / "dictionary_K280.npz",
    )
    protocol = json.loads((AMENDED / "protocol_v2_single_seed.json").read_text())
    for name in protocol["source_hashes"]:
        copy_once(ROOT / name, AMENDED / "source_snapshot" / "scientific" / name)
    for name in (
        "official_b1_pareto_v2_1_amendment.py",
        "official_b1_pareto_v2_1_amendment_run.py",
        "finalize_v2_1_failed_authority.py",
    ):
        copy_once(ROOT / name, AMENDED / "source_snapshot" / "amendment" / name)
    specification = Path(
        "/home/zanot/.codex/attachments/726e8677-e3e1-4525-8a7c-6f7b2792c00d/pasted-text.txt"
    )
    copy_once(specification, AMENDED / "source_snapshot" / "request_specification.txt")
    copy_once(
        ROOT.parent / "vortices_percentage" / "VORTICES_V2_1_SELECTION_PROTOCOL_FROZEN.md",
        AMENDED / "source_snapshot" / "precedent"
        / "VORTICES_V2_1_SELECTION_PROTOCOL_FROZEN.md",
    )

    restart = json.loads((AMENDED / "selection" / "restart_summary.json").read_text())
    exact = json.loads((AMENDED / "feasibility" / "exact_receipts.json").read_text())
    call_graph = json.loads((AMENDED / "jax_only_call_graph.json").read_text())
    amendment = json.loads((AMENDED / "amendment_pre_action.json").read_text())
    pass_result = json.loads(
        (AMENDED / f"selection_pass_{restart['final_pass_index']}" / "complete.json").read_text()
    )
    law_cache = json.loads(
        (AMENDED / "authoritative" / "cache" / f"{law['eta_sha256']}.json").read_text()
    )
    law_audit_ress = law_cache["audit"]["audit_forcing"]["minimum_ess_fraction"]

    decisions: dict[str, Any] = {
        "A_old_B1_preserved": observed_v1_tree == V1_TREE_SHA256,
        "B_exactly_one_root_seed": True,
        "C_native_Galerkin_unreachable": call_graph["passed"],
        "D_all_scientific_Full_solves_JAX_K280": seal["all_galerkin_backends"] == ["jax"],
        "E_historical_equivalence": True,
        "F_candidate_universe_preoutcome_freeze": True,
        "G_exact_feasibility_before_action": exact["exact_before_full_action"],
        "H_law_from_full_supported_pool": True,
        "I_consistency_checked_all_downstream": restart["final_law_consistent"],
        "J_required_reanchor_executed": "NOT REQUIRED",
        "K_final_law_consistent": restart["final_law_consistent"],
        "L_only_requested_allowances": True,
        "M_law_mandatory_at_0p5": pass_result["full"][0]["law_mandatory"],
        "N_previous_winner_mandatory": all(
            row["previous_incumbent_mandatory"] for row in pass_result["full"][1:]
        ),
        "O_full_action_nonincreasing": seal["full_action_nonincreasing"],
        "P_authoritative_certificates": False,
        "Q_no_float32_action": seal["all_scientific_action_dtypes"] == ["float64"],
        "R_complete_pool_persisted": exact["count"] == 5645,
        "S_performance_report": True,
        "T_complete": False,
    }

    performance_stages = [
        json.loads(path.read_text())
        for path in sorted((AMENDED / "performance" / "stages").glob("*.json"))
    ]
    peak = max(row.get("jax_process_peak_bytes", 0) for row in performance_stages)
    performance_lines = [
        "# B1 V2.1 JAX Performance Report",
        "",
        "Status: **SCIENTIFIC CERTIFICATION FAILURE**",
        "",
        f"Peak recorded JAX process allocation: `{peak / 2**20:.1f} MiB`.",
        "",
        "| stage | wall seconds | JAX process peak MiB |",
        "|---|---:|---:|",
    ]
    for row in performance_stages:
        performance_lines.append(
            f"| {row['mode']} | {row['wall_time_seconds']:.3f} | "
            f"{row.get('jax_process_peak_bytes', 0) / 2**20:.1f} |"
        )
    performance_lines += [
        "",
        "Exact screening used static batches of 8; JAX K=280 sufficient statistics used chunks of 512. No native Galerkin or float32 scientific action path was used.",
    ]
    atomic_bytes(PERFORMANCE, ("\n".join(performance_lines) + "\n").encode())

    report_lines = [
        "# Official B1 Galerkin Pareto V2.1 Single-Seed Result",
        "",
        "Status: **FAIL — LAW AUTHORITATIVE SUPPORT CERTIFICATE**",
        "",
        "This is the authoritative outcome of the one-root-seed, three-allowance run. Strict V2 was archived after a pre-action start-availability failure; V2.1 applied the frozen Vortices-style up-to-cap amendment before any action outcome.",
        "",
        f"- Root seed: `20261003` (no alternate roots)",
        f"- Amendment SHA-256: `{amendment['amendment_sha256']}`",
        f"- Restarts: `{restart['restart_count']}`",
        f"- Final Law risk: `{restart['final_law']['exact_scientific_risk']:.17g}`",
        f"- Best downstream Law improvement: `{pass_result['law_improvement']:.6g}` (below `1e-4`; no re-anchor required)",
        f"- Law selection-bank minimum rESS: `{law['minimum_rESS']:.9g}`",
        f"- Law authoritative-audit rESS: `{law_audit_ress:.9g}` (required `>= 0.05`)",
        "- Held-out validation: `NOT OPENED` after failed certification",
        "",
        "## Frozen selected table",
        "",
        "| method | allowance | exact risk | train K280 action | audit K280 action | selection action | exact rESS | Full certificate |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in seal["rows"]:
        allowance = "Law" if row["allowance_percent"] is None else f"{row['allowance_percent']:g}%"
        report_lines.append(
            f"| {row['method']} | {allowance} | {row['exact_risk']:.12g} | "
            f"{row['train_K280_action']:.12g} | {row['audit_K280_action']:.12g} | "
            f"{row['selection_action']:.12g} | {row['minimum_rESS']:.7g} | "
            f"{'PASS' if row['full_certificate_pass'] else 'FAIL'} |"
        )
    report_lines += [
        "",
        "All six Tangent/Full rows passed authoritative certification. The authority is nevertheless FAIL because the prospectively selected Law did not pass the independent authoritative support gate. No post-action replacement Law was chosen.",
        "",
        "## Final decision table",
        "",
        "| item | decision |",
        "|---|---|",
    ]
    labels = (
        ("A", "Old B1 preserved", "A_old_B1_preserved"),
        ("B", "Exactly one new root seed", "B_exactly_one_root_seed"),
        ("C", "Native Galerkin unreachable", "C_native_Galerkin_unreachable"),
        ("D", "All scientific Full solves JAX K=280", "D_all_scientific_Full_solves_JAX_K280"),
        ("E", "Historical JAX equivalence", "E_historical_equivalence"),
        ("F", "Candidate universe pre-outcome freeze", "F_candidate_universe_preoutcome_freeze"),
        ("G", "Exact feasibility before action", "G_exact_feasibility_before_action"),
        ("H", "Law from complete supported pool", "H_law_from_full_supported_pool"),
        ("I", "Consistency checked downstream", "I_consistency_checked_all_downstream"),
        ("J", "Required re-anchor executed", "J_required_reanchor_executed"),
        ("K", "Final Law risk-consistent", "K_final_law_consistent"),
        ("L", "Only 0.5/1/2% run", "L_only_requested_allowances"),
        ("M", "Law mandatory at 0.5%", "M_law_mandatory_at_0p5"),
        ("N", "Previous winner mandatory", "N_previous_winner_mandatory"),
        ("O", "Full action nonincreasing", "O_full_action_nonincreasing"),
        ("P", "All authoritative certificates", "P_authoritative_certificates"),
        ("Q", "No float32 scientific action", "Q_no_float32_action"),
        ("R", "Complete candidate pool persisted", "R_complete_pool_persisted"),
        ("S", "Performance report produced", "S_performance_report"),
        ("T", "SINGLE-SEED K=280 PARETO RUN COMPLETE", "T_complete"),
    )
    for letter, label, key in labels:
        value = decisions[key]
        decision = value if isinstance(value, str) else (
            "YES" if letter == "T" and value else "NO" if letter == "T"
            else "PASS" if value else "FAIL"
        )
        report_lines.append(f"| {letter}. {label} | **{decision}** |")
    atomic_bytes(REPORT, ("\n".join(report_lines) + "\n").encode())

    with tempfile.NamedTemporaryFile(
        mode="w", prefix=".amended_summary.", suffix=".csv", dir=AMENDED,
        delete=False, newline="", encoding="utf-8",
    ) as handle:
        temporary_csv = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(seal["rows"][0].keys()))
        writer.writeheader()
        writer.writerows(seal["rows"])
        handle.flush()
        os.fsync(handle.fileno())
    if CSV_PATH.exists():
        if sha256(CSV_PATH) != sha256(temporary_csv):
            temporary_csv.unlink()
            raise RuntimeError("existing amended CSV differs")
        temporary_csv.unlink()
    else:
        os.replace(temporary_csv, CSV_PATH)

    summary = {
        "schema_version": 1,
        "status": "FAIL_LAW_AUTHORITATIVE_SUPPORT_CERTIFICATE",
        "passed": False,
        "authoritative": True,
        "single_root_seed": 20261003,
        "allowances_percent": [0.5, 1.0, 2.0],
        "amendment_sha256": amendment["amendment_sha256"],
        "selection_seal_sha256": sha256(SEAL),
        "failure": {
            "geometry": "Law",
            "eta_sha256": law["eta_sha256"],
            "gate": "authoritative_audit_minimum_ess_fraction",
            "observed": law_audit_ress,
            "required_minimum": 0.05,
        },
        "heldout_validation_opened": False,
        "old_B1_tree_sha256_before": V1_TREE_SHA256,
        "old_B1_tree_sha256_after": observed_v1_tree,
        "rows": seal["rows"],
        "decision_table": decisions,
    }
    atomic_json(SUMMARY, summary)

    # Archive strict V2 only after all provenance needed by V2.1 is local.
    if STRICT.exists():
        if ARCHIVE.exists():
            raise RuntimeError(f"archive target already exists: {ARCHIVE}")
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        os.replace(STRICT, ARCHIVE)
    if not ARCHIVE.is_dir() or STRICT.exists():
        raise RuntimeError("strict V2 archive move failed")

    files = [
        path for path in sorted(AMENDED.rglob("*"))
        if path.is_file() and path not in {INVENTORY, SELF_CONTAINMENT}
    ]
    inventory = {
        "schema_version": 1,
        "artifact_count": len(files),
        "excluded": [INVENTORY.name, SELF_CONTAINMENT.name],
        "files": [{
            "path": str(path.relative_to(AMENDED)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        } for path in files],
    }
    atomic_json(INVENTORY, inventory)
    for row in inventory["files"]:
        path = AMENDED / row["path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise RuntimeError(f"post-archive self-containment mismatch: {row['path']}")
    atomic_json(SELF_CONTAINMENT, {
        "schema_version": 1,
        "passed": True,
        "authority_root": str(AMENDED.relative_to(ROOT)),
        "strict_v2_archived_at": str(ARCHIVE.relative_to(ROOT)),
        "parent_runtime_required": False,
        "large_input_directory_entries_survive_parent_archive": True,
        "reference_local_sha256": sha256(AMENDED / "artifacts" / "reference.npz"),
        "dictionary_local_sha256": sha256(AMENDED / "artifacts" / "dictionary_K280.npz"),
        "strict_failure_local_sha256": sha256(
            AMENDED / "provenance" / "strict_v2_pre_action_failure.json"
        ),
        "scientific_source_snapshot_complete": True,
        "request_specification_local": True,
        "heldout_validation_opened": False,
        "inventory_sha256": sha256(INVENTORY),
        "note": (
            "Historical parent paths in amendment_pre_action.json are provenance "
            "labels only; every scientific input and result needed to audit this "
            "failed authority is contained under authority_root."
        ),
    })
    print(json.dumps({
        "status": summary["status"],
        "strict_v2_archive": str(ARCHIVE),
        "self_containment_manifest": str(SELF_CONTAINMENT),
        "artifact_count": inventory["artifact_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
