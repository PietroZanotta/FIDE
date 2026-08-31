#!/usr/bin/env python3
"""Read-only integrity and headline-value checks for the published V2.1 result."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PUBLISHED = HERE / "outputs" / "published"
MANIFEST = PUBLISHED / "manifest.json"
DATA = PUBLISHED / "pareto_data.json"
PLOT_DATA = HERE / "plots" / "pareto_0p5_to_2pct_data.json"
EXPECTED_REDUCTIONS = {0.5: 8.276509445046583, 1.0: 12.195980824405499, 2.0: 15.713080350957208}
EXPECTED_RISK_CHANGES = {0.5: 0.3034405594846297, 1.0: 0.8498042944794334, 2.0: 1.5992462770772586}
REQUIRED_MEDIA = (
    "plots/pareto_0p5_to_2pct.png",
    "plots/pareto_methods_full_action_risk_0p5_to_2pct.png",
    "plots/pareto_frontier_3panel_0p5_to_2pct.png",
    "plots/vortices_v2_1_full_0p5_paper.png",
    "plots/vortices_v2_1_full_1p0_paper.png",
    "plots/vortices_v2_1_full_2p0_paper.png",
    "plots/vortices_v2_1_full_2p0.gif",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify() -> dict[str, Any]:
    checks: list[str] = []
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE_STANDALONE_PUBLICATION_BUNDLE" or manifest.get("scientific_execution_required_for_visualization") is not False:
        raise RuntimeError("publication-bundle manifest is not complete")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("publication-bundle manifest has no file inventory")
    for relative, expected in files.items():
        path = HERE / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"publication-bundle integrity failure: {relative}")
    checks.append("standalone publication-bundle hashes")

    payload = json.loads(DATA.read_text(encoding="utf-8"))
    if payload["allowances_in_scope"] != [0.5, 1.0, 2.0] or payload["allowances_paused"] != [3.0, 4.0, 5.0]:
        raise RuntimeError("published allowance scope changed")
    expected_receipt_hashes = {
        "primary_action_receipt_sha256": files["outputs/published/exact_action_evaluation_receipt.json"],
        "primary_inference_sha256": files["outputs/published/simultaneous_inference.json"],
        "primary_risk_receipt_sha256": files["outputs/published/finite_risk_evaluation_receipt.json"],
        "relative_metrics_receipt_sha256": files["outputs/published/relative_action_metrics_receipt.json"],
        "tangent_supplement_receipt_sha256": files["outputs/published/tangent_evaluation_receipt.json"],
    }
    if payload.get("inputs") != expected_receipt_hashes:
        raise RuntimeError("published Pareto data are not linked to the exposed receipts")
    full = {float(row["risk_allowance_percent"]): row for row in payload["records"] if row["method"] == "full"}
    if set(full) != set(EXPECTED_REDUCTIONS):
        raise RuntimeError("published Full records are incomplete")
    for allowance, expected in EXPECTED_REDUCTIONS.items():
        row = full[allowance]
        if abs(float(row["validation_full_action_reduction_vs_law_percent"]) - expected) > 1e-12:
            raise RuntimeError(f"Full-action reduction changed at {allowance:g}%")
        if abs(float(row["validation_R_change_vs_law_percent"]) - EXPECTED_RISK_CHANGES[allowance]) > 1e-12:
            raise RuntimeError(f"holdout risk change changed at {allowance:g}%")
        if not row["certified"]:
            raise RuntimeError(f"uncertified Full result at {allowance:g}%")
    checks.append("published Pareto coordinates")

    holdout_receipt = json.loads((PUBLISHED / "holdout_bank_receipt.json").read_text(encoding="utf-8"))
    reference_receipt = json.loads((PUBLISHED / "reference_qualification_receipt.json").read_text(encoding="utf-8"))
    geometry_receipt = json.loads((PUBLISHED / "selection_geometries.json").read_text(encoding="utf-8"))
    pause_receipt = json.loads((PUBLISHED / "selection_pause_receipt.json").read_text(encoding="utf-8"))
    if holdout_receipt.get("status") != "FROZEN_SHARED_C3_64_CONFIRMATORY_BANK" or holdout_receipt.get("bank_sha256") != files["inputs/visualization_holdout_bank.npz"]:
        raise RuntimeError("exposed holdout bank is not the frozen C3-64 bank")
    if reference_receipt.get("status") != "PASS" or reference_receipt.get("rollout_bank_sha256") != files["inputs/visualization_reference_bank.npz"]:
        raise RuntimeError("exposed reference rollout is not the qualified rollout")
    geometries = geometry_receipt.get("rows", [])
    if geometry_receipt.get("status") != "FROZEN_VISUALIZATION_GEOMETRY_ADAPTER" or [row.get("allowance_percent") for row in geometries] != [0.5, 1.0, 2.0] or any(len(row.get("full_centers", [])) != 4 for row in geometries):
        raise RuntimeError("published sensor geometries are incomplete")
    if pause_receipt.get("status") != "PAUSED_BY_USER_AFTER_2PCT_PARTIAL_PARETO" or pause_receipt.get("completed_allowance_percentages") != [0.5, 1.0, 2.0]:
        raise RuntimeError("published selection boundary changed")
    checks.append("frozen bank, reference, geometry, and scope receipts")

    if PLOT_DATA.is_file() and json.loads(PLOT_DATA.read_text(encoding="utf-8")) != payload:
        raise RuntimeError("rendered plot-data copy differs from the published source")
    missing = [relative for relative in REQUIRED_MEDIA if not (HERE / relative).is_file()]
    if missing:
        raise RuntimeError(f"published media missing: {missing}")
    pareto_manifest = json.loads((HERE / "plots" / "pareto_0p5_to_2pct_manifest.json").read_text(encoding="utf-8"))
    for name, expected in pareto_manifest.get("files", {}).items():
        path = HERE / "plots" / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Pareto-render integrity failure: {name}")
    static_manifest = json.loads((HERE / "plots" / "static_visualization_manifest.json").read_text(encoding="utf-8"))
    for artifact in static_manifest.get("artifacts", []):
        for kind in ("png", "pdf"):
            path = HERE.parent.parent / artifact[kind]
            if not path.is_file() or sha256_file(path) != artifact[f"{kind}_sha256"]:
                raise RuntimeError(f"static-render integrity failure: {artifact[kind]}")
    gif_manifest = json.loads((HERE / "plots" / "vortices_v2_1_full_2p0_manifest.json").read_text(encoding="utf-8"))
    for kind in ("gif", "preview"):
        path = HERE.parent.parent / gif_manifest[kind]
        if not path.is_file() or sha256_file(path) != gif_manifest[f"{kind}_sha256"]:
            raise RuntimeError(f"animation integrity failure: {gif_manifest[kind]}")
    checks.append("published figures and animation hashes")
    return {
        "status": "PASS",
        "allowances_verified_percent": sorted(full),
        "holdout_full_action_reduction_percent": {str(key): EXPECTED_REDUCTIONS[key] for key in sorted(EXPECTED_REDUCTIONS)},
        "checks": checks,
        "scientific_execution_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the complete machine-readable receipt")
    args = parser.parse_args()
    result = verify()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        reductions = ", ".join(f"{key}%: {value:.2f}%" for key, value in result["holdout_full_action_reduction_percent"].items())
        print(f"PASS — saved V2.1 result verified ({reductions})")


if __name__ == "__main__":
    main()
