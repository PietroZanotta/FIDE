"""Independent, seed-addressable B1 Galerkin Pareto studies.

Each reference-flow seed is a complete single-reference study over the frozen
0.5%, 1%, 2%, 3%, 4%, and 5% risk allowances.  Seeds use isolated output roots
but identical deterministic selection seeds, so additional reference flows can
be evaluated later without rerunning completed seeds.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import official_b1_pareto as single
from . import three_reference_pareto as engine
from .pareto_v3_common import file_sha256


ROOT = Path(__file__).resolve().parent
VERSION = "skyrmion_b1_galerkin_pareto_per_seed_v1"
OUTPUT_BASE = ROOT / "outputs" / VERSION
FLOW_PATHS = dict(engine.FLOW_PATHS)
FLOW_SHA256 = dict(engine.FLOW_SHA256)
SUPPORTED_SEEDS = tuple(FLOW_PATHS)
GLOBAL_SEED = 20260901
_BASE_PROTOCOL_PAYLOAD = engine.protocol_payload


def seed_output_root(seed_id: str) -> Path:
    if seed_id not in SUPPORTED_SEEDS:
        raise ValueError(
            f"unknown seed {seed_id!r}; choose one of {', '.join(SUPPORTED_SEEDS)}"
        )
    return OUTPUT_BASE / seed_id


def _seed_protocol_payload(cfg: dict[str, Any], seed_id: str) -> dict[str, Any]:
    payload = _BASE_PROTOCOL_PAYLOAD(cfg)
    payload["version"] = VERSION
    payload["study_type"] = "independent single-reference B1 Pareto selection"
    payload["reference"].update({
        "flow_ids": [seed_id],
        "checkpoint_sha256": {seed_id: FLOW_SHA256[seed_id]},
        "independently_runnable_seed": True,
        "future_seed_aggregation_requires_rerun": False,
    })
    payload["source_hashes"]["per_seed_pareto.py"] = file_sha256(Path(__file__))
    payload["source_hashes"]["run_per_seed_pareto.py"] = file_sha256(
        ROOT / "run_per_seed_pareto.py"
    )
    payload["source_hashes"]["eval.py"] = file_sha256(ROOT / "eval.py")
    payload["source_hashes"]["visualize_per_seed_pareto.py"] = file_sha256(
        ROOT / "visualize_per_seed_pareto.py"
    )
    payload["constants"].update({
        "flow_count": 1,
        "objective": "single-flow action",
    })
    payload["law"]["selection_objective"] = f"scientific risk for {seed_id}"
    payload["risk_rule"] = "R_seed(eta) <= (1+p/100) R_seed(Law)"
    payload["full_method"] = "single-flow fixed-feature K=280 Galerkin approximation"
    payload["seed_composition"] = {
        "mode": "independent additive runs",
        "current_seed": seed_id,
        "supported_seeds": list(SUPPORTED_SEEDS),
        "shared_allowances_percent": list(engine.ALLOWANCES),
        "aggregation_rule": "join completed per-seed receipts by allowance",
    }
    return payload


def configure_seed(seed_id: str) -> Path:
    """Point the reusable selection engine at one isolated seed root."""
    output_root = seed_output_root(seed_id)
    engine.VERSION = VERSION
    engine.OUTPUT_ROOT = output_root
    engine.GLOBAL_SEED = GLOBAL_SEED
    engine.FLOW_IDS = (seed_id,)
    engine.FLOW_PATHS = {seed_id: FLOW_PATHS[seed_id]}
    engine.FLOW_SHA256 = {seed_id: FLOW_SHA256[seed_id]}
    path_names = {
        "PROTOCOL_PATH": output_root / "protocol.json",
        "PROTOCOL_HASH_PATH": output_root / "protocol_hash.txt",
        "DESIGN_PATH": output_root / "design_truth" / "design_truth.npz",
        "DESIGN_RECORD": output_root / "design_truth" / "manifest.json",
        "ARTIFACT_DIR": output_root / "artifacts",
        "LAW_PATH": output_root / "law" / "official_law.json",
        "SCREENING_PATH": output_root / "screening" / "candidate_pool.json",
        "FINAL_PARETO": output_root / "pareto.json",
    }
    for name, value in path_names.items():
        setattr(engine, name, value)
    for index in range(1, 13):
        suffix = "" if index == 1 else f"_{index}"
        setattr(
            engine,
            "RUNTIME_PATCH_RECEIPT" if index == 1 else f"RUNTIME_PATCH_RECEIPT_{index}",
            output_root / f"runtime_implementation_patch{suffix}.json",
        )
    engine._SHARED_SELECTION.clear()
    engine._FLOW_SELECTION_SHARED.clear()
    engine.protocol_payload = lambda cfg: _seed_protocol_payload(cfg, seed_id)
    return output_root


def finalize_seed(cfg: dict[str, Any], seed_id: str) -> dict[str, Any]:
    output_root = configure_seed(seed_id)
    engine._activate()
    engine.freeze_runtime_patch()
    path = output_root / "pareto.json"
    if path.exists():
        return single.read_json(path)
    screening = single.read_json(output_root / "screening" / "candidate_pool.json")
    tangent = single.read_json(output_root / "tangent" / "selection.json")
    full = single.read_json(output_root / "full_search" / "selection.json")
    law_risk = float(screening["law_risk_by_flow"][seed_id])
    rows = []
    for index, allowance in enumerate(engine.ALLOWANCES):
        ceiling = (1.0 + allowance / 100.0) * law_risk
        tangent_winner = tangent["allowances"][index]["winner"]
        full_winner = full["allowances"][index]["winner"]
        row = {
            "allowance_percent": allowance,
            "risk_ceiling": ceiling,
            "Law": {
                "eta": screening["law_eta"],
                "risk": law_risk,
                "tangent_action": None,
                "full_action": None,
                "status": "REFERENCE",
            },
            "Tangent": None,
            "Full": None,
        }
        if tangent_winner is not None:
            row["Tangent"] = {
                "eta": tangent_winner["eta"],
                "risk": float(tangent_winner["risk_by_flow"][seed_id]),
                "tangent_action": float(tangent_winner["action"]),
                "full_action": None,
                "status": "CERTIFIED",
            }
        if full_winner is not None:
            row["Full"] = {
                "eta": full_winner["eta"],
                "risk": float(full_winner["risk_by_flow"][seed_id]),
                "tangent_action": None,
                "full_action": float(full_winner["action"]),
                "status": "CERTIFIED",
            }
        rows.append(row)
    tangent_gaps = sum(row["Tangent"] is None for row in rows)
    full_gaps = sum(row["Full"] is None for row in rows)
    result = {
        "schema_version": 1,
        "version": VERSION,
        "seed_id": seed_id,
        "mode": "single-reference multi-risk Pareto",
        "status": "COMPLETE" if tangent_gaps + full_gaps == 0 else "COMPLETE_WITH_GAPS",
        "tangent_gap_count": tangent_gaps,
        "full_gap_count": full_gaps,
        "protocol_sha256": engine.require_protocol(cfg)["protocol_sha256"],
        "runtime_patch_sha256": file_sha256(engine.RUNTIME_PATCH_RECEIPT_12),
        "allowances": rows,
        "validation_accessed": False,
        "deep_ritz_used": False,
    }
    single.atomic_json(path, result)
    with (output_root / "pareto.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("allowance_percent", "method", "risk", "risk_ceiling", "tangent_action", "full_action", "status"))
        for row in rows:
            for method in ("Law", "Tangent", "Full"):
                value = row[method]
                writer.writerow((
                    row["allowance_percent"], method,
                    None if value is None else value["risk"], row["risk_ceiling"],
                    None if value is None else value["tangent_action"],
                    None if value is None else value["full_action"],
                    "NO CERTIFIED POINT" if value is None else value["status"],
                ))
    lines = [
        f"# B1 Galerkin Pareto — {seed_id}", "",
        f"Status: {result['status']}", "",
        "Independent single-reference run over six risk allowances. Future seeds can be run independently and joined by allowance.", "",
        "Each ceiling is `(1 + p/100) * R_seed(Law)`; a candidate below Law has a negative risk change and remains valid.", "",
        "| allowance | method | risk | ceiling | Tangent action | Full K280 action | status |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        for method in ("Law", "Tangent", "Full"):
            value = row[method]
            if value is None:
                lines.append(f"| {row['allowance_percent']}% | {method} | — | {row['risk_ceiling']:.9g} | — | — | NO CERTIFIED POINT |")
            else:
                tangent_action = "—" if value["tangent_action"] is None else f"{value['tangent_action']:.9g}"
                full_action = "—" if value["full_action"] is None else f"{value['full_action']:.9g}"
                lines.append(f"| {row['allowance_percent']}% | {method} | {value['risk']:.9g} | {row['risk_ceiling']:.9g} | {tangent_action} | {full_action} | {value['status']} |")
    single.atomic_text(output_root / "report.md", "\n".join(lines) + "\n")
    evaluation_rows = []
    for row in rows:
        allowance = float(row["allowance_percent"])
        methods = {}
        failures = []
        for method in ("Tangent", "Full"):
            value = row[method]
            certified = bool(
                value is not None
                and value.get("status") == "CERTIFIED"
                and math.isfinite(float(value["risk"]))
                and float(value["risk"]) <= float(row["risk_ceiling"]) + 1e-12
                and math.isfinite(float(
                    value["tangent_action"] if method == "Tangent" else value["full_action"]
                ))
            )
            methods[method] = {
                "certified": certified,
                "risk": None if value is None else value["risk"],
                "risk_ceiling": row["risk_ceiling"],
                "risk_change_vs_law_percent": (
                    None if value is None else
                    100.0 * (float(value["risk"]) / law_risk - 1.0)
                ),
                "action": None if value is None else (
                    value["tangent_action"] if method == "Tangent" else value["full_action"]
                ),
            }
            if not certified:
                failures.append(f"{method} has no certified point")
        receipt = {
            "schema_version": 1,
            "seed_id": seed_id,
            "allowance_percent": allowance,
            "classification": "PASS" if not failures else "FAIL",
            "law_risk": law_risk,
            "risk_ceiling": row["risk_ceiling"],
            "methods": methods,
            "failures": failures,
            "selection_only": True,
            "validation_accessed": False,
        }
        evaluation_rows.append(receipt)
        directory = output_root / "evaluations" / f"allowance_{single.slug(allowance)}"
        single.atomic_json(directory / "eval.json", receipt)
        eval_lines = [
            f"# {seed_id} — {allowance:g}% risk evaluation", "",
            f"Classification: {receipt['classification']}", "",
            f"Law risk: {law_risk:.9g}", "",
            f"Risk ceiling: {row['risk_ceiling']:.9g}", "",
            "| method | certified | risk | change vs Law | action |", "|---|---|---:|---:|---:|",
        ]
        for method in ("Tangent", "Full"):
            method_row = methods[method]
            risk_text = "—" if method_row["risk"] is None else f"{method_row['risk']:.9g}"
            change_text = "—" if method_row["risk_change_vs_law_percent"] is None else f"{method_row['risk_change_vs_law_percent']:.4f}%"
            action_text = "—" if method_row["action"] is None else f"{method_row['action']:.9g}"
            eval_lines.append(f"| {method} | {method_row['certified']} | {risk_text} | {change_text} | {action_text} |")
        single.atomic_text(directory / "eval.md", "\n".join(eval_lines) + "\n")
    evaluation_summary = {
        "schema_version": 1,
        "seed_id": seed_id,
        "status": "PASS" if all(row["classification"] == "PASS" for row in evaluation_rows) else "FAIL",
        "allowance_count": len(evaluation_rows),
        "passed_allowance_count": sum(row["classification"] == "PASS" for row in evaluation_rows),
        "rows": evaluation_rows,
        "validation_accessed": False,
    }
    single.atomic_json(output_root / "evaluations" / "summary.json", evaluation_summary)
    return result


def run_seed_stage(
    cfg: dict[str, Any], seed_id: str, stage: str,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    configure_seed(seed_id)
    if stage == "finalize":
        return finalize_seed(cfg, seed_id)
    if stage == "all":
        result: dict[str, Any] = {}
        for name in ("protocol", "data", "law", "candidates", "screen", "tangent", "full"):
            result = engine.run_stage(cfg, name, progress)
        return finalize_seed(cfg, seed_id)
    return engine.run_stage(cfg, stage, progress)
