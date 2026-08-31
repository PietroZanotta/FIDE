"""Corrective lower-K follow-up to the failed three-Law qualification v1.

V1 established two facts on development-only banks: the physical residual was
lowest at the bottom of its K>=120 ladder, and alternate rank tolerances gave
stable scientific outputs but failed the hard algebra gate because they were
deliberately coarser solves.  V2 therefore extends the nested dictionary below
K=120.  The unchanged hard gates apply at the selected 1e-12 solve; the other
rank tolerances remain mandatory output-stability diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import three_law_qualification as base
from .pareto_v3_common import file_sha256, payload_sha256


VERSION = "skyrmion_b1_three_law_common_task_v2"
OUTPUT_ROOT = base.ROOT / "outputs" / VERSION
PROTOCOL_PATH = OUTPUT_ROOT / "protocol.json"
RESULT_PATH = OUTPUT_ROOT / "result.json"
HANDOFF_PATH = OUTPUT_ROOT / "pareto_handoff.json"
REPORT_PATH = OUTPUT_ROOT / "report.md"
V1_ROOT = base.ROOT / "outputs" / base.VERSION

K_LADDER = (40, 60, 80, 100, 120)
RANK_TOLERANCES = base.RANK_TOLERANCES
DEFAULT_RANK_TOLERANCE = base.DEFAULT_RANK_TOLERANCE


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _law_path(flow_id: str) -> Path:
    return V1_ROOT / "laws" / flow_id / "official_law.json"


def _case_path(flow_id: str) -> Path:
    return OUTPUT_ROOT / "development" / f"{flow_id}.json"


def _confirmation_path(flow_id: str) -> Path:
    return OUTPUT_ROOT / "confirmation" / f"{flow_id}.json"


def _configure_base() -> None:
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.K_LADDER = K_LADDER
    base.RANK_TOLERANCES = RANK_TOLERANCES
    base.DEFAULT_RANK_TOLERANCE = DEFAULT_RANK_TOLERANCE
    base._law_path = _law_path
    base._case_path = _case_path
    base._confirmation_path = _confirmation_path


def protocol_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    v1 = _read(V1_ROOT / "result.json")
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "status": "FROZEN_BEFORE_V2_LOWER_K_EVALUATION",
        "classification": "development-only corrective follow-up after v1 failed",
        "parent_v1_result_sha256": file_sha256(V1_ROOT / "result.json"),
        "parent_v1_status": v1["status"],
        "parent_v1_observations": {
            "lowest_tested_K": 120,
            "energy_residual_increased_with_K_for_all_three_flows": True,
            "alternate_tolerance_outputs_stable_but_coarse_algebra_invalid": True,
        },
        "flow_ids": list(base.FLOW_IDS),
        "K_ladder": list(K_LADDER),
        "rank_tolerances": list(RANK_TOLERANCES),
        "selected_rank_tolerance": DEFAULT_RANK_TOLERANCE,
        "selection_rule": (
            "smallest K whose 1e-12 solve passes every unchanged hard gate on all "
            "three Law/flow diagonal cases, whose scientific outputs are robust "
            "across the rank-tolerance ladder, and whose action/gradient are stable "
            "against the next K"
        ),
        "hard_gates": {
            "apply_to_selected_rank_tolerance": True,
            "thresholds_unchanged_from_v1": True,
            "certificate_thresholds": cfg["production_galerkin"]["certificate_thresholds"],
            "algebra_thresholds": {
                key: cfg["production_galerkin"][key]
                for key in (
                    "maximum_range_residual",
                    "maximum_stationarity_residual",
                    "maximum_identity_relerr",
                    "maximum_symmetry_residual",
                    "minimum_rank_fraction",
                    "maximum_retained_condition",
                )
            },
        },
        "rank_tolerance_robustness": {
            "hard_algebra_gate_on_alternate_tolerances": False,
            "reason": (
                "alternate tolerances are perturbation diagnostics, not candidate "
                "production solvers; require stable action, energy, and gradient"
            ),
            "maximum_action_spread": base.QUALIFICATION_THRESHOLDS[
                "rank_tolerance_action_spread"
            ],
            "maximum_energy_spread": base.QUALIFICATION_THRESHOLDS[
                "rank_tolerance_energy_spread"
            ],
            "minimum_gradient_cosine": base.QUALIFICATION_THRESHOLDS[
                "rank_tolerance_gradient_cosine_minimum"
            ],
        },
        "neighbor_stability": {
            key: value
            for key, value in base.QUALIFICATION_THRESHOLDS.items()
            if key.startswith("neighbor_")
        },
        "laws": {
            flow_id: file_sha256(_law_path(flow_id)) for flow_id in base.FLOW_IDS
        },
        "dictionary_sha256": file_sha256(base.resolution_study.DICTIONARY_PATH),
        "authoritative_confirmation": {
            "train_role": base.CONFIRMATION_TRAIN,
            "audit_role": base.CONFIRMATION_AUDIT,
            "all_three_complete_certificates_required": True,
        },
        "pareto_contract": {
            "mandatory_same_metric_Law_candidate_at_every_allowance": True,
            "fail_closed_without_handoff": True,
        },
        "source_sha256": file_sha256(Path(__file__)),
        "validation_accessed": False,
    }
    payload["protocol_sha256"] = payload_sha256(payload)
    return payload


def freeze_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    _configure_base()
    payload = protocol_payload(cfg)
    if PROTOCOL_PATH.exists():
        old = _read(PROTOCOL_PATH)
        if old != payload:
            raise RuntimeError("three-Law v2 protocol changed after freezing")
        return old
    base._atomic_json(PROTOCOL_PATH, payload)
    base._atomic_text(OUTPUT_ROOT / "protocol_hash.txt", payload["protocol_sha256"] + "\n")
    return payload


def qualify_rows(results: list[dict[str, Any]]) -> dict[str, Any]:
    settings = base.QUALIFICATION_THRESHOLDS
    candidates = []
    for K in K_LADDER:
        flow_checks = []
        for result in results:
            tolerance_rows = [row for row in result["rows"] if row["K"] == K]
            default = next(
                row
                for row in tolerance_rows
                if row["rank_tolerance"] == DEFAULT_RANK_TOLERANCE
            )
            actions = [row["train_action"] for row in tolerance_rows]
            energies = [
                row["heldout_certificate"]["maximum_energy_residual"]
                for row in tolerance_rows
            ]
            cosines = [
                base.gradient_comparison(right["gradient"], left["gradient"])["cosine"]
                for left, right in zip(tolerance_rows[:-1], tolerance_rows[1:])
            ]
            robust = bool(
                (max(actions) - min(actions))
                / max(abs(default["train_action"]), 1.0e-12)
                <= settings["rank_tolerance_action_spread"]
                and max(energies) - min(energies)
                <= settings["rank_tolerance_energy_spread"]
                and min(cosines, default=1.0)
                >= settings["rank_tolerance_gradient_cosine_minimum"]
            )
            neighbor = None
            if K != K_LADDER[-1]:
                next_K = K_LADDER[K_LADDER.index(K) + 1]
                next_row = next(
                    row
                    for row in result["rows"]
                    if row["K"] == next_K
                    and row["rank_tolerance"] == DEFAULT_RANK_TOLERANCE
                )
                neighbor = {
                    "action_relative_change": base.relative_change(
                        next_row["train_action"], default["train_action"]
                    ),
                    **base.gradient_comparison(
                        next_row["gradient"], default["gradient"]
                    ),
                }
            stable = bool(
                neighbor is None
                or (
                    neighbor["action_relative_change"]
                    <= settings["neighbor_action_relative_tolerance"]
                    and neighbor["cosine"]
                    >= settings["neighbor_gradient_cosine_minimum"]
                    and neighbor["relative_difference"]
                    <= settings["neighbor_gradient_relative_tolerance"]
                )
            )
            flow_checks.append(
                {
                    "flow_id": result["flow_id"],
                    "default_complete_certificate": default["complete_certificate"],
                    "robust_to_rank_tolerance": robust,
                    "stable_neighbor": stable,
                    "neighbor": neighbor,
                    "default_energy_residual": default["heldout_certificate"][
                        "maximum_energy_residual"
                    ],
                    "default_action": default["train_action"],
                    "default_algebra_valid": default["algebra"]["valid"],
                    "default_train_forcing_valid": default["train_forcing"]["valid"],
                    "default_audit_forcing_valid": default["audit_forcing"]["valid"],
                }
            )
        qualified = all(
            row["default_complete_certificate"]
            and row["robust_to_rank_tolerance"]
            and row["stable_neighbor"]
            for row in flow_checks
        )
        candidates.append(
            {"K": K, "qualified": qualified, "flow_checks": flow_checks}
        )
    chosen = next((row["K"] for row in candidates if row["qualified"]), None)
    return {
        "qualification_candidates": candidates,
        "recommended_K": chosen,
        "recommended_rank_tolerance": (
            DEFAULT_RANK_TOLERANCE if chosen is not None else None
        ),
        "development_qualified": chosen is not None,
    }


def run_development(cfg: dict[str, Any],
                    progress: Callable[[str], None] | None = print) -> dict[str, Any]:
    _configure_base()
    protocol = freeze_protocol(cfg)
    results = [
        base._evaluate_development_ladder(cfg, flow_id, progress)
        for flow_id in base.FLOW_IDS
    ]
    qualification = qualify_rows(results)
    payload = {
        "schema_version": 1,
        "status": "QUALIFIED" if qualification["development_qualified"] else "NOT_QUALIFIED",
        "protocol_sha256": protocol["protocol_sha256"],
        "flows": [
            {
                "flow_id": row["flow_id"],
                "law_sha256": row["law_sha256"],
                "development_sha256": file_sha256(_case_path(row["flow_id"])),
            }
            for row in results
        ],
        **qualification,
        "validation_accessed": False,
    }
    base._atomic_json(OUTPUT_ROOT / "development" / "summary.json", payload)
    return payload


def _write_report(result: dict[str, Any]) -> None:
    old_root, old_report = base.OUTPUT_ROOT, base.REPORT_PATH
    try:
        base.OUTPUT_ROOT = OUTPUT_ROOT
        base.REPORT_PATH = REPORT_PATH
        base._write_report(result)
    finally:
        base.OUTPUT_ROOT, base.REPORT_PATH = old_root, old_report


def run(cfg: dict[str, Any], progress: Callable[[str], None] | None = print) -> dict[str, Any]:
    _configure_base()
    if RESULT_PATH.exists():
        return _read(RESULT_PATH)
    protocol = freeze_protocol(cfg)
    development = run_development(cfg, progress)
    confirmation = base.run_confirmation(cfg, development, progress)
    passed = bool(development["development_qualified"] and confirmation["passed"])
    result = {
        "schema_version": 1,
        "version": VERSION,
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "protocol_sha256": protocol["protocol_sha256"],
        "development": development,
        "confirmation": confirmation,
        "parent_v1_result_sha256": file_sha256(V1_ROOT / "result.json"),
        "validation_accessed": False,
    }
    base._atomic_json(RESULT_PATH, result)
    if passed:
        handoff = {
            "schema_version": 1,
            "status": "FROZEN_FOR_PARETO",
            "qualification_result_sha256": file_sha256(RESULT_PATH),
            "protocol_sha256": protocol["protocol_sha256"],
            "K": confirmation["K"],
            "rank_tolerance": confirmation["rank_tolerance"],
            "dictionary_sha256": protocol["dictionary_sha256"],
            "laws": {
                row["flow_id"]: {
                    "eta": row["eta"],
                    "risk": row["scientific_risk"],
                    "law_sha256": row["law_sha256"],
                    "confirmation_sha256": file_sha256(
                        _confirmation_path(row["flow_id"])
                    ),
                }
                for row in confirmation["rows"]
            },
            "mandatory_candidate_rule": (
                "include Law_seed in every allowance and select by the same Full metric"
            ),
            "validation_accessed": False,
        }
        base._atomic_json(HANDOFF_PATH, handoff)
    _write_report(result)
    return result


__all__ = [
    "DEFAULT_RANK_TOLERANCE",
    "HANDOFF_PATH",
    "K_LADDER",
    "OUTPUT_ROOT",
    "PROTOCOL_PATH",
    "RANK_TOLERANCES",
    "freeze_protocol",
    "qualify_rows",
    "run",
    "run_development",
]
