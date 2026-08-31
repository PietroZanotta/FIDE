"""Support-robust refreeze and common-task qualification of the three B1 Laws."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import three_law_qualification as base
from . import three_law_qualification_v2 as v2
from . import three_law_qualification_v3 as v3
from .pareto_v3_common import file_sha256, payload_sha256


VERSION = "skyrmion_b1_three_law_common_task_v4"
OUTPUT_ROOT = base.ROOT / "outputs" / VERSION
PROTOCOL_PATH = OUTPUT_ROOT / "protocol.json"
RESULT_PATH = OUTPUT_ROOT / "result.json"
HANDOFF_PATH = OUTPUT_ROOT / "pareto_handoff.json"
REPORT_PATH = OUTPUT_ROOT / "report.md"

K_LADDER = (80, 100, 120)
RANK_TOLERANCES = base.RANK_TOLERANCES
DEFAULT_RANK_TOLERANCE = base.DEFAULT_RANK_TOLERANCE
LAW_SUPPORT_ROLES = ("screen", "search_train", "periodic_audit", "authoritative_train")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _law_path(flow_id: str) -> Path:
    return OUTPUT_ROOT / "laws" / flow_id / "official_law.json"


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
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "status": "FROZEN_BEFORE_SUPPORT_ROBUST_LAW_RESCREEN",
        "classification": "Law-freeze correction after v3 support/forcing failures",
        "parents": {
            "v1_result_sha256": file_sha256(base.ROOT / "outputs" / base.VERSION / "result.json"),
            "v2_result_sha256": file_sha256(v2.RESULT_PATH),
            "v3_result_sha256": file_sha256(v3.RESULT_PATH),
        },
        "flow_ids": list(base.FLOW_IDS),
        "law_refreeze": {
            "candidate_source": "each flow's frozen 24-candidate risk-anchor shortlist from v1",
            "support_roles": list(LAW_SUPPORT_ROLES),
            "every_role_must_pass_original_projection_ESS_forcing_covariance_gate": True,
            "selection_rule": "minimum frozen risk-anchor risk, then candidate_id",
            "authoritative_audit_used_for_selection": False,
        },
        "development_qualification": {
            "K_ladder": list(K_LADDER),
            "rank_tolerances": list(RANK_TOLERANCES),
            "selected_rank_tolerance": DEFAULT_RANK_TOLERANCE,
            "selected_solve_must_pass_every_original_hard_gate": True,
            "alternate_tolerances_are_output_stability_diagnostics": True,
            "neighbor_stability_thresholds": {
                key: value
                for key, value in base.QUALIFICATION_THRESHOLDS.items()
                if key.startswith("neighbor_")
            },
        },
        "authoritative_confirmation": {
            "train_role": base.CONFIRMATION_TRAIN,
            "audit_role": base.CONFIRMATION_AUDIT,
            "complete_original_certificate_required_for_all_three": True,
        },
        "pareto_contract": {
            "release_handoff_only_after_all_three_confirm": True,
            "mandatory_same_metric_Law_candidate_at_every_allowance": True,
        },
        "dictionary_sha256": file_sha256(base.resolution_study.DICTIONARY_PATH),
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
            raise RuntimeError("three-Law v4 protocol changed after freezing")
        return old
    base._atomic_json(PROTOCOL_PATH, payload)
    base._atomic_text(OUTPUT_ROOT / "protocol_hash.txt", payload["protocol_sha256"] + "\n")
    return payload


def _source_shortlist(flow_id: str) -> list[dict[str, Any]]:
    path = base.ROOT / "outputs" / base.VERSION / "laws" / flow_id / "search_results.json"
    return _read(path)["shortlist"]


def refreeze_law(cfg: dict[str, Any], flow_id: str,
                 progress: Callable[[str], None] | None = print) -> dict[str, Any]:
    destination = _law_path(flow_id)
    if destination.exists():
        return _read(destination)
    base._activate_source(flow_id)
    shortlist = _source_shortlist(flow_id)
    etas = np.asarray([row["eta"] for row in shortlist], dtype=np.float64)
    evaluations = {}
    for role in LAW_SUPPORT_ROLES:
        if progress:
            progress(f"Law support screen {flow_id}/{role}: {len(shortlist)} candidates")
        result = base.ensemble.evaluate_references(etas, cfg, role)
        evaluations[role] = {
            "support_valid": np.asarray(result["support_valid"], dtype=bool),
            "minimum_rESS": np.asarray(result["minimum_ress"], dtype=np.float64),
            "maximum_forcing_mean": np.asarray(
                result["per_flow_maximum_forcing_mean"], dtype=np.float64
            )[0],
        }
    rows = []
    for index, source in enumerate(shortlist):
        role_rows = {
            role: {
                "support_valid": bool(evaluations[role]["support_valid"][index]),
                "minimum_rESS": float(evaluations[role]["minimum_rESS"][index]),
                "maximum_forcing_mean": float(
                    evaluations[role]["maximum_forcing_mean"][index]
                ),
            }
            for role in LAW_SUPPORT_ROLES
        }
        rows.append(
            {
                **source,
                "support_by_role": role_rows,
                "support_robust": all(
                    row["support_valid"] for row in role_rows.values()
                ),
            }
        )
    eligible = [row for row in rows if row["support_robust"]]
    if not eligible:
        result = {
            "schema_version": 1,
            "status": "NO_SUPPORT_ROBUST_LAW",
            "flow_id": flow_id,
            "candidate_count": len(rows),
            "eligible_count": 0,
            "rows": rows,
            "validation_accessed": False,
        }
        base._atomic_json(destination.parent / "support_screen.json", result)
        raise RuntimeError(f"no support-robust Law candidate for {flow_id}")
    winner = min(
        eligible, key=lambda row: (float(row["anchor_risk"]), row["candidate_id"])
    )
    source_law = _read(base.ROOT / "outputs" / base.VERSION / "laws" / flow_id / "official_law.json")
    law = {
        "schema_version": 1,
        "status": "FROZEN_SUPPORT_ROBUST",
        "flow_id": flow_id,
        "eta_Law_official": winner["eta"],
        "R_Law_official": float(winner["anchor_risk"]),
        "checkpoint_sha256": source_law["checkpoint_sha256"],
        "design_truth_sha256": source_law["design_truth_sha256"],
        "source_v1_law_sha256": file_sha256(
            base.ROOT / "outputs" / base.VERSION / "laws" / flow_id / "official_law.json"
        ),
        "source_shortlist_sha256": payload_sha256(shortlist),
        "selection_roles": list(LAW_SUPPORT_ROLES),
        "authoritative_audit_used_for_selection": False,
        "selection_provenance": winner,
    }
    base._atomic_json(
        destination.parent / "support_screen.json",
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "flow_id": flow_id,
            "candidate_count": len(rows),
            "eligible_count": len(eligible),
            "winner_candidate_id": winner["candidate_id"],
            "rows": rows,
            "validation_accessed": False,
        },
    )
    base._atomic_json(destination, law)
    return law


def refreeze_all_laws(cfg: dict[str, Any],
                      progress: Callable[[str], None] | None = print) -> dict[str, Any]:
    freeze_protocol(cfg)
    laws = {flow_id: refreeze_law(cfg, flow_id, progress) for flow_id in base.FLOW_IDS}
    result = {
        "schema_version": 1,
        "status": "COMPLETE",
        "laws": {
            flow_id: {
                "eta": law["eta_Law_official"],
                "risk": law["R_Law_official"],
                "sha256": file_sha256(_law_path(flow_id)),
            }
            for flow_id, law in laws.items()
        },
        "validation_accessed": False,
    }
    base._atomic_json(OUTPUT_ROOT / "laws" / "summary.json", result)
    return result


def run_development(cfg: dict[str, Any],
                    progress: Callable[[str], None] | None = print) -> dict[str, Any]:
    _configure_base()
    protocol = freeze_protocol(cfg)
    refreeze_all_laws(cfg, progress)
    results = [
        base._evaluate_development_ladder(cfg, flow_id, progress)
        for flow_id in base.FLOW_IDS
    ]
    qualification = v2.qualify_rows(results)
    payload = {
        "schema_version": 1,
        "status": "QUALIFIED" if qualification["development_qualified"] else "NOT_QUALIFIED",
        "protocol_sha256": protocol["protocol_sha256"],
        **qualification,
        "flows": [
            {
                "flow_id": row["flow_id"],
                "law_sha256": row["law_sha256"],
                "development_sha256": file_sha256(_case_path(row["flow_id"])),
            }
            for row in results
        ],
        "validation_accessed": False,
    }
    base._atomic_json(OUTPUT_ROOT / "development" / "summary.json", payload)
    return payload


def _write_report(result: dict[str, Any]) -> None:
    lines = [
        "# Support-robust three-Law common-task qualification",
        "",
        f"Status: **{result['status']}**",
        "",
        "Each Law was refrozen from its original risk-anchor shortlist after "
        "passing the original support/forcing gates on screen, search-train, "
        "periodic-audit, and authoritative-train banks. Authoritative-audit was "
        "not used for selection.",
        "",
        f"Common setting: K={result['confirmation'].get('K')}, rank tolerance="
        f"{result['confirmation'].get('rank_tolerance')}.",
        "",
        "| flow | Law risk | Full action | train rESS | audit rESS | forcing | algebra | energy | complete |",
        "|:--|---:|---:|---:|---:|:---:|:---:|---:|:---:|",
    ]
    for row in result["confirmation"]["rows"]:
        lines.append(
            f"| {row['flow_id']} | {row['scientific_risk']:.9g} | {row['train_action']:.9g} | "
            f"{row['train_forcing']['minimum_ess_fraction']:.6g} | "
            f"{row['audit_forcing']['minimum_ess_fraction']:.6g} | "
            f"{'PASS' if row['train_forcing']['valid'] and row['audit_forcing']['valid'] else 'FAIL'} | "
            f"{'PASS' if row['algebra']['valid'] else 'FAIL'} | "
            f"{row['heldout_certificate']['maximum_energy_residual']:.6g} | "
            f"{'PASS' if row['complete_certificate'] else 'FAIL'} |"
        )
    base._atomic_text(REPORT_PATH, "\n".join(lines) + "\n")


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
                    "same_metric_Law_full_action": row["train_action"],
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
    "HANDOFF_PATH",
    "K_LADDER",
    "LAW_SUPPORT_ROLES",
    "OUTPUT_ROOT",
    "PROTOCOL_PATH",
    "freeze_protocol",
    "refreeze_all_laws",
    "refreeze_law",
    "run",
    "run_development",
]
