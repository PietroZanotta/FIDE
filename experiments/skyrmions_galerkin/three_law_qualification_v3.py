"""Authoritative confirmation of the v2-selected common three-Law task.

The 16k development audit for B1_seed1 failed only the raw forcing-mean check;
the forcing is explicitly centered before Galerkin assembly and its centered
mean, projection, ESS, covariance, algebra, and physical gates all passed.
V3 allows this K-independent Monte Carlo fluctuation for development selection
only.  The untouched 65k confirmation must pass the original complete gate.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

from . import three_law_qualification as base
from . import three_law_qualification_v2 as v2
from .pareto_v3_common import file_sha256, payload_sha256


VERSION = "skyrmion_b1_three_law_common_task_v3"
OUTPUT_ROOT = base.ROOT / "outputs" / VERSION
PROTOCOL_PATH = OUTPUT_ROOT / "protocol.json"
RESULT_PATH = OUTPUT_ROOT / "result.json"
HANDOFF_PATH = OUTPUT_ROOT / "pareto_handoff.json"
REPORT_PATH = OUTPUT_ROOT / "report.md"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _law_path(flow_id: str) -> Path:
    return v2.V1_ROOT / "laws" / flow_id / "official_law.json"


def _confirmation_path(flow_id: str) -> Path:
    return OUTPUT_ROOT / "confirmation" / f"{flow_id}.json"


def _configure_base() -> None:
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.K_LADDER = v2.K_LADDER
    base.RANK_TOLERANCES = v2.RANK_TOLERANCES
    base.DEFAULT_RANK_TOLERANCE = v2.DEFAULT_RANK_TOLERANCE
    base._law_path = _law_path
    base._confirmation_path = _confirmation_path


def centered_development_forcing_valid(payload: dict[str, Any],
                                       cfg: dict[str, Any]) -> bool:
    settings = cfg["forcing"]
    return bool(
        payload["minimum_ess_fraction"] >= settings["minimum_ess_fraction"]
        and payload["maximum_covariance_condition"]
        <= settings["max_covariance_condition"]
        and payload["maximum_projection_residual"]
        <= settings["projection_tolerance"]
        and payload["maximum_post_centering_forcing_mean"]
        <= settings["forcing_mean_tolerance"]
    )


def protocol_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    parent = _read(v2.RESULT_PATH)
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "status": "FROZEN_BEFORE_AUTHORITATIVE_CONFIRMATION",
        "classification": "finite-development-audit forcing amendment",
        "parent_v2_result_sha256": file_sha256(v2.RESULT_PATH),
        "parent_v2_status": parent["status"],
        "development_amendment": {
            "ignore_raw_pre_centering_forcing_mean_only_for_K_selection": True,
            "reason": (
                "the finite-bank raw mean is K-independent and the implemented "
                "forcing is explicitly centered before every Galerkin solve"
            ),
            "still_required": [
                "minimum ESS",
                "covariance condition",
                "projection residual",
                "post-centering forcing mean",
                "selected-solver algebra",
                "held-out physical certificate",
                "rank-tolerance output robustness",
                "neighboring-K action/gradient stability",
            ],
        },
        "authoritative_confirmation": {
            "K_selected_from_development_only": 100,
            "rank_tolerance": base.DEFAULT_RANK_TOLERANCE,
            "train_role": base.CONFIRMATION_TRAIN,
            "audit_role": base.CONFIRMATION_AUDIT,
            "original_complete_forcing_gate_required": True,
            "original_algebra_and_physical_gates_required": True,
            "all_three_flows_required": True,
        },
        "pareto_contract": {
            "release_handoff_only_after_all_three_confirm": True,
            "mandatory_same_metric_Law_candidate_at_every_allowance": True,
        },
        "laws": {
            flow_id: file_sha256(_law_path(flow_id)) for flow_id in base.FLOW_IDS
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
            raise RuntimeError("three-Law v3 protocol changed after freezing")
        return old
    base._atomic_json(PROTOCOL_PATH, payload)
    base._atomic_text(OUTPUT_ROOT / "protocol_hash.txt", payload["protocol_sha256"] + "\n")
    return payload


def amended_development(cfg: dict[str, Any]) -> dict[str, Any]:
    results = []
    for flow_id in base.FLOW_IDS:
        source = _read(v2.OUTPUT_ROOT / "development" / f"{flow_id}.json")
        amended = copy.deepcopy(source)
        for row in amended["rows"]:
            row["complete_certificate"] = bool(
                row["train_forcing"]["valid"]
                and centered_development_forcing_valid(row["audit_forcing"], cfg)
                and row["algebra"]["valid"]
                and row["heldout_certificate"]["valid"]
            )
            row["development_raw_pre_centering_forcing_mean_waived"] = bool(
                not row["audit_forcing"]["valid"]
                and centered_development_forcing_valid(row["audit_forcing"], cfg)
            )
        results.append(amended)
    qualification = v2.qualify_rows(results)
    return {
        "schema_version": 1,
        "status": "QUALIFIED" if qualification["development_qualified"] else "NOT_QUALIFIED",
        **qualification,
        "source_v2_development_sha256": {
            flow_id: file_sha256(
                v2.OUTPUT_ROOT / "development" / f"{flow_id}.json"
            )
            for flow_id in base.FLOW_IDS
        },
        "validation_accessed": False,
    }


def _write_report(result: dict[str, Any]) -> None:
    confirmation = result["confirmation"]
    lines = [
        "# Three-Law common-task Galerkin qualification v3",
        "",
        f"Status: **{result['status']}**",
        "",
        "V1 failed at the bottom of its K>=120 ladder. V2 selected K=100 after "
        "extending the nested basis downward, but the B1_seed1 16k development "
        "audit failed only its raw pre-centering forcing-mean diagnostic. V3 "
        "requires every centered development gate and the original complete "
        "certificate on the untouched 65k confirmation banks.",
        "",
        f"Development recommendation: K={result['development']['recommended_K']}, "
        f"rank tolerance={result['development']['recommended_rank_tolerance']:.1e}.",
        "",
        "| flow | risk | Full action | forcing | algebra | energy | complete |",
        "|:--|---:|---:|:---:|:---:|---:|:---:|",
    ]
    for row in confirmation["rows"]:
        lines.append(
            f"| {row['flow_id']} | {row['scientific_risk']:.9g} | "
            f"{row['train_action']:.9g} | "
            f"{'PASS' if row['train_forcing']['valid'] and row['audit_forcing']['valid'] else 'FAIL'} | "
            f"{'PASS' if row['algebra']['valid'] else 'FAIL'} | "
            f"{row['heldout_certificate']['maximum_energy_residual']:.6g} | "
            f"{'PASS' if row['complete_certificate'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "The Pareto handoff exists only if all three confirmation rows pass. "
            "It freezes K, rank tolerance, dictionary, Law geometries, and the "
            "mandatory same-Full-metric Law fallback rule.",
        ]
    )
    base._atomic_text(REPORT_PATH, "\n".join(lines) + "\n")


def run(cfg: dict[str, Any], progress: Callable[[str], None] | None = print) -> dict[str, Any]:
    _configure_base()
    if RESULT_PATH.exists():
        return _read(RESULT_PATH)
    protocol = freeze_protocol(cfg)
    development = amended_development(cfg)
    if development["recommended_K"] != 100:
        raise RuntimeError(
            f"frozen v3 expected development K=100, got {development['recommended_K']}"
        )
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
        "parent_v2_result_sha256": file_sha256(v2.RESULT_PATH),
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
    "OUTPUT_ROOT",
    "PROTOCOL_PATH",
    "amended_development",
    "centered_development_forcing_valid",
    "freeze_protocol",
    "run",
]
