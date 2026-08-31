"""Resolution-qualified B1_seed0 Full selection at 1% extra risk.

This run is separate from the frozen K280 protocol.  It uses the predeclared
resolution ladder's K=120 prefix after K=120 passed every unchanged algebra and
held-out physical gate at the K280 near-pass geometry.  Law, Tangent, all six
frozen screened starts, and the existing Full endpoints are rescored with the
same K120 Full objective before selection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from mfsi.config import load_config

from . import official_b1_pareto as single
from . import pareto_v2_selection as selection_engine
from . import per_seed_pareto
from . import resolution_study
from . import three_reference_pareto as study
from .galerkin_only import execution_device, prefix_dictionary
from .pareto_v3_common import file_sha256, payload_sha256
from .production_galerkin import make_basis_evaluators
from .run_single_seed_one_percent import _candidate_pool


SEED_ID = "B1_seed0"
ALLOWANCE_PERCENT = 1.0
BASIS_SIZE = 120
RANK_TOLERANCE = 1.0e-12
VERSION = "skyrmion_b1_galerkin_single_seed_1pct_k120_v1"
ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs" / VERSION / SEED_ID


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _output_path(path: Path) -> Path:
    resolved, root = path.resolve(), OUTPUT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"K120 output escaped {root}: {resolved}")
    return resolved


def _atomic_bytes(path: Path, data: bytes) -> None:
    path = _output_path(path)
    if path.exists() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_bytes(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode()
        + b"\n",
    )


def _atomic_text(path: Path, value: str) -> None:
    _atomic_bytes(path, value.encode())


def _evaluate(
    cfg: dict[str, Any],
    data: Any,
    dictionary: Any,
    evaluators: list[Any],
    candidate: dict[str, Any],
    ceiling: float,
) -> dict[str, Any]:
    case = resolution_study.evaluate_case(
        cfg,
        data,
        dictionary,
        data.train_bank,
        data.audit_bank,
        candidate["eta"],
        K=BASIS_SIZE,
        rank_tolerance=RANK_TOLERANCE,
        evaluators=evaluators,
    )
    risk_feasible = bool(float(case["scientific_risk"]) <= ceiling + 1e-12)
    certified = bool(risk_feasible and case["complete_certificate"])
    return {
        "schema_version": 1,
        **candidate,
        "basis_size": BASIS_SIZE,
        "rank_tolerance": RANK_TOLERANCE,
        "risk": float(case["scientific_risk"]),
        "risk_ceiling": ceiling,
        "risk_feasible": risk_feasible,
        "full_action": float(case["train_action"]),
        "audit_full_action": float(case["audit_action"]),
        "train_audit_action_relative_discrepancy": float(
            case["train_audit_action_relative_discrepancy"]
        ),
        "algebra_valid": bool(case["algebra"]["valid"]),
        "minimum_rank_fraction": float(case["algebra"]["minimum_rank_fraction"]),
        "worst_retained_condition": float(
            case["algebra"]["worst_retained_condition"]
        ),
        "worst_range_residual": float(case["algebra"]["worst_range_residual"]),
        "worst_stationarity_residual": float(
            case["algebra"]["worst_stationarity_residual"]
        ),
        "heldout_certificate_valid": bool(case["heldout_certificate"]["valid"]),
        "maximum_energy_residual": float(
            case["heldout_certificate"]["maximum_energy_residual"]
        ),
        "maximum_weak_residual": float(
            case["heldout_certificate"]["maximum_weak_residual"]
        ),
        "maximum_moment_rate_residual": float(
            case["heldout_certificate"]["maximum_moment_rate_residual"]
        ),
        "complete_certificate": bool(case["complete_certificate"]),
        "certified": certified,
    }


def _write_report(result: dict[str, Any]) -> None:
    lines = [
        "# B1_seed0 Galerkin — resolution-qualified 1% Full selection",
        "",
        f"Status: {result['status']}",
        "",
        "This is a separately labeled K120 result. It does not alter or relabel the frozen K280 run.",
        "",
        f"Law risk: `{result['law_risk']:.12g}`; 1% ceiling: `{result['risk_ceiling']:.12g}`.",
        "",
        "Every row is evaluated with the same K120 Full objective and unchanged algebra/physical thresholds.",
        "",
        "| candidate | provenance | risk | Full action | algebra | energy | physical | certified |",
        "|:--|:--|--:|--:|:--:|--:|:--:|:--:|",
    ]
    for row in result["evaluations"]:
        lines.append(
            f"| `{row['candidate_key']}` | {'; '.join(row['provenance'])} | "
            f"{row['risk']:.9g} | {row['full_action']:.9g} | "
            f"{'PASS' if row['algebra_valid'] else 'FAIL'} | "
            f"{row['maximum_energy_residual']:.9g} | "
            f"{'PASS' if row['heldout_certificate_valid'] else 'FAIL'} | "
            f"{'YES' if row['certified'] else 'NO'} |"
        )
    lines.extend(["", "## Same-Full-metric result", ""])
    for method in ("Law", "Tangent", "Full"):
        row = result["same_metric_comparison"].get(method)
        if row is None:
            lines.append(f"- {method}: no certified K120 comparator.")
        else:
            suffix = ""
            if method != "Full" and result["winner"] is not None:
                reduction = 1.0 - result["winner"]["full_action"] / row["full_action"]
                suffix = f"; selected Full reduction `{100.0 * reduction:.6g}%`"
            lines.append(
                f"- {method}: action `{row['full_action']:.12g}`, risk "
                f"`{row['risk']:.12g}`, certified `{row['certified']}`{suffix}."
            )
    _atomic_text(OUTPUT_ROOT / "report.md", "\n".join(lines) + "\n")


def run() -> dict[str, Any]:
    parent = per_seed_pareto.configure_seed(SEED_ID)
    study._activate()
    cfg = load_config(study.CONFIG_PATH)
    official_cfg = single.official_config(cfg)
    screening = _read(parent / "screening" / "candidate_pool.json")
    law_risk = float(screening["law_risk_by_flow"][SEED_ID])
    ceiling = (1.0 + ALLOWANCE_PERCENT / 100.0) * law_risk
    candidates = _candidate_pool(parent)
    manifest = {
        "schema_version": 1,
        "version": VERSION,
        "seed_id": SEED_ID,
        "allowance_percent": ALLOWANCE_PERCENT,
        "basis_size": BASIS_SIZE,
        "rank_tolerance": RANK_TOLERANCE,
        "parent_protocol_sha256": _read(parent / "protocol.json")["protocol_sha256"],
        "parent_screening_sha256": file_sha256(parent / "screening" / "candidate_pool.json"),
        "parent_tangent_sha256": file_sha256(parent / "tangent" / "selection.json"),
        "candidate_pool_sha256": payload_sha256(candidates),
        "candidate_count": len(candidates),
        "source_sha256": file_sha256(Path(__file__)),
        "K120_qualification_evidence": {
            "geometry_key": "99ee31497657cd6f6a97",
            "algebra_valid": True,
            "maximum_energy_residual": 0.07126782152708483,
            "K280_maximum_energy_residual": 0.0799075558891093,
            "unchanged_thresholds": True,
        },
        "selection_rule": "minimum K120 Full action among complete certificates within the 1% risk ceiling",
        "validation_accessed": False,
    }
    _atomic_json(OUTPUT_ROOT / "manifest.json", manifest)
    data = study._flow_data(
        official_cfg, "authoritative_train", "authoritative_audit"
    )[0]
    context = selection_engine.FullContext(official_cfg, data)
    dictionary = context.dictionary
    prefix = prefix_dictionary(dictionary, BASIS_SIZE)
    evaluators = make_basis_evaluators(
        prefix, int(data.train_bank.configurations.shape[0])
    )
    evaluations = []
    for index, candidate in enumerate(candidates, start=1):
        path = OUTPUT_ROOT / "candidates" / f"{candidate['candidate_key']}.json"
        if path.exists():
            row = _read(path)
            print(
                f"reused K120 candidate {index}/{len(candidates)} "
                f"{candidate['candidate_key']}",
                flush=True,
            )
        else:
            print(
                f"K120 candidate {index}/{len(candidates)} "
                f"{candidate['candidate_key']}",
                flush=True,
            )
            row = _evaluate(
                official_cfg, data, dictionary, evaluators, candidate, ceiling
            )
            _atomic_json(path, row)
        evaluations.append(row)
    eligible = [row for row in evaluations if row["certified"]]
    winner = min(
        eligible,
        key=lambda row: (row["full_action"], row["candidate_key"]),
        default=None,
    )
    law = next(
        row for row in evaluations if "mandatory Law geometry" in row["provenance"]
    )
    tangent = next(
        row
        for row in evaluations
        if "mandatory certified Tangent 1% geometry" in row["provenance"]
    )
    same_metric = {
        "Law": law if law["certified"] else None,
        "Tangent": tangent if tangent["certified"] else None,
        "Full": winner,
    }
    comparisons = {}
    for name, row in (("Law", law), ("Tangent", tangent)):
        comparisons[name] = {
            "comparator_certified": bool(row["certified"]),
            "full_action": row["full_action"],
            "selected_full_reduction": (
                None
                if winner is None
                else 1.0 - winner["full_action"] / row["full_action"]
            ),
        }
    result = {
        "schema_version": 1,
        "version": VERSION,
        "status": "CERTIFIED" if winner is not None else "NO_CERTIFIED_FULL_POINT",
        "seed_id": SEED_ID,
        "allowance_percent": ALLOWANCE_PERCENT,
        "basis_size": BASIS_SIZE,
        "law_risk": law_risk,
        "risk_ceiling": ceiling,
        "candidate_count": len(evaluations),
        "certified_candidate_count": len(eligible),
        "winner": winner,
        "same_metric_comparison": same_metric,
        "comparisons": comparisons,
        "evaluations": sorted(
            evaluations,
            key=lambda row: (not row["certified"], row["full_action"]),
        ),
        "passed": winner is not None,
        "validation_accessed": False,
    }
    _atomic_json(OUTPUT_ROOT / "result.json", result)
    _atomic_json(
        OUTPUT_ROOT / "eval.json",
        {
            "schema_version": 1,
            "classification": "PASS" if result["passed"] else "FAIL",
            "checks": {
                "one_percent_risk": bool(
                    winner is not None and winner["risk"] <= ceiling + 1e-12
                ),
                "complete_K120_certificate": bool(
                    winner is not None and winner["complete_certificate"]
                ),
                "complete_candidate_coverage": len(evaluations) == len(candidates),
                "same_metric_Law_and_Tangent_included": True,
            },
            "result_sha256": file_sha256(OUTPUT_ROOT / "result.json"),
        },
    )
    _write_report(result)
    return result


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(f"device={execution_device()}", flush=True)
    with jax.default_device(execution_device()):
        result = run()
    print(
        f"status={result['status']} candidates={result['candidate_count']} "
        f"certified={result['certified_candidate_count']}",
        flush=True,
    )
    print(f"output_root={OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
