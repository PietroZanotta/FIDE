"""Corrected authoritative B1_seed0 Full selection at 1% extra risk.

The original per-seed Full stage capped the frozen screened starts at three and
did not cross-evaluate the Law and Tangent geometries with the same K280 Full
objective.  This targeted run reuses the frozen parent banks and protocol,
audits the complete 1% candidate set, and chooses the lowest certified Full
action.  It does not mutate the parent run.
"""

from __future__ import annotations

import argparse
import json
import math
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
from . import three_reference_pareto as study
from .galerkin_only import execution_device
from .pareto_v3_common import file_sha256, payload_sha256


SEED_ID = "B1_seed0"
ALLOWANCE_PERCENT = 1.0
VERSION = "skyrmion_b1_galerkin_single_seed_1pct_v1"
ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs" / VERSION / SEED_ID


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _output_path(path: Path) -> Path:
    resolved = path.resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"targeted 1% output escaped {root}: {resolved}")
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


def _candidate_key(eta: Any) -> str:
    return study.eta_key(eta)


def _add_candidate(
    candidates: dict[str, dict[str, Any]],
    eta: Any,
    provenance: str,
) -> None:
    key = _candidate_key(eta)
    if key not in candidates:
        candidates[key] = {
            "candidate_key": key,
            "eta": np.asarray(eta, dtype=np.float64).tolist(),
            "provenance": [],
        }
    if provenance not in candidates[key]["provenance"]:
        candidates[key]["provenance"].append(provenance)


def _candidate_pool(parent: Path) -> list[dict[str, Any]]:
    screening = _read(parent / "screening" / "candidate_pool.json")
    tangent = _read(parent / "tangent" / "selection.json")
    candidates: dict[str, dict[str, Any]] = {}
    _add_candidate(candidates, screening["law_eta"], "mandatory Law geometry")
    tangent_point = next(
        row
        for row in tangent["allowances"]
        if float(row["allowance_percent"]) == ALLOWANCE_PERCENT
    )
    _add_candidate(
        candidates,
        tangent_point["winner"]["eta"],
        "mandatory certified Tangent 1% geometry",
    )
    for index, row in enumerate(screening["starts"]["1"]):
        _add_candidate(
            candidates,
            row["eta"],
            f"frozen screened 1% start {index}: {row.get('candidate_id', 'unknown')}",
        )
    trajectory_dir = parent / "full_search" / "allowance_1"
    for path in sorted(trajectory_dir.glob("trajectory_*.json")):
        row = _read(path)
        _add_candidate(candidates, row["start"]["eta"], f"existing Full start: {path.name}")
        _add_candidate(candidates, row["endpoint"]["eta"], f"existing Full endpoint: {path.name}")
    return sorted(candidates.values(), key=lambda row: row["candidate_key"])


def _public_evaluation_payload(
    public: dict[str, Any],
    audit: dict[str, Any],
    candidate: dict[str, Any],
    ceiling: float,
) -> dict[str, Any]:
    per_flow = audit["per_flow"][SEED_ID]
    certificate = per_flow["heldout_certificate"]
    risk_feasible = bool(float(public["risk"]) <= ceiling + 1e-12)
    certified = bool(risk_feasible and audit["valid"])
    return {
        "schema_version": 1,
        **candidate,
        "risk": float(public["risk"]),
        "risk_ceiling": ceiling,
        "risk_feasible": risk_feasible,
        "full_action": float(public["action"]),
        "search_valid": bool(public["search_valid"]),
        "algebra_valid": bool(public["algebra_valid"]),
        "geometry_valid": bool(public["geometry_valid"]),
        "train_forcing_valid": bool(public["train_forcing_audit"]["valid"]),
        "authoritative_audit_valid": bool(audit["valid"]),
        "audit_forcing_valid": bool(per_flow["audit_forcing"]["valid"]),
        "heldout_certificate_valid": bool(certificate["valid"]),
        "maximum_energy_residual": float(certificate["maximum_energy_residual"]),
        "maximum_weak_residual": float(certificate["maximum_weak_residual"]),
        "maximum_moment_rate_residual": float(
            certificate["maximum_moment_rate_residual"]
        ),
        "certified": certified,
    }


def _evaluation_payload(
    context: study.EnsembleFullContext,
    candidate: dict[str, Any],
    ceiling: float,
) -> dict[str, Any]:
    raw = context.evaluate(candidate["eta"], gradient=False)
    audit = context.audit(raw, require_physical=True)
    return _public_evaluation_payload(
        selection_engine._public(raw), audit, candidate, ceiling
    )


def _parent_cached_payload(
    parent: Path,
    candidate: dict[str, Any],
    ceiling: float,
) -> dict[str, Any] | None:
    path = parent / "authoritative" / "cache" / f"{candidate['candidate_key']}.json"
    if not path.exists():
        return None
    cached = _read(path)
    return {
        **_public_evaluation_payload(
            cached, cached["authoritative_audit"], candidate, ceiling
        ),
        "reused_parent_authoritative_cache": True,
        "parent_cache_sha256": file_sha256(path),
    }


def _write_report(result: dict[str, Any]) -> None:
    lines = [
        "# B1_seed0 Galerkin — corrected 1% Full selection",
        "",
        f"Status: {result['status']}",
        "",
        "All candidates use the same authoritative K280 Full action and the same held-out physical certificate.",
        "",
        f"Law risk: `{result['law_risk']:.12g}`; 1% ceiling: `{result['risk_ceiling']:.12g}`.",
        "",
        "| candidate | provenance | risk | Full action | algebra | physical audit | energy residual | certified |",
        "|:--|:--|--:|--:|:--:|:--:|--:|:--:|",
    ]
    for row in result["evaluations"]:
        provenance = "; ".join(row["provenance"])
        lines.append(
            f"| `{row['candidate_key']}` | {provenance} | {row['risk']:.9g} | "
            f"{row['full_action']:.9g} | {'PASS' if row['algebra_valid'] else 'FAIL'} | "
            f"{'PASS' if row['authoritative_audit_valid'] else 'FAIL'} | "
            f"{row['maximum_energy_residual']:.9g} | {'YES' if row['certified'] else 'NO'} |"
        )
    winner = result.get("winner")
    lines.extend(["", "## Result", ""])
    if winner is None:
        lines.append("No candidate passed both the 1% risk ceiling and the authoritative K280 certificate.")
    else:
        lines.append(
            f"Winner `{winner['candidate_key']}` has Full action `{winner['full_action']:.12g}` "
            f"at risk `{winner['risk']:.12g}`."
        )
    comparisons = result["same_metric_comparison"]
    lines.extend(["", "## Same-Full-metric comparison", ""])
    for name in ("Law", "Tangent", "Full"):
        row = comparisons.get(name)
        if row is None:
            lines.append(f"- {name}: no certified Full comparator.")
        else:
            lines.append(
                f"- {name}: Full action `{row['full_action']:.12g}`, risk `{row['risk']:.12g}`, "
                f"certified `{row['certified']}`."
            )
    _atomic_text(OUTPUT_ROOT / "report.md", "\n".join(lines) + "\n")


def run() -> dict[str, Any]:
    parent = per_seed_pareto.configure_seed(SEED_ID)
    study._activate()
    cfg = load_config(study.CONFIG_PATH)
    official_cfg = single.official_config(cfg)
    protocol = _read(parent / "protocol.json")
    screening = _read(parent / "screening" / "candidate_pool.json")
    law_risk = float(screening["law_risk_by_flow"][SEED_ID])
    ceiling = (1.0 + ALLOWANCE_PERCENT / 100.0) * law_risk
    candidates = _candidate_pool(parent)
    manifest = {
        "schema_version": 1,
        "version": VERSION,
        "seed_id": SEED_ID,
        "allowance_percent": ALLOWANCE_PERCENT,
        "parent_output": str(parent),
        "parent_protocol_sha256": protocol["protocol_sha256"],
        "parent_screening_sha256": file_sha256(parent / "screening" / "candidate_pool.json"),
        "parent_tangent_sha256": file_sha256(parent / "tangent" / "selection.json"),
        "source_sha256": file_sha256(Path(__file__)),
        "candidate_pool_sha256": payload_sha256(candidates),
        "candidate_count": len(candidates),
        "selection_rule": "minimum authoritative K280 Full action among candidates passing the 1% risk ceiling and physical audit",
        "validation_accessed": False,
    }
    _atomic_json(OUTPUT_ROOT / "manifest.json", manifest)
    data = study._flow_data(
        official_cfg, "authoritative_train", "authoritative_audit"
    )
    context = study.EnsembleFullContext(
        selection_engine, official_cfg, data, np.asarray([law_risk])
    )
    evaluations = []
    for index, candidate in enumerate(candidates, start=1):
        cache_path = OUTPUT_ROOT / "candidates" / f"{candidate['candidate_key']}.json"
        if cache_path.exists():
            row = _read(cache_path)
        else:
            row = _parent_cached_payload(parent, candidate, ceiling)
            if row is None:
                print(
                    f"authoritative candidate {index}/{len(candidates)} "
                    f"{candidate['candidate_key']}",
                    flush=True,
                )
                row = _evaluation_payload(context, candidate, ceiling)
            else:
                print(
                    f"reused authoritative candidate {index}/{len(candidates)} "
                    f"{candidate['candidate_key']}",
                    flush=True,
                )
            _atomic_json(cache_path, row)
        evaluations.append(row)
    eligible = [row for row in evaluations if row["certified"]]
    winner = min(
        eligible,
        key=lambda row: (float(row["full_action"]), row["candidate_key"]),
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
    same_metric = {"Law": law, "Tangent": tangent, "Full": winner}
    result = {
        "schema_version": 1,
        "version": VERSION,
        "status": "CERTIFIED" if winner is not None else "NO_CERTIFIED_FULL_POINT",
        "seed_id": SEED_ID,
        "allowance_percent": ALLOWANCE_PERCENT,
        "law_risk": law_risk,
        "risk_ceiling": ceiling,
        "candidate_count": len(evaluations),
        "certified_candidate_count": len(eligible),
        "winner": winner,
        "same_metric_comparison": same_metric,
        "evaluations": sorted(
            evaluations,
            key=lambda row: (not row["certified"], row["full_action"]),
        ),
        "passed": winner is not None,
        "validation_accessed": False,
    }
    _atomic_json(OUTPUT_ROOT / "result.json", result)
    evaluation = {
        "schema_version": 1,
        "classification": "PASS" if result["passed"] else "FAIL",
        "checks": {
            "one_percent_risk": bool(
                winner is not None and winner["risk"] <= ceiling + 1e-12
            ),
            "authoritative_full_certificate": bool(
                winner is not None and winner["authoritative_audit_valid"]
            ),
            "finite_full_action": bool(
                winner is not None and math.isfinite(winner["full_action"])
            ),
            "complete_frozen_start_coverage": len(candidates)
            >= 2 + len(screening["starts"]["1"]),
        },
        "result_sha256": file_sha256(OUTPUT_ROOT / "result.json"),
    }
    _atomic_json(OUTPUT_ROOT / "eval.json", evaluation)
    _write_report(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
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
