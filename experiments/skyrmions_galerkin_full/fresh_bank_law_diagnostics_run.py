"""Complete the required Law diagnostics on every frozen fresh bank pair.

The main robustness audit intentionally evaluates only screen-supported candidates.
That rule is correct for candidate eligibility, but the study contract separately
requires Law diagnostics on all 32 audit banks.  This diagnostic-only pass fills
the Law rows omitted by the generic audit subset without changing any sealed
candidate result or eligibility decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from mfsi.config import load_config

from .fresh_bank_robustness import (
    ARTIFACT_DIR,
    BANK_MANIFEST_PATH,
    CANDIDATE_FREEZE_PATH,
    OUTPUT_ROOT,
    REPLICATE_COUNT,
    _FreshBankEvaluator,
    _atomic_json,
    _bank_path,
    _load_bank,
    _verify_stage,
    freeze_bank_manifest,
    freeze_candidates,
)
from .galerkin_only import execution_device
from .galerkin_only_data import load_selection_galerkin_data
from .production_artifacts import file_sha256


RESULT_PATH = OUTPUT_ROOT / "law_all_bank_diagnostics.json"
INVENTORY_PATH = OUTPUT_ROOT / "law_all_bank_diagnostics_inventory.json"


def _verify_cached() -> dict[str, Any] | None:
    if not RESULT_PATH.exists() and not INVENTORY_PATH.exists():
        return None
    if not RESULT_PATH.exists() or not INVENTORY_PATH.exists():
        raise RuntimeError("incomplete sealed all-bank Law diagnostic")
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    if inventory["result_sha256"] != file_sha256(RESULT_PATH):
        raise RuntimeError("sealed all-bank Law diagnostic changed")
    if inventory["candidate_freeze_sha256"] != file_sha256(CANDIDATE_FREEZE_PATH):
        raise RuntimeError("candidate freeze changed after Law diagnostic")
    if inventory["bank_manifest_sha256"] != file_sha256(BANK_MANIFEST_PATH):
        raise RuntimeError("bank manifest changed after Law diagnostic")
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    if len(result["rows"]) != REPLICATE_COUNT:
        raise RuntimeError("Law diagnostic does not contain all replicate pairs")
    if not all(row["audit"]["audit_performed"] for row in result["rows"]):
        raise RuntimeError("Law was not evaluated on every audit bank")
    result["cache_hit"] = True
    return result


def run(cfg: dict[str, Any], progress: Any | None = None) -> dict[str, Any]:
    freeze = freeze_candidates(cfg)
    manifest = freeze_bank_manifest(cfg)
    cached = _verify_cached()
    if cached is not None:
        return cached

    law_eta = np.asarray([freeze["law_eta"]], dtype=np.float64)
    problem = load_selection_galerkin_data(cfg, ARTIFACT_DIR).selection_problem
    evaluator = _FreshBankEvaluator(problem)
    rows = []
    supplemental_ids = []

    for record in manifest["replicates"]:
        replicate = int(record["replicate_id"])
        screen_cached = _verify_stage(replicate, "screen")
        audit_cached = _verify_stage(replicate, "audit")
        if screen_cached is None or audit_cached is None:
            raise RuntimeError(f"replicate {replicate} is not fully sealed")
        screen_summary, screen = screen_cached
        audit_summary, audit = audit_cached
        screen_diag = dict(screen_summary["law_candidate_diagnostics"])

        if bool(audit["audit_performed"][0]):
            audit_diag = dict(audit_summary["law_candidate_diagnostics"])
            audit_diag["diagnostic_source"] = "sealed_candidate_audit_stage"
        else:
            bank_path = _bank_path(replicate, "audit")
            diagnostic = evaluator.evaluate(law_eta, _load_bank(bank_path))
            audit_diag = {
                name: values[0].item() for name, values in diagnostic.items()
            }
            audit_diag["audit_performed"] = True
            audit_diag["robust_ress_pair"] = min(
                float(screen["minimum_ress"][0]),
                float(diagnostic["minimum_ress"][0]),
            )
            audit_diag["diagnostic_source"] = "supplemental_required_law_audit"
            supplemental_ids.append(replicate)

        rows.append(
            {
                "replicate_id": replicate,
                "screen_bank_sha256": screen_summary["bank_sha256"],
                "audit_bank_sha256": audit_summary["bank_sha256"],
                "screen": screen_diag,
                "audit": audit_diag,
            }
        )
        if progress is not None:
            progress(replicate, audit_diag["diagnostic_source"])

    result = {
        "schema_version": 1,
        "development_only": True,
        "purpose": "diagnostic-only completion of Law evaluation on every frozen bank pair",
        "candidate_freeze_sha256": file_sha256(CANDIDATE_FREEZE_PATH),
        "bank_manifest_sha256": file_sha256(BANK_MANIFEST_PATH),
        "law_candidate_id": freeze["law_candidate_id"],
        "law_eta": freeze["law_eta"],
        "replicate_count": REPLICATE_COUNT,
        "all_screen_banks_evaluated": True,
        "all_audit_banks_evaluated": True,
        "supplemental_audit_replicate_ids": supplemental_ids,
        "supplemental_audit_count": len(supplemental_ids),
        "candidate_eligibility_recomputed": False,
        "sealed_candidate_results_modified": False,
        "rows": rows,
    }
    _atomic_json(RESULT_PATH, result)
    inventory = {
        "schema_version": 1,
        "result_path": str(RESULT_PATH.relative_to(OUTPUT_ROOT)),
        "result_sha256": file_sha256(RESULT_PATH),
        "candidate_freeze_sha256": file_sha256(CANDIDATE_FREEZE_PATH),
        "bank_manifest_sha256": file_sha256(BANK_MANIFEST_PATH),
    }
    _atomic_json(INVENTORY_PATH, inventory)
    result["cache_hit"] = False
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = jax.devices("cpu")[0] if args.force_cpu else execution_device()

    def progress(replicate: int, source: str) -> None:
        print(f"Law replicate={replicate:02d} audit_source={source}", flush=True)

    with jax.default_device(device):
        result = run(cfg, progress=progress)
    print(f"Law replicate pairs complete: {result['replicate_count']}")
    print(f"supplemental Law audit evaluations: {result['supplemental_audit_count']}")
    print(f"result SHA-256: {file_sha256(RESULT_PATH)}")
    print(f"cache_hit={result['cache_hit']}")


if __name__ == "__main__":
    main()
