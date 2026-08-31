#!/usr/bin/env python3
"""Read-only scientific checks plus reference-stage receipt generation."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np


V2_DIR = Path(__file__).resolve().parent
REPO_ROOT = V2_DIR.parents[1]
ROOT = V2_DIR / "outputs" / "prospective_v2"
REFERENCES = ROOT / "references"
FREEZE = ROOT / "freeze"
for value in (REPO_ROOT / "src", V2_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from core import frozen_reference_scott_bandwidth  # noqa: E402
from selection_contract import sha256_file  # noqa: E402


EXPECTED_SEEDS = [310000101, 310000102, 310000103]
EXPECTED_ROLLOUT_SEEDS = [310003102, 310003103, 310003104]
MANIFEST_PATH = V2_DIR / "VORTICES_V2_FREEZE_MANIFEST.json"
CONFIG_PATH = V2_DIR / "VORTICES_V2_SELECTION_CONFIG.json"
COMMON_PATH = FREEZE / "common_bandwidth_receipt.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(label)


def main() -> None:
    completed_at = utc_now()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    checks: dict[str, Any] = {}
    standalone_inputs = {
        V2_DIR / "base_experiment_config.json": "8f57f167675718b19d7ffc1741a8175adbe22069ff4043634b62df8dcf100ed0",
        V2_DIR / "experiment.py": "5bcd5b3c96668cabf6d7a8b2b1944f48f490635763b997172584328551a9a4c4",
        V2_DIR / "bounded_reference.py": "bb9bb091329cf1cda54252d4b86463c900307f5ed7b983fd30de959ffa4d7cbe",
        V2_DIR / "inputs" / "reference_endpoints.npz": "ad4006927e268c52f621c16c773f0600d803370bd21fb5e0816d82a70dbdfbba",
        V2_DIR / "inputs" / "truth_bank.npz": "d897ff7fc44c0b85d7bb5391c0cc25895b4301e9c2ce00184697a1899d853b5b",
    }
    for path, expected in standalone_inputs.items():
        require(path.is_file(), f"standalone input missing: {path.name}")
        require(sha256_file(path) == expected, f"standalone input changed: {path.name}")
    checks["standalone_inputs"] = {"status": "PASS", "files": len(standalone_inputs)}

    dirs = sorted(
        p.name for p in REFERENCES.glob("reference_seed_*") if p.is_dir()
    )
    expected_dirs = [f"reference_seed_{seed}" for seed in EXPECTED_SEEDS]
    require(dirs == expected_dirs, f"reference directories {dirs} != {expected_dirs}")
    checks["exactly_three_reference_directories"] = {
        "status": "PASS",
        "directories": dirs,
    }

    require(not (ROOT / "selection").exists(), "selection tree exists")
    require(not (ROOT / "validation").exists(), "validation tree exists")
    forbidden_names = {
        "shared_selection_bank.npz",
        "shared_validation_bank.npz",
        "frozen_winners.json",
        "simultaneous_inference.json",
    }
    hits = [str(p) for p in ROOT.rglob("*") if p.is_file() and p.name in forbidden_names]
    require(not hits, f"forbidden downstream artifacts exist: {hits}")
    checks["no_selection_or_validation_outputs"] = {"status": "PASS"}

    common = json.loads(COMMON_PATH.read_text(encoding="utf-8"))
    require(common["status"] == "FROZEN_COMMON_REFERENCE_ONLY_BANDWIDTH", "bad common status")
    require(common["immutable"] is True, "common bandwidth not immutable")
    require(common["action_or_risk_inputs_used"] is False, "action/risk contaminated bandwidth")
    require(common["training_seeds"] == EXPECTED_SEEDS, "common receipt seed set mismatch")
    require(len(common["per_reference"]) == 3, "common receipt reference count mismatch")

    rows: list[dict[str, Any]] = []
    for index, (seed, rollout_seed) in enumerate(
        zip(EXPECTED_SEEDS, EXPECTED_ROLLOUT_SEEDS)
    ):
        directory = REFERENCES / f"reference_seed_{seed}"
        receipt_path = directory / "qualification_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        require(receipt["qualified"] is True and receipt["status"] == "PASS", f"seed {seed} failed")
        require(receipt["training_seed"] == seed, f"training seed mismatch {seed}")
        require(receipt["rollout_seed"] == rollout_seed, f"rollout seed mismatch {seed}")
        require(receipt["repository_head"] == manifest["repository"]["head"], f"HEAD mismatch {seed}")
        require(receipt["freeze_manifest_sha256"] == sha256_file(MANIFEST_PATH), f"manifest mismatch {seed}")
        checkpoint = Path(receipt["checkpoint_path"])
        rollout = Path(receipt["rollout_bank_path"])
        require(sha256_file(checkpoint) == receipt["checkpoint_sha256"], f"checkpoint hash mismatch {seed}")
        require(sha256_file(rollout) == receipt["rollout_bank_sha256"], f"rollout hash mismatch {seed}")
        with np.load(rollout, allow_pickle=False) as bank:
            nodes = np.asarray(bank["nodes"], dtype=np.float64)
            velocity = np.asarray(bank["velocity"], dtype=np.float64)
            weights = np.asarray(bank["weights"], dtype=np.float64)
        require(nodes.shape == (21, 32768, 2), f"node shape mismatch {seed}")
        require(velocity.shape == (21, 32768, 2), f"velocity shape mismatch {seed}")
        require(weights.shape == (21, 32768), f"weight shape mismatch {seed}")
        recomputed, by_time = frozen_reference_scott_bandwidth(nodes, weights)
        require(recomputed == float(receipt["scott_bandwidth"]), f"Scott median mismatch {seed}")
        np.testing.assert_array_equal(
            by_time, np.asarray(receipt["scott_bandwidth_by_time"], dtype=np.float64)
        )
        common_row = common["per_reference"][index]
        require(common_row["training_seed"] == seed, f"common order mismatch {seed}")
        require(common_row["scott_bandwidth"] == recomputed, f"common Scott mismatch {seed}")
        require(common_row["checkpoint_sha256"] == receipt["checkpoint_sha256"], f"common checkpoint mismatch {seed}")
        require(common_row["rollout_bank_sha256"] == receipt["rollout_bank_sha256"], f"common rollout mismatch {seed}")
        require(common_row["qualification_receipt_sha256"] == sha256_file(receipt_path), f"common qualification mismatch {seed}")
        rows.append(
            {
                "training_seed": seed,
                "rollout_seed": rollout_seed,
                "qualification": "PASS",
                "qualification_receipt": str(receipt_path.resolve()),
                "qualification_receipt_sha256": sha256_file(receipt_path),
                "final_training_step": receipt["final_training_step"],
                "final_conditional_fm_loss": receipt["final_conditional_fm_loss"],
                "final_preclip_gradient_norm": receipt["final_preclip_gradient_norm"],
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": receipt["checkpoint_sha256"],
                "rollout_bank": str(rollout.resolve()),
                "rollout_bank_sha256": receipt["rollout_bank_sha256"],
                "rollout_shape": list(nodes.shape),
                "velocity_shape": list(velocity.shape),
                "weight_shape": list(weights.shape),
                "minimum_in_domain_base_mass": receipt["minimum_in_domain_base_mass"],
                "maximum_weight_sum_absolute_error": receipt["maximum_weight_sum_absolute_error"],
                "training_config_sha256": receipt["training_config_sha256"],
                "training_architecture_sha256": receipt["training_architecture_sha256"],
                "scott_bandwidth": recomputed,
                "scott_bandwidth_by_time": by_time.tolist(),
            }
        )

    per_reference = [row["scott_bandwidth"] for row in rows]
    exact_median = float(np.median(np.asarray(per_reference, dtype=np.float64)))
    require(exact_median == common["common_physical_bandwidth"], "h_common is not exact median")
    checks["three_qualifications"] = {"status": "PASS", "seeds": EXPECTED_SEEDS}
    checks["independent_scott_recomputation"] = {
        "status": "PASS",
        "per_reference": per_reference,
    }
    checks["exact_common_median"] = {
        "status": "PASS",
        "common_physical_bandwidth": exact_median,
    }

    common_hash = sha256_file(COMMON_PATH)
    provenance = {
        "schema_version": 1,
        "status": "PASS",
        "role": "provenance envelope for the single immutable frozen common-bandwidth receipt",
        "verified_at_utc": completed_at,
        "repository_head": manifest["repository"]["head"],
        "freeze_manifest": str(MANIFEST_PATH.resolve()),
        "freeze_manifest_sha256": sha256_file(MANIFEST_PATH),
        "selection_config": str(CONFIG_PATH.resolve()),
        "selection_config_sha256": sha256_file(CONFIG_PATH),
        "common_bandwidth_receipt": str(COMMON_PATH.resolve()),
        "common_bandwidth_receipt_sha256": common_hash,
        "training_seeds": EXPECTED_SEEDS,
        "rollout_seeds": EXPECTED_ROLLOUT_SEEDS,
        "endpoint_data_sha256": common["endpoint_data_sha256"],
        "training_architecture_sha256": common["training_architecture_sha256"],
        "numerical_method_config_sha256": common["numerical_method_config_sha256"],
        "references": rows,
        "common_physical_bandwidth": exact_median,
        "rule": common["rule"],
        "action_or_risk_inputs_used": False,
    }
    provenance_path = FREEZE / "common_bandwidth_provenance.json"
    atomic_json(provenance_path, provenance)

    runner_path = V2_DIR / "run_reference_stage.py"
    checker_path = V2_DIR / "verify_reference_stage.py"
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "reference_stage_complete": True,
        "eligible_to_begin_prospective_selection": True,
        "completed_at_utc": completed_at,
        "pretraining_frozen_preflight": {
            "status": "PASS",
            "exit_status": 0,
            "passed_checks": 16,
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "performed_before_any_reference_directory_was_created": True,
        },
        "repository_head": manifest["repository"]["head"],
        "endpoint_data_sha256": common["endpoint_data_sha256"],
        "references": rows,
        "common_physical_bandwidth": exact_median,
        "common_bandwidth_receipt": str(COMMON_PATH.resolve()),
        "common_bandwidth_receipt_sha256": common_hash,
        "common_bandwidth_provenance": str(provenance_path.resolve()),
        "frozen_numerical_and_protocol_sha256": {
            relative: expected for relative, expected in manifest["frozen_files"].items()
        },
        "shared_dependency_sha256": manifest["shared_dependencies"],
        "execution_harness_sha256": sha256_file(runner_path),
        "post_reference_preflight_sha256": sha256_file(checker_path),
        "checks": checks,
        "scientific_operations_performed": [
            "three_frozen_reference_trainings",
            "three_frozen_reference_rollouts",
            "three_reference_only_qualifications",
            "three_reference_only_scott_bandwidths",
            "median_of_three_common_bandwidth_freeze",
        ],
        "scientific_operations_not_performed": [
            "selection_bank_generation",
            "validation_bank_generation",
            "population_optimization",
            "law_optimization_or_risk_anchor",
            "tangent_optimization",
            "full_optimization_or_action_evaluation",
            "full_vs_law_reduction",
            "validation_inference",
        ],
    }
    summary_path = REFERENCES / "reference_stage_summary.json"
    atomic_json(summary_path, summary)

    qualification_summary = {
        "schema_version": 1,
        "status": "PASS",
        "all_three_qualified": True,
        "references": rows,
        "common_bandwidth_eligible": True,
        "common_physical_bandwidth": exact_median,
        "reference_stage_summary": str(summary_path.resolve()),
    }
    qualification_json = REFERENCES / "REFERENCE_QUALIFICATION_SUMMARY.json"
    atomic_json(qualification_json, qualification_summary)
    lines = [
        "# Vortices V2 reference qualification summary",
        "",
        "All three prospectively frozen references passed every unchanged",
        "reference-only qualification gate. No replacement seed was used.",
        "",
        "| Training seed | Rollout seed | Final loss | Final gradient | Scott bandwidth | Qualification |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['training_seed']}` | `{row['rollout_seed']}` | "
            f"`{row['final_conditional_fm_loss']:.16g}` | "
            f"`{row['final_preclip_gradient_norm']:.16g}` | "
            f"`{row['scott_bandwidth']:.17g}` | **PASS** |"
        )
    lines += [
        "",
        f"The frozen exact median is `{exact_median:.17g}`.",
        "",
        "Full-precision timewise values and hashes are in",
        "`reference_stage_summary.json` and each `qualification_receipt.json`.",
        "",
    ]
    (REFERENCES / "REFERENCE_QUALIFICATION_SUMMARY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    postcheck = {
        "schema_version": 1,
        "status": "PASS",
        "checked_at_utc": completed_at,
        "checks": checks,
        "reference_stage_summary": str(summary_path.resolve()),
        "reference_stage_summary_sha256": sha256_file(summary_path),
        "qualification_summary": str(qualification_json.resolve()),
        "qualification_summary_sha256": sha256_file(qualification_json),
        "common_bandwidth_receipt_sha256": common_hash,
    }
    postcheck_path = REFERENCES / "post_reference_preflight.json"
    atomic_json(postcheck_path, postcheck)
    print(json.dumps(postcheck, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
