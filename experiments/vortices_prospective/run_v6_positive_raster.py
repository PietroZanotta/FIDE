from __future__ import annotations

"""Resume V6 under the frozen positive-raster prevalidation amendment."""

import argparse
import copy
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np

from common import config_hash, load_config, software_metadata, write_json_atomic
from mfsi.cache import file_sha256
from prospective_data import TargetProspectiveData
from evaluator import ProspectiveEvaluator, make_common_reference_evaluators
from v4_objective import V4CRNBank
from v6_reference_ensemble import train_reference_split, v6_paths
from v6_select import (
    _multi_authoritative,
    _shared_signature,
    combine_freeze,
    select_arm,
)
from v6_validate import validate_v6


HERE = Path(__file__).resolve().parent
DEFAULT_REPAIR = HERE / "configs" / "production_v6_positive_raster_repair.json"
DEFAULT_OUTPUT = HERE / "outputs" / "prospective_v6_beta_ablation_positive_raster_v1"


def apply_execution_profile(
    cfg: dict[str, Any], path: str | Path
) -> dict[str, Any]:
    path = Path(path).resolve()
    profile = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version", "name", "status", "v6_fast_execution",
        "scientific_protocol_preserved",
    }
    if set(profile) != expected or int(profile["schema_version"]) != 1:
        raise ValueError("invalid V6 execution profile")
    allowed = {
        "law_start_batch_size", "tangent_start_batch_size",
        "full_start_batch_size", "polish_start_batch_size",
        "prescreen_optimize_starts", "prescreen_start_batch_size",
    }
    settings = profile["v6_fast_execution"]
    if not set(settings) <= allowed or any(int(value) < 1 for value in settings.values()):
        raise ValueError("invalid V6 fast-execution setting")
    preserved = profile["scientific_protocol_preserved"]
    checks = {
        "broad_full_starts": int(cfg["v4"]["full_optimizer"]["starts"]),
        "reference_particles": int(cfg["reference"]["particles"]),
        "full_lbfgs_enabled": bool(cfg["v4"]["full_lbfgs"]["enabled"]),
    }
    if any(preserved.get(key) != value for key, value in checks.items()):
        raise ValueError("execution profile does not preserve the resolved protocol")
    resolved = copy.deepcopy(cfg)
    resolved["v6_fast_execution"] = {
        key: int(value) for key, value in settings.items()
    }
    resolved["v6_execution_profile"] = {
        "name": str(profile["name"]),
        "path": str(path),
        "sha256": file_sha256(path),
        "status_at_load": str(profile["status"]),
    }
    return resolved


def load_repair_config(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path).resolve()
    repair = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version", "name", "base_config", "source_output", "output_name",
        "authoritative_positive_raster", "repair_reason",
    }
    if set(repair) != expected or int(repair["schema_version"]) != 6:
        raise ValueError("invalid V6 positive-raster repair specification")
    cfg = copy.deepcopy(load_config(path.parent / repair["base_config"]))
    cfg["name"] = str(repair["name"])
    cfg["v6"]["output_name"] = str(repair["output_name"])
    cfg["raster"] = {
        "authoritative_positive": copy.deepcopy(repair["authoritative_positive_raster"])
    }
    cfg["v6_prevalidation_repair"] = {
        "specification": path.name,
        "reason": str(repair["repair_reason"]),
        "source_output": str(repair["source_output"]),
    }
    return cfg, repair


def _copy_verified(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_sha256(source) != file_sha256(target):
            raise RuntimeError(f"incompatible adopted artifact: {target}")
        return
    shutil.copy2(source, target)


def prepare_repair(cfg: dict[str, Any], repair: dict[str, Any], output_dir: str | Path):
    source = HERE / "outputs" / str(repair["source_output"])
    target = v6_paths(output_dir)
    if target["hidden"].exists() and any(target["hidden"].iterdir()):
        raise RuntimeError("repair preparation refuses to run after hidden data exists")
    source_paths = v6_paths(source)
    required = [
        source_paths["endpoint"] / "endpoint_data.npz",
        source_paths["prospective"] / "aggregate_predictions.npz",
        source_paths["prospective"] / "build_receipt.json",
        source_paths["prospective"] / "v6_selection_crn.npz",
        source_paths["shared_results"] / "design_reference_manifest.json",
        source_paths["shared_results"] / "shared_selection_manifest.json",
        source_paths["shared_results"] / "shared_candidate_archive.json",
    ]
    if not all(path.exists() for path in required):
        raise RuntimeError("failed V6 source is missing a required prevalidation artifact")
    for relative in (
        Path("shared/endpoint_reference/endpoint_data.npz"),
        Path("shared/prospective/aggregate_predictions.npz"),
        Path("shared/prospective/build_receipt.json"),
        Path("shared/prospective/v6_selection_crn.npz"),
    ):
        _copy_verified(source / relative, Path(output_dir).resolve() / relative)

    parent_design_path = source_paths["shared_results"] / "design_reference_manifest.json"
    parent_design = json.loads(parent_design_path.read_text(encoding="utf-8"))
    adopted_design = copy.deepcopy(parent_design)
    adopted_design["adoption"] = {
        "source_output": str(source.resolve()),
        "source_manifest_sha256": file_sha256(parent_design_path),
        "reason": "byte-identical pre-hidden reference reuse under numerical amendment",
        "resolved_repair_config_hash": config_hash(cfg),
    }
    target["shared_results"].mkdir(parents=True, exist_ok=True)
    design_path = target["shared_results"] / "design_reference_manifest.json"
    if design_path.exists():
        if json.loads(design_path.read_text(encoding="utf-8")) != adopted_design:
            raise RuntimeError("existing repaired design manifest is incompatible")
    else:
        write_json_atomic(design_path, adopted_design)

    resolved_path = target["results"] / "resolved_config.json"
    target["results"].mkdir(parents=True, exist_ok=True)
    if resolved_path.exists():
        if json.loads(resolved_path.read_text(encoding="utf-8")) != cfg:
            raise RuntimeError("existing repaired resolved config is incompatible")
    else:
        write_json_atomic(resolved_path, cfg)
    receipt = {
        "schema_version": 6,
        "status": "frozen_before_repaired_full_selection_and_hidden_validation",
        "source_output": str(source.resolve()),
        "source_artifact_sha256": {path.name: file_sha256(path) for path in required},
        "resolved_config_sha256": file_sha256(resolved_path),
        "resolved_config_hash": config_hash(cfg),
        "repair_specification_sha256": file_sha256(DEFAULT_REPAIR),
        "hidden_data_loaded": False,
        "evaluation_references_loaded": False,
    }
    receipt_path = target["results"] / "prevalidation_repair_receipt.json"
    if receipt_path.exists():
        if json.loads(receipt_path.read_text(encoding="utf-8")) != receipt:
            raise RuntimeError("existing prevalidation repair receipt is incompatible")
    else:
        write_json_atomic(receipt_path, receipt)
    return receipt


def adopt_shared(cfg: dict[str, Any], repair: dict[str, Any], output_dir: str | Path):
    paths = v6_paths(output_dir)
    receipt_path = paths["results"] / "prevalidation_repair_receipt.json"
    if not receipt_path.exists():
        raise RuntimeError("prepare repaired inputs before adopting shared selection")
    source = HERE / "outputs" / str(repair["source_output"])
    parent_paths = v6_paths(source)
    parent_path = parent_paths["shared_results"] / "shared_selection_manifest.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    manifest_path = paths["shared_results"] / "shared_selection_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    design = json.loads(
        (paths["shared_results"] / "design_reference_manifest.json").read_text(encoding="utf-8")
    )
    ids = [row["reference_id"] for row in design["references"]]
    data = TargetProspectiveData.load(
        paths["endpoint"] / "endpoint_data.npz",
        paths["prospective"] / "aggregate_predictions.npz",
    )
    _, evaluators = make_common_reference_evaluators(
        cfg,
        data,
        [row["rollout"] for row in design["references"]],
    )
    with np.load(paths["prospective"] / "v6_selection_crn.npz", allow_pickle=False) as bank_data:
        bank = V4CRNBank(
            np.asarray(bank_data["sampling_z"], dtype=np.float64),
            np.asarray(bank_data["detector_z"], dtype=np.float64),
        ).prefix(int(cfg["v4"]["authoritative_crn_trials"]))
    law_eta = np.asarray(parent["selected"]["Law"]["eta"], dtype=np.float64)
    tangent_eta = np.asarray(parent["selected"]["Tangent"]["eta"], dtype=np.float64)
    limits = [float(parent["risk_limit_by_reference"][reference_id]) for reference_id in ids]
    started = time.perf_counter()
    law = _multi_authoritative(
        evaluators, ids, law_eta, bank, cfg, compute_full=True, beta=0.0
    )
    tangent = _multi_authoritative(
        evaluators, ids, tangent_eta, bank, cfg, compute_full=True, beta=0.0,
        risk_limits=limits,
    )
    if not law["valid"] or not tangent["valid"]:
        raise RuntimeError("positive-raster shared recertification failed")
    archive_path = parent_paths["shared_results"] / "shared_candidate_archive.json"
    manifest = {
        "schema_version": 6,
        "status": "frozen_before_hidden_validation",
        "role": "v6_repaired_shared_law_tangent_selection",
        "selection_input_hashes": _shared_signature(cfg, paths, design),
        "reference_ids": ids,
        "gradient_checks": parent["gradient_checks"],
        "risk_allowance": float(cfg["risk_allowance"]),
        "law_risk_by_reference": parent["law_risk_by_reference"],
        "risk_limit_by_reference": parent["risk_limit_by_reference"],
        "selected": {"Law": law, "Tangent": tangent},
        "adopted_optimization": {
            "source_manifest": str(parent_path.resolve()),
            "source_manifest_sha256": file_sha256(parent_path),
            "source_candidate_archive": str(archive_path.resolve()),
            "source_candidate_archive_sha256": file_sha256(archive_path),
            "inherited_selection_elapsed_seconds": parent["selection_elapsed_seconds"],
            "justification": "Law and Tangent objectives do not invoke authoritative rasterization",
        },
        "selection_elapsed_seconds": time.perf_counter() - started,
        "software": software_metadata(),
        "hidden_data_loaded": False,
        "evaluation_references_loaded": False,
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def run(cfg, repair, output_dir, stage):
    result = None
    if stage in {"prepare", "all"}:
        result = prepare_repair(cfg, repair, output_dir)
    if stage in {"adopt-shared", "all"}:
        result = adopt_shared(cfg, repair, output_dir)
    if stage in {"v6a", "all"}:
        result = select_arm(
            cfg, output_dir, HERE / "configs" / "production_v6a_positive_raster.json"
        )
    if stage in {"v6b", "all"}:
        result = select_arm(
            cfg, output_dir, HERE / "configs" / "production_v6b_positive_raster.json"
        )
    if stage in {"combine", "all"}:
        result = combine_freeze(cfg, output_dir)
    if stage in {"evaluation-references", "all"}:
        result = train_reference_split(cfg, output_dir, "evaluation")
    if stage in {"validate", "all"}:
        result = validate_v6(cfg, output_dir)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair", type=Path, default=DEFAULT_REPAIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execution-profile", type=Path)
    parser.add_argument(
        "--stage",
        choices=("prepare", "adopt-shared", "v6a", "v6b", "combine", "evaluation-references", "validate", "all"),
        required=True,
    )
    args = parser.parse_args()
    cfg, repair = load_repair_config(args.repair)
    if args.execution_profile is not None:
        if args.output_dir.resolve() == DEFAULT_OUTPUT.resolve():
            raise ValueError(
                "an execution profile requires a new explicit --output-dir"
            )
        cfg = apply_execution_profile(cfg, args.execution_profile)
    result = run(cfg, repair, args.output_dir, args.stage)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
