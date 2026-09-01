from __future__ import annotations

"""Repair the one-seed prospective frontier after the D0 Law reoptimization."""

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
from typing import Any

from common import config_hash, fingerprint, software_metadata, write_json_atomic
from mfsi.cache import file_sha256
from run_v6a_risk_study import validate_study
from v6_reference_ensemble import train_reference_split, v6_paths
from v6_select import select_arm


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "outputs" / "prospective_reflected_single_seed_pareto"
LAW_AUDIT = HERE / "outputs" / "law_reoptimization_audit" / "results" / "law_reoptimization_result.json"
DEFAULT_OUTPUT = HERE / "outputs" / "prospective_reflected_single_seed_pareto_repaired"
OVERLAY = HERE / "configs" / "production_v6a.json"
ALLOWANCES = (0.005, 0.01, 0.02)
SAVED_TWO_PERCENT_CANDIDATE = "full-grad-001-polished"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_or_link(source: Path, target: Path) -> None:
    source = source.resolve()
    if target.exists():
        if file_sha256(source) != file_sha256(target):
            raise RuntimeError(f"incompatible reused artifact: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _tag(allowance: float) -> str:
    return f"risk_{f'{100.0 * allowance:g}'.replace('.', 'p')}pct"


def resolve_config() -> dict[str, Any]:
    cfg = _read(SOURCE / "results" / "resolved_config.json")
    cfg["name"] = "prospective_reflected_single_seed_pareto_repaired"
    cfg["v6"]["output_name"] = cfg["name"]
    cfg["v6"]["evaluation_reference_ids"] = ["E1"]
    cfg["v6"]["evaluation_reference_training_seeds"] = [20266112]
    cfg["v6"]["evaluation_reference_rollout_seed"] = 20266161
    cfg["seeds"].update(
        {
            "validation_physical": 20270021,
            "validation_sampling": 20270022,
            "validation_detector": 20270023,
            "validation_bootstrap": 20270024,
        }
    )
    cfg["v6a_risk_study"] = {
        "schema_version": 2,
        "mode": "repaired-single-seed-pareto",
        "beta": 0.0,
        "allowances": list(ALLOWANCES),
        "repaired_law_audit": str(LAW_AUDIT.resolve()),
        "full_rerun_allowances": [0.005, 0.01],
        "two_percent_policy": "best of saved authoritative escape hatch and repaired 1% incumbent",
        "evaluation_reference": "fresh E1 after repaired three-point freeze",
        "v6b_excluded": True,
    }
    return cfg


def _prepare_root(cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    source = v6_paths(SOURCE)
    target = v6_paths(output)
    reused = {
        "endpoint": (source["endpoint"] / "endpoint_data.npz", target["endpoint"] / "endpoint_data.npz"),
        "aggregate": (source["prospective"] / "aggregate_predictions.npz", target["prospective"] / "aggregate_predictions.npz"),
        "build_receipt": (source["prospective"] / "build_receipt.json", target["prospective"] / "build_receipt.json"),
        "selection_crn": (source["prospective"] / "v6_selection_crn.npz", target["prospective"] / "v6_selection_crn.npz"),
    }
    for source_path, target_path in reused.values():
        _copy_or_link(source_path, target_path)
    target["results"].mkdir(parents=True, exist_ok=True)
    target["shared_results"].mkdir(parents=True, exist_ok=True)
    resolved = target["results"] / "resolved_config.json"
    if resolved.exists() and _read(resolved) != cfg:
        raise RuntimeError("existing repaired resolved config is incompatible")
    if not resolved.exists():
        write_json_atomic(resolved, cfg)
    law_audit = _read(LAW_AUDIT)
    law_hash = file_sha256(LAW_AUDIT)
    if law_audit["status"] != "d0_law_reoptimization_complete":
        raise RuntimeError("D0 Law audit is not complete")

    source_manifest_path = source["shared_results"] / "design_reference_manifest.json"
    source_manifest = _read(source_manifest_path)
    source_reference = source_manifest["references"][0]
    rollout = source["design_references"] / "D0" / "endpoint_reference" / "reference_rollout.npz"
    checkpoint = source["design_references"] / "D0" / "endpoint_reference" / "reference_checkpoint.npz"
    receipt = source["design_references"] / "D0" / "endpoint_reference" / "reference_receipt.json"
    if file_sha256(rollout) != source_reference["rollout_sha256"]:
        raise RuntimeError("reused D0 rollout hash differs from frozen source manifest")
    reference = {
        **source_reference,
        "rollout": str(rollout.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "receipt": str(receipt.resolve()),
    }
    design_manifest = {
        "schema_version": 6,
        "status": "reused_byte_identically_for_post_hoc_d0_frontier_repair",
        "role": "repaired_study_design_reference_ensemble",
        "references": [reference],
        "selection_access_allowed": True,
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "law_audit_sha256": law_hash,
    }
    write_json_atomic(target["shared_results"] / "design_reference_manifest.json", design_manifest)
    input_binding = {
        "schema_version": 1,
        "role": "repaired_study_byte_identical_d0_inputs",
        "config_hash": config_hash(cfg),
        "source_run": str(SOURCE.resolve()),
        "hashes": {name: file_sha256(target_path) for name, (_, target_path) in reused.items()},
        "design_reference_manifest_sha256": file_sha256(target["shared_results"] / "design_reference_manifest.json"),
        "law_audit_sha256": law_hash,
        "evaluation_reference_generated": False,
        "hidden_validation_loaded": False,
    }
    write_json_atomic(target["shared_results"] / "input_binding.json", input_binding)
    return law_audit


def _point_config(cfg: dict[str, Any], allowance: float, full_incumbent: list[float] | None) -> dict[str, Any]:
    point_cfg = copy.deepcopy(cfg)
    point_cfg["risk_allowance"] = float(allowance)
    point_cfg["name"] = f"{cfg['name']}_{_tag(allowance)}"
    point_cfg["v6"]["output_name"] = point_cfg["name"]
    point_cfg["v6a_pareto"] = {
        "allowance": float(allowance),
        "fixed_law_eta": _read(LAW_AUDIT)["reoptimized_law"]["eta"],
        "tangent_incumbent_eta": None,
        "full_incumbent_eta": full_incumbent,
    }
    return point_cfg


def _prepare_point(output: Path, point: Path, point_cfg: dict[str, Any]) -> None:
    root_paths = v6_paths(output)
    point_paths = v6_paths(point)
    for source, target in (
        (root_paths["endpoint"] / "endpoint_data.npz", point_paths["endpoint"] / "endpoint_data.npz"),
        (root_paths["prospective"] / "aggregate_predictions.npz", point_paths["prospective"] / "aggregate_predictions.npz"),
        (root_paths["prospective"] / "build_receipt.json", point_paths["prospective"] / "build_receipt.json"),
        (root_paths["prospective"] / "v6_selection_crn.npz", point_paths["prospective"] / "v6_selection_crn.npz"),
    ):
        _copy_or_link(source, target)
    point_paths["shared_results"].mkdir(parents=True, exist_ok=True)
    design = _read(root_paths["shared_results"] / "design_reference_manifest.json")
    write_json_atomic(point_paths["shared_results"] / "design_reference_manifest.json", design)
    point_paths["results"].mkdir(parents=True, exist_ok=True)
    resolved = point_paths["results"] / "resolved_config.json"
    if resolved.exists() and _read(resolved) != point_cfg:
        raise RuntimeError(f"incompatible repaired point config: {point}")
    if not resolved.exists():
        write_json_atomic(resolved, point_cfg)


def _write_shared_manifest(point: Path, cfg: dict[str, Any], allowance: float, law: dict[str, Any]) -> dict[str, Any]:
    paths = v6_paths(point)
    risk = float(law["d0_risk"])
    eta = list(law["eta"])
    law_selected = {
        "eta": eta,
        "centers": law["centers"],
        "reference_ids": ["D0"],
        "risk_by_reference": {"D0": risk},
        "mean_risk": risk,
        "valid": True,
        "geometry_valid": True,
        "risk_feasible_all_references": True,
        "role": "reoptimized D0 Law anchor",
    }
    manifest = {
        "schema_version": 7,
        "status": "repaired_law_frozen_before_full_reselection_and_e1",
        "role": "repaired_shared_law_anchor",
        "reference_ids": ["D0"],
        "risk_allowance": float(allowance),
        "law_risk_by_reference": {"D0": risk},
        "risk_limit_by_reference": {"D0": (1.0 + float(allowance)) * risk},
        "selected": {"Law": law_selected, "Tangent": law_selected},
        "law_audit": str(LAW_AUDIT.resolve()),
        "law_audit_sha256": file_sha256(LAW_AUDIT),
        "config_hash": config_hash(cfg),
        "hidden_data_loaded": False,
        "evaluation_references_loaded": False,
    }
    write_json_atomic(paths["shared_results"] / "shared_selection_manifest.json", manifest)
    return manifest


def _freeze_point(point: Path, allowance: float, shared: dict[str, Any], selected: dict[str, Any], arm_manifest: Path) -> dict[str, Any]:
    paths = v6_paths(point)
    shared_path = paths["shared_results"] / "shared_selection_manifest.json"
    result = {
        "schema_version": 2,
        "status": "repaired_v6a_point_frozen_before_e1_and_hidden_validation",
        "allowance": float(allowance),
        "beta": 0.0,
        "selected": {
            "Law": shared["selected"]["Law"],
            "Tangent": shared["selected"]["Tangent"],
            "v6a": selected,
        },
        "shared_manifest_sha256": file_sha256(shared_path),
        "v6a_manifest_sha256": file_sha256(arm_manifest),
        "hidden_data_loaded": False,
        "evaluation_references_loaded": False,
    }
    path = paths["results"] / "v6a_point_frozen_manifest.json"
    write_json_atomic(path, result)
    return result


def _saved_two_percent_candidate() -> tuple[dict[str, Any], Path]:
    archive = SOURCE / "points" / "risk_2pct" / "arms" / "v6a_beta_0" / "results" / "candidate_archive.json"
    data = _read(archive)
    matches = [row for row in data["authoritative_finalists"] if row["candidate_id"] == SAVED_TWO_PERCENT_CANDIDATE]
    if len(matches) != 1:
        raise RuntimeError("saved 2% escape-hatch candidate is missing or ambiguous")
    return matches[0], archive


def _adopt_two_percent(
    point: Path,
    cfg: dict[str, Any],
    shared: dict[str, Any],
    repaired_one_percent: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    saved, source_archive = _saved_two_percent_candidate()
    limit = float(shared["risk_limit_by_reference"]["D0"])
    candidates = [saved, repaired_one_percent]
    feasible = [
        row for row in candidates
        if float(row["risk_by_reference"]["D0"]) <= limit + float(cfg["validity"]["risk_constraint_tolerance"])
    ]
    if not feasible:
        raise RuntimeError("neither the saved 2% candidate nor repaired 1% incumbent is feasible")
    selected = min(feasible, key=lambda row: float(row["full_distribution"]["mean"]))
    paths = v6_paths(point)
    results = paths["arms"] / "v6a_beta_0" / "results"
    results.mkdir(parents=True, exist_ok=True)
    archive = {
        "schema_version": 1,
        "role": "2pct_authoritative_escape_hatch",
        "source_candidate_archive": str(source_archive.resolve()),
        "source_candidate_archive_sha256": file_sha256(source_archive),
        "saved_candidate": saved,
        "repaired_1pct_incumbent": repaired_one_percent,
        "risk_limit_d0": limit,
        "selected_candidate_id": selected["candidate_id"],
    }
    archive_path = results / "candidate_archive.json"
    write_json_atomic(archive_path, archive)
    manifest = {
        "schema_version": 7,
        "status": "frozen_before_e1_and_hidden_validation",
        "experiment": "v6a_beta_0",
        "beta": 0.0,
        "selection_mode": "authoritative_escape_hatch_without_new_2pct_optimization",
        "risk_limit_by_reference": shared["risk_limit_by_reference"],
        "selected": selected,
        "authoritative_finalists": feasible,
        "candidate_archive_sha256": file_sha256(archive_path),
        "hidden_data_loaded": False,
        "evaluation_references_loaded": False,
        "software": software_metadata(),
    }
    manifest_path = results / "frozen_manifest.json"
    write_json_atomic(manifest_path, manifest)
    return selected, manifest_path


def select_repaired(cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    paths = v6_paths(output)
    combined_path = paths["results"] / "combined_frozen_manifest.json"
    if combined_path.exists():
        existing = _read(combined_path)
        if existing.get("config_hash") != config_hash(cfg):
            raise RuntimeError("existing repaired combined freeze is incompatible")
        print("[repaired-study] reusing completed three-point freeze", flush=True)
        return existing
    if (paths["shared_results"] / "evaluation_reference_manifest.json").exists():
        raise RuntimeError("repaired selection refuses to run after E1 generation")
    law_audit = _prepare_root(cfg, output)
    law = law_audit["reoptimized_law"]
    points = []
    previous_full: list[float] | None = None
    repaired_one_percent: dict[str, Any] | None = None
    for allowance in ALLOWANCES:
        point = output / "points" / _tag(allowance)
        point_cfg = _point_config(cfg, allowance, previous_full)
        _prepare_point(output, point, point_cfg)
        shared = _write_shared_manifest(point, point_cfg, allowance, law)
        if allowance < 0.02:
            arm = select_arm(point_cfg, point, OVERLAY)
            selected = arm["selected"]
            manifest_path = v6_paths(point)["arms"] / "v6a_beta_0" / "results" / "frozen_manifest.json"
            previous_full = list(selected["eta"])
            if allowance == 0.01:
                repaired_one_percent = selected
        else:
            if repaired_one_percent is None:
                raise RuntimeError("repair the 1% point before adopting 2%")
            selected, manifest_path = _adopt_two_percent(
                point, point_cfg, shared, repaired_one_percent
            )
        frozen = _freeze_point(point, allowance, shared, selected, manifest_path)
        point_manifest = v6_paths(point)["results"] / "v6a_point_frozen_manifest.json"
        points.append(
            {
                "allowance": allowance,
                "point_root": str(point.resolve()),
                "point_manifest": str(point_manifest.resolve()),
                "point_manifest_sha256": file_sha256(point_manifest),
                "selected_predicted_full_mean": float(selected["full_distribution"]["mean"]),
                "selected": frozen["selected"],
                "selection_mode": "rerun" if allowance < 0.02 else "escape_hatch",
            }
        )
    combined = {
        "schema_version": 2,
        "status": "repaired_three_point_frontier_frozen_before_e1_and_hidden_validation",
        "experiment": cfg["name"],
        "config_hash": config_hash(cfg),
        "repair_source_hash": fingerprint(
            {
                "run_repaired_study.py": file_sha256(Path(__file__)),
                "v6_select.py": file_sha256(HERE / "v6_select.py"),
                "law_audit": file_sha256(LAW_AUDIT),
            }
        ),
        "allowances": list(ALLOWANCES),
        "points": points,
        "fixed_law_eta": law["eta"],
        "law_d0_risk": law["d0_risk"],
        "evaluation_reference_registry_frozen_but_not_generated": {
            "ids": cfg["v6"]["evaluation_reference_ids"],
            "training_seeds": cfg["v6"]["evaluation_reference_training_seeds"],
            "rollout_seed": cfg["v6"]["evaluation_reference_rollout_seed"],
        },
        "hidden_seed_registry": {
            key: cfg["seeds"][key]
            for key in ("validation_physical", "validation_sampling", "validation_detector", "validation_bootstrap")
        },
        "software": software_metadata(),
    }
    write_json_atomic(combined_path, combined)
    return combined


def run(cfg: dict[str, Any], output: Path, stage: str) -> dict[str, Any]:
    output = output.resolve()
    _prepare_root(cfg, output)
    result: dict[str, Any] = {}
    if stage in {"select", "all"}:
        result = select_repaired(cfg, output)
    if stage in {"evaluation-reference", "all"}:
        train_reference_split(cfg, output, "evaluation")
    if stage in {"validate", "all"}:
        result = validate_study(cfg, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stage",
        choices=("select", "evaluation-reference", "validate", "all"),
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = resolve_config()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "stage": args.stage,
                    "allowances": list(ALLOWANCES),
                    "full_rerun_allowances": [0.005, 0.01],
                    "two_percent": "authoritative escape hatch",
                    "evaluation_reference": cfg["v6"]["evaluation_reference_ids"],
                    "hidden_seeds": {
                        key: cfg["seeds"][key]
                        for key in (
                            "validation_physical",
                            "validation_sampling",
                            "validation_detector",
                            "validation_bootstrap",
                        )
                    },
                },
                indent=2,
            )
        )
        return
    result = run(cfg, args.output, args.stage)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
