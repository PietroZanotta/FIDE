from __future__ import annotations

"""Run the preregistered fast V6a-only one-reference preflight or Pareto study."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

from common import (
    config_hash,
    fingerprint,
    load_config,
    software_metadata,
    write_json_atomic,
)
from evaluator import ProspectiveEvaluator, make_common_reference_evaluators
from frozen_diagnostic_core import paired_statistics
from mfsi.cache import file_sha256
from prospective_data import TargetProspectiveData
from run_v6_positive_raster import apply_execution_profile
from v4_validate import _mean, _realized_bank_and_moments, _trial_values
from v4_objective import canonical_geometry_key, ensure_v4_crn_bank
from v6_reference_ensemble import prepare_common_inputs, train_reference_split, v6_paths
from v6_select import select_arm, select_shared
from v6_validate import (
    _ensure_hidden,
    _ensure_randomness,
    _freeze_binding,
    _method_summary,
    _two_level_bootstrap,
)


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASE_CONFIG = HERE / "configs" / "production_v6_common.json"
FAST_PROFILE = HERE / "configs" / "v6_fast_execution_exact_v1.json"
V6A_OVERLAY = HERE / "configs" / "production_v6a.json"
DEFAULT_PARETO_OUTPUT = HERE / "outputs" / "prospective_reflected_single_seed_pareto"
PARETO_ALLOWANCES = (0.005, 0.01, 0.02)


def _notify(message: str) -> None:
    if os.environ.get("V6A_RISK_STUDY_DISABLE_NOTIFY") == "1":
        return
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "notify.py"), message],
        cwd=REPO,
        check=True,
    )


def _copy_or_link(source: Path, target: Path) -> None:
    source = source.resolve()
    if target.exists():
        if file_sha256(source) != file_sha256(target):
            raise RuntimeError(f"incompatible frozen artifact: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _tag(allowance: float) -> str:
    percent = 100.0 * float(allowance)
    return f"risk_{f'{percent:g}'.replace('.', 'p')}pct"


def resolve_study_config(mode: str, execution_profile: Path = FAST_PROFILE) -> dict[str, Any]:
    if mode != "single-seed-pareto":
        raise ValueError(f"unknown study mode: {mode}")
    cfg = load_config(BASE_CONFIG)
    cfg["name"] = "prospective_reflected_single_seed_pareto"
    cfg["v6"]["output_name"] = cfg["name"]
    cfg["v6"]["design_reference_ids"] = ["D0"]
    cfg["v6"]["design_reference_training_seeds"] = [20266101]
    cfg["v6"]["design_reference_rollout_seed"] = 20266150
    cfg["v6"]["evaluation_reference_ids"] = ["E0"]
    cfg["v6"]["evaluation_reference_training_seeds"] = [20266111]
    cfg["v6"]["evaluation_reference_rollout_seed"] = 20266160
    cfg["v6"]["empirical_frontier_allowances"] = list(PARETO_ALLOWANCES)
    # This is a new protocol, so use a smaller predeclared search rather than
    # carrying the old three-reference 32-start budget into a one-seed study.
    cfg["v4"]["law_optimizer"]["starts"] = 16
    cfg["v4"]["full_optimizer"].update({"starts": 16, "law_perturbation_starts": 6})
    cfg["v4"]["tangent_optimizer"].update({"starts": 16, "law_perturbation_starts": 6})
    cfg["v4"]["funnel"].update({
        "rescore_candidates": 8,
        "polish_candidates": 3,
        "authoritative_full_finalists": 4,
    })
    cfg["v4"]["full_lbfgs"]["enabled"] = False
    cfg = apply_execution_profile(cfg, execution_profile)
    cfg["v6a_risk_study"] = {
        "schema_version": 1,
        "mode": mode,
        "beta": 0.0,
        "allowances": list(PARETO_ALLOWANCES),
        "reuse_existing_design_references": False,
        "v6b_excluded": True,
        "selection_completed_at_every_allowance_before_evaluation_references": True,
        "previous_allowance_incumbents_are_mandatory": True,
        "source_run": None,
    }
    return cfg


def _input_sources(cfg: dict[str, Any]) -> dict[str, Path]:
    source_root = Path(cfg["v6a_risk_study"]["source_run"]).resolve()
    source = v6_paths(source_root)
    return {
        "endpoint": source["endpoint"] / "endpoint_data.npz",
        "aggregate": source["prospective"] / "aggregate_predictions.npz",
        "build_receipt": source["prospective"] / "build_receipt.json",
        "selection_crn": source["prospective"] / "v6_selection_crn.npz",
        "design_manifest": source["shared_results"] / "design_reference_manifest.json",
        "shared_manifest": source["shared_results"] / "shared_selection_manifest.json",
    }


def prepare_study(cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    paths = v6_paths(output_dir)
    if paths["hidden"].exists() and any(paths["hidden"].iterdir()):
        raise RuntimeError("study preparation refuses to run after hidden validation")
    build_result = prepare_common_inputs(cfg, output_dir)
    bank_path = paths["prospective"] / "v6_selection_crn.npz"
    ensure_v4_crn_bank(bank_path, cfg, int(cfg["v4"]["selection_crn_trials"]))
    paths["results"].mkdir(parents=True, exist_ok=True)
    resolved = paths["results"] / "resolved_config.json"
    if resolved.exists():
        if json.loads(resolved.read_text(encoding="utf-8")) != cfg:
            raise RuntimeError("existing risk-study config is incompatible")
    else:
        write_json_atomic(resolved, cfg)
    receipt = {
        "schema_version": 1,
        "status": "prepared_before_design_selection_and_hidden_validation",
        "config_hash": config_hash(cfg),
        "source_run": None,
        "generated_inputs": {
            "endpoint_sha256": file_sha256(paths["endpoint"] / "endpoint_data.npz"),
            "aggregate_sha256": file_sha256(paths["prospective"] / "aggregate_predictions.npz"),
            "selection_crn_sha256": file_sha256(bank_path),
            "input_binding": build_result["binding"],
        },
        "resolved_config_sha256": file_sha256(resolved),
        "hidden_data_loaded": False,
    }
    write_json_atomic(paths["results"] / "preparation_receipt.json", receipt)
    return receipt


def _adopt_design_references(cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    paths = v6_paths(output_dir)
    target = paths["shared_results"] / "design_reference_manifest.json"
    source_path = _input_sources(cfg)["design_manifest"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    expected = list(cfg["v6"]["design_reference_ids"])
    references = [row for row in source["references"] if row["reference_id"] in expected]
    if [row["reference_id"] for row in references] != expected:
        raise RuntimeError("source design-reference registry does not cover the study")
    for row in references:
        if file_sha256(row["rollout"]) != row["rollout_sha256"]:
            raise RuntimeError("source design-reference rollout hash changed")
    manifest = {
        "schema_version": 6,
        "status": "frozen_before_risk_study_selection_and_hidden_validation",
        "role": "v6a_risk_study_design_reference_ensemble",
        "signature": {
            "config_hash": config_hash(cfg),
            "ids": expected,
            "training_seeds": list(cfg["v6"]["design_reference_training_seeds"]),
            "rollout_seed": int(cfg["v6"]["design_reference_rollout_seed"]),
            "adopted_source_manifest_sha256": file_sha256(source_path),
        },
        "references": references,
        "adoption": {
            "source_manifest": str(source_path.resolve()),
            "reason": "pre-hidden byte-identical reuse; reference training is allowance-independent",
        },
        "selection_access_allowed": True,
    }
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("existing adopted design manifest is incompatible")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(target, manifest)
    return manifest


def prepare_design_references(cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    if bool(cfg["v6a_risk_study"]["reuse_existing_design_references"]):
        result = _adopt_design_references(cfg, output_dir)
    else:
        result = train_reference_split(cfg, output_dir, "design")
    _notify(f"{cfg['name']}: design reference set frozen")
    return result


def _prepare_point_inputs(study_root: Path, point_root: Path, point_cfg: dict[str, Any]) -> None:
    study = v6_paths(study_root)
    point = v6_paths(point_root)
    for source, target in (
        (study["endpoint"] / "endpoint_data.npz", point["endpoint"] / "endpoint_data.npz"),
        (study["prospective"] / "aggregate_predictions.npz", point["prospective"] / "aggregate_predictions.npz"),
        (study["prospective"] / "build_receipt.json", point["prospective"] / "build_receipt.json"),
        (study["prospective"] / "v6_selection_crn.npz", point["prospective"] / "v6_selection_crn.npz"),
    ):
        _copy_or_link(source, target)
    source_manifest = study["shared_results"] / "design_reference_manifest.json"
    target_manifest = point["shared_results"] / "design_reference_manifest.json"
    _copy_or_link(source_manifest, target_manifest)
    point["results"].mkdir(parents=True, exist_ok=True)
    resolved = point["results"] / "resolved_config.json"
    if resolved.exists():
        if json.loads(resolved.read_text(encoding="utf-8")) != point_cfg:
            raise RuntimeError(f"incompatible point config: {point_root}")
    else:
        write_json_atomic(resolved, point_cfg)


def _point_freeze(point_root: Path, allowance: float) -> dict[str, Any]:
    paths = v6_paths(point_root)
    shared_path = paths["shared_results"] / "shared_selection_manifest.json"
    arm_path = paths["arms"] / "v6a_beta_0" / "results" / "frozen_manifest.json"
    shared = json.loads(shared_path.read_text(encoding="utf-8"))
    arm = json.loads(arm_path.read_text(encoding="utf-8"))
    result = {
        "schema_version": 1,
        "status": "v6a_point_frozen_before_any_study_evaluation_reference_or_hidden_data",
        "allowance": float(allowance),
        "beta": 0.0,
        "selected": {
            "Law": shared["selected"]["Law"],
            "Tangent": shared["selected"]["Tangent"],
            "v6a": arm["selected"],
        },
        "shared_manifest_sha256": file_sha256(shared_path),
        "v6a_manifest_sha256": file_sha256(arm_path),
        "v6b_generated": False,
        "hidden_data_loaded": False,
        "evaluation_references_loaded": False,
    }
    write_json_atomic(paths["results"] / "v6a_point_frozen_manifest.json", result)
    return result


def select_study(cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    paths = v6_paths(output_dir)
    combined_path = paths["results"] / "combined_frozen_manifest.json"
    if combined_path.exists():
        existing = json.loads(combined_path.read_text(encoding="utf-8"))
        if existing.get("config_hash") != config_hash(cfg):
            raise RuntimeError("existing combined risk-study freeze is incompatible")
        for point in existing["points"]:
            manifest = Path(point["point_manifest"])
            if file_sha256(manifest) != point["point_manifest_sha256"]:
                raise RuntimeError("frozen risk-study point manifest hash changed")
        print("[v6a-risk-study] reusing all frozen allowance selections", flush=True)
        return existing
    if (paths["shared_results"] / "evaluation_reference_manifest.json").exists():
        raise RuntimeError("risk-study selection refuses to run after evaluation references")
    allowances = [float(value) for value in cfg["v6a_risk_study"]["allowances"]]
    fixed_law = None
    previous_tangent = None
    previous_full = None
    points = []
    previous_predicted_action = float("inf")
    for allowance in allowances:
        point_root = output_dir / "points" / _tag(allowance)
        point_cfg = json.loads(json.dumps(cfg))
        point_cfg["risk_allowance"] = float(allowance)
        point_cfg["name"] = f"{cfg['name']}_{_tag(allowance)}"
        point_cfg["v6"]["output_name"] = point_cfg["name"]
        point_cfg["v6a_pareto"] = {
            "allowance": float(allowance),
            "fixed_law_eta": fixed_law,
            "tangent_incumbent_eta": previous_tangent,
            "full_incumbent_eta": previous_full,
        }
        _prepare_point_inputs(output_dir, point_root, point_cfg)
        shared = select_shared(point_cfg, point_root)
        if fixed_law is None:
            fixed_law = shared["selected"]["Law"]["eta"]
        arm = select_arm(point_cfg, point_root, V6A_OVERLAY)
        frozen = _point_freeze(point_root, allowance)
        current_action = float(arm["selected"]["full_distribution"]["mean"])
        if current_action > previous_predicted_action + 1.0e-10:
            raise RuntimeError("certified V6a selection frontier is not nested")
        previous_predicted_action = min(previous_predicted_action, current_action)
        previous_tangent = shared["selected"]["Tangent"]["eta"]
        previous_full = arm["selected"]["eta"]
        points.append({
            "allowance": allowance,
            "point_root": str(point_root.resolve()),
            "point_manifest": str((v6_paths(point_root)["results"] / "v6a_point_frozen_manifest.json").resolve()),
            "point_manifest_sha256": file_sha256(v6_paths(point_root)["results"] / "v6a_point_frozen_manifest.json"),
            "selected_predicted_full_mean": current_action,
            "selected": frozen["selected"],
        })
        _notify(f"{cfg['name']}: V6a allowance {100.0 * allowance:g}% frozen")
    combined = {
        "schema_version": 1,
        "status": "all_v6a_allowances_frozen_before_evaluation_references_and_hidden_validation",
        "experiment": cfg["name"],
        "config_hash": config_hash(cfg),
        "allowances": allowances,
        "points": points,
        "fixed_law_eta": fixed_law,
        "v6b_generated": False,
        "evaluation_reference_registry_frozen_but_not_generated": {
            "ids": cfg["v6"]["evaluation_reference_ids"],
            "training_seeds": cfg["v6"]["evaluation_reference_training_seeds"],
            "rollout_seed": cfg["v6"]["evaluation_reference_rollout_seed"],
        },
        "hidden_seed_registry": {
            key: cfg["seeds"][key] for key in (
                "validation_physical", "validation_sampling",
                "validation_detector", "validation_bootstrap",
            )
        },
        "software": software_metadata(),
    }
    write_json_atomic(combined_path, combined)
    return combined


def prepare_evaluation_references(cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    result = train_reference_split(cfg, output_dir, "evaluation")
    _notify(f"{cfg['name']}: evaluation reference set frozen after all selections")
    return result


def _evaluate_geometry(evaluators, ids, eta, states, indices, detector_z):
    first = evaluators[ids[0]]
    bank, mean, second, qoi = _realized_bank_and_moments(
        first, np.asarray(eta, dtype=np.float64), states, indices, detector_z
    )
    by_reference = {
        reference_id: evaluators[reference_id].evaluate_population(
            np.asarray(eta, dtype=np.float64), mean, second, qoi, bank, compute_full=True
        )
        for reference_id in ids
    }
    return _method_summary(eta, by_reference)


def validate_study(cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    paths = v6_paths(output_dir)
    binding, freeze_hashes = _freeze_binding(paths)
    validation_source_hash = fingerprint({
        "config_hash": config_hash(cfg),
        "run_v6a_risk_study.py": file_sha256(Path(__file__)),
        "v6_validate.py": file_sha256(HERE / "v6_validate.py"),
        "evaluator.py": file_sha256(HERE / "evaluator.py"),
        "reflected_raster.py": file_sha256(HERE / "reflected_raster.py"),
    })
    result_path = paths["results"] / "validation_result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            existing.get("freeze_binding") == binding
            and existing.get("validation_source_hash") == validation_source_hash
        ):
            return existing
        raise RuntimeError("existing study validation belongs to another freeze")
    combined = json.loads(
        (paths["results"] / "combined_frozen_manifest.json").read_text(encoding="utf-8")
    )
    evaluation_manifest = json.loads(
        (paths["shared_results"] / "evaluation_reference_manifest.json").read_text(encoding="utf-8")
    )
    ids = [row["reference_id"] for row in evaluation_manifest["references"]]
    if ids != list(cfg["v6"]["evaluation_reference_ids"]):
        raise RuntimeError("evaluation reference IDs differ from frozen registry")
    data = TargetProspectiveData.load(
        paths["endpoint"] / "endpoint_data.npz",
        paths["prospective"] / "aggregate_predictions.npz",
    )
    _, evaluator_list = make_common_reference_evaluators(
        cfg,
        data,
        [row["rollout"] for row in evaluation_manifest["references"]],
    )
    evaluators = dict(zip(ids, evaluator_list))
    hidden_path, states = _ensure_hidden(cfg, paths, binding)
    randomness_path, indices, detector_z = _ensure_randomness(
        cfg, paths, binding, states.shape[1]
    )
    started = time.perf_counter()
    point_results = []
    geometry_cache: dict[tuple[float, ...], dict[str, Any]] = {}
    geometry_cache_hits = 0

    def evaluate_once(eta):
        nonlocal geometry_cache_hits
        key = canonical_geometry_key(np.asarray(eta, dtype=np.float64))
        if key in geometry_cache:
            geometry_cache_hits += 1
            return geometry_cache[key]
        value = _evaluate_geometry(
            evaluators, ids, eta, states, indices, detector_z
        )
        geometry_cache[key] = value
        return value

    for point in combined["points"]:
        allowance = float(point["allowance"])
        methods = {
            method: evaluate_once(point["selected"][method]["eta"])
            for method in ("Law", "Tangent", "v6a")
        }
        law = methods["Law"]
        full = methods["v6a"]
        base = np.concatenate([
            _trial_values(law["realized_by_reference"][key], "full_action")
            for key in ids
        ])
        treatment = np.concatenate([
            _trial_values(full["realized_by_reference"][key], "full_action")
            for key in ids
        ])
        comparison = paired_statistics(
            base, treatment,
            bootstrap_seed=int(cfg["seeds"]["validation_bootstrap"])
            + int(round(10000 * allowance)),
        )
        risk_pass = {
            key: full["risk_by_reference"][key]
            <= (1.0 + allowance) * law["risk_by_reference"][key]
            for key in ids
        }
        comparison_by_reference = {}
        differences_by_reference = {}
        for offset, key in enumerate(ids):
            law_values = _trial_values(
                law["realized_by_reference"][key], "full_action"
            )
            full_values = _trial_values(
                full["realized_by_reference"][key], "full_action"
            )
            comparison_by_reference[key] = paired_statistics(
                law_values,
                full_values,
                bootstrap_seed=int(cfg["seeds"]["validation_bootstrap"])
                + int(round(10000 * allowance)) + offset + 1,
            )
            differences_by_reference[key] = full_values - law_values
        numerical_pass = bool(all(
            certificate["invalid_trial_count"] == 0
            and certificate["nan_or_inf_count"] == 0
            and certificate["all_full_solvers_converged"]
            for certificate in full["certification_by_reference"].values()
        ))
        point_result = {
            "allowance": allowance,
            "methods": methods,
            "v6a_minus_law": comparison,
            "v6a_minus_law_by_reference": comparison_by_reference,
            "two_level_reference_bootstrap": _two_level_bootstrap(
                differences_by_reference,
                int(cfg["seeds"]["validation_bootstrap"])
                + 1000 + int(round(10000 * allowance)),
            ),
            "risk_pass_by_reference": risk_pass,
            "all_reference_risk_pass": bool(all(risk_pass.values())),
            "paired_action_ci_below_zero": bool(comparison["paired_t_95_ci"][1] < 0.0),
            "numerical_certification_pass": numerical_pass,
            "strict_success": bool(
                all(risk_pass.values())
                and comparison["paired_t_95_ci"][1] < 0.0
                and numerical_pass
            ),
        }
        point_results.append(point_result)
        point_root = Path(point["point_root"])
        write_json_atomic(
            v6_paths(point_root)["results"] / "validation_result.json", point_result
        )
        _notify(f"{cfg['name']}: V6a allowance {100.0 * allowance:g}% validated")
    result = {
        "schema_version": 1,
        "experiment": cfg["name"],
        "freeze_binding": binding,
        "validation_source_hash": validation_source_hash,
        "freeze_hashes": freeze_hashes,
        "reference_ids": ids,
        "hidden_state_bank": str(hidden_path),
        "hidden_randomness": str(randomness_path),
        "points": point_results,
        "exact_geometry_validation_cache": {
            "unique_geometries": len(geometry_cache),
            "cache_hits": geometry_cache_hits,
        },
        "validation_elapsed_seconds": time.perf_counter() - started,
    }
    write_json_atomic(result_path, result)
    rows = []
    for point in point_results:
        law = point["methods"]["Law"]
        full = point["methods"]["v6a"]
        rows.append({
            "allowance_percent": 100.0 * point["allowance"],
            "law_full_action": law["equal_reference_mean_full_action"],
            "v6a_full_action": full["equal_reference_mean_full_action"],
            "full_reduction_vs_law": 1.0 - full["equal_reference_mean_full_action"] / law["equal_reference_mean_full_action"],
            "law_risk": law["equal_reference_mean_risk"],
            "v6a_risk": full["equal_reference_mean_risk"],
            "all_reference_risk_pass": point["all_reference_risk_pass"],
            "paired_action_ci_low": point["v6a_minus_law"]["paired_t_95_ci"][0],
            "paired_action_ci_high": point["v6a_minus_law"]["paired_t_95_ci"][1],
            "strict_success": point["strict_success"],
        })
    with (paths["results"] / "pareto.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json_atomic(paths["results"] / "pareto.json", {"points": rows})
    lines = [
        "# Prospective-vortices V6a risk study", "",
        "Every allowance was independently gradient-optimized and frozen before evaluation-reference training and hidden validation. V6b was not generated.", "",
        "| Allowance | Law action | V6a action | Reduction | Law risk | V6a risk | Risk pass | Strict success |", 
        "|--:|--:|--:|--:|--:|--:|:--|:--|",
    ]
    for row in rows:
        lines.append(
            f"| {row['allowance_percent']:g}% | {row['law_full_action']:.6g} | "
            f"{row['v6a_full_action']:.6g} | {100*row['full_reduction_vs_law']:.3f}% | "
            f"{row['law_risk']:.6g} | {row['v6a_risk']:.6g} | "
            f"{'PASS' if row['all_reference_risk_pass'] else 'FAIL'} | "
            f"{'PASS' if row['strict_success'] else 'FAIL'} |"
        )
    (paths["results"] / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def run(cfg: dict[str, Any], output_dir: Path, stage: str):
    paths = v6_paths(output_dir)
    if stage == "all" and (paths["results"] / "validation_result.json").exists():
        return validate_study(cfg, output_dir)
    if stage == "all" and paths["hidden"].exists() and any(paths["hidden"].iterdir()):
        return validate_study(cfg, output_dir)
    if stage in {"prepare", "all"}:
        prepare_study(cfg, output_dir)
    if stage in {"design-references", "all"}:
        prepare_design_references(cfg, output_dir)
    if stage in {"select", "all"}:
        select_study(cfg, output_dir)
    if stage in {"evaluation-references", "all"}:
        prepare_evaluation_references(cfg, output_dir)
    if stage in {"validate", "all"}:
        return validate_study(cfg, output_dir)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("single-seed-pareto",),
        default="single-seed-pareto",
    )
    parser.add_argument("--execution-profile", type=Path, default=FAST_PROFILE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("prepare", "design-references", "select", "evaluation-references", "validate", "all"),
        default="all",
    )
    args = parser.parse_args()
    cfg = resolve_study_config(args.mode, args.execution_profile.resolve())
    output = args.output_dir or DEFAULT_PARETO_OUTPUT
    if args.dry_run:
        print(json.dumps({
            "mode": args.mode,
            "output_dir": str(output.resolve()),
            "stage": args.stage,
            "allowances": cfg["v6a_risk_study"]["allowances"],
            "design_reference_ids": cfg["v6"]["design_reference_ids"],
            "evaluation_reference_ids": cfg["v6"]["evaluation_reference_ids"],
            "beta": 0.0,
            "v6b_excluded": True,
            "fast_execution": cfg["v6_fast_execution"],
        }, indent=2))
        return
    result = run(cfg, output.resolve(), args.stage)
    if result is not None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
