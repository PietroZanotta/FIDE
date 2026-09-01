from __future__ import annotations

"""Held-out multi-reference validation for the frozen v6 beta ablation."""

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from aggregate_qois import qoi_features
from common import config_hash, fingerprint, load_config, write_json_atomic
from evaluator import (
    AggregateObservationBank,
    ProspectiveEvaluator,
    make_common_reference_evaluators,
)
from frozen_diagnostic_core import paired_statistics
from mfsi.cache import file_sha256
from physical import truth_from_config
from prospective_data import TargetProspectiveData
from v4_objective import distribution
from v4_validate import _certification, _mean, _realized_bank_and_moments, _trial_values
from v6_reference_ensemble import DEFAULT_CONFIG, DEFAULT_OUTPUT, load_reference_manifest, v6_paths

jax.config.update("jax_enable_x64", True)


def _freeze_binding(paths) -> tuple[str, dict[str, str]]:
    files = {
        "combined_manifest": paths["results"] / "combined_frozen_manifest.json",
        "evaluation_reference_manifest": paths["shared_results"] / "evaluation_reference_manifest.json",
    }
    for path in files.values():
        if not path.exists():
            raise RuntimeError(f"v6 validation prerequisite missing: {path}")
    hashes = {key: file_sha256(path) for key, path in files.items()}
    return fingerprint(hashes), hashes


def _ensure_hidden(cfg, paths, binding):
    paths["hidden"].mkdir(parents=True, exist_ok=True)
    path = paths["hidden"] / "v6_hidden_state_bank.npz"
    seed = int(cfg["seeds"]["validation_physical"])
    signature = fingerprint({
        "schema_version": 6, "config_hash": config_hash(cfg), "freeze_binding": binding,
        "seed": seed, "particles": int(cfg["truth"]["hidden_validation_particles"]),
    })
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if str(np.asarray(data["signature"]).item()) != signature:
                raise RuntimeError("existing v6 hidden state bank is incompatible")
            return path, np.asarray(data["states"], dtype=np.float64)
    truth = truth_from_config(cfg)
    times = jnp.linspace(0.0, 1.0, int(cfg["time"]["scientific_nodes"]), dtype=jnp.float64)
    bank = truth.make_bank(
        seed=seed, n=int(cfg["truth"]["hidden_validation_particles"]), times=times,
        substeps_per_interval=int(cfg["truth"]["rk4_substeps_per_interval"]),
    )
    states = np.asarray(bank.particles, dtype=np.float64)
    np.savez_compressed(
        path, role=np.asarray("v6_hidden_physical_bank_created_after_both_arm_freezes"),
        signature=np.asarray(signature), freeze_binding=np.asarray(binding),
        physical_seed=np.asarray(seed), times=np.asarray(times), states=states,
    )
    return path, states


def _ensure_randomness(cfg, paths, binding, particle_n):
    path = paths["hidden"] / "v6_hidden_observation_randomness.npz"
    trials = int(cfg["v4"]["validation_trials"])
    sample_shape = (trials, int(cfg["time"]["acquisition_nodes"]), int(cfg["measurement"]["finite_n"]))
    detector_shape = (trials, int(cfg["time"]["acquisition_nodes"]), int(cfg["measurement"]["n_sensors"]))
    sampling_seed = int(cfg["seeds"]["validation_sampling"])
    detector_seed = int(cfg["seeds"]["validation_detector"])
    signature = fingerprint({
        "schema_version": 6, "freeze_binding": binding, "particle_n": int(particle_n),
        "sample_shape": sample_shape, "detector_shape": detector_shape,
        "sampling_seed": sampling_seed, "detector_seed": detector_seed,
    })
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if str(np.asarray(data["signature"]).item()) != signature:
                raise RuntimeError("existing v6 hidden observation randomness is incompatible")
            return path, np.asarray(data["sample_indices"]), np.asarray(data["detector_z"])
    sample_rng = np.random.default_rng(sampling_seed)
    detector_rng = np.random.default_rng(detector_seed)
    indices = sample_rng.integers(0, particle_n, size=sample_shape, dtype=np.int32)
    detector_z = detector_rng.standard_normal(detector_shape)
    np.savez_compressed(
        path, role=np.asarray("v6_paired_hidden_observation_randomness"),
        signature=np.asarray(signature), freeze_binding=np.asarray(binding),
        sampling_seed=np.asarray(sampling_seed), detector_seed=np.asarray(detector_seed),
        particle_n=np.asarray(particle_n), sample_indices=indices, detector_z=detector_z,
    )
    return path, indices, detector_z


def _two_level_bootstrap(differences: dict[str, np.ndarray], seed: int, draws: int = 20000):
    ids = list(differences)
    matrix = np.stack([np.asarray(differences[key], dtype=np.float64) for key in ids])
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError("two-level bootstrap requires aligned finite trials")
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        ref_idx = rng.integers(0, len(ids), size=len(ids))
        means = []
        for index in ref_idx:
            row = matrix[index]
            trial_idx = rng.integers(0, len(row), size=len(row))
            means.append(float(np.mean(row[trial_idx])))
        values[draw] = float(np.mean(means))
    return {
        "reference_ids": ids,
        "draws": int(draws),
        "mean": float(np.mean(matrix)),
        "95_ci": np.quantile(values, [0.025, 0.975]).tolist(),
        "interpretation": (
            f"low-resolution cluster bootstrap over {len(ids)} fixed held-out "
            "reference seed(s)"
        ),
    }


def _method_summary(eta, by_reference):
    all_actions = np.concatenate([_trial_values(result, "full_action") for result in by_reference.values()])
    return {
        "eta": np.asarray(eta).tolist(),
        "centers": np.asarray(eta).reshape((-1, 2)).tolist(),
        "risk_by_reference": {key: _mean(value, "risk") for key, value in by_reference.items()},
        "full_action_by_reference": {key: _mean(value, "full_action") for key, value in by_reference.items()},
        "equal_reference_mean_risk": float(np.mean([_mean(value, "risk") for value in by_reference.values()])),
        "equal_reference_mean_full_action": float(np.mean([_mean(value, "full_action") for value in by_reference.values()])),
        "pooled_full_distribution": distribution(all_actions),
        "certification_by_reference": {key: _certification(value) for key, value in by_reference.items()},
        "realized_by_reference": by_reference,
    }


def _write_outputs(paths, result):
    write_json_atomic(paths["results"] / "validation_result.json", result)
    trial_rows = []
    for method, block in result["methods"].items():
        for reference_id, realized in block["realized_by_reference"].items():
            for row in realized["trials"]:
                trial_rows.append({"method": method, "reference_id": reference_id, **row})
    with (paths["results"] / "validation_trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trial_rows[0]))
        writer.writeheader(); writer.writerows(trial_rows)
    summary_rows = []
    law = result["methods"]["Law"]
    for method, block in result["methods"].items():
        for reference_id, risk in block["risk_by_reference"].items():
            summary_rows.append({
                "method": method, "reference_id": reference_id, "risk": risk,
                "risk_increase_vs_law": risk / law["risk_by_reference"][reference_id] - 1.0,
                "risk_pass_2pct": risk <= 1.02 * law["risk_by_reference"][reference_id],
                "full_action": block["full_action_by_reference"][reference_id],
                "full_reduction_vs_law": 1.0 - block["full_action_by_reference"][reference_id] / law["full_action_by_reference"][reference_id],
            })
    with (paths["results"] / "reference_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader(); writer.writerows(summary_rows)

    lines = [
        "# Prospective vortices v6a/v6b multi-reference ablation",
        "",
        "Both beta arms, the shared Law/Tangent comparators, and all design-reference choices were frozen before held-out reference training and hidden physical validation. Evaluation uses three predeclared independently trained endpoint-only reference flows and aligned physical, sampling, and detector randomness.",
        "",
        "## Held-out reference results",
        "",
        "| Reference | Method | Full action | Reduction vs Law | Risk | Risk increase | 2% risk |",
        "|:--|:--|--:|--:|--:|--:|:--|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['reference_id']} | {row['method']} | {row['full_action']:.6g} | "
            f"{100*row['full_reduction_vs_law']:.3f}% | {row['risk']:.6g} | "
            f"{100*row['risk_increase_vs_law']:.3f}% | {'PASS' if row['risk_pass_2pct'] else 'FAIL'} |"
        )
    lines.extend(["", "## Equal-reference aggregate", "", "| Method | Full action | Reduction vs Law | Mean risk | Strict success |", "|:--|--:|--:|--:|:--|"])
    for method, block in result["methods"].items():
        lines.append(
            f"| {method} | {block['equal_reference_mean_full_action']:.6g} | "
            f"{100*(1-block['equal_reference_mean_full_action']/law['equal_reference_mean_full_action']):.3f}% | "
            f"{block['equal_reference_mean_risk']:.6g} | "
            f"{('PASS' if result['claims'].get(method, {}).get('strict_success') else 'FAIL') if method in ('v6a','v6b') else 'secondary'} |"
        )
    lines.extend(["", "## Paired ablation", ""])
    for comparison, block in result["comparisons"].items():
        pooled = block["pooled"]
        diff = pooled["difference_full_minus_law"]
        lines.append(
            f"- **{comparison}:** paired mean `{diff['mean']:.6g}`, paired t 95% CI "
            f"`[{pooled['paired_t_95_ci'][0]:.6g}, {pooled['paired_t_95_ci'][1]:.6g}]`; "
            f"two-level bootstrap CI `[{block['two_level_bootstrap']['95_ci'][0]:.6g}, "
            f"{block['two_level_bootstrap']['95_ci'][1]:.6g}]`."
        )
    lines.extend([
        "", "## Interpretation", "",
        result["interpretation"], "",
        "The two-level reference-cluster interval is low resolution because only three held-out reference seeds were preregistered; seed-specific results remain primary evidence.",
    ])
    (paths["results"] / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_v6(cfg: dict[str, Any], output_dir: str | Path):
    paths = v6_paths(output_dir)
    paths["results"].mkdir(parents=True, exist_ok=True)
    binding, freeze_hashes = _freeze_binding(paths)
    result_path = paths["results"] / "validation_result.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("freeze_binding") == binding:
            print("[v6-validation] reusing compatible completed validation", flush=True)
            return cached
        raise RuntimeError("existing v6 validation result belongs to another freeze")
    combined_path = paths["results"] / "combined_frozen_manifest.json"
    combined_bytes = combined_path.read_bytes()
    combined = json.loads(combined_bytes)
    evaluation_manifest = load_reference_manifest(output_dir, "evaluation")
    ids = [row["reference_id"] for row in evaluation_manifest["references"]]
    if ids != list(cfg["v6"]["evaluation_reference_ids"]):
        raise RuntimeError("evaluation reference IDs differ from frozen registry")
    data = TargetProspectiveData.load(
        paths["endpoint"] / "endpoint_data.npz", paths["prospective"] / "aggregate_predictions.npz"
    )
    _, evaluator_list = make_common_reference_evaluators(
        cfg,
        data,
        [row["rollout"] for row in evaluation_manifest["references"]],
    )
    evaluators = dict(zip(ids, evaluator_list))
    hidden_path, states = _ensure_hidden(cfg, paths, binding)
    random_path, indices, detector_z = _ensure_randomness(cfg, paths, binding, states.shape[1])
    started = time.perf_counter()
    selected = combined["selected"]
    method_names = ("Law", "Tangent", "v6a", "v6b")
    methods = {}
    action_arrays: dict[str, dict[str, np.ndarray]] = {}
    first_evaluator = evaluators[ids[0]]
    for method in method_names:
        eta = np.asarray(selected[method], dtype=np.float64)
        bank, mean, second, qoi = _realized_bank_and_moments(
            first_evaluator, eta, states, indices, detector_z
        )
        by_reference = {}
        action_arrays[method] = {}
        for reference_id in ids:
            result = evaluators[reference_id].evaluate_population(
                eta, mean, second, qoi, bank, compute_full=True
            )
            by_reference[reference_id] = result
            action_arrays[method][reference_id] = _trial_values(result, "full_action")
            print(f"[v6-validation] {method} on {reference_id}", flush=True)
        methods[method] = _method_summary(eta, by_reference)

    comparisons = {}
    comparison_pairs = {
        "v6a_minus_Law": ("Law", "v6a"),
        "v6b_minus_Law": ("Law", "v6b"),
        "Tangent_minus_Law": ("Law", "Tangent"),
        "v6b_minus_v6a": ("v6a", "v6b"),
    }
    for offset, (label, (base, treatment)) in enumerate(comparison_pairs.items()):
        by_ref = {}
        differences = {}
        for reference_id in ids:
            base_values = action_arrays[base][reference_id]
            treatment_values = action_arrays[treatment][reference_id]
            by_ref[reference_id] = paired_statistics(
                base_values, treatment_values,
                bootstrap_seed=int(cfg["seeds"]["validation_bootstrap"]) + offset * 10 + ids.index(reference_id),
            )
            differences[reference_id] = treatment_values - base_values
        pooled_base = np.concatenate([action_arrays[base][key] for key in ids])
        pooled_treatment = np.concatenate([action_arrays[treatment][key] for key in ids])
        comparisons[label] = {
            "base": base, "treatment": treatment, "by_reference": by_ref,
            "pooled": paired_statistics(
                pooled_base, pooled_treatment,
                bootstrap_seed=int(cfg["seeds"]["validation_bootstrap"]) + 100 + offset,
            ),
            "two_level_bootstrap": _two_level_bootstrap(
                differences, int(cfg["seeds"]["validation_bootstrap"]) + 200 + offset
            ),
        }

    claims = {}
    law = methods["Law"]
    for method in ("v6a", "v6b"):
        risk_pass = {
            reference_id: methods[method]["risk_by_reference"][reference_id]
            <= (1.0 + float(cfg["risk_allowance"])) * law["risk_by_reference"][reference_id]
            for reference_id in ids
        }
        comparison = comparisons[f"{method}_minus_Law"]["pooled"]
        numerical_pass = all(
            cert["invalid_trial_count"] == 0 and cert["nan_or_inf_count"] == 0
            and cert["all_full_solvers_converged"]
            for cert in methods[method]["certification_by_reference"].values()
        )
        claims[method] = {
            "risk_pass_by_reference": risk_pass,
            "all_reference_risk_pass": bool(all(risk_pass.values())),
            "pooled_paired_ci_below_zero": bool(comparison["paired_t_95_ci"][1] < 0.0),
            "numerical_certification_pass": bool(numerical_pass),
            "strict_success": bool(all(risk_pass.values()) and comparison["paired_t_95_ci"][1] < 0.0 and numerical_pass),
        }
    if claims["v6b"]["strict_success"] and claims["v6a"]["strict_success"]:
        interpretation = "Both nominal and beta=0.25 Full arms satisfy the strict preregistered held-out multi-reference success rule; the paired v6b-v6a comparison quantifies the robustness/action tradeoff."
    elif claims["v6b"]["strict_success"]:
        interpretation = "Only the beta=0.25 arm satisfies the strict held-out multi-reference success rule, supporting the preregistered robustness intervention for this finite reference ensemble."
    elif claims["v6a"]["strict_success"]:
        interpretation = "Only the nominal beta=0 arm satisfies the strict held-out multi-reference success rule; beta=0.25 did not improve the complete risk/action criterion."
    else:
        interpretation = "Neither beta arm satisfies the complete strict held-out multi-reference success rule; inspect seed-specific risk, action, and numerical evidence without tuning on this hidden bank."
    result = {
        "schema_version": 6, "experiment": cfg["name"], "freeze_binding": binding,
        "freeze_hashes": freeze_hashes,
        "fresh_hidden_validation": {
            "state_bank": str(hidden_path), "state_bank_sha256": file_sha256(hidden_path),
            "observation_randomness": str(random_path), "observation_randomness_sha256": file_sha256(random_path),
            "created_after_both_arm_freezes": True, "common_randomness_across_methods_and_references": True,
            "physical_seed": int(cfg["seeds"]["validation_physical"]),
            "sampling_seed": int(cfg["seeds"]["validation_sampling"]),
            "detector_seed": int(cfg["seeds"]["validation_detector"]),
        },
        "evaluation_reference_ids": ids, "methods": methods, "comparisons": comparisons,
        "claims": claims, "interpretation": interpretation,
        "validation_elapsed_seconds": time.perf_counter() - started,
    }
    if combined_path.read_bytes() != combined_bytes:
        raise RuntimeError("combined v6 frozen manifest changed during validation")
    _write_outputs(paths, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = validate_v6(load_config(args.config), args.output_dir)
    print(json.dumps(result["claims"], indent=2))


if __name__ == "__main__":
    main()
