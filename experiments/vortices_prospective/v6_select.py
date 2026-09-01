from __future__ import annotations

"""Select and freeze the paired v6a/v6b multi-reference ablation.

This module deliberately has no validation imports and no evaluation-reference
paths. It refuses to run once either held-out references or hidden data exist.
"""

import argparse
import copy
import json
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from common import config_hash, fingerprint, geometry_valid, load_config, software_metadata, write_json_atomic
from evaluator import ProspectiveEvaluator
from mfsi.cache import file_sha256
from prospective_data import TargetProspectiveData
from v4_objective import V4CRNBank, canonical_geometry_key, distribution, ensure_v4_crn_bank, geometry_penalty
from v4_select import _adam_multistart, _directional_check, generate_full_starts, generate_law_starts
from v6_objective import V6MultiReferenceObjective
from v6_reference_ensemble import DEFAULT_CONFIG, DEFAULT_OUTPUT, load_reference_manifest, v6_paths

jax.config.update("jax_enable_x64", True)


def _source_hash() -> str:
    here = Path(__file__).resolve().parent
    names = (
        "v6_select.py",
        "v6_objective.py",
        "v6_reference_ensemble.py",
        "v4_objective.py",
        "v4_select.py",
        "evaluator.py",
        "prospective_data.py",
        "reflected_raster.py",
    )
    return fingerprint({name: file_sha256(here / name) for name in names})


def _mean(result: dict[str, Any], key: str) -> float:
    value = result[key]["mean"]
    return float(value) if value is not None else float("inf")


def _trial_values(result: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(
        [row[key] for row in result["trials"] if row["valid"] and row[key] is not None],
        dtype=np.float64,
    )


def _risk_feasible(risks: list[float], limits: list[float], cfg: dict[str, Any]) -> bool:
    tolerance = float(cfg["validity"]["risk_constraint_tolerance"])
    return bool(np.all(np.asarray(risks) <= np.asarray(limits) + tolerance))


def _explicit_incumbent_run(eta: np.ndarray, stage: str, provenance: str) -> dict[str, Any]:
    """Represent a mandatory unmodified Pareto incumbent in candidate archives."""
    values = np.asarray(eta, dtype=np.float64).tolist()
    return {
        "run": 0,
        "stage": stage,
        "provenance": provenance,
        "initial_eta": values,
        "final_eta": values,
        "initial_penalized_objective": None,
        "final_penalized_objective": None,
        "final_gradient_norm": 0.0,
        "iterations": 0,
        "convergence_triggered": False,
        "numerical_invalid_steps": 0,
        "runtime_seconds": 0.0,
        "execution_chunk_seconds": 0.0,
        "execution_chunk_size": 0,
        "execution_batch_size": 0,
        "trace_objective": [],
        "trace_gradient_norm": [],
    }


_MANDATORY_FULL_CANDIDATE_IDS = (
    "full-pareto-incumbent",
    "full-law-incumbent",
)


def _retain_mandatory_candidates(
    selected: list[dict[str, Any]], eligible: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Retain exact-feasible safety anchors through a truncated funnel stage."""
    retained = list(selected)
    retained_ids = {str(row["candidate_id"]) for row in retained}
    by_id = {str(row["candidate_id"]): row for row in eligible}
    for candidate_id in _MANDATORY_FULL_CANDIDATE_IDS:
        if candidate_id in by_id and candidate_id not in retained_ids:
            retained.append(by_id[candidate_id])
            retained_ids.add(candidate_id)
    return retained


def _load_design(cfg: dict[str, Any], output_dir: str | Path):
    paths = v6_paths(output_dir)
    if paths["hidden"].exists() and any(paths["hidden"].iterdir()):
        raise RuntimeError("v6 selection refuses to run after hidden data exists")
    evaluation_manifest = paths["shared_results"] / "evaluation_reference_manifest.json"
    if evaluation_manifest.exists():
        raise RuntimeError("v6 selection cannot run after evaluation references exist")
    manifest = load_reference_manifest(output_dir, "design")
    expected_ids = list(cfg["v6"]["design_reference_ids"])
    ids = [row["reference_id"] for row in manifest["references"]]
    if ids != expected_ids:
        raise RuntimeError("design reference order differs from preregistration")
    for row in manifest["references"]:
        if file_sha256(row["rollout"]) != row["rollout_sha256"]:
            raise RuntimeError("design reference rollout hash changed")
    data = TargetProspectiveData.load(
        paths["endpoint"] / "endpoint_data.npz",
        paths["prospective"] / "aggregate_predictions.npz",
    )
    rollouts = [row["rollout"] for row in manifest["references"]]
    objective = V6MultiReferenceObjective(cfg, data, rollouts)
    evaluators = [
        ProspectiveEvaluator(
            cfg,
            data,
            path,
            raster_bandwidth=objective.common_raster_bandwidth,
        )
        for path in rollouts
    ]
    return paths, manifest, ids, data, objective, evaluators


def _multi_authoritative(
    evaluators: list[ProspectiveEvaluator],
    ids: list[str],
    eta: np.ndarray,
    bank: V4CRNBank,
    cfg: dict[str, Any],
    *,
    compute_full: bool,
    beta: float = 0.0,
    risk_limits: list[float] | None = None,
) -> dict[str, Any]:
    by_reference = {}
    risks, tangent_values, action_values = [], [], []
    for reference_id, evaluator in zip(ids, evaluators):
        result = evaluator.evaluate_prospective(
            eta, bank.as_observation_bank(), compute_full=compute_full
        )
        by_reference[reference_id] = result
        risks.append(_mean(result, "risk"))
        tangent_values.extend(_trial_values(result, "tangent_proxy").tolist())
        if compute_full:
            action_values.extend(_trial_values(result, "full_action").tolist())
    action_dist = distribution(action_values) if compute_full else distribution([])
    tangent_dist = distribution(tangent_values)
    valid = bool(all(row["valid"] for row in by_reference.values()))
    feasible = valid and geometry_valid(eta, cfg)
    if risk_limits is not None:
        feasible = feasible and _risk_feasible(risks, risk_limits, cfg)
    score = (
        float(action_dist["mean"] + float(beta) * action_dist["sd"])
        if compute_full and action_dist["mean"] is not None else None
    )
    return {
        "eta": np.asarray(eta, dtype=np.float64).tolist(),
        "centers": np.asarray(eta, dtype=np.float64).reshape((-1, 2)).tolist(),
        "reference_ids": ids,
        "risk_by_reference": dict(zip(ids, risks)),
        "mean_risk": float(np.mean(risks)),
        "tangent_distribution": tangent_dist,
        "full_distribution": action_dist,
        "robust_score": score,
        "beta": float(beta),
        "valid": valid,
        "geometry_valid": geometry_valid(eta, cfg),
        "risk_feasible_all_references": bool(feasible),
        "by_reference": by_reference,
    }


def _shared_signature(cfg, paths, design_manifest):
    crn = paths["prospective"] / "v6_selection_crn.npz"
    return {
        "config_hash": config_hash(cfg),
        "source_hash": _source_hash(),
        "endpoint_sha256": file_sha256(paths["endpoint"] / "endpoint_data.npz"),
        "aggregate_sha256": file_sha256(paths["prospective"] / "aggregate_predictions.npz"),
        "design_reference_manifest_sha256": file_sha256(paths["shared_results"] / "design_reference_manifest.json"),
        "design_reference_rollout_hashes": [row["rollout_sha256"] for row in design_manifest["references"]],
        "selection_crn_sha256": file_sha256(crn),
    }


def select_shared(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    paths, design_manifest, ids, _, objective, evaluators = _load_design(cfg, output_dir)
    paths["shared_results"].mkdir(parents=True, exist_ok=True)
    bank_path = paths["prospective"] / "v6_selection_crn.npz"
    bank = ensure_v4_crn_bank(bank_path, cfg, int(cfg["v4"]["selection_crn_trials"]))
    signature = _shared_signature(cfg, paths, design_manifest)
    manifest_path = paths["shared_results"] / "shared_selection_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("selection_input_hashes") == signature:
            print("[v6-shared] reusing frozen Law/Tangent", flush=True)
            return existing
        raise RuntimeError("existing v6 shared selection is incompatible")

    started = time.perf_counter()
    pareto = cfg.get("v6a_pareto", {})
    fixed_law_eta = pareto.get("fixed_law_eta")
    if fixed_law_eta is None:
        law_starts, law_sources = generate_law_starts(cfg)
    else:
        law_starts = np.asarray([fixed_law_eta], dtype=np.float64)
        law_sources = ["frozen cross-allowance Law anchor"]
    gradient_block = cfg["v4"]["gradient_checks"]
    check_bank = bank.prefix(int(gradient_block["multi_trial_count"]))
    rng = np.random.default_rng(int(cfg["seeds"]["gradient_directions"]))
    direction = rng.normal(size=law_starts[0].shape)
    direction /= np.linalg.norm(direction)
    risk_check = _directional_check(
        "v6_multi_reference_risk",
        lambda eta: objective.risk_mean(
            eta, jnp.asarray(check_bank.sampling_z), jnp.asarray(check_bank.detector_z)
        ),
        law_starts[0], direction,
        list(gradient_block["finite_difference_steps"]),
        float(gradient_block["relative_tolerance"]),
    )
    law_bank = bank.prefix(int(cfg["v4"]["law_optimizer"]["crn_trials"]))
    execution = cfg.get("v6_fast_execution", {})
    if fixed_law_eta is None:
        law_runs = _adam_multistart(
            law_starts, law_sources, law_bank, cfg["v4"]["law_optimizer"], cfg,
            lambda eta, s, d: objective.risk_mean(eta, s, d),
            schedule_seed=int(cfg["seeds"]["law_batch_schedule"]), stage="v6-law-adam",
            start_batch_size=int(execution.get("law_start_batch_size", 1)),
        )
    else:
        law_runs = [
            _explicit_incumbent_run(
                law_starts[0], "v6-law-fixed", "frozen cross-allowance Law anchor"
            )
        ]
    risk_bank = bank.prefix(int(cfg["v4"]["authoritative_crn_trials"]))
    law_candidates = []
    for index, run in enumerate(law_runs, start=1):
        eta = np.asarray(run["final_eta"], dtype=np.float64)
        audited = _multi_authoritative(evaluators, ids, eta, risk_bank, cfg, compute_full=False)
        law_candidates.append({"candidate_id": f"law-grad-{index:03d}", "gradient_run": run, **audited})
        print(f"[v6-law-audit] {index}/{len(law_runs)}", flush=True)
    valid_law = [row for row in law_candidates if row["valid"] and row["geometry_valid"]]
    if not valid_law:
        raise RuntimeError("v6 Law produced no valid candidate")
    law = min(valid_law, key=lambda row: row["mean_risk"])
    law_eta = np.asarray(law["eta"], dtype=np.float64)
    law_risks = [float(law["risk_by_reference"][reference_id]) for reference_id in ids]
    risk_limits = [(1.0 + float(cfg["risk_allowance"])) * value for value in law_risks]

    tangent_cfg = copy.deepcopy(cfg)
    tangent_cfg["v4"]["full_optimizer"].update({
        "starts": int(cfg["v4"]["tangent_optimizer"]["starts"]),
        "law_perturbation_starts": int(cfg["v4"]["tangent_optimizer"]["law_perturbation_starts"]),
        "law_perturbation_scale": float(cfg["v4"]["tangent_optimizer"]["law_perturbation_scale"]),
        "start_oversample": int(cfg["v4"]["tangent_optimizer"]["start_oversample"]),
    })
    tangent_cfg["seeds"]["full_global_starts"] = int(cfg["seeds"]["tangent_global_starts"])
    tangent_cfg["seeds"]["full_law_perturbations"] = int(cfg["seeds"]["tangent_law_perturbations"])
    tangent_starts, tangent_sources = generate_full_starts(tangent_cfg, law_eta)
    tangent_bank = bank.prefix(int(cfg["v4"]["tangent_optimizer"]["crn_trials"]))
    tangent_runs = _adam_multistart(
        tangent_starts,
        [source.replace("Full", "Tangent") for source in tangent_sources],
        tangent_bank, cfg["v4"]["tangent_optimizer"], cfg,
        lambda eta, s, d: objective.constrained_tangent_loss(eta, s, d, risk_limits),
        schedule_seed=int(cfg["seeds"]["tangent_batch_schedule"]), stage="v6-tangent-adam",
        start_batch_size=int(execution.get("tangent_start_batch_size", 1)),
    )
    tangent_candidates = []
    seen = set()
    for index, run in enumerate(tangent_runs, start=1):
        eta = np.asarray(run["final_eta"], dtype=np.float64)
        key = canonical_geometry_key(eta)
        if key in seen:
            continue
        seen.add(key)
        audited = _multi_authoritative(
            evaluators, ids, eta, risk_bank, cfg, compute_full=False, risk_limits=risk_limits
        )
        tangent_candidates.append({"candidate_id": f"tangent-grad-{index:03d}", "gradient_run": run, **audited})
        print(f"[v6-tangent-audit] {len(tangent_candidates)}/{len(tangent_runs)}", flush=True)
    tangent_incumbent = pareto.get("tangent_incumbent_eta")
    if tangent_incumbent is not None:
        eta = np.asarray(tangent_incumbent, dtype=np.float64)
        audited = _multi_authoritative(
            evaluators, ids, eta, risk_bank, cfg,
            compute_full=False, risk_limits=risk_limits,
        )
        tangent_candidates.append({
            "candidate_id": "tangent-pareto-incumbent",
            "gradient_run": _explicit_incumbent_run(
                eta, "v6-tangent-incumbent", "previous Pareto Tangent incumbent"
            ),
            **audited,
        })
    feasible_tangent = [row for row in tangent_candidates if row["risk_feasible_all_references"]]
    if not feasible_tangent:
        raise RuntimeError("v6 Tangent produced no all-reference-risk-feasible candidate")
    tangent = min(feasible_tangent, key=lambda row: row["tangent_distribution"]["mean"])
    law_full = _multi_authoritative(evaluators, ids, law_eta, risk_bank, cfg, compute_full=True, beta=0.0)
    tangent_full = _multi_authoritative(
        evaluators, ids, np.asarray(tangent["eta"]), risk_bank, cfg,
        compute_full=True, beta=0.0, risk_limits=risk_limits,
    )
    archive_path = paths["shared_results"] / "shared_candidate_archive.json"
    write_json_atomic(archive_path, {
        "law_gradient_runs": law_runs,
        "law_candidates": law_candidates,
        "tangent_gradient_runs": tangent_runs,
        "tangent_candidates": tangent_candidates,
    })
    manifest = {
        "schema_version": 6,
        "status": "frozen_before_hidden_validation",
        "role": "v6_shared_law_tangent_selection",
        "selection_input_hashes": signature,
        "reference_ids": ids,
        "gradient_checks": {"risk": risk_check},
        "risk_allowance": float(cfg["risk_allowance"]),
        "law_risk_by_reference": dict(zip(ids, law_risks)),
        "risk_limit_by_reference": dict(zip(ids, risk_limits)),
        "selected": {"Law": law_full, "Tangent": tangent_full},
        "pareto_execution": {
            "fixed_law_anchor": fixed_law_eta is not None,
            "tangent_incumbent_supplied": tangent_incumbent is not None,
        },
        "candidate_archive_sha256": file_sha256(archive_path),
        "selection_elapsed_seconds": time.perf_counter() - started,
        "software": software_metadata(),
        "hidden_data_loaded": False,
        "evaluation_references_loaded": False,
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def _load_arm_overlay(path: str | Path) -> dict[str, Any]:
    overlay = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(overlay) != {"schema_version", "name", "base_config", "beta"}:
        raise ValueError("v6 arm overlay contains unapproved fields")
    if int(overlay["schema_version"]) != 6:
        raise ValueError("v6 arm schema mismatch")
    return overlay


def _compile_joint_rescore(objective, bank, fidelity):
    sampling = jnp.asarray(bank.sampling_z)
    detector = jnp.asarray(bank.detector_z)
    return jax.jit(lambda e: objective.full_trials_by_reference(
        e, sampling, detector, fidelity
    ))


def _joint_rescore(objective, eta, bank, fidelity, beta, *, compiled=None):
    fn = compiled if compiled is not None else _compile_joint_rescore(
        objective, bank, fidelity
    )
    action, risks, residual, ess, poisson = jax.device_get(fn(jnp.asarray(eta)))
    flat = np.asarray(action).reshape(-1)
    dist = distribution(flat)
    return {
        "action": dist,
        "action_by_reference": [distribution(row) for row in np.asarray(action)],
        "risk_by_reference": [distribution(row) for row in np.asarray(risks)],
        "robust_score": float(dist["mean"] + float(beta) * dist["sd"]),
        "max_projection_residual": float(np.max(residual)),
        "min_ess_fraction": float(np.min(ess)),
        "max_differentiable_poisson_residual": float(np.max(poisson)),
    }


def _compile_v6_loss(objective, bank, fidelity, risk_limits, beta, cfg):
    sampling = jnp.asarray(bank.sampling_z)
    detector = jnp.asarray(bank.detector_z)
    return jax.jit(jax.value_and_grad(lambda e:
        objective.constrained_full_loss(
            e, sampling, detector, fidelity, risk_limits, beta
        ) + geometry_penalty(e, cfg)
    ))


def _v6_lbfgs(
    eta, objective, bank, fidelity, risk_limits, beta, cfg, *, compiled_loss=None
):
    block = cfg["v4"]["full_lbfgs"]
    if not bool(block["enabled"]):
        return np.asarray(eta), {"enabled": False, "iterations": 0}
    loss = compiled_loss if compiled_loss is not None else _compile_v6_loss(
        objective, bank, fidelity, risk_limits, beta, cfg
    )
    def fun(x):
        value, grad = jax.device_get(loss(jnp.asarray(x, dtype=jnp.float64)))
        return float(value), np.asarray(grad, dtype=np.float64)
    margin = float(cfg["measurement"]["boundary_margin"])
    bounds = []
    for _ in range(int(cfg["measurement"]["n_sensors"])):
        bounds.extend([(margin, 2.0 - margin), (margin, 1.0 - margin)])
    result = minimize(
        fun, np.asarray(eta, dtype=np.float64), jac=True, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": int(block["max_iterations"]), "ftol": float(block["ftol"]),
                 "gtol": float(block["gtol"]), "maxls": int(block["max_line_search_steps"])},
    )
    return np.asarray(result.x), {
        "enabled": True, "iterations": int(result.nit), "success": bool(result.success),
        "message": str(result.message), "final_penalized_objective": float(result.fun),
        "gradient_norm": float(np.linalg.norm(result.jac)),
    }


def _shared_full_starts(cfg, paths, law_eta):
    path = paths["shared_results"] / "full_starts.npz"
    starts, provenance = generate_full_starts(cfg, np.asarray(law_eta, dtype=np.float64))
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if not np.array_equal(np.asarray(data["starts"]), starts):
                raise RuntimeError("frozen v6 Full starts differ from config")
            saved_provenance = [str(x) for x in data["provenance"]]
            if saved_provenance != provenance:
                raise RuntimeError("frozen v6 Full provenance differs")
    else:
        np.savez_compressed(path, starts=starts, provenance=np.asarray(provenance))
    return starts, provenance, file_sha256(path)


def _prescreen_full_starts(
    starts,
    provenance,
    objective,
    bank,
    fidelity,
    risk_limits,
    beta,
    cfg,
    execution,
):
    keep = int(execution.get("prescreen_optimize_starts", len(starts)))
    if keep >= len(starts):
        return starts, provenance, {
            "enabled": False,
            "input_starts": len(starts),
            "optimized_starts": len(starts),
        }
    if keep < 1:
        raise ValueError("prescreen_optimize_starts must be >= 1")
    mandatory = min(
        len(starts),
        1 + int(cfg["v4"]["full_optimizer"]["law_perturbation_starts"]),
    )
    if keep < mandatory:
        raise ValueError(
            "prescreen_optimize_starts cannot discard the frozen Law-neighborhood starts"
        )
    batch_size = int(
        execution.get(
            "prescreen_start_batch_size",
            execution.get("full_start_batch_size", 1),
        )
    )
    if batch_size < 1:
        raise ValueError("prescreen_start_batch_size must be >= 1")
    sampling = jnp.asarray(bank.sampling_z)
    detector = jnp.asarray(bank.detector_z)
    score_one = lambda eta: (
        objective.constrained_full_loss(
            eta, sampling, detector, fidelity, risk_limits, beta
        ) + geometry_penalty(eta, cfg)
    )
    effective_batch = min(batch_size, len(starts))
    score_batch = jax.jit(jax.vmap(score_one))
    scores = []
    started = time.perf_counter()
    for begin in range(0, len(starts), effective_batch):
        chunk = jnp.asarray(
            starts[begin : begin + effective_batch], dtype=jnp.float64
        )
        take = len(chunk)
        if take < effective_batch:
            chunk = jnp.concatenate(
                (chunk, jnp.repeat(chunk[-1:], effective_batch - take, axis=0)),
                axis=0,
            )
        scores.extend(np.asarray(score_batch(chunk))[:take].tolist())
    scores = np.asarray(scores, dtype=np.float64)
    selected_indices = _select_prescreen_indices(scores, mandatory, keep)
    receipt = {
        "enabled": True,
        "definition": "exact differentiable initial constrained Full loss",
        "input_starts": len(starts),
        "optimized_starts": len(selected_indices),
        "mandatory_law_neighborhood_starts": mandatory,
        "evaluation_batch_size": effective_batch,
        "elapsed_seconds": time.perf_counter() - started,
        "selected_zero_based_indices": selected_indices,
        "selected_one_based_indices": [index + 1 for index in selected_indices],
        "scores": scores.tolist(),
    }
    return (
        np.asarray(starts)[selected_indices],
        [provenance[index] for index in selected_indices],
        receipt,
    )


def _select_prescreen_indices(
    scores: np.ndarray, mandatory_count: int, keep: int
) -> list[int]:
    scores = np.asarray(scores, dtype=np.float64)
    mandatory_count = int(mandatory_count)
    keep = int(keep)
    if not 0 <= mandatory_count <= keep <= len(scores):
        raise ValueError("invalid prescreen counts")
    mandatory = set(range(mandatory_count))
    optional = [
        int(index) for index in np.argsort(scores, kind="stable")
        if int(index) not in mandatory
    ]
    return sorted([*mandatory, *optional[: keep - mandatory_count]])


def _checkpointed_adam_multistart(
    checkpoint_path: Path,
    checkpoint_identity: dict[str, Any],
    starts: np.ndarray,
    provenance: list[str],
    bank: V4CRNBank,
    settings: dict[str, Any],
    cfg: dict[str, Any],
    loss,
    *,
    schedule_seed: int,
    stage: str,
    start_batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    identity_hash = fingerprint(checkpoint_identity)
    completed_rows: list[dict[str, Any]] = []
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("identity_hash") != identity_hash:
            raise RuntimeError(f"incompatible optimizer checkpoint: {checkpoint_path}")
        completed_rows = list(checkpoint.get("gradient_runs", []))

    def save(rows):
        write_json_atomic(checkpoint_path, {
            "schema_version": 1,
            "status": "complete" if len(rows) == len(starts) else "partial",
            "identity_hash": identity_hash,
            "completed_starts": len(rows),
            "total_starts": len(starts),
            "gradient_runs": rows,
        })

    rows = _adam_multistart(
        starts, provenance, bank, settings, cfg, loss,
        schedule_seed=schedule_seed,
        stage=stage,
        start_batch_size=start_batch_size,
        completed_rows=completed_rows,
        chunk_callback=save,
    )
    if not checkpoint_path.exists():
        save(rows)
    receipt = {
        "path": str(checkpoint_path.resolve()),
        "sha256": file_sha256(checkpoint_path),
        "identity_hash": identity_hash,
        "completed_starts": len(rows),
        "total_starts": len(starts),
    }
    return rows, receipt


def select_arm(cfg, output_dir, overlay_path):
    paths, design_manifest, ids, _, objective, evaluators = _load_design(cfg, output_dir)
    shared_path = paths["shared_results"] / "shared_selection_manifest.json"
    if not shared_path.exists():
        raise RuntimeError("select shared Law/Tangent before v6 arms")
    shared = json.loads(shared_path.read_text(encoding="utf-8"))
    overlay_path = Path(overlay_path).resolve()
    overlay = _load_arm_overlay(overlay_path)
    beta = float(overlay["beta"])
    expected_beta = float(cfg["v6"]["betas"]["v6a" if beta == 0.0 else "v6b"])
    if beta != expected_beta:
        raise RuntimeError("arm beta differs from common preregistration")
    arm_dir = paths["arms"] / str(overlay["name"])
    results_dir = arm_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = results_dir / "frozen_manifest.json"
    bank_path = paths["prospective"] / "v6_selection_crn.npz"
    bank = V4CRNBank(*[
        np.asarray(x, dtype=np.float64) for x in (
            np.load(bank_path, allow_pickle=False)["sampling_z"],
            np.load(bank_path, allow_pickle=False)["detector_z"],
        )
    ])
    law_eta = np.asarray(shared["selected"]["Law"]["eta"], dtype=np.float64)
    limits = [float(shared["risk_limit_by_reference"][reference_id]) for reference_id in ids]
    starts, provenance, starts_sha = _shared_full_starts(cfg, paths, law_eta)
    signature = {
        "common_config_hash": config_hash(cfg), "arm_overlay_sha256": file_sha256(overlay_path),
        "source_hash": _source_hash(), "shared_manifest_sha256": file_sha256(shared_path),
        "design_reference_manifest_sha256": file_sha256(paths["shared_results"] / "design_reference_manifest.json"),
        "selection_crn_sha256": file_sha256(bank_path), "full_starts_sha256": starts_sha,
        "beta": beta,
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("selection_input_hashes") == signature:
            print(f"[{overlay['name']}] reusing frozen arm", flush=True)
            return existing
        raise RuntimeError(f"existing {overlay['name']} manifest is incompatible")

    started = time.perf_counter()
    checks = cfg["v4"]["gradient_checks"]
    check_bank = bank.prefix(int(checks["multi_trial_count"]))
    rng = np.random.default_rng(int(cfg["seeds"]["gradient_directions"]) + (0 if beta == 0 else 1))
    direction = rng.normal(size=starts[0].shape); direction /= np.linalg.norm(direction)
    gradient_check = _directional_check(
        f"{overlay['name']}_joint_reference_full",
        lambda eta: objective.full_score(
            eta, jnp.asarray(check_bank.sampling_z), jnp.asarray(check_bank.detector_z),
            str(checks["fidelity"]), beta,
        ), starts[0], direction, list(checks["finite_difference_steps"]),
        float(checks["relative_tolerance"]),
    )
    settings = cfg["v4"]["full_optimizer"]
    execution = cfg.get("v6_fast_execution", {})
    search_fidelity = str(settings["fidelity"])
    search_bank = bank.prefix(objective.fidelity(search_fidelity).trials)
    starts, provenance, prescreen = _prescreen_full_starts(
        starts, provenance, objective, search_bank, search_fidelity,
        limits, beta, cfg, execution,
    )
    runs, broad_checkpoint = _checkpointed_adam_multistart(
        results_dir / "full_adam_checkpoint.json",
        {
            "selection_input_hashes": signature,
            "stage": "broad_full_adam",
            "beta": beta,
            "fidelity": search_fidelity,
            "risk_limits": limits,
            "settings": settings,
            "starts": np.asarray(starts).tolist(),
            "provenance": provenance,
            "start_prescreen_selected": prescreen.get("selected_zero_based_indices"),
        },
        starts, provenance, search_bank, settings, cfg,
        lambda eta, s, d: objective.constrained_full_loss(
            eta, s, d, search_fidelity, limits, beta
        ), schedule_seed=int(cfg["seeds"]["full_batch_schedule"]),
        stage=f"{overlay['name']}-full-adam",
        start_batch_size=int(execution.get("full_start_batch_size", 1)),
    )
    search_score_fn = jax.jit(lambda eta: objective.full_score(
        eta,
        jnp.asarray(search_bank.sampling_z),
        jnp.asarray(search_bank.detector_z),
        search_fidelity,
        beta,
    ))
    risk_bank = bank.prefix(int(cfg["v4"]["authoritative_crn_trials"]))
    candidates, seen = [], set()
    for index, run in enumerate(runs, start=1):
        eta = np.asarray(run["final_eta"], dtype=np.float64)
        key = canonical_geometry_key(eta)
        if key in seen:
            continue
        seen.add(key)
        risk_audit = _multi_authoritative(
            evaluators, ids, eta, risk_bank, cfg, compute_full=False, risk_limits=limits
        )
        search_score = float(jax.device_get(search_score_fn(jnp.asarray(eta))))
        candidates.append({
            "candidate_id": f"full-grad-{index:03d}", "source": run["provenance"],
            "gradient_run": run, "search_full_score": search_score, **risk_audit,
        })
        print(f"[{overlay['name']}-risk-audit] {len(candidates)}/{len(runs)}", flush=True)

    # Full is a superset of Law, so the frozen Law solution is always a valid
    # Full candidate. Keep it unmodified as the first-point feasibility anchor:
    # constrained gradient starts can all cross a tight exact-risk boundary.
    risk_audit = _multi_authoritative(
        evaluators, ids, law_eta, risk_bank, cfg,
        compute_full=False, risk_limits=limits,
    )
    search_score = float(jax.device_get(search_score_fn(jnp.asarray(law_eta))))
    candidates.append({
        "candidate_id": "full-law-incumbent",
        "source": "frozen Law feasibility anchor",
        "gradient_run": _explicit_incumbent_run(
            law_eta, f"{overlay['name']}-full-law-incumbent",
            "frozen Law feasibility anchor",
        ),
        "search_full_score": search_score,
        **risk_audit,
    })
    full_incumbent = cfg.get("v6a_pareto", {}).get("full_incumbent_eta")
    if full_incumbent is not None:
        eta = np.asarray(full_incumbent, dtype=np.float64)
        risk_audit = _multi_authoritative(
            evaluators, ids, eta, risk_bank, cfg,
            compute_full=False, risk_limits=limits,
        )
        search_score = float(jax.device_get(search_score_fn(jnp.asarray(eta))))
        candidates.append({
            "candidate_id": "full-pareto-incumbent",
            "source": "previous Pareto Full incumbent",
            "gradient_run": _explicit_incumbent_run(
                eta, f"{overlay['name']}-full-incumbent",
                "previous Pareto Full incumbent",
            ),
            "search_full_score": search_score,
            **risk_audit,
        })
    feasible = [row for row in candidates if row["risk_feasible_all_references"]]
    if not feasible:
        raise RuntimeError(f"{overlay['name']} produced no exact feasible candidate")
    feasible.sort(key=lambda row: row["search_full_score"])
    rescore_name = str(cfg["v4"]["funnel"]["rescore_fidelity"])
    rescore_bank = bank.prefix(objective.fidelity(rescore_name).trials)
    rescore_fn = _compile_joint_rescore(objective, rescore_bank, rescore_name)
    rescored = feasible[: int(cfg["v4"]["funnel"]["rescore_candidates"])]
    rescored = _retain_mandatory_candidates(rescored, feasible)
    for index, row in enumerate(rescored, start=1):
        row["prospective_rescore"] = _joint_rescore(
            objective, np.asarray(row["eta"]), rescore_bank, rescore_name, beta,
            compiled=rescore_fn,
        )
        print(f"[{overlay['name']}-rescore] {index}/{len(rescored)}", flush=True)
    rescored.sort(key=lambda row: row["prospective_rescore"]["robust_score"])
    polish_name = str(cfg["v4"]["funnel"]["polish_fidelity"])
    polish_bank = bank.prefix(objective.fidelity(polish_name).trials)
    polish_seeds = rescored[: int(cfg["v4"]["funnel"]["polish_candidates"])]
    polish_settings = {
        **settings, "steps": int(cfg["v4"]["funnel"]["polish_adam_steps"]),
        "batch_size": int(cfg["v4"]["funnel"]["polish_batch_size"]),
    }
    polish_starts = np.asarray([row["eta"] for row in polish_seeds])
    polish_provenance = ["multi-reference Full gradient polish"] * len(polish_seeds)
    polish_runs, polish_checkpoint = _checkpointed_adam_multistart(
        results_dir / "full_polish_adam_checkpoint.json",
        {
            "selection_input_hashes": signature,
            "stage": "full_polish_adam",
            "beta": beta,
            "fidelity": polish_name,
            "risk_limits": limits,
            "settings": polish_settings,
            "parent_candidate_ids": [row["candidate_id"] for row in polish_seeds],
            "starts": polish_starts.tolist(),
        },
        polish_starts,
        polish_provenance,
        polish_bank, polish_settings, cfg,
        lambda eta, s, d: objective.constrained_full_loss(
            eta, s, d, polish_name, limits, beta
        ), schedule_seed=int(cfg["seeds"]["full_polish_batch_schedule"]),
        stage=f"{overlay['name']}-full-polish",
        start_batch_size=int(execution.get("polish_start_batch_size", 1)),
    )
    polish_loss = _compile_v6_loss(
        objective, polish_bank, polish_name, limits, beta, cfg
    )
    polish_rescore_fn = _compile_joint_rescore(
        objective, polish_bank, polish_name
    )
    polished = []
    for source, run in zip(polish_seeds, polish_runs):
        eta, lbfgs = _v6_lbfgs(
            np.asarray(run["final_eta"]), objective, polish_bank, polish_name,
            limits, beta, cfg, compiled_loss=polish_loss,
        )
        risk_audit = _multi_authoritative(
            evaluators, ids, eta, risk_bank, cfg, compute_full=False, risk_limits=limits
        )
        polished.append({
            "candidate_id": source["candidate_id"] + "-polished",
            "source": "multi-reference Full Adam + L-BFGS polish",
            "parent_candidate_id": source["candidate_id"], "gradient_run": run,
            "lbfgs": lbfgs,
            "prospective_rescore": _joint_rescore(
                objective, eta, polish_bank, polish_name, beta,
                compiled=polish_rescore_fn,
            ),
            **risk_audit,
        })
    polished = [row for row in polished if row["risk_feasible_all_references"]]
    # Polishing is an opportunity to improve a certified broad candidate, not
    # permission to discard the already feasible pool. This fallback is shared
    # identically by both beta arms and remains subject to authoritative final
    # certification below.
    polish_feasible_fallback_used = not polished
    pool = polished + rescored
    pool.sort(key=lambda row: row["prospective_rescore"]["robust_score"])
    finalists, finalist_seen = [], set()
    mandatory_first = _retain_mandatory_candidates([], pool)
    for row in mandatory_first + pool:
        key = canonical_geometry_key(row["eta"])
        if key in finalist_seen:
            continue
        finalist_seen.add(key)
        finalists.append(row)
        if len(finalists) >= int(cfg["v4"]["funnel"]["authoritative_full_finalists"]):
            break
    authoritative = []
    for index, row in enumerate(finalists, start=1):
        audited = _multi_authoritative(
            evaluators, ids, np.asarray(row["eta"]), risk_bank, cfg,
            compute_full=True, beta=beta, risk_limits=limits,
        )
        authoritative.append({
            "candidate_id": row["candidate_id"], "source": row["source"],
            "gradient_norm": float(row["gradient_run"]["final_gradient_norm"]), **audited,
        })
        print(f"[{overlay['name']}-authoritative] {index}/{len(finalists)}", flush=True)
    certified = [row for row in authoritative if row["risk_feasible_all_references"] and row["valid"]]
    if not certified:
        raise RuntimeError(f"{overlay['name']} has no certified authoritative finalist")
    selected = min(certified, key=lambda row: row["robust_score"])
    archive_path = results_dir / "candidate_archive.json"
    write_json_atomic(archive_path, {
        "gradient_runs": runs, "candidates": candidates, "rescored": rescored,
        "polished": polished, "authoritative_finalists": authoritative,
        "start_prescreen": prescreen,
        "optimizer_checkpoints": {
            "broad_full_adam": broad_checkpoint,
            "full_polish_adam": polish_checkpoint,
        },
    })
    manifest = {
        "schema_version": 6, "status": "frozen_before_hidden_validation",
        "experiment": str(overlay["name"]), "beta": beta,
        "selection_input_hashes": signature, "reference_ids": ids,
        "risk_limit_by_reference": shared["risk_limit_by_reference"],
        "gradient_check": gradient_check, "selected": selected,
        "authoritative_finalists": authoritative,
        "selection_counts": {
            "gradient_starts": len(runs), "distinct_candidates": len(candidates),
            "risk_feasible_candidates": len(feasible), "polished_feasible": len(polished),
            "authoritative_finalists": len(authoritative),
        },
        "polish_feasible_fallback_used": polish_feasible_fallback_used,
        "start_prescreen": prescreen,
        "optimizer_checkpoints": {
            "broad_full_adam": broad_checkpoint,
            "full_polish_adam": polish_checkpoint,
        },
        "candidate_archive_sha256": file_sha256(archive_path),
        "selection_elapsed_seconds": time.perf_counter() - started,
        "hidden_data_loaded": False, "evaluation_references_loaded": False,
        "software": software_metadata(),
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def combine_freeze(cfg, output_dir):
    paths = v6_paths(output_dir)
    paths["results"].mkdir(parents=True, exist_ok=True)
    combined_path = paths["results"] / "combined_frozen_manifest.json"
    shared_path = paths["shared_results"] / "shared_selection_manifest.json"
    arm_paths = {
        "v6a": paths["arms"] / "v6a_beta_0" / "results" / "frozen_manifest.json",
        "v6b": paths["arms"] / "v6b_beta_0p25" / "results" / "frozen_manifest.json",
    }
    required = [shared_path, *arm_paths.values()]
    if not all(path.exists() for path in required):
        raise RuntimeError("combined freeze requires shared selection and both arms")
    manifests = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in arm_paths.items()}
    if manifests["v6a"]["selection_input_hashes"]["full_starts_sha256"] != manifests["v6b"]["selection_input_hashes"]["full_starts_sha256"]:
        raise RuntimeError("v6 arms did not use identical starts")
    if manifests["v6a"]["selection_input_hashes"]["selection_crn_sha256"] != manifests["v6b"]["selection_input_hashes"]["selection_crn_sha256"]:
        raise RuntimeError("v6 arms did not use identical CRNs")
    rows = []
    for arm, manifest in manifests.items():
        for row in manifest["authoritative_finalists"]:
            rows.append({
                "arm": arm, "candidate_id": row["candidate_id"], "eta": row["eta"],
                "mean_risk": row["mean_risk"], "robust_score": row["robust_score"],
                "mean_full": row["full_distribution"]["mean"],
                "risk_by_reference": row["risk_by_reference"],
            })
    shared = json.loads(shared_path.read_text(encoding="utf-8"))
    law_risks = shared["law_risk_by_reference"]
    frontier = {}
    for allowance in cfg["v6"]["empirical_frontier_allowances"]:
        feasible = [row for row in rows if all(
            row["risk_by_reference"][reference_id] <= (1.0 + float(allowance)) * law_risks[reference_id]
            + float(cfg["validity"]["risk_constraint_tolerance"])
            for reference_id in shared["reference_ids"]
        )]
        frontier[str(allowance)] = min(feasible, key=lambda row: row["mean_full"]) if feasible else None
    frontier_path = paths["results"] / "empirical_candidate_frontier.json"
    write_json_atomic(frontier_path, {
        "interpretation": "descriptive union of 2%-targeted frozen finalist pools; not an independently optimized Pareto sweep",
        "allowances": frontier,
    })
    combined = {
        "schema_version": 6, "status": "both_arms_frozen_before_evaluation_references_and_hidden_validation",
        "experiment": cfg["name"], "config_hash": config_hash(cfg),
        "source_hash": _source_hash(), "shared_manifest_sha256": file_sha256(shared_path),
        "arm_manifest_sha256": {key: file_sha256(path) for key, path in arm_paths.items()},
        "selected": {"Law": shared["selected"]["Law"]["eta"], "Tangent": shared["selected"]["Tangent"]["eta"],
                     "v6a": manifests["v6a"]["selected"]["eta"], "v6b": manifests["v6b"]["selected"]["eta"]},
        "betas": {"v6a": manifests["v6a"]["beta"], "v6b": manifests["v6b"]["beta"]},
        "design_reference_ids": shared["reference_ids"],
        "evaluation_reference_registry_frozen_but_not_generated": {
            "ids": cfg["v6"]["evaluation_reference_ids"],
            "training_seeds": cfg["v6"]["evaluation_reference_training_seeds"],
            "rollout_seed": cfg["v6"]["evaluation_reference_rollout_seed"],
        },
        "hidden_seed_registry": {key: cfg["seeds"][key] for key in
            ("validation_physical", "validation_sampling", "validation_detector", "validation_bootstrap")},
        "empirical_frontier_sha256": file_sha256(frontier_path),
        "hidden_data_loaded": False, "evaluation_references_loaded": False,
    }
    if combined_path.exists():
        existing = json.loads(combined_path.read_text(encoding="utf-8"))
        if existing != combined:
            raise RuntimeError("existing combined v6 freeze is incompatible")
    else:
        write_json_atomic(combined_path, combined)
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("shared", "v6a", "v6b", "combine", "all"), required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.stage in {"shared", "all"}:
        result = select_shared(cfg, args.output_dir)
    if args.stage in {"v6a", "all"}:
        result = select_arm(cfg, args.output_dir, Path(__file__).resolve().parent / "configs" / "production_v6a.json")
    if args.stage in {"v6b", "all"}:
        result = select_arm(cfg, args.output_dir, Path(__file__).resolve().parent / "configs" / "production_v6b.json")
    if args.stage in {"combine", "all"}:
        result = combine_freeze(cfg, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
