from __future__ import annotations

"""Gradient-based selection and pre-validation freeze for prospective v4."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from common import (
    REPO_ROOT,
    artifact_dirs,
    config_hash,
    fingerprint,
    geometry_valid,
    load_config,
    software_metadata,
    write_json_atomic,
)
from evaluator import ProspectiveEvaluator
from mfsi.cache import file_sha256
from mfsi.design import random_point_sensor_starts
from prospective_data import TargetProspectiveData
from v4_objective import (
    V4CRNBank,
    V4DifferentiableObjective,
    acquisition_reparameterization_formula,
    canonical_geometry_key,
    distribution,
    ensure_v4_crn_bank,
    geometry_penalty,
    project_box,
)
from v4_protocol import run_time_derivative_audit, v4_source_hash

jax.config.update("jax_enable_x64", True)


def _mean(result: dict[str, Any], key: str) -> float:
    value = result[key]["mean"]
    return float(value) if value is not None else float("inf")


def _trial_values(result: dict[str, Any], key: str) -> np.ndarray:
    values = [row[key] for row in result["trials"] if row["valid"] and row[key] is not None]
    return np.asarray(values, dtype=np.float64)


def _risk_feasible(risk: float, law_risk: float, cfg: dict[str, Any]) -> bool:
    ceiling = (1.0 + float(cfg["risk_allowance"])) * float(law_risk)
    return bool(risk <= ceiling + float(cfg["validity"]["risk_constraint_tolerance"]))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def generate_law_starts(cfg: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    block = cfg["v4"]["law_optimizer"]
    count = int(block["starts"])
    m = cfg["measurement"]
    margin = float(m["boundary_margin"])
    starts = random_point_sensor_starts(
        jax.random.PRNGKey(int(cfg["seeds"]["law_initial_starts"])),
        count,
        n_sensors=int(m["n_sensors"]),
        x_bounds=(margin, 2.0 - margin),
        y_bounds=(margin, 1.0 - margin),
        min_sep=float(m["min_separation"]),
        oversample=int(block["start_oversample"]),
    )
    return np.asarray(starts, dtype=np.float64), ["Law global gradient start"] * count


def generate_full_starts(
    cfg: dict[str, Any], law_eta: np.ndarray, tangent_eta: np.ndarray | None = None
) -> tuple[np.ndarray, list[str]]:
    """Generate Full starts independently; Tangent is optional and never required."""
    del tangent_eta  # v4 preregisters Tangent-disabled independence.
    block = cfg["v4"]["full_optimizer"]
    total = int(block["starts"])
    perturb_count = min(int(block["law_perturbation_starts"]), total - 1)
    global_count = total - 1 - perturb_count
    m = cfg["measurement"]
    margin = float(m["boundary_margin"])
    rows = [np.asarray(law_eta, dtype=np.float64)]
    provenance = ["Full Law-incumbent gradient start"]
    rng = np.random.default_rng(int(cfg["seeds"]["full_law_perturbations"]));
    attempts = 0
    while len(rows) < 1 + perturb_count and attempts < max(1000, perturb_count * 1000):
        attempts += 1
        scale = float(block["law_perturbation_scale"])
        proposal = np.asarray(law_eta) + rng.normal(size=np.asarray(law_eta).shape) * scale
        centers = proposal.reshape((-1, 2))
        centers[:, 0] = np.clip(centers[:, 0], margin, 2.0 - margin)
        centers[:, 1] = np.clip(centers[:, 1], margin, 1.0 - margin)
        proposal = centers.reshape(-1)
        if geometry_valid(proposal, cfg):
            rows.append(proposal)
            provenance.append("Full Law-perturbation gradient start")
    if len(rows) != 1 + perturb_count:
        raise RuntimeError("could not construct preregistered separated Law perturbations")
    if global_count:
        global_starts = random_point_sensor_starts(
            jax.random.PRNGKey(int(cfg["seeds"]["full_global_starts"])),
            global_count,
            n_sensors=int(m["n_sensors"]),
            x_bounds=(margin, 2.0 - margin),
            y_bounds=(margin, 1.0 - margin),
            min_sep=float(m["min_separation"]),
            oversample=int(block["start_oversample"]),
        )
        rows.extend(np.asarray(global_starts, dtype=np.float64))
        provenance.extend(["Full global gradient start"] * global_count)
    return np.asarray(rows, dtype=np.float64), provenance


def _batch_schedule(seed: int, steps: int, trials: int, batch_size: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    schedules = []
    pool: list[int] = []
    for _ in range(int(steps)):
        if len(pool) < int(batch_size):
            pool.extend(rng.permutation(int(trials)).tolist())
        schedules.append([pool.pop(0) for _ in range(int(batch_size))])
    return np.asarray(schedules, dtype=np.int32)


def _adam_multistart(
    starts: np.ndarray,
    provenance: list[str],
    bank: V4CRNBank,
    settings: dict[str, Any],
    cfg: dict[str, Any],
    loss: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    *,
    schedule_seed: int,
    stage: str,
) -> list[dict[str, Any]]:
    steps = int(settings["steps"])
    schedule = jnp.asarray(
        _batch_schedule(
            schedule_seed, steps, bank.trials, int(settings["batch_size"])
        )
    )
    sampling = jnp.asarray(bank.sampling_z, dtype=jnp.float64)
    detector = jnp.asarray(bank.detector_z, dtype=jnp.float64)
    value_and_grad = jax.value_and_grad(
        lambda eta, s, d: loss(eta, s, d) + geometry_penalty(eta, cfg)
    )
    lr = float(settings["learning_rate"])
    beta1 = float(settings["beta1"])
    beta2 = float(settings["beta2"])
    eps = float(settings["eps"])

    def optimize_one(eta0):
        def step(carry, idx):
            eta, m, v, iteration = carry
            s = sampling[idx]
            d = detector[idx]
            value, gradient = value_and_grad(eta, s, d)
            finite = jnp.isfinite(value) & jnp.all(jnp.isfinite(gradient))
            gradient = jnp.where(finite, gradient, jnp.zeros_like(gradient))
            next_m = beta1 * m + (1.0 - beta1) * gradient
            next_v = beta2 * v + (1.0 - beta2) * gradient * gradient
            t = iteration + 1
            mhat = next_m / (1.0 - beta1**t)
            vhat = next_v / (1.0 - beta2**t)
            proposal = project_box(
                eta - lr * mhat / (jnp.sqrt(vhat) + eps), cfg
            )
            next_eta = jnp.where(finite, proposal, eta)
            trace = jnp.stack(
                [value, jnp.linalg.norm(gradient), jnp.asarray(finite, dtype=jnp.float64)]
            )
            return (next_eta, next_m, next_v, t), trace

        initial = (
            jnp.asarray(eta0, dtype=jnp.float64),
            jnp.zeros_like(eta0),
            jnp.zeros_like(eta0),
            jnp.asarray(0, dtype=jnp.int32),
        )
        final, trace = jax.lax.scan(step, initial, schedule)
        return final[0], trace

    compiled = jax.jit(optimize_one)
    rows = []
    for index, (eta0, source) in enumerate(zip(starts, provenance), start=1):
        started = time.perf_counter()
        eta_final, trace = compiled(jnp.asarray(eta0, dtype=jnp.float64))
        eta_final, trace = jax.device_get((eta_final, trace))
        trace = np.asarray(trace, dtype=np.float64)
        rows.append(
            {
                "run": index,
                "stage": stage,
                "provenance": source,
                "initial_eta": np.asarray(eta0).tolist(),
                "final_eta": np.asarray(eta_final).tolist(),
                "initial_penalized_objective": float(trace[0, 0]),
                "final_penalized_objective": float(trace[-1, 0]),
                "final_gradient_norm": float(trace[-1, 1]),
                "iterations": steps,
                "convergence_triggered": False,
                "numerical_invalid_steps": int(np.sum(trace[:, 2] < 0.5)),
                "runtime_seconds": time.perf_counter() - started,
                "trace_objective": trace[:, 0].tolist(),
                "trace_gradient_norm": trace[:, 1].tolist(),
            }
        )
        print(f"[{stage}] gradient start {index}/{len(starts)}", flush=True)
    return rows


def _directional_check(
    name: str,
    fn: Callable[[jax.Array], jax.Array],
    eta: np.ndarray,
    direction: np.ndarray,
    steps: list[float],
    tolerance: float,
) -> dict[str, Any]:
    value_grad = jax.jit(jax.value_and_grad(fn))
    value, gradient = value_grad(jnp.asarray(eta, dtype=jnp.float64))
    value, gradient = jax.device_get((value, gradient))
    ad = float(np.dot(np.asarray(gradient), direction))
    eval_fn = jax.jit(fn)
    rows = []
    for h in steps:
        plus = jnp.asarray(eta + float(h) * direction, dtype=jnp.float64)
        minus = jnp.asarray(eta - float(h) * direction, dtype=jnp.float64)
        vp, vm = jax.device_get((eval_fn(plus), eval_fn(minus)))
        fd = float((vp - vm) / (2.0 * float(h)))
        relative = abs(ad - fd) / max(abs(ad), abs(fd), 1.0e-10)
        rows.append(
            {
                "h": float(h),
                "autodiff_directional_derivative": ad,
                "finite_difference_directional_derivative": fd,
                "relative_error": float(relative),
            }
        )
    best = min(row["relative_error"] for row in rows)
    passed = bool(np.isfinite(best) and best <= float(tolerance))
    if not passed:
        raise RuntimeError(f"v4 {name} gradient check failed: best relative error {best}")
    return {
        "name": name,
        "value": float(value),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "best_relative_error": float(best),
        "tolerance": float(tolerance),
        "passed": passed,
        "step_checks": rows,
    }


def run_gradient_checks(
    cfg: dict[str, Any], objective: V4DifferentiableObjective, bank: V4CRNBank,
    geometries: np.ndarray,
) -> dict[str, Any]:
    block = cfg["v4"]["gradient_checks"]
    h_values = [float(x) for x in block["finite_difference_steps"]]
    tolerance = float(block["relative_tolerance"])
    fidelity = str(block["fidelity"])
    trial_n = min(int(block["multi_trial_count"]), bank.trials)
    rng = np.random.default_rng(int(cfg["seeds"]["gradient_directions"]));
    output = []
    for index, eta in enumerate(geometries[: int(block["geometries"])]):
        direction = rng.normal(size=eta.shape)
        direction /= np.linalg.norm(direction)
        prefix = bank.prefix(trial_n)
        output.extend(
            [
                _directional_check(
                    f"risk_geometry_{index}",
                    lambda e, p=prefix: objective.risk_mean(
                        e, jnp.asarray(p.sampling_z), jnp.asarray(p.detector_z)
                    ),
                    eta,
                    direction,
                    h_values,
                    tolerance,
                ),
                _directional_check(
                    f"one_trial_full_geometry_{index}",
                    lambda e: objective.full_score(
                        e,
                        jnp.asarray(bank.sampling_z[:1]),
                        jnp.asarray(bank.detector_z[:1]),
                        fidelity,
                    ),
                    eta,
                    direction,
                    h_values,
                    tolerance,
                ),
                _directional_check(
                    f"multi_trial_full_geometry_{index}",
                    lambda e, p=prefix: objective.full_score(
                        e, jnp.asarray(p.sampling_z), jnp.asarray(p.detector_z), fidelity
                    ),
                    eta,
                    direction,
                    h_values,
                    tolerance,
                ),
            ]
        )
    return {
        "schema_version": 1,
        "all_passed": bool(all(row["passed"] for row in output)),
        "checks": output,
    }


def _exact_risk_rows(
    evaluator: ProspectiveEvaluator,
    candidates: list[dict[str, Any]],
    bank: V4CRNBank,
    cfg: dict[str, Any],
) -> None:
    for index, row in enumerate(candidates, start=1):
        eta = np.asarray(row["eta"], dtype=np.float64)
        result = evaluator.evaluate_prospective(
            eta, bank.as_observation_bank(), compute_full=False
        )
        row["authoritative_risk_result"] = result
        row["risk"] = _mean(result, "risk")
        row["geometry_valid"] = geometry_valid(eta, cfg)
        print(f"[risk-audit] candidate {index}/{len(candidates)}", flush=True)


def _score_differentiable_candidate(
    objective: V4DifferentiableObjective,
    eta: np.ndarray,
    bank: V4CRNBank,
    fidelity: str,
) -> dict[str, Any]:
    fn = jax.jit(
        lambda e, s, d: objective.full_trials(e, s, d, fidelity)
    )
    action, risks, residual, ess, poisson = jax.device_get(
        fn(
            jnp.asarray(eta, dtype=jnp.float64),
            jnp.asarray(bank.sampling_z),
            jnp.asarray(bank.detector_z),
        )
    )
    action_dist = distribution(action)
    return {
        "action": action_dist,
        "risk": distribution(risks),
        "robust_score": float(action_dist["mean"] + objective.beta * action_dist["sd"]),
        "max_projection_residual": float(np.max(residual)),
        "min_ess_fraction": float(np.min(ess)),
        "max_differentiable_poisson_residual": float(np.max(poisson)),
    }


def _lbfgs_polish(
    eta: np.ndarray,
    objective: V4DifferentiableObjective,
    bank: V4CRNBank,
    fidelity: str,
    risk_limit: float,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    block = cfg["v4"]["full_lbfgs"]
    loss = jax.jit(
        jax.value_and_grad(
            lambda e: objective.constrained_full_loss(
                e,
                jnp.asarray(bank.sampling_z),
                jnp.asarray(bank.detector_z),
                fidelity,
                risk_limit,
            )
            + geometry_penalty(e, cfg)
        )
    )

    def fun(x):
        value, grad = loss(jnp.asarray(x, dtype=jnp.float64))
        value, grad = jax.device_get((value, grad))
        return float(value), np.asarray(grad, dtype=np.float64)

    margin = float(cfg["measurement"]["boundary_margin"])
    bounds = []
    for _ in range(int(cfg["measurement"]["n_sensors"])):
        bounds.extend([(margin, 2.0 - margin), (margin, 1.0 - margin)])
    result = minimize(
        fun,
        np.asarray(eta, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": int(block["max_iterations"]),
            "ftol": float(block["ftol"]),
            "gtol": float(block["gtol"]),
            "maxls": int(block["max_line_search_steps"]),
        },
    )
    return np.asarray(result.x, dtype=np.float64), {
        "enabled": True,
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
        "final_penalized_objective": float(result.fun),
        "gradient_norm": float(np.linalg.norm(result.jac)),
    }


def _authoritative_candidate(
    evaluator: ProspectiveEvaluator,
    eta: np.ndarray,
    bank: V4CRNBank,
    law_risk: float,
    cfg: dict[str, Any],
    source: str,
    candidate_id: str,
    gradient_norm: float | None,
) -> dict[str, Any]:
    result = evaluator.evaluate_prospective(
        eta, bank.as_observation_bank(), compute_full=True
    )
    actions = _trial_values(result, "full_action")
    action_dist = distribution(actions)
    risk = _mean(result, "risk")
    robust = (
        float(action_dist["mean"] + float(cfg["v4"]["robustness_beta"]) * action_dist["sd"])
        if action_dist["mean"] is not None else float("inf")
    )
    trials = result["trials"]
    certified = bool(
        result["valid"]
        and geometry_valid(eta, cfg)
        and _risk_feasible(risk, law_risk, cfg)
    )
    return {
        "candidate_id": candidate_id,
        "source": source,
        "eta": np.asarray(eta).tolist(),
        "centers": np.asarray(eta).reshape((-1, 2)).tolist(),
        "risk": risk,
        "mean_full": action_dist["mean"],
        "full_sd": action_dist["sd"],
        "full_se": action_dist["se"],
        "full_distribution": action_dist,
        "robust_score": robust,
        "gradient_norm": gradient_norm,
        "certified": certified,
        "valid_fraction": float(result["valid_fraction"]),
        "max_projection_residual": float(max(row["max_projection_residual"] for row in trials)),
        "min_ess_fraction": float(min(row["min_ess_fraction"] for row in trials)),
        "min_covariance_eigenvalue": float(min(row["min_covariance_eigenvalue"] for row in trials)),
        "max_poisson_relative_residual": float(max(row["max_poisson_relative_residual"] for row in trials)),
        "max_component_compatibility_residual": float(max(row["max_component_compatibility_residual"] for row in trials)),
        "max_full_moment_rate_residual": float(max(row["max_full_moment_rate_residual"] for row in trials)),
        "all_full_solvers_converged": bool(all(row["full_solver_converged"] for row in trials)),
        "authoritative_result": result,
    }


def _git_snapshot() -> dict[str, Any]:
    def command(args):
        return subprocess.run(
            args, cwd=REPO_ROOT, check=False, capture_output=True, text=True
        ).stdout.strip()
    return {
        "commit": command(["git", "rev-parse", "HEAD"]),
        "status_short": command(["git", "status", "--short"]),
        "diff_stat": command(["git", "diff", "--stat"]),
    }


def _plot_traces(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    count = min(8, len(rows))
    for row in rows[:count]:
        ax.plot(row["trace_objective"], alpha=0.8, label=f"run {row['run']}")
    ax.set(xlabel="Adam iteration", ylabel="penalized prospective Full objective")
    if count <= 8:
        ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def select_and_freeze_v4(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    dirs = artifact_dirs(output_dir)
    dirs["results"].mkdir(parents=True, exist_ok=True)
    manifest_path = dirs["results"] / "frozen_manifest.json"
    required = {
        "endpoint": dirs["endpoint"] / "endpoint_data.npz",
        "aggregate": dirs["prospective"] / "aggregate_predictions.npz",
        "rollout": dirs["endpoint"] / "reference_rollout.npz",
        "checkpoint": dirs["endpoint"] / "reference_checkpoint.npz",
        "input_receipt": dirs["prospective"] / "v4_input_receipt.json",
    }
    for path in required.values():
        if not path.exists():
            raise FileNotFoundError(f"v4 prerequisite missing: {path}")
    master_trials = int(cfg["v4"]["selection_crn_trials"])
    crn_path = dirs["prospective"] / "v4_selection_crn.npz"
    master_bank = ensure_v4_crn_bank(crn_path, cfg, master_trials)
    time_audit = run_time_derivative_audit(cfg, output_dir)
    signature = {
        "config_hash": config_hash(cfg),
        "v4_source_hash": v4_source_hash(),
        "endpoint_sha256": file_sha256(required["endpoint"]),
        "aggregate_sha256": file_sha256(required["aggregate"]),
        "reference_rollout_sha256": file_sha256(required["rollout"]),
        "reference_checkpoint_sha256": file_sha256(required["checkpoint"]),
        "selection_crn_sha256": file_sha256(crn_path),
        "input_receipt_sha256": file_sha256(required["input_receipt"]),
        "time_audit_sha256": file_sha256(dirs["results"] / "time_derivative_audit.json"),
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("selection_input_hashes") == signature:
            print("[v4-selection] reusing compatible frozen manifest", flush=True)
            return manifest
        raise RuntimeError("a different v4 frozen manifest already exists; use a new run identity")
    if dirs["hidden"].exists() and any(dirs["hidden"].iterdir()):
        raise RuntimeError("v4 hidden-validation data exists before the selection freeze")

    started = time.perf_counter()
    timings: dict[str, float] = {}
    data = TargetProspectiveData.load(required["endpoint"], required["aggregate"])
    objective = V4DifferentiableObjective(cfg, data, required["rollout"])
    authoritative = ProspectiveEvaluator(cfg, data, required["rollout"])

    law_starts, law_sources = generate_law_starts(cfg)
    check_started = time.perf_counter()
    gradient_checks = run_gradient_checks(cfg, objective, master_bank, law_starts)
    gradient_checks["signature"] = signature
    write_json_atomic(dirs["results"] / "gradient_checks.json", gradient_checks)
    timings["gradient_checks"] = time.perf_counter() - check_started

    law_started = time.perf_counter()
    law_settings = cfg["v4"]["law_optimizer"]
    law_bank = master_bank.prefix(int(law_settings["crn_trials"]))
    law_runs = _adam_multistart(
        law_starts,
        law_sources,
        law_bank,
        law_settings,
        cfg,
        lambda eta, s, d: objective.risk_mean(eta, s, d),
        schedule_seed=int(cfg["seeds"]["law_batch_schedule"]),
        stage="v4-law-adam",
    )
    law_candidates = [
        {
            "candidate_id": f"law-grad-{row['run']:03d}",
            "eta": row["final_eta"],
            "source": row["provenance"],
            "gradient_run": row,
        }
        for row in law_runs
    ]
    risk_bank = master_bank.prefix(int(cfg["v4"]["authoritative_crn_trials"]))
    _exact_risk_rows(authoritative, law_candidates, risk_bank, cfg)
    valid_law = [
        row for row in law_candidates
        if row["authoritative_risk_result"]["valid"] and row["geometry_valid"]
    ]
    if not valid_law:
        raise RuntimeError("v4 Law gradient optimization produced no certified geometry")
    law_row = min(valid_law, key=lambda row: row["risk"])
    law_eta = np.asarray(law_row["eta"], dtype=np.float64)
    law_risk = float(law_row["risk"])
    risk_limit = (1.0 + float(cfg["risk_allowance"])) * law_risk
    timings["law_gradient_selection"] = time.perf_counter() - law_started

    full_started = time.perf_counter()
    full_starts, full_sources = generate_full_starts(cfg, law_eta, tangent_eta=None)
    full_settings = cfg["v4"]["full_optimizer"]
    search_fidelity = str(full_settings["fidelity"])
    search_bank = master_bank.prefix(objective.fidelity(search_fidelity).trials)
    full_runs = _adam_multistart(
        full_starts,
        full_sources,
        search_bank,
        full_settings,
        cfg,
        lambda eta, s, d: objective.constrained_full_loss(
            eta, s, d, search_fidelity, risk_limit
        ),
        schedule_seed=int(cfg["seeds"]["full_batch_schedule"]),
        stage="v4-full-adam",
    )
    _plot_traces(dirs["results"] / "full_optimization_traces.png", full_runs)
    full_candidates = []
    seen = set()
    search_score_fn = jax.jit(
        lambda e: objective.full_score(
            e,
            jnp.asarray(search_bank.sampling_z),
            jnp.asarray(search_bank.detector_z),
            search_fidelity,
        )
    )
    for row in full_runs:
        eta = np.asarray(row["final_eta"], dtype=np.float64)
        key = canonical_geometry_key(eta)
        if key in seen:
            continue
        seen.add(key)
        full_candidates.append(
            {
                "candidate_id": f"full-grad-{row['run']:03d}",
                "eta": eta.tolist(),
                "source": row["provenance"].replace("gradient start", "gradient refined"),
                "gradient_run": row,
                "search_full_score": float(search_score_fn(jnp.asarray(eta))),
            }
        )
    _exact_risk_rows(authoritative, full_candidates, risk_bank, cfg)
    feasible_full = [
        row for row in full_candidates
        if row["authoritative_risk_result"]["valid"]
        and row["geometry_valid"]
        and _risk_feasible(row["risk"], law_risk, cfg)
    ]
    if not feasible_full:
        raise RuntimeError("v4 Full gradient search produced no exact risk-feasible candidate")
    feasible_full.sort(key=lambda row: row["search_full_score"])
    timings["full_stage1_gradient_search"] = time.perf_counter() - full_started

    rescore_started = time.perf_counter()
    rescore_name = str(cfg["v4"]["funnel"]["rescore_fidelity"])
    rescore_bank = master_bank.prefix(objective.fidelity(rescore_name).trials)
    rescore_rows = feasible_full[: int(cfg["v4"]["funnel"]["rescore_candidates"])]
    for index, row in enumerate(rescore_rows, start=1):
        row["prospective_rescore"] = _score_differentiable_candidate(
            objective, np.asarray(row["eta"]), rescore_bank, rescore_name
        )
        print(f"[v4-rescore] candidate {index}/{len(rescore_rows)}", flush=True)
    rescore_rows.sort(key=lambda row: row["prospective_rescore"]["robust_score"])
    timings["prospective_rescoring"] = time.perf_counter() - rescore_started

    polish_started = time.perf_counter()
    polish_name = str(cfg["v4"]["funnel"]["polish_fidelity"])
    polish_bank = master_bank.prefix(objective.fidelity(polish_name).trials)
    polish_seed_rows = rescore_rows[: int(cfg["v4"]["funnel"]["polish_candidates"])]
    polish_settings = {
        **full_settings,
        "steps": int(cfg["v4"]["funnel"]["polish_adam_steps"]),
        "batch_size": int(cfg["v4"]["funnel"]["polish_batch_size"]),
    }
    polished_runs = _adam_multistart(
        np.asarray([row["eta"] for row in polish_seed_rows]),
        ["Full intermediate-fidelity gradient polish"] * len(polish_seed_rows),
        polish_bank,
        polish_settings,
        cfg,
        lambda eta, s, d: objective.constrained_full_loss(
            eta, s, d, polish_name, risk_limit
        ),
        schedule_seed=int(cfg["seeds"]["full_polish_batch_schedule"]),
        stage="v4-full-polish",
    )
    polished = []
    for source_row, run_row in zip(polish_seed_rows, polished_runs):
        eta = np.asarray(run_row["final_eta"], dtype=np.float64)
        lbfgs = {"enabled": False, "iterations": 0}
        if bool(cfg["v4"]["full_lbfgs"]["enabled"]):
            eta, lbfgs = _lbfgs_polish(
                eta, objective, polish_bank, polish_name, risk_limit, cfg
            )
        scored = {
            "candidate_id": source_row["candidate_id"] + "-polished",
            "eta": eta.tolist(),
            "source": "Full Adam + L-BFGS Full-objective polished",
            "parent_candidate_id": source_row["candidate_id"],
            "gradient_run": run_row,
            "lbfgs": lbfgs,
            "prospective_rescore": _score_differentiable_candidate(
                objective, eta, polish_bank, polish_name
            ),
        }
        polished.append(scored)
    _exact_risk_rows(authoritative, polished, risk_bank, cfg)
    polished = [
        row for row in polished
        if row["authoritative_risk_result"]["valid"]
        and row["geometry_valid"]
        and _risk_feasible(row["risk"], law_risk, cfg)
    ]
    if not polished:
        raise RuntimeError("v4 Full-objective polishing produced no exact feasible candidate")
    timings["full_objective_gradient_polish"] = time.perf_counter() - polish_started

    finalist_started = time.perf_counter()
    pool = polished + rescore_rows
    pool.sort(key=lambda row: row["prospective_rescore"]["robust_score"])
    finalist_inputs = []
    finalist_seen = set()
    for row in pool:
        key = canonical_geometry_key(row["eta"])
        if key in finalist_seen:
            continue
        finalist_seen.add(key)
        finalist_inputs.append(row)
        if len(finalist_inputs) >= int(cfg["v4"]["funnel"]["authoritative_full_finalists"]):
            break
    authoritative_rows = []
    for index, row in enumerate(finalist_inputs, start=1):
        gradient_norm = float(row["gradient_run"]["final_gradient_norm"])
        authoritative_rows.append(
            _authoritative_candidate(
                authoritative,
                np.asarray(row["eta"]),
                risk_bank,
                law_risk,
                cfg,
                row["source"],
                row["candidate_id"],
                gradient_norm,
            )
        )
        print(f"[v4-authoritative] Full finalist {index}/{len(finalist_inputs)}", flush=True)
    certified_full = [row for row in authoritative_rows if row["certified"]]
    if not certified_full:
        raise RuntimeError("no v4 authoritative Full finalist is certified and risk-feasible")
    full_row = min(certified_full, key=lambda row: row["robust_score"])
    law_authoritative = _authoritative_candidate(
        authoritative,
        law_eta,
        risk_bank,
        law_risk,
        cfg,
        "v4 Law gradient optimum",
        law_row["candidate_id"],
        float(law_row["gradient_run"]["final_gradient_norm"]),
    )
    if not law_authoritative["certified"]:
        raise RuntimeError("v4 authoritative Law comparator failed certification")
    timings["authoritative_finalist_selection"] = time.perf_counter() - finalist_started

    candidate_archive = {
        "schema_version": 1,
        "law_gradient_runs": law_runs,
        "law_candidates": law_candidates,
        "full_gradient_runs": full_runs,
        "full_candidates": full_candidates,
        "risk_feasible_full_candidates": [row["candidate_id"] for row in feasible_full],
        "rescored_candidates": rescore_rows,
        "polished_candidates": polished,
    }
    archive_path = dirs["results"] / "v4_candidate_archive.json"
    write_json_atomic(archive_path, candidate_archive)
    finalist_path = dirs["results"] / "v4_finalists.json"
    write_json_atomic(
        finalist_path,
        {"Law": law_authoritative, "Full_finalists": authoritative_rows},
    )
    table_rows = [
        {
            key: row[key]
            for key in (
                "candidate_id", "source", "risk", "mean_full", "full_sd",
                "robust_score", "gradient_norm", "certified",
                "max_projection_residual", "min_ess_fraction",
                "max_poisson_relative_residual", "max_full_moment_rate_residual",
            )
        }
        for row in [law_authoritative] + authoritative_rows
    ]
    _write_csv(dirs["results"] / "v4_finalists.csv", table_rows)

    manifest = {
        "schema_version": 4,
        "status": "frozen_before_hidden_validation",
        "experiment": cfg["name"],
        "config": cfg,
        "config_hash": config_hash(cfg),
        "selection_input_hashes": signature,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_snapshot(),
        "software": software_metadata(),
        "previous_hidden_validation_used_for_v4_design": False,
        "hidden_validation_loaded": False,
        "input_artifacts": json.loads(required["input_receipt"].read_text(encoding="utf-8")),
        "reference": {
            "checkpoint_sha256": signature["reference_checkpoint_sha256"],
            "rollout_sha256": signature["reference_rollout_sha256"],
            "training_inputs": ["x0", "x1"],
        },
        "aggregate_predictor": {
            "sha256": signature["aggregate_sha256"],
            "differentiable_interpolation": "piecewise bilinear on the declared continuous response table",
        },
        "qoi_definition_hash": fingerprint(cfg["qoi"]),
        "selection_crn": {
            "sha256": signature["selection_crn_sha256"],
            "trials": master_trials,
            "sampling_seed": int(cfg["seeds"]["selection_sampling_crn"]),
            "detector_seed": int(cfg["seeds"]["selection_detector_crn"]),
            "common_across_geometries": True,
            "resampled_during_optimization": False,
        },
        "observation_reparameterization": acquisition_reparameterization_formula(cfg),
        "risk_definition": {
            "kind": "mean noisy-reconstructed aggregate scientific QoI risk",
            "same_for_law_and_full": True,
            "allowance": float(cfg["risk_allowance"]),
            "anchor": law_risk,
            "ceiling": risk_limit,
            "scalarized_with_full": False,
        },
        "full_objective": {
            "definition": "mean_r FullAction(S(mean_eta + finite_se_eta*z_sampling + sigma*z_detector)) + beta*sample_sd",
            "primary_mean_action": True,
            "robustness_beta": float(cfg["v4"]["robustness_beta"]),
            "production_reconstruction_inside_objective": True,
            "implicit_information_projection_gradient": True,
            "implicit_poisson_gradient": True,
            "tangent_gradient_substituted": False,
        },
        "optimizer": {
            "law": cfg["v4"]["law_optimizer"],
            "full_adam": cfg["v4"]["full_optimizer"],
            "full_lbfgs": cfg["v4"]["full_lbfgs"],
            "funnel": cfg["v4"]["funnel"],
            "full_specific_search_independent_of_tangent": True,
            "tangent_enabled": False,
        },
        "gradient_checks": gradient_checks,
        "time_derivative_audit": time_audit,
        "candidate_archive_sha256": file_sha256(archive_path),
        "finalist_table_sha256": file_sha256(finalist_path),
        "selected": {
            "Law": {
                "eta": law_authoritative["eta"],
                "centers": law_authoritative["centers"],
                "predicted": law_authoritative,
            },
            "Full": {
                "eta": full_row["eta"],
                "centers": full_row["centers"],
                "predicted": full_row,
            },
        },
        "selection_metrics": {
            "law_gradient_starts": len(law_runs),
            "full_gradient_starts": len(full_runs),
            "genuinely_full_refined_distinct_candidates": len(full_candidates),
            "genuinely_full_refined_risk_feasible_candidates": len(feasible_full),
            "full_objective_polished_feasible_candidates": len(polished),
            "authoritative_full_finalists": len(authoritative_rows),
            "law_risk": law_risk,
            "risk_ceiling": risk_limit,
            "law_predicted_full_mean": law_authoritative["mean_full"],
            "full_predicted_full_mean": full_row["mean_full"],
            "predicted_full_reduction": 1.0 - full_row["mean_full"] / law_authoritative["mean_full"],
        },
        "numerical_certification": {
            "Law": {key: law_authoritative[key] for key in (
                "certified", "valid_fraction", "max_projection_residual",
                "min_ess_fraction", "min_covariance_eigenvalue",
                "max_poisson_relative_residual", "max_component_compatibility_residual",
                "max_full_moment_rate_residual", "all_full_solvers_converged",
            )},
            "Full": {key: full_row[key] for key in (
                "certified", "valid_fraction", "max_projection_residual",
                "min_ess_fraction", "min_covariance_eigenvalue",
                "max_poisson_relative_residual", "max_component_compatibility_residual",
                "max_full_moment_rate_residual", "all_full_solvers_converged",
            )},
        },
        "stage_runtime_seconds": timings,
        "selection_elapsed_seconds": time.perf_counter() - started,
    }
    write_json_atomic(manifest_path, manifest)
    print(f"[v4-selection] frozen manifest written: {manifest_path}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    select_and_freeze_v4(load_config(args.config), args.output_dir)


if __name__ == "__main__":
    main()
