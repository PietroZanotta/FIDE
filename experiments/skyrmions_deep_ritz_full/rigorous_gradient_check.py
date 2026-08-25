"""Rigorous fixed-design validation of the Deep Ritz envelope derivative.

This module never updates sensor coordinates.  It tracks one deterministic
local Ritz branch from a stationary center and compares the fixed-theta VJP
against optimized-value finite differences in several fixed directions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
from scipy.optimize import minimize

from .deep_ritz import (
    CertificateConfig,
    RitzParams,
    audit_deep_ritz,
    load_ritz_checkpoint,
    ritz_objective,
)
from .full_gradient import (
    envelope_full_value_and_grad,
    forcing_state,
    full_energy_from_state,
    periodic_branch_distance,
    reconstruct_moments,
    ritz_objective_eta,
    wrap_periodic,
)
from .workflow import (
    OUTPUT_ROOT,
    InnerSolution,
    PreparedExperiment,
    hard_forcing_audit,
    require_output_path,
    solve_inner,
    write_json,
)

Array = jax.Array


@dataclass(frozen=True)
class StationarySolution:
    params: RitzParams
    metrics: dict[str, Any]
    rounds: list[dict[str, Any]]


def _gradient_validation_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg["envelope"]["gradient_validation"]


def _parameter_gradient_metrics(gradient: RitzParams) -> dict[str, Any]:
    flat, _ = ravel_pytree(gradient)
    count = int(flat.size)
    return {
        "parameter_count": count,
        "raw_norm": float(jnp.linalg.norm(flat)),
        "rms": float(jnp.linalg.norm(flat) / jnp.sqrt(float(max(count, 1)))),
        "max_abs": float(jnp.max(jnp.abs(flat))),
    }


def _optimizer_payload(inner: InnerSolution) -> dict[str, Any]:
    last_lbfgs = next(
        (row for row in reversed(inner.result.history) if row.get("phase") == "lbfgs"),
        {},
    )
    return {
        "finite": bool(inner.result.finite),
        "lbfgs_converged": bool(inner.result.lbfgs_converged),
        "adam_final_objective": float(inner.result.adam_final_objective),
        "lbfgs_final_objective": float(inner.result.lbfgs_final_objective),
        "last_recorded_lbfgs_gradient_norm": (
            float(last_lbfgs["gradient_norm"])
            if "gradient_norm" in last_lbfgs else None
        ),
        "adam_seconds": float(inner.result.adam_seconds),
        "lbfgs_seconds": float(inner.result.lbfgs_seconds),
    }


def _scipy_lbfgs_polish(
    eta: Array,
    params: RitzParams,
    cfg: dict[str, Any],
    data: PreparedExperiment,
) -> tuple[RitzParams, dict[str, Any]]:
    """Robust deterministic full-bank L-BFGS polish for theta only.

    The NumPy conversion is strictly the SciPy optimizer boundary.  Every
    objective/gradient evaluation is still a JAX float64 full-bank evaluation,
    and this routine is never part of the differentiated eta graph.
    """

    problem = data.selection_problem
    reconstruction = reconstruct_moments(eta, problem)
    state = forcing_state(eta, problem, data.ritz_train_bank, reconstruction)
    flat0, unravel = ravel_pytree(params)

    def flat_objective(vector: Array) -> Array:
        return ritz_objective(
            unravel(vector),
            data.ritz_train_bank.configurations,
            state.projection.weights,
            state.forcing,
            problem.times,
            problem.time_weights,
            box=problem.box,
        )

    value_gradient = jax.jit(jax.value_and_grad(flat_objective))

    def scipy_value_gradient(vector: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = value_gradient(jnp.asarray(vector, dtype=jnp.float64))
        return float(value), np.asarray(jax.device_get(gradient), dtype=np.float64)

    validation_cfg = _gradient_validation_config(cfg)
    result = minimize(
        scipy_value_gradient,
        np.asarray(jax.device_get(flat0), dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(validation_cfg["theta_lbfgs_iterations"]),
            "maxcor": int(validation_cfg["theta_lbfgs_history"]),
            "maxls": int(validation_cfg["theta_lbfgs_line_search_steps"]),
            "gtol": float(validation_cfg["theta_lbfgs_gradient_tolerance"]),
            "ftol": float(validation_cfg["theta_lbfgs_function_tolerance"]),
        },
    )
    polished = unravel(jnp.asarray(result.x, dtype=jnp.float64))
    return polished, {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "final_objective": float(result.fun),
        "reported_max_abs_gradient": float(np.max(np.abs(result.jac))),
    }


def stationary_metrics(
    eta: Array,
    params: RitzParams,
    cfg: dict[str, Any],
    data: PreparedExperiment,
    *,
    optimizer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute exact full-bank stationarity and held-out physical diagnostics."""

    problem = data.selection_problem
    eta = wrap_periodic(eta, problem.family)
    reconstruction = reconstruct_moments(eta, problem)
    train_state = forcing_state(eta, problem, data.ritz_train_bank, reconstruction)

    def objective(theta: RitzParams) -> Array:
        return ritz_objective(
            theta,
            data.ritz_train_bank.configurations,
            train_state.projection.weights,
            train_state.forcing,
            problem.times,
            problem.time_weights,
            box=problem.box,
        )

    objective_value, parameter_gradient = jax.value_and_grad(objective)(params)
    kinetic_action = full_energy_from_state(
        params, problem, data.ritz_train_bank, train_state
    )
    absolute_identity = jnp.abs(kinetic_action + 2.0 * objective_value)
    relative_identity = absolute_identity / jnp.maximum(
        jnp.abs(kinetic_action), 1.0e-12
    )

    audit_state = forcing_state(
        eta, problem, data.ritz_audit_bank, reconstruction
    )
    certificate = audit_deep_ritz(
        params,
        data.ritz_audit_bank.configurations,
        audit_state.projection.weights,
        audit_state.forcing,
        problem.times,
        problem.time_weights,
        family=problem.family,
        eta=eta,
        reference_velocity=data.ritz_audit_bank.velocity,
        target_derivatives=reconstruction.derivatives,
        cfg=CertificateConfig(**cfg["certificates"]),
        box=problem.box,
        chunk_size=min(1024, int(data.ritz_audit_bank.configurations.shape[1])),
    )
    train_forcing = hard_forcing_audit(eta, problem, data.ritz_train_bank)
    audit_forcing = hard_forcing_audit(eta, problem, data.ritz_audit_bank)
    gradient_metrics = _parameter_gradient_metrics(parameter_gradient)
    thresholds = _gradient_validation_config(cfg)
    stationary = bool(
        float(relative_identity) <= float(thresholds["maximum_energy_identity_relerr"])
        and gradient_metrics["rms"] <= float(thresholds["maximum_gradient_rms"])
        and gradient_metrics["max_abs"]
        <= float(thresholds["maximum_gradient_max_abs"])
    )
    diagnostics_valid = bool(
        train_forcing["valid"]
        and audit_forcing["valid"]
        and bool(jax.device_get(problem.family.geometry_valid(eta)))
        and (
            certificate["valid"]
            or not bool(thresholds.get("require_certificate_valid", False))
        )
    )
    return {
        "eta": jax.device_get(eta).tolist(),
        "ritz_objective_J": float(objective_value),
        "optimized_value_V_minus_2J": float(-2.0 * objective_value),
        "kinetic_action_A": float(kinetic_action),
        "absolute_A_plus_2J": float(absolute_identity),
        "energy_identity_relerr": float(relative_identity),
        "parameter_gradient": gradient_metrics,
        "stationary": stationary,
        "diagnostics_valid": diagnostics_valid,
        "geometry_valid": bool(jax.device_get(problem.family.geometry_valid(eta))),
        "periodic_branch_distance": float(periodic_branch_distance(eta, problem.family)),
        "train_forcing": train_forcing,
        "audit_forcing": audit_forcing,
        "ritz_audit": certificate,
        "optimizer": optimizer,
        "thresholds": {
            "maximum_energy_identity_relerr": float(
                thresholds["maximum_energy_identity_relerr"]
            ),
            "maximum_gradient_rms": float(thresholds["maximum_gradient_rms"]),
            "maximum_gradient_max_abs": float(
                thresholds["maximum_gradient_max_abs"]
            ),
            "require_certificate_valid": bool(
                thresholds.get("require_certificate_valid", False)
            ),
        },
    }


def solve_to_stationarity(
    eta: Array,
    cfg: dict[str, Any],
    data: PreparedExperiment,
    *,
    initial_params: RitzParams | None,
    initialize: bool,
    maximum_rounds: int,
) -> StationarySolution:
    """Deterministic full-bank solve followed by repeatable L-BFGS polishes."""

    params = initial_params
    rounds: list[dict[str, Any]] = []
    metrics: dict[str, Any] | None = None
    for round_index in range(max(int(maximum_rounds), 1)):
        mode = (
            "gradient_validation_center"
            if initialize and round_index == 0
            else "gradient_validation_polish"
        )
        inner = solve_inner(
            eta,
            cfg,
            data.selection_problem,
            data.ritz_train_bank,
            mode=mode,
            initial_params=params,
        )
        params, scipy_optimizer = _scipy_lbfgs_polish(
            eta, inner.params, cfg, data
        )
        optimizer = _optimizer_payload(inner)
        optimizer["deterministic_full_bank_scipy_lbfgs"] = scipy_optimizer
        metrics = stationary_metrics(
            eta, params, cfg, data, optimizer=optimizer
        )
        rounds.append({
            "round": round_index + 1,
            "mode": mode,
            "optimizer": optimizer,
            "ritz_objective_J": metrics["ritz_objective_J"],
            "optimized_value_V_minus_2J": metrics["optimized_value_V_minus_2J"],
            "kinetic_action_A": metrics["kinetic_action_A"],
            "energy_identity_relerr": metrics["energy_identity_relerr"],
            "parameter_gradient": metrics["parameter_gradient"],
            "stationary": metrics["stationary"],
            "diagnostics_valid": metrics["diagnostics_valid"],
        })
        if metrics["stationary"]:
            break
    assert params is not None and metrics is not None
    metrics["polish_round_count"] = len(rounds)
    return StationarySolution(params=params, metrics=metrics, rounds=rounds)


def _load_local_warm_start() -> tuple[RitzParams | None, dict[str, Any] | None]:
    path = OUTPUT_ROOT / "gradient_checks" / "smoke" / "theta_center.npz"
    if not path.is_file():
        return None, None
    params, metadata = load_ritz_checkpoint(path)
    return params, {"path": str(path), "metadata": metadata}


def _choose_center(
    warm: StationarySolution | None, fresh: StationarySolution
) -> tuple[str, StationarySolution]:
    if warm is None:
        return "fresh", fresh
    candidates = [("warm", warm), ("fresh", fresh)]
    stationary = [item for item in candidates if item[1].metrics["stationary"]]
    if stationary:
        return min(stationary, key=lambda item: item[1].metrics["ritz_objective_J"])

    def failure_score(item: tuple[str, StationarySolution]) -> float:
        metrics = item[1].metrics
        thresholds = metrics["thresholds"]
        return (
            float(metrics["energy_identity_relerr"])
            / float(thresholds["maximum_energy_identity_relerr"])
            + float(metrics["parameter_gradient"]["rms"])
            / float(thresholds["maximum_gradient_rms"])
            + float(metrics["parameter_gradient"]["max_abs"])
            / float(thresholds["maximum_gradient_max_abs"])
        )

    # If neither branch is stationary, report the least nonstationary branch;
    # choosing the most negative J would reward the observed runaway.
    return min(candidates, key=failure_score)


def _basin_payload(
    warm: StationarySolution | None,
    fresh: StationarySolution,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    if warm is None:
        return {
            "warm_start_available": False,
            "similar_optimized_values": None,
            "relative_value_difference": None,
        }
    warm_value = float(warm.metrics["optimized_value_V_minus_2J"])
    fresh_value = float(fresh.metrics["optimized_value_V_minus_2J"])
    relative = abs(warm_value - fresh_value) / max(
        abs(warm_value), abs(fresh_value), 1.0e-12
    )
    tolerance = float(
        _gradient_validation_config(cfg)["basin_value_relative_tolerance"]
    )
    return {
        "warm_start_available": True,
        "warm_value": warm_value,
        "fresh_value": fresh_value,
        "absolute_value_difference": abs(warm_value - fresh_value),
        "relative_value_difference": relative,
        "relative_tolerance": tolerance,
        "similar_optimized_values": relative <= tolerance,
        "warm_stationary": bool(warm.metrics["stationary"]),
        "fresh_stationary": bool(fresh.metrics["stationary"]),
    }


def _determinism_payload(
    eta: Array,
    theta: RitzParams,
    cfg: dict[str, Any],
    data: PreparedExperiment,
) -> dict[str, Any]:
    first_value, first_gradient, _ = envelope_full_value_and_grad(
        eta, theta, data.selection_problem, data.ritz_train_bank
    )
    second_value, second_gradient, _ = envelope_full_value_and_grad(
        eta, theta, data.selection_problem, data.ritz_train_bank
    )
    first_j = ritz_objective_eta(
        theta, eta, data.selection_problem, data.ritz_train_bank
    )
    second_j = ritz_objective_eta(
        theta, eta, data.selection_problem, data.ritz_train_bank
    )
    tolerance = float(
        _gradient_validation_config(cfg)["determinism_absolute_tolerance"]
    )
    objective_difference = float(jnp.abs(first_j - second_j))
    value_difference = float(jnp.abs(first_value - second_value))
    gradient_difference = float(jnp.max(jnp.abs(first_gradient - second_gradient)))
    return {
        "first_J": float(first_j),
        "second_J": float(second_j),
        "objective_absolute_difference": objective_difference,
        "envelope_value_absolute_difference": value_difference,
        "gradient_max_absolute_difference": gradient_difference,
        "absolute_tolerance": tolerance,
        "passed": bool(
            objective_difference <= tolerance
            and value_difference <= tolerance
            and gradient_difference <= tolerance
        ),
    }


def _direction_is_admissible(
    eta0: Array,
    direction: Array,
    maximum_epsilon: float,
    cfg: dict[str, Any],
    data: PreparedExperiment,
) -> tuple[bool, dict[str, Any]]:
    family = data.selection_problem.family
    plus = wrap_periodic(eta0 + maximum_epsilon * direction, family)
    minus = wrap_periodic(eta0 - maximum_epsilon * direction, family)
    plus_geometry = bool(jax.device_get(family.geometry_valid(plus)))
    minus_geometry = bool(jax.device_get(family.geometry_valid(minus)))
    minimum_branch = float(
        _gradient_validation_config(cfg)["minimum_branch_distance"]
    )
    plus_branch = float(periodic_branch_distance(plus, family))
    minus_branch = float(periodic_branch_distance(minus, family))
    if not (
        plus_geometry and minus_geometry
        and plus_branch >= minimum_branch and minus_branch >= minimum_branch
    ):
        return False, {
            "plus_geometry_valid": plus_geometry,
            "minus_geometry_valid": minus_geometry,
            "plus_branch_distance": plus_branch,
            "minus_branch_distance": minus_branch,
        }
    plus_forcing = hard_forcing_audit(
        plus, data.selection_problem, data.ritz_train_bank
    )
    minus_forcing = hard_forcing_audit(
        minus, data.selection_problem, data.ritz_train_bank
    )
    admissible = bool(plus_forcing["valid"] and minus_forcing["valid"])
    return admissible, {
        "plus_geometry_valid": plus_geometry,
        "minus_geometry_valid": minus_geometry,
        "plus_branch_distance": plus_branch,
        "minus_branch_distance": minus_branch,
        "plus_forcing": plus_forcing,
        "minus_forcing": minus_forcing,
    }


def deterministic_directions(
    eta0: Array, cfg: dict[str, Any], data: PreparedExperiment
) -> tuple[list[Array], list[dict[str, Any]]]:
    validation_cfg = _gradient_validation_config(cfg)
    count = int(validation_cfg["direction_count"])
    maximum_epsilon = max(float(value) for value in validation_cfg["epsilon_ladder"])
    directions: list[Array] = []
    attempts: list[dict[str, Any]] = []
    key = jax.random.PRNGKey(int(validation_cfg["direction_seed"]))
    for attempt in range(int(validation_cfg["maximum_direction_attempts"])):
        direction = jax.random.normal(
            jax.random.fold_in(key, attempt), eta0.shape, dtype=jnp.float64
        )
        direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1.0e-30)
        admissible, diagnostics = _direction_is_admissible(
            eta0, direction, maximum_epsilon, cfg, data
        )
        attempts.append({
            "attempt": attempt,
            "direction": jax.device_get(direction).tolist(),
            "accepted": admissible,
            "diagnostics": diagnostics,
        })
        if admissible:
            directions.append(direction)
            if len(directions) == count:
                break
    return directions, attempts


def _has_consecutive(values: list[bool], count: int) -> bool:
    if count <= 0:
        return True
    return any(all(values[index:index + count]) for index in range(len(values) - count + 1))


def _continuity_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    plus = [float(row["continuity_plus_error"]) for row in rows]
    minus = [float(row["continuity_minus_error"]) for row in rows]

    def summarize(values: list[float]) -> dict[str, Any]:
        steps = [values[index + 1] <= values[index] for index in range(len(values) - 1)]
        required = max((len(steps) + 1) // 2, 1)
        return {
            "errors": values,
            "nonincreasing_step_count": sum(steps),
            "required_nonincreasing_steps": required,
            "final_below_initial": values[-1] <= values[0],
            "approaches_center": bool(values[-1] <= values[0] and sum(steps) >= required),
        }

    plus_summary = summarize(plus)
    minus_summary = summarize(minus)
    return {
        "plus": plus_summary,
        "minus": minus_summary,
        "passed": bool(
            plus_summary["approaches_center"] and minus_summary["approaches_center"]
        ),
    }


def _direction_summary(
    rows: list[dict[str, Any]], ad_directional: float, cfg: dict[str, Any]
) -> dict[str, Any]:
    validation_cfg = _gradient_validation_config(cfg)
    sign_count = int(validation_cfg["consecutive_sign_count"])
    accuracy_count = int(validation_cfg["consecutive_accuracy_count"])
    relative_tolerance = float(validation_cfg["relative_error_tolerance"])
    preferred = float(validation_cfg["preferred_relative_error"])
    ad_sign = 1 if ad_directional > 0.0 else -1 if ad_directional < 0.0 else 0
    fd_signs = [
        1 if row["fd_optimized_value"] > 0.0 else -1 if row["fd_optimized_value"] < 0.0 else 0
        for row in rows
    ]
    same_sign = [sign != 0 and sign == ad_sign for sign in fd_signs]
    accurate = [row["relative_error_V"] <= relative_tolerance for row in rows]
    stationary = [
        bool(row["plus_metrics"]["stationary"] and row["minus_metrics"]["stationary"])
        for row in rows
    ]
    diagnostics = [
        bool(
            row["plus_metrics"]["diagnostics_valid"]
            and row["minus_metrics"]["diagnostics_valid"]
        )
        for row in rows
    ]
    continuity = _continuity_summary(rows)
    common_epsilon = float(validation_cfg["common_epsilon"])
    common = next(
        (row for row in rows if abs(float(row["epsilon"]) - common_epsilon) <= 1.0e-15),
        None,
    )
    preferred_pass = any(row["relative_error_V"] <= preferred for row in rows)
    required_preferred = bool(validation_cfg.get("require_preferred_accuracy", False))
    rules = {
        "consecutive_fd_sign_and_AD_agreement": _has_consecutive(same_sign, sign_count),
        "consecutive_relative_accuracy": _has_consecutive(accurate, accuracy_count),
        "preferred_accuracy_observed": preferred_pass,
        "all_perturbed_solutions_stationary": all(stationary),
        "all_perturbed_diagnostics_valid": all(diagnostics),
        "optimized_value_continuity": continuity["passed"],
    }
    passed = bool(
        rules["consecutive_fd_sign_and_AD_agreement"]
        and rules["consecutive_relative_accuracy"]
        and rules["all_perturbed_solutions_stationary"]
        and rules["all_perturbed_diagnostics_valid"]
        and rules["optimized_value_continuity"]
        and (preferred_pass or not required_preferred)
    )
    stable_errors = [
        max(rows[index]["relative_error_V"], rows[index + 1]["relative_error_V"])
        for index in range(len(rows) - 1)
        if accurate[index] and accurate[index + 1]
    ]
    return {
        "passed": passed,
        "rules": rules,
        "ad_sign": ad_sign,
        "fd_signs": fd_signs,
        "best_stable_two_point_relative_error": min(stable_errors) if stable_errors else None,
        "common_epsilon": common_epsilon,
        "common_epsilon_relative_error": (
            float(common["relative_error_V"]) if common is not None else None
        ),
        "continuity": continuity,
        "thresholds": {
            "consecutive_sign_count": sign_count,
            "consecutive_accuracy_count": accuracy_count,
            "relative_error_tolerance": relative_tolerance,
            "preferred_relative_error": preferred,
            "require_preferred_accuracy": required_preferred,
        },
    }


def _stage_two(rows: list[dict[str, Any]], ad_directional: float) -> dict[str, Any]:
    relative_errors: list[float] = []
    for row in rows:
        epsilon = float(row["epsilon"])
        fd_action = (
            float(row["plus_metrics"]["kinetic_action_A"])
            - float(row["minus_metrics"]["kinetic_action_A"])
        ) / (2.0 * epsilon)
        absolute = abs(fd_action - ad_directional)
        relative = absolute / max(abs(fd_action), abs(ad_directional), 1.0e-12)
        row["fd_kinetic_action"] = fd_action
        row["absolute_error_A"] = absolute
        row["relative_error_A"] = relative
        relative_errors.append(relative)
    return {
        "enabled": True,
        "best_relative_error": min(relative_errors),
        "relative_errors": relative_errors,
    }


def run_rigorous_gradient_check(
    cfg: dict[str, Any],
    data: PreparedExperiment,
    output_dir: Path,
) -> dict[str, Any]:
    """Run both center solves and the five-direction optimized-value check."""

    output_dir = require_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_cfg = _gradient_validation_config(cfg)
    eta0 = wrap_periodic(
        jnp.asarray(cfg["envelope"]["eta0"], dtype=jnp.float64),
        data.selection_problem.family,
    )

    warm_params, warm_source = _load_local_warm_start()
    warm_solution = None
    if warm_params is not None:
        warm_solution = solve_to_stationarity(
            eta0,
            cfg,
            data,
            initial_params=warm_params,
            initialize=True,
            maximum_rounds=int(validation_cfg["center_polish_rounds"]),
        )
    fresh_solution = solve_to_stationarity(
        eta0,
        cfg,
        data,
        initial_params=None,
        initialize=True,
        maximum_rounds=int(validation_cfg["center_polish_rounds"]),
    )
    selected_label, center = _choose_center(warm_solution, fresh_solution)
    basin = _basin_payload(warm_solution, fresh_solution, cfg)
    envelope_value, gradient, envelope_diagnostics = envelope_full_value_and_grad(
        eta0, center.params, data.selection_problem, data.ritz_train_bank
    )
    determinism = _determinism_payload(eta0, center.params, cfg, data)
    center_payload = {
        "eta0": jax.device_get(eta0).tolist(),
        "selected_center": selected_label,
        "warm_start_source": warm_source,
        "warm": (
            {"metrics": warm_solution.metrics, "rounds": warm_solution.rounds}
            if warm_solution is not None else None
        ),
        "fresh": {"metrics": fresh_solution.metrics, "rounds": fresh_solution.rounds},
        "selected_metrics": center.metrics,
        "selected_rounds": center.rounds,
        "basin_sensitivity": basin,
        "envelope_value": float(envelope_value),
        "envelope_gradient": jax.device_get(gradient).tolist(),
        "envelope_diagnostics": {
            name: float(value)
            for name, value in zip(
                envelope_diagnostics._fields, envelope_diagnostics, strict=True
            )
        },
        "determinism": determinism,
    }
    write_json(output_dir / "center.json", center_payload)

    center_rules = {
        "stationary": bool(center.metrics["stationary"]),
        "diagnostics_valid": bool(center.metrics["diagnostics_valid"]),
        "deterministic": bool(determinism["passed"]),
    }
    if not all(center_rules.values()):
        summary = {
            "mode": "rigorous-gradient-check",
            "conclusion": "ENVELOPE GRADIENT NOT YET VALIDATED",
            "passed": False,
            "stage_1_optimized_value_passed": False,
            "stage_2_kinetic_action_enabled": False,
            "eta0": jax.device_get(eta0).tolist(),
            "envelope_gradient": jax.device_get(gradient).tolist(),
            "center_rules": center_rules,
            "center": center_payload,
            "direction_count": 0,
            "direction_attempts": [],
            "directions": [],
            "strict_thresholds": validation_cfg,
            "no_outer_eta_updates": True,
            "directional_checks_not_run_reason": (
                "mandatory center stationarity/determinism/diagnostic prerequisite failed"
            ),
            "dominant_numerical_issue": (
                "stronger full-bank theta minimization lowers empirical J while parameter "
                "gradients and A-versus-minus-2J error remain large, so no locally tracked "
                "stationary Ritz branch has been established"
            ),
        }
        write_json(output_dir / "summary.json", summary)
        return summary

    directions, direction_attempts = deterministic_directions(eta0, cfg, data)
    epsilon_ladder = [float(value) for value in validation_cfg["epsilon_ladder"]]
    direction_payloads: list[dict[str, Any]] = []
    for direction_index, direction in enumerate(directions):
        ad_directional = float(jnp.vdot(gradient, direction))
        rows: list[dict[str, Any]] = []
        for epsilon in epsilon_ladder:
            plus_eta = wrap_periodic(
                eta0 + epsilon * direction, data.selection_problem.family
            )
            minus_eta = wrap_periodic(
                eta0 - epsilon * direction, data.selection_problem.family
            )
            # Both sides start from exactly the same selected center parameters.
            plus = solve_to_stationarity(
                plus_eta,
                cfg,
                data,
                initial_params=center.params,
                initialize=False,
                maximum_rounds=int(validation_cfg["perturbation_polish_rounds"]),
            )
            minus = solve_to_stationarity(
                minus_eta,
                cfg,
                data,
                initial_params=center.params,
                initialize=False,
                maximum_rounds=int(validation_cfg["perturbation_polish_rounds"]),
            )
            plus_value = float(plus.metrics["optimized_value_V_minus_2J"])
            minus_value = float(minus.metrics["optimized_value_V_minus_2J"])
            center_value = float(center.metrics["optimized_value_V_minus_2J"])
            fd_value = (plus_value - minus_value) / (2.0 * epsilon)
            absolute = abs(fd_value - ad_directional)
            relative = absolute / max(abs(fd_value), abs(ad_directional), 1.0e-12)
            rows.append({
                "epsilon": epsilon,
                "ad_directional": ad_directional,
                "fd_optimized_value": fd_value,
                "absolute_error_V": absolute,
                "relative_error_V": relative,
                "V0": center_value,
                "Vplus": plus_value,
                "Vminus": minus_value,
                "continuity_plus_error": abs(plus_value - center_value),
                "continuity_minus_error": abs(minus_value - center_value),
                "Aplus": float(plus.metrics["kinetic_action_A"]),
                "Aminus": float(minus.metrics["kinetic_action_A"]),
                "fd_kinetic_action": None,
                "absolute_error_A": None,
                "relative_error_A": None,
                "plus_metrics": plus.metrics,
                "minus_metrics": minus.metrics,
                "plus_polish_rounds": plus.rounds,
                "minus_polish_rounds": minus.rounds,
            })
        summary = _direction_summary(rows, ad_directional, cfg)
        direction_payloads.append({
            "direction_index": direction_index,
            "direction": jax.device_get(direction).tolist(),
            "ad_directional": ad_directional,
            "rows": rows,
            "stage_1_summary": summary,
            "stage_2_summary": {"enabled": False, "reason": "awaiting global Stage-1 gate"},
        })

    center_rules = {
        **center_rules,
        "requested_direction_count_obtained": len(directions)
        == int(validation_cfg["direction_count"]),
    }
    all_direction_stage_one = bool(
        len(directions) == int(validation_cfg["direction_count"])
        and all(payload["stage_1_summary"]["passed"] for payload in direction_payloads)
    )
    stage_one_passed = bool(all(center_rules.values()) and all_direction_stage_one)
    if stage_one_passed:
        for payload in direction_payloads:
            payload["stage_2_summary"] = _stage_two(
                payload["rows"], float(payload["ad_directional"])
            )
    for payload in direction_payloads:
        write_json(
            output_dir / f"direction_{int(payload['direction_index']):02d}.json",
            payload,
        )

    summary = {
        "mode": "rigorous-gradient-check",
        "conclusion": (
            "ENVELOPE GRADIENT NUMERICALLY VALIDATED"
            if stage_one_passed
            else "ENVELOPE GRADIENT NOT YET VALIDATED"
        ),
        "passed": stage_one_passed,
        "stage_1_optimized_value_passed": stage_one_passed,
        "stage_2_kinetic_action_enabled": stage_one_passed,
        "eta0": jax.device_get(eta0).tolist(),
        "envelope_gradient": jax.device_get(gradient).tolist(),
        "center_rules": center_rules,
        "center": center_payload,
        "direction_count": len(directions),
        "direction_attempts": direction_attempts,
        "directions": [{
            "direction_index": payload["direction_index"],
            "direction": payload["direction"],
            "ad_directional": payload["ad_directional"],
            "stage_1_summary": payload["stage_1_summary"],
            "stage_2_summary": payload["stage_2_summary"],
        } for payload in direction_payloads],
        "strict_thresholds": validation_cfg,
        "no_outer_eta_updates": True,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


__all__ = [
    "run_rigorous_gradient_check",
    "solve_to_stationarity",
    "stationary_metrics",
]
