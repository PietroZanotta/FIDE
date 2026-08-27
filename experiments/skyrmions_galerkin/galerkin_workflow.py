"""Evaluation workflow for the fixed-basis Galerkin route."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .full_gradient import (
    minimum_sensor_separation,
    periodic_branch_distance,
    smooth_separation_penalty,
    wrap_periodic,
)
from .galerkin import (
    FrozenDeepSetsBasis,
    basis_size_ladder,
    evaluate_basis,
    evaluate_fixed_eta,
    galerkin_envelope_value_and_grad,
    load_basis_families,
    save_basis_checkpoint,
    save_galerkin_arrays,
)
from .workflow import (
    OUTPUT_ROOT,
    PreparedExperiment,
    hard_forcing_audit,
    law_risk_anchor,
    require_output_path,
    selection_risk,
    write_json,
)

Array = jax.Array


def _feature_checks(
    basis: FrozenDeepSetsBasis,
    data: PreparedExperiment,
    basis_size: int,
) -> dict[str, Any]:
    problem = data.selection_problem
    x = data.ritz_train_bank.configurations[:1, :4]
    times = problem.times[:1]
    original = evaluate_basis(basis, x, times, basis_size)
    permutation = jax.random.permutation(
        jax.random.PRNGKey(20261119), x.shape[-2]
    )
    permuted = evaluate_basis(basis, x[..., permutation, :], times, basis_size)
    repeated = evaluate_basis(basis, x, times, basis_size)
    invariance_error = float(jnp.max(jnp.abs(original.values - permuted.values)))
    determinism_error = float(jnp.max(jnp.abs(original.values - repeated.values)))
    return {
        "permutation_invariance_max_abs": invariance_error,
        "permutation_invariance_passed": invariance_error <= 1.0e-12,
        "determinism_max_abs": determinism_error,
        "determinism_passed": determinism_error == 0.0,
        "state_gradients_finite": bool(jnp.all(jnp.isfinite(original.state_gradients))),
        "basis_has_eta_argument": False,
        "basis_parameters_frozen": True,
    }


def _convergence_gate(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    if len(rows) < 2:
        return {
            "passed": False,
            "action_stable": False,
            "physical_diagnostics_controlled": False,
            "reason": "fewer than two usable basis sizes",
        }
    previous, current = rows[-2], rows[-1]
    denominator = max(abs(float(current["galerkin_action"])), 1.0e-12)
    action_change = abs(
        float(current["galerkin_action"]) - float(previous["galerkin_action"])
    ) / denominator
    previous_weak = float(previous["held_out_certificate"]["maximum_weak_residual"])
    current_weak = float(current["held_out_certificate"]["maximum_weak_residual"])
    action_stable = action_change <= float(
        cfg["galerkin"]["action_stability_relative_tolerance"]
    )
    weak_controlled = current_weak <= float(
        cfg["galerkin"]["weak_residual_growth_tolerance"]
    ) * max(previous_weak, 1.0e-12)
    algebra = all(bool(row["algebra_valid"]) for row in rows)
    physical = bool(current["physical_valid"] and weak_controlled)
    passed = bool(algebra and action_stable and physical)
    return {
        "passed": passed,
        "all_quadratic_systems_valid": algebra,
        "action_stable": action_stable,
        "last_step_relative_action_change": action_change,
        "physical_diagnostics_controlled": physical,
        "last_step_weak_residual_controlled": weak_controlled,
    }


def _matching_deep_ritz_comparison() -> dict[str, Any]:
    path = OUTPUT_ROOT / "gradient_checks" / "smoke" / "result.json"
    if not path.is_file():
        return {"available": False, "scope": "NOT ESTABLISHED"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    diagnostics = payload.get("center_envelope_diagnostics", {})
    return {
        "available": True,
        "scope": "smoke/local diagnostic comparison only",
        "source": str(path),
        "deep_ritz_action": diagnostics.get("full_energy"),
        "deep_ritz_objective": diagnostics.get("ritz_objective"),
        "deep_ritz_negative_twice_objective": payload.get("envelope_value"),
        "deep_ritz_energy_identity_relerr": diagnostics.get("energy_identity_relerr"),
        "warning": "Numeric comparison is local and is not commensurate with a published production action.",
    }


def run_galerkin_fixed(
    cfg: dict[str, Any], data: PreparedExperiment, output_dir: Path
) -> dict[str, Any]:
    output_dir = require_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eta = jnp.asarray(cfg["envelope"]["eta0"], dtype=jnp.float64)
    families = load_basis_families(cfg)
    basis = families[0]
    size = basis_size_ladder(basis, cfg)[-1]
    started = time.perf_counter()
    payload, solve, evaluation = evaluate_fixed_eta(eta, cfg, data, basis, size)
    payload["runtime_seconds"] = time.perf_counter() - started
    payload["feature_checks"] = _feature_checks(basis, data, size)
    payload["classification"] = (
        "FIXED GALERKIN BASIS PHYSICALLY ACCEPTABLE"
        if payload["physical_valid"]
        else "FIXED GALERKIN BASIS NOT PHYSICALLY ACCEPTABLE"
    )
    save_basis_checkpoint(output_dir / "basis.npz", basis)
    save_galerkin_arrays(output_dir / "solve.npz", solve, evaluation)
    write_json(output_dir / "result.json", payload)
    return payload


def run_galerkin_convergence(
    cfg: dict[str, Any], data: PreparedExperiment, output_dir: Path
) -> dict[str, Any]:
    output_dir = require_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eta = jnp.asarray(cfg["envelope"]["eta0"], dtype=jnp.float64)
    family_results: list[dict[str, Any]] = []
    for family in load_basis_families(cfg):
        family_dir = output_dir / family.name
        family_dir.mkdir(parents=True, exist_ok=True)
        save_basis_checkpoint(family_dir / "frozen_basis.npz", family)
        rows: list[dict[str, Any]] = []
        for size in basis_size_ladder(family, cfg):
            started = time.perf_counter()
            train_basis = evaluate_basis(
                family, data.ritz_train_bank.configurations,
                data.selection_problem.times, size,
            )
            audit_basis = evaluate_basis(
                family, data.ritz_audit_bank.configurations,
                data.selection_problem.times, size,
            )
            payload, solve, _ = evaluate_fixed_eta(
                eta, cfg, data, family, size,
                train_basis=train_basis, audit_basis=audit_basis,
            )
            payload["runtime_seconds"] = time.perf_counter() - started
            write_json(family_dir / f"K{size}.json", payload)
            save_galerkin_arrays(family_dir / f"K{size}.npz", solve, train_basis)
            rows.append(payload)
        family_results.append({
            "basis_family": family.name,
            "basis_source": family.source,
            "basis_source_sha256": family.source_sha256,
            "feature_checks": _feature_checks(family, data, rows[-1]["basis_size"]),
            "basis_sizes": rows,
            "convergence_gate": _convergence_gate(rows, cfg),
        })
    primary = family_results[0]
    crosscheck = {"available": len(family_results) > 1}
    if len(family_results) > 1:
        first = float(primary["basis_sizes"][-1]["galerkin_action"])
        second = float(family_results[1]["basis_sizes"][-1]["galerkin_action"])
        crosscheck.update({
            "primary_action": first,
            "control_action": second,
            "relative_action_difference": abs(first - second) / max(abs(first), 1.0e-12),
        })
    passed = bool(primary["convergence_gate"]["passed"])
    result = {
        "profile": cfg.get("execution_profile"),
        "eta0": jax.device_get(eta).tolist(),
        "primary_basis_family": primary["basis_family"],
        "families": family_results,
        "feature_family_crosscheck": crosscheck,
        "matching_nonlinear_deep_ritz": _matching_deep_ritz_comparison(),
        "basis_convergence_passed": passed,
        "outcome_classification": (
            "B. GALERKIN SOLVER VALID, ETA GRADIENT NOT YET VALIDATED"
            if passed
            else "C. GALERKIN BASIS NOT YET PHYSICALLY ADEQUATE"
        ),
    }
    write_json(output_dir / "result.json", result)
    return result


def _consecutive_true(values: list[bool], count: int) -> bool:
    return any(all(values[start:start + count]) for start in range(len(values) - count + 1))


def _direction_pass(rows: list[dict[str, Any]], ad: float, cfg: dict[str, Any]) -> dict[str, Any]:
    settings = cfg["galerkin"]["gradient"]
    signs = [bool(row["fd"] * ad > 0.0) for row in rows]
    accurate = [
        bool(row["relative_discrepancy"] <= float(settings["relative_error_tolerance"]))
        for row in rows
    ]
    errors = [float(row["absolute_discrepancy"]) for row in rows]
    decreasing_regime = any(
        errors[index + 1] < errors[index] and errors[index + 2] < errors[index + 1]
        for index in range(max(0, len(errors) - 2))
    )
    stable = all(bool(row["rank_stable"] and row["hard_gates_passed"]) for row in rows)
    sign_pass = _consecutive_true(signs, int(settings["consecutive_sign_count"]))
    accuracy_pass = _consecutive_true(accurate, int(settings["consecutive_accuracy_count"]))
    preferred = any(
        float(row["relative_discrepancy"]) <= float(settings["preferred_relative_error"])
        for row in rows
    )
    return {
        "passed": bool(stable and sign_pass and accuracy_pass and decreasing_regime),
        "rank_and_hard_gates_passed": stable,
        "consecutive_sign_passed": sign_pass,
        "consecutive_accuracy_passed": accuracy_pass,
        "preferred_accuracy_observed": preferred,
        "decreasing_error_regime_observed": decreasing_regime,
    }


def _admissible_directions(cfg: dict[str, Any], data: PreparedExperiment) -> list[Array]:
    settings = cfg["galerkin"]["gradient"]
    eta = jnp.asarray(cfg["envelope"]["eta0"], dtype=jnp.float64)
    epsilon = max(float(value) for value in settings["epsilon_ladder"])
    directions: list[Array] = []
    key = jax.random.PRNGKey(int(settings["direction_seed"]))
    for _ in range(int(settings["maximum_direction_attempts"])):
        key, subkey = jax.random.split(key)
        direction = jax.random.normal(subkey, eta.shape, dtype=jnp.float64)
        direction = direction / jnp.linalg.norm(direction)
        plus = wrap_periodic(eta + epsilon * direction, data.selection_problem.family)
        minus = wrap_periodic(eta - epsilon * direction, data.selection_problem.family)
        admissible = bool(jax.device_get(
            data.selection_problem.family.geometry_valid(plus)
            & data.selection_problem.family.geometry_valid(minus)
        ))
        branch = min(
            float(periodic_branch_distance(plus, data.selection_problem.family)),
            float(periodic_branch_distance(minus, data.selection_problem.family)),
        )
        if admissible and branch >= float(settings["minimum_branch_distance"]):
            directions.append(direction)
        if len(directions) == int(settings["direction_count"]):
            return directions
    raise RuntimeError("could not construct five deterministic admissible directions")


def run_galerkin_gradient_check(
    cfg: dict[str, Any], data: PreparedExperiment, output_dir: Path
) -> dict[str, Any]:
    output_dir = require_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    convergence = run_galerkin_convergence(cfg, data, output_dir / "convergence")
    if not convergence["basis_convergence_passed"]:
        result = {
            "passed": False,
            "skipped": True,
            "reason": "basis convergence or held-out physical gate failed",
            "eta_gradient": None,
            "directions": [],
            "outcome_classification": "C. GALERKIN BASIS NOT YET PHYSICALLY ADEQUATE",
        }
        write_json(output_dir / "result.json", result)
        return result
    primary = load_basis_families(cfg)[0]
    size = basis_size_ladder(primary, cfg)[-1]
    eta = jnp.asarray(cfg["envelope"]["eta0"], dtype=jnp.float64)
    basis_eval = evaluate_basis(
        primary, data.ritz_train_bank.configurations,
        data.selection_problem.times, size,
    )
    center, center_solve, _ = evaluate_fixed_eta(
        eta, cfg, data, primary, size, train_basis=basis_eval
    )
    value, gradient = galerkin_envelope_value_and_grad(
        eta, center_solve.coefficients, data.selection_problem,
        data.ritz_train_bank, basis_eval,
    )
    value_repeat, gradient_repeat = galerkin_envelope_value_and_grad(
        eta, center_solve.coefficients, data.selection_problem,
        data.ritz_train_bank, basis_eval,
    )
    tolerance = float(cfg["galerkin"]["gradient"]["determinism_absolute_tolerance"])
    deterministic = bool(
        float(jnp.max(jnp.abs(gradient - gradient_repeat))) <= tolerance
        and abs(float(value - value_repeat)) <= tolerance
    )
    directions_payload: list[dict[str, Any]] = []
    center_rank = center_solve.numerical_rank
    for index, direction in enumerate(_admissible_directions(cfg, data)):
        ad = float(jnp.vdot(gradient, direction))
        rows: list[dict[str, Any]] = []
        for epsilon in cfg["galerkin"]["gradient"]["epsilon_ladder"]:
            plus_eta = wrap_periodic(eta + float(epsilon) * direction, data.selection_problem.family)
            minus_eta = wrap_periodic(eta - float(epsilon) * direction, data.selection_problem.family)
            plus, plus_solve, _ = evaluate_fixed_eta(
                plus_eta, cfg, data, primary, size, train_basis=basis_eval
            )
            minus, minus_solve, _ = evaluate_fixed_eta(
                minus_eta, cfg, data, primary, size, train_basis=basis_eval
            )
            fd = (float(plus["galerkin_action"]) - float(minus["galerkin_action"])) / (
                2.0 * float(epsilon)
            )
            absolute = abs(fd - ad)
            relative = absolute / max(abs(fd), abs(ad), 1.0e-12)
            rank_stable = bool(
                jnp.array_equal(center_rank, plus_solve.numerical_rank)
                and jnp.array_equal(center_rank, minus_solve.numerical_rank)
            )
            hard = bool(plus["physical_valid"] and minus["physical_valid"])
            rows.append({
                "epsilon": float(epsilon), "fd": fd,
                "absolute_discrepancy": absolute,
                "relative_discrepancy": relative,
                "rank_center": jax.device_get(center_rank).tolist(),
                "rank_plus": jax.device_get(plus_solve.numerical_rank).tolist(),
                "rank_minus": jax.device_get(minus_solve.numerical_rank).tolist(),
                "rank_stable": rank_stable,
                "hard_gates_passed": hard,
                "plus_worst_range_residual": plus["worst_range_residual"],
                "minus_worst_range_residual": minus["worst_range_residual"],
                "plus_worst_condition_number": plus["worst_condition_number"],
                "minus_worst_condition_number": minus["worst_condition_number"],
                "plus_forcing_audit": plus["train_forcing_audit"],
                "minus_forcing_audit": minus["train_forcing_audit"],
            })
        gate = _direction_pass(rows, ad, cfg)
        directions_payload.append({
            "index": index,
            "direction": jax.device_get(direction).tolist(),
            "ad_directional_derivative": ad,
            "rows": rows,
            **gate,
        })
    passed_count = sum(bool(direction["passed"]) for direction in directions_payload)
    passed = bool(
        deterministic
        and passed_count >= int(cfg["galerkin"]["gradient"]["required_passed_directions"])
    )
    result = {
        "passed": passed,
        "skipped": False,
        "eta0": jax.device_get(eta).tolist(),
        "basis_family": primary.name,
        "basis_size": size,
        "center_action": center["galerkin_action"],
        "envelope_value": float(value),
        "eta_gradient": jax.device_get(gradient).tolist(),
        "gradient_finite": bool(jnp.all(jnp.isfinite(gradient))),
        "deterministic": deterministic,
        "passed_direction_count": passed_count,
        "directions": directions_payload,
        "outcome_classification": (
            "A. GALERKIN SOLVER AND ETA GRADIENT VALIDATED"
            if passed
            else "B. GALERKIN SOLVER VALID, ETA GRADIENT NOT YET VALIDATED"
        ),
    }
    write_json(output_dir / "result.json", result)
    return result


def run_galerkin_refinement(
    cfg: dict[str, Any], data: PreparedExperiment, output_dir: Path,
    *, allowance_percent: float,
) -> dict[str, Any]:
    output_dir = require_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gate = run_galerkin_gradient_check(cfg, data, output_dir / "prerequisite_gradient_check")
    if not gate["passed"]:
        result = {
            "ran": False,
            "reason": "strict Galerkin prerequisite gate did not pass",
            "outcome_classification": gate["outcome_classification"],
        }
        write_json(output_dir / "result.json", result)
        return result
    basis = load_basis_families(cfg)[0]
    size = basis_size_ladder(basis, cfg)[-1]
    basis_eval = evaluate_basis(
        basis, data.ritz_train_bank.configurations, data.selection_problem.times, size
    )
    eta = jnp.asarray(cfg["envelope"]["eta0"], dtype=jnp.float64)
    start_eta = eta
    start, _, _ = evaluate_fixed_eta(eta, cfg, data, basis, size, train_basis=basis_eval)
    law_risk = law_risk_anchor(cfg, data)
    risk_limit = law_risk * (1.0 + allowance_percent / 100.0)
    settings = cfg["galerkin"]["refinement"]
    history: list[dict[str, Any]] = []
    for step in range(int(settings["steps"])):
        current, solve, _ = evaluate_fixed_eta(
            eta, cfg, data, basis, size, train_basis=basis_eval
        )
        _, action_gradient = galerkin_envelope_value_and_grad(
            eta, solve.coefficients, data.selection_problem, data.ritz_train_bank, basis_eval
        )
        def penalties(design: Array) -> Array:
            risk_hinge = jax.nn.relu(selection_risk(design, data) / risk_limit - 1.0)
            return (
                float(settings["risk_penalty"]) * risk_hinge * risk_hinge
                + float(settings["separation_penalty"])
                * smooth_separation_penalty(design, data.selection_problem.family)
            )
        total_gradient = action_gradient + jax.grad(penalties)(eta)
        accepted = False
        for backtrack in range(int(settings["backtracking_steps"])):
            scale = float(settings["learning_rate"]) * (0.5 ** backtrack)
            proposal = wrap_periodic(eta - scale * total_gradient, data.selection_problem.family)
            if bool(jax.device_get(data.selection_problem.family.geometry_valid(proposal))):
                proposal_payload, _, _ = evaluate_fixed_eta(
                    proposal, cfg, data, basis, size, train_basis=basis_eval
                )
                if proposal_payload["galerkin_action"] <= current["galerkin_action"]:
                    eta = proposal
                    accepted = True
                    break
        history.append({
            "step": step, "accepted": accepted,
            "eta": jax.device_get(eta).tolist(),
            "action_before": current["galerkin_action"],
            "gradient_norm": float(jnp.linalg.norm(total_gradient)),
        })
        if not accepted:
            break
    end, _, _ = evaluate_fixed_eta(eta, cfg, data, basis, size, train_basis=basis_eval)
    result = {
        "ran": True,
        "authoritative_improvement_claimed": False,
        "start_eta": jax.device_get(start_eta).tolist(),
        "end_eta": jax.device_get(eta).tolist(),
        "start_galerkin_action": start["galerkin_action"],
        "end_galerkin_action": end["galerkin_action"],
        "end_selection_risk": float(selection_risk(eta, data)),
        "risk_ceiling": risk_limit,
        "end_minimum_separation": float(minimum_sensor_separation(eta, data.selection_problem.family)),
        "end_diagnostics": end,
        "history": history,
        "authoritative_crosscheck": "NOT ESTABLISHED",
        "outcome_classification": gate["outcome_classification"],
    }
    write_json(output_dir / "result.json", result)
    return result


__all__ = [
    "run_galerkin_convergence",
    "run_galerkin_fixed",
    "run_galerkin_gradient_check",
    "run_galerkin_refinement",
]
