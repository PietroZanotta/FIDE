"""Production Galerkin envelope derivative and strict multi-direction audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .full_gradient import (
    forcing_state,
    periodic_branch_distance,
    reconstruct_moments,
    wrap_periodic,
)
from .galerkin import aggregate_quadratic_values, rank_aware_quadratic_solve
from .production_artifacts import require_production_output_path
from .production_basis import load_dictionary
from .production_galerkin import assemble_hybrid_system, make_basis_evaluators
from .production_workflow import load_production_data
from .workflow import PreparedExperiment, write_json

Array = jax.Array


def precompute_fixed_potential_rows(
    dictionary,
    coefficients: Array,
    data: PreparedExperiment,
    evaluators: list[Any],
    *,
    chunk_size: int,
) -> tuple[Array, Array]:
    bank = data.ritz_train_bank
    time_count, sample_count = bank.configurations.shape[:2]
    potentials = []
    kinetic_rows = []
    for time_index in range(int(time_count)):
        values_by_chunk = []
        kinetic_by_chunk = []
        for start in range(0, int(sample_count), int(chunk_size)):
            stop = min(start + int(chunk_size), int(sample_count))
            values, gradients = evaluators[time_index](
                bank.configurations[time_index, start:stop]
            )
            potential = jnp.einsum("k,nk->n", coefficients[time_index], values)
            gradient = jnp.einsum(
                "k,nkpd->npd", coefficients[time_index], gradients
            )
            values_by_chunk.append(potential)
            kinetic_by_chunk.append(jnp.sum(gradient * gradient, axis=(-2, -1)))
        potentials.append(jnp.concatenate(values_by_chunk))
        kinetic_rows.append(jnp.concatenate(kinetic_by_chunk))
    return jnp.stack(potentials), jnp.stack(kinetic_rows)


def production_hybrid_envelope_value_and_grad(
    eta: Array,
    coefficients_fixed: Array,
    data: PreparedExperiment,
    potential_rows: Array,
    kinetic_rows: Array,
) -> tuple[Array, Array]:
    """Differentiate explicit eta dependence only; no coefficient solve is traced."""

    problem = data.selection_problem
    bank = data.ritz_train_bank

    def value(design: Array) -> Array:
        reconstruction = reconstruct_moments(design, problem)
        state = forcing_state(design, problem, bank, reconstruction)
        weights = state.projection.weights
        forcing = state.forcing
        kinetic = jnp.einsum("tn,tn->t", weights, kinetic_rows)
        potential_mean = jnp.einsum("tn,tn->t", weights, potential_rows)
        forcing_mean = jnp.einsum("tn,tn->t", weights, forcing)
        linear = jnp.einsum("tn,tn,tn->t", weights, forcing, potential_rows)
        linear = linear - forcing_mean * potential_mean
        objective = 0.5 * kinetic + linear
        return -2.0 * jnp.sum(problem.time_weights * objective)

    return jax.value_and_grad(value)(jnp.asarray(eta, dtype=jnp.float64))


def _forcing_state_payload(state, problem) -> dict[str, Any]:
    maximum_projection = float(jnp.max(jnp.linalg.norm(
        state.projection.residual, axis=-1
    )))
    minimum_ess = float(jnp.min(state.projection.ess_fraction))
    maximum_mean = float(jnp.max(jnp.abs(state.forcing_mean_before_centering)))
    maximum_condition = float(jnp.max(state.covariance_condition))
    post_mean = float(jnp.max(jnp.abs(jnp.einsum(
        "tn,tn->t", state.projection.weights, state.forcing
    ))))
    cfg = problem.forcing_config
    valid = bool(
        maximum_projection <= cfg.projection_tolerance
        and minimum_ess >= cfg.minimum_ess_fraction
        and maximum_mean <= cfg.forcing_mean_tolerance
        and maximum_condition <= cfg.max_covariance_condition
    )
    return {
        "valid": valid,
        "maximum_projection_residual": maximum_projection,
        "minimum_ess_fraction": minimum_ess,
        "maximum_forcing_mean": maximum_mean,
        "maximum_post_centering_forcing_mean": post_mean,
        "maximum_covariance_condition": maximum_condition,
    }


def evaluate_local_eta(
    eta: Array,
    cfg: dict[str, Any],
    data: PreparedExperiment,
    dictionary,
    evaluators: list[Any],
) -> tuple[dict[str, Any], Any]:
    settings = cfg["production_galerkin"]
    eta = wrap_periodic(eta, data.selection_problem.family)
    reconstruction = reconstruct_moments(eta, data.selection_problem)
    train_state = forcing_state(
        eta, data.selection_problem, data.ritz_train_bank, reconstruction
    )
    audit_state = forcing_state(
        eta, data.selection_problem, data.ritz_audit_bank, reconstruction
    )
    system = assemble_hybrid_system(
        dictionary,
        data.ritz_train_bank,
        train_state.projection.weights,
        train_state.forcing,
        chunk_size=int(settings["chunk_size"]),
        evaluators=evaluators,
    )
    solve = rank_aware_quadratic_solve(
        system.gram, system.load,
        relative_rank_tolerance=float(settings["relative_rank_tolerance"]),
    )
    aggregate = aggregate_quadratic_values(
        solve, data.selection_problem.time_weights
    )
    train_audit = _forcing_state_payload(train_state, data.selection_problem)
    heldout_audit = _forcing_state_payload(audit_state, data.selection_problem)
    algebra_valid = bool(
        float(aggregate["identity_relerr"]) <= float(settings["maximum_identity_relerr"])
        and float(jnp.max(solve.stationarity_residual)) <= float(settings["maximum_stationarity_residual"])
        and float(jnp.max(solve.range_residual)) <= float(settings["maximum_range_residual"])
        and float(jnp.max(system.raw_symmetry_residual)) <= float(settings["maximum_symmetry_residual"])
        and float(jnp.max(solve.condition_number)) <= float(settings["maximum_retained_condition"])
    )
    geometry = bool(jax.device_get(
        data.selection_problem.family.geometry_valid(eta)
    ))
    payload = {
        "eta": jax.device_get(eta).tolist(),
        "action": float(aggregate["action"]),
        "objective": float(aggregate["objective"]),
        "identity_relerr": float(aggregate["identity_relerr"]),
        "rank_by_time": jax.device_get(solve.numerical_rank).tolist(),
        "worst_range_residual": float(jnp.max(solve.range_residual)),
        "worst_stationarity_residual": float(jnp.max(solve.stationarity_residual)),
        "worst_retained_condition": float(jnp.max(solve.condition_number)),
        "worst_symmetry_residual": float(jnp.max(system.raw_symmetry_residual)),
        "train_forcing_audit": train_audit,
        "heldout_forcing_audit": heldout_audit,
        "geometry_valid": geometry,
        "algebra_valid": algebra_valid,
        "hard_gates_passed": bool(
            algebra_valid and geometry and train_audit["valid"] and heldout_audit["valid"]
        ),
    }
    return payload, solve


def _directions(cfg: dict[str, Any], data: PreparedExperiment) -> list[Array]:
    settings = cfg["production_galerkin"]["gradient"]
    eta = jnp.asarray(cfg["envelope"]["eta0"], dtype=jnp.float64)
    epsilon = max(float(value) for value in settings["epsilon_ladder"])
    key = jax.random.PRNGKey(int(settings["direction_seed"]))
    directions = []
    for _ in range(int(settings["maximum_direction_attempts"])):
        key, subkey = jax.random.split(key)
        direction = jax.random.normal(subkey, eta.shape, dtype=jnp.float64)
        direction = direction / jnp.linalg.norm(direction)
        plus = wrap_periodic(eta + epsilon * direction, data.selection_problem.family)
        minus = wrap_periodic(eta - epsilon * direction, data.selection_problem.family)
        geometry = bool(jax.device_get(
            data.selection_problem.family.geometry_valid(plus)
            & data.selection_problem.family.geometry_valid(minus)
        ))
        branch = min(
            float(periodic_branch_distance(plus, data.selection_problem.family)),
            float(periodic_branch_distance(minus, data.selection_problem.family)),
        )
        if geometry and branch >= float(settings["minimum_branch_distance"]):
            directions.append(direction)
        if len(directions) == int(settings["direction_count"]):
            return directions
    raise RuntimeError("could not construct deterministic production directions")


def _consecutive(values: list[bool], count: int) -> bool:
    return any(
        all(values[start:start + count])
        for start in range(max(0, len(values) - count + 1))
    )


def _direction_gate(rows: list[dict[str, Any]], ad: float, settings) -> dict[str, Any]:
    accepted = [row for row in rows if row["accepted"]]
    signs = [row["fd"] * ad > 0.0 for row in accepted]
    accurate = [
        row["relative_discrepancy"] <= float(settings["relative_error_tolerance"])
        for row in accepted
    ]
    errors = [row["absolute_discrepancy"] for row in accepted]
    decreasing = any(
        errors[index + 1] < errors[index]
        and errors[index + 2] < errors[index + 1]
        for index in range(max(0, len(errors) - 2))
    )
    sign_passed = _consecutive(signs, int(settings["consecutive_sign_count"]))
    accuracy_passed = _consecutive(
        accurate, int(settings["consecutive_accuracy_count"])
    )
    return {
        "passed": bool(sign_passed and accuracy_passed and decreasing),
        "accepted_epsilon_count": len(accepted),
        "consecutive_sign_passed": sign_passed,
        "consecutive_accuracy_passed": accuracy_passed,
        "decreasing_error_regime_observed": decreasing,
        "preferred_accuracy_observed": any(
            row["relative_discrepancy"] <= float(settings["preferred_relative_error"])
            for row in accepted
        ),
    }


def run_production_gradient_check(
    cfg: dict[str, Any], artifact_dir: Path, output_dir: Path
) -> dict[str, Any]:
    output_dir = require_production_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    convergence_path = output_dir.parent / "convergence" / "result.json"
    if not convergence_path.is_file():
        raise RuntimeError("production convergence result is missing")
    convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
    if not convergence.get("basis_convergence_passed"):
        result = {
            "passed": False, "skipped": True,
            "reason": "Gate B failed",
            "outcome_classification": "C. GALERKIN BASIS NOT YET PHYSICALLY ADEQUATE",
        }
        write_json(output_dir / "result.json", result)
        return result
    data = load_production_data(cfg, artifact_dir)
    feature_path = output_dir.parent / "convergence" / "features" / "hybrid_dictionary.npz"
    dictionary = load_dictionary(feature_path, box=tuple(cfg["physics"]["box"]))
    size = int(cfg["production_galerkin"]["basis_size_ladder"][-1])
    with np.load(output_dir.parent / "convergence" / f"K{size}.npz", allow_pickle=False) as arrays:
        coefficients = jnp.asarray(arrays["coefficients"], dtype=jnp.float64)
    evaluators = make_basis_evaluators(
        dictionary, int(data.selection_problem.times.shape[0])
    )
    potential_rows, kinetic_rows = precompute_fixed_potential_rows(
        dictionary, coefficients, data, evaluators,
        chunk_size=int(cfg["production_galerkin"]["chunk_size"]),
    )
    eta = jnp.asarray(cfg["envelope"]["eta0"], dtype=jnp.float64)
    value, gradient = production_hybrid_envelope_value_and_grad(
        eta, coefficients, data, potential_rows, kinetic_rows
    )
    repeated_value, repeated_gradient = production_hybrid_envelope_value_and_grad(
        eta, coefficients, data, potential_rows, kinetic_rows
    )
    settings = cfg["production_galerkin"]["gradient"]
    deterministic = bool(
        abs(float(value - repeated_value))
        <= float(settings["determinism_absolute_tolerance"])
        and float(jnp.max(jnp.abs(gradient - repeated_gradient)))
        <= float(settings["determinism_absolute_tolerance"])
    )
    center = convergence["basis_sizes"][-1]
    center_rank = center["numerical_rank_by_time"]
    directions_payload = []
    partial = {
        "passed": False, "skipped": False, "in_progress": True,
        "eta_gradient": jax.device_get(gradient).tolist(), "directions": [],
    }
    write_json(output_dir / "result.json", partial)
    for index, direction in enumerate(_directions(cfg, data)):
        ad = float(jnp.vdot(gradient, direction))
        rows = []
        for epsilon in settings["epsilon_ladder"]:
            plus_eta = wrap_periodic(
                eta + float(epsilon) * direction, data.selection_problem.family
            )
            minus_eta = wrap_periodic(
                eta - float(epsilon) * direction, data.selection_problem.family
            )
            plus, plus_solve = evaluate_local_eta(
                plus_eta, cfg, data, dictionary, evaluators
            )
            minus, minus_solve = evaluate_local_eta(
                minus_eta, cfg, data, dictionary, evaluators
            )
            fd = (plus["action"] - minus["action"]) / (2.0 * float(epsilon))
            absolute = abs(fd - ad)
            relative = absolute / max(abs(fd), abs(ad), 1.0e-12)
            ranks_stable = bool(
                plus["rank_by_time"] == center_rank
                and minus["rank_by_time"] == center_rank
            )
            accepted = bool(
                ranks_stable
                and plus["hard_gates_passed"]
                and minus["hard_gates_passed"]
            )
            rows.append({
                "epsilon": float(epsilon), "fd": fd,
                "absolute_discrepancy": absolute,
                "relative_discrepancy": relative,
                "rank_center": center_rank,
                "rank_plus": plus["rank_by_time"],
                "rank_minus": minus["rank_by_time"],
                "rank_stable": ranks_stable,
                "accepted": accepted,
                "plus": plus, "minus": minus,
            })
        direction_result = {
            "index": index,
            "direction": jax.device_get(direction).tolist(),
            "ad_directional_derivative": ad,
            "rows": rows,
            **_direction_gate(rows, ad, settings),
        }
        directions_payload.append(direction_result)
        partial["directions"] = directions_payload
        write_json(output_dir / "result.json", partial)
    passed_count = sum(bool(row["passed"]) for row in directions_payload)
    passed = bool(
        deterministic
        and bool(jnp.all(jnp.isfinite(gradient)))
        and passed_count >= int(settings["required_passed_directions"])
    )
    result = {
        "passed": passed,
        "skipped": False,
        "in_progress": False,
        "eta0": jax.device_get(eta).tolist(),
        "basis_size": size,
        "center_action": center["galerkin_action"],
        "envelope_value": float(value),
        "eta_gradient": jax.device_get(gradient).tolist(),
        "gradient_finite": bool(jnp.all(jnp.isfinite(gradient))),
        "deterministic": deterministic,
        "passed_direction_count": passed_count,
        "directions": directions_payload,
        "outcome_classification": (
            "A. PRODUCTION GALERKIN SOLVER AND ETA GRADIENT VALIDATED"
            if passed
            else "B. PRODUCTION GALERKIN SOLVER VALID, ETA GRADIENT NOT YET VALIDATED"
        ),
    }
    write_json(output_dir / "result.json", result)
    return result


__all__ = [
    "evaluate_local_eta", "precompute_fixed_potential_rows",
    "production_hybrid_envelope_value_and_grad", "run_production_gradient_check",
]
