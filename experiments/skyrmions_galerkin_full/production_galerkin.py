"""Chunked production-scale assembly and certification for the hybrid basis."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .deep_ritz import CertificateConfig, _audit_features
from .full_gradient import forcing_state, reconstruct_moments
from .galerkin import (
    GalerkinSystem,
    aggregate_quadratic_values,
    rank_aware_quadratic_solve,
)
from .production_artifacts import require_production_output_path
from .production_basis import (
    HybridInvariantDictionary,
    dictionary_metadata,
    fit_frozen_normalization,
    make_hybrid_dictionary,
    raw_values_and_gradients,
    save_dictionary,
)
from .production_workflow import run_production_reproduction
from .workflow import PreparedExperiment, write_json

Array = jax.Array


def _normalized_chunk(
    dictionary: HybridInvariantDictionary,
    configurations: Array,
    time_index: int,
) -> tuple[Array, Array]:
    values, gradients = raw_values_and_gradients(dictionary, configurations)
    means = dictionary.base_means[int(time_index)]
    scales = dictionary.energy_scales[int(time_index)]
    return (
        (values - means) / scales,
        gradients / scales[None, :, None, None],
    )


def make_basis_evaluators(
    dictionary: HybridInvariantDictionary, time_count: int
) -> list[Any]:
    return [
        jax.jit(lambda rows, t=t: _normalized_chunk(dictionary, rows, t))
        for t in range(int(time_count))
    ]


def assemble_hybrid_system(
    dictionary: HybridInvariantDictionary,
    bank,
    weights: Array,
    forcing: Array,
    *,
    chunk_size: int,
    evaluators: list[Any] | None = None,
) -> GalerkinSystem:
    """Accumulate full K/f without materializing all state Jacobians."""

    time_count, sample_count = bank.configurations.shape[:2]
    gram_rows = []
    load_rows = []
    mean_rows = []
    symmetry_rows = []
    forcing_means = []
    evaluators = evaluators or make_basis_evaluators(dictionary, int(time_count))
    for time_index in range(int(time_count)):
        gram = jnp.zeros((dictionary.size, dictionary.size), dtype=jnp.float64)
        load = jnp.zeros((dictionary.size,), dtype=jnp.float64)
        mean = jnp.zeros_like(load)
        forcing_mean = jnp.asarray(0.0, dtype=jnp.float64)
        for start in range(0, int(sample_count), int(chunk_size)):
            stop = min(start + int(chunk_size), int(sample_count))
            values, gradients = evaluators[time_index](
                bank.configurations[time_index, start:stop]
            )
            chunk_weights = weights[time_index, start:stop]
            chunk_forcing = forcing[time_index, start:stop]
            mean = mean + jnp.einsum("n,nk->k", chunk_weights, values)
            forcing_mean = forcing_mean + jnp.einsum(
                "n,n->", chunk_weights, chunk_forcing
            )
            load = load + jnp.einsum(
                "n,n,nk->k", chunk_weights, chunk_forcing, values
            )
            gram = gram + jnp.einsum(
                "n,njpd,nkpd->jk", chunk_weights, gradients, gradients
            )
        load = load - forcing_mean * mean
        transpose = gram.T
        symmetry = jnp.linalg.norm(gram - transpose) / jnp.maximum(
            jnp.linalg.norm(gram), 1.0e-30
        )
        gram_rows.append(0.5 * (gram + transpose))
        load_rows.append(load)
        mean_rows.append(mean)
        symmetry_rows.append(symmetry)
        forcing_means.append(forcing_mean)
    empty = jnp.zeros((0,), dtype=jnp.float64)
    return GalerkinSystem(
        gram=jnp.stack(gram_rows),
        load=jnp.stack(load_rows),
        basis_means=jnp.stack(mean_rows),
        centered_basis=empty,
        weights=empty,
        forcing=empty,
        raw_symmetry_residual=jnp.stack(symmetry_rows),
        forcing_mean=jnp.stack(forcing_means),
    )


def _pad_coefficients(solves: list[Any], sizes: list[int], maximum: int) -> Array:
    padded = []
    for solve, size in zip(solves, sizes, strict=True):
        padded.append(jnp.pad(solve.coefficients, ((0, 0), (0, maximum - size))))
    return jnp.stack(padded)


def audit_hybrid_solutions(
    dictionary: HybridInvariantDictionary,
    coefficients: Array,
    data: PreparedExperiment,
    eta: Array,
    reconstruction,
    audit_state,
    thresholds: CertificateConfig,
    *,
    chunk_size: int,
) -> list[dict[str, Any]]:
    """Evaluate all padded coefficient sets together on the held-out bank."""

    bank = data.ritz_audit_bank
    problem = data.selection_problem
    weights = audit_state.projection.weights
    forcing = audit_state.forcing
    solution_count, time_count, _ = coefficients.shape
    sample_count = int(bank.configurations.shape[1])
    test_count = int(_audit_features(bank.configurations[0, 0], problem.box).shape[0])
    value_means = jnp.zeros((solution_count, time_count), dtype=jnp.float64)
    test_means = jnp.zeros((time_count, test_count), dtype=jnp.float64)
    feature_fn = lambda state: _audit_features(state, problem.box)
    tests_only = jax.jit(jax.vmap(feature_fn))
    evaluators = [
        jax.jit(lambda rows, t=t: _normalized_chunk(dictionary, rows, t))
        for t in range(int(time_count))
    ]
    for time_index in range(int(time_count)):
        for start in range(0, sample_count, int(chunk_size)):
            stop = min(start + int(chunk_size), sample_count)
            x = bank.configurations[time_index, start:stop]
            values, _ = evaluators[time_index](x)
            tests = tests_only(x)
            chunk_weights = weights[time_index, start:stop]
            potentials = jnp.einsum(
                "lk,nk->ln", coefficients[:, time_index], values
            )
            value_means = value_means.at[:, time_index].add(
                jnp.einsum("n,ln->l", chunk_weights, potentials)
            )
            test_means = test_means.at[time_index].add(
                jnp.einsum("n,nf->f", chunk_weights, tests)
            )
    kinetic = jnp.zeros((solution_count, time_count), dtype=jnp.float64)
    linear = jnp.zeros_like(kinetic)
    gauge = jnp.zeros_like(kinetic)
    weak_left = jnp.zeros((solution_count, time_count, test_count), dtype=jnp.float64)
    weak_right = jnp.zeros((time_count, test_count), dtype=jnp.float64)
    grad_scale = jnp.zeros_like(weak_right)
    forcing_scale = jnp.zeros((time_count,), dtype=jnp.float64)
    test_scale = jnp.zeros_like(weak_right)
    corrected_rate = jnp.zeros(
        (solution_count, time_count, problem.family.n_sensors), dtype=jnp.float64
    )
    advective_rate = jnp.zeros((time_count, problem.family.n_sensors), dtype=jnp.float64)
    feature_with_gradient = jax.jit(jax.vmap(
        lambda state: (feature_fn(state), jax.jacrev(feature_fn)(state))
    ))
    for time_index in range(int(time_count)):
        for start in range(0, sample_count, int(chunk_size)):
            stop = min(start + int(chunk_size), sample_count)
            x = bank.configurations[time_index, start:stop]
            values, gradients = evaluators[time_index](x)
            tests, test_gradients = feature_with_gradient(x)
            chunk_weights = weights[time_index, start:stop]
            chunk_forcing = forcing[time_index, start:stop]
            potentials = jnp.einsum(
                "lk,nk->ln", coefficients[:, time_index], values
            )
            potential_gradients = jnp.einsum(
                "lk,nkpd->lnpd", coefficients[:, time_index], gradients
            )
            centered_potentials = potentials - value_means[:, time_index, None]
            centered_tests = tests - test_means[time_index, None, :]
            kinetic_rows = jnp.sum(
                potential_gradients * potential_gradients, axis=(-2, -1)
            )
            kinetic = kinetic.at[:, time_index].add(
                jnp.einsum("n,ln->l", chunk_weights, kinetic_rows)
            )
            linear = linear.at[:, time_index].add(jnp.einsum(
                "n,n,ln->l", chunk_weights, chunk_forcing, centered_potentials
            ))
            gauge = gauge.at[:, time_index].add(
                jnp.einsum("n,ln->l", chunk_weights, centered_potentials)
            )
            weak_left = weak_left.at[:, time_index].add(jnp.einsum(
                "n,lnpd,nfpd->lf", chunk_weights, potential_gradients, test_gradients
            ))
            weak_right = weak_right.at[time_index].add(jnp.einsum(
                "n,n,nf->f", chunk_weights, chunk_forcing, centered_tests
            ))
            grad_scale = grad_scale.at[time_index].add(jnp.einsum(
                "n,nfpd,nfpd->f", chunk_weights, test_gradients, test_gradients
            ))
            forcing_scale = forcing_scale.at[time_index].add(jnp.einsum(
                "n,n,n->", chunk_weights, chunk_forcing, chunk_forcing
            ))
            test_scale = test_scale.at[time_index].add(jnp.einsum(
                "n,nf,nf->f", chunk_weights, centered_tests, centered_tests
            ))
            corrected = jax.vmap(
                lambda gradient: problem.family.jvp(x, -gradient, eta)
            )(potential_gradients)
            corrected_rate = corrected_rate.at[:, time_index].add(jnp.einsum(
                "n,lnr->lr", chunk_weights, corrected
            ))
            advective = problem.family.jvp(
                x, bank.velocity[time_index, start:stop], eta
            )
            advective_rate = advective_rate.at[time_index].add(
                jnp.einsum("n,nr->r", chunk_weights, advective)
            )
    weak = jnp.abs(weak_left + weak_right[None]) / jnp.maximum(
        jnp.sqrt(kinetic)[:, :, None] * jnp.sqrt(grad_scale)[None]
        + jnp.sqrt(forcing_scale)[None, :, None] * jnp.sqrt(test_scale)[None],
        1.0e-12,
    )
    energy = jnp.abs(kinetic + linear) / jnp.maximum(
        kinetic + jnp.abs(linear), 1.0e-12
    )
    rhs = reconstruction.derivatives - advective_rate
    moment = jnp.max(
        jnp.linalg.norm(corrected_rate - rhs[None], axis=-1)
        / jnp.maximum(1.0, jnp.linalg.norm(rhs, axis=-1))[None],
        axis=-1,
    )
    actions = jnp.einsum("t,lt->l", problem.time_weights, kinetic)
    payloads = []
    for index in range(solution_count):
        maximum_weak = float(jnp.max(weak[index]))
        maximum_energy = float(jnp.max(energy[index]))
        maximum_gauge = float(jnp.max(jnp.abs(gauge[index])))
        maximum_moment = float(jnp.max(moment[index]))
        valid = bool(
            maximum_weak <= thresholds.maximum_weak_residual
            and maximum_energy <= thresholds.maximum_energy_residual
            and maximum_gauge <= thresholds.maximum_gauge_residual
            and maximum_moment <= thresholds.maximum_moment_rate_residual
        )
        payloads.append({
            "action": float(actions[index]),
            "maximum_weak_residual": maximum_weak,
            "maximum_energy_residual": maximum_energy,
            "maximum_gauge_residual": maximum_gauge,
            "maximum_moment_rate_residual": maximum_moment,
            "weak_residual_by_time_and_feature": jax.device_get(weak[index]).tolist(),
            "energy_residual_by_time": jax.device_get(energy[index]).tolist(),
            "moment_rate_residual_by_time": jax.device_get(moment[index]).tolist(),
            "valid": valid,
            "thresholds": {
                "maximum_weak_residual": thresholds.maximum_weak_residual,
                "maximum_energy_residual": thresholds.maximum_energy_residual,
                "maximum_gauge_residual": thresholds.maximum_gauge_residual,
                "maximum_moment_rate_residual": thresholds.maximum_moment_rate_residual,
            },
        })
    return payloads


def _monotonicity(rows: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    increments = []
    passed = True
    for previous, current in zip(rows[:-1], rows[1:], strict=True):
        delta = float(current["galerkin_action"] - previous["galerkin_action"])
        relative = delta / max(abs(float(current["galerkin_action"])), 1.0e-30)
        pair_passed = relative >= -float(tolerance)
        passed = passed and pair_passed
        increments.append({
            "from_basis_size": previous["basis_size"],
            "to_basis_size": current["basis_size"],
            "absolute_increase": delta,
            "relative_increase": relative,
            "passed": pair_passed,
        })
    return {"passed": passed, "increments": increments}


def run_production_galerkin_convergence(
    cfg: dict[str, Any], artifact_dir: Path, output_dir: Path
) -> tuple[dict[str, Any], PreparedExperiment]:
    output_dir = require_production_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reproduction, data = run_production_reproduction(
        cfg, artifact_dir, output_dir / "prerequisite_reproduction"
    )
    if not reproduction["gate_a_passed"]:
        result = {
            "ran": False,
            "gate_a_passed": False,
            "outcome_classification": "D. PRODUCTION PROJECTED-LAW REPRODUCTION FAILED",
        }
        write_json(output_dir / "result.json", result)
        return result, data
    settings = cfg["production_galerkin"]
    dictionary = make_hybrid_dictionary(
        box=tuple(cfg["physics"]["box"]),
        fourier_wavevector_count=int(settings["fourier_wavevector_count"]),
        radial_count=int(settings["radial_count"]),
    )
    started = time.perf_counter()
    dictionary = fit_frozen_normalization(
        dictionary,
        data.ritz_train_bank.configurations,
        data.ritz_train_bank.base_weights,
        chunk_size=int(settings["chunk_size"]),
    )
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    save_dictionary(feature_dir / "hybrid_dictionary.npz", dictionary)
    metadata = dictionary_metadata(dictionary)
    metadata["normalization_seconds"] = time.perf_counter() - started
    write_json(feature_dir / "metadata.json", metadata)
    eta = jnp.asarray(cfg["envelope"]["eta0"], dtype=jnp.float64)
    reconstruction = reconstruct_moments(eta, data.selection_problem)
    train_state = forcing_state(
        eta, data.selection_problem, data.ritz_train_bank, reconstruction
    )
    started = time.perf_counter()
    system = assemble_hybrid_system(
        dictionary,
        data.ritz_train_bank,
        train_state.projection.weights,
        train_state.forcing,
        chunk_size=int(settings["chunk_size"]),
    )
    assembly_seconds = time.perf_counter() - started
    sizes = [int(size) for size in settings["basis_size_ladder"]]
    if sizes != sorted(set(sizes)) or sizes[-1] > dictionary.size:
        raise ValueError("production Galerkin ladder is not a valid nested prefix ladder")
    solves = [
        rank_aware_quadratic_solve(
            system.gram[:, :size, :size],
            system.load[:, :size],
            relative_rank_tolerance=float(settings["relative_rank_tolerance"]),
        )
        for size in sizes
    ]
    padded = _pad_coefficients(solves, sizes, dictionary.size)
    audit_state = forcing_state(
        eta, data.selection_problem, data.ritz_audit_bank, reconstruction
    )
    started = time.perf_counter()
    certificates = audit_hybrid_solutions(
        dictionary, padded, data, eta, reconstruction, audit_state,
        CertificateConfig(**settings["certificate_thresholds"]),
        chunk_size=int(settings["chunk_size"]),
    )
    audit_seconds = time.perf_counter() - started
    rows = []
    for size, solve, certificate in zip(sizes, solves, certificates, strict=True):
        aggregate = aggregate_quadratic_values(
            solve, data.selection_problem.time_weights
        )
        algebra_valid = bool(
            float(aggregate["identity_relerr"]) <= float(settings["maximum_identity_relerr"])
            and float(jnp.max(solve.stationarity_residual)) <= float(settings["maximum_stationarity_residual"])
            and float(jnp.max(solve.range_residual)) <= float(settings["maximum_range_residual"])
            and float(jnp.max(system.raw_symmetry_residual)) <= float(settings["maximum_symmetry_residual"])
        )
        rank_fraction = solve.numerical_rank / float(size)
        row = {
            "basis_size": size,
            "galerkin_action": float(aggregate["action"]),
            "galerkin_objective": float(aggregate["objective"]),
            "aggregate_identity_relerr": float(aggregate["identity_relerr"]),
            "action_by_time": jax.device_get(solve.action_by_time).tolist(),
            "objective_by_time": jax.device_get(solve.objective_by_time).tolist(),
            "identity_relerr_by_time": jax.device_get(solve.identity_relerr_by_time).tolist(),
            "numerical_rank_by_time": jax.device_get(solve.numerical_rank).tolist(),
            "rank_fraction_by_time": jax.device_get(rank_fraction).tolist(),
            "eigenvalues_by_time": jax.device_get(solve.eigenvalues).tolist(),
            "retained_condition_by_time": jax.device_get(solve.condition_number).tolist(),
            "stationarity_residual_by_time": jax.device_get(solve.stationarity_residual).tolist(),
            "range_residual_by_time": jax.device_get(solve.range_residual).tolist(),
            "worst_stationarity_residual": float(jnp.max(solve.stationarity_residual)),
            "worst_range_residual": float(jnp.max(solve.range_residual)),
            "worst_retained_condition": float(jnp.max(solve.condition_number)),
            "minimum_rank_fraction": float(jnp.min(rank_fraction)),
            "worst_symmetry_residual": float(jnp.max(system.raw_symmetry_residual)),
            "algebra_valid": algebra_valid,
            "held_out_certificate": certificate,
        }
        rows.append(row)
        np.savez_compressed(
            require_production_output_path(output_dir / f"K{size}.npz"),
            coefficients=np.asarray(solve.coefficients),
            eigenvalues=np.asarray(solve.eigenvalues),
            retained=np.asarray(solve.retained),
        )
        write_json(output_dir / f"K{size}.json", row)
    np.savez_compressed(
        require_production_output_path(output_dir / "full_system.npz"),
        gram=np.asarray(system.gram), load=np.asarray(system.load),
        basis_means=np.asarray(system.basis_means),
    )
    monotonicity = _monotonicity(
        rows, float(settings["monotonicity_relative_tolerance"])
    )
    increments = monotonicity["increments"]
    stable_flags = [
        abs(float(row["relative_increase"]))
        <= float(settings["stability_relative_tolerance"])
        for row in increments
    ]
    required = int(settings["required_consecutive_stable_increments"])
    stabilized = any(
        all(stable_flags[start:start + required])
        for start in range(max(0, len(stable_flags) - required + 1))
    )
    largest = rows[-1]
    basis_passed = bool(
        monotonicity["passed"]
        and stabilized
        and largest["algebra_valid"]
        and largest["held_out_certificate"]["valid"]
        and largest["minimum_rank_fraction"] >= float(settings["minimum_rank_fraction"])
        and largest["worst_retained_condition"] <= float(settings["maximum_retained_condition"])
    )
    result = {
        "ran": True,
        "eta0": jax.device_get(eta).tolist(),
        "gate_a_passed": True,
        "dictionary": metadata,
        "basis_sizes": rows,
        "monotonicity": monotonicity,
        "two_consecutive_stable_increments": stabilized,
        "preferred_largest_increment_observed": bool(
            increments
            and abs(float(increments[-1]["relative_increase"]))
            <= float(settings["preferred_stability_relative_tolerance"])
        ),
        "assembly_seconds": assembly_seconds,
        "held_out_audit_seconds": audit_seconds,
        "basis_convergence_passed": basis_passed,
        "deep_ritz_reproduction": reproduction["deep_ritz_reproduction"],
        "forcing_audits": reproduction["forcing_audits"],
        "outcome_classification": (
            "B. PRODUCTION GALERKIN SOLVER VALID, ETA GRADIENT NOT YET VALIDATED"
            if basis_passed
            else "C. GALERKIN BASIS NOT YET PHYSICALLY ADEQUATE"
        ),
    }
    write_json(output_dir / "result.json", result)
    return result, data


__all__ = [
    "assemble_hybrid_system", "audit_hybrid_solutions",
    "make_basis_evaluators",
    "run_production_galerkin_convergence",
]
