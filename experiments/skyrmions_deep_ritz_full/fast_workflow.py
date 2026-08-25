"""Benchmarking, validation, and controlled 3% refinement for the fast route."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mfsi.cache import fingerprint
from mfsi.projection import EmpiricalIProjector

from .deep_ritz import CertificateConfig, load_ritz_checkpoint
from .fast_production import (
    FAST_ROOT, FastProductionContext, _sync, _timed, require_fast_output_path,
)
from .full_gradient import (
    forcing_state, minimum_sensor_separation, periodic_branch_distance,
    reconstruct_moments, wrap_periodic,
)
from .galerkin import aggregate_quadratic_values, rank_aware_quadratic_solve
from .production_artifacts import PRODUCTION_ROOT, file_sha256
from .production_basis import load_dictionary, raw_values_and_gradients
from .production_galerkin import audit_hybrid_solutions, make_basis_evaluators
from .production_gradient import (
    evaluate_local_eta, precompute_fixed_potential_rows,
    production_hybrid_envelope_value_and_grad,
)
from .production_workflow import load_production_data
from .workflow import (
    authoritative_evaluate, save_candidate_checkpoint, selection_risk,
    validation_risk, write_json,
)

Array = jax.Array


def _rel(left: Array, right: Array) -> float:
    a, b = np.asarray(left), np.asarray(right)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-30))


def _max_abs(left: Array, right: Array) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def _timing_pair(function, repeats: int = 3) -> dict[str, float]:
    _, first = _timed(function)
    steady = [_timed(function)[1] for _ in range(repeats)]
    return {"first_call_seconds": first, "steady_median_seconds": statistics.median(steady)}


def _old_complete(ctx: FastProductionContext, eta: Array, evaluators) -> tuple[Any, Any, Any, Any]:
    payload, solve = evaluate_local_eta(eta, ctx.cfg, ctx.data, ctx.dictionary, evaluators)
    potential, kinetic = precompute_fixed_potential_rows(
        ctx.dictionary, solve.coefficients, ctx.data, evaluators,
        chunk_size=int(ctx.cfg["production_galerkin"]["chunk_size"]),
    )
    value, gradient = production_hybrid_envelope_value_and_grad(
        eta, solve.coefficients, ctx.data, potential, kinetic
    )
    return payload, solve, value, gradient


def _equivalence_payload(ctx: FastProductionContext, eta: Array, evaluators) -> dict[str, Any]:
    old_payload, old_solve, old_value, old_gradient = _old_complete(ctx, eta, evaluators)
    fast = ctx.evaluate(eta, basis_size=160, with_gradient=True)
    _sync(fast)
    reconstruction = reconstruct_moments(eta, ctx.data.selection_problem)
    old_state = forcing_state(
        eta, ctx.data.selection_problem, ctx.data.ritz_train_bank, reconstruction
    )
    comparisons = {
        "risk_relative": abs(float(fast.risk) - float(selection_risk(eta, ctx.data)))
            / max(abs(float(fast.risk)), 1e-30),
        "c_max_abs": _max_abs(fast.reconstruction.values, reconstruction.values),
        "cdot_max_abs": _max_abs(fast.reconstruction.derivatives, reconstruction.derivatives),
        "lambda_max_abs": _max_abs(fast.train_state.projection.lam, old_state.projection.lam),
        "weights_max_abs": _max_abs(fast.train_state.projection.weights, old_state.projection.weights),
        "lambda_dot_max_abs": _max_abs(fast.train_state.lambda_dot, old_state.lambda_dot),
        "forcing_max_abs": _max_abs(fast.train_state.forcing, old_state.forcing),
        "gram_relative": 0.0,
        "coefficients_relative": _rel(fast.solve.coefficients, old_solve.coefficients),
        "action_relative": abs(float(fast.action) - float(old_value)) / max(abs(float(old_value)), 1e-30),
        "gradient_relative": _rel(fast.gradient, old_gradient),
        "rank_equal": bool(np.array_equal(
            np.asarray(fast.solve.numerical_rank), np.asarray(old_solve.numerical_rank)
        )),
    }
    # Reconstruct the old K/f explicitly so the cache contractions themselves are checked.
    old_reconstruction = reconstruct_moments(eta, ctx.data.selection_problem)
    old_train = forcing_state(
        eta, ctx.data.selection_problem, ctx.data.ritz_train_bank, old_reconstruction
    )
    from .production_galerkin import assemble_hybrid_system
    old_system = assemble_hybrid_system(
        ctx.dictionary, ctx.data.ritz_train_bank,
        old_train.projection.weights, old_train.forcing,
        chunk_size=int(ctx.cfg["production_galerkin"]["chunk_size"]),
        evaluators=evaluators,
    )
    comparisons["gram_relative"] = _rel(fast.system.gram, old_system.gram)
    comparisons["load_relative"] = _rel(fast.system.load, old_system.load)
    passed = bool(
        comparisons["action_relative"] <= 1e-10
        and comparisons["gradient_relative"] <= 1e-8
        and comparisons["gram_relative"] <= 1e-10
        and comparisons["load_relative"] <= 1e-10
        # Coefficients in numerically near-null retained directions are much
        # more sensitive than the represented potential.  The measured 1e-6
        # coefficient change leaves action/gradient at 1e-13/1e-10.
        and comparisons["coefficients_relative"] <= 1e-5
        and comparisons["rank_equal"]
    )
    return {
        "passed": passed,
        "tolerances": {"action_relative": 1e-10, "gradient_relative": 1e-8,
                       "gram_load_relative": 1e-10, "coefficients_relative": 1e-5},
        "coefficient_tolerance_justification": (
            "roundoff-order changes in ill-conditioned retained eigendirections; "
            "rank is identical and action/gradient satisfy their stricter invariance gates"
        ),
        "comparisons": comparisons,
        "old_action": float(old_value), "fast_action": float(fast.action),
        "old_gradient": np.asarray(old_gradient).tolist(),
        "fast_gradient": np.asarray(fast.gradient).tolist(),
        "old_payload_action": old_payload["action"],
    }


def run_fast_benchmark(cfg: dict[str, Any], artifact_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = require_fast_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dictionary_path = PRODUCTION_ROOT / "convergence" / "features" / "hybrid_dictionary.npz"
    eta = jnp.asarray(cfg["envelope"]["eta0"], dtype=jnp.float64)

    ctx, context_seconds = _timed(lambda: FastProductionContext(cfg, artifact_dir))
    evaluators = make_basis_evaluators(ctx.dictionary, len(ctx.values))
    old_timing = _timing_pair(lambda: _old_complete(ctx, eta, evaluators), repeats=3)
    fast_timing = _timing_pair(lambda: ctx.evaluate(eta, with_gradient=True), repeats=5)
    fast_value_timing = _timing_pair(lambda: ctx.evaluate(eta, with_gradient=False), repeats=5)
    equivalence = _equivalence_payload(ctx, eta, evaluators)

    problem, bank = ctx.data.selection_problem, ctx.data.ritz_train_bank
    reconstruction = reconstruct_moments(eta, problem)
    train_state = forcing_state(eta, problem, bank, reconstruction)
    system = ctx.assemble(train_state.projection.weights, train_state.forcing, 160)
    solve = rank_aware_quadratic_solve(
        system.gram, system.load,
        relative_rank_tolerance=float(cfg["production_galerkin"]["relative_rank_tolerance"]),
    )
    potential, kinetic = ctx.potential_rows(solve.coefficients, 160)

    # Granular synchronized stages. Each entry is a directly executed operation;
    # coupled stages are explicitly labeled rather than inferred by subtraction.
    sample_rows = bank.configurations[0]
    value_only = jax.jit(lambda rows: raw_values_and_gradients(ctx.dictionary, rows)[0])
    gradient_only = jax.jit(lambda rows: raw_values_and_gradients(ctx.dictionary, rows)[1])
    feature_ref = jax.jit(lambda: problem.family.features(bank.configurations, eta))
    jvp_ref = jax.jit(lambda: problem.family.jvp(bank.configurations, bank.velocity, eta))
    truth_features = jax.jit(lambda: problem.family.features(problem.truth_configurations, eta))
    features = feature_ref()
    projector = EmpiricalIProjector(problem.projection_config, trajectory_backend=problem.projection_backend)
    projection_call = lambda: projector.project_trajectory(
        features, bank.base_weights, reconstruction.values[None, ...]
    )
    projection = projection_call()
    projection = jax.tree_util.tree_map(lambda x: x[0], projection)
    projected_calculations = jax.jit(lambda: (
        jnp.einsum("tn,tnr->tr", projection.weights, features),
        jnp.min(projection.ess_fraction),
    ))
    timings = {
        "01_artifact_loading": {"steady_seconds": ctx.artifact_load_seconds},
        "02_basis_construction_dictionary_load": _timing_pair(
            lambda: load_dictionary(dictionary_path, box=tuple(cfg["physics"]["box"])), 2),
        "03_basis_values_reference_bank_one_time": _timing_pair(lambda: value_only(sample_rows), 2),
        "04_basis_state_jacobians_reference_bank_one_time": _timing_pair(lambda: gradient_only(sample_rows), 2),
        "05_sensor_observables_truth_bank": _timing_pair(truth_features, 3),
        "06_sensor_observables_reference_bank": _timing_pair(feature_ref, 3),
        "06b_sensor_jvp_reference_bank": _timing_pair(jvp_ref, 3),
        "07_moment_reconstruction": _timing_pair(lambda: reconstruct_moments(eta, problem), 3),
        "08_information_projection": _timing_pair(projection_call, 3),
        "09_projected_weight_calculations": _timing_pair(projected_calculations, 3),
        "10_lambda_dot_and_11_forcing_coupled": _timing_pair(
            lambda: forcing_state(eta, problem, bank, reconstruction), 3),
        "12_fast_K_matrix_assembly": _timing_pair(
            lambda: ctx.assemble(train_state.projection.weights, train_state.forcing, 160).gram, 3),
        "13_fast_f_vector_assembly": _timing_pair(
            lambda: ctx.assemble(train_state.projection.weights, train_state.forcing, 160).load, 3),
        "14_rank_eigendecomposition_solve": _timing_pair(lambda: rank_aware_quadratic_solve(
            system.gram, system.load,
            relative_rank_tolerance=float(cfg["production_galerkin"]["relative_rank_tolerance"])), 3),
        "15_galerkin_action": _timing_pair(lambda: aggregate_quadratic_values(
            solve, problem.time_weights)["action"], 3),
        "16_reverse_mode_eta_gradient": _timing_pair(
            lambda: ctx._envelope_value_grad(eta, potential, kinetic), 3),
        "18_risk_computation": _timing_pair(lambda: ctx._risk(eta), 3),
        "19_complete_value_only_fast": fast_value_timing,
        "20_complete_value_gradient_old": old_timing,
        "20_complete_value_gradient_fast": fast_timing,
        "21_authoritative_fixed_design_rescore": {
            "steady_seconds": None,
            "note": "not rerun for profiling; populated from the selectively scheduled finalist call",
        },
    }
    certificate_start = time.perf_counter()
    audit_state = forcing_state(eta, problem, ctx.data.ritz_audit_bank, reconstruction)
    certificate = audit_hybrid_solutions(
        ctx.dictionary, solve.coefficients[None], ctx.data, eta, reconstruction,
        audit_state, CertificateConfig(**cfg["production_galerkin"]["certificate_thresholds"]),
        chunk_size=int(cfg["production_galerkin"]["chunk_size"]),
    )[0]
    _sync(certificate)
    timings["17_heldout_galerkin_certification"] = {
        "first_call_seconds": time.perf_counter() - certificate_start,
        "steady_median_seconds": None,
    }
    result = {
        "schema_version": 1,
        "eta": np.asarray(eta).tolist(),
        "artifact_manifest_sha256": file_sha256(artifact_dir / "isolated_artifact_manifest.json"),
        "dictionary_sha256": file_sha256(dictionary_path),
        "context_initialization_seconds": context_seconds,
        "cache": ctx.cache_info,
        "timings": timings,
        "old_complete_value_gradient": old_timing,
        "fast_complete_value_gradient": fast_timing,
        "fast_complete_value_only": fast_value_timing,
        "steady_value_gradient_speedup": (
            old_timing["steady_median_seconds"] / fast_timing["steady_median_seconds"]
        ),
        "equivalence": equivalence,
        "heldout_certificate": certificate,
        "projection_warm_start": {
            "physical_time": "already used by EmpiricalIProjector.project_trajectory",
            "outer_design": "not exposed by the fixed native projection API; intentionally not emulated",
            "gradient_semantics_changed": False,
        },
    }
    write_json(output_dir / "result.json", result)
    return result


def _gradient_metrics(lower: dict[str, Any], upper: dict[str, Any]) -> dict[str, float]:
    a = np.asarray(lower["gradient"], dtype=np.float64)
    b = np.asarray(upper["gradient"], dtype=np.float64)
    return {
        "lower_K": lower["K"], "upper_K": upper["K"],
        "cosine_similarity": float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))),
        "relative_gradient_difference": float(np.linalg.norm(b - a) / max(np.linalg.norm(b), 1e-30)),
        "relative_action_difference": abs(upper["action"] - lower["action"]) / max(abs(upper["action"]), 1e-30),
    }


def run_gradient_convergence(cfg: dict[str, Any], artifact_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = require_fast_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = FAST_ROOT / "profiling" / "result.json"
    if not benchmark.is_file() or not json.loads(benchmark.read_text())["equivalence"]["passed"]:
        raise RuntimeError("fast numerical-equivalence prerequisite is missing or failed")
    ctx = FastProductionContext(cfg, artifact_dir)
    eta = jnp.asarray(cfg["envelope"]["eta0"], dtype=jnp.float64)
    rows = []
    for size in (100, 120, 140, 160):
        evaluated, seconds = _timed(lambda size=size: ctx.evaluate(eta, basis_size=size, with_gradient=True))
        rows.append({"K": size, "action": float(evaluated.action),
                     "gradient": np.asarray(evaluated.gradient).tolist(),
                     "gradient_norm": float(jnp.linalg.norm(evaluated.gradient)),
                     "seconds": seconds,
                     "rank_by_time": np.asarray(evaluated.solve.numerical_rank).tolist()})
    neighbors = [_gradient_metrics(rows[i], rows[i + 1]) for i in range(len(rows) - 1)]
    final = neighbors[-1]
    passed = bool(final["cosine_similarity"] >= 0.995 and final["relative_gradient_difference"] <= 0.05)
    result = {"passed": passed, "rows": rows, "neighbor_comparisons": neighbors,
              "declared_gate": {"cosine_similarity_minimum": .995,
                                "relative_gradient_difference_maximum": .05}}
    write_json(output_dir / "result.json", result)
    return result


def _nearby_points(ctx: FastProductionContext, count: int = 4) -> list[Array]:
    eta0 = jnp.asarray(ctx.cfg["envelope"]["eta0"], dtype=jnp.float64)
    law = float(ctx._risk(jnp.asarray(ctx.cfg["envelope"]["law_eta"], dtype=jnp.float64)))
    ceiling = 1.03 * law
    points = [eta0]
    previous = jnp.asarray([
        0.8953839921146673, 0.20595035907471138, 1.3345144773868762,
        0.8654744150451203, 0.7508077339024882, 0.5179727362721115,
        1.6423936578820195, 0.5884106107337586,
    ], dtype=jnp.float64)
    if float(ctx._risk(previous)) <= ceiling:
        points.append(previous)
    key = jax.random.PRNGKey(20260824)
    for attempt in range(256):
        key, subkey = jax.random.split(key)
        direction = jax.random.normal(subkey, eta0.shape, dtype=jnp.float64)
        direction /= jnp.linalg.norm(direction)
        radius = (1 + attempt % 4) * 5e-5
        candidate = wrap_periodic(eta0 + radius * direction, ctx.data.selection_problem.family)
        if (bool(ctx.data.selection_problem.family.geometry_valid(candidate))
                and float(ctx._risk(candidate)) <= ceiling):
            state = ctx.evaluate(candidate, basis_size=160, with_gradient=False)
            if ctx.search_payload(state)["train_forcing_audit"]["valid"]:
                points.append(candidate)
        if len(points) >= count:
            break
    if len(points) < count:
        raise RuntimeError(f"only found {len(points)} deterministic nearby feasible points")
    return points


def run_local_gradient_audit(cfg: dict[str, Any], artifact_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = require_fast_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    convergence_path = FAST_ROOT / "gradient_convergence" / "result.json"
    if not convergence_path.is_file() or not json.loads(convergence_path.read_text())["passed"]:
        raise RuntimeError("basis-gradient convergence prerequisite is missing or failed")
    ctx = FastProductionContext(cfg, artifact_dir)
    points = _nearby_points(ctx, 4)
    key = jax.random.PRNGKey(20260825)
    point_rows = []
    for point_index, eta in enumerate(points):
        k140 = ctx.evaluate(eta, basis_size=140, with_gradient=True)
        k160 = ctx.evaluate(eta, basis_size=160, with_gradient=True)
        row = {
            "point_index": point_index, "eta": np.asarray(eta).tolist(),
            "risk": float(k160.risk), "K140_action": float(k140.action),
            "K160_action": float(k160.action), "K140_gradient": np.asarray(k140.gradient).tolist(),
            "K160_gradient": np.asarray(k160.gradient).tolist(),
            **_gradient_metrics(
                {"K": 140, "action": float(k140.action), "gradient": np.asarray(k140.gradient).tolist()},
                {"K": 160, "action": float(k160.action), "gradient": np.asarray(k160.gradient).tolist()},
            ),
            "train_forcing_audit": ctx.search_payload(k160)["train_forcing_audit"],
            "directional_checks": [],
        }
        if point_index in (1, 2):
            for direction_index in range(2):
                key, subkey = jax.random.split(key)
                direction = jax.random.normal(subkey, eta.shape, dtype=jnp.float64)
                direction /= jnp.linalg.norm(direction)
                ad = float(jnp.dot(k160.gradient, direction))
                checks = []
                for epsilon in (3e-4, 1e-4, 3e-5, 1e-5):
                    plus = wrap_periodic(eta + epsilon * direction, ctx.data.selection_problem.family)
                    minus = wrap_periodic(eta - epsilon * direction, ctx.data.selection_problem.family)
                    ep = ctx.evaluate(plus, basis_size=160, with_gradient=False)
                    em = ctx.evaluate(minus, basis_size=160, with_gradient=False)
                    fd = (float(ep.action) - float(em.action)) / (2 * epsilon)
                    checks.append({"epsilon": epsilon, "fd": fd,
                                   "relative_discrepancy": abs(fd - ad) / max(abs(fd), abs(ad), 1e-12)})
                row["directional_checks"].append({
                    "direction_index": direction_index, "direction": np.asarray(direction).tolist(),
                    "ad": ad, "rows": checks,
                    "passed": bool(min(c["relative_discrepancy"] for c in checks) <= .02),
                })
        point_rows.append(row)
    convergence_pass = all(r["cosine_similarity"] >= .995 and r["relative_gradient_difference"] <= .05 for r in point_rows)
    fd_pass = all(d["passed"] for r in point_rows for d in r["directional_checks"])
    passed = bool(convergence_pass and fd_pass)
    result = {"passed": passed, "K_convergence_passed": convergence_pass,
              "directional_fd_passed": fd_pass, "points": point_rows,
              "gates": {"cosine_minimum": .995, "relative_gradient_maximum": .05,
                        "at_least_one_fd_relative_error_maximum": .02}}
    write_json(output_dir / "result.json", result)
    return result


def _periodic_delta(candidate: Array, center: Array, box: tuple[float, float]) -> Array:
    shaped = (candidate - center).reshape((-1, 2))
    box_array = jnp.asarray(box, dtype=jnp.float64)
    shaped = shaped - box_array * jnp.round(shaped / box_array)
    return shaped.reshape((-1,))


def _local_trajectory(
    ctx: FastProductionContext, start: Array, *, radius: float, steps: int,
) -> dict[str, Any]:
    family = ctx.data.selection_problem.family
    law_risk = float(ctx._risk(jnp.asarray(ctx.cfg["envelope"]["law_eta"], dtype=jnp.float64)))
    ceiling = 1.03 * law_risk
    center = jnp.asarray(start, dtype=jnp.float64)
    eta = center
    current = ctx.evaluate(eta, with_gradient=True)
    initial_action = float(current.action)
    history = []
    evaluations = 1
    for step in range(int(steps)):
        gradient = current.gradient
        direction = -gradient / jnp.maximum(jnp.linalg.norm(gradient), 1e-30)
        accepted = False
        attempts = []
        for backtrack in range(10):
            length = radius * (0.5 ** backtrack)
            proposal = wrap_periodic(eta + length * direction, family)
            delta = _periodic_delta(proposal, center, family.box)
            norm = float(jnp.linalg.norm(delta))
            if norm > radius * (1 + 1e-12) or not bool(family.geometry_valid(proposal)):
                attempts.append({"length": length, "accepted": False, "reason": "trust_or_geometry"})
                continue
            risk = float(ctx._risk(proposal))
            if risk > ceiling:
                attempts.append({"length": length, "accepted": False, "reason": "risk", "risk": risk})
                continue
            candidate = ctx.evaluate(proposal, with_gradient=True)
            evaluations += 1
            payload = ctx.search_payload(candidate)
            rank_stable = bool(np.array_equal(
                np.asarray(candidate.solve.numerical_rank), np.asarray(current.solve.numerical_rank)
            ))
            accepted = bool(
                float(candidate.action) < float(current.action)
                and payload["train_forcing_audit"]["valid"] and rank_stable
            )
            attempts.append({"length": length, "accepted": accepted, "risk": risk,
                             "action": float(candidate.action), "rank_stable": rank_stable,
                             "train_forcing_valid": payload["train_forcing_audit"]["valid"]})
            if accepted:
                eta, current = proposal, candidate
                break
        history.append({"step": step, "accepted": accepted,
                        "eta": np.asarray(eta).tolist(), "action": float(current.action),
                        "risk": float(current.risk), "attempts": attempts})
        if not accepted or float(jnp.linalg.norm(_periodic_delta(eta, center, family.box))) >= radius * .999:
            break
    return {"start_eta": np.asarray(start).tolist(), "end_eta": np.asarray(eta).tolist(),
            "radius": radius, "steps_requested": steps,
            "steps_accepted": sum(row["accepted"] for row in history),
            "start_action": initial_action, "end_action": float(current.action),
            "action_reduction": initial_action - float(current.action),
            "end_risk": float(current.risk), "evaluation_count": evaluations,
            "history": history}


def run_fast_refinement(cfg: dict[str, Any], artifact_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = require_fast_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = FAST_ROOT / "local_gradient_audit" / "result.json"
    if not audit_path.is_file() or not json.loads(audit_path.read_text())["passed"]:
        result = {"ran": False, "reason": "local gradient prerequisite failed"}
        write_json(output_dir / "result.json", result)
        return result
    ctx = FastProductionContext(cfg, artifact_dir)
    eta0 = jnp.asarray(cfg["envelope"]["eta0"], dtype=jnp.float64)
    trajectory = _local_trajectory(ctx, eta0, radius=2e-4, steps=6)
    result = {"ran": True, "trust_region": {"initial_radius": 2e-4,
              "norm": "periodic minimum-image Euclidean", "steps": 6},
              "trajectory": trajectory}
    write_json(output_dir / "result.json", result)
    return result


def run_fast_multistart(cfg: dict[str, Any], artifact_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = require_fast_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    refine_path = FAST_ROOT / "trajectories" / "result.json"
    if not refine_path.is_file() or not json.loads(refine_path.read_text()).get("ran"):
        raise RuntimeError("single-start trust-region prerequisite is missing")
    ctx = FastProductionContext(cfg, artifact_dir)
    points = _nearby_points(ctx, 4)
    trajectories = [_local_trajectory(ctx, point, radius=2e-4, steps=6) for point in points]
    trajectories.sort(key=lambda row: row["end_action"])
    # Full held-out certification is reserved for the two best distinct endpoints.
    finalists = []
    thresholds = CertificateConfig(**cfg["production_galerkin"]["certificate_thresholds"])
    for index, trajectory in enumerate(trajectories[:2]):
        eta = jnp.asarray(trajectory["end_eta"], dtype=jnp.float64)
        evaluation = ctx.evaluate(eta, with_gradient=False)
        reconstruction = evaluation.reconstruction
        audit_state = forcing_state(
            eta, ctx.data.selection_problem, ctx.data.ritz_audit_bank, reconstruction
        )
        certificate = audit_hybrid_solutions(
            ctx.dictionary, evaluation.solve.coefficients[None], ctx.data, eta,
            reconstruction, audit_state, thresholds,
            chunk_size=int(cfg["production_galerkin"]["chunk_size"]),
        )[0]
        forcing_valid = bool(ctx.search_payload(evaluation)["train_forcing_audit"]["valid"])
        eligible = bool(certificate["valid"] and forcing_valid)
        finalists.append({"rank": index + 1, **trajectory,
                          "heldout_certificate": certificate,
                          "train_forcing_valid": forcing_valid,
                          "eligible_for_authoritative": eligible})
    result = {"ran": True, "start_count": len(points), "trajectories": trajectories,
              "finalists": finalists,
              "galerkin_evaluation_count": sum(t["evaluation_count"] for t in trajectories),
              "authoritative_calls_avoided_in_hot_loop": sum(t["evaluation_count"] for t in trajectories)}
    write_json(output_dir / "result.json", result)
    return result


def _authoritative_signature(
    cfg: dict[str, Any], artifact_dir: Path, eta: Array, *, validation: bool,
) -> str:
    return fingerprint({
        "kind": "fast_production_authoritative_fixed_design_v1",
        "eta_float64": np.asarray(eta, dtype=np.float64).tolist(),
        "validation": validation,
        "artifact_manifest_sha256": file_sha256(
            artifact_dir / "isolated_artifact_manifest.json"
        ),
        "solver": cfg["deep_ritz"],
        "certificates": cfg["certificates"],
        "projection": cfg["projection"],
        "forcing": cfg["forcing"],
    })


def _cached_authoritative(
    cfg: dict[str, Any], artifact_dir: Path, eta: Array, cache_dir: Path,
    *, validation: bool, initial_checkpoint: Path,
) -> tuple[dict[str, Any], bool, float]:
    cache_dir = require_fast_output_path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = _authoritative_signature(cfg, artifact_dir, eta, validation=validation)
    result_path = cache_dir / "result.json"
    metadata_path = cache_dir / "metadata.json"
    if result_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature:
            return json.loads(result_path.read_text(encoding="utf-8")), True, float(metadata["elapsed_seconds"])
    data = load_production_data(cfg, artifact_dir)
    initial_params, checkpoint_metadata = load_ritz_checkpoint(initial_checkpoint)
    start = time.perf_counter()
    evaluation = authoritative_evaluate(
        eta, cfg, data, allowance_percent=3.0,
        initial_params=initial_params, validation=validation,
    )
    elapsed = time.perf_counter() - start
    write_json(result_path, evaluation.payload)
    if evaluation.params is not None:
        save_candidate_checkpoint(
            cache_dir / "checkpoint.npz", evaluation,
            role="fast_production_validation" if validation else "fast_production_selection",
        )
    write_json(metadata_path, {
        "signature": signature, "elapsed_seconds": elapsed,
        "initial_checkpoint": str(initial_checkpoint),
        "initial_checkpoint_sha256": file_sha256(initial_checkpoint),
        "initial_checkpoint_metadata": checkpoint_metadata,
    })
    return evaluation.payload, False, elapsed


def run_authoritative_selection(
    cfg: dict[str, Any], artifact_dir: Path, output_dir: Path,
) -> dict[str, Any]:
    output_dir = require_fast_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    multistart_path = FAST_ROOT / "multistart" / "result.json"
    if not multistart_path.is_file():
        raise RuntimeError("multistart result is missing")
    multistart = json.loads(multistart_path.read_text(encoding="utf-8"))
    eligible = [row for row in multistart["finalists"] if row["eligible_for_authoritative"]]
    if not eligible:
        result = {"ran": False, "reason": "no held-out-certified Galerkin finalist"}
        write_json(output_dir / "result.json", result)
        return result
    candidate_eta = jnp.asarray(eligible[0]["end_eta"], dtype=jnp.float64)
    candidate_key = fingerprint(np.asarray(candidate_eta).tolist())[:16]
    candidate, cache_hit, elapsed = _cached_authoritative(
        cfg, artifact_dir, candidate_eta, output_dir / candidate_key,
        validation=False, initial_checkpoint=artifact_dir / "ritz_full.npz",
    )
    prior_root = PRODUCTION_ROOT / "authoritative_crosscheck"
    incumbent = json.loads((prior_root / "eta0.json").read_text(encoding="utf-8"))
    previous_tiny = json.loads((prior_root / "eta1.json").read_text(encoding="utf-8"))
    candidates = [
        {"id": "old_3pct_incumbent", "source": str(prior_root / "eta0.json"), **incumbent},
        {"id": "previous_tiny_update", "source": str(prior_root / "eta1.json"), **previous_tiny},
        {"id": "best_new_continuous", "source": str(output_dir / candidate_key / "result.json"), **candidate},
    ]
    valid = [row for row in candidates if row.get("valid", False)]
    winner = min(valid, key=lambda row: float(row["action"])) if valid else candidates[0]
    tolerance = float(cfg["envelope"].get("minimum_improvement", 1e-6))
    improved_vs_old = bool(
        winner.get("valid", False)
        and float(winner["action"]) < float(incumbent["action"]) - tolerance
    )
    further_vs_previous = bool(
        candidate.get("valid", False)
        and float(candidate["action"]) < float(previous_tiny["action"]) - tolerance
    )
    result = {
        "ran": True, "authoritative_new_calls": 0 if cache_hit else 1,
        "authoritative_cache_hits": 2 + int(cache_hit),
        "new_candidate_elapsed_seconds": elapsed,
        "candidates": candidates, "winner": winner,
        "winner_frozen": bool(winner.get("valid", False)),
        "improved_vs_old_incumbent": improved_vs_old,
        "best_new_further_improved_vs_previous_tiny": further_vs_previous,
        "replacement_tolerance": tolerance,
        "original_production_incumbent_modified": False,
    }
    write_json(output_dir.parent / "selection" / "result.json", result)
    write_json(output_dir / "result.json", result)
    return result


def run_fast_validation(
    cfg: dict[str, Any], artifact_dir: Path, output_dir: Path,
    *, selection_result: Path,
) -> dict[str, Any]:
    output_dir = require_fast_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_result = require_fast_output_path(selection_result)
    selection = json.loads(selection_result.read_text(encoding="utf-8"))
    if not selection.get("winner_frozen"):
        raise RuntimeError("selection does not contain a frozen valid winner")
    winner = selection["winner"]
    winner_eta = jnp.asarray(winner["eta"], dtype=jnp.float64)

    # The old incumbent's disjoint-bank result is a frozen production artifact,
    # reused only because eta and authoritative configuration match exactly.
    source_result = json.loads((artifact_dir / "result.json").read_text(encoding="utf-8"))
    old_eta = np.asarray(source_result["full_3_percent"]["eta"], dtype=np.float64)
    expected_old = np.asarray(cfg["envelope"]["eta0"], dtype=np.float64)
    if not np.array_equal(old_eta, expected_old):
        raise RuntimeError("frozen incumbent validation eta does not match eta0")
    old_validation = source_result["validation"]["full"]
    winner_validation, cache_hit, elapsed = _cached_authoritative(
        cfg, artifact_dir, winner_eta, output_dir / "winner",
        validation=True, initial_checkpoint=artifact_dir / "ritz_validation_full.npz",
    )
    result = {
        "ran": True,
        "selection_was_frozen_before_validation": True,
        "validation_used_for_selection": False,
        "old_incumbent": old_validation,
        "old_incumbent_reused_frozen_authoritative_validation": True,
        "winner": winner_validation,
        "winner_cache_hit": cache_hit,
        "winner_elapsed_seconds": elapsed,
        "validation_reversal": bool(
            not winner_validation.get("valid", False)
            or float(winner_validation.get("action", math.inf))
               >= float(old_validation.get("action", math.inf))
        ),
    }
    write_json(output_dir / "result.json", result)
    return result


__all__ = ["run_fast_benchmark", "run_gradient_convergence", "run_local_gradient_audit",
           "run_fast_refinement", "run_fast_multistart",
           "run_authoritative_selection", "run_fast_validation"]
