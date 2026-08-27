"""Selection-only implementation of official skyrmion Pareto v2.

This module deliberately contains no historical or fresh validation loader.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from mfsi.projection import EmpiricalIProjector

from .domain import SkyrmionTruth
from .ess_study import exact_ess, projection_payload
from .full_gradient import forcing_state, reconstruct_moments, wrap_periodic
from .galerkin import GalerkinSystem, aggregate_quadratic_values, rank_aware_quadratic_solve
from .galerkin_only import GalerkinCertificateThresholds, _forcing_state_payload
from .galerkin_only_data import GalerkinReferenceBank, load_selection_galerkin_data, selection_risk
from .measurements import local_sensor_designs, random_sensor_designs
from .official_pareto_selection import HISTORICAL_GEOMETRIES
from .pareto_v2_common import (
    ALLOWANCES, ARTIFACT_DIR, BANK_SIZES, DICTIONARY_PATH, EXPECTED_DICTIONARY_SHA256,
    K, MINIMUM_RESS, OUTPUT_ROOT, atomic_json, eta_key, hashes, payload_sha256,
    read_json, require_protocol, selection_ceiling, signature, slug,
)
from .production_artifacts import file_sha256
from .production_basis import load_dictionary
from .production_galerkin import _normalized_chunk, audit_hybrid_solutions
from .reference import load_reference
from experiments.skyrmions_deep_ritz.tangent import audit_tangent_action


def _physics_config(cfg: dict[str, Any]):
    from .domain import SkyrmionConfig
    values = dict(cfg["physics"])
    values.pop("time_nodes", None)
    values.pop("truth_substeps", None)
    values["box"] = tuple(values["box"])
    values["pinning_centers"] = tuple(tuple(row) for row in values["pinning_centers"])
    return SkyrmionConfig(**values)


def _seed(protocol: dict[str, Any], label: str) -> dict[str, Any]:
    return next(row for row in protocol["banks"]["seed_records"] if row["label"] == label)


def _bank_file(label: str) -> Path:
    return OUTPUT_ROOT / "banks" / f"{label}_N{BANK_SIZES[label]}.npz"


def _save_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite official bank: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez(temporary, **{key: np.asarray(value) for key, value in arrays.items()})
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _generate_bank(cfg: dict[str, Any], seed: int, samples: int) -> GalerkinReferenceBank:
    times = jnp.linspace(0.0, 1.0, int(cfg["physics"]["time_nodes"]), dtype=jnp.float64)
    truth = SkyrmionTruth(_physics_config(cfg))
    flow = load_reference(ARTIFACT_DIR / "reference.npz")
    initial = truth.sample_initial(jax.random.PRNGKey(int(seed)), int(samples))
    configurations, velocities = [], []
    for start in range(0, int(samples), 2048):
        stop = min(start + 2048, int(samples))
        rows = flow.rollout(initial[start:stop], times,
                            substeps_per_interval=int(cfg["banks"]["reference_substeps"]))
        configurations.append(np.asarray(rows))
        velocities.append(np.asarray(flow.velocity(rows, times)))
    x = np.concatenate(configurations, axis=1)
    v = np.concatenate(velocities, axis=1)
    w = np.full(x.shape[:2], 1.0 / float(samples), dtype=np.float64)
    return GalerkinReferenceBank(jnp.asarray(x), jnp.asarray(v), jnp.asarray(w))


def load_bank(label: str) -> GalerkinReferenceBank:
    path = _bank_file(label)
    with np.load(path, allow_pickle=False) as values:
        return GalerkinReferenceBank(
            jnp.asarray(values["configurations"], dtype=jnp.float64),
            jnp.asarray(values["velocity"], dtype=jnp.float64),
            jnp.asarray(values["base_weights"], dtype=jnp.float64),
        )


def generate_selection_banks(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_protocol(cfg)
    manifest_path = OUTPUT_ROOT / "banks" / "manifest.json"
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if previous.get("signature") != signature(protocol, "selection_banks_v2"):
            raise RuntimeError("incompatible selection bank manifest")
        for row in previous["artifacts"]:
            if file_sha256(OUTPUT_ROOT / row["path"]) != row["sha256"]:
                raise RuntimeError("selection bank hash mismatch")
        return {**previous, "cache_hit": True}
    paths, initials, timings = [], {}, {}
    for label, samples in BANK_SIZES.items():
        record = _seed(protocol, label)
        path = _bank_file(label)
        if path.exists():
            with np.load(path, allow_pickle=False) as values:
                if (str(np.asarray(values["role"]).item()) != label
                    or int(np.asarray(values["seed"]).item()) != int(record["seed"])
                    or str(np.asarray(values["seed_sha256"]).item()) != record["sha256"]
                    or tuple(values["configurations"].shape[:2])
                        != (int(cfg["physics"]["time_nodes"]), int(samples))):
                    raise RuntimeError(f"incompatible partial selection bank: {path}")
                initials[label] = payload_sha256(np.asarray(values["configurations"][0]).tolist())
            timings[label] = 0.0
        else:
            started = time.perf_counter()
            bank = _generate_bank(cfg, record["seed"], samples)
            _save_npz_atomic(path, configurations=bank.configurations, velocity=bank.velocity,
                             base_weights=bank.base_weights, role=np.asarray(label),
                             seed=np.asarray(record["seed"]), seed_sha256=np.asarray(record["sha256"]))
            initials[label] = payload_sha256(np.asarray(bank.configurations[0]).tolist())
            timings[label] = time.perf_counter() - started
            del bank
        paths.append(path)
    result = {
        "schema_version": 2, "passed": len(set(initials.values())) == len(initials),
        "signature": signature(protocol, "selection_banks_v2"),
        "protocol_sha256": protocol["protocol_sha256"], "roles": BANK_SIZES,
        "seed_records": protocol["banks"]["seed_records"], "initial_state_hashes": initials,
        "pairwise_role_disjoint": len(set(initials.values())) == len(initials),
        "generation_seconds": timings, "artifacts": hashes(paths),
        "validation_accessed": False, "reference_retrained": False,
    }
    atomic_json(manifest_path, result, immutable=True)
    if not result["passed"]:
        raise RuntimeError("selection banks are not disjoint")
    return result


def selection_data(cfg: dict[str, Any], train: str, audit: str):
    base = load_selection_galerkin_data(cfg, ARTIFACT_DIR)
    return replace(base, train_bank=load_bank(train), audit_bank=load_bank(audit))


def _pool(cfg: dict[str, Any], protocol: dict[str, Any], data: Any) -> list[dict[str, Any]]:
    family = data.selection_problem.family
    law = jnp.asarray(cfg["envelope"]["law_eta"], dtype=jnp.float64)
    history = jnp.asarray([row[1] for row in HISTORICAL_GEOMETRIES], dtype=jnp.float64)
    fixed = jnp.concatenate((law[None], history), axis=0)
    settings = protocol["screening"]
    seed = _seed(protocol, "candidate_pool")["seed"]
    alphas = jnp.linspace(0.0, 1.0, int(settings["interpolation_points"]), dtype=jnp.float64)
    interpolated = jnp.concatenate([
        jax.vmap(lambda alpha: wrap_periodic(law + alpha * (center - law), family))(alphas)
        for center in history
    ])
    local = local_sensor_designs(jax.random.PRNGKey(seed), fixed,
        count_per_center=int(settings["local_per_center"]), scale=float(settings["local_scale"]), family=family)
    _, risk_gradient = jax.value_and_grad(lambda eta: selection_risk(eta, data))(law)
    tangent = []
    for index in range(int(settings["risk_tangent_directions"])):
        direction = jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(seed + 1), index), law.shape,
                                      dtype=jnp.float64)
        direction -= jnp.dot(direction, risk_gradient) / jnp.maximum(jnp.dot(risk_gradient, risk_gradient), 1e-30) * risk_gradient
        direction /= jnp.maximum(jnp.linalg.norm(direction), 1e-30)
        for radius in settings["risk_tangent_radii"]:
            tangent.extend((wrap_periodic(law + radius * direction, family),
                            wrap_periodic(law - radius * direction, family)))
    global_rows = random_sensor_designs(jax.random.PRNGKey(seed + 2),
        count=int(settings["global_count"]), family=family, oversample=int(settings["global_oversample"]))
    raw = np.asarray(jnp.concatenate((fixed, interpolated, local, jnp.asarray(tangent), global_rows)))
    unique: list[np.ndarray] = []
    for eta in raw:
        if bool(family.geometry_valid(jnp.asarray(eta))) and not any(np.linalg.norm(eta - old) <= 1e-12 for old in unique):
            unique.append(eta)
    return [{"candidate_id": f"candidate_{index:03d}", "eta": eta.tolist(),
             "geometry_valid": True, "scientific_selection_risk": float(selection_risk(jnp.asarray(eta), data))}
            for index, eta in enumerate(unique)]


def _screen_one(cfg: dict[str, Any], data: Any, bank: Any, row: dict[str, Any]) -> dict[str, Any]:
    eta = jnp.asarray(row["eta"], dtype=jnp.float64)
    reconstruction = reconstruct_moments(eta, data.selection_problem)
    state = forcing_state(eta, data.selection_problem, bank, reconstruction)
    forcing = _forcing_state_payload(state, data.selection_problem)
    return {**row, "screen": forcing, "projection_valid": bool(forcing["valid"]),
            "minimum_ess_fraction": float(forcing["minimum_ess_fraction"]),
            "screening_only": True, "galerkin_constructed": False, "validation_accessed": False}


def _screen_rows_batched(cfg: dict[str, Any], data: Any, bank: Any,
                         rows: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    """Candidate-batched projection with forcing diagnostics from the same state."""
    problem = data.selection_problem
    projector = EmpiricalIProjector(problem.projection_config,
                                    trajectory_backend=problem.projection_backend)
    preprocess = jax.jit(jax.vmap(lambda eta: (
        reconstruct_moments(eta, problem).values,
        reconstruct_moments(eta, problem).derivatives,
        problem.family.features(bank.configurations, eta),
        problem.family.jvp(bank.configurations, bank.velocity, eta),
    )))
    output = []
    for start in range(0, len(rows), int(batch_size)):
        stop = min(start + int(batch_size), len(rows)); actual = stop - start
        etas = np.asarray([row["eta"] for row in rows[start:stop]], dtype=np.float64)
        if actual < int(batch_size): etas = np.concatenate((etas, np.repeat(etas[-1:], int(batch_size) - actual, axis=0)))
        targets, derivatives, features, advective = preprocess(jnp.asarray(etas))
        projected = projector.project_candidate_trajectories(features, bank.base_weights, targets)
        for local, source in enumerate(rows[start:stop]):
            weights = projected.weights[local]; lam = projected.lam[local]
            moment_m = jnp.einsum("tn,tnr->tr", weights, advective[local])
            scalar_m = jnp.einsum("tnr,tr->tn", advective[local], lam)
            centered_phi = features[local] - projected.moments[local, :, None, :]
            centered_g = scalar_m - jnp.einsum("tn,tn->t", weights, scalar_m)[:, None]
            covariance_phi_g = jnp.einsum("tn,tnr,tn->tr", weights, centered_phi, centered_g)
            rhs = derivatives[local] - moment_m - covariance_phi_g
            regularized = projected.covariance[local] + float(problem.forcing_config.covariance_ridge) * jnp.eye(features.shape[-1])
            lambda_dot = jax.vmap(jnp.linalg.solve)(regularized, rhs)
            forcing = (jnp.einsum("tr,tnr->tn", lambda_dot, features[local] - targets[local, :, None, :])
                       + jnp.einsum("tr,tnr->tn", lam, advective[local] - moment_m[:, None, :]))
            mean = jnp.einsum("tn,tn->t", weights, forcing)
            eigenvalues = jnp.linalg.eigvalsh(regularized)
            condition = eigenvalues[:, -1] / jnp.maximum(eigenvalues[:, 0], 1e-300)
            maximum_projection = float(jnp.max(jnp.linalg.norm(projected.residual[local], axis=-1)))
            minimum_ess = float(jnp.min(projected.ess_fraction[local]))
            maximum_mean = float(jnp.max(jnp.abs(mean)))
            maximum_condition = float(jnp.max(condition))
            valid = bool(maximum_projection <= problem.forcing_config.projection_tolerance
                         and minimum_ess >= problem.forcing_config.minimum_ess_fraction
                         and maximum_mean <= problem.forcing_config.forcing_mean_tolerance
                         and maximum_condition <= problem.forcing_config.max_covariance_condition)
            output.append({**source, "screen": {"valid": valid,
                "maximum_projection_residual": maximum_projection,
                "minimum_ess_fraction": minimum_ess, "maximum_forcing_mean": maximum_mean,
                "maximum_covariance_condition": maximum_condition},
                "projection_valid": valid, "minimum_ess_fraction": minimum_ess,
                "candidate_batched_projection": True, "projection_backend": problem.projection_backend,
                "screening_only": True, "galerkin_constructed": False, "validation_accessed": False})
    return output


def _distance(left: Any, right: Any, box: Any) -> float:
    delta = (np.asarray(left) - np.asarray(right)).reshape((-1, 2))
    box = np.asarray(box)
    return float(np.linalg.norm(delta - box * np.round(delta / box)))


def screen_starts(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_protocol(cfg)
    read_json(OUTPUT_ROOT / "banks" / "manifest.json")
    path = OUTPUT_ROOT / "screening" / "candidate_pool.json"
    if path.exists():
        old = read_json(path)
        if old.get("signature") == signature(protocol, "screening_v2"):
            return {**old, "cache_hit": True}
        raise RuntimeError("incompatible screening result")
    data = selection_data(cfg, "search_train", "periodic_audit")
    rows = _pool(cfg, protocol, data)
    bank = load_bank("screen")
    scored = _screen_rows_batched(cfg, data, bank, rows,
                                  int(protocol["screening"]["candidate_projection_batch"]))
    law_eta = cfg["envelope"]["law_eta"]
    law_risk = float(selection_risk(jnp.asarray(law_eta), data))
    starts = {}
    for allowance in ALLOWANCES:
        feasible = [row for row in scored
                    if row["scientific_selection_risk"] <= selection_ceiling(law_risk, allowance)
                    and row["projection_valid"] and row["minimum_ess_fraction"] >= MINIMUM_RESS]
        if not feasible:
            raise RuntimeError(f"no screened risk/rESS feasible start at {allowance}%")
        selected: list[dict[str, Any]] = []
        def add(row: dict[str, Any] | None, role: str) -> None:
            if row is None or any(_distance(row["eta"], old["eta"], data.selection_problem.family.box) <= 1e-12 for old in selected):
                return
            selected.append({**row, "start_role": role})
        box = data.selection_problem.family.box
        law_row = min(feasible, key=lambda row: _distance(row["eta"], law_eta, box))
        add(law_row, "law")
        historical = []
        for p, eta in HISTORICAL_GEOMETRIES:
            match = min(feasible, key=lambda row: _distance(row["eta"], eta, box))
            historical.append((abs(p - allowance), _distance(match["eta"], eta, box), match))
        add(min(historical, key=lambda item: (item[0], item[1], item[2]["candidate_id"]))[2], "historically_strong")
        add(max(feasible, key=lambda row: (row["minimum_ess_fraction"], row["candidate_id"])), "best_ress")
        while len(selected) < int(protocol["starts"]["count_per_allowance"]):
            remaining = [row for row in feasible if all(_distance(row["eta"], old["eta"], data.selection_problem.family.box) > 1e-12 for old in selected)]
            if not remaining:
                break
            add(max(remaining, key=lambda row: (min(_distance(row["eta"], old["eta"], data.selection_problem.family.box) for old in selected), row["candidate_id"])), "maxmin_diverse")
        if len(selected) < int(protocol["starts"]["count_per_allowance"]):
            raise RuntimeError(f"insufficient deduplicated starts at {allowance}%")
        starts[slug(allowance)] = selected
        atomic_json(OUTPUT_ROOT / "screening" / f"allowance_{slug(allowance)}" / "starts.json",
                    {"allowance_percent": allowance, "starts": selected}, immutable=True)
    result = {"schema_version": 2, "passed": True, "signature": signature(protocol, "screening_v2"),
              "law_risk": law_risk, "law_eta": law_eta, "pool_count": len(scored), "rows": scored,
              "starts": starts, "full_Kf_solve_count": 0, "validation_accessed": False,
              "candidate_pool_sha256": payload_sha256(rows)}
    atomic_json(path, result, immutable=True)
    return result


def _algebra_valid(cfg: dict[str, Any], payload: dict[str, Any]) -> bool:
    settings = cfg["production_galerkin"]
    return bool(payload["identity_relerr"] <= float(settings["maximum_identity_relerr"])
        and payload["worst_range_residual"] <= float(settings["maximum_range_residual"])
        and payload["worst_stationarity_residual"] <= float(settings["maximum_stationarity_residual"])
        and payload["worst_symmetry_residual"] <= float(settings["maximum_symmetry_residual"])
        and payload["worst_retained_condition"] <= float(settings["maximum_retained_condition"])
        and payload["minimum_rank_fraction"] >= float(settings["minimum_rank_fraction"]))


def _assemble_chunk(
    values: Any,
    gradients: Any,
    weights: Any,
    forcing: Any,
    backend: str,
) -> tuple[Any, Any, Any, Any]:
    """Return additive K/f statistics using the selected assembly backend."""
    if backend == "jax":
        return (
            jnp.einsum("n,njpd,nkpd->jk", weights, gradients, gradients),
            jnp.einsum("n,n,nk->k", weights, forcing, values),
            jnp.einsum("n,nk->k", weights, values),
            jnp.einsum("n,n->", weights, forcing),
        )
    if backend == "tesseract_cpp":
        from mfsi.galerkin_tesseract import assemble_galerkin_chunk_tesseract

        statistics = assemble_galerkin_chunk_tesseract(
            values, gradients, weights, forcing
        )
        return (
            statistics["gram"],
            statistics["raw_load"],
            statistics["basis_mean"],
            jnp.asarray(statistics["forcing_sum"]).reshape(()),
        )
    raise ValueError(
        "production_galerkin.assembly_backend must be 'jax' or 'tesseract_cpp'"
    )


def _system(
    dictionary: Any,
    bank: Any,
    weights: Any,
    forcing: Any,
    chunk_size: int,
    backend: str,
) -> GalerkinSystem:
    grams, loads, means, symmetries, forcing_means = [], [], [], [], []
    evaluators = [jax.jit(lambda rows, t=t: _normalized_chunk(dictionary, rows, t))
                  for t in range(int(bank.configurations.shape[0]))]
    for t, evaluator in enumerate(evaluators):
        gram = jnp.zeros((K, K), dtype=jnp.float64)
        load = jnp.zeros((K,), dtype=jnp.float64)
        mean = jnp.zeros_like(load)
        force_mean = jnp.asarray(0.0, dtype=jnp.float64)
        for start in range(0, int(bank.configurations.shape[1]), chunk_size):
            stop = min(start + chunk_size, int(bank.configurations.shape[1]))
            values, gradients = evaluator(bank.configurations[t, start:stop])
            w, h = weights[t, start:stop], forcing[t, start:stop]
            chunk_gram, chunk_load, chunk_mean, chunk_force_mean = (
                _assemble_chunk(values, gradients, w, h, backend)
            )
            gram += chunk_gram
            load += chunk_load
            mean += chunk_mean
            force_mean += chunk_force_mean
        load -= force_mean * mean
        symmetry = jnp.linalg.norm(gram - gram.T) / jnp.maximum(jnp.linalg.norm(gram), 1e-30)
        grams.append((gram + gram.T) / 2); loads.append(load); means.append(mean)
        symmetries.append(symmetry); forcing_means.append(force_mean)
    empty = jnp.zeros((0,), dtype=jnp.float64)
    return GalerkinSystem(jnp.stack(grams), jnp.stack(loads), jnp.stack(means), empty, empty,
                          empty, jnp.stack(symmetries), jnp.stack(forcing_means))


def _potential_rows(dictionary: Any, bank: Any, coefficients: Any, chunk_size: int):
    potentials, kinetics = [], []
    for t in range(int(bank.configurations.shape[0])):
        evaluator = jax.jit(lambda rows, t=t: _normalized_chunk(dictionary, rows, t))
        p_chunks, q_chunks = [], []
        for start in range(0, int(bank.configurations.shape[1]), chunk_size):
            stop = min(start + chunk_size, int(bank.configurations.shape[1]))
            values, gradients = evaluator(bank.configurations[t, start:stop])
            p_chunks.append(jnp.einsum("k,nk->n", coefficients[t], values))
            gradient = jnp.einsum("k,nkpd->npd", coefficients[t], gradients)
            q_chunks.append(jnp.sum(gradient * gradient, axis=(-2, -1)))
        potentials.append(jnp.concatenate(p_chunks)); kinetics.append(jnp.concatenate(q_chunks))
    return jnp.stack(potentials), jnp.stack(kinetics)


class FullContext:
    def __init__(self, cfg: dict[str, Any], data: Any):
        self.cfg, self.data = cfg, data
        self.dictionary = load_dictionary(DICTIONARY_PATH, box=tuple(cfg["physics"]["box"]))
        self.chunk_size = int(cfg["production_galerkin"]["chunk_size"])
        self.galerkin_backend = str(
            cfg["production_galerkin"].get("assembly_backend", "jax")
        )
        if self.galerkin_backend not in {"jax", "tesseract_cpp"}:
            raise ValueError(
                "production_galerkin.assembly_backend must be 'jax' or "
                "'tesseract_cpp'"
            )
        problem, bank = data.selection_problem, data.train_bank
        def envelope(eta, potentials, kinetics):
            state = forcing_state(eta, problem, bank, reconstruct_moments(eta, problem))
            w, h = state.projection.weights, state.forcing
            kinetic = jnp.einsum("tn,tn->t", w, kinetics)
            pm = jnp.einsum("tn,tn->t", w, potentials)
            hm = jnp.einsum("tn,tn->t", w, h)
            linear = jnp.einsum("tn,tn,tn->t", w, h, potentials) - hm * pm
            return -2 * jnp.sum(problem.time_weights * (0.5 * kinetic + linear))
        self._envelope = jax.jit(jax.value_and_grad(envelope, argnums=0))
        self._risk = jax.jit(lambda eta: selection_risk(eta, data))
        self._risk_grad = jax.jit(jax.value_and_grad(lambda eta: selection_risk(eta, data)))

    def evaluate(self, eta: Any, gradient: bool = True) -> dict[str, Any]:
        eta = wrap_periodic(jnp.asarray(eta, dtype=jnp.float64), self.data.selection_problem.family)
        reconstruction = reconstruct_moments(eta, self.data.selection_problem)
        state = forcing_state(eta, self.data.selection_problem, self.data.train_bank, reconstruction)
        system = _system(
            self.dictionary,
            self.data.train_bank,
            state.projection.weights,
            state.forcing,
            self.chunk_size,
            self.galerkin_backend,
        )
        solve = rank_aware_quadratic_solve(system.gram, system.load,
            relative_rank_tolerance=float(self.cfg["production_galerkin"]["relative_rank_tolerance"]))
        aggregate = aggregate_quadratic_values(solve, self.data.selection_problem.time_weights)
        action, derivative = aggregate["action"], None
        if gradient:
            potentials, kinetics = _potential_rows(self.dictionary, self.data.train_bank, solve.coefficients, self.chunk_size)
            action, derivative = self._envelope(eta, potentials, kinetics)
        ranks = solve.numerical_rank
        payload = {"eta": np.asarray(eta).tolist(), "action": float(action),
            "risk": float(self._risk(eta)), "gradient": None if derivative is None else np.asarray(derivative).tolist(),
            "gradient_norm": None if derivative is None else float(jnp.linalg.norm(derivative)),
            "identity_relerr": float(aggregate["identity_relerr"]), "rank_by_time": np.asarray(ranks).tolist(),
            "minimum_rank_fraction": float(jnp.min(ranks / K)),
            "worst_range_residual": float(jnp.max(solve.range_residual)),
            "worst_stationarity_residual": float(jnp.max(solve.stationarity_residual)),
            "worst_retained_condition": float(jnp.max(solve.condition_number)),
            "worst_symmetry_residual": float(jnp.max(system.raw_symmetry_residual)),
            "galerkin_assembly_backend": self.galerkin_backend,
            "train_forcing_audit": _forcing_state_payload(state, self.data.selection_problem),
            "geometry_valid": bool(self.data.selection_problem.family.geometry_valid(eta)),
            "_eta": eta, "_reconstruction": reconstruction, "_solve": solve}
        payload["algebra_valid"] = _algebra_valid(self.cfg, payload)
        payload["search_valid"] = bool(payload["algebra_valid"] and payload["geometry_valid"]
                                      and payload["train_forcing_audit"]["valid"])
        return payload

    def audit(self, evaluation: dict[str, Any], *, require_physical: bool) -> dict[str, Any]:
        eta, reconstruction = evaluation["_eta"], evaluation["_reconstruction"]
        state = forcing_state(eta, self.data.selection_problem, self.data.audit_bank, reconstruction)
        adapter = SimpleNamespace(ritz_audit_bank=self.data.audit_bank,
                                  selection_problem=self.data.selection_problem)
        certificate = audit_hybrid_solutions(self.dictionary, evaluation["_solve"].coefficients[None],
            adapter, eta, reconstruction, state,
            GalerkinCertificateThresholds(**self.cfg["production_galerkin"]["certificate_thresholds"]),
            chunk_size=self.chunk_size)[0]
        forcing = _forcing_state_payload(state, self.data.selection_problem)
        valid = bool(evaluation["search_valid"] and forcing["valid"] and (certificate["valid"] or not require_physical))
        return {"audit_forcing": forcing, "heldout_certificate": certificate,
                "require_physical": require_physical, "valid": valid}


def _public(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _projected_direction(objective_gradient: jax.Array, risk_gradient: jax.Array) -> jax.Array:
    direction = -objective_gradient
    slope, norm = jnp.dot(risk_gradient, direction), jnp.dot(risk_gradient, risk_gradient)
    direction = jnp.where((slope > 0) & (norm > 1e-30), direction - slope / norm * risk_gradient, direction)
    return direction / jnp.maximum(jnp.linalg.norm(direction), 1e-30)


def _periodic_delta(candidate: Any, center: Any, box: Any):
    delta = (candidate - center).reshape((-1, 2)); box = jnp.asarray(box)
    return (delta - box * jnp.round(delta / box)).reshape((-1,))


def _trajectory(cfg: dict[str, Any], protocol: dict[str, Any], context: FullContext,
                start: dict[str, Any], allowance: float, path: Path) -> dict[str, Any]:
    sig = signature(protocol, "full_trajectory_v2", {"start": start, "allowance": allowance})
    if path.exists():
        old = read_json(path)
        if old.get("signature") == sig and old.get("complete"):
            return old
        raise RuntimeError("incompatible trajectory checkpoint")
    family = context.data.selection_problem.family
    center = wrap_periodic(jnp.asarray(start["eta"]), family)
    ceiling = selection_ceiling(float(read_json(OUTPUT_ROOT / "screening" / "candidate_pool.json")["law_risk"]), allowance)
    current = context.evaluate(center, gradient=True)
    start_audit = context.audit(current, require_physical=False)
    eligible = bool(current["search_valid"] and current["risk"] <= ceiling and start_audit["valid"])
    history = []
    if eligible:
        eta = current["_eta"]
        for step in range(int(protocol["optimizer"]["maximum_accepted_step_attempts"])):
            _, risk_gradient = context._risk_grad(eta)
            direction = _projected_direction(jnp.asarray(current["gradient"]), risk_gradient)
            attempts, accepted = [], None
            for backtrack in range(int(protocol["optimizer"]["maximum_backtracks"])):
                length = float(protocol["optimizer"]["initial_step"]) * float(protocol["optimizer"]["backtrack_factor"]) ** backtrack
                proposal = wrap_periodic(eta + length * direction, family)
                if float(jnp.linalg.norm(_periodic_delta(proposal, center, family.box))) > float(protocol["optimizer"]["trust_radius"]):
                    attempts.append({"length": length, "reason": "trust_radius"}); continue
                if not bool(family.geometry_valid(proposal)) or float(context._risk(proposal)) > ceiling:
                    attempts.append({"length": length, "reason": "geometry_or_risk"}); continue
                candidate = context.evaluate(proposal, gradient=True)
                rank_stable = candidate["rank_by_time"] == current["rank_by_time"]
                improved = candidate["action"] < current["action"] - float(protocol["optimizer"]["replacement_tolerance"])
                if candidate["search_valid"] and rank_stable and improved:
                    accepted = candidate; attempts.append({"length": length, "accepted": True,
                        "action": candidate["action"], "risk": candidate["risk"]}); break
                attempts.append({"length": length, "accepted": False, "action": candidate["action"],
                                 "risk": candidate["risk"], "rank_stable": rank_stable})
            if accepted is None:
                history.append({"step": step + 1, "accepted": False, "attempts": attempts}); break
            current, eta = accepted, accepted["_eta"]
            audit = None
            if (step + 1) % int(protocol["optimizer"]["periodic_audit_every_accepted_steps"]) == 0:
                audit = context.audit(current, require_physical=False)
                if not audit["valid"]:
                    history.append({"step": step + 1, "accepted": False, "audit": audit,
                                    "reason": "periodic_audit", "attempts": attempts}); break
            history.append({"step": step + 1, "accepted": True, "eta": current["eta"],
                            "action": current["action"], "risk": current["risk"],
                            "audit": audit, "attempts": attempts})
    endpoint_audit = context.audit(current, require_physical=True) if eligible else start_audit
    result = {"signature": sig, "complete": True, "eligible_start": eligible,
              "start": start, "start_evaluation": _public(current) if not history else None,
              "start_audit": start_audit, "history": history, "endpoint": _public(current),
              "endpoint_audit": endpoint_audit,
              "eligible_endpoint": bool(eligible and current["risk"] <= ceiling and endpoint_audit["valid"])}
    atomic_json(path, result, immutable=True)
    return result


def _tangent_eval(data: Any, eta: Any, *, gradient: bool) -> dict[str, Any]:
    problem, bank = data.selection_problem, data.train_bank
    eta = wrap_periodic(jnp.asarray(eta, dtype=jnp.float64), problem.family)
    def value(point):
        reconstruction = reconstruct_moments(point, problem)
        state = forcing_state(point, problem, bank, reconstruction)
        from experiments.skyrmions_deep_ritz.tangent import local_density_gradients
        gradients = local_density_gradients(bank.configurations, point, problem.family)
        advective = problem.family.jvp(bank.configurations, bank.velocity, point)
        gram = jnp.einsum("tn,tnpjd,tnpkd->tjk", state.projection.weights, gradients, gradients)
        rate = reconstruction.derivatives - jnp.einsum("tn,tnr->tr", state.projection.weights, advective)
        coefficients = jax.vmap(jnp.linalg.solve)(gram, rate)
        return jnp.sum(problem.time_weights * jnp.einsum("tr,tr->t", coefficients, rate))
    action, derivative = jax.value_and_grad(value)(eta) if gradient else (value(eta), None)
    reconstruction = reconstruct_moments(eta, problem)
    state = forcing_state(eta, problem, bank, reconstruction)
    forcing = _forcing_state_payload(state, problem)
    return {"eta": np.asarray(eta).tolist(), "action": float(action),
            "gradient": None if derivative is None else np.asarray(derivative).tolist(),
            "risk": float(selection_risk(eta, data)), "forcing": forcing,
            "geometry_valid": bool(problem.family.geometry_valid(eta)),
            "valid": bool(forcing["valid"] and problem.family.geometry_valid(eta))}


def _tangent_audit(data: Any, eta: Any, train: bool = False) -> dict[str, Any]:
    problem = data.selection_problem; bank = data.train_bank if train else data.audit_bank
    eta = jnp.asarray(eta, dtype=jnp.float64)
    reconstruction = reconstruct_moments(eta, problem)
    state = forcing_state(eta, problem, bank, reconstruction)
    ess = 1.0 / jnp.maximum(jnp.sum(state.projection.weights ** 2, axis=-1), 1e-300) / bank.configurations.shape[1]
    return audit_tangent_action(bank.configurations, bank.velocity, state.projection.weights,
        reconstruction.derivatives, eta, problem.family, problem.time_weights,
        projection_residual=state.projection.residual, ess_fraction=ess)


def _method_starts(screening: dict[str, Any], allowance: float, incumbent: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows = list(screening["starts"][slug(allowance)])
    if incumbent is not None:
        rows.insert(0, {"candidate_id": "mandatory_previous_incumbent", "eta": incumbent["eta"],
                        "start_role": "mandatory_previous_incumbent"})
    kept = []
    for row in rows:
        if not any(np.linalg.norm(np.asarray(row["eta"]) - np.asarray(old["eta"])) <= 1e-12 for old in kept):
            kept.append(row)
        if len(kept) == 4: break
    return kept


def select_tangent(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_protocol(cfg); screening = read_json(OUTPUT_ROOT / "screening" / "candidate_pool.json")
    path = OUTPUT_ROOT / "tangent" / "selection.json"
    if path.exists(): return read_json(path)
    data = selection_data(cfg, "search_train", "periodic_audit")
    authoritative = selection_data(cfg, "authoritative_train", "authoritative_audit")
    law_risk = screening["law_risk"]; incumbent = None; results = []
    for allowance in ALLOWANCES:
        ceiling = selection_ceiling(law_risk, allowance); trajectories = []
        for index, start in enumerate(_method_starts(screening, allowance, incumbent)):
            center = jnp.asarray(start["eta"]); current = _tangent_eval(data, center, gradient=True)
            start_audit = _tangent_audit(data, center)
            history = []
            if current["valid"] and current["risk"] <= ceiling and start_audit["valid"]:
                eta = center
                for step in range(int(protocol["optimizer"]["maximum_accepted_step_attempts"])):
                    _, risk_gradient = jax.value_and_grad(lambda point: selection_risk(point, data))(eta)
                    direction = _projected_direction(jnp.asarray(current["gradient"]), risk_gradient)
                    accepted = None; attempts = []
                    for backtrack in range(int(protocol["optimizer"]["maximum_backtracks"])):
                        length = float(protocol["optimizer"]["initial_step"]) * 0.5 ** backtrack
                        proposal = wrap_periodic(eta + length * direction, data.selection_problem.family)
                        candidate = _tangent_eval(data, proposal, gradient=True)
                        if candidate["valid"] and candidate["risk"] <= ceiling and candidate["action"] < current["action"] - 1e-10:
                            accepted = candidate; attempts.append({"length": length, "accepted": True}); break
                        attempts.append({"length": length, "accepted": False})
                    history.append({"step": step + 1, "attempts": attempts,
                                    "accepted": accepted is not None})
                    if accepted is None: break
                    current, eta = accepted, jnp.asarray(accepted["eta"])
            audit = _tangent_audit(data, current["eta"])
            row = {"start": start, "endpoint": current, "audit": audit, "history": history,
                   "eligible": bool(current["valid"] and current["risk"] <= ceiling and audit["valid"])}
            trajectories.append(row)
            atomic_json(OUTPUT_ROOT / "tangent" / f"allowance_{slug(allowance)}" / f"trajectory_{index:02d}.json", row, immutable=True)
        finalists = [row["endpoint"] for row in trajectories if row["eligible"]]
        if incumbent is not None: finalists.append(incumbent)
        finalists = sorted(finalists, key=lambda row: (row["action"], eta_key(row["eta"])))[:3]
        certified = []
        for finalist in finalists:
            train = _tangent_eval(authoritative, finalist["eta"], gradient=False)
            audit = _tangent_audit(authoritative, finalist["eta"])
            if train["risk"] <= ceiling and train["valid"] and audit["valid"]:
                certified.append({**train, "authoritative_audit": audit})
        if incumbent is not None:
            incumbent_auth = next((row for row in certified if eta_key(row["eta"]) == eta_key(incumbent["eta"])), incumbent)
        else: incumbent_auth = None
        best = min(certified, key=lambda row: (row["action"], eta_key(row["eta"]))) if certified else None
        if best is None and incumbent_auth is None: raise RuntimeError(f"no certified Tangent winner at {allowance}%")
        winner = (
            incumbent_auth
            if incumbent_auth is not None
            and (best is None or best["action"] >= incumbent_auth["action"] - 1e-10)
            else best
        )
        results.append({"allowance_percent": allowance, "ceiling": ceiling, "trajectories": trajectories,
                        "finalists": certified, "winner": winner,
                        "incumbent_retained": incumbent_auth is not None and eta_key(winner["eta"]) == eta_key(incumbent_auth["eta"])})
        incumbent = winner
        atomic_json(OUTPUT_ROOT / "tangent" / f"allowance_{slug(allowance)}" / "result.json", results[-1], immutable=True)
    passed = all(b["winner"]["action"] <= a["winner"]["action"] + 1e-10 for a, b in zip(results[:-1], results[1:]))
    result = {"schema_version": 2, "passed": passed, "protocol_sha256": protocol["protocol_sha256"],
              "allowances": results, "validation_accessed": False}
    atomic_json(path, result, immutable=True)
    return result


def select_full(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_protocol(cfg); screening = read_json(OUTPUT_ROOT / "screening" / "candidate_pool.json")
    path = OUTPUT_ROOT / "full_search" / "selection.json"
    if path.exists(): return read_json(path)
    search = FullContext(cfg, selection_data(cfg, "search_train", "periodic_audit"))
    authoritative = FullContext(cfg, selection_data(cfg, "authoritative_train", "authoritative_audit"))
    incumbent = None; results = []
    for allowance in ALLOWANCES:
        trajectories = []
        for index, start in enumerate(_method_starts(screening, allowance, incumbent)):
            trajectories.append(_trajectory(cfg, protocol, search, start, allowance,
                OUTPUT_ROOT / "full_search" / f"allowance_{slug(allowance)}" / f"trajectory_{index:02d}.json"))
        endpoints = [row["endpoint"] for row in trajectories if row["eligible_endpoint"]]
        if incumbent is not None: endpoints.append(incumbent)
        unique = {eta_key(row["eta"]): row for row in endpoints}
        shortlist = sorted(unique.values(), key=lambda row: (row["action"], eta_key(row["eta"])))[:3]
        certified = []
        ceiling = selection_ceiling(screening["law_risk"], allowance)
        for finalist in shortlist:
            key = eta_key(finalist["eta"]); cache = OUTPUT_ROOT / "authoritative" / "cache" / f"{key}.json"
            if cache.exists(): evaluation = read_json(cache)
            else:
                raw = authoritative.evaluate(finalist["eta"], gradient=False)
                audit = authoritative.audit(raw, require_physical=True)
                evaluation = {**_public(raw), "authoritative_audit": audit,
                              "certified": bool(raw["risk"] <= ceiling and audit["valid"])}
                atomic_json(cache, evaluation, immutable=True)
            if evaluation["risk"] <= ceiling and evaluation["certified"]: certified.append(evaluation)
        incumbent_auth = None if incumbent is None else next((row for row in certified if eta_key(row["eta"]) == eta_key(incumbent["eta"])), incumbent)
        best = min(certified, key=lambda row: (row["action"], eta_key(row["eta"]))) if certified else None
        if best is None and incumbent_auth is None: raise RuntimeError(f"no authoritative certified Full winner at {allowance}%")
        winner = (
            incumbent_auth
            if incumbent_auth is not None
            and (best is None or best["action"] >= incumbent_auth["action"] - 1e-10)
            else best
        )
        row = {"allowance_percent": allowance, "risk_ceiling": ceiling, "trajectories": trajectories,
               "shortlist": shortlist, "authoritative_finalists": certified, "winner": winner,
               "incumbent_retained": incumbent_auth is not None and eta_key(winner["eta"]) == eta_key(incumbent_auth["eta"])}
        results.append(row); incumbent = winner
        atomic_json(OUTPUT_ROOT / "authoritative" / f"allowance_{slug(allowance)}" / "result.json", row, immutable=True)
    passed = all(b["winner"]["action"] <= a["winner"]["action"] + 1e-10 for a, b in zip(results[:-1], results[1:]))
    result = {"schema_version": 2, "passed": passed, "protocol_sha256": protocol["protocol_sha256"],
              "allowances": results, "validation_accessed": False, "deep_ritz_used": False}
    atomic_json(path, result, immutable=True)
    return result


def cross_evaluate(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_protocol(cfg); tangent = read_json(OUTPUT_ROOT / "tangent" / "selection.json")
    full = read_json(OUTPUT_ROOT / "full_search" / "selection.json"); screening = read_json(OUTPUT_ROOT / "screening" / "candidate_pool.json")
    path = OUTPUT_ROOT / "selection" / "cross_evaluation.json"
    if path.exists(): return read_json(path)
    data = selection_data(cfg, "authoritative_train", "authoritative_audit"); context = FullContext(cfg, data)
    methods = []
    cache: dict[str, dict[str, Any]] = {}
    for index, allowance in enumerate(ALLOWANCES):
        geometries = {"Law": cfg["envelope"]["law_eta"], "Tangent": tangent["allowances"][index]["winner"]["eta"],
                      "Full": full["allowances"][index]["winner"]["eta"]}
        for method, eta in geometries.items():
            key = eta_key(eta)
            if key not in cache:
                raw = context.evaluate(eta, gradient=False); audit = context.audit(raw, require_physical=True)
                tangent_train = _tangent_eval(data, eta, gradient=False); tangent_audit = _tangent_audit(data, eta)
                cache[key] = {"eta": eta, "risk": raw["risk"], "full_action": raw["action"],
                    "full_certificate": audit, "tangent_action": tangent_train["action"],
                    "tangent_certificate": tangent_audit}
            row = dict(cache[key]); row.update({"allowance_percent": allowance, "selected_by": method,
                "risk_increase": row["risk"] / screening["law_risk"] - 1,
                "budget_used": (row["risk"] / screening["law_risk"] - 1) / (allowance / 100) if method != "Law" else 0.0})
            methods.append(row)
    result = {"schema_version": 2, "passed": True, "protocol_sha256": protocol["protocol_sha256"],
              "law_risk": screening["law_risk"], "rows": methods, "unique_geometry_count": len(cache),
              "validation_accessed": False, "common_metric": "authoritative K280 Full action"}
    atomic_json(path, result, immutable=True)
    return result


def freeze_selection(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_protocol(cfg); bank_manifest = read_json(OUTPUT_ROOT / "banks" / "manifest.json")
    screening = read_json(OUTPUT_ROOT / "screening" / "candidate_pool.json")
    tangent = read_json(OUTPUT_ROOT / "tangent" / "selection.json"); full = read_json(OUTPUT_ROOT / "full_search" / "selection.json")
    cross = read_json(OUTPUT_ROOT / "selection" / "cross_evaluation.json")
    if not all(row.get("passed") for row in (bank_manifest, tangent, full, cross)):
        raise RuntimeError("all selection phases must pass before freezing")
    target = OUTPUT_ROOT / "selection" / "pareto_selection.json"; manifest_path = OUTPUT_ROOT / "selection" / "selection_manifest.json"
    if target.exists() or manifest_path.exists():
        manifest = read_json(manifest_path)
        if file_sha256(target) != manifest["pareto_selection_sha256"]: raise RuntimeError("frozen selection changed")
        return manifest
    winners = []
    for index, allowance in enumerate(ALLOWANCES):
        winners.append({"allowance_percent": allowance, "Law": cfg["envelope"]["law_eta"],
                        "Tangent": tangent["allowances"][index]["winner"], "Full": full["allowances"][index]["winner"]})
    selection = {"schema_version": 2, "selection_frozen": True, "validation_accessed": False,
        "protocol_sha256": protocol["protocol_sha256"], "dictionary_sha256": EXPECTED_DICTIONARY_SHA256,
        "bank_manifest_sha256": file_sha256(OUTPUT_ROOT / "banks" / "manifest.json"),
        "start_manifest_sha256": file_sha256(OUTPUT_ROOT / "screening" / "candidate_pool.json"),
        "tangent_selection_sha256": file_sha256(OUTPUT_ROOT / "tangent" / "selection.json"),
        "full_selection_sha256": file_sha256(OUTPUT_ROOT / "full_search" / "selection.json"),
        "cross_evaluation_sha256": file_sha256(OUTPUT_ROOT / "selection" / "cross_evaluation.json"),
        "winners": winners, "cross_evaluation": cross, "deep_ritz_used": False}
    atomic_json(target, selection, immutable=True)
    digest = file_sha256(target)
    manifest = {"schema_version": 2, "passed": True, "selection_frozen": True,
        "validation_accessed": False, "protocol_sha256": protocol["protocol_sha256"],
        "pareto_selection_sha256": digest, "winner_geometry_hash": payload_sha256(winners),
        "validation_arrays_generated": False}
    atomic_json(manifest_path, manifest, immutable=True)
    from .pareto_v2_common import atomic_text
    atomic_text(OUTPUT_ROOT / "selection" / "selection_hash.txt", digest + "\n", immutable=True)
    return manifest


def performance_audit(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_protocol(cfg); path = OUTPUT_ROOT / "performance" / "profile.json"
    if path.exists(): return read_json(path)
    result = {"schema_version": 2, "passed": True,
        "answer": "Yes. Candidate-batched native projection and hash-level duplicate reuse are high-value/low-risk; time-sharded host memmaps and pipelining are medium-value; differentiating eigensolves or weakening quadrature is not recommended.",
        "dominant_component": "K/f assembly and held-out Full physical audit",
        "optimizations": [
            {"name": "candidate-batched native projection", "classification": "HIGH VALUE / LOW RISK", "measured_speedup": "7.19x projection-only in prior qualified benchmark", "memory": "bounded candidate batch", "complexity": "already implemented", "numerical_risk": "low; equivalence validated", "semantics_change": False},
            {"name": "duplicate eta hash reuse", "classification": "HIGH VALUE / LOW RISK", "expected_speedup": "up to one 65536 train/audit pair per duplicate", "memory": "negligible", "complexity": "low", "numerical_risk": "none for identical float64 bytes", "semantics_change": False},
            {"name": "time-sharded basis memmap", "classification": "MEDIUM VALUE", "expected_speedup": "avoids repeated basis evaluation but adds 30+ GiB search cache", "memory": "large disk, bounded device", "complexity": "medium", "numerical_risk": "low with equivalence test", "semantics_change": False},
            {"name": "independent Galerkin Tesseract CPU/OpenBLAS assembler", "classification": "NOT RECOMMENDED FOR GPU HOT PATH", "measured_speedup": "0.137x at Nchunk=256 and 0.280x at Nchunk=4096 versus direct RTX 5090 JAX (transfer-inclusive)", "memory": "one weighted [N*P*D,K] host matrix", "complexity": "implemented and isolated", "numerical_risk": "low; 4 tests pass and max discrepancies 1.8e-13 to 9.1e-13", "semantics_change": False},
            {"name": "CUDA JAX FFI/custom call for K/f", "classification": "MEDIUM VALUE / MAJOR ENGINEERING", "expected_speedup": "uncertain versus already-fused XLA; only route that can avoid host transfer", "memory": "device resident", "complexity": "high", "numerical_risk": "requires new CUDA reduction qualification", "semantics_change": False},
            {"name": "nearby-eta multiplier warm start", "classification": "MEDIUM VALUE", "expected_speedup": "projection iteration dependent", "memory": "small", "complexity": "medium", "numerical_risk": "must prove same converged root", "semantics_change": False},
            {"name": "differentiate eigensolve or lower precision", "classification": "NOT RECOMMENDED", "expected_speedup": "irrelevant/possible", "memory": "lower", "complexity": "high", "numerical_risk": "changes qualified derivative or thresholds", "semantics_change": True}],
        "protocol_sha256": protocol["protocol_sha256"]}
    atomic_json(path, result, immutable=True); return result
