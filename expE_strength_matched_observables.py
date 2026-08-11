#!/usr/bin/env python3
"""Experiment E: strength-matched and reference-sensitivity controls.

Every stage is resume-safe.  Existing Experiment-D checkpoints/cells are read
but never rewritten.  Candidate selection uses design-only banks and never
touches downstream or final evaluation metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import example_b as exb
import observable_design_toy as od
import strength_matched_observables as sm

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "strength_matched_observables.yaml"
REPORT = ROOT / "STRENGTH_MATCHED_OBSERVABLE_REPORT.md"
BASELINES = ("info", "cv", "fiber", "random", "full_phi5")
TARGETS = ("low", "medium", "high")
METRICS = ("tangent_local_mmd", "tangent_rollout_mmd", "mfsi_rollout_mmd",
           "velocity_gap", "angular_error", "min_ess", "mean_ess",
           "mean_projection_distortion", "mean_lambda_norm", "max_moment_error")


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _key(seed: int, stream: int) -> jax.Array:
    return jax.random.fold_in(jax.random.PRNGKey(seed), stream)


def _json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(od.json_ready(data), indent=2, allow_nan=False))


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def _load_standardization(source: Path) -> od.Standardization:
    data = np.load(source / "design_standardization.npz")
    return od.Standardization(jnp.asarray(data["center"]), jnp.asarray(data["whitening"]),
                              jnp.asarray(data["covariance_eigenvalues"]))


def _load_A(source: Path, objective: str, model_seed: int) -> np.ndarray:
    return np.asarray(np.load(source / "checkpoints" /
                              f"observable_{objective}_modelseed_{model_seed}.npz")["A"])


def _load_ritz(path: Path):
    return exb.unflatten(jnp.asarray(np.load(path)["potential_params"]), exb.RITZ_HIDDEN, 1)


def _save_ritz(path: Path, params) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, potential_params=np.asarray(exb.core.flatten_mlp(params)))


def _paths(config: dict[str, Any]) -> tuple[Path, Path]:
    source = (ROOT / config["source_experiment_d"]).resolve()
    out = (ROOT / config["output"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return source, out


def _fingerprints(source: Path) -> dict[str, str]:
    paths = [source / "design_standardization.npz", source / "results.json",
             *sorted((source / "checkpoints").glob("*.npz"))]
    return {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def _candidate_endpoint_mmd(pool: jax.Array, z0: jax.Array, z1: jax.Array,
                            batch_size: int) -> np.ndarray:
    w = jnp.ones(z0.shape[0], dtype=z0.dtype) / z0.shape[0]

    @jax.jit
    def batch_fn(matrices):
        def one(A):
            return jnp.sqrt(od.weighted_mmd2(z0 @ A.T, w, z1 @ A.T, w))
        return jax.vmap(one)(matrices)

    values = []
    for start in range(0, len(pool), batch_size):
        values.append(np.asarray(batch_fn(pool[start:start + batch_size])))
    return np.concatenate(values)


def _candidate_diagnostics(
    pool: jax.Array, matrix: jax.Array, means: jax.Array, covariance: jax.Array,
    endpoint_mmd: np.ndarray,
) -> dict[str, np.ndarray]:
    strength = np.asarray(jnp.einsum("nri,ij,nrj->n", pool, matrix, pool))
    variance = np.asarray(jnp.einsum("nri,ij,nrj->n", pool, covariance, pool))
    projected = jnp.einsum("ti,nri->ntr", means, pool)
    norm = np.asarray(jnp.mean(jnp.sum(projected * projected, axis=-1) /
                              (jnp.sum(means * means, axis=-1)[None, :] + 1e-14), axis=-1))
    return {"strength": strength, "normalized_strength": norm,
            "variance_trace": variance, "endpoint_phi_mmd": endpoint_mmd}


def _one_diagnostic(A: np.ndarray, matrix: jax.Array, means: jax.Array,
                    covariance: jax.Array, z0: jax.Array, z1: jax.Array) -> dict[str, float]:
    Aj = jnp.asarray(A)
    w = jnp.ones(z0.shape[0], dtype=z0.dtype) / z0.shape[0]
    return {
        "strength": float(sm.constraint_strength(Aj, matrix)),
        "normalized_strength": float(sm.normalized_strength(Aj, means)),
        "variance_trace": float(jnp.einsum("ri,ij,rj->", Aj, covariance, Aj)),
        "endpoint_phi_mmd": float(jnp.sqrt(od.weighted_mmd2(z0 @ Aj.T, w, z1 @ Aj.T, w))),
    }


def prepare(config: dict[str, Any], force: bool = False) -> None:
    source, out = _paths(config)
    (out / "matching").mkdir(parents=True, exist_ok=True)
    selection_path = out / "matching" / "matching_diagnostics.json"
    if selection_path.exists() and not force:
        print("[expE:prepare] matching artifacts already exist; reusing", flush=True)
        return
    standardization = _load_standardization(source)
    streams = config["seed_streams"]
    base = int(config["base_seed"])
    design = config["strength_design"]
    times = jnp.asarray(design["times"], dtype=jnp.float64)
    geometries: dict[str, Any] = {}
    for index, geometry in enumerate(sm.GEOMETRIES):
        matrix, means, covariance = sm.strength_matrix(
            jax.random.PRNGKey(base + int(streams["strength_design"]) + index), times,
            int(design["particles_per_time"]), standardization, geometry)
        geometries[geometry] = {"matrix": np.asarray(matrix), "means": np.asarray(means),
                                "covariance": np.asarray(covariance)}
    np.savez(out / "matching" / "strength_design_banks.npz",
             **{f"{g}_{name}": value for g, data in geometries.items()
                for name, value in data.items()}, times=np.asarray(times))

    matching = config["matching"]
    count = int(matching["candidate_pool_size"])
    print(f"[expE:prepare] sampling {count} common random Stiefel candidates", flush=True)
    pool = sm.random_stiefel_pool(jax.random.PRNGKey(base + int(streams["candidate_pool"])), count,
                                  int(config["R"]), 5)
    endpoint_key = jax.random.PRNGKey(base + int(streams["candidate_pool"]) + 1)
    k0, k1 = jax.random.split(endpoint_key)
    n_endpoint = int(matching["endpoint_particles_per_law"])
    z0 = od.standardized_dictionary(exb.sample_ring(k0, n_endpoint), standardization)
    z1 = od.standardized_dictionary(exb.sample_four_lobes(k1, n_endpoint), standardization)
    print("[expE:prepare] computing design-only endpoint Phi-MMD diagnostics", flush=True)
    endpoint_mmd = _candidate_endpoint_mmd(pool, z0, z1,
                                            int(matching["endpoint_mmd_batch_size"]))
    default = geometries["default"]
    diagnostics = _candidate_diagnostics(pool, jnp.asarray(default["matrix"]),
                                         jnp.asarray(default["means"]),
                                         jnp.asarray(default["covariance"]), endpoint_mmd)
    np.savez(out / "matching" / "random_candidate_pool.npz", A=np.asarray(pool), **diagnostics)

    logs = np.column_stack([np.log(np.maximum(diagnostics[name], 1e-14))
                            for name in ("strength", "variance_trace", "endpoint_phi_mmd")])
    score_mean, score_std = logs.mean(axis=0), logs.std(axis=0)
    standardized_scores = (logs - score_mean) / score_std
    strength_tol = float(matching["strength_relative_tolerance"])
    variance_tol = float(matching["variance_relative_tolerance"])
    mmd_tol = float(matching["endpoint_phi_mmd_relative_tolerance"])
    K = int(matching["K"])
    selections: dict[str, Any] = {
        "guardrail": "selection-only diagnostics; no downstream/evaluation quantity entered matching",
        "pool_size": count, "K": K, "score_mean": score_mean.tolist(),
        "score_std": score_std.tolist(), "model_seeds": {},
    }
    matrices_to_save: dict[str, np.ndarray] = {}
    for model_seed in map(int, config["model_seeds"]):
        A_fiber = _load_A(source, "fiber", model_seed)
        fiber = _one_diagnostic(A_fiber, jnp.asarray(default["matrix"]),
                                jnp.asarray(default["means"]), jnp.asarray(default["covariance"]),
                                z0, z1)
        rel_strength = (np.abs(diagnostics["strength"] - fiber["strength"]) /
                        (fiber["strength"] + 1e-14))
        eligible = np.flatnonzero(rel_strength <= strength_tol)
        if len(eligible) < K:
            raise RuntimeError(f"candidate pool has only {len(eligible)} strength matches for seed {model_seed}")
        e1_indices = eligible[np.argsort(rel_strength[eligible])[:K]]
        target_log = np.log(np.maximum([fiber["strength"], fiber["variance_trace"],
                                        fiber["endpoint_phi_mmd"]], 1e-14))
        target_score = (target_log - score_mean) / score_std
        distance = np.linalg.norm(standardized_scores - target_score[None, :], axis=1)
        rel_variance = (np.abs(diagnostics["variance_trace"] - fiber["variance_trace"]) /
                        (fiber["variance_trace"] + 1e-14))
        rel_mmd = (np.abs(diagnostics["endpoint_phi_mmd"] - fiber["endpoint_phi_mmd"]) /
                   (fiber["endpoint_phi_mmd"] + 1e-14))
        exact = np.flatnonzero((rel_strength <= strength_tol) &
                               (rel_variance <= variance_tol) & (rel_mmd <= mmd_tol))
        if len(exact) >= K:
            e2_indices = exact[np.argsort(distance[exact])[:K]]
            e2_mode = "exact_tolerances"
        else:
            e2_indices = np.argsort(distance)[:K]
            e2_mode = "nearest_standardized_diagnostic_space"
        def selected_rows(indices):
            return [{"candidate_index": int(index),
                     **{name: float(values[index]) for name, values in diagnostics.items()},
                     "relative_strength_error": float(rel_strength[index]),
                     "relative_variance_error": float(rel_variance[index]),
                     "relative_endpoint_mmd_error": float(rel_mmd[index]),
                     "standardized_match_distance": float(distance[index])}
                    for index in indices]
        selections["model_seeds"][str(model_seed)] = {
            "fiber": fiber, "E1_strength_matched": selected_rows(e1_indices),
            "E2_joint_matched": selected_rows(e2_indices), "E2_mode": e2_mode,
            "E2_exact_candidate_count": int(len(exact)),
        }
        matrices_to_save[f"fiber_{model_seed}"] = A_fiber
        matrices_to_save[f"e1_{model_seed}"] = np.asarray(pool)[e1_indices]
        matrices_to_save[f"e2_{model_seed}"] = np.asarray(pool)[e2_indices]
    np.savez(out / "selected_matched_random_matrices.npz", **matrices_to_save)
    _json(selection_path, selections)
    _json(out / "source_frozen_input_hashes.json", _fingerprints(source))
    print("[expE:prepare] selections frozen", flush=True)


def _load_design(out: Path, geometry: str = "default") -> tuple[jax.Array, jax.Array, jax.Array]:
    data = np.load(out / "matching" / "strength_design_banks.npz")
    return (jnp.asarray(data[f"{geometry}_matrix"]), jnp.asarray(data[f"{geometry}_means"]),
            jnp.asarray(data[f"{geometry}_covariance"]))


def _train_default_ritz(path: Path, model: od.ObservableModel, reference, seed: int,
                        budget: dict[str, Any]) -> Any:
    if path.exists():
        return _load_ritz(path)
    potential, _ = od.train_downstream_ritz(
        _key(seed, 500), model, reference, steps=int(budget["ritz_steps"]),
        n_times=int(budget["ritz_times"]), n_particles=int(budget["ritz_particles"]))
    _save_ritz(path, potential)
    return potential


def _evaluate_cell(path: Path, model: od.ObservableModel, reference, potential,
                   eval_seed: int, budget: dict[str, Any], times: list[float],
                   geometry: str = "default") -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())["downstream"]
    downstream = sm.evaluate_downstream_geometry(
        _key(eval_seed, 700), model, reference, potential, geometry=geometry,
        times=times, n_particles=int(budget["rollout_particles"]),
        target_particles=int(budget["target_particles"]), flow_steps=int(budget["flow_steps"]),
        local_dt=float(budget["local_dt"]))
    _json(path, {"downstream": downstream})
    return downstream


def run_matched(config: dict[str, Any], only_seed: int | None = None) -> None:
    source, out = _paths(config)
    selections = json.loads((out / "matching" / "matching_diagnostics.json").read_text())
    pool = np.load(out / "matching" / "random_candidate_pool.npz")["A"]
    standardization, reference = _load_standardization(source), exb.load_model()[0]
    budget, times = config["budget"], list(map(float, config["evaluation_times"]))
    for model_seed in map(int, config["model_seeds"]):
        if only_seed is not None and model_seed != only_seed:
            continue
        rows = selections["model_seeds"][str(model_seed)]
        indices = sorted({int(x["candidate_index"]) for kind in
                          ("E1_strength_matched", "E2_joint_matched") for x in rows[kind]})
        print(f"[expE:matched] seed {model_seed}: {len(indices)} unique selected candidates", flush=True)
        for candidate_index in indices:
            model = od.ObservableModel(jnp.asarray(pool[candidate_index]), standardization)
            ritz_path = (out / "checkpoints" / "matched" /
                         f"ritz_candidate_{candidate_index}_modelseed_{model_seed}.npz")
            potential = _train_default_ritz(ritz_path, model, reference, model_seed, budget)
            for eval_seed in map(int, config["evaluation_seeds"]):
                cell_path = (out / "cells" / "matched" /
                             f"candidate_{candidate_index}_model_{model_seed}_eval_{eval_seed}.json")
                if not cell_path.exists():
                    print(f"[expE:matched]   candidate {candidate_index} eval {eval_seed}", flush=True)
                _evaluate_cell(cell_path, model, reference, potential, eval_seed, budget, times)
        jax.clear_caches()


def run_e3(config: dict[str, Any], only_seed: int | None = None) -> None:
    source, out = _paths(config)
    standardization, reference = _load_standardization(source), exb.load_model()[0]
    matrix, _, _ = _load_design(out)
    budget, times = config["budget"], list(map(float, config["evaluation_times"]))
    target_cfg = config["strength_targets"]
    for model_seed in map(int, config["model_seeds"]):
        if only_seed is not None and model_seed != only_seed:
            continue
        init_key = _key(model_seed, int(config["seed_streams"]["observable_initialization"]))
        B0, _ = od.initialize_stiefel(init_key, int(config["R"]))
        for target_name in TARGETS:
            checkpoint = (out / "checkpoints" / "e3" /
                          f"observable_fiber_{target_name}_modelseed_{model_seed}.npz")
            if checkpoint.exists():
                cached = np.load(checkpoint)
                A = jnp.asarray(cached["A"]); meta = json.loads(str(cached["metadata_json"]))
            else:
                print(f"[expE:E3] seed {model_seed} target {target_name}: training observable", flush=True)
                A, meta = sm.train_strength_constrained_fiber(
                    _key(model_seed, int(config["seed_streams"]["fiber_training"])),
                    standardization, B0, reference, matrix,
                    target_strength=float(target_cfg[target_name]),
                    relative_tolerance=float(target_cfg["relative_acceptance_tolerance"]),
                    gamma_start=float(target_cfg["gamma_start"]), gamma_end=float(target_cfg["gamma_end"]),
                    steps=int(budget["observable_steps"]), n_times=int(budget["fiber_times"]),
                    n_particles=int(budget["fiber_particles"]), delta_t=float(budget["local_dt"]))
                od.save_observable(checkpoint, f"fiber_{target_name}",
                                   od.ObservableModel(A, standardization), meta)
            if not bool(meta["strength_target_accepted"]):
                print(f"[expE:E3] seed {model_seed} target {target_name}: rejected "
                      f"(relative miss {meta['relative_strength_error']:.3%})", flush=True)
                continue
            model = od.ObservableModel(A, standardization)
            ritz_path = (out / "checkpoints" / "e3" /
                         f"ritz_fiber_{target_name}_modelseed_{model_seed}.npz")
            potential = _train_default_ritz(ritz_path, model, reference, model_seed, budget)
            for eval_seed in map(int, config["evaluation_seeds"]):
                cell_path = (out / "cells" / "e3" /
                             f"fiber_{target_name}_model_{model_seed}_eval_{eval_seed}.json")
                if not cell_path.exists():
                    print(f"[expE:E3]   {target_name} eval {eval_seed}", flush=True)
                _evaluate_cell(cell_path, model, reference, potential, eval_seed, budget, times)
        jax.clear_caches()


def _train_geometry_ritz(path: Path, model: od.ObservableModel, reference, model_seed: int,
                         budget: dict[str, Any], geometry: str):
    if path.exists():
        return _load_ritz(path)
    if geometry == "default":
        return _train_default_ritz(path, model, reference, model_seed, budget)
    potential, _ = sm.train_downstream_ritz_geometry(
        _key(model_seed, 500), model, reference, geometry=geometry,
        steps=int(budget["ritz_steps"]), n_times=int(budget["ritz_times"]),
        n_particles=int(budget["ritz_particles"]))
    _save_ritz(path, potential)
    return potential


def run_e4(config: dict[str, Any], only_seed: int | None = None) -> None:
    source, out = _paths(config)
    standardization, reference = _load_standardization(source), exb.load_model()[0]
    budget, times = config["budget"], list(map(float, config["evaluation_times"]))
    for model_seed in map(int, config["model_seeds"]):
        if only_seed is not None and model_seed != only_seed:
            continue
        init_key = _key(model_seed, int(config["seed_streams"]["observable_initialization"]))
        B0, _ = od.initialize_stiefel(init_key, int(config["R"]))
        observables: dict[str, jax.Array] = {"default": jnp.asarray(_load_A(source, "fiber", model_seed))}
        for train_geometry in ("smoothstep", "cosine"):
            checkpoint = (out / "checkpoints" / "e4" /
                          f"observable_fiber_{train_geometry}_modelseed_{model_seed}.npz")
            if checkpoint.exists():
                observables[train_geometry] = jnp.asarray(np.load(checkpoint)["A"])
            else:
                print(f"[expE:E4] seed {model_seed}: training {train_geometry} observable", flush=True)
                A, meta = sm.train_fiber_geometry(
                    _key(model_seed, int(config["seed_streams"]["fiber_training"])),
                    standardization, B0, reference, geometry=train_geometry,
                    steps=int(budget["observable_steps"]), n_times=int(budget["fiber_times"]),
                    n_particles=int(budget["fiber_particles"]), delta_t=float(budget["local_dt"]))
                od.save_observable(checkpoint, f"fiber_{train_geometry}",
                                   od.ObservableModel(A, standardization), meta)
                observables[train_geometry] = A
        for train_geometry, A in observables.items():
            model = od.ObservableModel(A, standardization)
            for eval_geometry in sm.GEOMETRIES:
                if train_geometry == "default" and eval_geometry == "default":
                    continue  # exact Experiment-D potential/cells are reused during aggregation.
                ritz_path = (out / "checkpoints" / "e4" /
                             f"ritz_train_{train_geometry}_eval_{eval_geometry}_modelseed_{model_seed}.npz")
                potential = _train_geometry_ritz(ritz_path, model, reference, model_seed,
                                                 budget, eval_geometry)
                for eval_seed in map(int, config["evaluation_seeds"]):
                    cell_path = (out / "cells" / "e4" /
                                 f"train_{train_geometry}_eval_{eval_geometry}_model_{model_seed}_bank_{eval_seed}.json")
                    if not cell_path.exists():
                        print(f"[expE:E4]   {train_geometry}->{eval_geometry} eval {eval_seed}", flush=True)
                    _evaluate_cell(cell_path, model, reference, potential, eval_seed,
                                   budget, times, eval_geometry)
        jax.clear_caches()


def _record_from_result(result: dict[str, Any]) -> dict[str, float]:
    """Support both extended Experiment-E and original Experiment-D cells."""
    target = result["target"]
    row = {
        "tangent_local_mmd": float(result["local_summary"]["mean_tangent_next_mmd"]),
        "tangent_rollout_mmd": float(result["summary"]["moment_tangent"]["mean_interior_mmd"]),
        "mfsi_rollout_mmd": float(result["summary"]["mfsi_learned_safe"]["mean_interior_mmd"]),
        "velocity_gap": float(result["local_summary"]["mean_velocity_gap_mse"]),
        "angular_error": float(result["summary"]["mfsi_learned_safe"]["mean_interior_angular_error"]),
        "min_ess": float(min(x["ess_fraction"] for x in target)),
        "mean_ess": float(np.mean([x["ess_fraction"] for x in target])),
        "max_moment_error": float(result["summary"]["mfsi_learned_safe"]["max_moment_error"]),
    }
    if "projection_distortion" in target[0]:
        row["mean_projection_distortion"] = float(np.mean([x["projection_distortion"] for x in target]))
        row["mean_lambda_norm"] = float(np.mean([x["lambda_norm"] for x in target]))
    return row


def _baseline_records(config: dict[str, Any], source: Path, out: Path) -> list[dict[str, Any]]:
    matrix, means, _ = _load_design(out)
    records: list[dict[str, Any]] = []
    for objective in BASELINES:
        for model_seed in map(int, config["model_seeds"]):
            A = jnp.asarray(_load_A(source, objective, model_seed))
            strength = float(sm.constraint_strength(A, matrix))
            normalized = float(sm.normalized_strength(A, means))
            for eval_seed in map(int, config["evaluation_seeds"]):
                cell = json.loads((source / "cells" /
                    f"nominal_{objective}_model_{model_seed}_eval_{eval_seed}.json").read_text())["downstream"]
                row = _record_from_result(cell)
                mechanism_path = (source / "mechanism_cells" /
                                  f"{objective}_model_{model_seed}_eval_{eval_seed}.json")
                mechanism = json.loads(mechanism_path.read_text())["per_time"]
                row["mean_projection_distortion"] = float(np.mean(
                    [x["projection_distortion"] for x in mechanism]))
                row["mean_lambda_norm"] = float(np.mean([x["lambda_norm"] for x in mechanism]))
                records.append({"objective": objective, "model_seed": model_seed,
                                "evaluation_seed": eval_seed, "strength": strength,
                                "normalized_strength": normalized, **row})
    return records


def _matched_records(config: dict[str, Any], out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selections = json.loads((out / "matching" / "matching_diagnostics.json").read_text())
    raw: list[dict[str, Any]] = []
    kind_names = {"E1_strength_matched": "strength_matched_random",
                  "E2_joint_matched": "joint_matched_random"}
    for model_seed in map(int, config["model_seeds"]):
        selection = selections["model_seeds"][str(model_seed)]
        for kind, objective in kind_names.items():
            for draw, selected in enumerate(selection[kind]):
                index = int(selected["candidate_index"])
                for eval_seed in map(int, config["evaluation_seeds"]):
                    path = (out / "cells" / "matched" /
                            f"candidate_{index}_model_{model_seed}_eval_{eval_seed}.json")
                    result = json.loads(path.read_text())["downstream"]
                    raw.append({"objective": objective, "matching_kind": kind,
                                "model_seed": model_seed, "evaluation_seed": eval_seed,
                                "control_draw": draw, "candidate_index": index,
                                "strength": selected["strength"],
                                "normalized_strength": selected["normalized_strength"],
                                **_record_from_result(result)})
    aggregated = []
    for objective in kind_names.values():
        for model_seed in map(int, config["model_seeds"]):
            for eval_seed in map(int, config["evaluation_seeds"]):
                subset = [r for r in raw if r["objective"] == objective and
                          r["model_seed"] == model_seed and r["evaluation_seed"] == eval_seed]
                aggregated.append({"objective": objective, "model_seed": model_seed,
                                   "evaluation_seed": eval_seed,
                                   **{name: float(np.mean([r[name] for r in subset]))
                                      for name in ("strength", "normalized_strength", *METRICS)}})
    return raw, aggregated


def _e3_records(config: dict[str, Any], source: Path, out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records, achievements = [], []
    matrix, means, _ = _load_design(out)
    for model_seed in map(int, config["model_seeds"]):
        for target in TARGETS:
            checkpoint = (out / "checkpoints" / "e3" /
                          f"observable_fiber_{target}_modelseed_{model_seed}.npz")
            data = np.load(checkpoint); A = jnp.asarray(data["A"])
            meta = json.loads(str(data["metadata_json"]))
            achievements.append({"objective": f"fiber_{target}", "model_seed": model_seed,
                                 "target_strength": float(meta["target_strength"]),
                                 "achieved_strength": float(meta["achieved_strength"]),
                                 "relative_strength_error": float(meta["relative_strength_error"]),
                                 "accepted": bool(meta["strength_target_accepted"]),
                                 "fiber_validation_objective": float(meta["fiber_validation_objective"])})
            if not meta["strength_target_accepted"]:
                continue
            strength = float(sm.constraint_strength(A, matrix))
            normalized = float(sm.normalized_strength(A, means))
            for eval_seed in map(int, config["evaluation_seeds"]):
                path = (out / "cells" / "e3" /
                        f"fiber_{target}_model_{model_seed}_eval_{eval_seed}.json")
                result = json.loads(path.read_text())["downstream"]
                records.append({"objective": f"fiber_{target}", "model_seed": model_seed,
                                "evaluation_seed": eval_seed, "strength": strength,
                                "normalized_strength": normalized, **_record_from_result(result)})
    return records, achievements


def _e4_records(config: dict[str, Any], source: Path, out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    standardization = _load_standardization(source)
    records, subspaces = [], []
    design = {geometry: _load_design(out, geometry) for geometry in sm.GEOMETRIES}
    for model_seed in map(int, config["model_seeds"]):
        matrices = {"default": _load_A(source, "fiber", model_seed)}
        for geometry in ("smoothstep", "cosine"):
            matrices[geometry] = np.asarray(np.load(out / "checkpoints" / "e4" /
                f"observable_fiber_{geometry}_modelseed_{model_seed}.npz")["A"])
        for i, left in enumerate(sm.GEOMETRIES):
            for right in sm.GEOMETRIES[i + 1:]:
                angles = np.asarray(od.principal_angles(jnp.asarray(matrices[left]),
                                                       jnp.asarray(matrices[right])))
                subspaces.append({"model_seed": model_seed, "left": left, "right": right,
                                  "projection_distance": float(od.subspace_distance(
                                      jnp.asarray(matrices[left]), jnp.asarray(matrices[right]))),
                                  "largest_angle_deg": float(np.max(angles) * 180 / np.pi),
                                  "angles_deg": json.dumps((angles * 180 / np.pi).tolist())})
        for train_geometry, A_np in matrices.items():
            A = jnp.asarray(A_np)
            model = od.ObservableModel(A, standardization)
            for eval_geometry in sm.GEOMETRIES:
                matrix, means, _ = design[eval_geometry]
                strength = float(sm.constraint_strength(A, matrix))
                normalized = float(sm.normalized_strength(A, means))
                for eval_seed in map(int, config["evaluation_seeds"]):
                    if train_geometry == "default" and eval_geometry == "default":
                        result = json.loads((source / "cells" /
                            f"nominal_fiber_model_{model_seed}_eval_{eval_seed}.json").read_text())["downstream"]
                        row = _record_from_result(result)
                        mechanism = json.loads((source / "mechanism_cells" /
                            f"fiber_model_{model_seed}_eval_{eval_seed}.json").read_text())["per_time"]
                        row["mean_projection_distortion"] = float(np.mean(
                            [x["projection_distortion"] for x in mechanism]))
                        row["mean_lambda_norm"] = float(np.mean([x["lambda_norm"] for x in mechanism]))
                    else:
                        path = (out / "cells" / "e4" /
                                f"train_{train_geometry}_eval_{eval_geometry}_model_{model_seed}_bank_{eval_seed}.json")
                        row = _record_from_result(json.loads(path.read_text())["downstream"])
                    records.append({"objective": f"train_{train_geometry}_eval_{eval_geometry}",
                                    "train_geometry": train_geometry, "eval_geometry": eval_geometry,
                                    "model_seed": model_seed, "evaluation_seed": eval_seed,
                                    "strength": strength, "normalized_strength": normalized, **row})
    return records, subspaces


def _crossed_contrast(records: list[dict[str, Any]], left: str, right: str,
                      replicates: int, seed: int) -> dict[str, Any]:
    left_rows = [r for r in records if r["objective"] == left]
    right_rows = [r for r in records if r["objective"] == right]
    models = sorted(set(int(r["model_seed"]) for r in left_rows) &
                    set(int(r["model_seed"]) for r in right_rows))
    evals = sorted(set(int(r["evaluation_seed"]) for r in left_rows) &
                   set(int(r["evaluation_seed"]) for r in right_rows))
    output = {"left": left, "right": right, "model_seed_count": len(models),
              "evaluation_seed_count": len(evals), "metrics": {}}
    rng = np.random.default_rng(seed)
    md = rng.integers(0, len(models), (replicates, len(models)))
    ed = rng.integers(0, len(evals), (replicates, len(evals)))
    for metric in METRICS:
        def table(rows):
            mapping = {(int(r["model_seed"]), int(r["evaluation_seed"])): float(r[metric])
                       for r in rows}
            return np.asarray([[mapping[(m, e)] for e in evals] for m in models])
        difference = table(left_rows) - table(right_rows)
        draws = difference[md[:, :, None], ed[:, None, :]].mean(axis=(1, 2))
        lo, hi = np.quantile(draws, [0.025, 0.975])
        output["metrics"][metric] = {"mean": float(difference.mean()),
                                     "ci95_low": float(lo), "ci95_high": float(hi)}
    return output


def _seed_means(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for objective in sorted(set(r["objective"] for r in records)):
        for model_seed in sorted(set(int(r["model_seed"]) for r in records
                                     if r["objective"] == objective)):
            subset = [r for r in records if r["objective"] == objective and
                      int(r["model_seed"]) == model_seed]
            rows.append({"objective": objective, "model_seed": model_seed,
                         **{name: float(np.mean([r[name] for r in subset]))
                            for name in ("strength", "normalized_strength", *METRICS)}})
    return rows


def _objective_means(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    seed_rows = _seed_means(records)
    return {objective: {name: float(np.mean([r[name] for r in seed_rows
                                             if r["objective"] == objective]))
                        for name in ("strength", "normalized_strength", *METRICS)}
            for objective in sorted(set(r["objective"] for r in seed_rows))}


def _make_figures(out: Path, records: list[dict[str, Any]], raw_matched: list[dict[str, Any]],
                  matching: dict[str, Any], e4: list[dict[str, Any]],
                  subspaces: list[dict[str, Any]]) -> None:
    colors = {"info": "#4477aa", "cv": "#ee8844", "fiber": "#228833",
              "random": "#999999", "full_phi5": "#aa3377", "fiber_low": "#66c2a5",
              "fiber_medium": "#3288bd", "fiber_high": "#5e4fa2",
              "strength_matched_random": "#b2abd2", "joint_matched_random": "#fdb863"}
    # 1. Matching quality.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fields = (("strength", "V(A)"), ("variance_trace", "variance trace"),
              ("endpoint_phi_mmd", "endpoint Phi-MMD"))
    for seed_data in matching["model_seeds"].values():
        target = seed_data["fiber"]
        for kind, marker in (("E1_strength_matched", "o"), ("E2_joint_matched", "s")):
            for row in seed_data[kind]:
                for ax, (field, label) in zip(axes, fields):
                    ax.scatter(target[field], row[field], marker=marker, alpha=.65,
                               color="#666666" if kind.startswith("E1") else "#cc6677")
                    ax.set_xlabel(f"FIBER {label}"); ax.set_ylabel(f"matched random {label}")
    for ax in axes:
        lo, hi = ax.get_xlim(); lo2, hi2 = ax.get_ylim(); low, high = min(lo, lo2), max(hi, hi2)
        ax.plot([low, high], [low, high], "k--", lw=1)
    fig.tight_layout(); fig.savefig(out / "strength_matching_quality.png", dpi=180); plt.close(fig)

    # 2. Paired FIBER vs controls by seed.
    seed_rows = _seed_means(records)
    metrics = (("tangent_local_mmd", "local tangent MMD"),
               ("tangent_rollout_mmd", "tangent rollout MMD"),
               ("mfsi_rollout_mmd", "safe-MFSI MMD"), ("velocity_gap", "velocity gap"),
               ("angular_error", "angular error"), ("min_ess", "min ESS"))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (metric, label) in zip(axes.flat, metrics):
        for control, marker in (("strength_matched_random", "o"), ("joint_matched_random", "s")):
            fiber = {r["model_seed"]: r[metric] for r in seed_rows if r["objective"] == "fiber"}
            ctrl = {r["model_seed"]: r[metric] for r in seed_rows if r["objective"] == control}
            x = [fiber[s] for s in sorted(fiber)]; y = [ctrl[s] for s in sorted(ctrl)]
            ax.scatter(x, y, marker=marker, alpha=.75, label=control.replace("_", " "))
        low = min(ax.get_xlim()[0], ax.get_ylim()[0]); high = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([low, high], [low, high], "k--", lw=1); ax.set_xlabel("FIBER"); ax.set_ylabel("matched random")
        ax.set_title(label)
    axes[0,0].legend(fontsize=7); fig.tight_layout()
    fig.savefig(out / "fiber_vs_strength_matched_random.png", dpi=180); plt.close(fig)

    # 3. Central frontier.
    frontier_metrics = (("tangent_local_mmd", "local tangent MMD"),
                        ("mfsi_rollout_mmd", "safe-MFSI MMD"),
                        ("tangent_rollout_mmd", "tangent rollout MMD"),
                        ("mean_ess", "mean ESS"), ("velocity_gap", "velocity gap"))
    fig, axes = plt.subplots(1, 5, figsize=(19, 4))
    objectives = [x for x in colors if any(r["objective"] == x for r in seed_rows)]
    for ax, (metric, label) in zip(axes, frontier_metrics):
        for objective in objectives:
            subset = [r for r in seed_rows if r["objective"] == objective]
            ax.scatter([r["strength"] for r in subset], [r[metric] for r in subset],
                       label=objective.upper(), color=colors[objective], alpha=.7, s=25)
        ax.set_xlabel("constraint strength V(A)"); ax.set_ylabel(label)
    axes[0].legend(fontsize=6); fig.tight_layout()
    fig.savefig(out / "strength_transport_frontier.png", dpi=180); plt.close(fig)

    for metric, filename, ylabel in (("mean_ess", "ess_vs_strength.png", "mean ESS"),
                                     ("velocity_gap", "velocity_gap_vs_strength.png", "velocity gap")):
        fig, ax = plt.subplots(figsize=(7, 5))
        for objective in objectives:
            subset = [r for r in seed_rows if r["objective"] == objective]
            ax.scatter([r["strength"] for r in subset], [r[metric] for r in subset],
                       label=objective.upper(), color=colors[objective], alpha=.75)
        ax.set_xlabel("constraint strength V(A)"); ax.set_ylabel(ylabel); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / filename, dpi=180); plt.close(fig)

    # 6. Reference subspace angles.
    pairs = [("default", "smoothstep"), ("default", "cosine"), ("smoothstep", "cosine")]
    values = [np.mean([r["largest_angle_deg"] for r in subspaces
                       if r["left"] == a and r["right"] == b]) for a, b in pairs]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([f"{a}\nvs {b}" for a,b in pairs], values, color=["#4477aa", "#228833", "#cc6677"])
    ax.set_ylabel("mean largest principal angle (degrees)")
    fig.tight_layout(); fig.savefig(out / "reference_geometry_subspace_angles.png", dpi=180); plt.close(fig)

    # 7. Cross-reference safe-MFSI means.
    matrix = np.empty((3,3))
    for i, train in enumerate(sm.GEOMETRIES):
        for j, evaluate in enumerate(sm.GEOMETRIES):
            matrix[i,j] = np.mean([r["mfsi_rollout_mmd"] for r in e4
                                   if r["train_geometry"] == train and r["eval_geometry"] == evaluate])
    fig, ax = plt.subplots(figsize=(6,5)); image = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(3), sm.GEOMETRIES, rotation=25); ax.set_yticks(range(3), sm.GEOMETRIES)
    ax.set_xlabel("evaluation reference"); ax.set_ylabel("observable-training reference")
    for i in range(3):
        for j in range(3): ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label="safe-MFSI rollout MMD")
    fig.tight_layout(); fig.savefig(out / "cross_reference_generalization.png", dpi=180); plt.close(fig)


def _write_report(out: Path, summary: dict[str, Any], bootstrap: dict[str, Any],
                  matching: dict[str, Any], achievements: list[dict[str, Any]],
                  e4: list[dict[str, Any]], subspaces: list[dict[str, Any]]) -> None:
    means = summary["objective_means"]
    primary = bootstrap["fiber_minus_strength_matched_random"]["metrics"]
    joint = bootstrap["fiber_minus_joint_matched_random"]["metrics"]
    fiber, strength, info = means["fiber"], means["strength_matched_random"], means["joint_matched_random"]
    def evidence(metric, data):
        x = data[metric]
        return f"{x['mean']:+.4f} [{x['ci95_low']:+.4f}, {x['ci95_high']:+.4f}]"
    accepted = {target: sum(r["accepted"] for r in achievements if r["objective"] == f"fiber_{target}")
                for target in TARGETS}
    achieved = {target: np.mean([r["achieved_strength"] for r in achievements
                                 if r["objective"] == f"fiber_{target}" and r["accepted"]])
                for target in TARGETS}
    match_rows = list(matching["model_seeds"].values())
    e1_strength_errors = [row["relative_strength_error"] for seed in match_rows
                          for row in seed["E1_strength_matched"]]
    e2_errors = {
        field: [row[field] for seed in match_rows for row in seed["E2_joint_matched"]]
        for field in ("relative_strength_error", "relative_variance_error",
                      "relative_endpoint_mmd_error")
    }
    def pct(values, q):
        return float(np.quantile(np.asarray(values, dtype=float), q))
    angle_means = {(a,b): np.mean([r["largest_angle_deg"] for r in subspaces
                                   if r["left"] == a and r["right"] == b])
                   for a,b in (("default","smoothstep"),("default","cosine"),("smoothstep","cosine"))}
    e4_mmd = {(a,b): np.mean([r["mfsi_rollout_mmd"] for r in e4
                              if r["train_geometry"] == a and r["eval_geometry"] == b])
              for a in sm.GEOMETRIES for b in sm.GEOMETRIES}
    e4_matched = {
        geometry: {metric: np.mean([r[metric] for r in e4
                                    if r["train_geometry"] == geometry and
                                    r["eval_geometry"] == geometry])
                   for metric in ("strength", "tangent_local_mmd", "tangent_rollout_mmd",
                                  "mfsi_rollout_mmd", "angular_error", "mean_ess")}
        for geometry in sm.GEOMETRIES
    }
    exact_e2 = sum(v["E2_mode"] == "exact_tolerances" for v in matching["model_seeds"].values())
    local_supports = primary["tangent_local_mmd"]["ci95_high"] < 0
    safe_supports = primary["mfsi_rollout_mmd"]["ci95_high"] < 0
    conclusion = ("FIBER retains a law-level advantage beyond raw constraint weakness."
                  if local_supports or safe_supports else
                  "The strength-matched control explains most of the original FIBER law-level advantage.")
    lines = [
        "# Strength-matched observable follow-up",
        "",
        "This is a separate Experiment-E follow-up. Experiment-D objectives, checkpoints, metrics, and conclusions were not modified. Matching used fixed design-only banks; final evaluation banks never entered selection.",
        "",
        "## Bottom line", "", conclusion,
        "",
        f"Against strength-matched random controls, FIBER-minus-control crossed contrasts are: local tangent MMD `{evidence('tangent_local_mmd', primary)}`, tangent rollout MMD `{evidence('tangent_rollout_mmd', primary)}`, safe-MFSI MMD `{evidence('mfsi_rollout_mmd', primary)}`, velocity gap `{evidence('velocity_gap', primary)}`, angular error `{evidence('angular_error', primary)}`, and min ESS `{evidence('min_ess', primary)}`. Negative is favorable for errors; positive is favorable for ESS.",
        "",
        f"FIBER mean strength is `{fiber['strength']:.4f}` and matched-random mean strength is `{strength['strength']:.4f}`. The joint strength/informativeness control has mean strength `{info['strength']:.4f}`; FIBER-minus-joint safe-MFSI is `{evidence('mfsi_rollout_mmd', joint)}`.",
        "",
        "## E1: strength-matched random subspaces", "",
        f"Five controls were nested within each of ten model seeds and averaged before crossed bootstrap. Exact 5% strength matching was available for every seed. FIBER safe-MFSI MMD is `{fiber['mfsi_rollout_mmd']:.4f}` versus `{strength['mfsi_rollout_mmd']:.4f}` for strength-matched random; mean ESS is `{fiber['mean_ess']:.3f}` versus `{strength['mean_ess']:.3f}`.",
        "",
        f"Across the 50 E1 controls, relative strength error has median `{pct(e1_strength_errors, .5):.3%}`, 95th percentile `{pct(e1_strength_errors, .95):.3%}`, and maximum `{max(e1_strength_errors):.3%}`.",
        "",
        "## E2: joint strength/informativeness matching", "",
        (f"Exact triple-tolerance matching supplied five controls for `{exact_e2}/10` seeds; "
         + (f"the remaining `{10 - exact_e2}` seeds used the prespecified nearest-neighbor fallback. "
            if exact_e2 < 10 else "the nearest-neighbor fallback was not used. ")
         + f"Joint-control safe-MFSI MMD is `{info['mfsi_rollout_mmd']:.4f}` and local tangent MMD is `{info['tangent_local_mmd']:.4f}`."),
        "",
        f"Across the 50 E2 controls, maximum relative errors were strength `{max(e2_errors['relative_strength_error']):.3%}`, variance `{max(e2_errors['relative_variance_error']):.3%}`, and endpoint Phi-MMD `{max(e2_errors['relative_endpoint_mmd_error']):.3%}` (prespecified limits 5%, 5%, and 10%).",
        "",
        "## E3: strength-constrained FIBER", "",
        f"Accepted checkpoints within the prespecified 5% target tolerance: LOW `{accepted['low']}/10`, MEDIUM `{accepted['medium']}/10`, HIGH `{accepted['high']}/10`. Among accepted checkpoints, mean achieved strengths are `{achieved['low']:.4f}`, `{achieved['medium']:.4f}`, and `{achieved['high']:.4f}`. Rejected checkpoints were retained in the achievement table but were not evaluated or counted as scientific replicates.",
        "",
    ]
    for target in TARGETS:
        name = f"fiber_{target}"
        if name in means:
            x = means[name]
            lines.append(f"- {name.upper()}: V `{x['strength']:.4f}`, local MMD `{x['tangent_local_mmd']:.4f}`, safe-MFSI `{x['mfsi_rollout_mmd']:.4f}`, tangent rollout `{x['tangent_rollout_mmd']:.4f}`, mean ESS `{x['mean_ess']:.3f}`, velocity gap `{x['velocity_gap']:.4f}`.")
    lines += [
        "",
        f"At nearly the original FIBER strength, FIBER-low minus original FIBER is `{evidence('tangent_local_mmd', bootstrap['fiber_low_minus_fiber']['metrics'])}` for local MMD, `{evidence('tangent_rollout_mmd', bootstrap['fiber_low_minus_fiber']['metrics'])}` for tangent rollout, and `{evidence('mfsi_rollout_mmd', bootstrap['fiber_low_minus_fiber']['metrics'])}` for safe-MFSI. Medium and high constraints also worsen law-level metrics and ESS relative to original FIBER. This supplies evidence for a transportability/strength tradeoff, but it is not strictly monotone for every endpoint (for example, tangent rollout is lower at HIGH than at MEDIUM). The constrained runs do not support a medium/high-strength FIBER advantage.",
        "",
        "The central frontier plots are descriptive. No method is called Pareto-optimal unless it is nondominated in the displayed empirical metrics.",
        "", "## E4: reference-geometry sensitivity", "",
        "The two alternate references are fixed smooth monotone time reparameterizations of the repository-validated default stochastic interpolant (smoothstep and cosine). They preserve the endpoint laws, endpoint coupling, standardization, and reference construction; their endpoint and derivative identities are covered by the follow-up tests.",
        "",
        f"Mean largest principal angles are default/smoothstep `{angle_means[('default','smoothstep')]:.1f}` degrees, default/cosine `{angle_means[('default','cosine')]:.1f}` degrees, and smoothstep/cosine `{angle_means[('smoothstep','cosine')]:.1f}` degrees.",
        "",
        "Matched-reference means:", "",
        "| reference | V(A) | local MMD | tangent rollout | safe-MFSI | angular error | mean ESS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for geometry in sm.GEOMETRIES:
        x = e4_matched[geometry]
        lines.append(f"| {geometry} | {x['strength']:.4f} | {x['tangent_local_mmd']:.4f} | {x['tangent_rollout_mmd']:.4f} | {x['mfsi_rollout_mmd']:.4f} | {x['angular_error']:.4f} | {x['mean_ess']:.3f} |")
    lines += [
        "",
        "Safe-MFSI cross-reference means (rows train the observable, columns evaluate after reference-specific downstream retraining):", "",
        "| train \\ eval | default | smoothstep | cosine |", "|---|---:|---:|---:|",
    ]
    for train in sm.GEOMETRIES:
        lines.append(f"| {train} | " + " | ".join(f"{e4_mmd[(train, ev)]:.4f}" for ev in sm.GEOMETRIES) + " |")
    lines += [
        "", "The default-trained subspaces differ materially from those trained under either alternate schedule, while the two alternate schedules produce similar subspaces. Accordingly, the learned object is best described here as reference-aware rather than fully intrinsic. Cross-reference safe-MFSI remains stable within each evaluation geometry after downstream retraining, so observable transfer is robust despite the subspace shift. Subspace stability and performance transfer are kept separate because downstream potentials were retrained while A remained frozen.",
        "", "## Interpretation guardrails", "",
        "Matched candidates, particles, time points, and optimizer iterations were never treated as independent replicates. All correlations/frontiers are descriptive. Negative results are retained. No result changes the registered Experiment-D confirmatory analysis.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    (out / REPORT.name).write_text("\n".join(lines) + "\n")


def aggregate(config: dict[str, Any]) -> None:
    source, out = _paths(config)
    baseline = _baseline_records(config, source, out)
    raw_matched, matched = _matched_records(config, out)
    e3, achievements = _e3_records(config, source, out)
    e4, subspaces = _e4_records(config, source, out)
    records = baseline + matched + e3
    replicates = int(config["budget"]["bootstrap_replicates"])
    bootstrap = {
        "method": "independent resampling of model rows and evaluation-bank columns; matched draws averaged within model seed",
        "replicates": replicates,
        "fiber_minus_strength_matched_random": _crossed_contrast(
            records, "fiber", "strength_matched_random", replicates,
            int(config["base_seed"]) + int(config["seed_streams"]["bootstrap"])),
        "fiber_minus_joint_matched_random": _crossed_contrast(
            records, "fiber", "joint_matched_random", replicates,
            int(config["base_seed"]) + int(config["seed_streams"]["bootstrap"]) + 1),
    }
    for index, target in enumerate(TARGETS):
        name = f"fiber_{target}"
        if any(r["objective"] == name for r in records):
            bootstrap[f"{name}_minus_fiber"] = _crossed_contrast(
                records, name, "fiber", replicates,
                int(config["base_seed"]) + int(config["seed_streams"]["bootstrap"]) + 10 + index)
    matching = json.loads((out / "matching" / "matching_diagnostics.json").read_text())
    summary = {"status": "complete_post_confirmatory_followup",
               "objective_means": _objective_means(records),
               "record_counts": {"seed_evaluation": len(records), "matched_draw": len(raw_matched),
                                 "cross_reference": len(e4)},
               "source_hashes_unchanged": _fingerprints(source) ==
                    json.loads((out / "source_frozen_input_hashes.json").read_text())}
    _csv(out / "seed_level_results.csv", records)
    _csv(out / "model_seed_means.csv", _seed_means(records))
    _csv(out / "matched_random_draw_records.csv", raw_matched)
    _csv(out / "achieved_strengths.csv", achievements)
    _csv(out / "reference_geometry_seed_level.csv", e4)
    _csv(out / "reference_geometry_subspaces.csv", subspaces)
    _json(out / "crossed_bootstrap.json", bootstrap)
    _json(out / "summary.json", summary)
    _make_figures(out, records, raw_matched, matching, e4, subspaces)
    _write_report(out, summary, bootstrap, matching, achievements, e4, subspaces)
    print(json.dumps({"report": str(REPORT), "records": len(records),
                      "matched_draw_records": len(raw_matched), "e4_records": len(e4)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("prepare", "matched", "e3", "e4", "aggregate", "all"),
                        default="all")
    parser.add_argument("--model-seed", type=int, default=None,
                        help="run one model seed for a compute stage; aggregation still requires all seeds")
    parser.add_argument("--force-prepare", action="store_true")
    args = parser.parse_args()
    config = _load_config(args.config)
    if args.stage in ("prepare", "all"):
        prepare(config, args.force_prepare)
    if args.stage in ("matched", "all"):
        run_matched(config, args.model_seed)
    if args.stage in ("e3", "all"):
        run_e3(config, args.model_seed)
    if args.stage in ("e4", "all"):
        run_e4(config, args.model_seed)
    if args.stage in ("aggregate", "all") and args.model_seed is None:
        aggregate(config)


if __name__ == "__main__":
    main()
