from __future__ import annotations

"""Post-hoc D0-only repair audit for the prospective Law baseline."""

import argparse
import copy
import json
from pathlib import Path
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from common import fingerprint, geometry_valid, software_metadata, write_json_atomic
from mfsi.cache import file_sha256
from mfsi.design import random_point_sensor_starts
from prospective_data import TargetProspectiveData
from v4_objective import (
    V4CRNBank,
    canonical_geometry_key,
    ensure_v4_crn_bank,
    geometry_penalty,
)
from v4_select import _adam_multistart
from v6_objective import V6MultiReferenceObjective

jax.config.update("jax_enable_x64", True)


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "outputs" / "prospective_reflected_single_seed_pareto"
DEFAULT_OUTPUT = HERE / "outputs" / "law_reoptimization_audit"
ALLOWANCES = (0.005, 0.01, 0.02)


def _source_hash() -> str:
    names = (
        "reaudit_law.py",
        "v6_objective.py",
        "v4_objective.py",
        "v4_select.py",
        "evaluator.py",
        "prospective_data.py",
        "reflected_raster.py",
    )
    return fingerprint({name: file_sha256(HERE / name) for name in names})


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_eta(value: Any, provenance: str, rows: list[tuple[np.ndarray, str]]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_provenance = f"{provenance}/{key}"
            if key in {"eta", "final_eta"}:
                array = np.asarray(child, dtype=np.float64)
                if array.shape == (8,) and np.all(np.isfinite(array)):
                    rows.append((array, child_provenance))
            elif key not in {"trace_objective", "trace_gradient_norm", "trials"}:
                _collect_eta(child, child_provenance, rows)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, (dict, list)):
                _collect_eta(child, f"{provenance}[{index}]", rows)


def _discovered_geometries(source: Path, cfg: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    files = [source / "results" / "combined_frozen_manifest.json"]
    files.extend(sorted((source / "points").glob("*/shared/results/shared_candidate_archive.json")))
    files.extend(sorted((source / "points").glob("*/arms/v6a_beta_0/results/candidate_archive.json")))
    rows: list[tuple[np.ndarray, str]] = []
    for path in files:
        _collect_eta(_load_json(path), str(path.relative_to(source)), rows)
    unique: dict[tuple[float, ...], tuple[np.ndarray, str]] = {}
    for eta, provenance in rows:
        if geometry_valid(eta, cfg):
            unique.setdefault(canonical_geometry_key(eta), (eta, provenance))
    if not unique:
        raise RuntimeError("no valid discovered geometries found")
    return (
        np.stack([row[0] for row in unique.values()]),
        [row[1] for row in unique.values()],
    )


def _global_geometries(cfg: dict[str, Any], count: int, seed: int) -> tuple[np.ndarray, list[str]]:
    measurement = cfg["measurement"]
    margin = float(measurement["boundary_margin"])
    starts = random_point_sensor_starts(
        jax.random.PRNGKey(int(seed)),
        int(count),
        n_sensors=int(measurement["n_sensors"]),
        x_bounds=(margin, 2.0 - margin),
        y_bounds=(margin, 1.0 - margin),
        min_sep=float(measurement["min_separation"]),
        oversample=max(256, 8 * int(count)),
    )
    values = np.asarray(starts, dtype=np.float64)
    return values, [f"independent D0 Law global start seed={seed}"] * len(values)


def _score_geometries(
    objective: V6MultiReferenceObjective,
    bank: V4CRNBank,
    geometries: np.ndarray,
    cfg: dict[str, Any],
    *,
    batch_size: int = 8,
) -> tuple[np.ndarray, float]:
    sampling = jnp.asarray(bank.sampling_z, dtype=jnp.float64)
    detector = jnp.asarray(bank.detector_z, dtype=jnp.float64)
    score_one = lambda eta: objective.risk_mean(eta, sampling, detector)
    compiled = jax.jit(jax.vmap(score_one))
    effective = min(int(batch_size), len(geometries))
    scores: list[float] = []
    started = time.perf_counter()
    for begin in range(0, len(geometries), effective):
        chunk = jnp.asarray(geometries[begin : begin + effective], dtype=jnp.float64)
        take = int(chunk.shape[0])
        if take < effective:
            chunk = jnp.concatenate(
                (chunk, jnp.repeat(chunk[-1:], effective - take, axis=0)), axis=0
            )
        scores.extend(np.asarray(compiled(chunk))[:take].tolist())
    return np.asarray(scores, dtype=np.float64), time.perf_counter() - started


def _select_best(
    geometries: np.ndarray,
    provenance: list[str],
    scores: np.ndarray,
    keep: int,
) -> tuple[np.ndarray, list[str], list[int]]:
    valid = [
        int(index)
        for index in np.argsort(scores, kind="stable")
        if np.isfinite(scores[index])
    ]
    selected = valid[: min(int(keep), len(valid))]
    if not selected:
        raise RuntimeError("candidate prescreen produced no finite score")
    return geometries[selected], [provenance[index] for index in selected], selected


def _checkpointed_adam(
    path: Path,
    identity: dict[str, Any],
    starts: np.ndarray,
    provenance: list[str],
    bank: V4CRNBank,
    settings: dict[str, Any],
    cfg: dict[str, Any],
    objective: V6MultiReferenceObjective,
    *,
    seed: int,
    stage: str,
) -> list[dict[str, Any]]:
    identity_hash = fingerprint(identity)
    completed: list[dict[str, Any]] = []
    if path.exists():
        saved = _load_json(path)
        if saved.get("identity_hash") != identity_hash:
            raise RuntimeError(f"incompatible optimizer checkpoint: {path}")
        completed = list(saved["gradient_runs"])

    def save(rows: list[dict[str, Any]]) -> None:
        write_json_atomic(
            path,
            {
                "schema_version": 1,
                "identity_hash": identity_hash,
                "completed_starts": len(rows),
                "total_starts": len(starts),
                "gradient_runs": rows,
            },
        )

    rows = _adam_multistart(
        starts,
        provenance,
        bank,
        settings,
        cfg,
        lambda eta, sampling, detector: objective.risk_mean(eta, sampling, detector),
        schedule_seed=int(seed),
        stage=stage,
        start_batch_size=min(8, len(starts)),
        completed_rows=completed,
        chunk_callback=save,
    )
    save(rows)
    return rows


def _lbfgs_polish(
    objective: V6MultiReferenceObjective,
    bank: V4CRNBank,
    starts: np.ndarray,
    cfg: dict[str, Any],
    *,
    max_iterations: int,
) -> list[dict[str, Any]]:
    sampling = jnp.asarray(bank.sampling_z, dtype=jnp.float64)
    detector = jnp.asarray(bank.detector_z, dtype=jnp.float64)
    loss = jax.jit(
        jax.value_and_grad(
            lambda eta: objective.risk_mean(eta, sampling, detector)
            + geometry_penalty(eta, cfg)
        )
    )

    def fun(eta: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = jax.device_get(loss(jnp.asarray(eta, dtype=jnp.float64)))
        return float(value), np.asarray(gradient, dtype=np.float64)

    margin = float(cfg["measurement"]["boundary_margin"])
    bounds = []
    for _ in range(int(cfg["measurement"]["n_sensors"])):
        bounds.extend(((margin, 2.0 - margin), (margin, 1.0 - margin)))
    rows = []
    for index, eta in enumerate(starts, start=1):
        started = time.perf_counter()
        result = minimize(
            fun,
            np.asarray(eta, dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": int(max_iterations), "ftol": 1.0e-12, "gtol": 1.0e-7, "maxls": 30},
        )
        row = {
            "run": index,
            "initial_eta": np.asarray(eta).tolist(),
            "final_eta": np.asarray(result.x).tolist(),
            "final_objective": float(result.fun),
            "gradient_norm": float(np.linalg.norm(result.jac)),
            "iterations": int(result.nit),
            "evaluations": int(result.nfev),
            "success": bool(result.success),
            "message": str(result.message),
            "geometry_valid": geometry_valid(result.x, cfg),
            "elapsed_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        print(f"[law-lbfgs] polished {index}/{len(starts)}", flush=True)
    return rows


def _input_binding(source: Path) -> dict[str, str]:
    paths = {
        "resolved_config": source / "results" / "resolved_config.json",
        "combined_frozen_manifest": source / "results" / "combined_frozen_manifest.json",
        "endpoint_data": source / "shared" / "endpoint_reference" / "endpoint_data.npz",
        "aggregate_predictions": source / "shared" / "prospective" / "aggregate_predictions.npz",
        "selection_crn": source / "shared" / "prospective" / "v6_selection_crn.npz",
        "design_reference_manifest": source / "shared" / "results" / "design_reference_manifest.json",
        "d0_rollout": source / "shared" / "references" / "design" / "D0" / "endpoint_reference" / "reference_rollout.npz",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing Law audit inputs: " + ", ".join(missing))
    manifest = _load_json(paths["design_reference_manifest"])
    reference = manifest["references"][0]
    if reference["reference_id"] != "D0":
        raise RuntimeError("Law audit requires D0 as the sole design reference")
    if file_sha256(paths["d0_rollout"]) != reference["rollout_sha256"]:
        raise RuntimeError("canonical D0 rollout differs from its frozen manifest hash")
    return {name: file_sha256(path) for name, path in paths.items()}


def run(
    source: Path,
    output: Path,
    *,
    global_pool: int = 48,
    warm_keep: int = 16,
    global_keep: int = 16,
    polish_keep: int = 8,
    lbfgs_keep: int = 4,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    results_dir = output / "results"
    checkpoints = output / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    final_path = results_dir / "law_reoptimization_result.json"
    binding = _input_binding(source)
    source_hash = _source_hash()
    identity = {
        "schema_version": 1,
        "source_hash": source_hash,
        "input_binding": binding,
        "global_pool": int(global_pool),
        "warm_keep": int(warm_keep),
        "global_keep": int(global_keep),
        "polish_keep": int(polish_keep),
        "lbfgs_keep": int(lbfgs_keep),
    }
    identity_hash = fingerprint(identity)
    if final_path.exists():
        existing = _load_json(final_path)
        if existing.get("identity_hash") != identity_hash:
            raise RuntimeError("existing Law audit result has incompatible inputs/settings")
        print("[law-audit] reusing completed compatible result", flush=True)
        return existing

    started = time.perf_counter()
    cfg = _load_json(source / "results" / "resolved_config.json")
    data = TargetProspectiveData.load(
        source / "shared" / "endpoint_reference" / "endpoint_data.npz",
        source / "shared" / "prospective" / "aggregate_predictions.npz",
    )
    rollout = source / "shared" / "references" / "design" / "D0" / "endpoint_reference" / "reference_rollout.npz"
    objective = V6MultiReferenceObjective(cfg, data, [rollout])
    bank = ensure_v4_crn_bank(
        source / "shared" / "prospective" / "v6_selection_crn.npz",
        cfg,
        int(cfg["v4"]["selection_crn_trials"]),
    )

    discovered, discovered_provenance = _discovered_geometries(source, cfg)
    global_starts, global_provenance = _global_geometries(cfg, global_pool, 20266991)
    prescreen_bank = bank.prefix(16)
    discovered_scores, discovered_seconds = _score_geometries(
        objective, prescreen_bank, discovered, cfg
    )
    global_scores, global_seconds = _score_geometries(
        objective, prescreen_bank, global_starts, cfg
    )
    warm, warm_provenance, warm_indices = _select_best(
        discovered, discovered_provenance, discovered_scores, warm_keep
    )
    global_selected, global_selected_provenance, global_indices = _select_best(
        global_starts, global_provenance, global_scores, global_keep
    )
    stage1_starts = np.concatenate((warm, global_selected), axis=0)
    stage1_provenance = warm_provenance + global_selected_provenance

    stage1_settings = copy.deepcopy(cfg["v4"]["law_optimizer"])
    stage1_settings.update({"steps": 60, "batch_size": 8, "learning_rate": 0.008})
    stage1_runs = _checkpointed_adam(
        checkpoints / "stage1_adam.json",
        {**identity, "stage": "stage1", "starts": stage1_starts.tolist()},
        stage1_starts,
        stage1_provenance,
        prescreen_bank,
        stage1_settings,
        cfg,
        objective,
        seed=20266992,
        stage="law-repair-stage1-adam",
    )
    stage1_final = np.asarray([row["final_eta"] for row in stage1_runs], dtype=np.float64)
    authoritative_scores, authoritative_rescore_seconds = _score_geometries(
        objective, bank, stage1_final, cfg
    )
    stage2_starts, stage2_provenance, stage2_indices = _select_best(
        stage1_final,
        [f"stage1 run {index + 1}" for index in range(len(stage1_final))],
        authoritative_scores,
        polish_keep,
    )

    stage2_settings = copy.deepcopy(stage1_settings)
    stage2_settings.update({"steps": 60, "batch_size": 32, "learning_rate": 0.003})
    stage2_runs = _checkpointed_adam(
        checkpoints / "stage2_adam.json",
        {**identity, "stage": "stage2", "starts": stage2_starts.tolist()},
        stage2_starts,
        stage2_provenance,
        bank,
        stage2_settings,
        cfg,
        objective,
        seed=20266993,
        stage="law-repair-stage2-adam",
    )
    stage2_final = np.asarray([row["final_eta"] for row in stage2_runs], dtype=np.float64)
    stage2_scores, stage2_rescore_seconds = _score_geometries(
        objective, bank, stage2_final, cfg
    )
    lbfgs_starts, _, lbfgs_indices = _select_best(
        stage2_final,
        [f"stage2 run {index + 1}" for index in range(len(stage2_final))],
        stage2_scores,
        lbfgs_keep,
    )
    lbfgs_path = checkpoints / "lbfgs.json"
    if lbfgs_path.exists():
        saved = _load_json(lbfgs_path)
        if saved.get("identity_hash") != identity_hash:
            raise RuntimeError("incompatible L-BFGS checkpoint")
        lbfgs_rows = list(saved["runs"])
    else:
        lbfgs_rows = _lbfgs_polish(
            objective, bank, lbfgs_starts, cfg, max_iterations=40
        )
        write_json_atomic(
            lbfgs_path,
            {"schema_version": 1, "identity_hash": identity_hash, "runs": lbfgs_rows},
        )

    finalist_eta = list(stage2_final)
    finalist_provenance = [f"stage2-adam-{index + 1}" for index in range(len(stage2_final))]
    for row in lbfgs_rows:
        if row["geometry_valid"]:
            finalist_eta.append(np.asarray(row["final_eta"], dtype=np.float64))
            finalist_provenance.append(f"lbfgs-{row['run']}")
    finalist_eta_array = np.asarray(finalist_eta, dtype=np.float64)
    finalist_scores, finalist_seconds = _score_geometries(
        objective, bank, finalist_eta_array, cfg
    )
    feasible_indices = [
        index
        for index, eta in enumerate(finalist_eta_array)
        if geometry_valid(eta, cfg) and np.isfinite(finalist_scores[index])
    ]
    if not feasible_indices:
        raise RuntimeError("stronger Law search produced no valid finalist")
    selected_index = min(feasible_indices, key=lambda index: finalist_scores[index])
    selected_eta = finalist_eta_array[selected_index]
    selected_risk = float(finalist_scores[selected_index])

    combined = _load_json(source / "results" / "combined_frozen_manifest.json")
    old_law = combined["points"][0]["selected"]["Law"]
    point_rows = []
    all_feasible = True
    tolerance = float(cfg["validity"]["risk_constraint_tolerance"])
    for point in combined["points"]:
        allowance = float(point["allowance"])
        full = point["selected"]["v6a"]
        full_risk = float(full["mean_risk"])
        ceiling = (1.0 + allowance) * selected_risk
        feasible = bool(full_risk <= ceiling + tolerance)
        all_feasible = all_feasible and feasible
        point_rows.append(
            {
                "allowance": allowance,
                "full_eta": full["eta"],
                "full_d0_risk": full_risk,
                "repaired_law_d0_risk_ceiling": ceiling,
                "margin_to_ceiling": ceiling - full_risk,
                "existing_full_feasible": feasible,
            }
        )

    result = {
        "schema_version": 1,
        "status": "d0_law_reoptimization_complete",
        "role": "post_hoc_d0_only_law_baseline_repair_audit",
        "identity_hash": identity_hash,
        "identity": identity,
        "selection_information_boundary": {
            "design_reference_used": "D0",
            "evaluation_reference_used": False,
            "hidden_validation_used": False,
            "full_discovered_geometries_used_as_warm_starts": True,
            "interpretation": "post-hoc baseline repair; not the original prospective freeze",
        },
        "search": {
            "discovered_geometry_pool": len(discovered),
            "independent_global_pool": len(global_starts),
            "stage1_starts": len(stage1_starts),
            "stage2_starts": len(stage2_starts),
            "lbfgs_starts": len(lbfgs_starts),
            "warm_selected_indices": warm_indices,
            "global_selected_indices": global_indices,
            "stage2_selected_indices": stage2_indices,
            "lbfgs_selected_indices": lbfgs_indices,
            "prescreen_seconds": discovered_seconds + global_seconds,
            "authoritative_rescore_seconds": authoritative_rescore_seconds,
            "stage2_rescore_seconds": stage2_rescore_seconds,
            "finalist_rescore_seconds": finalist_seconds,
            "stage1_runs": stage1_runs,
            "stage1_authoritative_scores": authoritative_scores.tolist(),
            "stage2_runs": stage2_runs,
            "stage2_authoritative_scores": stage2_scores.tolist(),
            "lbfgs_runs": lbfgs_rows,
            "finalist_provenance": finalist_provenance,
            "finalist_scores": finalist_scores.tolist(),
        },
        "old_law": {
            "eta": old_law["eta"],
            "d0_risk": float(old_law["mean_risk"]),
        },
        "reoptimized_law": {
            "eta": selected_eta.tolist(),
            "centers": selected_eta.reshape((-1, 2)).tolist(),
            "d0_risk": selected_risk,
            "provenance": finalist_provenance[selected_index],
            "relative_risk_change_vs_old_law": selected_risk / float(old_law["mean_risk"]) - 1.0,
            "geometry_valid": geometry_valid(selected_eta, cfg),
        },
        "existing_full_points": point_rows,
        "all_existing_full_points_feasible": all_feasible,
        "elapsed_seconds": time.perf_counter() - started,
        "software": software_metadata(),
    }
    write_json_atomic(final_path, result)
    lines = [
        "# D0 Law Reoptimization Audit",
        "",
        "This is a post-hoc D0-only baseline repair. It does not overwrite the original prospective freeze.",
        "",
        f"- Old Law D0 risk: `{result['old_law']['d0_risk']:.12g}`",
        f"- Reoptimized Law D0 risk: `{selected_risk:.12g}`",
        f"- Relative change: `{100.0 * result['reoptimized_law']['relative_risk_change_vs_old_law']:.6f}%`",
        f"- All existing Full points feasible: `{'YES' if all_feasible else 'NO'}`",
        f"- Runtime: `{result['elapsed_seconds']:.1f} s`",
        "",
        "| Allowance | Full D0 risk | Repaired ceiling | Margin | Feasible |",
        "|--:|--:|--:|--:|:--:|",
    ]
    for row in point_rows:
        lines.append(
            f"| {100.0 * row['allowance']:g}% | {row['full_d0_risk']:.9g} | "
            f"{row['repaired_law_d0_risk_ceiling']:.9g} | {row['margin_to_ceiling']:.9g} | "
            f"{'PASS' if row['existing_full_feasible'] else 'FAIL'} |"
        )
    (results_dir / "LAW_REOPTIMIZATION_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--global-pool", type=int, default=48)
    parser.add_argument("--warm-keep", type=int, default=16)
    parser.add_argument("--global-keep", type=int, default=16)
    parser.add_argument("--polish-keep", type=int, default=8)
    parser.add_argument("--lbfgs-keep", type=int, default=4)
    args = parser.parse_args()
    result = run(
        args.source_run,
        args.output,
        global_pool=args.global_pool,
        warm_keep=args.warm_keep,
        global_keep=args.global_keep,
        polish_keep=args.polish_keep,
        lbfgs_keep=args.lbfgs_keep,
    )
    print(json.dumps({
        "status": result["status"],
        "old_law_d0_risk": result["old_law"]["d0_risk"],
        "reoptimized_law_d0_risk": result["reoptimized_law"]["d0_risk"],
        "all_existing_full_points_feasible": result["all_existing_full_points_feasible"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
