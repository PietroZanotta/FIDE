from __future__ import annotations

"""Isolated performance/equivalence probes for prospective-vortices V6."""

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for path in (REPO / "src", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import load_config
from prospective_data import TargetProspectiveData
from v4_objective import V4CRNBank, geometry_penalty, project_box
from v4_select import _adam_multistart, _batch_schedule, generate_full_starts
from v6_objective import V6MultiReferenceObjective
from v6_select import _compile_joint_rescore, _joint_rescore, _load_design


DEFAULT_RUN = HERE / "outputs" / "prospective_v6_beta_ablation_positive_raster_v1"
DEFAULT_CONFIG = DEFAULT_RUN / "results" / "resolved_config.json"
DEFAULT_OUTPUT = DEFAULT_RUN / "diagnostics" / "v6_acceleration_benchmark_v1"

jax.config.update("jax_enable_x64", True)


def _seconds(function, *args):
    started = time.perf_counter()
    value = function(*args)
    jax.block_until_ready(value)
    return value, time.perf_counter() - started


def _optimizer_function(
    objective, bank, cfg, limits, *, beta: float, steps: int, fidelity: str
):
    settings = cfg["v4"]["full_optimizer"]
    schedule = jnp.asarray(
        _batch_schedule(
            int(cfg["seeds"]["full_batch_schedule"]),
            steps,
            bank.trials,
            int(settings["batch_size"]),
        )
    )
    sampling = jnp.asarray(bank.sampling_z, dtype=jnp.float64)
    detector = jnp.asarray(bank.detector_z, dtype=jnp.float64)

    def loss(eta, s, d):
        return (
            objective.constrained_full_loss(eta, s, d, fidelity, limits, beta)
            + geometry_penalty(eta, cfg)
        )

    value_and_grad = jax.value_and_grad(loss)
    learning_rate = float(settings["learning_rate"])
    beta1 = float(settings["beta1"])
    beta2 = float(settings["beta2"])
    eps = float(settings["eps"])

    def optimize_one(eta0):
        def step(carry, indices):
            eta, m, v, iteration = carry
            value, gradient = value_and_grad(eta, sampling[indices], detector[indices])
            finite = jnp.isfinite(value) & jnp.all(jnp.isfinite(gradient))
            gradient = jnp.where(finite, gradient, jnp.zeros_like(gradient))
            next_m = beta1 * m + (1.0 - beta1) * gradient
            next_v = beta2 * v + (1.0 - beta2) * gradient * gradient
            iteration = iteration + 1
            mhat = next_m / (1.0 - beta1**iteration)
            vhat = next_v / (1.0 - beta2**iteration)
            proposal = project_box(
                eta - learning_rate * mhat / (jnp.sqrt(vhat) + eps), cfg
            )
            next_eta = jnp.where(finite, proposal, eta)
            trace = jnp.stack((value, jnp.linalg.norm(gradient), finite.astype(jnp.float64)))
            return (next_eta, next_m, next_v, iteration), trace

        initial = (
            jnp.asarray(eta0, dtype=jnp.float64),
            jnp.zeros_like(eta0),
            jnp.zeros_like(eta0),
            jnp.asarray(0, dtype=jnp.int32),
        )
        final, trace = jax.lax.scan(step, initial, schedule)
        return final[0], trace

    return optimize_one


def benchmark_multistart(
    run_root: Path,
    cfg: dict,
    *,
    steps: int,
    batch_size: int,
    scalar_count: int | None,
    fidelity: str,
) -> dict:
    _, _, ids, _, objective, _ = _load_design(cfg, run_root)
    shared = json.loads(
        (run_root / "shared" / "results" / "shared_selection_manifest.json").read_text()
    )
    limits = [float(shared["risk_limit_by_reference"][key]) for key in ids]
    with np.load(run_root / "shared" / "prospective" / "v6_selection_crn.npz") as data:
        master = V4CRNBank(
            np.asarray(data["sampling_z"], dtype=np.float64),
            np.asarray(data["detector_z"], dtype=np.float64),
        )
    fidelity_spec = objective.fidelity(fidelity)
    bank = master.prefix(fidelity_spec.trials)
    with np.load(run_root / "shared" / "results" / "full_starts.npz") as data:
        starts = jnp.asarray(data["starts"][: int(batch_size)], dtype=jnp.float64)
    optimize_one = _optimizer_function(
        objective, bank, cfg, limits, beta=0.0, steps=steps, fidelity=fidelity
    )

    compile_started = time.perf_counter()
    scalar = jax.jit(optimize_one).lower(starts[0]).compile()
    scalar_compile = time.perf_counter() - compile_started
    compile_started = time.perf_counter()
    batch = jax.jit(jax.vmap(optimize_one)).lower(starts).compile()
    batch_compile = time.perf_counter() - compile_started

    measured_scalar_count = len(starts) if scalar_count is None else min(int(scalar_count), len(starts))
    scalar_outputs = []
    scalar_started = time.perf_counter()
    for start in starts[:measured_scalar_count]:
        output = scalar(start)
        jax.block_until_ready(output)
        scalar_outputs.append(output)
    scalar_seconds = time.perf_counter() - scalar_started
    batch_output, batch_seconds = _seconds(batch, starts)
    scalar_eta = np.stack([np.asarray(value[0]) for value in scalar_outputs])
    scalar_trace = np.stack([np.asarray(value[1]) for value in scalar_outputs])
    batch_eta = np.asarray(batch_output[0])
    batch_trace = np.asarray(batch_output[1])
    return {
        "probe": "chunked_real_v6_search_gradient",
        "fidelity": fidelity,
        "steps": int(steps),
        "batch_size": int(batch_size),
        "reference_ids": ids,
        "scalar_compile_seconds": scalar_compile,
        "batch_compile_seconds": batch_compile,
        "scalar_count_measured": int(measured_scalar_count),
        "scalar_measured_seconds": scalar_seconds,
        "scalar_chunk_seconds_extrapolated": scalar_seconds / measured_scalar_count * len(starts),
        "batched_chunk_seconds": batch_seconds,
        "steady_state_speedup": (
            scalar_seconds / measured_scalar_count * len(starts) / batch_seconds
        ),
        "maximum_eta_absolute_difference": float(
            np.max(np.abs(scalar_eta - batch_eta[:measured_scalar_count]))
        ),
        "maximum_trace_absolute_difference": float(
            np.max(np.abs(scalar_trace - batch_trace[:measured_scalar_count]))
        ),
        "scalar_final_eta": scalar_eta.tolist(),
        "batch_final_eta": batch_eta.tolist(),
    }


def benchmark_compile_reuse(
    run_root: Path, cfg: dict, *, fidelity: str, candidate_count: int
) -> dict:
    _, _, _, _, objective, _ = _load_design(cfg, run_root)
    with np.load(run_root / "shared" / "prospective" / "v6_selection_crn.npz") as data:
        master = V4CRNBank(
            np.asarray(data["sampling_z"], dtype=np.float64),
            np.asarray(data["detector_z"], dtype=np.float64),
        )
    bank = master.prefix(objective.fidelity(fidelity).trials)
    archive = json.loads(
        (
            run_root
            / "arms"
            / "v6a_beta_0"
            / "results"
            / "candidate_archive.json"
        ).read_text()
    )
    candidates = [
        np.asarray(row["eta"], dtype=np.float64)
        for row in archive["candidates"][: int(candidate_count)]
    ]

    fresh_values = []
    fresh_started = time.perf_counter()
    for eta in candidates:
        fresh_values.append(
            _joint_rescore(objective, eta, bank, fidelity, 0.0)
        )
    fresh_seconds = time.perf_counter() - fresh_started

    compiled = _compile_joint_rescore(objective, bank, fidelity)
    reused_values = []
    reused_started = time.perf_counter()
    for eta in candidates:
        reused_values.append(
            _joint_rescore(
                objective, eta, bank, fidelity, 0.0, compiled=compiled
            )
        )
    reused_seconds = time.perf_counter() - reused_started
    fresh_scores = np.asarray([row["robust_score"] for row in fresh_values])
    reused_scores = np.asarray([row["robust_score"] for row in reused_values])
    return {
        "probe": "joint_rescore_compilation_reuse",
        "fidelity": fidelity,
        "candidate_count": len(candidates),
        "fresh_jit_per_candidate_seconds": fresh_seconds,
        "one_reused_jit_seconds": reused_seconds,
        "speedup": fresh_seconds / reused_seconds,
        "maximum_score_absolute_difference": float(
            np.max(np.abs(fresh_scores - reused_scores))
        ),
        "fresh_scores": fresh_scores.tolist(),
        "reused_scores": reused_scores.tolist(),
    }


def benchmark_tangent_batching(
    run_root: Path, cfg: dict, *, steps: int, start_batch_size: int
) -> dict:
    _, _, ids, _, objective, _ = _load_design(cfg, run_root)
    shared = json.loads(
        (run_root / "shared" / "results" / "shared_selection_manifest.json").read_text()
    )
    law_eta = np.asarray(shared["selected"]["Law"]["eta"], dtype=np.float64)
    limits = [float(shared["risk_limit_by_reference"][key]) for key in ids]
    tangent_cfg = copy.deepcopy(cfg)
    tangent_cfg["v4"]["full_optimizer"].update({
        key: cfg["v4"]["tangent_optimizer"][key]
        for key in (
            "starts", "law_perturbation_starts", "law_perturbation_scale",
            "start_oversample",
        )
    })
    tangent_cfg["seeds"]["full_global_starts"] = int(cfg["seeds"]["tangent_global_starts"])
    tangent_cfg["seeds"]["full_law_perturbations"] = int(cfg["seeds"]["tangent_law_perturbations"])
    starts, provenance = generate_full_starts(tangent_cfg, law_eta)
    starts = starts[: int(start_batch_size)]
    provenance = provenance[: int(start_batch_size)]
    with np.load(run_root / "shared" / "prospective" / "v6_selection_crn.npz") as data:
        master = V4CRNBank(
            np.asarray(data["sampling_z"], dtype=np.float64),
            np.asarray(data["detector_z"], dtype=np.float64),
        )
    settings = {
        **cfg["v4"]["tangent_optimizer"],
        "steps": int(steps),
    }
    bank = master.prefix(int(settings["crn_trials"]))
    arguments = (
        starts, provenance, bank, settings, cfg,
        lambda eta, s, d: objective.constrained_tangent_loss(
            eta, s, d, limits
        ),
    )
    started = time.perf_counter()
    scalar = _adam_multistart(
        *arguments,
        schedule_seed=int(cfg["seeds"]["tangent_batch_schedule"]),
        stage="benchmark-tangent-scalar",
        start_batch_size=1,
    )
    scalar_seconds = time.perf_counter() - started
    started = time.perf_counter()
    batched = _adam_multistart(
        *arguments,
        schedule_seed=int(cfg["seeds"]["tangent_batch_schedule"]),
        stage="benchmark-tangent-batched",
        start_batch_size=int(start_batch_size),
    )
    batched_seconds = time.perf_counter() - started
    return {
        "probe": "real_v6_tangent_multistart_batching",
        "steps": int(steps),
        "batch_size": int(start_batch_size),
        "scalar_compile_and_run_seconds": scalar_seconds,
        "batched_compile_and_run_seconds": batched_seconds,
        "cold_speedup": scalar_seconds / batched_seconds,
        "maximum_eta_absolute_difference": float(np.max(np.abs(
            np.asarray([row["final_eta"] for row in scalar])
            - np.asarray([row["final_eta"] for row in batched])
        ))),
        "maximum_trace_absolute_difference": float(np.max(np.abs(
            np.asarray([row["trace_objective"] for row in scalar])
            - np.asarray([row["trace_objective"] for row in batched])
        ))),
    }


def benchmark_start_prescreen(
    run_root: Path, cfg: dict, *, keep: int, batch_size: int
) -> dict:
    _, _, ids, _, objective, _ = _load_design(cfg, run_root)
    shared = json.loads(
        (run_root / "shared" / "results" / "shared_selection_manifest.json").read_text()
    )
    limits = [float(shared["risk_limit_by_reference"][key]) for key in ids]
    with np.load(run_root / "shared" / "prospective" / "v6_selection_crn.npz") as data:
        master = V4CRNBank(
            np.asarray(data["sampling_z"], dtype=np.float64),
            np.asarray(data["detector_z"], dtype=np.float64),
        )
    fidelity = str(cfg["v4"]["full_optimizer"]["fidelity"])
    bank = master.prefix(objective.fidelity(fidelity).trials)
    with np.load(run_root / "shared" / "results" / "full_starts.npz") as data:
        starts = jnp.asarray(data["starts"], dtype=jnp.float64)
        provenance = [str(value) for value in data["provenance"]]
    sampling = jnp.asarray(bank.sampling_z)
    detector = jnp.asarray(bank.detector_z)
    score_one = lambda eta: (
        objective.constrained_full_loss(
            eta, sampling, detector, fidelity, limits, 0.0
        ) + geometry_penalty(eta, cfg)
    )
    compiled = jax.jit(jax.vmap(score_one))
    scores = []
    started = time.perf_counter()
    for begin in range(0, len(starts), int(batch_size)):
        chunk = starts[begin : begin + int(batch_size)]
        take = len(chunk)
        if take < int(batch_size):
            chunk = jnp.concatenate(
                (chunk, jnp.repeat(chunk[-1:], int(batch_size) - take, axis=0))
            )
        scores.extend(np.asarray(compiled(chunk))[:take].tolist())
    jax.block_until_ready(jnp.asarray(scores))
    seconds = time.perf_counter() - started
    order = np.argsort(np.asarray(scores))
    kept = order[: int(keep)]
    archive = json.loads(
        (
            run_root / "arms" / "v6a_beta_0" / "results" / "candidate_archive.json"
        ).read_text()
    )
    feasible = {
        index for index, row in enumerate(archive["candidates"])
        if row["risk_feasible_all_references"]
    }
    selected_id = json.loads(
        (
            run_root / "arms" / "v6a_beta_0" / "results" / "frozen_manifest.json"
        ).read_text()
    )["selected"]["candidate_id"]
    selected_index = int(selected_id.split("-")[-1]) - 1
    stored_initial = np.asarray([
        row["initial_penalized_objective"] for row in archive["gradient_runs"]
    ])
    return {
        "probe": "batched_initial_full_loss_prescreen",
        "candidate_count": int(len(starts)),
        "keep": int(keep),
        "batch_size": int(batch_size),
        "compile_and_evaluate_seconds": seconds,
        "kept_one_based_indices": (kept + 1).tolist(),
        "kept_provenance": [provenance[index] for index in kept],
        "feasible_candidate_indices": sorted(index + 1 for index in feasible),
        "feasible_recall": len(feasible.intersection(set(kept))) / max(len(feasible), 1),
        "selected_candidate_index": selected_index + 1,
        "selected_candidate_retained": bool(selected_index in set(kept)),
        "maximum_difference_from_recorded_initial_trace": float(
            np.max(np.abs(np.asarray(scores) - stored_initial))
        ),
        "scores": scores,
    }


def _subset_rollouts(run_root: Path, output_dir: Path, count: int) -> list[Path]:
    manifest = json.loads(
        (run_root / "shared" / "results" / "design_reference_manifest.json").read_text()
    )
    paths = []
    for reference in manifest["references"]:
        source = Path(reference["rollout"])
        target = output_dir / "particle_rollouts" / f"{reference['reference_id']}_{count}.npz"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with np.load(source, allow_pickle=False) as data:
                particles = int(data["nodes"].shape[1])
                indices = np.floor(
                    np.arange(int(count), dtype=np.float64) * particles / int(count)
                ).astype(np.int64)
                np.savez_compressed(
                    target,
                    role=np.asarray(data["role"]),
                    signature=np.asarray(data["signature"]),
                    times=np.asarray(data["times"]),
                    nodes=np.asarray(data["nodes"])[:, indices],
                    velocity=np.asarray(data["velocity"])[:, indices],
                    weights=np.full(
                        (len(data["times"]), int(count)), 1.0 / int(count), dtype=np.float64
                    ),
                    diagnostic_parent_particles=np.asarray(particles),
                    diagnostic_systematic_indices=indices,
                )
        paths.append(target)
    return paths


def benchmark_particles(
    run_root: Path, cfg: dict, output_dir: Path, *, counts: list[int]
) -> dict:
    data = TargetProspectiveData.load(
        run_root / "shared" / "endpoint_reference" / "endpoint_data.npz",
        run_root / "shared" / "prospective" / "aggregate_predictions.npz",
    )
    with np.load(run_root / "shared" / "prospective" / "v6_selection_crn.npz") as arrays:
        master = V4CRNBank(
            np.asarray(arrays["sampling_z"], dtype=np.float64),
            np.asarray(arrays["detector_z"], dtype=np.float64),
        )
    archive = json.loads(
        (run_root / "arms" / "v6a_beta_0" / "results" / "candidate_archive.json").read_text()
    )
    feasible = [row for row in archive["candidates"] if row["risk_feasible_all_references"]]
    etas = [np.asarray(row["eta"], dtype=np.float64) for row in feasible]
    candidate_ids = [str(row["candidate_id"]) for row in feasible]
    selected = json.loads(
        (run_root / "arms" / "v6a_beta_0" / "results" / "frozen_manifest.json").read_text()
    )["selected"]
    selected_eta = jnp.asarray(selected["eta"], dtype=jnp.float64)
    rows = []
    reference_gradient = None
    reference_scores = None
    for count in counts:
        rollout_paths = _subset_rollouts(run_root, output_dir, int(count))
        objective = V6MultiReferenceObjective(cfg, data, rollout_paths)
        fidelity = "search"
        bank = master.prefix(objective.fidelity(fidelity).trials)
        sampling = jnp.asarray(bank.sampling_z)
        detector = jnp.asarray(bank.detector_z)
        score = lambda eta: objective.full_score(
            eta, sampling, detector, fidelity, 0.0
        )
        value_gradient = jax.jit(jax.value_and_grad(score))
        started = time.perf_counter()
        selected_value, gradient = jax.device_get(value_gradient(selected_eta))
        selected_seconds = time.perf_counter() - started
        score_fn = jax.jit(score)
        started = time.perf_counter()
        scores = np.asarray([
            float(jax.device_get(score_fn(jnp.asarray(eta)))) for eta in etas
        ])
        ranking_seconds = time.perf_counter() - started
        gradient = np.asarray(gradient, dtype=np.float64)
        if int(count) == max(counts):
            reference_gradient = gradient
            reference_scores = scores
        rows.append({
            "particles": int(count),
            "selected_value": float(selected_value),
            "selected_value_gradient_compile_and_run_seconds": selected_seconds,
            "six_candidate_ranking_seconds": ranking_seconds,
            "scores": scores.tolist(),
            "gradient": gradient.tolist(),
        })
    assert reference_gradient is not None and reference_scores is not None
    full_norm = max(float(np.linalg.norm(reference_gradient)), np.finfo(np.float64).tiny)
    selected_id = str(selected["candidate_id"])
    for row in rows:
        gradient = np.asarray(row["gradient"])
        scores = np.asarray(row["scores"])
        row["gradient_relative_l2_vs_full"] = float(
            np.linalg.norm(gradient - reference_gradient) / full_norm
        )
        row["gradient_cosine_vs_full"] = float(
            np.dot(gradient, reference_gradient)
            / max(np.linalg.norm(gradient) * np.linalg.norm(reference_gradient), np.finfo(np.float64).tiny)
        )
        row["score_max_absolute_vs_full"] = float(np.max(np.abs(scores - reference_scores)))
        row["score_spearman_vs_full"] = float(spearmanr(scores, reference_scores).statistic)
        row["best_candidate_id"] = candidate_ids[int(np.argmin(scores))]
        row["selected_candidate_remains_best"] = row["best_candidate_id"] == selected_id
    return {
        "probe": "nested_reference_particle_fidelity",
        "candidate_ids": candidate_ids,
        "selected_candidate_id": selected_id,
        "systematic_common_particle_subsets": True,
        "rows": rows,
    }


def benchmark_lbfgs_impact(run_root: Path, cfg: dict) -> dict:
    _, _, _, _, objective, _ = _load_design(cfg, run_root)
    with np.load(run_root / "shared" / "prospective" / "v6_selection_crn.npz") as arrays:
        master = V4CRNBank(
            np.asarray(arrays["sampling_z"], dtype=np.float64),
            np.asarray(arrays["detector_z"], dtype=np.float64),
        )
    bank = master.prefix(objective.fidelity("polish").trials)
    archive = json.loads(
        (run_root / "arms" / "v6a_beta_0" / "results" / "candidate_archive.json").read_text()
    )
    compiled = _compile_joint_rescore(objective, bank, "polish")
    rows = []
    started = time.perf_counter()
    for row in archive["polished"]:
        before_eta = np.asarray(row["gradient_run"]["final_eta"], dtype=np.float64)
        after_eta = np.asarray(row["eta"], dtype=np.float64)
        before = _joint_rescore(
            objective, before_eta, bank, "polish", 0.0, compiled=compiled
        )
        after_score = float(row["prospective_rescore"]["robust_score"])
        rows.append({
            "candidate_id": row["candidate_id"],
            "lbfgs_iterations": int(row["lbfgs"]["iterations"]),
            "eta_change_l2": float(np.linalg.norm(after_eta - before_eta)),
            "before_score": float(before["robust_score"]),
            "after_score": after_score,
            "score_change_after_minus_before": after_score - float(before["robust_score"]),
            "after_risk_feasible": bool(row["risk_feasible_all_references"]),
        })
    return {
        "probe": "completed_v6a_lbfgs_before_after",
        "elapsed_seconds": time.perf_counter() - started,
        "selected_v6a_candidate_was_polished": str(
            json.loads(
                (run_root / "arms" / "v6a_beta_0" / "results" / "frozen_manifest.json").read_text()
            )["selected"]["candidate_id"]
        ).endswith("-polished"),
        "rows": rows,
    }
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--scalar-count", type=int)
    parser.add_argument(
        "--mode", choices=("multistart", "tangent", "compile-reuse", "prescreen", "particles", "lbfgs-impact"), default="multistart"
    )
    parser.add_argument("--fidelity", choices=("search", "rescore", "polish"), default="search")
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--prescreen-keep", type=int, default=12)
    parser.add_argument("--particle-counts", nargs="+", type=int, default=[4096, 8192, 16384, 32768])
    args = parser.parse_args()
    cfg = load_config(args.config.resolve())
    if args.mode == "multistart":
        result = benchmark_multistart(
            args.run_root.resolve(), cfg,
            steps=int(args.steps), batch_size=int(args.batch_size), scalar_count=args.scalar_count,
            fidelity=str(args.fidelity),
        )
        filename = (
            f"multistart_{args.fidelity}_batch_{int(args.batch_size)}"
            f"_steps_{int(args.steps)}.json"
        )
    elif args.mode == "tangent":
        result = benchmark_tangent_batching(
            args.run_root.resolve(), cfg,
            steps=int(args.steps), start_batch_size=int(args.batch_size),
        )
        filename = f"tangent_batch_{int(args.batch_size)}_steps_{int(args.steps)}.json"
    elif args.mode == "compile-reuse":
        result = benchmark_compile_reuse(
            args.run_root.resolve(), cfg,
            fidelity=str(args.fidelity), candidate_count=int(args.candidate_count),
        )
        filename = f"compile_reuse_{args.fidelity}_{int(args.candidate_count)}.json"
    elif args.mode == "prescreen":
        result = benchmark_start_prescreen(
            args.run_root.resolve(), cfg,
            keep=int(args.prescreen_keep), batch_size=int(args.batch_size),
        )
        filename = f"prescreen_keep_{int(args.prescreen_keep)}_batch_{int(args.batch_size)}.json"
    elif args.mode == "particles":
        result = benchmark_particles(
            args.run_root.resolve(), cfg, args.output_dir.resolve(),
            counts=[int(value) for value in args.particle_counts],
        )
        filename = "particle_fidelity.json"
    else:
        result = benchmark_lbfgs_impact(args.run_root.resolve(), cfg)
        filename = "lbfgs_impact.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / filename
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
