#!/usr/bin/env python3
"""Phase 2D latent-coordinate domain-preserving deterministic reference."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from phase2_common import EmpiricalEndpointSource, load_phase2_config, resolve, sha256, write_csv, write_json  # noqa: E402
from repair_reference_support import endpoint_metrics, flow_config, native_single, sparse_simplex_lp  # noqa: E402

sys.path.insert(0, str(SCRIPT_DIR.parents[2] / "src"))
from mfsi.flow_matching import train_reference_flow  # noqa: E402
from mfsi.projection import IProjectionConfig  # noqa: E402
from mfsi.projection_tesseract import is_tesseract_iprojection_available  # noqa: E402
from mfsi.reference import DomainPreservingReferenceFlow, MLPReferenceFlow, save_npz_checkpoint  # noqa: E402

jax.config.update("jax_enable_x64", True)


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else SCRIPT_DIR.parent / "configs/domain_preserving_reference.json"
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    result["_config_path"] = str(path.resolve())
    return result


def numpy_to_latent_with_audit(
    points: np.ndarray, bounds: np.ndarray, epsilon: float, label: str
) -> tuple[np.ndarray, dict]:
    lower = np.asarray([bounds[0], bounds[2]], dtype=np.float64)
    upper = np.asarray([bounds[1], bounds[3]], dtype=np.float64)
    width = upper - lower
    ratio = (np.asarray(points, dtype=np.float64) - lower) / width
    clip_mask = (ratio < epsilon) | (ratio > 1.0 - epsilon)
    safe = np.clip(ratio, epsilon, 1.0 - epsilon)
    adjusted = lower + width * safe
    displacement = np.abs(adjusted - points)
    latent = np.log(safe) - np.log1p(-safe)
    return latent, {
        "dataset": label, "sample_count": len(points),
        "coordinate_count": int(points.size),
        "samples_requiring_map_clipping": int(clip_mask.any(axis=1).sum()),
        "coordinates_requiring_map_clipping": int(clip_mask.sum()),
        "maximum_physical_coordinate_displacement_km": float(displacement.max()),
        "maximum_physical_euclidean_displacement_km": float(np.linalg.norm(displacement, axis=1).max()),
        "minimum_raw_normalized_coordinate": float(ratio.min()),
        "maximum_raw_normalized_coordinate": float(ratio.max()),
        "exact_boundary_sample_count": int(((ratio == 0.0) | (ratio == 1.0)).any(axis=1).sum()),
    }


def accepted(row: dict, thresholds: dict) -> bool:
    return bool(
        row["biased_rbf_mmd2"] <= float(thresholds["maximum_biased_rbf_mmd2_each_endpoint"])
        and row["mean_error_over_empirical_rms_spread"] <= float(thresholds["maximum_mean_error_over_empirical_rms_spread"])
        and row["relative_covariance_frobenius_error"] <= float(thresholds["maximum_relative_covariance_frobenius_error"])
        and row["outside_domain_fraction"] <= float(thresholds["maximum_generated_endpoint_outside_domain_fraction"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--force-training", action="store_true")
    parser.add_argument("--force-bank", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    phase2 = load_phase2_config(resolve(cfg["base_phase2_config"]))
    if not is_tesseract_iprojection_available():
        raise RuntimeError("native I-projection Tesseract is required")
    processed = resolve(phase2["processed_dir"])
    analysis = resolve(phase2["analysis_dir"])
    model_dir = SCRIPT_DIR.parent / "models/reference_flow_domain_preserving"
    conditioned_model_dir = SCRIPT_DIR.parent / "models/reference_flow_conditioned_endpoints"
    old_model_dir = resolve(phase2["model_dir"])
    table_dir = analysis / "tables"
    endpoint_fig_dir = analysis / "figures/reference_support/domain_preserving_endpoints"
    path_fig_dir = analysis / "figures/reference_support/domain_preserving_paths"
    lp_fig_dir = analysis / "figures/reference_support/domain_preserving_lp_scaling"
    for directory in [model_dir, table_dir, endpoint_fig_dir, path_fig_dir, lp_fig_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    with np.load(processed / "development_270.npz", allow_pickle=False) as data:
        X = np.asarray(data["X"], dtype=np.float64)
        split = np.asarray(data["split"]).astype(str)
        times = np.asarray(data["normalized_time"], dtype=np.float64)
        days = np.asarray(data["relative_days"], dtype=np.float64)
        frozen_domain = np.asarray(data["domain_km"], dtype=np.float64)
    inference = X[split == "inference"]
    validation = X[split == "validation"]
    transform_cfg = cfg["domain_transform"]
    bounds = np.asarray(transform_cfg["bounds_km"], dtype=np.float64)
    np.testing.assert_array_equal(bounds, frozen_domain)
    epsilon = float(transform_cfg["map_epsilon"])
    pool_path = resolve(cfg["conditioned_training_pool"])
    initial_bank_path = resolve(cfg["conditioned_initial_bank"])
    with np.load(pool_path, allow_pickle=False) as data:
        training0 = np.asarray(data["x0_km"], dtype=np.float64)
        training1 = np.asarray(data["x1_km"], dtype=np.float64)
        if bool(data["final_test_accessed"]):
            raise RuntimeError("conditioned training pool reports final-test access")
    with np.load(initial_bank_path, allow_pickle=False) as data:
        initial = np.asarray(data["initial_km"], dtype=np.float64)
        if bool(data["final_test_accessed"]):
            raise RuntimeError("conditioned initial bank reports final-test access")
    latent0, audit0 = numpy_to_latent_with_audit(training0, bounds, epsilon, "conditioned_training_day0")
    latent1, audit1 = numpy_to_latent_with_audit(training1, bounds, epsilon, "conditioned_training_day45")
    latent_initial, audit_bank = numpy_to_latent_with_audit(initial, bounds, epsilon, "nested_reference_initial")
    _, audit_empirical0 = numpy_to_latent_with_audit(inference[:, 0], bounds, epsilon, "empirical_inference_day0")
    _, audit_empirical1 = numpy_to_latent_with_audit(inference[:, -1], bounds, epsilon, "empirical_inference_day45")
    map_rows = [audit0, audit1, audit_bank, audit_empirical0, audit_empirical1]
    theoretical_coordinate_shift = max(
        epsilon * (bounds[1] - bounds[0]), epsilon * (bounds[3] - bounds[2])
    )
    for row in map_rows:
        row["map_epsilon"] = epsilon
        row["theoretical_max_boundary_coordinate_shift_km"] = theoretical_coordinate_shift
    write_csv(table_dir / "domain_preserving_map_audit.csv", map_rows)
    if max(row["maximum_physical_coordinate_displacement_km"] for row in map_rows) > 0.01:
        raise RuntimeError("inverse-map numerical margin causes scientifically material displacement")

    block = phase2["reference_training"]
    train_cfg = flow_config(block, int(cfg["flow"]["training_seed"]))
    signature = json.dumps({
        "training_pool_sha256": sha256(pool_path), "training": asdict(train_cfg),
        "domain_transform": transform_cfg,
        "architecture_optimizer_bridge_integration_changed": False,
    }, sort_keys=True)
    checkpoint = model_dir / "reference.npz"
    flow = None
    if checkpoint.exists() and not args.force_training:
        candidate = DomainPreservingReferenceFlow.from_npz(
            checkpoint,
            substeps_per_interval=int(phase2["reference"]["rk4_substeps_per_time_interval"]),
        )
        if (candidate.metadata or {}).get("domain_preserving_signature") == signature:
            flow = candidate
            print("[domain preserving] reusing checkpoint", flush=True)
    if flow is None:
        source = EmpiricalEndpointSource(jnp.asarray(latent0), jnp.asarray(latent1))
        started = time.perf_counter()
        trained, history = train_reference_flow(
            source, train_cfg,
            substeps_per_interval=int(phase2["reference"]["rk4_substeps_per_time_interval"]),
        )
        training_seconds = time.perf_counter() - started
        metadata = dict(trained.metadata or {})
        metadata.update({
            "domain_preserving_signature": signature,
            "coordinate_space": "unconstrained_logistic_latent",
            "domain_transform": {"bounds_km": bounds.tolist(), "map_epsilon": epsilon},
            "physical_velocity": "J_T(z) times latent_velocity",
            "endpoint_source": "frozen conditioned endpoint KDE training pool",
            "endpoint_only": True, "intermediate_positions_used_for_training": False,
            "architecture_optimizer_bridge_integration_changed": False,
            "history": history, "training_seconds": training_seconds,
        })
        save_npz_checkpoint(checkpoint, trained.params, metadata)
        write_csv(table_dir / "domain_preserving_training_history.csv", history)
        flow = DomainPreservingReferenceFlow(
            trained.params, jnp.asarray(bounds), map_epsilon=epsilon,
            substeps_per_interval=int(phase2["reference"]["rk4_substeps_per_time_interval"]),
            metadata=metadata,
        )
        print(f"[domain preserving] trained in {training_seconds:.1f}s", flush=True)

    with np.load(processed / "iprojection_primary.npz", allow_pickle=False) as data:
        evaluation_indices = np.asarray(data["evaluation_indices"], dtype=int)
        frozen_mmd_bandwidth = float(data["bandwidth_km"])
    evaluation_times = times[evaluation_indices]
    evaluation_days = days[evaluation_indices]
    sizes = [int(value) for value in cfg["reference_bank"]["sizes"]]
    largest = max(sizes)
    if len(initial) != largest:
        raise RuntimeError("reused conditioned initial bank does not match requested largest bank")
    bank_path = model_dir / f"reference_bank_eval_{largest}.npz"
    bank_signature = json.dumps({
        "checkpoint_sha256": sha256(checkpoint), "initial_bank_sha256": sha256(initial_bank_path),
        "evaluation_times": evaluation_times.tolist(), "domain_transform": transform_cfg,
    }, sort_keys=True)
    nodes = velocities = None
    if bank_path.exists() and not args.force_bank:
        with np.load(bank_path, allow_pickle=False) as data:
            if str(data["signature"].item()) == bank_signature:
                nodes = np.asarray(data["nodes_km"], dtype=np.float64)
                velocities = np.asarray(data["velocity_km_per_normalized_time"], dtype=np.float64)
                print(f"[domain preserving] reusing {largest:,}-path bank", flush=True)
    if nodes is None:
        nodes = np.empty((len(evaluation_times), largest, 2), dtype=np.float64)
        velocities = np.empty_like(nodes)
        chunk = int(cfg["reference_bank"]["rollout_chunk_size"])
        started = time.perf_counter()
        for start in range(0, largest, chunk):
            stop = min(start + chunk, largest)
            z_nodes = flow.latent_rollout(
                jnp.asarray(latent_initial[start:stop]), jnp.asarray(evaluation_times)
            )
            x_nodes = flow.to_physical(z_nodes)
            u_nodes = jax.vmap(
                lambda t, z: flow.physical_velocity_from_latent(z, t)
            )(jnp.asarray(evaluation_times), z_nodes)
            nodes[:, start:stop] = np.asarray(x_nodes)
            velocities[:, start:stop] = np.asarray(u_nodes)
            if stop == chunk or stop % 50000 == 0 or stop == largest:
                print(f"[domain preserving] rollout {stop:,}/{largest:,}", flush=True)
        np.savez_compressed(
            bank_path, nodes_km=nodes, velocity_km_per_normalized_time=velocities,
            initial_km=initial, evaluation_indices=evaluation_indices,
            evaluation_times=evaluation_times, evaluation_days=evaluation_days,
            signature=np.asarray(bank_signature), checkpoint_sha256=np.asarray(sha256(checkpoint)),
            source_initial_bank_sha256=np.asarray(sha256(initial_bank_path)),
            final_test_accessed=np.asarray(False),
        )
        print(f"[domain preserving] rollout completed in {time.perf_counter()-started:.1f}s", flush=True)
    if not np.isfinite(nodes).all() or not np.isfinite(velocities).all():
        raise RuntimeError("nonfinite domain-preserving nodes or physical velocities")
    strict_outside = (
        (nodes[..., 0] < bounds[0]) | (nodes[..., 0] > bounds[1])
        | (nodes[..., 1] < bounds[2]) | (nodes[..., 1] > bounds[3])
    )
    if strict_outside.any():
        raise RuntimeError("logistic domain-preserving rollout produced a substantive outside particle")
    boundary_equal = (
        (nodes[..., 0] == bounds[0]) | (nodes[..., 0] == bounds[1])
        | (nodes[..., 1] == bounds[2]) | (nodes[..., 1] == bounds[3])
    )
    unique_rows = []
    midpoint = len(evaluation_times) // 2
    for size in sizes:
        unique_rows.append({
            "particle_count": size,
            "unique_initial_exact": len(np.unique(nodes[0, :size], axis=0)),
            "unique_midpoint_exact": len(np.unique(nodes[midpoint, :size], axis=0)),
            "unique_day45_exact": len(np.unique(nodes[-1, :size], axis=0)),
        })
    write_csv(table_dir / "domain_preserving_unique_paths.csv", unique_rows)
    if any(
        int(row[key]) != int(row["particle_count"])
        for row in unique_rows
        for key in ["unique_initial_exact", "unique_midpoint_exact", "unique_day45_exact"]
    ):
        raise RuntimeError("domain-preserving flow lost unique support scaling")

    estimator_path = processed / "endpoint_density_estimator_conditioned/conditioned_kde_endpoints.npz"
    with np.load(estimator_path, allow_pickle=False) as data:
        conditioned0 = np.asarray(data["conditioned_audit_x0_km"], dtype=np.float64)
        conditioned1 = np.asarray(data["conditioned_audit_x1_km"], dtype=np.float64)
    with np.load(conditioned_model_dir / "reference_bank_eval_200000.npz", allow_pickle=False) as data:
        previous_nodes = np.asarray(data["nodes_km"], dtype=np.float64)
    thresholds = cfg["endpoint_acceptance"]
    metric_rng = np.random.default_rng(int(cfg["seed"]) + 101)
    metric_rows = [
        endpoint_metrics(inference[:, 0], conditioned0, "conditioned_endpoint_target", "day0", frozen_mmd_bandwidth, bounds, metric_rng),
        endpoint_metrics(inference[:, -1], conditioned1, "conditioned_endpoint_target", "day45", frozen_mmd_bandwidth, bounds, metric_rng),
        endpoint_metrics(inference[:, 0], previous_nodes[0], "previous_unconstrained", "day0", frozen_mmd_bandwidth, bounds, metric_rng),
        endpoint_metrics(inference[:, -1], previous_nodes[-1], "previous_unconstrained", "day45", frozen_mmd_bandwidth, bounds, metric_rng),
        endpoint_metrics(inference[:, 0], nodes[0], "domain_preserving", "day0", frozen_mmd_bandwidth, bounds, metric_rng),
        endpoint_metrics(inference[:, -1], nodes[-1], "domain_preserving", "day45", frozen_mmd_bandwidth, bounds, metric_rng),
    ]
    for row in metric_rows:
        row["accepted"] = accepted(row, thresholds) if row["model"] == "domain_preserving" else "comparison"
    generated_rows = [row for row in metric_rows if row["model"] == "domain_preserving"]
    endpoint_passed = all(row["accepted"] is True for row in generated_rows)
    write_csv(table_dir / "domain_preserving_endpoint_metrics.csv", metric_rows)
    write_json(table_dir / "domain_preserving_endpoint_acceptance.json", {
        "criteria_frozen_from": thresholds["frozen_from"], "criteria": thresholds,
        "passed": endpoint_passed, "generated_metrics": generated_rows,
        "strict_outside_particle_count_all_times": int(strict_outside.sum()),
        "exact_boundary_coordinate_count_all_times": int(boundary_equal.sum()),
        "final_test_artifact_loaded": False, "intermediate_positions_used_for_tuning": False,
    })
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex="col", sharey="col")
    plot_rng = np.random.default_rng(int(cfg["seed"]) + 201)
    for column, (label, target, prior, current) in enumerate([
        ("day 0", conditioned0, previous_nodes[0], nodes[0]),
        ("day 45", conditioned1, previous_nodes[-1], nodes[-1]),
    ]):
        for row, (name, sample, color) in enumerate([
            ("conditioned target", target, "#4c78a8"),
            ("domain-preserving generated", current, "#54a24b"),
        ]):
            index = plot_rng.choice(len(sample), 3000, replace=False)
            axes[row, column].scatter(sample[index, 0], sample[index, 1], s=3, alpha=.14, color=color)
            empirical = inference[:, 0] if column == 0 else inference[:, -1]
            axes[row, column].scatter(empirical[:, 0], empirical[:, 1], s=11, alpha=.65, color="#e45756")
            axes[row, column].set_title(f"{name}: {label}"); axes[row, column].set_aspect("equal"); axes[row, column].grid(alpha=.15)
        prior_index = plot_rng.choice(len(prior), 3000, replace=False)
        axes[0, 2].scatter(previous_nodes[-1, prior_index, 0], previous_nodes[-1, prior_index, 1], s=3, alpha=.12, color="#f58518")
        axes[1, 2].scatter(nodes[-1, prior_index, 0], nodes[-1, prior_index, 1], s=3, alpha=.12, color="#54a24b")
    axes[0, 2].set_title("previous unconstrained: day 45")
    axes[1, 2].set_title("domain preserving: day 45")
    for ax in axes[:, 2]: ax.set_aspect("equal"); ax.grid(alpha=.15)
    fig.supxlabel("x (km)"); fig.supylabel("y (km)"); fig.tight_layout()
    fig.savefig(endpoint_fig_dir / "conditioned_previous_domain_preserving_endpoints.png", dpi=190); plt.close(fig)
    if not endpoint_passed:
        write_json(table_dir / "domain_preserving_reference_summary.json", {
            "endpoint_gate_passed": False, "support_audit_run": False,
            "full_bank_sweep_run": False, "stochastic_bridge_justified": False,
            "stop_reason": "domain-preserving reference failed a frozen endpoint fidelity metric",
            "final_test_artifact_loaded": False,
        })
        raise RuntimeError("domain-preserving reference failed endpoint gate")
    print("[domain preserving] endpoint gate passed", flush=True)

    with (table_dir / "reference_support_cases.json").open(encoding="utf-8") as handle:
        cases = json.load(handle)["cases"]
    if len(cases) != int(cfg["support_acceptance"]["same_case_count"]):
        raise RuntimeError("frozen 20-case manifest changed")
    with np.load(processed / "sensor_bank.npz", allow_pickle=False) as data:
        centers = np.asarray(data["centers_km"], dtype=np.float64)
        sigma = float(data["sigma_km"])
    p = phase2["projection"]
    projection_cfg = IProjectionConfig(
        max_steps=int(p["max_steps"]), residual_tol=float(p["residual_tol"]),
        newton_ridge=float(p["newton_ridge"]), step_cap=float(p["step_cap"]),
        lambda_clip=float(p["lambda_clip"]), line_search_steps=int(p["line_search_steps"]),
        implicit_ridge=float(p["implicit_ridge"]),
    )
    native_acceptance = {
        "accept_residual": float(p["accept_residual"]),
        "min_ess_fraction": float(cfg["support_acceptance"]["minimum_native_ess_fraction"]),
        "max_covariance_condition": float(cfg["support_acceptance"]["maximum_native_covariance_condition"]),
    }
    audit_rows = []
    started = time.perf_counter()
    for size in sizes:
        print(f"[domain preserving LP] M={size:,}", flush=True)
        for case in cases:
            source_index = int(case["source_time_index"])
            eval_position = int(np.flatnonzero(evaluation_indices == source_index)[0])
            design_index = int(case["design_index"])
            points = nodes[eval_position, :size]
            delta = points[:, None] - centers[design_index]
            phi = np.exp(-0.5 * np.sum(delta * delta, axis=-1) / sigma**2)
            target = np.asarray(case["target_moments"], dtype=np.float64)
            lp = sparse_simplex_lp(phi, target, float(case["frozen_lp_tolerance"]))
            native = native_single(phi, target, projection_cfg, native_acceptance)
            coordinate_margin = float(np.min(np.minimum(
                target - phi.min(axis=0), phi.max(axis=0) - target
            )))
            if lp["minimum_linf_residual"] > float(case["frozen_lp_tolerance"]):
                category = "C_convex_hull_failure"
            elif native["native_healthy"]:
                category = "A_support_repaired"
            else:
                category = "B_feasible_but_unhealthy"
            reasons = []
            if not native["native_converged"]: reasons.append("native_nonconvergence")
            if native["native_verified_l2_residual"] > float(p["accept_residual"]): reasons.append("moment_residual")
            if native["native_ess_fraction"] < native_acceptance["min_ess_fraction"]: reasons.append("low_ess")
            if native["native_covariance_condition_regularized"] > native_acceptance["max_covariance_condition"]: reasons.append("covariance_condition")
            audit_rows.append({
                "case": case["case"], "design_id": case["design_id"],
                "design_index": design_index, "style": case["style"],
                "day": case["day"], "source_time_index": source_index,
                "particle_count": size, "unique_reference_particles": size,
                "old_minimum_linf_residual": case["old_minimum_linf_residual"],
                "coordinate_support_margin": coordinate_margin,
                "classification": category, "native_failure_reason": ";".join(reasons),
                **lp, **native,
            })
        print(f"[domain preserving LP] completed M={size:,}; elapsed={time.perf_counter()-started:.1f}s", flush=True)
    write_csv(table_dir / "reference_support_lp_domain_preserving.csv", audit_rows)
    largest_rows = [row for row in audit_rows if row["particle_count"] == largest]
    tolerance = float(p["accept_residual"])
    lp_feasible = sum(row["minimum_linf_residual"] <= tolerance for row in largest_rows)
    native_healthy = sum(row["native_healthy"] for row in largest_rows)
    support_passed = bool(
        lp_feasible >= int(cfg["support_acceptance"]["minimum_lp_feasible_cases_at_200000"])
        and native_healthy >= int(cfg["support_acceptance"]["minimum_native_healthy_cases_at_200000"])
    )
    feasible_rows = [row for row in largest_rows if row["minimum_linf_residual"] <= tolerance]
    categories = {
        name: sum(row["classification"] == name for row in largest_rows)
        for name in ["A_support_repaired", "B_feasible_but_unhealthy", "C_convex_hull_failure"]
    }
    quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    ess = np.asarray([row["native_ess_fraction"] for row in feasible_rows])
    multiplier = np.asarray([row["native_multiplier_norm"] for row in feasible_rows])
    summary = {
        "endpoint_gate_passed": True, "support_audit_run": True,
        "support_audit_passed": support_passed, "particle_count": largest,
        "lp_feasible_cases": lp_feasible, "native_healthy_cases": native_healthy,
        "case_count": len(cases), "categories": categories,
        "feasible_ess_quantiles": {str(q): float(np.quantile(ess, q)) if len(ess) else None for q in quantiles},
        "feasible_multiplier_norm_quantiles": {str(q): float(np.quantile(multiplier, q)) if len(multiplier) else None for q in quantiles},
        "full_bank_sweep_authorized": support_passed, "full_bank_sweep_run": False,
        "stochastic_bridge_justified": not support_passed,
        "final_test_artifact_loaded": False,
    }
    write_json(table_dir / "domain_preserving_reference_summary.json", summary)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for case_index in range(len(cases)):
        rows = [row for row in audit_rows if row["case"] == case_index]
        ax.plot(
            [row["particle_count"] for row in rows],
            [max(float(row["minimum_linf_residual"]), 1e-12) for row in rows],
            marker="o", ms=3, alpha=.65,
        )
    ax.axhline(tolerance, color="black", ls="--", label="frozen LP tolerance")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=.2); ax.legend()
    ax.set_xlabel("unique reference particles M"); ax.set_ylabel("minimum LP residual (L-infinity)")
    ax.set_title("Frozen 20-case audit: domain-preserving deterministic reference")
    fig.tight_layout(); fig.savefig(lp_fig_dir / "domain_preserving_lp_scaling.png", dpi=190); plt.close(fig)

    display_days = [2.5, 5.0, 10.0]
    with np.load(old_model_dir / "reference_bank.npz", allow_pickle=False) as data:
        old_nodes = np.asarray(data["nodes_km"], dtype=np.float64)[evaluation_indices]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for column, day in enumerate(display_days):
        t = int(np.argmin(np.abs(evaluation_days - day)))
        source_index = evaluation_indices[t]
        old_index = plot_rng.choice(old_nodes.shape[1], 2000, replace=False)
        new_index = plot_rng.choice(nodes.shape[1], 3000, replace=False)
        axes[0, column].scatter(old_nodes[t, old_index, 0], old_nodes[t, old_index, 1], s=3, alpha=.16, color="#4c78a8")
        axes[1, column].scatter(nodes[t, new_index, 0], nodes[t, new_index, 1], s=3, alpha=.13, color="#54a24b")
        for row in range(2):
            axes[row, column].scatter(inference[:, source_index, 0], inference[:, source_index, 1], s=10, alpha=.5, color="black")
            axes[row, column].set_aspect("equal"); axes[row, column].grid(alpha=.15)
        axes[0, column].set_title(f"old discrete, day {day:g}")
        axes[1, column].set_title(f"domain preserving, day {day:g}")
    fig.supxlabel("x (km)"); fig.supylabel("y (km)"); fig.tight_layout()
    fig.savefig(path_fig_dir / "old_vs_domain_preserving_support.png", dpi=190); plt.close(fig)
    print(
        f"[domain preserving] 20-case LP={lp_feasible}/20, native healthy={native_healthy}/20, "
        f"passed={support_passed}", flush=True,
    )
    if support_passed:
        print("[domain preserving] full-bank sweep authorized", flush=True)
    else:
        print("[domain preserving] deterministic support insufficient; stochastic bridge justified", flush=True)


if __name__ == "__main__":
    main()
