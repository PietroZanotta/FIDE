#!/usr/bin/env python3
"""Phase 2C domain-conditioned continuous endpoint reference experiment."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from phase2_common import load_phase2_config, resolve, sha256, write_csv, write_json  # noqa: E402
from repair_reference_support import (  # noqa: E402
    biased_mmd2,
    endpoint_metrics,
    energy_distance,
    flow_config,
    log_kde_density,
    native_single,
    sparse_simplex_lp,
)

sys.path.insert(0, str(SCRIPT_DIR.parents[2] / "src"))
from mfsi.flow_matching import train_reference_flow  # noqa: E402
from mfsi.projection import IProjectionConfig  # noqa: E402
from mfsi.projection_tesseract import is_tesseract_iprojection_available  # noqa: E402
from mfsi.reference import MLPReferenceFlow, save_npz_checkpoint  # noqa: E402

jax.config.update("jax_enable_x64", True)


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else SCRIPT_DIR.parent / "configs/domain_conditioned_reference.json"
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    result["_config_path"] = str(path.resolve())
    return result


def inside(points: np.ndarray, domain: np.ndarray) -> np.ndarray:
    return (
        (points[:, 0] >= domain[0]) & (points[:, 0] <= domain[1])
        & (points[:, 1] >= domain[2]) & (points[:, 1] <= domain[3])
    )


def propose_kde(
    atoms: np.ndarray,
    cholesky: np.ndarray,
    rng: np.random.Generator,
    count: int,
) -> np.ndarray:
    indices = rng.integers(0, len(atoms), size=int(count))
    return atoms[indices] + rng.normal(size=(int(count), 2)) @ cholesky.T


def rejection_sample_whole_mixture(
    atoms: np.ndarray,
    bandwidth: np.ndarray,
    domain: np.ndarray,
    rng: np.random.Generator,
    count: int,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Exact rejection sampler for the complete equal-weight KDE mixture."""
    cholesky = np.linalg.cholesky(bandwidth)
    accepted: list[np.ndarray] = []
    accepted_n = 0
    proposed_n = 0
    while accepted_n < count:
        needed = count - accepted_n
        batch_n = max(4096, int(math.ceil(needed * 1.04)))
        proposed = propose_kde(atoms, cholesky, rng, batch_n)
        kept = proposed[inside(proposed, domain)]
        accepted.append(kept)
        accepted_n += len(kept)
        proposed_n += batch_n
    result = np.concatenate(accepted, axis=0)[:count]
    return result, {
        "requested": int(count), "proposed": proposed_n,
        "accepted_before_trim": accepted_n,
        "effective_acceptance_rate": float(accepted_n / proposed_n),
    }


@dataclass(frozen=True)
class ConditionedEndpointPoolSource:
    x0_pool: jax.Array
    x1_pool: jax.Array

    def sample(self, key: jax.Array, n: int, endpoint: int) -> jax.Array:
        if endpoint not in (0, 1):
            raise ValueError("endpoint must be zero or one")
        pool = self.x0_pool if endpoint == 0 else self.x1_pool
        indices = jax.random.randint(key, (int(n),), 0, pool.shape[0])
        return pool[indices]


def metric_acceptance(row: dict, thresholds: dict) -> bool:
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
    old_model_dir = resolve(phase2["model_dir"])
    unbounded_model_dir = SCRIPT_DIR.parent / "models/reference_flow_continuous_endpoints"
    model_dir = SCRIPT_DIR.parent / "models/reference_flow_conditioned_endpoints"
    estimator_dir = processed / "endpoint_density_estimator_conditioned"
    table_dir = analysis / "tables"
    endpoint_fig_dir = analysis / "figures/reference_support/conditioned_endpoints"
    compare_fig_dir = analysis / "figures/reference_support/conditioned_old_vs_new"
    lp_fig_dir = analysis / "figures/reference_support/conditioned_lp_scaling"
    for directory in [model_dir, estimator_dir, table_dir, endpoint_fig_dir, compare_fig_dir, lp_fig_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    with np.load(processed / "development_270.npz", allow_pickle=False) as data:
        X = np.asarray(data["X"], dtype=np.float64)
        split = np.asarray(data["split"]).astype(str)
        times = np.asarray(data["normalized_time"], dtype=np.float64)
        days = np.asarray(data["relative_days"], dtype=np.float64)
        domain = np.asarray(data["domain_km"], dtype=np.float64)
    inference = X[split == "inference"]
    validation = X[split == "validation"]
    if inference.shape != (200, 181, 2) or validation.shape != (70, 181, 2):
        raise RuntimeError("frozen 200/70 development split changed")
    x0 = inference[:, 0]
    x1 = inference[:, -1]
    bandwidth_path = resolve(cfg["endpoint_law"]["bandwidth_source"])
    with np.load(bandwidth_path, allow_pickle=False) as data:
        original_x0 = np.asarray(data["x0_atoms_km"], dtype=np.float64)
        original_x1 = np.asarray(data["x1_atoms_km"], dtype=np.float64)
        H0 = np.asarray(data["H0_km2"], dtype=np.float64)
        H1 = np.asarray(data["H1_km2"], dtype=np.float64)
    np.testing.assert_array_equal(original_x0, x0)
    np.testing.assert_array_equal(original_x1, x1)
    normalization_n = int(cfg["endpoint_law"]["normalization_proposals"])
    normalization_rng = np.random.default_rng(int(cfg["endpoint_law"]["normalization_seed"]))
    normalization_rows = []
    for label, atoms, bandwidth in [("day0", x0, H0), ("day45", x1, H1)]:
        proposals = propose_kde(atoms, np.linalg.cholesky(bandwidth), normalization_rng, normalization_n)
        accepted = int(inside(proposals, domain).sum())
        probability = accepted / normalization_n
        standard_error = math.sqrt(probability * (1.0 - probability) / normalization_n)
        normalization_rows.append({
            "endpoint": label, "proposals": normalization_n, "inside_count": accepted,
            "Z_hat": probability, "outside_mass_hat": 1.0 - probability,
            "monte_carlo_standard_error": standard_error,
            "ci95_lower": probability - 1.96 * standard_error,
            "ci95_upper": probability + 1.96 * standard_error,
            "domain_xmin_km": domain[0], "domain_xmax_km": domain[1],
            "domain_ymin_km": domain[2], "domain_ymax_km": domain[3],
        })
    write_csv(table_dir / "conditioned_kde_normalization.csv", normalization_rows)
    print(
        "[conditioned] " + ", ".join(
            f"{row['endpoint']} Z={row['Z_hat']:.6f}" for row in normalization_rows
        ), flush=True,
    )

    audit_n = int(cfg["endpoint_law"]["audit_samples"])
    audit_rng = np.random.default_rng(int(cfg["seed"]) + 101)
    unbounded0 = propose_kde(x0, np.linalg.cholesky(H0), audit_rng, audit_n)
    unbounded1 = propose_kde(x1, np.linalg.cholesky(H1), audit_rng, audit_n)
    conditioned0, audit_stats0 = rejection_sample_whole_mixture(x0, H0, domain, audit_rng, audit_n)
    conditioned1, audit_stats1 = rejection_sample_whole_mixture(x1, H1, domain, audit_rng, audit_n)
    np.savez_compressed(
        estimator_dir / "conditioned_kde_endpoints.npz",
        x0_atoms_km=x0, x1_atoms_km=x1, H0_km2=H0, H1_km2=H1,
        domain_km=domain, conditioned_audit_x0_km=conditioned0,
        conditioned_audit_x1_km=conditioned1,
        method=np.asarray("exact rejection from whole equal-weight KDE mixture"),
        final_test_accessed=np.asarray(False),
    )
    with np.load(processed / "iprojection_primary.npz", allow_pickle=False) as data:
        evaluation_indices = np.asarray(data["evaluation_indices"], dtype=int)
        frozen_mmd_bandwidth = float(data["bandwidth_km"])
    thresholds = cfg["endpoint_acceptance"]
    endpoint_rng = np.random.default_rng(int(cfg["seed"]) + 201)
    estimator_metrics = []
    for label, empirical, unconditional, conditioned, stats in [
        ("day0", x0, unbounded0, conditioned0, audit_stats0),
        ("day45", x1, unbounded1, conditioned1, audit_stats1),
    ]:
        unconditional_row = endpoint_metrics(
            empirical, unconditional, "unbounded_kde", label,
            frozen_mmd_bandwidth, domain, endpoint_rng,
        )
        conditioned_row = endpoint_metrics(
            empirical, conditioned, "conditioned_kde", label,
            frozen_mmd_bandwidth, domain, endpoint_rng,
        )
        comparison_n = 2500
        a = unconditional[endpoint_rng.choice(len(unconditional), comparison_n, replace=False)]
        b = conditioned[endpoint_rng.choice(len(conditioned), comparison_n, replace=False)]
        conditioned_row.update({
            "conditional_vs_unconditional_mmd2": biased_mmd2(a, b, frozen_mmd_bandwidth),
            "conditional_vs_unconditional_energy_distance_km": energy_distance(a, b),
            "rejection_effective_acceptance_rate": stats["effective_acceptance_rate"],
        })
        unconditional_row.update({
            "conditional_vs_unconditional_mmd2": 0.0,
            "conditional_vs_unconditional_energy_distance_km": 0.0,
            "rejection_effective_acceptance_rate": 1.0,
        })
        unconditional_row["accepted"] = metric_acceptance(unconditional_row, thresholds)
        conditioned_row["accepted"] = metric_acceptance(conditioned_row, thresholds)
        estimator_metrics.extend([unconditional_row, conditioned_row])
    write_csv(table_dir / "conditioned_endpoint_estimator_metrics.csv", estimator_metrics)
    conditioned_estimator_passed = all(
        row["accepted"] for row in estimator_metrics if row["model"] == "conditioned_kde"
    )
    if not conditioned_estimator_passed:
        raise RuntimeError("conditioned endpoint estimator failed frozen endpoint metrics")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex="col", sharey="col")
    for column, (label, empirical, unconditional, conditioned) in enumerate([
        ("day 0", x0, unbounded0, conditioned0),
        ("day 45", x1, unbounded1, conditioned1),
    ]):
        for row, (name, sample, color) in enumerate([
            ("unbounded KDE", unconditional, "#4c78a8"),
            ("domain-conditioned KDE", conditioned, "#54a24b"),
        ]):
            ax = axes[row, column]
            index = endpoint_rng.choice(len(sample), 3000, replace=False)
            ax.scatter(sample[index, 0], sample[index, 1], s=3, alpha=.14, color=color)
            ax.scatter(empirical[:, 0], empirical[:, 1], s=12, alpha=.7, color="#e45756")
            ax.plot(
                [domain[0], domain[1], domain[1], domain[0], domain[0]],
                [domain[2], domain[2], domain[3], domain[3], domain[2]],
                color="black", lw=1,
            )
            ax.set_title(f"{name}: {label}"); ax.set_aspect("equal"); ax.grid(alpha=.15)
    fig.supxlabel("x (km)"); fig.supylabel("y (km)"); fig.tight_layout()
    fig.savefig(endpoint_fig_dir / "unbounded_vs_conditioned_endpoints.png", dpi=190); plt.close(fig)

    training_pool_n = int(cfg["endpoint_law"]["training_pool_samples"])
    pool_path = estimator_dir / f"conditioned_training_pool_{training_pool_n}.npz"
    if pool_path.exists():
        with np.load(pool_path, allow_pickle=False) as data:
            training0 = np.asarray(data["x0_km"], dtype=np.float64)
            training1 = np.asarray(data["x1_km"], dtype=np.float64)
    else:
        pool_rng = np.random.default_rng(int(cfg["seed"]) + 301)
        training0, stats0 = rejection_sample_whole_mixture(x0, H0, domain, pool_rng, training_pool_n)
        training1, stats1 = rejection_sample_whole_mixture(x1, H1, domain, pool_rng, training_pool_n)
        np.savez_compressed(
            pool_path, x0_km=training0, x1_km=training1,
            x0_acceptance=np.asarray(stats0["effective_acceptance_rate"]),
            x1_acceptance=np.asarray(stats1["effective_acceptance_rate"]),
            final_test_accessed=np.asarray(False),
        )
    block = phase2["reference_training"]
    center = np.asarray(block["normalization_center_km"], dtype=np.float64)
    scale = float(block["normalization_scale_km"])
    train_cfg = flow_config(block, int(cfg["flow"]["training_seed"]))
    estimator_path = estimator_dir / "conditioned_kde_endpoints.npz"
    signature = json.dumps({
        "conditioned_estimator_sha256": sha256(estimator_path),
        "training_pool_sha256": sha256(pool_path), "training": asdict(train_cfg),
        "center": center.tolist(), "scale": scale,
        "only_change": "whole-mixture KDE conditioned on frozen rectangle",
    }, sort_keys=True)
    checkpoint = model_dir / "reference.npz"
    flow = None
    if checkpoint.exists() and not args.force_training:
        candidate = MLPReferenceFlow.from_npz(
            checkpoint,
            substeps_per_interval=int(phase2["reference"]["rk4_substeps_per_time_interval"]),
        )
        if (candidate.metadata or {}).get("conditioned_signature") == signature:
            flow = candidate
            print("[conditioned] reusing checkpoint", flush=True)
    if flow is None:
        source = ConditionedEndpointPoolSource(
            jnp.asarray((training0 - center) / scale),
            jnp.asarray((training1 - center) / scale),
        )
        started = time.perf_counter()
        flow, history = train_reference_flow(
            source, train_cfg,
            substeps_per_interval=int(phase2["reference"]["rk4_substeps_per_time_interval"]),
        )
        training_seconds = time.perf_counter() - started
        metadata = dict(flow.metadata or {})
        metadata.update({
            "conditioned_signature": signature,
            "endpoint_source": "whole Gaussian KDE mixture conditioned on frozen domain",
            "endpoint_only": True, "intermediate_positions_used_for_training": False,
            "architecture_optimizer_bridge_integration_changed": False,
            "training_pool_samples_each_endpoint": training_pool_n,
            "history": history, "training_seconds": training_seconds,
        })
        flow = MLPReferenceFlow(
            flow.params,
            substeps_per_interval=int(phase2["reference"]["rk4_substeps_per_time_interval"]),
            metadata=metadata,
        )
        save_npz_checkpoint(checkpoint, flow.params, metadata)
        write_csv(table_dir / "conditioned_reference_training_history.csv", history)
        print(f"[conditioned] trained in {training_seconds:.1f}s", flush=True)

    evaluation_times = times[evaluation_indices]
    evaluation_days = days[evaluation_indices]
    sizes = [int(value) for value in cfg["reference_bank"]["sizes"]]
    largest = max(sizes)
    bank_path = model_dir / f"reference_bank_eval_{largest}.npz"
    bank_signature = json.dumps({
        "checkpoint_sha256": sha256(checkpoint), "estimator_sha256": sha256(estimator_path),
        "seed": int(cfg["reference_bank"]["seed"]), "particles": largest,
        "evaluation_times": evaluation_times.tolist(),
    }, sort_keys=True)
    nodes = initial = None
    if bank_path.exists() and not args.force_bank:
        with np.load(bank_path, allow_pickle=False) as data:
            if str(data["signature"].item()) == bank_signature:
                nodes = np.asarray(data["nodes_km"], dtype=np.float64)
                initial = np.asarray(data["initial_km"], dtype=np.float64)
                print(f"[conditioned] reusing {largest:,}-path bank", flush=True)
    if nodes is None:
        bank_rng = np.random.default_rng(int(cfg["reference_bank"]["seed"]))
        initial, bank_stats = rejection_sample_whole_mixture(x0, H0, domain, bank_rng, largest)
        nodes = np.empty((len(evaluation_times), largest, 2), dtype=np.float64)
        chunk = int(cfg["reference_bank"]["rollout_chunk_size"])
        started = time.perf_counter()
        for start in range(0, largest, chunk):
            stop = min(start + chunk, largest)
            normalized = (initial[start:stop] - center) / scale
            nodes[:, start:stop] = np.asarray(flow.rollout(
                jnp.asarray(normalized), jnp.asarray(evaluation_times)
            )) * scale + center
            if stop == chunk or stop % 50000 == 0 or stop == largest:
                print(f"[conditioned] rollout {stop:,}/{largest:,}", flush=True)
        np.savez_compressed(
            bank_path, nodes_km=nodes, initial_km=initial,
            evaluation_indices=evaluation_indices, evaluation_times=evaluation_times,
            evaluation_days=evaluation_days, signature=np.asarray(bank_signature),
            checkpoint_sha256=np.asarray(sha256(checkpoint)),
            estimator_sha256=np.asarray(sha256(estimator_path)),
            sampling_acceptance=np.asarray(bank_stats["effective_acceptance_rate"]),
            final_test_accessed=np.asarray(False),
        )
        print(f"[conditioned] rollout completed in {time.perf_counter()-started:.1f}s", flush=True)
    unique_rows = []
    intermediate_position = len(evaluation_times) // 2
    for size in sizes:
        unique_rows.append({
            "particle_count": size,
            "unique_initial_exact": len(np.unique(initial[:size], axis=0)),
            "unique_midpoint_exact": len(np.unique(nodes[intermediate_position, :size], axis=0)),
            "unique_day45_exact": len(np.unique(nodes[-1, :size], axis=0)),
        })
    write_csv(table_dir / "conditioned_reference_unique_paths.csv", unique_rows)
    if any(any(int(row[key]) != int(row["particle_count"]) for key in [
        "unique_initial_exact", "unique_midpoint_exact", "unique_day45_exact"
    ]) for row in unique_rows):
        raise RuntimeError("conditioned reference lost unique support scaling")

    generated_metrics = [
        endpoint_metrics(x0, nodes[0], "conditioned_generated", "day0", frozen_mmd_bandwidth, domain, endpoint_rng),
        endpoint_metrics(x1, nodes[-1], "conditioned_generated", "day45", frozen_mmd_bandwidth, domain, endpoint_rng),
    ]
    for row in generated_metrics:
        row["accepted"] = metric_acceptance(row, thresholds)
    all_endpoint_rows = estimator_metrics + generated_metrics
    write_csv(table_dir / "conditioned_endpoint_metrics.csv", all_endpoint_rows)
    generated_endpoint_passed = all(row["accepted"] for row in generated_metrics)
    endpoint_passed = conditioned_estimator_passed and generated_endpoint_passed
    write_json(table_dir / "conditioned_endpoint_acceptance.json", {
        "criteria_frozen_from": cfg["endpoint_acceptance"]["frozen_from"],
        "criteria": thresholds, "conditioned_estimator_passed": conditioned_estimator_passed,
        "generated_reference_passed": generated_endpoint_passed,
        "passed": endpoint_passed, "generated_metrics": generated_metrics,
        "final_test_artifact_loaded": False, "intermediate_positions_used_for_tuning": False,
    })
    if not endpoint_passed:
        day45_generated = next(row for row in generated_metrics if row["endpoint"] == "day45")
        write_json(table_dir / "domain_conditioned_reference_summary.json", {
            "endpoint_gate_passed": False, "support_audit_run": False,
            "full_bank_sweep_run": False, "stochastic_bridge_justified": False,
            "stop_reason": "conditioned generated day-45 outside-domain mass exceeds the frozen threshold",
            "generated_day45_outside_domain_fraction": day45_generated["outside_domain_fraction"],
            "frozen_maximum_outside_domain_fraction": thresholds["maximum_generated_endpoint_outside_domain_fraction"],
            "conditioned_estimator_passed": conditioned_estimator_passed,
            "generated_reference_passed": generated_endpoint_passed,
            "unique_paths": unique_rows,
            "stochastic_bridge_decision_reason": "not justified because the generated endpoint gate failed before the 20-case structural-mismatch test",
            "final_test_artifact_loaded": False,
        })
        raise RuntimeError("conditioned generated reference failed endpoint gate; intermediate audit stopped")
    print("[conditioned] endpoint gate passed", flush=True)

    with (table_dir / "reference_support_cases.json").open(encoding="utf-8") as handle:
        cases = json.load(handle)["cases"]
    if len(cases) != int(cfg["support_acceptance"]["same_case_count"]):
        raise RuntimeError("frozen diagnostic case count changed")
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
        print(f"[conditioned LP] M={size:,}", flush=True)
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
                category = "C_structural_intermediate_mismatch"
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
        print(f"[conditioned LP] completed M={size:,}; elapsed={time.perf_counter()-started:.1f}s", flush=True)
    write_csv(table_dir / "reference_support_lp_conditioned.csv", audit_rows)
    largest_rows = [row for row in audit_rows if row["particle_count"] == largest]
    tolerance = float(p["accept_residual"])
    lp_feasible = sum(row["minimum_linf_residual"] <= tolerance for row in largest_rows)
    native_healthy = sum(row["native_healthy"] for row in largest_rows)
    support_passed = bool(
        lp_feasible >= int(cfg["support_acceptance"]["minimum_lp_feasible_cases_at_200000"])
        and native_healthy >= int(cfg["support_acceptance"]["minimum_native_healthy_cases_at_200000"])
    )
    feasible_ess = np.asarray([
        row["native_ess_fraction"] for row in largest_rows
        if row["minimum_linf_residual"] <= tolerance
    ])
    feasible_multiplier = np.asarray([
        row["native_multiplier_norm"] for row in largest_rows
        if row["minimum_linf_residual"] <= tolerance
    ])
    categories = {
        category: sum(row["classification"] == category for row in largest_rows)
        for category in [
            "A_support_repaired", "B_feasible_but_unhealthy",
            "C_structural_intermediate_mismatch",
        ]
    }
    audit_summary = {
        "endpoint_gate_passed": True, "support_audit_run": True,
        "support_audit_passed": support_passed, "particle_count": largest,
        "lp_feasible_cases": lp_feasible, "native_healthy_cases": native_healthy,
        "case_count": len(cases), "categories": categories,
        "feasible_ess_quantiles": {
            str(q): float(np.quantile(feasible_ess, q)) if len(feasible_ess) else None
            for q in [0.0, 0.25, 0.5, 0.75, 1.0]
        },
        "feasible_multiplier_norm_quantiles": {
            str(q): float(np.quantile(feasible_multiplier, q)) if len(feasible_multiplier) else None
            for q in [0.0, 0.25, 0.5, 0.75, 1.0]
        },
        "full_bank_sweep_authorized": support_passed,
        "full_bank_sweep_run": False, "stochastic_bridge_justified": not support_passed,
        "final_test_artifact_loaded": False,
    }
    write_json(table_dir / "domain_conditioned_reference_summary.json", audit_summary)
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
    ax.set_title("Frozen 20-case audit: domain-conditioned deterministic reference")
    fig.tight_layout(); fig.savefig(lp_fig_dir / "conditioned_lp_scaling.png", dpi=190); plt.close(fig)

    with np.load(old_model_dir / "reference_bank.npz", allow_pickle=False) as data:
        old_nodes = np.asarray(data["nodes_km"], dtype=np.float64)[evaluation_indices]
    with np.load(unbounded_model_dir / "reference_bank_eval_200000.npz", allow_pickle=False) as data:
        unbounded_nodes = np.asarray(data["nodes_km"], dtype=np.float64)
    plot_rng = np.random.default_rng(int(cfg["seed"]) + 501)
    display_days = [2.5, 5.0, 10.0]
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), sharex=True, sharey=True)
    for column, day in enumerate(display_days):
        t = int(np.argmin(np.abs(evaluation_days - day)))
        source_index = evaluation_indices[t]
        collections = [
            ("old discrete", old_nodes[t], "#4c78a8"),
            ("unbounded KDE", unbounded_nodes[t], "#f58518"),
            ("conditioned KDE", nodes[t], "#54a24b"),
        ]
        for row, (label, bank, color) in enumerate(collections):
            sample_index = plot_rng.choice(len(bank), min(2500, len(bank)), replace=False)
            axes[row, column].scatter(bank[sample_index, 0], bank[sample_index, 1], s=3, alpha=.14, color=color)
            axes[row, column].scatter(inference[:, source_index, 0], inference[:, source_index, 1], s=10, alpha=.55, color="black")
            axes[row, column].set_title(f"{label}, day {day:g}"); axes[row, column].set_aspect("equal"); axes[row, column].grid(alpha=.15)
    fig.supxlabel("x (km)"); fig.supylabel("y (km)"); fig.tight_layout()
    fig.savefig(compare_fig_dir / "old_unbounded_conditioned_support.png", dpi=190); plt.close(fig)
    print(
        f"[conditioned] 20-case audit LP={lp_feasible}/20, native healthy={native_healthy}/20, "
        f"passed={support_passed}", flush=True,
    )
    if support_passed:
        print("[conditioned] full-bank sweep authorized; run conditioned_full_bank_feasibility.py", flush=True)
    else:
        print("[conditioned] stopped before full bank; endpoint-only stochastic bridge is justified", flush=True)


if __name__ == "__main__":
    main()
