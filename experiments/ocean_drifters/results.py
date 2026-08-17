"""Reusable law-level tables and publication figures for ocean drifters."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

from mfsi.cache import write_json_atomic


COLORS = {
    "blue": "#2B6CB0",
    "orange": "#D97706",
    "green": "#2F855A",
    "red": "#C53030",
    "gray": "#718096",
    "light": "#BEE3F8",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _projection_diagnostics(experiment) -> dict[int, dict[str, float]]:
    regular: dict[tuple[int, int], dict[str, float]] = {}
    for row in _read_csv(experiment.paths["projection_table"]):
        key = (int(row["design_index"]), int(row["source_time_index"]))
        regular[key] = {
            "kl": float(row["kl_divergence"]),
            "log10_ess": float(row["log10_intrinsic_ess"]),
            "lambda_norm": float(row["lambda_norm"]),
            "condition": float(row["covariance_condition_regularized"]),
            "residual": float(row["verified_l2_residual"]),
        }
    final_adaptive: dict[tuple[int, int], dict[str, str]] = {}
    for row in _read_csv(experiment.paths["adaptive_level_table"]):
        key = (int(row["design_index"]), int(row["source_time_index"]))
        if key not in final_adaptive or int(row["refinement_level"]) > int(final_adaptive[key]["refinement_level"]):
            final_adaptive[key] = row
    for key, row in final_adaptive.items():
        regular[key] = {
            "kl": float(row["kl_divergence"]),
            "log10_ess": float(row["log10_intrinsic_ess"]),
            "lambda_norm": float(row["lambda_norm"]),
            "condition": float(row["covariance_condition_regularized"]),
            "residual": float(row["verified_l2_residual"]),
        }
    output: dict[int, dict[str, float]] = {}
    for design in range(len(experiment.sensor_bank.design_ids)):
        values = [value for (index, _), value in regular.items() if index == design]
        output[design] = {
            "maximum_kl": max(value["kl"] for value in values),
            "integrated_kl_uniform": float(np.mean([value["kl"] for value in values])),
            "minimum_log10_intrinsic_ess": min(value["log10_ess"] for value in values),
            "maximum_lambda_norm": max(value["lambda_norm"] for value in values),
            "maximum_covariance_condition": max(value["condition"] for value in values),
            "maximum_projection_residual": max(value["residual"] for value in values),
        }
    return output


def _geometry(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    distance = cdist(a, b)
    row, col = linear_sum_assignment(distance)
    matched_rms = float(np.sqrt(np.mean(distance[row, col] ** 2)))
    hausdorff = float(max(np.max(np.min(distance, axis=1)), np.max(np.min(distance, axis=0))))
    average_nearest = float(0.5 * (np.mean(np.min(distance, axis=1)) + np.mean(np.min(distance, axis=0))))
    return matched_rms, hausdorff, average_nearest


def _classical_mds(distance: np.ndarray) -> np.ndarray:
    n = len(distance)
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (distance**2) @ centering
    eigenvalue, eigenvector = np.linalg.eigh(gram)
    order = np.argsort(eigenvalue)[::-1][:2]
    return eigenvector[:, order] * np.sqrt(np.maximum(eigenvalue[order], 0.0))[None]


def _risk_ranked(summary: list[dict[str, Any]], best: str, r_star: float, ceiling: float, stem: Path) -> None:
    rows = sorted(summary, key=lambda row: row["validation_risk"])
    rank = np.arange(1, len(rows) + 1)
    risk = np.asarray([row["validation_risk"] for row in rows])
    low = np.asarray([row["bootstrap_CI_low"] for row in rows])
    high = np.asarray([row["bootstrap_CI_high"] for row in rows])
    near = np.asarray([row["near_optimal"] for row in rows])
    fig, axis = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    axis.fill_between(rank, low, high, color=COLORS["light"], alpha=0.45, linewidth=0, label="95% whole-ID bootstrap interval")
    axis.plot(rank[~near], risk[~near], color=COLORS["gray"], linewidth=1.3, label="other admissible layouts")
    axis.plot(rank[near], risk[near], color=COLORS["blue"], linewidth=2.2, label="frozen near-optimal set")
    best_position = next(i for i, row in enumerate(rows) if row["design_id"] == best)
    axis.scatter(rank[best_position], risk[best_position], marker="*", s=150, color=COLORS["orange"], edgecolor="black", linewidth=0.5, zorder=5, label=best)
    axis.axhline(r_star, color="black", linestyle="--", linewidth=1, label=r"$R^\star$")
    axis.axhline(ceiling, color=COLORS["red"], linestyle=":", linewidth=1.5, label=r"$R^\star+\epsilon$")
    axis.set(xlabel="layout rank", ylabel=r"validation RFF-MMD$^2$ risk", xlim=(0.5, len(rows) + 0.5))
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    _save_figure(fig, stem)


def _risk_distribution(summary: list[dict[str, Any]], r_star: float, ceiling: float, stem: Path) -> None:
    risk = np.asarray([row["validation_risk"] for row in summary])
    fig, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.hist(risk, bins=34, color=COLORS["blue"], alpha=0.82, edgecolor="white")
    axis.axvline(r_star, color="black", linestyle="--", label=r"$R^\star$")
    axis.axvline(ceiling, color=COLORS["red"], linestyle=":", linewidth=1.7, label=r"$R^\star+\epsilon$")
    axis.axvline(np.median(risk), color=COLORS["orange"], linestyle="-.", label="median")
    axis.set(xlabel=r"validation RFF-MMD$^2$ risk", ylabel="layout count")
    axis.grid(alpha=0.18, axis="y")
    axis.legend(frameon=False)
    _save_figure(fig, stem)


def _sensor_maps(experiment, near_indices: np.ndarray, representatives: list[int], stem: Path) -> None:
    inference = experiment.cohort.inference[:, ::8].reshape(-1, 2)
    centers = experiment.sensor_bank.centers_km
    bounds = experiment.cfg["scientific"]["domain_km"]
    fig, axes = plt.subplots(1, len(representatives), figsize=(4.1 * len(representatives), 4.1), sharex=True, sharey=True, constrained_layout=True)
    for axis, design in zip(np.atleast_1d(axes), representatives, strict=True):
        axis.hexbin(inference[:, 0], inference[:, 1], gridsize=48, bins="log", mincnt=1, cmap="Greys", alpha=0.65)
        axis.scatter(centers[design, :, 0], centers[design, :, 1], c=np.arange(4), cmap="tab10", marker="X", s=95, edgecolor="black", linewidth=0.6)
        axis.set_title(experiment.sensor_bank.design_ids[design])
        axis.set_aspect("equal")
        axis.grid(alpha=0.15)
    axes[0].set_xlim(bounds[0], bounds[1]); axes[0].set_ylim(bounds[2], bounds[3])
    fig.supxlabel("projected x (km)"); fig.supylabel("projected y (km)")
    _save_figure(fig, stem.with_name("near_optimal_sensor_maps"))

    fig, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    axis.hexbin(inference[:, 0], inference[:, 1], gridsize=56, bins="log", mincnt=1, cmap="Greys", alpha=0.5)
    all_centers = centers[near_indices].reshape(-1, 2)
    axis.scatter(all_centers[:, 0], all_centers[:, 1], s=14, color=COLORS["blue"], alpha=0.45, edgecolor="none", label="272 sensor placements from 68 layouts")
    axis.set(xlabel="projected x (km)", ylabel="projected y (km)", xlim=(bounds[0], bounds[1]), ylim=(bounds[2], bounds[3]))
    axis.set_aspect("equal"); axis.grid(alpha=0.15); axis.legend(frameon=False)
    _save_figure(fig, stem.with_name("near_optimal_sensor_centers_aggregate"))


def _burden_plot(summary: list[dict[str, Any]], stem: Path) -> dict[str, float]:
    metrics = [
        ("maximum_KL", "maximum KL", False),
        ("minimum_log10_intrinsic_ess", r"minimum $\log_{10}$ intrinsic ESS", False),
        ("maximum_lambda_norm", r"maximum $\|\lambda\|_2$", True),
        ("maximum_covariance_condition", "worst covariance condition", True),
    ]
    risk = np.asarray([row["validation_risk"] for row in summary])
    near = np.asarray([row["near_optimal"] for row in summary])
    correlations: dict[str, float] = {}
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.5), constrained_layout=True)
    for axis, (key, label, logx) in zip(axes.ravel(), metrics, strict=True):
        value = np.asarray([row[key] for row in summary], dtype=np.float64)
        correlations[f"all_{key}"] = float(spearmanr(value, risk).statistic)
        correlations[f"near_optimal_{key}"] = float(spearmanr(value[near], risk[near]).statistic)
        plot_value = np.log10(np.maximum(value, 1e-300)) if logx else value
        axis.scatter(plot_value[~near], risk[~near], s=13, color=COLORS["gray"], alpha=0.4)
        axis.scatter(plot_value[near], risk[near], s=19, color=COLORS["blue"], alpha=0.8, label="near-optimal")
        if logx:
            label = rf"$\log_{{10}}$({label})"
        axis.set(xlabel=label, ylabel="validation risk")
        axis.grid(alpha=0.18)
        axis.text(
            0.03, 0.95,
            f"Spearman ρ: all={correlations[f'all_{key}']:.2f}, near={correlations[f'near_optimal_{key}']:.2f}",
            transform=axis.transAxes, va="top", fontsize=8,
        )
    axes[0, 0].legend(frameon=False)
    _save_figure(fig, stem)
    return correlations


def _mmd_time_plots(experiment, risk: dict[str, Any], representatives: list[int], near_indices: np.ndarray, stem: Path) -> dict[str, float]:
    design_to_local = {int(design): local for local, design in enumerate(risk["design_indices"])}
    fig, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for design in representatives:
        local = design_to_local[design]
        axis.plot(risk["evaluation_days"], risk["risk_by_time"][local], marker="o", markersize=3, linewidth=1.4, label=experiment.sensor_bank.design_ids[design])
    axis.set(xlabel="day", ylabel=r"validation MMD$^2$", yscale="log")
    axis.grid(alpha=0.2); axis.legend(frameon=False, fontsize=8, ncol=2)
    _save_figure(fig, stem.with_name("mmd_by_time"))

    near_order = near_indices[np.argsort([risk["risk"][design_to_local[index]] for index in near_indices])]
    matrix = np.asarray([risk["risk_by_time"][design_to_local[index]] for index in near_order])
    fig, axis = plt.subplots(figsize=(9.2, 7.0), constrained_layout=True)
    image = axis.imshow(np.log10(np.maximum(matrix, 1e-12)), aspect="auto", origin="upper", cmap="viridis", extent=[risk["evaluation_days"][0], risk["evaluation_days"][-1], len(near_order), 1])
    axis.set(xlabel="day", ylabel="near-optimal layout (risk rank)")
    colorbar = fig.colorbar(image, ax=axis); colorbar.set_label(r"$\log_{10}$ validation MMD$^2$")
    _save_figure(fig, stem.with_name("near_optimal_mmd_time_heatmap"))
    mean_by_time = matrix.mean(axis=0)
    range_by_time = np.ptp(matrix, axis=0)
    return {
        "largest_mean_near_optimal_mmd_day": float(risk["evaluation_days"][np.argmax(mean_by_time)]),
        "largest_mean_near_optimal_mmd": float(np.max(mean_by_time)),
        "largest_near_optimal_mmd_range_day": float(risk["evaluation_days"][np.argmax(range_by_time)]),
        "largest_near_optimal_mmd_range": float(np.max(range_by_time)),
    }


def _representative_laws(experiment, representatives: list[int], output_dir: Path) -> None:
    rows = _read_csv(experiment.paths["projection_table"])
    lookup = {(int(row["design_index"]), int(row["source_time_index"])): row for row in rows}
    selected_sources = [10, 90, 180]
    nx, ny = (int(value) for value in experiment.cfg["projection"]["grid_resolution"])
    bounds = np.asarray(experiment.cfg["scientific"]["domain_km"], dtype=np.float64)
    xmin, xmax, ymin, ymax = bounds
    x = xmin + (np.arange(nx) + 0.5) * (xmax - xmin) / nx
    y = ymin + (np.arange(ny) + 0.5) * (ymax - ymin) / ny
    xx, yy = np.meshgrid(x, y, indexing="xy")
    points = np.stack((xx.ravel(), yy.ravel()), axis=-1)
    checkpoint = experiment.paths["reference_checkpoint"]
    prefix = experiment.cfg["artifacts"]["reference_checkpoint"]["sha256"][:12]
    validation = experiment.cohort.validation
    for design in representatives:
        centers = experiment.sensor_bank.centers_km[design]
        phi = np.exp(-0.5 * np.sum((points[:, None] - centers[None]) ** 2, axis=-1) / experiment.sensor_bank.sigma_km**2)
        fig, axes = plt.subplots(len(selected_sources), 3, figsize=(12.2, 10.4), sharex=True, sharey=True, constrained_layout=True)
        for row_index, source in enumerate(selected_sources):
            cache = experiment.reference_density_cache / f"density_{prefix}_t{source:03d}_{nx}x{ny}.npz"
            with np.load(cache, allow_pickle=False) as data:
                log_base = np.asarray(data["log_base_mass"], dtype=np.float64)
            projection = lookup[(design, source)]
            lam = np.asarray([float(projection[f"lambda_{index}"]) for index in range(4)])
            log_projected = log_base + phi @ lam
            log_projected -= np.max(log_projected)
            projected_mass = np.exp(log_projected); projected_mass /= projected_mass.sum()
            reference_mass = np.exp(log_base)
            for axis, mass, title in [
                (axes[row_index, 0], reference_mass, "frozen reference"),
                (axes[row_index, 1], projected_mass, "I-projected law"),
            ]:
                image = np.log10(np.maximum(mass.reshape(ny, nx), 1e-300))
                lower, upper = np.quantile(image[np.isfinite(image)], [0.15, 0.995])
                axis.contourf(xx, yy, image, levels=np.linspace(lower, upper, 18), cmap="mako" if "mako" in plt.colormaps() else "viridis", extend="both")
                axis.scatter(centers[:, 0], centers[:, 1], marker="x", color="white", s=25, linewidth=1)
                axis.set_title(title)
            axes[row_index, 2].hexbin(validation[:, source, 0], validation[:, source, 1], gridsize=20, mincnt=1, cmap="viridis")
            axes[row_index, 2].scatter(centers[:, 0], centers[:, 1], marker="x", color="white", s=25, linewidth=1)
            axes[row_index, 2].set_title("70 validation drifters")
            axes[row_index, 0].set_ylabel(f"day {experiment.cohort.relative_days[source]:g}\ny (km)")
        for axis in axes[-1]: axis.set_xlabel("x (km)")
        for axis in axes.ravel(): axis.set_xlim(xmin, xmax); axis.set_ylim(ymin, ymax); axis.set_aspect("equal")
        fig.suptitle(experiment.sensor_bank.design_ids[design])
        _save_figure(fig, output_dir / experiment.sensor_bank.design_ids[design])


def generate_law_results(experiment, risk: dict[str, Any], analysis_dir: Path) -> dict[str, Any]:
    """Generate all frozen law-level outputs without changing scientific choices."""
    table_dir = analysis_dir / "tables"
    figure_dir = analysis_dir / "figures/results"
    table_dir.mkdir(parents=True, exist_ok=True); figure_dir.mkdir(parents=True, exist_ok=True)
    burden = _projection_diagnostics(experiment)
    freeze = json.loads(experiment.paths["risk_freeze"].read_text(encoding="utf-8"))
    near_ids = set(freeze["near_optimal_design_ids"])
    near_indices = np.asarray([int(value.split("_")[-1]) for value in freeze["near_optimal_design_ids"]], dtype=int)
    risk_by_design = {int(row["design_index"]): row for row in risk["rows"]}
    summary: list[dict[str, Any]] = []
    for design in range(len(experiment.sensor_bank.design_ids)):
        risk_row = risk_by_design[design]
        diag = burden[design]
        row = {
            "design_index": design,
            "design_id": experiment.sensor_bank.design_ids[design],
            "rank": int(risk_row["rank"]),
            "validation_risk": float(risk_row["risk"]),
            "bootstrap_SE": float(risk_row["bootstrap_standard_error"]),
            "bootstrap_CI_low": float(risk_row["bootstrap_ci_lower"]),
            "bootstrap_CI_high": float(risk_row["bootstrap_ci_upper"]),
            "bootstrap_rank_median": float(risk_row["bootstrap_rank_median"]),
            "bootstrap_rank_CI_low": float(risk_row["bootstrap_rank_ci_lower"]),
            "bootstrap_rank_CI_high": float(risk_row["bootstrap_rank_ci_upper"]),
            "near_optimal": experiment.sensor_bank.design_ids[design] in near_ids,
            "maximum_KL": diag["maximum_kl"],
            "integrated_KL_uniform": diag["integrated_kl_uniform"],
            "minimum_intrinsic_ESS": 10.0 ** diag["minimum_log10_intrinsic_ess"] if diag["minimum_log10_intrinsic_ess"] > -308 else 0.0,
            "minimum_log10_intrinsic_ess": diag["minimum_log10_intrinsic_ess"],
            "maximum_lambda_norm": diag["maximum_lambda_norm"],
            "maximum_covariance_condition": diag["maximum_covariance_condition"],
            "maximum_projection_residual": diag["maximum_projection_residual"],
        }
        for sensor, (x, y) in enumerate(experiment.sensor_bank.centers_km[design], start=1):
            row[f"sensor_{sensor}_x_km"] = float(x); row[f"sensor_{sensor}_y_km"] = float(y)
        summary.append(row)
    _write_csv(table_dir / "law_level_summary.csv", sorted(summary, key=lambda row: row["rank"]))

    pair_rows: list[dict[str, Any]] = []
    distance_matrix = np.zeros((len(near_indices), len(near_indices)), dtype=np.float64)
    for left_position, left in enumerate(near_indices):
        for right_position in range(left_position + 1, len(near_indices)):
            right = near_indices[right_position]
            rms, hausdorff, nearest = _geometry(experiment.sensor_bank.centers_km[left], experiment.sensor_bank.centers_km[right])
            distance_matrix[left_position, right_position] = distance_matrix[right_position, left_position] = rms
            pair_rows.append({
                "left_design_id": experiment.sensor_bank.design_ids[left],
                "right_design_id": experiment.sensor_bank.design_ids[right],
                "hungarian_matched_rms_km": rms,
                "hausdorff_km": hausdorff,
                "symmetric_average_nearest_sensor_km": nearest,
            })
    _write_csv(table_dir / "near_optimal_geometry.csv", pair_rows)
    coordinates = _classical_mds(distance_matrix)
    _write_csv(table_dir / "near_optimal_geometry_embedding.csv", [{
        "design_id": experiment.sensor_bank.design_ids[design],
        "mds_1_km": float(coordinates[position, 0]),
        "mds_2_km": float(coordinates[position, 1]),
    } for position, design in enumerate(near_indices)])

    r_star = float(freeze["best_validation_risk"]); ceiling = float(freeze["risk_ceiling"])
    _risk_ranked(summary, freeze["best_design_id"], r_star, ceiling, figure_dir / "validation_risk_ranked")
    _risk_distribution(summary, r_star, ceiling, figure_dir / "validation_risk_distribution")
    best = int(freeze["best_design_id"].split("_")[-1])
    farthest_positions = np.argsort(distance_matrix[np.flatnonzero(near_indices == best)[0]])[::-1][:3]
    map_representatives = [best, *near_indices[farthest_positions].tolist()]
    _sensor_maps(experiment, near_indices, map_representatives, figure_dir / "sensor_maps")
    fig, axis = plt.subplots(figsize=(7.0, 5.6), constrained_layout=True)
    scatter = axis.scatter(coordinates[:, 0], coordinates[:, 1], c=[risk_by_design[int(index)]["risk"] for index in near_indices], cmap="viridis", s=40)
    axis.scatter(coordinates[near_indices == best, 0], coordinates[near_indices == best, 1], marker="*", color=COLORS["orange"], edgecolor="black", s=160)
    axis.set(xlabel="classical MDS coordinate 1 (km)", ylabel="classical MDS coordinate 2 (km)")
    axis.grid(alpha=0.18); fig.colorbar(scatter, ax=axis, label="validation risk")
    _save_figure(fig, figure_dir / "near_optimal_geometry_mds")
    correlations = _burden_plot(summary, figure_dir / "risk_vs_projection_burden")

    ordered_near = sorted(near_indices, key=lambda index: risk_by_design[int(index)]["risk"])
    median_near = int(ordered_near[len(ordered_near) // 2])
    high_burden = int(max(near_indices, key=lambda index: burden[int(index)]["maximum_kl"]))
    non_near = [index for index in range(512) if experiment.sensor_bank.design_ids[index] not in near_ids]
    poor = int(max(non_near, key=lambda index: risk_by_design[index]["risk"]))
    representatives = [best, median_near, high_burden, poor]
    time_summary = _mmd_time_plots(experiment, risk, representatives, near_indices, figure_dir / "mmd")
    _representative_laws(experiment, representatives, figure_dir / "representative_laws")

    rms_values = np.asarray([row["hungarian_matched_rms_km"] for row in pair_rows])
    summary_payload = {
        "schema_version": 1,
        "layout_count": 512,
        "near_optimal_layout_count": 68,
        "best_design_id": freeze["best_design_id"],
        "R_star": r_star,
        "epsilon": float(freeze["frozen_additive_epsilon"]),
        "risk_ceiling": ceiling,
        "risk_minimum": float(min(row["validation_risk"] for row in summary)),
        "risk_median": float(np.median([row["validation_risk"] for row in summary])),
        "risk_maximum": float(max(row["validation_risk"] for row in summary)),
        "geometry_hungarian_rms_km": {
            "minimum": float(rms_values.min()), "median": float(np.median(rms_values)),
            "maximum": float(rms_values.max()),
        },
        "spearman_risk_vs_projection_burden": correlations,
        "representative_design_ids": [experiment.sensor_bank.design_ids[index] for index in representatives],
        "map_design_ids": [experiment.sensor_bank.design_ids[index] for index in map_representatives],
        **time_summary,
        "final_test_accessed": False,
    }
    write_json_atomic(table_dir / "law_level_results_summary.json", summary_payload)
    return summary_payload
