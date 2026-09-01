#!/usr/bin/env python3
"""Render frozen prospective-vortices validation results and held-out media.

The renderer is deterministic post-processing. It reads the already frozen
selection, held-out reference, and hidden validation banks; it never trains a model,
optimizes a geometry, or changes a scientific result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, PowerNorm  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

from evaluator import AggregateObservationBank, ProspectiveEvaluator  # noqa: E402
from prospective_data import TargetProspectiveData  # noqa: E402
from validate import _realized_bank_and_moments  # noqa: E402


jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_RUN = HERE / "outputs" / "prospective_reflected_single_seed_pareto"
DEFAULT_PLOTS = HERE / "plots"
DEFAULT_RESULTS = HERE / "results"
ALLOWANCES = ((0.005, "0p5"), (0.01, "1p0"), (0.02, "2p0"))
TIME_INDICES = (0, 5, 15, 20)
SENSOR_COLORS = ("#1CA6A3", "#F28E5B", "#9271C2", "#5AAA70")
ALLOWANCE_COLORS = ("#2A9D8F", "#E78A45", "#8B5FBF")
FLOW_CMAP = LinearSegmentedColormap.from_list(
    "prospective_vortex_density",
    ("#F7F3EA", "#DCC8A6", "#D77A61", "#8D3D55", "#272442"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--static-dpi", type=int, default=240)
    parser.add_argument("--gif-dpi", type=int, default=92)
    parser.add_argument("--gif-grid-nx", type=int, default=180)
    parser.add_argument("--gif-fps", type=float, default=7.0)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#C9C3B8",
            "axes.linewidth": 0.7,
            "figure.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def verify_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    validation_path = run_dir / "results" / "validation_result.json"
    if not validation_path.exists():
        raise FileNotFoundError(f"missing frozen rendering input: {validation_path}")
    validation = load_json(validation_path)
    reference_ids = list(validation.get("reference_ids", []))
    if len(reference_ids) != 1:
        raise RuntimeError("renderer requires exactly one held-out reference")
    reference_id = str(reference_ids[0])
    required = {
        "config": run_dir / "results" / "resolved_config.json",
        "combined": run_dir / "results" / "combined_frozen_manifest.json",
        "validation": validation_path,
        "states": run_dir / "hidden_validation" / "v6_hidden_state_bank.npz",
        "randomness": run_dir / "hidden_validation" / "v6_hidden_observation_randomness.npz",
        "aggregate": run_dir / "shared" / "prospective" / "aggregate_predictions.npz",
        "endpoint": run_dir / "shared" / "endpoint_reference" / "endpoint_data.npz",
        "rollout": run_dir / "shared" / "references" / "evaluation" / reference_id
        / "endpoint_reference" / "reference_rollout.npz",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing frozen rendering input(s): " + ", ".join(missing))
    cfg = load_json(required["config"])
    combined = load_json(required["combined"])
    allowed_statuses = {
        "all_v6a_allowances_frozen_before_evaluation_references_and_hidden_validation",
        "repaired_three_point_frontier_frozen_before_e1_and_hidden_validation",
    }
    if combined.get("status") not in allowed_statuses:
        raise RuntimeError("the Pareto set is not in its required frozen state")
    if [float(row["allowance"]) for row in validation["points"]] != [
        value for value, _ in ALLOWANCES
    ]:
        raise RuntimeError("held-out validation does not contain exactly 0.5%, 1%, and 2%")
    if not all(
        bool(row["all_reference_risk_pass"])
        and bool(row["numerical_certification_pass"])
        for row in validation["points"]
    ):
        raise RuntimeError("refusing to publish a risk- or numerically-invalid validation")
    return cfg, combined, validation, reference_id


def publish_summary(
    run_dir: Path,
    results_dir: Path,
    combined: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    results_dir.mkdir(parents=True, exist_ok=True)
    reference_id = str(validation["reference_ids"][0])
    held_key = f"held_out_{reference_id}"
    common_law_action = None
    for frozen in combined["points"]:
        law_selection = frozen["selected"]["Law"]
        full_selection = frozen["selected"]["v6a"]
        if np.array_equal(
            np.asarray(law_selection["eta"]), np.asarray(full_selection["eta"])
        ):
            common_law_action = full_selection["full_distribution"]["mean"]
            break
    points: list[dict[str, Any]] = []
    for frozen, held_out in zip(combined["points"], validation["points"], strict=True):
        law_selection = frozen["selected"]["Law"]
        full_selection = frozen["selected"]["v6a"]
        law_validation = held_out["methods"]["Law"]
        full_validation = held_out["methods"]["v6a"]
        comparison = held_out["v6a_minus_law"]
        points.append(
            {
                "allowance_percent": 100.0 * float(held_out["allowance"]),
                "law_eta": law_validation["eta"],
                "full_eta": full_validation["eta"],
                "selection_D0": {
                    "law_risk": law_selection["risk_by_reference"]["D0"],
                    "full_risk": full_selection["risk_by_reference"]["D0"],
                    "risk_limit": (
                        (1.0 + float(held_out["allowance"]))
                        * law_selection["risk_by_reference"]["D0"]
                    ),
                    "law_full_action": (
                        law_selection.get("full_distribution", {}).get("mean")
                        if law_selection.get("full_distribution") is not None
                        else common_law_action
                    ),
                    "full_full_action": full_selection["full_distribution"]["mean"],
                },
                held_key: {
                    "law_risk": law_validation["risk_by_reference"][reference_id],
                    "full_risk": full_validation["risk_by_reference"][reference_id],
                    "risk_ratio": (
                        full_validation["risk_by_reference"][reference_id]
                        / law_validation["risk_by_reference"][reference_id]
                    ),
                    "law_full_action": law_validation["full_action_by_reference"][reference_id],
                    "full_full_action": full_validation["full_action_by_reference"][reference_id],
                    "action_reduction": comparison["ratio_of_means_reduction"],
                    "paired_action_difference_95_ci": comparison["paired_t_95_ci"],
                    "valid_trials": comparison["valid_pair_count"],
                },
                "risk_pass": held_out["all_reference_risk_pass"],
                "paired_action_ci_below_zero": held_out["paired_action_ci_below_zero"],
                "numerical_certification_pass": held_out["numerical_certification_pass"],
                "strict_success": held_out["strict_success"],
            }
        )
    summary = {
        "schema_version": 1,
        "status": (
            "STRICT_SUCCESS_ALL_THREE_HELD_OUT_POINTS"
            if all(row["strict_success"] for row in points)
            else f"VALIDATION_COMPLETE_{sum(bool(row['strict_success']) for row in points)}_OF_3_STRICT_POINTS"
        ),
        "experiment": validation["experiment"],
        "selection_reference": "D0",
        "validation_reference": reference_id,
        "selection_frozen_before_validation_reference_training_and_hidden_validation": True,
        "validation_trials": 64,
        "points": points,
        "source_artifacts": {
            "combined_frozen_manifest_sha256": sha256_file(
                run_dir / "results" / "combined_frozen_manifest.json"
            ),
            "validation_result_sha256": sha256_file(
                run_dir / "results" / "validation_result.json"
            ),
        },
    }
    json_path = results_dir / "validation_summary.json"
    write_json_atomic(json_path, summary)
    csv_path = results_dir / "validation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "allowance_percent",
            "law_full_action_validation",
            "full_full_action_validation",
            "action_reduction",
            "law_risk_validation",
            "full_risk_validation",
            "risk_ratio",
            "paired_action_ci_low",
            "paired_action_ci_high",
            "strict_success",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in points:
            held = row[held_key]
            writer.writerow(
                {
                    "allowance_percent": row["allowance_percent"],
                    "law_full_action_validation": held["law_full_action"],
                    "full_full_action_validation": held["full_full_action"],
                    "action_reduction": held["action_reduction"],
                    "law_risk_validation": held["law_risk"],
                    "full_risk_validation": held["full_risk"],
                    "risk_ratio": held["risk_ratio"],
                    "paired_action_ci_low": held["paired_action_difference_95_ci"][0],
                    "paired_action_ci_high": held["paired_action_difference_95_ci"][1],
                    "strict_success": row["strict_success"],
                }
            )
    return summary


def clean_domain_axis(axis: plt.Axes) -> None:
    axis.set_xlim(0.0, 2.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def add_sensor_markers(axis: plt.Axes, centers: np.ndarray, width: float) -> None:
    for index, (center, color) in enumerate(zip(centers, SENSOR_COLORS, strict=True)):
        axis.add_patch(
            plt.Circle(
                center,
                width,
                fill=False,
                color=color,
                linewidth=1.0,
                linestyle=(0, (2.2, 2.2)),
                zorder=5,
            )
        )
        axis.scatter(
            center[0], center[1], marker="x", s=28, linewidth=1.5, color=color, zorder=6
        )
        axis.text(
            center[0] + 0.025,
            center[1] + 0.025,
            f"S{index + 1}",
            color=color,
            fontsize=7,
            fontweight="bold",
            zorder=7,
        )


def render_dashboard(
    summary: dict[str, Any], plots_dir: Path, *, dpi: int
) -> list[Path]:
    style()
    points = summary["points"]
    reference_id = str(summary["validation_reference"])
    held_key = f"held_out_{reference_id}"
    allowances = np.asarray([row["allowance_percent"] for row in points])
    law_action = np.asarray([row[held_key]["law_full_action"] for row in points])
    full_action = np.asarray([row[held_key]["full_full_action"] for row in points])
    difference_ci = np.asarray(
        [row[held_key]["paired_action_difference_95_ci"] for row in points]
    )
    action_percent = 100.0 * full_action / law_action
    action_ci = 100.0 * (law_action[:, None] + difference_ci) / law_action[:, None]
    risk_change = 100.0 * np.asarray(
        [row[held_key]["risk_ratio"] - 1.0 for row in points]
    )

    fig = plt.figure(figsize=(12.8, 7.7), constrained_layout=False)
    grid = fig.add_gridspec(
        2, 4, height_ratios=(1.15, 0.85), left=0.07, right=0.98,
        bottom=0.09, top=0.86, hspace=0.38, wspace=0.28,
    )
    action_ax = fig.add_subplot(grid[0, :2])
    risk_ax = fig.add_subplot(grid[0, 2:])
    geometry_axes = [fig.add_subplot(grid[1, index]) for index in range(4)]
    fig.suptitle(
        "Prospective vortices · held-out Pareto validation",
        x=0.07, y=0.96, ha="left", fontsize=18, fontweight="bold", color="#20242B",
    )
    fig.text(
        0.07, 0.905,
        f"One frozen D0 selection reference · one post-freeze {reference_id} reference · 64 paired hidden trials",
        ha="left", fontsize=9, color="#686D74",
    )

    action_ax.axhline(100.0, color="#555A62", linestyle="--", linewidth=1.1, label="Law")
    action_ax.errorbar(
        allowances,
        action_percent,
        yerr=np.vstack((action_percent - action_ci[:, 0], action_ci[:, 1] - action_percent)),
        color="#D84C5B",
        marker="o",
        markersize=6,
        capsize=4,
        linewidth=2.1,
        label=f"Full/FIDE ({reference_id})",
    )
    for index, (x, y, row) in enumerate(
        zip(allowances, action_percent, points, strict=True)
    ):
        horizontal = "left" if index == 0 else ("right" if index == len(points) - 1 else "center")
        x_offset = 5 if index == 0 else (-5 if index == len(points) - 1 else 0)
        action_ax.annotate(
            (
                "same as Law"
                if np.isclose(row[held_key]["action_reduction"], 0.0)
                else f"{100.0 * row[held_key]['action_reduction']:.2f}% lower"
            ),
            (x, y), xytext=(x_offset, -18), textcoords="offset points", ha=horizontal,
            fontsize=8, color="#9D3542",
        )
    action_ax.set_title("A  Full action relative to Law", loc="left", fontweight="bold")
    action_ax.set_xlabel("allowed extra scientific risk")
    action_ax.set_ylabel("held-out Full action (% of Law)")
    action_ax.set_xticks(allowances, [f"{value:g}%" for value in allowances])
    action_ax.set_ylim(min(78.0, float(np.min(action_ci)) - 3.0), 103.0)
    action_ax.grid(axis="y", color="#E8E4DC", linewidth=0.7)
    action_ax.legend(frameon=False, loc="lower left")

    risk_ax.axhline(0.0, color="#555A62", linewidth=1.0)
    risk_ax.plot(
        allowances, allowances, color="#9B9FA5", linestyle="--", linewidth=1.4,
        marker="s", markersize=4, label="allowed increase",
    )
    risk_ax.plot(
        allowances, risk_change, color="#3B82A0", linewidth=2.1,
        marker="o", markersize=6, label=f"observed {reference_id} change",
    )
    for x, y in zip(allowances, risk_change, strict=True):
        risk_ax.annotate(
            f"{y:.2f}%", (x, y), xytext=(0, -17), textcoords="offset points",
            ha="center", fontsize=8, color="#286A84",
        )
    risk_ax.fill_between(allowances, 0.0, allowances, color="#A7C7B7", alpha=0.18)
    risk_ax.set_title("B  Held-out scientific-risk check", loc="left", fontweight="bold")
    risk_ax.set_xlabel("allowed extra scientific risk")
    risk_ax.set_ylabel("risk change versus Law (%)")
    risk_ax.set_xticks(allowances, [f"{value:g}%" for value in allowances])
    risk_ax.set_ylim(min(-3.0, float(np.min(risk_change)) - 0.8), 2.6)
    risk_ax.grid(axis="y", color="#E8E4DC", linewidth=0.7)
    risk_ax.legend(frameon=False, loc="upper left")

    layouts = [("Law", np.asarray(points[0]["law_eta"]).reshape(4, 2))]
    layouts.extend(
        (f"Full {row['allowance_percent']:g}%", np.asarray(row["full_eta"]).reshape(4, 2))
        for row in points
    )
    for axis, (title, centers) in zip(geometry_axes, layouts, strict=True):
        axis.set_facecolor("#F8F5EF")
        axis.axvline(1.0, color="#D7D1C7", linestyle=":", linewidth=0.8)
        add_sensor_markers(axis, centers, 0.12)
        clean_domain_axis(axis)
        axis.set_title(title, fontsize=10, fontweight="bold", pad=5)
    fig.text(
        0.07, 0.405, "C  Frozen sensor geometries", fontsize=10.5,
        fontweight="bold", color="#20242B",
    )

    plots_dir.mkdir(parents=True, exist_ok=True)
    png = plots_dir / "prospective_pareto_validation.png"
    pdf = plots_dir / "prospective_pareto_validation.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return [png, pdf]


def render_toy_style_frontier(
    summary: dict[str, Any], plots_dir: Path, results_dir: Path, *, dpi: int
) -> list[Path]:
    """Render the repaired frontier with the Toy experiment's shared grammar."""
    import sys

    experiments_dir = HERE.parent
    if str(experiments_dir) not in sys.path:
        sys.path.insert(0, str(experiments_dir))
    from percentage_pareto_visualization import make_figure

    reference_id = str(summary["validation_reference"])
    held_key = f"held_out_{reference_id}"
    rows = []
    for point in summary["points"]:
        selection = point["selection_D0"]
        held = point[held_key]
        law_action = float(held["law_full_action"])
        difference_low, difference_high = held["paired_action_difference_95_ci"]
        rows.append(
            {
                "risk_allowance_percent": point["allowance_percent"],
                "R_star": selection["law_risk"],
                "full_R_excess_selection": (
                    float(selection["full_risk"]) - float(selection["law_risk"])
                ),
                "full_A_selection": selection["full_full_action"],
                "law_A_selection": selection["law_full_action"],
                "full_certified": bool(point["risk_pass"]),
                "law_R_validation": held["law_risk"],
                "full_R_validation": held["full_risk"],
                "law_A_validation": held["law_full_action"],
                "full_A_validation": held["full_full_action"],
                "validation_action_reduction": held["action_reduction"],
                "validation_ci_lower": -float(difference_high) / law_action,
                "validation_ci_upper": -float(difference_low) / law_action,
                "strict_success": bool(point["strict_success"]),
            }
        )
    data_path = results_dir / "repaired_pareto_frontier_data.json"
    write_json_atomic(
        data_path,
        {
            "schema_version": 1,
            "experiment": summary["experiment"],
            "selection_reference": summary["selection_reference"],
            "validation_reference": reference_id,
            "tangent_included": False,
            "tangent_reason": (
                "There was not enough experiment time to rerun Tangent after the "
                "Law repair; repaired point manifests use the Law geometry as a "
                "validation-cache placeholder."
            ),
            "rows": rows,
        },
    )
    fig = make_figure(rows, experiment_label="Vortices Prospective · repaired")
    fig.axes[1].set_title(f"B   Independent {reference_id} validation", loc="left")
    for item in fig.texts:
        if item.get_text().startswith("Selection-bank certification is authoritative"):
            item.set_text(
                "D0/E1 are workflow artifact labels for selection/validation seeds. "
                "Tangent is omitted because there was not enough time to rerun it."
            )
    stem = plots_dir / "pareto_frontier_repaired_e1"
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return [png, pdf, data_path]


def sensor_features(points: np.ndarray, centers: np.ndarray, width: float) -> np.ndarray:
    delta = points[..., None, :] - centers
    return np.exp(-0.5 * np.sum(delta * delta, axis=-1) / width**2)


def particle_density(
    points: np.ndarray, weights: np.ndarray, *, nx: int, ny: int
) -> np.ndarray:
    histogram, _, _ = np.histogram2d(
        points[:, 1], points[:, 0], bins=(ny, nx),
        range=((0.0, 1.0), (0.0, 2.0)), weights=weights,
    )
    smoothed = gaussian_filter(histogram, sigma=0.015 / (2.0 / nx), mode="constant")
    mass = float(np.sum(smoothed))
    if mass <= 0.0:
        raise RuntimeError("visual density has zero mass")
    return smoothed / (mass * (2.0 / nx) ** 2)


def double_gyre_velocity(
    xx: np.ndarray, yy: np.ndarray, time: float, truth_cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    physical_time = float(truth_cfg["horizon"]) * time
    a = float(truth_cfg["epsilon"]) * np.sin(
        2.0 * np.pi * physical_time / float(truth_cfg["period"])
    )
    b = 1.0 - 2.0 * a
    f = a * xx**2 + b * xx
    dfdx = 2.0 * a * xx + b
    scale = float(truth_cfg["horizon"]) * np.pi * float(truth_cfg["amplitude"])
    return (
        -scale * np.sin(np.pi * f) * np.cos(np.pi * yy),
        scale * np.cos(np.pi * f) * np.sin(np.pi * yy) * dfdx,
    )


def load_visual_inputs(
    run_dir: Path, cfg: dict[str, Any], reference_id: str
) -> dict[str, Any]:
    with np.load(
        run_dir / "hidden_validation" / "v6_hidden_state_bank.npz", allow_pickle=False
    ) as arrays:
        times = np.asarray(arrays["times"], dtype=np.float64)
        states = np.asarray(arrays["states"], dtype=np.float64)
    with np.load(
        run_dir / "hidden_validation" / "v6_hidden_observation_randomness.npz",
        allow_pickle=False,
    ) as arrays:
        sample_indices = np.asarray(arrays["sample_indices"], dtype=np.int32)
        detector_z = np.asarray(arrays["detector_z"], dtype=np.float64)
    data = TargetProspectiveData.load(
        run_dir / "shared" / "endpoint_reference" / "endpoint_data.npz",
        run_dir / "shared" / "prospective" / "aggregate_predictions.npz",
    )
    evaluator = ProspectiveEvaluator(
        cfg,
        data,
        run_dir / "shared" / "references" / "evaluation" / reference_id
        / "endpoint_reference" / "reference_rollout.npz",
    )
    return {
        "times": times,
        "states": states,
        "sample_indices": sample_indices,
        "detector_z": detector_z,
        "evaluator": evaluator,
        "reference_id": reference_id,
    }


def prepare_projected_law(
    inputs: dict[str, Any], eta: np.ndarray, trial: int
) -> dict[str, Any]:
    evaluator: ProspectiveEvaluator = inputs["evaluator"]
    if trial < 0 or trial >= len(inputs["sample_indices"]):
        raise ValueError(f"trial must lie in [0, {len(inputs['sample_indices']) - 1}]")
    bank, mean, second, _ = _realized_bank_and_moments(
        evaluator,
        eta,
        inputs["states"],
        inputs["sample_indices"][trial : trial + 1],
        inputs["detector_z"][trial : trial + 1],
    )
    one_bank = AggregateObservationBank(
        np.asarray(bank.sampling_z), np.asarray(bank.detector_z)
    )
    targets, _, _ = evaluator.reconstruct(mean, second, one_bank)
    projection, weights, _, _, _ = evaluator._project(
        eta, mean, second, one_bank
    )
    return {
        "times": inputs["times"],
        "truth_particles": inputs["states"],
        "reference_particles": np.asarray(evaluator.nodes),
        "weights": np.asarray(weights[0]),
        "targets": np.asarray(targets[0]),
        "centers": eta.reshape(4, 2),
        "width": float(evaluator.sensors.width),
        "max_projection_residual": float(
            np.max(np.linalg.norm(np.asarray(projection.residual[0]), axis=-1))
        ),
        "minimum_ess_fraction": float(np.min(np.asarray(projection.ess_fraction[0]))),
        "config": evaluator.cfg,
        "reference_id": inputs["reference_id"],
    }


def precompute_fields(data: dict[str, Any], nx: int) -> dict[str, Any]:
    if nx < 128:
        raise ValueError("visualization grid must have at least 128 x cells")
    ny = nx // 2
    uniform = np.full(data["truth_particles"].shape[1], 1.0 / data["truth_particles"].shape[1])
    hidden = [
        particle_density(points, uniform, nx=nx, ny=ny)
        for points in data["truth_particles"]
    ]
    corrected = [
        particle_density(points, weights, nx=nx, ny=ny)
        for points, weights in zip(
            data["reference_particles"], data["weights"], strict=True
        )
    ]
    x = (np.arange(nx) + 0.5) * 2.0 / nx
    y = (np.arange(ny) + 0.5) / ny
    xx, yy = np.meshgrid(x, y, indexing="xy")
    features = sensor_features(
        np.column_stack((xx.ravel(), yy.ravel())), data["centers"], data["width"]
    ).reshape(ny, nx, 4)
    sensor_views = [
        [density * features[:, :, sensor] for sensor in range(4)] for density in hidden
    ]
    density_values = np.concatenate([field.ravel() for field in hidden + corrected])
    sensor_values = np.concatenate(
        [field.ravel() for group in sensor_views for field in group]
    )
    return {
        "hidden": hidden,
        "corrected": corrected,
        "sensor_views": sensor_views,
        "density_norm": PowerNorm(
            gamma=0.52, vmin=0.0, vmax=float(np.quantile(density_values, 0.998))
        ),
        "sensor_norm": PowerNorm(
            gamma=0.45, vmin=0.0, vmax=float(np.quantile(sensor_values, 0.998))
        ),
    }


def add_flow(axis: plt.Axes, time: float, cfg: dict[str, Any]) -> None:
    gx = np.linspace(0.02, 1.98, 31)
    gy = np.linspace(0.02, 0.98, 17)
    xx, yy = np.meshgrid(gx, gy, indexing="xy")
    vx, vy = double_gyre_velocity(xx, yy, time, cfg["truth"])
    axis.streamplot(
        gx, gy, vx, vy, density=0.52, linewidth=0.38, arrowsize=0.45,
        color=(1.0, 1.0, 1.0, 0.38),
    )


def render_static_law(
    data: dict[str, Any], fields: dict[str, Any], allowance: float,
    plots_dir: Path, *, dpi: int,
) -> list[Path]:
    style()
    fig = plt.figure(figsize=(14.8, 8.8), constrained_layout=False)
    outer = fig.add_gridspec(
        3, 4, height_ratios=(1.0, 1.0, 0.95), left=0.10, right=0.96,
        bottom=0.08, top=0.85, wspace=0.06, hspace=0.16,
    )
    fig.suptitle(
        "Prospective four-sensor design in a moving double gyre",
        x=0.10, y=0.96, ha="left", fontsize=18, fontweight="bold", color="#20242B",
    )
    fig.text(
        0.10, 0.90,
        f"Full/FIDE geometry at {100.0 * allowance:g}% allowance · held-out {data['reference_id']} trial 0",
        ha="left", fontsize=9, color="#686D74",
    )
    density_image = None
    for column, time_index in enumerate(TIME_INDICES):
        hidden_ax = fig.add_subplot(outer[0, column])
        corrected_ax = fig.add_subplot(outer[1, column])
        density_image = hidden_ax.imshow(
            fields["hidden"][time_index], origin="lower", extent=(0, 2, 0, 1),
            cmap=FLOW_CMAP, norm=fields["density_norm"], interpolation="bilinear",
        )
        add_flow(hidden_ax, float(data["times"][time_index]), data["config"])
        corrected_ax.imshow(
            fields["corrected"][time_index], origin="lower", extent=(0, 2, 0, 1),
            cmap=FLOW_CMAP, norm=fields["density_norm"], interpolation="bilinear",
        )
        add_sensor_markers(corrected_ax, data["centers"], data["width"])
        clean_domain_axis(hidden_ax)
        clean_domain_axis(corrected_ax)
        hidden_ax.set_title(
            rf"$t={float(data['times'][time_index]):.2f}$", fontsize=11,
            fontweight="bold", pad=5,
        )
        subgrid = outer[2, column].subgridspec(2, 2, wspace=0.05, hspace=0.08)
        for sensor in range(4):
            axis = fig.add_subplot(subgrid[sensor // 2, sensor % 2])
            sensor_map = LinearSegmentedColormap.from_list(
                f"sensor_{sensor}", ("#FFFFFF", SENSOR_COLORS[sensor], "#151A21")
            )
            axis.imshow(
                fields["sensor_views"][time_index][sensor], origin="lower",
                extent=(0, 2, 0, 1), cmap=sensor_map, norm=fields["sensor_norm"],
                interpolation="bilinear",
            )
            clean_domain_axis(axis)
            axis.text(
                0.03, 0.86, f"S{sensor + 1}  y={data['targets'][time_index, sensor]:.3f}",
                transform=axis.transAxes, fontsize=6.3, fontweight="bold",
                color=SENSOR_COLORS[sensor],
            )
    fig.text(0.028, 0.705, "HIDDEN\nPOPULATION", rotation=90, ha="center", va="center",
             fontsize=9, fontweight="bold", color="#4A3A62")
    fig.text(0.028, 0.445, "MEASUREMENT-\nIMPLIED LAW", rotation=90, ha="center", va="center",
             fontsize=9, fontweight="bold", color="#8B3E46")
    fig.text(0.028, 0.185, "SENSOR\nVIEWS", rotation=90, ha="center", va="center",
             fontsize=9, fontweight="bold", color="#436C68")
    if density_image is not None:
        cax = fig.add_axes((0.965, 0.42, 0.012, 0.31))
        colorbar = fig.colorbar(density_image, cax=cax)
        colorbar.set_label("probability density", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
        colorbar.outline.set_visible(False)
    tag = next(tag for value, tag in ALLOWANCES if np.isclose(value, allowance))
    png = plots_dir / f"vortices_prospective_full_{tag}_paper.png"
    pdf = plots_dir / f"vortices_prospective_full_{tag}_paper.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return [png, pdf]


def render_gif(
    data: dict[str, Any], fields: dict[str, Any], plots_dir: Path, *, dpi: int, fps: float
) -> tuple[Path, Path]:
    style()
    fig = plt.figure(figsize=(13.0, 4.6), dpi=dpi, constrained_layout=False)
    grid = fig.add_gridspec(
        4, 3, width_ratios=(1.0, 1.0, 0.29), left=0.035, right=0.925,
        bottom=0.15, top=0.80, wspace=0.06, hspace=0.28,
    )
    hidden_ax = fig.add_subplot(grid[:, 0])
    corrected_ax = fig.add_subplot(grid[:, 1])
    sensor_axes = [fig.add_subplot(grid[index, 2]) for index in range(4)]
    progress_ax = fig.add_axes((0.055, 0.05, 0.85, 0.025))
    colorbar_ax = fig.add_axes((0.947, 0.28, 0.012, 0.40))
    scalar = mpl.cm.ScalarMappable(norm=fields["density_norm"], cmap=FLOW_CMAP)
    colorbar = fig.colorbar(scalar, cax=colorbar_ax)
    colorbar.set_label("probability density", fontsize=8.6)
    colorbar.ax.tick_params(labelsize=7.2, length=2.5)
    colorbar.outline.set_visible(False)
    fig.suptitle(
        "Prospective four-sensor design in a moving double gyre",
        x=0.035, y=0.965, ha="left", fontsize=18, fontweight="bold", color="#20242B",
    )
    fig.text(
        0.035, 0.895,
        f"Full/FIDE geometry at 2% allowance · held-out {data['reference_id']} trial 0",
        ha="left", fontsize=8.5, color="#686D74",
    )
    time_text = fig.text(0.925, 0.845, "", ha="right", fontsize=12, fontweight="bold")
    frames: list[Image.Image] = []
    times = data["times"]
    for index, time in enumerate(times):
        for axis in (hidden_ax, corrected_ax, *sensor_axes):
            axis.clear()
        hidden_ax.imshow(
            fields["hidden"][index], origin="lower", extent=(0, 2, 0, 1),
            cmap=FLOW_CMAP, norm=fields["density_norm"], interpolation="bilinear",
        )
        add_flow(hidden_ax, float(time), data["config"])
        corrected_ax.imshow(
            fields["corrected"][index], origin="lower", extent=(0, 2, 0, 1),
            cmap=FLOW_CMAP, norm=fields["density_norm"], interpolation="bilinear",
        )
        add_sensor_markers(corrected_ax, data["centers"], data["width"])
        hidden_ax.set_title("HIDDEN POPULATION", fontsize=10, fontweight="bold", color="#4A3A62")
        corrected_ax.set_title("MEASUREMENT-IMPLIED LAW", fontsize=10, fontweight="bold", color="#8B3E46")
        for sensor, axis in enumerate(sensor_axes):
            sensor_map = LinearSegmentedColormap.from_list(
                f"gif_sensor_{sensor}", ("#FFFFFF", SENSOR_COLORS[sensor], "#151A21")
            )
            axis.imshow(
                fields["sensor_views"][index][sensor], origin="lower", extent=(0, 2, 0, 1),
                cmap=sensor_map, norm=fields["sensor_norm"], interpolation="bilinear",
            )
            axis.set_title(
                f"S{sensor + 1} SEES  y={data['targets'][index, sensor]:.3f}",
                fontsize=6.5, fontweight="bold", color=SENSOR_COLORS[sensor], pad=1.5,
            )
        for axis in (hidden_ax, corrected_ax, *sensor_axes):
            clean_domain_axis(axis)
        time_text.set_text(rf"$t={float(time):.2f}$")
        progress = index / (len(times) - 1)
        progress_ax.clear()
        progress_ax.set_xlim(0.0, 1.0)
        progress_ax.set_ylim(-1.0, 1.0)
        progress_ax.axis("off")
        progress_ax.plot([0, 1], [0, 0], color="#D1CBC0", linewidth=4, solid_capstyle="round")
        progress_ax.plot([0, progress], [0, 0], color="#596675", linewidth=4, solid_capstyle="round")
        progress_ax.scatter([progress], [0], s=38, color="#D84C5B", edgecolor="white", zorder=3)
        progress_ax.text(0, -0.75, "0", ha="center", va="top", fontsize=7)
        progress_ax.text(1, -0.75, "1", ha="center", va="top", fontsize=7)
        progress_ax.text(0.5, -0.75, "normalized time", ha="center", va="top", fontsize=7)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba[:, :, :3].copy()))
    plt.close(fig)
    gif = plots_dir / "vortices_prospective_full_2p0.gif"
    preview = plots_dir / "vortices_prospective_full_2p0_preview.png"
    duration = [700] + [int(round(1000.0 / fps))] * (len(frames) - 2) + [700]
    frames[0].save(
        gif, save_all=True, append_images=frames[1:], duration=duration,
        loop=0, optimize=False, disposal=2,
    )
    frames[0].save(preview)
    return gif, preview


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    plots_dir = args.plots_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    cfg, combined, validation, reference_id = verify_run(run_dir)
    summary = publish_summary(run_dir, results_dir, combined, validation)
    artifacts = render_dashboard(summary, plots_dir, dpi=args.static_dpi)
    artifacts.extend(
        render_toy_style_frontier(
            summary, plots_dir, results_dir, dpi=args.static_dpi
        )
    )
    visual_inputs = load_visual_inputs(run_dir, cfg, reference_id)
    prepared: dict[float, dict[str, Any]] = {}
    for allowance, _ in ALLOWANCES:
        point = next(
            row for row in summary["points"]
            if np.isclose(row["allowance_percent"], 100.0 * allowance)
        )
        data = prepare_projected_law(
            visual_inputs, np.asarray(point["full_eta"], dtype=np.float64), args.trial
        )
        fields = precompute_fields(data, args.gif_grid_nx)
        prepared[allowance] = {"data": data, "fields": fields}
        artifacts.extend(
            render_static_law(data, fields, allowance, plots_dir, dpi=args.static_dpi)
        )
    gif, preview = render_gif(
        prepared[0.02]["data"], prepared[0.02]["fields"], plots_dir,
        dpi=args.gif_dpi, fps=args.gif_fps,
    )
    artifacts.extend((gif, preview))
    manifest = {
        "schema_version": 1,
        "status": f"COMPLETE_HELD_OUT_{reference_id}_VISUALIZATION_SET",
        "validation_reference": reference_id,
        "data_role": "POST_FREEZE_HELD_OUT_VISUALIZATION_ONLY",
        "trial": args.trial,
        "allowance_percentages": [100.0 * value for value, _ in ALLOWANCES],
        "selection_state_changed": False,
        "renderer": str(Path(__file__).resolve().relative_to(REPO)),
        "renderer_sha256": sha256_file(Path(__file__)),
        "validation_summary_sha256": sha256_file(results_dir / "validation_summary.json"),
        "projection_checks": {
            f"{100.0 * allowance:g}%": {
                "maximum_residual": prepared[allowance]["data"]["max_projection_residual"],
                "minimum_ess_fraction": prepared[allowance]["data"]["minimum_ess_fraction"],
            }
            for allowance, _ in ALLOWANCES
        },
        "artifacts": [
            {
                "path": str(path.relative_to(REPO)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in artifacts
        ],
    }
    write_json_atomic(plots_dir / "visualization_manifest.json", manifest)
    for path in artifacts:
        print(f"saved {path}")
    print(f"saved {plots_dir / 'visualization_manifest.json'}")
    print(f"saved {results_dir / 'validation_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
