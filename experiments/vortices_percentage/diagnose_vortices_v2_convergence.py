#!/usr/bin/env python3
"""Run the preregistered V2 discretization-development convergence study.

Historical V1 confirmatory trials are used only as mechanism/development cases.
All outputs are written beneath the separate V2 development directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np

from core import (
    HERE,
    V2_VERSION,
    config_fingerprint,
    continuity_check,
    diagonal_condition_estimate,
    edge_energy_density,
    frozen_reference_scott_bandwidth,
    hard_fiber_particle_state,
    independent_poisson,
    load_development_context,
    make_grid,
    rasterize_trajectory_v2,
    sha256_file,
    solve_v2,
    top_fraction_share,
    weighted_gradient_relative_error,
)
from mfsi.decomposition import raster_tangent_projection
from mfsi.poisson import weighted_laplacian
from mfsi.raster import (
    rasterize_projected_particles_reflected_rect,
    reflected_flux_divergence_rect,
    reflected_particle_flux_rect,
)


DEFAULT_PARETO = HERE / "inputs" / "development_pareto"
DEFAULT_BANK = (
    DEFAULT_PARETO / "confirmatory_validation_2048" / "bank_19892"
    / "fresh_validation_bank.npz"
)
DEFAULT_OUTPUT = HERE / "outputs" / "vortices_v2_reflection_prefreeze"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pareto-dir", type=Path, default=DEFAULT_PARETO)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--bandwidth", type=float, default=None)
    parser.add_argument("--include-512", action="store_true")
    parser.add_argument(
        "--case-limit", type=int, default=None,
        help="Development/debug convenience only; omit for the declared case set.",
    )
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def git_state() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
    )
    return {"head": head, "working_tree_dirty": dirty}


def load_geometries(bank_root: Path) -> dict[str, np.ndarray]:
    summary = json.loads((bank_root / "summary.json").read_text(encoding="utf-8"))
    return {
        row["key"]: np.asarray(row["geometry"], dtype=np.float64)
        for row in summary["manifest"]["geometries"]
    }


def attach_legacy_receipts(cases: list[dict[str, Any]], bank_root: Path) -> None:
    """Attach immutable V1 values for causal attribution, never V2 validation."""
    cache: dict[str, dict[str, Any]] = {}
    for case in cases:
        key = case["geometry_key"]
        if key not in cache:
            raw = json.loads(
                (bank_root / "raw" / f"geometry_{key}.json").read_text(encoding="utf-8")
            )
            cache[key] = {str(row["trial"]): row for row in raw["rows"]}
        receipt = cache[key][str(case["trial"])]
        case["legacy_v1_integrated_action"] = float(receipt["full_action"])
        case["legacy_v1_action_at_t_0p5"] = float(receipt["full_action_by_time"][10])


def development_cases(
    geometries: dict[str, np.ndarray], trial_count: int, seed: int, ordinary_count: int
) -> list[dict[str, Any]]:
    # These tail cases were declared from the immutable V1 report.  The ordinary
    # cases are selected only by a fixed RNG seed, before any V2 action is read.
    cases = [
        {"name": "golden_full_4pct", "geometry_key": "ea6c90af64ce4356", "trial": 130,
         "role": "golden_known_v1_tail"},
        {"name": "golden_law", "geometry_key": "f8fdd998b4627969", "trial": 130,
         "role": "same_observation_law_control"},
        {"name": "golden_full_2pct", "geometry_key": "ce783572fe3170da", "trial": 130,
         "role": "same_observation_full_control"},
        {"name": "known_law_tail", "geometry_key": "f8fdd998b4627969", "trial": 65,
         "role": "known_v1_tail"},
        {"name": "known_full_1pct_tail", "geometry_key": "41ca33ec45daa976", "trial": 240,
         "role": "known_v1_tail"},
    ]
    excluded = {case["trial"] for case in cases}
    eligible = np.asarray([trial for trial in range(trial_count) if trial not in excluded])
    rng = np.random.default_rng(int(seed))
    ordinary = rng.choice(eligible, size=int(ordinary_count), replace=False)
    for index, trial in enumerate(ordinary):
        cases.append({
            "name": f"ordinary_law_{index + 1}",
            "geometry_key": "f8fdd998b4627969",
            "trial": int(trial),
            "role": "deterministic_action_blind_ordinary_sample",
        })
    for case in cases:
        case["geometry"] = geometries[case["geometry_key"]]
    return cases


def evaluate_grid(
    state,
    grid,
    bandwidth: float,
    time_weights: np.ndarray,
    *,
    image_pairs: int,
):
    raster = rasterize_trajectory_v2(
        state, grid, bandwidth=bandwidth, image_pairs=image_pairs
    )
    solved = solve_v2(raster["q"], raster["source"], grid)
    action_by_time = np.asarray(solved.action, dtype=np.float64)
    potential = np.asarray(solved.potential, dtype=np.float64)
    top_shares = []
    condition = []
    energy_identity = []
    for t_index in range(len(action_by_time)):
        density, energy = edge_energy_density(raster["q"][t_index], potential[t_index], grid)
        top_shares.append(top_fraction_share(density))
        condition.append(diagonal_condition_estimate(raster["q"][t_index], grid))
        energy_identity.append(
            abs(energy - action_by_time[t_index]) / max(abs(action_by_time[t_index]), 1e-14)
        )
    q_flat = raster["q"].ravel()
    return {
        "action": float(np.sum(time_weights * action_by_time)),
        "action_by_time": action_by_time,
        "max_action_by_time": float(np.max(action_by_time)),
        "action_at_t_0p5": float(action_by_time[10]),
        "max_poisson_relative_residual": float(np.max(solved.relative_residual)),
        "max_component_compatibility_residual": float(
            np.max(solved.maximum_component_compatibility_residual)
        ),
        "min_q": float(np.min(q_flat)),
        "q_quantiles": np.quantile(q_flat, [0.0, 0.001, 0.01, 0.1, 0.5]).tolist(),
        "condition_estimate": float(np.max(condition)),
        "condition_estimate_definition": "max/min positive finite-volume diagonal",
        "max_top_1pct_energy_share": float(np.max(top_shares)),
        "max_component_count": int(np.max(solved.component_count)),
        "all_solver_converged": bool(np.all(solved.solver_converged)),
        "all_component_compatible": bool(np.all(solved.compatible)),
        "max_energy_identity_relative_error": float(np.max(energy_identity)),
        "max_mass_error": float(
            np.max(np.abs(np.sum(raster["mass"], axis=(-2, -1)) - 1.0))
        ),
        "max_source_integral_absolute": float(
            np.max(np.abs(np.sum(raster["source"], axis=(-2, -1)) * grid.cell_area))
        ),
        "strictly_positive_q": bool(np.all(raster["q"] > 0.0)),
        "raster": raster,
        "potential": potential,
    }


def add_relative_changes(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, float, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (row["case"], row["bandwidth_multiplier"], row["particle_count"]), []
        ).append(row)
    for group in grouped.values():
        group.sort(key=lambda row: row["grid_nx"])
        for index, row in enumerate(group):
            row["relative_change_from_previous_grid"] = (
                float("nan") if index == 0 else
                abs(row["action"] - group[index - 1]["action"])
                / max(abs(row["action"]), 1e-14)
            )


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(jsonable(value))
                if isinstance(value, (list, dict, np.ndarray))
                else value
                for key, value in row.items()
            })


def plot_action_grid(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for case in sorted({row["case"] for row in rows}):
        chosen = [row for row in rows if row["case"] == case and row["bandwidth_multiplier"] == 1.0]
        chosen.sort(key=lambda row: row["grid_nx"])
        ax.plot([row["grid_nx"] for row in chosen], [row["action"] for row in chosen], "o-", label=case)
    ax.set_yscale("log")
    ax.set_xlabel("grid cells in x (ny = nx/2)")
    ax.set_ylabel("integrated V2 Full action")
    ax.set_title("Fixed-physical-bandwidth PDE-grid convergence")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_particle(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for case in sorted({row["case"] for row in rows}):
        chosen = sorted((row for row in rows if row["case"] == case), key=lambda row: row["particle_count"])
        ax.plot([row["particle_count"] for row in chosen], [row["action"] for row in chosen], "o-", label=case)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("recalibrated empirical reference particles")
    ax.set_ylabel("integrated V2 Full action")
    ax.set_title("Hard-fiber empirical-law convergence")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_bandwidth(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    fine = max(row["grid_nx"] for row in rows)
    for case in sorted({row["case"] for row in rows}):
        chosen = sorted(
            (row for row in rows if row["case"] == case and row["grid_nx"] == fine),
            key=lambda row: row["bandwidth_multiplier"],
        )
        ax.plot([row["bandwidth_multiplier"] for row in chosen], [row["action"] for row in chosen], "o-", label=case)
    ax.set_yscale("log")
    ax.set_xlabel("predeclared bandwidth multiplier")
    ax.set_ylabel(f"integrated action on {fine}x{fine // 2}")
    ax.set_title("Reference-derived physical-bandwidth sensitivity")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_golden_fields(q, source, potential, energy, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    entries = [
        (np.log10(np.maximum(q, np.finfo(np.float64).tiny)), r"$\log_{10}q_h$", "viridis"),
        (source, r"positive defect $s_h$", "coolwarm"),
        (potential, r"$\psi_h$ from $K\psi=-s$", "coolwarm"),
        (np.log10(np.maximum(energy, np.finfo(np.float64).tiny)), r"$\log_{10}$ energy density", "magma"),
    ]
    for ax, (field, title, cmap) in zip(axes.ravel(), entries):
        image = ax.imshow(field, origin="lower", extent=[0, 2, 0, 1], aspect="auto", cmap=cmap)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.82)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_field(field, path: Path, *, title: str, cmap: str, log: bool = False) -> None:
    values = np.asarray(field, dtype=np.float64)
    if log:
        values = np.log10(np.maximum(values, np.finfo(np.float64).tiny))
    fig, ax = plt.subplots(figsize=(7.5, 3.8), constrained_layout=True)
    image = ax.imshow(values, origin="lower", extent=[0, 2, 0, 1], aspect="auto", cmap=cmap)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _continuity_tests(grid) -> np.ndarray:
    points = np.asarray(grid.points(), dtype=np.float64)
    tests = [np.ones(grid.shape)]
    for center_x, center_y, scale in (
        (0.25, 0.20, 0.035),
        (0.25, 0.80, 0.035),
        (1.75, 0.20, 0.035),
        (1.75, 0.80, 0.035),
        (0.50, 0.50, 0.08),
        (1.50, 0.50, 0.08),
    ):
        tests.append(
            np.exp(
                -(
                    (points[..., 0] - center_x) ** 2
                    + (points[..., 1] - center_y) ** 2
                )
                / scale
            )
        )
    return np.asarray(tests)


def _continuity_metrics(residual, source, grid) -> dict[str, Any]:
    residual = np.asarray(residual, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    tests = _continuity_tests(grid)
    cell_area = float(grid.cell_area)
    source_l2 = float(np.sqrt(np.sum(source**2) * cell_area))
    weak = np.sum(tests * residual[None], axis=(-2, -1)) * cell_area
    weak_scale = np.maximum(
        np.sum(np.abs(tests * source[None]), axis=(-2, -1)) * cell_area,
        1.0e-14,
    )
    return {
        "relative_l2_error": float(
            np.sqrt(np.sum(residual**2) * cell_area) / max(source_l2, 1.0e-14)
        ),
        "l1_error": float(np.sum(np.abs(residual)) * cell_area),
        "maximum_absolute_error": float(np.max(np.abs(residual))),
        "weak_errors": weak.tolist(),
        "weak_relative_errors": (np.abs(weak) / weak_scale).tolist(),
        "maximum_weak_relative_error": float(np.max(np.abs(weak) / weak_scale)),
    }


def _tangent_velocity(points: np.ndarray) -> np.ndarray:
    x_coordinate, y_coordinate = points[:, 0], points[:, 1]
    return 0.08 * np.column_stack(
        [
            np.sin(0.5 * np.pi * x_coordinate) * np.cos(np.pi * y_coordinate),
            -0.5
            * np.cos(0.5 * np.pi * x_coordinate)
            * np.sin(np.pi * y_coordinate),
        ]
    )


def _rk4_positions(points: np.ndarray, step: float) -> np.ndarray:
    k1 = _tangent_velocity(points)
    k2 = _tangent_velocity(points + 0.5 * step * k1)
    k3 = _tangent_velocity(points + 0.5 * step * k2)
    k4 = _tangent_velocity(points + step * k3)
    return points + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def reflected_manufactured_rows(
    grids: list[tuple[int, int]],
    *,
    bandwidth: float,
    image_pairs: int,
    epsilon: float = 2.0e-5,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(20260830)
    interior = np.column_stack(
        [rng.uniform(0.35, 1.65, 600), rng.uniform(0.22, 0.78, 600)]
    )
    boundary_count = 160
    near_boundary = np.concatenate(
        [
            np.column_stack(
                [np.full(boundary_count, 0.012), rng.uniform(0.02, 0.98, boundary_count)]
            ),
            np.column_stack(
                [np.full(boundary_count, 1.988), rng.uniform(0.02, 0.98, boundary_count)]
            ),
            np.column_stack(
                [rng.uniform(0.02, 1.98, boundary_count), np.full(boundary_count, 0.012)]
            ),
            np.column_stack(
                [rng.uniform(0.02, 1.98, boundary_count), np.full(boundary_count, 0.988)]
            ),
        ]
    )
    tangent = np.column_stack(
        [rng.uniform(0.002, 1.998, 800), rng.uniform(0.002, 0.998, 800)]
    )
    systems = (
        (
            "constant_velocity_interior",
            interior,
            np.tile(np.asarray([0.055, -0.025]), (len(interior), 1)),
            False,
        ),
        (
            "particles_near_each_boundary",
            near_boundary,
            np.concatenate(
                [
                    np.tile([0.045, 0.0], (boundary_count, 1)),
                    np.tile([-0.045, 0.0], (boundary_count, 1)),
                    np.tile([0.0, 0.035], (boundary_count, 1)),
                    np.tile([0.0, -0.035], (boundary_count, 1)),
                ]
            ),
            False,
        ),
        ("tangent_to_boundary_field", tangent, _tangent_velocity(tangent), True),
    )
    rows = []
    for name, nodes, velocity, use_rk4 in systems:
        weights = rng.uniform(0.5, 1.5, len(nodes))
        weights /= np.sum(weights)
        forcing = (
            0.3 * np.sin(np.pi * nodes[:, 0])
            + 0.2 * np.cos(2.0 * np.pi * nodes[:, 1])
        )
        forcing -= weights @ forcing
        plus_weights = weights * np.exp(epsilon * forcing)
        minus_weights = weights * np.exp(-epsilon * forcing)
        plus_weights /= np.sum(plus_weights)
        minus_weights /= np.sum(minus_weights)
        plus_nodes = (
            _rk4_positions(nodes, epsilon)
            if use_rk4
            else nodes + epsilon * velocity
        )
        minus_nodes = (
            _rk4_positions(nodes, -epsilon)
            if use_rk4
            else nodes - epsilon * velocity
        )
        for nx, ny in grids:
            grid = make_grid(nx, ny)
            minus = rasterize_projected_particles_reflected_rect(
                minus_nodes,
                minus_weights,
                np.zeros_like(minus_weights),
                grid,
                bandwidth=bandwidth,
                image_pairs=image_pairs,
            )
            plus = rasterize_projected_particles_reflected_rect(
                plus_nodes,
                plus_weights,
                np.zeros_like(plus_weights),
                grid,
                bandwidth=bandwidth,
                image_pairs=image_pairs,
            )
            center = rasterize_projected_particles_reflected_rect(
                nodes,
                weights,
                forcing,
                grid,
                bandwidth=bandwidth,
                image_pairs=image_pairs,
            )
            flux_x, flux_y = reflected_particle_flux_rect(
                nodes,
                weights,
                velocity,
                grid,
                bandwidth=bandwidth,
                image_pairs=image_pairs,
            )
            residual = (
                (np.asarray(plus.q) - np.asarray(minus.q)) / (2.0 * epsilon)
                + np.asarray(reflected_flux_divergence_rect(flux_x, flux_y, grid))
                - np.asarray(center.source)
            )
            rows.append(
                {
                    "case": name,
                    "grid_nx": nx,
                    "grid_ny": ny,
                    "epsilon": epsilon,
                    **_continuity_metrics(residual, center.source, grid),
                    "mass_error": float(abs(np.sum(center.mass) - 1.0)),
                    "source_integral_absolute": float(
                        abs(np.sum(center.source) * grid.cell_area)
                    ),
                    "minimum_q": float(np.min(center.q)),
                    "maximum_absolute_normal_boundary_flux": float(
                        max(
                            np.max(np.abs(np.asarray(flux_x)[:, [0, -1]])),
                            np.max(np.abs(np.asarray(flux_y)[[0, -1], :])),
                        )
                    ),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    context = load_development_context(
        args.pareto_dir.expanduser().resolve(), args.bank.expanduser().resolve(), namespace=19892
    )
    computed_bandwidth, bandwidth_by_time = frozen_reference_scott_bandwidth(
        context.exp.reference_nodes, context.exp.reference_weights
    )
    declared = float(config["raster"]["physical_bandwidth"])
    if args.bandwidth is None and abs(computed_bandwidth - declared) > 5e-15:
        raise RuntimeError("config physical bandwidth does not match frozen reference rule")
    reference_bandwidth = computed_bandwidth if args.bandwidth is None else float(args.bandwidth)
    image_pairs = int(config["raster"]["reflected_image_pairs"])
    multipliers = [float(value) for value in config["raster"]["bandwidth_multipliers"]]
    grids = [tuple(value) for value in config["development"]["grid_sizes"]]
    if args.include_512:
        grids.append((512, 256))
    geometries = load_geometries(args.bank.expanduser().resolve().parent)
    cases = development_cases(
        geometries,
        int(context.bank.sample_indices.shape[0]),
        int(config["development"]["ordinary_trial_seed"]),
        int(config["development"]["ordinary_trial_count"]),
    )
    attach_legacy_receipts(cases, args.bank.expanduser().resolve().parent)
    if args.case_limit is not None:
        cases = cases[: int(args.case_limit)]
    print(
        f"[v2] reflection bandwidth={reference_bandwidth:.15g} "
        f"image_pairs={image_pairs} cases={len(cases)} grids={grids}",
        flush=True,
    )
    time_weights = np.asarray(context.exp.time_w, dtype=np.float64)
    state_cache = {}
    grid_rows: list[dict[str, Any]] = []
    retained = {}
    for case in cases:
        print(f"[v2] projecting {case['name']} trial={case['trial']}", flush=True)
        state = hard_fiber_particle_state(context, case["geometry"], case["trial"])
        state_cache[case["name"]] = state
        for multiplier in multipliers:
            bandwidth = reference_bandwidth * multiplier
            for nx, ny in grids:
                print(f"[v2] grid {case['name']} h={multiplier:g} G={nx}x{ny}", flush=True)
                grid = make_grid(nx, ny)
                result = evaluate_grid(
                    state,
                    grid,
                    bandwidth,
                    time_weights,
                    image_pairs=image_pairs,
                )
                row = {
                    "study": "pde_grid",
                    "case": case["name"],
                    "role": case["role"],
                    "namespace": 19892,
                    "trial": case["trial"],
                    "geometry_key": case["geometry_key"],
                    "projection_type": "hard",
                    "particle_count": state.particle_count,
                    "grid_nx": nx,
                    "grid_ny": ny,
                    "physical_bandwidth": bandwidth,
                    "bandwidth_multiplier": multiplier,
                    "reflected_image_pairs": image_pairs,
                    "action": result["action"],
                    "action_by_time": result["action_by_time"],
                    "max_action_by_time": result["max_action_by_time"],
                    "action_at_t_0p5": result["action_at_t_0p5"],
                    "legacy_v1_integrated_action": case["legacy_v1_integrated_action"],
                    "legacy_v1_action_at_t_0p5": case["legacy_v1_action_at_t_0p5"],
                    "max_poisson_relative_residual": result["max_poisson_relative_residual"],
                    "max_component_compatibility_residual": result["max_component_compatibility_residual"],
                    "min_q": result["min_q"],
                    "q_quantiles": result["q_quantiles"],
                    "condition_estimate": result["condition_estimate"],
                    "condition_estimate_definition": result["condition_estimate_definition"],
                    "max_top_1pct_energy_share": result["max_top_1pct_energy_share"],
                    "max_component_count": result["max_component_count"],
                    "all_solver_converged": result["all_solver_converged"],
                    "all_component_compatible": result["all_component_compatible"],
                    "max_energy_identity_relative_error": result["max_energy_identity_relative_error"],
                    "max_mass_error": result["max_mass_error"],
                    "max_source_integral_absolute": result["max_source_integral_absolute"],
                    "strictly_positive_q": result["strictly_positive_q"],
                    "max_calibration_residual": float(np.max(state.calibration_residual)),
                    "min_ess_fraction": float(np.min(state.ess_fraction)),
                    "min_covariance_eigenvalue": float(np.min(state.covariance_min_eigenvalue)),
                    "max_lambda_dot_norm": float(np.max(np.linalg.norm(state.lambda_dot, axis=-1))),
                    "lambda_dot_norm_at_t_0p5": float(np.linalg.norm(state.lambda_dot[10])),
                }
                grid_rows.append(row)
                retained[(case["name"], multiplier, nx)] = result
    add_relative_changes(grid_rows)

    fine_nx, fine_ny = grids[-1]
    particle_rows: list[dict[str, Any]] = []
    for case in cases[:3]:
        previous = None
        for count in config["development"]["particle_counts"]:
            print(f"[v2] particles {case['name']} N={count}", flush=True)
            state = hard_fiber_particle_state(
                context, case["geometry"], case["trial"], particle_count=int(count)
            )
            result = evaluate_grid(
                state,
                make_grid(fine_nx, fine_ny),
                reference_bandwidth,
                time_weights,
                image_pairs=image_pairs,
            )
            action = result["action"]
            particle_rows.append({
                "study": "empirical_particle",
                "case": case["name"],
                "role": case["role"],
                "namespace": 19892,
                "trial": case["trial"],
                "geometry_key": case["geometry_key"],
                "projection_type": "hard_recalibrated_at_each_N",
                "particle_count": int(count),
                "grid_nx": fine_nx,
                "grid_ny": fine_ny,
                "physical_bandwidth": reference_bandwidth,
                "bandwidth_multiplier": 1.0,
                "reflected_image_pairs": image_pairs,
                "action": action,
                "action_by_time": result["action_by_time"],
                "max_action_by_time": result["max_action_by_time"],
                "action_at_t_0p5": result["action_at_t_0p5"],
                "relative_change_from_previous_particles": (
                    float("nan") if previous is None else
                    abs(action - previous) / max(abs(action), 1e-14)
                ),
                "max_poisson_relative_residual": result["max_poisson_relative_residual"],
                "max_component_compatibility_residual": result["max_component_compatibility_residual"],
                "min_q": result["min_q"],
                "condition_estimate": result["condition_estimate"],
                "max_top_1pct_energy_share": result["max_top_1pct_energy_share"],
                "max_component_count": result["max_component_count"],
                "all_solver_converged": result["all_solver_converged"],
                "all_component_compatible": result["all_component_compatible"],
                "max_calibration_residual": float(np.max(state.calibration_residual)),
                "min_ess_fraction": float(np.min(state.ess_fraction)),
                "min_covariance_eigenvalue": float(np.min(state.covariance_min_eigenvalue)),
                "max_lambda_dot_norm": float(np.max(np.linalg.norm(state.lambda_dot, axis=-1))),
                "lambda_dot_norm_at_t_0p5": float(np.linalg.norm(state.lambda_dot[10])),
            })
            previous = action

    golden = cases[0]
    golden_state = state_cache[golden["name"]]
    continuity_rows = []
    for nx, ny in grids:
        for epsilon in config["development"]["continuity_epsilons"]:
            print(f"[v2] continuity G={nx}x{ny} eps={epsilon}", flush=True)
            continuity_rows.append(
                continuity_check(
                    context,
                    golden["geometry"],
                    golden["trial"],
                    grid=make_grid(nx, ny),
                    bandwidth=reference_bandwidth,
                    epsilon=float(epsilon),
                    state=golden_state,
                    image_pairs=image_pairs,
                )
            )

    print("[v2] reflected manufactured continuity", flush=True)
    manufactured_rows = reflected_manufactured_rows(
        grids,
        bandwidth=reference_bandwidth,
        image_pairs=image_pairs,
    )

    solver_rows = []
    solver_targets = (
        ("golden_full_4pct", fine_nx),
        ("golden_law", 128),
        ("golden_law", fine_nx),
        ("golden_full_2pct", 128),
        ("golden_full_2pct", fine_nx),
    )
    for name, nx in solver_targets:
        if (name, 1.0, nx) not in retained:
            continue
        result = retained[(name, 1.0, nx)]
        grid = make_grid(nx, nx // 2)
        t_index = 10
        independent = independent_poisson(
            result["raster"]["q"][t_index], result["raster"]["source"][t_index], grid
        )
        production_action = float(result["action_by_time"][t_index])
        sign_flip = solve_v2(
            result["raster"]["q"][t_index], -result["raster"]["source"][t_index], grid
        )
        _, edge_energy = edge_energy_density(
            result["raster"]["q"][t_index], result["potential"][t_index], grid
        )
        bilinear_energy = float(
            grid.cell_area
            * np.sum(
                result["potential"][t_index]
                * np.asarray(
                    weighted_laplacian(
                        jnp.asarray(result["potential"][t_index]),
                        jnp.asarray(result["raster"]["q"][t_index]),
                        grid.dx,
                    )
                )
            )
        )
        solver_rows.append({
            "case": name,
            "time": float(context.times[t_index]),
            "grid_nx": nx,
            "grid_ny": nx // 2,
            "production_action": production_action,
            "independent_action": independent["action"],
            "relative_action_error": abs(production_action - independent["action"]) / max(abs(production_action), 1e-14),
            "weighted_gradient_relative_error": weighted_gradient_relative_error(
                result["raster"]["q"][t_index],
                result["potential"][t_index],
                independent["potential"],
                grid,
            ),
            "production_residual": float(result["max_poisson_relative_residual"]),
            "independent_residual": independent["relative_residual"],
            "bilinear_energy": bilinear_energy,
            "edge_energy": edge_energy,
            "bilinear_edge_relative_error": abs(bilinear_energy - edge_energy)
            / max(abs(bilinear_energy), 1.0e-14),
            "sign_flip_action_relative_error": abs(
                production_action - float(sign_flip.action[0])
            ) / max(abs(production_action), 1e-14),
            "sign_flip_potential_max_error": float(
                np.max(np.abs(result["potential"][t_index] + sign_flip.potential[0]))
            ),
        })

    decomposition_rows = []
    for case in cases:
        name = case["name"]
        result = retained[(name, 1.0, fine_nx)]
        state = state_cache[name]
        grid = make_grid(fine_nx, fine_ny)
        features = context.exp.family.features(
            grid.points(), jnp.asarray(state.eta, dtype=jnp.float64)
        )
        # raster_tangent_projection uses the historical convention
        # L(delta)=-r(source). V2 solves K psi=-s, so supplying -s makes both
        # Full and Tangent fields satisfy the V2 target L(delta)=+r(s).
        decomposition = raster_tangent_projection(
            jnp.asarray(result["potential"], dtype=jnp.float64),
            jnp.asarray(result["raster"]["q"], dtype=jnp.float64),
            -jnp.asarray(result["raster"]["source"], dtype=jnp.float64),
            features,
            dx=float(grid.dx),
            cell_area=float(grid.cell_area),
            pinv_rcond=1.0e-10,
            operator_floor_rel=0.0,
            gauge_strength=0.0,
            source_is_density=True,
        )
        full_energy = np.asarray(decomposition.full_energy, dtype=np.float64)
        tangent_energy = np.asarray(decomposition.tangent_energy, dtype=np.float64)
        hidden_energy = np.asarray(decomposition.hidden_energy, dtype=np.float64)
        full_residual = np.asarray(
            decomposition.full_moment_residual, dtype=np.float64
        )
        tangent_residual = np.asarray(
            decomposition.tangent_moment_residual, dtype=np.float64
        )
        hidden_residual = np.asarray(
            decomposition.hidden_moment_residual, dtype=np.float64
        )
        cross = np.asarray(
            decomposition.tangent_hidden_inner_product, dtype=np.float64
        )
        pythagorean = np.asarray(
            decomposition.pythagorean_residual, dtype=np.float64
        )
        hierarchy = np.asarray(
            decomposition.hierarchy_raw_violation, dtype=np.float64
        )
        decomposition_rows.append(
            {
                "case": name,
                "grid_nx": fine_nx,
                "grid_ny": fine_ny,
                "physical_bandwidth": reference_bandwidth,
                "reflected_image_pairs": image_pairs,
                "full_action": float(np.sum(time_weights * full_energy)),
                "tangent_action": float(np.sum(time_weights * tangent_energy)),
                "hidden_action": float(np.sum(time_weights * hidden_energy)),
                "full_action_match_relative": float(
                    np.max(
                        np.abs(full_energy - result["action_by_time"])
                        / np.maximum(np.abs(result["action_by_time"]), 1.0e-14)
                    )
                ),
                "maximum_full_moment_rate_residual": float(
                    np.max(np.linalg.norm(full_residual, axis=-1))
                ),
                "maximum_tangent_moment_rate_residual": float(
                    np.max(np.linalg.norm(tangent_residual, axis=-1))
                ),
                "maximum_hidden_nullspace_residual": float(
                    np.max(np.linalg.norm(hidden_residual, axis=-1))
                ),
                "maximum_absolute_tangent_hidden_inner_product": float(
                    np.max(np.abs(cross))
                ),
                "maximum_absolute_pythagorean_residual": float(
                    np.max(np.abs(pythagorean))
                ),
                "maximum_raw_hierarchy_violation": float(np.max(hierarchy)),
                "minimum_gram_rank": int(np.min(np.asarray(decomposition.gram_rank))),
                "full_energy_by_time": full_energy,
                "tangent_energy_by_time": tangent_energy,
                "hidden_energy_by_time": hidden_energy,
            }
        )

    golden_result = retained[("golden_full_4pct", 1.0, fine_nx)]
    golden_grid = make_grid(fine_nx, fine_ny)
    golden_energy, _ = edge_energy_density(
        golden_result["raster"]["q"][10], golden_result["potential"][10], golden_grid
    )
    np.savez_compressed(
        output / "golden_v2_fields.npz",
        q=golden_result["raster"]["q"][10],
        source=golden_result["raster"]["source"][10],
        potential=golden_result["potential"][10],
        energy_density=golden_energy,
        bandwidth=np.asarray(reference_bandwidth),
        grid=np.asarray([fine_nx, fine_ny]),
    )
    plot_action_grid(grid_rows, output / "action_vs_grid_resolution.png")
    plot_particle(particle_rows, output / "action_vs_particle_count.png")
    plot_bandwidth(grid_rows, output / "bandwidth_sensitivity.png")
    plot_golden_fields(
        golden_result["raster"]["q"][10],
        golden_result["raster"]["source"][10],
        golden_result["potential"][10],
        golden_energy,
        output / "golden_q_source_energy.png",
    )
    plot_field(
        golden_result["raster"]["q"][10], output / "golden_q.png",
        title="Golden V2 physical density at t=0.5", cmap="viridis", log=True,
    )
    plot_field(
        golden_result["raster"]["source"][10], output / "golden_source.png",
        title="Golden V2 positive continuity defect at t=0.5", cmap="coolwarm",
    )
    plot_field(
        golden_energy, output / "golden_energy_density.png",
        title="Golden V2 weighted correction energy density at t=0.5", cmap="magma", log=True,
    )

    gates = config["numerical_gates"]
    default_rows = [row for row in grid_rows if row["bandwidth_multiplier"] == 1.0]
    coarse_to_medium = {
        row["case"]: row["relative_change_from_previous_grid"]
        for row in default_rows
        if row["grid_nx"] == 128
    }
    medium_to_fine = [
        row["relative_change_from_previous_grid"]
        for row in default_rows if row["grid_nx"] == fine_nx
    ]
    grid_change_shrinks = all(
        row["relative_change_from_previous_grid"] < coarse_to_medium[row["case"]]
        for row in default_rows
        if row["grid_nx"] == fine_nx
    )
    particle_final = [
        row["relative_change_from_previous_particles"]
        for row in particle_rows if row["particle_count"] == max(config["development"]["particle_counts"])
    ]
    grid_pass = bool(
        default_rows
        and all(row["all_solver_converged"] and row["all_component_compatible"] for row in default_rows)
        and max(row["max_poisson_relative_residual"] for row in default_rows) <= float(gates["poisson_relative_residual"])
        and max(row["max_mass_error"] for row in default_rows)
        <= float(gates["mass_absolute"])
        and max(row["max_source_integral_absolute"] for row in default_rows)
        <= float(gates["source_compatibility_absolute"])
        and all(row["strictly_positive_q"] for row in default_rows)
        and max(row["max_component_count"] for row in default_rows) == 1
        and max(medium_to_fine) <= float(gates["medium_to_fine_action_relative"])
        and grid_change_shrinks
    )
    particle_pass = bool(
        particle_final
        and max(particle_final) <= float(gates["particle_16384_to_32768_action_relative"])
    )
    solver_pass = bool(
        solver_rows
        and max(row["relative_action_error"] for row in solver_rows)
        <= float(gates["independent_action_relative"])
        and max(row["production_residual"] for row in solver_rows)
        <= float(gates["poisson_relative_residual"])
        and max(row["independent_residual"] for row in solver_rows)
        <= float(gates["poisson_relative_residual"])
        and max(row["bilinear_edge_relative_error"] for row in solver_rows)
        <= float(gates["independent_action_relative"])
    )
    smallest_epsilon = min(config["development"]["continuity_epsilons"])
    continuity_fine_epsilon = sorted(
        (row for row in continuity_rows if row["epsilon"] == smallest_epsilon),
        key=lambda row: row["grid_nx"],
    )
    continuity_grid_range = (
        max(row["relative_l2_error"] for row in continuity_fine_epsilon)
        - min(row["relative_l2_error"] for row in continuity_fine_epsilon)
    )
    continuity_epsilon_monotone = True
    for nx, _ in grids:
        ordered = sorted(
            (row for row in continuity_rows if row["grid_nx"] == nx),
            key=lambda row: row["epsilon"], reverse=True,
        )
        continuity_epsilon_monotone &= all(
            ordered[index]["relative_l2_error"] <= ordered[index - 1]["relative_l2_error"]
            for index in range(1, len(ordered))
        )
    continuity_pass = bool(
        continuity_epsilon_monotone
        and max(row["relative_l2_error"] for row in continuity_fine_epsilon)
        <= float(gates["golden_continuity_relative_l2"])
        and max(row["maximum_weak_relative_error"] for row in continuity_fine_epsilon)
        <= float(gates["golden_continuity_max_weak_relative"])
        and continuity_grid_range <= float(gates["golden_continuity_grid_range"])
        and min(row["correlation"] for row in continuity_fine_epsilon)
        >= float(gates["golden_continuity_correlation"])
        and max(
            row["particle_moment_identity_absolute"] for row in continuity_rows
        )
        <= 1.0e-8
        and max(
            row["maximum_absolute_normal_boundary_flux"] for row in continuity_rows
        )
        <= float(gates["manufactured_zero_normal_flux_absolute"])
    )
    manufactured_pass = bool(
        manufactured_rows
        and max(row["relative_l2_error"] for row in manufactured_rows)
        <= float(gates["manufactured_continuity_relative_l2"])
        and max(row["maximum_weak_relative_error"] for row in manufactured_rows)
        <= float(gates["manufactured_continuity_max_weak_relative"])
        and max(row["mass_error"] for row in manufactured_rows)
        <= float(gates["mass_absolute"])
        and max(
            row["maximum_absolute_normal_boundary_flux"]
            for row in manufactured_rows
        )
        <= float(gates["manufactured_zero_normal_flux_absolute"])
    )
    decomposition_tolerance = float(gates["decomposition_absolute"])
    decomposition_pass = bool(
        decomposition_rows
        and max(
            row["maximum_full_moment_rate_residual"]
            for row in decomposition_rows
        )
        <= decomposition_tolerance
        and max(
            row["maximum_tangent_moment_rate_residual"]
            for row in decomposition_rows
        )
        <= decomposition_tolerance
        and max(
            row["maximum_hidden_nullspace_residual"]
            for row in decomposition_rows
        )
        <= decomposition_tolerance
        and max(
            row["maximum_absolute_tangent_hidden_inner_product"]
            for row in decomposition_rows
        )
        <= decomposition_tolerance
        and max(
            row["maximum_absolute_pythagorean_residual"]
            for row in decomposition_rows
        )
        <= decomposition_tolerance
        and max(
            row["maximum_raw_hierarchy_violation"]
            for row in decomposition_rows
        )
        <= decomposition_tolerance
        and max(row["full_action_match_relative"] for row in decomposition_rows)
        <= decomposition_tolerance
    )
    golden_central = float(golden_result["action_by_time"][10])
    maximum_curated_instantaneous = max(
        row["max_action_by_time"] for row in default_rows if row["grid_nx"] == fine_nx
    )
    catastrophic_threshold = float(
        config["development"]["catastrophic_instantaneous_action_threshold"]
    )
    catastrophic_tail = bool(maximum_curated_instantaneous >= catastrophic_threshold)
    all_required_pass = bool(
        grid_pass
        and particle_pass
        and solver_pass
        and manufactured_pass
        and continuity_pass
        and decomposition_pass
    )
    summary = {
        "schema_version": 2,
        "method_version": V2_VERSION,
        "status": (
            "READY_TO_FREEZE_BEFORE_REFERENCE_TRAINING_AND_SELECTION"
            if all_required_pass
            else "NOT_READY_TO_FREEZE"
        ),
        "scientific_scope": "development_only; old trials are not V2 validation",
        "decision_gate": {
            "grid_convergence_demonstrated": grid_pass,
            "grid_action_changes_shrink": grid_change_shrinks,
            "particle_convergence_demonstrated": particle_pass,
            "independent_solver_agreement": solver_pass,
            "manufactured_reflected_continuity": manufactured_pass,
            "golden_reflected_continuity": continuity_pass,
            "continuity_epsilon_monotone": continuity_epsilon_monotone,
            "continuity_grid_range": continuity_grid_range,
            "common_raster_decomposition": decomposition_pass,
            "golden_central_action": golden_central,
            "legacy_golden_central_action": 179303.0912515869,
            "maximum_curated_instantaneous_action": maximum_curated_instantaneous,
            "development_catastrophic_threshold": catastrophic_threshold,
            "development_catastrophic_threshold_role": (
                "mechanism classification only; not an action-performance or validation gate"
            ),
            "catastrophic_hard_fiber_tail_persists_in_curated_v2_cases": catastrophic_tail,
            "soft_fiber_required": False,
            "ready_to_freeze_before_reference_training_and_new_selection": all_required_pass,
        },
        "provenance": {
            "git": git_state(),
            "config_path": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config.resolve()),
            "config_fingerprint": config_fingerprint(config),
            "core_sha256": sha256_file(HERE / "core.py"),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "v1_pareto_dir": str(args.pareto_dir.resolve()),
            "development_bank": str(args.bank.resolve()),
            "development_bank_sha256": sha256_file(args.bank.resolve()),
            "random_namespace": 19892,
            "projection_type": "hard",
            "scientific_raster": "direct cell-integrated even reflection; matched odd-normal flux",
            "reference_particle_policy": "nested deterministic prefixes; projection recalibrated at every N",
        },
        "raster": {
            "definition": config["raster"],
            "computed_reference_bandwidth": computed_bandwidth,
            "used_reference_bandwidth": reference_bandwidth,
            "bandwidth_by_time": bandwidth_by_time,
        },
        "cases": cases,
        "grid_rows": grid_rows,
        "particle_rows": particle_rows,
        "continuity_rows": continuity_rows,
        "manufactured_continuity_rows": manufactured_rows,
        "independent_solver_rows": solver_rows,
        "decomposition_rows": decomposition_rows,
    }
    shared_csv = {
        "method_version": V2_VERSION,
        "config_sha256": summary["provenance"]["config_sha256"],
        "config_fingerprint": summary["provenance"]["config_fingerprint"],
        "core_sha256": summary["provenance"]["core_sha256"],
        "script_sha256": summary["provenance"]["script_sha256"],
        "random_namespace": 19892,
        "projection_type_global": "hard",
        "raster_definition": config["raster"],
    }
    compact_rows = []
    for row in grid_rows + particle_rows:
        compact_rows.append({
            **shared_csv,
            **row,
        })
    for row in continuity_rows:
        compact_rows.append({
            **shared_csv,
            "study": "continuity",
            "particle_count": 32768,
            "physical_bandwidth": reference_bandwidth,
            **row,
        })
    for row in solver_rows:
        compact_rows.append({
            **shared_csv,
            "study": "independent_poisson",
            "particle_count": 32768,
            "physical_bandwidth": reference_bandwidth,
            **row,
        })
    for row in manufactured_rows:
        compact_rows.append({
            **shared_csv,
            "study": "manufactured_reflected_continuity",
            "physical_bandwidth": reference_bandwidth,
            **row,
        })
    for row in decomposition_rows:
        compact_rows.append({
            **shared_csv,
            "study": "reflected_common_raster_decomposition",
            "particle_count": 32768,
            **row,
        })
    save_csv(output / "convergence_summary.csv", compact_rows)
    (output / "convergence_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(jsonable(summary["decision_gate"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
