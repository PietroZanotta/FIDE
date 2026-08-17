#!/usr/bin/env python3
"""Separate finite-bank Monte Carlo sparsity from reference convex-hull mismatch."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from phase2_common import (  # noqa: E402
    gaussian_features_numpy,
    load_phase2_config,
    resolve,
    write_csv,
    write_json,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def representative_failures(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    failed = [row for row in rows if row["valid"] == "False"]
    coordinate = [row for row in failed if "coordinate_support_infeasible" in row["failure_reason"]]
    joint_or_numerical = [row for row in failed if "coordinate_support_infeasible" not in row["failure_reason"]]

    def spread(candidates: list[dict[str, str]], needed: int, used: set[tuple[int, int]]) -> list[dict[str, str]]:
        candidates = sorted(
            candidates,
            key=lambda row: (
                int(row["source_time_index"]),
                float(row["verified_moment_residual"]),
                int(row["design_index"]),
            ),
        )
        chosen: list[dict[str, str]] = []
        for raw_index in np.linspace(0, len(candidates) - 1, max(needed * 4, needed), dtype=int):
            row = candidates[int(raw_index)]
            key = (int(row["design_index"]), int(row["source_time_index"]))
            if key not in used:
                used.add(key); chosen.append(row)
            if len(chosen) == needed:
                break
        if len(chosen) < needed:
            for row in candidates:
                key = (int(row["design_index"]), int(row["source_time_index"]))
                if key not in used:
                    used.add(key); chosen.append(row)
                if len(chosen) == needed:
                    break
        return chosen

    used: set[tuple[int, int]] = set()
    coordinate_n = count // 2
    selected = spread(coordinate, coordinate_n, used)
    selected += spread(joint_or_numerical, count - len(selected), used)
    if len(selected) != count:
        selected += spread(failed, count - len(selected), used)
    return sorted(selected, key=lambda row: (int(row["source_time_index"]), int(row["design_index"])))


def simplex_hull_lp(phi: np.ndarray, target: np.ndarray) -> dict[str, object]:
    """Test exact feasibility and independently minimize L-infinity residual."""
    phi = np.asarray(phi, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    particle_n, moment_n = phi.shape
    equality = np.vstack([np.ones(particle_n), phi.T])
    rhs = np.r_[1.0, target]
    exact = linprog(
        np.zeros(particle_n), A_eq=equality, b_eq=rhs,
        bounds=(0.0, None), method="highs",
        options={"dual_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9},
    )
    # Variables are [w_1,...,w_M,r], minimizing the maximum absolute
    # componentwise moment residual under the probability-simplex constraint.
    objective = np.r_[np.zeros(particle_n), 1.0]
    upper = np.vstack([
        np.c_[phi.T, -np.ones(moment_n)],
        np.c_[-phi.T, -np.ones(moment_n)],
    ])
    upper_rhs = np.r_[target, -target]
    equality_residual = np.c_[np.ones((1, particle_n)), np.zeros((1, 1))]
    minimum = linprog(
        objective, A_ub=upper, b_ub=upper_rhs,
        A_eq=equality_residual, b_eq=np.asarray([1.0]),
        bounds=[(0.0, None)] * particle_n + [(0.0, None)], method="highs",
        options={"dual_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9},
    )
    if not minimum.success:
        raise RuntimeError(f"residual LP failed: {minimum.message}")
    weights = minimum.x[:-1]
    achieved = weights @ phi
    residual = achieved - target
    return {
        "exact_lp_success": bool(exact.success),
        "minimum_linf_residual": float(np.max(np.abs(residual))),
        "minimum_l2_residual_at_linf_solution": float(np.linalg.norm(residual)),
        "active_weight_count": int(np.sum(weights > 1e-10)),
        "maximum_weight": float(weights.max()),
        "lp_status": minimum.message,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--cases", type=int, default=20)
    args = parser.parse_args()
    cfg = load_phase2_config(args.config)
    processed = resolve(cfg["processed_dir"])
    model_dir = resolve(cfg["model_dir"])
    analysis = resolve(cfg["analysis_dir"])
    table_dir = analysis / "tables"
    figure_dir = analysis / "figures/iprojection"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = read_csv(table_dir / "iprojection_diagnostics.csv")
    selected = representative_failures(diagnostics, args.cases)
    if len(selected) != args.cases:
        raise RuntimeError(f"could not select {args.cases} failed design-time cases")
    with np.load(model_dir / "reference_bank.npz", allow_pickle=False) as data:
        bank = np.asarray(data["nodes_km"], dtype=np.float64)
        initial_indices = np.asarray(data["initial_inference_indices"], dtype=int)
    with np.load(processed / "sensor_bank.npz", allow_pickle=False) as data:
        centers = np.asarray(data["centers_km"], dtype=np.float64)
        sigma = float(data["sigma_km"])
        design_ids = np.asarray(data["design_id"]).astype(str)
        styles = np.asarray(data["style"]).astype(str)
    with np.load(processed / "measurement_trajectories.npz", allow_pickle=False) as data:
        measurements = np.asarray(data["c"], dtype=np.float64)
    # The current reference is a deterministic ODE pushforward of a 200-atom
    # empirical P0. Extract its exact unique trajectory support once.
    unique_initial = np.unique(initial_indices)
    if len(unique_initial) != 200:
        raise RuntimeError(f"expected 200 unique initial atoms, got {len(unique_initial)}")
    representative_columns = np.asarray([
        np.flatnonzero(initial_indices == index)[0] for index in unique_initial
    ], dtype=int)
    unique_paths = bank[:, representative_columns]
    np.testing.assert_allclose(
        bank,
        unique_paths[:, np.searchsorted(unique_initial, initial_indices)],
        rtol=0.0, atol=1e-12,
    )
    sizes = [2_000, 10_000, 50_000, 200_000]
    rng = np.random.default_rng(int(cfg["seed"]) + 88001)
    resampled_atoms = rng.integers(0, len(unique_initial), size=max(sizes))
    support_by_size = {size: np.unique(resampled_atoms[:size]) for size in sizes}
    rows: list[dict[str, object]] = []
    for case_index, source in enumerate(selected):
        design_index = int(source["design_index"])
        source_time_index = int(source["source_time_index"])
        target = measurements[design_index, source_time_index]
        for size in sizes:
            atoms = support_by_size[size]
            points = unique_paths[source_time_index, atoms]
            phi = gaussian_features_numpy(points, centers[design_index], sigma)
            result = simplex_hull_lp(phi, target)
            coordinate_margin = float(np.min(np.minimum(
                target - phi.min(axis=0), phi.max(axis=0) - target
            )))
            rows.append({
                "case": case_index,
                "design_id": design_ids[design_index],
                "design_index": design_index,
                "style": styles[design_index],
                "day": float(source["day"]),
                "source_time_index": source_time_index,
                "original_failure_reason": source["failure_reason"],
                "original_newton_residual": float(source["verified_moment_residual"]),
                "nominal_particle_count": size,
                "unique_reference_paths": len(atoms),
                "duplicate_paths_collapsed": size - len(atoms),
                "coordinate_support_margin": coordinate_margin,
                **result,
            })
    write_csv(table_dir / "reference_support_lp.csv", rows)
    largest = [row for row in rows if row["nominal_particle_count"] == max(sizes)]
    feasible_tol = float(cfg["projection"]["accept_residual"])
    exact_count = sum(
        row["exact_lp_success"] and row["minimum_linf_residual"] <= feasible_tol
        for row in largest
    )
    invariant_count = 0
    for case in range(args.cases):
        values = [row["minimum_linf_residual"] for row in rows if row["case"] == case]
        invariant_count += bool(np.ptp(values) <= 1e-12)
    summary = {
        "diagnostic": "simplex convex-hull feasibility independent of Newton/L-BFGS",
        "representative_failed_cases": args.cases,
        "nominal_particle_counts": sizes,
        "unique_support_counts": {str(size): int(len(support_by_size[size])) for size in sizes},
        "reference_initial_law": "200-atom empirical inference P0",
        "reference_dynamics": "deterministic endpoint-flow ODE",
        "duplicate_collapse_is_exact": True,
        "lp_feasible_at_200000": exact_count,
        "lp_infeasible_at_200000": args.cases - exact_count,
        "cases_with_residual_invariant_across_sizes_at_1e-12": invariant_count,
        "accept_residual": feasible_tol,
        "interpretation": (
            "Increasing nominal M does not add support: all sizes contain the same 200 transported atoms. "
            "Positive LP residuals are genuine convex-hull mismatch for the current reference law; "
            "zero-residual LP cases isolate Newton regularity/ESS failures rather than Monte Carlo sparsity."
        ),
        "final_test_artifact_loaded": False,
    }
    write_json(table_dir / "reference_support_lp_summary.json", summary)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for case in range(args.cases):
        case_rows = [row for row in rows if row["case"] == case]
        ax.plot(
            [row["nominal_particle_count"] for row in case_rows],
            [max(float(row["minimum_linf_residual"]), 1e-12) for row in case_rows],
            marker="o", ms=3, alpha=.65,
        )
    ax.axhline(feasible_tol, color="black", ls="--", label="projection residual contract")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("nominal reference particles M")
    ax.set_ylabel("minimum simplex LP moment residual (L-infinity)")
    ax.set_title("Reference-support diagnostic: duplicate particles add no convex-hull support")
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(figure_dir / "reference_support_lp_scaling.png", dpi=190)
    plt.close(fig)
    report = f"""# Reference-support convex-hull diagnostic

The diagnostic evaluated {args.cases} representative failed design-time pairs at nominal bank sizes {', '.join(f'{size:,}' for size in sizes)}. It used a pure probability-simplex linear program, independent of the native Newton solver, both to test exact moment feasibility and to minimize the maximum componentwise moment residual.

The current endpoint reference is a deterministic ODE pushforward of the 200-atom empirical inference initial law. Consequently, its 4,000-particle bank contains 20 exact copies of each transported initial atom. Ordinary regeneration at larger nominal M also samples only those same 200 paths. All tested sizes contained all {len(unique_initial)} unique paths; collapsing duplicates before the LP is mathematically exact.

At M=200,000, **{exact_count}/{args.cases}** failed native cases are convex-hull feasible to the frozen residual tolerance, while **{args.cases - exact_count}/{args.cases}** retain a positive minimum LP residual. Residuals are invariant across the four nominal sizes in {invariant_count}/{args.cases} cases to 1e-12.

## Interpretation

Nominal particle count is not the cheap fix for this implementation: it cannot enlarge the support of a deterministic flow driven by a discrete 200-atom initial law. Positive LP residual cases are therefore genuine support mismatch for the current reference law (problem B), not finite-bank Monte Carlo sparsity. LP-feasible cases that the native projection rejects instead identify boundary/regularity or ESS failures; adding duplicate particles will not repair their effective support either.

This result does not prove that every continuous endpoint-only reference would fail. It says that the currently frozen discrete deterministic reference cannot answer the proposed M-scaling question through brute-force replication. A scientifically meaningful next experiment must change the proposal support—such as a predeclared continuous initial-law estimator or an endpoint-flow ensemble—and then repeat this same LP audit before rerunning the full bank.

See [the full LP table](tables/reference_support_lp.csv), [machine-readable summary](tables/reference_support_lp_summary.json), and [scaling figure](figures/iprojection/reference_support_lp_scaling.png).
"""
    (analysis / "reference_support_diagnostic.md").write_text(report, encoding="utf-8")
    print(
        f"[support LP] M={sizes}; unique={[len(support_by_size[size]) for size in sizes]}; "
        f"feasible={exact_count}/{args.cases}; invariant={invariant_count}/{args.cases}",
        flush=True,
    )


if __name__ == "__main__":
    main()
