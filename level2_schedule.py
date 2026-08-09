#!/usr/bin/env python3
"""Small MFSI level-2 experiment: learn a fiber-adapted SI noise schedule.

The exact one-dimensional Experiment-A bridge is intentionally reused so the
level-2 claim can be checked without disturbing or retraining Experiments A/B.
The direct backend executes the Tesseract's JAX recipe in-process; the
Tesseract backend sends the identical payload across the served container's
REST boundary.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from backend_runtime import _post, normalize_backend

ROOT = Path(__file__).resolve().parent
API_PATH = ROOT / "tesseracts" / "fiber_path_adapter" / "tesseract_api.py"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "level2_schedule"


def _load_jax_api():
    spec = importlib.util.spec_from_file_location("mfsi_level2_tesseract_api", API_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load level-2 JAX recipe: {API_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inverse_softplus(value: np.ndarray | float) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return np.log(np.expm1(value))


def _payload(quick: bool) -> dict[str, Any]:
    n_grid = 401 if quick else 801
    n_times = 11 if quick else 17
    n_landscape = 41 if quick else 71
    grid_x = np.linspace(-7.0, 7.0, n_grid, dtype=np.float64)
    dx = grid_x[1] - grid_x[0]
    grid_weights = np.full_like(grid_x, dx)
    grid_weights[[0, -1]] = 0.5 * dx
    landscape_beta = np.linspace(0.05, 1.80, n_landscape, dtype=np.float64)
    return {
        "grid_x": grid_x,
        "grid_weights": grid_weights,
        "times": np.linspace(0.0, 1.0, n_times, dtype=np.float64),
        "landscape_raw": _inverse_softplus(landscape_beta),
        "amplitude": 0.8,
        "target": np.asarray([0.0, 1.0], dtype=np.float64),
        "schedule_raw": float(_inverse_softplus(0.25)),
        "ess_floor": 0.60,
        "ess_penalty": 25.0,
        "learning_rate": 0.05,
        "finite_difference_epsilon": 2e-4,
    }


def _run_backend(backend: str, payload: dict[str, Any]) -> dict[str, Any]:
    if backend == "jax":
        return _load_jax_api().apply_payload(payload)
    url = os.environ.get("MFSI_LEVEL2_TESSERACT_URL")
    if not url:
        raise RuntimeError(
            "level-2 Tesseract URL is missing; invoke this through "
            "scripts/run_level2.sh --backend tesseract"
        )
    return _post(url, "apply", {"inputs": payload})


def _scalar(result: dict[str, Any], key: str) -> float:
    return float(np.asarray(result[key]))


def _summary(backend: str, result: dict[str, Any], payload: dict[str, Any], quick: bool) -> dict[str, Any]:
    times = np.asarray(result["times"])
    initial_energy = float(np.trapezoid(result["initial_correction_energy"], times))
    optimized_energy = float(np.trapezoid(result["optimized_correction_energy"], times))
    initial_ess = float(np.min(result["initial_ess_fraction"]))
    optimized_ess = float(np.min(result["optimized_ess_fraction"]))
    target = np.asarray(payload["target"])
    initial_moment_error = float(np.max(np.abs(np.asarray(result["initial_moments"]) - target)))
    optimized_moment_error = float(np.max(np.abs(np.asarray(result["optimized_moments"]) - target)))
    max_calibration_residual = float(np.max(result["optimized_calibration_residual"]))
    reduction = 1.0 - optimized_energy / max(initial_energy, 1e-30)
    gates = {
        "correction_energy_reduced": optimized_energy < 0.05 * initial_energy,
        "optimized_path_meets_ess_floor": optimized_ess >= payload["ess_floor"],
        "implicit_gradient_matches_finite_difference": _scalar(result, "gradient_relative_error") < 2e-5,
        "fiber_moments_calibrated": optimized_moment_error < 1e-9,
        "calibration_solver_converged": max_calibration_residual < 1e-9,
    }
    return {
        "experiment": "level-2-fiber-adapted-reference-schedule",
        "backend": backend,
        "mode": "quick" if quick else "standard",
        "description": (
            "Optimize the constant stochastic-interpolant noise amplitude beta "
            "for Experiment A using integrated exact correction energy plus an ESS penalty."
        ),
        "configuration": {
            "amplitude": payload["amplitude"],
            "target_moments": target.tolist(),
            "time_points": len(payload["times"]),
            "quadrature_points": len(payload["grid_x"]),
            "initial_beta": _scalar(result, "initial_beta"),
            "ess_floor": payload["ess_floor"],
            "ess_penalty": payload["ess_penalty"],
            "optimization_steps": int(np.asarray(result["optimization_steps"])),
            "learning_rate": payload["learning_rate"],
        },
        "metrics": {
            "optimized_beta": _scalar(result, "optimized_beta"),
            "initial_objective": _scalar(result, "initial_objective"),
            "optimized_objective": _scalar(result, "optimized_objective"),
            "initial_integrated_correction_energy": initial_energy,
            "optimized_integrated_correction_energy": optimized_energy,
            "correction_energy_reduction_fraction": reduction,
            "initial_min_ess_fraction": initial_ess,
            "optimized_min_ess_fraction": optimized_ess,
            "initial_max_moment_error": initial_moment_error,
            "optimized_max_moment_error": optimized_moment_error,
            "optimized_max_calibration_residual": max_calibration_residual,
            "implicit_gradient": _scalar(result, "initial_gradient"),
            "finite_difference_gradient": _scalar(result, "finite_difference_gradient"),
            "gradient_relative_error": _scalar(result, "gradient_relative_error"),
        },
        "acceptance_gates": gates,
        "passed": all(gates.values()),
    }


def _style():
    plt.rcParams.update({
        "figure.facecolor": "#f7f3ea",
        "axes.facecolor": "#fffdf8",
        "axes.edgecolor": "#293241",
        "axes.labelcolor": "#293241",
        "text.color": "#293241",
        "font.size": 10,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "--",
    })


def _plot_summary(result: dict[str, Any], payload: dict[str, Any], output: Path, backend: str):
    _style()
    navy, coral, teal, gold = "#1d3557", "#e76f51", "#2a9d8f", "#e9c46a"
    beta = np.asarray(result["landscape_beta"])
    landscape_objective = np.asarray(result["landscape_objective"])
    times = np.asarray(result["times"])
    trace_objective = np.asarray(result["optimization_objective"])
    trace_beta = np.asarray(result["optimization_beta"])

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(beta, landscape_objective, color=navy, lw=2.4)
    ax.axvline(1.0, color=gold, lw=1.8, ls="--", label=r"variance-matched $\beta=1$")
    ax.scatter([_scalar(result, "initial_beta")], [_scalar(result, "initial_objective")],
               s=70, color=coral, zorder=4, label="initial")
    ax.scatter([_scalar(result, "optimized_beta")], [_scalar(result, "optimized_objective")],
               s=80, color=teal, edgecolor="white", zorder=5, label="optimized")
    ax.set(xlabel=r"schedule amplitude $\beta$", ylabel="level-2 objective", title="Fiber-adapted objective landscape")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    steps = np.arange(1, len(trace_objective) + 1)
    ax.semilogy(steps, np.maximum(trace_objective, 1e-12), color=navy, lw=2.2, label="objective")
    ax.set(xlabel="Adam step", ylabel="objective (log scale)", title="Implicit-gradient optimization")
    twin = ax.twinx()
    twin.plot(steps, trace_beta, color=teal, lw=1.8, alpha=0.9, label=r"$\beta$")
    twin.axhline(1.0, color=gold, ls="--", lw=1.4)
    twin.set_ylabel(r"schedule amplitude $\beta$", color=teal)
    twin.grid(False)

    ax = axes[1, 0]
    ax.plot(times, result["initial_correction_energy"], color=coral, marker="o", ms=4, label="initial")
    ax.plot(times, result["optimized_correction_energy"], color=teal, marker="o", ms=4, label="optimized")
    ax.fill_between(times, 0, result["initial_correction_energy"], color=coral, alpha=0.10)
    ax.set(xlabel="path time t", ylabel=r"$E_{q_t}[|\nabla\psi_t|^2]$", title="Correction energy along the path")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(times, result["initial_ess_fraction"], color=coral, marker="o", ms=4, label="initial")
    ax.plot(times, result["optimized_ess_fraction"], color=teal, marker="o", ms=4, label="optimized")
    ax.axhline(payload["ess_floor"], color=navy, ls="--", lw=1.5, label="ESS floor")
    ax.set(xlabel="path time t", ylabel="relative ESS", ylim=(0.0, 1.04), title="Projection overlap constraint")
    ax.legend(frameon=False)

    fig.suptitle(f"MFSI level 2 · fiber-adapted reference path · {backend.upper()} backend", fontsize=15, fontweight="bold")
    fig.savefig(output / "level2_schedule_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_density_paths(result: dict[str, Any], output: Path):
    _style()
    navy, coral, teal = "#1d3557", "#e76f51", "#2a9d8f"
    x = np.asarray(result["grid_x"])
    times = np.asarray(result["times"])
    chosen = [int(np.argmin(np.abs(times - value))) for value in (0.23, 0.50, 0.77)]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.5), sharex=True, sharey=True, constrained_layout=True)
    rows = [
        ("initial", result["initial_reference_density"], result["initial_projected_density"], coral),
        ("optimized", result["optimized_reference_density"], result["optimized_projected_density"], teal),
    ]
    for row, (label, reference, projected, accent) in enumerate(rows):
        reference, projected = np.asarray(reference), np.asarray(projected)
        for col, index in enumerate(chosen):
            ax = axes[row, col]
            ax.plot(x, reference[index], color=navy, ls="--", lw=1.8, label="raw reference")
            ax.plot(x, projected[index], color=accent, lw=2.4, label="I-projected fiber law")
            ax.fill_between(x, 0, projected[index], color=accent, alpha=0.12)
            ax.set_xlim(-3.5, 3.5)
            ax.set_title(f"{label.capitalize()} path · t={times[index]:.2f}")
            if col == 0:
                ax.set_ylabel("density")
            if row == 1:
                ax.set_xlabel("x")
            if row == 0 and col == 2:
                ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Raw bridge versus moment-fiber projection", fontsize=15, fontweight="bold")
    fig.savefig(output / "level2_density_paths.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_outputs(result: dict[str, Any], summary: dict[str, Any], output: Path, payload: dict[str, Any], plots: bool):
    output.mkdir(parents=True, exist_ok=True)
    (output / "level2_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        output / "level2_arrays.npz",
        **{key: np.asarray(value) for key, value in result.items()},
    )
    if plots:
        _plot_summary(result, payload, output, summary["backend"])
        _plot_density_paths(result, output)


def _print_summary(summary: dict[str, Any], output: Path):
    metrics = summary["metrics"]
    print("\nMFSI level-2 fiber-adapted schedule")
    print(f"  backend                         {summary['backend']}")
    print(f"  beta: initial -> optimized      {summary['configuration']['initial_beta']:.6f} -> {metrics['optimized_beta']:.6f}")
    print(f"  integrated correction energy    {metrics['initial_integrated_correction_energy']:.6e} -> {metrics['optimized_integrated_correction_energy']:.6e}")
    print(f"  correction-energy reduction     {100.0 * metrics['correction_energy_reduction_fraction']:.3f}%")
    print(f"  minimum relative ESS            {metrics['initial_min_ess_fraction']:.6f} -> {metrics['optimized_min_ess_fraction']:.6f}")
    print(f"  implicit/FD gradient rel. error  {metrics['gradient_relative_error']:.3e}")
    print(f"  acceptance gates                {'PASS' if summary['passed'] else 'FAIL'}")
    print(f"  outputs                         {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("jax", "tesseract"), default=None)
    parser.add_argument("--quick", action="store_true", help="smaller quadrature/plot grid for a smoke run")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    backend = normalize_backend(args.backend)
    output = args.output_dir or (DEFAULT_OUTPUT_ROOT / backend)
    payload = _payload(args.quick)
    result = _run_backend(backend, payload)
    summary = _summary(backend, result, payload, args.quick)
    _write_outputs(result, summary, output, payload, not args.no_plots)
    _print_summary(summary, output)
    if not summary["passed"]:
        raise SystemExit("level-2 acceptance gates failed")


if __name__ == "__main__":
    main()
