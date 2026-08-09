#!/usr/bin/env python3
"""Advanced level-2 MFSI suite: finite-neural and 32D many-body studies."""
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
OUTPUT_ROOT = ROOT / "results" / "level2_suite"
APIS = {
    "finite_neural": ROOT / "tesseracts" / "finite_neural_path" / "tesseract_api.py",
    "manybody": ROOT / "tesseracts" / "manybody_neural_path" / "tesseract_api.py",
}
URL_VARIABLES = {
    "finite_neural": "MFSI_FINITE_NEURAL_TESSERACT_URL",
    "manybody": "MFSI_MANYBODY_TESSERACT_URL",
}


def _load_api(experiment: str):
    path = APIS[experiment]
    spec = importlib.util.spec_from_file_location(f"mfsi_{experiment}_api", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load JAX recipe: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inverse_softplus(value: float) -> float:
    return float(np.log(np.expm1(value)))


def _finite_endpoint_banks(rng: np.random.Generator, times: np.ndarray, count: int):
    shape = (len(times), count)
    thickness = 0.16
    radius = np.sqrt(2.0 * (1.0 - thickness**2))
    angle_minus = rng.uniform(0.0, 2.0 * np.pi, shape)
    minus = radius * np.stack([np.cos(angle_minus), np.sin(angle_minus)], axis=-1)
    minus += thickness * rng.normal(size=(*shape, 2))
    lobe = rng.integers(0, 4, shape)
    angle_plus = 0.5 * np.pi * lobe
    plus = radius * np.stack([np.cos(angle_plus), np.sin(angle_plus)], axis=-1)
    plus += thickness * rng.normal(size=(*shape, 2))
    noise = rng.normal(size=(*shape, 2))
    return minus, plus, noise


def finite_payload(quick: bool) -> dict[str, Any]:
    rng = np.random.default_rng(202608091)
    times = np.linspace(0.05, 0.95, 7 if quick else 9, dtype=np.float64)
    train_count = 192 if quick else 512
    validation_count = 384 if quick else 1024
    feature_count = 40 if quick else 64
    test_count = 20 if quick else 32
    train = _finite_endpoint_banks(rng, times, train_count)
    validation = _finite_endpoint_banks(rng, times, validation_count)
    return {
        "times": times,
        "train_minus": train[0],
        "train_plus": train[1],
        "train_noise": train[2],
        "validation_minus": validation[0],
        "validation_plus": validation[1],
        "validation_noise": validation[2],
        "target": np.asarray([0.0, 0.0, 1.0, 0.0, 1.0]),
        "feature_weight": rng.normal(scale=0.60, size=(feature_count, 2)),
        "feature_bias": rng.normal(scale=0.40, size=feature_count),
        "feature_time": rng.normal(scale=0.50, size=feature_count),
        "test_feature_weight": rng.normal(scale=0.60, size=(test_count, 2)),
        "test_feature_bias": rng.normal(scale=0.40, size=test_count),
        "test_feature_time": rng.normal(scale=0.50, size=test_count),
        "initial_raw": np.asarray([_inverse_softplus(0.25), 0.50, -0.40]),
        "ess_floor": 0.30,
        "ess_penalty": 12.0,
        "schedule_regularization": 1e-3,
        "learning_rate": 0.025,
        "gradient_check_direction": np.asarray([1.0, 0.0, 0.0]),
        "finite_difference_epsilon": 2e-4,
    }


def _normalize_configuration(configuration: np.ndarray) -> np.ndarray:
    centered = configuration - np.mean(configuration, axis=0, keepdims=True)
    covariance = centered.T @ centered / centered.shape[0]
    values, vectors = np.linalg.eigh(covariance)
    whitening = vectors @ np.diag(np.sqrt(0.5 / np.maximum(values, 1e-6))) @ vectors.T
    return centered @ whitening


def _manybody_banks(rng: np.random.Generator, times: np.ndarray, count: int, particles: int):
    minus = np.empty((len(times), count, particles, 2), dtype=np.float64)
    plus = np.empty_like(minus)
    noise = np.empty_like(minus)
    base_angles = 2.0 * np.pi * np.arange(particles) / particles
    cluster_ids = np.repeat(np.arange(4), particles // 4)
    for time_index in range(len(times)):
        for sample_index in range(count):
            rotation = rng.uniform(0.0, 2.0 * np.pi)
            ring_angle = rotation + base_angles + rng.normal(0.0, 0.045, particles)
            ring_radius = 1.0 + rng.normal(0.0, 0.045, particles)
            ring = np.stack(
                [ring_radius * np.cos(ring_angle), ring_radius * np.sin(ring_angle)], axis=-1
            )
            cluster_angle = rotation + 0.5 * np.pi * cluster_ids
            clusters = np.stack([np.cos(cluster_angle), np.sin(cluster_angle)], axis=-1)
            clusters += rng.normal(0.0, 0.13, size=(particles, 2))
            minus[time_index, sample_index] = _normalize_configuration(ring)
            plus[time_index, sample_index] = _normalize_configuration(clusters)
            noise[time_index, sample_index] = _normalize_configuration(
                rng.normal(size=(particles, 2))
            )
    return minus, plus, noise


def manybody_payload(quick: bool) -> dict[str, Any]:
    rng = np.random.default_rng(202608092)
    particles = 16
    times = np.linspace(0.08, 0.92, 5 if quick else 7, dtype=np.float64)
    train_count = 32 if quick else 72
    validation_count = 72 if quick else 160
    radial_centers = np.linspace(0.30, 2.40, 7 if quick else 9)
    feature_count = 10 if quick else 18
    test_count = 8 if quick else 12
    train = _manybody_banks(rng, times, train_count, particles)
    validation = _manybody_banks(rng, times, validation_count, particles)
    descriptor_count = len(radial_centers)
    return {
        "times": times,
        "train_minus": train[0],
        "train_plus": train[1],
        "train_noise": train[2],
        "validation_minus": validation[0],
        "validation_plus": validation[1],
        "validation_noise": validation[2],
        "target": np.asarray([1.0, 0.0, 0.0]),
        "radial_centers": radial_centers,
        "radial_width": 0.36,
        "box_size": 8.0,
        "feature_weight": rng.normal(scale=0.65, size=(feature_count, descriptor_count)),
        "feature_bias": rng.normal(scale=0.30, size=feature_count),
        "feature_time": rng.normal(scale=0.40, size=feature_count),
        "test_feature_weight": rng.normal(scale=0.65, size=(test_count, descriptor_count)),
        "test_feature_bias": rng.normal(scale=0.30, size=test_count),
        "test_feature_time": rng.normal(scale=0.40, size=test_count),
        # Deliberately time-dependent but still empirically feasible; schedules
        # with beta near 0.3 collapse the finite-bank convex hull at mid-path.
        "initial_raw": np.asarray([_inverse_softplus(0.55), 0.22, -0.16]),
        "ess_floor": 0.22,
        "ess_penalty": 12.0,
        "schedule_regularization": 2e-3,
        "learning_rate": 0.020,
        "gradient_check_direction": np.asarray([1.0, 0.0, 0.0]),
        "finite_difference_epsilon": 2e-4,
    }


def _run_backend(experiment: str, backend: str, payload: dict[str, Any]):
    if backend == "jax":
        return _load_api(experiment).apply_payload(payload)
    url = os.environ.get(URL_VARIABLES[experiment])
    if not url:
        raise RuntimeError(
            f"{URL_VARIABLES[experiment]} is missing; invoke through scripts/run_level2_suite.sh"
        )
    return _post(url, "apply", {"inputs": payload}, timeout=900.0)


def _integral(values, times):
    return float(np.trapezoid(np.asarray(values), np.asarray(times)))


def _summary(experiment: str, backend: str, result: dict[str, Any], payload: dict[str, Any], quick: bool):
    times = np.asarray(result["times"])
    initial_energy = _integral(result["initial_validation_energy"], times)
    optimized_energy = _integral(result["optimized_validation_energy"], times)
    initial_ritz_gain = _integral(result["initial_validation_ritz_gain"], times)
    optimized_ritz_gain = _integral(result["optimized_validation_ritz_gain"], times)
    min_initial_ess = float(np.min(result["initial_validation_ess"]))
    min_optimized_ess = float(np.min(result["optimized_validation_ess"]))
    max_residual = float(np.max(result["optimized_validation_calibration_residual"]))
    gradient_error = float(np.asarray(result["gradient_relative_error"]))
    energy_reduction = 1.0 - optimized_energy / max(initial_energy, 1e-30)
    gates = {
        "fresh_bank_energy_reduced": optimized_energy < 0.55 * initial_energy,
        "fresh_bank_overlap_improved": min_optimized_ess > min_initial_ess,
        "fresh_bank_calibrated": max_residual < 1e-8,
        "implicit_gradient_matches_finite_difference": gradient_error < 2e-4,
    }
    # The small quick bank is a plumbing check; fresh-bank neural generalization
    # is a scientific gate only at the standard budget.
    if not quick:
        gates["neural_correction_beats_zero_on_fresh_initial_path"] = initial_ritz_gain > 0.0
    if experiment == "manybody":
        q4 = np.asarray(result["optimized_validation_q4"])
        gates["hidden_manybody_structure_moves"] = float(q4[-1] - q4[0]) > 0.45
        gates["state_is_high_dimensional"] = int(np.asarray(result["state_dimension"])) >= 32
    return {
        "experiment": experiment,
        "backend": backend,
        "mode": "quick" if quick else "standard",
        "configuration": {
            "time_points": len(times),
            "training_bank_size": int(payload["train_minus"].shape[1]),
            "validation_bank_size": int(payload["validation_minus"].shape[1]),
            "schedule_parameters": 3,
            "neural_features": int(payload["feature_weight"].shape[0]),
            "ess_floor": float(payload["ess_floor"]),
            "optimization_steps": int(np.asarray(result["optimization_steps"])),
            "state_dimension": int(np.asarray(result.get("state_dimension", 2))),
            "particle_count": int(np.asarray(result.get("particle_count", 1))),
        },
        "metrics": {
            "initial_integrated_fresh_energy": initial_energy,
            "optimized_integrated_fresh_energy": optimized_energy,
            "fresh_energy_reduction_fraction": energy_reduction,
            "initial_min_fresh_ess": min_initial_ess,
            "optimized_min_fresh_ess": min_optimized_ess,
            "initial_integrated_fresh_ritz_gain": initial_ritz_gain,
            "optimized_integrated_fresh_ritz_gain": optimized_ritz_gain,
            "max_fresh_calibration_residual": max_residual,
            "gradient_relative_error": gradient_error,
            "initial_raw": np.asarray(result["initial_raw"]).tolist(),
            "optimized_raw": np.asarray(result["optimized_raw"]).tolist(),
            "initial_beta_range": [float(np.min(result["initial_beta"])), float(np.max(result["initial_beta"]))],
            "optimized_beta_range": [float(np.min(result["optimized_beta"])), float(np.max(result["optimized_beta"]))],
        },
        "acceptance_gates": gates,
        "passed": all(gates.values()),
    }


def _style():
    plt.rcParams.update({
        "figure.facecolor": "#f4f1ea", "axes.facecolor": "#fffdf8",
        "axes.edgecolor": "#263238", "text.color": "#263238",
        "axes.labelcolor": "#263238", "axes.titleweight": "bold",
        "axes.grid": True, "grid.alpha": 0.18, "grid.linestyle": "--",
    })


def _plot_dashboard(experiment: str, backend: str, result: dict[str, Any], payload: dict[str, Any], output: Path):
    _style()
    navy, coral, teal, gold = "#17324d", "#e76f51", "#2a9d8f", "#e9c46a"
    times = np.asarray(result["times"])
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.0), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(times, result["initial_beta"], color=coral, marker="o", label="initial")
    ax.plot(times, result["optimized_beta"], color=teal, marker="o", label="adapted")
    ax.set(title="Three-parameter schedule", xlabel="t", ylabel=r"$\beta_\xi(t)$")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    trace = np.maximum(np.asarray(result["optimization_objective"]), 1e-12)
    ax.semilogy(np.arange(1, len(trace) + 1), trace, color=navy, lw=2.2)
    ax.set(title="Implicit-gradient optimization", xlabel="step", ylabel="training objective")

    ax = axes[0, 2]
    ax.plot(times, result["initial_validation_energy"], color=coral, marker="o", label="initial")
    ax.plot(times, result["optimized_validation_energy"], color=teal, marker="o", label="adapted")
    ax.set(title="Fresh-bank neural correction", xlabel="t", ylabel="correction energy")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(times, result["initial_validation_ess"], color=coral, marker="o", label="initial")
    ax.plot(times, result["optimized_validation_ess"], color=teal, marker="o", label="adapted")
    ax.axhline(payload["ess_floor"], color=navy, ls="--", label="ESS floor")
    ax.set(title="Independent finite-bank overlap", xlabel="t", ylabel="relative ESS", ylim=(0, 1.04))
    ax.legend(frameon=False)

    ax = axes[1, 1]
    structural_key = "optimized_validation_q4" if experiment == "manybody" else "optimized_validation_angular4"
    initial_key = "initial_validation_q4" if experiment == "manybody" else "initial_validation_angular4"
    ax.plot(times, result[initial_key], color=coral, marker="o", label="initial reference")
    ax.plot(times, result[structural_key], color=teal, marker="o", label="adapted reference")
    ax.set(title="Held-out fourfold structure", xlabel="t", ylabel=r"$|q_4|$")
    ax.legend(frameon=False)

    ax = axes[1, 2]
    width = 0.34
    initial_gain = np.asarray(result["initial_validation_ritz_gain"])
    optimized_gain = np.asarray(result["optimized_validation_ritz_gain"])
    ax.bar(times - width / (2 * len(times)), initial_gain, width / len(times), color=coral, label="initial")
    ax.bar(times + width / (2 * len(times)), optimized_gain, width / len(times), color=teal, label="adapted")
    ax.axhline(0.0, color=navy, lw=1)
    ax.set(title="Fresh-bank Deep-Ritz gain vs zero", xlabel="t", ylabel="gain")
    ax.legend(frameon=False)

    title = "Finite-bank neural bridge" if experiment == "finite_neural" else "16-particle periodic bridge (32D)"
    fig.suptitle(f"MFSI level 2 · {title} · {backend.upper()}", fontsize=15, fontweight="bold")
    fig.savefig(output / f"{experiment}_dashboard.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_snapshots(experiment: str, result: dict[str, Any], output: Path):
    _style()
    times = np.asarray(result["times"])
    chosen = [0, len(times) // 2, len(times) - 1]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0), constrained_layout=True)
    for row, prefix in enumerate(("initial", "optimized")):
        states = np.asarray(result[f"{prefix}_validation_state"])
        weights = np.asarray(result[f"{prefix}_validation_weights"])
        for column, time_index in enumerate(chosen):
            ax = axes[row, column]
            if experiment == "finite_neural":
                order = np.argsort(weights[time_index])[-300:]
                sizes = 10.0 + 1000.0 * weights[time_index, order]
                ax.scatter(states[time_index, order, 0], states[time_index, order, 1],
                           c=weights[time_index, order], s=sizes, cmap="viridis", alpha=0.68,
                           edgecolors="none")
                limit = 3.2
            else:
                index = int(np.argmax(weights[time_index]))
                configuration = states[time_index, index]
                colors = np.arange(configuration.shape[0])
                ax.scatter(configuration[:, 0], configuration[:, 1], c=colors, cmap="twilight",
                           s=62, edgecolor="white", linewidth=0.6)
                ax.plot(np.r_[configuration[:, 0], configuration[0, 0]],
                        np.r_[configuration[:, 1], configuration[0, 1]],
                        color="#17324d", alpha=0.25, lw=1)
                limit = 2.4
            ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
            ax.set_title(f"{prefix.capitalize()} · t={times[time_index]:.2f}")
            ax.set_xlabel("x")
            if column == 0:
                ax.set_ylabel("y")
    fig.suptitle("Fresh-bank projected path snapshots", fontsize=15, fontweight="bold")
    fig.savefig(output / f"{experiment}_snapshots.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write(experiment: str, backend: str, result: dict[str, Any], summary: dict[str, Any], payload: dict[str, Any], plots: bool):
    output = OUTPUT_ROOT / experiment / backend
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(output / "arrays.npz", **{key: np.asarray(value) for key, value in result.items()})
    if plots:
        _plot_dashboard(experiment, backend, result, payload, output)
        _plot_snapshots(experiment, result, output)
    return output


def _print(summary: dict[str, Any], output: Path):
    metrics = summary["metrics"]
    print(f"\n{summary['experiment']} ({summary['backend']})")
    print(f"  state dimension / finite banks  {summary['configuration']['state_dimension']} / {summary['configuration']['training_bank_size']} train + {summary['configuration']['validation_bank_size']} fresh")
    print(f"  integrated fresh energy         {metrics['initial_integrated_fresh_energy']:.6e} -> {metrics['optimized_integrated_fresh_energy']:.6e}")
    print(f"  minimum fresh ESS               {metrics['initial_min_fresh_ess']:.4f} -> {metrics['optimized_min_fresh_ess']:.4f}")
    print(f"  implicit/FD gradient error      {metrics['gradient_relative_error']:.3e}")
    print(f"  gates                           {'PASS' if summary['passed'] else 'FAIL'}")
    print(f"  outputs                         {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("finite_neural", "manybody", "both"), default="both")
    parser.add_argument("--backend", choices=("jax", "tesseract"), default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    backend = normalize_backend(args.backend)
    experiments = ("finite_neural", "manybody") if args.experiment == "both" else (args.experiment,)
    failed = []
    for experiment in experiments:
        payload = finite_payload(args.quick) if experiment == "finite_neural" else manybody_payload(args.quick)
        result = _run_backend(experiment, backend, payload)
        summary = _summary(experiment, backend, result, payload, args.quick)
        output = _write(experiment, backend, result, summary, payload, not args.no_plots)
        _print(summary, output)
        if not summary["passed"]:
            failed.append(experiment)
    if failed:
        raise SystemExit("advanced level-2 acceptance gates failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
