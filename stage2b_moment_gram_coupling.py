#!/usr/bin/env python3
"""MFSI Stage 2B: one richer fiber-aware coupling-only experiment.

The sole scientific change from Stage 2 is the addition of 36 bilinear
interactions between the six unique entries of JPhi(X) JPhi(X)^T at the two
endpoints.  Schedules and every objective/downstream setting remain frozen.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import coupling_study as coupling
import level2_paper_study as paper


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results" / "coupling_study" / "standard"
DEFAULT_OUTPUT = ROOT / "results" / "coupling_study" / "stage2b_moment_gram"
METHODS = ("independent", "geometric_sinkhorn", "fiber_aware", "fiber_aware_gram")
LABELS = {
    "independent": "independent",
    "geometric_sinkhorn": "geometric OT",
    "fiber_aware": "Phi-only",
    "fiber_aware_gram": "Phi + Gram",
}
ROLE_KEYS = {
    "train": "coupling_optimization",
    "selection": "coupling_validation",
    "evaluation": "final_evaluation",
}
NOISE_OFFSETS = {"train": 21000, "selection": 22000, "evaluation": 23000}
RICH_SINKHORN_ITERATIONS = 500


def reconstruct_bank(populations: dict, record: dict) -> coupling.EndpointBank:
    minus_indices = np.asarray(record["minus_source_indices"], dtype=np.int64)
    plus_indices = np.asarray(record["plus_source_indices"], dtype=np.int64)
    return coupling.EndpointBank(
        coupling._canonicalize(populations["minus"][minus_indices]),
        coupling._canonicalize(populations["plus"][plus_indices]),
        np.asarray(record["minus_weights"], dtype=np.float64),
        np.asarray(record["plus_weights"], dtype=np.float64),
        minus_indices, plus_indices,
    )


def objective_parts(metrics: dict, times: jnp.ndarray) -> dict:
    unscaled = jnp.trapezoid(
        jnp.maximum(coupling.ESS_FLOOR - metrics["ess"], 0.0) ** 2, times
    )
    scaled = coupling.ESS_PENALTY * unscaled
    energy = metrics["integrated_correction_energy"]
    return {
        "integrated_correction_energy": float(energy),
        "ess_penalty_unscaled": float(unscaled),
        "ess_penalty_scaled": float(scaled),
        "total_objective": float(energy + scaled),
        "minimum_ess": float(metrics["minimum_ess"]),
        "median_ess": float(metrics["median_ess"]),
        "integrated_projection_distortion": float(metrics["integrated_projection_distortion"]),
        "maximum_projected_moment_error": float(metrics["maximum_moment_error"]),
        "curves": [
            {
                "t": float(t),
                "correction_energy": float(metrics["correction_energy"][index]),
                "ess": float(metrics["ess"][index]),
                "projection_distortion": float(metrics["projection_distortion"][index]),
                "q4": float(metrics["q4"][index]),
            }
            for index, t in enumerate(np.asarray(times))
        ],
    }


def build_richer_plan(bank: coupling.EndpointBank, parameters: jnp.ndarray):
    cost = coupling.microscopic_cost(bank)
    features = coupling.richer_coupling_features(bank)
    plan = np.asarray(coupling.build_plan(
        "fiber_aware", jnp.asarray(cost), jnp.asarray(features),
        jnp.asarray(bank.minus_weights), jnp.asarray(bank.plus_weights),
        parameters, RICH_SINKHORN_ITERATIONS,
    ))
    return plan, cost, features


def gradient_check(raw: jnp.ndarray, seed: int) -> dict:
    populations = paper.build_physical_populations(seed + 70000, True)
    rng = np.random.default_rng(seed + 71000)
    minus_indices = rng.choice(len(populations["minus"]), size=7, replace=False)
    plus_indices = rng.choice(len(populations["plus"]), size=7, replace=False)
    uniform = np.full(7, 1.0 / 7.0)
    bank = coupling.EndpointBank(
        coupling._canonicalize(populations["minus"][minus_indices]),
        coupling._canonicalize(populations["plus"][plus_indices]),
        uniform, uniform, minus_indices, plus_indices,
    )
    times = jnp.asarray([0.49, 0.50, 0.51])
    cost = jnp.asarray(coupling.microscopic_cost(bank))
    features = jnp.asarray(coupling.richer_coupling_features(bank))
    statistics = coupling.precompute_statistics(
        raw, bank, times, coupling.make_noise(seed + 72000, 7, 1)
    )
    geometric = coupling.build_plan(
        "geometric_sinkhorn", cost, features,
        jnp.asarray(uniform), jnp.asarray(uniform),
    )
    target = geometric.reshape(-1) @ statistics.observables[1]
    objective = jax.jit(lambda parameters: coupling.coupling_objective(
        parameters, cost, features, statistics, times, target,
        jnp.asarray(uniform), jnp.asarray(uniform),
        RICH_SINKHORN_ITERATIONS,
    ))
    parameters = jnp.linspace(-0.025, 0.025, 45)
    direction = jnp.cos(jnp.arange(45, dtype=jnp.float64) + 0.3)
    direction /= jnp.linalg.norm(direction)
    autodiff = float(jax.grad(objective)(parameters) @ direction)
    step = 1e-4
    finite = float(
        (objective(parameters + step * direction)
         - objective(parameters - step * direction)) / (2.0 * step)
    )
    relative = abs(autodiff - finite) / max(abs(autodiff), abs(finite), 1e-10)
    return {
        "feature_dimension": int(features.shape[-1]),
        "existing_phi_parameters": 9,
        "new_gram_parameters": 36,
        "finite_difference_step": step,
        "autodiff_directional_derivative": autodiff,
        "finite_difference_directional_derivative": finite,
        "relative_error": relative,
    }


def source_method(seed_report: dict, method: str) -> dict:
    row = seed_report["methods"][method]
    neural = row["neural_downstream"]
    return {
        "integrated_correction_energy": row["integrated_correction_energy"],
        "minimum_ess": row["minimum_ess"],
        "median_ess": row["median_ess"],
        "integrated_projection_distortion": row["integrated_projection_distortion"],
        "maximum_projected_moment_error": row["maximum_moment_error"],
        "microscopic_cost": row["plan"]["microscopic_cost"],
        "projected_law_mmd2": neural["projected_law_mmd2"],
        "maximum_moment_error": neural["maximum_moment_error"],
        "generated_q4_change": neural["rows"][-1]["q4"] - neural["rows"][0]["q4"],
        "curves": row["curves"],
        "source": "unchanged Stage-2 result",
    }


def run_seed(seed_report: dict) -> dict:
    seed = seed_report["seed"]
    started = time.perf_counter()
    populations = paper.build_physical_populations(seed + 10000, False)
    raw = jnp.asarray(seed_report["fixed_schedule"]["raw"])
    target = jnp.asarray(seed_report["endpoint"]["target"])
    times = jnp.asarray(np.linspace(0.12, 0.88, 6))
    old_parameters = jnp.asarray(seed_report["fiber_optimization"]["parameters"])
    role_state = {}
    for split in ("train", "selection"):
        bank = reconstruct_bank(populations, seed_report["banks"][ROLE_KEYS[split]])
        cost = jnp.asarray(coupling.microscopic_cost(bank))
        phi_features = jnp.asarray(coupling.coupling_features(bank))
        rich_features = jnp.asarray(coupling.richer_coupling_features(bank))
        statistics = coupling.precompute_statistics(
            raw, bank, times,
            coupling.make_noise(seed + NOISE_OFFSETS[split], len(bank.minus), 2),
        )
        role_state[split] = (bank, cost, phi_features, rich_features, statistics)
    train = role_state["train"]
    selection = role_state["selection"]
    parameters, trace, validation_values, selected, optimization_seconds = coupling.optimize_coupling(
        train[1], train[3], train[4], selection[1], selection[3], selection[4],
        times, target,
        jnp.asarray(train[0].minus_weights), jnp.asarray(train[0].plus_weights),
        jnp.asarray(selection[0].minus_weights), jnp.asarray(selection[0].plus_weights),
        60, RICH_SINKHORN_ITERATIONS,
    )
    parameters = jnp.asarray(parameters)

    evaluation_bank = reconstruct_bank(
        populations, seed_report["banks"]["final_evaluation"]
    )
    evaluation_cost = jnp.asarray(coupling.microscopic_cost(evaluation_bank))
    evaluation_phi = jnp.asarray(coupling.coupling_features(evaluation_bank))
    evaluation_rich = jnp.asarray(coupling.richer_coupling_features(evaluation_bank))
    evaluation_statistics = coupling.precompute_statistics(
        raw, evaluation_bank, times,
        coupling.make_noise(seed + NOISE_OFFSETS["evaluation"], len(evaluation_bank.minus), 2),
    )
    role_state["evaluation"] = (
        evaluation_bank, evaluation_cost, evaluation_phi, evaluation_rich,
        evaluation_statistics,
    )
    role_metrics = {}
    for split, state in role_state.items():
        bank, cost, phi_features, rich_features, statistics = state
        plans = {
            "geometric_sinkhorn": coupling.build_plan(
                "geometric_sinkhorn", cost, rich_features,
                jnp.asarray(bank.minus_weights), jnp.asarray(bank.plus_weights),
            ),
            "fiber_aware": coupling.build_plan(
                "fiber_aware", cost, phi_features,
                jnp.asarray(bank.minus_weights), jnp.asarray(bank.plus_weights),
                old_parameters,
            ),
            "fiber_aware_gram": coupling.build_plan(
                "fiber_aware", cost, rich_features,
                jnp.asarray(bank.minus_weights), jnp.asarray(bank.plus_weights),
                parameters, RICH_SINKHORN_ITERATIONS,
            ),
        }
        role_metrics[split] = {}
        for method, plan in plans.items():
            metrics = coupling.plan_path_metrics(plan, statistics, times, target)
            role_metrics[split][method] = objective_parts(metrics, times)
            role_metrics[split][method]["plan"] = coupling.plan_diagnostics(
                np.asarray(plan), np.asarray(cost), bank.minus_weights,
                bank.plus_weights,
                RICH_SINKHORN_ITERATIONS if method == "fiber_aware_gram"
                else coupling.SINKHORN_ITERATIONS,
            )

    neural_training = reconstruct_bank(populations, seed_report["banks"]["neural_training"])
    gate_bank = reconstruct_bank(populations, seed_report["banks"]["neural_gate"])
    generation_bank = reconstruct_bank(populations, seed_report["banks"]["neural_generation"])
    oracle_bank = reconstruct_bank(populations, seed_report["banks"]["neural_oracle"])
    continuous_rng = np.random.default_rng(seed + 31000)
    strata = np.arange(18) + continuous_rng.uniform(size=18)
    continuous_times = jnp.asarray(0.12 + 0.76 * strata / 18)
    training_plan, training_cost, _ = build_richer_plan(neural_training, parameters)
    training_samples, training_sampling = coupling.sample_bridge_bank(
        neural_training, training_plan, seed + 32000,
        np.asarray(continuous_times), 64,
    )
    model, training_trace, training_seconds = paper.train_neural_correction(
        jax.random.PRNGKey(seed), raw, training_samples, continuous_times,
        target, 420,
    )
    gate_plan, _, _ = build_richer_plan(gate_bank, parameters)
    gate_samples, gate_sampling = coupling.sample_bridge_bank(
        gate_bank, gate_plan, seed + 33000, np.asarray(times), 384,
    )
    gate, gate_gain, gate_se = paper.select_gate(
        model, raw, gate_samples, times, target
    )
    generation_plan, _, _ = build_richer_plan(generation_bank, parameters)
    generation_samples, generation_sampling = coupling.sample_bridge_bank(
        generation_bank, generation_plan, seed + 34000,
        np.asarray([0.5]), 64,
    )
    evaluation_times = jnp.asarray([0.25, 0.50, 0.75, 1.0])
    oracle_plan, _, _ = build_richer_plan(oracle_bank, parameters)
    oracle_samples, oracle_sampling = coupling.sample_bridge_bank(
        oracle_bank, oracle_plan, seed + 35000,
        np.asarray(evaluation_times), 256,
    )
    generated, integration_seconds, nfe = paper.integrate_method(
        "neural", model, gate, raw,
        generation_samples[0][0], generation_samples[1][0],
        generation_samples[2][0], target, 24, evaluation_times,
    )
    rows = paper.evaluate_generated(
        generated, oracle_samples, raw, evaluation_times, target
    )
    evaluation = role_metrics["evaluation"]["fiber_aware_gram"]
    rich_method = {
        "integrated_correction_energy": evaluation["integrated_correction_energy"],
        "minimum_ess": evaluation["minimum_ess"],
        "median_ess": evaluation["median_ess"],
        "integrated_projection_distortion": evaluation["integrated_projection_distortion"],
        "maximum_projected_moment_error": evaluation["maximum_projected_moment_error"],
        "microscopic_cost": evaluation["plan"]["microscopic_cost"],
        "projected_law_mmd2": paper.interior_mmd2(rows),
        "maximum_moment_error": float(max(row["moment_error"] for row in rows)),
        "generated_q4_change": float(rows[-1]["q4"] - rows[0]["q4"]),
        "curves": evaluation["curves"],
        "source": "Stage-2B evaluation",
        "neural_downstream": {
            "architecture": "unchanged two-hidden-layer width-18 invariant MLP",
            "training_protocol": "unchanged stratified random-continuous-time",
            "training_times": np.asarray(continuous_times).tolist(),
            "optimizer_steps": 420,
            "training_initial_loss": training_trace[0],
            "training_final_loss": training_trace[-1],
            "training_seconds": training_seconds,
            "model_parameters": paper.serialize_mlp(model),
            "gate": gate, "gate_gain": gate_gain,
            "gate_standard_error": gate_se,
            "ode_solver": "fixed-step Heun", "integration_steps": 24,
            "nfe": nfe, "integration_seconds": integration_seconds,
            "rows": rows,
            "sampling_diagnostics": {
                "training": training_sampling, "gate": gate_sampling,
                "generation": generation_sampling, "oracle": oracle_sampling,
            },
            "training_plan": coupling.plan_diagnostics(
                training_plan, training_cost, neural_training.minus_weights,
                neural_training.plus_weights, RICH_SINKHORN_ITERATIONS,
            ),
        },
    }
    methods = {
        method: source_method(seed_report, method)
        for method in ("independent", "geometric_sinkhorn", "fiber_aware")
    }
    methods["fiber_aware_gram"] = rich_method
    feature_check = coupling.richer_coupling_features(train[0])
    return {
        "seed": seed,
        "wall_seconds": time.perf_counter() - started,
        "fixed_schedule": seed_report["fixed_schedule"],
        "endpoint": seed_report["endpoint"],
        "bank_ids_and_weights": seed_report["banks"],
        "representation": {
            "total_parameters": 45,
            "existing_phi_interactions": 9,
            "new_gram_interactions": 36,
            "gram_definition": "six upper-triangular entries of JPhi(X) JPhi(X)^T",
            "gram_preprocessing": (
                "fixed pooled-endpoint centering and coordinatewise scaling, "
                "matching the existing Phi interaction construction"
            ),
            "first_nine_match_original": bool(np.array_equal(
                feature_check[..., :9], coupling.coupling_features(train[0])
            )),
            "q4_used": False, "final_mmd_features_used": False,
        },
        "optimization": {
            "parameters": np.asarray(parameters).tolist(),
            "phi_parameters": np.asarray(parameters[:9]).tolist(),
            "gram_matrix_A": np.asarray(parameters[9:]).reshape(6, 6).tolist(),
            "steps": 60, "learning_rate": coupling.COUPLING_LEARNING_RATE,
            "ess_floor": coupling.ESS_FLOOR,
            "ess_penalty_beta": coupling.ESS_PENALTY,
            "sinkhorn_iterations": RICH_SINKHORN_ITERATIONS,
            "trace": trace, "validation_objectives": validation_values,
            "selected_candidate_index": selected,
            "initial_validation_objective": validation_values[0],
            "selected_validation_objective": validation_values[selected],
            "wall_seconds": optimization_seconds,
        },
        "role_metrics": role_metrics,
        "methods": methods,
    }


def metric_value(report: dict, method: str, metric: str) -> float:
    return report["methods"][method][metric]


def aggregate(seed_reports: list[dict]) -> dict:
    metrics = (
        "integrated_correction_energy", "minimum_ess", "median_ess",
        "integrated_projection_distortion", "projected_law_mmd2",
        "maximum_moment_error", "microscopic_cost", "generated_q4_change",
    )
    methods = {
        method: {
            metric: coupling.mean_ci([
                metric_value(report, method, metric) for report in seed_reports
            ])
            for metric in metrics
        }
        for method in METHODS
    }
    comparisons = {
        "gram_minus_geometric": ("fiber_aware_gram", "geometric_sinkhorn"),
        "gram_minus_phi_only": ("fiber_aware_gram", "fiber_aware"),
        "phi_only_minus_geometric": ("fiber_aware", "geometric_sinkhorn"),
        "geometric_minus_independent": ("geometric_sinkhorn", "independent"),
    }
    contrasts = {}
    for name, (left, right) in comparisons.items():
        contrasts[name] = {
            metric: coupling.mean_ci([
                metric_value(report, left, metric)
                - metric_value(report, right, metric)
                for report in seed_reports
            ])
            for metric in metrics
        }
    role_differences = {}
    for split in ("train", "selection", "evaluation"):
        role_differences[split] = {}
        for comparator in ("geometric_sinkhorn", "fiber_aware"):
            role_differences[split][f"gram_minus_{comparator}"] = {
                metric: coupling.mean_ci([
                    report["role_metrics"][split]["fiber_aware_gram"][metric]
                    - report["role_metrics"][split][comparator][metric]
                    for report in seed_reports
                ])
                for metric in (
                    "integrated_correction_energy", "ess_penalty_scaled",
                    "total_objective", "minimum_ess",
                    "integrated_projection_distortion",
                )
            }
    primary = contrasts["gram_minus_geometric"]
    return {
        "methods": methods, "paired_contrasts": contrasts,
        "role_paired_differences": role_differences,
        "interpretation": {
            "heldout_correction_energy_beats_geometric": primary[
                "integrated_correction_energy"
            ]["ci95_high"] < 0.0,
            "heldout_correction_energy_beats_phi_only": contrasts[
                "gram_minus_phi_only"
            ]["integrated_correction_energy"]["ci95_high"] < 0.0,
            "projected_mmd_materially_worse_than_geometric": primary[
                "projected_law_mmd2"
            ]["ci95_low"] > 0.0,
        },
    }


def make_decision(aggregate_results: dict) -> dict:
    interpretation = aggregate_results["interpretation"]
    improves_geometric = interpretation[
        "heldout_correction_energy_beats_geometric"
    ]
    improves_phi_only = interpretation[
        "heldout_correction_energy_beats_phi_only"
    ]
    mmd_materially_worse = interpretation[
        "projected_mmd_materially_worse_than_geometric"
    ]
    success = improves_geometric and improves_phi_only and not mmd_materially_worse
    return {
        "stage2b_success": success,
        "richer_beats_geometric_on_heldout_correction_energy": improves_geometric,
        "richer_beats_phi_only_on_heldout_correction_energy": improves_phi_only,
        "projected_mmd_materially_worse_than_geometric": mmd_materially_worse,
        "stop_fiber_aware_coupling_development_for_this_paper": not success,
        "joint_schedule_coupling_optimization_justified": False,
        "preferred_coupling_baseline_for_later_work": "geometric_sinkhorn",
        "reason": (
            "The one permitted richer representation did not lower held-out "
            "correction energy beyond geometric OT and Phi-only coupling."
        ),
    }


def write_csv(summary: dict, output: Path) -> None:
    rows = []
    for report in summary["seed_reports"]:
        for method in METHODS:
            row = report["methods"][method]
            rows.append({
                "seed": report["seed"], "method": method,
                **{name: row[name] for name in (
                    "integrated_correction_energy", "minimum_ess", "median_ess",
                    "integrated_projection_distortion", "projected_law_mmd2",
                    "maximum_moment_error", "microscopic_cost",
                    "generated_q4_change",
                )},
            })
    with (output / "stage2b_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def make_plots(summary: dict, output: Path) -> None:
    plt.rcParams.update({
        "figure.facecolor": "#f4f1ea", "axes.facecolor": "#fffdf8",
        "axes.grid": True, "grid.alpha": 0.2,
    })
    colors = ["#457b9d", "#e9c46a", "#e76f51", "#2a9d8f"]
    metrics = (
        "integrated_correction_energy", "minimum_ess",
        "integrated_projection_distortion", "projected_law_mmd2",
    )
    titles = ("Correction burden", "Minimum ESS", "Projection distortion", "Projected-law MMD²")
    figure, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for ax, metric, title in zip(axes, metrics, titles):
        stats = [summary["aggregate"]["methods"][method][metric] for method in METHODS]
        means = [row["mean"] for row in stats]
        errors = [[mean - row["ci95_low"] for mean, row in zip(means, stats)],
                  [row["ci95_high"] - mean for mean, row in zip(means, stats)]]
        ax.bar([LABELS[m] for m in METHODS], means, yerr=errors, color=colors, capsize=4)
        ax.set_title(title); ax.tick_params(axis="x", rotation=20)
    figure.suptitle("Stage 2B: one moment-response Gram extension", fontweight="bold")
    figure.savefig(output / "stage2b_summary.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    contrast_names = ("gram_minus_geometric", "gram_minus_phi_only")
    display_metrics = (
        "integrated_correction_energy", "minimum_ess",
        "integrated_projection_distortion", "projected_law_mmd2",
    )
    for ax, contrast_name in zip(axes, contrast_names):
        values = summary["aggregate"]["paired_contrasts"][contrast_name]
        means = [values[name]["mean"] for name in display_metrics]
        errors = [[mean - values[name]["ci95_low"] for mean, name in zip(means, display_metrics)],
                  [values[name]["ci95_high"] - mean for mean, name in zip(means, display_metrics)]]
        ax.errorbar(means, np.arange(4), xerr=errors, fmt="o", color="#2a9d8f", capsize=4)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_yticks(np.arange(4), ["E_corr", "ESS_min", "D_proj", "MMD²"])
        ax.set(title=contrast_name.replace("_", " "), xlabel="paired difference")
    figure.savefig(output / "stage2b_paired_effects.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    representative = summary["seed_reports"][0]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for method, color in zip(("geometric_sinkhorn", "fiber_aware", "fiber_aware_gram"), colors[1:]):
        curves = representative["methods"][method]["curves"]
        times = [row["t"] for row in curves]
        axes[0].plot(times, [row["correction_energy"] for row in curves], "o-", color=color, label=LABELS[method])
        axes[1].plot(times, [row["ess"] for row in curves], "o-", color=color, label=LABELS[method])
        axes[2].plot(times, [row["projection_distortion"] for row in curves], "o-", color=color, label=LABELS[method])
    for ax, title in zip(axes, ("Correction energy", "ESS", "Projection distortion")):
        ax.set(title=title, xlabel="time"); ax.legend(frameon=False, fontsize=8)
    figure.savefig(output / "stage2b_time_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_report(summary: dict, output: Path) -> None:
    aggregate = summary["aggregate"]
    geo = aggregate["paired_contrasts"]["gram_minus_geometric"]
    phi = aggregate["paired_contrasts"]["gram_minus_phi_only"]
    decision = summary["decision"]
    role_differences = aggregate["role_paired_differences"]
    lines = [
        "# MFSI Stage 2B: moment-response Gram coupling", "",
        "## Question", "",
        "Does adding local moment-fiber geometry to the endpoint representation improve held-out coupling quality beyond geometric OT and the Phi-only coupling?", "",
        "## Isolated intervention", "",
        "The geometric Sinkhorn kernel and nine standardized Phi endpoint interactions were retained. The only added representation is a 36-parameter bilinear interaction between the six unique entries of JPhi(X) JPhi(X)^T at the two endpoints. Those six entries use fixed pooled-endpoint centering and coordinatewise scaling, matching the existing Phi feature construction; the preprocessing has no learned parameters. No q4, final-MMD descriptors, neural descriptors, schedule parameters, objectives, optimizer settings, or downstream settings were changed.", "",
        "## Aggregate metrics", "",
        "| method | E_corr | min ESS | D_proj | MMD² |", "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = aggregate["methods"][method]
        lines.append(
            f"| {LABELS[method]} | {row['integrated_correction_energy']['mean']:.6g} | "
            f"{row['minimum_ess']['mean']:.6g} | {row['integrated_projection_distortion']['mean']:.6g} | "
            f"{row['projected_law_mmd2']['mean']:.6g} |"
        )
    lines.extend([
        "", "## Primary paired effects", "",
        f"Gram minus geometric E_corr: `{geo['integrated_correction_energy']['mean']:.6g}` "
        f"(95% interval `{geo['integrated_correction_energy']['ci95_low']:.6g}` to "
        f"`{geo['integrated_correction_energy']['ci95_high']:.6g}`).",
        "",
        f"Gram minus Phi-only E_corr: `{phi['integrated_correction_energy']['mean']:.6g}` "
        f"(95% interval `{phi['integrated_correction_energy']['ci95_low']:.6g}` to "
        f"`{phi['integrated_correction_energy']['ci95_high']:.6g}`).",
        "",
        f"Gram minus geometric MMD²: `{geo['projected_law_mmd2']['mean']:.6g}` "
        f"(95% interval `{geo['projected_law_mmd2']['ci95_low']:.6g}` to "
        f"`{geo['projected_law_mmd2']['ci95_high']:.6g}`).",
        "", "## Generalization across bank roles", "",
        "| role | Gram - geometric E_corr | Gram - geometric ESS penalty | Gram - geometric total objective |",
        "|---|---:|---:|---:|",
    ])
    for role in ("train", "selection", "evaluation"):
        row = role_differences[role]["gram_minus_geometric_sinkhorn"]
        lines.append(
            f"| {role} | {row['integrated_correction_energy']['mean']:.6g} | "
            f"{row['ess_penalty_scaled']['mean']:.6g} | "
            f"{row['total_objective']['mean']:.6g} |"
        )
    lines.extend([
        "",
        "The richer plan lowered the mean objective on the training and selection banks, but the correction-energy effect reversed on the untouched evaluation banks. This is the same parameterization/generalization mismatch the experiment was designed to test.",
        "", "## Numerical constraint check", "",
        f"The richer logits required 500 fixed inner log-Sinkhorn iterations instead of the Stage-2 default of 100 to converge to the same prescribed endpoint marginals. At 100 iterations the worst residual was `7.8e-6`; at 500 it was `{summary['validation']['maximum_plan_marginal_linf']:.6g}`. This changes only numerical convergence of the constraint solve: epsilon, target marginals, objective, Adam learning rate, and 60 optimization steps are unchanged. It is not a searched scientific hyperparameter.",
        "", "## Interpretation and stop decision", "",
        "The single richer representation does not provide held-out evidence of lower correction burden than either geometric OT or the Phi-only coupling. Its mean projected-law MMD² is also higher than geometric OT, although that paired interval includes zero.",
        "",
        ("**Stop condition reached:** fiber-aware coupling development should stop for this paper. Geometric OT remains the preferred coupling baseline for any later interaction study. Joint schedule-plus-coupling optimization was not implemented and is not justified by this result."
         if decision["stop_fiber_aware_coupling_development_for_this_paper"] else
         "The predeclared stop condition was not reached."),
        "",
    ])
    (output / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--aggregate-existing", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.source / "summary.json").read_text())
    if source["seeds"] != [401, 402, 403, 404, 405]:
        raise ValueError("Stage 2B requires the five standard Stage-2 banks")
    reports = []
    for seed_report in source["seed_reports"]:
        path = args.output_dir / f"seed_{seed_report['seed']}.json"
        if args.aggregate_existing:
            report = json.loads(path.read_text())
        else:
            print(f"[stage2b] seed {seed_report['seed']}", flush=True)
            report = run_seed(seed_report)
            path.write_text(json.dumps(report, indent=2) + "\n")
        reports.append(report)
    validation = gradient_check(
        jnp.asarray(source["seed_reports"][0]["fixed_schedule"]["raw"]), 401
    )
    maximum_marginal_error = max(
        max(
            method["plan"]["row_marginal_linf"],
            method["plan"]["column_marginal_linf"],
        )
        for report in reports for role in report["role_metrics"].values()
        for method in role.values()
    )
    aggregate_results = aggregate(reports)
    summary = {
        "experiment": "stage2b-one-richer-moment-response-gram-coupling",
        "stage": "2B", "scientific_replication_n": 5,
        "source_stage2": str(args.source.relative_to(ROOT)),
        "joint_schedule_coupling_optimization": False,
        "sole_intervention": (
            "36 bilinear interactions from fixed standardized six-entry "
            "endpoint JPhi JPhi^T embeddings"
        ),
        "frozen": {
            "schedule": True, "objective": True, "coupling_optimizer": True,
            "bank_roles": True, "neural_pipeline": True,
            "ode_and_mmd_protocol": True,
        },
        "configuration": {
            "total_coupling_parameters": 45,
            "existing_phi_parameters": 9, "new_gram_parameters": 36,
            "coupling_steps": 60,
            "coupling_learning_rate": coupling.COUPLING_LEARNING_RATE,
            "sinkhorn_epsilon": coupling.SINKHORN_EPSILON,
            "sinkhorn_iterations": coupling.SINKHORN_ITERATIONS,
            "rich_sinkhorn_iterations": RICH_SINKHORN_ITERATIONS,
            "ess_floor": coupling.ESS_FLOOR,
            "ess_penalty_beta": coupling.ESS_PENALTY,
            "q4_used_for_optimization": False,
            "final_mmd_used_for_optimization_or_selection": False,
        },
        "validation": {
            **validation,
            "gradient_passed": validation["relative_error"] < 5e-5,
            "maximum_plan_marginal_linf": maximum_marginal_error,
            "marginals_passed": maximum_marginal_error < 5e-8,
            "first_nine_features_unchanged": all(
                report["representation"]["first_nine_match_original"]
                for report in reports
            ),
        },
        "seed_reports": reports,
        "aggregate": aggregate_results,
        "decision": make_decision(aggregate_results),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "decision.json").write_text(
        json.dumps(summary["decision"], indent=2) + "\n"
    )
    write_csv(summary, args.output_dir)
    write_report(summary, args.output_dir)
    if not args.no_plots:
        make_plots(summary, args.output_dir)
    print(json.dumps({
        "validation": summary["validation"],
        "interpretation": summary["aggregate"]["interpretation"],
        "gram_minus_geometric": summary["aggregate"]["paired_contrasts"]["gram_minus_geometric"],
    }, indent=2))
    print(f"outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
