#!/usr/bin/env python3
"""MFSI Stage 3: rollout-aware adaptation of a frozen neural correction.

The sole learned object is a bounded three-parameter smooth time modulation of
the completed random-continuous-time neural correction.  Neural weights,
schedule, endpoint construction, Heun solver, and final evaluation are frozen.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import level2_paper_study as paper


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results" / "level2_paper_study" / "jax" / "summary.json"
DEFAULT_OUTPUT = ROOT / "results" / "stage3_rollout_adaptation"
SEEDS = [401, 402, 403, 404, 405]
EVALUATION_TIMES = jnp.asarray([0.25, 0.50, 0.75, 1.0])
HEUN_STEPS = 24
GENERATION_COUNT = 64
ORACLE_COUNT = 256
OPTIMIZER_STEPS = 40
LEARNING_RATE = 0.04
CANDIDATE_INTERVAL = 5
METHODS = ("raw", "tangent", "frozen_neural", "rollout_adapted")
LABELS = {
    "raw": "raw SI",
    "tangent": "tangent",
    "frozen_neural": "frozen neural",
    "rollout_adapted": "rollout-adapted",
}


def deserialize_mlp(record: dict) -> paper.MLP:
    return paper.MLP(*(jnp.asarray(record[name]) for name in paper.MLP._fields))


def array_hash(*arrays) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def model_hash(model: paper.MLP) -> str:
    return array_hash(*(getattr(model, name) for name in paper.MLP._fields))


def modulation(alpha: jax.Array, t: jax.Array, frozen_gate: float) -> jax.Array:
    """One smooth harmonic with a learned offset, initialized at frozen_gate."""
    gate = jnp.clip(jnp.asarray(frozen_gate), 1e-6, 1.0 - 1e-6)
    baseline_logit = jnp.log(gate) - jnp.log1p(-gate)
    basis = jnp.asarray([
        1.0,
        jnp.cos(2.0 * jnp.pi * t),
        jnp.sin(2.0 * jnp.pi * t),
    ])
    return jax.nn.sigmoid(baseline_logit + alpha @ basis)


def differentiable_rollout(alpha: jax.Array, model: paper.MLP,
                           frozen_gate: float, raw: jax.Array,
                           minus: jax.Array, plus: jax.Array,
                           noise: jax.Array) -> jax.Array:
    """The established 24-step Heun rollout expressed as a JAX scan."""
    dt = jnp.asarray(1.0 / HEUN_STEPS, dtype=minus.dtype)

    def velocity(state, t):
        gamma_derivative = jax.grad(
            lambda value: paper.gamma_schedule(raw, value)
        )(t)
        reference = plus - minus + gamma_derivative * noise
        correction = paper.v_correction(model, t, state)
        return reference + modulation(alpha, t, frozen_gate) * correction

    def step(state, index):
        t = index.astype(minus.dtype) * dt
        first = velocity(state, t)
        proposal = state + dt * first
        second = velocity(proposal, t + dt)
        updated = state + 0.5 * dt * (first + second)
        return updated, updated

    _, states = jax.lax.scan(step, minus, jnp.arange(HEUN_STEPS))
    snapshot_indices = jnp.asarray([5, 11, 17, 23])
    return states[snapshot_indices]


def differentiable_weighted_mmd(features_a, weights_a,
                                features_b, weights_b):
    """Exact static-shape form of the established median-RBF weighted MMD."""
    combined = jnp.concatenate([features_a, features_b], axis=0)
    distance = jnp.sum(
        (combined[:, None, :] - combined[None, :, :]) ** 2, axis=-1
    )
    rows, cols = jnp.triu_indices(combined.shape[0], 1)
    bandwidth = jnp.maximum(jnp.median(distance[rows, cols]), 1e-6)

    def kernel(x, y):
        squared = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
        return jnp.exp(-squared / (2.0 * bandwidth))

    value = (
        weights_a @ kernel(features_a, features_a) @ weights_a
        + weights_b @ kernel(features_b, features_b) @ weights_b
        - 2.0 * weights_a @ kernel(features_a, features_b) @ weights_b
    )
    return jnp.maximum(value, 0.0)


def oracle_projection(raw, bank, target):
    states, weights = paper.path_metrics(
        raw, bank, EVALUATION_TIMES, target
    )[:2]
    return (
        jax.lax.stop_gradient(jax.vmap(paper.v_observables)(states)),
        jax.lax.stop_gradient(weights),
    )


def projected_rollout_loss(alpha, model, gate, raw, generation, oracle):
    minus, plus, noise = (value[0] for value in generation)
    generated = differentiable_rollout(
        alpha, model, gate, raw, minus, plus, noise
    )
    generated_features = jax.vmap(paper.v_observables)(generated)
    oracle_features, oracle_weights = oracle
    uniform = jnp.full(generated.shape[1], 1.0 / generated.shape[1])
    per_time = jax.vmap(
        differentiable_weighted_mmd, in_axes=(0, None, 0, 0)
    )(generated_features, uniform, oracle_features, oracle_weights)
    # Match the established interior-MMD convention: the t=1 endpoint is
    # reported but is not part of the path objective.
    return jnp.trapezoid(per_time[:-1], EVALUATION_TIMES[:-1])


def make_rollout_role(populations, seed, offset):
    rng = np.random.default_rng(seed + offset)
    generation = paper.make_bridge_bank(
        populations, rng, np.asarray([0.5]), GENERATION_COUNT
    )
    oracle = paper.make_bridge_bank(
        populations, rng, np.asarray(EVALUATION_TIMES), ORACLE_COUNT
    )
    fingerprint = array_hash(*generation, *oracle)
    return generation, oracle, fingerprint


def optimize_modulation(model, gate, raw, target, adaptation, selection):
    adaptation_oracle = oracle_projection(raw, adaptation[1], target)
    selection_oracle = oracle_projection(raw, selection[1], target)
    train_objective = jax.jit(lambda alpha: projected_rollout_loss(
        alpha, model, gate, raw, adaptation[0], adaptation_oracle
    ))
    selection_objective = jax.jit(lambda alpha: projected_rollout_loss(
        alpha, model, gate, raw, selection[0], selection_oracle
    ))
    value_gradient = jax.jit(jax.value_and_grad(train_objective))
    alpha = jnp.zeros(3, dtype=jnp.float64)
    first = jnp.zeros_like(alpha)
    second = jnp.zeros_like(alpha)
    candidates = [np.asarray(alpha)]
    candidate_steps = [0]
    trace = []
    started = time.perf_counter()
    for iteration in range(1, OPTIMIZER_STEPS + 1):
        value, gradient = value_gradient(alpha)
        gradient_norm = jnp.linalg.norm(gradient)
        gradient = gradient * jnp.minimum(
            1.0, 5.0 / jnp.maximum(gradient_norm, 1e-12)
        )
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient * gradient
        first_hat = first / (1.0 - 0.9**iteration)
        second_hat = second / (1.0 - 0.999**iteration)
        alpha = alpha - LEARNING_RATE * first_hat / (jnp.sqrt(second_hat) + 1e-8)
        trace.append({
            "step": iteration,
            "adaptation_loss": float(value),
            "gradient_norm": float(gradient_norm),
            "alpha": np.asarray(alpha).tolist(),
        })
        if iteration % CANDIDATE_INTERVAL == 0:
            candidates.append(np.asarray(alpha))
            candidate_steps.append(iteration)
    selection_losses = [
        float(selection_objective(jnp.asarray(candidate)))
        for candidate in candidates
    ]
    selected_index = int(np.argmin(selection_losses))
    selected = jnp.asarray(candidates[selected_index])
    jax.block_until_ready(selected)
    return {
        "alpha": selected,
        "trace": trace,
        "candidate_steps": candidate_steps,
        "selection_losses": selection_losses,
        "selected_candidate_index": selected_index,
        "selected_step": candidate_steps[selected_index],
        "initial_adaptation_loss": float(train_objective(jnp.zeros(3))),
        "selected_adaptation_loss": float(train_objective(selected)),
        "initial_selection_loss": selection_losses[0],
        "selected_selection_loss": selection_losses[selected_index],
        "wall_seconds": time.perf_counter() - started,
    }


def evaluate_method(method, model, gate, alpha, raw, generation, oracle_bank,
                    target):
    minus, plus, noise = (value[0] for value in generation)
    started = time.perf_counter()
    if method == "rollout_adapted":
        generated = np.asarray(differentiable_rollout(
            alpha, model, gate, raw, minus, plus, noise
        ))
        nfe = 2 * HEUN_STEPS
        wall = time.perf_counter() - started
    elif method == "frozen_neural":
        generated = np.asarray(differentiable_rollout(
            jnp.zeros(3), model, gate, raw, minus, plus, noise
        ))
        nfe = 2 * HEUN_STEPS
        wall = time.perf_counter() - started
    else:
        generated, wall, nfe = paper.integrate_method(
            method, model, gate, raw, minus, plus, noise, target,
            HEUN_STEPS, EVALUATION_TIMES,
        )
    rows = paper.evaluate_generated(
        generated, oracle_bank, raw, EVALUATION_TIMES, target
    )
    projected_oracle = oracle_projection(raw, oracle_bank, target)
    projected_generated = jax.vmap(paper.v_observables)(jnp.asarray(generated))
    uniform = jnp.full(generated.shape[1], 1.0 / generated.shape[1])
    projected_mmd = jax.vmap(
        differentiable_weighted_mmd, in_axes=(0, None, 0, 0)
    )(projected_generated, uniform, *projected_oracle)
    return {
        "interior_mmd2": paper.interior_mmd2(rows),
        "integrated_mmd2": float(np.trapezoid(
            [row["mmd2"] for row in rows], np.asarray(EVALUATION_TIMES)
        )),
        "endpoint_mmd2": rows[-1]["mmd2"],
        "maximum_moment_error": max(row["moment_error"] for row in rows),
        "interior_phi_mmd2": float(jnp.trapezoid(
            projected_mmd[:-1], EVALUATION_TIMES[:-1]
        )),
        "q4_change": rows[-1]["q4"] - rows[0]["q4"],
        "wall_seconds": wall,
        "nfe": int(nfe),
        "rows": rows,
    }


def gradient_check(model, gate, raw, target, role):
    oracle = oracle_projection(raw, role[1], target)
    objective = jax.jit(lambda alpha: projected_rollout_loss(
        alpha, model, gate, raw, role[0], oracle
    ))
    alpha = jnp.asarray([0.03, -0.02, 0.01])
    direction = jnp.asarray([0.4, -0.7, 0.5])
    direction /= jnp.linalg.norm(direction)
    autodiff = float(jax.grad(objective)(alpha) @ direction)
    step = 1e-4
    finite = float(
        (objective(alpha + step * direction)
         - objective(alpha - step * direction)) / (2.0 * step)
    )
    relative = abs(autodiff - finite) / max(abs(autodiff), abs(finite), 1e-10)
    return {
        "autodiff_directional_derivative": autodiff,
        "finite_difference_directional_derivative": finite,
        "finite_difference_step": step,
        "relative_error": relative,
        "passed": relative < 5e-4,
    }


def mmd_parity_check():
    rng = np.random.default_rng(73191)
    first = jnp.asarray(rng.normal(size=(11, 3)))
    second = jnp.asarray(rng.normal(size=(13, 3)))
    first_weights = jnp.asarray(rng.dirichlet(np.ones(11)))
    second_weights = jnp.asarray(rng.dirichlet(np.ones(13)))
    established = paper.weighted_mmd(
        first, first_weights, second, second_weights
    )
    differentiable = differentiable_weighted_mmd(
        first, first_weights, second, second_weights
    )
    error = float(abs(established - differentiable))
    return {"absolute_error": error, "passed": error < 1e-12}


def run_seed(source_report):
    seed = source_report["seed"]
    continuous = source_report["rollout_diagnostics"]["continuous_time_training"]
    model = deserialize_mlp(continuous["model_parameters"])
    frozen_model_hash = model_hash(model)
    gate = float(continuous["gate"])
    raw = jnp.asarray(source_report["schedules"]["optimized_multi"]["raw"])
    populations = paper.build_physical_populations(seed + 10000, False)
    target = jnp.asarray(populations["target"])
    adaptation = make_rollout_role(populations, seed, 81000)
    selection = make_rollout_role(populations, seed, 82000)
    evaluation = make_rollout_role(populations, seed, 83000)
    roles = {"adaptation": adaptation, "selection": selection, "evaluation": evaluation}
    if len({role[2] for role in roles.values()}) != 3:
        raise RuntimeError("rollout bank roles are not distinct")
    print(f"[stage3] seed {seed}: differentiable rollout optimization", flush=True)
    optimization = optimize_modulation(
        model, gate, raw, target, adaptation[:2], selection[:2]
    )
    alpha = optimization.pop("alpha")
    methods = {
        method: evaluate_method(
            method, model, gate, alpha, raw, evaluation[0], evaluation[1], target
        )
        for method in METHODS
    }
    # Compare the functional rollout with the established Python Heun loop at
    # the exact frozen scalar gate on the untouched bank.
    minus, plus, noise = (value[0] for value in evaluation[0])
    established, _, _ = paper.integrate_method(
        "neural", model, gate, raw, minus, plus, noise, target,
        HEUN_STEPS, EVALUATION_TIMES,
    )
    functional = differentiable_rollout(
        jnp.zeros(3), model, gate, raw, minus, plus, noise
    )
    rollout_parity = float(np.max(np.abs(established - np.asarray(functional))))
    amplitude_times = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0])
    amplitudes = [
        float(modulation(alpha, jnp.asarray(t), gate)) for t in amplitude_times
    ]
    return {
        "seed": seed,
        "source": "completed random-continuous-time paper-facing model",
        "frozen_schedule_raw": np.asarray(raw).tolist(),
        "frozen_gate": gate,
        "frozen_model_hash_before": frozen_model_hash,
        "frozen_model_hash_after": model_hash(model),
        "bank_fingerprints": {name: role[2] for name, role in roles.items()},
        "optimization": {
            **optimization,
            "selected_alpha": np.asarray(alpha).tolist(),
            "amplitude_times": amplitude_times.tolist(),
            "selected_amplitudes": amplitudes,
            "optimizer": "Adam",
            "steps": OPTIMIZER_STEPS,
            "learning_rate": LEARNING_RATE,
            "candidate_interval": CANDIDATE_INTERVAL,
        },
        "validation": {
            "functional_heun_max_abs_error": rollout_parity,
            "functional_heun_parity_passed": rollout_parity < 2e-10,
        },
        "methods": methods,
    }


def mean_ci(values):
    return paper.mean_ci([float(value) for value in values])


def aggregate(reports):
    metric_names = (
        "interior_mmd2", "integrated_mmd2", "endpoint_mmd2",
        "maximum_moment_error", "interior_phi_mmd2", "q4_change",
    )
    methods = {
        method: {
            metric: mean_ci([report["methods"][method][metric] for report in reports])
            for metric in metric_names
        }
        for method in METHODS
    }
    comparisons = {
        "adapted_minus_frozen": ("rollout_adapted", "frozen_neural"),
        "adapted_minus_tangent": ("rollout_adapted", "tangent"),
        "adapted_minus_raw": ("rollout_adapted", "raw"),
        "frozen_minus_tangent": ("frozen_neural", "tangent"),
    }
    contrasts = {
        name: {
            metric: mean_ci([
                report["methods"][left][metric]
                - report["methods"][right][metric]
                for report in reports
            ])
            for metric in metric_names
        }
        for name, (left, right) in comparisons.items()
    }
    adaptation = {
        metric: mean_ci([
            report["optimization"][metric] for report in reports
        ])
        for metric in (
            "initial_adaptation_loss", "selected_adaptation_loss",
            "initial_selection_loss", "selected_selection_loss",
            "selected_step",
        )
    }
    primary = contrasts["adapted_minus_frozen"]["interior_mmd2"]
    return {
        "methods": methods,
        "paired_contrasts": contrasts,
        "optimization": adaptation,
        "interpretation": {
            "adapted_mean_improves_frozen": primary["mean"] < 0.0,
            "adapted_ci_supports_improvement_over_frozen": primary["ci95_high"] < 0.0,
            "adapted_ci_supports_improvement_over_tangent": contrasts[
                "adapted_minus_tangent"
            ]["interior_mmd2"]["ci95_high"] < 0.0,
        },
    }


def write_csv(summary, output):
    rows = []
    for report in summary["seed_reports"]:
        for method in METHODS:
            metrics = report["methods"][method]
            rows.append({
                "seed": report["seed"], "method": method,
                **{name: metrics[name] for name in (
                    "interior_mmd2", "integrated_mmd2", "endpoint_mmd2",
                    "maximum_moment_error", "interior_phi_mmd2", "q4_change",
                )},
            })
    with (output / "stage3_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(summary, output):
    plt.rcParams.update({
        "figure.facecolor": "#f4f1ea", "axes.facecolor": "#fffdf8",
        "axes.grid": True, "grid.alpha": 0.2,
    })
    colors = ["#457b9d", "#e9c46a", "#e76f51", "#2a9d8f"]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        ("interior_mmd2", "maximum_moment_error", "interior_phi_mmd2"),
        ("Final law MMD²", "Maximum moment error", "Measured-Φ MMD²"),
    ):
        stats = [summary["aggregate"]["methods"][method][metric] for method in METHODS]
        means = [row["mean"] for row in stats]
        errors = [[mean - row["ci95_low"] for mean, row in zip(means, stats)],
                  [row["ci95_high"] - mean for mean, row in zip(means, stats)]]
        ax.bar([LABELS[method] for method in METHODS], means, yerr=errors,
               color=colors, capsize=4)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    figure.suptitle("Stage 3: frozen-field rollout adaptation", fontweight="bold")
    figure.savefig(output / "stage3_summary.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for report, color in zip(summary["seed_reports"], colors + ["#6d597a"]):
        axes[0].plot(
            report["optimization"]["amplitude_times"],
            report["optimization"]["selected_amplitudes"], "o-",
            label=str(report["seed"]), color=color,
        )
        axes[1].plot(
            report["optimization"]["candidate_steps"],
            report["optimization"]["selection_losses"], "o-",
            label=str(report["seed"]), color=color,
        )
    axes[0].set(title="Selected time modulation", xlabel="time", ylabel="a(t)")
    axes[1].set(title="Selection-bank objective", xlabel="optimizer step", ylabel="Φ-MMD²")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    figure.savefig(output / "stage3_optimization.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for method, color in zip(METHODS, colors):
        values = [report["methods"][method]["rows"] for report in summary["seed_reports"]]
        mmd = np.mean([[row["mmd2"] for row in rows] for rows in values], axis=0)
        q4 = np.mean([[row["q4"] for row in rows] for rows in values], axis=0)
        axes[0].plot(np.asarray(EVALUATION_TIMES), mmd, "o-", label=LABELS[method], color=color)
        axes[1].plot(np.asarray(EVALUATION_TIMES), q4, "o-", label=LABELS[method], color=color)
    axes[0].set(title="Held-out law discrepancy", xlabel="time", ylabel="MMD²")
    axes[1].set(title="Held-out q4 trajectory", xlabel="time", ylabel="q4")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    figure.savefig(output / "stage3_paths.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_report(summary, output):
    aggregate_results = summary["aggregate"]
    primary = aggregate_results["paired_contrasts"]["adapted_minus_frozen"]["interior_mmd2"]
    tangent = aggregate_results["paired_contrasts"]["adapted_minus_tangent"]["interior_mmd2"]
    lines = [
        "# MFSI Stage 3: rollout-aware differentiable correction adaptation", "",
        "## Isolated intervention", "",
        "The completed random-continuous-time invariant MLP, its weights, the selected schedule, endpoint construction, reference velocity, 24-step Heun solver, evaluation times, and final radial-plus-q4 MMD were frozen. The only learned object was a three-parameter bounded harmonic modulation of the frozen conservative correction. Alpha=0 exactly reproduces the frozen scalar Ritz gate.", "",
        "Adaptation used interior MMD² on the three measured Phi observables, with the established weighted median-RBF kernel. q4 and the final radial-plus-q4 law metric were untouched until evaluation. Forty Adam steps at learning rate 0.04 were run on an adaptation bank; candidates at steps 0,5,...,40 were selected once on an independent selection bank and evaluated on a third untouched bank.", "",
        "## Held-out evaluation", "",
        "| method | interior law MMD² | max moment error | interior Phi MMD² | q4 change |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = aggregate_results["methods"][method]
        lines.append(
            f"| {LABELS[method]} | {row['interior_mmd2']['mean']:.8g} | "
            f"{row['maximum_moment_error']['mean']:.8g} | "
            f"{row['interior_phi_mmd2']['mean']:.8g} | "
            f"{row['q4_change']['mean']:.8g} |"
        )
    lines.extend([
        "", "## Primary paired effects", "",
        f"Adapted minus frozen interior law MMD²: `{primary['mean']:.8g}` "
        f"(95% interval `{primary['ci95_low']:.8g}` to `{primary['ci95_high']:.8g}`).", "",
        f"Adapted minus tangent interior law MMD²: `{tangent['mean']:.8g}` "
        f"(95% interval `{tangent['ci95_low']:.8g}` to `{tangent['ci95_high']:.8g}`).", "",
        "## Validation", "",
        f"Directional gradient relative error: `{summary['validation']['gradient']['relative_error']:.3e}`. Functional-Heun maximum parity error: `{summary['validation']['maximum_functional_heun_error']:.3e}`. MMD static-shape parity error: `{summary['validation']['mmd_parity']['absolute_error']:.3e}`. Neural parameter hashes were unchanged and all bank-role fingerprints were distinct.", "",
        "## Interpretation", "",
        ("The paired interval supports a held-out improvement over the frozen neural correction."
         if aggregate_results["interpretation"]["adapted_ci_supports_improvement_over_frozen"]
         else "The experiment does not establish a held-out improvement over the frozen neural correction."),
        "",
        ("It also establishes an improvement over tangent."
         if aggregate_results["interpretation"]["adapted_ci_supports_improvement_over_tangent"]
         else "It does not establish an improvement over tangent."), "",
    ])
    (output / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--aggregate-existing", action="store_true")
    parser.add_argument("--seeds", default="401 402 403 404 405")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.source.read_text())
    previous_elapsed = None
    previous_summary_path = args.output_dir / "summary.json"
    if args.aggregate_existing and previous_summary_path.exists():
        previous_elapsed = json.loads(previous_summary_path.read_text()).get(
            "elapsed_seconds"
        )
    source_by_seed = {report["seed"]: report for report in source["seed_reports"]}
    reports = []
    started = time.perf_counter()
    for seed in seeds:
        path = args.output_dir / f"seed_{seed}.json"
        if args.aggregate_existing:
            report = json.loads(path.read_text())
        else:
            report = run_seed(source_by_seed[seed])
            path.write_text(json.dumps(report, indent=2) + "\n")
        reports.append(report)
    first_source = source_by_seed[seeds[0]]
    first_continuous = first_source["rollout_diagnostics"]["continuous_time_training"]
    first_model = deserialize_mlp(first_continuous["model_parameters"])
    first_raw = jnp.asarray(first_source["schedules"]["optimized_multi"]["raw"])
    first_populations = paper.build_physical_populations(seeds[0] + 10000, False)
    gradient_role = make_rollout_role(first_populations, seeds[0], 84000)
    gradient = gradient_check(
        first_model, float(first_continuous["gate"]), first_raw,
        jnp.asarray(first_populations["target"]),
        gradient_role[:2],
    )
    aggregate_results = aggregate(reports)
    validation = {
        "gradient": gradient,
        "mmd_parity": mmd_parity_check(),
        "maximum_functional_heun_error": max(
            report["validation"]["functional_heun_max_abs_error"] for report in reports
        ),
        "functional_heun_parity_passed": all(
            report["validation"]["functional_heun_parity_passed"] for report in reports
        ),
        "neural_parameters_frozen": all(
            report["frozen_model_hash_before"] == report["frozen_model_hash_after"]
            for report in reports
        ),
        "bank_roles_distinct": all(
            len(set(report["bank_fingerprints"].values())) == 3 for report in reports
        ),
        "q4_used_for_adaptation_or_selection": False,
        "final_evaluation_used_for_adaptation_or_selection": False,
    }
    summary = {
        "experiment": "stage3-rollout-aware-differentiable-correction-adaptation",
        "stage": 3,
        "scientific_replication_n": len(reports),
        "seeds": seeds,
        "source": str(args.source.relative_to(ROOT)),
        "sole_trainable_object": "three coefficients of smooth bounded time modulation",
        "modulation": "sigmoid(logit(frozen_gate) + alpha0 + alpha1*cos(2*pi*t) + alpha2*sin(2*pi*t))",
        "configuration": {
            "new_trainable_parameters": 3,
            "optimizer": "Adam", "optimizer_steps": OPTIMIZER_STEPS,
            "learning_rate": LEARNING_RATE,
            "candidate_steps": list(range(0, OPTIMIZER_STEPS + 1, CANDIDATE_INTERVAL)),
            "heun_steps": HEUN_STEPS, "nfe": 2 * HEUN_STEPS,
            "evaluation_times": np.asarray(EVALUATION_TIMES).tolist(),
            "generation_count_per_role": GENERATION_COUNT,
            "oracle_count_per_time_per_role": ORACLE_COUNT,
            "bank_seed_offsets": {
                "adaptation": 81000, "selection": 82000,
                "evaluation": 83000, "gradient_validation": 84000,
            },
            "adaptation_loss": "interior weighted median-RBF MMD2 on measured Phi only",
            "selection_loss": "same objective on independent rollout and oracle banks",
            "final_metric": "unchanged radial-descriptor-plus-q4 projected-law MMD2",
        },
        "frozen": {
            "endpoint_populations_and_calibration": True,
            "target_and_observables": True,
            "endpoint_pairing": True,
            "schedule_and_reference_velocity": True,
            "neural_architecture_and_weights": True,
            "random_continuous_time_ritz_training": True,
            "heun_solver_steps_and_evaluation_times": True,
            "final_mmd_and_q4_definitions": True,
        },
        "elapsed_seconds": (
            previous_elapsed if previous_elapsed is not None
            else time.perf_counter() - started
        ),
        "last_aggregation_seconds": (
            time.perf_counter() - started if args.aggregate_existing else None
        ),
        "validation": validation,
        "seed_reports": reports,
        "aggregate": aggregate_results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(summary, args.output_dir)
    write_report(summary, args.output_dir)
    if not args.no_plots:
        make_plots(summary, args.output_dir)
    print(json.dumps({
        "validation": validation,
        "interpretation": aggregate_results["interpretation"],
        "adapted_minus_frozen": aggregate_results["paired_contrasts"]["adapted_minus_frozen"],
    }, indent=2))
    print(f"outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
