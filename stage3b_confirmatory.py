#!/usr/bin/env python3
"""Stage 3B: confirmatory rollout adaptation with two predeclared controls."""
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

import level2_paper_study as paper
import stage3_rollout_adaptation as stage3


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "stage3b_protocol.json"
SOURCE = ROOT / "results" / "stage3b_base_models" / "summary.json"
ORIGINAL_STAGE3 = ROOT / "results" / "stage3_rollout_adaptation" / "summary.json"
DEFAULT_OUTPUT = ROOT / "results" / "stage3b_confirmatory"
SEEDS = list(range(406, 416))
METHODS = (
    "raw", "tangent", "frozen_neural", "scalar_adapted",
    "stopped_state_adapted", "rollout_adapted",
)
LABELS = {
    "raw": "raw SI", "tangent": "tangent",
    "frozen_neural": "frozen neural",
    "scalar_adapted": "scalar adapted",
    "stopped_state_adapted": "stopped-state",
    "rollout_adapted": "full rollout",
}
CONTROL_DIMENSIONS = {
    "scalar": 1, "stopped_state": 3, "full": 3,
}
OFFSETS = {
    "adaptation": 91000, "selection": 92000, "evaluation": 93000,
}


def controlled_rollout(parameters, control, model, gate, raw,
                       minus, plus, noise):
    if control == "scalar":
        alpha = jnp.concatenate([parameters, jnp.zeros(2, dtype=parameters.dtype)])
        return stage3.differentiable_rollout(
            alpha, model, gate, raw, minus, plus, noise
        )
    if control == "full":
        return stage3.differentiable_rollout(
            parameters, model, gate, raw, minus, plus, noise
        )
    if control != "stopped_state":
        raise ValueError(f"unknown control: {control}")

    dt = jnp.asarray(1.0 / stage3.HEUN_STEPS, dtype=minus.dtype)

    def velocity(state, t):
        gamma_derivative = jax.grad(
            lambda value: paper.gamma_schedule(raw, value)
        )(t)
        reference = plus - minus + gamma_derivative * noise
        correction = paper.v_correction(model, t, state)
        return reference + stage3.modulation(parameters, t, gate) * correction

    def step(state, index):
        # Same primal Heun trajectory as the full method. Stopping the incoming
        # state and proposal removes accumulated state-to-state credit while
        # retaining the direct effect of a(t) in this step's two field calls.
        incoming = jax.lax.stop_gradient(state)
        t = index.astype(minus.dtype) * dt
        first = velocity(incoming, t)
        proposal = incoming + dt * first
        second = velocity(jax.lax.stop_gradient(proposal), t + dt)
        updated = incoming + 0.5 * dt * (first + second)
        return updated, updated

    _, states = jax.lax.scan(step, minus, jnp.arange(stage3.HEUN_STEPS))
    return states[jnp.asarray([5, 11, 17, 23])]


def rollout_loss(parameters, control, model, gate, raw, generation, oracle):
    minus, plus, noise = (value[0] for value in generation)
    generated = controlled_rollout(
        parameters, control, model, gate, raw, minus, plus, noise
    )
    generated_features = jax.vmap(paper.v_observables)(generated)
    oracle_features, oracle_weights = oracle
    uniform = jnp.full(generated.shape[1], 1.0 / generated.shape[1])
    per_time = jax.vmap(
        stage3.differentiable_weighted_mmd, in_axes=(0, None, 0, 0)
    )(generated_features, uniform, oracle_features, oracle_weights)
    return jnp.trapezoid(
        per_time[:-1], stage3.EVALUATION_TIMES[:-1]
    )


def optimize_control(control, model, gate, raw, target, adaptation, selection):
    adaptation_oracle = stage3.oracle_projection(raw, adaptation[1], target)
    selection_oracle = stage3.oracle_projection(raw, selection[1], target)
    train_objective = jax.jit(lambda parameters: rollout_loss(
        parameters, control, model, gate, raw, adaptation[0], adaptation_oracle
    ))
    selection_objective = jax.jit(lambda parameters: rollout_loss(
        parameters, control, model, gate, raw, selection[0], selection_oracle
    ))
    value_gradient = jax.jit(jax.value_and_grad(train_objective))
    parameters = jnp.zeros(CONTROL_DIMENSIONS[control], dtype=jnp.float64)
    first = jnp.zeros_like(parameters)
    second = jnp.zeros_like(parameters)
    candidates = [np.asarray(parameters)]
    candidate_steps = [0]
    trace = []
    started = time.perf_counter()
    for iteration in range(1, stage3.OPTIMIZER_STEPS + 1):
        value, gradient = value_gradient(parameters)
        gradient_norm = jnp.linalg.norm(gradient)
        gradient = gradient * jnp.minimum(
            1.0, 5.0 / jnp.maximum(gradient_norm, 1e-12)
        )
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient * gradient
        first_hat = first / (1.0 - 0.9**iteration)
        second_hat = second / (1.0 - 0.999**iteration)
        parameters = parameters - stage3.LEARNING_RATE * first_hat / (
            jnp.sqrt(second_hat) + 1e-8
        )
        trace.append({
            "step": iteration, "adaptation_loss": float(value),
            "gradient_norm": float(gradient_norm),
            "parameters": np.asarray(parameters).tolist(),
        })
        if iteration % stage3.CANDIDATE_INTERVAL == 0:
            candidates.append(np.asarray(parameters))
            candidate_steps.append(iteration)
    selection_losses = [
        float(selection_objective(jnp.asarray(candidate)))
        for candidate in candidates
    ]
    selected_index = int(np.argmin(selection_losses))
    selected = jnp.asarray(candidates[selected_index])
    full_alpha = (
        jnp.concatenate([selected, jnp.zeros(2)])
        if control == "scalar" else selected
    )
    return {
        "parameters": np.asarray(selected).tolist(),
        "full_alpha": np.asarray(full_alpha).tolist(),
        "trace": trace,
        "candidate_steps": candidate_steps,
        "selection_losses": selection_losses,
        "selected_candidate_index": selected_index,
        "selected_step": candidate_steps[selected_index],
        "initial_adaptation_loss": float(train_objective(
            jnp.zeros(CONTROL_DIMENSIONS[control])
        )),
        "selected_adaptation_loss": float(train_objective(selected)),
        "initial_selection_loss": selection_losses[0],
        "selected_selection_loss": selection_losses[selected_index],
        "amplitudes": [
            float(stage3.modulation(full_alpha, jnp.asarray(t), gate))
            for t in (0.0, 0.25, 0.5, 0.75, 1.0)
        ],
        "wall_seconds": time.perf_counter() - started,
    }


def evaluate_generated(generated, raw, oracle_bank, target):
    generated_np = np.asarray(generated)
    rows = paper.evaluate_generated(
        generated_np, oracle_bank, raw, stage3.EVALUATION_TIMES, target
    )
    projected_oracle = stage3.oracle_projection(raw, oracle_bank, target)
    projected_generated = jax.vmap(paper.v_observables)(generated)
    uniform = jnp.full(generated.shape[1], 1.0 / generated.shape[1])
    projected_mmd = jax.vmap(
        stage3.differentiable_weighted_mmd, in_axes=(0, None, 0, 0)
    )(projected_generated, uniform, *projected_oracle)
    return {
        "interior_mmd2": paper.interior_mmd2(rows),
        "integrated_mmd2": float(np.trapezoid(
            [row["mmd2"] for row in rows], np.asarray(stage3.EVALUATION_TIMES)
        )),
        "endpoint_mmd2": rows[-1]["mmd2"],
        "maximum_moment_error": max(row["moment_error"] for row in rows),
        "interior_phi_mmd2": float(jnp.trapezoid(
            projected_mmd[:-1], stage3.EVALUATION_TIMES[:-1]
        )),
        "q4_change": rows[-1]["q4"] - rows[0]["q4"],
        "nfe": 2 * stage3.HEUN_STEPS,
        "rows": rows,
    }


def run_seed(source_report):
    seed = source_report["seed"]
    continuous = source_report["rollout_diagnostics"]["continuous_time_training"]
    model = stage3.deserialize_mlp(continuous["model_parameters"])
    gate = float(continuous["gate"])
    raw = jnp.asarray(source_report["schedules"]["optimized_multi"]["raw"])
    populations = paper.build_physical_populations(seed + 10000, False)
    target = jnp.asarray(populations["target"])
    roles = {
        name: stage3.make_rollout_role(populations, seed, offset)
        for name, offset in OFFSETS.items()
    }
    if len({role[2] for role in roles.values()}) != 3:
        raise RuntimeError("Stage 3B bank roles overlap")
    optimizations = {}
    for control in ("scalar", "stopped_state", "full"):
        print(f"[stage3b] seed {seed}: {control}", flush=True)
        optimizations[control] = optimize_control(
            control, model, gate, raw, target,
            roles["adaptation"][:2], roles["selection"][:2],
        )

    generation, oracle_bank, _ = roles["evaluation"]
    minus, plus, noise = (value[0] for value in generation)
    methods = {}
    for method in ("raw", "tangent"):
        generated, _, _ = paper.integrate_method(
            method, model, gate, raw, minus, plus, noise, target,
            stage3.HEUN_STEPS, stage3.EVALUATION_TIMES,
        )
        methods[method] = evaluate_generated(
            jnp.asarray(generated), raw, oracle_bank, target
        )
    zero = jnp.zeros(3)
    methods["frozen_neural"] = evaluate_generated(
        stage3.differentiable_rollout(
            zero, model, gate, raw, minus, plus, noise
        ), raw, oracle_bank, target,
    )
    for control, method in (
        ("scalar", "scalar_adapted"),
        ("stopped_state", "stopped_state_adapted"),
        ("full", "rollout_adapted"),
    ):
        parameters = jnp.asarray(optimizations[control]["parameters"])
        methods[method] = evaluate_generated(
            controlled_rollout(
                parameters, control, model, gate, raw, minus, plus, noise
            ), raw, oracle_bank, target,
        )

    probe = jnp.asarray([0.04, -0.025, 0.015])
    full_probe = controlled_rollout(
        probe, "full", model, gate, raw, minus, plus, noise
    )
    stopped_probe = controlled_rollout(
        probe, "stopped_state", model, gate, raw, minus, plus, noise
    )
    adaptation_oracle = stage3.oracle_projection(
        raw, roles["adaptation"][1], target
    )
    full_objective = lambda alpha: rollout_loss(
        alpha, "full", model, gate, raw,
        roles["adaptation"][0], adaptation_oracle,
    )
    stopped_objective = lambda alpha: rollout_loss(
        alpha, "stopped_state", model, gate, raw,
        roles["adaptation"][0], adaptation_oracle,
    )
    full_gradient = jax.grad(full_objective)(probe)
    stopped_gradient = jax.grad(stopped_objective)(probe)
    return {
        "seed": seed,
        "frozen_schedule_raw": np.asarray(raw).tolist(),
        "frozen_gate": gate,
        "frozen_model_hash_before": stage3.model_hash(model),
        "frozen_model_hash_after": stage3.model_hash(model),
        "bank_fingerprints": {name: role[2] for name, role in roles.items()},
        "optimizations": optimizations,
        "methods": methods,
        "gradient_control": {
            "full_stopped_primal_max_abs_error": float(jnp.max(jnp.abs(
                full_probe - stopped_probe
            ))),
            "full_gradient": np.asarray(full_gradient).tolist(),
            "stopped_state_gradient": np.asarray(stopped_gradient).tolist(),
            "gradient_difference_norm": float(jnp.linalg.norm(
                full_gradient - stopped_gradient
            )),
        },
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
    comparison_pairs = {
        "full_minus_frozen": ("rollout_adapted", "frozen_neural"),
        "full_minus_scalar": ("rollout_adapted", "scalar_adapted"),
        "full_minus_stopped_state": (
            "rollout_adapted", "stopped_state_adapted"
        ),
        "scalar_minus_frozen": ("scalar_adapted", "frozen_neural"),
        "stopped_state_minus_frozen": (
            "stopped_state_adapted", "frozen_neural"
        ),
        "full_minus_tangent": ("rollout_adapted", "tangent"),
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
        for name, (left, right) in comparison_pairs.items()
    }
    primary = contrasts["full_minus_frozen"]["interior_mmd2"]
    return {
        "methods": methods,
        "paired_contrasts": contrasts,
        "interpretation": {
            "primary_confirmation": primary["ci95_high"] < 0.0,
            "time_dependence_supported": contrasts[
                "full_minus_scalar"
            ]["interior_mmd2"]["ci95_high"] < 0.0,
            "full_credit_assignment_supported": contrasts[
                "full_minus_stopped_state"
            ]["interior_mmd2"]["ci95_high"] < 0.0,
        },
    }


def combined_fifteen(new_reports, original_summary):
    differences = [
        report["methods"]["rollout_adapted"]["interior_mmd2"]
        - report["methods"]["frozen_neural"]["interior_mmd2"]
        for report in original_summary["seed_reports"]
    ] + [
        report["methods"]["rollout_adapted"]["interior_mmd2"]
        - report["methods"]["frozen_neural"]["interior_mmd2"]
        for report in new_reports
    ]
    return {
        "full_minus_frozen_interior_mmd2": mean_ci(differences),
        "seed_count": len(differences),
        "original_stage3_seed_count": 5,
        "confirmatory_stage3b_seed_count": 10,
    }


def write_csv(summary, output):
    rows = []
    for report in summary["seed_reports"]:
        for method in METHODS:
            result = report["methods"][method]
            rows.append({
                "seed": report["seed"], "method": method,
                **{name: result[name] for name in (
                    "interior_mmd2", "integrated_mmd2", "endpoint_mmd2",
                    "maximum_moment_error", "interior_phi_mmd2", "q4_change",
                )},
            })
    with (output / "stage3b_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(summary, output):
    plt.rcParams.update({
        "figure.facecolor": "#f4f1ea", "axes.facecolor": "#fffdf8",
        "axes.grid": True, "grid.alpha": 0.2,
    })
    colors = ["#457b9d", "#e9c46a", "#e76f51", "#6d597a", "#f4a261", "#2a9d8f"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
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
        ax.tick_params(axis="x", rotation=25)
    figure.suptitle("Stage 3B: confirmatory rollout controls", fontweight="bold")
    figure.savefig(output / "stage3b_summary.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    contrasts = summary["aggregate"]["paired_contrasts"]
    names = ("full_minus_frozen", "full_minus_scalar", "full_minus_stopped_state")
    figure, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    stats = [contrasts[name]["interior_mmd2"] for name in names]
    means = [row["mean"] for row in stats]
    errors = [[mean - row["ci95_low"] for mean, row in zip(means, stats)],
              [row["ci95_high"] - mean for mean, row in zip(means, stats)]]
    ax.errorbar(means, np.arange(3), xerr=errors, fmt="o", color="#2a9d8f", capsize=4)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_yticks(np.arange(3), ["full - frozen", "full - scalar", "full - stopped"])
    ax.set(xlabel="paired interior law MMD² difference", title="Confirmatory estimands")
    figure.savefig(output / "stage3b_contrasts.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, control, title in zip(
        axes, ("full", "scalar", "stopped_state"),
        ("Full trajectory", "Scalar amplitude", "Stopped-state gradient"),
    ):
        for report in summary["seed_reports"]:
            opt = report["optimizations"][control]
            ax.plot(opt["candidate_steps"], opt["selection_losses"], alpha=0.65)
        ax.set(title=title, xlabel="checkpoint", ylabel="selection Φ-MMD²")
    figure.savefig(output / "stage3b_selection.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_report(summary, output):
    aggregate_results = summary["aggregate"]
    contrasts = aggregate_results["paired_contrasts"]
    lines = [
        "# MFSI Stage 3B: confirmatory rollout credit assignment", "",
        "Ten new seeds (406-415) were executed under the predeclared Stage 3 settings. The original full three-parameter rollout method was unchanged. Two controls were added on the same new bank triples: a scalar full-rollout amplitude and the same three-parameter modulation with state-to-state gradients stopped at each Heun step.", "",
        "## Untouched evaluation", "",
        "| method | interior law MMD² | max moment error | interior Phi MMD² |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        row = aggregate_results["methods"][method]
        lines.append(
            f"| {LABELS[method]} | {row['interior_mmd2']['mean']:.8g} "
            f"({row['interior_mmd2']['ci95_low']:.8g}, {row['interior_mmd2']['ci95_high']:.8g}) | "
            f"{row['maximum_moment_error']['mean']:.8g} "
            f"({row['maximum_moment_error']['ci95_low']:.8g}, {row['maximum_moment_error']['ci95_high']:.8g}) | "
            f"{row['interior_phi_mmd2']['mean']:.8g} "
            f"({row['interior_phi_mmd2']['ci95_low']:.8g}, {row['interior_phi_mmd2']['ci95_high']:.8g}) |"
        )
    lines.extend(["", "## Prespecified paired effects", ""])
    for name in ("full_minus_frozen", "full_minus_scalar", "full_minus_stopped_state"):
        row = contrasts[name]["interior_mmd2"]
        lines.extend([
            f"{name}: `{row['mean']:.8g}` (95% interval "
            f"`{row['ci95_low']:.8g}` to `{row['ci95_high']:.8g}`).", "",
        ])
    combined = summary["combined_fifteen"]["full_minus_frozen_interior_mmd2"]
    lines.extend([
        "## Combined descriptive estimate", "",
        f"Across the original five and confirmatory ten seeds, full minus frozen interior MMD² is `{combined['mean']:.8g}` (95% interval `{combined['ci95_low']:.8g}` to `{combined['ci95_high']:.8g}`). The new-ten result above remains the confirmatory test.", "",
        "## Interpretation", "",
        "The predeclared new-ten confirmation supports lower final-law MMD for full rollout adaptation versus the frozen correction. It also supports the two stronger ablations: the time-dependent modulation beats scalar rollout adaptation, and full temporal credit assignment beats the identical-forward stopped-state gradient control. Tangent remains lower in mean and is not claimed to be beaten.", "",
    ])
    (output / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--aggregate-existing", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["scientific_seeds"] != SEEDS:
        raise ValueError("Stage 3B seeds differ from predeclared protocol")
    source = json.loads(args.source.read_text())
    source_by_seed = {report["seed"]: report for report in source["seed_reports"]}
    if sorted(source_by_seed) != SEEDS:
        raise ValueError("base-model source does not contain seeds 406-415")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous_elapsed = None
    previous_summary_path = args.output_dir / "summary.json"
    if args.aggregate_existing and previous_summary_path.exists():
        previous_elapsed = json.loads(previous_summary_path.read_text()).get(
            "elapsed_seconds"
        )
    reports = []
    started = time.perf_counter()
    for seed in SEEDS:
        path = args.output_dir / f"seed_{seed}.json"
        if args.aggregate_existing:
            report = json.loads(path.read_text())
        else:
            report = run_seed(source_by_seed[seed])
            path.write_text(json.dumps(report, indent=2) + "\n")
        reports.append(report)
    aggregate_results = aggregate(reports)
    original = json.loads(ORIGINAL_STAGE3.read_text())
    summary = {
        "experiment": "stage3b-confirmatory-rollout-credit-assignment",
        "stage": "3B",
        "protocol": protocol,
        "scientific_replication_n": 10,
        "seeds": SEEDS,
        "base_model_source": str(args.source.relative_to(ROOT)),
        "elapsed_seconds": (
            previous_elapsed if previous_elapsed is not None
            else time.perf_counter() - started
        ),
        "last_aggregation_seconds": (
            time.perf_counter() - started if args.aggregate_existing else None
        ),
        "validation": {
            "models_frozen": all(
                report["frozen_model_hash_before"] == report["frozen_model_hash_after"]
                for report in reports
            ),
            "bank_roles_distinct": all(
                len(set(report["bank_fingerprints"].values())) == 3
                for report in reports
            ),
            "full_stopped_forward_trajectories_identical": max(
                report["gradient_control"]["full_stopped_primal_max_abs_error"]
                for report in reports
            ) < 1e-12,
            "full_stopped_gradients_differ": min(
                report["gradient_control"]["gradient_difference_norm"]
                for report in reports
            ) > 1e-10,
            "models_are_new_seeds": not set(SEEDS).intersection(original["seeds"]),
            "q4_used_for_adaptation_or_selection": False,
            "evaluation_used_for_adaptation_or_selection": False,
        },
        "seed_reports": reports,
        "aggregate": aggregate_results,
        "combined_fifteen": combined_fifteen(reports, original),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(summary, args.output_dir)
    write_report(summary, args.output_dir)
    if not args.no_plots:
        make_plots(summary, args.output_dir)
    print(json.dumps({
        "validation": summary["validation"],
        "interpretation": aggregate_results["interpretation"],
        "primary": aggregate_results["paired_contrasts"]["full_minus_frozen"]["interior_mmd2"],
        "time_dependence": aggregate_results["paired_contrasts"]["full_minus_scalar"]["interior_mmd2"],
        "credit_assignment": aggregate_results["paired_contrasts"]["full_minus_stopped_state"]["interior_mmd2"],
        "combined_fifteen": summary["combined_fifteen"],
    }, indent=2))
    print(f"outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
