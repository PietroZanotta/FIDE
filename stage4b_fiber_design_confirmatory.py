#!/usr/bin/env python3
"""MFSI Stage 4B: predeclared confirmatory fiber-design experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

import level2_paper_study as paper
import stage4_fiber_design as stage4


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "stage4b_fiber_design_confirmatory"
PROTOCOL_NAME = "stage4b_protocol.json"
SEEDS = tuple(range(426, 436))
METHODS = ("hand", "stop_grad", "full_grad")
OFFSETS = {
    "adaptation": 111000,
    "selection": 112000,
    "evaluation": 113000,
    "gradient_validation": 114000,
}
COUNTS = {
    "adaptation": stage4.ADAPTATION_COUNT,
    "selection": stage4.SELECTION_COUNT,
    "evaluation": stage4.EVALUATION_COUNT,
    "gradient_validation": 48,
}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_role(populations, seed: int, role: str):
    rng = np.random.default_rng(seed + OFFSETS[role])
    bank = paper.make_bridge_bank(
        populations, rng, np.asarray(stage4.TIMES), COUNTS[role]
    )
    return bank, stage4.array_hash(*bank)


def select_frozen_schedule(seed: int, populations):
    """Run only the established standard-mode Level-2 schedule prefix."""
    rng = np.random.default_rng(seed)
    times = stage4.TIMES
    train = paper.make_bridge_bank(populations, rng, np.asarray(times), 72)
    selection = paper.make_bridge_bank(populations, rng, np.asarray(times), 192)
    # Preserve the exact established RNG prefix, although the test bank is not
    # inspected here and has no effect on schedule selection.
    test = paper.make_bridge_bank(populations, rng, np.asarray(times), 192)
    target = jnp.asarray(populations["target"])
    hand = np.asarray([paper._inverse_softplus(0.55)])
    scalar, scalar_trace, scalar_seconds = paper.optimize_schedule(
        hand, train, times, target, 45
    )
    multi_initial = np.asarray([scalar[0], 0.0, 0.0])
    candidate, multi_trace, multi_seconds = paper.optimize_schedule(
        multi_initial, train, times, target, 45
    )
    nested = np.asarray([scalar[0], 0.0, 0.0])
    selection_objectives = {
        "nested_scalar": float(
            paper.schedule_objective(jnp.asarray(nested), selection, times, target)
        ),
        "multi_candidate": float(
            paper.schedule_objective(jnp.asarray(candidate), selection, times, target)
        ),
    }
    choice = min(selection_objectives, key=selection_objectives.get)
    selected = nested if choice == "nested_scalar" else candidate
    return jnp.asarray(selected), {
        "source": "frozen standard-mode Level-2 optimized_multi schedule prefix",
        "selected_family": choice,
        "selected_raw": np.asarray(selected).tolist(),
        "multi_candidate_raw": np.asarray(candidate).tolist(),
        "selection_objectives": selection_objectives,
        "scalar_trace": scalar_trace,
        "multi_trace": multi_trace,
        "scalar_optimizer_seconds": scalar_seconds,
        "multi_optimizer_seconds": multi_seconds,
        "schedule_train_fingerprint": stage4.array_hash(*train),
        "schedule_selection_fingerprint": stage4.array_hash(*selection),
        "schedule_test_fingerprint_uninspected": stage4.array_hash(*test),
    }


def _fiber_state(raw, t, minus, plus, noise, common_mean, theta, basis, stopped):
    """Frozen Stage-4 forward path with an optional backward-only ablation."""
    coefficients = stage4.observable_coefficients(theta, basis)
    state, velocity = paper.bridge_state(raw, t, minus, plus, noise)
    dictionary = stage4.v_dictionary(state)
    observables = dictionary @ coefficients.T
    target = coefficients @ common_mean
    lam = stage4.calibrate_converged(observables, target)
    weights, moments, covariance = paper._tilt(lam, observables)
    if stopped:
        lam = jax.lax.stop_gradient(lam)
        weights = jax.lax.stop_gradient(weights)
        moments = jax.lax.stop_gradient(moments)
        covariance = jax.lax.stop_gradient(covariance)
    dictionary_jacobians = stage4.v_jdictionary(state)
    jacobians = jnp.einsum("rk,mknd->mrnd", coefficients, dictionary_jacobians)
    jphi_u = jnp.einsum("mrnd,mnd->mr", jacobians, velocity)
    expected = weights @ jphi_u
    scalar = jphi_u @ lam
    covariance_term = jnp.sum(
        weights[:, None] * (observables - target) * scalar[:, None], axis=0
    )
    if stopped:
        expected = jax.lax.stop_gradient(expected)
        covariance_term = jax.lax.stop_gradient(covariance_term)
    lambda_dot = paper._solve(
        covariance, -expected - covariance_term, paper.CALIBRATION_RIDGE
    )
    if stopped:
        lambda_dot = jax.lax.stop_gradient(lambda_dot)
    forcing = (observables - target) @ lambda_dot + (jphi_u - expected) @ lam
    forcing = forcing - weights @ forcing
    descriptor_values = paper.v_descriptors(state)
    descriptor_jacobians = paper.v_jdesc(state)
    gram = jnp.einsum(
        "m,mknd,mlnd->kl", weights, descriptor_jacobians, descriptor_jacobians
    )
    rhs = jnp.einsum("m,mk,m->k", weights, descriptor_values, forcing)
    ritz_coefficients = paper._solve(gram, rhs, paper.RITZ_RIDGE)
    correction = jnp.einsum(
        "mknd,k->mnd", descriptor_jacobians, ritz_coefficients
    )
    correction_energy = jnp.sum(
        weights * jnp.sum(correction * correction, axis=(1, 2))
    )
    forcing_power = jnp.sum(weights * forcing * forcing)
    ess = 1.0 / (state.shape[0] * jnp.sum(weights * weights))
    residual = jnp.linalg.norm(moments - target)
    return state, weights, correction_energy, forcing_power, ess, residual


def path_metrics(raw, bank, common_mean, theta, basis, stopped=False):
    return jax.vmap(
        lambda t, xm, xp, z: _fiber_state(
            raw, t, xm, xp, z, common_mean, theta, basis, stopped
        )
    )(stage4.TIMES, bank[0], bank[1], bank[2])


def objective(raw, bank, common_mean, theta, basis, stopped=False):
    values = path_metrics(raw, bank, common_mean, theta, basis, stopped)
    energy, forcing, ess = values[2], values[3], values[4]
    penalty = 15.0 * jnp.trapezoid(
        jax.nn.relu(stage4.ESS_FLOOR - ess) ** 2, stage4.TIMES
    )
    return jnp.trapezoid(
        energy + stage4.FORCING_WEIGHT * forcing, stage4.TIMES
    ) + penalty


def deterministic_candidates(theta0):
    index = jnp.arange(theta0.size, dtype=jnp.float64).reshape(theta0.shape)
    return (
        theta0,
        theta0 + 0.005 * jnp.sin(index + 0.2),
        theta0 + 0.010 * jnp.cos(0.7 * index + 0.4),
        theta0 + 0.015 * jnp.sin(1.3 * index + 0.8),
    )


def run_numerical_checks(output, raw, geometry, bank):
    full = jax.jit(lambda theta: objective(
        raw, bank, geometry["common_mean"], theta, geometry["basis"], False
    ))
    stopped = jax.jit(lambda theta: objective(
        raw, bank, geometry["common_mean"], theta, geometry["basis"], True
    ))
    forward_rows = []
    for index, theta in enumerate(deterministic_candidates(geometry["theta0"])):
        full_value = float(full(theta))
        stop_value = float(stopped(theta))
        forward_rows.append({
            "candidate": index,
            "full_grad_forward": full_value,
            "stop_grad_forward": stop_value,
            "absolute_difference": abs(full_value - stop_value),
        })
    forward_check = {
        "tolerance": 1e-10,
        "candidates": forward_rows,
        "maximum_absolute_difference": max(
            row["absolute_difference"] for row in forward_rows
        ),
    }
    forward_check["passed"] = (
        forward_check["maximum_absolute_difference"] <= forward_check["tolerance"]
    )

    theta = geometry["theta0"] + 0.01 * jnp.sin(
        jnp.arange(geometry["theta0"].size, dtype=jnp.float64)
    ).reshape(geometry["theta0"].shape)
    direction = jnp.cos(
        jnp.arange(theta.size, dtype=jnp.float64) + 0.3
    ).reshape(theta.shape)
    direction /= jnp.linalg.norm(direction)
    full_gradient = jax.grad(full)(theta)
    stop_gradient = jax.grad(stopped)(theta)
    autodiff = float(jnp.sum(full_gradient * direction))
    finite_step = 2e-5
    finite = float(
        (full(theta + finite_step * direction) - full(theta - finite_step * direction))
        / (2.0 * finite_step)
    )
    relative_error = abs(autodiff - finite) / max(
        abs(autodiff), abs(finite), 1e-12
    )
    gradient_check = {
        "autodiff_directional_derivative": autodiff,
        "central_finite_difference_directional_derivative": finite,
        "finite_difference_step": finite_step,
        "relative_error": relative_error,
        "comparison_target": "comparable to prior Stage-4 relative error 2.58e-8",
        "tolerance": 2e-6,
        "passed": relative_error < 2e-6,
        "gradient_ablation": {
            "full_gradient_norm": float(jnp.linalg.norm(full_gradient)),
            "stop_gradient_norm": float(jnp.linalg.norm(stop_gradient)),
            "difference_norm": float(jnp.linalg.norm(full_gradient - stop_gradient)),
            "relative_gradient_discrepancy": float(
                jnp.linalg.norm(full_gradient - stop_gradient)
                / jnp.maximum(
                    jnp.maximum(
                        jnp.linalg.norm(full_gradient), jnp.linalg.norm(stop_gradient)
                    ),
                    1e-12,
                )
            ),
            "forward_difference_at_gradient_candidate": abs(
                float(full(theta)) - float(stopped(theta))
            ),
        },
    }
    gradient_check["gradient_ablation"]["passed"] = (
        gradient_check["gradient_ablation"]["forward_difference_at_gradient_candidate"]
        <= 1e-10
        and gradient_check["gradient_ablation"]["difference_norm"] > 1e-10
    )
    write_json(output / "forward_equivalence.json", forward_check)
    write_json(output / "gradient_check.json", gradient_check)
    if not (
        forward_check["passed"]
        and gradient_check["passed"]
        and gradient_check["gradient_ablation"]["passed"]
    ):
        raise RuntimeError("Stage-4B numerical prechecks failed")
    return forward_check, gradient_check


def optimize(raw, geometry, adaptation, selection, stopped):
    train = jax.jit(lambda theta: objective(
        raw, adaptation, geometry["common_mean"], theta, geometry["basis"], stopped
    ))
    select = jax.jit(lambda theta: objective(
        raw, selection, geometry["common_mean"], theta, geometry["basis"], stopped
    ))
    value_gradient = jax.jit(jax.value_and_grad(train))
    theta = geometry["theta0"]
    first = jnp.zeros_like(theta)
    second = jnp.zeros_like(theta)
    candidates = [np.asarray(theta)]
    candidate_steps = [0]
    trace = []
    started = time.perf_counter()
    for iteration in range(1, stage4.OPTIMIZER_STEPS + 1):
        value, gradient = value_gradient(theta)
        norm = jnp.linalg.norm(gradient)
        gradient = gradient * jnp.minimum(1.0, 5.0 / jnp.maximum(norm, 1e-12))
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient * gradient
        first_hat = first / (1.0 - 0.9**iteration)
        second_hat = second / (1.0 - 0.999**iteration)
        theta = theta - stage4.LEARNING_RATE * first_hat / (
            jnp.sqrt(second_hat) + 1e-8
        )
        theta = stage4.canonical_rows(theta)
        trace.append({
            "step": iteration,
            "adaptation_objective": float(value),
            "gradient_norm": float(norm),
        })
        if iteration % stage4.CANDIDATE_INTERVAL == 0:
            candidates.append(np.asarray(theta))
            candidate_steps.append(iteration)
    selection_values = [float(select(jnp.asarray(candidate))) for candidate in candidates]
    finite = np.where(np.isfinite(selection_values), selection_values, np.inf)
    if not np.isfinite(finite[0]):
        raise FloatingPointError("non-finite hand checkpoint objective")
    selected_index = int(np.argmin(finite))
    selected = jnp.asarray(candidates[selected_index])
    return selected, {
        "candidate_steps": candidate_steps,
        "selection_objectives": selection_values,
        "selected_candidate_index": selected_index,
        "selected_step": candidate_steps[selected_index],
        "initial_adaptation_objective": float(train(geometry["theta0"])),
        "selected_adaptation_objective": float(train(selected)),
        "initial_selection_objective": selection_values[0],
        "selected_selection_objective": selection_values[selected_index],
        "trace": trace,
        "wall_seconds": time.perf_counter() - started,
    }


def evaluate(raw, bank, geometry, theta):
    values = path_metrics(
        raw, bank, geometry["common_mean"], theta, geometry["basis"], False
    )
    states, weights, energy, forcing, ess, residual = values
    coefficients = stage4.observable_coefficients(theta, geometry["basis"])
    endpoint_gap = coefficients @ jnp.asarray(geometry["raw_endpoint_gap"])
    null_residual = jnp.linalg.norm(endpoint_gap)
    orthonormality = jnp.max(jnp.abs(
        coefficients @ coefficients.T - jnp.eye(coefficients.shape[0])
    ))
    return {
        "construction_objective": float(objective(
            raw, bank, geometry["common_mean"], theta, geometry["basis"], False
        )),
        "correction_energy": float(jnp.trapezoid(energy, stage4.TIMES)),
        "forcing_power": float(jnp.trapezoid(forcing, stage4.TIMES)),
        "minimum_ess": float(jnp.min(ess)),
        "median_ess": float(jnp.median(ess)),
        "maximum_calibration_residual": float(jnp.max(residual)),
        "endpoint_equivalence_residual": float(jnp.linalg.norm(endpoint_gap)),
        "orthonormality_max_abs_residual": float(orthonormality),
        "nullspace_residual": float(null_residual),
        "states": states,
        "weights": weights,
        "coefficients": np.asarray(coefficients),
    }


def principal_geometry(left, right):
    singular = np.linalg.svd(left @ right.T, compute_uv=False)
    singular = np.clip(singular, -1.0, 1.0)
    angles = np.arccos(singular)
    return {
        "principal_angles_radians": angles.tolist(),
        "principal_angles_degrees": np.degrees(angles).tolist(),
        "subspace_chordal_distance": float(np.sqrt(np.sum(np.sin(angles) ** 2))),
    }


def paired_summary(values):
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1))
    half = float(stats.t.ppf(0.975, len(array) - 1) * sd / math.sqrt(len(array)))
    return {
        "n": len(array),
        "mean_paired_difference": mean,
        "paired_sd": sd,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "favorable_seed_count": int(np.sum(array < 0.0)),
        "seed_level_differences": array.tolist(),
        "success": mean + half < 0.0,
    }


def aggregate_method(rows, method, metric):
    values = np.asarray([row["methods"][method][metric] for row in rows])
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def write_csv_outputs(rows, geometry_rows, output):
    metric_names = (
        "construction_objective", "correction_energy", "forcing_power",
        "minimum_ess", "median_ess", "maximum_calibration_residual",
        "endpoint_equivalence_residual", "orthonormality_max_abs_residual",
        "nullspace_residual",
    )
    metric_rows = []
    for row in rows:
        for method in METHODS:
            metric_rows.append({
                "seed": row["seed"], "method": method,
                **{name: row["methods"][method][name] for name in metric_names},
            })
    with (output / "per_seed_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    with (output / "fiber_geometry.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(geometry_rows[0]))
        writer.writeheader()
        writer.writerows(geometry_rows)


def write_report(summary, output):
    primary = summary["contrasts"]["full_grad_minus_hand"]
    mechanism = summary["contrasts"]["full_grad_minus_stop_grad"]
    lines = [
        "# MFSI Stage 4B — Confirmatory Fiber Design", "",
        "## Confirmatory decisions", "",
        f"- `full_grad - hand` **{'confirmed' if primary['success'] else 'did not confirm'}**: mean paired difference `{primary['mean_paired_difference']:.8g}`, paired SD `{primary['paired_sd']:.8g}`, 95% paired t interval `[{primary['ci95_low']:.8g}, {primary['ci95_high']:.8g}]`, favorable seeds `{primary['favorable_seed_count']}/10`.",
        f"- `full_grad - stop_grad` **{'confirmed' if mechanism['success'] else 'did not confirm'}**: mean paired difference `{mechanism['mean_paired_difference']:.8g}`, paired SD `{mechanism['paired_sd']:.8g}`, 95% paired t interval `[{mechanism['ci95_low']:.8g}, {mechanism['ci95_high']:.8g}]`, favorable seeds `{mechanism['favorable_seed_count']}/10`.",
        f"- Stage 5 is **{'scientifically justified' if summary['stage5_scientifically_justified'] else 'not scientifically justified by the strong-success rule'}**. Stage 5 was not implemented.", "",
        "The decision uses only the ten predeclared new seeds 426–435. No Stage-4 pilot seed was pooled into either interval.", "",
        "## Untouched evaluation-bank metrics", "",
        "| fiber | objective | correction energy | forcing power | minimum ESS | median ESS | max calibration residual | endpoint residual |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = summary["evaluation_metrics"][method]
        lines.append(
            f"| {method} | {row['construction_objective']['mean']:.8g} | "
            f"{row['correction_energy']['mean']:.8g} | {row['forcing_power']['mean']:.8g} | "
            f"{row['minimum_ess']['mean']:.8g} | {row['median_ess']['mean']:.8g} | "
            f"{row['maximum_calibration_residual']['maximum']:.3e} | "
            f"{row['endpoint_equivalence_residual']['maximum']:.3e} |"
        )
    lines.extend(["", "## Seed-level objective differences", "",
        "| seed | full - hand | full - stop |", "|---:|---:|---:|"])
    for seed, a, b in zip(
        summary["seeds"], primary["seed_level_differences"],
        mechanism["seed_level_differences"],
    ):
        lines.append(f"| {seed} | {a:.8g} | {b:.8g} |")
    lines.extend([
        "", "## Numerical and protocol checks", "",
        f"Forward equivalence passed at four deterministic fibers; maximum absolute difference was `{summary['checks']['forward_max_abs_difference']:.3e}`.", "",
        f"The full-gradient directional check had relative error `{summary['checks']['full_gradient_relative_error']:.3e}`. The full-versus-stop gradient discrepancy norm was `{summary['checks']['gradient_difference_norm']:.8g}` (relative discrepancy `{summary['checks']['relative_gradient_discrepancy']:.8g}`).", "",
        f"Across all selected fibers, maximum row-orthonormality, endpoint-equivalence, and nullspace residuals were `{summary['checks']['maximum_orthonormality_residual']:.3e}`, `{summary['checks']['maximum_endpoint_residual']:.3e}`, and `{summary['checks']['maximum_nullspace_residual']:.3e}`.", "",
        "q4 and angular descriptors were excluded from the dictionary, objective, optimization, and checkpoint selection. q4 was computed only after every checkpoint choice had been frozen and is retained only as an evaluation diagnostic.", "",
        "`D_proj` is not emitted because it is not available in the frozen Stage-4 construction code. The optional downstream test was not run.", "",
        "## Interpretation", "",
        summary["interpretation"], "",
    ])
    (output / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / PROTOCOL_NAME
    if not protocol_path.exists():
        raise FileNotFoundError(
            f"predeclared protocol must exist before execution: {protocol_path}"
        )
    protocol = json.loads(protocol_path.read_text())
    if tuple(protocol["seeds"]) != SEEDS:
        raise ValueError("protocol seed block does not match frozen Stage-4B driver")

    config = {
        **protocol["frozen_config"],
        "seeds": list(SEEDS),
        "scientific_replication_n": len(SEEDS),
        "bank_offsets": OFFSETS,
        "bank_counts": COUNTS,
        "methods": list(METHODS),
        "d_proj_available_in_frozen_code": False,
    }
    write_json(output / "config.json", config)
    started = time.perf_counter()

    # Physical populations and the selected schedule are frozen prerequisites.
    contexts = []
    provenance_rows = []
    print("[stage4b] preparing frozen per-seed schedules", flush=True)
    for seed in SEEDS:
        populations = paper.build_physical_populations(seed + 10000, False)
        geometry = stage4.endpoint_geometry(populations)
        raw, schedule = select_frozen_schedule(seed, populations)
        roles = {
            role: make_role(populations, seed, role)
            for role in ("adaptation", "selection", "evaluation")
        }
        if len({fingerprint for _, fingerprint in roles.values()}) != 3:
            raise RuntimeError(f"bank roles overlap for seed {seed}")
        endpoint_hash = stage4.array_hash(
            populations["minus"], populations["plus"],
            populations["minus_weights"], populations["plus_weights"],
        )
        contexts.append({
            "seed": seed, "populations": populations, "geometry": geometry,
            "raw": raw, "schedule": schedule, "roles": roles,
            "endpoint_hash": endpoint_hash,
        })
        provenance_rows.append({
            "seed": seed,
            "physical_population_seed": seed + 10000,
            "endpoint_law_hash": endpoint_hash,
            "schedule": schedule,
            "bank_fingerprints": {
                name: value[1] for name, value in roles.items()
            },
        })

    # Required checks occur before either scientific optimizer is launched.
    first = contexts[0]
    validation_bank, validation_hash = make_role(
        first["populations"], first["seed"], "gradient_validation"
    )
    forward_check, gradient_check = run_numerical_checks(
        output, first["raw"], first["geometry"], validation_bank
    )
    print(
        "[stage4b] numerical checks passed: "
        f"forward={forward_check['maximum_absolute_difference']:.3e}, "
        f"gradient relative error={gradient_check['relative_error']:.3e}",
        flush=True,
    )

    # Freeze every selection before accessing evaluation-only q4.
    selection_rows = []
    for context in contexts:
        seed = context["seed"]
        print(f"[stage4b] seed {seed}: full_grad", flush=True)
        full_theta, full_optimization = optimize(
            context["raw"], context["geometry"],
            context["roles"]["adaptation"][0],
            context["roles"]["selection"][0], False,
        )
        print(f"[stage4b] seed {seed}: stop_grad", flush=True)
        stop_theta, stop_optimization = optimize(
            context["raw"], context["geometry"],
            context["roles"]["adaptation"][0],
            context["roles"]["selection"][0], True,
        )
        context["selected"] = {
            "hand": context["geometry"]["theta0"],
            "full_grad": full_theta,
            "stop_grad": stop_theta,
        }
        context["optimization"] = {
            "full_grad": full_optimization, "stop_grad": stop_optimization
        }
        coefficients = {
            method: np.asarray(stage4.observable_coefficients(
                context["selected"][method], context["geometry"]["basis"]
            )) for method in METHODS
        }
        context["coefficients"] = coefficients
        selection_rows.append({
            "seed": seed,
            "full_grad": {
                **full_optimization,
                "theta": np.asarray(full_theta).tolist(),
                "coefficients": coefficients["full_grad"].tolist(),
            },
            "stop_grad": {
                **stop_optimization,
                "theta": np.asarray(stop_theta).tolist(),
                "coefficients": coefficients["stop_grad"].tolist(),
            },
            "hand": {
                "selected_step": 0,
                "theta": np.asarray(context["geometry"]["theta0"]).tolist(),
                "coefficients": coefficients["hand"].tolist(),
            },
        })
    write_json(output / "selected_subspaces.json", {
        "selection_frozen_before_evaluation_q4": True,
        "seeds": selection_rows,
    })

    print("[stage4b] all selections frozen; evaluating untouched banks", flush=True)
    result_rows = []
    geometry_rows = []
    for context in contexts:
        evaluated = {}
        for method in METHODS:
            values = evaluate(
                context["raw"], context["roles"]["evaluation"][0],
                context["geometry"], context["selected"][method],
            )
            # q4 is first explicitly evaluated here, after all selections.
            q4_path = jax.vmap(
                lambda states, weights: weights @ paper.v_q4(states)
            )(values.pop("states"), values.pop("weights"))
            coefficients = values.pop("coefficients")
            values["q4_path_evaluation_only"] = np.asarray(q4_path).tolist()
            evaluated[method] = values
            if method != "hand":
                hand_geometry = principal_geometry(
                    coefficients, context["coefficients"]["hand"]
                )
                geometry_rows.append({
                    "seed": context["seed"],
                    "comparison": f"{method}_vs_hand",
                    "principal_angle_1_degrees": hand_geometry["principal_angles_degrees"][0],
                    "principal_angle_2_degrees": hand_geometry["principal_angles_degrees"][1],
                    "principal_angle_3_degrees": hand_geometry["principal_angles_degrees"][2],
                    "subspace_chordal_distance": hand_geometry["subspace_chordal_distance"],
                })
        full_stop = principal_geometry(
            context["coefficients"]["full_grad"],
            context["coefficients"]["stop_grad"],
        )
        geometry_rows.append({
            "seed": context["seed"], "comparison": "full_grad_vs_stop_grad",
            "principal_angle_1_degrees": full_stop["principal_angles_degrees"][0],
            "principal_angle_2_degrees": full_stop["principal_angles_degrees"][1],
            "principal_angle_3_degrees": full_stop["principal_angles_degrees"][2],
            "subspace_chordal_distance": full_stop["subspace_chordal_distance"],
        })
        row = {
            "seed": context["seed"],
            "frozen_schedule_raw": np.asarray(context["raw"]).tolist(),
            "optimization": context["optimization"],
            "methods": evaluated,
        }
        result_rows.append(row)
        write_json(output / f"seed_{context['seed']}.json", row)

    metric_names = (
        "construction_objective", "correction_energy", "forcing_power",
        "minimum_ess", "median_ess", "maximum_calibration_residual",
        "endpoint_equivalence_residual", "orthonormality_max_abs_residual",
        "nullspace_residual",
    )
    primary_values = [
        row["methods"]["full_grad"]["construction_objective"]
        - row["methods"]["hand"]["construction_objective"]
        for row in result_rows
    ]
    mechanism_values = [
        row["methods"]["full_grad"]["construction_objective"]
        - row["methods"]["stop_grad"]["construction_objective"]
        for row in result_rows
    ]
    contrasts = {
        "full_grad_minus_hand": paired_summary(primary_values),
        "full_grad_minus_stop_grad": paired_summary(mechanism_values),
    }
    primary_success = contrasts["full_grad_minus_hand"]["success"]
    mechanism_success = contrasts["full_grad_minus_stop_grad"]["success"]
    if primary_success and mechanism_success:
        interpretation = (
            "Strong success: differentiable moment-fiber design replicated, and "
            "correct differentiation through the I-projection materially contributed."
        )
    elif primary_success:
        interpretation = (
            "Fiber-design success only: optimization beat the hand fiber, but this "
            "experiment did not establish that the full implicit gradient is essential."
        )
    elif mechanism_success:
        interpretation = (
            "Gradient-mechanism success only: the correct gradient improved optimization, "
            "but the hand fiber remained statistically competitive."
        )
    else:
        interpretation = (
            "Neither confirmatory criterion passed. The Stage-4 five-seed pilot did not "
            "replicate strongly enough; this branch should stop without tuning."
        )
    maximum = lambda name: max(
        row["methods"][method][name] for row in result_rows for method in METHODS
    )
    summary = {
        "experiment": "stage4b-confirmatory-fiber-design",
        "scientific_replication_n": len(SEEDS),
        "seeds": list(SEEDS),
        "protocol_file": PROTOCOL_NAME,
        "evaluation_metrics": {
            method: {
                metric: aggregate_method(result_rows, method, metric)
                for metric in metric_names
            } for method in METHODS
        },
        "contrasts": contrasts,
        "stage5_scientifically_justified": primary_success and mechanism_success,
        "interpretation": interpretation,
        "checks": {
            "forward_max_abs_difference": forward_check["maximum_absolute_difference"],
            "full_gradient_relative_error": gradient_check["relative_error"],
            "gradient_difference_norm": gradient_check["gradient_ablation"]["difference_norm"],
            "relative_gradient_discrepancy": gradient_check["gradient_ablation"]["relative_gradient_discrepancy"],
            "maximum_orthonormality_residual": maximum("orthonormality_max_abs_residual"),
            "maximum_endpoint_residual": maximum("endpoint_equivalence_residual"),
            "maximum_nullspace_residual": maximum("nullspace_residual"),
            "all_bank_roles_disjoint": True,
            "q4_used_for_adaptation_or_selection": False,
            "evaluation_used_for_selection": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    provenance = {
        "protocol_sha256": file_hash(protocol_path),
        "driver_sha256": file_hash(Path(__file__)),
        "frozen_stage4_driver_sha256": file_hash(ROOT / "stage4_fiber_design.py"),
        "level2_source_sha256": file_hash(ROOT / "level2_paper_study.py"),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jax_backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "gradient_validation_bank_fingerprint": validation_hash,
        "rows": provenance_rows,
    }
    write_json(output / "seed_provenance.json", provenance)
    write_json(output / "paired_contrasts.json", contrasts)
    write_json(output / "summary.json", summary)
    write_csv_outputs(result_rows, geometry_rows, output)
    write_report(summary, output)
    print(json.dumps({
        "full_grad_minus_hand": contrasts["full_grad_minus_hand"],
        "full_grad_minus_stop_grad": contrasts["full_grad_minus_stop_grad"],
        "stage5_scientifically_justified": summary["stage5_scientifically_justified"],
    }, indent=2), flush=True)
    print(f"outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
