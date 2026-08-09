#!/usr/bin/env python3
"""MFSI Stage 4: differentiable design of a three-observable moment fiber.

The physical endpoint laws, their deterministic angular-sort coupling, the
selected reference schedule, calibration, and Ritz realization are fixed.  The
only optimized object is the rank-three subspace spanned by three radial RBF
observables.  q4 is excluded from both the candidate dictionary and every
adaptation/selection calculation.
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
PROTOCOL = ROOT / "stage4_protocol.json"
DEFAULT_OUTPUT = ROOT / "results" / "stage4_fiber_design"
SEEDS = [401, 402, 403, 404, 405]
TIMES = jnp.asarray(np.linspace(0.12, 0.88, 6), dtype=jnp.float64)
DICTIONARY_CENTERS = jnp.concatenate([paper.OBS_CENTERS, paper.DESC_CENTERS])
ADAPTATION_COUNT = 96
SELECTION_COUNT = 128
EVALUATION_COUNT = 256
OPTIMIZER_STEPS = 40
LEARNING_RATE = 0.03
CANDIDATE_INTERVAL = 5
ESS_FLOOR = 0.18
FORCING_WEIGHT = 0.02
CALIBRATION_STEPS = 640
METHODS = ("hand", "designed")
OFFSETS = {"adaptation": 101000, "selection": 102000, "evaluation": 103000}


def array_hash(*arrays) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def dictionary_single(configuration):
    """Candidate measurements: radial only, with the hand basis nested."""
    return paper.radial_descriptors_single(configuration, DICTIONARY_CENTERS)


v_dictionary = jax.vmap(dictionary_single)
v_jdictionary = jax.vmap(jax.jacrev(dictionary_single))


def endpoint_geometry(populations):
    """Build a unit-normalized endpoint-equal coefficient coordinate system.

    Columns of ``basis`` span coefficient vectors orthogonal to the difference
    of the two fixed endpoint feature means.  They are Euclidean-orthonormal;
    because every dictionary entry is the same bounded RBF family, this removes
    coefficient scaling/collapse while preserving the original numerical scale
    of the hand observables and their fixed calibration algorithm.
    """
    minus_values = np.asarray(v_dictionary(jnp.asarray(populations["minus"])))
    plus_values = np.asarray(v_dictionary(jnp.asarray(populations["plus"])))
    minus_weights = np.asarray(populations["minus_weights"])
    plus_weights = np.asarray(populations["plus_weights"])
    minus_mean = minus_weights @ minus_values
    plus_mean = plus_weights @ plus_values
    common_mean = 0.5 * (minus_mean + plus_mean)
    gap = plus_mean - minus_mean

    # The gap has nonzero components only outside the calibrated hand block up
    # to numerical precision, but use an explicit nullspace for exactness.
    _, _, right = np.linalg.svd(gap[None, :], full_matrices=True)
    null = right[1:].T
    basis = null

    # Project the original three measurements into the exact endpoint-null
    # space, then change coordinates/scale only.  This is the hand fiber at
    # optimizer step zero (up to endpoint calibration roundoff).
    hand = np.eye(len(DICTIONARY_CENTERS), dtype=np.float64)[:3]
    gap_norm2 = float(gap @ gap)
    if gap_norm2 > 0.0:
        hand = hand - (hand @ gap)[:, None] * gap[None, :] / gap_norm2
    normalized_hand = np.asarray(canonical_rows(jnp.asarray(hand)))
    theta0 = normalized_hand @ basis
    theta0 = np.asarray(canonical_rows(jnp.asarray(theta0)))
    return {
        "basis": jnp.asarray(basis),
        "theta0": jnp.asarray(theta0),
        "common_mean": jnp.asarray(common_mean),
        "raw_endpoint_gap": gap,
        "minus_dictionary_mean": minus_mean,
        "plus_dictionary_mean": plus_mean,
    }


def canonical_rows(theta):
    """Return a rank-three row-orthonormal representative of a subspace."""
    # Cholesky avoids the undefined eigenvector derivative at the hand
    # initialization, whose three singular values are exactly repeated.
    factor = jnp.linalg.cholesky(theta @ theta.T + 1e-12 * jnp.eye(theta.shape[0]))
    return jnp.linalg.solve(factor, theta)


def observable_coefficients(theta, basis):
    return canonical_rows(theta) @ basis.T


def _calibrate_converged_primal(observables, target):
    initial = jnp.zeros(target.shape[0], dtype=observables.dtype)

    def body(_, lam):
        _, moments, covariance = paper._tilt(lam, observables)
        step = paper._solve(
            covariance, moments - target, paper.CALIBRATION_RIDGE
        )
        norm = jnp.linalg.norm(step)
        return lam - step * jnp.minimum(1.0, 2.0 / jnp.maximum(norm, 1e-12))

    return jax.lax.fori_loop(0, CALIBRATION_STEPS, body, initial)


@jax.custom_jvp
def calibrate_converged(observables, target):
    """Converged exponential calibration with its implicit derivative."""
    return _calibrate_converged_primal(observables, target)


@calibrate_converged.defjvp
def _calibrate_converged_jvp(primals, tangents):
    observables, target = primals
    dobservables, dtarget = tangents
    lam = _calibrate_converged_primal(observables, target)
    weights, moments, covariance = paper._tilt(lam, observables)
    centered = observables - moments
    dlogit = jnp.sum(dobservables * lam[None, :], axis=-1)
    d_equation = (
        weights @ dobservables
        + jnp.sum(
            weights[:, None] * centered * dlogit[:, None], axis=0
        )
        - dtarget
    )
    # The calibrated equation is E_lambda[Phi] - target = 0, whose exact
    # Jacobian is the covariance.  CALIBRATION_RIDGE stabilizes the Newton
    # iterations but is not part of that equation and must not bias its JVP.
    tangent = paper._solve(covariance, -d_equation, 0.0)
    return lam, tangent


def designed_fiber_state(raw, t, minus, plus, noise, common_mean, theta, basis):
    """The established empirical moment-fiber construction with learned Phi."""
    coefficients = observable_coefficients(theta, basis)
    state, velocity = paper.bridge_state(raw, t, minus, plus, noise)
    dictionary = v_dictionary(state)
    observables = dictionary @ coefficients.T
    target = coefficients @ common_mean
    lam = calibrate_converged(observables, target)
    weights, moments, covariance = paper._tilt(lam, observables)
    dictionary_jacobians = v_jdictionary(state)
    jacobians = jnp.einsum("rk,mknd->mrnd", coefficients, dictionary_jacobians)
    jphi_u = jnp.einsum("mrnd,mnd->mr", jacobians, velocity)
    expected = weights @ jphi_u
    scalar = jphi_u @ lam
    covariance_term = jnp.sum(
        weights[:, None] * (observables - target) * scalar[:, None], axis=0
    )
    lambda_dot = paper._solve(
        covariance, -expected - covariance_term, paper.CALIBRATION_RIDGE
    )
    forcing = (observables - target) @ lambda_dot + (jphi_u - expected) @ lam
    forcing = forcing - weights @ forcing

    # The realization dictionary is deliberately unchanged from the paper
    # study.  Only the three measured observables above are variable.
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


def design_path_metrics(raw, bank, common_mean, theta, basis):
    return jax.vmap(
        lambda t, xm, xp, z: designed_fiber_state(
            raw, t, xm, xp, z, common_mean, theta, basis
        )
    )(TIMES, bank[0], bank[1], bank[2])


def construction_objective(raw, bank, common_mean, theta, basis):
    values = design_path_metrics(raw, bank, common_mean, theta, basis)
    energy, forcing, ess = values[2], values[3], values[4]
    penalty = 15.0 * jnp.trapezoid(jax.nn.relu(ESS_FLOOR - ess) ** 2, TIMES)
    return (
        jnp.trapezoid(energy + FORCING_WEIGHT * forcing, TIMES) + penalty
    )


def make_role(populations, seed, name):
    counts = {
        "adaptation": ADAPTATION_COUNT,
        "selection": SELECTION_COUNT,
        "evaluation": EVALUATION_COUNT,
    }
    rng = np.random.default_rng(seed + OFFSETS[name])
    bank = paper.make_bridge_bank(
        populations, rng, np.asarray(TIMES), counts[name]
    )
    return bank, array_hash(*bank)


def optimize_observables(raw, geometry, adaptation, selection, steps):
    common = geometry["common_mean"]
    basis = geometry["basis"]
    train_objective = jax.jit(
        lambda theta: construction_objective(raw, adaptation, common, theta, basis)
    )
    selection_objective = jax.jit(
        lambda theta: construction_objective(raw, selection, common, theta, basis)
    )
    value_gradient = jax.jit(jax.value_and_grad(train_objective))
    theta = geometry["theta0"]
    first = jnp.zeros_like(theta)
    second = jnp.zeros_like(theta)
    candidates = [np.asarray(theta)]
    candidate_steps = [0]
    trace = []
    started = time.perf_counter()
    for iteration in range(1, steps + 1):
        value, gradient = value_gradient(theta)
        norm = jnp.linalg.norm(gradient)
        gradient = gradient * jnp.minimum(1.0, 5.0 / jnp.maximum(norm, 1e-12))
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient * gradient
        first_hat = first / (1.0 - 0.9**iteration)
        second_hat = second / (1.0 - 0.999**iteration)
        theta = theta - LEARNING_RATE * first_hat / (jnp.sqrt(second_hat) + 1e-8)
        # Keep the optimizer representation well conditioned.  The objective
        # depends only on its canonical row space.
        theta = canonical_rows(theta)
        trace.append({
            "step": iteration,
            "adaptation_objective": float(value),
            "gradient_norm": float(norm),
        })
        if iteration % CANDIDATE_INTERVAL == 0:
            candidates.append(np.asarray(theta))
            candidate_steps.append(iteration)
    selection_values = [
        float(selection_objective(jnp.asarray(candidate))) for candidate in candidates
    ]
    finite_selection = np.where(np.isfinite(selection_values), selection_values, np.inf)
    if not np.isfinite(finite_selection[0]):
        raise FloatingPointError("non-finite Stage-4 hand-control objective")
    selected_index = int(np.argmin(finite_selection))
    selected = jnp.asarray(candidates[selected_index])
    return {
        "theta": selected,
        "trace": trace,
        "candidate_steps": candidate_steps,
        "selection_objectives": selection_values,
        "selected_candidate_index": selected_index,
        "selected_step": candidate_steps[selected_index],
        "initial_adaptation_objective": float(train_objective(geometry["theta0"])),
        "selected_adaptation_objective": float(train_objective(selected)),
        "initial_selection_objective": selection_values[0],
        "selected_selection_objective": selection_values[selected_index],
        "wall_seconds": time.perf_counter() - started,
    }


def evaluate_fiber(raw, bank, geometry, theta):
    values = design_path_metrics(
        raw, bank, geometry["common_mean"], theta, geometry["basis"]
    )
    states, weights, energy, forcing, ess, residual = values
    # q4 is first touched here, after candidate selection, as a hidden
    # diagnostic.  It is not an input to the designed observables.
    q4 = jax.vmap(lambda x, w: w @ paper.v_q4(x))(states, weights)
    coefficients = observable_coefficients(theta, geometry["basis"])
    endpoint_gap = coefficients @ jnp.asarray(geometry["raw_endpoint_gap"])
    return {
        "construction_objective": float(construction_objective(
            raw, bank, geometry["common_mean"], theta, geometry["basis"]
        )),
        "integrated_correction_energy": float(jnp.trapezoid(energy, TIMES)),
        "integrated_forcing_power": float(jnp.trapezoid(forcing, TIMES)),
        "minimum_ess": float(jnp.min(ess)),
        "maximum_calibration_residual": float(jnp.max(residual)),
        "endpoint_equivalence_residual": float(jnp.linalg.norm(endpoint_gap)),
        "q4_path": np.asarray(q4).tolist(),
        "coefficients": np.asarray(coefficients).tolist(),
    }


def gradient_check(raw, geometry, bank):
    objective = jax.jit(lambda theta: construction_objective(
        raw, bank, geometry["common_mean"], theta, geometry["basis"]
    ))
    theta = geometry["theta0"] + 0.01 * jnp.reshape(
        jnp.sin(jnp.arange(geometry["theta0"].size, dtype=jnp.float64)),
        geometry["theta0"].shape,
    )
    direction = jnp.reshape(
        jnp.cos(jnp.arange(theta.size, dtype=jnp.float64) + 0.3), theta.shape
    )
    direction /= jnp.linalg.norm(direction)
    autodiff = float(jnp.sum(jax.grad(objective)(theta) * direction))
    step = 2e-5
    finite = float(
        (objective(theta + step * direction) - objective(theta - step * direction))
        / (2.0 * step)
    )
    relative = abs(autodiff - finite) / max(abs(autodiff), abs(finite), 1e-9)
    return {
        "autodiff_directional_derivative": autodiff,
        "finite_difference_directional_derivative": finite,
        "finite_difference_step": step,
        "relative_error": relative,
        "passed": relative < 2e-3,
    }


def run_seed(source_report, optimizer_steps):
    seed = int(source_report["seed"])
    populations = paper.build_physical_populations(seed + 10000, False)
    endpoint_hash_before = array_hash(
        populations["minus"], populations["plus"],
        populations["minus_weights"], populations["plus_weights"],
    )
    geometry = endpoint_geometry(populations)
    raw = jnp.asarray(source_report["schedules"]["optimized_multi"]["raw"])
    roles = {name: make_role(populations, seed, name) for name in OFFSETS}
    if len({fingerprint for _, fingerprint in roles.values()}) != len(roles):
        raise RuntimeError("Stage-4 bank roles overlap")
    print(f"[stage4] seed {seed}: differentiable observable design", flush=True)
    optimization = optimize_observables(
        raw, geometry, roles["adaptation"][0], roles["selection"][0],
        optimizer_steps,
    )
    selected = optimization.pop("theta")
    methods = {
        "hand": evaluate_fiber(raw, roles["evaluation"][0], geometry, geometry["theta0"]),
        "designed": evaluate_fiber(raw, roles["evaluation"][0], geometry, selected),
    }
    hand_coefficients = np.asarray(observable_coefficients(
        geometry["theta0"], geometry["basis"]
    ))
    hand_projector = hand_coefficients.T @ np.linalg.solve(
        hand_coefficients @ hand_coefficients.T, hand_coefficients
    )
    original = np.eye(len(DICTIONARY_CENTERS), dtype=np.float64)[:3]
    original_projector = original.T @ original
    hand_subspace_error = float(np.max(np.abs(
        hand_projector - original_projector
    )))
    endpoint_hash_after = array_hash(
        populations["minus"], populations["plus"],
        populations["minus_weights"], populations["plus_weights"],
    )
    q4_gap = populations["plus_q4"] - populations["minus_q4"]
    return {
        "seed": seed,
        "frozen_schedule_raw": np.asarray(raw).tolist(),
        "endpoint_law_hash_before": endpoint_hash_before,
        "endpoint_law_hash_after": endpoint_hash_after,
        "endpoint_q4": {
            "minus": populations["minus_q4"],
            "plus": populations["plus_q4"],
            "gap": q4_gap,
        },
        "dictionary_endpoint_gap_norm": float(np.linalg.norm(
            geometry["raw_endpoint_gap"]
        )),
        "bank_fingerprints": {name: value[1] for name, value in roles.items()},
        "hand_observable_subspace_max_abs_error": hand_subspace_error,
        "optimization": {
            **optimization,
            "selected_coefficients": methods["designed"]["coefficients"],
        },
        "methods": methods,
    }


def mean_ci(values):
    return paper.mean_ci([float(value) for value in values])


def aggregate(reports):
    metrics = (
        "construction_objective", "integrated_correction_energy",
        "integrated_forcing_power", "minimum_ess",
        "maximum_calibration_residual", "endpoint_equivalence_residual",
    )
    methods = {
        method: {
            metric: mean_ci([r["methods"][method][metric] for r in reports])
            for metric in metrics
        }
        for method in METHODS
    }
    contrasts = {
        metric: mean_ci([
            r["methods"]["designed"][metric] - r["methods"]["hand"][metric]
            for r in reports
        ])
        for metric in metrics
    }
    primary = contrasts["construction_objective"]
    return {
        "methods": methods,
        "designed_minus_hand": contrasts,
        "interpretation": {
            "designed_mean_improves_hand": primary["mean"] < 0.0,
            "designed_ci_supports_improvement": primary["ci95_high"] < 0.0,
        },
    }


def write_csv(summary, output):
    fields = (
        "construction_objective", "integrated_correction_energy",
        "integrated_forcing_power", "minimum_ess",
        "maximum_calibration_residual", "endpoint_equivalence_residual",
    )
    rows = []
    for report in summary["seed_reports"]:
        for method in METHODS:
            rows.append({
                "seed": report["seed"], "method": method,
                **{name: report["methods"][method][name] for name in fields},
            })
    with (output / "stage4_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(summary, output):
    plt.rcParams.update({
        "figure.facecolor": "#f4f1ea", "axes.facecolor": "#fffdf8",
        "axes.grid": True, "grid.alpha": 0.2,
    })
    colors = {"hand": "#457b9d", "designed": "#e76f51"}
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        ("construction_objective", "integrated_correction_energy", "minimum_ess"),
        ("Construction objective", "Correction energy", "Minimum ESS"),
    ):
        stats = [summary["aggregate"]["methods"][method][metric] for method in METHODS]
        means = [row["mean"] for row in stats]
        errors = [
            [mean - row["ci95_low"] for mean, row in zip(means, stats)],
            [row["ci95_high"] - mean for mean, row in zip(means, stats)],
        ]
        ax.bar(METHODS, means, yerr=errors, color=[colors[m] for m in METHODS], capsize=4)
        ax.set_title(title)
    figure.suptitle("Stage 4: differentiable moment-fiber design", fontweight="bold")
    figure.savefig(output / "stage4_summary.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for report in summary["seed_reports"]:
        axes[0].plot(
            report["optimization"]["candidate_steps"],
            report["optimization"]["selection_objectives"], "o-",
            label=str(report["seed"]),
        )
        for method in METHODS:
            axes[1].plot(
                np.asarray(TIMES), report["methods"][method]["q4_path"],
                "o-", color=colors[method], alpha=0.35,
            )
    axes[0].set(title="Independent selection bank", xlabel="optimizer step", ylabel="objective")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set(title="Evaluation-only hidden q4", xlabel="time", ylabel="projected q4")
    figure.savefig(output / "stage4_selection_q4.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_report(summary, output):
    aggregate_results = summary["aggregate"]
    primary = aggregate_results["designed_minus_hand"]["construction_objective"]
    lines = [
        "# MFSI Stage 4: differentiable moment-fiber design", "",
        "## Controlled intervention", "",
        "The endpoint configurations and weights, hidden endpoint q4 gap, angular-sort endpoint coupling, selected reference schedule, six construction times, converged exponential-calibration equations, and eight-function Ritz realization dictionary were fixed. The only optimized object was the rank-three subspace defining the measured observables.", "",
        "The candidate dictionary contains eleven radial RBF measurements and nests the three hand observables exactly. It contains no angular descriptor or q4. An endpoint-nullspace constraint makes the two fixed weighted endpoint laws exactly equivalent under every candidate, while row-orthonormal coefficients prevent scale or rank collapse. Adaptation, checkpoint selection, and evaluation used disjoint bridge banks.", "",
        "## Untouched evaluation banks", "",
        "| fiber | construction objective | correction energy | forcing power | minimum ESS |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = aggregate_results["methods"][method]
        lines.append(
            f"| {method} | {row['construction_objective']['mean']:.8g} | "
            f"{row['integrated_correction_energy']['mean']:.8g} | "
            f"{row['integrated_forcing_power']['mean']:.8g} | "
            f"{row['minimum_ess']['mean']:.8g} |"
        )
    lines.extend([
        "", "## Primary paired effect", "",
        f"Designed minus hand construction objective: `{primary['mean']:.8g}` "
        f"(95% interval `{primary['ci95_low']:.8g}` to `{primary['ci95_high']:.8g}`).", "",
        "The construction objective is the established integrated correction energy plus 0.02 times forcing power and the unchanged ESS-floor penalty.", "",
        "## Interpretation", "",
        (
            "The held-out paired interval supports a better three-observable equivalence class under the predeclared construction objective."
            if aggregate_results["interpretation"]["designed_ci_supports_improvement"]
            else "The experiment does not establish a held-out improvement over the hand-designed fiber."
        ), "",
        "q4 was accessed only after selection and is reported strictly as a hidden evaluation diagnostic; no rollout, schedule optimization, coupling change, or neural training was performed.", "",
    ])
    (output / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="401 402 403 404 405")
    parser.add_argument("--optimizer-steps", type=int, default=OPTIMIZER_STEPS)
    parser.add_argument("--aggregate-existing", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split()]
    source = json.loads(args.source.read_text())
    protocol = json.loads(args.protocol.read_text())
    source_by_seed = {int(row["seed"]): row for row in source["seed_reports"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    started = time.perf_counter()
    for seed in seeds:
        path = args.output_dir / f"seed_{seed}.json"
        if args.aggregate_existing:
            report = json.loads(path.read_text())
        else:
            report = run_seed(source_by_seed[seed], args.optimizer_steps)
            path.write_text(json.dumps(report, indent=2) + "\n")
        reports.append(report)

    # A fourth, independent bank validates differentiation without entering
    # adaptation, selection, or evaluation.
    first_seed = seeds[0]
    first_populations = paper.build_physical_populations(first_seed + 10000, False)
    first_geometry = endpoint_geometry(first_populations)
    gradient_rng = np.random.default_rng(first_seed + 104000)
    gradient_bank = paper.make_bridge_bank(
        first_populations, gradient_rng, np.asarray(TIMES), 48
    )
    first_raw = jnp.asarray(source_by_seed[first_seed]["schedules"]["optimized_multi"]["raw"])
    gradient = gradient_check(first_raw, first_geometry, gradient_bank)
    aggregate_results = aggregate(reports)
    validation = {
        "gradient": gradient,
        "endpoint_laws_unchanged": all(
            row["endpoint_law_hash_before"] == row["endpoint_law_hash_after"]
            for row in reports
        ),
        "endpoint_equivalence_enforced": max(
            row["methods"][method]["endpoint_equivalence_residual"]
            for row in reports for method in METHODS
        ) < 1e-10,
        "hidden_q4_gap_preserved": min(
            row["endpoint_q4"]["gap"] for row in reports
        ) > 0.30,
        "bank_roles_distinct": all(
            len(set(row["bank_fingerprints"].values())) == 3 for row in reports
        ),
        "hand_candidate_nested": all(
            row["optimization"]["candidate_steps"][0] == 0 for row in reports
        ),
        "hand_observable_subspace_matches_original": max(
            row["hand_observable_subspace_max_abs_error"] for row in reports
        ) < 2e-10,
        "q4_used_for_adaptation_or_selection": False,
        "rollout_used": False,
        "schedule_optimized": False,
        "coupling_changed": False,
        "neural_model_trained": False,
    }
    summary = {
        "experiment": "stage4-differentiable-moment-fiber-design",
        "stage": 4,
        "scientific_replication_n": len(reports),
        "seeds": seeds,
        "source": str(args.source.relative_to(ROOT)),
        "protocol": protocol,
        "sole_scientific_intervention": "rank-three measured-observable subspace",
        "configuration": {
            "particle_count": paper.N_PARTICLES,
            "state_dimension": paper.STATE_DIMENSION,
            "observable_count": 3,
            "candidate_dictionary_size": int(len(DICTIONARY_CENTERS)),
            "candidate_dictionary": "radial pair-distance RBFs only",
            "dictionary_centers": np.asarray(DICTIONARY_CENTERS).tolist(),
            "construction_times": np.asarray(TIMES).tolist(),
            "adaptation_count_per_time": ADAPTATION_COUNT,
            "selection_count_per_time": SELECTION_COUNT,
            "evaluation_count_per_time": EVALUATION_COUNT,
            "optimizer": "Adam",
            "optimizer_steps": args.optimizer_steps,
            "learning_rate": LEARNING_RATE,
            "candidate_interval": CANDIDATE_INTERVAL,
            "ess_floor": ESS_FLOOR,
            "forcing_weight": FORCING_WEIGHT,
            "calibration_steps": CALIBRATION_STEPS,
            "bank_seed_offsets": OFFSETS,
            "primary_metric": "held-out construction objective",
        },
        "frozen": {
            "endpoint_configurations": True,
            "endpoint_weights": True,
            "hidden_endpoint_q4_difference": True,
            "endpoint_coupling": True,
            "reference_schedule": True,
            "construction_times": True,
            "calibration_equations_steps_and_ridge": True,
            "ritz_dictionary_and_ridges": True,
            "optimizer_protocol_across_methods": True,
        },
        "validation": validation,
        "elapsed_seconds": time.perf_counter() - started,
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
        "designed_minus_hand": aggregate_results["designed_minus_hand"],
    }, indent=2))
    print(f"outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
