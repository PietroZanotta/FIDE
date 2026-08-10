"""Phase-3B reference CNN training after a confirmed method-blind path design."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .benchmark_design import ROOT
from .feasibility import calibrate_iprojection_instrumented, solve_target_hull_lp
from .field_transport import (
    init_periodic_reference_cnn,
    maximal_same_index_coupling,
    noisy_field_interpolant,
    periodic_reference_cnn,
    smooth_hidden_observables,
    standardized_noise_bank,
)
from .observables import ShellDefinition, field_observables
from .phase2_continuation import _write_csv, _write_json
from .phase3_reference_design import DEFAULT_CONFIG, _read_json
from .simulator import generate_initial_conditions, simulate


def _make_endpoint_bank(config, phase2, candidate, seed_start, count):
    design = _read_json(ROOT / config["source_design_config"])
    grid, ic, simulator = design["grid"], design["initial_conditions"], design["simulator"]
    seeds = np.arange(int(seed_start), int(seed_start) + int(count))
    initial_u, initial_v, _ = generate_initial_conditions(
        seeds, height=int(grid["height"]), width=int(grid["width"]),
        blob_count=tuple(ic["blob_count"]), radius_range=tuple(ic["radius_range"]),
        u_depletion_range=tuple(ic["u_depletion_range"]),
        v_amplitude_range=tuple(ic["v_amplitude_range"]), noise_std=float(ic["noise_std"]),
    )
    tiled_u = np.concatenate([initial_u, initial_u])
    tiled_v = np.concatenate([initial_v, initial_v])
    feeds = np.concatenate([
        np.full(count, float(candidate["spot_feed"])),
        np.full(count, float(candidate["labyrinth_feed"])),
    ])
    kills = np.concatenate([
        np.full(count, float(candidate["spot_kill"])),
        np.full(count, float(candidate["labyrinth_kill"])),
    ])
    _, final_v = simulate(
        tiled_u, tiled_v, feed=feeds, kill=kills,
        diffusion_u=float(simulator["diffusion_u"]), diffusion_v=float(simulator["diffusion_v"]),
        dt=float(simulator["dt"]), physical_time=float(simulator["physical_time"]),
        spacing=float(grid["spacing"]),
    )
    final_v = np.asarray(final_v)
    return final_v[:count], final_v[count:], seeds


def _calibrate_endpoints(minus, plus, target, center, scale, shells, calibration):
    components = ("mean", "second_moment")[:len(target)]
    minus_phi = np.asarray(field_observables(jnp.asarray(minus), shells, components), dtype=np.float64)
    plus_phi = np.asarray(field_observables(jnp.asarray(plus), shells, components), dtype=np.float64)
    minus_std, plus_std = (minus_phi - center) / scale, (plus_phi - center) / scale
    minus_hull, plus_hull = solve_target_hull_lp(minus_std, target), solve_target_hull_lp(plus_std, target)
    if not minus_hull["success"] or not plus_hull["success"]:
        raise RuntimeError("fixed Phase-2 target is outside a Phase-3B endpoint bank hull")
    minus_cal = calibrate_iprojection_instrumented(
        minus_std, target, tolerance=float(calibration["residual_tolerance"]),
        max_iterations=int(calibration["maximum_iterations"]),
    )
    plus_cal = calibrate_iprojection_instrumented(
        plus_std, target, tolerance=float(calibration["residual_tolerance"]),
        max_iterations=int(calibration["maximum_iterations"]),
    )
    return minus_cal, plus_cal, {"minus": minus_hull, "plus": plus_hull}


def _trainable_parameters(full):
    return {
        "layers": [{"weight": layer["weight"], "bias": layer["bias"]} for layer in full["layers"]],
        "output": {"weight": full["output"]["weight"], "bias": full["output"]["bias"]},
    }


def _assemble_parameters(trainable, dilations, time_frequencies, kernel_size):
    return {
        "layers": [
            {**layer, "dilation": dilation}
            for layer, dilation in zip(trainable["layers"], dilations)
        ],
        "output": trainable["output"], "time_frequencies": time_frequencies,
        "kernel_size": kernel_size,
    }


def _sample_batch(rng, minus, plus, coupling, batch_size, amplitude):
    flat = coupling.ravel()
    selected = rng.choice(len(flat), size=batch_size, p=flat)
    minus_index, plus_index = np.unravel_index(selected, coupling.shape)
    noise = standardized_noise_bank(batch_size, minus.shape[1:], int(rng.integers(0, 2**31 - 1)))
    times = rng.uniform(0.0, 1.0, size=batch_size).astype(np.float32)
    states, targets = noisy_field_interpolant(
        minus[minus_index], plus[plus_index], noise, times, amplitude
    )
    return np.asarray(times), np.asarray(states), np.asarray(targets)


def _evaluate_loss(trainable, architecture, rng, minus, plus, coupling, amplitude, count, batch_size):
    squared_error, squared_target = 0.0, 0.0
    seen = 0
    apply = jax.jit(lambda t, x: periodic_reference_cnn(
        _assemble_parameters(trainable, **architecture), t, x
    ))
    while seen < count:
        current = min(batch_size, count - seen)
        times, states, targets = _sample_batch(rng, minus, plus, coupling, current, amplitude)
        prediction = np.asarray(apply(jnp.asarray(times), jnp.asarray(states)))
        squared_error += float(np.sum((prediction - targets) ** 2))
        squared_target += float(np.sum(targets ** 2))
        seen += current
    return squared_error / (seen * np.prod(minus.shape[1:])), squared_target / (seen * np.prod(minus.shape[1:]))


def _rollout(trainable, architecture, initial, steps):
    parameters = _assemble_parameters(trainable, **architecture)
    dt = jnp.asarray(1.0 / steps, dtype=initial.dtype)

    @jax.jit
    def integrate(fields):
        def body(index, state):
            time_value = index * dt
            first = periodic_reference_cnn(parameters, time_value, state)
            proposal = state + dt * first
            second = periodic_reference_cnn(parameters, time_value + dt, proposal)
            return state + 0.5 * dt * (first + second)
        return jax.lax.fori_loop(0, steps, body, fields)
    return np.asarray(integrate(jnp.asarray(initial)))


def run_reference_training(config_path: Path = DEFAULT_CONFIG) -> dict:
    config = _read_json(config_path.resolve())
    output = ROOT / config["output_directory"]
    confirmation = _read_json(output / "selected_confirmation_summary.json")
    if not confirmation["phase3a_pass_confirmed"]:
        raise RuntimeError("Phase 3A must be independently confirmed before reference training")
    phase2 = _read_json(ROOT / config["source_phase2_directory"] / "large_bank_phase2_summary.json")
    selected = confirmation["screen_selected_path"]
    candidate = phase2["passing_candidates_ranked"][int(selected["candidate_rank"]) - 1]
    reference = config["phase3b_reference"]
    design = _read_json(ROOT / config["source_design_config"])
    obs = design["observables"]
    shells = ShellDefinition(tuple(obs["shell_centers_cycles_per_pixel"]), tuple(obs["shell_widths_cycles_per_pixel"]))
    dimension = int(candidate["observation_dimension"])
    center = np.asarray(phase2["standardization"]["center"][:dimension])
    scale = np.asarray(phase2["standardization"]["scale"][:dimension])
    target = np.asarray(candidate["target_standardized"])
    amplitude = float(selected["schedule_amplitude"])
    train_minus, train_plus, train_seeds = _make_endpoint_bank(
        config, phase2, candidate, int(reference["training_initial_condition_seed_start"]),
        int(reference["training_initial_condition_count"]),
    )
    validation_minus, validation_plus, validation_seeds = _make_endpoint_bank(
        config, phase2, candidate, int(reference["model_selection_seed_start"]),
        int(reference["model_selection_initial_condition_count"]),
    )
    design_seeds = set(range(32000, 32256))
    if design_seeds & set(map(int, train_seeds)) or design_seeds & set(map(int, validation_seeds)):
        raise RuntimeError("Phase-3B seeds overlap Phase-2 design seeds")
    if set(map(int, train_seeds)) & set(map(int, validation_seeds)):
        raise RuntimeError("training and model-selection IC seeds overlap")
    train_minus_cal, train_plus_cal, train_hulls = _calibrate_endpoints(
        train_minus, train_plus, target, center, scale, shells, config["calibration"]
    )
    validation_minus_cal, validation_plus_cal, validation_hulls = _calibrate_endpoints(
        validation_minus, validation_plus, target, center, scale, shells, config["calibration"]
    )
    train_coupling = maximal_same_index_coupling(train_minus_cal["weights"], train_plus_cal["weights"])
    validation_coupling = maximal_same_index_coupling(
        validation_minus_cal["weights"], validation_plus_cal["weights"]
    )
    hidden_channels = tuple(map(int, reference["hidden_channels"]))
    dilations = tuple(map(int, reference["dilations"]))
    full = init_periodic_reference_cnn(
        jax.random.PRNGKey(int(reference["optimizer_seed"])),
        hidden_channels=hidden_channels, dilations=dilations, dtype=jnp.float32,
    )
    trainable = _trainable_parameters(full)
    architecture = {
        "dilations": dilations, "time_frequencies": int(full["time_frequencies"]),
        "kernel_size": int(full["kernel_size"]),
    }
    first = jax.tree_util.tree_map(jnp.zeros_like, trainable)
    second = jax.tree_util.tree_map(jnp.zeros_like, trainable)
    learning_rate = float(reference["learning_rate"])

    @jax.jit
    def train_step(parameters, first_moment, second_moment, step, times, states, targets):
        def objective(value):
            prediction = periodic_reference_cnn(
                _assemble_parameters(value, **architecture), times, states
            )
            return jnp.mean((prediction - targets) ** 2)
        loss, gradient = jax.value_and_grad(objective)(parameters)
        first_moment = jax.tree_util.tree_map(lambda old, grad: 0.9 * old + 0.1 * grad, first_moment, gradient)
        second_moment = jax.tree_util.tree_map(lambda old, grad: 0.999 * old + 0.001 * grad * grad, second_moment, gradient)
        corrected_first = jax.tree_util.tree_map(lambda value: value / (1.0 - 0.9 ** step), first_moment)
        corrected_second = jax.tree_util.tree_map(lambda value: value / (1.0 - 0.999 ** step), second_moment)
        parameters = jax.tree_util.tree_map(
            lambda value, m, v: value - learning_rate * m / (jnp.sqrt(v) + 1e-8),
            parameters, corrected_first, corrected_second,
        )
        grad_norm = jnp.sqrt(sum(jnp.sum(value * value) for value in jax.tree_util.tree_leaves(gradient)))
        return parameters, first_moment, second_moment, loss, grad_norm

    rng = np.random.default_rng(int(reference["optimizer_seed"]))
    trace, start = [], time.perf_counter()
    steps, batch_size = int(reference["training_steps"]), int(reference["batch_size"])
    for step in range(1, steps + 1):
        times, states, targets = _sample_batch(
            rng, train_minus, train_plus, train_coupling, batch_size, amplitude
        )
        trainable, first, second, loss, grad_norm = train_step(
            trainable, first, second, step, jnp.asarray(times), jnp.asarray(states), jnp.asarray(targets)
        )
        if step == 1 or step % 50 == 0 or step == steps:
            trace.append({"step": step, "training_mse_per_pixel": float(loss), "gradient_norm": float(grad_norm)})
        if step % 500 == 0:
            print(f"reference CNN step {step}/{steps}: mse={float(loss):.6e}", flush=True)
    jax.block_until_ready(trainable["output"]["weight"])
    training_seconds = time.perf_counter() - start
    evaluation_rng = np.random.default_rng(int(reference["optimizer_seed"]) + 1)
    heldout_mse, zero_mse = _evaluate_loss(
        trainable, architecture, evaluation_rng, validation_minus, validation_plus,
        validation_coupling, amplitude, int(reference["heldout_interpolant_count"]), batch_size,
    )
    rollout_rng = np.random.default_rng(int(reference["optimizer_seed"]) + 2)
    rollout_count = int(reference["rollout_sample_count"])
    initial_indices = rollout_rng.choice(
        len(validation_minus), size=rollout_count, p=validation_minus_cal["weights"]
    )
    rollout = _rollout(
        trainable, architecture, validation_minus[initial_indices], int(reference["rollout_steps"])
    )
    rollout_phi = np.asarray(field_observables(
        jnp.asarray(rollout), shells, ("mean", "second_moment")[:dimension]
    ))
    rollout_standardized_mean = ((rollout_phi - center) / scale).mean(axis=0)
    rollout_residual = rollout_standardized_mean - target
    threshold = float(phase2["fixed_threshold"])
    hidden_scales = np.asarray(_read_json(output / "linear_phase3a_summary.json")["hidden_observable_scales"])
    rollout_hidden = np.asarray(smooth_hidden_observables(jnp.asarray(rollout), threshold=threshold)).mean(axis=0)
    target_hidden_values = np.asarray(smooth_hidden_observables(jnp.asarray(validation_plus), threshold=threshold))
    target_hidden = validation_plus_cal["weights"] @ target_hidden_values
    hidden_endpoint_error = float(np.linalg.norm((rollout_hidden - target_hidden) / hidden_scales) / np.sqrt(len(hidden_scales)))
    loss_fraction = heldout_mse / zero_mse
    quality = reference["quality_gates"]
    loss_gate = loss_fraction <= float(quality["heldout_mse_fraction_of_zero_predictor"])
    rollout_gate = float(np.max(np.abs(rollout_residual))) <= float(quality["rollout_maximum_standardized_target_residual"])
    phase3_pass = bool(loss_gate and rollout_gate)
    checkpoint = {
        "trainable": jax.tree_util.tree_map(lambda value: np.asarray(value), trainable),
        "architecture": architecture, "candidate": candidate, "selected_path": selected,
        "center": center, "scale": scale, "target": target,
    }
    with (output / "reference_cnn_checkpoint.pkl").open("wb") as handle:
        pickle.dump(checkpoint, handle)
    np.savez_compressed(
        output / "reference_training_endpoint_banks.npz",
        training_minus=train_minus, training_plus=train_plus,
        validation_minus=validation_minus, validation_plus=validation_plus,
        training_minus_weights=train_minus_cal["weights"], training_plus_weights=train_plus_cal["weights"],
        validation_minus_weights=validation_minus_cal["weights"], validation_plus_weights=validation_plus_cal["weights"],
        training_seeds=train_seeds, validation_seeds=validation_seeds,
    )
    endpoint_diagnostics = {
        "training": {"minus": train_minus_cal, "plus": train_plus_cal, "hulls": train_hulls},
        "model_selection": {"minus": validation_minus_cal, "plus": validation_plus_cal, "hulls": validation_hulls},
    }
    # Avoid duplicating large weight arrays in JSON; they are in the NPZ above.
    for split in endpoint_diagnostics.values():
        for side in ("minus", "plus"):
            split[side] = {key: value for key, value in split[side].items() if key != "weights"}
        for side in ("minus", "plus"):
            split["hulls"][side] = {key: value for key, value in split["hulls"][side].items() if key != "weights"}
    result = {
        "status": "phase3_pass" if phase3_pass else "phase3_reference_quality_failed",
        "phase3a_confirmation": confirmation, "candidate": candidate,
        "training_initial_condition_seeds": [int(train_seeds[0]), int(train_seeds[-1])],
        "model_selection_initial_condition_seeds": [int(validation_seeds[0]), int(validation_seeds[-1])],
        "seed_sets_disjoint": True, "training_steps": steps, "training_seconds": training_seconds,
        "heldout_mse_per_pixel": heldout_mse, "heldout_zero_predictor_mse_per_pixel": zero_mse,
        "heldout_mse_fraction_of_zero_predictor": loss_fraction,
        "heldout_flow_matching_gate_pass": loss_gate,
        "rollout_maximum_standardized_target_residual": float(np.max(np.abs(rollout_residual))),
        "rollout_standardized_target_residual": rollout_residual,
        "rollout_hidden_endpoint_error_standardized_rms": hidden_endpoint_error,
        "rollout_field_minimum": float(rollout.min()), "rollout_field_maximum": float(rollout.max()),
        "rollout_endpoint_gate_pass": rollout_gate, "phase3_pass": phase3_pass,
        "phase4_authorized": phase3_pass, "deep_ritz_training_performed": False,
        "method_comparison_performed": False, "training_trace": trace,
        "endpoint_calibration_diagnostics": endpoint_diagnostics,
    }
    _write_csv(output / "reference_cnn_training_trace.csv", trace)
    _write_json(output / "phase3_reference_training_summary.json", result)
    return result


def evaluate_saved_reference(config_path: Path = DEFAULT_CONFIG) -> dict:
    """Reproduce held-out/rollout evaluation from an already-written checkpoint."""
    config = _read_json(config_path.resolve())
    output = ROOT / config["output_directory"]
    with (output / "reference_cnn_checkpoint.pkl").open("rb") as handle:
        checkpoint = pickle.load(handle)
    endpoints = np.load(output / "reference_training_endpoint_banks.npz")
    trainable, architecture = checkpoint["trainable"], checkpoint["architecture"]
    candidate, selected = checkpoint["candidate"], checkpoint["selected_path"]
    center, scale, target = map(np.asarray, (checkpoint["center"], checkpoint["scale"], checkpoint["target"]))
    reference = config["phase3b_reference"]
    validation_minus = np.asarray(endpoints["validation_minus"])
    validation_plus = np.asarray(endpoints["validation_plus"])
    validation_minus_weights = np.asarray(endpoints["validation_minus_weights"])
    validation_plus_weights = np.asarray(endpoints["validation_plus_weights"])
    validation_coupling = maximal_same_index_coupling(validation_minus_weights, validation_plus_weights)
    evaluation_rng = np.random.default_rng(int(reference["optimizer_seed"]) + 1)
    heldout_mse, zero_mse = _evaluate_loss(
        trainable, architecture, evaluation_rng, validation_minus, validation_plus,
        validation_coupling, float(selected["schedule_amplitude"]),
        int(reference["heldout_interpolant_count"]), int(reference["batch_size"]),
    )
    rollout_rng = np.random.default_rng(int(reference["optimizer_seed"]) + 2)
    rollout_count = int(reference["rollout_sample_count"])
    initial_indices = rollout_rng.choice(
        len(validation_minus), size=rollout_count, p=validation_minus_weights
    )
    rollout = _rollout(
        trainable, architecture, validation_minus[initial_indices], int(reference["rollout_steps"])
    )
    design = _read_json(ROOT / config["source_design_config"])
    obs = design["observables"]
    shells = ShellDefinition(tuple(obs["shell_centers_cycles_per_pixel"]), tuple(obs["shell_widths_cycles_per_pixel"]))
    dimension = int(candidate["observation_dimension"])
    rollout_phi = np.asarray(field_observables(
        jnp.asarray(rollout), shells, ("mean", "second_moment")[:dimension]
    ))
    rollout_standardized_mean = ((rollout_phi - center) / scale).mean(axis=0)
    rollout_residual = rollout_standardized_mean - target
    phase2 = _read_json(ROOT / config["source_phase2_directory"] / "large_bank_phase2_summary.json")
    threshold = float(phase2["fixed_threshold"])
    hidden_scales = np.asarray(_read_json(output / "linear_phase3a_summary.json")["hidden_observable_scales"])
    rollout_hidden = np.asarray(smooth_hidden_observables(jnp.asarray(rollout), threshold=threshold)).mean(axis=0)
    target_hidden_values = np.asarray(smooth_hidden_observables(jnp.asarray(validation_plus), threshold=threshold))
    target_hidden = validation_plus_weights @ target_hidden_values
    hidden_endpoint_error = float(np.linalg.norm((rollout_hidden - target_hidden) / hidden_scales) / np.sqrt(len(hidden_scales)))
    loss_fraction = float(heldout_mse / zero_mse)
    quality = reference["quality_gates"]
    loss_gate = bool(loss_fraction <= float(quality["heldout_mse_fraction_of_zero_predictor"]))
    rollout_maximum = float(np.max(np.abs(rollout_residual)))
    rollout_gate = bool(rollout_maximum <= float(quality["rollout_maximum_standardized_target_residual"]))
    phase3_pass = bool(loss_gate and rollout_gate)
    with (output / "reference_cnn_training_trace.csv").open() as handle:
        trace = [
            {"step": int(row["step"]), "training_mse_per_pixel": float(row["training_mse_per_pixel"]),
             "gradient_norm": float(row["gradient_norm"])}
            for row in csv.DictReader(handle)
        ]
    result = {
        "status": "phase3_pass" if phase3_pass else "phase3_reference_quality_failed",
        "candidate": candidate, "selected_path": selected,
        "evaluation_source": "saved checkpoint from the single completed 4000-step training run",
        "training_rerun_for_evaluation": False, "training_steps": int(reference["training_steps"]),
        "training_seconds": None,
        "training_seconds_unavailable_reason": "initial summary serialization failed after checkpoint creation",
        "training_initial_condition_seeds": [int(endpoints["training_seeds"][0]), int(endpoints["training_seeds"][-1])],
        "model_selection_initial_condition_seeds": [int(endpoints["validation_seeds"][0]), int(endpoints["validation_seeds"][-1])],
        "seed_sets_disjoint": True,
        "heldout_mse_per_pixel": float(heldout_mse),
        "heldout_zero_predictor_mse_per_pixel": float(zero_mse),
        "heldout_mse_fraction_of_zero_predictor": loss_fraction,
        "heldout_flow_matching_gate_pass": loss_gate,
        "rollout_maximum_standardized_target_residual": rollout_maximum,
        "rollout_standardized_target_residual": rollout_residual,
        "rollout_hidden_endpoint_error_standardized_rms": hidden_endpoint_error,
        "rollout_field_minimum": float(rollout.min()), "rollout_field_maximum": float(rollout.max()),
        "rollout_endpoint_gate_pass": rollout_gate, "phase3_pass": phase3_pass,
        "phase4_authorized": phase3_pass, "deep_ritz_training_performed": False,
        "method_comparison_performed": False, "training_trace": trace,
    }
    _write_json(output / "phase3_reference_training_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--evaluate-saved", action="store_true")
    args = parser.parse_args()
    result = evaluate_saved_reference(args.config) if args.evaluate_saved else run_reference_training(args.config)
    print(json.dumps(result, indent=2, default=lambda value: np.asarray(value).tolist()))


if __name__ == "__main__":
    main()
