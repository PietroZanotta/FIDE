"""Corrected Gray--Scott Phase-3B reference-flow audit and diagnostics (v8)."""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .benchmark_design import ROOT
from .feasibility import calibrate_iprojection_instrumented, solve_target_hull_lp
from .field_transport import (
    _time_channels,
    maximal_same_index_coupling,
    noisy_field_interpolant,
    periodic_reference_cnn,
    periodic_conv2d,
    smooth_hidden_observables,
    standardized_noise_bank,
)
from .observables import ShellDefinition, field_observables
from .phase2_continuation import _write_csv, _write_json
from .phase3_reference_design import _read_json
from .phase3_reference_training import (
    _assemble_parameters,
    _calibrate_endpoints,
    _make_endpoint_bank,
)


DEFAULT_CONFIG = ROOT / "configs" / "expC_grayscott_phase3_quality_v8.json"
DEFAULT_SWEEP_CONFIG = ROOT / "configs" / "expC_grayscott_phase3_quality_v8_sweep.json"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_manifest(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): _hash_file(path)
        for path in sorted(directory.rglob("*")) if path.is_file()
    }


def sample_frozen_bridge(
    rng: np.random.Generator,
    minus: np.ndarray,
    plus: np.ndarray,
    coupling: np.ndarray,
    times: np.ndarray,
    amplitude: float = 0.07,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sample states/analytic derivatives using shared endpoints and shared Z."""
    times = np.asarray(times, dtype=np.float32)
    selected = rng.choice(coupling.size, size=len(times), p=coupling.ravel())
    minus_indices, plus_indices = np.unravel_index(selected, coupling.shape)
    noise_seed = int(rng.integers(0, 2**31 - 1))
    noise = standardized_noise_bank(len(times), minus.shape[1:], noise_seed)
    states, targets = noisy_field_interpolant(
        minus[minus_indices], plus[plus_indices], noise, times, amplitude
    )
    return np.asarray(states), np.asarray(targets), {
        "minus_indices": minus_indices, "plus_indices": plus_indices,
        "noise": noise, "noise_seed": noise_seed, "times": times,
    }


def bridge_target_consistency(
    minus: np.ndarray, plus: np.ndarray, noise: np.ndarray, times: np.ndarray,
    amplitude: float = 0.07, epsilon: float = 1e-4,
) -> dict:
    analytic_state, analytic_target = noisy_field_interpolant(minus, plus, noise, times, amplitude)
    plus_state, _ = noisy_field_interpolant(minus, plus, noise, times + epsilon, amplitude)
    minus_state, _ = noisy_field_interpolant(minus, plus, noise, times - epsilon, amplitude)
    finite = (plus_state - minus_state) / (2.0 * epsilon)
    error = np.asarray(finite - analytic_target)
    relative = float(np.linalg.norm(error) / max(np.linalg.norm(np.asarray(analytic_target)), 1e-30))
    formula = (
        np.asarray(plus) - np.asarray(minus)
        + amplitude * np.pi * np.cos(np.pi * np.asarray(times))[:, None, None, None] * np.asarray(noise)
    )
    return {
        "relative_finite_difference_error": relative,
        "maximum_absolute_finite_difference_error": float(np.max(np.abs(error))),
        "maximum_analytic_formula_error": float(np.max(np.abs(np.asarray(analytic_target) - formula))),
        "state_dtype": str(np.asarray(analytic_state).dtype),
        "target_dtype": str(np.asarray(analytic_target).dtype),
        "same_noise_used_for_state_and_target": True,
    }


def weighted_mmd2_four_weight(
    features_a: np.ndarray, weights_a: np.ndarray,
    features_b: np.ndarray, weights_b: np.ndarray,
) -> float:
    """Established four-weight, median-RBF MMD generalized to field features."""
    a = jnp.asarray(features_a, dtype=jnp.float32)
    b = jnp.asarray(features_b, dtype=jnp.float32)
    wa = jnp.asarray(weights_a, dtype=jnp.float32); wa /= jnp.sum(wa)
    wb = jnp.asarray(weights_b, dtype=jnp.float32); wb /= jnp.sum(wb)
    combined = jnp.concatenate([a, b], axis=0)
    distances = jnp.sum((combined[:, None] - combined[None, :]) ** 2, axis=-1)
    positive = distances[distances > 0]
    bandwidth = jnp.maximum(jnp.median(positive), 1e-8)
    def kernel(x, y):
        return jnp.exp(-jnp.sum((x[:, None] - y[None, :]) ** 2, axis=-1) / (2.0 * bandwidth))
    value = wa @ kernel(a, a) @ wa + wb @ kernel(b, b) @ wb - 2.0 * wa @ kernel(a, b) @ wb
    return float(jnp.maximum(value, 0.0))


def downsample_fields(fields: np.ndarray, factor: int = 4) -> np.ndarray:
    fields = np.asarray(fields)
    batch, channels, height, width = fields.shape
    if height % factor or width % factor:
        raise ValueError("field shape must be divisible by downsample factor")
    return fields.reshape(batch, channels, height // factor, factor, width // factor, factor).mean(axis=(3, 5))


def heun_rollout_snapshots(apply, initial: np.ndarray, steps: int, snapshot_times: np.ndarray) -> np.ndarray:
    """Integrate an autonomous-in-state time-dependent field and retain grid snapshots."""
    snapshot_times = np.asarray(snapshot_times, dtype=np.float64)
    indices = np.rint(snapshot_times * steps).astype(int)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("requested snapshot times collapse onto the same integration step")
    state = jnp.asarray(initial)
    snapshots = [np.asarray(state)] if indices[0] == 0 else []
    wanted = {int(value) for value in indices[1:]}
    dt = 1.0 / steps
    for index in range(steps):
        time = index * dt
        first = apply(time, state)
        proposal = state + dt * first
        second = apply(time + dt, proposal)
        state = state + 0.5 * dt * (first + second)
        if index + 1 in wanted:
            snapshots.append(np.asarray(state))
    return np.stack(snapshots)


def _checkpoint_apply(checkpoint):
    if checkpoint.get("model_kind", "v7_sequential") == "residual_periodic":
        return jax.jit(lambda time, fields: residual_reference_cnn(
            checkpoint["trainable"], checkpoint["architecture"], time, fields
        ))
    parameters = _assemble_parameters(checkpoint["trainable"], **checkpoint["architecture"])
    return jax.jit(lambda time, fields: periodic_reference_cnn(parameters, time, fields))


def _init_conv_array(key, output_channels, input_channels, dtype=jnp.float32):
    scale = jnp.asarray(np.sqrt(2.0 / (9 * input_channels)), dtype=dtype)
    return {
        "weight": scale * jax.random.normal(key, (output_channels, input_channels, 3, 3), dtype=dtype),
        "bias": jnp.zeros((output_channels,), dtype=dtype),
    }


def init_residual_reference_cnn(
    key, *, channels: int = 28, dilations=(1, 2, 4, 8, 4),
    time_frequencies: int = 3, dtype=jnp.float32,
):
    keys = jax.random.split(key, len(dilations) + 2)
    input_channels = 1 + 1 + 2 * time_frequencies
    return {
        "input": _init_conv_array(keys[0], channels, input_channels, dtype),
        "blocks": [_init_conv_array(k, channels, channels, dtype) for k in keys[1:-1]],
        "output": _init_conv_array(keys[-1], 1, channels, dtype),
    }, {
        "channels": channels, "dilations": tuple(map(int, dilations)),
        "time_frequencies": time_frequencies, "kernel_size": 3,
    }


def residual_reference_cnn(trainable, architecture, time, fields):
    fields = jnp.asarray(fields)
    batch, _, height, width = fields.shape
    time_features = _time_channels(
        time, batch, height, width, fields.dtype, int(architecture["time_frequencies"])
    )
    hidden = jax.nn.silu(periodic_conv2d(
        jnp.concatenate([fields, time_features], axis=1),
        trainable["input"]["weight"], trainable["input"]["bias"], 1,
    ))
    for block, dilation in zip(trainable["blocks"], architecture["dilations"]):
        update = jax.nn.silu(periodic_conv2d(
            hidden, block["weight"], block["bias"], int(dilation)
        ))
        hidden = (hidden + update) / jnp.sqrt(jnp.asarray(2.0, dtype=hidden.dtype))
    return periodic_conv2d(hidden, trainable["output"]["weight"], trainable["output"]["bias"], 1)


def _stratified_times(rng: np.random.Generator, count: int) -> np.ndarray:
    return ((np.arange(count) + rng.random(count)) / count).astype(np.float32)


def _fm_summary(prediction: np.ndarray, target: np.ndarray) -> dict:
    prediction, target = np.asarray(prediction), np.asarray(target)
    error = prediction - target
    mse = float(np.mean(error * error))
    zero = float(np.mean(target * target))
    pred_energy = float(np.mean(prediction * prediction))
    dot = float(np.mean(prediction * target))
    cosine = dot / max(np.sqrt(pred_energy * zero), 1e-30)
    error_spectrum = np.abs(np.fft.fft2(error[:, 0], norm="ortho")) ** 2
    target_spectrum = np.abs(np.fft.fft2(target[:, 0], norm="ortho")) ** 2
    height, width = error.shape[-2:]
    fy, fx = np.fft.fftfreq(height), np.fft.fftfreq(width)
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    bands = ((0.0, 0.125), (0.125, 0.25), (0.25, np.inf))
    spectral = []
    for low, high in bands:
        mask = (radius >= low) & (radius < high)
        error_energy = float(np.mean(error_spectrum[:, mask]))
        target_energy = float(np.mean(target_spectrum[:, mask]))
        spectral.append({
            "minimum_frequency": low, "maximum_frequency": high,
            "error_energy": error_energy, "target_energy": target_energy,
            "error_fraction_of_target": error_energy / max(target_energy, 1e-30),
        })
    return {
        "fm_mse_per_pixel": mse, "zero_predictor_mse_per_pixel": zero,
        "normalized_fm_mse": mse / max(zero, 1e-30),
        "target_velocity_rms": float(np.sqrt(zero)),
        "predicted_velocity_rms": float(np.sqrt(pred_energy)),
        "cosine_alignment": cosine, "spatial_frequency_diagnostics": spectral,
    }


def evaluate_fm_bank(
    checkpoint, minus, plus, coupling, *, sample_count: int, fixed_time_count: int,
    time_grid: list[float], seed: int, amplitude: float,
) -> dict:
    apply = _checkpoint_apply(checkpoint)
    rng = np.random.default_rng(seed)
    times = _stratified_times(rng, sample_count)
    states, targets, sample = sample_frozen_bridge(rng, minus, plus, coupling, times, amplitude)
    predictions = []
    for start in range(0, sample_count, 256):
        predictions.append(np.asarray(apply(
            jnp.asarray(times[start:start + 256]), jnp.asarray(states[start:start + 256])
        )))
    predictions = np.concatenate(predictions)
    overall = _fm_summary(predictions, targets)
    endpoint_mask = (times <= 0.10) | (times >= 0.90)
    middle_mask = (times >= 0.25) & (times <= 0.75)
    regions = {
        "near_endpoints": _fm_summary(predictions[endpoint_mask], targets[endpoint_mask]),
        "middle": _fm_summary(predictions[middle_mask], targets[middle_mask]),
    }
    per_time = []
    for index, time in enumerate(time_grid):
        fixed_times = np.full(fixed_time_count, time, dtype=np.float32)
        fixed_states, fixed_targets, _ = sample_frozen_bridge(
            rng, minus, plus, coupling, fixed_times, amplitude
        )
        fixed_predictions = []
        for start in range(0, fixed_time_count, 256):
            fixed_predictions.append(np.asarray(apply(
                jnp.asarray(fixed_times[start:start + 256]),
                jnp.asarray(fixed_states[start:start + 256]),
            )))
        row = {"t": float(time), **_fm_summary(np.concatenate(fixed_predictions), fixed_targets)}
        per_time.append(row)
    empirical_minus = np.bincount(sample["minus_indices"], minlength=len(minus)) / sample_count
    empirical_plus = np.bincount(sample["plus_indices"], minlength=len(plus)) / sample_count
    return {
        "overall": overall, "regions": regions, "per_time": per_time,
        "sampling": {
            "sample_count": sample_count,
            "maximum_minus_marginal_sampling_error": float(np.max(np.abs(empirical_minus - coupling.sum(axis=1)))),
            "maximum_plus_marginal_sampling_error": float(np.max(np.abs(empirical_plus - coupling.sum(axis=0)))),
            "time_minimum": float(times.min()), "time_maximum": float(times.max()),
            "time_mean": float(times.mean()), "time_sampling": "stratified uniform on [0,1]",
        },
        "dtype": {
            "state": str(states.dtype), "target": str(targets.dtype),
            "prediction": str(predictions.dtype),
            "convolution_weight": str(checkpoint["trainable"]["layers"][0]["weight"].dtype),
        },
    }


def _extended_hidden(fields: np.ndarray, threshold: float) -> np.ndarray:
    base = np.asarray(smooth_hidden_observables(jnp.asarray(fields), threshold=threshold))
    high = np.asarray(field_observables(
        jnp.asarray(fields), ShellDefinition((0.38,), (0.055,)), ("shell_1",)
    ))
    return np.concatenate([base, high], axis=1)


def _law_row(fields: np.ndarray, center, scale, target, shells, threshold) -> dict:
    phi_physical = np.asarray(field_observables(
        jnp.asarray(fields), shells, ("mean", "second_moment")
    )).mean(axis=0)
    standardized = (phi_physical - center) / scale
    hidden = _extended_hidden(fields, threshold).mean(axis=0)
    return {
        "phi_physical": phi_physical, "phi_standardized": standardized,
        "phi_minus_frozen_c": standardized - target,
        "smooth_tv": hidden[0], "anisotropy": hidden[1], "soft_area": hidden[2],
        "soft_perimeter": hidden[3], "heldout_power_1": hidden[4],
        "heldout_power_2": hidden[5], "field_minimum": float(fields.min()),
        "field_maximum": float(fields.max()),
    }


def _direct_si_banks(rng, minus, plus, coupling, times, count, amplitude):
    banks = []
    for time in times:
        states, _, _ = sample_frozen_bridge(
            rng, minus, plus, coupling, np.full(count, time, dtype=np.float32), amplitude
        )
        banks.append(states)
    return np.stack(banks)


def reference_split_thresholds(
    rng, minus, plus, coupling, times, *, replicates, count, raw_count,
    downsampled_count, center, scale, shells, amplitude,
) -> tuple[dict, list[dict]]:
    rows = []
    raw_by_time, down_by_time, phi_by_time = [[] for _ in times], [[] for _ in times], [[] for _ in times]
    for replicate in range(replicates):
        first = _direct_si_banks(rng, minus, plus, coupling, times, count, amplitude)
        second = _direct_si_banks(rng, minus, plus, coupling, times, count, amplitude)
        for index, time in enumerate(times):
            a, b = first[index], second[index]
            raw = weighted_mmd2_four_weight(
                a[:raw_count].reshape(raw_count, -1), np.full(raw_count, 1/raw_count),
                b[:raw_count].reshape(raw_count, -1), np.full(raw_count, 1/raw_count),
            )
            ad, bd = downsample_fields(a[:downsampled_count]), downsample_fields(b[:downsampled_count])
            down = weighted_mmd2_four_weight(
                ad.reshape(downsampled_count, -1), np.full(downsampled_count, 1/downsampled_count),
                bd.reshape(downsampled_count, -1), np.full(downsampled_count, 1/downsampled_count),
            )
            pa = np.asarray(field_observables(jnp.asarray(a), shells, ("mean", "second_moment"))).mean(0)
            pb = np.asarray(field_observables(jnp.asarray(b), shells, ("mean", "second_moment"))).mean(0)
            phi_error = float(np.max(np.abs((pa - pb) / scale)))
            raw_by_time[index].append(raw); down_by_time[index].append(down); phi_by_time[index].append(phi_error)
            rows.append({"replicate": replicate, "t": float(time), "raw_field_mmd2": raw,
                         "downsampled_field_mmd2": down, "maximum_standardized_phi_split_error": phi_error})
    thresholds = []
    for index, time in enumerate(times):
        thresholds.append({
            "t": float(time),
            "raw_field_mmd2": 4.0 * float(np.quantile(raw_by_time[index], 0.95)),
            "downsampled_field_mmd2": 4.0 * float(np.quantile(down_by_time[index], 0.95)),
            "maximum_standardized_phi_error": max(0.10, 3.0 * float(np.quantile(phi_by_time[index], 0.95))),
        })
    return {"per_time": thresholds, "rule_frozen_before_model_sweep": True}, rows


def evaluate_rollout_against_direct_si(
    checkpoint, minus, plus, minus_weights, plus_weights, config, *, steps: int,
    thresholds: dict | None = None, seed_offset: int = 0, time_grid=None,
) -> dict:
    diagnostic = config["rollout_diagnostics"]
    rng = np.random.default_rng(int(diagnostic["diagnostic_seed"]) + seed_offset)
    coupling = maximal_same_index_coupling(minus_weights, plus_weights)
    requested_times = np.asarray(
        config["fm_diagnostics"]["time_grid"] if time_grid is None else time_grid,
        dtype=np.float64,
    )
    indices = np.rint(requested_times * steps).astype(int)
    times = indices / steps
    count = int(diagnostic["particle_count"])
    initial_indices = rng.choice(len(minus), size=count, p=minus_weights)
    initial = minus[initial_indices]
    apply = _checkpoint_apply(checkpoint)
    rollout = heun_rollout_snapshots(apply, initial, steps, requested_times)
    direct = _direct_si_banks(
        rng, minus, plus, coupling, times, count, float(config["frozen_phase3a"]["schedule_amplitude"])
    )
    phase2 = _read_json(ROOT / config["source_phase2_directory"] / "large_bank_phase2_summary.json")
    design_config = _read_json(ROOT / config["source_design_config"])
    obs = design_config["observables"]
    shells = ShellDefinition(tuple(obs["shell_centers_cycles_per_pixel"]), tuple(obs["shell_widths_cycles_per_pixel"]))
    center = np.asarray(checkpoint["center"]); scale = np.asarray(checkpoint["scale"]); target = np.asarray(checkpoint["target"])
    threshold_value = float(phase2["fixed_threshold"])
    rows = []
    raw_count = int(diagnostic["mmd_particle_count_raw"])
    down_count = int(diagnostic["mmd_particle_count_downsampled"])
    for index, time in enumerate(times):
        learned, oracle = rollout[index], direct[index]
        learned_law = _law_row(learned, center, scale, target, shells, threshold_value)
        direct_law = _law_row(oracle, center, scale, target, shells, threshold_value)
        raw_mmd = weighted_mmd2_four_weight(
            learned[:raw_count].reshape(raw_count, -1), np.full(raw_count, 1/raw_count),
            oracle[:raw_count].reshape(raw_count, -1), np.full(raw_count, 1/raw_count),
        )
        ld, od = downsample_fields(learned[:down_count]), downsample_fields(oracle[:down_count])
        down_mmd = weighted_mmd2_four_weight(
            ld.reshape(down_count, -1), np.full(down_count, 1/down_count),
            od.reshape(down_count, -1), np.full(down_count, 1/down_count),
        )
        standardized_difference = np.asarray(learned_law["phi_standardized"]) - np.asarray(direct_law["phi_standardized"])
        direct_phi = np.asarray(field_observables(jnp.asarray(oracle), shells, ("mean", "second_moment")))
        standardized_direct = (direct_phi - center) / scale
        hull = solve_target_hull_lp(standardized_direct, target)
        projected_residual = None; projected_ess = None
        if hull["success"]:
            projected = calibrate_iprojection_instrumented(standardized_direct, target, tolerance=1e-10, max_iterations=500)
            projected_residual = projected["reported_residual"]
            projected_ess = projected["ess_fraction"]
        row = {
            "t": float(time), "requested_t": float(requested_times[index]),
            "learned": learned_law, "direct_si": direct_law,
            "learned_minus_direct_standardized_phi": standardized_difference,
            "maximum_learned_minus_direct_standardized_phi": float(np.max(np.abs(standardized_difference))),
            "raw_field_mmd2": raw_mmd, "downsampled_field_mmd2": down_mmd,
            "projected_target_hull_feasible": hull["success"],
            "projected_phi_minus_c": projected_residual,
            "projected_ess_fraction": projected_ess,
        }
        if thresholds is not None:
            threshold_row = thresholds["per_time"][index]
            row.update({
                "phi_fidelity_gate_pass": row["maximum_learned_minus_direct_standardized_phi"] <= threshold_row["maximum_standardized_phi_error"],
                "raw_mmd_gate_pass": raw_mmd <= threshold_row["raw_field_mmd2"],
                "downsampled_mmd_gate_pass": down_mmd <= threshold_row["downsampled_field_mmd2"],
                "thresholds": threshold_row,
            })
        rows.append(row)
    integrated_raw = float(np.trapezoid([row["raw_field_mmd2"] for row in rows], times))
    integrated_down = float(np.trapezoid([row["downsampled_field_mmd2"] for row in rows], times))
    maximum_phi = max(row["maximum_learned_minus_direct_standardized_phi"] for row in rows)
    result = {
        "ode_steps": steps, "times": times, "rows": rows,
        "integrated_raw_field_mmd2": integrated_raw,
        "integrated_downsampled_field_mmd2": integrated_down,
        "maximum_learned_minus_direct_standardized_phi": maximum_phi,
        "endpoint_raw_field_mmd2": rows[-1]["raw_field_mmd2"],
        "endpoint_maximum_standardized_phi_error": rows[-1]["maximum_learned_minus_direct_standardized_phi"],
        "old_incorrect_rollout_phi_minus_c": rows[-1]["learned"]["phi_minus_frozen_c"],
        "serious_field_range_fraction": float(np.mean(
            (rollout < float(config["reference_quality_gates"]["serious_field_minimum"]))
            | (rollout > float(config["reference_quality_gates"]["serious_field_maximum"]))
        )),
    }
    if thresholds is not None:
        result["rollout_fidelity_gate_pass"] = bool(all(
            row["phi_fidelity_gate_pass"] and row["raw_mmd_gate_pass"] and row["downsampled_mmd_gate_pass"]
            for row in rows
        ))
    return result


def endpoint_calibration_summary(result: dict) -> dict:
    keys = (
        "converged", "convergence_reason", "iterations", "maximum_absolute_standardized_residual",
        "ess_fraction", "entropy_fraction", "maximum_weight", "lambda_norm",
        "covariance_eigenvalues", "covariance_rank", "covariance_condition",
    )
    return {key: result[key] for key in keys}


def validation_bank_action(minimum_ess: float, attempted_chunks: int, maximum_chunks: int, threshold: float = 0.20) -> str:
    if minimum_ess >= threshold:
        return "accept"
    return "append_next_chunk" if attempted_chunks < maximum_chunks else "exhausted"


def build_healthy_validation_bank(config_path: Path = DEFAULT_CONFIG) -> dict:
    config = _read_json(config_path.resolve())
    output = ROOT / config["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    v7 = ROOT / config["source_phase3_v7_directory"]
    manifest = directory_manifest(v7)
    _write_json(output / "preserved_phase3_v7_sha256.json", manifest)
    _write_json(output / "phase3_quality_v8_config.json", config)
    with (v7 / "reference_cnn_checkpoint.pkl").open("rb") as handle:
        checkpoint = pickle.load(handle)
    candidate = checkpoint["candidate"]
    target = np.asarray(checkpoint["target"]); center = np.asarray(checkpoint["center"]); scale = np.asarray(checkpoint["scale"])
    design = _read_json(ROOT / config["source_phase2_directory"] / "large_bank_phase2_summary.json")
    design_config = _read_json(ROOT / "configs" / "expC_grayscott_design.yaml")
    obs = design_config["observables"]
    shells = ShellDefinition(tuple(obs["shell_centers_cycles_per_pixel"]), tuple(obs["shell_widths_cycles_per_pixel"]))
    rule = config["validation_endpoint_bank"]
    chunks_minus, chunks_plus, chunks_seed = [], [], []
    attempts = []
    accepted = None
    for chunk_index in range(int(rule["maximum_chunks"])):
        seed_start = int(rule["initial_seed_start"]) + chunk_index * int(rule["chunk_size"])
        minus, plus, seeds = _make_endpoint_bank(
            config, design, candidate, seed_start, int(rule["chunk_size"])
        )
        chunks_minus.append(minus); chunks_plus.append(plus); chunks_seed.append(seeds)
        joined_minus, joined_plus = np.concatenate(chunks_minus), np.concatenate(chunks_plus)
        minus_cal, plus_cal, hulls = _calibrate_endpoints(
            joined_minus, joined_plus, target, center, scale, shells,
            {"residual_tolerance": 1e-10, "maximum_iterations": 500},
        )
        row = {
            "attempt": chunk_index + 1, "seed_start": int(chunks_seed[0][0]),
            "seed_stop": int(chunks_seed[-1][-1]), "bank_size_per_endpoint": len(joined_minus),
            "minus": endpoint_calibration_summary(minus_cal),
            "plus": endpoint_calibration_summary(plus_cal),
            "minimum_ess_fraction": min(minus_cal["ess_fraction"], plus_cal["ess_fraction"]),
        }
        attempts.append(row)
        action = validation_bank_action(
            row["minimum_ess_fraction"], chunk_index + 1, int(rule["maximum_chunks"]),
            float(rule["minimum_ess_fraction"]),
        )
        row["deterministic_rule_action"] = action
        if action == "accept":
            accepted = (joined_minus, joined_plus, np.concatenate(chunks_seed), minus_cal, plus_cal, hulls)
            break
    if accepted is None:
        raise RuntimeError("deterministic validation-bank enlargement exhausted without ESS >= 0.20")
    minus, plus, seeds, minus_cal, plus_cal, hulls = accepted
    np.savez_compressed(
        output / "healthy_validation_endpoint_bank.npz", minus=minus, plus=plus,
        seeds=seeds, minus_weights=minus_cal["weights"], plus_weights=plus_cal["weights"],
        target=target, center=center, scale=scale,
    )
    result = {
        "status": "healthy_validation_bank_ready", "attempts": attempts,
        "accepted_bank_size_per_endpoint": len(minus),
        "accepted_seed_start": int(seeds[0]), "accepted_seed_stop": int(seeds[-1]),
        "minus_calibration": endpoint_calibration_summary(minus_cal),
        "plus_calibration": endpoint_calibration_summary(plus_cal),
        "minimum_ess_fraction": min(minus_cal["ess_fraction"], plus_cal["ess_fraction"]),
        "frozen_target": target, "target_changed": False,
        "phase3_v7_unchanged_after_run": directory_manifest(v7) == manifest,
    }
    _write_json(output / "validation_bank_summary.json", result)
    return result


def audit_v7_checkpoint(config_path: Path = DEFAULT_CONFIG) -> dict:
    config = _read_json(config_path.resolve())
    output = ROOT / config["output_directory"]
    v7 = ROOT / config["source_phase3_v7_directory"]
    preserved = _read_json(output / "preserved_phase3_v7_sha256.json")
    if directory_manifest(v7) != preserved:
        raise RuntimeError("v7 artifacts changed before the checkpoint audit")
    with (v7 / "reference_cnn_checkpoint.pkl").open("rb") as handle:
        checkpoint = pickle.load(handle)
    v7_banks = np.load(v7 / "reference_training_endpoint_banks.npz")
    healthy = np.load(output / "healthy_validation_endpoint_bank.npz")
    diagnostic = config["fm_diagnostics"]
    amplitude = float(config["frozen_phase3a"]["schedule_amplitude"])
    roles = {
        "v7_training_endpoint_bank_fresh_samples": (
            np.asarray(v7_banks["training_minus"]), np.asarray(v7_banks["training_plus"]),
            np.asarray(v7_banks["training_minus_weights"]), np.asarray(v7_banks["training_plus_weights"]),
        ),
        "v7_degenerate_model_selection_bank": (
            np.asarray(v7_banks["validation_minus"]), np.asarray(v7_banks["validation_plus"]),
            np.asarray(v7_banks["validation_minus_weights"]), np.asarray(v7_banks["validation_plus_weights"]),
        ),
        "v8_healthy_independent_validation_bank": (
            np.asarray(healthy["minus"]), np.asarray(healthy["plus"]),
            np.asarray(healthy["minus_weights"]), np.asarray(healthy["plus_weights"]),
        ),
    }
    fm = {}
    for role_index, (name, (minus, plus, minus_weights, plus_weights)) in enumerate(roles.items()):
        fm[name] = evaluate_fm_bank(
            checkpoint, minus, plus,
            maximal_same_index_coupling(minus_weights, plus_weights),
            sample_count=int(diagnostic["sample_count_per_bank"]),
            fixed_time_count=int(diagnostic["fixed_time_sample_count"]),
            time_grid=list(map(float, diagnostic["time_grid"])),
            seed=int(diagnostic["diagnostic_seed"]) + 1000 * role_index,
            amplitude=amplitude,
        )
    new_minus, new_plus, new_mw, new_pw = roles["v8_healthy_independent_validation_bank"]
    new_coupling = maximal_same_index_coupling(new_mw, new_pw)
    verification_rng = np.random.default_rng(int(diagnostic["diagnostic_seed"]) + 9000)
    verification_times = np.linspace(0.05, 0.95, 256, dtype=np.float32)
    states, targets, sampled = sample_frozen_bridge(
        verification_rng, new_minus, new_plus, new_coupling, verification_times, amplitude
    )
    target_check = bridge_target_consistency(
        new_minus[sampled["minus_indices"]], new_plus[sampled["plus_indices"]],
        sampled["noise"], verification_times, amplitude, epsilon=2e-3,
    )
    apply = _checkpoint_apply(checkpoint)
    input_gradient = jax.grad(lambda value: jnp.mean(apply(jnp.asarray(0.37), value)))(jnp.asarray(states[:1]))
    target_check.update({
        "cnn_input_gradient_norm": float(jnp.linalg.norm(input_gradient)),
        "no_accidental_stop_gradient": bool(jnp.linalg.norm(input_gradient) > 0),
        "normalization": "none: raw float32 V fields and raw float32 velocity targets in training and inference",
        "time_sampling_objective": "uniform continuous time; v8 diagnostics use stratified uniform draws",
        "coupling_exact_minus_marginal_error": float(np.max(np.abs(new_coupling.sum(1) - new_mw))),
        "coupling_exact_plus_marginal_error": float(np.max(np.abs(new_coupling.sum(0) - new_pw))),
        "noise_spatial_mean_maximum": float(np.max(np.abs(sampled["noise"].mean(axis=(1, 2, 3))))),
        "noise_rms_maximum_error": float(np.max(np.abs(np.mean(sampled["noise"] ** 2, axis=(1, 2, 3)) - 1.0))),
    })
    rollout_config = config["rollout_diagnostics"]
    requested = np.asarray(diagnostic["time_grid"], dtype=np.float64)
    primary_steps = int(rollout_config["primary_ode_steps"])
    actual_times = np.rint(requested * primary_steps).astype(int) / primary_steps
    split_rng = np.random.default_rng(int(rollout_config["diagnostic_seed"]) + 5000)
    thresholds, split_rows = reference_split_thresholds(
        split_rng, new_minus, new_plus, new_coupling, actual_times,
        replicates=int(rollout_config["reference_split_replicates"]),
        count=int(rollout_config["particle_count"]),
        raw_count=int(rollout_config["mmd_particle_count_raw"]),
        downsampled_count=int(rollout_config["mmd_particle_count_downsampled"]),
        center=np.asarray(checkpoint["center"]), scale=np.asarray(checkpoint["scale"]),
        shells=ShellDefinition((0.0625, 0.125, 0.1875), (0.045, 0.050, 0.055)),
        amplitude=amplitude,
    )
    rollout = evaluate_rollout_against_direct_si(
        checkpoint, new_minus, new_plus, new_mw, new_pw, config,
        steps=primary_steps, thresholds=thresholds,
    )
    fresh_train_ratio = fm["v7_training_endpoint_bank_fresh_samples"]["overall"]["normalized_fm_mse"]
    old_validation_ratio = fm["v7_degenerate_model_selection_bank"]["overall"]["normalized_fm_mse"]
    new_validation_ratio = fm["v8_healthy_independent_validation_bank"]["overall"]["normalized_fm_mse"]
    endpoint_npz = v7_banks
    old_min_ess = min(
        1.0 / (len(endpoint_npz["validation_minus_weights"]) * np.sum(endpoint_npz["validation_minus_weights"] ** 2)),
        1.0 / (len(endpoint_npz["validation_plus_weights"]) * np.sum(endpoint_npz["validation_plus_weights"] ** 2)),
    )
    trace = list(csv.DictReader((v7 / "reference_cnn_training_trace.csv").open()))
    tail_losses = np.asarray([float(row["training_mse_per_pixel"]) for row in trace[-20:]])
    slope = float(np.polyfit(np.arange(len(tail_losses)), tail_losses, 1)[0])
    flags = {
        "optimization_failure_possible": bool(slope < -1e-4),
        "capacity_or_underfit": bool(fresh_train_ratio > float(config["reference_quality_gates"]["heldout_fm_mse_fraction_of_zero_predictor"])),
        "generalization_failure": bool(fresh_train_ratio <= 0.35 and new_validation_ratio > 0.35),
        "finite_bank_importance_pathology_in_v7_validation": bool(old_min_ess < 0.10),
        "rollout_accumulation_despite_acceptable_local_fm": bool(new_validation_ratio <= 0.35 and not rollout.get("rollout_fidelity_gate_pass", False)),
    }
    active = [name for name, value in flags.items() if value]
    result = {
        "status": "v7_checkpoint_audited",
        "gate_semantics_audit": {
            "experiment_b_reference_selection": "held-out stochastic-interpolant velocity regression only",
            "validated_rollout_comparison": "generated law versus independently sampled oracle/interpolant law by MMD",
            "v7_old_rollout_maximum_standardized_phi_minus_c": 1.786652653481443,
            "v7_old_gate": 0.10,
            "v7_old_gate_is_reference_fidelity_gate": False,
            "reason": "the raw reference SI defines an unprojected moving prior and is not required to remain on the fixed moment fiber",
            "v7_history_rewritten": False,
            "corrected_v8_primary_rollout_comparison": "learned raw rollout versus direct raw SI marginal",
        },
        "fm_target_verification": target_check,
        "fm_diagnostics": fm,
        "reference_split_thresholds": thresholds,
        "v7_checkpoint_rollout_vs_direct_si": rollout,
        "failure_classification": {
            "flags": flags, "active_mechanisms": active,
            "supporting_metrics": {
                "fresh_training_bank_normalized_fm_mse": fresh_train_ratio,
                "old_degenerate_validation_normalized_fm_mse": old_validation_ratio,
                "new_healthy_validation_normalized_fm_mse": new_validation_ratio,
                "old_validation_minimum_ess": old_min_ess,
                "new_validation_minimum_ess": _read_json(output / "validation_bank_summary.json")["minimum_ess_fraction"],
                "training_trace_tail_loss_slope_per_checkpoint": slope,
                "corrected_rollout_fidelity_gate_pass": rollout.get("rollout_fidelity_gate_pass", False),
            },
        },
        "deep_ritz_training_performed": False, "phase4_performed": False,
        "phase3_v7_unchanged_after_run": directory_manifest(v7) == preserved,
    }
    _write_csv(output / "reference_split_variability.csv", split_rows)
    for name, values in fm.items():
        _write_csv(output / f"v7_{name}_fm_by_time.csv", values["per_time"])
    _write_json(output / "v7_checkpoint_audit.json", result)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validation-bank", "audit-v7"), nargs="?", default="validation-bank")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    result = build_healthy_validation_bank(args.config) if args.command == "validation-bank" else audit_v7_checkpoint(args.config)
    print(json.dumps(result, indent=2, default=lambda value: np.asarray(value).tolist()))


if __name__ == "__main__":
    main()
