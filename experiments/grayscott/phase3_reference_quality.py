"""Corrected Gray--Scott Phase-3B reference-flow audit and diagnostics (v8)."""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
import platform
import subprocess
import sys
import time
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
    spectral_reference_model,
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
    if checkpoint.get("model_kind") == "spectral_global":
        return jax.jit(lambda time, fields: spectral_reference_model(
            checkpoint["trainable"], checkpoint["architecture"], time, fields
        ))
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
            "convolution_weight": str(jax.tree_util.tree_leaves(checkpoint["trainable"])[0].dtype),
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
    spectrum = np.abs(np.fft.fft2(np.asarray(fields)[:, 0], norm="ortho")) ** 2
    height, width = fields.shape[-2:]
    fy, fx = np.fft.fftfreq(height), np.fft.fftfreq(width)
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    radial_power = [float(spectrum[:, (radius >= low) & (radius < high)].mean())
                    for low, high in ((0.0, 0.125), (0.125, 0.25), (0.25, np.inf))]
    return {
        "phi_physical": phi_physical, "phi_standardized": standardized,
        "phi_minus_frozen_c": standardized - target,
        "smooth_tv": hidden[0], "anisotropy": hidden[1], "soft_area": hidden[2],
        "soft_perimeter": hidden[3], "heldout_power_1": hidden[4],
        "heldout_power_2": hidden[5],
        "radial_power_low": radial_power[0], "radial_power_middle": radial_power[1],
        "radial_power_high": radial_power[2], "field_minimum": float(fields.min()),
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


def _apply_trainable(model_kind, trainable, architecture, times, states):
    if model_kind == "spectral_global":
        return spectral_reference_model(trainable, architecture, times, states)
    if model_kind == "residual_periodic":
        return residual_reference_cnn(trainable, architecture, times, states)
    return periodic_reference_cnn(
        _assemble_parameters(trainable, **architecture), times, states
    )


def _tree_global_norm(tree):
    return jnp.sqrt(sum(jnp.sum(value * value) for value in jax.tree_util.tree_leaves(tree)))


def _evaluate_fixed_fm(model_kind, trainable, architecture, times, states, targets, batch_size=256):
    apply = jax.jit(lambda t, x: _apply_trainable(model_kind, trainable, architecture, t, x))
    squared = 0.0
    for start in range(0, len(times), batch_size):
        prediction = np.asarray(apply(
            jnp.asarray(times[start:start + batch_size]), jnp.asarray(states[start:start + batch_size])
        ))
        squared += float(np.sum((prediction - targets[start:start + batch_size]) ** 2))
    return squared / (len(times) * np.prod(states.shape[1:]))


def _train_reference_variant(
    *, variant, initial_trainable, model_kind, architecture,
    train_minus, train_plus, train_coupling,
    validation_times, validation_states, validation_targets,
    amplitude, protocol, output,
):
    steps = int(variant.get("additional_training_steps", variant.get("training_steps", 0)))
    batch_size = int(protocol["batch_size"])
    clip = float(protocol["gradient_clip"]); weight_decay = float(protocol["weight_decay"])
    lr_start = float(variant["learning_rate_start"]); lr_end = float(variant["learning_rate_end"])
    trainable = jax.tree_util.tree_map(jnp.asarray, initial_trainable)
    first = jax.tree_util.tree_map(jnp.zeros_like, trainable)
    second = jax.tree_util.tree_map(jnp.zeros_like, trainable)

    @jax.jit
    def step_fn(parameters, first_moment, second_moment, step_index, times, states, targets):
        def objective(value):
            prediction = _apply_trainable(model_kind, value, architecture, times, states)
            return jnp.mean((prediction - targets) ** 2)
        loss, gradient = jax.value_and_grad(objective)(parameters)
        gradient_norm = _tree_global_norm(gradient)
        factor = jnp.minimum(1.0, clip / jnp.maximum(gradient_norm, 1e-30))
        gradient = jax.tree_util.tree_map(lambda value: value * factor, gradient)
        first_moment = jax.tree_util.tree_map(lambda old, grad: 0.9 * old + 0.1 * grad, first_moment, gradient)
        second_moment = jax.tree_util.tree_map(lambda old, grad: 0.999 * old + 0.001 * grad * grad, second_moment, gradient)
        corrected_first = jax.tree_util.tree_map(lambda value: value / (1.0 - 0.9 ** step_index), first_moment)
        corrected_second = jax.tree_util.tree_map(lambda value: value / (1.0 - 0.999 ** step_index), second_moment)
        fraction = jnp.minimum((step_index - 1) / jnp.maximum(steps - 1, 1), 1.0)
        learning_rate = lr_end + 0.5 * (lr_start - lr_end) * (1.0 + jnp.cos(jnp.pi * fraction))
        parameters = jax.tree_util.tree_map(
            lambda value, m, v: value - learning_rate * (
                m / (jnp.sqrt(v) + 1e-8) + weight_decay * value
            ), parameters, corrected_first, corrected_second,
        )
        return parameters, first_moment, second_moment, loss, gradient_norm, learning_rate

    rng = np.random.default_rng(int(protocol["fixed_training_interpolant_seed"]))
    validation_zero = float(np.mean(validation_targets * validation_targets))
    best, best_ratio, trace = trainable, float("inf"), []
    start_time = time.perf_counter()
    for step in range(1, steps + 1):
        batch_times = _stratified_times(rng, batch_size)
        batch_states, batch_targets, _ = sample_frozen_bridge(
            rng, train_minus, train_plus, train_coupling, batch_times, amplitude
        )
        trainable, first, second, loss, gradient_norm, learning_rate = step_fn(
            trainable, first, second, step, jnp.asarray(batch_times),
            jnp.asarray(batch_states), jnp.asarray(batch_targets),
        )
        if step == 1 or step % int(protocol["evaluation_interval"]) == 0 or step == steps:
            validation_mse = _evaluate_fixed_fm(
                model_kind, trainable, architecture,
                validation_times, validation_states, validation_targets,
            )
            ratio = validation_mse / validation_zero
            if ratio < best_ratio:
                best_ratio = ratio
                best = jax.tree_util.tree_map(lambda value: np.asarray(value), trainable)
            trace.append({
                "step": step, "training_mse_per_pixel": float(loss),
                "validation_mse_per_pixel": validation_mse,
                "validation_normalized_fm_mse": ratio,
                "gradient_norm_before_clip": float(gradient_norm),
                "learning_rate": float(learning_rate),
            })
        if step % 1000 == 0:
            print(f"{variant['id']} step {step}/{steps}: validation ratio={trace[-1]['validation_normalized_fm_mse']:.5f}", flush=True)
    jax.block_until_ready(jax.tree_util.tree_leaves(trainable)[0])
    return best, trace, time.perf_counter() - start_time


def run_controlled_sweep(
    config_path: Path = DEFAULT_CONFIG,
    sweep_path: Path = DEFAULT_SWEEP_CONFIG,
) -> dict:
    config = _read_json(config_path.resolve()); protocol = _read_json(sweep_path.resolve())
    output = ROOT / config["output_directory"]
    v7 = ROOT / config["source_phase3_v7_directory"]
    audit = _read_json(output / "v7_checkpoint_audit.json")
    if not audit["fm_target_verification"]["same_noise_used_for_state_and_target"]:
        raise RuntimeError("FM target verification must pass before the sweep")
    healthy_summary = _read_json(output / "validation_bank_summary.json")
    if healthy_summary["minimum_ess_fraction"] < float(config["reference_quality_gates"]["validation_endpoint_minimum_ess_fraction"]):
        raise RuntimeError("validation endpoint bank is not healthy")
    _write_json(output / "controlled_sweep_protocol.json", protocol)
    with (v7 / "reference_cnn_checkpoint.pkl").open("rb") as handle:
        v7_checkpoint = pickle.load(handle)
    training = np.load(v7 / "reference_training_endpoint_banks.npz")
    validation = np.load(output / "healthy_validation_endpoint_bank.npz")
    train_minus, train_plus = np.asarray(training["training_minus"]), np.asarray(training["training_plus"])
    train_mw, train_pw = np.asarray(training["training_minus_weights"]), np.asarray(training["training_plus_weights"])
    validation_minus, validation_plus = np.asarray(validation["minus"]), np.asarray(validation["plus"])
    validation_mw, validation_pw = np.asarray(validation["minus_weights"]), np.asarray(validation["plus_weights"])
    train_coupling = maximal_same_index_coupling(train_mw, train_pw)
    validation_coupling = maximal_same_index_coupling(validation_mw, validation_pw)
    rng = np.random.default_rng(int(protocol["fixed_validation_interpolant_seed"]))
    validation_times = _stratified_times(rng, int(protocol["fixed_validation_interpolant_count"]))
    validation_states, validation_targets, _ = sample_frozen_bridge(
        rng, validation_minus, validation_plus, validation_coupling, validation_times,
        float(config["frozen_phase3a"]["schedule_amplitude"]),
    )
    validation_zero = float(np.mean(validation_targets * validation_targets))
    thresholds = audit["reference_split_thresholds"]
    variants = []
    checkpoints = {"A_v7_saved_baseline": v7_checkpoint}
    for variant in protocol["variants"]:
        variant_id = variant["id"]
        if variant_id == "A_v7_saved_baseline":
            checkpoint = v7_checkpoint
            training_trace, training_seconds = [], None
        elif variant_id == "B_baseline_longer":
            best, training_trace, training_seconds = _train_reference_variant(
                variant=variant, initial_trainable=v7_checkpoint["trainable"],
                model_kind="v7_sequential", architecture=v7_checkpoint["architecture"],
                train_minus=train_minus, train_plus=train_plus, train_coupling=train_coupling,
                validation_times=validation_times, validation_states=validation_states,
                validation_targets=validation_targets,
                amplitude=float(config["frozen_phase3a"]["schedule_amplitude"]),
                protocol=protocol, output=output,
            )
            checkpoint = {**v7_checkpoint, "trainable": best, "model_kind": "v7_sequential",
                          "variant_id": variant_id, "training_protocol": variant}
        else:
            initial, architecture = init_residual_reference_cnn(
                jax.random.PRNGKey(int(protocol["moderate_model_initialization_seed"])),
                channels=28, dilations=(1, 2, 4, 8, 4),
                time_frequencies=3, dtype=jnp.float32,
            )
            best, training_trace, training_seconds = _train_reference_variant(
                variant=variant, initial_trainable=initial,
                model_kind="residual_periodic", architecture=architecture,
                train_minus=train_minus, train_plus=train_plus, train_coupling=train_coupling,
                validation_times=validation_times, validation_states=validation_states,
                validation_targets=validation_targets,
                amplitude=float(config["frozen_phase3a"]["schedule_amplitude"]),
                protocol=protocol, output=output,
            )
            checkpoint = {
                "trainable": best, "architecture": architecture, "model_kind": "residual_periodic",
                "variant_id": variant_id, "training_protocol": variant,
                "candidate": v7_checkpoint["candidate"], "selected_path": v7_checkpoint["selected_path"],
                "center": v7_checkpoint["center"], "scale": v7_checkpoint["scale"], "target": v7_checkpoint["target"],
            }
        checkpoints[variant_id] = checkpoint
        validation_mse = _evaluate_fixed_fm(
            checkpoint.get("model_kind", "v7_sequential"), checkpoint["trainable"],
            checkpoint["architecture"], validation_times, validation_states, validation_targets,
        )
        local_ratio = validation_mse / validation_zero
        rollout = evaluate_rollout_against_direct_si(
            checkpoint, validation_minus, validation_plus, validation_mw, validation_pw,
            config, steps=int(config["rollout_diagnostics"]["primary_ode_steps"]),
            thresholds=thresholds,
        )
        local_pass = bool(local_ratio <= float(config["reference_quality_gates"]["heldout_fm_mse_fraction_of_zero_predictor"]))
        pathology_pass = bool(rollout["serious_field_range_fraction"] <= float(config["reference_quality_gates"]["maximum_serious_range_fraction"]))
        pass_all = bool(local_pass and rollout["rollout_fidelity_gate_pass"] and pathology_pass)
        row = {
            "variant_id": variant_id, "validation_mse_per_pixel": validation_mse,
            "validation_zero_predictor_mse_per_pixel": validation_zero,
            "validation_normalized_fm_mse": local_ratio,
            "local_fm_gate_pass": local_pass,
            "rollout_fidelity_gate_pass": rollout["rollout_fidelity_gate_pass"],
            "field_pathology_gate_pass": pathology_pass,
            "phase3b_candidate_pass": pass_all,
            "integrated_raw_field_mmd2": rollout["integrated_raw_field_mmd2"],
            "integrated_downsampled_field_mmd2": rollout["integrated_downsampled_field_mmd2"],
            "maximum_rollout_standardized_phi_error": rollout["maximum_learned_minus_direct_standardized_phi"],
            "endpoint_rollout_standardized_phi_error": rollout["endpoint_maximum_standardized_phi_error"],
            "training_seconds": training_seconds,
            "parameter_count": int(sum(np.asarray(value).size for value in jax.tree_util.tree_leaves(checkpoint["trainable"]))),
            "training_trace": training_trace, "rollout": rollout,
        }
        variants.append(row)
        with (output / f"checkpoint_{variant_id}.pkl").open("wb") as handle:
            pickle.dump(checkpoint, handle)
        _write_csv(output / f"training_trace_{variant_id}.csv", training_trace)
        _write_json(output / f"variant_{variant_id}_summary.json", row)
    passing = [row for row in variants if row["phase3b_candidate_pass"]]
    passing.sort(key=lambda row: (
        row["validation_normalized_fm_mse"], row["integrated_downsampled_field_mmd2"],
        row["integrated_raw_field_mmd2"], row["parameter_count"],
    ))
    local_candidates = [row for row in variants if row["local_fm_gate_pass"]]
    local_candidates.sort(key=lambda row: row["validation_normalized_fm_mse"])
    diagnostic_variant = passing[0] if passing else (local_candidates[0] if local_candidates else min(
        variants, key=lambda row: row["validation_normalized_fm_mse"]
    ))
    selected_id = passing[0]["variant_id"] if passing else None
    result = {
        "status": "phase3b_sweep_has_passer" if selected_id else "phase3b_sweep_no_passer",
        "variants": variants, "selected_variant_id": selected_id,
        "ode_diagnostic_variant_id": diagnostic_variant["variant_id"],
        "selection_uses_reference_quality_only": True,
        "deep_ritz_training_performed": False, "phase4_performed": False,
    }
    _write_csv(output / "controlled_sweep_summary.csv", [
        {key: value for key, value in row.items() if key not in ("training_trace", "rollout")}
        for row in variants
    ])
    _write_json(output / "controlled_sweep_summary.json", result)
    return result


def run_ode_resolution_and_finalize(
    config_path: Path = DEFAULT_CONFIG,
    sweep_path: Path = DEFAULT_SWEEP_CONFIG,
) -> dict:
    config = _read_json(config_path.resolve()); protocol = _read_json(sweep_path.resolve())
    output = ROOT / config["output_directory"]
    sweep = _read_json(output / "controlled_sweep_summary.json")
    variant_id = sweep["selected_variant_id"] or sweep["ode_diagnostic_variant_id"]
    with (output / f"checkpoint_{variant_id}.pkl").open("rb") as handle:
        checkpoint = pickle.load(handle)
    validation = np.load(output / "healthy_validation_endpoint_bank.npz")
    minus, plus = np.asarray(validation["minus"]), np.asarray(validation["plus"])
    mw, pw = np.asarray(validation["minus_weights"]), np.asarray(validation["plus_weights"])
    common_times = protocol["ode_common_time_grid"]
    resolutions = {}
    for steps in map(int, config["rollout_diagnostics"]["ode_steps"]):
        resolutions[str(steps)] = evaluate_rollout_against_direct_si(
            checkpoint, minus, plus, mw, pw, config, steps=steps,
            thresholds=None, seed_offset=17000, time_grid=common_times,
        )
    m128 = resolutions["128"]["integrated_downsampled_field_mmd2"]
    m256 = resolutions["256"]["integrated_downsampled_field_mmd2"]
    relative_improvement = (m128 - m256) / max(m128, 1e-30)
    ode_sensitive = bool(relative_improvement > 0.20)
    variant_row = next(row for row in sweep["variants"] if row["variant_id"] == variant_id)
    target_ok = _read_json(output / "v7_checkpoint_audit.json")["fm_target_verification"]["relative_finite_difference_error"] < 1e-3
    bank_ok = _read_json(output / "validation_bank_summary.json")["minimum_ess_fraction"] >= 0.20
    phase3b_pass = bool(
        sweep["selected_variant_id"] is not None and target_ok and bank_ok
        and not ode_sensitive and variant_row["phase3b_candidate_pass"]
    )
    remaining = []
    if not variant_row["local_fm_gate_pass"]: remaining.append("local conditional-velocity regression quality")
    if not variant_row["rollout_fidelity_gate_pass"]: remaining.append("rollout accumulation / conditional-velocity approximation")
    if ode_sensitive: remaining.append("ODE under-resolution")
    if not variant_row["field_pathology_gate_pass"]: remaining.append("field-range pathology")
    result = {
        "status": "phase3b_pass" if phase3b_pass else "phase3b_failed_after_controlled_sweep",
        "diagnostic_variant_id": variant_id, "selected_variant_id": sweep["selected_variant_id"],
        "ode_resolution": resolutions,
        "downsampled_mmd_relative_improvement_128_to_256": relative_improvement,
        "ode_under_resolution": ode_sensitive,
        "fm_target_verification_pass": target_ok, "validation_bank_health_pass": bank_ok,
        "phase3b_pass": phase3b_pass, "remaining_obstacles": remaining,
        "phase4_authorized": phase3b_pass,
        "deep_ritz_training_performed": False, "learned_method_comparison_performed": False,
    }
    _write_json(output / "phase3b_final_decision.json", result)
    return result


def materialize_final_audit(
    config_path: Path = DEFAULT_CONFIG,
    sweep_path: Path = DEFAULT_SWEEP_CONFIG,
) -> dict:
    """Write compact, human-auditable tables and reproducibility metadata."""
    config_path = config_path.resolve(); sweep_path = sweep_path.resolve()
    config = _read_json(config_path); protocol = _read_json(sweep_path)
    output = ROOT / config["output_directory"]
    sweep = _read_json(output / "controlled_sweep_summary.json")
    decision = _read_json(output / "phase3b_final_decision.json")

    rollout_files = []
    for variant in sweep["variants"]:
        rows = []
        for row in variant["rollout"]["rows"]:
            thresholds = row.get("thresholds", {})
            rows.append({
                "variant_id": variant["variant_id"], "time": row["t"],
                "maximum_standardized_phi_error": row["maximum_learned_minus_direct_standardized_phi"],
                "phi_threshold": thresholds.get("maximum_standardized_phi_error"),
                "raw_field_mmd2": row["raw_field_mmd2"],
                "raw_mmd2_threshold": thresholds.get("raw_field_mmd2"),
                "downsampled_field_mmd2": row["downsampled_field_mmd2"],
                "downsampled_mmd2_threshold": thresholds.get("downsampled_field_mmd2"),
                "phi_gate_pass": row.get("phi_fidelity_gate_pass"),
                "raw_mmd_gate_pass": row.get("raw_mmd_gate_pass"),
                "downsampled_mmd_gate_pass": row.get("downsampled_mmd_gate_pass"),
                "projected_phi_residual_max": float(np.max(np.abs(row["projected_phi_minus_c"]))),
                "projected_ess_fraction": row["projected_ess_fraction"],
                "learned_field_minimum": row["learned"]["field_minimum"],
                "learned_field_maximum": row["learned"]["field_maximum"],
            })
        path = output / f"rollout_{variant['variant_id']}_by_time.csv"
        _write_csv(path, rows); rollout_files.append(str(path.relative_to(ROOT)))

    resolution_rows = []
    for steps, result in sorted(decision["ode_resolution"].items(), key=lambda item: int(item[0])):
        resolution_rows.append({
            "ode_steps": int(steps),
            "integrated_raw_field_mmd2": result["integrated_raw_field_mmd2"],
            "integrated_downsampled_field_mmd2": result["integrated_downsampled_field_mmd2"],
            "maximum_standardized_phi_error": result["maximum_learned_minus_direct_standardized_phi"],
            "endpoint_standardized_phi_error": result["endpoint_maximum_standardized_phi_error"],
            "endpoint_raw_field_mmd2": result["endpoint_raw_field_mmd2"],
        })
    _write_csv(output / "ode_resolution_summary.csv", resolution_rows)

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.splitlines()
    source_directories = {
        "phase2_v6": ROOT / config["source_phase2_directory"],
        "phase3_v7": ROOT / config["source_phase3_v7_directory"],
    }
    source_manifests = {
        name: directory_manifest(path) for name, path in source_directories.items()
    }
    experiment_b_paths = [
        ROOT / "example_b.py", ROOT / "sweep_example_b.py",
        ROOT / "scripts" / "run_example_b.sh", ROOT / "checkpoints" / "example_b.npz",
        ROOT / "results" / "example_b", ROOT / "results" / "reference" / "example_b",
    ]
    experiment_b_manifest = {}
    for path in experiment_b_paths:
        if path.is_file():
            experiment_b_manifest[str(path.relative_to(ROOT))] = _hash_file(path)
        elif path.is_dir():
            experiment_b_manifest.update(directory_manifest(path))
    _write_json(output / "preserved_source_manifests.json", {
        **source_manifests, "experiment_b": experiment_b_manifest,
    })

    checkpoints = {}
    for path in sorted(output.glob("checkpoint_*.pkl")):
        with path.open("rb") as handle:
            checkpoint = pickle.load(handle)
        checkpoints[str(path.relative_to(ROOT))] = {
            "sha256": _hash_file(path),
            "parameter_count": int(sum(
                np.asarray(value).size for value in jax.tree_util.tree_leaves(checkpoint["trainable"])
            )),
            "model_kind": checkpoint.get("model_kind", "v7_sequential"),
        }
    metadata = {
        "experiment": "C", "version": config["version"],
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit, "git_status_at_materialization": git_status,
        "python": sys.version, "platform": platform.platform(),
        "jax_version": jax.__version__, "numpy_version": np.__version__,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_devices": [str(device) for device in jax.devices()],
        "config": str(config_path.relative_to(ROOT)), "config_sha256": _hash_file(config_path),
        "sweep_protocol": str(sweep_path.relative_to(ROOT)),
        "sweep_protocol_sha256": _hash_file(sweep_path),
        "frozen_phase3a": config["frozen_phase3a"],
        "validation_bank": {
            "seeds": [61001, 62024], "size_per_endpoint": 1024,
            "minimum_ess_fraction": _read_json(output / "validation_bank_summary.json")["minimum_ess_fraction"],
        },
        "training_and_evaluation_seeds": {
            key: value for key, value in protocol.items() if key.endswith("seed")
        },
        "optimizer": protocol["optimizer"], "batch_size": protocol["batch_size"],
        "time_sampling": protocol["time_sampling"],
        "ode_steps": config["rollout_diagnostics"]["ode_steps"],
        "checkpoints": checkpoints,
        "phase3b_status": decision["status"],
        "phase4_performed": False, "deep_ritz_training_performed": False,
        "learned_method_comparison_performed": False,
    }
    _write_json(output / "run_metadata.json", metadata)
    artifact_hashes = {
        str(path.relative_to(ROOT)): _hash_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "v8_artifact_sha256.json"
    }
    _write_json(output / "v8_artifact_sha256.json", artifact_hashes)
    return {
        "status": "v8_audit_materialized", "rollout_tables": rollout_files,
        "resolution_table": str((output / "ode_resolution_summary.csv").relative_to(ROOT)),
        "metadata": str((output / "run_metadata.json").relative_to(ROOT)),
        "artifact_count": len(artifact_hashes),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("validation-bank", "audit-v7", "sweep", "ode-finalize", "materialize"),
        nargs="?", default="validation-bank",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sweep-config", type=Path, default=DEFAULT_SWEEP_CONFIG)
    args = parser.parse_args()
    if args.command == "validation-bank":
        result = build_healthy_validation_bank(args.config)
    elif args.command == "audit-v7":
        result = audit_v7_checkpoint(args.config)
    elif args.command == "sweep":
        result = run_controlled_sweep(args.config, args.sweep_config)
    elif args.command == "ode-finalize":
        result = run_ode_resolution_and_finalize(args.config, args.sweep_config)
    else:
        result = materialize_final_audit(args.config, args.sweep_config)
    print(json.dumps(result, indent=2, default=lambda value: np.asarray(value).tolist()))


if __name__ == "__main__":
    main()
