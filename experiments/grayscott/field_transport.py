"""Field-state reference-path, CNN, tangent, and blind-spot infrastructure."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

import mfsi_components as core
from .observables import ShellDefinition, field_observables

Array = jax.Array


def maximal_same_index_coupling(minus_weights: np.ndarray, plus_weights: np.ndarray) -> np.ndarray:
    """Maximize same-IC mass while preserving separately tilted marginals."""
    minus = np.asarray(minus_weights, dtype=np.float64)
    plus = np.asarray(plus_weights, dtype=np.float64)
    if minus.shape != plus.shape or minus.ndim != 1:
        raise ValueError("paired endpoint weights must be equal-length vectors")
    minus, plus = minus / minus.sum(), plus / plus.sum()
    diagonal = np.minimum(minus, plus)
    remaining_minus, remaining_plus = minus - diagonal, plus - diagonal
    remaining_mass = float(remaining_minus.sum())
    coupling = np.diag(diagonal)
    if remaining_mass > 1e-15:
        coupling += np.outer(remaining_minus, remaining_plus) / remaining_mass
    coupling /= coupling.sum()
    return coupling


def independent_coupling(minus_weights: np.ndarray, plus_weights: np.ndarray) -> np.ndarray:
    """Independent exact-marginal endpoint coupling used as a diagnostic."""
    minus = np.asarray(minus_weights, dtype=np.float64)
    plus = np.asarray(plus_weights, dtype=np.float64)
    if minus.ndim != 1 or plus.ndim != 1:
        raise ValueError("endpoint weights must be vectors")
    minus, plus = minus / minus.sum(), plus / plus.sum()
    return np.outer(minus, plus)


def field_l2_cost(minus_bank: np.ndarray, plus_bank: np.ndarray) -> np.ndarray:
    """Pairwise mean squared field displacement without materializing [n,m,H,W]."""
    minus = np.asarray(minus_bank, dtype=np.float64).reshape((len(minus_bank), -1))
    plus = np.asarray(plus_bank, dtype=np.float64).reshape((len(plus_bank), -1))
    dimension = minus.shape[1]
    if plus.shape[1] != dimension:
        raise ValueError("endpoint fields must have the same flattened dimension")
    cost = (
        np.sum(minus * minus, axis=1)[:, None]
        + np.sum(plus * plus, axis=1)[None, :]
        - 2.0 * minus @ plus.T
    ) / dimension
    return np.maximum(cost, 0.0)


def geometric_l2_transport_coupling(
    minus_bank: np.ndarray,
    plus_bank: np.ndarray,
    minus_weights: np.ndarray,
    plus_weights: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Exact weighted geometric L2 OT via a sparse HiGHS transportation LP."""
    minus_weights = np.asarray(minus_weights, dtype=np.float64)
    plus_weights = np.asarray(plus_weights, dtype=np.float64)
    minus_weights /= minus_weights.sum()
    plus_weights /= plus_weights.sum()
    n_minus, n_plus = len(minus_weights), len(plus_weights)
    if len(minus_bank) != n_minus or len(plus_bank) != n_plus:
        raise ValueError("bank sizes and marginal weights do not agree")
    cost = field_l2_cost(minus_bank, plus_bank)

    # All row constraints and all but the last column constraint form a
    # full-row-rank representation of the transportation polytope.
    rows = np.repeat(np.arange(n_minus), n_plus)
    columns = np.arange(n_minus * n_plus)
    row_matrix = sparse.coo_matrix(
        (np.ones(n_minus * n_plus), (rows, columns)),
        shape=(n_minus, n_minus * n_plus),
    )
    kept_columns = n_plus - 1
    column_rows = np.tile(np.arange(kept_columns), n_minus)
    column_variables = (
        np.arange(n_minus)[:, None] * n_plus + np.arange(kept_columns)[None, :]
    ).ravel()
    column_matrix = sparse.coo_matrix(
        (np.ones(len(column_variables)), (column_rows, column_variables)),
        shape=(kept_columns, n_minus * n_plus),
    )
    equality = sparse.vstack([row_matrix, column_matrix], format="csr")
    rhs = np.concatenate([minus_weights, plus_weights[:-1]])
    result = linprog(
        cost.ravel(), A_eq=equality, b_eq=rhs, bounds=(0.0, None), method="highs",
        options={"dual_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9},
    )
    if not result.success:
        raise RuntimeError(f"geometric L2 transport LP failed: {result.message}")
    coupling = np.maximum(result.x.reshape((n_minus, n_plus)), 0.0)
    # Correct only roundoff-sized marginal errors, retaining the LP solution.
    coupling *= (minus_weights / np.maximum(coupling.sum(axis=1), 1e-300))[:, None]
    coupling *= (plus_weights / np.maximum(coupling.sum(axis=0), 1e-300))[None, :]
    coupling *= (minus_weights / np.maximum(coupling.sum(axis=1), 1e-300))[:, None]
    row_residual = float(np.max(np.abs(coupling.sum(axis=1) - minus_weights)))
    column_residual = float(np.max(np.abs(coupling.sum(axis=0) - plus_weights)))
    return coupling, {
        "solver": "scipy.optimize.linprog(method='highs') sparse transportation LP",
        "status": int(result.status), "message": str(result.message),
        "iterations": int(result.nit),
        "transport_cost_mean_squared_per_pixel": float(np.sum(coupling * cost)),
        "transport_displacement_rms": float(np.sqrt(np.sum(coupling * cost))),
        "maximum_marginal_residual": max(row_residual, column_residual),
        "positive_edge_count": int(np.count_nonzero(coupling > 1e-14)),
    }


def standardized_noise_bank(
    count: int, shape: tuple[int, int, int], seed: int, dtype=np.float32
) -> np.ndarray:
    """Repository-style centered, unit-RMS Gaussian noise for scalar fields."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(count,) + tuple(shape))
    noise -= noise.mean(axis=(-3, -2, -1), keepdims=True)
    rms = np.sqrt(np.mean(noise * noise, axis=(-3, -2, -1), keepdims=True))
    return (noise / np.maximum(rms, 1e-12)).astype(dtype)


def noisy_field_interpolant(
    minus: Array, plus: Array, noise: Array, time: Array | float, amplitude: float
) -> tuple[Array, Array]:
    """Linear bridge plus the validated smooth endpoint-zero sin(pi t) schedule."""
    minus, plus, noise = map(jnp.asarray, (minus, plus, noise))
    time = jnp.asarray(time, dtype=minus.dtype)
    gamma = jnp.asarray(amplitude, dtype=minus.dtype) * jnp.sin(jnp.pi * time)
    gamma_dot = jnp.asarray(amplitude, dtype=minus.dtype) * jnp.pi * jnp.cos(jnp.pi * time)
    while time.ndim < minus.ndim:
        time = time[..., None]
        gamma = gamma[..., None]
        gamma_dot = gamma_dot[..., None]
    state = (1.0 - time) * minus + time * plus + gamma * noise
    velocity = plus - minus + gamma_dot * noise
    return state, velocity


def sample_coupled_indices(coupling: np.ndarray, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    coupling = np.asarray(coupling, dtype=np.float64)
    rng = np.random.default_rng(seed)
    flat = rng.choice(coupling.size, size=count, replace=True, p=coupling.ravel())
    return np.unravel_index(flat, coupling.shape)


def linear_field_interpolant(minus: Array, plus: Array, time: Array | float) -> tuple[Array, Array]:
    """Experiment-B linear stochastic-interpolant family for paired fields."""
    minus, plus = jnp.asarray(minus), jnp.asarray(plus)
    time = jnp.asarray(time, dtype=minus.dtype)
    while time.ndim < minus.ndim:
        time = time[..., None]
    return (1.0 - time) * minus + time * plus, plus - minus


def sample_reference_interpolant(
    minus_bank: np.ndarray,
    plus_bank: np.ndarray,
    coupling: np.ndarray,
    time: float,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    minus_indices, plus_indices = sample_coupled_indices(coupling, count, seed)
    states, derivatives = linear_field_interpolant(
        jnp.asarray(minus_bank[minus_indices]), jnp.asarray(plus_bank[plus_indices]), time
    )
    return np.asarray(states), np.asarray(derivatives), minus_indices, plus_indices


def _time_channels(time: Array | float, batch: int, height: int, width: int, dtype, frequencies: int):
    time = jnp.asarray(time, dtype=dtype)
    if time.ndim == 0:
        time = jnp.full((batch,), time, dtype=dtype)
    time = jnp.broadcast_to(time, (batch,))
    k = 2.0 ** jnp.arange(frequencies, dtype=dtype)
    angles = 2.0 * jnp.pi * time[:, None] * k[None]
    embedding = jnp.concatenate([time[:, None], jnp.sin(angles), jnp.cos(angles)], axis=1)
    return jnp.broadcast_to(embedding[:, :, None, None], (batch, embedding.shape[1], height, width))


def _init_conv(key: Array, output_channels: int, input_channels: int, kernel_size: int, dtype=jnp.float32):
    scale = jnp.asarray(np.sqrt(2.0 / (input_channels * kernel_size * kernel_size)), dtype=dtype)
    weight = scale * jax.random.normal(
        key, (output_channels, input_channels, kernel_size, kernel_size), dtype=dtype
    )
    return {"weight": weight, "bias": jnp.zeros((output_channels,), dtype=dtype)}


def init_periodic_reference_cnn(
    key: Array,
    *,
    input_channels: int = 1,
    hidden_channels: Sequence[int] = (24, 24, 24, 24),
    dilations: Sequence[int] = (1, 2, 4, 1),
    kernel_size: int = 3,
    time_frequencies: int = 3,
    dtype=jnp.float32,
) -> dict:
    if len(hidden_channels) != len(dilations):
        raise ValueError("hidden_channels and dilations must match")
    keys = jax.random.split(key, len(hidden_channels) + 1)
    channels = input_channels + 1 + 2 * time_frequencies
    layers = []
    for subkey, output_channels, dilation in zip(keys[:-1], hidden_channels, dilations):
        layers.append({**_init_conv(subkey, output_channels, channels, kernel_size, dtype),
                       "dilation": int(dilation)})
        channels = output_channels
    output = _init_conv(keys[-1], input_channels, channels, kernel_size, dtype)
    return {
        "layers": layers, "output": output, "time_frequencies": int(time_frequencies),
        "kernel_size": int(kernel_size),
    }


def periodic_conv2d(inputs: Array, weight: Array, bias: Array, dilation: int = 1) -> Array:
    radius = (weight.shape[-1] // 2) * dilation
    padded = jnp.pad(inputs, ((0, 0), (0, 0), (radius, radius), (radius, radius)), mode="wrap")
    output = jax.lax.conv_general_dilated(
        padded, weight, window_strides=(1, 1), padding="VALID",
        rhs_dilation=(dilation, dilation), dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    return output + bias[None, :, None, None]


def periodic_reference_cnn(params: dict, time: Array | float, fields: Array) -> Array:
    fields = jnp.asarray(fields)
    if fields.ndim != 4:
        raise ValueError("fields must have shape [B,C,H,W]")
    batch, _, height, width = fields.shape
    time_features = _time_channels(
        time, batch, height, width, fields.dtype, params["time_frequencies"]
    )
    hidden = jnp.concatenate([fields, time_features], axis=1)
    for layer in params["layers"]:
        hidden = jax.nn.silu(periodic_conv2d(
            hidden, layer["weight"], layer["bias"], layer["dilation"]
        ))
    return periodic_conv2d(hidden, params["output"]["weight"], params["output"]["bias"])


def reference_flow_matching_loss(params: dict, times: Array, states: Array, targets: Array) -> Array:
    prediction = periodic_reference_cnn(params, times, states)
    return jnp.mean(jnp.sum((prediction - targets) ** 2, axis=(1, 2, 3)))


def standardized_field_observables(
    fields: Array,
    shells: ShellDefinition,
    components: Sequence[str],
    center: Array,
    scale: Array,
) -> Array:
    return (field_observables(fields, shells, components) - jnp.asarray(center)) / jnp.asarray(scale)


def field_jphi_times_velocity(
    fields: Array,
    velocity: Array,
    observable_fn: Callable[[Array], Array],
) -> Array:
    """Compute one field-valued J_Phi @ u per batch item without a Jacobian."""
    return jax.jvp(observable_fn, (fields,), (velocity,))[1]


def field_observable_jacobian(fields: Array, observable_single: Callable[[Array], Array]) -> Array:
    return jax.vmap(jax.jacrev(observable_single))(fields)


def weighted_field_tangent_velocity(
    fields: Array,
    reference_velocity: Array,
    weights: Array,
    observable_single: Callable[[Array], Array],
    target_rate: Array | None = None,
) -> tuple[Array, dict]:
    """Moment tangent correction with channel/spatial contractions."""
    fields, reference_velocity, weights = map(jnp.asarray, (fields, reference_velocity, weights))
    weights = weights / jnp.sum(weights)
    jacobian = field_observable_jacobian(fields, observable_single)
    flat_jacobian = jacobian.reshape((fields.shape[0], jacobian.shape[1], -1))
    flat_velocity = reference_velocity.reshape((fields.shape[0], -1))
    rates = jnp.einsum("brd,bd->br", flat_jacobian, flat_velocity)
    current_rate = jnp.einsum("b,br->r", weights, rates)
    if target_rate is None:
        target_rate = jnp.zeros_like(current_rate)
    gram = jnp.einsum("b,brd,bsd->rs", weights, flat_jacobian, flat_jacobian)
    coefficient, rank, condition = core._stable_cov_solve(
        gram, current_rate - target_rate, damping=core.DEFAULT_DAMPING
    )
    correction = -jnp.einsum("brd,r->bd", flat_jacobian, coefficient).reshape(fields.shape)
    corrected = reference_velocity + correction
    corrected_rates = jnp.einsum(
        "brd,bd->br", flat_jacobian, corrected.reshape((fields.shape[0], -1))
    )
    residual_rate = jnp.einsum("b,br->r", weights, corrected_rates) - target_rate
    return corrected, {
        "gram": gram, "coefficient": coefficient, "rank": rank, "condition": condition,
        "uncorrected_rate": current_rate, "corrected_rate_residual": residual_rate,
        "correction_energy": jnp.einsum("b,b...->", weights, correction * correction),
    }


def smooth_hidden_observables(
    fields: Array,
    *,
    threshold: float,
    temperature: float = 0.02,
    epsilon: float = 1e-8,
    heldout_shell: ShellDefinition = ShellDefinition((0.27,), (0.055,)),
) -> Array:
    """Smooth TV, anisotropy, soft area/perimeter, and held-out power."""
    fields = jnp.asarray(fields)
    scalar = fields[:, 0]
    dx = 0.5 * (jnp.roll(scalar, -1, -1) - jnp.roll(scalar, 1, -1))
    dy = 0.5 * (jnp.roll(scalar, -1, -2) - jnp.roll(scalar, 1, -2))
    smooth_tv = jnp.mean(jnp.sqrt(dx * dx + dy * dy + epsilon), axis=(-2, -1))
    jxx, jyy = jnp.mean(dx * dx, axis=(-2, -1)), jnp.mean(dy * dy, axis=(-2, -1))
    jxy = jnp.mean(dx * dy, axis=(-2, -1))
    anisotropy = jnp.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy + epsilon) / (jxx + jyy + epsilon)
    soft = jax.nn.sigmoid((scalar - threshold) / temperature)
    soft_area = jnp.mean(soft, axis=(-2, -1))
    sx = jnp.roll(soft, -1, -1) - soft
    sy = jnp.roll(soft, -1, -2) - soft
    soft_perimeter = jnp.mean(jnp.sqrt(sx * sx + sy * sy + epsilon), axis=(-2, -1))
    heldout_power = field_observables(fields, heldout_shell, ("shell_1",))[:, 0]
    return jnp.stack([smooth_tv, anisotropy, soft_area, soft_perimeter, heldout_power], axis=-1)


def tangent_blindspot_diagnostic(
    times: Array,
    states_by_time: Array,
    weights_by_time: Array,
    reference_velocity_fn: Callable[[Array, Array], Array],
    observable_single: Callable[[Array], Array],
    hidden_single: Callable[[Array], Array],
    hidden_scales: Array,
) -> dict:
    """Oracle-at-target tangent hidden-rate residual on an interior time grid."""
    times = jnp.asarray(times)
    target_hidden = jax.vmap(
        lambda states, weights: jnp.einsum("b,br->r", weights, jax.vmap(hidden_single)(states))
    )(states_by_time, weights_by_time)
    finite_difference = (target_hidden[2:] - target_hidden[:-2]) / (times[2:, None] - times[:-2, None])
    predicted = []
    tangent_rate_residuals = []
    for index in range(1, len(times) - 1):
        states, weights, time = states_by_time[index], weights_by_time[index], times[index]
        reference = reference_velocity_fn(time, states)
        tangent, diagnostics = weighted_field_tangent_velocity(
            states, reference, weights, observable_single
        )
        hidden_rate = jax.jvp(
            lambda value: jax.vmap(hidden_single)(value), (states,), (tangent,)
        )[1]
        predicted.append(jnp.einsum("b,br->r", weights, hidden_rate))
        tangent_rate_residuals.append(diagnostics["corrected_rate_residual"])
    predicted = jnp.stack(predicted)
    residual = finite_difference - predicted
    normalized = residual / jnp.asarray(hidden_scales)
    denominator = jnp.sum((finite_difference / jnp.asarray(hidden_scales)) ** 2, axis=1) + 1e-12
    score_by_time = jnp.sum(normalized * normalized, axis=1) / denominator
    return {
        "target_hidden_means": target_hidden, "target_hidden_derivative": finite_difference,
        "tangent_predicted_hidden_derivative": predicted, "residual": residual,
        "normalized_residual": normalized, "score_by_time": score_by_time,
        "B_tan": jnp.mean(score_by_time),
        "tangent_measured_rate_residual": jnp.stack(tangent_rate_residuals),
    }
