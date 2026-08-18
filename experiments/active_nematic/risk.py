"""Periodic scientific-risk evaluation for normalized defect laws."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array


@dataclass(frozen=True)
class PeriodicHistogramGrid:
    """Cell-centered histogram grid on a 2-D or 3-D product of circles."""

    periods: tuple[float, ...]
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.periods) not in (2, 3) or len(self.periods) != len(self.shape):
            raise ValueError("periods and shape must define a 2-D or 3-D torus")
        if any(period <= 0.0 for period in self.periods) or any(int(n) < 2 for n in self.shape):
            raise ValueError("periods must be positive and every grid dimension >= 2")

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    def flat_bin_index(self, x: Array) -> Array:
        x = jnp.mod(jnp.asarray(x, dtype=jnp.float64), jnp.asarray(self.periods))
        scaled = jnp.floor(x / jnp.asarray(self.periods) * jnp.asarray(self.shape)).astype(jnp.int32)
        index = scaled[..., 0]
        for dimension in range(1, len(self.shape)):
            index = index * int(self.shape[dimension]) + scaled[..., dimension]
        return index


def histogram_mass(x: Array, weights: Array, grid: PeriodicHistogramGrid) -> Array:
    index = grid.flat_bin_index(x)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    weights = weights / jnp.maximum(jnp.sum(weights), 1.0e-300)
    return jnp.zeros(grid.size, dtype=jnp.float64).at[index].add(weights).reshape(grid.shape)


def multiscale_periodic_kernel_fft(
    grid: PeriodicHistogramGrid, bandwidths: Array
) -> Array:
    """FFT of an equal-weight chord-Gaussian kernel mixture."""
    bandwidths = jnp.asarray(bandwidths, dtype=jnp.float64)
    axes = []
    for period, count in zip(grid.periods, grid.shape):
        phase = 2.0 * jnp.pi * jnp.arange(count, dtype=jnp.float64) / count
        scale = float(period) / (2.0 * jnp.pi)
        axes.append(2.0 * scale**2 * (1.0 - jnp.cos(phase)))
    distance2 = jnp.zeros(grid.shape, dtype=jnp.float64)
    for dimension, values in enumerate(axes):
        reshape = [1] * len(grid.shape)
        reshape[dimension] = grid.shape[dimension]
        distance2 = distance2 + values.reshape(reshape)
    kernels = jnp.exp(-0.5 * distance2[..., None] / bandwidths**2)
    return jnp.fft.fftn(jnp.mean(kernels, axis=-1))


def periodic_grid_mmd2(p: Array, q: Array, kernel_fft: Array) -> Array:
    """Circular-convolution MMD² for normalized periodic histogram masses."""
    delta = jnp.asarray(p, dtype=jnp.float64) - jnp.asarray(q, dtype=jnp.float64)
    potential = jnp.fft.ifftn(jnp.fft.fftn(delta) * kernel_fft).real
    return jnp.maximum(jnp.sum(delta * potential), 0.0)


def trapezoid_weights(times: Array) -> Array:
    times = jnp.asarray(times, dtype=jnp.float64)
    if times.ndim != 1 or times.shape[0] < 2:
        raise ValueError("times must be one-dimensional with at least two entries")
    widths = times[1:] - times[:-1]
    if bool(jnp.any(widths <= 0.0)):
        raise ValueError("times must be strictly increasing")
    weights = jnp.zeros_like(times)
    weights = weights.at[0].set(0.5 * widths[0])
    weights = weights.at[-1].set(0.5 * widths[-1])
    if len(times) > 2:
        weights = weights.at[1:-1].set(0.5 * (widths[:-1] + widths[1:]))
    return weights / jnp.sum(weights)


def periodic_gaussian_kernel(
    x: Array,
    y: Array,
    *,
    periods: Array,
    bandwidths: Array,
) -> Array:
    """Product Gaussian kernel using smooth chord distances on each circle."""
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    periods = jnp.asarray(periods, dtype=jnp.float64)
    bandwidths = jnp.asarray(bandwidths, dtype=jnp.float64)
    if x.shape[-1] != y.shape[-1] or periods.shape != (x.shape[-1],):
        raise ValueError("samples and periods must share the same state dimension")
    if bandwidths.ndim != 1 or bool(jnp.any(bandwidths <= 0.0)):
        raise ValueError("bandwidths must be a positive one-dimensional array")
    phase = 2.0 * jnp.pi * (x[..., :, None, :] - y[..., None, :, :]) / periods
    scale = periods / (2.0 * jnp.pi)
    distance2 = jnp.sum(2.0 * scale**2 * (1.0 - jnp.cos(phase)), axis=-1)
    kernels = jnp.exp(-0.5 * distance2[..., None] / bandwidths**2)
    return jnp.mean(kernels, axis=-1)


def periodic_mmd2(
    x: Array,
    y: Array,
    *,
    periods: Array,
    bandwidths: Array,
) -> Array:
    """Biased empirical MMD², matching the vortices grid-mass convention."""
    kxx = periodic_gaussian_kernel(x, x, periods=periods, bandwidths=bandwidths)
    kyy = periodic_gaussian_kernel(y, y, periods=periods, bandwidths=bandwidths)
    kxy = periodic_gaussian_kernel(x, y, periods=periods, bandwidths=bandwidths)
    return jnp.maximum(jnp.mean(kxx) + jnp.mean(kyy) - 2.0 * jnp.mean(kxy), 0.0)


def periodic_weighted_mmd2(
    x: Array,
    x_weights: Array,
    y: Array,
    y_weights: Array,
    *,
    periods: Array,
    bandwidths: Array,
) -> Array:
    """Weighted form used for an I-projected reference particle law."""
    wx = jnp.asarray(x_weights, dtype=jnp.float64)
    wy = jnp.asarray(y_weights, dtype=jnp.float64)
    wx = wx / jnp.maximum(jnp.sum(wx), 1.0e-300)
    wy = wy / jnp.maximum(jnp.sum(wy), 1.0e-300)
    kxx = periodic_gaussian_kernel(x, x, periods=periods, bandwidths=bandwidths)
    kyy = periodic_gaussian_kernel(y, y, periods=periods, bandwidths=bandwidths)
    kxy = periodic_gaussian_kernel(x, y, periods=periods, bandwidths=bandwidths)
    value = wx @ kxx @ wx + wy @ kyy @ wy - 2.0 * (wx @ kxy @ wy)
    return jnp.maximum(value, 0.0)


@dataclass(frozen=True)
class RiskConfig:
    bandwidths: tuple[float, ...] = (0.5, 1.0, 2.0)
    count_weight: float = 0.0


class ScientificRisk(tuple):
    """Tuple-like result keeping normalized-law risk and count error distinct."""

    __slots__ = ()

    def __new__(cls, law_mmd2: Array, count_mse: Array, combined: Array):
        return tuple.__new__(cls, (law_mmd2, count_mse, combined))

    law_mmd2 = property(lambda self: self[0])
    count_mse = property(lambda self: self[1])
    combined = property(lambda self: self[2])


def trajectory_risk(
    reconstructed: Array,
    hidden: Array,
    times: Array,
    *,
    periods: Array,
    reconstructed_count: Array | None = None,
    hidden_count: Array | None = None,
    config: RiskConfig = RiskConfig(),
) -> ScientificRisk:
    """Time-integrated normalized-law MMD with optional auxiliary count MSE."""
    reconstructed = jnp.asarray(reconstructed, dtype=jnp.float64)
    hidden = jnp.asarray(hidden, dtype=jnp.float64)
    if reconstructed.ndim != 3 or hidden.ndim != 3 or reconstructed.shape[0] != hidden.shape[0]:
        raise ValueError("trajectory samples must have shape [time,sample,state_dim]")
    weights = trapezoid_weights(times)
    bandwidths = jnp.asarray(config.bandwidths, dtype=jnp.float64)
    rows = jax.vmap(
        lambda x, y: periodic_mmd2(x, y, periods=periods, bandwidths=bandwidths)
    )(reconstructed, hidden)
    law = jnp.sum(weights * rows)
    if (reconstructed_count is None) != (hidden_count is None):
        raise ValueError("both count trajectories must be supplied together")
    count = jnp.asarray(0.0, dtype=jnp.float64)
    if reconstructed_count is not None:
        count = jnp.sum(
            weights
            * (jnp.asarray(reconstructed_count) - jnp.asarray(hidden_count)) ** 2
        )
    return ScientificRisk(law, count, law + float(config.count_weight) * count)
