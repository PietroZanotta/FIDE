"""Finite-measure periodic RKHS risk for the two defect populations."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

try:
    from .risk import (
        PeriodicHistogramGrid,
        periodic_gaussian_kernel,
        periodic_grid_mmd2,
    )
except ImportError:  # pragma: no cover
    from risk import (
        PeriodicHistogramGrid,
        periodic_gaussian_kernel,
        periodic_grid_mmd2,
    )


Array = jax.Array


class FiniteMeasureRisk(NamedTuple):
    finite_measure_risk: Array
    shape_mmd: Array
    mass_error: Array


class TwoSpeciesRisk(NamedTuple):
    total: Array
    plus: FiniteMeasureRisk
    minus: FiniteMeasureRisk


def finite_measure_mmd2(
    x: Array,
    x_weights: Array,
    y: Array,
    y_weights: Array,
    *,
    periods: Array,
    bandwidths: Array,
) -> FiniteMeasureRisk:
    """Ordinary RKHS embedding distance with unnormalized nonnegative weights."""
    wx = jnp.asarray(x_weights, dtype=jnp.float64)
    wy = jnp.asarray(y_weights, dtype=jnp.float64)
    if wx.ndim != 1 or wy.ndim != 1 or len(wx) != len(x) or len(wy) != len(y):
        raise ValueError("finite-measure weights must align with samples")
    if bool(jnp.any(wx < 0.0) or jnp.any(wy < 0.0)):
        raise ValueError("finite-measure weights must be nonnegative")
    kxx = periodic_gaussian_kernel(x, x, periods=periods, bandwidths=bandwidths)
    kyy = periodic_gaussian_kernel(y, y, periods=periods, bandwidths=bandwidths)
    kxy = periodic_gaussian_kernel(x, y, periods=periods, bandwidths=bandwidths)
    finite = wx @ kxx @ wx + wy @ kyy @ wy - 2.0 * (wx @ kxy @ wy)
    mass_x = jnp.sum(wx)
    mass_y = jnp.sum(wy)
    px = wx / jnp.maximum(mass_x, 1.0e-300)
    py = wy / jnp.maximum(mass_y, 1.0e-300)
    shape = px @ kxx @ px + py @ kyy @ py - 2.0 * (px @ kxy @ py)
    return FiniteMeasureRisk(
        finite_measure_risk=jnp.maximum(finite, 0.0),
        shape_mmd=jnp.maximum(shape, 0.0),
        mass_error=(mass_x - mass_y) ** 2,
    )


def finite_histogram_mass(
    x: Array,
    weights: Array,
    grid: PeriodicHistogramGrid,
) -> Array:
    """Histogram a finite measure without normalizing away its total mass."""
    index = grid.flat_bin_index(x)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    return jnp.zeros(grid.size, dtype=jnp.float64).at[index].add(weights).reshape(
        grid.shape
    )


def periodic_grid_finite_mmd2(
    finite_p: Array,
    finite_q: Array,
    kernel_fft: Array,
) -> FiniteMeasureRisk:
    finite_p = jnp.asarray(finite_p, dtype=jnp.float64)
    finite_q = jnp.asarray(finite_q, dtype=jnp.float64)
    delta = finite_p - finite_q
    potential = jnp.fft.ifftn(jnp.fft.fftn(delta) * kernel_fft).real
    finite = jnp.maximum(jnp.sum(delta * potential), 0.0)
    mass_p = jnp.sum(finite_p)
    mass_q = jnp.sum(finite_q)
    normalized_p = finite_p / jnp.maximum(mass_p, 1.0e-300)
    normalized_q = finite_q / jnp.maximum(mass_q, 1.0e-300)
    shape = periodic_grid_mmd2(normalized_p, normalized_q, kernel_fft)
    return FiniteMeasureRisk(finite, shape, (mass_p - mass_q) ** 2)


def aggregate_two_species_risk(
    plus: FiniteMeasureRisk,
    minus: FiniteMeasureRisk,
    *,
    weight_plus: float = 1.0,
    weight_minus: float = 1.0,
) -> TwoSpeciesRisk:
    return TwoSpeciesRisk(
        total=float(weight_plus) * plus.finite_measure_risk
        + float(weight_minus) * minus.finite_measure_risk,
        plus=plus,
        minus=minus,
    )
