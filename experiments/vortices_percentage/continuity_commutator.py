"""Boundary-kernel operators for the Vortices V2 continuity audit.

This module is diagnostic only.  It neither changes the V2 scientific raster
nor writes selection/validation artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from scipy.special import ndtr

from core import (
    diagonal_condition_estimate,
    make_grid,
    solve_v2,
)
from mfsi.raster import (
    _bilinear_cell_center_deposition_rect,
    _full_support_gaussian_matrix_1d,
    rasterize_projected_particles_positive_rect,
    rasterize_projected_particles_reflected_rect,
    reflected_flux_divergence_rect,
    reflected_gaussian_cell_mass_matrix_1d,
    reflected_particle_flux_rect,
)


SQRT_TWO_PI = float(np.sqrt(2.0 * np.pi))


def legacy_column_raster(nodes, weights, forcing, grid, *, bandwidth: float):
    """Historical column-normalized raster retained only for audit replay."""
    return rasterize_projected_particles_positive_rect(
        jnp.asarray(nodes, dtype=jnp.float64),
        jnp.asarray(weights, dtype=jnp.float64),
        jnp.asarray(forcing, dtype=jnp.float64),
        grid,
        bandwidth=float(bandwidth),
    )


def legacy_column_smoothed_field(
    nodes: Any,
    values: Any,
    grid,
    *,
    bandwidth: float,
) -> np.ndarray:
    """Historical CIC/column-normalized field retained only for this audit."""
    raw = _bilinear_cell_center_deposition_rect(
        jnp.asarray(nodes, dtype=jnp.float64),
        jnp.asarray(values, dtype=jnp.float64),
        grid,
    )
    kernel_x = _full_support_gaussian_matrix_1d(
        grid.x_centers(), float(bandwidth)
    )
    kernel_y = _full_support_gaussian_matrix_1d(
        grid.y_centers(), float(bandwidth)
    )
    return np.asarray(kernel_y @ raw @ kernel_x.T) / grid.cell_area


def legacy_zero_face_divergence(
    flux_x: np.ndarray,
    flux_y: np.ndarray,
    grid,
) -> np.ndarray:
    """Historical forced-zero-face divergence; never a V2 scientific gate."""
    faces_x = np.zeros((grid.ny, grid.nx + 1), dtype=np.float64)
    faces_y = np.zeros((grid.ny + 1, grid.nx), dtype=np.float64)
    faces_x[:, 1:-1] = 0.5 * (flux_x[:, :-1] + flux_x[:, 1:])
    faces_y[1:-1, :] = 0.5 * (flux_y[:-1, :] + flux_y[1:, :])
    return np.diff(faces_x, axis=1) / grid.dx + np.diff(faces_y, axis=0) / grid.dy


def truncated_gaussian_log_normalizer_gradient(
    points: Any,
    *,
    bandwidth: float,
    x_bounds: tuple[float, float] = (0.0, 2.0),
    y_bounds: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Analytic ``grad_x log Z(x)`` for a rectangular truncated Gaussian.

    For one coordinate, with normalized Gaussian density ``phi_h``,

    ``Z(x)=Phi((b-x)/h)-Phi((a-x)/h)`` and
    ``Z'(x)=[phi((a-x)/h)-phi((b-x)/h)]/h``.
    """
    points = np.asarray(points, dtype=np.float64)
    h = float(bandwidth)
    out = np.empty_like(points)
    for axis, (lower, upper) in enumerate((x_bounds, y_bounds)):
        coordinate = points[..., axis]
        lower_z = (lower - coordinate) / h
        upper_z = (upper - coordinate) / h
        normalization = ndtr(upper_z) - ndtr(lower_z)
        density_lower = np.exp(-0.5 * lower_z**2) / SQRT_TWO_PI
        density_upper = np.exp(-0.5 * upper_z**2) / SQRT_TWO_PI
        derivative = (density_lower - density_upper) / h
        out[..., axis] = derivative / np.maximum(normalization, 1.0e-300)
    return out


@dataclass(frozen=True)
class ColumnKernel1D:
    kernel: np.ndarray
    target_derivative: np.ndarray
    source_log_normalizer_gradient: np.ndarray


def column_normalized_kernel_1d(centers: Any, bandwidth: float) -> ColumnKernel1D:
    """Exact discrete column-normalized kernel and its derivative identities."""
    centers = np.asarray(centers, dtype=np.float64)
    displacement = centers[:, None] - centers[None, :]
    raw = np.exp(-0.5 * (displacement / float(bandwidth)) ** 2)
    normalization = np.sum(raw, axis=0, keepdims=True)
    kernel = raw / normalization
    target_derivative = -(displacement / float(bandwidth) ** 2) * kernel
    source_log_gradient = np.sum(
        (displacement / float(bandwidth) ** 2) * raw, axis=0
    ) / normalization[0]
    # d_x K = -d_y K - K d_x log Z, including the sign requested by the audit.
    source_derivative = -target_derivative - kernel * source_log_gradient[None, :]
    epsilon = max(1.0e-7, 1.0e-5 * float(bandwidth))
    plus_displacement = centers[:, None] - (centers[None, :] + epsilon)
    minus_displacement = centers[:, None] - (centers[None, :] - epsilon)
    plus = np.exp(-0.5 * (plus_displacement / float(bandwidth)) ** 2)
    minus = np.exp(-0.5 * (minus_displacement / float(bandwidth)) ** 2)
    plus /= np.sum(plus, axis=0, keepdims=True)
    minus /= np.sum(minus, axis=0, keepdims=True)
    finite = (plus - minus) / (2.0 * epsilon)
    if not np.allclose(source_derivative, finite, rtol=2.0e-6, atol=2.0e-8):
        raise RuntimeError("column-kernel derivative identity failed")
    return ColumnKernel1D(kernel, target_derivative, source_log_gradient)


def column_flux_and_commutator(
    nodes: Any,
    weights: Any,
    velocity: Any,
    grid,
    *,
    bandwidth: float,
) -> dict[str, np.ndarray]:
    """Natural ``S(q u)`` flux, analytic divergence, and commutator fields."""
    nodes = np.asarray(nodes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    raw_flux_x = np.asarray(
        _bilinear_cell_center_deposition_rect(
            jnp.asarray(nodes), jnp.asarray(weights * velocity[:, 0]), grid
        ),
        dtype=np.float64,
    )
    raw_flux_y = np.asarray(
        _bilinear_cell_center_deposition_rect(
            jnp.asarray(nodes), jnp.asarray(weights * velocity[:, 1]), grid
        ),
        dtype=np.float64,
    )
    x_operator = column_normalized_kernel_1d(np.asarray(grid.x_centers()), bandwidth)
    y_operator = column_normalized_kernel_1d(np.asarray(grid.y_centers()), bandwidth)
    kx, ky = x_operator.kernel, y_operator.kernel
    dx_kernel, dy_kernel = x_operator.target_derivative, y_operator.target_derivative
    flux_x = (ky @ raw_flux_x @ kx.T) / grid.cell_area
    flux_y = (ky @ raw_flux_y @ kx.T) / grid.cell_area
    divergence = (
        ky @ raw_flux_x @ dx_kernel.T
        + dy_kernel @ raw_flux_y @ kx.T
    ) / grid.cell_area
    grid_scalar = -(
        raw_flux_x * x_operator.source_log_normalizer_gradient[None, :]
        + raw_flux_y * y_operator.source_log_normalizer_gradient[:, None]
    )
    commutator_grid = (ky @ grid_scalar @ kx.T) / grid.cell_area

    continuous_gradient = truncated_gaussian_log_normalizer_gradient(
        nodes, bandwidth=bandwidth
    )
    particle_scalar = -np.sum(velocity * continuous_gradient, axis=-1)
    commutator_particle = legacy_column_smoothed_field(
        nodes,
        weights * particle_scalar,
        grid,
        bandwidth=bandwidth,
    )
    return {
        "flux_x": flux_x,
        "flux_y": flux_y,
        "analytic_divergence": divergence,
        "commutator_grid": commutator_grid,
        "commutator_particle": commutator_particle,
        "particle_grad_log_z": continuous_gradient,
    }


def reflected_particle_raster(
    nodes: Any,
    weights: Any,
    forcing: Any,
    grid,
    *,
    bandwidth: float,
    image_pairs: int = 4,
) -> dict[str, np.ndarray | float]:
    """Compatibility wrapper around the shared reflected raster primitive."""
    nodes = np.asarray(nodes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    forcing = np.asarray(forcing, dtype=np.float64)
    x_edges = np.linspace(grid.x_min, grid.x_max, grid.nx + 1)
    y_edges = np.linspace(grid.y_min, grid.y_max, grid.ny + 1)
    kx = np.asarray(reflected_gaussian_cell_mass_matrix_1d(
        x_edges, nodes[:, 0], bandwidth=bandwidth, image_pairs=image_pairs
    ))
    ky = np.asarray(reflected_gaussian_cell_mass_matrix_1d(
        y_edges, nodes[:, 1], bandwidth=bandwidth, image_pairs=image_pairs
    ))
    result = rasterize_projected_particles_reflected_rect(
        nodes,
        weights,
        forcing,
        grid,
        bandwidth=bandwidth,
        image_pairs=image_pairs,
    )
    mass = np.asarray(result.mass, dtype=np.float64)
    source = np.asarray(result.source, dtype=np.float64)
    normalization = float(np.sum(mass))
    return {
        "mass": mass,
        "q": np.asarray(result.q, dtype=np.float64),
        "source": source,
        "mass_error_before_normalization": normalization - 1.0,
        "source_before_center": float(result.source_mass_before_center),
        "kernel_x_column_error": float(np.max(np.abs(np.sum(kx, axis=0) - 1.0))),
        "kernel_y_column_error": float(np.max(np.abs(np.sum(ky, axis=0) - 1.0))),
    }


def reflected_particle_flux(
    nodes: Any,
    weights: Any,
    velocity: Any,
    grid,
    *,
    bandwidth: float,
    image_pairs: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper around the shared matched reflected flux."""
    flux_x, flux_y = reflected_particle_flux_rect(
        nodes,
        weights,
        velocity,
        grid,
        bandwidth=bandwidth,
        image_pairs=image_pairs,
    )
    return np.asarray(flux_x, dtype=np.float64), np.asarray(flux_y, dtype=np.float64)


def reflected_divergence(
    flux_x_faces: np.ndarray,
    flux_y_faces: np.ndarray,
    grid,
) -> np.ndarray:
    return np.asarray(reflected_flux_divergence_rect(flux_x_faces, flux_y_faces, grid))


def reflected_action_diagnostic(
    nodes: Any,
    weights: Any,
    forcing: Any,
    *,
    nx: int,
    ny: int,
    bandwidth: float,
) -> dict[str, float | bool]:
    grid = make_grid(nx, ny)
    raster = reflected_particle_raster(
        nodes, weights, forcing, grid, bandwidth=bandwidth
    )
    solved = solve_v2(raster["q"], raster["source"], grid)
    return {
        "action": float(solved.action[0]),
        "poisson_relative_residual": float(solved.relative_residual[0]),
        "compatible": bool(solved.compatible[0]),
        "component_count": int(solved.component_count[0]),
        "min_q": float(np.min(raster["q"])),
        "mass_error_before_normalization": float(raster["mass_error_before_normalization"]),
        "source_before_center": float(raster["source_before_center"]),
        "condition_estimate": diagonal_condition_estimate(raster["q"], grid),
    }
