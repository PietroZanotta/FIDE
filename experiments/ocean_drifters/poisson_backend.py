"""Ocean-only adapter for weighted-Poisson backends.

The ocean laws are represented by log cell masses whose dynamic range can be
far larger than a float64 density can represent.  This module keeps that
experiment-specific requirement out of the shared Poisson API used by the toy
and vortices experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from mfsi.poisson_variational_tesseract import (
    VARIATIONAL_SOLVER_REVISION,
    VariationalPoissonConfig,
    solve_variational_poisson_batch_tesseract,
)


OCEAN_VARIATIONAL_POISSON_BACKEND = "tesseract_cpp_variational_ritz"


@dataclass(frozen=True)
class OceanVariationalPoissonConfig:
    """Configuration owned by the ocean experiment, not the shared solver."""

    dx: float
    maximum_mode: int = 5
    rank_relative_tolerance: float = 1.0e-12
    weak_relative_tolerance: float = 1.0e-9
    eigensolver_tolerance: float = 1.0e-20
    maximum_eigensolver_sweeps: int = 120

    def native_config(self) -> VariationalPoissonConfig:
        return VariationalPoissonConfig(
            dx=self.dx,
            maximum_mode=self.maximum_mode,
            rank_relative_tolerance=self.rank_relative_tolerance,
            weak_relative_tolerance=self.weak_relative_tolerance,
            eigensolver_tolerance=self.eigensolver_tolerance,
            maximum_eigensolver_sweeps=self.maximum_eigensolver_sweeps,
        )


class OceanPoissonBatchResult(NamedTuple):
    """Poisson-like common fields followed by variational audit diagnostics.

    ``relative_residual`` is the diagonally scaled Galerkin optimality
    residual in the explicitly retained numerical trial space. The complete
    and discarded-space residual diagnostics remain available separately. It
    is intentionally not labelled or interpreted as a strong-form
    finite-volume PDE residual.
    """

    action: np.ndarray
    potential: np.ndarray
    relative_residual: np.ndarray
    weighted_mean_potential: np.ndarray
    operator_floor: np.ndarray
    converged: np.ndarray
    diagnostics: dict[str, np.ndarray]


def solve_ocean_variational_poisson_batch(
    log_q_mass: Any,
    forcing: Any,
    cfg: OceanVariationalPoissonConfig,
) -> OceanPoissonBatchResult:
    """Solve the ocean weak problem without materializing or flooring ``q``."""
    raw = solve_variational_poisson_batch_tesseract(
        log_q_mass,
        forcing,
        cfg.native_config(),
    )
    diagnostics = {name: np.asarray(value) for name, value in raw.items()}
    action = np.asarray(diagnostics.pop("action"), dtype=np.float64)
    potential = np.asarray(diagnostics.pop("potential"), dtype=np.float64)
    relative_residual = np.asarray(
        diagnostics["retained_scaled_weak_relative_residual"], dtype=np.float64
    )
    weighted_mean_potential = np.asarray(
        diagnostics["gauge_residual"], dtype=np.float64
    )
    converged = np.asarray(diagnostics["converged"], dtype=bool)
    return OceanPoissonBatchResult(
        action=action,
        potential=potential,
        relative_residual=relative_residual,
        weighted_mean_potential=weighted_mean_potential,
        operator_floor=np.zeros_like(action),
        converged=converged,
        diagnostics=diagnostics,
    )


def solve_ocean_variational_poisson_quadrature(
    points: Any,
    log_q_mass: Any,
    forcing: Any,
    bounds: Any,
    cfg: OceanVariationalPoissonConfig,
) -> OceanPoissonBatchResult:
    """Solve the same Ritz problem on an ocean-local nonuniform quadrature.

    This path is used only to audit cells adaptively refined around a
    concentrated projected law. It retains the same result contract as the
    structured Tesseract endpoint.
    """
    x = np.asarray(points, dtype=np.float64)
    log_q = np.asarray(log_q_mass, dtype=np.float64)
    h = np.asarray(forcing, dtype=np.float64)
    domain = np.asarray(bounds, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("points must have shape [N,2]")
    if log_q.shape != (len(x),) or h.shape != (len(x),):
        raise ValueError("log_q_mass and forcing must have shape [N]")
    if domain.shape != (4,):
        raise ValueError("bounds must have shape [4]")
    if not np.isfinite(x).all() or not np.isfinite(log_q).all() or not np.isfinite(h).all():
        raise ValueError("quadrature inputs must contain only finite values")

    shifted = log_q - np.max(log_q)
    unnormalized = np.exp(shifted)
    normalization = float(np.sum(unnormalized))
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("quadrature log masses cannot be normalized")
    weights = unnormalized / normalization
    underflow_count = int(np.sum(unnormalized == 0.0))

    xmin, xmax, ymin, ymax = domain
    length_x = xmax - xmin
    length_y = ymax - ymin
    phase_x = np.pi * (x[:, 0] - xmin) / length_x
    phase_y = np.pi * (x[:, 1] - ymin) / length_y
    values = []
    gradient_x = []
    gradient_y = []
    for y_mode in range(cfg.maximum_mode + 1):
        for x_mode in range(cfg.maximum_mode + 1):
            if x_mode == 0 and y_mode == 0:
                continue
            x_phase = x_mode * phase_x
            y_phase = y_mode * phase_y
            values.append(np.cos(x_phase) * np.cos(y_phase))
            gradient_x.append(
                -(x_mode * np.pi / length_x)
                * np.sin(x_phase)
                * np.cos(y_phase)
            )
            gradient_y.append(
                -(y_mode * np.pi / length_y)
                * np.cos(x_phase)
                * np.sin(y_phase)
            )
    basis = np.column_stack(values)
    grad_x = np.column_stack(gradient_x)
    grad_y = np.column_stack(gradient_y)
    basis_mean = weights @ basis
    load = (basis - basis_mean).T @ (weights * h)
    gram = (
        grad_x.T @ (weights[:, None] * grad_x)
        + grad_y.T @ (weights[:, None] * grad_y)
    )
    diagonal = np.diag(gram)
    scale = np.zeros_like(diagonal)
    positive_diagonal = diagonal > 0.0
    scale[positive_diagonal] = 1.0 / np.sqrt(diagonal[positive_diagonal])
    scaled_gram = scale[:, None] * gram * scale[None, :]
    scaled_load = scale * load
    eigenvalues, eigenvectors = np.linalg.eigh(scaled_gram)
    maximum_eigenvalue = float(eigenvalues[-1])
    retained = eigenvalues > cfg.rank_relative_tolerance * max(
        maximum_eigenvalue, np.finfo(np.float64).tiny
    )
    retained_projection = eigenvectors[:, retained].T @ scaled_load
    amplitudes = -retained_projection / eigenvalues[retained]
    scaled_coefficients = eigenvectors[:, retained] @ amplitudes
    coefficients = scale * scaled_coefficients
    action = float(np.sum(retained_projection**2 / eigenvalues[retained]))
    load_value = float(coefficients @ load)
    objective = 0.5 * action + load_value

    scaled_residual = scaled_gram @ scaled_coefficients + scaled_load
    complete_residual = float(
        np.linalg.norm(scaled_residual)
        / max(np.linalg.norm(scaled_load), np.finfo(np.float64).tiny)
    )
    retained_algebraic_residual = (
        eigenvalues[retained] * amplitudes + retained_projection
    )
    retained_residual = float(
        np.linalg.norm(retained_algebraic_residual)
        / max(np.linalg.norm(retained_projection), np.finfo(np.float64).tiny)
    )
    discarded_projection = eigenvectors[:, ~retained].T @ scaled_load
    discarded_load = float(
        np.linalg.norm(discarded_projection)
        / max(np.linalg.norm(scaled_load), np.finfo(np.float64).tiny)
    )
    potential = (basis - basis_mean) @ coefficients
    # Center once more after the float64 matrix product.  The basis is centered
    # analytically, but concentrated laws and large coefficients can otherwise
    # leave a cancellation-sized numerical constant in the reconstructed field.
    # This is a pure Neumann gauge shift and cannot change the action.
    potential -= float(weights @ potential)
    gauge = float(weights @ potential)
    gauge_relative = abs(gauge) / max(
        float(np.sqrt(weights @ (potential * potential))),
        np.finfo(np.float64).tiny,
    )
    forcing_mean = float(weights @ h)
    forcing_rms = float(np.sqrt(weights @ (h * h)))
    compatibility_relative = abs(forcing_mean) / max(
        forcing_rms, np.finfo(np.float64).tiny
    )
    identity_error = abs(action + load_value) / max(
        abs(action), abs(load_value), np.finfo(np.float64).tiny
    )
    condition = (
        maximum_eigenvalue / float(eigenvalues[retained][0])
        if np.any(retained) else np.inf
    )
    converged = bool(
        np.any(retained)
        and np.isfinite(action)
        and retained_residual <= cfg.weak_relative_tolerance
    )
    diagnostics = {
        "objective": np.asarray([objective]),
        "weak_relative_residual": np.asarray([complete_residual]),
        "scaled_weak_relative_residual": np.asarray([complete_residual]),
        "retained_scaled_weak_relative_residual": np.asarray([retained_residual]),
        "discarded_scaled_load_relative_residual": np.asarray([discarded_load]),
        "gauge_residual": np.asarray([gauge]),
        "gauge_relative_residual": np.asarray([gauge_relative]),
        "compatibility_residual": np.asarray([forcing_mean]),
        "compatibility_relative_residual": np.asarray([compatibility_relative]),
        "energy_load_identity_relative_error": np.asarray([identity_error]),
        "condition_proxy": np.asarray([condition]),
        "retained_rank": np.asarray([int(np.sum(retained))]),
        "basis_size": np.asarray([basis.shape[1]]),
        "eigensolver_sweeps": np.asarray([1]),
        "quadrature_underflow_count": np.asarray([underflow_count]),
        "converged": np.asarray([float(converged)]),
    }
    return OceanPoissonBatchResult(
        action=np.asarray([action]),
        potential=potential[None, :],
        relative_residual=np.asarray([retained_residual]),
        weighted_mean_potential=np.asarray([gauge]),
        operator_floor=np.zeros(1),
        converged=np.asarray([converged]),
        diagnostics=diagnostics,
    )
__all__ = [
    "OCEAN_VARIATIONAL_POISSON_BACKEND",
    "OceanPoissonBatchResult",
    "OceanVariationalPoissonConfig",
    "VARIATIONAL_SOLVER_REVISION",
    "solve_ocean_variational_poisson_batch",
    "solve_ocean_variational_poisson_quadrature",
]
