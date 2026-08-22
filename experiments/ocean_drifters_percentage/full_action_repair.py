"""Ocean-local, cutoff-free Ritz machinery for the full-action repair pilot.

This module is deliberately separate from the production backend.  Its primary
solve never deletes a positive weighted-stiffness mode.  Relative spectral
cutoffs are exposed only through explicitly named diagnostic functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
import math
from typing import Iterable, Sequence

import numpy as np
from scipy import linalg


DEFAULT_STRUCTURAL_RELATIVE_TOLERANCE = 1.0e-12
DEFAULT_CUTOFFS = (1.0e-10, 1.0e-12, 1.0e-14)


@dataclass(frozen=True)
class TrialBasis:
    """Values and physical gradients of scalar trial potentials."""

    values: np.ndarray
    gradient_x: np.ndarray
    gradient_y: np.ndarray
    names: tuple[str, ...]

    def subset(self, indices: Sequence[int]) -> "TrialBasis":
        selected = np.asarray(indices, dtype=int)
        return TrialBasis(
            values=self.values[:, selected],
            gradient_x=self.gradient_x[:, selected],
            gradient_y=self.gradient_y[:, selected],
            names=tuple(self.names[index] for index in selected),
        )


@dataclass(frozen=True)
class StructuralOrthonormalization:
    """Map from an H-orthonormal coordinate vector to raw coefficients."""

    transform: np.ndarray
    raw_size: int
    rank: int
    gram_eigenvalues: np.ndarray
    threshold: float


@dataclass(frozen=True)
class RitzSolve:
    """Complete diagnostics for an untruncated finite-space solve."""

    certified: bool
    coefficients: np.ndarray
    orthonormal_coefficients: np.ndarray
    action: float
    action_energy: float
    action_load: float
    ritz_identity_relative_error: float
    relative_residual: float
    backward_error: float
    generalized_eigenvalues: np.ndarray
    generalized_forcing: np.ndarray
    generalized_contributions: np.ndarray
    cumulative_action: np.ndarray
    spectral_action: float
    spectral_identity_relative_error: float
    minimum_generalized_eigenvalue: float
    maximum_generalized_eigenvalue: float
    condition_proxy: float
    structural_rank: int
    raw_basis_size: int
    factorization: str
    failure_reason: str


def cosine_mode_pairs(maximum_mode: int) -> tuple[tuple[int, int], ...]:
    """Return the existing Neumann cosine ordering, excluding the constant."""
    if maximum_mode < 1:
        raise ValueError("maximum_mode must be positive")
    return tuple(
        (x_mode, y_mode)
        for y_mode in range(maximum_mode + 1)
        for x_mode in range(maximum_mode + 1)
        if (x_mode, y_mode) != (0, 0)
    )


def cosine_basis(
    points: np.ndarray,
    bounds: np.ndarray,
    maximum_mode: int,
) -> TrialBasis:
    """Evaluate the production cosine potentials and analytic gradients."""
    x = np.asarray(points, dtype=np.float64)
    domain = np.asarray(bounds, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("points must have shape [N,2]")
    if domain.shape != (4,):
        raise ValueError("bounds must have shape [4]")
    xmin, xmax, ymin, ymax = domain
    length_x = xmax - xmin
    length_y = ymax - ymin
    phase_x = np.pi * (x[:, 0] - xmin) / length_x
    phase_y = np.pi * (x[:, 1] - ymin) / length_y
    values: list[np.ndarray] = []
    gradient_x: list[np.ndarray] = []
    gradient_y: list[np.ndarray] = []
    names: list[str] = []
    for x_mode, y_mode in cosine_mode_pairs(maximum_mode):
        x_phase = x_mode * phase_x
        y_phase = y_mode * phase_y
        cos_x = np.cos(x_phase)
        cos_y = np.cos(y_phase)
        values.append(cos_x * cos_y)
        gradient_x.append(
            -(x_mode * np.pi / length_x) * np.sin(x_phase) * cos_y
        )
        gradient_y.append(
            -(y_mode * np.pi / length_y) * cos_x * np.sin(y_phase)
        )
        names.append(f"cosine_x{x_mode}_y{y_mode}")
    return TrialBasis(
        values=np.column_stack(values),
        gradient_x=np.column_stack(gradient_x),
        gradient_y=np.column_stack(gradient_y),
        names=tuple(names),
    )


def gaussian_sensor_basis(
    points: np.ndarray,
    centers: np.ndarray,
    sigma: float,
) -> TrialBasis:
    """Evaluate the same Gaussian observables and derivatives as tangent action."""
    x = np.asarray(points, dtype=np.float64)
    sensor_centers = np.asarray(centers, dtype=np.float64)
    width = float(sigma)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("points must have shape [N,2]")
    if sensor_centers.ndim != 2 or sensor_centers.shape[1] != 2:
        raise ValueError("centers must have shape [M,2]")
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("sigma must be positive")
    delta = x[:, None, :] - sensor_centers[None, :, :]
    values = np.exp(-0.5 * np.sum(delta * delta, axis=-1) / width**2)
    gradient = -(delta / width**2) * values[:, :, None]
    return TrialBasis(
        values=values,
        gradient_x=gradient[:, :, 0],
        gradient_y=gradient[:, :, 1],
        names=tuple(f"gaussian_sensor_{index}" for index in range(len(sensor_centers))),
    )


def enriched_basis(
    points: np.ndarray,
    bounds: np.ndarray,
    centers: np.ndarray,
    sigma: float,
    maximum_mode: int,
) -> TrialBasis:
    """Return the nested cosine-plus-sensor scalar-potential space."""
    cosine = cosine_basis(points, bounds, maximum_mode)
    sensor = gaussian_sensor_basis(points, centers, sigma)
    return TrialBasis(
        values=np.column_stack((cosine.values, sensor.values)),
        gradient_x=np.column_stack((cosine.gradient_x, sensor.gradient_x)),
        gradient_y=np.column_stack((cosine.gradient_y, sensor.gradient_y)),
        names=cosine.names + sensor.names,
    )


def cell_centers(bounds: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    """Fixed uniform reference points; independent of every projected law q."""
    xmin, xmax, ymin, ymax = np.asarray(bounds, dtype=np.float64)
    nx, ny = (int(value) for value in resolution)
    if nx < 2 or ny < 2:
        raise ValueError("reference resolution must be at least 2x2")
    x = xmin + (np.arange(nx) + 0.5) * (xmax - xmin) / nx
    y = ymin + (np.arange(ny) + 0.5) * (ymax - ymin) / ny
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.column_stack((xx.ravel(), yy.ravel()))


def fixed_physical_gram(
    basis: TrialBasis,
    *,
    length_scale: float,
) -> np.ndarray:
    """Uniform-reference mean-zero H1 Gram matrix.

    H(u,v) = E_mu[(u-E_mu u)(v-E_mu v)]
             + length_scale**2 E_mu[grad u . grad v].

    ``mu`` is the fixed uniform reference grid supplied by the caller.  Thus H
    depends on the physical trial functions but never on q or K equilibration.
    """
    scale = float(length_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("length_scale must be positive")
    values = np.asarray(basis.values, dtype=np.float64)
    centered = values - np.mean(values, axis=0, keepdims=True)
    inverse_count = 1.0 / len(values)
    gram = inverse_count * (
        centered.T @ centered
        + scale**2
        * (
            basis.gradient_x.T @ basis.gradient_x
            + basis.gradient_y.T @ basis.gradient_y
        )
    )
    return 0.5 * (gram + gram.T)


def structurally_orthonormalize(
    physical_gram: np.ndarray,
    *,
    relative_tolerance: float = DEFAULT_STRUCTURAL_RELATIVE_TOLERANCE,
) -> StructuralOrthonormalization:
    """Remove trial-representation dependence using H, never weighted K."""
    h = np.asarray(physical_gram, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("physical_gram must be square")
    if not np.isfinite(h).all():
        raise ValueError("physical_gram must be finite")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (h + h.T))
    maximum = max(float(eigenvalues[-1]), np.finfo(float).tiny)
    threshold = float(relative_tolerance) * maximum
    retained = eigenvalues > threshold
    if not np.any(retained):
        raise np.linalg.LinAlgError("the fixed physical Gram matrix has zero rank")
    transform = eigenvectors[:, retained] / np.sqrt(eigenvalues[retained])[None, :]
    return StructuralOrthonormalization(
        transform=transform,
        raw_size=h.shape[0],
        rank=int(np.sum(retained)),
        gram_eigenvalues=eigenvalues,
        threshold=threshold,
    )


def normalized_weights(log_q_mass: np.ndarray) -> np.ndarray:
    """Normalize log masses without flooring, clipping, or thresholding."""
    log_q = np.asarray(log_q_mass, dtype=np.float64).ravel()
    if not len(log_q) or not np.isfinite(log_q).all():
        raise ValueError("log_q_mass must be a nonempty finite array")
    shifted = log_q - float(np.max(log_q))
    unnormalized = np.exp(shifted)
    normalization = float(np.sum(unnormalized))
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise FloatingPointError("projected log masses cannot be normalized")
    return unnormalized / normalization


def assemble_variational_system(
    basis: TrialBasis,
    weights: np.ndarray,
    forcing: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble K and f=-E_q[h(phi-E_q phi)] in repository sign convention."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    h = np.asarray(forcing, dtype=np.float64).ravel()
    if len(w) != len(basis.values) or h.shape != w.shape:
        raise ValueError("basis, weights, and forcing must share their sample axis")
    if not np.isfinite(w).all() or not np.isfinite(h).all() or np.any(w < 0.0):
        raise ValueError("weights and forcing must be finite and weights nonnegative")
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("weights must have positive mass")
    w = w / total
    mean = w @ basis.values
    centered = basis.values - mean[None, :]
    f = -(centered.T @ (w * h))
    sqrt_weight = np.sqrt(w)[:, None]
    weighted_x = sqrt_weight * basis.gradient_x
    weighted_y = sqrt_weight * basis.gradient_y
    k = weighted_x.T @ weighted_x + weighted_y.T @ weighted_y
    return 0.5 * (k + k.T), f


def assemble_variational_system_longdouble(
    basis: TrialBasis,
    log_q_mass: np.ndarray,
    forcing: np.ndarray,
    *,
    chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Reassemble K and f with extended-range/precision accumulation.

    Trial functions are evaluated at the same float64 quadrature coordinates,
    but log-mass normalization, products, and reductions use ``np.longdouble``.
    This is a pilot certification path, not a different estimand.
    """
    log_q = np.asarray(log_q_mass, dtype=np.longdouble).ravel()
    h = np.asarray(forcing, dtype=np.longdouble).ravel()
    if len(log_q) != len(basis.values) or h.shape != log_q.shape:
        raise ValueError("basis, log masses, and forcing must share their sample axis")
    shifted = log_q - np.max(log_q)
    unnormalized = np.exp(shifted)
    normalization = np.sum(unnormalized, dtype=np.longdouble)
    if not np.isfinite(normalization) or normalization <= 0:
        raise FloatingPointError("long-double log masses cannot be normalized")
    weights = unnormalized / normalization
    size = basis.values.shape[1]
    mean = np.zeros(size, dtype=np.longdouble)
    for start in range(0, len(weights), int(chunk_size)):
        stop = min(start + int(chunk_size), len(weights))
        values = np.asarray(basis.values[start:stop], dtype=np.longdouble)
        mean += values.T @ weights[start:stop]
    stiffness = np.zeros((size, size), dtype=np.longdouble)
    load = np.zeros(size, dtype=np.longdouble)
    for start in range(0, len(weights), int(chunk_size)):
        stop = min(start + int(chunk_size), len(weights))
        local_weight = weights[start:stop]
        values = np.asarray(basis.values[start:stop], dtype=np.longdouble)
        gradient_x = np.asarray(basis.gradient_x[start:stop], dtype=np.longdouble)
        gradient_y = np.asarray(basis.gradient_y[start:stop], dtype=np.longdouble)
        load += (values - mean).T @ (local_weight * h[start:stop])
        stiffness += gradient_x.T @ (local_weight[:, None] * gradient_x)
        stiffness += gradient_y.T @ (local_weight[:, None] * gradient_y)
    stiffness = 0.5 * (stiffness + stiffness.T)
    return stiffness, -load


def _relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), np.finfo(float).tiny)


def solve_full_rank_ritz(
    stiffness: np.ndarray,
    forcing: np.ndarray,
    physical_gram: np.ndarray,
    *,
    structural_relative_tolerance: float = DEFAULT_STRUCTURAL_RELATIVE_TOLERANCE,
    residual_tolerance: float = 1.0e-9,
    backward_error_tolerance: float = 1.0e-12,
) -> RitzSolve:
    """Solve the structurally independent SPD problem without mode truncation."""
    k = np.asarray(stiffness, dtype=np.float64)
    f = np.asarray(forcing, dtype=np.float64).ravel()
    h = np.asarray(physical_gram, dtype=np.float64)
    if k.shape != h.shape or k.shape != (len(f), len(f)):
        raise ValueError("K, H, and f dimensions do not agree")
    if not np.isfinite(k).all() or not np.isfinite(f).all():
        raise ValueError("K and f must be finite")
    structural = structurally_orthonormalize(
        h, relative_tolerance=structural_relative_tolerance
    )
    transform = structural.transform
    reduced_k = transform.T @ (0.5 * (k + k.T)) @ transform
    reduced_k = 0.5 * (reduced_k + reduced_k.T)
    reduced_f = transform.T @ f

    eigenvalues, eigenvectors = np.linalg.eigh(reduced_k)
    generalized_forcing = eigenvectors.T @ reduced_f
    positive = eigenvalues > 0.0
    contributions = np.full_like(eigenvalues, np.nan)
    contributions[positive] = generalized_forcing[positive] ** 2 / eigenvalues[positive]
    cumulative = np.cumsum(np.where(positive, contributions, 0.0))
    spectral_action = (
        float(np.sum(contributions)) if np.all(positive) else math.nan
    )
    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])
    condition = maximum / minimum if minimum > 0.0 else math.inf

    factorization = "scipy.linalg.cho_factor/cho_solve"
    failure = ""
    coefficients_reduced = np.full(structural.rank, np.nan)
    try:
        factor, lower = linalg.cho_factor(
            reduced_k, lower=True, check_finite=True, overwrite_a=False
        )
        coefficients_reduced = linalg.cho_solve(
            (factor, lower), reduced_f, check_finite=True
        )
    except linalg.LinAlgError as exc:
        failure = f"full positive-mode Cholesky failed: {exc}"

    if np.isfinite(coefficients_reduced).all():
        residual = reduced_k @ coefficients_reduced - reduced_f
        residual_norm = float(
            np.linalg.norm(residual)
            / max(np.linalg.norm(reduced_f), np.finfo(float).tiny)
        )
        backward = float(
            np.linalg.norm(residual)
            / max(
                np.linalg.norm(reduced_k, ord=2)
                * np.linalg.norm(coefficients_reduced)
                + np.linalg.norm(reduced_f),
                np.finfo(float).tiny,
            )
        )
        raw_coefficients = transform @ coefficients_reduced
        action_energy = float(coefficients_reduced @ reduced_k @ coefficients_reduced)
        action_load = float(reduced_f @ coefficients_reduced)
        action = action_load
        identity = _relative_difference(action_energy, action_load)
        spectral_identity = (
            _relative_difference(action, spectral_action)
            if np.isfinite(spectral_action) else math.inf
        )
        certified = bool(
            minimum > 0.0
            and action >= 0.0
            and np.isfinite(action)
            and residual_norm <= residual_tolerance
            and backward <= backward_error_tolerance
            and identity <= 100.0 * np.finfo(float).eps * max(condition, 1.0)
        )
        if not certified and not failure:
            failure = (
                "untruncated double-precision solve did not meet the residual, "
                "backward-error, positivity, and Ritz-identity certificate"
            )
    else:
        raw_coefficients = np.full(k.shape[0], np.nan)
        action = action_energy = action_load = math.nan
        identity = residual_norm = backward = spectral_identity = math.inf
        certified = False

    return RitzSolve(
        certified=certified,
        coefficients=raw_coefficients,
        orthonormal_coefficients=coefficients_reduced,
        action=action,
        action_energy=action_energy,
        action_load=action_load,
        ritz_identity_relative_error=identity,
        relative_residual=residual_norm,
        backward_error=backward,
        generalized_eigenvalues=eigenvalues,
        generalized_forcing=generalized_forcing,
        generalized_contributions=contributions,
        cumulative_action=cumulative,
        spectral_action=spectral_action,
        spectral_identity_relative_error=spectral_identity,
        minimum_generalized_eigenvalue=minimum,
        maximum_generalized_eigenvalue=maximum,
        condition_proxy=condition,
        structural_rank=structural.rank,
        raw_basis_size=structural.raw_size,
        factorization=factorization,
        failure_reason=failure,
    )


def structurally_reduced_system(
    stiffness: np.ndarray,
    forcing: np.ndarray,
    physical_gram: np.ndarray,
    *,
    structural_relative_tolerance: float = DEFAULT_STRUCTURAL_RELATIVE_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray, StructuralOrthonormalization]:
    """Expose the H-orthonormal full system for precision certification."""
    structural = structurally_orthonormalize(
        physical_gram, relative_tolerance=structural_relative_tolerance
    )
    transform = structural.transform
    reduced_k = transform.T @ np.asarray(stiffness) @ transform
    reduced_k = 0.5 * (reduced_k + reduced_k.T)
    reduced_f = transform.T @ np.asarray(forcing).ravel()
    return reduced_k, reduced_f, structural


def generalized_cutoff_actions(
    solve: RitzSolve,
    cutoffs: Iterable[float] = DEFAULT_CUTOFFS,
) -> dict[float, float]:
    """Invariant truncation diagnostics; never the primary repaired action."""
    eigenvalues = solve.generalized_eigenvalues
    forcing = solve.generalized_forcing
    maximum = max(float(eigenvalues[-1]), np.finfo(float).tiny)
    output: dict[float, float] = {}
    for cutoff in cutoffs:
        tolerance = float(cutoff)
        retained = eigenvalues / maximum >= tolerance
        retained &= eigenvalues > 0.0
        output[tolerance] = float(
            np.sum(forcing[retained] ** 2 / eigenvalues[retained])
        )
    return output


def old_equilibrated_cutoff_actions(
    stiffness: np.ndarray,
    forcing: np.ndarray,
    cutoffs: Iterable[float] = DEFAULT_CUTOFFS,
) -> dict[float, float]:
    """Reproduce the old coordinate-equilibrated cutoff action for comparison."""
    k = np.asarray(stiffness, dtype=np.float64)
    f = np.asarray(forcing, dtype=np.float64).ravel()
    diagonal = np.diag(k)
    scale = np.zeros_like(diagonal)
    scale[diagonal > 0.0] = 1.0 / np.sqrt(diagonal[diagonal > 0.0])
    scaled_k = scale[:, None] * k * scale[None, :]
    scaled_k = 0.5 * (scaled_k + scaled_k.T)
    scaled_f = scale * f
    eigenvalues, eigenvectors = np.linalg.eigh(scaled_k)
    maximum = max(float(eigenvalues[-1]), np.finfo(float).tiny)
    projection = eigenvectors.T @ scaled_f
    output: dict[float, float] = {}
    for cutoff in cutoffs:
        retained = eigenvalues > float(cutoff) * maximum
        output[float(cutoff)] = float(
            np.sum(projection[retained] ** 2 / eigenvalues[retained])
        )
    return output


def transformed_system(
    stiffness: np.ndarray,
    forcing: np.ndarray,
    physical_gram: np.ndarray,
    coordinate_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Covariantly transform K, f, H for basis Phi_new=Phi_old B."""
    b = np.asarray(coordinate_map, dtype=np.float64)
    return b.T @ stiffness @ b, b.T @ forcing, b.T @ physical_gram @ b


def deterministic_coordinate_maps(size: int) -> dict[str, np.ndarray]:
    """Fixed, invertible maps used by every pilot basis-invariance audit."""
    if size < 1:
        raise ValueError("size must be positive")
    rng = np.random.default_rng(20260818)
    orthogonal, _ = np.linalg.qr(rng.normal(size=(size, size)))
    permutation = np.eye(size)[:, rng.permutation(size)]
    rescaling = np.diag(np.geomspace(0.5, 2.0, size))
    nonorthogonal = np.eye(size)
    if size > 1:
        nonorthogonal[np.arange(size - 1), np.arange(1, size)] = 0.05
    return {
        "diagonal_rescaling": rescaling,
        "orthogonal_mixing": orthogonal,
        "permutation": permutation,
        "moderate_nonorthogonal": nonorthogonal,
    }


def decimal_cholesky_action(
    stiffness: np.ndarray,
    forcing: np.ndarray,
    *,
    decimal_digits: int = 80,
) -> dict[str, float | bool | str | int]:
    """Certify the assembled full-rank linear algebra using Decimal arithmetic.

    This checks finite-precision solution error for the already assembled
    finite-dimensional matrix.  It does not certify quadrature assembly.
    """
    matrix = np.asarray(stiffness)
    vector = np.asarray(forcing).ravel()
    if matrix.shape != (len(vector), len(vector)):
        raise ValueError("stiffness and forcing dimensions do not agree")
    size = len(vector)
    with localcontext() as context:
        context.prec = int(decimal_digits)
        def decimal_value(value: np.generic | float) -> Decimal:
            if isinstance(value, np.longdouble):
                rendered = np.format_float_scientific(
                    value, precision=35, unique=False, trim="k"
                )
            else:
                rendered = repr(float(value))
            return Decimal(rendered)

        a = [
            [decimal_value(matrix[i, j]) for j in range(size)]
            for i in range(size)
        ]
        b = [decimal_value(value) for value in vector]
        lower = [[Decimal(0) for _ in range(size)] for _ in range(size)]
        try:
            for i in range(size):
                for j in range(i + 1):
                    value = a[i][j] - sum(
                        lower[i][k] * lower[j][k] for k in range(j)
                    )
                    if i == j:
                        if value <= 0:
                            raise ArithmeticError(
                                f"nonpositive Cholesky pivot {i}: {value}"
                            )
                        lower[i][j] = value.sqrt()
                    else:
                        lower[i][j] = value / lower[j][j]
            y = [Decimal(0) for _ in range(size)]
            for i in range(size):
                y[i] = (
                    b[i] - sum(lower[i][j] * y[j] for j in range(i))
                ) / lower[i][i]
            x = [Decimal(0) for _ in range(size)]
            for i in range(size - 1, -1, -1):
                x[i] = (
                    y[i]
                    - sum(lower[j][i] * x[j] for j in range(i + 1, size))
                ) / lower[i][i]
            action = sum(b[i] * x[i] for i in range(size))
            residual = [
                sum(a[i][j] * x[j] for j in range(size)) - b[i]
                for i in range(size)
            ]
            relative = (
                sum(value * value for value in residual).sqrt()
                / max(
                    sum(value * value for value in b).sqrt(),
                    Decimal(10) ** (1 - context.prec),
                )
            )
            return {
                "certified": bool(action >= 0 and relative < Decimal(10) ** (-decimal_digits // 2)),
                "action": float(action),
                "relative_residual": float(relative),
                "precision_decimal_digits": int(decimal_digits),
                "failure_reason": "",
            }
        except ArithmeticError as exc:
            return {
                "certified": False,
                "action": math.nan,
                "relative_residual": math.inf,
                "precision_decimal_digits": int(decimal_digits),
                "failure_reason": str(exc),
            }


__all__ = [
    "DEFAULT_CUTOFFS",
    "DEFAULT_STRUCTURAL_RELATIVE_TOLERANCE",
    "RitzSolve",
    "StructuralOrthonormalization",
    "TrialBasis",
    "assemble_variational_system",
    "assemble_variational_system_longdouble",
    "cell_centers",
    "cosine_basis",
    "cosine_mode_pairs",
    "decimal_cholesky_action",
    "deterministic_coordinate_maps",
    "enriched_basis",
    "fixed_physical_gram",
    "gaussian_sensor_basis",
    "generalized_cutoff_actions",
    "normalized_weights",
    "old_equilibrated_cutoff_actions",
    "solve_full_rank_ritz",
    "structurally_reduced_system",
    "structurally_orthonormalize",
    "transformed_system",
]
