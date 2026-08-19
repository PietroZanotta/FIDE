"""Float64 concentration and direct weighted-gradient diagnostics for ocean q_t."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
from scipy import linalg
from scipy.linalg.lapack import dgeqrf

from .full_action_repair import TrialBasis, normalized_weights


FLOAT64_UNIT_ROUNDOFF = 2.0**-53


@dataclass(frozen=True)
class DirectGradientDiagnostic:
    """Scalars and spectra from a direct QR/SVD of C=B R_H^{-1}."""

    scalars: dict[str, float | int | bool | str]
    singular_values: np.ndarray
    squared_singular_values: np.ndarray
    generalized_load_coefficients: np.ndarray
    action_contributions: np.ndarray
    normal_eigenvalues: np.ndarray
    normal_spectrum_discrepancy: np.ndarray


def _safe_log10(value: float) -> float:
    if value > 0.0:
        return math.log10(value)
    if value == 0.0:
        return -math.inf
    return math.nan


def _ratio(numerator: float, denominator: float) -> float:
    if denominator > 0.0:
        return numerator / denominator
    if numerator > 0.0:
        return math.inf
    return math.nan


def concentration_statistics(
    points: np.ndarray,
    log_q_mass: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Measure actual projected-law concentration using float64 weights."""
    x = np.asarray(points, dtype=np.float64)
    log_q = np.asarray(log_q_mass, dtype=np.float64).ravel()
    if x.shape != (len(log_q), 2):
        raise ValueError("points and log_q_mass dimensions do not agree")
    weights = normalized_weights(log_q)
    positive = weights > 0.0
    mean = weights @ x
    centered = x - mean
    covariance = centered.T @ (weights[:, None] * centered)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    minor = max(float(eigenvalues[0]), 0.0)
    major = max(float(eigenvalues[-1]), 0.0)
    area = math.sqrt(major * minor)
    anisotropy = _ratio(major, minor)
    digest = hashlib.sha256(np.ascontiguousarray(weights).tobytes()).hexdigest()
    return weights, {
        "projected_weight_normalization": "float64_max_shifted_exponential_sum",
        "projected_weight_sha256": digest,
        "projected_weight_sum": float(np.sum(weights)),
        "log_weight_range": float(np.max(log_q) - np.min(log_q)),
        "zero_weight_count": int(np.sum(~positive)),
        "zero_weight_fraction": float(np.mean(~positive)),
        "minimum_positive_weight": (
            float(np.min(weights[positive])) if np.any(positive) else math.nan
        ),
        "projected_weight_ess": float(1.0 / np.sum(weights * weights)),
        "projected_mean_x_km": float(mean[0]),
        "projected_mean_y_km": float(mean[1]),
        "cov_eig_major": major,
        "cov_eig_minor": minor,
        "cov_anisotropy": anisotropy,
        "cov_det": major * minor,
        "cov_area_scale": area,
        "major_std": math.sqrt(major),
        "minor_std": math.sqrt(minor),
        "log10_cov_eig_minor": _safe_log10(minor),
        "log10_cov_area_scale": _safe_log10(area),
    }


def direct_weighted_gradient_diagnostic(
    basis: TrialBasis,
    weights: np.ndarray,
    forcing: np.ndarray,
    physical_orthonormal_transform: np.ndarray,
) -> DirectGradientDiagnostic:
    """Diagnose C directly in float64 before assembling its normal matrix.

    A LAPACK QR factorization reduces the tall matrix C without forming C^T C;
    a LAPACK divide-and-conquer SVD of the small R factor then gives the direct
    singular spectrum and right singular vectors.  Only after that calculation
    is the float64 normal matrix assembled for comparison.
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    h = np.asarray(forcing, dtype=np.float64).ravel()
    transform = np.asarray(physical_orthonormal_transform, dtype=np.float64)
    if len(w) != len(basis.values) or h.shape != w.shape:
        raise ValueError("basis, weights, and forcing must share their sample axis")
    if transform.shape[0] != basis.values.shape[1]:
        raise ValueError("physical transform and basis dimensions do not agree")
    if not np.isfinite(w).all() or not np.isfinite(h).all():
        raise ValueError("weights and forcing must be finite")

    sample_count = len(w)
    dimension = transform.shape[1]
    sqrt_weight = np.sqrt(w)
    direct = np.empty((2 * sample_count, dimension), dtype=np.float64, order="F")
    direct[:sample_count] = (basis.gradient_x @ transform) * sqrt_weight[:, None]
    direct[sample_count:] = (basis.gradient_y @ transform) * sqrt_weight[:, None]

    # Primary conditioning diagnostic: direct QR/SVD, with no normal matrix.
    qr_storage, _, _, info = dgeqrf(direct, overwrite_a=True)
    if info != 0:
        raise np.linalg.LinAlgError(f"LAPACK dgeqrf failed with info={info}")
    triangular = np.triu(qr_storage[:dimension, :dimension])
    _, singular_values, right_vectors_t = linalg.svd(
        triangular,
        full_matrices=False,
        compute_uv=True,
        overwrite_a=False,
        check_finite=True,
        lapack_driver="gesdd",
    )
    sigma_max = float(singular_values[0])
    sigma_min = float(singular_values[-1])
    kappa_c = _ratio(sigma_max, sigma_min)
    direct_roundoff = FLOAT64_UNIT_ROUNDOFF * kappa_c
    normal_roundoff = FLOAT64_UNIT_ROUNDOFF * kappa_c**2
    eig_resolvability = _ratio(1.0, normal_roundoff)

    basis_mean = w @ basis.values
    raw_load = -(basis.values - basis_mean).T @ (w * h)
    whitened_load = transform.T @ raw_load
    generalized_load = right_vectors_t @ whitened_load
    squared_singular = singular_values * singular_values
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        contributions = generalized_load * generalized_load / squared_singular
    direct_action = float(np.sum(contributions))
    normal_unresolved = squared_singular <= (
        FLOAT64_UNIT_ROUNDOFF * sigma_max * sigma_max
    )
    unresolved_action = float(np.sum(contributions[normal_unresolved]))
    unresolved_fraction = _ratio(unresolved_action, direct_action)

    # Comparison diagnostic assembled only after the direct spectrum is known.
    raw_normal = (
        basis.gradient_x.T @ (w[:, None] * basis.gradient_x)
        + basis.gradient_y.T @ (w[:, None] * basis.gradient_y)
    )
    whitened_normal = transform.T @ raw_normal @ transform
    whitened_normal = 0.5 * (whitened_normal + whitened_normal.T)
    normal_eigenvalues = np.linalg.eigvalsh(whitened_normal)
    squared_ascending = squared_singular[::-1]
    discrepancy = normal_eigenvalues - squared_ascending
    scaled_discrepancy = float(
        np.max(np.abs(discrepancy))
        / max(sigma_max * sigma_max, np.finfo(float).tiny)
    )

    active = w > 0.0
    eq_h2 = float(w @ (h * h))
    return DirectGradientDiagnostic(
        scalars={
            "sigma_max": sigma_max,
            "sigma_min": sigma_min,
            "log10_sigma_min": _safe_log10(sigma_min),
            "kappa_C": kappa_c,
            "log10_kappa_C": _safe_log10(kappa_c),
            "float64_unit_roundoff": FLOAT64_UNIT_ROUNDOFF,
            "direct_roundoff_amplification": direct_roundoff,
            "normal_roundoff_amplification": normal_roundoff,
            "log10_normal_roundoff_amplification": _safe_log10(normal_roundoff),
            "normal_eigenvalue_resolvability_ratio": eig_resolvability,
            "direct_numerical_rank_resolved": bool(
                np.isfinite(direct_roundoff) and direct_roundoff < 1.0
            ),
            "direct_svd_rank": int(np.sum(singular_values > 0.0)),
            "smallest_direct_sigma_squared": float(squared_singular[-1]),
            "smallest_assembled_normal_eigenvalue": float(normal_eigenvalues[0]),
            "most_negative_assembled_normal_eigenvalue": float(
                min(normal_eigenvalues[0], 0.0)
            ),
            "normal_spectrum_maximum_scaled_discrepancy": scaled_discrepancy,
            "direct_svd_action": direct_action,
            "normal_unresolved_direction_action": unresolved_action,
            "normal_unresolved_direction_action_fraction": unresolved_fraction,
            "Eq_h2": eq_h2,
            "max_abs_h_on_positive_weight_support": (
                float(np.max(np.abs(h[active]))) if np.any(active) else math.nan
            ),
            "forcing_all_finite": bool(np.isfinite(h).all()),
            "diagnostic_precision": "float64",
            "positive_mode_truncation_used": False,
        },
        singular_values=singular_values,
        squared_singular_values=squared_singular,
        generalized_load_coefficients=generalized_load,
        action_contributions=contributions,
        normal_eigenvalues=normal_eigenvalues,
        normal_spectrum_discrepancy=discrepancy,
    )


__all__ = [
    "DirectGradientDiagnostic",
    "FLOAT64_UNIT_ROUNDOFF",
    "concentration_statistics",
    "direct_weighted_gradient_diagnostic",
]
