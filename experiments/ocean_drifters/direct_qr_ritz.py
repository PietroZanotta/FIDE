"""Float64 direct-operator Ritz solver for the ocean full-action repair.

The primary path in this module never forms a weighted stiffness/normal
matrix.  Structural dependence is resolved once in the fixed, q-independent
physical norm.  The resulting weighted-gradient operator is whitened with a
triangular solve and its action is evaluated from a Householder QR factor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import linalg
from scipy.linalg.lapack import dgeqrf

from .full_action_repair import (
    DEFAULT_STRUCTURAL_RELATIVE_TOLERANCE,
    TrialBasis,
)


FLOAT64_UNIT_ROUNDOFF = 2.0**-53


@dataclass(frozen=True)
class PreparedDirectRitzBasis:
    """A fixed trial space whitened in the q-independent physical norm."""

    values: np.ndarray
    gradient_x_whitened: np.ndarray
    gradient_y_whitened: np.ndarray
    raw_to_whitened: np.ndarray
    structural_basis: np.ndarray
    physical_cholesky_upper: np.ndarray
    raw_basis_size: int
    structural_rank: int
    physical_gram_eigenvalues: np.ndarray
    structural_threshold: float


@dataclass(frozen=True)
class WhitenedDirectRitzSolve:
    """Direct QR action plus an independent, untruncated SVD audit."""

    action_qr: float
    action_svd: float
    qr_svd_relative_discrepancy: float
    qr_triangular: np.ndarray
    singular_values: np.ndarray
    right_singular_vectors_t: np.ndarray
    generalized_load_coefficients: np.ndarray
    action_contributions: np.ndarray
    whitened_coefficients: np.ndarray
    sigma_max: float
    sigma_min: float
    kappa_c: float
    u_kappa_c: float
    u_kappa_c_squared: float
    lapack_full_column_rank: bool
    qr_success: bool
    svd_success: bool
    failure_reason: str


@dataclass(frozen=True)
class DirectRitzSolve:
    """Prepared-space solve with load and trial-space metadata."""

    direct: WhitenedDirectRitzSolve
    raw_load: np.ndarray
    whitened_load: np.ndarray
    raw_coefficients: np.ndarray
    structural_rank: int
    raw_basis_size: int


def _symmetric_relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(
        abs(left), abs(right), np.finfo(np.float64).tiny
    )


def _structural_basis_from_fixed_gram(
    physical_gram: np.ndarray,
    relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Select structural directions using only the fixed physical Gram."""
    gram = np.asarray(physical_gram, dtype=np.float64)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("physical_gram must be square")
    if not np.isfinite(gram).all():
        raise ValueError("physical_gram must be finite")
    eigenvalues, eigenvectors = linalg.eigh(
        0.5 * (gram + gram.T), check_finite=True, driver="evd"
    )
    maximum = max(float(eigenvalues[-1]), np.finfo(np.float64).tiny)
    threshold = float(relative_tolerance) * maximum
    retained = eigenvalues > threshold
    if not np.any(retained):
        raise np.linalg.LinAlgError("fixed physical Gram has zero structural rank")
    return eigenvectors[:, retained], eigenvalues, threshold


def prepare_direct_ritz_basis(
    basis: TrialBasis,
    physical_gram: np.ndarray,
    *,
    structural_relative_tolerance: float = DEFAULT_STRUCTURAL_RELATIVE_TOLERANCE,
    structural_basis: np.ndarray | None = None,
) -> PreparedDirectRitzBasis:
    """Whiten a structurally independent trial space without matrix inversion.

    ``structural_basis`` is optional and is primarily useful for coordinate
    invariance audits, where the already-frozen structural subspace must be
    represented consistently after an invertible basis change.
    """
    gram = np.asarray(physical_gram, dtype=np.float64)
    raw_size = basis.values.shape[1]
    if gram.shape != (raw_size, raw_size):
        raise ValueError("basis and physical_gram dimensions do not agree")
    if structural_basis is None:
        independent, eigenvalues, threshold = _structural_basis_from_fixed_gram(
            gram, structural_relative_tolerance
        )
    else:
        independent = np.asarray(structural_basis, dtype=np.float64)
        if independent.ndim != 2 or independent.shape[0] != raw_size:
            raise ValueError("structural_basis has incompatible dimensions")
        eigenvalues = linalg.eigvalsh(
            0.5 * (gram + gram.T), check_finite=True
        )
        threshold = math.nan

    reduced_gram = independent.T @ gram @ independent
    reduced_gram = 0.5 * (reduced_gram + reduced_gram.T)
    upper = linalg.cholesky(
        reduced_gram, lower=False, overwrite_a=False, check_finite=True
    )

    # T = S R_H^{-1}.  Solve R_H^T T^T = S^T; never form R_H^{-1}.
    raw_to_whitened_t = linalg.solve_triangular(
        upper,
        independent.T,
        trans="T",
        lower=False,
        check_finite=True,
        overwrite_b=False,
    )
    raw_to_whitened = raw_to_whitened_t.T
    whitened_gram = raw_to_whitened.T @ gram @ raw_to_whitened
    np.testing.assert_allclose(
        whitened_gram,
        np.eye(whitened_gram.shape[0]),
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    return PreparedDirectRitzBasis(
        values=np.asarray(basis.values, dtype=np.float64),
        gradient_x_whitened=np.asarray(
            basis.gradient_x @ raw_to_whitened, dtype=np.float64
        ),
        gradient_y_whitened=np.asarray(
            basis.gradient_y @ raw_to_whitened, dtype=np.float64
        ),
        raw_to_whitened=raw_to_whitened,
        structural_basis=independent,
        physical_cholesky_upper=upper,
        raw_basis_size=raw_size,
        structural_rank=independent.shape[1],
        physical_gram_eigenvalues=eigenvalues,
        structural_threshold=threshold,
    )


def solve_whitened_direct_ritz(
    weighted_gradient: np.ndarray,
    whitened_load: np.ndarray,
) -> WhitenedDirectRitzSolve:
    """Compute the Ritz action by direct Householder QR and audit by SVD."""
    c_matrix = np.asarray(weighted_gradient, dtype=np.float64, order="F")
    load = np.asarray(whitened_load, dtype=np.float64).ravel()
    if c_matrix.ndim != 2 or c_matrix.shape[1] != len(load):
        raise ValueError("weighted_gradient and whitened_load dimensions disagree")
    if c_matrix.shape[0] < c_matrix.shape[1]:
        raise ValueError("weighted_gradient must be tall or square")
    if not np.isfinite(c_matrix).all() or not np.isfinite(load).all():
        raise ValueError("direct Ritz inputs must be finite")

    storage, _, _, qr_info = dgeqrf(
        np.array(c_matrix, dtype=np.float64, order="F", copy=True),
        overwrite_a=True,
    )
    if qr_info != 0:
        raise np.linalg.LinAlgError(f"LAPACK dgeqrf failed with info={qr_info}")
    dimension = c_matrix.shape[1]
    triangular = np.triu(storage[:dimension, :dimension])

    failure_reasons: list[str] = []
    try:
        z = linalg.solve_triangular(
            triangular,
            load,
            trans="T",
            lower=False,
            check_finite=True,
        )
        action_qr = float(z @ z)
        whitened_coefficients = linalg.solve_triangular(
            triangular,
            z,
            trans="N",
            lower=False,
            check_finite=True,
        )
        qr_success = bool(
            np.isfinite(action_qr)
            and action_qr >= 0.0
            and np.isfinite(whitened_coefficients).all()
        )
        if not qr_success:
            failure_reasons.append("QR triangular solves returned nonfinite values")
    except linalg.LinAlgError as exc:
        z = np.full(dimension, np.nan)
        action_qr = math.nan
        whitened_coefficients = np.full(dimension, np.nan)
        qr_success = False
        failure_reasons.append(f"QR triangular solve failed: {exc}")

    try:
        _, singular_values, right_vectors_t = linalg.svd(
            triangular,
            full_matrices=False,
            compute_uv=True,
            overwrite_a=False,
            check_finite=True,
            lapack_driver="gesdd",
        )
        generalized_load = right_vectors_t @ load
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            contributions = (
                generalized_load * generalized_load
                / (singular_values * singular_values)
            )
        action_svd = float(np.sum(contributions))
        svd_success = bool(
            np.all(singular_values > 0.0)
            and np.isfinite(action_svd)
            and action_svd >= 0.0
            and np.isfinite(contributions).all()
        )
        if not svd_success:
            failure_reasons.append(
                "SVD did not return a finite full-positive-spectrum action"
            )
    except linalg.LinAlgError as exc:
        singular_values = np.full(dimension, np.nan)
        right_vectors_t = np.full((dimension, dimension), np.nan)
        generalized_load = np.full(dimension, np.nan)
        contributions = np.full(dimension, np.nan)
        action_svd = math.nan
        svd_success = False
        failure_reasons.append(f"SVD failed: {exc}")

    sigma_max = float(singular_values[0]) if svd_success else math.nan
    sigma_min = float(singular_values[-1]) if svd_success else math.nan
    kappa_c = (
        sigma_max / sigma_min
        if svd_success and sigma_min > 0.0 else math.inf
    )
    u_kappa = FLOAT64_UNIT_ROUNDOFF * kappa_c
    return WhitenedDirectRitzSolve(
        action_qr=action_qr,
        action_svd=action_svd,
        qr_svd_relative_discrepancy=(
            _symmetric_relative_difference(action_qr, action_svd)
            if qr_success and svd_success else math.inf
        ),
        qr_triangular=triangular,
        singular_values=singular_values,
        right_singular_vectors_t=right_vectors_t,
        generalized_load_coefficients=generalized_load,
        action_contributions=contributions,
        whitened_coefficients=whitened_coefficients,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        kappa_c=kappa_c,
        u_kappa_c=u_kappa,
        u_kappa_c_squared=FLOAT64_UNIT_ROUNDOFF * kappa_c * kappa_c,
        lapack_full_column_rank=bool(svd_success and np.all(singular_values > 0.0)),
        qr_success=qr_success,
        svd_success=svd_success,
        failure_reason="; ".join(failure_reasons),
    )


def solve_prepared_direct_ritz(
    prepared: PreparedDirectRitzBasis,
    weights: np.ndarray,
    forcing: np.ndarray,
) -> DirectRitzSolve:
    """Solve an ocean trial problem directly from weights and forcing."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    h = np.asarray(forcing, dtype=np.float64).ravel()
    if len(w) != len(prepared.values) or h.shape != w.shape:
        raise ValueError("prepared basis, weights, and forcing disagree")
    if not np.isfinite(w).all() or np.any(w < 0.0) or not np.isfinite(h).all():
        raise ValueError("weights and forcing must be finite; weights nonnegative")
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("weights must have finite positive mass")
    w = w / total
    mean = w @ prepared.values
    raw_load = -(prepared.values - mean).T @ (w * h)
    whitened_load = prepared.raw_to_whitened.T @ raw_load
    sqrt_weight = np.sqrt(w)
    count = len(w)
    dimension = prepared.structural_rank
    direct = np.empty((2 * count, dimension), dtype=np.float64, order="F")
    direct[:count] = prepared.gradient_x_whitened * sqrt_weight[:, None]
    direct[count:] = prepared.gradient_y_whitened * sqrt_weight[:, None]
    result = solve_whitened_direct_ritz(direct, whitened_load)
    raw_coefficients = prepared.raw_to_whitened @ result.whitened_coefficients
    return DirectRitzSolve(
        direct=result,
        raw_load=raw_load,
        whitened_load=whitened_load,
        raw_coefficients=raw_coefficients,
        structural_rank=prepared.structural_rank,
        raw_basis_size=prepared.raw_basis_size,
    )


def solve_raw_direct_ritz(
    weighted_gradient: np.ndarray,
    raw_load: np.ndarray,
    physical_gram: np.ndarray,
    *,
    structural_relative_tolerance: float = DEFAULT_STRUCTURAL_RELATIVE_TOLERANCE,
    structural_basis: np.ndarray | None = None,
) -> DirectRitzSolve:
    """Synthetic/audit entry point for raw B, f, H arrays."""
    b_matrix = np.asarray(weighted_gradient, dtype=np.float64)
    load = np.asarray(raw_load, dtype=np.float64).ravel()
    gram = np.asarray(physical_gram, dtype=np.float64)
    if b_matrix.ndim != 2 or b_matrix.shape[1] != len(load):
        raise ValueError("weighted_gradient and raw_load dimensions disagree")
    placeholder = TrialBasis(
        values=np.zeros((b_matrix.shape[0], b_matrix.shape[1]), dtype=np.float64),
        gradient_x=b_matrix,
        gradient_y=np.zeros_like(b_matrix),
        names=tuple(f"raw_{index}" for index in range(b_matrix.shape[1])),
    )
    prepared = prepare_direct_ritz_basis(
        placeholder,
        gram,
        structural_relative_tolerance=structural_relative_tolerance,
        structural_basis=structural_basis,
    )
    whitened_load = prepared.raw_to_whitened.T @ load
    result = solve_whitened_direct_ritz(
        prepared.gradient_x_whitened, whitened_load
    )
    return DirectRitzSolve(
        direct=result,
        raw_load=load,
        whitened_load=whitened_load,
        raw_coefficients=prepared.raw_to_whitened @ result.whitened_coefficients,
        structural_rank=prepared.structural_rank,
        raw_basis_size=prepared.raw_basis_size,
    )


__all__ = [
    "DirectRitzSolve",
    "FLOAT64_UNIT_ROUNDOFF",
    "PreparedDirectRitzBasis",
    "WhitenedDirectRitzSolve",
    "prepare_direct_ritz_basis",
    "solve_prepared_direct_ritz",
    "solve_raw_direct_ritz",
    "solve_whitened_direct_ritz",
]
