import numpy as np

from experiments.ocean_drifters.concentration_conditioning import (
    FLOAT64_UNIT_ROUNDOFF,
    concentration_statistics,
    direct_weighted_gradient_diagnostic,
)
from experiments.ocean_drifters.full_action_repair import TrialBasis


def test_concentration_statistics_use_exact_float64_projected_weights() -> None:
    points = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    log_weights = np.log(np.array([0.5, 0.25, 0.25]))
    weights, result = concentration_statistics(points, log_weights)
    expected_mean = weights @ points
    centered = points - expected_mean
    expected_covariance = centered.T @ (weights[:, None] * centered)
    expected_eigenvalues = np.linalg.eigvalsh(expected_covariance)
    assert np.isclose(weights.sum(), 1.0)
    assert result["zero_weight_count"] == 0
    assert np.isclose(result["projected_weight_ess"], 1.0 / np.sum(weights**2))
    assert np.isclose(result["cov_eig_minor"], expected_eigenvalues[0])
    assert np.isclose(result["cov_eig_major"], expected_eigenvalues[1])


def test_direct_qr_svd_matches_explicit_weighted_gradient_svd() -> None:
    rng = np.random.default_rng(19)
    sample_count = 80
    raw_size = 4
    points = rng.normal(size=(sample_count, 2))
    values = rng.normal(size=(sample_count, raw_size))
    gradient_x = rng.normal(size=(sample_count, raw_size))
    gradient_y = rng.normal(size=(sample_count, raw_size))
    basis = TrialBasis(
        values=values,
        gradient_x=gradient_x,
        gradient_y=gradient_y,
        names=tuple(f"phi_{index}" for index in range(raw_size)),
    )
    weights = rng.uniform(0.1, 1.0, size=sample_count)
    weights /= weights.sum()
    forcing = rng.normal(size=sample_count)
    transform = np.diag([0.5, 0.8, 1.1, 1.7])
    result = direct_weighted_gradient_diagnostic(
        basis, weights, forcing, transform
    )
    explicit = np.vstack((
        np.sqrt(weights)[:, None] * gradient_x @ transform,
        np.sqrt(weights)[:, None] * gradient_y @ transform,
    ))
    expected = np.linalg.svd(explicit, compute_uv=False)
    assert np.allclose(result.singular_values, expected, rtol=2e-14)
    assert np.isclose(
        result.scalars["normal_roundoff_amplification"],
        FLOAT64_UNIT_ROUNDOFF
        * float(result.scalars["kappa_C"]) ** 2,
    )
    assert np.allclose(
        result.normal_eigenvalues,
        result.squared_singular_values[::-1],
        rtol=2e-13,
        atol=2e-15,
    )
    assert result.scalars["positive_mode_truncation_used"] is False
