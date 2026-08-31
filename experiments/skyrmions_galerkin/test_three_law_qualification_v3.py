from __future__ import annotations

from . import three_law_qualification_v3 as qualification


CFG = {
    "forcing": {
        "minimum_ess_fraction": 0.05,
        "max_covariance_condition": 1.0e10,
        "projection_tolerance": 2.0e-6,
        "forcing_mean_tolerance": 2.0e-7,
    }
}


def test_centered_development_gate_does_not_use_raw_mean() -> None:
    payload = {
        "minimum_ess_fraction": 0.08,
        "maximum_covariance_condition": 100.0,
        "maximum_projection_residual": 1.0e-10,
        "maximum_forcing_mean": 6.0e-7,
        "maximum_post_centering_forcing_mean": 1.0e-15,
    }
    assert qualification.centered_development_forcing_valid(payload, CFG)


def test_centered_development_gate_keeps_projection_and_support_checks() -> None:
    payload = {
        "minimum_ess_fraction": 0.01,
        "maximum_covariance_condition": 100.0,
        "maximum_projection_residual": 1.0e-10,
        "maximum_forcing_mean": 0.0,
        "maximum_post_centering_forcing_mean": 0.0,
    }
    assert not qualification.centered_development_forcing_valid(payload, CFG)
