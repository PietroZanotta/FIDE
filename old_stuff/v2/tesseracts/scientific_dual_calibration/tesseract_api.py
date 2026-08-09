"""Packaged API for covariance-Newton dual calibration."""

from __future__ import annotations

from manybody_completion.homometric import build_population_support
from manybody_completion.network import PriorParameters
from manybody_completion.solvers import calibrate_dual
from manybody_completion.tesseract_backend import serialize_calibration


def apply(payload: dict) -> dict:
    support = build_population_support(int(payload.get("n_spins", 8)))
    params = PriorParameters.from_mapping(payload["prior_parameters"])
    result = calibrate_dual(
        params,
        support,
        float(payload["target_moment"]),
        sampler_options=dict(payload["sampler_options"]),
        calibration_options=dict(payload["calibration_options"]),
        seed=int(payload.get("seed", 0)),
    )
    return serialize_calibration(result)
