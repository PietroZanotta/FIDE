"""Packaged API for covariance-Newton dual calibration."""

from __future__ import annotations

import numpy as np

from manybody_completion.adaptive_components import ProposalModel, WarmStartModel
from manybody_completion.homometric import build_population_support
from manybody_completion.network import PriorParameters
from manybody_completion.solvers import calibrate_dual, calibrate_dual_from_probabilities
from manybody_completion.tesseract_backend import serialize_calibration


def apply(payload: dict) -> dict:
    support = build_population_support(int(payload.get("n_spins", 8)))
    common = {
        "support": support,
        "target_moment": float(payload["target_moment"]),
        "sampler_options": dict(payload["sampler_options"]),
        "calibration_options": dict(payload["calibration_options"]),
        "seed": int(payload.get("seed", 0)),
    }
    if "prior_probabilities" in payload:
        proposal_model = (
            ProposalModel.from_mapping(payload["proposal_model"])
            if "proposal_model" in payload
            else None
        )
        warm_start_model = (
            WarmStartModel.from_mapping(payload["warm_start_model"])
            if "warm_start_model" in payload
            else None
        )
        result = calibrate_dual_from_probabilities(
            np.asarray(payload["prior_probabilities"], dtype=np.float64),
            proposal_model=proposal_model,
            warm_start_model=warm_start_model,
            **common,
        )
    else:
        result = calibrate_dual(
            PriorParameters.from_mapping(payload["prior_parameters"]),
            **common,
        )
    return serialize_calibration(result)
