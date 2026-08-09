"""Packaged API for the tilted-ensemble scientific component."""

from __future__ import annotations

import numpy as np

from manybody_completion.adaptive_components import ProposalModel
from manybody_completion.homometric import build_population_support
from manybody_completion.network import PriorParameters
from manybody_completion.solvers import tilted_ensemble, tilted_ensemble_from_probabilities
from manybody_completion.tesseract_backend import serialize_tilted


def apply(payload: dict) -> dict:
    support = build_population_support(int(payload.get("n_spins", 8)))
    options = dict(payload["sampler_options"])
    if "prior_probabilities" in payload:
        proposal_model = (
            ProposalModel.from_mapping(payload["proposal_model"])
            if "proposal_model" in payload
            else None
        )
        result = tilted_ensemble_from_probabilities(
            np.asarray(payload["prior_probabilities"], dtype=np.float64),
            support,
            float(payload["dual"]),
            seed=int(payload.get("seed", 0)),
            proposal_model=proposal_model,
            **options,
        )
    else:
        params = PriorParameters.from_mapping(payload["prior_parameters"])
        result = tilted_ensemble(
            params,
            support,
            float(payload["dual"]),
            seed=int(payload.get("seed", 0)),
            **options,
        )
    return serialize_tilted(result)
