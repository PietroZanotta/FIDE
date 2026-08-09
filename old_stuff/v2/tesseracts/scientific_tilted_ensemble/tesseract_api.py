"""Packaged API for the tilted-ensemble scientific component."""

from __future__ import annotations

from manybody_completion.homometric import build_population_support
from manybody_completion.network import PriorParameters
from manybody_completion.solvers import tilted_ensemble
from manybody_completion.tesseract_backend import serialize_tilted


def apply(payload: dict) -> dict:
    support = build_population_support(int(payload.get("n_spins", 8)))
    params = PriorParameters.from_mapping(payload["prior_parameters"])
    options = dict(payload["sampler_options"])
    result = tilted_ensemble(
        params,
        support,
        float(payload["dual"]),
        seed=int(payload.get("seed", 0)),
        **options,
    )
    return serialize_tilted(result)
