"""Explicit stochastic ablation routes and stage semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp
from jax import Array

from .solvers import SolverBackend, batch_project, batch_relax


class AblationMode(str, Enum):
    BASE = "base"
    POST_HOC = "post_hoc"
    RELAX_E2E = "relax_e2e"
    FULL_E2E = "full_e2e"


@dataclass(frozen=True)
class RouteSpec:
    training_stage: str
    serving_stage: str
    train_relaxation: bool
    train_projection: bool


ROUTES = {
    AblationMode.BASE: RouteSpec("initial", "initial", False, False),
    AblationMode.POST_HOC: RouteSpec("initial", "projected", False, False),
    AblationMode.RELAX_E2E: RouteSpec("relaxed", "projected", True, False),
    AblationMode.FULL_E2E: RouteSpec("projected", "projected", True, True),
}


def training_stage(
    mode: AblationMode,
    generated: Array,
    target_moments: Array,
    backend: SolverBackend,
) -> tuple[Array, dict[str, Array]]:
    """Execute only solvers whose gradients belong to the selected route."""
    spec = ROUTES[mode]
    batch_size = generated.shape[0]
    dtype = generated.dtype
    empty = {
        "relaxation_used": jnp.asarray(float(spec.train_relaxation), dtype),
        "projection_used": jnp.asarray(float(spec.train_projection), dtype),
        "relaxation_correction_rms": jnp.zeros((batch_size,), dtype),
        "projection_correction_rms": jnp.zeros((batch_size,), dtype),
    }
    if not spec.train_relaxation:
        return generated, empty
    relaxed, relaxation = batch_relax(backend, generated)
    diagnostics = {
        **empty,
        "relaxation_correction_rms": relaxation["correction_rms"],
    }
    if not spec.train_projection:
        return relaxed, diagnostics
    projected, projection = batch_project(backend, relaxed, target_moments)
    return projected, {
        **diagnostics,
        "projection_correction_rms": projection["correction_rms"],
    }


def evaluate_all_stages(
    generated: Array,
    target_moments: Array,
    backend: SolverBackend,
) -> dict[str, object]:
    """Run both solvers for stagewise diagnostics, regardless of serving route."""
    relaxed, relaxation = batch_relax(backend, generated)
    projected, projection = batch_project(backend, relaxed, target_moments)
    return {
        "initial": generated,
        "relaxed": relaxed,
        "projected": projected,
        "relaxation": relaxation,
        "projection": projection,
    }
