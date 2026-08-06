"""Configuration-selectable solver backends for the homometric experiment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from jax import Array

from .energy import PhysicalParameters
from .observables import PairBasis
from .solvers import (
    LocalJaxBackend,
    ProjectionOptions,
    RelaxationOptions,
    SolverBackend,
)


def _backend_configuration(config: dict[str, Any]) -> dict[str, Any]:
    selected = dict(config.get("solver_backend", {"kind": "local_jax"}))
    environment_kind = os.environ.get("MBC_SOLVER_BACKEND")
    if environment_kind:
        selected["kind"] = environment_kind
    environment_transport = os.environ.get("MBC_TESSERACT_TRANSPORT")
    if environment_transport:
        selected["transport"] = environment_transport
    return selected


def build_solver_backend(
    config: dict[str, Any],
    *,
    repository_root: Path,
    box: Array,
    basis: PairBasis,
    moment_scales: Array,
    physical: PhysicalParameters,
    projection_overrides: dict[str, Any] | None = None,
) -> SolverBackend:
    """Build the selected backend without importing Tesseract for local runs."""
    relaxation_options = RelaxationOptions(**config["relaxation"])
    projection_config = {**config["projection"], **(projection_overrides or {})}
    projection_options = ProjectionOptions(**projection_config)
    backend_config = _backend_configuration(config)
    kind = str(backend_config.get("kind", "local_jax")).lower()
    common = {
        "box": box,
        "basis": basis,
        "moment_scales": moment_scales,
        "physical": physical,
        "relaxation_options": relaxation_options,
        "projection_options": projection_options,
    }
    if kind == "local_jax":
        return LocalJaxBackend(**common)
    if kind != "tesseract":
        raise ValueError(
            f"unknown solver_backend.kind {kind!r}; expected 'local_jax' or 'tesseract'"
        )

    from .tesseract_backend import TesseractBackend

    return TesseractBackend.from_configuration(
        backend_config,
        repository_root=repository_root,
        **common,
    )
