"""Tesseract implementation of the scientific homometric solver contract."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jax import Array

from .energy import PhysicalParameters
from .observables import PairBasis
from .solvers import ProjectionOptions, RelaxationOptions


def _make_client(
    configuration: dict[str, Any],
    name: str,
    repository_root: Path,
):
    try:
        from tesseract_core import Tesseract
    except ImportError as error:
        raise RuntimeError(
            "The Tesseract backend requires tesseract-core and tesseract-jax. "
            "Install requirements/host-cpu.txt or requirements/host-cuda.txt."
        ) from error

    transport = str(configuration.get("transport", "url")).lower()
    if transport == "url":
        environment_name = f"MBC_{name.upper()}_TESSERACT_URL"
        url = os.environ.get(environment_name) or configuration.get(f"{name}_url")
        if not url:
            raise ValueError(
                f"solver_backend.{name}_url (or {environment_name}) is required "
                "for Tesseract URL transport"
            )
        return Tesseract.from_url(str(url))
    if transport == "local_api":
        default = f"tesseracts/scientific_{name}/tesseract_api.py"
        path = repository_root / configuration.get(f"{name}_api", default)
        return Tesseract.from_tesseract_api(path)
    raise ValueError("solver_backend.transport must be 'url' or 'local_api'")


@dataclass(frozen=True)
class TesseractBackend:
    """Host-side JAX bridge to equation-identical scientific Tesseracts."""

    box: Array
    basis: PairBasis
    moment_scales: Array
    physical: PhysicalParameters
    relaxation_options: RelaxationOptions
    projection_options: ProjectionOptions
    relaxation_client: Any
    projection_client: Any

    @classmethod
    def from_configuration(
        cls,
        configuration: dict[str, Any],
        *,
        repository_root: Path,
        **common: Any,
    ) -> TesseractBackend:
        return cls(
            **common,
            relaxation_client=_make_client(
                configuration, "relaxation", repository_root
            ),
            projection_client=_make_client(
                configuration, "projection", repository_root
            ),
        )

    def relax(self, coordinates: Array) -> tuple[Array, dict[str, Array]]:
        from tesseract_jax import apply_tesseract

        options = self.relaxation_options
        result = apply_tesseract(
            self.relaxation_client,
            {
                "coordinates": coordinates,
                "box": self.box,
                "r0": self.physical.r0,
                "kappa": self.physical.kappa,
                "num_steps": options.num_steps,
                "step_size": options.step_size,
                "prox_strength": options.prox_strength,
                "max_particle_step": options.max_particle_step,
                "tolerance": options.tolerance,
            },
        )
        relaxed = result.pop("relaxed_coordinates")
        return relaxed, result

    def project(
        self, coordinates: Array, target_moments: Array
    ) -> tuple[Array, dict[str, Array]]:
        from tesseract_jax import apply_tesseract

        options = self.projection_options
        result = apply_tesseract(
            self.projection_client,
            {
                "coordinates": coordinates,
                "target_moments": target_moments,
                "box": self.box,
                "basis_centers": self.basis.centers,
                "basis_widths": self.basis.widths,
                "moment_scales": self.moment_scales,
                "num_steps": options.num_steps,
                "ridge": options.ridge,
                "max_particle_step": options.max_particle_step,
                "tolerance": options.tolerance,
                "rank_tolerance": options.rank_tolerance,
                "line_search_steps": options.line_search_steps,
                "line_search_shrink": options.line_search_shrink,
                "sufficient_decrease": options.sufficient_decrease,
            },
        )
        projected = result.pop("projected_coordinates")
        return projected, result
