"""Reusable composition of the generator, relaxation, and projection stages.

The functions in this module remain framework-native JAX code.  Solver calls
are kept at coarse granularity so the same scalar objective can later replace
the local solver functions with Tesseract-backed callables.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import Array

from .geometry import wrap_positions
from .observables import PairBasis, ensemble_pair_moments
from .projection import ProjectionOptions, project_ensemble_moments
from .relaxation import (
    RelaxationOptions,
    relax_proximal,
    smooth_periodic_displacement,
)


@dataclass(frozen=True)
class PhysicalParameters:
    """Parameters of the smooth repulsive relaxation energy."""

    r0: float = 0.08
    kappa: float = 30.0
    prox_strength: float = 0.05


@dataclass(frozen=True)
class CompletionOptions:
    """Numerical options for the two local solver stages."""

    physical: PhysicalParameters = field(default_factory=PhysicalParameters)
    relaxation: RelaxationOptions = field(default_factory=RelaxationOptions)
    projection: ProjectionOptions = field(default_factory=ProjectionOptions)


def scalar_generator(
    parameter: Array | float,
    base_coordinates: Array,
    latent_displacements: Array,
    box: Array,
) -> Array:
    """The S3 generator ``G_a(Z,c) = wrap(X_base + a Z)``."""
    base_coordinates = jnp.asarray(base_coordinates)
    latent_displacements = jnp.asarray(latent_displacements, dtype=base_coordinates.dtype)
    box = jnp.asarray(box, dtype=base_coordinates.dtype)
    parameter = jnp.asarray(parameter, dtype=base_coordinates.dtype)
    if base_coordinates.shape != latent_displacements.shape:
        raise ValueError("base_coordinates and latent_displacements must have identical shapes")
    if base_coordinates.ndim != 3 or base_coordinates.shape[-1] != 2:
        raise ValueError(
            f"base_coordinates must have shape (M, N, 2); got {base_coordinates.shape}"
        )
    return wrap_positions(base_coordinates + parameter * latent_displacements, box)


def periodic_correction_mse(coordinates: Array, reference: Array, box: Array) -> Array:
    """Mean squared smooth periodic displacement per particle."""
    displacement = smooth_periodic_displacement(coordinates, reference, box)
    return jnp.mean(jnp.sum(displacement * displacement, axis=-1))


def run_local_completion(
    initial_coordinates: Array,
    target_moments: Array,
    box: Array,
    basis: PairBasis,
    moment_scales: Array | None = None,
    basis_mask: Array | None = None,
    options: CompletionOptions | None = None,
) -> dict[str, object]:
    """Run one complete local relaxation-plus-projection solve.

    The returned dictionary uses the same stage names as the planned end-to-end
    model: ``initial_coordinates``, ``relaxed_coordinates``, and
    ``projected_coordinates``.  Nested solver diagnostics are included for
    reporting but are not required to define an outer loss.
    """
    if options is None:
        options = CompletionOptions()

    initial_coordinates = jnp.asarray(initial_coordinates)
    dtype = initial_coordinates.dtype
    target_moments = jnp.asarray(target_moments, dtype=dtype)
    box = jnp.asarray(box, dtype=dtype)
    if moment_scales is None:
        moment_scales = jnp.ones_like(target_moments)
    else:
        moment_scales = jnp.asarray(moment_scales, dtype=dtype)
    if basis_mask is None:
        basis_mask = jnp.ones_like(target_moments)
    else:
        basis_mask = jnp.asarray(basis_mask, dtype=dtype)

    initial_coordinates = wrap_positions(initial_coordinates, box)
    moments_initial = ensemble_pair_moments(initial_coordinates, box, basis)
    relaxed_coordinates, relaxation_diagnostics = relax_proximal(
        initial_coordinates=initial_coordinates,
        box=box,
        r0=options.physical.r0,
        kappa=options.physical.kappa,
        prox_strength=options.physical.prox_strength,
        options=options.relaxation,
    )
    moments_relaxed = ensemble_pair_moments(relaxed_coordinates, box, basis)
    projected_coordinates, projection_diagnostics = project_ensemble_moments(
        coordinates=relaxed_coordinates,
        target_moments=target_moments,
        box=box,
        basis=basis,
        moment_scales=moment_scales,
        basis_mask=basis_mask,
        options=options.projection,
    )
    moments_projected = ensemble_pair_moments(projected_coordinates, box, basis)

    return {
        "initial_coordinates": initial_coordinates,
        "relaxed_coordinates": relaxed_coordinates,
        "projected_coordinates": projected_coordinates,
        "moments_initial": moments_initial,
        "moments_relaxed": moments_relaxed,
        "moments_projected": moments_projected,
        "relaxation": relaxation_diagnostics,
        "projection": projection_diagnostics,
    }


def stop_gradient_diagnostics(tree: object) -> object:
    """Detach an arbitrary diagnostics pytree while preserving its structure."""
    return jax.tree_util.tree_map(jax.lax.stop_gradient, tree)
