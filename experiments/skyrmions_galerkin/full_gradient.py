"""Pure-JAX sensor-to-Ritz graph and fixed-theta envelope derivative.

This module deliberately contains no optimizer or file I/O.  A differentiated
call sees only ``eta``; reference/truth banks and Ritz parameters are closed-over
constants.  Hard scientific checks live in :mod:`workflow`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from mfsi.moments import AnchoredCubicSplineReconstructor
from mfsi.projection import EmpiricalIProjector, IProjectionConfig

from .deep_ritz import RitzParams, potential_values_and_gradients, ritz_objective
from .domain import minimum_image
from .forcing import ForcingConfig, ForcingTrajectory, continuity_forcing
from .measurements import LocalDensitySensors
from .risk import integrated_risk

Array = jax.Array


class ReferenceBank(NamedTuple):
    configurations: Array
    velocity: Array
    base_weights: Array


class Reconstruction(NamedTuple):
    values: Array
    derivatives: Array
    residual_sum_squares: Array
    roughness: Array


class EnvelopeDiagnostics(NamedTuple):
    ritz_objective: Array
    full_energy: Array
    energy_identity_relerr: Array
    maximum_projection_residual: Array
    minimum_ess_fraction: Array
    maximum_forcing_mean: Array
    maximum_covariance_condition: Array


@dataclass(frozen=True)
class FrozenEtaProblem:
    """Static inputs for one deterministic eta objective.

    ``reconstructor`` contains only fixed spline basis matrices.  It is closed
    over by JAX traces; its data-dependent coefficient solve remains in JAX.
    """

    truth_configurations: Array
    times: Array
    time_weights: Array
    acquisition_indices: Array
    finite_configuration_count: int
    detector_noise: Array
    family: LocalDensitySensors
    reconstructor: AnchoredCubicSplineReconstructor
    projection_config: IProjectionConfig
    forcing_config: ForcingConfig
    projection_backend: str = "jax"
    box: tuple[float, float] = (2.0, 1.0)


def wrap_periodic(eta: Array, family: LocalDensitySensors) -> Array:
    centers = family.centers(jnp.asarray(eta, dtype=jnp.float64))
    return jnp.mod(centers, jnp.asarray(family.box, dtype=centers.dtype)).reshape(-1)


def minimum_sensor_separation(eta: Array, family: LocalDensitySensors) -> Array:
    centers = family.centers(wrap_periodic(eta, family))
    box = jnp.asarray(family.box, dtype=centers.dtype)
    displacement = minimum_image(centers[:, None, :] - centers[None, :, :], box)
    distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1) + 1.0e-24)
    masked = jnp.where(jnp.eye(family.n_sensors, dtype=bool), jnp.inf, distance)
    return jnp.min(masked)


def smooth_separation_penalty(eta: Array, family: LocalDensitySensors) -> Array:
    violation = jax.nn.relu(float(family.min_separation) - minimum_sensor_separation(eta, family))
    scale = jnp.maximum(float(family.min_separation), 1.0e-12)
    return (violation / scale) ** 2


def periodic_branch_distance(eta: Array, family: LocalDensitySensors) -> Array:
    """Distance to a wrap or minimum-image branch boundary, for logging only."""

    centers = family.centers(wrap_periodic(eta, family))
    box = jnp.asarray(family.box, dtype=centers.dtype)
    wrap_distance = jnp.min(jnp.minimum(centers, box - centers))
    displacement = centers[:, None, :] - centers[None, :, :]
    half_box_distance = jnp.abs(jnp.abs(displacement) - 0.5 * box)
    pair_mask = ~jnp.eye(family.n_sensors, dtype=bool)
    image_distance = jnp.min(jnp.where(pair_mask[..., None], half_box_distance, jnp.inf))
    return jnp.minimum(wrap_distance, image_distance)


def reconstruct_moments(eta: Array, problem: FrozenEtaProblem) -> Reconstruction:
    """Fixed-CRN observations followed by a differentiable anchored spline fit."""

    eta = wrap_periodic(eta, problem.family)
    features = problem.family.features(problem.truth_configurations, eta)
    acquired = features[problem.acquisition_indices]
    finite = acquired[:, : int(problem.finite_configuration_count), :]
    observations = jnp.mean(finite, axis=1) + problem.detector_noise
    endpoint0 = jnp.mean(features[0], axis=0)
    endpoint1 = jnp.mean(features[-1], axis=0)
    fit = problem.reconstructor.reconstruct(observations, endpoint0, endpoint1)
    return Reconstruction(
        values=fit.c,
        derivatives=fit.c_dot,
        residual_sum_squares=fit.residual_sum_squares,
        roughness=fit.roughness,
    )


def forcing_state(
    eta: Array,
    problem: FrozenEtaProblem,
    reference: ReferenceBank,
    reconstruction: Reconstruction | None = None,
) -> ForcingTrajectory:
    """Evaluate the smooth information-projection and continuity-forcing graph."""

    eta = wrap_periodic(eta, problem.family)
    if reconstruction is None:
        reconstruction = reconstruct_moments(eta, problem)
    return continuity_forcing(
        reference.configurations,
        reference.velocity,
        reference.base_weights,
        reconstruction.values,
        reconstruction.derivatives,
        eta,
        problem.family,
        projection_cfg=problem.projection_config,
        cfg=problem.forcing_config,
        fail_loudly=False,
        projection_backend=problem.projection_backend,
    )


def projected_weights(
    eta: Array,
    problem: FrozenEtaProblem,
    reference: ReferenceBank,
    reconstruction: Reconstruction | None = None,
) -> Array:
    """Cheaper law-risk path: calibration without the forcing solve."""

    eta = wrap_periodic(eta, problem.family)
    if reconstruction is None:
        reconstruction = reconstruct_moments(eta, problem)
    features = problem.family.features(reference.configurations, eta)
    projector = EmpiricalIProjector(
        problem.projection_config,
        trajectory_backend=problem.projection_backend,
    )
    trajectory = projector.project_trajectory(
        features,
        reference.base_weights,
        reconstruction.values[None, ...],
    )
    return trajectory.weights[0]


def projected_law_risk(
    eta: Array,
    problem: FrozenEtaProblem,
    reference: ReferenceBank,
    reference_features: Array,
    truth_feature_means: Array,
    whitening: Array,
) -> Array:
    reconstruction = reconstruct_moments(eta, problem)
    weights = projected_weights(eta, problem, reference, reconstruction)
    return integrated_risk(
        weights,
        reference_features,
        truth_feature_means,
        whitening,
        problem.time_weights,
    )


def ritz_objective_eta(
    theta: RitzParams,
    eta: Array,
    problem: FrozenEtaProblem,
    ritz_bank: ReferenceBank,
) -> Array:
    """The original centered Ritz functional with all eta inputs recomputed."""

    reconstruction = reconstruct_moments(eta, problem)
    state = forcing_state(eta, problem, ritz_bank, reconstruction)
    return ritz_objective(
        theta,
        ritz_bank.configurations,
        state.projection.weights,
        state.forcing,
        problem.times,
        problem.time_weights,
        box=problem.box,
    )


def full_energy_from_state(
    theta: RitzParams,
    problem: FrozenEtaProblem,
    ritz_bank: ReferenceBank,
    state: ForcingTrajectory,
) -> Array:
    _, gradients = potential_values_and_gradients(
        theta,
        ritz_bank.configurations,
        problem.times,
        box=problem.box,
    )
    kinetic = jnp.sum(gradients * gradients, axis=(-2, -1))
    rows = jnp.einsum("tn,tn->t", state.projection.weights, kinetic)
    return jnp.sum(problem.time_weights * rows)


def full_energy(
    theta: RitzParams,
    eta: Array,
    problem: FrozenEtaProblem,
    ritz_bank: ReferenceBank,
) -> Array:
    reconstruction = reconstruct_moments(eta, problem)
    state = forcing_state(eta, problem, ritz_bank, reconstruction)
    return full_energy_from_state(theta, problem, ritz_bank, state)


def envelope_diagnostics(
    theta: RitzParams,
    eta: Array,
    problem: FrozenEtaProblem,
    ritz_bank: ReferenceBank,
) -> EnvelopeDiagnostics:
    reconstruction = reconstruct_moments(eta, problem)
    state = forcing_state(eta, problem, ritz_bank, reconstruction)
    objective = ritz_objective(
        theta,
        ritz_bank.configurations,
        state.projection.weights,
        state.forcing,
        problem.times,
        problem.time_weights,
        box=problem.box,
    )
    energy = full_energy_from_state(theta, problem, ritz_bank, state)
    return EnvelopeDiagnostics(
        ritz_objective=objective,
        full_energy=energy,
        energy_identity_relerr=jnp.abs(energy + 2.0 * objective)
        / jnp.maximum(jnp.abs(energy), 1.0e-12),
        maximum_projection_residual=jnp.max(
            jnp.linalg.norm(state.projection.residual, axis=-1)
        ),
        minimum_ess_fraction=jnp.min(state.projection.ess_fraction),
        maximum_forcing_mean=jnp.max(
            jnp.abs(state.forcing_mean_before_centering)
        ),
        maximum_covariance_condition=jnp.max(state.covariance_condition),
    )


def envelope_full_value_and_grad(
    eta: Array,
    theta_fixed: RitzParams,
    problem: FrozenEtaProblem,
    ritz_bank: ReferenceBank,
) -> tuple[Array, Array, EnvelopeDiagnostics]:
    """Return ``-2 J`` and ``-2 partial_eta J`` with ``theta_fixed`` closed over.

    This function never trains ``theta_fixed`` and never differentiates through
    the process that produced it.  The closure makes the envelope semantics
    explicit without inserting stop-gradients into eta-dependent quantities.
    """

    objective = lambda design: -2.0 * ritz_objective_eta(
        theta_fixed, design, problem, ritz_bank
    )
    value, gradient = jax.value_and_grad(objective)(jnp.asarray(eta, dtype=jnp.float64))
    diagnostics = envelope_diagnostics(theta_fixed, eta, problem, ritz_bank)
    return value, gradient, diagnostics
