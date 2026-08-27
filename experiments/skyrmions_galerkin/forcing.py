from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from mfsi.projection import EmpiricalIProjector, IProjectionConfig, IProjectionTrajectoryState

from .measurements import LocalDensitySensors

Array = jax.Array


class ForcingTrajectory(NamedTuple):
    projection: IProjectionTrajectoryState
    features: Array
    advective_rates: Array
    lambda_dot: Array
    forcing: Array
    forcing_mean_before_centering: Array
    covariance_condition: Array


@dataclass(frozen=True)
class ForcingConfig:
    projection_tolerance: float = 2.0e-7
    minimum_ess_fraction: float = 0.03
    covariance_ridge: float = 1.0e-8
    max_covariance_condition: float = 1.0e10
    forcing_mean_tolerance: float = 2.0e-8


def strict_project_trajectory(
    features: Array,
    base_weights: Array,
    targets: Array,
    *,
    projection_cfg: IProjectionConfig,
    tolerance: float,
    trajectory_backend: str = "jax",
) -> IProjectionTrajectoryState:
    """Hard empirical projection with explicit failure instead of a soft fallback."""

    projector = EmpiricalIProjector(projection_cfg, trajectory_backend=trajectory_backend)
    state = projector.project_trajectory(
        jnp.asarray(features), jnp.asarray(base_weights), jnp.asarray(targets)[None, ...]
    )
    state = jax.tree_util.tree_map(lambda value: value[0], state)
    residual = np.asarray(state.residual)
    if not np.all(np.isfinite(residual)) or float(np.max(np.linalg.norm(residual, axis=-1))) > float(tolerance):
        raise RuntimeError(
            "empirical I-projection failed hard calibration: "
            f"max residual={float(np.max(np.linalg.norm(residual, axis=-1))):.6e}, "
            f"tolerance={float(tolerance):.6e}; target may be outside empirical support"
        )
    return state


def continuity_forcing(
    configurations: Array,
    reference_velocity: Array,
    base_weights: Array,
    targets: Array,
    target_derivatives: Array,
    eta: Array,
    family: LocalDensitySensors,
    *,
    projection_cfg: IProjectionConfig = IProjectionConfig(),
    cfg: ForcingConfig = ForcingConfig(),
    fail_loudly: bool = True,
    projection_backend: str = "jax",
) -> ForcingTrajectory:
    """Density-free FIDE forcing in full many-body configuration space."""

    configurations = jnp.asarray(configurations, dtype=jnp.float64)
    velocity = jnp.asarray(reference_velocity, dtype=jnp.float64)
    features = family.features(configurations, eta)
    advective = family.jvp(configurations, velocity, eta)
    projector = EmpiricalIProjector(projection_cfg, trajectory_backend=projection_backend)
    projected = projector.project_trajectory(features, base_weights, targets[None, ...])
    projection = jax.tree_util.tree_map(lambda value: value[0], projected)

    weights = projection.weights
    lam = projection.lam
    moment_m = jnp.einsum("tn,tnr->tr", weights, advective)
    scalar_m = jnp.einsum("tnr,tr->tn", advective, lam)
    centered_phi = features - projection.moments[:, None, :]
    centered_g = scalar_m - jnp.einsum("tn,tn->t", weights, scalar_m)[:, None]
    covariance_phi_g = jnp.einsum("tn,tnr,tn->tr", weights, centered_phi, centered_g)
    rhs = target_derivatives - moment_m - covariance_phi_g
    eye = jnp.eye(features.shape[-1], dtype=features.dtype)
    regularized = projection.covariance + float(cfg.covariance_ridge) * eye
    lambda_dot = jax.vmap(jnp.linalg.solve)(regularized, rhs)
    forcing = (
        jnp.einsum("tr,tnr->tn", lambda_dot, features - targets[:, None, :])
        + jnp.einsum("tr,tnr->tn", lam, advective - moment_m[:, None, :])
    )
    mean_before = jnp.einsum("tn,tn->t", weights, forcing)
    # Only a floating-point gauge offset is removed.  Material offsets fail below.
    forcing = forcing - mean_before[:, None]
    eigenvalues = jnp.linalg.eigvalsh(regularized)
    condition = eigenvalues[:, -1] / jnp.maximum(eigenvalues[:, 0], 1.0e-300)

    if fail_loudly:
        residual = np.linalg.norm(np.asarray(projection.residual), axis=-1)
        min_ess = float(np.min(np.asarray(projection.ess_fraction)))
        max_mean = float(np.max(np.abs(np.asarray(mean_before))))
        max_condition = float(np.max(np.asarray(condition)))
        reasons = []
        if not np.all(np.isfinite(residual)) or float(np.max(residual)) > cfg.projection_tolerance:
            reasons.append("projection_calibration")
        if not np.isfinite(min_ess) or min_ess < cfg.minimum_ess_fraction:
            reasons.append("projection_support_ess")
        if not np.isfinite(max_mean) or max_mean > cfg.forcing_mean_tolerance:
            reasons.append("forcing_mean_compatibility")
        if not np.isfinite(max_condition) or max_condition > cfg.max_covariance_condition:
            reasons.append("observable_covariance_condition")
        if reasons:
            raise RuntimeError(
                "invalid continuity forcing: " + ", ".join(reasons)
                + f" (max_projection_residual={float(np.max(residual)):.6e},"
                + f" min_ess_fraction={min_ess:.6e},"
                + f" max_forcing_mean={max_mean:.6e},"
                + f" max_covariance_condition={max_condition:.6e})"
            )

    return ForcingTrajectory(
        projection=projection,
        features=features,
        advective_rates=advective,
        lambda_dot=lambda_dot,
        forcing=forcing,
        forcing_mean_before_centering=mean_before,
        covariance_condition=condition,
    )
