"""Closed-form Tangent action for the many-body local-density sensors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .domain import minimum_image
from .measurements import LocalDensitySensors

Array = jax.Array


@dataclass(frozen=True)
class TangentCertificateConfig:
    maximum_gram_condition: float = 1.0e8
    maximum_moment_rate_residual: float = 1.0e-10
    minimum_ess_fraction: float = 0.05
    maximum_projection_residual: float = 2.0e-6


def local_density_gradients(
    configurations: Array,
    eta: Array,
    family: LocalDensitySensors,
) -> Array:
    """Return ``grad_X Phi`` with shape ``[..., particle, sensor, xy]``."""

    x = jnp.asarray(configurations, dtype=jnp.float64)
    centers = family.centers(eta)
    delta = minimum_image(
        x[..., :, None, :] - centers,
        jnp.asarray(family.box, dtype=x.dtype),
    )
    kernel = jnp.exp(-0.5 * jnp.sum(delta * delta, axis=-1) / float(family.width) ** 2)
    return (
        -kernel[..., None]
        * delta
        / (float(x.shape[-2]) * float(family.width) ** 2)
    )


def audit_tangent_action(
    configurations: Array,
    reference_velocity: Array,
    projected_weights: Array,
    target_derivatives: Array,
    eta: Array,
    family: LocalDensitySensors,
    time_weights: Array,
    *,
    projection_residual: Array | None = None,
    ess_fraction: Array | None = None,
    cfg: TangentCertificateConfig = TangentCertificateConfig(),
) -> dict[str, Any]:
    """Evaluate the exact minimum-norm moment correction on a frozen bank.

    The correction at each time is constrained to the span of the selected
    observable gradients.  No Deep Ritz fit or iterative field optimization is
    involved; only one small Gram solve per time node is required.
    """

    x = jnp.asarray(configurations, dtype=jnp.float64)
    velocity = jnp.asarray(reference_velocity, dtype=jnp.float64)
    weights = jnp.asarray(projected_weights, dtype=jnp.float64)
    target_dot = jnp.asarray(target_derivatives, dtype=jnp.float64)
    quadrature = jnp.asarray(time_weights, dtype=jnp.float64)
    gradients = local_density_gradients(x, eta, family)
    advective = family.jvp(x, velocity, eta)

    gram = jnp.einsum("tn,tnpjd,tnpkd->tjk", weights, gradients, gradients)
    advective_rate = jnp.einsum("tn,tnr->tr", weights, advective)
    required_rate = target_dot - advective_rate
    coefficients = jax.vmap(jnp.linalg.solve)(gram, required_rate)
    tangent_velocity = jnp.einsum("tr,tnprd->tnpd", coefficients, gradients)
    energy_rows = jnp.sum(tangent_velocity * tangent_velocity, axis=(-2, -1))
    kinetic = jnp.einsum("tn,tn->t", weights, energy_rows)
    kinetic_second = jnp.einsum("tn,tn->t", weights, energy_rows * energy_rows)
    action = jnp.sum(quadrature * kinetic)

    corrected_rate = advective_rate + jnp.einsum("tjk,tk->tj", gram, coefficients)
    raw_moment_residual = corrected_rate - target_dot
    normalized_moment_residual = jnp.linalg.norm(raw_moment_residual, axis=-1) / jnp.maximum(
        1.0, jnp.linalg.norm(required_rate, axis=-1)
    )
    eigenvalues = jnp.linalg.eigvalsh(gram)
    gram_condition = eigenvalues[:, -1] / jnp.maximum(eigenvalues[:, 0], 1.0e-300)

    kinetic_variance = jnp.maximum(kinetic_second - kinetic * kinetic, 0.0)
    effective_samples = 1.0 / jnp.maximum(jnp.sum(weights * weights, axis=-1), 1.0e-300)
    action_standard_error = jnp.sqrt(jnp.sum(
        (
            quadrature
            * jnp.sqrt(kinetic_variance / jnp.maximum(effective_samples, 1.0))
        ) ** 2
    ))

    maximum_projection_residual = (
        float(jnp.max(jnp.linalg.norm(jnp.asarray(projection_residual), axis=-1)))
        if projection_residual is not None
        else 0.0
    )
    minimum_ess_fraction = (
        float(jnp.min(jnp.asarray(ess_fraction)))
        if ess_fraction is not None
        else float(jnp.min(effective_samples) / x.shape[1])
    )
    maximum_gram_condition = float(jnp.max(gram_condition))
    maximum_moment_rate_residual = float(jnp.max(normalized_moment_residual))
    action_value = float(action)
    finite = all(np.isfinite(value) for value in (
        action_value,
        float(action_standard_error),
        maximum_projection_residual,
        minimum_ess_fraction,
        maximum_gram_condition,
        maximum_moment_rate_residual,
    ))
    valid = bool(
        finite
        and maximum_projection_residual <= cfg.maximum_projection_residual
        and minimum_ess_fraction >= cfg.minimum_ess_fraction
        and maximum_gram_condition <= cfg.maximum_gram_condition
        and maximum_moment_rate_residual <= cfg.maximum_moment_rate_residual
        and float(jnp.min(eigenvalues)) > 0.0
    )
    return {
        "action": action_value,
        "action_standard_error": float(action_standard_error),
        "action_by_time": np.asarray(kinetic).tolist(),
        "maximum_projection_residual": maximum_projection_residual,
        "minimum_ess_fraction": minimum_ess_fraction,
        "maximum_gram_condition": maximum_gram_condition,
        "minimum_gram_eigenvalue": float(jnp.min(eigenvalues)),
        "maximum_moment_rate_residual": maximum_moment_rate_residual,
        "moment_rate_residual_by_time": np.asarray(normalized_moment_residual).tolist(),
        "valid": valid,
        "thresholds": {
            "maximum_projection_residual": cfg.maximum_projection_residual,
            "minimum_ess_fraction": cfg.minimum_ess_fraction,
            "maximum_gram_condition": cfg.maximum_gram_condition,
            "maximum_moment_rate_residual": cfg.maximum_moment_rate_residual,
        },
    }
