from __future__ import annotations

import jax
import jax.numpy as jnp

from .domain import minimum_image

Array = jax.Array


def many_body_features(configurations: Array, box: tuple[float, float] = (2.0, 1.0)) -> Array:
    """Held-out invariant features; no optimized sensor output appears here."""

    x = jnp.asarray(configurations, dtype=jnp.float64)
    box_a = jnp.asarray(box, dtype=x.dtype)
    n = x.shape[-2]
    delta = minimum_image(x[..., :, None, :] - x[..., None, :, :], box_a)
    dist = jnp.sqrt(jnp.sum(delta * delta, axis=-1) + 1.0e-12)
    mask = 1.0 - jnp.eye(n, dtype=x.dtype)
    radial_centers = jnp.asarray([0.10, 0.20, 0.32, 0.48], dtype=x.dtype)
    radial = jnp.sum(
        mask[..., None]
        * jnp.exp(-0.5 * ((dist[..., None] - radial_centers) / 0.055) ** 2),
        axis=(-3, -2),
    ) / float(n * (n - 1))

    wavevectors = 2.0 * jnp.pi * jnp.asarray(
        [[1.0 / box[0], 0.0], [0.0, 1.0 / box[1]], [1.0 / box[0], 1.0 / box[1]], [2.0 / box[0], 0.0]],
        dtype=x.dtype,
    )
    phase = jnp.einsum("...nd,kd->...nk", x, wavevectors)
    structure = (jnp.sum(jnp.cos(phase), axis=-2) ** 2 + jnp.sum(jnp.sin(phase), axis=-2) ** 2) / float(n * n)

    theta = jnp.arctan2(delta[..., 1], delta[..., 0])
    neighbor = mask * jnp.exp(-0.5 * (dist / 0.26) ** 2)
    psi6 = jnp.sum(neighbor * jnp.exp(6.0j * theta), axis=-1) / jnp.maximum(jnp.sum(neighbor, axis=-1), 1.0e-12)
    hexatic = jnp.mean(jnp.abs(psi6), axis=-1, keepdims=True)
    return jnp.concatenate([radial, structure, hexatic], axis=-1)


def whitening_from_truth(features: Array, ridge: float = 1.0e-5) -> Array:
    flat = jnp.asarray(features).reshape(-1, features.shape[-1])
    centered = flat - jnp.mean(flat, axis=0)
    covariance = centered.T @ centered / jnp.maximum(float(flat.shape[0] - 1), 1.0)
    scale = jnp.maximum(jnp.trace(covariance) / covariance.shape[0], 1.0e-8)
    return jnp.linalg.inv(covariance + float(ridge) * scale * jnp.eye(covariance.shape[0]))


def integrated_risk(
    projected_weights: Array,
    reference_features: Array,
    truth_feature_means: Array,
    whitening: Array,
    time_weights: Array,
) -> Array:
    predicted = jnp.einsum("tn,tnf->tf", projected_weights, reference_features)
    error = predicted - truth_feature_means
    rows = jnp.einsum("ti,ij,tj->t", error, whitening, error)
    return jnp.sum(jnp.asarray(time_weights) * rows)

