from __future__ import annotations

import jax.numpy as jnp


def qoi_features(x):
    """Predeclared global QoIs, all scaled to natural O(1) magnitudes."""
    values = jnp.asarray(x, dtype=jnp.float64)
    xn = 0.5 * values[..., 0]
    y = values[..., 1]
    return jnp.stack([xn, y, xn * xn, y * y, xn * y], axis=-1)


__all__ = ["qoi_features"]

