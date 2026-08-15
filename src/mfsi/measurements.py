from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

Array = jax.Array


@dataclass(frozen=True)
class GaussianSensor2D:
    """Two-dimensional localized Gaussian sensors.

    ``eta`` contains sensor angles in radians. The implementation is JAX-only so
    gradients can flow through eta and through x.
    """

    radius: float = 1.5
    width: float = 0.45

    @staticmethod
    def _direction(theta: Array) -> Array:
        return jnp.stack((jnp.cos(theta), jnp.sin(theta)), axis=-1)

    def features(self, x: Array, eta: Array) -> Array:
        x = jnp.asarray(x, dtype=jnp.float64)
        eta = jnp.asarray(eta, dtype=jnp.float64)
        centers = self.radius * self._direction(eta)  # [K, 2]
        diff = x[..., None, :] - centers
        ell2 = self.width**2
        return jnp.exp(-0.5 * jnp.sum(diff * diff, axis=-1) / ell2)

    def feature_gradients(self, x: Array, eta: Array) -> Array:
        """Return d Phi / d x with shape ``x.shape[:-1] + (K, 2)``."""
        x = jnp.asarray(x, dtype=jnp.float64)
        eta = jnp.asarray(eta, dtype=jnp.float64)
        centers = self.radius * self._direction(eta)
        diff = x[..., None, :] - centers
        phi = self.features(x, eta)
        return -(diff / self.width**2) * phi[..., None]

    @staticmethod
    def canonicalize(eta: Array) -> Array:
        eta = jnp.mod(jnp.asarray(eta, dtype=jnp.float64), 2.0 * jnp.pi)
        return jnp.sort(eta)

    def eta_jacobian(self, x: Array, eta: Array) -> Array:
        """Return d Phi / d eta, useful for design optimization diagnostics."""
        return jax.jacfwd(lambda e: self.features(x, e))(eta)
