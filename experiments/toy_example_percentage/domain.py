from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import numpy as np

from mfsi.grid import CartesianGrid2D

Array = jax.Array


@dataclass(frozen=True)
class ToyEndpointSource:
    """Two-lobe endpoint laws used only by the toy benchmark."""

    radius: float = 1.5
    sigma: float = 0.30

    def sample(self, key: Array, n: int, endpoint: int) -> Array:
        if endpoint not in (0, 1):
            raise ValueError("endpoint must be 0 or 1")
        ksign, knoise = jax.random.split(key)
        signs = jnp.where(jax.random.bernoulli(ksign, 0.5, (n,)), 1.0, -1.0)
        noise = self.sigma * jax.random.normal(knoise, (n, 2), dtype=jnp.float64)
        if endpoint == 0:
            mean = jnp.stack([self.radius * signs, jnp.zeros_like(signs)], axis=-1)
        else:
            mean = jnp.stack([jnp.zeros_like(signs), self.radius * signs], axis=-1)
        return mean + noise

    def gauss_hermite_bank(self, order: int) -> tuple[Array, Array]:
        """Deterministic quadrature bank for the initial two-Gaussian mixture."""
        z, wz = np.polynomial.hermite.hermgauss(int(order))
        one_w = wz / math.sqrt(math.pi)
        zz1, zz2 = np.meshgrid(z, z, indexing="ij")
        ww1, ww2 = np.meshgrid(one_w, one_w, indexing="ij")
        noise = math.sqrt(2.0) * self.sigma * np.stack([zz1.reshape(-1), zz2.reshape(-1)], axis=-1)
        w2 = (ww1 * ww2).reshape(-1)
        plus = noise + np.array([self.radius, 0.0])
        minus = noise + np.array([-self.radius, 0.0])
        x0 = np.concatenate([plus, minus], axis=0)
        weights = np.concatenate([0.5 * w2, 0.5 * w2], axis=0)
        weights /= weights.sum()
        return jnp.asarray(x0, dtype=jnp.float64), jnp.asarray(weights, dtype=jnp.float64)


@dataclass(frozen=True)
class ToyPopulation:
    """Hidden population path used only as the toy benchmark oracle."""

    grid: CartesianGrid2D
    radius: float = 1.5
    sigma: float = 0.30
    alpha_min: float = math.pi / 6.0
    alpha_max: float = math.pi / 3.0

    @staticmethod
    def direction(theta: Array) -> Array:
        return jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)

    def component_density(self, alpha: Array) -> Array:
        x = self.grid.points()
        mu = self.radius * self.direction(alpha)
        s2 = self.sigma**2
        norm = 1.0 / (2.0 * jnp.pi * s2)
        dplus = jnp.sum((x - mu) ** 2, axis=-1)
        dminus = jnp.sum((x + mu) ** 2, axis=-1)
        return 0.5 * norm * (jnp.exp(-0.5 * dplus / s2) + jnp.exp(-0.5 * dminus / s2))

    def mass(self, t: Array, alpha: Array) -> Array:
        t = jnp.asarray(t, dtype=jnp.float64)
        w0 = (1.0 - t) ** 2
        wm = 2.0 * t * (1.0 - t)
        w1 = t**2
        density = (
            w0 * self.component_density(jnp.asarray(0.0))
            + wm * self.component_density(alpha)
            + w1 * self.component_density(jnp.asarray(0.5 * jnp.pi))
        )
        mass = density * self.grid.cell_area
        return mass / jnp.sum(mass)

    def masses(self, times: Array, alpha: Array) -> Array:
        return jax.vmap(lambda t: self.mass(t, alpha))(times)

    def moment(self, family, eta: Array, t: Array, alpha: Array) -> Array:
        phi = family.features(self.grid.flat_points(), eta)
        return self.mass(t, alpha).reshape(-1) @ phi

    def covariance(self, family, eta: Array, t: Array, alpha: Array) -> Array:
        phi = family.features(self.grid.flat_points(), eta)
        mass = self.mass(t, alpha).reshape(-1)
        mean = mass @ phi
        centered = phi - mean[None, :]
        return centered.T @ (mass[:, None] * centered)

    def alpha_quadrature(self, n: int) -> tuple[Array, Array]:
        z, w = np.polynomial.legendre.leggauss(int(n))
        a, b = self.alpha_min, self.alpha_max
        x = 0.5 * (b - a) * z + 0.5 * (a + b)
        weights = 0.5 * w  # normalized uniform expectation on [a,b]
        return jnp.asarray(x, dtype=jnp.float64), jnp.asarray(weights, dtype=jnp.float64)
