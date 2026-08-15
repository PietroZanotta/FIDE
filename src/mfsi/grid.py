from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

Array = jax.Array


@dataclass(frozen=True)
class CartesianGrid2D:
    half_width: float
    n: int

    @property
    def dx(self) -> float:
        return 2.0 * self.half_width / self.n

    @property
    def cell_area(self) -> float:
        return self.dx * self.dx

    def centers_1d(self) -> Array:
        return -self.half_width + (jnp.arange(self.n, dtype=jnp.float64) + 0.5) * self.dx

    def points(self) -> Array:
        c = self.centers_1d()
        xx, yy = jnp.meshgrid(c, c, indexing="xy")
        return jnp.stack([xx, yy], axis=-1)

    def flat_points(self) -> Array:
        return self.points().reshape((-1, 2))

    def in_domain(self, x: Array) -> Array:
        x = jnp.asarray(x, dtype=jnp.float64)
        L = self.half_width
        return (
            (x[..., 0] >= -L)
            & (x[..., 0] < L)
            & (x[..., 1] >= -L)
            & (x[..., 1] < L)
        )

    def flat_bin_index(self, x: Array) -> Array:
        """Hard histogram cell index; differentiable quantities should be weights."""
        x = jnp.asarray(x, dtype=jnp.float64)
        ij = jnp.floor((x + self.half_width) / self.dx).astype(jnp.int32)
        ix = jnp.clip(ij[..., 0], 0, self.n - 1)
        iy = jnp.clip(ij[..., 1], 0, self.n - 1)
        return iy * self.n + ix
