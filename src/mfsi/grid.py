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


@dataclass(frozen=True)
class RectangularGrid2D:
    """Axis-aligned 2-D cell-centered grid on a rectangular domain.

    This class is additive: ``CartesianGrid2D`` retains its historical centered-square
    semantics.  ``RectangularGrid2D`` is intended for benchmarks such as the double
    gyre on ``[0, 2] x [0, 1]``.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    nx: int
    ny: int

    def __post_init__(self) -> None:
        if not self.x_max > self.x_min:
            raise ValueError("x_max must be greater than x_min")
        if not self.y_max > self.y_min:
            raise ValueError("y_max must be greater than y_min")
        if int(self.nx) < 1 or int(self.ny) < 1:
            raise ValueError("nx and ny must be >= 1")

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.nx

    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / self.ny

    @property
    def cell_area(self) -> float:
        return self.dx * self.dy

    @property
    def shape(self) -> tuple[int, int]:
        """Array shape in NumPy/JAX row-major convention: ``(ny, nx)``."""
        return (self.ny, self.nx)

    @property
    def size(self) -> int:
        return self.nx * self.ny

    def x_centers(self) -> Array:
        return self.x_min + (jnp.arange(self.nx, dtype=jnp.float64) + 0.5) * self.dx

    def y_centers(self) -> Array:
        return self.y_min + (jnp.arange(self.ny, dtype=jnp.float64) + 0.5) * self.dy

    def points(self) -> Array:
        xx, yy = jnp.meshgrid(self.x_centers(), self.y_centers(), indexing="xy")
        return jnp.stack([xx, yy], axis=-1)

    def flat_points(self) -> Array:
        return self.points().reshape((-1, 2))

    def in_domain(self, x: Array) -> Array:
        x = jnp.asarray(x, dtype=jnp.float64)
        return (
            (x[..., 0] >= self.x_min)
            & (x[..., 0] < self.x_max)
            & (x[..., 1] >= self.y_min)
            & (x[..., 1] < self.y_max)
        )

    def flat_bin_index(self, x: Array) -> Array:
        """Hard histogram cell index; eta-dependent quantities should be weights."""
        x = jnp.asarray(x, dtype=jnp.float64)
        ix = jnp.floor((x[..., 0] - self.x_min) / self.dx).astype(jnp.int32)
        iy = jnp.floor((x[..., 1] - self.y_min) / self.dy).astype(jnp.int32)
        ix = jnp.clip(ix, 0, self.nx - 1)
        iy = jnp.clip(iy, 0, self.ny - 1)
        return iy * self.nx + ix

    def require_isotropic_spacing(self, *, atol: float = 1.0e-12) -> float:
        """Return the common spacing or raise if ``dx != dy``.

        The current weighted-Poisson solver uses one spatial spacing.  Rectangular
        domains are therefore supported without changing that solver whenever the
        grid aspect ratio is chosen so physical cells are square (e.g. ``nx=2*ny``
        on ``[0,2]x[0,1]``).
        """
        if abs(float(self.dx) - float(self.dy)) > float(atol):
            raise ValueError(
                "current Poisson solver requires square physical cells; "
                f"got dx={self.dx:g}, dy={self.dy:g}"
            )
        return float(self.dx)
