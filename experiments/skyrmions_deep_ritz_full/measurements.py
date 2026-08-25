from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .domain import minimum_image

Array = jax.Array


@dataclass(frozen=True)
class LocalDensitySensors:
    n_sensors: int
    width: float
    box: tuple[float, float] = (2.0, 1.0)
    min_separation: float = 0.18

    def __post_init__(self) -> None:
        if self.n_sensors < 1 or self.width <= 0.0:
            raise ValueError("sensor count and width must be positive")

    def centers(self, eta: Array) -> Array:
        eta = jnp.asarray(eta, dtype=jnp.float64)
        if eta.shape != (2 * self.n_sensors,):
            raise ValueError(f"eta must have shape ({2 * self.n_sensors},)")
        return eta.reshape(self.n_sensors, 2)

    def features(self, configurations: Array, eta: Array) -> Array:
        x = jnp.asarray(configurations, dtype=jnp.float64)
        centers = self.centers(eta)
        delta = minimum_image(x[..., :, None, :] - centers, jnp.asarray(self.box))
        r2 = jnp.sum(delta * delta, axis=-1)
        return jnp.mean(jnp.exp(-0.5 * r2 / float(self.width) ** 2), axis=-2)

    def jvp(self, configurations: Array, directions: Array, eta: Array) -> Array:
        return jax.jvp(lambda z: self.features(z, eta), (configurations,), (directions,))[1]

    def geometry_valid(self, eta: Array) -> Array:
        centers = self.centers(eta)
        box = jnp.asarray(self.box, dtype=centers.dtype)
        inside = jnp.all((centers >= 0.0) & (centers <= box))
        delta = minimum_image(centers[:, None, :] - centers[None, :, :], box)
        distances = jnp.sqrt(jnp.sum(delta * delta, axis=-1) + jnp.eye(self.n_sensors))
        separated = jnp.all(jnp.where(jnp.eye(self.n_sensors, dtype=bool), True, distances >= self.min_separation))
        return inside & separated


def random_sensor_designs(
    key: Array,
    *,
    count: int,
    family: LocalDensitySensors,
    oversample: int = 16,
) -> Array:
    """Deterministic feasible CRN candidate pool."""

    box = jnp.asarray(family.box, dtype=jnp.float64)
    draws = jax.random.uniform(
        key, (int(count) * int(oversample), family.n_sensors, 2), dtype=jnp.float64
    ) * box
    valid = jax.vmap(lambda x: family.geometry_valid(x.reshape(-1)))(draws)
    # Fixed-size selection is intentionally performed on the host boundary.
    import numpy as np

    accepted = np.asarray(draws)[np.asarray(valid)]
    if len(accepted) < int(count):
        raise RuntimeError("could not generate enough separated sensor designs")
    return jnp.asarray(accepted[: int(count)].reshape(int(count), -1))


def local_sensor_designs(
    key: Array,
    centers: Array,
    *,
    count_per_center: int,
    scale: float,
    family: LocalDensitySensors,
) -> Array:
    """Deterministic periodic local clouds around promising sensor geometries."""

    import numpy as np

    centers_np = np.asarray(centers, dtype=np.float64).reshape(-1, family.n_sensors, 2)
    if not len(centers_np) or int(count_per_center) <= 0:
        return jnp.empty((0, 2 * family.n_sensors), dtype=jnp.float64)
    noise = np.asarray(
        jax.random.normal(
            key,
            (len(centers_np), int(count_per_center), family.n_sensors, 2),
            dtype=jnp.float64,
        )
    )
    # Quadratic radii provide both near-anchor refinement and broader escapes.
    radii = (np.arange(1, int(count_per_center) + 1, dtype=np.float64) / int(count_per_center)) ** 2
    draws = centers_np[:, None, :, :] + float(scale) * radii[None, :, None, None] * noise
    draws = np.mod(draws, np.asarray(family.box, dtype=np.float64))
    flat = draws.reshape(-1, family.n_sensors, 2)
    valid = np.asarray(
        jax.vmap(lambda x: family.geometry_valid(x.reshape(-1)))(jnp.asarray(flat))
    )
    return jnp.asarray(flat[valid].reshape(-1, 2 * family.n_sensors), dtype=jnp.float64)
