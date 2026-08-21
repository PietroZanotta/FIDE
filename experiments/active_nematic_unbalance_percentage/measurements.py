"""Differentiable sparse virtual imaging windows on the periodic square."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

Array = jax.Array
Channel = Literal["occupancy", "polarity_cos", "polarity_sin"]


@dataclass(frozen=True)
class PeriodicGaussianSensors:
    """Smooth periodic Gaussian-like windows with optional polarity channels.

    ``eta=[x1,y1,...,xR,yR]`` gives labelled sensor centers.  The spatial kernel
    uses squared chord distance, a smooth periodic surrogate for geodesic squared
    distance with the same local quadratic behavior.  Channel ordering is
    sensor-major, then the order declared in ``channels``.
    """

    box_size: float
    width: float = 1.5
    n_sensors: int = 4
    channels: tuple[Channel, ...] = ("occupancy", "polarity_cos", "polarity_sin")

    def __post_init__(self) -> None:
        valid = {"occupancy", "polarity_cos", "polarity_sin"}
        if self.box_size <= 0.0 or self.width <= 0.0:
            raise ValueError("box_size and width must be positive")
        if int(self.n_sensors) < 1 or not self.channels:
            raise ValueError("at least one sensor and one channel are required")
        if len(set(self.channels)) != len(self.channels) or not set(self.channels) <= valid:
            raise ValueError("measurement channels must be unique supported channel names")

    @property
    def n_observables(self) -> int:
        return int(self.n_sensors) * len(self.channels)

    @property
    def requires_polarity(self) -> bool:
        return any(channel != "occupancy" for channel in self.channels)

    def centers(self, eta: Array) -> Array:
        eta = jnp.asarray(eta, dtype=jnp.float64)
        expected = 2 * int(self.n_sensors)
        if eta.shape[-1] != expected:
            raise ValueError(f"eta must have trailing dimension {expected}")
        return eta.reshape(eta.shape[:-1] + (self.n_sensors, 2))

    def canonicalize(self, eta: Array) -> Array:
        return jnp.mod(jnp.asarray(eta, dtype=jnp.float64), float(self.box_size))

    def _window_and_gradient(self, x: Array, eta: Array) -> tuple[Array, Array]:
        centers = self.centers(self.canonicalize(eta))
        position = x[..., :2]
        delta = position[..., None, :] - centers
        phase = (2.0 * jnp.pi / float(self.box_size)) * delta
        scale = float(self.box_size) / (2.0 * jnp.pi)
        distance2 = 2.0 * scale**2 * jnp.sum(1.0 - jnp.cos(phase), axis=-1)
        window = jnp.exp(-0.5 * distance2 / float(self.width) ** 2)
        ddistance2 = (float(self.box_size) / jnp.pi) * jnp.sin(phase)
        gradient = -window[..., None] * ddistance2 / (2.0 * float(self.width) ** 2)
        return window, gradient

    def features(self, x: Array, eta: Array) -> Array:
        x = jnp.asarray(x, dtype=jnp.float64)
        if x.shape[-1] not in (2, 3):
            raise ValueError("state must have trailing dimension 2 or 3")
        if self.requires_polarity and x.shape[-1] != 3:
            raise ValueError("polarity channels require state (x,y,polarity)")
        window, _ = self._window_and_gradient(x, eta)
        values = []
        for channel in self.channels:
            if channel == "occupancy":
                values.append(window)
            elif channel == "polarity_cos":
                values.append(window * jnp.cos(x[..., 2, None]))
            else:
                values.append(window * jnp.sin(x[..., 2, None]))
        return jnp.stack(values, axis=-1).reshape(x.shape[:-1] + (self.n_observables,))

    def feature_gradients(self, x: Array, eta: Array) -> Array:
        """Return ``d Phi/d X`` with shape ``[...,observable,state_dim]``."""
        x = jnp.asarray(x, dtype=jnp.float64)
        if x.shape[-1] not in (2, 3):
            raise ValueError("state must have trailing dimension 2 or 3")
        if self.requires_polarity and x.shape[-1] != 3:
            raise ValueError("polarity channels require state (x,y,polarity)")
        window, spatial = self._window_and_gradient(x, eta)
        rows = []
        for channel in self.channels:
            if channel == "occupancy":
                factor = jnp.ones_like(window)
                angular = jnp.zeros_like(window)
            elif channel == "polarity_cos":
                factor = jnp.cos(x[..., 2, None])
                angular = -window * jnp.sin(x[..., 2, None])
            else:
                factor = jnp.sin(x[..., 2, None])
                angular = window * jnp.cos(x[..., 2, None])
            grad = spatial * factor[..., None]
            if x.shape[-1] == 3:
                grad = jnp.concatenate([grad, angular[..., None]], axis=-1)
            rows.append(grad)
        stacked = jnp.stack(rows, axis=-2)
        return stacked.reshape(x.shape[:-1] + (self.n_observables, x.shape[-1]))

    def raw_signals(self, x: Array, eta: Array, positive_count: Array) -> Array:
        """Count-scaled sensor signals, kept separate from normalized-law moments."""
        normalized = jnp.mean(self.features(x, eta), axis=-2)
        return jnp.asarray(positive_count, dtype=jnp.float64)[..., None] * normalized

    def eta_jacobian(self, x: Array, eta: Array) -> Array:
        return jax.jacfwd(lambda design: self.features(x, design))(eta)


def periodic_separation_violation(
    min_separation: float, *, n_sensors: int, box_size: float
):
    """Return a differentiable constraint using smooth periodic chord distances."""
    min_separation = float(min_separation)
    n_sensors = int(n_sensors)
    box_size = float(box_size)
    if min_separation < 0.0 or box_size <= 0.0:
        raise ValueError("min_separation must be nonnegative and box_size positive")
    if n_sensors < 2:
        return lambda eta: jnp.asarray(-jnp.inf, dtype=jnp.float64)

    def violation(eta: Array) -> Array:
        centers = jnp.mod(jnp.asarray(eta).reshape((n_sensors, 2)), box_size)
        delta = centers[:, None, :] - centers[None, :, :]
        phase = 2.0 * jnp.pi * delta / box_size
        scale = box_size / (2.0 * jnp.pi)
        distance = jnp.sqrt(
            jnp.maximum(2.0 * scale**2 * jnp.sum(1.0 - jnp.cos(phase), axis=-1), 1.0e-300)
        )
        distance = jnp.where(jnp.eye(n_sensors, dtype=bool), jnp.inf, distance)
        return min_separation - jnp.min(distance)

    return violation


def random_periodic_sensor_starts(
    key: Array,
    count: int,
    *,
    n_sensors: int,
    box_size: float,
    min_separation: float = 0.0,
    oversample: int = 64,
) -> Array:
    """Periodic counterpart of the vortices random point-sensor start API."""
    total = max(int(count) * int(oversample), int(count))
    starts = float(box_size) * jax.random.uniform(
        key, (total, int(n_sensors), 2), dtype=jnp.float64
    )
    constraint = periodic_separation_violation(
        min_separation, n_sensors=n_sensors, box_size=box_size
    )
    valid = jax.vmap(lambda row: constraint(row.reshape(-1)) <= 0.0)(starts)
    order = jnp.argsort(~valid)
    if int(jnp.sum(valid)) < int(count):
        raise ValueError("could not generate enough separated periodic sensor starts")
    return starts[order[: int(count)]].reshape((int(count), 2 * int(n_sensors)))
