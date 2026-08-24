from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


class ConfigurationBank(NamedTuple):
    """One row is one complete many-body configuration, never one particle."""

    times: Array
    configurations: Array  # [time, sample, particle, xy]


@dataclass(frozen=True)
class SkyrmionConfig:
    n_particles: int = 16
    box: tuple[float, float] = (2.0, 1.0)
    interaction_strength: float = 0.035
    interaction_length: float = 0.16
    pinning_strength: float = 0.055
    pinning_width: float = 0.10
    pinning_centers: tuple[tuple[float, float], ...] = (
        (0.36, 0.24), (0.72, 0.74), (1.05, 0.46), (1.43, 0.78), (1.72, 0.25)
    )
    drive_start: float = 0.015
    drive_end: float = 0.13
    drive_transverse: float = 0.018
    dissipation: float = 1.0
    magnus: float = 0.32
    initial_jitter: float = 0.035
    noise_std: float = 0.006

    def __post_init__(self) -> None:
        if self.n_particles < 2:
            raise ValueError("n_particles must be >= 2")
        if min(self.box) <= 0.0:
            raise ValueError("box lengths must be positive")
        if self.interaction_length <= 0.0 or self.pinning_width <= 0.0:
            raise ValueError("force length scales must be positive")
        if self.dissipation <= 0.0:
            raise ValueError("dissipation must be positive")
        if self.noise_std < 0.0:
            raise ValueError("noise_std must be nonnegative")


def minimum_image(displacement: Array, box: Array) -> Array:
    """Shortest periodic displacement on the rectangular torus."""

    return displacement - box * jnp.round(displacement / box)


@dataclass(frozen=True)
class SkyrmionTruth:
    """Coarse-grained driven Thiele-type point-skyrmion system.

    The deterministic force is transformed by the inverse dissipative/Magnus
    mobility.  Weak Brownian increments and randomized initial configurations
    generate the ensemble; particle count is fixed throughout.
    """

    cfg: SkyrmionConfig = SkyrmionConfig()

    @property
    def box(self) -> Array:
        return jnp.asarray(self.cfg.box, dtype=jnp.float64)

    def deterministic_velocity(self, x: Array, t: Array) -> Array:
        x = jnp.asarray(x, dtype=jnp.float64)
        box = self.box
        pair = minimum_image(x[..., :, None, :] - x[..., None, :, :], box)
        r2 = jnp.sum(pair * pair, axis=-1)
        eye = jnp.eye(self.cfg.n_particles, dtype=bool)
        while eye.ndim < r2.ndim:
            eye = eye[None, ...]
        safe_r = jnp.sqrt(jnp.where(eye, 1.0, r2) + 1.0e-8)
        magnitude = (
            float(self.cfg.interaction_strength)
            * jnp.exp(-safe_r / float(self.cfg.interaction_length))
            / safe_r
        )
        magnitude = jnp.where(eye, 0.0, magnitude)
        interaction = jnp.sum(magnitude[..., None] * pair / safe_r[..., None], axis=-2)

        pins = jnp.asarray(self.cfg.pinning_centers, dtype=x.dtype)
        pin_delta = minimum_image(x[..., :, None, :] - pins, box)
        pin_r2 = jnp.sum(pin_delta * pin_delta, axis=-1)
        sigma2 = float(self.cfg.pinning_width) ** 2
        # Attractive Gaussian wells: -grad V.
        pinning = -float(self.cfg.pinning_strength) * jnp.sum(
            jnp.exp(-0.5 * pin_r2 / sigma2)[..., None] * pin_delta / sigma2,
            axis=-2,
        )

        ramp = jax.nn.sigmoid(10.0 * (jnp.asarray(t, x.dtype) - 0.46))
        drive_x = float(self.cfg.drive_start) + (
            float(self.cfg.drive_end) - float(self.cfg.drive_start)
        ) * ramp
        drive_y = float(self.cfg.drive_transverse) * jnp.sin(2.0 * jnp.pi * t)
        drive = jnp.stack([drive_x, drive_y])
        force = interaction + pinning + drive

        alpha = float(self.cfg.dissipation)
        gyro = float(self.cfg.magnus)
        denom = alpha * alpha + gyro * gyro
        rotated = jnp.stack([-force[..., 1], force[..., 0]], axis=-1)
        return (alpha * force + gyro * rotated) / denom

    def sample_initial(self, key: Array, samples: int) -> Array:
        n = self.cfg.n_particles
        nx = int(jnp.ceil(jnp.sqrt(n * self.cfg.box[0] / self.cfg.box[1])))
        ny = int(jnp.ceil(n / nx))
        gx = (jnp.arange(nx, dtype=jnp.float64) + 0.5) * self.cfg.box[0] / nx
        gy = (jnp.arange(ny, dtype=jnp.float64) + 0.5) * self.cfg.box[1] / ny
        grid = jnp.stack(jnp.meshgrid(gx, gy, indexing="xy"), axis=-1).reshape(-1, 2)[:n]
        kj, kp = jax.random.split(key)
        jitter = float(self.cfg.initial_jitter) * jax.random.normal(
            kj, (int(samples), n, 2), dtype=jnp.float64
        )
        configurations = jnp.mod(grid[None, :, :] + jitter, self.box)
        # Random labels ensure no downstream component can rely on lattice order.
        permutations = jax.vmap(lambda k: jax.random.permutation(k, n))(
            jax.random.split(kp, int(samples))
        )
        return jax.vmap(lambda row, p: row[p])(configurations, permutations)

    def rollout(
        self,
        x0: Array,
        times: Array,
        *,
        key: Array,
        substeps_per_interval: int = 24,
    ) -> Array:
        if substeps_per_interval < 1:
            raise ValueError("substeps_per_interval must be >= 1")
        x0 = jnp.asarray(x0, dtype=jnp.float64)
        times = jnp.asarray(times, dtype=jnp.float64)
        keys = jax.random.split(key, (len(times) - 1) * int(substeps_per_interval)).reshape(
            len(times) - 1, int(substeps_per_interval), 2
        )

        def interval(x, inputs):
            t0, t1, interval_keys = inputs
            dt = (t1 - t0) / float(substeps_per_interval)

            def substep(state, inp):
                idx, noise_key = inp
                ti = t0 + idx.astype(jnp.float64) * dt
                velocity = self.deterministic_velocity(state, ti)
                noise = float(self.cfg.noise_std) * jnp.sqrt(dt) * jax.random.normal(
                    noise_key, state.shape, dtype=state.dtype
                )
                nxt = jnp.mod(state + dt * velocity + noise, self.box)
                return nxt, None

            indices = jnp.arange(int(substeps_per_interval), dtype=jnp.int32)
            xn, _ = jax.lax.scan(substep, x, (indices, interval_keys))
            return xn, xn

        _, nodes = jax.lax.scan(interval, x0, (times[:-1], times[1:], keys))
        return jnp.concatenate([x0[None, ...], nodes], axis=0)

    def make_bank(
        self,
        *,
        seed: int,
        samples: int,
        times: Array,
        substeps_per_interval: int = 24,
    ) -> ConfigurationBank:
        initial_key, noise_key = jax.random.split(jax.random.PRNGKey(int(seed)))
        x0 = self.sample_initial(initial_key, int(samples))
        nodes = self.rollout(
            x0, times, key=noise_key, substeps_per_interval=substeps_per_interval
        )
        return ConfigurationBank(jnp.asarray(times, dtype=jnp.float64), nodes)

