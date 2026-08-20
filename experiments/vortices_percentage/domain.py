from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array


@dataclass(frozen=True)
class DoubleGyreConfig:
    """Normalized-time double-gyre truth on [0,2] x [0,1]."""

    amplitude: float = 0.1
    epsilon: float = 0.25
    horizon: float = 10.0
    period: float = 10.0

    def __post_init__(self) -> None:
        if self.amplitude <= 0.0:
            raise ValueError("amplitude must be positive")
        if self.horizon <= 0.0 or self.period <= 0.0:
            raise ValueError("horizon and period must be positive")


@dataclass(frozen=True)
class InitialLawConfig:
    background_weight: float = 0.10
    mixture_weights: tuple[float, ...] = (0.30, 0.20, 0.25, 0.25)
    centers: tuple[tuple[float, float], ...] = (
        (0.45, 0.25),
        (0.78, 0.72),
        (1.28, 0.28),
        (1.62, 0.68),
    )
    std_x: float = 0.07
    std_y: float = 0.07

    def __post_init__(self) -> None:
        if not 0.0 <= self.background_weight < 1.0:
            raise ValueError("background_weight must lie in [0,1)")
        if len(self.mixture_weights) != len(self.centers) or not self.centers:
            raise ValueError("mixture_weights and centers must have the same nonzero length")
        if any(w < 0.0 for w in self.mixture_weights):
            raise ValueError("mixture weights must be nonnegative")
        if sum(self.mixture_weights) <= 0.0:
            raise ValueError("mixture weights must have positive total mass")
        if self.std_x <= 0.0 or self.std_y <= 0.0:
            raise ValueError("Gaussian standard deviations must be positive")


class TruthBank(NamedTuple):
    times: Array
    particles: Array  # [T,N,2]


@dataclass(frozen=True)
class DoubleGyreTruth:
    flow: DoubleGyreConfig = DoubleGyreConfig()
    initial: InitialLawConfig = InitialLawConfig()

    @staticmethod
    def domain_bounds() -> tuple[tuple[float, float], tuple[float, float]]:
        return (0.0, 2.0), (0.0, 1.0)

    def velocity(self, x: Array, t: Array) -> Array:
        """Velocity for normalized time t in [0,1].

        The textbook double gyre is written in physical time tau.  Since the MFSI
        code uses normalized time, this returns dX/dt = horizon * dX/dtau.
        """
        x = jnp.asarray(x, dtype=jnp.float64)
        t = jnp.asarray(t, dtype=jnp.float64)
        tau = float(self.flow.horizon) * t
        omega = 2.0 * jnp.pi / float(self.flow.period)
        a = float(self.flow.epsilon) * jnp.sin(omega * tau)
        b = 1.0 - 2.0 * a
        xx = x[..., 0]
        yy = x[..., 1]
        f = a * xx * xx + b * xx
        dfdx = 2.0 * a * xx + b
        A = float(self.flow.amplitude)
        vx_tau = -jnp.pi * A * jnp.sin(jnp.pi * f) * jnp.cos(jnp.pi * yy)
        vy_tau = jnp.pi * A * jnp.cos(jnp.pi * f) * jnp.sin(jnp.pi * yy) * dfdx
        return float(self.flow.horizon) * jnp.stack([vx_tau, vy_tau], axis=-1)

    def rollout(self, x0: Array, times: Array, *, substeps_per_interval: int = 32) -> Array:
        if int(substeps_per_interval) < 1:
            raise ValueError("substeps_per_interval must be >= 1")
        x0 = jnp.asarray(x0, dtype=jnp.float64)
        times = jnp.asarray(times, dtype=jnp.float64)
        if times.ndim != 1:
            raise ValueError("times must be one-dimensional")

        def interval_step(x: Array, pair: tuple[Array, Array]):
            t0, t1 = pair
            dt = (t1 - t0) / float(substeps_per_interval)

            def one_substep(i: int, state: Array) -> Array:
                ti = t0 + i.astype(jnp.float64) * dt
                k1 = self.velocity(state, ti)
                k2 = self.velocity(state + 0.5 * dt * k1, ti + 0.5 * dt)
                k3 = self.velocity(state + 0.5 * dt * k2, ti + 0.5 * dt)
                k4 = self.velocity(state + dt * k3, ti + dt)
                return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            xn = jax.lax.fori_loop(0, int(substeps_per_interval), one_substep, x)
            return xn, xn

        pairs = (times[:-1], times[1:])
        _, nodes = jax.lax.scan(interval_step, x0, pairs)
        return jnp.concatenate([x0[None, ...], nodes], axis=0)

    def sample_initial_numpy(self, seed: int, n: int) -> np.ndarray:
        """Sample exactly from the uniform-plus-truncated-Gaussian initial law."""
        n = int(n)
        if n < 1:
            raise ValueError("n must be >= 1")
        rng = np.random.default_rng(int(seed))
        out = np.empty((n, 2), dtype=np.float64)
        bg = rng.random(n) < float(self.initial.background_weight)
        n_bg = int(np.sum(bg))
        out[bg, 0] = rng.uniform(0.0, 2.0, size=n_bg)
        out[bg, 1] = rng.uniform(0.0, 1.0, size=n_bg)

        mixture_idx = np.flatnonzero(~bg)
        weights = np.asarray(self.initial.mixture_weights, dtype=np.float64)
        weights /= weights.sum()
        components = rng.choice(len(weights), size=len(mixture_idx), p=weights)
        std = np.asarray([self.initial.std_x, self.initial.std_y], dtype=np.float64)
        centers = np.asarray(self.initial.centers, dtype=np.float64)

        # Rejection is cheap for the declared centers/std and preserves the intended
        # truncated law instead of silently clipping Gaussian tails onto boundaries.
        for comp in range(len(weights)):
            target_idx = mixture_idx[components == comp]
            remaining = target_idx.copy()
            while remaining.size:
                draw = centers[comp] + rng.normal(size=(remaining.size, 2)) * std
                good = (
                    (draw[:, 0] >= 0.0)
                    & (draw[:, 0] <= 2.0)
                    & (draw[:, 1] >= 0.0)
                    & (draw[:, 1] <= 1.0)
                )
                out[remaining[good]] = draw[good]
                remaining = remaining[~good]
        return out

    def make_bank(
        self,
        *,
        seed: int,
        n: int,
        times: Array,
        substeps_per_interval: int = 32,
    ) -> TruthBank:
        x0 = jnp.asarray(self.sample_initial_numpy(seed, n), dtype=jnp.float64)
        nodes = self.rollout(x0, times, substeps_per_interval=substeps_per_interval)
        return TruthBank(jnp.asarray(times, dtype=jnp.float64), nodes)


@dataclass(frozen=True)
class EmpiricalEndpointSource:
    """EndpointSource backed by fixed endpoint arrays from the hidden simulator."""

    x0: Array
    x1: Array

    def __post_init__(self) -> None:
        if self.x0.shape != self.x1.shape or self.x0.ndim != 2 or self.x0.shape[-1] != 2:
            raise ValueError("x0 and x1 must both have shape [N,2]")

    def sample(self, key: Array, n: int, endpoint: int) -> Array:
        if endpoint not in (0, 1):
            raise ValueError("endpoint must be 0 or 1")
        bank = self.x0 if endpoint == 0 else self.x1
        idx = jax.random.randint(key, (int(n),), 0, int(bank.shape[0]))
        return bank[idx]
