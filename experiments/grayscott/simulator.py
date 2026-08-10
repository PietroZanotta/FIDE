"""Deterministic batched Gray–Scott simulation on a periodic square grid."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array


@dataclass(frozen=True)
class GrayScottParameters:
    feed: float
    kill: float
    diffusion_u: float = 0.16
    diffusion_v: float = 0.08


def periodic_laplacian(field: Array, spacing: float = 1.0) -> Array:
    """Five-point periodic finite-difference Laplacian on the last two axes."""
    return (
        jnp.roll(field, 1, axis=-2)
        + jnp.roll(field, -1, axis=-2)
        + jnp.roll(field, 1, axis=-1)
        + jnp.roll(field, -1, axis=-1)
        - 4.0 * field
    ) / (spacing * spacing)


@partial(jax.jit, static_argnames=("steps",))
def _simulate_jax(
    initial_u: Array,
    initial_v: Array,
    feed: Array,
    kill: Array,
    diffusion_u: float,
    diffusion_v: float,
    dt: float,
    spacing: float,
    steps: int,
) -> tuple[Array, Array]:
    feed = jnp.asarray(feed, dtype=initial_u.dtype)
    kill = jnp.asarray(kill, dtype=initial_u.dtype)
    while feed.ndim < initial_u.ndim:
        feed = feed[..., None]
        kill = kill[..., None]

    def update(_, state):
        u, v = state
        uvv = u * v * v
        du = diffusion_u * periodic_laplacian(u, spacing) - uvv + feed * (1.0 - u)
        dv = diffusion_v * periodic_laplacian(v, spacing) + uvv - (feed + kill) * v
        return u + dt * du, v + dt * dv

    return jax.lax.fori_loop(0, steps, update, (initial_u, initial_v))


def simulate(
    initial_u: Array | np.ndarray,
    initial_v: Array | np.ndarray,
    parameters: GrayScottParameters | None = None,
    *,
    feed: Array | np.ndarray | float | None = None,
    kill: Array | np.ndarray | float | None = None,
    diffusion_u: float | None = None,
    diffusion_v: float | None = None,
    dt: float = 1.0,
    physical_time: float = 2500.0,
    spacing: float = 1.0,
) -> tuple[Array, Array]:
    """Simulate a batch shaped `[B, C, H, W]` without state clipping.

    `feed` and `kill` can be scalars or arrays broadcastable over the leading
    batch axes. A fixed initial state and arguments produce a deterministic
    result. Physical and interpolation times are intentionally separate.
    """
    if parameters is not None:
        if feed is not None or kill is not None:
            raise ValueError("pass either parameters or feed/kill, not both")
        feed, kill = parameters.feed, parameters.kill
        diffusion_u = parameters.diffusion_u if diffusion_u is None else diffusion_u
        diffusion_v = parameters.diffusion_v if diffusion_v is None else diffusion_v
    if feed is None or kill is None:
        raise ValueError("feed and kill are required")
    diffusion_u = 0.16 if diffusion_u is None else float(diffusion_u)
    diffusion_v = 0.08 if diffusion_v is None else float(diffusion_v)
    steps_float = physical_time / dt
    steps = int(round(steps_float))
    if not np.isclose(steps, steps_float, rtol=0.0, atol=1e-10):
        raise ValueError("physical_time must be an integer multiple of dt")
    u = jnp.asarray(initial_u)
    v = jnp.asarray(initial_v)
    if u.shape != v.shape or u.ndim < 3:
        raise ValueError("initial_u and initial_v must have matching [..., C, H, W] shapes")
    return _simulate_jax(
        u, v, jnp.asarray(feed), jnp.asarray(kill), diffusion_u, diffusion_v,
        float(dt), float(spacing), steps,
    )


def generate_initial_conditions(
    seeds: list[int] | np.ndarray,
    *,
    height: int = 32,
    width: int = 32,
    blob_count: tuple[int, int] = (2, 5),
    radius_range: tuple[float, float] = (1.8, 3.2),
    u_depletion_range: tuple[float, float] = (0.40, 0.52),
    v_amplitude_range: tuple[float, float] = (0.24, 0.34),
    noise_std: float = 0.002,
    dtype=np.float32,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Create a stochastic population with toroidal blobs and paired transforms."""
    if blob_count[0] < 1 or blob_count[1] < blob_count[0]:
        raise ValueError("invalid blob_count range")
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    us, vs, metadata = [], [], []
    for seed_value in np.asarray(seeds, dtype=np.int64):
        rng = np.random.default_rng(int(seed_value))
        profile = np.zeros((height, width), dtype=np.float64)
        count = int(rng.integers(blob_count[0], blob_count[1] + 1))
        blobs = []
        for _ in range(count):
            cy, cx = rng.uniform(0.0, height), rng.uniform(0.0, width)
            radius = rng.uniform(*radius_range)
            dy = np.minimum(np.abs(yy - cy), height - np.abs(yy - cy))
            dx = np.minimum(np.abs(xx - cx), width - np.abs(xx - cx))
            bump = np.exp(-0.5 * (dx * dx + dy * dy) / (radius * radius))
            profile = np.maximum(profile, bump)
            blobs.append({"cy": float(cy), "cx": float(cx), "radius": float(radius)})
        u_depletion = rng.uniform(*u_depletion_range)
        v_amplitude = rng.uniform(*v_amplitude_range)
        u = 1.0 - u_depletion * profile + rng.normal(0.0, noise_std, profile.shape)
        v = v_amplitude * profile + rng.normal(0.0, noise_std, profile.shape)

        # A label-blind D4 transform and periodic translation are applied to the
        # IC once; paired regimes receive exactly the same transformed state.
        rotation = int(rng.integers(0, 4))
        reflection = bool(rng.integers(0, 2))
        shift_y = int(rng.integers(0, height))
        shift_x = int(rng.integers(0, width))
        u = np.rot90(u, rotation)
        v = np.rot90(v, rotation)
        if reflection:
            u, v = np.flip(u, axis=-1), np.flip(v, axis=-1)
        u = np.roll(u, (shift_y, shift_x), axis=(-2, -1))
        v = np.roll(v, (shift_y, shift_x), axis=(-2, -1))
        us.append(u[None])
        vs.append(v[None])
        metadata.append({
            "seed": int(seed_value), "blobs": blobs, "rotation_quarters": rotation,
            "reflection": reflection, "shift_y": shift_y, "shift_x": shift_x,
            "u_depletion": float(u_depletion), "v_amplitude": float(v_amplitude),
        })
    return np.asarray(us, dtype=dtype), np.asarray(vs, dtype=dtype), metadata
