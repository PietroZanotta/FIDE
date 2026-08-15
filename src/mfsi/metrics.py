from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy as jsp

Array = jax.Array


def gaussian_mmd_kernel(grid_n: int, dx: float, bandwidth: float) -> Array:
    offsets = jnp.arange(-(grid_n - 1), grid_n, dtype=jnp.float64) * float(dx)
    ox, oy = jnp.meshgrid(offsets, offsets, indexing="xy")
    return jnp.exp(-(ox * ox + oy * oy) / (2.0 * float(bandwidth) ** 2))


def gaussian_mmd2_grid_mass(p: Array, q: Array, kernel: Array) -> Array:
    p = jnp.asarray(p, dtype=jnp.float64)
    q = jnp.asarray(q, dtype=jnp.float64)
    kernel = jnp.asarray(kernel, dtype=jnp.float64)
    kp = jsp.signal.fftconvolve(p, kernel, mode="same")
    kq = jsp.signal.fftconvolve(q, kernel, mode="same")
    value = jnp.sum(p * kp) + jnp.sum(q * kq) - 2.0 * jnp.sum(p * kq)
    return jnp.maximum(value, 0.0)
