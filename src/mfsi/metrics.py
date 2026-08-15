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


def gaussian_mmd_kernel_rect(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    bandwidth: float,
) -> Array:
    """Gaussian convolution kernel for a ``(ny, nx)`` rectangular mass grid."""
    if int(nx) < 1 or int(ny) < 1:
        raise ValueError("nx and ny must be >= 1")
    if float(dx) <= 0.0 or float(dy) <= 0.0 or float(bandwidth) <= 0.0:
        raise ValueError("dx, dy, and bandwidth must be positive")
    ox = jnp.arange(-(int(nx) - 1), int(nx), dtype=jnp.float64) * float(dx)
    oy = jnp.arange(-(int(ny) - 1), int(ny), dtype=jnp.float64) * float(dy)
    xx, yy = jnp.meshgrid(ox, oy, indexing="xy")
    return jnp.exp(-(xx * xx + yy * yy) / (2.0 * float(bandwidth) ** 2))


def multiscale_gaussian_mmd_kernel_rect(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    bandwidths,
) -> Array:
    """Equal-weight mixture of Gaussian kernels, still usable by one FFT MMD call."""
    values = tuple(float(b) for b in bandwidths)
    if not values:
        raise ValueError("bandwidths must be non-empty")
    if any(b <= 0.0 for b in values):
        raise ValueError("all bandwidths must be positive")
    kernels = [gaussian_mmd_kernel_rect(nx, ny, dx, dy, b) for b in values]
    return jnp.mean(jnp.stack(kernels, axis=0), axis=0)
