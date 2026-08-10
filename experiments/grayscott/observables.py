"""Low-dimensional differentiable observations of periodic scalar fields."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array


@dataclass(frozen=True)
class ShellDefinition:
    centers: tuple[float, ...] = (0.0625, 0.125, 0.1875)
    widths: tuple[float, ...] = (0.045, 0.050, 0.055)

    def __post_init__(self):
        if len(self.centers) != len(self.widths) or not self.centers:
            raise ValueError("shell centers and widths must have equal nonzero length")


def radial_frequencies(height: int, width: int, dtype=jnp.float32) -> Array:
    fy = jnp.fft.fftfreq(height).astype(dtype)
    fx = jnp.fft.fftfreq(width).astype(dtype)
    return jnp.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)


def shell_weights(height: int, width: int, shells: ShellDefinition, dtype=jnp.float32) -> Array:
    radius = radial_frequencies(height, width, dtype)
    centers = jnp.asarray(shells.centers, dtype=dtype)
    widths = jnp.asarray(shells.widths, dtype=dtype)
    return jnp.exp(-0.5 * ((radius[None] - centers[:, None, None]) / widths[:, None, None]) ** 2)


def shell_powers(fields: Array, shells: ShellDefinition) -> Array:
    """Return `(1/N) sum_k w_j(k)|FFT(V)_k|^2` with orthonormal FFT."""
    fields = jnp.asarray(fields)
    if fields.shape[-3] != 1:
        raise ValueError("the first benchmark expects one V channel")
    height, width = fields.shape[-2:]
    spectrum = jnp.fft.fft2(fields[..., 0, :, :], norm="ortho")
    power = jnp.real(spectrum * jnp.conj(spectrum))
    weights = shell_weights(height, width, shells, fields.dtype)
    return jnp.einsum("...hw,rhw->...r", power, weights) / (height * width)


def field_observables(
    fields: Array,
    shells: ShellDefinition = ShellDefinition(),
    components: Sequence[str] = ("mean", "second_moment", "shell_1", "shell_2"),
) -> Array:
    """Evaluate configurable Phi-2 through Phi-5 features in physical units."""
    fields = jnp.asarray(fields)
    if fields.ndim < 3 or fields.shape[-3] != 1:
        raise ValueError("fields must end in [1, H, W]")
    mean = jnp.mean(fields, axis=(-3, -2, -1))
    second = jnp.mean(fields * fields, axis=(-3, -2, -1))
    powers = shell_powers(fields, shells)
    values = {"mean": mean, "second_moment": second}
    values.update({f"shell_{index + 1}": powers[..., index] for index in range(powers.shape[-1])})
    try:
        return jnp.stack([values[name] for name in components], axis=-1)
    except KeyError as exc:
        raise ValueError(f"unknown or unavailable observable {exc.args[0]}") from exc


def fit_standardization(features: np.ndarray, floor: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(features, axis=0, dtype=np.float64)
    scale = np.std(features, axis=0, ddof=1, dtype=np.float64)
    return center, np.maximum(scale, floor)


def standardize(features: Array | np.ndarray, center: Array | np.ndarray, scale: Array | np.ndarray):
    return (jnp.asarray(features) - jnp.asarray(center)) / jnp.asarray(scale)
