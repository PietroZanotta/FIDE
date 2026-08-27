"""Large deterministic analytic invariant dictionary for production Galerkin work."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .production_artifacts import require_production_output_path

Array = jax.Array


@dataclass(frozen=True)
class HybridInvariantDictionary:
    box: tuple[float, float]
    wavevectors: Array
    radial_centers: Array
    radial_widths: Array
    feature_kind: Array
    feature_source_index: Array
    base_means: Array | None = None
    energy_scales: Array | None = None

    @property
    def size(self) -> int:
        return int(self.feature_kind.shape[0])


def _ordered_wavevectors(count: int, box: tuple[float, float]) -> Array:
    candidates: list[tuple[float, int, int]] = []
    limit = 1
    while len(candidates) < count:
        limit += 1
        candidates = []
        for nx in range(0, limit + 1):
            for ny in range(-limit, limit + 1):
                if nx == 0 and ny <= 0:
                    continue
                if max(abs(nx), abs(ny)) > limit:
                    continue
                kx = 2.0 * math.pi * nx / box[0]
                ky = 2.0 * math.pi * ny / box[1]
                candidates.append((kx * kx + ky * ky, nx, ny))
        candidates.sort(key=lambda row: (row[0], abs(row[1]) + abs(row[2]), row[1], row[2]))
    selected = candidates[:count]
    return jnp.asarray([
        [2.0 * math.pi * nx / box[0], 2.0 * math.pi * ny / box[1]]
        for _, nx, ny in selected
    ], dtype=jnp.float64)


def make_hybrid_dictionary(
    *, box: tuple[float, float] = (2.0, 1.0),
    fourier_wavevector_count: int = 64,
    radial_count: int = 32,
) -> HybridInvariantDictionary:
    if fourier_wavevector_count != 2 * radial_count:
        raise ValueError("nested hybrid ordering currently requires two Fourier vectors per radial mode")
    wavevectors = _ordered_wavevectors(fourier_wavevector_count, box)
    maximum_chord_radius = math.sqrt((box[0] / math.pi) ** 2 + (box[1] / math.pi) ** 2)
    centers = jnp.linspace(0.03, maximum_chord_radius, radial_count, dtype=jnp.float64)
    spacing = float(centers[1] - centers[0]) if radial_count > 1 else maximum_chord_radius
    widths = jnp.full((radial_count,), max(1.75 * spacing, 0.035), dtype=jnp.float64)
    kinds: list[int] = []
    sources: list[int] = []
    for radial in range(radial_count):
        for wavevector in (2 * radial, 2 * radial + 1):
            kinds.extend((0, 1))  # cosine, sine
            sources.extend((wavevector, wavevector))
        kinds.append(2)
        sources.append(radial)
    return HybridInvariantDictionary(
        box=box,
        wavevectors=wavevectors,
        radial_centers=centers,
        radial_widths=widths,
        feature_kind=jnp.asarray(kinds, dtype=jnp.int32),
        feature_source_index=jnp.asarray(sources, dtype=jnp.int32),
    )


def _fourier_values_and_gradients(
    dictionary: HybridInvariantDictionary, configurations: Array
) -> tuple[Array, Array]:
    x = jnp.asarray(configurations, dtype=jnp.float64)
    phases = jnp.einsum("...pd,md->...pm", x, dictionary.wavevectors)
    particle_count = x.shape[-2]
    cosine = jnp.mean(jnp.cos(phases), axis=-2)
    sine = jnp.mean(jnp.sin(phases), axis=-2)
    cosine_grad = (
        -jnp.sin(phases)[..., :, :, None]
        * dictionary.wavevectors[None, None, :, :]
        / float(particle_count)
    )
    sine_grad = (
        jnp.cos(phases)[..., :, :, None]
        * dictionary.wavevectors[None, None, :, :]
        / float(particle_count)
    )
    cosine_grad = jnp.moveaxis(cosine_grad, -3, -2)
    sine_grad = jnp.moveaxis(sine_grad, -3, -2)
    return (
        jnp.stack((cosine, sine), axis=-1),
        jnp.stack((cosine_grad, sine_grad), axis=-3),
    )


def _radial_values_and_gradients(
    dictionary: HybridInvariantDictionary, configurations: Array
) -> tuple[Array, Array]:
    x = jnp.asarray(configurations, dtype=jnp.float64)
    particle_count = int(x.shape[-2])
    pair_i, pair_j = jnp.triu_indices(particle_count, 1)
    delta = x[..., pair_i, :] - x[..., pair_j, :]
    box = jnp.asarray(dictionary.box, dtype=x.dtype)
    chord = box / jnp.pi * jnp.sin(jnp.pi * delta / box)
    chord_derivative = jnp.cos(jnp.pi * delta / box)
    radius = jnp.sqrt(jnp.sum(chord * chord, axis=-1) + 1.0e-24)
    center = dictionary.radial_centers
    width = dictionary.radial_widths
    standardized = (radius[..., :, None] - center) / width
    radial = jnp.exp(-0.5 * standardized * standardized)
    values = jnp.mean(radial, axis=-2)
    dr_ddelta = chord * chord_derivative / radius[..., :, None]
    dg_ddelta = (
        radial[..., :, :, None]
        * (-(radius[..., :, None] - center) / (width * width))[..., :, :, None]
        * dr_ddelta[..., :, None, :]
        / float(pair_i.shape[0])
    )
    leading = x.shape[:-2]
    gradients = jnp.zeros(
        leading + (dictionary.radial_centers.shape[0], particle_count, 2),
        dtype=x.dtype,
    )
    contribution = jnp.moveaxis(dg_ddelta, -2, -3)
    gradients = gradients.at[..., pair_i, :].add(contribution)
    gradients = gradients.at[..., pair_j, :].add(-contribution)
    return values, gradients


def raw_values_and_gradients(
    dictionary: HybridInvariantDictionary, configurations: Array
) -> tuple[Array, Array]:
    """Return raw features ``[...,K]`` and state gradients ``[...,K,P,2]``."""

    fourier_values, fourier_gradients = _fourier_values_and_gradients(
        dictionary, configurations
    )
    radial_values, radial_gradients = _radial_values_and_gradients(
        dictionary, configurations
    )
    kind = dictionary.feature_kind
    source = dictionary.feature_source_index
    safe_wave = jnp.minimum(source, dictionary.wavevectors.shape[0] - 1)
    safe_radial = jnp.minimum(source, dictionary.radial_centers.shape[0] - 1)
    cos_values = fourier_values[..., safe_wave, 0]
    sin_values = fourier_values[..., safe_wave, 1]
    pair_values = radial_values[..., safe_radial]
    values = jnp.where(kind == 0, cos_values, jnp.where(kind == 1, sin_values, pair_values))
    cos_grad = fourier_gradients[..., safe_wave, 0, :, :]
    sin_grad = fourier_gradients[..., safe_wave, 1, :, :]
    pair_grad = radial_gradients[..., safe_radial, :, :]
    gradients = jnp.where(
        (kind == 0)[..., None, None], cos_grad,
        jnp.where((kind == 1)[..., None, None], sin_grad, pair_grad),
    )
    return values, gradients


def normalized_values_and_gradients(
    dictionary: HybridInvariantDictionary, configurations: Array, time_index: int
) -> tuple[Array, Array]:
    if dictionary.base_means is None or dictionary.energy_scales is None:
        raise ValueError("dictionary normalization is not fitted")
    values, gradients = raw_values_and_gradients(dictionary, configurations)
    means = dictionary.base_means[int(time_index)]
    scales = dictionary.energy_scales[int(time_index)]
    return (
        (values - means) / scales,
        gradients / scales[:, None, None],
    )


def fit_frozen_normalization(
    dictionary: HybridInvariantDictionary,
    configurations: Array,
    base_weights: Array,
    *,
    chunk_size: int = 256,
) -> HybridInvariantDictionary:
    """Fit diagonal base means/Dirichlet scales once on an eta-independent bank."""

    time_count, sample_count = configurations.shape[:2]
    means = []
    scales = []
    evaluator = jax.jit(lambda rows: raw_values_and_gradients(dictionary, rows))
    for time_index in range(int(time_count)):
        mean = jnp.zeros((dictionary.size,), dtype=jnp.float64)
        energy = jnp.zeros_like(mean)
        for start in range(0, int(sample_count), int(chunk_size)):
            stop = min(start + int(chunk_size), int(sample_count))
            values, gradients = evaluator(configurations[time_index, start:stop])
            weights = base_weights[time_index, start:stop]
            mean = mean + jnp.einsum("n,nk->k", weights, values)
            energy = energy + jnp.einsum("n,nkpd,nkpd->k", weights, gradients, gradients)
        means.append(mean)
        scales.append(jnp.sqrt(jnp.maximum(energy, 1.0e-12)))
    return HybridInvariantDictionary(
        box=dictionary.box,
        wavevectors=dictionary.wavevectors,
        radial_centers=dictionary.radial_centers,
        radial_widths=dictionary.radial_widths,
        feature_kind=dictionary.feature_kind,
        feature_source_index=dictionary.feature_source_index,
        base_means=jnp.stack(means),
        energy_scales=jnp.stack(scales),
    )


def dictionary_metadata(dictionary: HybridInvariantDictionary) -> dict[str, Any]:
    kinds = np.asarray(dictionary.feature_kind)
    return {
        "kind": "analytic_periodic_permutation_invariant_hybrid_v1",
        "size": dictionary.size,
        "box": list(dictionary.box),
        "fourier_wavevector_count": int(dictionary.wavevectors.shape[0]),
        "fourier_coordinate_count": int(np.sum(kinds != 2)),
        "radial_coordinate_count": int(np.sum(kinds == 2)),
        "ordering": "two reciprocal vectors (cos,sin each), then one radial mode, repeated",
        "normalization": "per-time fixed base-bank mean subtraction and diagonal Dirichlet-energy scaling",
        "eta_independent": True,
        "prefix_nested": True,
        "wavevectors": np.asarray(dictionary.wavevectors).tolist(),
        "radial_centers": np.asarray(dictionary.radial_centers).tolist(),
        "radial_widths": np.asarray(dictionary.radial_widths).tolist(),
        "feature_kind": kinds.tolist(),
        "feature_source_index": np.asarray(dictionary.feature_source_index).tolist(),
    }


def save_dictionary(path: Path, dictionary: HybridInvariantDictionary) -> None:
    path = require_production_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        wavevectors=np.asarray(dictionary.wavevectors),
        radial_centers=np.asarray(dictionary.radial_centers),
        radial_widths=np.asarray(dictionary.radial_widths),
        feature_kind=np.asarray(dictionary.feature_kind),
        feature_source_index=np.asarray(dictionary.feature_source_index),
        base_means=np.asarray(dictionary.base_means),
        energy_scales=np.asarray(dictionary.energy_scales),
    )


def load_dictionary(path: Path, *, box: tuple[float, float]) -> HybridInvariantDictionary:
    with np.load(path, allow_pickle=False) as arrays:
        return HybridInvariantDictionary(
            box=box,
            wavevectors=jnp.asarray(arrays["wavevectors"], dtype=jnp.float64),
            radial_centers=jnp.asarray(arrays["radial_centers"], dtype=jnp.float64),
            radial_widths=jnp.asarray(arrays["radial_widths"], dtype=jnp.float64),
            feature_kind=jnp.asarray(arrays["feature_kind"], dtype=jnp.int32),
            feature_source_index=jnp.asarray(
                arrays["feature_source_index"], dtype=jnp.int32
            ),
            base_means=jnp.asarray(arrays["base_means"], dtype=jnp.float64),
            energy_scales=jnp.asarray(arrays["energy_scales"], dtype=jnp.float64),
        )


__all__ = [
    "HybridInvariantDictionary", "dictionary_metadata", "fit_frozen_normalization",
    "load_dictionary", "make_hybrid_dictionary", "normalized_values_and_gradients",
    "raw_values_and_gradients", "save_dictionary",
]
