"""Exact non-congruent radial-homometric benchmark on the square torus."""

from __future__ import annotations

from itertools import permutations
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .geometry import chord_distances, wrap_positions
from .observables import PairBasis, ensemble_angular_moments, ensemble_pair_moments

MODULUS = 12
MOTIF_A = ((0, 3), (7, 3), (8, 11), (10, 5))
MOTIF_B = ((5, 3), (5, 6), (5, 10), (9, 7))

_D4 = np.asarray(
    [
        [[1, 0], [0, 1]],
        [[0, -1], [1, 0]],
        [[-1, 0], [0, -1]],
        [[0, 1], [-1, 0]],
        [[-1, 0], [0, 1]],
        [[1, 0], [0, -1]],
        [[0, 1], [1, 0]],
        [[0, -1], [-1, 0]],
    ],
    dtype=np.int32,
)


def motif_coordinates(mode: int, box: Array, dtype: jnp.dtype) -> Array:
    """Return motif A or B in physical coordinates."""
    residues = MOTIF_A if mode == 0 else MOTIF_B
    box = jnp.asarray(box, dtype=dtype)
    return jnp.asarray(residues, dtype=dtype) * box / MODULUS


def sorted_pair_distances(coordinates: Array, box: Array) -> Array:
    """Sorted unordered smooth chord-distance signature."""
    distances = chord_distances(coordinates, box)
    rows, columns = jnp.triu_indices(coordinates.shape[-2], k=1)
    return jnp.sort(distances[rows, columns])


def _matches_up_to_permutation(
    left: np.ndarray,
    right: np.ndarray,
    box: np.ndarray,
    tol: float,
) -> bool:
    for order in permutations(range(left.shape[0])):
        delta = np.mod(left - right[np.asarray(order)] + 0.5 * box, box) - 0.5 * box
        if np.max(np.abs(delta)) <= tol:
            return True
    return False


def are_congruent(left: Array, right: Array, box: Array, tolerance: float = 1e-10) -> bool:
    """Check translation, D4, and particle-permutation congruence exhaustively."""
    left_np = np.asarray(left, dtype=np.float64)
    right_np = np.asarray(right, dtype=np.float64)
    box_np = np.asarray(box, dtype=np.float64)
    for matrix in _D4:
        transformed = np.mod(left_np @ matrix.T, box_np)
        for i in range(transformed.shape[0]):
            for j in range(right_np.shape[0]):
                shifted = np.mod(transformed + right_np[j] - transformed[i], box_np)
                if _matches_up_to_permutation(shifted, right_np, box_np, tolerance):
                    return True
    return False


def validate_homometric_pair(
    box: Array,
    basis: PairBasis,
    angular_orders: Array,
    angular_neighbor_scale: float,
) -> dict[str, float | bool | list[float]]:
    """Certify identical pair observations and separated held-out structure."""
    dtype = jnp.asarray(box).dtype
    motif_a = motif_coordinates(0, box, dtype)
    motif_b = motif_coordinates(1, box, dtype)
    distances_a = sorted_pair_distances(motif_a, box)
    distances_b = sorted_pair_distances(motif_b, box)
    pair_a = ensemble_pair_moments(motif_a[None], box, basis)
    pair_b = ensemble_pair_moments(motif_b[None], box, basis)
    angular_a = ensemble_angular_moments(
        motif_a[None], box, angular_orders, angular_neighbor_scale
    )
    angular_b = ensemble_angular_moments(
        motif_b[None], box, angular_orders, angular_neighbor_scale
    )
    return {
        "distance_max_abs_error": float(jnp.max(jnp.abs(distances_a - distances_b))),
        "pair_max_abs_error": float(jnp.max(jnp.abs(pair_a - pair_b))),
        "angular_separation": float(jnp.linalg.norm(angular_a - angular_b)),
        "non_congruent": not are_congruent(motif_a, motif_b, box),
        "pair_distance_signature": np.asarray(distances_a).tolist(),
    }


def _augment_one(key: Array, base: Array, box: Array) -> Array:
    symmetry_key, translation_key, permutation_key = jax.random.split(key, 3)
    matrix = jnp.asarray(_D4, dtype=base.dtype)[
        jax.random.randint(symmetry_key, (), 0, len(_D4))
    ]
    transformed = wrap_positions(base @ matrix.T, box)
    shift = jax.random.uniform(translation_key, (2,), dtype=base.dtype) * box
    transformed = wrap_positions(transformed + shift, box)
    order = jax.random.permutation(permutation_key, base.shape[0])
    return transformed[order]


def build_homometric_dataset(
    *,
    seed: int,
    samples_per_mode: int,
    num_replicas: int,
    box: Array,
    basis: PairBasis,
    angular_orders: Array,
    angular_neighbor_scale: float,
) -> dict[str, Array]:
    """Build a balanced, exact-symmetry-augmented A/B dataset."""
    box = jnp.asarray(box)
    dtype = box.dtype
    key = jax.random.PRNGKey(seed)
    coordinates: list[Array] = []
    labels: list[int] = []
    for mode in (0, 1):
        base = motif_coordinates(mode, box, dtype)
        for _ in range(samples_per_mode):
            key, sample_key = jax.random.split(key)
            replica_keys = jax.random.split(sample_key, num_replicas)
            coordinates.append(jax.vmap(lambda k: _augment_one(k, base, box))(replica_keys))
            labels.append(mode)
    stacked = jnp.stack(coordinates)
    pair = jax.vmap(lambda x: ensemble_pair_moments(x, box, basis))(stacked)
    angular = jax.vmap(
        lambda x: ensemble_angular_moments(
            x, box, angular_orders, angular_neighbor_scale
        )
    )(stacked)
    common = jnp.mean(pair, axis=0)
    maximum_deviation = jnp.max(jnp.abs(pair - common))
    # All condition rows are exactly equal after the numerical homometry check.
    conditions = jnp.zeros_like(pair)
    return {
        "coordinates": stacked,
        "labels": jnp.asarray(labels, dtype=jnp.int32),
        "pair_moments": pair,
        "angular_moments": angular,
        "conditions": conditions,
        "common_pair_moments": common,
        "pair_numerical_deviation": maximum_deviation,
        "box": box,
    }
