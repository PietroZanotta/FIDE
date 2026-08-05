"""Exact radial-homometric benchmark on a periodic square box.

The benchmark embeds two four-point subsets of the discrete torus ``Z_12^2``::

    A = {(0, 3), (7, 3), (8, 11), (10, 5)}
    B = {(5, 3), (5, 6), (5, 10), (9, 7)}.

After scaling by the box length, the two configurations have exactly the same
multiset of smooth chord distances used by the project's observed radial pair
statistics.  They are nevertheless not related by a global torus translation,
a square-box D4 transformation, or particle relabeling, and a held-out smooth
angular descriptor separates them strongly.

Dataset augmentation applies only exact symmetries: global translations, D4
box transformations, and an optional particle relabeling.  Flow training also
performs a shared source-target permutation, so the stored benchmark can retain
canonical particle labels while the learned distribution remains exchangeable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations
from typing import Any, Final

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .energies import EnergyParameters, total_energy_per_configuration
from .geometry import chord_distances, wrap_positions
from .observables import (
    PairBasis,
    angular_cosine_moments,
    ensemble_angular_cosine_moments,
    ensemble_pair_moments,
    pair_diagnostics,
)

HOMOMETRIC_MODULUS: Final[int] = 12
HOMOMETRIC_A: Final[tuple[tuple[int, int], ...]] = (
    (0, 3),
    (7, 3),
    (8, 11),
    (10, 5),
)
HOMOMETRIC_B: Final[tuple[tuple[int, int], ...]] = (
    (5, 3),
    (5, 6),
    (5, 10),
    (9, 7),
)
HOMOMETRIC_NAMES: Final[tuple[str, str]] = ("A", "B")


@dataclass(frozen=True)
class HomometricDatasetConfig:
    """Validated settings for the exact two-mode benchmark."""

    seed: int
    dtype: str
    box: tuple[float, float]
    num_replicas: int
    samples_per_mode: int
    pair_basis_num: int
    pair_basis_r_min: float
    pair_basis_r_max: float
    pair_basis_width: float
    angular_orders: tuple[int, ...]
    angular_neighbor_scale: float
    overlap_threshold: float
    physical_r0: float
    physical_kappa: float
    random_d4: bool = True
    random_translation: bool = True
    random_permutation: bool = False

    def validate(self) -> None:
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")
        if len(self.box) != 2 or min(self.box) <= 0:
            raise ValueError("box must contain two positive side lengths")
        if not np.isclose(self.box[0], self.box[1]):
            raise ValueError("the homometric D4 benchmark requires a square box")
        if self.num_replicas < 1:
            raise ValueError("num_replicas must be positive")
        if self.samples_per_mode < 2:
            raise ValueError("samples_per_mode must be at least two")
        if self.pair_basis_num < 1:
            raise ValueError("pair_basis_num must be positive")
        if not 0 <= self.pair_basis_r_min < self.pair_basis_r_max:
            raise ValueError("pair basis range must satisfy 0 <= r_min < r_max")
        if self.pair_basis_width <= 0:
            raise ValueError("pair_basis_width must be positive")
        if not self.angular_orders or min(self.angular_orders) < 1:
            raise ValueError("angular_orders must contain positive integers")
        if self.angular_neighbor_scale <= 0:
            raise ValueError("angular_neighbor_scale must be positive")
        if self.overlap_threshold <= 0:
            raise ValueError("overlap_threshold must be positive")
        if self.physical_r0 <= 0 or self.physical_kappa <= 0:
            raise ValueError("physical parameters must be positive")

    @property
    def jax_dtype(self) -> jnp.dtype:
        return jnp.float64 if self.dtype == "float64" else jnp.float32

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "HomometricDatasetConfig":
        """Construct from the repository YAML schema with unknown-key checks."""
        expected = {
            "seed",
            "dtype",
            "box",
            "num_replicas",
            "samples_per_mode",
            "pair_basis",
            "angular_orders",
            "angular_neighbor_scale",
            "overlap_threshold",
            "physical",
            "augmentations",
        }
        unknown = sorted(set(raw) - expected)
        missing = sorted(expected - set(raw))
        if unknown:
            raise ValueError(f"unknown homometric config keys: {unknown}")
        if missing:
            raise ValueError(f"missing homometric config keys: {missing}")
        pair_basis = raw["pair_basis"]
        physical = raw["physical"]
        augmentations = raw["augmentations"]
        config = cls(
            seed=int(raw["seed"]),
            dtype=str(raw["dtype"]),
            box=tuple(float(value) for value in raw["box"]),
            num_replicas=int(raw["num_replicas"]),
            samples_per_mode=int(raw["samples_per_mode"]),
            pair_basis_num=int(pair_basis["num_basis"]),
            pair_basis_r_min=float(pair_basis["r_min"]),
            pair_basis_r_max=float(pair_basis["r_max"]),
            pair_basis_width=float(pair_basis["width"]),
            angular_orders=tuple(int(value) for value in raw["angular_orders"]),
            angular_neighbor_scale=float(raw["angular_neighbor_scale"]),
            overlap_threshold=float(raw["overlap_threshold"]),
            physical_r0=float(physical["r0"]),
            physical_kappa=float(physical["kappa"]),
            random_d4=bool(augmentations["d4"]),
            random_translation=bool(augmentations["translation"]),
            random_permutation=bool(augmentations["permutation"]),
        )
        config.validate()
        return config


def base_homometric_coordinates(
    mode: int | str,
    box: Array,
    *,
    dtype: jnp.dtype = jnp.float64,
) -> Array:
    """Embed one four-point lattice motif in a square physical box."""
    if mode in (0, "A", "a"):
        residues = HOMOMETRIC_A
    elif mode in (1, "B", "b"):
        residues = HOMOMETRIC_B
    else:
        raise ValueError("mode must be 0/'A' or 1/'B'")
    box = jnp.asarray(box, dtype=dtype)
    if box.shape != (2,) or bool(jnp.any(box <= 0)):
        raise ValueError("box must have shape (2,) with positive entries")
    lattice = jnp.asarray(residues, dtype=dtype) / HOMOMETRIC_MODULUS
    return lattice * box


def d4_matrices(*, dtype: jnp.dtype = jnp.float64) -> Array:
    """Return the eight orthogonal integer matrices of the square symmetry group."""
    matrices = np.asarray(
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
    return jnp.asarray(matrices, dtype=dtype)


def apply_d4(coordinates: Array, matrix: Array, box: Array) -> Array:
    """Apply a square-box D4 transformation and wrap the result."""
    coordinates = jnp.asarray(coordinates)
    matrix = jnp.asarray(matrix, dtype=coordinates.dtype)
    box = jnp.asarray(box, dtype=coordinates.dtype)
    if coordinates.ndim < 2 or coordinates.shape[-1] != 2:
        raise ValueError("coordinates must have shape (..., N, 2)")
    if matrix.shape != (2, 2):
        raise ValueError("matrix must have shape (2, 2)")
    if box.shape != (2,) or not bool(jnp.isclose(box[0], box[1])):
        raise ValueError("D4 transformations require a square box")
    return wrap_positions(coordinates @ matrix.T, box)


def _configuration_match(
    left: np.ndarray,
    right: np.ndarray,
    box: np.ndarray,
    tolerance: float,
) -> bool:
    for permutation in permutations(range(left.shape[0])):
        delta = np.mod(left - right[np.asarray(permutation)] + 0.5 * box, box) - 0.5 * box
        if np.max(np.abs(delta)) <= tolerance:
            return True
    return False


def are_torus_d4_congruent(
    left: Array,
    right: Array,
    box: Array,
    *,
    tolerance: float = 1e-10,
) -> bool:
    """Test congruence under D4, global torus translation, and permutation."""
    left_np = np.asarray(left, dtype=np.float64)
    right_np = np.asarray(right, dtype=np.float64)
    box_np = np.asarray(box, dtype=np.float64)
    if left_np.shape != right_np.shape or left_np.ndim != 2 or left_np.shape[1] != 2:
        raise ValueError("left and right must have the same shape (N, 2)")
    for matrix in np.asarray(d4_matrices(dtype=jnp.float64)):
        transformed = np.mod(left_np @ matrix.T, box_np)
        for left_index in range(transformed.shape[0]):
            for right_index in range(right_np.shape[0]):
                shift = right_np[right_index] - transformed[left_index]
                shifted = np.mod(transformed + shift, box_np)
                if _configuration_match(shifted, right_np, box_np, tolerance):
                    return True
    return False


def sorted_unordered_chord_distances(coordinates: Array, box: Array) -> Array:
    """Return the sorted unordered pair-distance multiset for one configuration."""
    coordinates = jnp.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[-1] != 2:
        raise ValueError("coordinates must have shape (N, 2)")
    distances = chord_distances(coordinates, box)
    row, column = jnp.triu_indices(coordinates.shape[0], k=1)
    return jnp.sort(distances[row, column])


def homometric_reference_descriptors(
    box: Array,
    pair_basis: PairBasis,
    angular_orders: Array,
    angular_neighbor_scale: float,
    *,
    dtype: jnp.dtype = jnp.float64,
) -> dict[str, Array]:
    """Return observed pair and held-out angular descriptors for both motifs."""
    motif_a = base_homometric_coordinates("A", box, dtype=dtype)
    motif_b = base_homometric_coordinates("B", box, dtype=dtype)
    ensemble_a = motif_a[None, ...]
    ensemble_b = motif_b[None, ...]
    return {
        "coordinates_a": motif_a,
        "coordinates_b": motif_b,
        "pair_a": ensemble_pair_moments(ensemble_a, box, pair_basis),
        "pair_b": ensemble_pair_moments(ensemble_b, box, pair_basis),
        "angular_a": ensemble_angular_cosine_moments(
            ensemble_a, box, angular_orders, angular_neighbor_scale
        ),
        "angular_b": ensemble_angular_cosine_moments(
            ensemble_b, box, angular_orders, angular_neighbor_scale
        ),
        "distances_a": sorted_unordered_chord_distances(motif_a, box),
        "distances_b": sorted_unordered_chord_distances(motif_b, box),
    }


def _augment_replica(
    key: Array,
    base: Array,
    box: Array,
    config: HomometricDatasetConfig,
) -> Array:
    symmetry_key, translation_key, permutation_key = jax.random.split(key, 3)
    coordinates = base
    if config.random_d4:
        symmetry_index = jax.random.randint(symmetry_key, (), 0, 8)
        coordinates = apply_d4(coordinates, d4_matrices(dtype=base.dtype)[symmetry_index], box)
    if config.random_translation:
        shift = jax.random.uniform(
            translation_key,
            (2,),
            minval=0.0,
            maxval=1.0,
            dtype=base.dtype,
        ) * box
        coordinates = wrap_positions(coordinates + shift, box)
    if config.random_permutation:
        particle_order = jax.random.permutation(permutation_key, base.shape[0])
        coordinates = coordinates[particle_order]
    return coordinates


def generate_homometric_dataset(
    config: HomometricDatasetConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Generate balanced exact-symmetry copies of the two homometric modes."""
    config.validate()
    if config.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    dtype = config.jax_dtype
    box = jnp.asarray(config.box, dtype=dtype)
    pair_basis = PairBasis.uniform(
        config.pair_basis_num,
        config.pair_basis_r_min,
        config.pair_basis_r_max,
        config.pair_basis_width,
        dtype=dtype,
    )
    angular_orders = jnp.asarray(config.angular_orders, dtype=dtype)
    physical = EnergyParameters(
        r0=config.physical_r0,
        kappa=config.physical_kappa,
        pair_strength=1.0,
        angular_strength=0.0,
        angular_target_cosine=-0.5,
        angular_neighbor_scale=config.angular_neighbor_scale,
    )

    coordinates_out: list[np.ndarray] = []
    pair_out: list[np.ndarray] = []
    angular_out: list[np.ndarray] = []
    energy_out: list[np.ndarray] = []
    minimum_out: list[np.ndarray] = []
    overlap_out: list[np.ndarray] = []
    labels_out: list[int] = []
    names_out: list[str] = []

    key = jax.random.PRNGKey(config.seed)
    for label, name in enumerate(HOMOMETRIC_NAMES):
        base = base_homometric_coordinates(label, box, dtype=dtype)
        for _ in range(config.samples_per_mode):
            key, sample_key = jax.random.split(key)
            replica_keys = jax.random.split(sample_key, config.num_replicas)
            ensemble = jax.vmap(
                lambda replica_key: _augment_replica(replica_key, base, box, config)
            )(replica_keys)
            pair = ensemble_pair_moments(ensemble, box, pair_basis)
            angular = ensemble_angular_cosine_moments(
                ensemble,
                box,
                angular_orders,
                config.angular_neighbor_scale,
            )
            energy = total_energy_per_configuration(ensemble, box, physical, "pair")
            diagnostics = pair_diagnostics(ensemble, box, config.overlap_threshold)
            coordinates_out.append(np.asarray(ensemble))
            pair_out.append(np.asarray(pair))
            angular_out.append(np.asarray(angular))
            energy_out.append(np.asarray(energy))
            minimum_out.append(np.asarray(diagnostics["minimum_pair_distance"]))
            overlap_out.append(np.asarray(diagnostics["overlap_fraction"]))
            labels_out.append(label)
            names_out.append(name)

    reference = homometric_reference_descriptors(
        box,
        pair_basis,
        angular_orders,
        config.angular_neighbor_scale,
        dtype=dtype,
    )
    arrays = {
        "coordinates": np.stack(coordinates_out),
        "pair_moments": np.stack(pair_out),
        "angular_moments": np.stack(angular_out),
        "energy_per_replica": np.stack(energy_out),
        "minimum_pair_distance": np.stack(minimum_out),
        "overlap_fraction": np.stack(overlap_out),
        "regime_label": np.asarray(labels_out, dtype=np.int32),
        "regime_name": np.asarray(names_out),
        "pair_basis_centers": np.asarray(pair_basis.centers),
        "pair_basis_widths": np.asarray(pair_basis.widths),
        "angular_orders": np.asarray(angular_orders),
        "box": np.asarray(box),
        "base_coordinates_a": np.asarray(reference["coordinates_a"]),
        "base_coordinates_b": np.asarray(reference["coordinates_b"]),
        "reference_pair_moments": np.asarray(reference["pair_a"]),
        "reference_angular_a": np.asarray(reference["angular_a"]),
        "reference_angular_b": np.asarray(reference["angular_b"]),
        "reference_pair_distances": np.asarray(reference["distances_a"]),
    }
    metadata = {
        "schema_version": 2,
        "benchmark": "exact-periodic-radial-homometric-z12x12",
        "seed": config.seed,
        "dtype": config.dtype,
        "modulus": HOMOMETRIC_MODULUS,
        "residues_a": [list(value) for value in HOMOMETRIC_A],
        "residues_b": [list(value) for value in HOMOMETRIC_B],
        "pair_distance_signature": np.asarray(reference["distances_a"]).tolist(),
        "num_particles": len(HOMOMETRIC_A),
        "num_replicas": config.num_replicas,
        "samples_per_mode": config.samples_per_mode,
        "angular_neighbor_scale": config.angular_neighbor_scale,
        "configuration": asdict(config),
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
    }
    return arrays, metadata


def classify_homometric_configurations(
    coordinates: Array,
    box: Array,
    angular_orders: Array,
    angular_neighbor_scale: float,
    reference_angular_a: Array,
    reference_angular_b: Array,
    *,
    angular_scale: Array | None = None,
) -> dict[str, Array]:
    """Classify configurations by an evaluation-only angular descriptor."""
    coordinates = jnp.asarray(coordinates)
    if coordinates.ndim < 3 or coordinates.shape[-1] != 2:
        raise ValueError("coordinates must have shape (..., N, 2)")
    leading_shape = coordinates.shape[:-2]
    flattened = coordinates.reshape((-1,) + coordinates.shape[-2:])
    descriptor = angular_cosine_moments(
        flattened,
        box,
        angular_orders,
        angular_neighbor_scale,
    )
    reference_a = jnp.asarray(reference_angular_a, dtype=descriptor.dtype)
    reference_b = jnp.asarray(reference_angular_b, dtype=descriptor.dtype)
    if reference_a.shape != descriptor.shape[-1:] or reference_b.shape != descriptor.shape[-1:]:
        raise ValueError("reference angular descriptors have incompatible shape")
    if angular_scale is None:
        scale = jnp.ones_like(reference_a)
    else:
        scale = jnp.maximum(jnp.asarray(angular_scale, dtype=descriptor.dtype), 1e-12)
    distance_a = jnp.linalg.norm((descriptor - reference_a) / scale, axis=-1)
    distance_b = jnp.linalg.norm((descriptor - reference_b) / scale, axis=-1)
    label = (distance_b < distance_a).astype(jnp.int32)
    minimum_distance = jnp.minimum(distance_a, distance_b)
    margin = jnp.abs(distance_a - distance_b)
    return {
        "angular_descriptor": descriptor.reshape(leading_shape + descriptor.shape[-1:]),
        "distance_a": distance_a.reshape(leading_shape),
        "distance_b": distance_b.reshape(leading_shape),
        "label": label.reshape(leading_shape),
        "minimum_distance": minimum_distance.reshape(leading_shape),
        "classification_margin": margin.reshape(leading_shape),
    }


def homometric_mode_metrics(
    classification: dict[str, Array],
    *,
    ambiguous_distance_threshold: float,
) -> dict[str, Array]:
    """Summarize mode coverage and descriptor fidelity from classifier output."""
    labels = jnp.asarray(classification["label"])
    minimum_distance = jnp.asarray(classification["minimum_distance"])
    margin = jnp.asarray(classification["classification_margin"])
    probability_b = jnp.mean(labels.astype(minimum_distance.dtype))
    probability_a = 1.0 - probability_b
    probabilities = jnp.stack((probability_a, probability_b))
    entropy = -jnp.sum(
        jnp.where(probabilities > 0, probabilities * jnp.log(probabilities), 0.0)
    ) / jnp.log(jnp.asarray(2.0, dtype=minimum_distance.dtype))
    return {
        "mode_a_fraction": probability_a,
        "mode_b_fraction": probability_b,
        "mode_total_variation_from_balanced": jnp.abs(probability_a - 0.5),
        "normalized_mode_entropy": entropy,
        "both_modes_present": jnp.asarray(
            (probability_a > 0.0) & (probability_b > 0.0), dtype=jnp.bool_
        ),
        "mean_reference_distance": jnp.mean(minimum_distance),
        "max_reference_distance": jnp.max(minimum_distance),
        "mean_classification_margin": jnp.mean(margin),
        "ambiguous_fraction": jnp.mean(
            minimum_distance
            > jnp.asarray(ambiguous_distance_threshold, dtype=minimum_distance.dtype)
        ),
    }


def validate_homometric_dataset(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    *,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Validate exact pair homometry, non-congruence, and hidden separation."""
    coordinates = np.asarray(arrays["coordinates"])
    pair_moments = np.asarray(arrays["pair_moments"])
    angular_moments = np.asarray(arrays["angular_moments"])
    labels = np.asarray(arrays["regime_label"], dtype=np.int32)
    box = np.asarray(arrays["box"])
    basis = PairBasis(
        centers=jnp.asarray(arrays["pair_basis_centers"]),
        widths=jnp.asarray(arrays["pair_basis_widths"]),
    )
    orders = jnp.asarray(arrays["angular_orders"])
    neighbor_scale = float(metadata["angular_neighbor_scale"])
    if coordinates.ndim != 4 or coordinates.shape[-2:] != (4, 2):
        raise ValueError("coordinates must have shape (S, M, 4, 2)")
    if pair_moments.shape[0] != coordinates.shape[0]:
        raise ValueError("pair_moments sample count does not match coordinates")
    if angular_moments.shape[0] != coordinates.shape[0]:
        raise ValueError("angular_moments sample count does not match coordinates")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("regime labels must contain exactly 0 and 1")

    base_a = jnp.asarray(arrays["base_coordinates_a"])
    base_b = jnp.asarray(arrays["base_coordinates_b"])
    reference = homometric_reference_descriptors(
        jnp.asarray(box), basis, orders, neighbor_scale, dtype=base_a.dtype
    )
    pair_reference_error = float(
        jnp.max(jnp.abs(reference["pair_a"] - reference["pair_b"]))
    )
    distance_multiset_error = float(
        jnp.max(jnp.abs(reference["distances_a"] - reference["distances_b"]))
    )
    non_congruent = not are_torus_d4_congruent(base_a, base_b, box, tolerance=tolerance)
    common_condition = np.mean(pair_moments, axis=0)
    pair_dataset_deviation = float(np.max(np.abs(pair_moments - common_condition)))
    mode_pair_centroid_error = float(
        np.max(
            np.abs(
                np.mean(pair_moments[labels == 0], axis=0)
                - np.mean(pair_moments[labels == 1], axis=0)
            )
        )
    )
    angular_centroid_a = np.mean(angular_moments[labels == 0], axis=0)
    angular_centroid_b = np.mean(angular_moments[labels == 1], axis=0)
    angular_separation = float(np.linalg.norm(angular_centroid_a - angular_centroid_b))

    recomputed_pair = np.asarray(
        jax.vmap(lambda sample: ensemble_pair_moments(sample, jnp.asarray(box), basis))(
            jnp.asarray(coordinates)
        )
    )
    recomputed_angular = np.asarray(
        jax.vmap(
            lambda sample: ensemble_angular_cosine_moments(
                sample, jnp.asarray(box), orders, neighbor_scale
            )
        )(jnp.asarray(coordinates))
    )
    pair_recompute_error = float(np.max(np.abs(recomputed_pair - pair_moments)))
    angular_recompute_error = float(np.max(np.abs(recomputed_angular - angular_moments)))
    balanced = int(np.sum(labels == 0)) == int(np.sum(labels == 1))
    passed = all(
        (
            pair_reference_error <= tolerance,
            distance_multiset_error <= tolerance,
            non_congruent,
            pair_dataset_deviation <= 10 * tolerance,
            mode_pair_centroid_error <= 10 * tolerance,
            angular_separation > 100 * tolerance,
            pair_recompute_error <= 10 * tolerance,
            angular_recompute_error <= 10 * tolerance,
            balanced,
        )
    )
    return {
        "passed": passed,
        "shape": list(coordinates.shape),
        "pair_distance_signature_a": np.asarray(reference["distances_a"]).tolist(),
        "pair_distance_signature_b": np.asarray(reference["distances_b"]).tolist(),
        "pair_reference_max_abs_error": pair_reference_error,
        "distance_multiset_max_abs_error": distance_multiset_error,
        "non_congruent_under_translation_d4_permutation": non_congruent,
        "dataset_pair_max_deviation_from_common_condition": pair_dataset_deviation,
        "mode_pair_centroid_max_abs_error": mode_pair_centroid_error,
        "heldout_angular_centroid_separation": angular_separation,
        "pair_recomputation_max_abs_error": pair_recompute_error,
        "angular_recomputation_max_abs_error": angular_recompute_error,
        "label_counts": {
            "A": int(np.sum(labels == 0)),
            "B": int(np.sum(labels == 1)),
        },
        "balanced": balanced,
    }
