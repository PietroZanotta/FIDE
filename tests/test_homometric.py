from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from manybody_completion.homometric import (
    HomometricDatasetConfig,
    apply_d4,
    are_torus_d4_congruent,
    base_homometric_coordinates,
    classify_homometric_configurations,
    d4_matrices,
    generate_homometric_dataset,
    homometric_mode_metrics,
    homometric_reference_descriptors,
    validate_homometric_dataset,
)
from manybody_completion.observables import PairBasis


def _config() -> HomometricDatasetConfig:
    return HomometricDatasetConfig(
        seed=17,
        dtype="float64",
        box=(1.0, 1.0),
        num_replicas=3,
        samples_per_mode=3,
        pair_basis_num=7,
        pair_basis_r_min=0.18,
        pair_basis_r_max=0.42,
        pair_basis_width=0.04,
        angular_orders=(1, 2, 3, 4, 5, 6, 7, 8),
        angular_neighbor_scale=0.2,
        overlap_threshold=0.22,
        physical_r0=0.27,
        physical_kappa=35.0,
    )


def test_exact_pair_distance_multiset_is_not_d4_congruent() -> None:
    box = jnp.ones((2,), dtype=jnp.float64)
    motif_a = base_homometric_coordinates("A", box)
    motif_b = base_homometric_coordinates("B", box)
    basis = PairBasis.uniform(9, 0.18, 0.42, 0.035, dtype=jnp.float64)
    orders = jnp.arange(1, 9, dtype=jnp.float64)
    reference = homometric_reference_descriptors(box, basis, orders, 0.2)
    np.testing.assert_allclose(
        reference["distances_a"], reference["distances_b"], atol=1e-14
    )
    assert not are_torus_d4_congruent(motif_a, motif_b, box)
    assert are_torus_d4_congruent(motif_a, motif_a, box)


def test_pair_descriptors_match_and_angular_descriptor_separates() -> None:
    box = jnp.ones((2,), dtype=jnp.float64)
    basis = PairBasis.uniform(11, 0.18, 0.42, 0.03, dtype=jnp.float64)
    orders = jnp.arange(1, 9, dtype=jnp.float64)
    reference = homometric_reference_descriptors(box, basis, orders, 0.2)
    np.testing.assert_allclose(reference["pair_a"], reference["pair_b"], atol=1e-14)
    assert float(jnp.linalg.norm(reference["angular_a"] - reference["angular_b"])) > 1.0


def test_d4_and_translation_preserve_mode_classification() -> None:
    box = jnp.ones((2,), dtype=jnp.float64)
    basis = PairBasis.uniform(8, 0.18, 0.42, 0.04, dtype=jnp.float64)
    orders = jnp.arange(1, 9, dtype=jnp.float64)
    reference = homometric_reference_descriptors(box, basis, orders, 0.2)
    motif = base_homometric_coordinates("A", box)
    for matrix in d4_matrices(dtype=jnp.float64):
        transformed = apply_d4(motif, matrix, box)
        transformed = jnp.mod(transformed + jnp.asarray([0.371, 0.823]), box)
        classification = classify_homometric_configurations(
            transformed[None],
            box,
            orders,
            0.2,
            reference["angular_a"],
            reference["angular_b"],
        )
        assert int(classification["label"][0]) == 0
        assert float(classification["distance_a"][0]) < 1e-10


def test_dataset_generation_is_deterministic_and_valid() -> None:
    first_arrays, first_metadata = generate_homometric_dataset(_config())
    second_arrays, second_metadata = generate_homometric_dataset(_config())
    assert first_metadata == second_metadata
    for name in first_arrays:
        np.testing.assert_array_equal(first_arrays[name], second_arrays[name])
    report = validate_homometric_dataset(first_arrays, first_metadata)
    assert report["passed"]
    assert report["label_counts"] == {"A": 3, "B": 3}
    assert report["dataset_pair_max_deviation_from_common_condition"] < 1e-12
    assert report["heldout_angular_centroid_separation"] > 1.0


def test_mode_classifier_recovers_exact_augmented_modes() -> None:
    arrays, metadata = generate_homometric_dataset(_config())
    coordinates = jnp.asarray(arrays["coordinates"])
    labels = jnp.asarray(arrays["regime_label"])
    classification = classify_homometric_configurations(
        coordinates,
        jnp.asarray(arrays["box"]),
        jnp.asarray(arrays["angular_orders"]),
        float(metadata["angular_neighbor_scale"]),
        jnp.asarray(arrays["reference_angular_a"]),
        jnp.asarray(arrays["reference_angular_b"]),
    )
    expected = jnp.broadcast_to(labels[:, None], classification["label"].shape)
    np.testing.assert_array_equal(classification["label"], expected)
    metrics = homometric_mode_metrics(
        classification,
        ambiguous_distance_threshold=0.1,
    )
    np.testing.assert_allclose(metrics["mode_a_fraction"], 0.5, atol=1e-15)
    np.testing.assert_allclose(metrics["mode_b_fraction"], 0.5, atol=1e-15)
    assert float(metrics["ambiguous_fraction"]) == 0.0
