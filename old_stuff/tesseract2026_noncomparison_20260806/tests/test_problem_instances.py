import jax.numpy as jnp
import numpy as np

from manybody_completion.geometry import chord_distances
from manybody_completion.observables import PairBasis, ensemble_pair_moments
from manybody_completion.problem_instances import build_smoke_problem_instances


def test_s1_s2_s3_problem_contracts():
    arrays, metadata = build_smoke_problem_instances(seed=3)
    box = jnp.asarray(arrays["box"])

    s1 = jnp.asarray(arrays["s1_coordinates"])
    s1_distance = chord_distances(s1, box)[0, 0, 1]
    assert float(s1_distance) < float(arrays["s1_r0"])

    s2_basis = PairBasis(
        jnp.asarray(arrays["s2_basis_centers"]),
        jnp.asarray(arrays["s2_basis_widths"]),
    )
    reconstructed_target = ensemble_pair_moments(
        jnp.asarray(arrays["s2_reference_coordinates"]), box, s2_basis
    )
    np.testing.assert_allclose(reconstructed_target, arrays["s2_target_moments"], atol=1e-12)
    assert arrays["s2_relaxed_coordinates"].shape == (8, 4, 2)

    s3_basis = PairBasis(
        jnp.asarray(arrays["s3_basis_centers"]),
        jnp.asarray(arrays["s3_basis_widths"]),
    )
    x_star = jnp.mod(
        jnp.asarray(arrays["s3_base_coordinates"])
        + jnp.asarray(arrays["s3_a_star"])
        * jnp.asarray(arrays["s3_latent_displacements"]),
        box,
    )
    reconstructed_s3_target = ensemble_pair_moments(x_star, box, s3_basis)
    np.testing.assert_allclose(reconstructed_s3_target, arrays["s3_target_moments"], atol=1e-12)
    assert metadata["instances"]["S3"]["generator"] == "wrap(X_base + a * Z)"
