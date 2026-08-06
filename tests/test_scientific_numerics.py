import jax
import jax.numpy as jnp
import numpy as np

from manybody_completion.classical_baselines import _pair_potential_energy_batch
from manybody_completion.fast_training import clip_by_global_norm, tree_global_norm
from manybody_completion.observables import PairBasis, ensemble_pair_moments
from manybody_completion.solvers import ProjectionOptions, project_ensemble


def test_ibi_replica_energy_is_extensive_and_local_move_is_not_rescaled():
    box = jnp.asarray([1.0, 1.0])
    centers = jnp.linspace(0.03, 0.45, 12)
    potential = jnp.linspace(3.0, 0.0, 12) ** 2
    replica = jnp.asarray(
        [[[0.10, 0.12], [0.31, 0.18], [0.73, 0.61], [0.82, 0.91]]]
    )
    one_replica = replica[None, ...]
    two_replicas = jnp.stack((replica[0], replica[0]), axis=0)[None, ...]

    energy_one = _pair_potential_energy_batch(
        one_replica, box, centers, potential
    )
    energy_two = _pair_potential_energy_batch(
        two_replicas, box, centers, potential
    )
    np.testing.assert_allclose(energy_two, 2.0 * energy_one, rtol=1e-6)

    moved_one = one_replica.at[0, 0, 0].set(jnp.asarray([0.16, 0.09]))
    moved_two = two_replicas.at[0, 0, 0].set(jnp.asarray([0.16, 0.09]))
    delta_one = _pair_potential_energy_batch(
        moved_one, box, centers, potential
    ) - energy_one
    delta_two = _pair_potential_energy_batch(
        moved_two, box, centers, potential
    ) - energy_two
    np.testing.assert_allclose(delta_two, delta_one, rtol=1e-6, atol=1e-7)


def test_scientific_projection_is_residual_monotone_and_stops_at_tolerance():
    dtype = jnp.float64
    box = jnp.asarray([1.0, 1.0], dtype=dtype)
    basis = PairBasis.uniform(4, 0.12, 0.44, 0.07, dtype=dtype)
    coordinates = jax.random.uniform(
        jax.random.PRNGKey(11), (3, 5, 2), dtype=dtype
    )
    reference = jax.random.uniform(
        jax.random.PRNGKey(12), (3, 5, 2), dtype=dtype
    )
    target = ensemble_pair_moments(reference, box, basis)
    scales = jnp.ones_like(target)

    residuals = []
    for steps in (1, 4, 12):
        _, diagnostics = project_ensemble(
            coordinates,
            target,
            box,
            basis,
            scales,
            ProjectionOptions(
                num_steps=steps,
                ridge=1e-3,
                max_particle_step=0.05,
                tolerance=1e-6,
            ),
        )
        residuals.append(float(diagnostics["constraint_residual"]))
        assert residuals[-1] <= float(
            diagnostics["constraint_residual_before"]
        ) + 1e-12

    assert residuals[1] <= residuals[0] + 1e-12
    assert residuals[2] <= residuals[1] + 1e-12

    identity_target = ensemble_pair_moments(coordinates, box, basis)
    projected, diagnostics = project_ensemble(
        coordinates,
        identity_target,
        box,
        basis,
        scales,
        ProjectionOptions(num_steps=4, tolerance=1e-10),
    )
    np.testing.assert_allclose(projected, coordinates, atol=1e-12, rtol=0.0)
    assert bool(diagnostics["converged"])
    assert int(diagnostics["iterations"]) == 0


def test_global_norm_and_clipping_handle_huge_and_nonfinite_gradients():
    huge = {
        "a": jnp.asarray([1e30, -1e30], dtype=jnp.float32),
        "b": jnp.asarray([2e30], dtype=jnp.float32),
    }
    norm = tree_global_norm(huge)
    assert bool(jnp.isfinite(norm))
    np.testing.assert_allclose(float(norm / 1e30), np.sqrt(6.0), rtol=2e-6)

    clipped, original_norm = clip_by_global_norm(huge, 2.0)
    assert bool(jnp.isfinite(original_norm))
    assert float(tree_global_norm(clipped)) <= 2.0 + 1e-5

    invalid = {"a": jnp.asarray([jnp.inf, 1.0], dtype=jnp.float32)}
    clipped_invalid, invalid_norm = clip_by_global_norm(invalid, 2.0)
    assert bool(jnp.isinf(invalid_norm))
    np.testing.assert_array_equal(clipped_invalid["a"], jnp.zeros((2,)))
