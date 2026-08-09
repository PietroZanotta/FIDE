import jax
import jax.numpy as jnp
import numpy as np

from manybody_completion.geometry import chord_distances, translate
from manybody_completion.problem_instances import build_smoke_problem_instances
from manybody_completion.relaxation import RelaxationOptions, relax_proximal

jax.config.update("jax_enable_x64", True)


def _s1():
    arrays, _ = build_smoke_problem_instances()
    return arrays


def test_s1_relaxation_increases_separation_and_decreases_objective():
    arrays = _s1()
    x = jnp.asarray(arrays["s1_coordinates"])
    box = jnp.asarray(arrays["box"])
    options = RelaxationOptions(num_steps=128, step_size=2.5e-3, tolerance=1e-7)
    relaxed, diagnostics = relax_proximal(
        x,
        box,
        arrays["s1_r0"],
        arrays["s1_kappa"],
        arrays["s1_prox_strength"],
        options,
    )

    before = chord_distances(x, box)[0, 0, 1]
    after = chord_distances(relaxed, box)[0, 0, 1]
    assert float(after) > float(before)
    assert float(diagnostics["physical_energy_after"]) < float(
        diagnostics["physical_energy_before"]
    )
    assert float(diagnostics["proximal_objective_after"]) < float(
        diagnostics["proximal_objective_before"]
    )
    np.testing.assert_allclose(
        np.mean(np.asarray(relaxed), axis=1),
        np.mean(np.asarray(x), axis=1),
        atol=2e-10,
        rtol=2e-10,
    )
    assert bool(diagnostics["converged"])


def test_relaxation_is_translation_and_permutation_equivariant():
    arrays = _s1()
    x = jnp.asarray(arrays["s1_coordinates"])
    box = jnp.asarray(arrays["box"])
    options = RelaxationOptions(num_steps=48, step_size=1.5e-3, tolerance=1e-8)

    reference, reference_diag = relax_proximal(
        x, box, arrays["s1_r0"], arrays["s1_kappa"], arrays["s1_prox_strength"], options
    )
    shift = jnp.asarray([0.43, -0.37], dtype=x.dtype)
    shifted_input = translate(x, shift, box)
    shifted_output, shifted_diag = relax_proximal(
        shifted_input,
        box,
        arrays["s1_r0"],
        arrays["s1_kappa"],
        arrays["s1_prox_strength"],
        options,
    )
    np.testing.assert_allclose(
        np.asarray(shifted_output),
        np.asarray(translate(reference, shift, box)),
        atol=2e-10,
        rtol=2e-10,
    )

    permutation = jnp.asarray([1, 0])
    permuted_output, permuted_diag = relax_proximal(
        x[:, permutation, :],
        box,
        arrays["s1_r0"],
        arrays["s1_kappa"],
        arrays["s1_prox_strength"],
        options,
    )
    np.testing.assert_allclose(
        np.asarray(permuted_output),
        np.asarray(reference[:, permutation, :]),
        atol=2e-10,
        rtol=2e-10,
    )
    for key in (
        "physical_energy_after",
        "proximal_objective_after",
        "prox_displacement",
        "minimum_pair_distance_after",
    ):
        np.testing.assert_allclose(reference_diag[key], shifted_diag[key], atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(reference_diag[key], permuted_diag[key], atol=2e-10, rtol=2e-10)


def test_unrolled_relaxation_directional_derivative_matches_finite_difference():
    arrays = _s1()
    x = jnp.asarray(arrays["s1_coordinates"])
    box = jnp.asarray(arrays["box"])
    options = RelaxationOptions(
        num_steps=20,
        step_size=1e-3,
        tolerance=0.0,
        max_update_norm=0.02,
        line_search_steps=8,
    )
    direction = jax.random.normal(jax.random.PRNGKey(11), x.shape, dtype=x.dtype)
    direction = direction / jnp.linalg.norm(direction)
    cotangent = jax.random.normal(jax.random.PRNGKey(12), x.shape, dtype=x.dtype)

    def scalar_probe(value):
        relaxed, _ = relax_proximal(
            value,
            box,
            arrays["s1_r0"],
            arrays["s1_kappa"],
            arrays["s1_prox_strength"],
            options,
        )
        return jnp.vdot(cotangent, relaxed)

    autodiff = jax.jvp(scalar_probe, (x,), (direction,))[1]
    errors = []
    for epsilon in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
        finite_difference = (
            scalar_probe(x + epsilon * direction) - scalar_probe(x - epsilon * direction)
        ) / (2.0 * epsilon)
        errors.append(float(jnp.abs(finite_difference - autodiff)))

    # A sweep is required: the truncation error should decrease before roundoff dominates.
    assert min(errors) < 2e-7
    assert errors[-1] < errors[0] * 1e-4
