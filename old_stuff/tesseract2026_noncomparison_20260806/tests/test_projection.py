import jax
import jax.numpy as jnp
import numpy as np

from manybody_completion.geometry import translate
from manybody_completion.observables import PairBasis, ensemble_pair_moments
from manybody_completion.problem_instances import build_smoke_problem_instances
from manybody_completion.projection import ProjectionOptions, project_ensemble_moments

jax.config.update("jax_enable_x64", True)


def _s2():
    arrays, _ = build_smoke_problem_instances()
    coordinates = jnp.asarray(arrays["s2_relaxed_coordinates"])
    box = jnp.asarray(arrays["box"])
    target = jnp.asarray(arrays["s2_target_moments"])
    basis = PairBasis(
        centers=jnp.asarray(arrays["s2_basis_centers"]),
        widths=jnp.asarray(arrays["s2_basis_widths"]),
    )
    scales = jnp.ones_like(target)
    mask = jnp.ones_like(target)
    return coordinates, box, target, basis, scales, mask


def _options(**overrides):
    values = {
        "num_steps": 12,
        "tolerance": 1e-10,
        "ridge": 1e-8,
        "svd_rcond": 1e-7,
        "damping": 1.0,
        "max_step_norm": 0.05,
        "max_correction_norm": 0.25,
        "line_search_steps": 10,
        "line_search_shrink": 0.5,
        "sufficient_decrease": 0.0,
    }
    values.update(overrides)
    return ProjectionOptions(**values)


def test_s2_projection_reduces_residual_by_at_least_two_orders():
    coordinates, box, target, basis, scales, mask = _s2()
    projected, diagnostics = project_ensemble_moments(
        coordinates, target, box, basis, scales, mask, _options()
    )

    before = float(diagnostics["constraint_residual_before"])
    after = float(diagnostics["constraint_residual"])
    assert after <= before * 1e-2
    assert float(diagnostics["correction_norm"]) < 0.02
    assert bool(diagnostics["converged"])
    assert not bool(diagnostics["rank_deficient"])
    np.testing.assert_allclose(
        np.asarray(ensemble_pair_moments(projected, box, basis)),
        np.asarray(target),
        atol=2e-10,
        rtol=2e-10,
    )


def test_projection_identity_has_zero_correction_and_dual():
    coordinates, box, _, basis, scales, mask = _s2()
    target = ensemble_pair_moments(coordinates, box, basis)
    projected, diagnostics = project_ensemble_moments(
        coordinates, target, box, basis, scales, mask, _options()
    )

    np.testing.assert_allclose(projected, coordinates, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(diagnostics["dual_variables"], 0.0, atol=2e-12, rtol=0.0)
    assert float(diagnostics["correction_norm"]) < 2e-12
    assert float(diagnostics["constraint_residual"]) < 2e-12
    assert bool(diagnostics["converged"])


def test_projection_is_translation_and_permutation_equivariant():
    coordinates, box, target, basis, scales, mask = _s2()
    options = _options(num_steps=6)
    reference, reference_diagnostics = project_ensemble_moments(
        coordinates, target, box, basis, scales, mask, options
    )

    shift = jnp.asarray([0.31, -0.27], dtype=coordinates.dtype)
    translated_input = translate(coordinates, shift, box)
    translated, translated_diagnostics = project_ensemble_moments(
        translated_input, target, box, basis, scales, mask, options
    )
    np.testing.assert_allclose(
        translated,
        translate(reference, shift, box),
        atol=5e-10,
        rtol=5e-10,
    )

    permutation = jnp.asarray([2, 0, 3, 1])
    permuted, permuted_diagnostics = project_ensemble_moments(
        coordinates[:, permutation, :], target, box, basis, scales, mask, options
    )
    np.testing.assert_allclose(
        permuted,
        reference[:, permutation, :],
        atol=5e-10,
        rtol=5e-10,
    )

    for key in (
        "constraint_residual",
        "correction_norm",
        "largest_singular_value",
        "smallest_singular_value",
    ):
        np.testing.assert_allclose(
            reference_diagnostics[key], translated_diagnostics[key], atol=5e-10, rtol=5e-10
        )
        np.testing.assert_allclose(
            reference_diagnostics[key], permuted_diagnostics[key], atol=5e-10, rtol=5e-10
        )


def test_projection_directional_derivatives_match_finite_differences():
    coordinates, box, target, basis, scales, mask = _s2()
    options = _options(num_steps=3, tolerance=0.0)
    probe = jax.random.normal(jax.random.PRNGKey(31), coordinates.shape, dtype=coordinates.dtype)
    coordinate_direction = jax.random.normal(
        jax.random.PRNGKey(32), coordinates.shape, dtype=coordinates.dtype
    )
    coordinate_direction = coordinate_direction / jnp.linalg.norm(coordinate_direction)
    target_direction = jax.random.normal(jax.random.PRNGKey(33), target.shape, dtype=target.dtype)
    target_direction = target_direction / jnp.linalg.norm(target_direction)

    def coordinate_probe(value):
        projected, _ = project_ensemble_moments(
            value, target, box, basis, scales, mask, options
        )
        return jnp.vdot(probe, projected)

    def target_probe(value):
        projected, _ = project_ensemble_moments(
            coordinates, value, box, basis, scales, mask, options
        )
        return jnp.vdot(probe, projected)

    for function, value, direction in (
        (coordinate_probe, coordinates, coordinate_direction),
        (target_probe, target, target_direction),
    ):
        autodiff = jax.jvp(function, (value,), (direction,))[1]
        errors = []
        for epsilon in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
            finite_difference = (
                function(value + epsilon * direction)
                - function(value - epsilon * direction)
            ) / (2.0 * epsilon)
            errors.append(float(jnp.abs(finite_difference - autodiff)))
        assert min(errors) < 2e-7
        assert errors[-1] < errors[0] * 1e-4


def test_projection_jits_with_closed_over_validation_arrays():
    coordinates, box, target, basis, scales, mask = _s2()
    options = _options(num_steps=3, tolerance=0.0)

    @jax.jit
    def scalar_probe(offset):
        projected, _ = project_ensemble_moments(
            coordinates,
            target + offset,
            box,
            basis,
            scales,
            mask,
            options,
        )
        return jnp.sum(projected)

    value, gradient = jax.value_and_grad(scalar_probe)(jnp.zeros_like(target))
    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(gradient))


def test_rank_deficiency_is_reported_and_explicit_pruning_stays_finite():
    coordinates, box, _, original_basis, _, _ = _s2()
    duplicate_basis = PairBasis(
        centers=jnp.repeat(original_basis.centers, 2),
        widths=jnp.repeat(original_basis.widths, 2),
    )
    raw_target = ensemble_pair_moments(coordinates, box, duplicate_basis)
    perturbed_target = raw_target + jnp.asarray([2e-3, 2e-3], dtype=coordinates.dtype)
    scales = jnp.ones((2,), dtype=coordinates.dtype)

    unpruned, unpruned_diagnostics = project_ensemble_moments(
        coordinates,
        perturbed_target,
        box,
        duplicate_basis,
        scales,
        jnp.ones((2,), dtype=coordinates.dtype),
        _options(num_steps=4),
    )
    assert bool(unpruned_diagnostics["rank_deficient"])
    assert int(unpruned_diagnostics["effective_rank"]) == 1
    assert np.all(np.isfinite(np.asarray(unpruned)))
    assert np.all(np.isfinite(np.asarray(unpruned_diagnostics["dual_variables"])))

    pruned, pruned_diagnostics = project_ensemble_moments(
        coordinates,
        perturbed_target,
        box,
        duplicate_basis,
        scales,
        jnp.asarray([1.0, 0.0], dtype=coordinates.dtype),
        _options(num_steps=4),
    )
    assert not bool(pruned_diagnostics["rank_deficient"])
    assert int(pruned_diagnostics["active_constraints"]) == 1
    assert np.all(np.isfinite(np.asarray(pruned)))
    assert float(pruned_diagnostics["constraint_residual"]) < 1e-8
