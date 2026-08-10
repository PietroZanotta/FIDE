from __future__ import annotations

import unittest
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grayscott.calibration import calibrate_shared_target
from experiments.grayscott.feasibility import (
    calibrate_iprojection_instrumented,
    solve_common_hull_lp,
    solve_maximum_entropy_common_target,
    solve_maximum_minimum_weight_lp,
    solve_target_hull_lp,
)
from experiments.grayscott.field_transport import (
    field_l2_cost,
    field_jphi_times_velocity,
    geometric_l2_transport_coupling,
    independent_coupling,
    init_periodic_reference_cnn,
    linear_field_interpolant,
    maximal_same_index_coupling,
    noisy_field_interpolant,
    periodic_reference_cnn,
    smooth_hidden_observables,
    standardized_noise_bank,
    weighted_field_tangent_velocity,
)
from experiments.grayscott.morphology_metrics import (
    field_metrics,
    periodic_component_count,
    periodic_euler_characteristic,
    periodic_interface_length,
)
from experiments.grayscott.observables import ShellDefinition, field_observables
from experiments.grayscott.simulator import (
    GrayScottParameters,
    generate_initial_conditions,
    periodic_laplacian,
    simulate,
)
from experiments.grayscott.phase3_reference_design import _second_moment_identity
from experiments.grayscott.phase2_continuation import _json_default
from experiments.grayscott.phase3_reference_quality import (
    bridge_target_consistency,
    heun_rollout_snapshots,
    init_residual_reference_cnn,
    residual_reference_cnn,
    sample_frozen_bridge,
    validation_bank_action,
    weighted_mmd2_four_weight,
)

jax.config.update("jax_enable_x64", True)


class GrayScottSimulatorTests(unittest.TestCase):
    def test_periodic_laplacian_discrete_eigenmode(self):
        size = 12
        y, x = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        mode = np.cos(2.0 * np.pi * x / size) + 0.4 * np.sin(4.0 * np.pi * y / size)
        expected = (
            (2.0 * np.cos(2.0 * np.pi / size) - 2.0) * np.cos(2.0 * np.pi * x / size)
            + 0.4 * (2.0 * np.cos(4.0 * np.pi / size) - 2.0) * np.sin(4.0 * np.pi * y / size)
        )
        actual = np.asarray(periodic_laplacian(jnp.asarray(mode)))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_fixed_seed_and_simulator_are_deterministic(self):
        first = generate_initial_conditions([17, 18], height=12, width=12)
        second = generate_initial_conditions([17, 18], height=12, width=12)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        params = GrayScottParameters(0.035, 0.060)
        out_a = simulate(first[0], first[1], params, dt=0.5, physical_time=10.0)
        out_b = simulate(second[0], second[1], params, dt=0.5, physical_time=10.0)
        np.testing.assert_array_equal(np.asarray(out_a[0]), np.asarray(out_b[0]))
        np.testing.assert_array_equal(np.asarray(out_a[1]), np.asarray(out_b[1]))

    def test_physical_timestep_convergence(self):
        u, v, _ = generate_initial_conditions([91], height=12, width=12)
        params = GrayScottParameters(0.035, 0.060)
        _, coarse = simulate(u, v, params, dt=0.5, physical_time=20.0)
        _, medium = simulate(u, v, params, dt=0.25, physical_time=20.0)
        _, fine = simulate(u, v, params, dt=0.125, physical_time=20.0)
        coarse_error = np.linalg.norm(np.asarray(coarse - fine))
        medium_error = np.linalg.norm(np.asarray(medium - fine))
        self.assertLess(medium_error, coarse_error)


class ObservableTests(unittest.TestCase):
    def test_values_and_gradients_against_finite_differences(self):
        rng = np.random.default_rng(3)
        field = rng.normal(size=(1, 1, 8, 8)).astype(np.float64)
        shells = ShellDefinition((0.10, 0.20), (0.07, 0.08))

        def one(x):
            return field_observables(
                x[None], shells, ("mean", "second_moment", "shell_1", "shell_2")
            )[0]

        jacobian = np.asarray(jax.jacrev(one)(jnp.asarray(field[0])))
        epsilon = 1e-5
        for y, x in ((0, 0), (3, 5), (7, 2)):
            plus, minus = field.copy(), field.copy()
            plus[0, 0, y, x] += epsilon
            minus[0, 0, y, x] -= epsilon
            finite = (np.asarray(one(jnp.asarray(plus[0]))) - np.asarray(one(jnp.asarray(minus[0])))) / (2 * epsilon)
            np.testing.assert_allclose(jacobian[:, 0, y, x], finite, rtol=2e-6, atol=2e-8)

    def test_parseval_normalization(self):
        rng = np.random.default_rng(7)
        field = rng.normal(size=(5, 1, 10, 12))
        spectrum = np.fft.fft2(field[:, 0], norm="ortho")
        spectral_second_moment = np.sum(np.abs(spectrum) ** 2, axis=(-2, -1)) / (10 * 12)
        np.testing.assert_allclose(spectral_second_moment, np.mean(field * field, axis=(1, 2, 3)), rtol=1e-13)

    def test_affine_standardization_does_not_change_fiber(self):
        rng = np.random.default_rng(23)
        values = rng.normal(size=(40, 4))
        weights = rng.random(40)
        weights /= weights.sum()
        target = weights @ values
        center, scale = values.mean(0), values.std(0) + 0.3
        transformed_target = (target - center) / scale
        np.testing.assert_allclose(weights @ ((values - center) / scale), transformed_target)

    def test_field_jvp_matches_explicit_small_grid_jacobian(self):
        rng = np.random.default_rng(44)
        fields = jnp.asarray(rng.normal(size=(3, 1, 5, 5)))
        velocity = jnp.asarray(rng.normal(size=fields.shape))
        shells = ShellDefinition((0.12, 0.25), (0.08, 0.08))

        def batched(value):
            return field_observables(value, shells, ("mean", "second_moment", "shell_1"))

        jvp = field_jphi_times_velocity(fields, velocity, batched)
        explicit = []
        for field, direction in zip(fields, velocity):
            jacobian = jax.jacrev(lambda x: batched(x[None])[0])(field)
            explicit.append(jnp.einsum("rchw,chw->r", jacobian, direction))
        np.testing.assert_allclose(jvp, jnp.stack(explicit), rtol=1e-10, atol=1e-10)


class MorphologyTests(unittest.TestCase):
    def test_periodic_components_and_euler(self):
        mask = np.zeros((6, 6), dtype=bool)
        mask[2, 0] = True
        mask[2, -1] = True
        self.assertEqual(periodic_component_count(mask), 1)
        self.assertEqual(periodic_euler_characteristic(mask), 1)
        self.assertEqual(periodic_euler_characteristic(np.ones((6, 6), dtype=bool)), 0)
        self.assertEqual(periodic_interface_length(np.ones((6, 6), dtype=bool)), 0.0)

    def test_spot_count_is_invariant_to_field_polarity(self):
        field = np.zeros((8, 8), dtype=np.float64)
        field[1, 1] = field[5, 5] = 1.0
        bright = field_metrics(field, 0.5)
        dark = field_metrics(1.0 - field, 0.5)
        self.assertEqual(bright["minority_component_count"], 2.0)
        self.assertEqual(dark["minority_component_count"], 2.0)


class CalibrationAndTangentTests(unittest.TestCase):
    def test_exact_common_hull_lp_and_centrality(self):
        rng = np.random.default_rng(101)
        common = rng.normal(size=(80, 3))
        minus = common + rng.normal(scale=0.03, size=common.shape)
        plus = common + rng.normal(scale=0.03, size=common.shape)
        # Shared anchor points guarantee a nonempty, full-dimensional overlap.
        anchors = rng.normal(scale=0.2, size=(8, 3))
        minus = np.concatenate([minus, anchors])
        plus = np.concatenate([plus, anchors])
        feasible = solve_common_hull_lp(minus, plus)
        central = solve_maximum_minimum_weight_lp(minus, plus)
        entropy = solve_maximum_entropy_common_target(minus, plus)
        self.assertTrue(feasible["success"])
        self.assertLess(feasible["maximum_equality_residual"], 1e-8)
        self.assertTrue(central["success"])
        self.assertGreater(central["maximum_minimum_weight"], 0.0)
        self.assertTrue(entropy["converged"])
        self.assertLess(entropy["maximum_equality_residual"], 1e-9)
        target_result = solve_target_hull_lp(minus, feasible["target"])
        self.assertTrue(target_result["success"])
        self.assertLess(target_result["maximum_equality_residual"], 1e-8)

    def test_instrumented_iprojection_recovers_feasible_positive_target(self):
        rng = np.random.default_rng(202)
        features = rng.normal(size=(240, 4))
        known_weights = np.exp(rng.normal(scale=0.35, size=len(features)))
        known_weights /= known_weights.sum()
        target = known_weights @ features
        result = calibrate_iprojection_instrumented(
            features, target, tolerance=1e-10, max_iterations=300
        )
        self.assertTrue(result["converged"], result["convergence_reason"])
        self.assertLess(result["maximum_absolute_standardized_residual"], 1e-8)
        self.assertLess(result["residual_identity_maximum_difference"], 1e-12)
        np.testing.assert_allclose(result["moments"] - target, result["reported_residual"], atol=1e-14)

    def test_design_training_and_evaluation_seeds_are_disjoint(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "configs" / "expC_grayscott_design.yaml").read_text())
        seeds = config["seeds"]
        design = set(range(
            seeds["design_initial_condition_start"],
            seeds["design_initial_condition_start"] + seeds["design_initial_condition_count"],
        ))
        training = set(seeds["training_model"])
        evaluation = set(seeds["final_evaluation_bank"])
        self.assertFalse(design & training)
        self.assertFalse(design & evaluation)
        self.assertFalse(training & evaluation)

    def test_shared_endpoint_projection_and_ess(self):
        rng = np.random.default_rng(11)
        common = rng.normal(size=(128, 4))
        minus = common + np.array([0.15, -0.10, 0.05, 0.0])
        plus = common + np.array([-0.15, 0.10, -0.05, 0.0])
        result = calibrate_shared_target(minus, plus)
        self.assertLess(result["minus"]["max_abs_residual"], 1e-8)
        self.assertLess(result["plus"]["max_abs_residual"], 1e-8)
        self.assertGreater(result["minus"]["ess_fraction"], 0.2)
        self.assertGreater(result["plus"]["ess_fraction"], 0.2)
        self.assertAlmostEqual(float(result["minus"]["weights"].sum()), 1.0, places=12)

    def test_field_tangent_correction_zeroes_measured_rate(self):
        rng = np.random.default_rng(5)
        fields = jnp.asarray(rng.normal(size=(24, 1, 6, 6)))
        velocity = jnp.asarray(rng.normal(size=fields.shape))
        shells = ShellDefinition((0.10, 0.22), (0.08, 0.09))

        def phi_single(field):
            return field_observables(
                field[None], shells, ("mean", "second_moment", "shell_1", "shell_2")
            )[0]

        jacobian = jax.vmap(jax.jacrev(phi_single))(fields).reshape((len(fields), 4, -1))
        flat_velocity = velocity.reshape((len(fields), -1))
        rates = jnp.einsum("brd,bd->br", jacobian, flat_velocity)
        gram = jnp.einsum("brd,bsd->rs", jacobian, jacobian) / len(fields)
        coefficient = jnp.linalg.pinv(gram, rtol=1e-12) @ rates.mean(axis=0)
        corrected = flat_velocity - jnp.einsum("brd,r->bd", jacobian, coefficient)
        corrected_rate = jnp.einsum("brd,bd->br", jacobian, corrected).mean(axis=0)
        self.assertLess(float(jnp.max(jnp.abs(corrected_rate))), 1e-9)

    def test_weighted_field_tangent_contractions(self):
        rng = np.random.default_rng(54)
        fields = jnp.asarray(rng.normal(size=(18, 1, 5, 5)))
        velocity = jnp.asarray(rng.normal(size=fields.shape))
        weights = jnp.asarray(rng.random(len(fields)))
        weights /= weights.sum()
        shells = ShellDefinition((0.12,), (0.08,))

        def observable(field):
            return field_observables(field[None], shells, ("mean", "second_moment", "shell_1"))[0]

        _, diagnostics = weighted_field_tangent_velocity(fields, velocity, weights, observable)
        self.assertLess(float(jnp.max(jnp.abs(diagnostics["corrected_rate_residual"]))), 1e-9)


class FieldInfrastructureTests(unittest.TestCase):
    def test_maximal_coupling_has_exact_marginals(self):
        minus = np.array([0.10, 0.20, 0.30, 0.40])
        plus = np.array([0.25, 0.15, 0.35, 0.25])
        coupling = maximal_same_index_coupling(minus, plus)
        np.testing.assert_allclose(coupling.sum(axis=1), minus, atol=1e-14)
        np.testing.assert_allclose(coupling.sum(axis=0), plus, atol=1e-14)
        self.assertAlmostEqual(float(np.trace(coupling)), float(np.minimum(minus, plus).sum()), places=14)

    def test_independent_and_geometric_couplings_have_exact_marginals(self):
        minus_weights = np.array([0.10, 0.20, 0.30, 0.40])
        plus_weights = np.array([0.25, 0.15, 0.35, 0.25])
        minus_fields = np.array([0.0, 0.2, 0.7, 1.0])[:, None, None, None]
        plus_fields = np.array([0.1, 0.4, 0.8, 1.2])[:, None, None, None]
        independent = independent_coupling(minus_weights, plus_weights)
        geometric, diagnostics = geometric_l2_transport_coupling(
            minus_fields, plus_fields, minus_weights, plus_weights
        )
        for coupling in (independent, geometric):
            np.testing.assert_allclose(coupling.sum(axis=1), minus_weights, atol=1e-10)
            np.testing.assert_allclose(coupling.sum(axis=0), plus_weights, atol=1e-10)
        cost = field_l2_cost(minus_fields, plus_fields)
        self.assertLessEqual(float(np.sum(geometric * cost)), float(np.sum(independent * cost)) + 1e-12)
        self.assertLess(diagnostics["maximum_marginal_residual"], 1e-9)

    def test_noisy_interpolant_has_fixed_endpoints_and_repository_envelope(self):
        minus = np.zeros((3, 1, 5, 5))
        plus = np.ones_like(minus)
        noise = standardized_noise_bank(3, (1, 5, 5), 71, dtype=np.float64)
        np.testing.assert_allclose(noise.mean(axis=(1, 2, 3)), 0.0, atol=1e-15)
        np.testing.assert_allclose(np.mean(noise * noise, axis=(1, 2, 3)), 1.0, atol=1e-14)
        at_zero, _ = noisy_field_interpolant(minus, plus, noise, 0.0, 0.07)
        at_one, _ = noisy_field_interpolant(minus, plus, noise, 1.0, 0.07)
        np.testing.assert_allclose(at_zero, minus, atol=1e-14)
        np.testing.assert_allclose(at_one, plus, atol=1e-14)

    def test_linear_bridge_second_moment_identity_and_reparameterization(self):
        rng = np.random.default_rng(83)
        minus = rng.normal(size=(5, 1, 4, 4))
        plus = rng.normal(size=(6, 1, 4, 4))
        coupling = independent_coupling(np.full(5, 0.2), np.full(6, 1.0 / 6.0))
        # The identity uses the common endpoint second moment. Rescale plus so
        # its empirical second moment agrees with minus exactly.
        plus *= np.sqrt(np.mean(minus * minus) / np.mean(plus * plus))
        target = float(np.mean(minus * minus))
        rows = _second_moment_identity(minus, plus, coupling, target, [0.2, 0.5, 0.8])
        self.assertEqual(len(rows), 6)
        self.assertLess(max(row["absolute_identity_error"] for row in rows), 1e-12)
        self.assertTrue(all(row["second_moment_deficit"] > 0.0 for row in rows))

    def test_bridge_target_matches_finite_difference_with_shared_noise(self):
        rng = np.random.default_rng(91)
        minus = rng.normal(size=(6, 1, 5, 5)).astype(np.float32)
        plus = rng.normal(size=(6, 1, 5, 5)).astype(np.float32)
        noise = standardized_noise_bank(6, (1, 5, 5), 92)
        times = np.linspace(0.1, 0.9, 6, dtype=np.float32)
        result = bridge_target_consistency(minus, plus, noise, times, epsilon=2e-3)
        self.assertLess(result["relative_finite_difference_error"], 2e-4)
        self.assertLess(result["maximum_analytic_formula_error"], 2e-6)
        self.assertTrue(result["same_noise_used_for_state_and_target"])

    def test_frozen_bridge_sampler_has_correct_empirical_coupling_marginals(self):
        minus = np.arange(4, dtype=np.float32)[:, None, None, None]
        plus = np.arange(3, dtype=np.float32)[:, None, None, None]
        mw = np.array([0.1, 0.2, 0.3, 0.4])
        pw = np.array([0.25, 0.25, 0.5])
        coupling = independent_coupling(mw, pw)
        _, _, details = sample_frozen_bridge(
            np.random.default_rng(93), minus, plus, coupling,
            np.full(100000, 0.4, dtype=np.float32),
        )
        observed_minus = np.bincount(details["minus_indices"], minlength=4) / 100000
        observed_plus = np.bincount(details["plus_indices"], minlength=3) / 100000
        np.testing.assert_allclose(observed_minus, mw, atol=0.005)
        np.testing.assert_allclose(observed_plus, pw, atol=0.005)

    def test_reference_rollout_metric_compares_to_raw_si_not_fiber_target(self):
        rng = np.random.default_rng(94)
        direct = rng.normal(loc=2.0, size=(64, 4))
        identical = direct.copy()
        fiber_target = np.zeros(4)
        self.assertLess(weighted_mmd2_four_weight(direct, np.full(64, 1/64), identical, np.full(64, 1/64)), 1e-7)
        self.assertGreater(np.linalg.norm(direct.mean(0) - fiber_target), 1.0)

    def test_numpy_bool_serialization_default(self):
        self.assertIs(_json_default(np.bool_(True)), True)
        self.assertEqual(_json_default(np.int64(7)), 7)

    def test_validation_bank_replacement_rule(self):
        self.assertEqual(validation_bank_action(0.03, 1, 4), "append_next_chunk")
        self.assertEqual(validation_bank_action(0.21, 1, 4), "accept")
        self.assertEqual(validation_bank_action(0.19, 4, 4), "exhausted")

    def test_heun_converges_on_known_linear_velocity(self):
        initial = np.ones((3, 1, 2, 2), dtype=np.float32)
        apply = lambda t, x: -x
        errors = []
        for steps in (16, 32, 64):
            final = heun_rollout_snapshots(apply, initial, steps, np.array([0.0, 1.0]))[-1]
            errors.append(np.max(np.abs(final - np.exp(-1.0))))
        self.assertLess(errors[1], errors[0] / 3.5)
        self.assertLess(errors[2], errors[1] / 3.5)

    def test_residual_reference_cnn_is_periodic_translation_equivariant(self):
        params, architecture = init_residual_reference_cnn(
            jax.random.PRNGKey(95), channels=6, dilations=(1, 2), dtype=jnp.float32
        )
        fields = jax.random.normal(jax.random.PRNGKey(96), (3, 1, 9, 9))
        shifted = jnp.roll(fields, (2, -1), axis=(-2, -1))
        expected = jnp.roll(residual_reference_cnn(params, architecture, 0.4, fields), (2, -1), axis=(-2, -1))
        actual = residual_reference_cnn(params, architecture, 0.4, shifted)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def test_oracle_bridge_velocity_heun_tracks_known_stochastic_path(self):
        rng = np.random.default_rng(97)
        minus = rng.normal(size=(4, 1, 3, 3)).astype(np.float32)
        plus = rng.normal(size=(4, 1, 3, 3)).astype(np.float32)
        noise = standardized_noise_bank(4, (1, 3, 3), 98)
        amplitude = 0.07
        apply = lambda t, x: plus - minus + amplitude * np.pi * np.cos(np.pi * t) * noise
        rollout = heun_rollout_snapshots(apply, minus, 128, np.array([0.0, 0.5, 1.0]))
        middle, _ = noisy_field_interpolant(minus, plus, noise, 0.5, amplitude)
        self.assertLess(float(np.max(np.abs(rollout[1] - middle))), 2e-5)
        self.assertLess(float(np.max(np.abs(rollout[-1] - plus))), 2e-5)

    def test_linear_interpolant_endpoints_and_derivative(self):
        minus = jnp.zeros((2, 1, 4, 4))
        plus = jnp.ones((2, 1, 4, 4))
        at_zero, derivative = linear_field_interpolant(minus, plus, 0.0)
        at_one, _ = linear_field_interpolant(minus, plus, 1.0)
        np.testing.assert_array_equal(at_zero, minus)
        np.testing.assert_array_equal(at_one, plus)
        np.testing.assert_array_equal(derivative, plus - minus)

    def test_periodic_reference_cnn_is_translation_equivariant(self):
        params = init_periodic_reference_cnn(
            jax.random.PRNGKey(2), hidden_channels=(8, 8), dilations=(1, 2), dtype=jnp.float64
        )
        fields = jax.random.normal(jax.random.PRNGKey(3), (3, 1, 9, 9), dtype=jnp.float64)
        output = periodic_reference_cnn(params, 0.37, fields)
        shifted = jnp.roll(fields, (2, -3), axis=(-2, -1))
        shifted_output = periodic_reference_cnn(params, 0.37, shifted)
        self.assertEqual(output.shape, fields.shape)
        np.testing.assert_allclose(
            shifted_output, jnp.roll(output, (2, -3), axis=(-2, -1)), rtol=1e-11, atol=1e-11
        )

    def test_periodic_reference_cnn_respects_requested_float32_dtype(self):
        params = init_periodic_reference_cnn(jax.random.PRNGKey(72), hidden_channels=(4,), dilations=(1,))
        self.assertEqual(params["layers"][0]["weight"].dtype, jnp.float32)
        fields = jnp.zeros((2, 1, 6, 6), dtype=jnp.float32)
        self.assertEqual(periodic_reference_cnn(params, 0.2, fields).dtype, jnp.float32)

    def test_smooth_hidden_observables_have_finite_gradients(self):
        field = jax.random.normal(jax.random.PRNGKey(8), (1, 1, 7, 7), dtype=jnp.float64)
        values = smooth_hidden_observables(field, threshold=0.15)
        gradient = jax.grad(lambda x: jnp.sum(smooth_hidden_observables(x, threshold=0.15)))(field)
        self.assertEqual(values.shape, (1, 5))
        self.assertTrue(bool(jnp.all(jnp.isfinite(values))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(gradient))))


if __name__ == "__main__":
    unittest.main()
