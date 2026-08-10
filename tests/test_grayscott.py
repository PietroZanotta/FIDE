from __future__ import annotations

import unittest
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grayscott.calibration import calibrate_shared_target
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


if __name__ == "__main__":
    unittest.main()
