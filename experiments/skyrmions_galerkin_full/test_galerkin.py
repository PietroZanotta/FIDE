"""Fast unit checks for the fixed-basis Galerkin implementation."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
for search_path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from . import galerkin
from .deep_ritz import init_ritz_params
from .galerkin import (
    BasisEvaluation,
    FrozenDeepSetsBasis,
    GalerkinSystem,
    aggregate_quadratic_values,
    build_galerkin_system,
    evaluate_basis,
    frozen_deepsets_latent,
    galerkin_envelope_value_and_grad,
    rank_aware_quadratic_solve,
)
from .workflow import OUTPUT_ROOT, require_output_path


class GalerkinChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        params = init_ritz_params(
            jax.random.PRNGKey(17), hidden_width=6, hidden_layers=1
        )
        cls.basis = FrozenDeepSetsBasis(
            params=params, name="test", source="unit-test",
            source_sha256=None, box=(2.0, 1.0),
        )
        cls.x = jax.random.uniform(
            jax.random.PRNGKey(19), (2, 5, 4, 2), dtype=jnp.float64
        ) * jnp.asarray([2.0, 1.0])
        cls.times = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
        cls.evaluation = evaluate_basis(cls.basis, cls.x, cls.times, 4)

    def test_permutation_invariance(self) -> None:
        permutation = jnp.asarray([2, 0, 3, 1])
        permuted = evaluate_basis(self.basis, self.x[..., permutation, :], self.times, 4)
        self.assertTrue(jnp.allclose(self.evaluation.values, permuted.values, atol=1e-13))

    def test_state_gradients_are_finite(self) -> None:
        self.assertTrue(bool(jnp.all(jnp.isfinite(self.evaluation.state_gradients))))

    def test_feature_evaluation_is_deterministic(self) -> None:
        repeated = evaluate_basis(self.basis, self.x, self.times, 4)
        self.assertTrue(bool(jnp.array_equal(self.evaluation.values, repeated.values)))
        self.assertTrue(bool(jnp.array_equal(
            self.evaluation.state_gradients, repeated.state_gradients
        )))

    def test_basis_is_eta_independent(self) -> None:
        signature = inspect.signature(frozen_deepsets_latent)
        self.assertNotIn("eta", signature.parameters)
        leaves_before = jax.tree_util.tree_leaves(self.basis.params)
        evaluate_basis(self.basis, self.x, self.times, 4)
        leaves_after = jax.tree_util.tree_leaves(self.basis.params)
        self.assertTrue(all(bool(jnp.array_equal(a, b)) for a, b in zip(
            leaves_before, leaves_after, strict=True
        )))

    def _system(self) -> GalerkinSystem:
        weights = jnp.full((2, 5), 0.2, dtype=jnp.float64)
        forcing = jnp.arange(10, dtype=jnp.float64).reshape(2, 5) - 4.5
        return build_galerkin_system(self.evaluation, weights, forcing)

    def test_gram_is_symmetric(self) -> None:
        system = self._system()
        self.assertTrue(jnp.allclose(system.gram, system.gram.swapaxes(-1, -2), atol=1e-14))

    def test_raw_symmetry_is_machine_precision(self) -> None:
        self.assertLess(float(jnp.max(self._system().raw_symmetry_residual)), 1e-14)

    def test_rank_aware_minimum_norm_solution(self) -> None:
        gram = jnp.asarray([[[4.0, 0.0], [0.0, 0.0]]])
        load = jnp.asarray([[2.0, 0.0]])
        solve = rank_aware_quadratic_solve(gram, load, relative_rank_tolerance=1e-12)
        self.assertTrue(jnp.allclose(solve.coefficients, jnp.asarray([[-0.5, 0.0]])))
        self.assertEqual(int(solve.numerical_rank[0]), 1)

    def test_range_residual_detects_incompatibility(self) -> None:
        gram = jnp.asarray([[[1.0, 0.0], [0.0, 0.0]]])
        load = jnp.asarray([[0.0, 2.0]])
        solve = rank_aware_quadratic_solve(gram, load, relative_rank_tolerance=1e-12)
        self.assertAlmostEqual(float(solve.range_residual[0]), 1.0, places=14)

    def test_stationarity_residual(self) -> None:
        gram = jnp.asarray([[[2.0, 0.2], [0.2, 1.0]]])
        load = jnp.asarray([[0.4, -0.3]])
        solve = rank_aware_quadratic_solve(gram, load, relative_rank_tolerance=1e-12)
        self.assertLess(float(solve.stationarity_residual[0]), 1e-13)

    def test_exact_quadratic_identity(self) -> None:
        gram = jnp.asarray([[[2.0, 0.2], [0.2, 1.0]]])
        load = jnp.asarray([[0.4, -0.3]])
        solve = rank_aware_quadratic_solve(gram, load, relative_rank_tolerance=1e-12)
        self.assertLess(float(solve.identity_relerr_by_time[0]), 1e-13)

    def test_time_aggregation(self) -> None:
        gram = jnp.asarray([[[1.0]], [[2.0]]])
        load = jnp.asarray([[1.0], [2.0]])
        solve = rank_aware_quadratic_solve(gram, load, relative_rank_tolerance=1e-12)
        aggregate = aggregate_quadratic_values(solve, jnp.asarray([0.25, 0.75]))
        expected = 0.25 * float(solve.action_by_time[0]) + 0.75 * float(solve.action_by_time[1])
        self.assertAlmostEqual(float(aggregate["action"]), expected, places=14)

    @staticmethod
    def _mock_system(eta, _problem, _bank, _basis):
        gram = jnp.asarray([[[1.0 + eta[0] * eta[0]]]])
        load = jnp.asarray([[eta[1] - 0.25]])
        zero = jnp.zeros((1,), dtype=jnp.float64)
        system = GalerkinSystem(
            gram=gram, load=load, basis_means=jnp.zeros((1, 1)),
            centered_basis=jnp.zeros((1, 1, 1)), weights=jnp.ones((1, 1)),
            forcing=jnp.zeros((1, 1)), raw_symmetry_residual=zero,
            forcing_mean=zero,
        )
        return system, None

    def test_eta_gradient_is_finite_shape_eight(self) -> None:
        eta = jnp.arange(8, dtype=jnp.float64) / 10.0
        problem = SimpleNamespace(time_weights=jnp.asarray([1.0]))
        with patch.object(galerkin, "system_at_eta", self._mock_system):
            value, gradient = galerkin_envelope_value_and_grad(
                eta, jnp.asarray([[0.7]]), problem, None, None
            )
        self.assertTrue(bool(jnp.isfinite(value)))
        self.assertEqual(gradient.shape, (8,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(gradient))))

    def test_no_eigensolve_in_differentiated_envelope(self) -> None:
        source = inspect.getsource(galerkin_envelope_value_and_grad)
        self.assertNotIn("jnp.linalg.eigh", source)
        self.assertNotIn("rank_aware_quadratic_solve", source)
        self.assertIn("coefficients_fixed", source)

    def test_directional_finite_difference_agreement(self) -> None:
        eta = jnp.arange(8, dtype=jnp.float64) / 10.0
        direction = jnp.arange(1, 9, dtype=jnp.float64)
        direction = direction / jnp.linalg.norm(direction)
        problem = SimpleNamespace(time_weights=jnp.asarray([1.0]))
        coefficients = jnp.asarray([[0.7]])
        with patch.object(galerkin, "system_at_eta", self._mock_system):
            value, gradient = galerkin_envelope_value_and_grad(
                eta, coefficients, problem, None, None
            )
            epsilon = 1e-5
            plus, _ = galerkin_envelope_value_and_grad(
                eta + epsilon * direction, coefficients, problem, None, None
            )
            minus, _ = galerkin_envelope_value_and_grad(
                eta - epsilon * direction, coefficients, problem, None, None
            )
        finite_difference = (plus - minus) / (2.0 * epsilon)
        self.assertAlmostEqual(
            float(jnp.vdot(gradient, direction)), float(finite_difference), places=9
        )
        self.assertTrue(bool(jnp.isfinite(value)))

    def test_rank_change_detection(self) -> None:
        solve_a = rank_aware_quadratic_solve(
            jnp.asarray([[[1.0, 0.0], [0.0, 1e-13]]]), jnp.ones((1, 2)),
            relative_rank_tolerance=1e-10,
        )
        solve_b = rank_aware_quadratic_solve(
            jnp.asarray([[[1.0, 0.0], [0.0, 1e-8]]]), jnp.ones((1, 2)),
            relative_rank_tolerance=1e-10,
        )
        self.assertFalse(bool(jnp.array_equal(
            solve_a.numerical_rank, solve_b.numerical_rank
        )))

    def test_output_path_isolation(self) -> None:
        self.assertEqual(
            require_output_path(OUTPUT_ROOT / "galerkin" / "unit"),
            (OUTPUT_ROOT / "galerkin" / "unit").resolve(),
        )
        with self.assertRaises(ValueError):
            require_output_path(Path("/tmp/galerkin-escape"))

    def test_no_production_experiment_import_or_write_target(self) -> None:
        forbidden_import = "experiments" + ".skyrmions_deep_ritz."
        forbidden_output = "skyrmions_deep_ritz" + "/outputs"
        for path in PACKAGE_ROOT.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn(forbidden_import, source, path.name)
            self.assertNotIn(forbidden_output, source, path.name)


if __name__ == "__main__":
    unittest.main()
