"""Fast regression checks for the production Galerkin workflow.

Scientific acceptance is recorded by the frozen-bank JSON outputs.  These tests
exercise the implementation contracts without regenerating production data.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
import tempfile
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

from mfsi.config import load_config
from mfsi.projection import EmpiricalIProjector, IProjectionConfig

from .galerkin import (
    BasisEvaluation,
    build_galerkin_system,
    rank_aware_quadratic_solve,
)
from .production_artifacts import (
    FROZEN_FILES,
    PRODUCTION_ROOT,
    discover_artifact_sets,
    require_production_output_path,
)
from .production_basis import (
    fit_frozen_normalization,
    make_hybrid_dictionary,
    normalized_values_and_gradients,
    raw_values_and_gradients,
)
from .production_galerkin import _monotonicity, audit_hybrid_solutions
from . import production_gradient, production_workflow


class ProductionGalerkinChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config(PACKAGE_ROOT / "config.json", smoke=False)
        cls.dictionary = make_hybrid_dictionary(
            fourier_wavevector_count=4, radial_count=2
        )
        cls.x = jax.random.uniform(
            jax.random.PRNGKey(41), (2, 7, 5, 2), dtype=jnp.float64
        ) * jnp.asarray([2.0, 1.0])
        cls.weights = jnp.full((2, 7), 1.0 / 7.0, dtype=jnp.float64)

    # 1
    def test_artifact_discovery_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as raw:
            root = Path(raw)
            for name in FROZEN_FILES:
                (root / name).touch()
            before = sorted(path.name for path in root.iterdir())
            self.assertEqual(discover_artifact_sets([root]), [root.resolve()])
            self.assertEqual(before, sorted(path.name for path in root.iterdir()))

    # 2
    def test_artifact_copy_destination_isolation(self) -> None:
        self.assertEqual(
            require_production_output_path(PRODUCTION_ROOT / "artifacts"),
            (PRODUCTION_ROOT / "artifacts").resolve(),
        )
        with self.assertRaises(ValueError):
            require_production_output_path(PACKAGE_ROOT.parent / "skyrmions_deep_ritz")

    # 3
    def test_exact_eta0_reproduction_path(self) -> None:
        expected = [
            0.8954153767761239, 0.20592631632470587,
            1.3343788098383822, 0.8654288352917223,
            0.7508355365766083, 0.5179100329264751,
            1.6423735249784726, 0.5883599695898114,
        ]
        self.assertEqual(self.cfg["envelope"]["eta0"], expected)
        source = inspect.getsource(production_workflow.run_production_reproduction)
        for operation in ("reconstruct_moments", "_forcing_payload", "selection_risk"):
            self.assertIn(operation, source)

    # 4
    def test_reconstruction_determinism_gate_exists(self) -> None:
        source = inspect.getsource(production_workflow.run_production_reproduction)
        self.assertIn("reconstruction_repeat", source)
        self.assertIn("deterministic_error == 0.0", source)

    # 5
    def test_forcing_centering_in_load(self) -> None:
        values, gradients = raw_values_and_gradients(self.dictionary, self.x)
        evaluation = BasisEvaluation(values, gradients)
        forcing = jnp.arange(14, dtype=jnp.float64).reshape(2, 7)
        first = build_galerkin_system(evaluation, self.weights, forcing)
        second = build_galerkin_system(evaluation, self.weights, forcing + 37.0)
        self.assertTrue(jnp.allclose(first.load, second.load, atol=2e-13))

    # 6
    def test_projection_validity(self) -> None:
        phi = jnp.asarray([[-1.0], [0.0], [1.0]], dtype=jnp.float64)
        projector = EmpiricalIProjector(IProjectionConfig(residual_tol=1e-12))
        state = projector.project(phi, jnp.ones(3), jnp.asarray([0.2]))
        self.assertLess(float(jnp.linalg.norm(state.residual)), 1e-10)
        self.assertAlmostEqual(float(jnp.sum(state.weights)), 1.0, places=14)

    # 7
    def test_invariant_fourier_basis_is_periodic(self) -> None:
        values, _ = raw_values_and_gradients(self.dictionary, self.x)
        shifted = self.x.at[..., 0, :].add(jnp.asarray([2.0, -1.0]))
        repeated, _ = raw_values_and_gradients(self.dictionary, shifted)
        fourier = self.dictionary.feature_kind != 2
        self.assertTrue(jnp.allclose(values[..., fourier], repeated[..., fourier], atol=2e-13))

    # 8
    def test_invariant_pairwise_basis_is_periodic(self) -> None:
        values, _ = raw_values_and_gradients(self.dictionary, self.x)
        shifted = self.x.at[..., 1, :].add(jnp.asarray([-2.0, 1.0]))
        repeated, _ = raw_values_and_gradients(self.dictionary, shifted)
        radial = self.dictionary.feature_kind == 2
        self.assertTrue(jnp.allclose(values[..., radial], repeated[..., radial], atol=2e-13))

    # 9
    def test_particle_permutation_invariance(self) -> None:
        values, _ = raw_values_and_gradients(self.dictionary, self.x)
        repeated, _ = raw_values_and_gradients(
            self.dictionary, self.x[..., jnp.asarray([3, 0, 4, 1, 2]), :]
        )
        self.assertTrue(jnp.allclose(values, repeated, atol=2e-13))

    # 10
    def test_state_gradients_are_finite_and_exact(self) -> None:
        _, gradients = raw_values_and_gradients(self.dictionary, self.x[:1, :1])
        jacobian = jax.jacrev(
            lambda state: raw_values_and_gradients(self.dictionary, state)[0]
        )(self.x[0, 0])
        self.assertTrue(bool(jnp.all(jnp.isfinite(gradients))))
        self.assertTrue(jnp.allclose(gradients[0, 0], jacobian, atol=2e-12))

    # 11
    def test_basis_parameters_are_eta_independent(self) -> None:
        for function in (make_hybrid_dictionary, raw_values_and_gradients, fit_frozen_normalization):
            self.assertNotIn("eta", inspect.signature(function).parameters)

    # 12
    def test_frozen_normalization_is_deterministic(self) -> None:
        first = fit_frozen_normalization(self.dictionary, self.x, self.weights, chunk_size=3)
        second = fit_frozen_normalization(self.dictionary, self.x, self.weights, chunk_size=3)
        self.assertTrue(jnp.array_equal(first.base_means, second.base_means))
        self.assertTrue(jnp.array_equal(first.energy_scales, second.energy_scales))

    # 13
    def test_nested_basis_prefixes(self) -> None:
        fitted = fit_frozen_normalization(self.dictionary, self.x, self.weights, chunk_size=4)
        values, _ = normalized_values_and_gradients(fitted, self.x[0], 0)
        self.assertTrue(jnp.array_equal(values[:, :4], values[:, :7][:, :4]))

    def _system(self):
        values, gradients = raw_values_and_gradients(self.dictionary, self.x)
        forcing = jnp.linspace(-1.0, 1.0, 14).reshape(2, 7)
        return build_galerkin_system(BasisEvaluation(values, gradients), self.weights, forcing)

    # 14
    def test_gram_symmetry(self) -> None:
        system = self._system()
        self.assertTrue(jnp.allclose(system.gram, system.gram.swapaxes(-1, -2), atol=1e-14))

    # 15
    def test_rank_aware_solve(self) -> None:
        solve = rank_aware_quadratic_solve(
            jnp.asarray([[[3.0, 0.0], [0.0, 0.0]]]),
            jnp.asarray([[1.5, 0.0]]), relative_rank_tolerance=1e-12,
        )
        self.assertTrue(jnp.allclose(solve.coefficients, jnp.asarray([[-0.5, 0.0]])))
        self.assertEqual(int(solve.numerical_rank[0]), 1)

    # 16
    def test_range_residual(self) -> None:
        solve = rank_aware_quadratic_solve(
            jnp.asarray([[[1.0, 0.0], [0.0, 0.0]]]),
            jnp.asarray([[0.0, 1.0]]), relative_rank_tolerance=1e-12,
        )
        self.assertAlmostEqual(float(solve.range_residual[0]), 1.0, places=13)

    # 17
    def test_coefficient_stationarity(self) -> None:
        solve = rank_aware_quadratic_solve(
            jnp.asarray([[[2.0, 0.1], [0.1, 1.0]]]),
            jnp.asarray([[0.2, -0.4]]), relative_rank_tolerance=1e-12,
        )
        self.assertLess(float(solve.stationarity_residual[0]), 1e-13)

    # 18
    def test_exact_restricted_identity(self) -> None:
        solve = rank_aware_quadratic_solve(
            jnp.asarray([[[2.0, 0.1], [0.1, 1.0]]]),
            jnp.asarray([[0.2, -0.4]]), relative_rank_tolerance=1e-12,
        )
        self.assertLess(float(solve.identity_relerr_by_time[0]), 1e-13)

    # 19
    def test_monotone_nested_action_control(self) -> None:
        rows = [
            {"basis_size": 1, "galerkin_action": 1.0},
            {"basis_size": 2, "galerkin_action": 1.25},
            {"basis_size": 3, "galerkin_action": 1.2501},
        ]
        self.assertTrue(_monotonicity(rows, 1e-10)["passed"])

    # 20
    def test_heldout_weak_residual_is_computed(self) -> None:
        source = inspect.getsource(audit_hybrid_solutions)
        self.assertIn("weak_left + weak_right", source)
        self.assertIn("maximum_weak_residual", source)

    # 21
    def test_heldout_moment_rate_residual_is_computed(self) -> None:
        source = inspect.getsource(audit_hybrid_solutions)
        self.assertIn("corrected_rate - rhs", source)
        self.assertIn("maximum_moment_rate_residual", source)

    @staticmethod
    def _mock_state(design, _problem, _bank, _reconstruction):
        weights = jax.nn.softmax(jnp.stack((design[0], -design[0])))[None, :]
        forcing = jnp.stack((design[1], -design[1]))[None, :]
        return SimpleNamespace(projection=SimpleNamespace(weights=weights), forcing=forcing)

    def _mock_data(self):
        return SimpleNamespace(
            selection_problem=SimpleNamespace(time_weights=jnp.asarray([1.0])),
            ritz_train_bank=object(),
        )

    # 22
    def test_finite_eta_gradient_shape_eight(self) -> None:
        eta = jnp.arange(8, dtype=jnp.float64) / 10.0
        with patch.object(production_gradient, "reconstruct_moments", return_value=object()), \
             patch.object(production_gradient, "forcing_state", self._mock_state):
            value, gradient = production_gradient.production_hybrid_envelope_value_and_grad(
                eta, jnp.ones((1, 1)), self._mock_data(),
                jnp.asarray([[0.3, -0.1]]), jnp.asarray([[0.4, 0.7]]),
            )
        self.assertTrue(bool(jnp.isfinite(value)))
        self.assertEqual(gradient.shape, (8,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(gradient))))

    # 23
    def test_no_differentiation_through_pseudoinverse(self) -> None:
        source = inspect.getsource(
            production_gradient.production_hybrid_envelope_value_and_grad
        )
        self.assertNotIn("jnp.linalg.eigh", source)
        self.assertNotIn("rank_aware_quadratic_solve", source)
        self.assertIn("coefficients_fixed", source)

    # 24
    def test_centered_ad_fd_surrogate_agreement(self) -> None:
        eta = jnp.arange(8, dtype=jnp.float64) / 10.0
        direction = jnp.arange(1, 9, dtype=jnp.float64)
        direction /= jnp.linalg.norm(direction)
        args = (jnp.ones((1, 1)), self._mock_data(), jnp.asarray([[0.3, -0.1]]), jnp.asarray([[0.4, 0.7]]))
        with patch.object(production_gradient, "reconstruct_moments", return_value=object()), \
             patch.object(production_gradient, "forcing_state", self._mock_state):
            _, gradient = production_gradient.production_hybrid_envelope_value_and_grad(eta, *args)
            epsilon = 1e-5
            plus, _ = production_gradient.production_hybrid_envelope_value_and_grad(eta + epsilon * direction, *args)
            minus, _ = production_gradient.production_hybrid_envelope_value_and_grad(eta - epsilon * direction, *args)
        self.assertAlmostEqual(float(gradient @ direction), float((plus - minus) / (2 * epsilon)), places=9)

    # 25
    def test_rank_change_detection(self) -> None:
        low = rank_aware_quadratic_solve(
            jnp.asarray([[[1.0, 0.0], [0.0, 1e-14]]]), jnp.ones((1, 2)),
            relative_rank_tolerance=1e-12,
        )
        high = rank_aware_quadratic_solve(
            jnp.asarray([[[1.0, 0.0], [0.0, 1e-8]]]), jnp.ones((1, 2)),
            relative_rank_tolerance=1e-12,
        )
        self.assertFalse(bool(jnp.array_equal(low.numerical_rank, high.numerical_rank)))

    # 26
    def test_output_path_isolation(self) -> None:
        with self.assertRaises(ValueError):
            require_production_output_path(Path("/tmp/production-galerkin-escape"))

    # 27
    def test_forbidden_production_write_scan(self) -> None:
        forbidden_import = "experiments" + ".skyrmions_deep_ritz."
        forbidden_output = "skyrmions_deep_ritz" + "/outputs"
        for path in PACKAGE_ROOT.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn(forbidden_import, source, path.name)
            self.assertNotIn(forbidden_output, source, path.name)


if __name__ == "__main__":
    unittest.main()
