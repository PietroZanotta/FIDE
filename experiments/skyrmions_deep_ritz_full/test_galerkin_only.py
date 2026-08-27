"""Isolated regression contracts for the Galerkin-only 3% workflow."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
for search_path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from mfsi.config import load_config

from .galerkin import (
    BasisEvaluation, aggregate_quadratic_values, build_galerkin_system,
    rank_aware_quadratic_solve,
)
from .galerkin_only import (
    GALERKIN_ONLY_ROOT, OLD_DICTIONARY, _cache_signature,
    require_galerkin_only_output_path,
)
from .galerkin_only_workflow import (
    galerkin_only_static_audit, run_galerkin_only_optimization,
    run_galerkin_only_validation,
)
from .production_basis import (
    fit_frozen_normalization, load_dictionary, make_hybrid_dictionary,
    normalized_values_and_gradients, raw_values_and_gradients,
)
from .production_galerkin import assemble_hybrid_system


class GalerkinOnlyChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config(PACKAGE_ROOT / "config.json", smoke=False)
        cls.dictionary = make_hybrid_dictionary(
            fourier_wavevector_count=4, radial_count=2
        )
        cls.x = jax.random.uniform(
            jax.random.PRNGKey(29), (2, 7, 5, 2), dtype=jnp.float64
        ) * jnp.asarray([2.0, 1.0])
        cls.weights = jnp.full((2, 7), 1.0 / 7.0, dtype=jnp.float64)

    # 1
    def test_new_workflow_has_no_deep_ritz_call(self) -> None:
        self.assertTrue(galerkin_only_static_audit()["passed"])

    @staticmethod
    def _fixed_galerkin_device_result(device):
        matrix = jnp.asarray(
            [[[2.0, 0.1], [0.1, 1.0]]], dtype=jnp.float64, device=device
        )
        rhs = jnp.asarray([[0.2, -0.4]], dtype=jnp.float64, device=device)
        solve = rank_aware_quadratic_solve(
            matrix, rhs, relative_rank_tolerance=1.0e-12
        )
        eta = jnp.asarray([0.3, -0.2], dtype=jnp.float64, device=device)
        coefficient = jax.lax.stop_gradient(solve.coefficients)

        def fixed_envelope(design):
            load = rhs + jnp.asarray([[design[0], design[1]]]) * 1.0e-3
            return -2.0 * (
                0.5 * jnp.einsum("ti,tij,tj->", coefficient, matrix, coefficient)
                + jnp.einsum("ti,ti->", load, coefficient)
            )

        value, gradient = jax.value_and_grad(fixed_envelope)(eta)
        return np.asarray(value), np.asarray(gradient)

    # 2
    def test_cpu_gpu_fixed_galerkin_value_equivalence(self) -> None:
        try:
            gpu = jax.devices("gpu")[0]
        except (RuntimeError, IndexError):
            self.skipTest("CUDA device unavailable")
        cpu_value, _ = self._fixed_galerkin_device_result(jax.devices("cpu")[0])
        gpu_value, _ = self._fixed_galerkin_device_result(gpu)
        self.assertTrue(np.allclose(cpu_value, gpu_value, rtol=1e-12, atol=1e-13))

    # 3
    def test_cpu_gpu_fixed_galerkin_gradient_equivalence(self) -> None:
        try:
            gpu = jax.devices("gpu")[0]
        except (RuntimeError, IndexError):
            self.skipTest("CUDA device unavailable")
        _, cpu_gradient = self._fixed_galerkin_device_result(jax.devices("cpu")[0])
        _, gpu_gradient = self._fixed_galerkin_device_result(gpu)
        self.assertTrue(np.allclose(cpu_gradient, gpu_gradient, rtol=1e-12, atol=1e-13))

    def _production_dictionary(self, size: int):
        path = GALERKIN_ONLY_ROOT / "cache" / "dictionaries" / f"dictionary_K{size}.npz"
        if not path.is_file() or not OLD_DICTIONARY.is_file():
            self.skipTest("production dictionary artifacts unavailable")
        return load_dictionary(path, box=(2.0, 1.0))

    # 4
    def test_first_160_coordinates_are_unchanged(self) -> None:
        old = load_dictionary(OLD_DICTIONARY, box=(2.0, 1.0))
        extended = self._production_dictionary(280)
        for name in ("feature_kind", "feature_source_index", "base_means", "energy_scales"):
            self.assertTrue(np.array_equal(
                np.asarray(getattr(old, name)),
                np.asarray(getattr(extended, name))[..., :160],
            ))

    # 5
    def test_extended_dictionary_is_nested(self) -> None:
        lower = self._production_dictionary(240)
        upper = self._production_dictionary(280)
        self.assertTrue(np.array_equal(
            np.asarray(lower.feature_kind), np.asarray(upper.feature_kind[:240])
        ))
        self.assertTrue(np.array_equal(
            np.asarray(lower.base_means), np.asarray(upper.base_means[:, :240])
        ))

    # 6
    def test_basis_parameters_are_eta_independent(self) -> None:
        for function in (
            make_hybrid_dictionary, raw_values_and_gradients,
            fit_frozen_normalization,
        ):
            self.assertNotIn("eta", inspect.signature(function).parameters)

    # 7
    def test_permutation_invariance(self) -> None:
        values, _ = raw_values_and_gradients(self.dictionary, self.x)
        permuted, _ = raw_values_and_gradients(
            self.dictionary, self.x[..., jnp.asarray([3, 0, 4, 1, 2]), :]
        )
        self.assertTrue(jnp.allclose(values, permuted, atol=2e-13))

    # 8
    def test_periodicity(self) -> None:
        values, _ = raw_values_and_gradients(self.dictionary, self.x)
        shifted = self.x.at[..., 0, :].add(jnp.asarray([2.0, -1.0]))
        repeated, _ = raw_values_and_gradients(self.dictionary, shifted)
        self.assertTrue(jnp.allclose(values, repeated, atol=2e-13))

    # 9
    def test_state_gradients_are_finite(self) -> None:
        _, gradients = raw_values_and_gradients(self.dictionary, self.x)
        self.assertTrue(bool(jnp.all(jnp.isfinite(gradients))))

    # 10
    def test_cache_signature_covers_basis_and_normalization(self) -> None:
        fitted = fit_frozen_normalization(
            self.dictionary, self.x, self.weights, chunk_size=3
        )
        data = SimpleNamespace(train_bank=SimpleNamespace(configurations=self.x))
        with tempfile.TemporaryDirectory(dir="/tmp") as raw:
            root = Path(raw)
            artifact = root / "artifacts"
            artifact.mkdir()
            (artifact / "isolated_artifact_manifest.json").write_text("{}")
            first_path = root / "first.npz"
            second_path = root / "second.npz"
            first_path.write_bytes(b"first")
            second_path.write_bytes(b"second")
            first, metadata = _cache_signature(
                self.cfg, artifact, first_path, fitted, data
            )
            second, _ = _cache_signature(
                self.cfg, artifact, second_path, fitted, data
            )
        self.assertNotEqual(first, second)
        for key in (
            "artifact_manifest_sha256", "basis_definition_sha256", "basis_size",
            "normalization_hash", "dtype", "configuration_hash",
        ):
            self.assertIn(key, metadata)

    # 11
    def test_streamed_K_f_matches_direct_contractions(self) -> None:
        fitted = fit_frozen_normalization(
            self.dictionary, self.x, self.weights, chunk_size=3
        )
        values, gradients = [], []
        for time_index in range(2):
            v, g = normalized_values_and_gradients(fitted, self.x[time_index], time_index)
            values.append(v)
            gradients.append(g)
        forcing = jnp.linspace(-1.0, 1.0, 14).reshape(2, 7)
        direct = build_galerkin_system(
            BasisEvaluation(jnp.stack(values), jnp.stack(gradients)),
            self.weights, forcing,
        )
        bank = SimpleNamespace(configurations=self.x)
        streamed = assemble_hybrid_system(
            fitted, bank, self.weights, forcing, chunk_size=3
        )
        self.assertTrue(jnp.allclose(direct.gram, streamed.gram, atol=2e-13))
        self.assertTrue(jnp.allclose(direct.load, streamed.load, atol=2e-13))

    # 12
    def test_rank_aware_solve(self) -> None:
        solve = rank_aware_quadratic_solve(
            jnp.asarray([[[3.0, 0.0], [0.0, 0.0]]]),
            jnp.asarray([[1.5, 0.0]]), relative_rank_tolerance=1e-12,
        )
        self.assertEqual(int(solve.numerical_rank[0]), 1)
        self.assertTrue(jnp.allclose(
            solve.coefficients, jnp.asarray([[-0.5, 0.0]])
        ))

    # 13
    def test_restricted_A_equals_minus_2J(self) -> None:
        solve = rank_aware_quadratic_solve(
            jnp.asarray([[[2.0, 0.1], [0.1, 1.0]]]),
            jnp.asarray([[0.2, -0.4]]), relative_rank_tolerance=1e-12,
        )
        aggregate = aggregate_quadratic_values(solve, jnp.asarray([1.0]))
        self.assertLess(float(aggregate["identity_relerr"]), 1e-13)

    # 14
    def test_recorded_production_gradient_shape_and_finiteness(self) -> None:
        path = GALERKIN_ONLY_ROOT / "benchmark" / "result.json"
        if not path.is_file():
            self.skipTest("production benchmark artifact unavailable")
        gradient = np.asarray(json.loads(path.read_text())["evaluation"]["gradient"])
        self.assertEqual(gradient.shape, (8,))
        self.assertTrue(np.all(np.isfinite(gradient)))

    # 15
    def test_exact_three_percent_risk_gate(self) -> None:
        source = inspect.getsource(run_galerkin_only_optimization)
        self.assertIn("risk_ceiling = 1.03 * law_risk", source)
        path = GALERKIN_ONLY_ROOT / "selection" / "result.json"
        if path.is_file():
            result = json.loads(path.read_text())
            self.assertLessEqual(result["winner"]["risk"], result["risk_ceiling"])

    # 16
    def test_validation_requires_frozen_selection(self) -> None:
        source = inspect.getsource(run_galerkin_only_validation)
        self.assertIn('selection.get("selection_frozen", False)', source)
        self.assertIn("winner_frozen_before_validation", source)

    # 17
    def test_selection_does_not_access_validation(self) -> None:
        source = inspect.getsource(run_galerkin_only_optimization)
        self.assertNotIn("load_validation_galerkin_data", source)
        self.assertIn('"validation_accessed": False', source)

    # 18
    def test_validation_invokes_no_deep_ritz_solver(self) -> None:
        source = inspect.getsource(run_galerkin_only_validation)
        for token in ("solve_deep_ritz", "audit_deep_ritz", "authoritative_evaluate"):
            self.assertNotIn(token, source)

    # 19
    def test_output_path_isolation(self) -> None:
        self.assertEqual(
            require_galerkin_only_output_path(GALERKIN_ONLY_ROOT / "validation"),
            (GALERKIN_ONLY_ROOT / "validation").resolve(),
        )
        with self.assertRaises(ValueError):
            require_galerkin_only_output_path(PACKAGE_ROOT.parent / "skyrmions_deep_ritz")

    # 20
    def test_notify_helper_interface_without_sending(self) -> None:
        source = (REPO_ROOT / "scripts" / "notify.py").read_text(encoding="utf-8")
        self.assertIn('add_argument("message"', source)
        self.assertIn('add_argument("--title"', source)
        self.assertIn('add_argument("--timeout-ms"', source)


if __name__ == "__main__":
    unittest.main()
