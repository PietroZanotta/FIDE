"""Contracts for fixed-K=280 empirical-quadrature qualification."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from .galerkin import BasisEvaluation, build_galerkin_system
from .production_artifacts import file_sha256
from .production_basis import (
    fit_frozen_normalization,
    make_hybrid_dictionary,
    normalized_values_and_gradients,
)
from .production_galerkin import assemble_hybrid_system
from . import k280_quadrature as study
from . import k280_quadrature_run as runner


class K280QuadratureTests(unittest.TestCase):
    def test_01_v1_pareto_immutability(self):
        if study.PROTOCOL_PATH.is_file():
            frozen = study.read_json(study.PROTOCOL_PATH)["historical_snapshot"]
            current = study.historical_snapshot()
            self.assertEqual(frozen["records"]["OFFICIAL_GALERKIN_PARETO_PROTOCOL.md"],
                             current["records"]["OFFICIAL_GALERKIN_PARETO_PROTOCOL.md"])

    def test_02_prior_resolution_immutability(self):
        if study.PROTOCOL_PATH.is_file():
            self.assertEqual(study.read_json(study.PROTOCOL_PATH)["historical_snapshot"],
                             study.historical_snapshot())

    def test_03_no_validation_access(self):
        source = Path(study.__file__).read_text()
        self.assertNotIn("load_validation_galerkin_data", source)
        self.assertNotIn("reference_bank_validation_fit.npz", source)
        self.assertNotIn("reference_bank_validation_audit.npz", source)

    def test_04_fixed_K280(self):
        self.assertEqual(study.K_FIXED, 280)

    def test_05_fixed_dictionary_hash(self):
        self.assertEqual(file_sha256(study.DICTIONARY_PATH), study.EXPECTED_DICTIONARY_SHA256)

    def test_06_fixed_rank_tolerance(self):
        self.assertEqual(study.RANK_TOLERANCE, 1.0e-12)

    def test_07_fixed_energy_threshold(self):
        self.assertEqual(study.ENERGY_THRESHOLD, 0.08)

    def test_08_deterministic_nested_bank_seed(self):
        self.assertEqual(study.derive_seed(20260822, "train"),
                         study.derive_seed(20260822, "train"))
        self.assertNotEqual(study.derive_seed(20260822, "train")["seed"],
                            study.derive_seed(20260822, "audit")["seed"])

    def test_09_exact_prefix_nesting(self):
        self.assertEqual(study.MANDATORY_SUPPORTS, (
            (32768, 16384), (32768, 32768), (65536, 32768), (65536, 65536)
        ))
        self.assertEqual(study.OPTIONAL_SUPPORT, (131072, 65536))
        if study.TRAIN_BANK_PATH.is_file():
            with np.load(study.TRAIN_BANK_PATH) as arrays:
                rows = arrays["configurations"]
                np.testing.assert_array_equal(rows[:, :32768], rows[:, :65536][:, :32768])

    def test_10_train_audit_disjointness(self):
        if not study.BANK_MANIFEST_PATH.is_file():
            self.skipTest("banks not generated")
        manifest = study.read_json(study.BANK_MANIFEST_PATH)
        self.assertTrue(all(row["overlap"] == 0 for row in manifest["exact_overlap_checks"]))

    def test_11_streamed_K_f_equivalence(self):
        dictionary = make_hybrid_dictionary(fourier_wavevector_count=4, radial_count=2)
        x = jax.random.uniform(jax.random.PRNGKey(7), (2, 7, 5, 2), dtype=jnp.float64)
        weights = jnp.full((2, 7), 1.0 / 7.0, dtype=jnp.float64)
        fitted = fit_frozen_normalization(dictionary, x, weights, chunk_size=3)
        values, gradients = [], []
        for time_index in range(2):
            value, gradient = normalized_values_and_gradients(fitted, x[time_index], time_index)
            values.append(value); gradients.append(gradient)
        forcing = jnp.linspace(-1.0, 1.0, 14).reshape(2, 7)
        direct = build_galerkin_system(BasisEvaluation(jnp.stack(values), jnp.stack(gradients)),
                                       weights, forcing)
        streamed = assemble_hybrid_system(fitted, SimpleNamespace(configurations=x), weights,
                                          forcing, chunk_size=3)
        self.assertTrue(jnp.allclose(direct.gram, streamed.gram, atol=2.0e-13))
        self.assertTrue(jnp.allclose(direct.load, streamed.load, atol=2.0e-13))

    def test_12_old_certificate_gate_decomposition(self):
        if not study.GATE_AUDIT_PATH.is_file():
            self.skipTest("old gate audit not generated")
        rows = {row["geometry_id"]: row for row in study.read_json(study.GATE_AUDIT_PATH)["rows"]}
        for geometry in ("law", "historical_1", "historical_2"):
            self.assertFalse(rows[geometry]["gates"]["ESS_valid_train"])
            self.assertFalse(rows[geometry]["physical_numerical_certificate"])

    def test_13_physical_vs_study_qualification(self):
        if not study.GATE_AUDIT_PATH.is_file():
            self.skipTest("old gate audit not generated")
        rows = study.read_json(study.GATE_AUDIT_PATH)["rows"]
        self.assertTrue(any(row["physical_numerical_certificate"] and
                            not row["resolution_study_qualification"] for row in rows))

    def test_14_action_deterministic_reproducibility(self):
        if not study.ANALYSIS_PATH.is_file():
            self.skipTest("support results unavailable")
        for item in study.read_json(study.ANALYSIS_PATH)["geometries"]:
            self.assertEqual(item["rows"][0]["train_action"], item["rows"][1]["train_action"])

    def test_15_gradient_deterministic_reproducibility(self):
        if not study.ANALYSIS_PATH.is_file():
            self.skipTest("support results unavailable")
        for item in study.read_json(study.ANALYSIS_PATH)["geometries"]:
            self.assertEqual(item["rows"][0]["gradient"], item["rows"][1]["gradient"])

    def test_16_gradient_cosine(self):
        self.assertAlmostEqual(study.gradient_comparison([1, 2], [1, 2])["cosine"], 1.0)

    def test_17_gradient_relative_change(self):
        result = study.gradient_comparison([1, 0], [0, 1])
        self.assertAlmostEqual(result["relative_difference"], 2.0**0.5)

    def test_18_paired_support_comparison(self):
        cert = {"maximum_weak_residual": 0.1, "maximum_energy_residual": 0.07,
                "maximum_moment_rate_residual": 0.02}
        low = {"train_samples": 1, "audit_samples": 1, "train_action": 2.0,
               "audit_action": 2.0, "gradient": [1.0, 0.0], "heldout_certificate": cert}
        high = {**low, "train_samples": 2, "audit_samples": 2, "train_action": 1.9,
                "audit_action": 2.1}
        result = study.paired_comparison(low, high)
        self.assertAlmostEqual(result["train_action_relative_change"], 0.1 / 1.9)

    def test_19_finite_difference_helper(self):
        self.assertAlmostEqual(study.centered_fd(1.01, 0.99, 0.01), 1.0)

    def test_20_no_eta_optimization(self):
        source = inspect.getsource(runner.main)
        self.assertNotIn('"optimize"', source)
        self.assertNotIn('"pareto"', source)

    def test_21_no_K_or_rank_retuning(self):
        source = Path(study.__file__).read_text()
        self.assertNotIn("K_LADDER", source)
        self.assertNotIn("RANK_TOLERANCES", source)
        self.assertEqual(study.K_FIXED, 280)
        self.assertEqual(study.RANK_TOLERANCE, 1.0e-12)

    def test_22_output_isolation(self):
        accepted = study.require_output_path(study.OUTPUT_ROOT / "x.json")
        self.assertTrue(str(accepted).startswith(str(study.OUTPUT_ROOT.resolve())))
        with self.assertRaises(ValueError):
            study.require_output_path(Path(tempfile.gettempdir()) / "outside.json")


if __name__ == "__main__":
    unittest.main()
