"""Contracts for the selection-development Galerkin resolution study."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .galerkin_only import prefix_dictionary
from .production_artifacts import file_sha256
from .production_basis import load_dictionary
from . import resolution_study as study
from . import resolution_study_run as runner


class ResolutionStudyTests(unittest.TestCase):
    def test_01_v1_protocol_and_outputs_immutable(self):
        self.assertTrue(study.verify_v1_immutable()["passed"])

    def test_02_no_validation_access(self):
        source = Path(study.__file__).read_text(encoding="utf-8")
        self.assertNotIn("load_validation_galerkin_data", source)
        self.assertNotIn("reference_bank_validation_fit.npz", source)
        self.assertNotIn("reference_bank_validation_audit.npz", source)

    def test_03_development_seed_derivation(self):
        first = study.derive_seed(20260822, "train", 32768)
        self.assertEqual(first, study.derive_seed(20260822, "train", 32768))
        self.assertNotEqual(first["seed"], study.derive_seed(20260822, "audit", 16384)["seed"])

    def test_04_nested_bank_ladder(self):
        self.assertEqual(study.SUPPORT_LADDER, ((8192, 4096), (16384, 8192), (16384, 16384), (32768, 16384)))
        self.assertEqual(sorted(set(row[0] for row in study.SUPPORT_LADDER)), [8192, 16384, 32768])

    def test_05_exact_bank_size_accounting_when_present(self):
        if not study.BANK_MANIFEST_PATH.is_file():
            self.skipTest("banks not generated")
        manifest = json.loads(study.BANK_MANIFEST_PATH.read_text())
        self.assertEqual(manifest["nested_prefixes"]["train"], [8192, 16384, 32768])
        self.assertEqual(manifest["nested_prefixes"]["audit"], [4096, 8192, 16384])

    def test_06_K280_dictionary_preserved(self):
        self.assertEqual(study.K_PRIMARY, 280)
        self.assertEqual(file_sha256(study.DICTIONARY_PATH), study.EXPECTED_DICTIONARY_SHA256)

    def test_07_fixed_geometries_preserved(self):
        self.assertEqual([row[0] for row in study.FIXED_GEOMETRIES], [
            "law", "historical_0p5", "historical_1", "historical_2", "eta0_3pct", "eta_grad_3pct"
        ])
        self.assertTrue(all(len(row[2]) == 8 for row in study.FIXED_GEOMETRIES))

    def test_08_action_reproducibility_contract(self):
        self.assertEqual(study.relative_change(1.0, 1.0), 0.0)
        if (study.OUTPUT_ROOT / "quadrature" / "K280" / "result.json").is_file():
            raw = json.loads((study.OUTPUT_ROOT / "quadrature" / "K280" / "result.json").read_text())
            self.assertTrue(all(np.isfinite(row["train_action"]) for item in raw["geometries"] for row in item["rows"]))

    def test_09_gradient_reproducibility_contract(self):
        same = study.gradient_comparison([1.0, 2.0], [1.0, 2.0])
        self.assertAlmostEqual(same["cosine"], 1.0)
        self.assertEqual(same["relative_difference"], 0.0)

    def test_10_cross_support_gradient_comparison(self):
        result = study.gradient_comparison([1.0, 0.0], [0.0, 1.0])
        self.assertEqual(result["cosine"], 0.0)
        self.assertAlmostEqual(result["relative_difference"], 2**0.5)

    def test_11_energy_residual_threshold_never_changes(self):
        protocol_source = Path(study.__file__).read_text()
        self.assertIn('"safety_margin_energy": 0.075', protocol_source)
        self.assertIn('<= 0.08', protocol_source)
        self.assertNotIn('energy_threshold_unchanged": 0.09', protocol_source)

    def test_12_rank_eigenvalue_diagnostics_exist(self):
        source = inspect.getsource(study._algebra)
        for key in ("smallest_retained_eigenvalue", "largest_eigenvalue", "worst_retained_condition"):
            self.assertIn(key, source)

    def test_13_K_prefix_nestedness(self):
        dictionary = load_dictionary(study.DICTIONARY_PATH, box=(2.0, 1.0))
        lower, upper = prefix_dictionary(dictionary, 120), prefix_dictionary(dictionary, 160)
        np.testing.assert_array_equal(np.asarray(lower.feature_kind), np.asarray(upper.feature_kind)[:120])
        np.testing.assert_array_equal(np.asarray(lower.base_means), np.asarray(upper.base_means)[:, :120])

    def test_14_rank_tolerance_set_is_predeclared(self):
        self.assertEqual(study.RANK_TOLERANCES, (1e-10, 1e-11, 1e-12))

    def test_15_no_certificate_threshold_relaxation(self):
        source = Path(study.__file__).read_text()
        self.assertNotIn("maximum_energy_residual\"] =", source)
        self.assertNotIn("0.09", source)
        self.assertNotIn("0.10, 0.11", source)

    def test_16_no_eta_optimization_command(self):
        source = inspect.getsource(runner.main)
        self.assertNotIn('"optimize"', source)
        self.assertNotIn('"select"', source)
        self.assertNotIn("run_selection_sweep", source)

    def test_17_candidate_v2_initialization_gate(self):
        payload = {"exact_risk_valid": True, "geometry_valid": True,
                   "train_forcing_valid": True, "algebra_valid": True,
                   "complete_heldout_certificate": False}
        self.assertTrue(study.initialization_gate(payload))

    def test_18_endpoint_requires_complete_certificate(self):
        payload = {"exact_risk_valid": True, "geometry_valid": True,
                   "train_forcing_valid": True, "algebra_valid": True,
                   "complete_heldout_certificate": False}
        self.assertFalse(study.endpoint_gate(payload))
        payload["complete_heldout_certificate"] = True
        self.assertTrue(study.endpoint_gate(payload))

    def test_19_start_generator_uses_selection_risk_only(self):
        source = inspect.getsource(study.run_start_generator_diagnostic)
        self.assertIn("selection_risk", source)
        self.assertNotIn("validation_risk", source)
        self.assertNotIn("evaluate_case", source)

    def test_20_output_isolation(self):
        accepted = study.require_output_path(study.OUTPUT_ROOT / "x.json")
        self.assertTrue(str(accepted).startswith(str(study.OUTPUT_ROOT.resolve())))
        with self.assertRaises(ValueError):
            study.require_output_path(Path(tempfile.gettempdir()) / "outside.json")


if __name__ == "__main__":
    unittest.main()
