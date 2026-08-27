"""Contract tests for the official K=280 Galerkin Pareto workflow."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from . import official_pareto_common as common
from . import official_pareto_selection as selection
from . import official_pareto_validation as validation
from .production_artifacts import file_sha256


class OfficialParetoTests(unittest.TestCase):
    def test_01_official_k_is_fixed(self):
        self.assertEqual(common.OFFICIAL_K, 280)

    def test_02_dictionary_hash_is_fixed(self):
        self.assertEqual(file_sha256(common.DICTIONARY_PATH), common.EXPECTED_DICTIONARY_SHA256)

    def test_03_selection_risk_arithmetic(self):
        self.assertEqual(common.selection_ceiling(10.0, 3.0), 10.3)
        self.assertTrue(common.risk_feasible(10.3, 10.0, 3.0))
        self.assertFalse(common.risk_feasible(10.3000001, 10.0, 3.0))

    def test_04_validation_plus_five_points_arithmetic(self):
        self.assertEqual(common.validation_ceiling(10.0, 3.0), 10.8)
        self.assertEqual(common.strict_validation_ceiling(10.0, 3.0), 10.3)

    def test_05_seed_derivation_is_deterministic(self):
        left = common.derived_seed(20260822, "truth")
        right = common.derived_seed(20260822, "truth")
        self.assertEqual(left, right)
        self.assertNotEqual(left["seed"], common.derived_seed(20260822, "reference_fit")["seed"])

    def test_06_validation_generation_requires_selection_seal(self):
        source = inspect.getsource(validation.generate_fresh_validation)
        self.assertLess(source.index("_selection_seal()"), source.index("FRESH_ROOT.mkdir"))

    def test_07_selection_path_has_no_old_validation_loader(self):
        source = Path(selection.__file__).read_text(encoding="utf-8")
        self.assertNotIn("load_validation_galerkin_data", source)
        self.assertNotIn("reference_bank_validation_fit", source)
        self.assertNotIn("reference_bank_validation_audit", source)

    def test_08_feasible_sets_are_nested(self):
        ceilings = [common.selection_ceiling(5.0, p) for p in common.ALLOWANCES]
        self.assertTrue(all(right > left for left, right in zip(ceilings[:-1], ceilings[1:])))

    def test_09_incumbent_retention_uses_tolerance(self):
        self.assertTrue(common.retain_incumbent(1.0 - 0.5e-10, 1.0, 1e-10))
        self.assertFalse(common.retain_incumbent(1.0 - 2e-10, 1.0, 1e-10))

    def test_10_nonincreasing_action_contract(self):
        self.assertTrue(common.actions_nonincreasing([3.0, 2.0, 2.0, 1.0], 1e-10))
        self.assertFalse(common.actions_nonincreasing([1.0, 1.001], 1e-10))

    def test_11_exact_risk_gate_rejection_is_strict(self):
        ceiling = common.selection_ceiling(5.0, 0.5)
        self.assertTrue(common.risk_feasible(ceiling, 5.0, 0.5))
        self.assertFalse(common.risk_feasible(np.nextafter(ceiling, np.inf), 5.0, 0.5))

    def test_12_frozen_winner_path_is_write_once(self):
        source = inspect.getsource(selection.freeze_selection)
        self.assertIn("selection_path.exists() or manifest_path.exists()", source)
        self.assertIn("refusing overwrite", source)

    def test_13_common_solver_reduction(self):
        self.assertAlmostEqual(common.common_solver_reduction(4.0, 3.0), 0.25)
        with self.assertRaises(ValueError):
            common.common_solver_reduction(0.0, 0.0)

    def test_14_fresh_initial_rows_disjoint(self):
        left = validation._row_hashes(np.asarray([[1.0, 2.0], [3.0, 4.0]]))
        right = validation._row_hashes(np.asarray([[5.0, 6.0], [7.0, 8.0]]))
        overlap = validation._row_hashes(np.asarray([[3.0, 4.0], [9.0, 10.0]]))
        self.assertFalse(left & right)
        self.assertTrue(left & overlap)

    def test_15_validation_classification_cannot_change_winner(self):
        self.assertEqual(common.validation_classification(numerical_valid=True, declared_risk_pass=True), "PASS")
        self.assertEqual(common.validation_classification(numerical_valid=True, declared_risk_pass=False), "VALIDATION RISK REVERSAL")
        self.assertEqual(common.validation_classification(numerical_valid=False, declared_risk_pass=True), "VALIDATION NUMERICAL FAILURE")
        source = inspect.getsource(validation.run_fresh_validation)
        self.assertNotIn("winner =", source)

    def test_16_deep_ritz_cannot_enter_decision_path(self):
        for module in (selection, validation):
            source = Path(module.__file__).read_text(encoding="utf-8").lower()
            self.assertNotIn("solve_deep_ritz", source)
            self.assertNotIn("audit_deep_ritz", source)

    def test_17_output_isolation(self):
        accepted = common.require_official_output_path(common.OUTPUT_ROOT / "selection" / "x.json")
        self.assertTrue(str(accepted).startswith(str(common.OUTPUT_ROOT.resolve())))
        with self.assertRaises(ValueError):
            common.require_official_output_path(Path(tempfile.gettempdir()) / "outside.json")

    def test_18_protocol_hash_is_order_independent(self):
        self.assertEqual(common.payload_sha256({"a": 1, "b": 2}), common.payload_sha256({"b": 2, "a": 1}))
        self.assertNotEqual(common.payload_sha256({"a": 1}), common.payload_sha256({"a": 2}))

    def test_19_allowance_resume_signature_includes_ceiling(self):
        source = inspect.getsource(selection._trajectory_signature)
        self.assertIn("risk_ceiling", source)
        self.assertIn("allowance_percent", source)
        self.assertIn("protocol", source)

    def test_20_final_table_consistency_when_present(self):
        path = common.OUTPUT_ROOT / "final_summary.json"
        if not path.is_file():
            self.skipTest("official final summary not generated yet")
        result = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(result["selection_table"]), 6)
        self.assertEqual(len(result["validation_table"]), 6)
        self.assertEqual([row["allowance_percent"] for row in result["selection_table"]], list(common.ALLOWANCES))
        self.assertTrue(result["selection_winners_unchanged_after_validation"])


if __name__ == "__main__":
    unittest.main()
