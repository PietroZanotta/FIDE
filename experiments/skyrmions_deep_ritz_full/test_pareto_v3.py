"""Phase-0/1 fail-closed tests for the gated Pareto-v3 attempt."""

from __future__ import annotations

import unittest

from . import pareto_v3_common as common


class ParetoV3DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = common.verify_v2_frozen()
        cls.summary = common.read_json(
            common.DIAGNOSTIC_ROOT / "summary.json"
        )

    def test_01_v2_is_frozen(self):
        self.assertTrue(self.inventory["v2_frozen"])

    def test_02_v2_protocol_hash_is_unchanged(self):
        self.assertEqual(
            self.inventory["expected_hashes"]["inner_protocol"],
            "22a33ce47b2a3cc17ff063d100b878ac32c3ef6cc1a2b3e10a6eb8cd076488f1",
        )

    def test_03_v2_validation_remained_unopened(self):
        self.assertFalse(self.inventory["validation_accessed"])

    def test_04_new_seed_namespace(self):
        row = common.derive_seed(7, "selection", "screen")
        self.assertIn("official_pareto_v3:selection:screen", row["derivation_text"])

    def test_05_K_is_280(self):
        self.assertEqual(common.K, 280)

    def test_06_dictionary_hash_is_unchanged(self):
        self.assertEqual(
            common.file_sha256(common.DICTIONARY_PATH),
            common.EXPECTED_DICTIONARY_SHA256,
        )

    def test_07_rank_tolerance_is_unchanged(self):
        self.assertEqual(common.RANK_TOLERANCE, 1.0e-12)

    def test_08_ress_threshold_is_unchanged(self):
        self.assertEqual(common.MINIMUM_RESS, 0.05)

    def test_09_energy_threshold_is_unchanged(self):
        self.assertEqual(common.MAXIMUM_ENERGY_RESIDUAL, 0.08)

    def test_10_exact_selection_risk_arithmetic(self):
        self.assertEqual(common.selection_ceiling(2.0, 0.5), 2.01)

    def test_11_dual_bank_eligibility_count(self):
        self.assertEqual(self.summary["total_screening_feasible"], 193)
        self.assertEqual(self.summary["audit_ress_valid_count"], 0)

    def test_12_robust_ress_formula(self):
        for row in self.summary["all_rows"]:
            self.assertEqual(
                row["robust_ress"], min(row["screen_ress"], row["audit_ress"])
            )

    def test_13_all_audit_projections_pass(self):
        self.assertEqual(self.summary["audit_projection_valid_count"], 193)

    def test_14_phase_one_classification_is_C(self):
        self.assertTrue(self.summary["classification"].startswith("C."))
        self.assertFalse(self.summary["proceed_to_v3"])

    def test_15_no_tangent_or_full_work_entered_diagnostic(self):
        self.assertFalse(self.summary["tangent_optimization_run"])
        self.assertFalse(self.summary["full_kf_constructed"])

    def test_16_no_v3_protocol_was_frozen_after_C(self):
        self.assertFalse(common.PROTOCOL_PATH.exists())
        self.assertFalse(common.PROTOCOL_HASH_PATH.exists())

    def test_17_no_v3_selection_or_validation_exists(self):
        self.assertFalse((common.OUTPUT_ROOT / "banks").exists())
        self.assertFalse((common.OUTPUT_ROOT / "selection").exists())
        self.assertFalse((common.OUTPUT_ROOT / "fresh_validation").exists())

    def test_18_output_is_experiment_local(self):
        self.assertTrue(str(common.OUTPUT_ROOT).startswith(str(common.ROOT)))

    def test_19_diagnostic_is_development_only(self):
        self.assertTrue(self.summary["development_diagnostic_only"])
        self.assertFalse(self.summary["official_v3_result"])

    def test_20_validation_was_not_accessed(self):
        self.assertFalse(self.summary["validation_accessed"])


class ParetoV3AllAllowancesDiagnosticTests(unittest.TestCase):
    """Regression tests for the separate development-only risk sweep."""

    @classmethod
    def setUpClass(cls):
        cls.summary_path = common.ALL_ALLOWANCES_DIAGNOSTIC_ROOT / "summary.json"
        cls.inventory_path = common.ALL_ALLOWANCES_DIAGNOSTIC_ROOT / "inventory.json"
        cls.summary = common.read_json(cls.summary_path)
        cls.inventory = common.read_json(cls.inventory_path)
        cls.phase1 = common.read_json(common.DIAGNOSTIC_ROOT / "summary.json")
        cls.rows_by_id = {
            row["candidate_id"]: row
            for row in cls.summary["per_candidate_records"]
        }

    def test_21_frozen_v2_and_phase1_hashes_remain_unchanged(self):
        common.verify_v2_frozen()
        verified = common.verify_v3_phase1_frozen()
        self.assertEqual(
            verified["verified_hashes"], common.V3_PHASE1_EXPECTED
        )

    def test_22_new_artifact_pair_is_sealed(self):
        self.assertEqual(
            common.file_sha256(self.summary_path),
            self.inventory["summary_sha256"],
        )

    def test_23_candidate_pool_has_337_unique_candidates(self):
        rows = self.summary["per_candidate_records"]
        self.assertEqual(len(rows), 337)
        self.assertEqual(len({row["candidate_id"] for row in rows}), 337)
        self.assertEqual(len({tuple(row["eta"]) for row in rows}), 337)

    def test_24_screen_feasible_counts_match_frozen_v2(self):
        self.assertEqual(
            [row["screen_feasible_count"] for row in self.summary["allowances"]],
            [193, 216, 242, 274, 292, 299],
        )

    def test_25_risk_ceilings_use_exact_frozen_arithmetic(self):
        law_risk = self.summary["frozen_constants"]["law_selection_risk"]
        self.assertEqual(law_risk, 5.186549474478042)
        for row in self.summary["allowances"]:
            self.assertEqual(
                row["risk_ceiling"],
                common.selection_ceiling(law_risk, row["allowance_percent"]),
            )

    def test_26_robust_ress_is_the_dual_bank_minimum(self):
        for row in self.summary["per_candidate_records"]:
            self.assertEqual(
                row["robust_ress"],
                min(row["screen_minimum_ress"], row["audit_minimum_ress"]),
            )

    def test_27_phase1_rows_are_reused_exactly(self):
        self.assertEqual(
            self.summary["audit_reuse"]["reused_exact_phase1_count"], 193
        )
        for old in self.phase1["all_rows"]:
            new = self.rows_by_id[old["candidate_id"]]
            self.assertTrue(new["audit_result_reused_from_phase1"])
            self.assertEqual(new["eta"], old["eta"])
            self.assertEqual(new["screen_minimum_ress"], old["screen_ress"])
            self.assertEqual(new["audit_minimum_ress"], old["audit_ress"])
            self.assertEqual(new["robust_ress"], old["robust_ress"])
            self.assertEqual(
                new["audit_maximum_projection_residual"],
                old["audit"]["maximum_projection_residual"],
            )
            self.assertEqual(
                new["audit_maximum_covariance_condition"],
                old["audit"]["maximum_covariance_condition"],
            )
            self.assertEqual(
                new["audit_maximum_forcing_mean"],
                old["audit"]["maximum_forcing_mean"],
            )

    def test_28_half_percent_failure_is_unchanged(self):
        half = self.summary["allowances"][0]
        self.assertEqual(half["allowance_percent"], 0.5)
        self.assertEqual(half["dual_bank_eligible_count"], 0)
        self.assertAlmostEqual(
            half["best_audit_minimum_ress"],
            0.0483757148952091,
            places=15,
        )

    def test_29_no_forbidden_work_or_validation_access(self):
        self.assertFalse(self.summary["validation_accessed"])
        self.assertFalse(self.summary["tangent_optimization_run"])
        self.assertFalse(self.summary["full_kf_constructed"])
        self.assertFalse(self.summary["selection_or_validation_data_created"])
        for row in self.summary["per_candidate_records"]:
            self.assertFalse(row["validation_accessed"])
            self.assertFalse(row["tangent_optimization_run"])
            self.assertFalse(row["full_kf_constructed"])

    def test_30_official_v3_firewall_remains_closed(self):
        self.assertFalse(self.summary["official_v3_continuation"])
        self.assertFalse(self.summary["official_protocol_created"])
        self.assertFalse(common.PROTOCOL_PATH.exists())
        self.assertFalse(common.PROTOCOL_HASH_PATH.exists())
        self.assertFalse((common.OUTPUT_ROOT / "banks").exists())
        self.assertFalse((common.OUTPUT_ROOT / "selection").exists())
        self.assertFalse((common.OUTPUT_ROOT / "fresh_validation").exists())


if __name__ == "__main__":
    unittest.main()
