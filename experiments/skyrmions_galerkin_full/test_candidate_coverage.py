"""Fail-closed tests for the development-only candidate-coverage study."""

from __future__ import annotations

import inspect
import unittest

import numpy as np

from . import candidate_coverage as coverage
from .pareto_v3_common import (
    ALL_ALLOWANCES_DIAGNOSTIC_ROOT,
    V3_PHASE1_EXPECTED,
    eta_key,
    file_sha256,
    read_json,
    selection_ceiling,
    verify_v2_frozen,
    verify_v3_phase1_frozen,
)


class CandidateCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = read_json(coverage.GENERATOR_SPEC_PATH)
        cls.pool = read_json(coverage.CANDIDATE_POOL_PATH)
        cls.screen = read_json(coverage.SCREEN_RESULTS_PATH)
        cls.audit = read_json(coverage.AUDIT_RESULTS_PATH)
        cls.summary = read_json(coverage.SUMMARY_PATH)
        cls.inventory = read_json(coverage.INVENTORY_PATH)

    def test_01_original_v2_and_phase1_are_immutable(self):
        self.assertTrue(verify_v2_frozen()["v2_frozen"])
        self.assertEqual(
            verify_v3_phase1_frozen()["verified_hashes"], V3_PHASE1_EXPECTED
        )

    def test_02_full_pool_diagnostic_is_immutable(self):
        for name, expected in coverage.EXPECTED_ALL_ALLOWANCES_HASHES.items():
            self.assertEqual(
                file_sha256(ALL_ALLOWANCES_DIAGNOSTIC_ROOT / name), expected
            )

    def test_03_no_forbidden_work_or_validation(self):
        source = inspect.getsource(coverage)
        self.assertNotIn("load_validation_galerkin_data", source)
        self.assertNotIn("FullContext", source)
        self.assertNotIn("select_tangent", source)
        self.assertFalse(self.summary["validation_accessed"])
        self.assertFalse(self.summary["tangent_optimization_run"])
        self.assertFalse(self.summary["full_kf_constructed"])
        self.assertFalse(self.summary["eigensolve_run"])

    def test_04_generator_seed_namespace_is_deterministic(self):
        self.assertEqual(coverage.derive_seed(7, "local"), coverage.derive_seed(7, "local"))
        self.assertIn(coverage.VERSION, coverage.derive_seed(7, "local")["derivation_text"])

    def test_05_pool_has_requested_component_mixture(self):
        self.assertEqual(self.pool["final_new_unique_count"], coverage.REQUESTED_NEW_COUNT)
        self.assertEqual(self.pool["component_counts"], coverage.COMPONENT_TARGETS)
        self.assertEqual(len(self.pool["rows"]), coverage.REQUESTED_NEW_COUNT)

    def test_06_candidate_pool_is_canonicalized(self):
        law = self.spec["law_eta"]
        box = self.spec["box"]
        for row in self.pool["rows"]:
            actual = coverage.canonicalize_eta(row["eta"], law, box)
            np.testing.assert_array_equal(actual, np.asarray(row["eta"]))
            self.assertEqual(row["canonical_eta"], row["eta"])

    def test_07_minimum_separation_always_holds(self):
        threshold = self.spec["minimum_sensor_separation"]
        for row in self.pool["rows"]:
            self.assertGreaterEqual(
                coverage.minimum_periodic_separation(row["eta"], self.spec["box"]),
                threshold,
            )

    def test_08_new_candidates_are_unique(self):
        keys = [row["eta_sha256"] for row in self.pool["rows"]]
        self.assertEqual(len(keys), len(set(keys)))

    def test_09_new_candidates_are_disjoint_from_v2(self):
        old = read_json(
            coverage.V2_OUTPUT_ROOT / "screening" / "candidate_pool.json"
        )["rows"]
        old_keys = {
            eta_key(
                coverage.canonicalize_eta(
                    row["eta"], self.spec["law_eta"], self.spec["box"]
                )
            )
            for row in old
        }
        self.assertTrue(old_keys.isdisjoint(row["eta_sha256"] for row in self.pool["rows"]))

    def test_10_pool_was_frozen_before_audit(self):
        self.assertTrue(self.pool["generated_before_audit_evaluation"])
        self.assertTrue(self.audit["candidate_pool_frozen_before_audit"])
        self.assertLessEqual(
            self.audit["candidate_pool_mtime_ns_before_audit"],
            self.audit["audit_started_ns"],
        )
        self.assertEqual(
            self.audit["candidate_pool_sha256"], file_sha256(coverage.CANDIDATE_POOL_PATH)
        )

    def test_11_exact_risk_arithmetic_and_threshold(self):
        law = self.screen["law_selection_risk"]
        for row in self.summary["allowances"]:
            self.assertEqual(
                row["risk_ceiling"],
                selection_ceiling(law, row["allowance_percent"]),
            )
        self.assertEqual(coverage.MINIMUM_RESS, 0.05)

    def test_12_robust_ress_is_exact_minimum(self):
        for row in self.audit["rows"]:
            self.assertEqual(
                row["robust_ress"],
                min(row["screen"]["minimum_ress"], row["audit"]["minimum_ress"]),
            )

    def test_13_original_dual_bank_counts_reproduce(self):
        self.assertEqual(
            [row["original_v2_dual_bank_count"] for row in self.summary["allowances"]],
            [0, 1, 12, 35, 53, 59],
        )

    def test_14_risk_ress_statistics_are_sealed(self):
        summary_hash = file_sha256(coverage.SUMMARY_PATH)
        artifact = next(
            row for row in self.inventory["artifacts"] if row["path"] == "summary.json"
        )
        self.assertEqual(summary_hash, artifact["sha256"])
        self.assertEqual(
            set(self.summary["spearman_correlations"]), {"original", "new", "combined"}
        )

    def test_15_periodic_path_uses_minimum_image(self):
        law = np.asarray(self.spec["law_eta"])
        shifted = law.copy()
        shifted[0] += self.spec["box"][0] - 0.02
        midpoint = coverage.periodic_interpolate(
            law, shifted, 0.5, law, self.spec["box"]
        )
        self.assertAlmostEqual(
            coverage._symmetry_aware_distance(law, midpoint, self.spec["box"]),
            0.01,
            places=12,
        )

    def test_16_path_and_diversity_are_symmetry_aware(self):
        path_payload = read_json(coverage.PATH_DIAGNOSTICS_PATH)
        self.assertTrue(path_payload["crossings_are_evaluated_brackets_not_interpolated_claims"])
        self.assertIn("_symmetry_aware_distance", inspect.getsource(coverage._maxmin_combined))

    def test_17_cached_artifacts_are_fully_inventoried(self):
        for row in self.inventory["artifacts"]:
            self.assertEqual(file_sha256(coverage.OUTPUT_ROOT / row["path"]), row["sha256"])

    def test_18_official_v3_firewall_remains_closed(self):
        self.assertFalse(self.summary["official_protocol_created"])
        self.assertFalse(self.summary["selection_frozen"])
        self.assertTrue(all(value is False for value in self.summary["firewall_after"].values()))


if __name__ == "__main__":
    unittest.main()
