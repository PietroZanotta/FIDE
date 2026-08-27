"""Fail-closed tests for the development-only fresh-bank robustness study."""

from __future__ import annotations

import inspect
import unittest

import numpy as np

from . import fresh_bank_robustness as fresh
from .pareto_v3_common import file_sha256, read_json, selection_ceiling


class FreshBankRobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.freeze = read_json(fresh.CANDIDATE_FREEZE_PATH)
        cls.manifest = read_json(fresh.BANK_MANIFEST_PATH)
        cls.bank_inventory = read_json(fresh.BANK_INVENTORY_PATH)
        cls.candidates = read_json(fresh.CANDIDATE_SUMMARY_PATH)
        cls.allowances = read_json(fresh.ALLOWANCE_SUMMARY_PATH)
        cls.failures = read_json(fresh.FAILURE_SUMMARY_PATH)
        cls.summary = read_json(fresh.SUMMARY_PATH)
        cls.inventory = read_json(fresh.INVENTORY_PATH)

    def test_01_candidate_pool_is_exactly_4433_unique_canonical_rows(self):
        self.assertEqual(self.freeze["candidate_count"], 4433)
        self.assertEqual(self.freeze["unique_canonical_geometry_count"], 4433)
        self.assertEqual(len({row["canonical_eta_sha256"] for row in self.freeze["rows"]}), 4433)

    def test_02_candidate_sources_match_coverage_study(self):
        self.assertEqual(self.freeze["original_v2_membership_count"], 337)
        self.assertEqual(self.freeze["coverage_v1_membership_count"], 4096)
        for name, digest in fresh.EXPECTED_COVERAGE_HASHES.items():
            self.assertEqual(file_sha256(fresh.COVERAGE_ROOT / name), digest)

    def test_03_no_candidate_generation_path_exists(self):
        source = inspect.getsource(fresh)
        self.assertNotIn("generate_candidate_pool", source)
        self.assertFalse(self.freeze["candidate_generation_permitted"])

    def test_04_manifest_contains_32_independent_pairs(self):
        self.assertEqual(self.manifest["replicate_count"], 32)
        self.assertEqual(self.manifest["total_bank_count"], 64)
        self.assertEqual(len(self.manifest["replicates"]), 32)
        self.assertEqual(self.bank_inventory["bank_count"], 64)

    def test_05_all_fresh_seeds_are_unique(self):
        seeds = [
            row[f"{role}_seed"]["seed"]
            for row in self.manifest["replicates"]
            for role in ("screen", "audit")
        ]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertTrue(self.manifest["all_seeds_unique"])

    def test_06_fresh_seeds_differ_from_v2(self):
        self.assertTrue(self.manifest["fresh_seeds_disjoint_from_v2"])

    def test_07_manifest_precedes_candidate_evaluation(self):
        first_screen = fresh._stage_paths(0, "screen")[0]
        self.assertLessEqual(fresh.BANK_MANIFEST_PATH.stat().st_mtime_ns, first_screen.stat().st_mtime_ns)
        self.assertLessEqual(fresh.BANK_INVENTORY_PATH.stat().st_mtime_ns, first_screen.stat().st_mtime_ns)
        self.assertTrue(self.manifest["prospectively_frozen_before_candidate_evaluation"])

    def test_08_scientific_constants_are_unchanged(self):
        constants = self.summary["scientific_constants"]
        self.assertEqual(constants["fixed_law_selection_risk"], 5.186549474478042)
        self.assertEqual(constants["allowances_percent"], [0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(constants["projection_tolerance"], 2e-6)
        self.assertEqual(constants["forcing_mean_tolerance"], 2e-7)
        self.assertEqual(constants["maximum_covariance_condition"], 1e10)

    def test_09_ress_gate_is_exact(self):
        self.assertEqual(fresh.MINIMUM_RESS, 0.05)
        self.assertEqual(self.summary["scientific_constants"]["minimum_ress"], 0.05)

    def test_10_risk_arithmetic_is_authoritative(self):
        for row in self.summary["allowances"]:
            self.assertEqual(
                row["risk_ceiling"],
                selection_ceiling(fresh.LAW_RISK, row["allowance_percent"]),
            )
        self.assertFalse(self.summary["risk_semantics"]["fresh_bank_dependent_risk_recomputed"])

    def test_11_screen_and_audit_within_pair_are_independent(self):
        for row in self.manifest["replicates"]:
            self.assertNotEqual(row["screen_seed"]["seed"], row["audit_seed"]["seed"])
        self.assertTrue(self.bank_inventory["pairwise_distinct_initial_state_hashes"])

    def test_12_no_validation_access(self):
        self.assertNotIn("load_validation_galerkin_data", inspect.getsource(fresh))
        self.assertFalse(self.summary["validation_accessed"])

    def test_13_no_tangent_optimizer(self):
        self.assertNotIn("select_tangent", inspect.getsource(fresh))
        self.assertFalse(self.summary["tangent_optimization_run"])

    def test_14_no_full_kf_or_eigensolve(self):
        source = inspect.getsource(fresh)
        self.assertNotIn("FullContext", source)
        self.assertNotIn("GalerkinSystem", source)
        self.assertNotIn("rank_aware_quadratic_solve", source)
        self.assertFalse(self.summary["full_kf_constructed"])
        self.assertFalse(self.summary["eigensolve_run"])

    def test_15_no_deep_ritz(self):
        self.assertNotIn("from .deep_ritz", inspect.getsource(fresh))
        self.assertFalse(self.summary["deep_ritz_run"])

    def test_16_no_official_protocol_or_selection(self):
        self.assertFalse(self.summary["official_protocol_created"])
        self.assertFalse(self.summary["selection_frozen"])
        self.assertTrue(all(value is False for value in self.summary["firewall_after"].values()))

    def test_17_pass_fractions_equal_integer_counts_over_32(self):
        for candidate in self.candidates["rows"]:
            for row in candidate["allowances"]:
                self.assertEqual(
                    row["pass_fraction"], row["complete_dual_bank_pass_count"] / 32
                )

    def test_18_failure_modes_partition_every_old_witness(self):
        for row in self.failures["old_0p5_percent_witnesses"]:
            self.assertEqual(sum(row["failure_modes"].values()), 32)

    def test_19_diversity_is_symmetry_aware(self):
        self.assertIn("_symmetry_aware_distance", inspect.getsource(fresh._maxmin_shortlist))

    def test_20_controlling_time_indices_are_prospective(self):
        with np.load(fresh._stage_paths(0, "screen")[0], allow_pickle=False) as arrays:
            indices = arrays["controlling_ress_time_index"]
        self.assertTrue(np.all((indices >= 0) & (indices < 13)))

    def test_21_resumability_verifies_every_hash(self):
        for row in self.inventory["artifacts"]:
            self.assertEqual(file_sha256(fresh.OUTPUT_ROOT / row["path"]), row["sha256"])

    def test_22_summary_is_sealed_deterministically(self):
        summary_row = next(
            row for row in self.inventory["artifacts"] if row["path"] == "summary.json"
        )
        self.assertEqual(file_sha256(fresh.SUMMARY_PATH), summary_row["sha256"])

    def test_23_all_replicates_completed(self):
        self.assertEqual(self.summary["replicate_count"], 32)
        self.assertEqual(len(self.summary["cache_resume"]["completed"]), 32)

    def test_24_candidate_pool_precedes_all_banks(self):
        self.assertLessEqual(
            fresh.CANDIDATE_FREEZE_PATH.stat().st_mtime_ns,
            min((fresh.OUTPUT_ROOT / row["path"]).stat().st_mtime_ns for row in self.bank_inventory["banks"]),
        )


if __name__ == "__main__":
    unittest.main()
