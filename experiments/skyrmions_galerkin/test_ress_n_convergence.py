"""Focused tests for the nested-N rESS convergence diagnostic."""

from __future__ import annotations

import inspect
import json
import unittest

import numpy as np

from mfsi.config import load_config

from . import ress_n_convergence as convergence
from .pareto_v3_common import file_sha256


class RessNConvergenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.panel = json.loads(convergence.PANEL_PATH.read_text(encoding="utf-8"))
        cls.manifest = json.loads(convergence.MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.summary = json.loads(convergence.SUMMARY_PATH.read_text(encoding="utf-8"))

    def test_01_source_hashes(self) -> None:
        seal = json.loads(convergence.SOURCE_SEAL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(seal["fresh_bank_hashes"], convergence.EXPECTED_FRESH_HASHES)
        self.assertEqual(seal["preflight_hashes"], convergence.EXPECTED_PREFLIGHT_HASHES)
        self.assertEqual(seal["analysis_source_hashes"], convergence._code_hashes())

    def test_02_panel_size_and_roles(self) -> None:
        self.assertEqual(self.panel["candidate_count"], 64)
        self.assertEqual(len(self.panel["rows"]), 64)
        self.assertEqual(sum(row["panel_role"] == "law" for row in self.panel["rows"]), 1)
        self.assertEqual(sum(row["panel_role"] == "high_pass_ge24_of_32" for row in self.panel["rows"]), 55)
        self.assertEqual(sum("control" in row["panel_role"] for row in self.panel["rows"]), 8)

    def test_03_all_high_pass_candidates_present(self) -> None:
        source = json.loads((convergence.FRESH_ROOT / "candidate_robustness_summary.json").read_text())
        expected = {
            row["candidate_id"] for row in source["rows"]
            if row["candidate_id"] != "candidate_000"
            and row["fixed_scientific_selection_risk"] <= convergence.HALF_PERCENT_CEILING
            and convergence._row_pass_count(row) >= 24
        }
        observed = {row["candidate_id"] for row in self.panel["rows"] if row["panel_role"] == "high_pass_ge24_of_32"}
        self.assertEqual(observed, expected)

    def test_04_controls_are_deterministic_and_disjoint(self) -> None:
        controls = [row for row in self.panel["rows"] if "control" in row["panel_role"]]
        self.assertEqual(len({row["candidate_id"] for row in controls}), 8)
        self.assertEqual(self.panel["panel_rows_sha256"], convergence._payload_sha256(self.panel["rows"]))

    def test_05_panel_precedes_manifest_and_banks(self) -> None:
        self.assertLess(convergence.PANEL_PATH.stat().st_mtime_ns, convergence.MANIFEST_PATH.stat().st_mtime_ns)
        earliest_bank = min(path.stat().st_mtime_ns for path in (convergence.OUTPUT_ROOT / "banks").glob("*.npz"))
        self.assertLess(convergence.PANEL_PATH.stat().st_mtime_ns, earliest_bank)

    def test_06_manifest_dimensions_and_unique_seeds(self) -> None:
        self.assertEqual(self.manifest["master_pair_count"], 16)
        self.assertEqual(self.manifest["master_bank_count"], 32)
        seeds = [row["banks"][role]["seed"] for row in self.manifest["replicates"] for role in convergence.ROLES]
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_07_nested_ladder_exact(self) -> None:
        self.assertEqual(self.manifest["N_ladder"], [8192, 16384, 32768, 65536])
        self.assertTrue(self.manifest["nested_prefixes"])
        self.assertEqual(self.manifest["generation_configuration"]["prefix_base_weight_rule"], "slice then renormalize independently at each physical time")

    def test_08_thresholds_unchanged(self) -> None:
        self.assertEqual(convergence.MINIMUM_RESS, 0.05)
        self.assertEqual(self.summary["per_bank_ress_threshold"], 0.05)
        cfg = convergence.load_selection_galerkin_data(
            load_config(convergence.ROOT / "config.json"),
            convergence.ARTIFACT_DIR,
        ).selection_problem.forcing_config
        self.assertEqual(cfg.projection_tolerance, 2e-6)
        self.assertEqual(cfg.forcing_mean_tolerance, 2e-7)
        self.assertEqual(cfg.max_covariance_condition, 1e10)

    def test_09_no_forbidden_work(self) -> None:
        source = inspect.getsource(convergence)
        forbidden = (
            "load_validation_galerkin_data(", "select_tangent(", "select_full(",
            "assemble_full", "run_eigensolve(", "run_deep_ritz(", "freeze_selection(",
        )
        for token in forbidden:
            self.assertNotIn(token, source)
        self.assertTrue(all(value is False for value in self.summary["firewalls"].values()))

    def test_10_controlling_index_is_actual_argmin(self) -> None:
        path = convergence._result_path(0, "A", 8192)
        with np.load(path, allow_pickle=False) as arrays:
            np.testing.assert_array_equal(
                arrays["controlling_time_index"], np.argmin(arrays["ress_trajectory"], axis=1)
            )

    def test_11_true_nested_bank_prefix(self) -> None:
        path = convergence._bank_path(0, "A")
        with np.load(path, allow_pickle=False) as arrays:
            full = arrays["configurations"]
            self.assertTrue(np.array_equal(full[:, :8192], full[:, :16384][:, :8192]))
            self.assertTrue(np.array_equal(full[:, :16384], full[:, :32768][:, :16384]))
            self.assertTrue(np.array_equal(full[:, :32768], full[:, :65536][:, :32768]))

    def test_12_result_dimensions_and_trajectory_count(self) -> None:
        self.assertEqual(self.summary["total_candidate_bank_N_trajectories"], 8192)
        for N in convergence.N_LADDER:
            with np.load(convergence._result_path(0, "A", N), allow_pickle=False) as arrays:
                self.assertEqual(arrays["ress_trajectory"].shape, (64, 13))

    def test_13_pair_inventories_sealed(self) -> None:
        for replicate in range(16):
            inventory = json.loads((convergence._result_root(replicate) / "pair_inventory.json").read_text())
            for row in inventory["artifacts"]:
                self.assertEqual(file_sha256(convergence.OUTPUT_ROOT / row["path"]), row["sha256"])

    def test_13b_master_bank_inventory_sealed(self) -> None:
        inventory = json.loads(convergence.BANK_INVENTORY_PATH.read_text())
        self.assertEqual(inventory["bank_count"], 32)
        for row in inventory["banks"]:
            self.assertEqual(file_sha256(convergence.OUTPUT_ROOT / row["path"]), row["sha256"])

    def test_14_summary_inventory_sealed(self) -> None:
        inventory = json.loads(convergence.INVENTORY_PATH.read_text())
        for row in inventory["artifacts"]:
            self.assertEqual(file_sha256(convergence.OUTPUT_ROOT / row["path"]), row["sha256"])

    def test_15_interpretation_allowed(self) -> None:
        self.assertIn(
            self.summary["interpretation"]["label"],
            {
                "N_LIMITED_SUPPORT_ESTIMATION", "BORDERLINE_POPULATION_OVERLAP",
                "PERSISTENT_REFERENCE_PROPOSAL_MISMATCH", "MIXED_N_AND_PROPOSAL_EFFECT",
            },
        )

    def test_16_sealed_summary_cache_is_reproducible(self) -> None:
        summary_hash = file_sha256(convergence.SUMMARY_PATH)
        inventory_hash = file_sha256(convergence.INVENTORY_PATH)
        cached = convergence._verify_cached_summary()
        self.assertIsNotNone(cached)
        self.assertTrue(cached["cache_hit"])
        self.assertEqual(file_sha256(convergence.SUMMARY_PATH), summary_hash)
        self.assertEqual(file_sha256(convergence.INVENTORY_PATH), inventory_hash)


if __name__ == "__main__":
    unittest.main()
