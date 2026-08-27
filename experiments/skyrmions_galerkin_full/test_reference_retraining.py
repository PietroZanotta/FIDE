"""Focused tests for the three-model reference retraining ensemble."""

from __future__ import annotations

import inspect
import json
import unittest

import numpy as np

from . import reference_retraining as retraining
from .pareto_v3_common import file_sha256


class ReferenceRetrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.seal = json.loads(retraining.SOURCE_SEAL_PATH.read_text(encoding="utf-8"))
        cls.protocol = json.loads(retraining.PROTOCOL_PATH.read_text(encoding="utf-8"))
        cls.summary = json.loads(retraining.SUMMARY_PATH.read_text(encoding="utf-8"))

    def test_01_sources_and_production_checkpoint_unchanged(self) -> None:
        self.assertEqual(self.seal["source_hashes"], retraining.EXPECTED_SOURCE_HASHES)
        self.assertEqual(self.seal["analysis_source_hashes"], retraining._code_hashes())
        self.assertEqual(
            file_sha256(retraining.ORIGINAL_CHECKPOINT_PATH),
            retraining.EXPECTED_SOURCE_HASHES["reference.npz"],
        )

    def test_02_protocol_precedes_generated_data(self) -> None:
        self.assertLess(
            retraining.PROTOCOL_PATH.stat().st_mtime_ns,
            retraining.TRAIN_DATA_PATH.stat().st_mtime_ns,
        )
        self.assertLess(
            retraining.PROTOCOL_PATH.stat().st_mtime_ns,
            retraining.EVAL_DATA_PATH.stat().st_mtime_ns,
        )
        self.assertTrue(self.protocol["frozen_before_data_generation"])

    def test_03_regenerated_datasets_are_disjoint_and_correct_size(self) -> None:
        data = json.loads(retraining.DATA_INVENTORY_PATH.read_text(encoding="utf-8"))
        self.assertTrue(data["seeds_disjoint"])
        self.assertNotEqual(data["training_endpoint_seed"], data["evaluation_truth_seed"])
        with np.load(retraining.TRAIN_DATA_PATH, allow_pickle=False) as arrays:
            self.assertEqual(arrays["endpoint0"].shape, (12_000, 16, 2))
            self.assertEqual(arrays["endpoint1"].shape, (12_000, 16, 2))
        with np.load(retraining.EVAL_DATA_PATH, allow_pickle=False) as arrays:
            self.assertEqual(arrays["configurations"].shape, (13, 6_000, 16, 2))

    def test_04_exactly_three_models_share_every_config_except_seed(self) -> None:
        configs = self.protocol["fresh_training_configs"]
        self.assertEqual(set(configs), set(retraining.MODEL_LABELS))
        original = self.protocol["original_training_config"]
        seeds = []
        for label in retraining.MODEL_LABELS:
            config = configs[label]
            differences = {key for key in config if config[key] != original[key]}
            self.assertEqual(differences, {"seed"})
            seeds.append(config["seed"])
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_05_three_models_finished_and_checkpoints_are_distinct(self) -> None:
        hashes = []
        data_inventory_hashes = []
        for label in retraining.MODEL_LABELS:
            result = json.loads(retraining._training_result_path(label).read_text(encoding="utf-8"))
            self.assertEqual(result["training_steps_completed"], 6000)
            self.assertFalse(result["installed"])
            self.assertEqual(result["checkpoint_sha256"], file_sha256(retraining._checkpoint_path(label)))
            hashes.append(result["checkpoint_sha256"])
            data_inventory_hashes.append(result["data_inventory_sha256"])
        self.assertEqual(len(hashes), len(set(hashes)))
        self.assertEqual(len(set(data_inventory_hashes)), 1)
        self.assertEqual(data_inventory_hashes[0], file_sha256(retraining.DATA_INVENTORY_PATH))
        self.assertNotIn(retraining.EXPECTED_SOURCE_HASHES["reference.npz"], hashes)

    def test_06_matched_banks_use_identical_initial_configurations(self) -> None:
        initial_states = []
        for label in retraining.ALL_FLOW_LABELS:
            with np.load(retraining._bank_path(label), allow_pickle=False) as arrays:
                self.assertEqual(arrays["configurations"].shape, (13, 65_536, 16, 2))
                self.assertEqual(int(arrays["initial_seed"]), retraining.BANK_INITIAL_SEED)
                initial_states.append(np.asarray(arrays["configurations"][0]))
        for state in initial_states[1:]:
            self.assertTrue(np.array_equal(initial_states[0], state))

    def test_07_evaluation_shapes_and_fixed_crn_design(self) -> None:
        fixed = self.protocol["fixed_crn_evaluation"]
        self.assertEqual((fixed["batch_count"], fixed["batch_size"]), (256, 512))
        self.assertTrue(fixed["same_crn_for_all_flows"])
        for label in retraining.ALL_FLOW_LABELS:
            with np.load(retraining._evaluation_path(label), allow_pickle=False) as arrays:
                self.assertEqual(arrays["ress_trajectory"].shape, (64, 13))
                self.assertEqual(arrays["minimum_ress"].shape, (64,))
                self.assertEqual(arrays["truth_moment_error"].shape, (64, 3))
                self.assertEqual(arrays["fixed_crn_loss"].shape, (256,))

    def test_08_candidate_panel_and_threshold_are_unchanged(self) -> None:
        panel = json.loads(retraining.PANEL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(panel["candidate_count"], 64)
        self.assertEqual(len(panel["rows"]), 64)
        self.assertEqual(retraining.MINIMUM_RESS, 0.05)
        self.assertEqual(self.protocol["rESS_threshold"], 0.05)

    def test_09_interpretation_is_predeclared(self) -> None:
        self.assertIn(
            self.summary["development_interpretation"],
            {
                "RETRAINS_CONSISTENTLY_REPAIR_SUPPORT",
                "RETRAINS_IMPROVE_FIT_BUT_SUPPORT_REMAINS",
                "RETRAINS_REPRODUCE_SUPPORT_PROBLEM",
                "REFERENCE_TRAINING_SEED_SENSITIVE",
            },
        )

    def test_10_firewall(self) -> None:
        source = inspect.getsource(retraining)
        for token in (
            "load_validation_galerkin_data(", "select_tangent(", "select_full(",
            "assemble_full", "run_eigensolve(", "run_deep_ritz(", "freeze_selection(",
        ):
            self.assertNotIn(token, source)
        self.assertFalse(self.summary["production_checkpoint_installed"])
        self.assertFalse(self.summary["official_protocol_created"])
        self.assertFalse(self.summary["validation_accessed"])
        self.assertFalse(self.summary["downstream_tangent_or_full_run"])

    def test_11_inventory_is_sealed(self) -> None:
        inventory = json.loads(retraining.INVENTORY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(inventory["artifact_count"], len(inventory["artifacts"]))
        for row in inventory["artifacts"]:
            self.assertEqual(file_sha256(retraining.OUTPUT_ROOT / row["path"]), row["sha256"])

    def test_12_cached_summary_is_reproducible(self) -> None:
        summary_hash = file_sha256(retraining.SUMMARY_PATH)
        inventory_hash = file_sha256(retraining.INVENTORY_PATH)
        cached = retraining._verify_cached_summary()
        self.assertIsNotNone(cached)
        self.assertTrue(cached["cache_hit"])
        self.assertEqual(file_sha256(retraining.SUMMARY_PATH), summary_hash)
        self.assertEqual(file_sha256(retraining.INVENTORY_PATH), inventory_hash)


if __name__ == "__main__":
    unittest.main()
