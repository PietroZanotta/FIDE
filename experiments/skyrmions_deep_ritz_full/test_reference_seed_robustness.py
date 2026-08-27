"""Focused safeguards for the endpoint-only reference-seed study."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import unittest

import numpy as np

from mfsi.config import load_config

from . import reference_seed_robustness as study
from .pareto_v3_common import file_sha256


class ReferenceSeedRobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config(Path(study.__file__).with_name("config.json"))
        cls.seal = json.loads(study.SOURCE_SEAL_PATH.read_text(encoding="utf-8"))
        cls.manifest = json.loads(study.MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_01_expected_source_hashes_match(self) -> None:
        self.assertEqual(
            self.seal["n_convergence_expected_and_observed_hashes"],
            study.EXPECTED_N_CONVERGENCE_HASHES,
        )
        self.assertEqual(self.seal["analysis_source_hashes"], study._code_hashes())
        self.assertEqual(
            file_sha256(study.BASELINE_CHECKPOINT_PATH),
            study.EXPECTED_BASELINE_CHECKPOINT_SHA256,
        )

    def test_02_candidate_panel_is_exact_immutable_copy(self) -> None:
        self.assertEqual(file_sha256(study.PANEL_PATH), study.EXPECTED_PANEL_SHA256)
        self.assertEqual(study.PANEL_PATH.read_bytes(), study.PANEL_SOURCE_PATH.read_bytes())
        panel = json.loads(study.PANEL_PATH.read_text(encoding="utf-8"))
        self.assertEqual((panel["candidate_count"], len(panel["rows"])), (64, 64))

    def test_03_exactly_six_new_training_seeds_were_prefrozen(self) -> None:
        rows = self.manifest["new_training_seeds"]
        self.assertEqual(len(rows), 6)
        self.assertEqual(len({row["seed"] for row in rows}), 6)
        self.assertEqual(set(self.manifest["new_training_configs"]), set(study.NEW_MODEL_LABELS))

    def test_04_original_checkpoint_is_immutable_and_outside_output(self) -> None:
        self.assertFalse(study.OUTPUT_ROOT.resolve() in study.BASELINE_CHECKPOINT_PATH.resolve().parents)
        self.assertTrue(self.seal["baseline_checkpoint_immutable"])
        self.assertEqual(file_sha256(study.BASELINE_CHECKPOINT_PATH), self.manifest["baseline"]["checkpoint_sha256"])

    def test_05_all_training_configs_differ_only_by_seed(self) -> None:
        baseline = self.manifest["baseline"]["training_config"]
        for label, current in self.manifest["new_training_configs"].items():
            differences = {key for key in current if current[key] != baseline[key]}
            self.assertEqual(differences, {"seed"}, label)
            self.assertEqual(current["hidden_width"], 64)
            self.assertEqual(current["hidden_layers"], 3)
            self.assertEqual(current["bridge_noise_std"], 0.01)

    def test_06_training_schedule_and_optimizer_inputs_unchanged(self) -> None:
        baseline = self.manifest["baseline"]["training_config"]
        self.assertEqual(baseline["train_steps"], 6000)
        self.assertEqual(baseline["batch_size"], 512)
        self.assertEqual(baseline["learning_rate"], 8e-4)
        self.assertEqual(baseline["min_learning_rate_ratio"], 0.08)
        self.assertEqual(baseline["grad_clip_norm"], 8.0)
        reference_source = (Path(study.__file__).with_name("reference.py")).read_text(encoding="utf-8")
        self.assertIn("class _AdamState", reference_source)
        self.assertIn("beta1, beta2 = 0.9, 0.999", reference_source)

    def test_07_endpoint_only_training_firewall(self) -> None:
        source = inspect.getsource(study)
        self.assertNotIn('arrays["validation"]', source)
        self.assertNotIn("load_validation_galerkin_data", source)
        self.assertIn('arrays["endpoint0"]', source)
        self.assertIn('arrays["endpoint1"]', source)
        self.assertFalse(self.manifest["intermediate_truth_training_permitted"])
        self.assertFalse(self.manifest["validation_access_permitted"])

    def test_08_all_bank_seeds_prefrozen_and_phase_disjoint(self) -> None:
        phase_a = self.manifest["phase_a"]["common_bank_seeds"]
        phase_b = self.manifest["phase_b"]["common_bank_seeds"]
        self.assertEqual(len(phase_a), 8)
        self.assertEqual(len(phase_b), 4)
        self.assertTrue({row["seed"] for row in phase_a}.isdisjoint({row["seed"] for row in phase_b}))
        self.assertLess(study.MANIFEST_PATH.stat().st_mtime_ns, min(study._model_record_path(label).stat().st_mtime_ns for label in study.NEW_MODEL_LABELS) if all(study._model_record_path(label).exists() for label in study.NEW_MODEL_LABELS) else 2**63 - 1)

    def test_09_thresholds_and_rollout_are_unchanged(self) -> None:
        constants = self.manifest["fixed_constants"]
        self.assertEqual(constants["minimum_rESS"], 0.05)
        self.assertEqual(constants["projection_tolerance"], 2e-6)
        self.assertEqual(constants["forcing_mean_tolerance"], 2e-7)
        self.assertEqual(constants["maximum_covariance_condition"], 1e10)
        self.assertEqual(constants["reference_substeps"], 14)

    def test_10_no_forbidden_scientific_operations(self) -> None:
        source = inspect.getsource(study)
        forbidden_calls = (
            "random_sensor_designs(",
            "local_sensor_designs(",
            "select_tangent(",
            "select_full(",
            "assemble_galerkin",
            "eigh(",
            "eigsh(",
            "run_deep_ritz(",
            "freeze_selection(",
        )
        for token in forbidden_calls:
            self.assertNotIn(token, source)
        self.assertFalse(self.manifest["candidate_generation_permitted"])
        self.assertFalse(self.manifest["geometry_optimization_permitted"])
        self.assertFalse(self.manifest["tangent_full_galerkin_eigensolve_deep_ritz_permitted"])
        self.assertFalse(self.manifest["official_outputs_permitted"])

    def test_11_risk_and_measurement_implementations_are_sealed(self) -> None:
        hashes = self.seal["immutable_source_hashes"]
        self.assertEqual(hashes["scientific_risk_source"], file_sha256(Path(study.__file__).with_name("risk.py")))
        self.assertEqual(hashes["measurement_reconstruction_and_projected_risk_source"], file_sha256(Path(study.__file__).with_name("full_gradient.py")))
        self.assertTrue(self.manifest["measurement_targets_fixed_across_models"])

    def test_12_all_six_models_complete_when_training_artifacts_exist(self) -> None:
        if not all(study._model_record_path(label).exists() for label in study.NEW_MODEL_LABELS):
            self.skipTest("training stage not complete yet")
        hashes = []
        for label in study.NEW_MODEL_LABELS:
            record = json.loads(study._model_record_path(label).read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "complete")
            self.assertEqual(record["training_steps_completed"], 6000)
            self.assertEqual(record["training_config"]["batch_size"], 512)
            self.assertTrue(record["endpoint_only"])
            self.assertFalse(record["intermediate_truth_used"])
            self.assertEqual(record["checkpoint_sha256"], file_sha256(study._checkpoint_path(label)))
            hashes.append(record["checkpoint_sha256"])
        self.assertEqual(len(hashes), len(set(hashes)))
        self.assertNotIn(study.EXPECTED_BASELINE_CHECKPOINT_SHA256, hashes)

    def _check_common_hashes(self, path: Path, expected_models: int) -> None:
        if not path.exists():
            self.skipTest(f"bank stage not complete yet: {path.name}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for bank in manifest["banks"]:
            rows = bank["model_results"]
            self.assertEqual(len(rows), expected_models)
            self.assertEqual(len({row["initial_state_sha256"] for row in rows}), 1)
            self.assertEqual(len({row["measurement_target_sha256"] for row in rows}), 1)

    def test_13_phase_a_common_initials_and_measurements(self) -> None:
        self._check_common_hashes(study.PHASE_A_BANK_MANIFEST_PATH, 7)

    def test_14_phase_b_common_initials_and_measurements(self) -> None:
        self._check_common_hashes(study.PHASE_B_BANK_MANIFEST_PATH, 3)

    def test_15_ranking_is_deterministic_and_advances_exactly_two(self) -> None:
        if not study.RANKING_PATH.exists():
            self.skipTest("ranking stage not complete yet")
        ranking = json.loads(study.RANKING_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(ranking["selected_models"]), 2)
        self.assertTrue(set(ranking["selected_models"]).issubset(set(study.NEW_MODEL_LABELS)))
        self.assertEqual(ranking["baseline_always_advances"], study.BASELINE_LABEL)
        self.assertEqual(len(ranking["phase_b_model_labels"]), 3)
        self.assertEqual(ranking["ranking_rule"], self.manifest["ranking_rule"])
        self.assertTrue(ranking["selection_frozen_before_phase_b"])

    def test_16_phase_b_cannot_alter_selection(self) -> None:
        if not study.PHASE_B_SUMMARY_PATH.exists():
            self.skipTest("Phase B summary not complete yet")
        summary = json.loads(study.PHASE_B_SUMMARY_PATH.read_text(encoding="utf-8"))
        self.assertTrue(summary["phase_b_cannot_change_selection"])
        self.assertEqual(summary["phase_a_ranking_sha256"], file_sha256(study.RANKING_PATH))

    def test_17_result_shapes_and_threshold_application(self) -> None:
        if not study.PHASE_A_BANK_MANIFEST_PATH.exists():
            self.skipTest("Phase A not complete yet")
        values = study._load_result("a", study.BASELINE_LABEL, 0)
        self.assertEqual(values["ress_trajectory"].shape, (64, 13))
        self.assertEqual(values["lambda_norm"].shape, (64, 13))
        self.assertEqual(values["scientific_risk"].shape, (64,))
        self.assertTrue(np.array_equal(values["ress_valid"], values["minimum_ress"] >= 0.05))

    def test_18_final_summary_firewall_and_inventory(self) -> None:
        if not study.SUMMARY_PATH.exists():
            self.skipTest("final summary not complete yet")
        summary = json.loads(study.SUMMARY_PATH.read_text(encoding="utf-8"))
        self.assertFalse(summary["intermediate_truth_used"])
        self.assertFalse(summary["validation_accessed"])
        self.assertFalse(summary["tangent_run"])
        self.assertFalse(summary["full_run"])
        self.assertFalse(summary["galerkin_constructed"])
        self.assertFalse(summary["eigensolve_run"])
        self.assertFalse(summary["deep_ritz_run"])
        self.assertFalse(summary["official_reference_replaced"])
        self.assertFalse(summary["official_protocol_created"])
        inventory = json.loads(study.INVENTORY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(inventory["artifact_count"], len(inventory["artifacts"]))
        for row in inventory["artifacts"]:
            self.assertEqual(row["sha256"], file_sha256(study.OUTPUT_ROOT / row["path"]))

    def test_19_cached_analysis_reproduces_hashes(self) -> None:
        if not study.SUMMARY_PATH.exists():
            self.skipTest("final summary not complete yet")
        before = (file_sha256(study.SUMMARY_PATH), file_sha256(study.REPORT_PATH), file_sha256(study.INVENTORY_PATH))
        result = study.summarize(self.cfg)
        after = (file_sha256(study.SUMMARY_PATH), file_sha256(study.REPORT_PATH), file_sha256(study.INVENTORY_PATH))
        self.assertTrue(result["cache_hit"])
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
