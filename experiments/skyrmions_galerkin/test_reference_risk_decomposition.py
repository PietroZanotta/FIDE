from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest

import numpy as np

from . import reference_risk_decomposition as study
from .pareto_v3_common import file_sha256


class ReferenceRiskDecompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(study.__file__).read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.summary = json.loads(study.SUMMARY_PATH.read_text(encoding="utf-8"))
        cls.seal = json.loads(study.SOURCE_SEAL_PATH.read_text(encoding="utf-8"))

    def test_01_upstream_artifacts_remain_byte_identical(self):
        for relative, expected in self.seal["input_hashes"].items():
            self.assertEqual(file_sha256(study.REPO_ROOT / relative), expected)

    def test_02_exactly_seven_existing_checkpoints(self):
        hashes = self.seal["checkpoint_hashes"]
        self.assertEqual(set(hashes), set(study.MODEL_LABELS))
        self.assertEqual(len(set(hashes.values())), 7)
        self.assertTrue(all(study._checkpoint_path(label).exists() for label in study.MODEL_LABELS))

    def test_03_immutable_baseline_hash(self):
        self.assertEqual(
            file_sha256(study.BASELINE_CHECKPOINT_PATH),
            study.EXPECTED_BASELINE_CHECKPOINT_SHA256,
        )

    def test_04_no_training_or_new_random_seed_path(self):
        imported = {
            alias.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        self.assertFalse(any(name.startswith("train") for name in imported))
        self.assertNotIn("jax.random", self.source)
        self.assertNotIn("default_rng", self.source)
        self.assertEqual(self.summary["guardrails"]["new_reference_training"], 0)
        self.assertEqual(self.summary["guardrails"]["new_reference_seeds"], 0)

    def test_05_no_candidate_generation_or_validation_member_access(self):
        called_names = {
            node.func.id
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertFalse(any(name.startswith("optimize_") for name in called_names))
        self.assertNotIn('arrays["validation"]', self.source)
        self.assertNotIn("load_validation", self.source)
        self.assertFalse(self.summary["guardrails"]["validation_accessed"])
        self.assertEqual(self.summary["guardrails"]["candidate_generation"], 0)

    def test_06_fixed_W_time_weights_truth_and_targets_across_all_banks(self):
        keys = ("whitening", "time_weights", "truth_hidden_mean", "sensor_target")
        baseline = study._load_bank("model_00", 0)
        expected = {key: study._array_sha256(baseline[key]) for key in keys}
        for label in study.MODEL_LABELS:
            for bank in range(study.PHASE_A_BANK_COUNT):
                row = study._load_bank(label, bank)
                self.assertEqual({key: study._array_sha256(row[key]) for key in keys}, expected)

    def test_07_time_and_whitened_sums_reproduce_risk(self):
        for label in study.MODEL_LABELS:
            for bank in range(study.PHASE_A_BANK_COUNT):
                row = study._load_bank(label, bank)
                self.assertAlmostEqual(float(row["raw_risk_by_time"].sum()), float(row["raw_total_risk"]), places=10)
                self.assertAlmostEqual(float(row["projected_risk_by_time"].sum()), float(row["projected_total_risk"]), places=10)
                np.testing.assert_allclose(
                    row["raw_whitened_components"].sum(axis=1), row["raw_risk_by_time"], rtol=2e-11, atol=2e-11
                )
                np.testing.assert_allclose(
                    row["projected_whitened_components"].sum(axis=1), row["projected_risk_by_time"], rtol=2e-11, atol=2e-11
                )

    def test_08_raw_risk_is_explicitly_diagnostic(self):
        raw = json.loads(study.RAW_PATH.read_text(encoding="utf-8"))
        law = json.loads(study.LAW_PATH.read_text(encoding="utf-8"))
        self.assertTrue(raw["raw_reference_risk_diagnostic_only"])
        self.assertTrue(law["raw_reference_risk_diagnostic_only"])
        for label in study.MODEL_LABELS:
            for bank in range(study.PHASE_A_BANK_COUNT):
                _, record = study._result_paths(label, bank)
                self.assertTrue(json.loads(record.read_text())["raw_reference_risk_diagnostic_only"])

    def test_09_projected_law_risk_reproduces_upstream_per_bank(self):
        for label in study.MODEL_LABELS:
            for bank in range(study.PHASE_A_BANK_COUNT):
                observed = float(study._load_bank(label, bank)["projected_total_risk"])
                expected = float(study._load_result("a", label, bank)["scientific_risk"][0])
                self.assertAlmostEqual(observed, expected, places=8)

    def test_10_headline_node7_ress_reproduces_00_04_06(self):
        expected = {"model_00": 0.051405, "model_04": 0.117247, "model_06": 0.101545}
        for label, value in expected.items():
            self.assertAlmostEqual(self.summary["models"][label]["node7"]["rESS"], value, places=6)

    def test_11_common_bank_pairing_preserved(self):
        for bank in range(study.PHASE_A_BANK_COUNT):
            records = [json.loads(study._result_paths(label, bank)[1].read_text()) for label in study.MODEL_LABELS]
            self.assertEqual(len({row["seed_record"]["seed"] for row in records}), 1)
            self.assertEqual(len({row["initial_state_sha256"] for row in records}), 1)

    def test_12_no_tangent_full_galerkin_eigensolve_or_ritz_call(self):
        calls = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)
        forbidden = {"tangent", "run_full", "assemble_system", "solve_galerkin", "deep_ritz", "train_ritz"}
        self.assertFalse(forbidden.intersection(calls))
        # The sole `eigh` is the explicitly required 9x9 symmetric square root of W,
        # not a Galerkin K/f scientific eigensolve.
        self.assertEqual(calls.count("eigh"), 1)

    def test_13_cross_benchmark_evidence_paths_exist(self):
        audit = json.loads(study.CROSS_AUDIT_JSON.read_text())
        self.assertEqual(len(audit["what_code_does"]), 4)
        for row in audit["what_code_does"]:
            for relative in row["code_evidence"] + row["report_evidence"]:
                self.assertTrue((study.REPO_ROOT / relative).exists(), relative)

    def test_14_active_nematic_counts_come_from_config_and_artifact(self):
        cfg = json.loads((study.REPO_ROOT / "old_stuff/active_nematic_unbalance_percentage/config.json").read_text())
        audit = json.loads(study.ACTIVE_AUDIT_PATH.read_text())
        self.assertEqual(audit["reference_seeds"], cfg["reference_training"]["seeds"])
        self.assertEqual(audit["physical_view_count"], cfg["robust_selection"]["design_views"])
        self.assertEqual(audit["selection_view_count"], 12)
        self.assertEqual(len(study._active_nematic_law_rows()), 12)

    def test_15_gaussian_and_gyre_handling_has_code_artifact_evidence(self):
        audit = json.loads(study.CROSS_AUDIT_JSON.read_text())
        rows = {row["benchmark"]: row for row in audit["what_code_does"]}
        self.assertIn("src/mfsi/selection.py", rows["Analytic Gaussian mixture"]["code_evidence"])
        self.assertIn("src/mfsi/selection.py", rows["Double gyre"]["code_evidence"])
        self.assertTrue(any("frozen_inputs/manifest.json" in value for value in rows["Analytic Gaussian mixture"]["code_evidence"]))
        self.assertTrue(any("reference_seed_sensitivity/summary.json" in value for value in rows["Double gyre"]["code_evidence"]))

    def test_16_no_official_protocol_output_created(self):
        self.assertFalse(self.summary["no_official_protocol"] is False)
        names = [path.name.lower() for path in study.OUTPUT_ROOT.iterdir()]
        self.assertFalse(any("official" in name or "protocol" in name for name in names))

    def test_17_historical_risk_semantics_no_drift(self):
        semantics = self.summary["risk_semantics"]
        self.assertAlmostEqual(semantics["historical"]["risk"], study.HISTORICAL_LAW_RISK, places=10)
        checks = semantics["identity_checks"]
        self.assertTrue(checks["same_whitening"] and checks["same_truth_means"])
        self.assertTrue(checks["same_sensor_targets"] and checks["same_time_weights"])
        self.assertFalse(checks["risk_definition_drift"])

    def test_18_deterministic_cache_and_summary_hashes(self):
        cfg = json.loads(study.CONFIG_PATH.read_text())
        first = study.recompute_law_bank(cfg, "model_00", 0)
        second = study.recompute_law_bank(cfg, "model_00", 0)
        self.assertTrue(first["cache_hit"] and second["cache_hit"])
        self.assertEqual(first["result_sha256"], second["result_sha256"])
        inventory = json.loads(study.INVENTORY_PATH.read_text())
        self.assertEqual(inventory["summary_sha256"], file_sha256(study.SUMMARY_PATH))

    def test_19_required_artifacts_are_present_and_sealed(self):
        required = [
            study.SOURCE_SEAL_PATH, study.CROSS_AUDIT_JSON, study.CROSS_AUDIT_MD,
            study.SEMANTICS_PATH, study.RAW_PATH, study.PROJECTED_PATH, study.TIME_PATH,
            study.COMPONENT_PATH, study.LAW_PATH, study.PANEL_COMPARISON_PATH,
            study.ACTIVE_AUDIT_PATH, study.OPTIONS_PATH, study.SUMMARY_PATH,
            study.INVENTORY_PATH, study.REPORT_PATH,
        ]
        self.assertTrue(all(path.exists() for path in required))
        inventory = json.loads(study.INVENTORY_PATH.read_text())
        indexed = {row["path"]: row["sha256"] for row in inventory["files"]}
        for path in required:
            if path == study.INVENTORY_PATH:
                continue
            self.assertEqual(indexed[str(path.relative_to(study.OUTPUT_ROOT))], file_sha256(path))


if __name__ == "__main__":
    unittest.main()
