from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest

import numpy as np

from . import reference_semantics_audit as study
from .pareto_v3_common import file_sha256


class ReferenceSemanticsAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(study.__file__).read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.seal = json.loads(study.SOURCE_SEAL_PATH.read_text())
        cls.summary = json.loads(study.JOINT_SUMMARY_PATH.read_text())

    def test_01_all_source_study_hashes_match(self):
        for relative, digest in self.seal["upstream_direct_hashes"].items():
            self.assertEqual(file_sha256(study.REPO_ROOT / relative), digest)
        study._verify_inventory(study.ROBUST_ROOT)
        study._verify_inventory(study.DECOMP_ROOT)

    def test_02_all_seven_checkpoint_hashes_match(self):
        self.assertEqual(set(self.seal["checkpoint_hashes"]), set(study.MODEL_LABELS))
        for label, digest in self.seal["checkpoint_hashes"].items():
            self.assertEqual(file_sha256(study._checkpoint_path(label)), digest)
        self.assertEqual(self.seal["checkpoint_hashes"]["model_00"], study.EXPECTED_BASELINE_CHECKPOINT_SHA256)

    def test_03_no_training_or_random_seed_generation(self):
        imports = {alias.name for node in ast.walk(self.tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
        self.assertFalse(any(name.startswith("train") for name in imports))
        self.assertNotIn("jax.random", self.source)
        self.assertEqual(self.summary["guardrails"]["new_training"], 0)
        self.assertEqual(self.summary["guardrails"]["new_random_seeds"], 0)

    def test_04_only_endpoint_members_are_read_from_truth_archive(self):
        keys = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "arrays":
                if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                    keys.add(node.slice.value)
        self.assertTrue({"endpoint0", "endpoint1"}.issubset(keys))
        self.assertNotIn("design", keys)
        self.assertNotIn("validation", keys)
        self.assertFalse(self.summary["guardrails"]["intermediate_truth_arrays_accessed"])
        self.assertFalse(self.summary["guardrails"]["validation_accessed"])

    def test_05_no_sensor_generation_or_optimization(self):
        calls = self._calls()
        self.assertFalse(any(name.startswith("random_sensor") or name.startswith("local_sensor") for name in calls))
        self.assertFalse(any(name.startswith("optimize") for name in calls))
        self.assertFalse(self.summary["guardrails"]["sensor_generation"])
        self.assertFalse(self.summary["guardrails"]["sensor_optimization"])

    def test_06_no_tangent_full_galerkin_or_ritz_calls(self):
        calls = set(self._calls())
        forbidden = {"run_tangent", "run_full", "assemble_system", "solve_galerkin", "deep_ritz", "train_ritz"}
        self.assertFalse(calls.intersection(forbidden))
        imports = {node.module or "" for node in ast.walk(self.tree) if isinstance(node, ast.ImportFrom)}
        self.assertFalse(any("deep_ritz" in name or name.endswith("galerkin") for name in imports))
        # The metric's 9x9 diagnostic decomposition is required; no Full eigensolver is imported.
        self.assertEqual(self._calls().count("eigh"), 1)

    def test_07_authoritative_metric_factor_and_time_weights_reused(self):
        definition = study._metric_definition()
        upstream = study._load_law_bank("model_00", 0)
        self.assertTrue(np.array_equal(definition["raw_metric"], upstream["whitening"]))
        np.testing.assert_allclose(definition["metric"], upstream["whitening"], rtol=0, atol=2e-11)
        self.assertTrue(np.array_equal(definition["factor"], upstream["whitener_L"]))
        self.assertTrue(np.array_equal(definition["time_weights"], upstream["time_weights"]))
        self.assertTrue(np.array_equal(definition["metric"], definition["metric"].T))
        np.testing.assert_allclose(definition["factor"].T @ definition["factor"], definition["metric"], rtol=2e-12, atol=2e-12)

    def test_08_authoritative_Psi_source_hash_reused(self):
        definition = json.loads(study.WHITENING_DEFINITION_PATH.read_text())
        self.assertEqual(definition["hashes"]["Psi_source"], file_sha256(study.ROOT / "risk.py"))
        self.assertEqual(len(definition["feature_definitions"]), 9)

    def test_09_factor_metric_and_modal_risks_reconstruct_every_bank(self):
        definition = study._metric_definition()
        spectrum = json.loads(study.WHITENING_SPECTRUM_PATH.read_text())
        modal_payload = json.loads(study.MODAL_PATH.read_text())
        alpha = np.asarray(spectrum["risk_metric_eigenvalues_alpha_descending"])
        vectors = np.asarray(modal_payload["risk_mode_vectors_columns"])
        for label in study.MODEL_LABELS:
            for bank_index in range(8):
                row = study._load_law_bank(label, bank_index)
                for error_key, time_key, total_key in (
                    ("raw_hidden_error", "raw_risk_by_time", "raw_total_risk"),
                    ("projected_hidden_error", "projected_risk_by_time", "projected_total_risk"),
                ):
                    error, omega = row[error_key], row["time_weights"]
                    z = error @ definition["factor"].T
                    metric_time = omega * np.einsum("ti,ij,tj->t", error, definition["metric"], error)
                    modal = omega[:, None] * alpha[None, :] * (error @ vectors) ** 2
                    np.testing.assert_allclose(omega * np.sum(z**2, axis=1), row[time_key], rtol=2e-10, atol=2e-10)
                    np.testing.assert_allclose(metric_time, row[time_key], rtol=2e-10, atol=2e-10)
                    np.testing.assert_allclose(modal.sum(axis=1), row[time_key], rtol=2e-10, atol=2e-10)
                    self.assertAlmostEqual(float(modal.sum()), float(row[total_key]), places=8)

    def test_10_coordinate7_indexing_is_explicit_and_exact(self):
        payload = json.loads(study.COORDINATE7_PATH.read_text())
        convention = payload["indexing_convention"]
        self.assertEqual(convention["reported_coordinate"], 7)
        self.assertEqual(convention["array_indexing"], "zero-based")
        self.assertEqual(convention["human_ordinal"], "eighth whitening-factor row")
        row = study._load_law_bank("model_04", 0)
        receipt = payload["models"]["model_04"]["exact_per_bank_important_nodes"][0]["projected"]["7"]
        terms = np.asarray(list(receipt["signed_w7_times_delta_terms"].values()))
        self.assertAlmostEqual(float(terms.sum()), receipt["z7_direct"], places=11)
        expected = row["projected_hidden_error"][7] @ row["whitener_L"][7]
        self.assertAlmostEqual(expected, receipt["z7_direct"], places=11)

    def test_11_endpoint_common_initial_hash_and_exact_RK4_convention(self):
        results = json.loads(study.ENDPOINT_RESULTS_PATH.read_text())
        self.assertTrue(results["all_models_identical_initial_state_hash"])
        hashes = {row["initial_state_sha256"] for row in results["models"]}
        self.assertEqual(len(hashes), 1)
        for row in results["models"]:
            rollout = row["rollout"]
            self.assertEqual(rollout["integrator"], "deterministic periodic RK4")
            self.assertEqual(rollout["scientific_intervals"], 12)
            self.assertEqual(rollout["substeps_per_scientific_interval"], 14)
            self.assertEqual(rollout["total_RK4_steps"], 168)
            self.assertEqual(rollout["dtype"], "float64")

    def test_12_endpoint_quality_uses_only_final_endpoint_truth(self):
        manifest = json.loads(study.ENDPOINT_MANIFEST_PATH.read_text())
        separation = manifest["data_separation"]
        self.assertFalse(separation["intermediate_truth_used"])
        self.assertFalse(separation["validation_accessed"])
        self.assertFalse(separation["new_truth_simulation"])
        for row in json.loads(study.ENDPOINT_RESULTS_PATH.read_text())["models"]:
            self.assertFalse(row["rollout"]["intermediate_states_compared_to_truth"])
            self.assertEqual(row["diagnostic_independence"], "IN-SAMPLE OR NON-INDEPENDENT ENDPOINT DIAGNOSTIC")

    def test_13_no_reference_selection_or_official_protocol(self):
        self.assertTrue(self.summary["no_reference_replacement"])
        self.assertTrue(self.summary["no_intermediate_truth_model_selection"])
        self.assertTrue(self.summary["no_official_protocol"])
        names = [path.name.lower() for path in study.OUTPUT_ROOT.iterdir()]
        self.assertFalse(any("official" in name or "protocol" in name or "selected_checkpoint" in name for name in names))

    def test_14_deterministic_cached_rerun_reproduces_summary_hashes(self):
        before = (file_sha256(study.JOINT_SUMMARY_PATH), file_sha256(study.REPORT_PATH), file_sha256(study.INVENTORY_PATH))
        study.finalize()
        after = (file_sha256(study.JOINT_SUMMARY_PATH), file_sha256(study.REPORT_PATH), file_sha256(study.INVENTORY_PATH))
        self.assertEqual(before, after)

    def test_15_inventory_seals_all_outputs(self):
        inventory = json.loads(study.INVENTORY_PATH.read_text())
        self.assertEqual(inventory["joint_summary_sha256"], file_sha256(study.JOINT_SUMMARY_PATH))
        for row in inventory["files"]:
            self.assertEqual(file_sha256(study.OUTPUT_ROOT / row["path"]), row["sha256"])

    def _calls(self):
        calls = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)
        return calls


if __name__ == "__main__":
    unittest.main()
