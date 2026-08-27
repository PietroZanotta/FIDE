from __future__ import annotations

import inspect
import json
from pathlib import Path
import unittest

import numpy as np

from . import ess_study as study


class EssStudyContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parent
        self.source = (self.root / "ess_study.py").read_text(encoding="utf-8")

    def test_01_no_validation_access(self) -> None:
        self.assertNotIn("load_validation", self.source)
        self.assertNotIn("official_pareto_validation", self.source)
        self.assertNotIn("reference_bank_validation", self.source)

    def test_02_no_eta_full_optimization(self) -> None:
        self.assertNotIn("optimize", inspect.getsource(study.run_candidate_screen).lower())
        self.assertNotIn("optimizer", inspect.getsource(study.run_staged_rescore).lower())

    def test_03_historical_output_immutability_frozen(self) -> None:
        if not study.PROTOCOL_PATH.exists(): self.skipTest("protocol not frozen yet")
        protocol = json.loads(study.PROTOCOL_PATH.read_text())
        for name, expected in protocol["historical_immutable_sha256"].items():
            self.assertEqual(study.file_sha256(self.root / name), expected)

    def test_04_fixed_K280_metadata(self) -> None:
        self.assertEqual(study.EXPECTED_DICTIONARY_SHA256,
                         "37e9b60fcb92c4e5a0ee7ec1651fb7f8889f7ac6bdb02d3bd314e9ef40833326")

    def test_05_unchanged_ress_threshold(self) -> None:
        self.assertEqual(study.RESS_THRESHOLD, .05)

    def test_06_unchanged_energy_threshold(self) -> None:
        self.assertEqual(study.ENERGY_THRESHOLD, .08)

    def test_07_exact_ess_formula(self) -> None:
        w = np.asarray([[.5, .25, .25]])
        b = np.full_like(w, 1/3)
        absolute, relative, base = study.exact_ess(w, b)
        self.assertAlmostEqual(absolute[0], 1/(.5**2+.25**2+.25**2), places=14)
        self.assertAlmostEqual(base[0], 3, places=14)
        self.assertAlmostEqual(relative[0], absolute[0]/3, places=14)

    def test_08_absolute_relative_consistency(self) -> None:
        rng = np.random.default_rng(7)
        w = rng.random((13, 17)); w /= w.sum(axis=1, keepdims=True)
        absolute, relative, base = study.exact_ess(w, np.full((13,17), 1/17))
        np.testing.assert_allclose(relative, absolute/17, rtol=0, atol=1e-15)
        np.testing.assert_allclose(base, 17, rtol=0, atol=1e-14)

    def test_09_deterministic_replicate_seeds(self) -> None:
        a = study.derive_seed(20260822, 8192, 0)
        b = study.derive_seed(20260822, 8192, 0)
        c = study.derive_seed(20260822, 8192, 1)
        self.assertEqual(a, b); self.assertNotEqual(a["seed"], c["seed"])
        self.assertEqual(a["text"], "20260822:skyrmion:ess_qualification:v1:N8192:rep0")

    def test_10_candidate_pool_reproducibility_manifest(self) -> None:
        path = study.OUTPUT_ROOT / "candidate_pool" / "manifest.json"
        if not path.exists(): self.skipTest("candidate pool not run yet")
        payload = json.loads(path.read_text())
        self.assertEqual(payload["pool_count"], 337)
        self.assertEqual(study.payload_sha256(payload["candidates"]), payload["pool_sha256"])

    def test_11_exact_selection_risk_arithmetic(self) -> None:
        law = 5.0
        for allowance in study.ALLOWANCES:
            self.assertAlmostEqual((1+allowance/100)*law, law + allowance*law/100, places=14)

    def test_12_staged_filtering_logic(self) -> None:
        rows = [{"candidate_id": f"c{i}", "scientific_selection_risk": 1.0,
                 "geometry_valid": True, "projection_valid": True,
                 "minimum_ess_fraction": value}
                for i, value in enumerate((.03, .039, .04, .06))]
        selected = study._stage_b_ids({"rows": rows}, 1.0)
        self.assertIn("c2", selected); self.assertIn("c3", selected)

    def test_13_no_full_action_in_bulk_screen(self) -> None:
        body = inspect.getsource(study.run_candidate_screen) + inspect.getsource(study.score_candidates)
        self.assertNotIn("assemble", body)
        self.assertNotIn("rank_aware", body)
        self.assertNotIn("action", body.lower())

    def test_14_batched_scalar_ess_equivalence(self) -> None:
        path = study.OUTPUT_ROOT / "performance" / "benchmark.json"
        if not path.exists(): self.skipTest("performance audit not run yet")
        row = json.loads(path.read_text())["candidate_preprocessing"]
        self.assertLessEqual(row["max_absolute_discrepancy"], 1e-12)
        self.assertLessEqual(row["max_ess_discrepancy_after_projection"], 1e-12)

    def test_15_before_after_numerical_equivalence(self) -> None:
        path = study.OUTPUT_ROOT / "performance" / "benchmark.json"
        if not path.exists(): self.skipTest("performance audit not run yet")
        self.assertTrue(json.loads(path.read_text())["historical_before_after_equivalence"]["passed"])

    def test_16_output_path_isolation(self) -> None:
        self.assertEqual(study.require_output_path(study.OUTPUT_ROOT / "x"),
                         (study.OUTPUT_ROOT / "x").resolve())
        with self.assertRaises(ValueError): study.require_output_path(self.root / "x")

    def test_17_no_deep_ritz_invocation(self) -> None:
        self.assertNotIn("from .deep_ritz", self.source)
        self.assertNotIn("import deep_ritz", self.source)

    def test_18_no_old_validation_imports(self) -> None:
        run_source = (self.root / "ess_study_run.py").read_text()
        self.assertNotIn("validation", run_source.lower())

    def test_19_performance_benchmark_schema(self) -> None:
        path = study.OUTPUT_ROOT / "performance" / "benchmark.json"
        if not path.exists(): self.skipTest("performance audit not run yet")
        row = json.loads(path.read_text())
        for key in ("device", "candidate_preprocessing", "current_N8192",
                    "current_K280_cached", "historical_before_after_equivalence"):
            self.assertIn(key, row)

    def test_20_summary_consistency(self) -> None:
        path = study.OUTPUT_ROOT / "summary.json"
        if not path.exists(): self.skipTest("report not run yet")
        self.assertTrue(study.verify_summary_consistency(json.loads(path.read_text())))


if __name__ == "__main__":
    unittest.main()
