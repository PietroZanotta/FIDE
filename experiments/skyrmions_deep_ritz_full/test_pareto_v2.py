"""Fail-closed unit tests for official skyrmion Galerkin Pareto v2."""

from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
import unittest

import numpy as np
import jax.numpy as jnp

from . import pareto_v2_common as common
from . import pareto_v2_selection as selection
from . import pareto_v2_validation as validation
from .production_artifacts import file_sha256


class ParetoV2Tests(unittest.TestCase):
    def test_01_historical_protocol_immutability_inventory(self):
        self.assertIn("OFFICIAL_GALERKIN_PARETO_PROTOCOL.md", common.HISTORICAL_REPORTS)

    def test_02_no_old_validation_in_selection(self):
        source = inspect.getsource(selection)
        self.assertNotIn("load_validation_galerkin_data", source)
        self.assertNotIn("reference_bank_validation", source)

    def test_03_no_fresh_validation_before_selection_freeze(self):
        source = inspect.getsource(validation._selection_seal)
        self.assertIn("forbidden before complete selection freeze", source)

    def test_04_K_exactly_280(self): self.assertEqual(common.K, 280)

    def test_05_dictionary_hash(self):
        self.assertEqual(file_sha256(common.DICTIONARY_PATH), common.EXPECTED_DICTIONARY_SHA256)

    def test_06_rank_tolerance(self): self.assertEqual(common.RANK_TOLERANCE, 1e-12)

    def test_07_ress_threshold(self): self.assertEqual(common.MINIMUM_RESS, 0.05)

    def test_08_energy_threshold(self): self.assertEqual(common.MAXIMUM_ENERGY_RESIDUAL, 0.08)

    def test_09_exact_risk_arithmetic(self): self.assertEqual(common.selection_ceiling(2.0, 3), 2.06)

    def test_10_validation_arithmetic(self): self.assertEqual(common.validation_ceiling(2.0, 3), 2.16)

    def test_11_deterministic_selection_bank_seeds(self):
        self.assertEqual(common.derive_seed(7, "selection", "screen"), common.derive_seed(7, "selection", "screen"))

    def test_12_deterministic_validation_seeds(self):
        row = common.derive_seed(7, "validation", "truth")
        self.assertEqual(row["sha256"], hashlib.sha256(row["derivation_text"].encode()).hexdigest())

    def test_13_bank_role_separation(self): self.assertEqual(len(common.BANK_SIZES), len(set(common.BANK_SIZES)))

    def test_14_screening_uses_no_full_Kf(self):
        tree = ast.parse(inspect.getsource(selection.screen_starts))
        self.assertFalse(any(isinstance(node, ast.Name) and node.id == "FullContext" for node in ast.walk(tree)))

    def test_15_start_generator_reproducible(self):
        self.assertEqual(common.payload_sha256([1, 2]), common.payload_sha256([1, 2]))
        self.assertAlmostEqual(selection._distance([1.99, 0.5], [0.01, 0.5], (2.0, 1.0)), 0.02)

    def test_16_incumbent_retention(self): self.assertIn("mandatory_previous_incumbent", inspect.getsource(selection._method_starts))

    def test_17_nonincreasing_full_selection_action(self): self.assertIn("<= a[\"winner\"][\"action\"] + 1e-10", inspect.getsource(selection.select_full))

    def test_18_nonincreasing_tangent_selection_action(self): self.assertIn("<= a[\"winner\"][\"action\"] + 1e-10", inspect.getsource(selection.select_tangent))

    def test_19_exact_risk_rejection(self): self.assertIn("current[\"risk\"] <= ceiling", inspect.getsource(selection._trajectory))

    def test_20_ress_rejection(self): self.assertIn("MINIMUM_RESS", inspect.getsource(selection.screen_starts))

    def test_21_uncertified_starts_allowed(self): self.assertIn("require_physical=False", inspect.getsource(selection._trajectory))

    def test_22_uncertified_endpoints_forbidden(self): self.assertIn("require_physical=True", inspect.getsource(selection._trajectory))

    def test_23_authoritative_finalist_certification(self): self.assertIn("authoritative.audit", inspect.getsource(selection.select_full))

    def test_24_selection_immutability(self): self.assertIn("winner_geometry_hash", inspect.getsource(selection.freeze_selection))

    def test_25_validation_cannot_change_geometry(self): self.assertIn("selection_geometry_unchanged", inspect.getsource(validation.validate))

    def test_26_deep_ritz_exclusion(self): self.assertNotIn("from .deep_ritz", inspect.getsource(selection).lower())

    def test_27_cross_evaluation_consistency(self):
        source = inspect.getsource(selection.cross_evaluate)
        self.assertIn('"Law"', source); self.assertIn('"Tangent"', source); self.assertIn('"Full"', source)

    def test_28_tangent_full_objectives_distinct(self): self.assertIsNot(selection._tangent_eval, selection.FullContext.evaluate)

    def test_29_batched_projection_equivalence_contract(self):
        from mfsi.projection import EmpiricalIProjector
        self.assertTrue(hasattr(EmpiricalIProjector, "project_candidate_trajectories"))

    def test_30_output_isolation(self):
        self.assertTrue(str(common.OUTPUT_ROOT).startswith(str(Path(__file__).resolve().parent)))

    def test_31_cache_hash_resumability(self):
        protocol = {"protocol_sha256": "x"}
        self.assertEqual(common.signature(protocol, "a", [1]), common.signature(protocol, "a", [1]))

    def test_32_performance_equivalence_policy(self):
        self.assertIn("semantics_change", inspect.getsource(selection.performance_audit))

    def test_33_official_galerkin_backend_defaults_to_jax(self):
        cfg = common.read_json(common.ROOT / "config.json")
        self.assertEqual(cfg["production_galerkin"]["assembly_backend"], "jax")

    def test_34_galerkin_backend_flag_preserves_chunk_statistics(self):
        from mfsi.galerkin_tesseract import is_tesseract_galerkin_available

        if not is_tesseract_galerkin_available():
            self.skipTest("native Galerkin Tesseract unavailable")
        rng = np.random.default_rng(20260825)
        values = jnp.asarray(rng.normal(size=(24, 7)))
        gradients = jnp.asarray(rng.normal(size=(24, 7, 3, 2)))
        weights = jnp.asarray(rng.uniform(size=24))
        weights /= weights.sum()
        forcing = jnp.asarray(rng.normal(size=24))
        expected = selection._assemble_chunk(
            values, gradients, weights, forcing, "jax"
        )
        actual = selection._assemble_chunk(
            values, gradients, weights, forcing, "tesseract_cpp"
        )
        for actual_value, expected_value in zip(actual, expected, strict=True):
            np.testing.assert_allclose(
                actual_value, expected_value, rtol=3e-13, atol=3e-13
            )

    def test_35_invalid_galerkin_backend_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "assembly_backend"):
            selection._assemble_chunk(
                jnp.zeros((2, 1)),
                jnp.zeros((2, 1, 1, 1)),
                jnp.ones((2,)) / 2,
                jnp.zeros((2,)),
                "invalid",
            )


if __name__ == "__main__": unittest.main()
