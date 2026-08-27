"""Focused tests for the efficient replicate-gate preflight v2."""

from __future__ import annotations

import inspect
import itertools
import json
import math
import unittest

import numpy as np

from . import replicate_gate_preflight_v2 as preflight
from .pareto_v3_common import file_sha256


class ReplicateGatePreflightV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(preflight.SUMMARY_PATH.read_text(encoding="utf-8"))
        cls.manifest = json.loads(preflight.SUBSET_MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_01_source_hashes_match(self) -> None:
        seal = json.loads(preflight.SOURCE_SEAL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(seal["source_hashes"], preflight.EXPECTED_SOURCE_HASHES)
        for name, digest in seal["source_hashes"].items():
            self.assertEqual(file_sha256(preflight.SOURCE_ROOT / name), digest)

    def test_02_dimensions(self) -> None:
        self.assertEqual(self.summary["candidate_count"], 4433)
        self.assertEqual(self.summary["fresh_development_pairs"], 32)

    def test_03_reconstruction_matches_full32(self) -> None:
        _, counts, _ = preflight.reconstruct_eligibility()
        np.testing.assert_array_equal(np.max(counts, axis=1), [26, 29, 31, 31, 32, 32])
        np.testing.assert_array_equal(np.sum(counts >= 24, axis=1), [55, 310, 957, 1314, 1663, 1886])

    def test_04_no_scientific_work_imports(self) -> None:
        source = inspect.getsource(preflight)
        forbidden = (
            "from mfsi.projection import", "_generate_bank(", "reconstruct_moments(",
            "select_tangent(", "select_full(", "run_eigensolve(",
            "run_deep_ritz(", "load_validation_galerkin_data(",
            "generate_fresh_validation(",
        )
        for token in forbidden:
            self.assertNotIn(token, source)

    def test_05_architecture_grid_exact(self) -> None:
        grid = json.loads(preflight.ARCHITECTURE_GRID_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            [row["label"] for row in grid["architectures"]],
            ["1/1", "2/2", "3/4", "4/4", "6/8", "7/8", "8/8", "12/16", "14/16", "16/16"],
        )

    def test_06_subset_counts_and_uniqueness(self) -> None:
        expected = {1: 32, 2: 496, 4: 35960, 8: 5000, 16: 2500}
        for M, count in expected.items():
            masks = self.manifest["schedules"][str(M)]["subset_masks_uint32"]
            self.assertEqual(len(masks), count)
            self.assertEqual(len(set(masks)), count)
            self.assertTrue(all(int(mask).bit_count() == M for mask in masks))

    def test_07_subset_manifest_is_deterministic(self) -> None:
        frozen = preflight._payload_sha256(self.manifest)
        self.assertEqual(frozen, preflight._payload_sha256(json.loads(preflight.SUBSET_MANIFEST_PATH.read_text())))

    def test_08_hypergeometric_matches_enumeration(self) -> None:
        for successes, M, required in ((0, 4, 3), (17, 4, 3), (26, 4, 3), (31, 2, 2)):
            observed = preflight.exact_hypergeometric_probability(successes, M, required)
            good = set(range(successes))
            hits = sum(len(good.intersection(subset)) >= required for subset in itertools.combinations(range(32), M))
            expected = hits / math.comb(32, M)
            self.assertAlmostEqual(observed, expected, places=15)

    def test_09_popcount_matches_direct(self) -> None:
        rng = np.random.default_rng(77)
        values = rng.integers(0, 2**32, size=(17, 23), dtype=np.uint32)
        expected = np.vectorize(lambda value: int(value).bit_count())(values)
        np.testing.assert_array_equal(preflight.popcount_uint32(values), expected)

    def test_10_bitmask_scoring_matches_boolean_sums(self) -> None:
        rng = np.random.default_rng(91)
        outcomes = rng.integers(0, 2, size=(41, 32), dtype=np.uint8).astype(bool)
        candidate_masks = np.sum(
            outcomes.astype(np.uint32) * np.left_shift(np.uint32(1), np.arange(32, dtype=np.uint32)),
            axis=1,
            dtype=np.uint32,
        )
        subsets = np.asarray([preflight._mask(rng.choice(32, 8, replace=False)) for _ in range(50)], dtype=np.uint32)
        reference = np.sum(outcomes, axis=1) >= 24
        observed, _ = preflight.score_subsets(candidate_masks, subsets, 6, reference, 20)
        expected = np.asarray([
            np.sum(np.sum(outcomes[:, [i for i in range(32) if int(mask) & (1 << i)]], axis=1) >= 6)
            for mask in subsets
        ])
        np.testing.assert_array_equal(observed, expected)

    def test_11_threshold_and_firewalls(self) -> None:
        self.assertEqual(preflight.PER_BANK_RESS, 0.05)
        self.assertEqual(self.summary["new_scientific_evaluations"], 0)
        self.assertTrue(all(value is False for value in self.summary["firewalls"].values()))

    def test_12_design_frozen_before_results(self) -> None:
        self.assertLess(preflight.ARCHITECTURE_GRID_PATH.stat().st_mtime_ns, preflight.SUMMARY_PATH.stat().st_mtime_ns)
        self.assertLess(preflight.SUBSET_MANIFEST_PATH.stat().st_mtime_ns, preflight.SUMMARY_PATH.stat().st_mtime_ns)

    def test_13_inventory_seals_every_result(self) -> None:
        inventory = json.loads(preflight.INVENTORY_PATH.read_text(encoding="utf-8"))
        for row in inventory["artifacts"]:
            self.assertEqual(file_sha256(preflight.OUTPUT_ROOT / row["path"]), row["sha256"])

    def test_14_recommendation_is_allowed(self) -> None:
        allowed = {
            "RECOMMEND_3_OF_4", "RECOMMEND_6_OF_8", "RECOMMEND_12_OF_16",
            "RECOMMEND_7_OF_8", "RECOMMEND_14_OF_16", "NO_REPLICATE_GATE_ARCHITECTURE_READY",
        }
        self.assertIn(self.summary["recommendation"]["recommendation"], allowed)


if __name__ == "__main__":
    unittest.main()
