"""Tests for the diagnostic-only all-bank Law completion artifact."""

from __future__ import annotations

import json
import unittest

from .fresh_bank_law_diagnostics_run import INVENTORY_PATH, RESULT_PATH
from .fresh_bank_robustness import BANK_MANIFEST_PATH, CANDIDATE_FREEZE_PATH
from .production_artifacts import file_sha256


class FreshBankLawDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        cls.inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))

    def test_all_pairs_are_present(self) -> None:
        self.assertEqual(self.result["replicate_count"], 32)
        self.assertEqual(len(self.result["rows"]), 32)
        self.assertEqual(
            [row["replicate_id"] for row in self.result["rows"]], list(range(32))
        )

    def test_law_was_evaluated_on_every_bank(self) -> None:
        self.assertTrue(self.result["all_screen_banks_evaluated"])
        self.assertTrue(self.result["all_audit_banks_evaluated"])
        self.assertTrue(all(row["audit"]["audit_performed"] for row in self.result["rows"]))

    def test_completion_did_not_change_candidate_results(self) -> None:
        self.assertFalse(self.result["candidate_eligibility_recomputed"])
        self.assertFalse(self.result["sealed_candidate_results_modified"])

    def test_freeze_and_manifest_seals_match(self) -> None:
        self.assertEqual(
            self.result["candidate_freeze_sha256"], file_sha256(CANDIDATE_FREEZE_PATH)
        )
        self.assertEqual(
            self.result["bank_manifest_sha256"], file_sha256(BANK_MANIFEST_PATH)
        )

    def test_result_is_sealed(self) -> None:
        self.assertEqual(self.inventory["result_sha256"], file_sha256(RESULT_PATH))


if __name__ == "__main__":
    unittest.main()
