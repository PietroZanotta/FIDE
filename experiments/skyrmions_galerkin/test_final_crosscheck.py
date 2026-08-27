"""Regression contracts for the retrospective final 3% cross-check."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
for search_path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from mfsi.config import load_config

from . import final_crosscheck
from .final_crosscheck import (
    DICTIONARY_PATH, OUTPUT_ROOT, common_solver_comparison,
    directional_fd_row, frozen_pair_rows, require_crosscheck_output_path,
    risk_ratio, validation_protocol,
)
from .galerkin_only import GALERKIN_ONLY_ROOT
from .measurements import LocalDensitySensors
from .production_basis import load_dictionary


class FinalCrosscheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config(PACKAGE_ROOT / "config.json", smoke=False)

    def test_K280_gradient_is_deterministically_reproduced(self) -> None:
        path = OUTPUT_ROOT / "gradient" / "result.json"
        if not path.is_file():
            self.skipTest("direct K=280 gradient result has not run")
        result = json.loads(path.read_text())
        self.assertTrue(result["deterministic"])
        self.assertEqual(np.asarray(result["eta_gradient"]).shape, (8,))
        self.assertTrue(np.all(np.isfinite(result["eta_gradient"])))

    def test_directional_fd_helper_uses_centered_difference(self) -> None:
        family = LocalDensitySensors(4, 0.12, (2.0, 1.0), 0.2)
        context = SimpleNamespace(
            data=SimpleNamespace(selection_problem=SimpleNamespace(family=family))
        )
        payloads = [
            {"action": 1.3, "rank_by_time": [2], "rank_stable": True,
             "hard_gates_passed": True},
            {"action": 0.7, "rank_by_time": [2], "rank_stable": True,
             "hard_gates_passed": True},
        ]
        with patch.object(final_crosscheck, "_gradient_point_payload", side_effect=payloads):
            row = directional_fd_row(
                context, jnp.asarray(self.cfg["envelope"]["eta0"]),
                jnp.ones(8) / jnp.sqrt(8.0), 0.1, 3.0, [2],
            )
        self.assertAlmostEqual(row["fd"], 3.0, places=14)
        self.assertAlmostEqual(row["relative_discrepancy"], 0.0, places=14)
        self.assertTrue(row["accepted"])

    def test_cross_K_dictionary_reuses_exact_nested_prefixes(self) -> None:
        lower_path = GALERKIN_ONLY_ROOT / "cache" / "dictionaries" / "dictionary_K240.npz"
        if not lower_path.is_file() or not DICTIONARY_PATH.is_file():
            self.skipTest("production nested dictionaries unavailable")
        lower = load_dictionary(lower_path, box=(2.0, 1.0))
        upper = load_dictionary(DICTIONARY_PATH, box=(2.0, 1.0))
        self.assertTrue(np.array_equal(
            np.asarray(lower.feature_kind), np.asarray(upper.feature_kind[:240])
        ))
        self.assertTrue(np.array_equal(
            np.asarray(lower.base_means), np.asarray(upper.base_means[:, :240])
        ))

    @staticmethod
    def _synthetic_ladder():
        def design(actions):
            return {
                "risk": 1.0,
                "ladder": [
                    {"basis_size": size, "fit_action": action,
                     "audit_action": action + 0.01, "valid": True}
                    for size, action in zip((160, 200, 240, 280), actions, strict=True)
                ],
            }
        return {
            "designs": {
                "law": design((2.0, 2.1, 2.2, 2.3)),
                "eta0": design((1.5, 1.6, 1.7, 1.8)),
                "eta_grad": design((1.4, 1.5, 1.6, 1.7)),
            }
        }

    def test_frozen_pair_comparison(self) -> None:
        rows = frozen_pair_rows(self._synthetic_ladder(), "fit_action")
        self.assertEqual([row["basis_size"] for row in rows], [160, 200, 240, 280])
        self.assertTrue(all(row["delta"] < 0.0 for row in rows))
        self.assertTrue(all(row["eta_grad_better"] for row in rows))

    def test_validation_protocol_is_three_percent_plus_five_points(self) -> None:
        protocol = validation_protocol(self.cfg, 5.0)
        self.assertAlmostEqual(protocol["strict_3pct_ceiling"], 5.15)
        self.assertAlmostEqual(protocol["declared_plus5pp_ceiling"], 5.4)
        self.assertAlmostEqual(protocol["declared_validation_multiplier"], 1.08)

    def test_exact_risk_ratio(self) -> None:
        self.assertAlmostEqual(risk_ratio(5.4, 5.0), 0.08)

    def test_common_solver_comparison(self) -> None:
        result = common_solver_comparison(self._synthetic_ladder(), "fit_action")
        self.assertAlmostEqual(
            result["fide_improvement_eta_grad_over_law"], (2.3 - 1.7) / 2.3
        )
        self.assertAlmostEqual(
            result["continuous_improvement_eta_grad_over_eta0"], (1.8 - 1.7) / 1.8
        )

    def test_validation_cannot_mutate_or_rerun_selection(self) -> None:
        source = inspect.getsource(final_crosscheck)
        for forbidden in (
            "run_galerkin_only_optimization(", "_run_trust_trajectory(",
            "solve_deep_ritz(", "audit_deep_ritz(",
        ):
            self.assertNotIn(forbidden, source)
        validation_source = inspect.getsource(final_crosscheck.run_validation_ladder)
        self.assertIn('"selection_mutated": False', validation_source)
        self.assertIn('"no_optimization": True', validation_source)

    def test_output_path_isolation(self) -> None:
        self.assertEqual(
            require_crosscheck_output_path(OUTPUT_ROOT / "summary"),
            (OUTPUT_ROOT / "summary").resolve(),
        )
        with self.assertRaises(ValueError):
            require_crosscheck_output_path(GALERKIN_ONLY_ROOT / "selection")


if __name__ == "__main__":
    unittest.main()
