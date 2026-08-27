"""Experiment-local checks for the continuous sensor-gradient implementation."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
for search_path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from mfsi.config import load_config

from . import full_gradient
from .full_gradient import (
    envelope_full_value_and_grad,
    forcing_state,
    minimum_sensor_separation,
    projected_weights,
    smooth_separation_penalty,
    wrap_periodic,
)
from .workflow import (
    OUTPUT_ROOT,
    prepare_experiment,
    require_output_path,
    run_gradient_check,
    selection_risk,
)
from .rigorous_gradient_check import _direction_summary


class FullGradientChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        package = Path(__file__).resolve().parent
        cls.cfg = load_config(package / "config.json", smoke=True)
        cls.data = prepare_experiment(cls.cfg, OUTPUT_ROOT / "test_artifacts")
        cls.eta = jnp.asarray(cls.cfg["envelope"]["eta0"], dtype=jnp.float64)
        cls.direction = jnp.arange(1.0, 9.0, dtype=jnp.float64)
        cls.direction = cls.direction / jnp.linalg.norm(cls.direction)
        cls.gradient_result, cls.theta = run_gradient_check(
            cls.cfg, cls.data, inner_mode="smoke"
        )

    def test_import_and_no_legacy_dependency(self) -> None:
        package = Path(__file__).resolve().parent
        forbidden = "experiments" + ".skyrmions_deep_ritz."
        for path in package.glob("*.py"):
            self.assertNotIn(forbidden, path.read_text(encoding="utf-8"), path.name)

    def test_output_isolation_guard(self) -> None:
        self.assertEqual(require_output_path(OUTPUT_ROOT / "x"), (OUTPUT_ROOT / "x").resolve())
        with self.assertRaises(ValueError):
            require_output_path(Path("/tmp/not-this-experiment"))

    def test_reference_is_closed_over_not_an_eta_variable(self) -> None:
        signature = inspect.signature(envelope_full_value_and_grad)
        self.assertEqual(tuple(signature.parameters)[:2], ("eta", "theta_fixed"))
        value, gradient, _ = envelope_full_value_and_grad(
            self.eta,
            self.theta,
            self.data.selection_problem,
            self.data.ritz_train_bank,
        )
        self.assertTrue(bool(jnp.isfinite(value)))
        self.assertEqual(gradient.shape, (8,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(gradient))))

    def test_information_projection_gradient(self) -> None:
        weight_shape = projected_weights(
            self.eta, self.data.selection_problem, self.data.projection_bank
        ).shape
        cotangent = jnp.arange(
            weight_shape[0] * weight_shape[1], dtype=jnp.float64
        ).reshape(weight_shape)
        gradient = jax.grad(
            lambda eta: jnp.vdot(
                projected_weights(
                    eta, self.data.selection_problem, self.data.projection_bank
                ),
                cotangent,
            )
        )(self.eta)
        tangent = jnp.vdot(gradient, self.direction)
        self.assertTrue(bool(jnp.isfinite(tangent)))
        self.assertGreater(float(jnp.abs(tangent)), 0.0)

    def test_forcing_gradient(self) -> None:
        forcing_shape = forcing_state(
            self.eta, self.data.selection_problem, self.data.ritz_train_bank
        ).forcing.shape
        cotangent = jnp.arange(
            forcing_shape[0] * forcing_shape[1], dtype=jnp.float64
        ).reshape(forcing_shape)
        gradient = jax.grad(
            lambda eta: jnp.vdot(
                forcing_state(
                    eta, self.data.selection_problem, self.data.ritz_train_bank
                ).forcing,
                cotangent,
            )
        )(self.eta)
        tangent = jnp.vdot(gradient, self.direction)
        self.assertTrue(bool(jnp.isfinite(tangent)))
        self.assertGreater(float(jnp.abs(tangent)), 0.0)

    def test_energy_identity_and_reoptimized_directional_check(self) -> None:
        result = self.gradient_result
        self.assertTrue(result["passed"])
        self.assertLessEqual(
            result["center_envelope_diagnostics"]["energy_identity_relerr"],
            result["energy_identity_tolerance"],
        )
        self.assertLessEqual(
            result["best_relative_discrepancy"], result["relative_tolerance"]
        )
        self.assertEqual(len(result["rows"]), 4)
        self.assertTrue(all("plus_ritz_solve" in row for row in result["rows"]))

    def test_periodic_wrapping(self) -> None:
        family = self.data.selection_problem.family
        wrapped = wrap_periodic(self.eta + jnp.tile(jnp.asarray([4.0, -3.0]), 4), family)
        centers = wrapped.reshape(4, 2)
        self.assertTrue(bool(jnp.all(centers >= 0.0)))
        self.assertTrue(bool(jnp.all(centers < jnp.asarray(family.box))))

    def test_smooth_and_exact_separation_agree(self) -> None:
        family = self.data.selection_problem.family
        self.assertTrue(bool(family.geometry_valid(self.eta)))
        self.assertEqual(float(smooth_separation_penalty(self.eta, family)), 0.0)
        bad = self.eta.at[2:4].set(self.eta[:2])
        self.assertFalse(bool(family.geometry_valid(bad)))
        self.assertGreater(float(smooth_separation_penalty(bad, family)), 0.0)
        self.assertLess(float(minimum_sensor_separation(bad, family)), family.min_separation)

    def test_risk_has_gradient_but_hard_gate_is_separate(self) -> None:
        risk, gradient = jax.value_and_grad(lambda eta: selection_risk(eta, self.data))(self.eta)
        self.assertTrue(bool(jnp.isfinite(risk)))
        self.assertTrue(bool(jnp.all(jnp.isfinite(gradient))))
        source = inspect.getsource(full_gradient.projected_law_risk)
        self.assertNotIn("geometry_valid", source)
        workflow_source = (Path(__file__).resolve().parent / "workflow.py").read_text(encoding="utf-8")
        self.assertIn('risk <= risk_limit', workflow_source)

    def test_report_is_json_serializable(self) -> None:
        json.dumps(self.gradient_result, allow_nan=False)

    def _strict_rows(self, relative_errors: list[float], fd_values: list[float]):
        rows = []
        for index, (relative, fd_value) in enumerate(zip(relative_errors, fd_values)):
            error = 1.0 / (index + 1)
            rows.append({
                "epsilon": self.cfg["envelope"]["gradient_validation"]["epsilon_ladder"][index],
                "fd_optimized_value": fd_value,
                "relative_error_V": relative,
                "continuity_plus_error": error,
                "continuity_minus_error": 0.8 * error,
                "plus_metrics": {"stationary": True, "diagnostics_valid": True},
                "minus_metrics": {"stationary": True, "diagnostics_valid": True},
            })
        return rows

    def test_strict_gate_rejects_one_lucky_epsilon(self) -> None:
        rows = self._strict_rows(
            [0.4, 0.3, 0.01, 0.3, 0.4, 0.5, 0.6],
            [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
        )
        summary = _direction_summary(rows, 1.0, self.cfg)
        self.assertFalse(summary["passed"])
        self.assertFalse(summary["rules"]["consecutive_relative_accuracy"])

    def test_strict_gate_accepts_a_convergence_window(self) -> None:
        rows = self._strict_rows(
            [0.2, 0.08, 0.04, 0.03, 0.01, 0.03, 0.08],
            [0.8, 0.92, 0.96, 0.97, 0.99, 0.97, 0.92],
        )
        summary = _direction_summary(rows, 1.0, self.cfg)
        self.assertTrue(summary["passed"])
        self.assertTrue(summary["rules"]["consecutive_fd_sign_and_AD_agreement"])


if __name__ == "__main__":
    unittest.main()
