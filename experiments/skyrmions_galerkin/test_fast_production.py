"""Small structural tests for the accelerated production workflow."""

from __future__ import annotations

from types import SimpleNamespace
from dataclasses import replace
import unittest

import jax
import jax.numpy as jnp

from .fast_production import FAST_ROOT, basis_cache_memory_estimates, require_fast_output_path
from .fast_workflow import _gradient_metrics, _periodic_delta
from .deep_ritz import DeepRitzConfig, init_ritz_params, solve_deep_ritz
from .authoritative_stability import ordering_summary

jax.config.update("jax_enable_x64", True)


class FastProductionTests(unittest.TestCase):
    def test_output_guard_accepts_only_fast_subtree(self):
        self.assertEqual(require_fast_output_path(FAST_ROOT / "unit"), (FAST_ROOT / "unit").resolve())
        with self.assertRaises(ValueError):
            require_fast_output_path(FAST_ROOT.parent / "escape")

    def test_memory_estimate_rejects_per_sample_gram(self):
        def bank(samples):
            return SimpleNamespace(configurations=jnp.zeros((13, samples, 16, 2)))
        data = SimpleNamespace(
            ritz_train_bank=bank(8192), ritz_audit_bank=bank(4096),
            validation_fit_bank=bank(16384), validation_audit_bank=bank(16384),
        )
        result = basis_cache_memory_estimates(data, 160)
        self.assertFalse(result["rejected_train_per_sample_gram"]["cached"])
        self.assertEqual(result["train"]["basis_values_bytes"], 13 * 8192 * 160 * 8)
        self.assertEqual(result["train"]["basis_gradients_bytes"], 13 * 8192 * 160 * 32 * 8)

    def test_periodic_delta_uses_minimum_image(self):
        center = jnp.asarray([1.99, .99, .1, .2])
        candidate = jnp.asarray([.01, .01, .1, .2])
        delta = _periodic_delta(candidate, center, (2.0, 1.0))
        self.assertTrue(jnp.allclose(delta, jnp.asarray([.02, .02, 0., 0.])))

    def test_gradient_metrics(self):
        lower = {"K": 140, "action": 1.0, "gradient": [1.0, 0.0]}
        upper = {"K": 160, "action": .9, "gradient": [1.0, 0.0]}
        result = _gradient_metrics(lower, upper)
        self.assertAlmostEqual(result["cosine_similarity"], 1.0)
        self.assertAlmostEqual(result["relative_gradient_difference"], 0.0)

    def test_compiled_full_bank_matches_reference_solver(self):
        key = jax.random.PRNGKey(19)
        configurations = jax.random.uniform(
            key, (3, 8, 4, 2), dtype=jnp.float64
        )
        weights = jnp.full((3, 8), 1.0 / 8.0, dtype=jnp.float64)
        forcing = jax.random.normal(
            jax.random.fold_in(key, 1), (3, 8), dtype=jnp.float64
        )
        forcing = forcing - jnp.mean(forcing, axis=1, keepdims=True)
        times = jnp.linspace(0.0, 1.0, 3, dtype=jnp.float64)
        time_weights = jnp.asarray([0.25, 0.5, 0.25], dtype=jnp.float64)
        initial = init_ritz_params(
            jax.random.fold_in(key, 2), hidden_width=5, hidden_layers=1
        )
        base = DeepRitzConfig(
            hidden_width=5, hidden_layers=1, adam_steps=4,
            adam_batch_size=8, lbfgs_iterations=2, lbfgs_batch_size=4,
            lbfgs_line_search_steps=3, log_every=1,
        )
        reference = solve_deep_ritz(
            configurations, weights, forcing, times, time_weights, base,
            initial_params=initial,
        )
        compiled = solve_deep_ritz(
            configurations, weights, forcing, times, time_weights,
            replace(base, compiled_full_bank=True), initial_params=initial,
        )
        reference_flat = jnp.concatenate([
            leaf.reshape(-1) for leaf in jax.tree_util.tree_leaves(reference.params)
        ])
        compiled_flat = jnp.concatenate([
            leaf.reshape(-1) for leaf in jax.tree_util.tree_leaves(compiled.params)
        ])
        self.assertTrue(jnp.allclose(reference_flat, compiled_flat, rtol=1e-11, atol=1e-12))
        self.assertAlmostEqual(
            reference.lbfgs_final_objective, compiled.lbfgs_final_objective,
            delta=1e-11,
        )

    def test_authoritative_ordering_requires_pairwise_consensus(self):
        def pair(left, right, valid=True):
            return {
                "incumbent": {"action": left, "valid": valid},
                "challenger": {"action": right, "valid": valid},
            }

        stable = ordering_summary(
            [pair(1.0, 0.9), pair(1.1, 1.0)], minimum_improvement=1e-6
        )
        self.assertEqual(stable["decision"], "stable_improvement")
        self.assertTrue(stable["passed"])
        mixed = ordering_summary(
            [pair(1.0, 0.9), pair(1.0, 1.1)], minimum_improvement=1e-6
        )
        self.assertEqual(mixed["decision"], "indeterminate")
        self.assertFalse(mixed["passed"])
        invalid = ordering_summary(
            [pair(1.0, 0.9), pair(1.0, 0.8, valid=False)],
            minimum_improvement=1e-6,
        )
        self.assertFalse(invalid["passed"])


if __name__ == "__main__":
    unittest.main()
