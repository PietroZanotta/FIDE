from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import pytest

from mfsi.design import OptimizeResult

from experiments.active_nematic_unbalance_percentage.eval import (
    _mean_se,
    _selected_full_audit,
)
from experiments.active_nematic_unbalance_percentage.percentage_selection import (
    ExactAuditCache,
    ObservationBankPrefixCache,
    _explicit_candidate,
    _audit_full,
    _risk_passes,
    percentage_risk_budget,
)
from experiments.active_nematic_unbalance_percentage.unbalanced_experiment import (
    UnbalancedActiveNematicExperiment,
)


ROOT = Path(__file__).resolve().parent


def test_config_declares_only_five_percent_relative_budget() -> None:
    cfg = json.loads((ROOT / "config.json").read_text())
    assert cfg["law"] == {
        "mmd_bandwidths": [0.5, 1.0, 2.0, 4.0],
        "grid_shape": [48, 48, 24],
        "max_relative_risk_violation": 0.05,
    }
    assert "epsilon_r" not in cfg["law"]
    optimization = cfg["optimization"]
    assert (
        optimization["law_steps"],
        optimization["tangent_steps"],
        optimization["full_steps"],
    ) == (60, 50, 40)
    assert optimization["tangent_local_starts"] == 8
    assert optimization["full_local_starts"] == 10
    assert optimization["full_exact_rescore_candidates"] == 4


def test_percentage_budget_is_anchored_to_each_risk_star() -> None:
    epsilon, ceiling = percentage_risk_budget(1.6, 0.05)
    assert epsilon == pytest.approx(0.08)
    assert ceiling == pytest.approx(1.68)
    assert ceiling / 1.6 == pytest.approx(1.05)


def test_fast_eval_marks_omitted_selection_baseline_unavailable() -> None:
    result = {
        "designs": {"law": [1.0, 2.0]},
        "selection_candidates": {"full": []},
    }
    assert _selected_full_audit(result, "law") is None
    assert _mean_se([]) is None


def test_full_audit_always_rescores_mandatory_incumbent() -> None:
    class Bank(NamedTuple):
        plus_sample_indices: jnp.ndarray

    class Sensors:
        @staticmethod
        def canonicalize(eta):
            return eta

    class Experiment:
        sensors = Sensors()

        @staticmethod
        def audit_metric(eta, bank, name):
            assert name == "full_action"
            return {"value": float(eta[0]), "valid": True}

    bank = Bank(jnp.arange(4))
    candidates = [
        OptimizeResult(jnp.asarray([value]), value, True, ())
        for value in (0.0, 1.0, 2.0)
    ]
    screened = [(candidate, {"value": 1.0, "valid": True}) for candidate in candidates]
    best, rows = _audit_full(
        Experiment(), bank, screened,
        prescreen_trials=1,
        finalists=1,
        mandatory=[jnp.asarray([2.0])],
    )
    assert float(best[0].eta[0]) == 0.0
    assert {float(row[0].eta[0]) for row in rows} == {0.0, 2.0}


def test_explicit_incumbent_is_not_changed_by_optimizer_semantics() -> None:
    class Sensors:
        @staticmethod
        def canonicalize(eta):
            return jnp.asarray(eta)

    class Experiment:
        sensors = Sensors()

    candidate = _explicit_candidate(Experiment(), jnp.asarray([3.0, 4.0]))
    assert candidate.feasible
    assert tuple(candidate.violations) == ()
    assert candidate.eta.tolist() == [3.0, 4.0]


def test_exact_audit_cache_requires_byte_identical_geometry_and_bank() -> None:
    class Sensors:
        @staticmethod
        def canonicalize(eta):
            return jnp.asarray(eta, dtype=jnp.float64)

    class Experiment:
        sensors = Sensors()

        def __init__(self):
            self.calls = 0

        def audit_metric(self, eta, bank, name):
            self.calls += 1
            return {"value": float(jnp.sum(eta)), "valid": True}

    exp = Experiment()
    cache = ExactAuditCache()
    first_bank = object()
    second_bank = object()
    eta = jnp.asarray([1.0, 2.0])
    assert cache.evaluate(exp, eta, first_bank, "law_risk") == cache.evaluate(
        exp, eta, first_bank, "law_risk"
    )
    cache.evaluate(exp, eta.at[0].set(1.0 + 1.0e-12), first_bank, "law_risk")
    cache.evaluate(exp, eta, second_bank, "law_risk")
    assert exp.calls == 3
    assert cache.stats() == {"hits": 1, "misses": 3, "entries": 3}

    seeded = ExactAuditCache()
    receipt = {"value": 7.0, "valid": True}
    seeded.store(exp, eta, first_bank, "full_action", receipt)
    assert seeded.evaluate(exp, eta, first_bank, "full_action") is receipt
    assert seeded.stats() == {"hits": 1, "misses": 0, "entries": 1}


def test_prefix_cache_reuses_only_the_same_root_and_exact_count() -> None:
    class Bank(NamedTuple):
        plus_sample_indices: jnp.ndarray
        minus_sample_indices: jnp.ndarray

    first = Bank(jnp.arange(6), jnp.arange(6) + 10)
    second = Bank(jnp.arange(6), jnp.arange(6) + 10)
    cache = ObservationBankPrefixCache()
    prefix = cache.prefix(first, 3)
    assert cache.prefix(first, 3) is prefix
    assert cache.prefix(first, 4) is not prefix
    assert cache.prefix(second, 3) is not prefix
    assert prefix.plus_sample_indices.tolist() == [0, 1, 2]
    assert cache.stats() == {"hits": 1, "misses": 3, "entries": 3}


def test_stable_prefix_enables_repeated_full_prescreen_audit_hit() -> None:
    class Bank(NamedTuple):
        plus_sample_indices: jnp.ndarray

    class Sensors:
        @staticmethod
        def canonicalize(eta):
            return eta

    class Experiment:
        sensors = Sensors()

        def __init__(self):
            self.calls = 0

        def audit_metric(self, eta, bank, name):
            self.calls += 1
            return {"value": float(eta[0]), "valid": True}

    exp = Experiment()
    bank = Bank(jnp.arange(4))
    candidate = OptimizeResult(jnp.asarray([1.0]), 1.0, True, ())
    screened = [(candidate, {"value": 0.0, "valid": True})]
    audit_cache = ExactAuditCache()
    prefix_cache = ObservationBankPrefixCache()
    for _ in range(2):
        _audit_full(
            exp,
            bank,
            screened,
            prescreen_trials=2,
            finalists=1,
            mandatory=[candidate.eta],
            audit_cache=audit_cache,
            prefix_cache=prefix_cache,
        )
    assert exp.calls == 2  # one prefix audit plus one complete-bank audit
    assert audit_cache.stats() == {"hits": 2, "misses": 2, "entries": 2}


def test_remembered_scalar_authority_skips_batch_probe() -> None:
    exp = object.__new__(UnbalancedActiveNematicExperiment)
    exp.remember_certification_scalar_fallback = True
    exp._certification_scalar_required = True
    expected = [{"trial": 0, "valid": True}]
    exp.exact_trial_rows = lambda eta, bank: expected
    exp.trial_values_batch = lambda *args, **kwargs: pytest.fail(
        "batch probe should have been skipped"
    )
    assert exp.certified_trial_rows(jnp.zeros(1), object()) is expected


def test_exact_scalar_rows_compute_geometry_once_per_species() -> None:
    class Bank(NamedTuple):
        plus_sample_indices: jnp.ndarray

    class Sensors:
        @staticmethod
        def centers(eta):
            return jnp.asarray([[0.0, 0.0]])

    class FakeExperiment:
        cfg = {
            "validity": {
                "max_calibration_residual": 1.0,
                "min_ess_fraction": 0.0,
                "max_screened_pde_relative_residual": 1.0,
            }
        }
        species_weights = {"plus": 1.0, "minus": 1.0}
        sensors = Sensors()
        reuse_exact_trial_geometry = True

        def __init__(self):
            self.geometry_calls = []
            self.trial_geometry = []

        def _geometry(self, species, eta):
            self.geometry_calls.append(species)
            return (f"{species}-truth", f"{species}-ref")

        def trial_values(self, eta, bank, trial, *, geometry_by_species=None):
            self.trial_geometry.append(geometry_by_species)
            metric = {
                "law_risk": jnp.asarray(1.0),
                "shape_mmd": jnp.asarray(0.5),
                "mass_error": jnp.asarray(0.25),
                "tangent_action": jnp.asarray(2.0),
                "tangent_transport": jnp.asarray(1.5),
                "tangent_reaction": jnp.asarray(0.5),
                "full_action": jnp.asarray(3.0),
                "move_action": jnp.asarray(2.0),
                "reaction_action": jnp.asarray(1.0),
                "reaction_fraction": jnp.asarray(1.0 / 3.0),
                "max_calibration_residual": jnp.asarray(0.0),
                "min_ess_fraction": jnp.asarray(1.0),
                "max_pde_relative_residual": jnp.asarray(0.0),
            }
            return (
                jnp.asarray(2.0),
                jnp.asarray(4.0),
                jnp.asarray(6.0),
                metric,
                metric,
            )

    exp = FakeExperiment()
    rows = UnbalancedActiveNematicExperiment.exact_trial_rows(
        exp, jnp.zeros(2), Bank(jnp.arange(5))
    )
    assert len(rows) == 5
    assert exp.geometry_calls == ["plus", "minus"]
    assert all(geometry is exp.trial_geometry[0] for geometry in exp.trial_geometry)
    assert exp.trial_geometry[0]["plus"] == ("plus-truth", "plus-ref")


def test_full_only_metric_can_skip_discarded_tangent_work() -> None:
    class FakeExperiment:
        skip_unused_tangent_for_full_metric = True

        def __init__(self):
            self.calls = []

        def trial_values_batch(
            self, eta, bank, *, compute_tangent=True, compute_full=True
        ):
            self.calls.append((compute_tangent, compute_full))
            return (
                jnp.asarray([1.0]),
                jnp.asarray([2.0]),
                jnp.asarray([3.0]),
                {},
                {},
            )

    exp = FakeExperiment()
    full = UnbalancedActiveNematicExperiment.mean_metric(
        exp, jnp.zeros(1), object(), "full_action"
    )
    tangent = UnbalancedActiveNematicExperiment.mean_metric(
        exp, jnp.zeros(1), object(), "tangent_action"
    )
    assert float(full) == 3.0
    assert float(tangent) == 2.0
    assert exp.calls == [(False, True), (True, False)]


def test_robust_risk_screen_requires_every_view_to_pass() -> None:
    maxima = [1.05, 2.10]
    assert _risk_passes(
        {"value": 2.0, "view_values": [1.04, 2.0]}, 2.10, maxima
    )
    assert not _risk_passes(
        {"value": 2.0, "view_values": [1.06, 2.0]}, 2.10, maxima
    )


def test_production_manifest_has_exact_five_percent_ceilings_if_present() -> None:
    manifest = ROOT / "outputs" / "run" / "manifest.json"
    if not manifest.is_file():
        pytest.skip("production results are not present")
    for entry in json.loads(manifest.read_text())["runs"]:
        result = json.loads(Path(entry["result"]).read_text())
        assert result["risk_max"] / result["risk_star"] == pytest.approx(1.05)
        for design in result["validation_designs"].values():
            summary = design["summary"]
            assert summary["valid_trials"] == summary["trials"] == 32
