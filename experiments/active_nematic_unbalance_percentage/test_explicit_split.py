from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from experiments.active_nematic_unbalance_percentage import run as active_run
from experiments.active_nematic_unbalance_percentage.domain import EmpiricalEndpointSource
from experiments.active_nematic_unbalance_percentage.domain import SplitConfig, make_run_split
from experiments.active_nematic_unbalance_percentage.run import normalized_times
from experiments.active_nematic_unbalance_percentage.train_reference_endpoints import (
    _time_grid,
)


def test_explicit_split_preserves_declared_roles() -> None:
    config = SplitConfig(
        train_runs=4,
        design_runs=2,
        validation_runs=2,
        train_indices=(0, 2, 4, 6),
        design_indices=(1, 3),
        validation_indices=(5, 7),
    )
    split = make_run_split(config)
    assert np.array_equal(split.train, [0, 2, 4, 6])
    assert np.array_equal(split.design, [1, 3])
    assert np.array_equal(split.validation, [5, 7])


def test_explicit_split_must_be_a_complete_partition() -> None:
    with pytest.raises(ValueError, match="partition"):
        SplitConfig(
            train_runs=2,
            design_runs=1,
            validation_runs=1,
            train_indices=(0, 4),
            design_indices=(1,),
            validation_indices=(2,),
        )


def test_reference_time_normalization_preserves_production_interval() -> None:
    bank = SimpleNamespace(times=np.asarray([21.0, 26.0, 31.0]))
    assert np.array_equal(normalized_times(bank), [0.0, 0.5, 1.0])


def test_reference_time_normalization_supports_prospective_interval() -> None:
    bank = SimpleNamespace(times=np.asarray([5.0, 15.0, 25.0]))
    assert np.array_equal(normalized_times(bank), [0.0, 0.5, 1.0])


def test_reference_time_grid_can_retain_intermediate_rollout_times() -> None:
    assert _time_grid(21.0, 31.0, 1.0) == list(
        np.arange(21.0, 32.0, dtype=np.float64)
    )


def test_reference_time_grid_rejects_nondividing_step() -> None:
    with pytest.raises(ValueError, match="divide"):
        _time_grid(5.0, 15.0, 3.0)


def test_matched_kde_training_source_changes_only_initial_samples(monkeypatch) -> None:
    empirical = EmpiricalEndpointSource(
        jnp.zeros((4, 3), dtype=jnp.float64),
        jnp.ones((4, 3), dtype=jnp.float64),
    )
    calls = {}

    monkeypatch.setattr(
        active_run,
        "endpoint_source_for_species",
        lambda *args, **kwargs: empirical,
    )

    def sample_kde(states, probabilities, **kwargs):
        calls.update(kwargs)
        return np.full((4, 3), 2.0)

    monkeypatch.setattr(active_run, "sample_periodic_kde_bank", sample_kde)
    measure = SimpleNamespace(
        states=np.zeros((2, 3)),
        normalized_probabilities=lambda minimum_mass: np.asarray([0.5, 0.5]),
    )
    bank = SimpleNamespace(measure=lambda species, endpoint, runs: measure)
    cfg = {
        "seed": 7,
        "unbalanced": {"minimum_mass": 1.0e-6},
        "reference": {
            "endpoint_particles": 4,
            "endpoint_seed_offset": 20,
            "bank_position_jitter_std": 1.0,
            "bank_beta_jitter_std": 0.25,
        },
        "reference_training": {"initial_endpoint_density_model": "periodic_kde"},
    }
    source = active_run.reference_training_source(
        cfg,
        bank,
        np.asarray([0, 1]),
        "minus",
        np.asarray([10.0, 10.0, 2.0 * np.pi]),
    )

    assert np.array_equal(source.x0, np.full((4, 3), 2.0))
    assert np.array_equal(source.x1, empirical.x1)
    assert calls["seed"] == 28
    assert calls["position_std"] == 1.0
    assert calls["beta_std"] == 0.25


def test_empirical_training_source_remains_backward_compatible(monkeypatch) -> None:
    empirical = EmpiricalEndpointSource(
        jnp.zeros((2, 3), dtype=jnp.float64),
        jnp.ones((2, 3), dtype=jnp.float64),
    )
    monkeypatch.setattr(
        active_run,
        "endpoint_source_for_species",
        lambda *args, **kwargs: empirical,
    )
    cfg = {
        "seed": 7,
        "unbalanced": {"minimum_mass": 1.0e-6},
        "reference": {"endpoint_particles": 2},
        "reference_training": {},
    }
    source = active_run.reference_training_source(
        cfg, SimpleNamespace(), np.asarray([0]), "plus", np.ones(3)
    )
    assert source is empirical
