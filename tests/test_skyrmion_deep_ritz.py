from dataclasses import replace
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from mfsi.projection import IProjectionConfig
from experiments.skyrmions_deep_ritz.deep_ritz import (
    audit_deep_ritz,
    DeepRitzConfig,
    invariant_potential,
    init_ritz_params,
    load_ritz_checkpoint,
    manufactured_cosine_weak_residual,
    promote_ritz_params_to_independent,
    ritz_objective,
    save_ritz_checkpoint,
    solve_deep_ritz,
)
from experiments.skyrmions_deep_ritz.forcing import (
    ForcingConfig,
    continuity_forcing,
    strict_project_trajectory,
)
from experiments.skyrmions_deep_ritz.measurements import LocalDensitySensors
from experiments.skyrmions_deep_ritz.reference import (
    equivariant_velocity,
    init_equivariant_reference,
)


def _configurations(seed=0, samples=24, particles=5):
    key = jax.random.PRNGKey(seed)
    return jax.random.uniform(key, (samples, particles, 2), dtype=jnp.float64) * jnp.asarray([2.0, 1.0])


def test_measurement_is_permutation_invariant():
    x = _configurations(samples=3)
    permutation = jnp.asarray([3, 0, 4, 1, 2])
    sensors = LocalDensitySensors(3, 0.15)
    eta = jnp.asarray([0.2, 0.3, 0.9, 0.6, 1.7, 0.8])
    assert jnp.allclose(sensors.features(x, eta), sensors.features(x[:, permutation], eta), atol=1e-13)


def test_ritz_potential_invariance_and_gradient_equivariance():
    x = _configurations(samples=1)[0]
    permutation = jnp.asarray([2, 4, 0, 3, 1])
    params = init_ritz_params(jax.random.PRNGKey(4), hidden_width=10, hidden_layers=2)
    psi = lambda row: invariant_potential(params, row, jnp.asarray(0.37))
    assert jnp.allclose(psi(x), psi(x[permutation]), atol=1e-13)
    gradient = jax.grad(psi)(x)
    permuted_gradient = jax.grad(psi)(x[permutation])
    assert jnp.allclose(permuted_gradient, gradient[permutation], rtol=1e-11, atol=1e-12)


def test_ritz_checkpoint_round_trip(tmp_path):
    x = _configurations(samples=1)[0]
    params = init_ritz_params(jax.random.PRNGKey(5), hidden_width=8, hidden_layers=1)
    path = tmp_path / "ritz.npz"
    save_ritz_checkpoint(path, params, metadata={"role": "test"})
    restored, metadata = load_ritz_checkpoint(path)
    assert metadata == {"role": "test"}
    assert jnp.allclose(
        invariant_potential(params, x, jnp.asarray(0.4)),
        invariant_potential(restored, x, jnp.asarray(0.4)),
        atol=1e-14,
    )


def test_independent_time_nodes_preserve_symmetry_and_isolate_parameters(tmp_path):
    x = _configurations(samples=1)[0]
    shared = init_ritz_params(jax.random.PRNGKey(6), hidden_width=8, hidden_layers=1)
    params = promote_ritz_params_to_independent(shared, 3)
    changed = {
        group: tuple({name: value for name, value in layer.items()} for layer in layers)
        for group, layers in params.items()
    }
    last = changed["head"][-1]
    changed["head"] = changed["head"][:-1] + ({
        "W": last["W"].at[0].add(0.5),
        "b": last["b"],
    },)
    assert not jnp.allclose(
        invariant_potential(params, x, jnp.asarray(0.0)),
        invariant_potential(changed, x, jnp.asarray(0.0)),
    )
    assert jnp.allclose(
        invariant_potential(params, x, jnp.asarray(0.5)),
        invariant_potential(changed, x, jnp.asarray(0.5)),
        atol=1e-14,
    )
    permutation = jnp.asarray([2, 4, 0, 3, 1])
    assert jnp.allclose(
        invariant_potential(params, x, jnp.asarray(1.0)),
        invariant_potential(params, x[permutation], jnp.asarray(1.0)),
        atol=1e-13,
    )
    path = tmp_path / "independent.npz"
    save_ritz_checkpoint(path, params)
    restored, _ = load_ritz_checkpoint(path)
    assert jnp.allclose(
        invariant_potential(params, x, jnp.asarray(1.0)),
        invariant_potential(restored, x, jnp.asarray(1.0)),
        atol=1e-14,
    )


def test_chunked_full_objective_matches_monolithic_gauge_exactly():
    x = _configurations(seed=14, samples=8, particles=2)[None, ...]
    raw = jnp.linspace(0.2, 1.0, 8, dtype=jnp.float64)[None, :]
    weights = raw / jnp.sum(raw, axis=-1, keepdims=True)
    forcing = jnp.sin(jnp.arange(8, dtype=jnp.float64))[None, :]
    forcing = forcing - jnp.sum(weights * forcing, axis=-1, keepdims=True)
    times, time_weights = jnp.asarray([0.4]), jnp.asarray([1.0])
    params = init_ritz_params(jax.random.PRNGKey(15), hidden_width=7, hidden_layers=1)
    expected = float(ritz_objective(params, x, weights, forcing, times, time_weights))
    result = solve_deep_ritz(
        x, weights, forcing, times, time_weights,
        DeepRitzConfig(
            hidden_width=7, hidden_layers=1, adam_steps=0,
            lbfgs_iterations=0, lbfgs_batch_size=3,
        ),
        initial_params=params,
    )
    assert np.isclose(result.adam_final_objective, expected, rtol=2e-12, atol=2e-12)


def test_chunked_audit_matches_single_chunk_exactly():
    rows = jnp.stack([
        _configurations(seed=16, samples=7, particles=3),
        _configurations(seed=17, samples=7, particles=3),
    ])
    raw = jnp.asarray([
        [0.2, 0.4, 0.6, 0.8, 1.0, 0.7, 0.3],
        [1.0, 0.3, 0.8, 0.2, 0.5, 0.9, 0.4],
    ])
    weights = raw / jnp.sum(raw, axis=-1, keepdims=True)
    forcing = jnp.sin(jnp.arange(14, dtype=jnp.float64)).reshape(2, 7)
    forcing = forcing - jnp.einsum("tn,tn->t", weights, forcing)[:, None]
    times = jnp.asarray([0.0, 1.0])
    params = init_ritz_params(jax.random.PRNGKey(18), hidden_width=7, hidden_layers=1)
    whole = audit_deep_ritz(
        params, rows, weights, forcing, times, jnp.asarray([0.5, 0.5]),
        chunk_size=7,
    )
    chunked = audit_deep_ritz(
        params, rows, weights, forcing, times, jnp.asarray([0.5, 0.5]),
        chunk_size=3,
    )
    for name in (
        "action", "action_standard_error", "maximum_weak_residual",
        "maximum_energy_residual", "maximum_gauge_residual",
    ):
        assert np.isclose(chunked[name], whole[name], rtol=2e-12, atol=2e-12)


def test_reference_velocity_is_permutation_equivariant():
    x = _configurations(samples=2)
    permutation = jnp.asarray([4, 1, 3, 0, 2])
    params = init_equivariant_reference(jax.random.PRNGKey(8), hidden_width=9, hidden_layers=1)
    original = equivariant_velocity(params, jnp.asarray([0.2, 0.7]), x, box=(2.0, 1.0))
    permuted = equivariant_velocity(params, jnp.asarray([0.2, 0.7]), x[:, permutation], box=(2.0, 1.0))
    assert jnp.allclose(permuted, original[:, permutation], rtol=1e-11, atol=1e-12)


def test_strict_i_projection_calibrates_and_rejects_infeasible_target():
    phi = jnp.asarray(
        [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]], dtype=jnp.float64
    )
    base = jnp.full((1, 4), 0.25)
    target = jnp.asarray([[0.35, 0.65]])
    config = IProjectionConfig(max_steps=100, residual_tol=1e-12, newton_ridge=1e-12)
    state = strict_project_trajectory(phi, base, target, projection_cfg=config, tolerance=1e-9)
    assert float(jnp.linalg.norm(state.residual)) < 1e-9
    with pytest.raises(RuntimeError, match="failed hard calibration"):
        strict_project_trajectory(
            phi, base, jnp.asarray([[1.2, 0.5]]), projection_cfg=config, tolerance=1e-8
        )


def test_continuity_forcing_is_weighted_centered_for_valid_upstream_solve():
    x = _configurations(samples=40, particles=4)[None, ...]
    velocity = jnp.zeros_like(x)
    base = jnp.full((1, 40), 1.0 / 40.0)
    family = LocalDensitySensors(2, 0.22)
    eta = jnp.asarray([0.4, 0.3, 1.3, 0.7])
    target = jnp.mean(family.features(x, eta), axis=1)
    state = continuity_forcing(
        x, velocity, base, target, jnp.zeros_like(target), eta, family,
        projection_cfg=IProjectionConfig(max_steps=50, residual_tol=1e-12),
        cfg=ForcingConfig(
            projection_tolerance=1e-9, minimum_ess_fraction=0.1,
            forcing_mean_tolerance=1e-9,
        ),
    )
    assert float(jnp.max(jnp.abs(jnp.einsum("tn,tn->t", state.projection.weights, state.forcing)))) < 1e-12
    assert float(jnp.max(jnp.abs(state.forcing_mean_before_centering))) < 1e-9


def test_manufactured_torus_problem_and_lbfgs_preserves_heldout_residual():
    train = _configurations(seed=21, samples=128, particles=2)[None, ...]
    audit = _configurations(seed=22, samples=160, particles=2)
    weights = jnp.full((1, train.shape[1]), 1.0 / train.shape[1])
    audit_weights = jnp.full((audit.shape[0],), 1.0 / audit.shape[0])
    wave = 2.0 * jnp.pi / 2.0
    forcing = jnp.mean(jnp.cos(wave * train[..., 0]), axis=-1)
    times = jnp.asarray([0.5])
    time_weights = jnp.asarray([1.0])
    adam_cfg = DeepRitzConfig(
        seed=33, hidden_width=12, hidden_layers=1, adam_steps=100,
        adam_learning_rate=2e-3, lbfgs_iterations=0, log_every=100,
    )
    adam = solve_deep_ritz(train, weights, forcing, times, time_weights, adam_cfg)
    before = float(manufactured_cosine_weak_residual(adam.params, audit, audit_weights))
    refine_cfg = replace(adam_cfg, adam_steps=0, lbfgs_iterations=12, lbfgs_history=5)
    refined = solve_deep_ritz(
        train, weights, forcing, times, time_weights, refine_cfg, initial_params=adam.params
    )
    after = float(manufactured_cosine_weak_residual(refined.params, audit, audit_weights))
    assert np.isfinite(before) and np.isfinite(after)
    assert after <= before + 2e-2
    # The exact psi=-h/k^2 satisfies this weak equation; the modest network should
    # make a clearly nontrivial dent without turning the test into a benchmark.
    assert after < 0.65
