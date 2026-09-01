from __future__ import annotations

import ast
import copy
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for path in (HERE, REPO / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aggregate_qois import qoi_features
from build_prospective_data import _response_fields
from certify_v4_covariance import _condition_rows
from common import load_config
from evaluator import _correlated_sampling_delta, _smooth_bound_moment_curve
from experiment import _smooth_bound_moment_curve as physical_smooth_bound
from mfsi.moments import AnchoredCubicSplineConfig, AnchoredCubicSplineReconstructor
from prospective_data import TargetProspectiveData
from tangent_supplement_select import _tangent_trials
from v4_objective import V4DifferentiableObjective, canonical_geometry_key, make_v4_crn_bank
from v4_select import _adam_multistart, _risk_feasible, generate_full_starts
from v4_validate import validate_v4

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def tiny_objective(tmp_path_factory):
    root = tmp_path_factory.mktemp("v4-objective")
    cfg = copy.deepcopy(load_config(HERE / "configs" / "smoke_v4.json"))
    cfg["time"] = {"scientific_nodes": 3, "acquisition_nodes": 3}
    cfg["moment_reconstruction"]["internal_knots"] = 1
    cfg["projection"].update({"max_steps": 100, "backend": "jax"})
    for block in cfg["v4"]["full_fidelities"].values():
        block.update(
            {
                "trials": 2,
                "time_nodes": 3,
                "grid_nx": 10,
                "grid_ny": 5,
                "cg_tol": 1e-7,
                "cg_maxiter": 160,
            }
        )
    rng = np.random.default_rng(917)
    n = 160
    times = np.linspace(0.0, 1.0, 3)
    base = rng.uniform([0.08, 0.08], [1.92, 0.92], size=(n, 2))
    nodes = np.stack(
        [
            base,
            np.clip(base + np.asarray([0.035, -0.012]), [0.02, 0.02], [1.98, 0.98]),
            np.clip(base + np.asarray([0.065, -0.018]), [0.02, 0.02], [1.98, 0.98]),
        ]
    )
    prospective = np.stack(
        [
            np.clip(nodes[0] + np.asarray([0.012, 0.006]), [0.02, 0.02], [1.98, 0.98]),
            np.clip(nodes[1] + np.asarray([0.018, 0.004]), [0.02, 0.02], [1.98, 0.98]),
            np.clip(nodes[2] + np.asarray([0.024, 0.002]), [0.02, 0.02], [1.98, 0.98]),
        ]
    )
    x_grid = np.linspace(0.0, 2.0, 33)
    y_grid = np.linspace(0.0, 1.0, 17)
    mean, second = _response_fields(
        prospective, x_grid, y_grid, cfg["measurement"]["sensor_width"], 80
    )
    qoi = np.asarray(jnp.mean(qoi_features(jnp.asarray(prospective)), axis=1))
    endpoint = root / "endpoint.npz"
    aggregate = root / "aggregate.npz"
    rollout = root / "rollout.npz"
    np.savez_compressed(
        endpoint,
        role=np.asarray("endpoint_only_reference_training"),
        x0=prospective[0],
        x1=prospective[-1],
    )
    np.savez_compressed(
        aggregate,
        role=np.asarray("prospective_aggregate_only"),
        times=times,
        x_grid=x_grid,
        y_grid=y_grid,
        response_mean_field=mean,
        response_second_field=second,
        scientific_qoi_predictions=qoi,
        qoi_scales=np.ones(5),
    )
    velocity = np.gradient(nodes, times, axis=0)
    np.savez_compressed(
        rollout,
        role=np.asarray("frozen_endpoint_only_reference_rollout"),
        times=times,
        nodes=nodes,
        velocity=velocity,
        weights=np.full((3, n), 1.0 / n),
    )
    data = TargetProspectiveData.load(endpoint, aggregate)
    objective = V4DifferentiableObjective(cfg, data, rollout)
    bank = make_v4_crn_bank(cfg, 2)
    eta = np.asarray([0.35, 0.28, 0.78, 0.68, 1.22, 0.31, 1.64, 0.70])
    return cfg, objective, bank, eta


def _directional_agreement(fn, eta):
    direction = np.asarray([0.2, -0.1, -0.15, 0.2, 0.1, 0.18, -0.12, -0.16])
    direction /= np.linalg.norm(direction)
    value, gradient = jax.value_and_grad(fn)(jnp.asarray(eta))
    h = 2.0e-4
    fd = float((fn(jnp.asarray(eta + h * direction)) - fn(jnp.asarray(eta - h * direction))) / (2.0 * h))
    ad = float(jnp.dot(gradient, jnp.asarray(direction)))
    relative = abs(ad - fd) / max(abs(ad), abs(fd), 1.0e-9)
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert relative < 0.08


def test_v4_risk_and_full_gradients_match_centered_fd(tiny_objective):
    _, objective, bank, eta = tiny_objective
    _directional_agreement(
        lambda e: objective.risk_mean(e, bank.sampling_z, bank.detector_z), eta
    )
    _directional_agreement(
        lambda e: objective.full_score(
            e, bank.sampling_z[:1], bank.detector_z[:1], "search"
        ),
        eta,
    )
    _directional_agreement(
        lambda e: objective.full_score(e, bank.sampling_z, bank.detector_z, "search"),
        eta,
    )


def test_tangent_supplement_gradient_matches_centered_fd(tiny_objective):
    _, objective, bank, eta = tiny_objective
    _directional_agreement(
        lambda e: jnp.mean(
            _tangent_trials(
                objective, e, bank.sampling_z, bank.detector_z
            )[0]
        ),
        eta,
    )


def test_v4_crn_objective_and_gradient_are_deterministic(tiny_objective):
    _, objective, bank, eta = tiny_objective
    fn = jax.jit(
        jax.value_and_grad(
            lambda e: objective.full_score(e, bank.sampling_z, bank.detector_z, "search")
        )
    )
    first = jax.device_get(fn(jnp.asarray(eta)))
    second = jax.device_get(fn(jnp.asarray(eta)))
    # Fixed CRNs make the mathematical objective identical. Parallel GPU
    # reductions may differ by a final floating-point ulp across launches.
    assert np.allclose(np.asarray(first[0]), np.asarray(second[0]), rtol=1e-12, atol=1e-14)
    assert np.allclose(np.asarray(first[1]), np.asarray(second[1]), rtol=1e-12, atol=1e-14)


def test_v4_crn_is_shared_and_seeds_are_separate(tiny_objective):
    cfg, _, bank, _ = tiny_objective
    again = make_v4_crn_bank(cfg, 2)
    assert np.array_equal(bank.sampling_z, again.sampling_z)
    assert np.array_equal(bank.detector_z, again.detector_z)
    assert not np.array_equal(bank.sampling_z, bank.detector_z)


def test_correlated_sampling_uses_off_diagonal_sensor_covariance():
    mean = jnp.asarray([[0.4, 0.5]])
    covariance = jnp.asarray([[[0.04, 0.03], [0.03, 0.09]]])
    cross_second = covariance + mean[..., :, None] * mean[..., None, :]
    sampling_z = jnp.asarray([[[1.0, 0.0]]])
    delta = np.asarray(
        _correlated_sampling_delta(mean, cross_second, 1, sampling_z)
    )
    assert delta[0, 0, 0] == pytest.approx(0.2, abs=1.0e-10)
    assert delta[0, 0, 1] == pytest.approx(0.15, abs=1.0e-10)


def test_prospective_bound_transform_matches_percentage_experiment():
    values = jnp.asarray([0.001, 0.0025, 0.2, 0.9975, 0.999])
    derivatives = jnp.asarray([1.0, -2.0, 0.5, 3.0, -4.0])
    expected = physical_smooth_bound(values, derivatives, 0.002, 0.998, 0.002)
    actual = _smooth_bound_moment_curve(
        values, derivatives, 0.002, 0.998, 0.002
    )
    assert np.allclose(actual[0], expected[0])
    assert np.allclose(actual[1], expected[1])


def test_labelled_sensor_permutations_have_distinct_cache_keys():
    eta = np.asarray([0.3, 0.4, 1.2, 0.7, 0.8, 0.25, 1.6, 0.6])
    permuted = eta.reshape((-1, 2))[[1, 0, 2, 3]].reshape(-1)
    assert canonical_geometry_key(eta) != canonical_geometry_key(permuted)


def test_chunked_adam_matches_scalar_execution(tiny_objective):
    cfg, objective, bank, eta = tiny_objective
    starts = np.stack((eta, eta + np.asarray([0.01, 0.0, 0.0, -0.01, 0.0, 0.01, -0.01, 0.0])))
    settings = {
        **cfg["v4"]["full_optimizer"],
        "steps": 2,
        "batch_size": 1,
    }
    arguments = (
        starts,
        ["test"] * len(starts),
        bank,
        settings,
        cfg,
        lambda e, s, d: objective.full_score(e, s, d, "search"),
    )
    scalar = _adam_multistart(
        *arguments, schedule_seed=73, stage="scalar-test", start_batch_size=1
    )
    chunked = _adam_multistart(
        *arguments, schedule_seed=73, stage="chunked-test", start_batch_size=2
    )
    assert np.allclose(
        [row["final_eta"] for row in scalar],
        [row["final_eta"] for row in chunked],
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    assert np.allclose(
        [row["trace_objective"] for row in scalar],
        [row["trace_objective"] for row in chunked],
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    assert all(row["execution_batch_size"] == 2 for row in chunked)


def test_chunked_adam_resumes_only_at_verified_chunk_boundary(tiny_objective):
    cfg, objective, bank, eta = tiny_objective
    starts = np.stack((
        eta,
        eta + np.asarray([0.01, 0.0, 0.0, -0.01, 0.0, 0.01, -0.01, 0.0]),
        eta + np.asarray([0.0, 0.01, -0.01, 0.0, 0.01, 0.0, 0.0, -0.01]),
        eta + np.asarray([-0.01, 0.0, 0.01, 0.0, 0.0, -0.01, 0.01, 0.0]),
    ))
    settings = {**cfg["v4"]["full_optimizer"], "steps": 1, "batch_size": 1}
    arguments = (
        starts,
        ["resume-test"] * len(starts),
        bank,
        settings,
        cfg,
        lambda e, s, d: objective.full_score(e, s, d, "search"),
    )
    complete = _adam_multistart(
        *arguments, schedule_seed=79, stage="complete-test", start_batch_size=2
    )
    callback_lengths = []
    resumed = _adam_multistart(
        *arguments,
        schedule_seed=79,
        stage="resume-test",
        start_batch_size=2,
        completed_rows=complete[:2],
        chunk_callback=lambda rows: callback_lengths.append(len(rows)),
    )
    assert resumed[:2] == complete[:2]
    assert np.allclose(
        [row["final_eta"] for row in resumed],
        [row["final_eta"] for row in complete],
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    assert np.allclose(
        [row["trace_objective"] for row in resumed],
        [row["trace_objective"] for row in complete],
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    assert callback_lengths == [4]
    with pytest.raises(ValueError, match="chunk boundary"):
        _adam_multistart(
            *arguments,
            schedule_seed=79,
            stage="bad-resume-test",
            start_batch_size=2,
            completed_rows=complete[:1],
        )


def test_v4_reconstruction_gradient_matches_fd():
    t_obs = np.linspace(0.0, 1.0, 5)
    reconstructor = AnchoredCubicSplineReconstructor(
        t_obs,
        np.linspace(0.0, 1.0, 9),
        AnchoredCubicSplineConfig(internal_knots=1, smoothing=1e-4),
    )
    y = jnp.asarray(np.linspace(0.2, 0.7, 10).reshape(5, 2))
    direction = jnp.asarray(np.linspace(-0.2, 0.3, 10).reshape(5, 2))

    def fn(values):
        fit = reconstructor.reconstruct(values, values[0], values[-1])
        return jnp.sum(fit.c * fit.c) + 0.01 * jnp.sum(fit.c_dot * fit.c_dot)

    gradient = jax.grad(fn)(y)
    h = 1e-5
    fd = float((fn(y + h * direction) - fn(y - h * direction)) / (2.0 * h))
    ad = float(jnp.sum(gradient * direction))
    assert abs(ad - fd) / max(abs(ad), abs(fd), 1e-10) < 2e-5


def test_v4_full_starts_do_not_depend_on_tangent():
    cfg = load_config(HERE / "configs" / "smoke_v4.json")
    law = np.asarray([0.35, 0.28, 0.78, 0.68, 1.22, 0.31, 1.64, 0.70])
    starts, provenance = generate_full_starts(cfg, law, tangent_eta=None)
    assert len(starts) == cfg["v4"]["full_optimizer"]["starts"]
    assert len(provenance) == len(starts)
    assert all("Full" in source for source in provenance)


def test_v4_exact_risk_constraint_rule():
    cfg = load_config(HERE / "configs" / "smoke_v4.json")
    assert _risk_feasible(1.02, 1.0, cfg)
    assert not _risk_feasible(1.020001, 1.0, cfg)


def test_v4_selection_has_no_validation_or_hidden_import():
    tree = ast.parse((HERE / "v4_select.py").read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "v4_validate" not in imports
    assert "validate" not in imports
    assert "diagnose_frozen_selection" not in imports


def test_v4_validation_refuses_to_run_without_freeze(tmp_path):
    cfg = load_config(HERE / "configs" / "smoke_v4.json")
    with pytest.raises(RuntimeError, match="frozen pre-validation manifest"):
        validate_v4(cfg, tmp_path)


def test_v4_covariance_condition_certificate():
    covariance = np.asarray(
        [[[[2.0, 0.0], [0.0, 0.5]], [[3.0, 0.0], [0.0, 1.0]]]],
        dtype=np.float64,
    )
    row = _condition_rows(covariance, ridge=0.5)[0]
    assert row["min_covariance_eigenvalue"] == pytest.approx(0.5)
    assert row["max_covariance_eigenvalue"] == pytest.approx(3.0)
    assert row["max_raw_covariance_condition_number"] == pytest.approx(4.0)
    assert row["max_ridge_regularized_condition_number"] == pytest.approx(2.5)
