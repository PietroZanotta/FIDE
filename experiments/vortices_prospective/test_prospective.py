from __future__ import annotations

import ast
import importlib.util
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

from build_prospective_data import _response_fields, build
from common import load_config
from mfsi.projection import EmpiricalIProjector, IProjectionConfig
from physical import gaussian_response_direct
from prospective_data import TargetProspectiveData, _bilinear_table
from train_reference import EndpointOnlySource
from validate import validate

jax.config.update("jax_enable_x64", True)


def _load_selection_module():
    spec = importlib.util.spec_from_file_location("prospective_selection_test", HERE / "select.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_minimal_artifacts(root: Path):
    endpoint = root / "endpoint.npz"
    aggregate = root / "aggregate.npz"
    x = np.asarray([[0.2, 0.3], [1.2, 0.7]], dtype=np.float64)
    np.savez_compressed(
        endpoint, x0=x, x1=x[::-1], role=np.asarray("endpoint_only_reference_training")
    )
    np.savez_compressed(
        aggregate,
        role=np.asarray("prospective_aggregate_only"),
        times=np.asarray([0.0, 1.0]),
        x_grid=np.asarray([0.0, 2.0]),
        y_grid=np.asarray([0.0, 1.0]),
        response_mean_field=np.zeros((2, 2, 2)),
        response_second_field=np.zeros((2, 2, 2)),
        scientific_qoi_predictions=np.zeros((2, 5)),
        qoi_scales=np.ones(5),
    )
    return endpoint, aggregate


def test_prospective_interface_has_no_hidden_state_access(tmp_path):
    endpoint, aggregate = _write_minimal_artifacts(tmp_path)
    data = TargetProspectiveData.load(endpoint, aggregate)
    names = set(data.__dataclass_fields__)
    assert "states" not in names
    assert "hidden_path" not in names
    assert not any("hidden" in name for name in names)

    hidden = tmp_path / "hidden_validation"
    hidden.mkdir()
    hidden_endpoint = hidden / "endpoint.npz"
    hidden_endpoint.write_bytes(endpoint.read_bytes())
    with pytest.raises(PermissionError):
        TargetProspectiveData.load(hidden_endpoint, aggregate)

    tree = ast.parse((HERE / "select.py").read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "validate" not in imported
    assert "physical" not in imported
    assert "domain" not in imported
    evaluator_tree = ast.parse((HERE / "evaluator.py").read_text(encoding="utf-8"))
    evaluator_imports = {
        alias.name
        for node in ast.walk(evaluator_tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "physical" not in evaluator_imports
    assert "domain" not in evaluator_imports
    assert "validate" not in evaluator_imports


def test_response_interpolation_agrees_with_direct_aggregate():
    rng = np.random.default_rng(19)
    states = rng.uniform([0.0, 0.0], [2.0, 1.0], size=(2, 600, 2))
    x_grid = np.linspace(0.0, 2.0, 65)
    y_grid = np.linspace(0.0, 1.0, 33)
    mean, second = _response_fields(states, x_grid, y_grid, 0.12, 200)
    centers = np.asarray([[0.37, 0.42], [1.21, 0.73], [1.65, 0.28]])
    predicted = np.asarray(_bilinear_table(mean, x_grid, y_grid, centers))
    direct = np.asarray(gaussian_response_direct(states, centers, 0.12))
    assert np.max(np.abs(predicted - direct)) < 2.5e-3
    assert np.all(second >= mean * mean - 1.0e-12)


def test_response_table_recovers_cross_sensor_second_moments():
    rng = np.random.default_rng(23)
    states = rng.uniform([0.0, 0.0], [2.0, 1.0], size=(2, 1200, 2))
    x_grid = np.linspace(0.0, 2.0, 97)
    y_grid = np.linspace(0.0, 1.0, 49)
    width = 0.12
    mean, second = _response_fields(states, x_grid, y_grid, width, 300)
    data = TargetProspectiveData(
        endpoint_path=Path("endpoint.npz"),
        aggregate_path=Path("aggregate.npz"),
        endpoint_ensemble_0=states[0],
        endpoint_ensemble_1=states[-1],
        times=np.asarray([0.0, 1.0]),
        x_grid=x_grid,
        y_grid=y_grid,
        response_mean_field=mean,
        response_second_field=second,
        scientific_qoi_predictions=np.zeros((2, 5)),
        qoi_scales=np.ones(5),
        metadata={},
    )
    centers = np.asarray([[0.71, 0.46], [0.88, 0.53], [1.42, 0.67]])
    predicted = np.asarray(data.response_cross_second(centers, width))
    delta = states[:, :, None, :] - centers[None, None, :, :]
    phi = np.exp(-0.5 * np.sum(delta * delta, axis=-1) / width**2)
    direct = np.einsum("tnj,tnk->tjk", phi, phi) / phi.shape[1]
    assert np.max(np.abs(predicted - direct)) < 2.5e-3
    assert np.allclose(predicted, np.swapaxes(predicted, -1, -2))
    assert np.max(np.abs(predicted[:, 0, 1])) > 1.0e-3


def test_endpoint_source_exposes_endpoints_only():
    source = EndpointOnlySource(jnp.zeros((4, 2)), jnp.ones((4, 2)))
    assert np.allclose(source.sample(jax.random.key(1), 3, 0), 0.0)
    assert np.allclose(source.sample(jax.random.key(2), 3, 1), 1.0)
    with pytest.raises(ValueError):
        source.sample(jax.random.key(3), 3, 2)


def test_information_projection_calibrates_declared_moment():
    phi = jnp.asarray([[0.0], [0.25], [0.5], [0.75], [1.0]])
    base = jnp.full((5,), 0.2)
    target = jnp.asarray([0.63])
    projector = EmpiricalIProjector(
        IProjectionConfig(max_steps=100, residual_tol=1.0e-10, newton_ridge=1.0e-9)
    )
    state = projector.project(phi, base, target)
    assert float(jnp.linalg.norm(state.residual)) < 1.0e-8
    assert np.isclose(float(state.weights @ phi[:, 0]), 0.63, atol=1.0e-8)


def test_full_risk_constraint_rule():
    selection = _load_selection_module()
    assert selection.risk_feasible(1.02, 1.0, 0.02, 1.0e-12)
    assert not selection.risk_feasible(1.02001, 1.0, 0.02, 1.0e-12)


def test_validation_requires_frozen_manifest(tmp_path):
    cfg = load_config(HERE / "configs" / "smoke.json")
    with pytest.raises(RuntimeError, match="frozen manifest"):
        validate(cfg, tmp_path)


def test_aggregate_preprocessing_cache_reuse(tmp_path):
    cfg = load_config(HERE / "configs" / "smoke.json")
    cfg["truth"]["endpoint_particles"] = 24
    cfg["truth"]["prospective_particles"] = 32
    cfg["truth"]["endpoint_rk4_substeps"] = 2
    cfg["truth"]["rk4_substeps_per_interval"] = 1
    cfg["time"]["scientific_nodes"] = 3
    cfg["time"]["acquisition_nodes"] = 3
    cfg["aggregate_predictor"].update({"grid_nx": 7, "grid_ny": 5, "particle_chunk": 16})
    first = build(cfg, tmp_path)
    endpoint = Path(first["endpoint_path"])
    aggregate = Path(first["aggregate_path"])
    mtimes = (endpoint.stat().st_mtime_ns, aggregate.stat().st_mtime_ns)
    second = build(cfg, tmp_path)
    assert second == first
    assert (endpoint.stat().st_mtime_ns, aggregate.stat().st_mtime_ns) == mtimes
