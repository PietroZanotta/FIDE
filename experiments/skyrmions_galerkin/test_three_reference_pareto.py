from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from mfsi.config import load_config

from . import three_reference_pareto as study
from .pareto_v3_common import file_sha256


def test_three_frozen_b1_references() -> None:
    assert len(study.FLOW_IDS) == 3
    assert len(set(study.FLOW_SHA256.values())) == 3
    for flow_id in study.FLOW_IDS:
        assert file_sha256(study.FLOW_PATHS[flow_id]) == study.FLOW_SHA256[flow_id]


def test_robust_protocol_semantics() -> None:
    cfg = load_config(study.CONFIG_PATH)
    payload = study.protocol_payload(cfg)
    assert payload["reference"]["equal_weight"] is True
    assert payload["reference"]["matched_initial_configurations_across_flows"] is True
    assert "separately for every flow" in payload["risk_rule"]
    assert payload["allowances_percent"] == list(study.ALLOWANCES)
    assert payload["deep_ritz_used"] is False


def test_bank_paths_are_flow_isolated() -> None:
    paths = {study._bank_path("screen", flow_id) for flow_id in study.FLOW_IDS}
    assert len(paths) == 3
    assert all(study.OUTPUT_ROOT in path.parents for path in paths)


def test_reference_screen_uses_target_centered_forcing_mean() -> None:
    """Screening must reproduce the production forcing compatibility residual."""
    problem = SimpleNamespace(
        forcing_config=SimpleNamespace(covariance_ridge=0.0),
        time_weights=jnp.ones((1,), dtype=jnp.float64),
    )
    evaluator = object.__new__(study._TargetCenteredReferenceEvaluator)
    evaluator.problem = problem
    evaluator.truth_means = jnp.zeros((1, 1), dtype=jnp.float64)
    evaluator.whitening = jnp.eye(1, dtype=jnp.float64)
    evaluator.postprocessors = {}
    postprocess = evaluator._postprocessor(2)
    result = postprocess(
        jnp.asarray([[[0.5, 0.5]]]),
        jnp.zeros((1, 1, 1)),
        jnp.asarray([[[1.0]]]),
        jnp.asarray([[[[1.0]]]]),
        jnp.asarray([[[0.1]]]),
        jnp.ones((1, 1)),
        jnp.asarray([[[[0.0], [2.0]]]]),
        jnp.zeros((1, 1, 2, 1)),
        jnp.ones((1, 1, 1)),
        jnp.zeros((1, 2, 1)),
    )
    forcing_mean = np.asarray(result[7])
    np.testing.assert_allclose(forcing_mean, [[0.1]], rtol=0.0, atol=1e-14)
