"""Pre-freeze regression tests for the V4 authority path."""

from __future__ import annotations

import ast

import jax
import jax.numpy as jnp
import numpy as np

from . import official_b1_pareto_v4_single_seed as study


def test_v4_frozen_constants() -> None:
    values = study._values()
    assert study.ROOT_SEED == 20261004
    assert values["allowances_percent"] == [0.5, 1.0, 2.0]
    assert values["K"] == 280 == study.K
    assert values["relative_rank_tolerance"] == 1e-12
    assert values["law_consistency_tolerance"] == 1e-4
    assert values["maximum_anchor_refinement_restarts"] == 2
    assert values["validation"]["reference_fit_samples"] == 65536
    assert values["validation"]["reference_audit_samples"] == 65536
    assert values["feature_sample_chunk"] == 8192


def test_all_random_roles_are_unique_fold_in_descendants() -> None:
    records = study.randomness_records()
    assert len(records) == len(study.ROLE_IDS)
    assert len({row["role_id"] for row in records}) == len(records)
    assert len({tuple(row["jax_key_words_uint32"]) for row in records}) == len(records)
    assert all(f"PRNGKey({study.ROOT_SEED})" in row["derivation"] for row in records)


def test_static_v4_scientific_path_has_no_native_branch() -> None:
    graph = study._static_call_graph()
    assert graph["passed"]
    assert not graph["native_galerkin_reachable"]
    assert not graph["violations"]
    source = ast.parse(study.SOURCE_PATH.read_text(encoding="utf-8"))
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(source) if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "assemble_galerkin_chunk_tesseract" not in calls
    assert "evaluate_galerkin_action" not in calls


def test_chunked_many_body_features_is_scientifically_identical(monkeypatch) -> None:
    key = jax.random.PRNGKey(91827)
    rows = jax.random.uniform(
        key, (2, 17, 24, 2), minval=0.0, maxval=1.0, dtype=jnp.float64
    )
    expected = study.ORIGINAL_MANY_BODY_FEATURES(rows, (2.0, 1.0))
    monkeypatch.setattr(study, "FEATURE_SAMPLE_CHUNK", 8)
    observed = study.chunked_many_body_features(rows, (2.0, 1.0))
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


def test_energy_contract_is_unchanged() -> None:
    cfg = study.base.effective_config()
    assert cfg["production_galerkin"]["certificate_thresholds"] == {
        "maximum_weak_residual": 0.12,
        "maximum_energy_residual": 0.08,
        "maximum_gauge_residual": 1e-9,
        "maximum_moment_rate_residual": 0.10,
    }
