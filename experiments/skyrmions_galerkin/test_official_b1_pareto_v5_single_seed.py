"""Pre-freeze regression tests for the V5 authority path."""

from __future__ import annotations

import ast

import jax
import jax.numpy as jnp
import numpy as np

from . import official_b1_pareto_v5_single_seed as study


def test_v5_frozen_constants() -> None:
    values = study._values()
    assert study.ROOT_SEED == 20261005
    assert values["allowances_percent"] == [0.5, 1.0, 2.0]
    assert values["K"] == 280 == study.K
    assert values["relative_rank_tolerance"] == 1e-12
    assert values["law_consistency_tolerance"] == 1e-4
    assert values["maximum_anchor_refinement_restarts"] == 2
    assert values["bank_sizes"]["authoritative_train"] == 131072
    assert values["bank_sizes"]["authoritative_audit"] == 131072
    assert values["validation"]["reference_fit_samples"] == 131072
    assert values["validation"]["reference_audit_samples"] == 131072
    assert values["feature_sample_chunk"] == 8192
    assert len(study.RISK_GUARD_COUNTS) == 4
    assert all(row == {"truth": 5000, "reference": 65536} for row in study.RISK_GUARD_COUNTS.values())
    assert study.RISK_CANDIDATE_BATCH_SIZE == 32


def test_all_random_roles_are_unique_fold_in_descendants() -> None:
    records = study.randomness_records()
    assert len(records) == len(study.ROLE_IDS)
    assert len({row["role_id"] for row in records}) == len(records)
    assert len({tuple(row["jax_key_words_uint32"]) for row in records}) == len(records)
    assert all(f"PRNGKey({study.ROOT_SEED})" in row["derivation"] for row in records)


def test_static_v5_scientific_path_has_no_native_branch() -> None:
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
    assert cfg["production_galerkin"]["maximum_range_residual"] == 1e-8
    assert cfg["production_galerkin"]["maximum_stationarity_residual"] == 1e-8


def test_five_role_feasibility_is_role_paired_and_conjunctive() -> None:
    law = {
        "risk_by_role": {role: 2.0 + index for index, role in enumerate(study.RISK_ROLE_NAMES)}
    }
    receipt = {
        "jointly_supported": True,
        "all_five_risk_roles_valid": True,
        "risk_by_role": {
            role: law["risk_by_role"][role] * 1.004
            for role in study.RISK_ROLE_NAMES
        },
    }
    assert study.V5SelectionRuntime.robust_feasible(receipt, law, 0.5)
    failed_role = study.RISK_ROLE_NAMES[-1]
    receipt["risk_by_role"][failed_role] = law["risk_by_role"][failed_role] * 1.006
    assert not study.V5SelectionRuntime.robust_feasible(receipt, law, 0.5)
    assert study.V5SelectionRuntime.robust_feasible(receipt, law, 1.0)
    receipt["all_five_risk_roles_valid"] = False
    assert not study.V5SelectionRuntime.robust_feasible(receipt, law, 2.0)
