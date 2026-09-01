from __future__ import annotations

import ast
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from . import official_b1_pareto_v2_single_seed as study
from .galerkin import rank_aware_quadratic_solve


def test_frozen_semantics_constants() -> None:
    values = study._v2_values()
    assert values["root_seed"] == 20261003
    assert values["allowances_percent"] == [0.5, 1.0, 2.0]
    assert values["K"] == 280
    assert values["relative_rank_tolerance"] == 1e-12
    assert values["projection_backend"] == "jax"
    assert values["galerkin_backend"] == "jax"
    assert sum(values["candidate_components"].values()) == 5645


def test_jax_rank_rule_and_sign() -> None:
    gram = jnp.asarray([[[4.0, 0.0], [0.0, 1.0e-14]]], dtype=jnp.float64)
    load = jnp.asarray([[2.0, 3.0]], dtype=jnp.float64)
    solve = rank_aware_quadratic_solve(
        gram, load, relative_rank_tolerance=1.0e-12
    )
    assert int(solve.numerical_rank[0]) == 1
    assert jnp.allclose(solve.coefficients[0], jnp.asarray([-0.5, 0.0]))
    assert jnp.allclose(solve.action_by_time[0], 1.0)


def test_native_galerkin_unreachable_from_v2_sources() -> None:
    source_names = (
        "official_b1_pareto_v2_single_seed.py",
        "jax_galerkin_v2.py",
        "official_b1_pareto_v2_single_seed_run.py",
    )
    for name in source_names:
        source = (study.ROOT / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = set()
        calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(("." * node.level) + (node.module or ""))
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
        assert "mfsi.galerkin_tesseract" not in imports
        assert ".pareto_v2_selection" not in imports
        assert "assemble_galerkin_chunk_tesseract" not in calls


def test_protocol_seal_when_present() -> None:
    if study.PROTOCOL_PATH.exists():
        protocol = study.require_protocol()
        assert protocol["single_seed"] is True
        assert protocol["solver"]["galerkin_backend"] == "jax"
        assert protocol["solver"]["native_fallback"] is False
        assert protocol["candidate_universe"]["frozen_before_outcomes"] is True
        assert protocol["feasibility_first"] is True


def test_old_authority_hash_when_present() -> None:
    assert study.tree_sha256(study.V1_ROOT) == study.V1_TREE_SHA256_BEFORE
