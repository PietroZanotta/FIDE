from __future__ import annotations

import ast
from pathlib import Path

from . import v4_energy_development as study


def test_frozen_development_grid() -> None:
    assert study.N_LADDER == (16384, 32768, 65536, 131072)
    assert study.K_LADDER == (120, 180, 220, 280, 360, 440)
    assert study.K_N_GRID == (180, 280, 360, 440)
    assert study.RANK_TOLERANCE == 1e-12
    assert study.ENERGY_FLOOR == 1e-12
    assert study.SPLIT_COUNT == 4


def test_diagnostic_role_seeds_unique() -> None:
    assert len(study.ROLE_IDS) == len(set(study.ROLE_IDS.values()))
    assert len({study._role_seed(role) for role in study.ROLE_IDS}) == len(study.ROLE_IDS)


def test_energy_formula_known_identity() -> None:
    import jax.numpy as jnp
    from .galerkin import GalerkinSystem

    empty = jnp.zeros((0,), dtype=jnp.float64)
    fit = GalerkinSystem(
        gram=jnp.asarray([[[2.0]]]), load=jnp.asarray([[1.0]]),
        basis_means=jnp.zeros((1, 1)), centered_basis=empty,
        weights=empty, forcing=empty, raw_symmetry_residual=jnp.zeros((1,)),
        forcing_mean=jnp.zeros((1,)),
    )
    audit = GalerkinSystem(
        gram=jnp.asarray([[[3.0]]]), load=jnp.asarray([[1.4]]),
        basis_means=jnp.zeros((1, 1)), centered_basis=empty,
        weights=empty, forcing=empty, raw_symmetry_residual=jnp.zeros((1,)),
        forcing_mean=jnp.zeros((1,)),
    )
    row = study._energy_rows(jnp.asarray([[-0.5]]), fit, audit)[0]
    assert abs(row["audit_quadratic"] - 0.75) < 1e-14
    assert abs(row["audit_linear"] + 0.7) < 1e-14
    assert abs(row["numerator"] - 0.05) < 1e-14
    assert abs(row["denominator"] - 1.45) < 1e-14


def test_development_sources_cannot_reach_native_galerkin() -> None:
    forbidden_modules = {"mfsi.galerkin_tesseract", ".pareto_v2_selection"}
    forbidden_calls = {"assemble_galerkin_chunk_tesseract", "evaluate_galerkin_action"}
    for source in (Path(study.__file__), study.RUNNER_PATH):
        tree = ast.parse(source.read_text())
        imports, calls = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add("." * node.level + (node.module or ""))
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
        assert not imports & forbidden_modules
        assert not calls & forbidden_calls


def test_dictionary_npz_writer_is_development_scoped() -> None:
    source = Path(study.__file__).read_text()
    assert "save_dictionary(" not in source
    assert "require_production_output_path" not in source
