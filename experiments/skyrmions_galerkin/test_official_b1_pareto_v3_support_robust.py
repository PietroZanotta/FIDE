from __future__ import annotations

import ast
from pathlib import Path

from . import official_b1_pareto_v3_support_robust as study


def _row(value: float, risk: float, ress: float = 0.1) -> dict:
    eta = [value, 0.0, 0.0, 0.0]
    return {
        "candidate_id": f"row_{value}",
        "eta": eta,
        "eta_sha256": study.base.eta_key(eta),
        "exact_scientific_risk": risk,
        "minimum_rESS": ress,
        "jointly_supported": True,
    }


def test_v3_single_root_and_fresh_roles_are_unique() -> None:
    assert study.ROOT_SEED == 20261003
    assert len(study.ROLE_IDS) == len(set(study.ROLE_IDS.values()))
    records = [study._role_record(role) for role in study.ROLE_IDS]
    assert len({tuple(row["jax_key_words_uint32"]) for row in records}) == len(records)
    assert len({row["integer_seed_adapter"] for row in records}) == len(records)
    assert "authoritative_audit" not in study.GUARD_COUNTS
    assert study.FRESH_STANDARD_ROLES["authoritative_audit"] == 65536


def test_vortices_up_to_cap_rule_preserves_mandatory_starts() -> None:
    rows = [_row(0.1, 1.0), _row(0.2, 1.1), _row(0.3, 1.2)]
    selected = study._amended_select_starts(rows, rows[0], None, count=6)
    assert len(selected) == 3
    assert selected[0]["mandatory_roles"] == ["mandatory_law"]
    assert all(row["start_availability"]["all_available_used_when_below_cap"] for row in selected)


def test_v3_sources_cannot_reach_native_galerkin() -> None:
    forbidden_modules = {"mfsi.galerkin_tesseract", ".pareto_v2_selection"}
    forbidden_calls = {"assemble_galerkin_chunk_tesseract", "evaluate_galerkin_action"}
    for source in (Path(study.__file__), study.RUNNER_PATH):
        tree = ast.parse(source.read_text(encoding="utf-8"))
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
        assert not (imports & forbidden_modules)
        assert not (calls & forbidden_calls)
