from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

import jax.numpy as jnp
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for path in (HERE, REPO / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import load_config
from v6_objective import V6MultiReferenceObjective
from v6_reference_ensemble import _reference_cfg
from run_v6_positive_raster import apply_execution_profile, load_repair_config
from run_v6a_risk_study import (
    PARETO_ALLOWANCES,
    _adopt_design_references,
    prepare_study,
    resolve_study_config,
)
from v6_select import (
    _explicit_incumbent_run,
    _retain_mandatory_candidates,
    _risk_feasible,
    _select_prescreen_indices,
    combine_freeze,
)
from v6_validate import validate_v6


def test_v6_beta_arms_differ_only_in_identity_and_beta():
    a = json.loads((HERE / "configs" / "production_v6a.json").read_text())
    b = json.loads((HERE / "configs" / "production_v6b.json").read_text())
    assert a["base_config"] == b["base_config"]
    assert a["beta"] == 0.0
    assert b["beta"] == 0.25
    assert set(a) == {"schema_version", "name", "base_config", "beta"}
    assert set(b) == set(a)


def test_v6_base_config_uses_final_reflected_raster():
    base = load_config(HERE / "configs" / "production_v6_common.json")
    raster = base["raster"]["reflected"]
    assert raster["image_pairs"] == 4
    assert raster["source_column_normalization"] is False
    assert raster["density_floor"] == 0.0
    assert raster["grid_bandwidth_floor"] is False


def test_v6_exact_execution_profile_is_opt_in_and_preserves_protocol():
    resolved = resolve_study_config("single-seed-pareto")
    assert resolved["v6_fast_execution"] == {
        "law_start_batch_size": 8,
        "tangent_start_batch_size": 8,
        "full_start_batch_size": 8,
        "polish_start_batch_size": 4,
        "prescreen_optimize_starts": 16,
        "prescreen_start_batch_size": 8,
    }
    assert resolved["v4"]["full_optimizer"]["starts"] == 16
    assert resolved["v4"]["full_lbfgs"]["enabled"] is False


def test_v6a_risk_study_has_one_seed_and_only_requested_allowances():
    pareto = resolve_study_config("single-seed-pareto")
    assert pareto["v6a_risk_study"]["allowances"] == list(PARETO_ALLOWANCES)
    assert pareto["v6"]["design_reference_ids"] == ["D0"]
    assert pareto["v6"]["evaluation_reference_ids"] == ["E0"]
    assert set(pareto["v6"]["design_reference_training_seeds"]).isdisjoint(
        pareto["v6"]["evaluation_reference_training_seeds"]
    )
    assert pareto["v6a_risk_study"]["v6b_excluded"] is True
    assert pareto["v6a_risk_study"]["source_run"] is None
    assert pareto["v6_fast_execution"]["full_start_batch_size"] == 8
    assert pareto["v6_fast_execution"]["polish_start_batch_size"] == 4
    assert pareto["v6_fast_execution"]["tangent_start_batch_size"] == 8


def test_v6a_pareto_incumbent_archive_row_is_unmodified_and_zero_cost():
    eta = jnp.linspace(0.24, 0.76, 8)
    row = _explicit_incumbent_run(eta, "pareto-test", "previous winner")
    assert row["initial_eta"] == row["final_eta"]
    assert row["iterations"] == 0
    assert row["runtime_seconds"] == 0.0
    assert row["provenance"] == "previous winner"


def test_v6a_full_funnel_retains_law_and_pareto_feasibility_anchors():
    optional = {"candidate_id": "full-grad-001"}
    law = {"candidate_id": "full-law-incumbent"}
    pareto = {"candidate_id": "full-pareto-incumbent"}
    retained = _retain_mandatory_candidates([optional], [optional, law, pareto])
    assert [row["candidate_id"] for row in retained] == [
        "full-grad-001",
        "full-pareto-incumbent",
        "full-law-incumbent",
    ]
    assert _retain_mandatory_candidates(retained, [optional, law, pareto]) == retained


def test_v6a_pareto_does_not_adopt_an_existing_experiment():
    cfg = resolve_study_config("single-seed-pareto")
    assert cfg["v6a_risk_study"]["reuse_existing_design_references"] is False
    assert cfg["v6a_risk_study"]["source_run"] is None
    assert cfg["v6"]["design_reference_training_seeds"] == [20266101]


def test_v6_reference_and_hidden_seed_registry_is_disjoint():
    cfg = load_config(HERE / "configs" / "production_v6_common.json")
    reference = set(cfg["v6"]["design_reference_training_seeds"])
    reference |= set(cfg["v6"]["evaluation_reference_training_seeds"])
    reference |= {
        cfg["v6"]["design_reference_rollout_seed"],
        cfg["v6"]["evaluation_reference_rollout_seed"],
    }
    selection = {value for key, value in cfg["seeds"].items() if not key.startswith("validation_")}
    hidden = {value for key, value in cfg["seeds"].items() if key.startswith("validation_")}
    assert len(reference) == 4
    assert reference.isdisjoint(selection)
    assert reference.isdisjoint(hidden)
    assert selection.isdisjoint(hidden)


def test_v6_reference_cfg_varies_training_seed_but_pairs_rollout_seed():
    cfg = load_config(HERE / "configs" / "production_v6_common.json")
    first = _reference_cfg(cfg, "D0", 101, 900)
    second = _reference_cfg(cfg, "D1", 102, 900)
    assert first["seed"] != second["seed"]
    assert first["seed"] + first["reference"]["seed_offset"] == 900
    assert second["seed"] + second["reference"]["seed_offset"] == 900
    assert first["truth"] == second["truth"]
    assert first["reference_training"] == second["reference_training"]


def test_v6_joint_robust_score_uses_reference_trial_distribution():
    values = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    score0, mean, sd = V6MultiReferenceObjective.robust_score(values, 0.0)
    score25, _, _ = V6MultiReferenceObjective.robust_score(values, 0.25)
    assert float(mean) == pytest.approx(2.5)
    assert float(sd) == pytest.approx(1.2909944487358056)
    assert float(score0) == pytest.approx(2.5)
    assert float(score25) == pytest.approx(2.5 + 0.25 * 1.2909944487358056)


def test_v6_exact_risk_gate_requires_every_reference():
    cfg = load_config(HERE / "configs" / "production_v6_common.json")
    assert _risk_feasible([1.01, 2.01, 3.01], [1.02, 2.02, 3.02], cfg)
    assert not _risk_feasible([1.01, 2.020001, 3.01], [1.02, 2.02, 3.02], cfg)


def test_v6_selection_has_no_validation_import():
    tree = ast.parse((HERE / "v6_select.py").read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "v6_validate" not in imports
    assert "v4_validate" not in imports


def test_canonical_tree_has_no_retired_or_sibling_experiment_dependency():
    forbidden = (
        "vortices_prospective_new",
        "old_stuff/vortices_prospective",
        'HERE.parent / "vortices_percentage"',
    )
    for path in HERE.glob("*.py"):
        if path.name == Path(__file__).name:
            continue
        source = path.read_text(encoding="utf-8")
        assert not any(value in source for value in forbidden), path


def test_v6_prescreen_keeps_mandatory_neighborhood_and_best_optional_starts():
    scores = jnp.asarray([100.0, 90.0, 80.0, 5.0, 3.0, 4.0])
    assert _select_prescreen_indices(scores, mandatory_count=3, keep=5) == [0, 1, 2, 4, 5]
    with pytest.raises(ValueError, match="invalid prescreen counts"):
        _select_prescreen_indices(scores, mandatory_count=3, keep=2)


def test_v6_freeze_and_validation_fail_closed_without_prerequisites(tmp_path):
    cfg = load_config(HERE / "configs" / "production_v6_common.json")
    with pytest.raises(RuntimeError, match="requires shared selection and both arms"):
        combine_freeze(cfg, tmp_path)
    with pytest.raises(RuntimeError, match="prerequisite missing"):
        validate_v6(cfg, tmp_path)
