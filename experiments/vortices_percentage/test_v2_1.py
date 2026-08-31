from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from v2_1_contract import CONFIG, generated_starts, load_resolved_config


HERE = Path(__file__).resolve().parent


def test_v2_1_overlay_preserves_frozen_science_and_uses_only_two_digit_new_randomness():
    resolved, overlay = load_resolved_config(require_frozen=False)
    randomness = overlay["two_digit_randomness"]
    values = [
        randomness["observation_generation_seed"],
        randomness["selection_namespace"],
        randomness["stress_test_namespace"],
        randomness["validation_namespace"],
        randomness["optimizer_root_seed"],
        randomness["bootstrap_seed"],
        *randomness["tangent_local_cloud_seed_by_allowance"].values(),
    ]
    for row in randomness["full_local_cloud_seeds_by_allowance_and_round"].values():
        values.extend(row)
    assert len(values) == len(set(values))
    assert all(10 <= int(value) <= 99 for value in values)
    assert resolved["reference_replicates"]["training_seeds"] == [310000101, 310000102, 310000103]
    assert resolved["reference_replicates"]["rollout"]["seeds"] == [310003102, 310003103, 310003104]
    assert resolved["scientific_evaluator"]["exact_grid"] == [256, 128]
    assert resolved["scientific_evaluator"]["scientific_time_indices"] == list(range(21))
    assert resolved["scientific_evaluator"]["precision"] == "float64"
    assert resolved["scientific_evaluator"]["density_floor"] == 0.0


def test_v2_1_generated_start_pool_is_deterministic_and_fresh():
    resolved, _ = load_resolved_config(require_frozen=False)
    first = generated_starts(resolved)
    second = generated_starts(resolved)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (64, 8)
    base = json.loads((HERE / "VORTICES_V2_SELECTION_CONFIG.json").read_text(encoding="utf-8"))
    assert resolved["optimization"]["optimizer_root_seed"] == 14
    assert base["optimization"]["optimizer_root_seed"] != 14
    assert not np.array_equal(first, generated_starts({**resolved, "optimization": {**resolved["optimization"], "optimizer_root_seed": 15}}))


def test_v2_1_full_rule_is_feasibility_first_and_has_mandatory_baselines():
    overlay = json.loads(CONFIG.read_text(encoding="utf-8"))
    full = overlay["optimization_overrides"]["full"]
    assert full["exact_population_and_risk_audit"] == "every_unique_candidate_before_proxy"
    assert full["proxy_eligibility"] == "jointly_feasible_candidates_only"
    assert full["mandatory_finalists_at_0p5"] == ["new_law", "current_tangent"]
    assert full["mandatory_finalists_later"] == ["previous_tighter_full", "current_tangent"]
    assert full["promoted_candidates"] == 8
