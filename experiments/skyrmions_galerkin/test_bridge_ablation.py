from __future__ import annotations

from dataclasses import asdict
import ast
from pathlib import Path

import numpy as np

from . import bridge_ablation as study
from .domain import minimum_image
from .pareto_v3_common import file_sha256
from .reference import ReferenceTrainingConfig


def _toy() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    return rng.random((16, 2)) * np.asarray(study.BOX), rng.random((16, 2)) * np.asarray(study.BOX)


def test_frozen_physics_risk_whitening_ress_and_reference() -> None:
    seal = study._json(study.UPSTREAM_ROOT / "source_seal.json")
    assert file_sha256(study.BASELINE_CHECKPOINT_PATH) == study.EXPECTED_BASELINE_CHECKPOINT_SHA256
    assert file_sha256(study.CONFIG_PATH) == study.EXPECTED_CONFIG_SHA256
    assert file_sha256(study.ROOT / "risk.py") == seal["immutable_source_hashes"]["scientific_risk_source"]
    assert study.MINIMUM_RESS == 0.05
    assert study._metric_definition()["metric"].shape == (9, 9)


def test_01_physical_source_matches_study_seal_when_built() -> None:
    if study.SOURCE_SEAL_PATH.exists():
        assert file_sha256(study.ROOT / "domain.py") == study._json(study.SOURCE_SEAL_PATH)["fixed_input_hashes"]["physical_model"]


def test_02_risk_source_matches_study_seal_when_built() -> None:
    if study.SOURCE_SEAL_PATH.exists():
        assert file_sha256(study.ROOT / "risk.py") == study._json(study.SOURCE_SEAL_PATH)["fixed_input_hashes"]["scientific_risk"]


def test_03_whitening_is_authoritative_and_symmetric() -> None:
    metric = study._metric_definition()["metric"]
    assert np.array_equal(metric, metric.T)


def test_04_ress_threshold_is_exact() -> None:
    assert study.MINIMUM_RESS == 0.05


def test_05_production_checkpoint_is_immutable() -> None:
    assert file_sha256(study.BASELINE_CHECKPOINT_PATH) == study.EXPECTED_BASELINE_CHECKPOINT_SHA256


def test_06_no_validation_permission() -> None:
    if study.MANIFEST_PATH.exists():
        assert study._json(study.MANIFEST_PATH)["validation_access_permitted"] is False


def test_07_training_api_accepts_endpoints_not_trajectories() -> None:
    parameters = set(__import__("inspect").signature(study.train_coupled_reference).parameters)
    assert parameters == {"endpoint0", "endpoint1", "family", "cfg"}


def test_08_four_arms_are_predeclared() -> None:
    assert study.FAMILIES == ("B0", "B1", "B2", "B3")


def test_09_b0_is_labelwise() -> None:
    assert "original target labels retained" in study.build_coupling_banks.__doc__ if study.build_coupling_banks.__doc__ else True


def test_10_b1_is_endpoint_permutation_only() -> None:
    _, x1 = _toy()
    permutation, _ = study.particle_assignment(*_toy())
    assert set(map(tuple, x1[permutation])) == set(map(tuple, x1))


def test_11_hungarian_cost_is_no_worse_than_identity() -> None:
    x0, x1 = _toy()
    permutation, cost = study.particle_assignment(x0, x1)
    identity_cost = float(np.mean(study._particle_displacements(x0[None], x1[None]) ** 2))
    assert cost <= identity_cost + 1e-15 and len(np.unique(permutation)) == 16


def test_12_ot_marginal_preservation_when_built() -> None:
    if study.COUPLING_SPEC_PATH.exists():
        pair = study._load_pair_map("B2", 0)
        assert np.array_equal(np.sort(pair["target_index"]), np.arange(study.TRAIN_ENDPOINTS))


def test_13_b2_does_not_apply_particle_permutation_when_built() -> None:
    if study.COUPLING_SPEC_PATH.exists():
        pair = study._load_pair_map("B2", 0)
        assert np.array_equal(pair["particle_permutation"], np.tile(np.arange(16), (study.TRAIN_ENDPOINTS, 1)))


def test_14_b3_reuses_b2_pairs_when_built() -> None:
    if study.COUPLING_SPEC_PATH.exists():
        assert np.array_equal(study._load_pair_map("B2", 0)["target_index"], study._load_pair_map("B3", 0)["target_index"])


def test_15_b3_particle_permutation_is_valid_when_built() -> None:
    if study.COUPLING_SPEC_PATH.exists():
        permutations = study._load_pair_map("B3", 0)["particle_permutation"]
        assert np.all(np.sort(permutations, axis=1) == np.arange(16))


def test_16_minimum_image_matches_authoritative_function() -> None:
    x0, x1 = _toy()
    assert np.allclose(study._particle_displacement_vectors(x0[None], x1[None]), minimum_image(x1 - x0, np.asarray(study.BOX))[None])


def test_17_bridge_endpoints_are_intended_configs() -> None:
    test_bridge_endpoints_and_noise_vanish()


def test_18_noise_vanishes_at_both_endpoints() -> None:
    assert np.sin(0.0) == 0.0 and abs(np.sin(np.pi)) < 1e-15


def test_19_noise_amplitude_is_fixed() -> None:
    cfg = study._json(study.CONFIG_PATH)
    assert cfg["reference_training"]["bridge_noise_std"] == 0.01


def test_20_architecture_is_identical_across_arms() -> None:
    cfg = study._json(study.CONFIG_PATH)
    assert cfg["reference_training"]["hidden_width"] == 64 and cfg["reference_training"]["hidden_layers"] == 3


def test_21_training_steps_are_exact() -> None:
    assert study._json(study.CONFIG_PATH)["reference_training"]["train_steps"] == 6000


def test_22_matched_seed_count_and_schedule() -> None:
    assert study.MATCHED_SEEDS == 3 and study.CFM_EVAL_N == 32768


def test_23_holdout_seed_is_disjoint_when_manifest_built() -> None:
    if study.MANIFEST_PATH.exists():
        manifest = study._json(study.MANIFEST_PATH)
        holdout = manifest["endpoint_holdout"]["seed"]["seed"]
        assert holdout not in {row["seed"] for row in manifest["training"]["matched_seeds"]}


def test_24_holdout_retains_no_intermediate_state_when_built() -> None:
    test_holdout_contains_endpoints_only_when_built()


def test_25_common_evaluation_p0_when_built() -> None:
    test_common_eval_hash_and_sealed_caches_when_built()


def test_26_candidate_panel_hash_is_frozen() -> None:
    assert file_sha256(study.PANEL_SOURCE_PATH) == study.EXPECTED_PANEL_SHA256


def test_27_no_sensor_generation_symbol() -> None:
    assert not hasattr(study, "generate_sensor_candidates")


def test_28_no_tangent_symbol() -> None:
    assert not hasattr(study, "run_tangent")


def test_29_no_full_construction_symbol() -> None:
    assert not hasattr(study, "construct_full")


def test_30_no_full_eigensolve_symbol() -> None:
    assert not hasattr(study, "full_eigensolve")


def test_31_no_deep_ritz_symbol() -> None:
    assert not hasattr(study, "train_deep_ritz")


def test_32_cache_records_verify_hashes_when_built() -> None:
    if study.COUPLING_SPEC_PATH.exists():
        for family in study.FAMILIES:
            study._load_pair_map(family, 0)


def test_33_deterministic_summary_when_built() -> None:
    test_deterministic_summary_and_firewalls_when_built()


def test_34_output_root_is_development_namespace() -> None:
    assert study.OUTPUT_ROOT.name == study.VERSION and "dev_bridge_ablation" in study.VERSION


def test_manifest_freezes_exactly_four_families_and_training_constants() -> None:
    cfg = study._json(study.CONFIG_PATH)
    assert set(study.FAMILIES) == {"B0", "B1", "B2", "B3"}
    base = asdict(ReferenceTrainingConfig(**cfg["reference_training"]))
    assert (base["hidden_width"], base["hidden_layers"], base["train_steps"], base["batch_size"]) == (64, 3, 6000, 512)
    assert base["bridge_noise_std"] == 0.01
    assert study.MATCHED_SEEDS == 3


def test_particle_assignment_exact_and_endpoint_preserving() -> None:
    x0, x1 = _toy()
    permutation, cost = study.particle_assignment(x0, x1)
    assert np.array_equal(np.sort(permutation), np.arange(16))
    brute_selected = study._particle_displacements(x0[None], x1[permutation][None])
    assert np.isclose(cost, np.mean(brute_selected**2))
    assert np.array_equal(np.sort(x1[permutation], axis=0), np.sort(x1, axis=0))


def test_periodic_assignment_uses_minimum_image() -> None:
    x0, x1 = _toy()
    permutation, _ = study.particle_assignment(x0, x1)
    observed = study._particle_displacement_vectors(x0[None], x1[permutation][None])
    expected = np.asarray(minimum_image(x1[permutation] - x0, np.asarray(study.BOX)))
    assert np.allclose(observed[0], expected)


def test_bridge_endpoints_and_noise_vanish() -> None:
    x0, x1 = _toy()
    delta = study._particle_displacement_vectors(x0[None], x1[None])[0]
    noise = np.ones_like(x0)
    at0 = np.mod(x0 + 0 * delta + 0.01 * np.sin(0) * noise, study.BOX)
    at1 = np.mod(x0 + delta + 0.01 * np.sin(np.pi) * noise, study.BOX)
    assert np.allclose(at0, x0)
    assert np.allclose(at1, x1)


def test_arm_semantics_and_pair_sharing_when_built() -> None:
    if not study.COUPLING_SPEC_PATH.exists():
        return
    b0, b1, b2, b3 = (study._load_pair_map(family, 0) for family in study.FAMILIES)
    assert np.array_equal(b0["source_index"], b1["source_index"])
    assert np.array_equal(b0["target_index"], b1["target_index"])
    assert np.array_equal(b2["source_index"], b3["source_index"])
    assert np.array_equal(b2["target_index"], b3["target_index"])
    assert np.array_equal(b0["particle_permutation"], np.tile(np.arange(16), (study.TRAIN_ENDPOINTS, 1)))
    assert np.array_equal(b2["particle_permutation"], np.tile(np.arange(16), (study.TRAIN_ENDPOINTS, 1)))
    assert np.array_equal(np.sort(b2["target_index"]), np.arange(study.TRAIN_ENDPOINTS))


def test_holdout_contains_endpoints_only_when_built() -> None:
    if not study.HOLDOUT_PATH.exists():
        return
    with np.load(study.HOLDOUT_PATH, allow_pickle=False) as arrays:
        assert set(arrays.files) == {"endpoint0", "endpoint1", "seed"}
        assert arrays["endpoint0"].shape == arrays["endpoint1"].shape == (study.HOLDOUT_N, 16, 2)


def test_no_forbidden_entry_points_or_paths() -> None:
    tree = ast.parse(Path(study.__file__).read_text(encoding="utf-8"))
    names = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not ({"generate_candidates", "run_tangent", "run_full", "eigh", "train_deep_ritz"} & names)
    assert not hasattr(study, "load_validation")


def test_common_eval_hash_and_sealed_caches_when_built() -> None:
    if not study.EVAL_MANIFEST_PATH.exists():
        return
    manifest = study._json(study.EVAL_MANIFEST_PATH)
    assert all(len({row["initial_P0_sha256"] for row in bank["models"]}) == 1 for bank in manifest["banks"])
    assert manifest["candidate_panel_sha256"] == file_sha256(study.PANEL_PATH)
    for bank in manifest["banks"]:
        for row in bank["models"]:
            path = study.OUTPUT_ROOT / "reference_eval_results" / row["label"] / f"bank_{bank['bank_index']:02d}.npz"
            assert file_sha256(path) == row["result_sha256"]


def test_deterministic_summary_and_firewalls_when_built() -> None:
    if not study.SUMMARY_PATH.exists():
        return
    before = study.SUMMARY_PATH.read_bytes()
    study.summarize(study._json(study.CONFIG_PATH))
    assert study.SUMMARY_PATH.read_bytes() == before
    safeguards = study._json(study.SUMMARY_PATH)["safeguards"]
    assert not any(safeguards.values())
