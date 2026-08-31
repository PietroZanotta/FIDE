from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from core import frozen_reference_scott_bandwidth
from dry_run_vortices_v2_workflow import workflow_manifest
from freeze_common_bandwidth import freeze_common_bandwidth
from preflight_vortices_v2_freeze import DEFAULT_MANIFEST, run_preflight
from selection_contract import (
    CONFIG_PATH,
    EVALUATOR_IDENTITY,
    canonical_json_sha256,
    generated_starts,
    geometry_is_feasible,
    load_selection_config,
    observation_bank_identity,
    sha256_file,
    validate_selection_config,
)
from v2_inference import effects_for_common_indices


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent


def test_fresh_reference_seeds_and_rollout_seed_policy_are_frozen():
    config = load_selection_config()
    seeds = config["reference_replicates"]["training_seeds"]
    assert seeds == [310000101, 310000102, 310000103]
    assert not set(seeds) & {20260815, 20260816, 20260817}
    assert config["reference_replicates"]["rollout"]["seeds"] == [
        seed + 3001 for seed in seeds
    ]


def test_namespaces_are_distinct_and_observation_bank_is_shared_across_references():
    config = load_selection_config()
    selection = observation_bank_identity(config, "selection")
    validation = observation_bank_identity(config, "validation")
    assert selection["namespace"] == 410000101
    assert validation["namespace"] == 410000102
    assert selection["namespace"] != validation["namespace"]
    assert selection["trials"] == 128
    assert validation["trials"] == 1024
    assert selection["shared_across_reference_seeds"] == [310000101, 310000102, 310000103]
    assert validation["shared_across_reference_seeds"] == selection["shared_across_reference_seeds"]
    assert config["observation_banks"]["selection_prefixes"] == {
        "tangent_full_prescreen": [0, 32],
        "law_finite_risk": [0, 64],
        "tangent_full_final": [0, 128],
    }


def test_optimizer_config_is_complete_and_mutations_fail_closed():
    config = load_selection_config()
    validate_selection_config(config)
    starts_a = generated_starts(config)
    starts_b = generated_starts(config)
    np.testing.assert_array_equal(starts_a, starts_b)
    assert starts_a.shape == (64, 8)
    for eta in starts_a:
        assert geometry_is_feasible(
            eta,
            box=config["risk_and_geometry"]["center_box"],
            minimum_separation=config["risk_and_geometry"]["minimum_pairwise_separation"],
            tolerance=config["optimization"]["common_adam"]["feasibility_tolerance"],
        )
    changed = copy.deepcopy(config)
    del changed["optimization"]["full"]["promoted_candidates"]
    with pytest.raises(ValueError, match="incomplete"):
        validate_selection_config(changed)
    changed = copy.deepcopy(config)
    changed["scientific_evaluator"]["density_floor"] = 1e-8
    with pytest.raises(ValueError, match="density floor"):
        validate_selection_config(changed)


def test_reflected_v2_evaluator_and_proxy_identity_are_frozen():
    config = load_selection_config()
    evaluator = config["scientific_evaluator"]
    proxy = config["optimization"]["full_search_proxy"]
    assert evaluator["identity"] == EVALUATOR_IDENTITY
    assert evaluator["scalar_raster"] == "direct_particle_cell_integrated_even_reflection_neumann_gaussian"
    assert evaluator["source_raster"] == "identical_reflected_scalar_kernel"
    assert evaluator["density_floor"] == 0.0
    assert evaluator["exact_grid"] == [256, 128]
    assert proxy["grid"] == [64, 32]
    assert proxy["time_indices"] == list(range(21))
    assert proxy["density_floor"] == 0.0
    source = (HERE / "selection_contract.py").read_text(encoding="utf-8")
    assert "rasterize_trajectory_v2" in source
    assert "rasterize_projected_particles_positive_rect" not in source


def test_common_bootstrap_indices_pair_all_references_and_methods():
    law = np.vstack([
        np.linspace(1.0, 2.0, 8),
        np.linspace(2.0, 3.0, 8),
        np.linspace(3.0, 4.0, 8),
    ])
    full = np.stack(
        [np.stack([(0.50 + 0.03 * p) * law[r] for p in range(6)]) for r in range(3)]
    )
    indices = np.asarray([[0, 0, 7, 7], [1, 2, 3, 4]], dtype=np.int64)
    effects = effects_for_common_indices(law, full, indices)
    assert effects.shape == (2, 3, 6)
    expected = 1.0 - np.asarray([0.50 + 0.03 * p for p in range(6)])
    np.testing.assert_allclose(
        effects,
        np.broadcast_to(expected, effects.shape),
        rtol=2e-15,
        atol=2e-15,
    )


def _qualification_receipt(tmp_path: Path, seed: int, scale: float, qualified: bool = True) -> Path:
    config = load_selection_config()
    rng = np.random.default_rng(seed)
    nodes = rng.uniform([0.2, 0.2], [1.8, 0.8], size=(21, 32, 2)) * scale
    weights = np.full((21, 32), 1.0 / 32.0)
    rollout = tmp_path / f"rollout_{seed}.npz"
    np.savez(rollout, nodes=nodes, weights=weights)
    checkpoint = tmp_path / f"checkpoint_{seed}.npz"
    np.savez(checkpoint, parameter=np.asarray([seed], dtype=np.int64))
    bandwidth, by_time = frozen_reference_scott_bandwidth(nodes, weights)
    receipt_path = tmp_path / f"qualification_{seed}.json"
    payload = {
        "qualified": qualified,
        "training_seed": seed,
        "training_config_sha256": f"{seed:064x}"[-64:],
        "training_architecture_sha256": canonical_json_sha256(
            config["reference_replicates"]["training"]
        ),
        "numerical_method_config_sha256": sha256_file(HERE / "config.json"),
        "endpoint_data_sha256": config["reference_replicates"]["endpoint_dataset"]["sha256"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "rollout_bank_path": str(rollout),
        "rollout_bank_sha256": sha256_file(rollout),
        "qualification_receipt_path": str(receipt_path),
        "scott_bandwidth": bandwidth,
        "scott_bandwidth_by_time": by_time.tolist(),
    }
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    return receipt_path


def test_common_bandwidth_receipt_is_exactly_three_and_fail_closed(tmp_path: Path):
    seeds = [310000101, 310000102, 310000103]
    receipts = [
        _qualification_receipt(tmp_path, seed, scale)
        for seed, scale in zip(seeds, (0.9, 1.0, 1.1))
    ]
    output = tmp_path / "common.json"
    result = freeze_common_bandwidth(receipts, output)
    assert result["status"] == "FROZEN_COMMON_REFERENCE_ONLY_BANDWIDTH"
    assert result["training_seeds"] == seeds
    assert result["common_physical_bandwidth"] == pytest.approx(
        np.median([row["scott_bandwidth"] for row in result["per_reference"]])
    )
    with pytest.raises(ValueError, match="exactly three"):
        freeze_common_bandwidth(receipts[:2], tmp_path / "bad_count.json")
    bad = _qualification_receipt(tmp_path, seeds[0], 0.9, qualified=False)
    with pytest.raises(ValueError, match="not qualified"):
        freeze_common_bandwidth([bad, *receipts[1:]], tmp_path / "bad_quality.json")


def test_toy_and_v1_immutable_hashes():
    expected = {
        "experiments/toy_example_percentage/config.json": "e0462f4ce60fc2105905bd6e54fb673e14de0ea2ba120a951d02053b1caf9f1f",
        "experiments/toy_example_percentage/experiment.py": "35bbce25a92790f152664c8304adcd837e117598213c0c80b821254b50e7dcd3",
        "experiments/toy_example_percentage/outputs/pareto/corrected_nested_full_sweep.json": "114df72191c0b519e6e45cf7c574060a47ac6c64201eba7ed7432f2f11fc2c7e",
        "experiments/toy_example_percentage/outputs/pareto/authoritative_run_summary.json": "2e29a178e3850ccb35c067006fb565bcb4f5fb845740ab4b6d4e4859034b80db",
        "experiments/vortices_percentage/config.json": "8f57f167675718b19d7ffc1741a8175adbe22069ff4043634b62df8dcf100ed0",
        "experiments/vortices_percentage/experiment.py": "5bcd5b3c96668cabf6d7a8b2b1944f48f490635763b997172584328551a9a4c4",
        "experiments/vortices_percentage/outputs/pareto/confirmatory_validation_2048/confirmatory_summary.json": "3d1f8e6f95adcdc3e61918d789c5cf83c25c802e3b9ae9d3a64620a91a329901",
    }
    for relative, digest in expected.items():
        assert sha256_file(REPO_ROOT / relative) == digest


def test_dry_run_writes_nothing_and_lists_all_future_stages(tmp_path: Path):
    before = set(tmp_path.iterdir())
    payload = workflow_manifest()
    after = set(tmp_path.iterdir())
    assert before == after
    assert payload["mode"] == "DRY_RUN_NO_SCIENTIFIC_EXECUTION"
    assert payload["files_written"] == []
    assert payload["scientific_actions_evaluated"] == 0
    assert [row["id"] for row in payload["stages"]] == [
        "A-C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"
    ]


def test_final_manifest_preflight_passes_without_scientific_execution():
    result = run_preflight(DEFAULT_MANIFEST, check_git_history=False)
    assert result["status"] == "PASS"
    assert result["scientific_operations_performed"] == []
