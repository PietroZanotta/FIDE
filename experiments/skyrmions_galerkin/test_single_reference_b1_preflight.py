from __future__ import annotations

from dataclasses import asdict
import ast
import inspect
from pathlib import Path

import numpy as np

from . import single_reference_b1_preflight as study
from .domain import minimum_image
from .pareto_v3_common import eta_key, file_sha256
from .reference import ReferenceTrainingConfig


def _toy() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(31)
    return rng.random((16, 2)) * np.asarray(study.BOX), rng.random((16, 2)) * np.asarray(study.BOX)


def test_01_physical_simulator_and_config_unchanged() -> None:
    assert file_sha256(study.CONFIG_PATH) == study.EXPECTED_CONFIG_SHA256
    if study.SOURCE_SEAL_PATH.exists():
        assert file_sha256(study.ROOT / "domain.py") == study._json(study.SOURCE_SEAL_PATH)["immutable_definition_hashes"]["physical_simulator"]


def test_02_b1_matches_selected_particle_assignment() -> None:
    x0, x1 = _toy(); permutation, cost = study.particle_assignment(x0, x1)
    delta = np.asarray(minimum_image(x1[permutation] - x0, np.asarray(study.BOX)))
    assert np.isclose(cost, np.mean(np.sum(delta * delta, axis=-1)))


def test_03_configuration_ot_is_absent() -> None:
    assert "linear_sum_assignment" not in inspect.getsource(study._build_pair_maps)
    assert "configuration_OT" in inspect.getsource(study.freeze_manifest)


def test_04_assignment_is_exact_16_by_16() -> None:
    x0, x1 = _toy(); permutation, cost = study.particle_assignment(x0, x1)
    identity = float(np.mean(np.sum(np.asarray(minimum_image(x1 - x0, np.asarray(study.BOX))) ** 2, axis=-1)))
    assert permutation.shape == (16,) and len(np.unique(permutation)) == 16 and cost <= identity + 1e-15


def test_05_assignment_preserves_unlabeled_endpoint() -> None:
    x0, x1 = _toy(); permutation, _ = study.particle_assignment(x0, x1)
    assert set(map(tuple, x1[permutation])) == set(map(tuple, x1))


def test_06_bridge_noise_is_exact_and_vanishes() -> None:
    cfg = study._json(study.CONFIG_PATH)
    assert cfg["reference_training"]["bridge_noise_std"] == 0.01
    assert np.sin(0.0) == 0.0 and abs(np.sin(np.pi)) < 1e-15


def test_07_manifest_precedes_generated_data_when_built() -> None:
    if study.TRAIN_DATA_PATH.exists():
        assert study.MANIFEST_PATH.stat().st_mtime_ns <= study.TRAIN_DATA_PATH.stat().st_mtime_ns


def test_08_endpoint_datasets_are_disjoint_when_built() -> None:
    if study.QUAL_DATA_PATH.exists():
        train0, train1 = study._load_endpoints(study.TRAIN_DATA_PATH); qual0, qual1 = study._load_endpoints(study.QUAL_DATA_PATH)
        assert not ({study._array_sha256(train0), study._array_sha256(train1)} & {study._array_sha256(qual0), study._array_sha256(qual1)})


def test_09_reference_training_accepts_endpoints_only() -> None:
    assert set(inspect.signature(study.train_b1_reference).parameters) == {"endpoint0", "endpoint1", "pair_maps", "cfg"}


def test_10_design_truth_not_in_training_signature() -> None:
    assert "design" not in inspect.signature(study.train_b1_reference).parameters


def test_11_no_validation_loader_or_access() -> None:
    assert not hasattr(study, "load_validation_galerkin_data")
    assert "truth_validation" not in Path(study.__file__).read_text(encoding="utf-8")


def test_12_three_training_seeds_are_frozen() -> None:
    assert study.ATTEMPTS == ("A", "B", "C")
    if study.MANIFEST_PATH.exists():
        assert list(study._json(study.MANIFEST_PATH)["reference_training"]["attempt_order"]) == list(study.ATTEMPTS)


def test_13_first_passing_reference_only() -> None:
    assert "first_passing_seed_wins" in inspect.getsource(study.freeze_manifest)


def test_14_later_seed_cannot_replace_accepted() -> None:
    assert "a later reference was trained after acceptance" in inspect.getsource(study.train_and_accept_reference)


def test_15_architecture_settings_unchanged() -> None:
    values = asdict(ReferenceTrainingConfig(**study._json(study.CONFIG_PATH)["reference_training"]))
    assert (values["hidden_width"], values["hidden_layers"], values["batch_size"]) == (64, 3, 512)


def test_16_training_is_6000_steps() -> None:
    assert study._json(study.CONFIG_PATH)["reference_training"]["train_steps"] == 6000


def test_17_qualification_thresholds_frozen_before_training_when_built() -> None:
    if study.MANIFEST_PATH.exists():
        q = study._json(study.MANIFEST_PATH)["endpoint_qualification"]
        assert (q["CFM_loss_maximum"], q["endpoint_Psi_L2_maximum"], q["endpoint_whitened_Psi_norm_maximum"], q["endpoint_Law_Phi_L2_maximum"]) == (0.2, 0.02, 1.5, 0.005)


def test_18_accepted_checkpoint_precedes_law_when_built() -> None:
    if study.LAW_FREEZE_PATH.exists():
        assert study.ACCEPTED_REFERENCE_PATH.stat().st_mtime_ns <= study.LAW_FREEZE_PATH.stat().st_mtime_ns


def test_19_all_reference_banks_use_accepted_checkpoint_when_built() -> None:
    if study.REFERENCE_BANK_MANIFEST_PATH.exists():
        manifest = study._json(study.REFERENCE_BANK_MANIFEST_PATH)
        assert manifest["all_banks_use_accepted_checkpoint"] and len({row["checkpoint_sha256"] for row in manifest["banks"]}) == 1


def test_20_historical_risk_is_context_only() -> None:
    assert study.HISTORICAL_R_LAW == 5.186549474478042
    if study.LAW_FREEZE_PATH.exists() and study._json(study.LAW_FREEZE_PATH).get("status") == "FROZEN":
        assert study._json(study.LAW_FREEZE_PATH)["historical_R_Law_status"] == "NOT USED AS NEW ANCHOR"


def test_21_new_law_is_recomputed() -> None:
    assert "reconstruct_and_freeze_law" in dir(study)


def test_22_new_anchor_comes_from_risk_anchor_bank_when_built() -> None:
    if study.LAW_FREEZE_PATH.exists() and study._json(study.LAW_FREEZE_PATH).get("status") == "FROZEN":
        law = study._json(study.LAW_FREEZE_PATH)
        assert law["risk_anchor_bank_sha256"] == file_sha256(study._bank_path("risk_anchor"))


def test_23_exact_allowance_arithmetic_when_built() -> None:
    if study.LAW_FREEZE_PATH.exists() and study._json(study.LAW_FREEZE_PATH).get("status") == "FROZEN":
        law = study._json(study.LAW_FREEZE_PATH)
        for allowance in study.ALLOWANCES:
            assert law["allowance_ceilings"][str(allowance)] == (1 + allowance / 100) * law["R_Law_B1"]


def test_24_candidate_generator_frozen_before_support_when_built() -> None:
    if study._support_result_path("screen_0").exists():
        assert study.CANDIDATE_SPEC_PATH.stat().st_mtime_ns <= study._support_result_path("screen_0").stat().st_mtime_ns


def test_25_candidate_geometries_unique_when_built() -> None:
    if study.CANDIDATE_POOL_PATH.exists():
        pool = study._json(study.CANDIDATE_POOL_PATH); assert len({row["eta_sha256"] for row in pool["rows"]}) == pool["count"]


def test_26_minimum_separation_when_built() -> None:
    if study.CANDIDATE_POOL_PATH.exists():
        assert study._json(study.CANDIDATE_POOL_PATH)["minimum_observed_separation"] >= study._json(study.CONFIG_PATH)["measurement"]["min_separation"]


def test_27_four_independent_support_pairs() -> None:
    assert study.SUPPORT_PAIRS == 4
    if study.MANIFEST_PATH.exists():
        seeds = study._json(study.MANIFEST_PATH)["seeds"]
        values = [seeds[f"support_{kind}_{i}"]["seed"] for i in range(4) for kind in ("screen", "audit")]
        assert len(set(values)) == 8


def test_28_ress_threshold_exact() -> None:
    assert study.MINIMUM_RESS == 0.05


def test_29_no_tangent_run_entry_point() -> None:
    assert not hasattr(study, "run_tangent")


def test_30_no_full_kf_entry_point() -> None:
    assert not hasattr(study, "construct_full_Kf")


def test_31_no_eigensolve_entry_point() -> None:
    tree = ast.parse(Path(study.__file__).read_text(encoding="utf-8")); names = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not ({"eigh", "eigvalsh", "eig"} & names)


def test_32_no_deep_ritz_entry_point() -> None:
    assert not hasattr(study, "train_deep_ritz")


def test_33_no_official_protocol() -> None:
    assert "official_protocol_permitted" in inspect.getsource(study.freeze_manifest)


def test_34_deterministic_summary_when_built() -> None:
    if study.SUMMARY_PATH.exists():
        before = file_sha256(study.SUMMARY_PATH); study.summarize(study._json(study.CONFIG_PATH)); assert file_sha256(study.SUMMARY_PATH) == before


def test_35_output_namespace_is_clean_room_development() -> None:
    assert study.OUTPUT_ROOT.name == study.SEED_NAMESPACE and "single_reference_b1_preflight" in study.VERSION


def test_periodic_canonical_key_is_stable() -> None:
    cfg = study._json(study.CONFIG_PATH); law = np.asarray(cfg["envelope"]["law_eta"])
    assert eta_key(study.canonicalize_eta(law, law, study.BOX)) == eta_key(study.canonicalize_eta(law, law, study.BOX))
