from __future__ import annotations

import ast
import inspect
from pathlib import Path

from . import final_b1_support_confirmation as study
from .pareto_v3_common import file_sha256


def test_01_accepted_checkpoint_hash_matches() -> None:
    assert file_sha256(study._checkpoint_path()) == study.EXPECTED_CHECKPOINT_SHA256


def test_02_development_law_and_pool_are_frozen() -> None:
    law, pool, risk = (study._json(path) for path in (study.LAW_FREEZE_PATH, study.CANDIDATE_POOL_PATH, study.CANDIDATE_RISK_PATH))
    assert law["status"] == "FROZEN" and pool["count"] == study.CANDIDATE_COUNT
    assert risk["candidate_pool_sha256"] == file_sha256(study.CANDIDATE_POOL_PATH)


def test_03_manifest_has_exactly_eight_independent_pairs_when_built() -> None:
    if study.BANK_MANIFEST_PATH.exists():
        manifest = study._json(study.BANK_MANIFEST_PATH)
        assert manifest["pair_count"] == 8 and len(manifest["banks"]) == 16
        assert len({row["seed"] for row in manifest["banks"]}) == 16


def test_04_confirmation_seeds_do_not_overlap_prior_work_when_built() -> None:
    if study.BANK_MANIFEST_PATH.exists():
        manifest = study._json(study.BANK_MANIFEST_PATH); prior = study._all_integer_seeds(study._json(study.PREFLIGHT_MANIFEST_PATH))
        assert not ({row["seed"] for row in manifest["banks"]} & prior)


def test_05_no_candidate_generation() -> None:
    assert not hasattr(study, "generate_candidate_pool")


def test_06_no_model_training() -> None:
    assert not hasattr(study, "train_b1_reference") and not hasattr(study, "train_endpoint_reference")


def test_07_no_validation_access() -> None:
    assert not hasattr(study, "load_validation_galerkin_data")
    assert "truth_validation" not in Path(study.__file__).read_text(encoding="utf-8")


def test_08_no_tangent_or_full() -> None:
    assert not hasattr(study, "run_tangent") and not hasattr(study, "run_full")


def test_09_launch_requires_ready_classification_when_built() -> None:
    if study.SUMMARY_PATH.exists():
        summary = study._json(study.SUMMARY_PATH)
        assert summary["production_launched"] is False


def test_10_ready_requires_all_eight_conditions_when_built() -> None:
    if study.SUMMARY_PATH.exists():
        summary = study._json(study.SUMMARY_PATH)
        assert (summary["classification"] == "PRODUCTION_LAUNCH_READY") == all(row["passed"] for row in summary["conditions"].values())


def test_thresholds_are_exact() -> None:
    assert study.MINIMUM_RESS == 0.05
    assert study.READINESS_LAW_MARGIN == 0.060
    assert study.LOW_RISK_P10_MARGIN == 0.055
    assert study.MINIMUM_CANDIDATES == 25 and study.MINIMUM_DIVERSE == 5


def test_only_frozen_candidates_are_evaluated() -> None:
    source = inspect.getsource(study.evaluate_all)
    assert "CANDIDATE_POOL_PATH" in source and "generate_candidate" not in source


def test_no_eigensolve_or_galerkin_call() -> None:
    tree = ast.parse(Path(study.__file__).read_text(encoding="utf-8"))
    calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not ({"eigh", "eigvalsh", "assemble_galerkin", "solve_full"} & calls)


def test_all_cache_records_verify_hashes_when_built() -> None:
    if study.SUMMARY_PATH.exists():
        for pair in range(8):
            for role in ("screen", "audit"):
                label = f"{role}_{pair}"; result = study._result_path(label); record = study._json(result.with_suffix(".json"))
                assert file_sha256(result) == record["result_sha256"]


def test_deterministic_summary_when_built() -> None:
    if study.SUMMARY_PATH.exists():
        before = file_sha256(study.SUMMARY_PATH); study.summarize(study._json(study.CONFIG_PATH)); assert file_sha256(study.SUMMARY_PATH) == before
