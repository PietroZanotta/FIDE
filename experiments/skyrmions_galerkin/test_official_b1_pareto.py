from __future__ import annotations

import ast
from pathlib import Path

from . import official_b1_pareto as study
from .pareto_v3_common import file_sha256


def test_confirmation_authorizes_launch() -> None:
    summary=study.read_json(study.CONFIRMATION_ROOT/"summary.json")
    assert summary["classification"]=="PRODUCTION_LAUNCH_READY"
    assert all(row["passed"] for row in summary["conditions"].values())


def test_reference_and_dictionary_are_frozen() -> None:
    assert file_sha256(study.CHECKPOINT)==study.CHECKPOINT_SHA256
    assert file_sha256(study.DICTIONARY_PATH)==study.DICTIONARY_SHA256


def test_protocol_semantics_when_frozen() -> None:
    if study.PROTOCOL_PATH.exists():
        protocol=study.read_json(study.PROTOCOL_PATH)
        assert protocol["reference"]["B1_particle_matching"] is True
        assert protocol["reference"]["configuration_OT"] is False
        assert protocol["reference"]["retrained"] is False
        assert protocol["law"]["development_R_Law_used_as_anchor"] is False
        assert protocol["constants"]["minimum_rESS"]==0.05
        assert protocol["constants"]["K"]==280
        assert protocol["constants"]["galerkin_backend"]=="jax"
        assert protocol["allowance_failures_independent"] is True


def test_fresh_data_and_seed_contract() -> None:
    assert study.DESIGN_N==6000 and study.CANDIDATE_COUNT==4096
    assert study.BANK_SIZES=={"law_search":32768,"risk_anchor":32768,"screen":8192,"search_train":32768,"periodic_audit":16384,"authoritative_train":65536,"authoritative_audit":65536}
    assert study.VALIDATION_SIZES=={"truth":5000,"reference_fit":16384,"reference_audit":16384}


def test_validation_firewall() -> None:
    source=ast.parse(Path(study.__file__).read_text(encoding="utf-8"))
    assert study.generate_validation.__code__.co_names.count("exists")
    if not (study.OUTPUT_ROOT/"selection"/"selection_hash.json").exists():
        assert not (study.OUTPUT_ROOT/"fresh_validation"/"truth.npz").exists()
    calls={node.func.attr for node in ast.walk(source) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)}
    assert "train_b1_reference" not in calls and "train_endpoint_reference" not in calls


def test_exact_risk_and_no_deep_ritz() -> None:
    assert study.selection_ceiling(2.0,3.0)==2.06
    source=Path(study.__file__).read_text(encoding="utf-8")
    assert "deep_ritz" not in source.lower() or '"deep_ritz_used":False' in source.replace(" ","")


def test_cache_hashes_when_built() -> None:
    if (study.OUTPUT_ROOT/"banks"/"manifest.json").exists():
        for label in study.BANK_SIZES:
            record=study.read_json(study._bank_path(label).with_suffix(".json"))
            assert file_sha256(study._bank_path(label))==record["sha256"]


def test_validation_does_not_modify_selection_when_built() -> None:
    if (study.OUTPUT_ROOT/"fresh_validation"/"results.json").exists():
        result=study.read_json(study.OUTPUT_ROOT/"fresh_validation"/"results.json")
        assert result["selection_geometry_unchanged"] is True and result["optimization_run"] is False
