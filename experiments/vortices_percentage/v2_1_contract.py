"""V2.1 overlay loader and fail-closed deterministic search helpers."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np

from mfsi.design import random_point_sensor_starts
from selection_contract import validate_selection_config


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE_CONFIG = HERE / "VORTICES_V2_SELECTION_CONFIG.json"
CONFIG = HERE / "VORTICES_V2_1_SELECTION_CONFIG.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_overlay(path: Path = CONFIG) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_resolved_config(path: Path = CONFIG, *, require_frozen: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    overlay = load_overlay(path)
    expected_status = "FROZEN_PROSPECTIVE_BEFORE_V2_1_SELECTION_BANK"
    if require_frozen and overlay.get("status") != expected_status:
        raise ValueError(f"V2.1 config is not frozen: {overlay.get('status')!r}")
    base_record = overlay["base_v2_config"]
    if sha256_file(BASE_CONFIG) != base_record["sha256"]:
        raise ValueError("base V2 selection config hash mismatch")
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    validate_selection_config(base)
    resolved = deepcopy(base)

    banks = overlay["observation_bank_overrides"]
    resolved["observation_banks"].update(
        {
            "generation_seed": int(banks["generation_seed"]),
            "selection_namespace": int(banks["selection_namespace"]),
            "validation_namespace": int(banks["validation_namespace"]),
            "selection_master_trials": int(banks["selection_master_trials"]),
            "validation_trials": int(banks["validation_trials"]),
        }
    )
    resolved["optimization"]["optimizer_root_seed"] = int(
        overlay["optimization_overrides"]["optimizer_root_seed"]
    )
    resolved["optimization"]["full"].update(
        {
            "exact_risk_audit_candidates": "all_unique_candidates",
            "promoted_candidates": int(overlay["optimization_overrides"]["full"]["promoted_candidates"]),
            "selection_order": "exact_L_and_R_then_feasible_proxy_rank",
        }
    )
    resolved["validation"]["bootstrap_seed"] = int(
        overlay["validation_overrides"]["bootstrap_seed"]
    )

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
    values = list(map(int, values))
    if len(values) != len(set(values)) or any(value < 10 or value > 99 for value in values):
        raise ValueError("new V2.1 randomness must be unique two-digit integers")
    if resolved["reference_replicates"]["training_seeds"] != [310000101, 310000102, 310000103]:
        raise ValueError("historical reference seeds changed")
    if resolved["reference_replicates"]["rollout"]["seeds"] != [310003102, 310003103, 310003104]:
        raise ValueError("historical rollout seeds changed")
    if resolved["scientific_evaluator"]["exact_grid"] != [256, 128]:
        raise ValueError("exact scientific grid changed")
    if resolved["scientific_evaluator"]["scientific_time_indices"] != list(range(21)):
        raise ValueError("scientific time nodes changed")
    if resolved["scientific_evaluator"]["precision"] != "float64":
        raise ValueError("scientific precision changed")
    if float(resolved["scientific_evaluator"]["density_floor"]) != 0.0:
        raise ValueError("scientific density floor changed")
    return resolved, overlay


def generated_starts(config: dict[str, Any]) -> np.ndarray:
    opt = config["optimization"]
    risk = config["risk_and_geometry"]
    box = risk["center_box"]
    return np.asarray(
        random_point_sensor_starts(
            jax.random.PRNGKey(int(opt["optimizer_root_seed"])),
            int(opt["generated_start_count"]),
            n_sensors=4,
            x_bounds=tuple(map(float, box[0])),
            y_bounds=tuple(map(float, box[1])),
            min_sep=float(risk["minimum_pairwise_separation"]),
            oversample=int(opt["start_oversampling_factor"]),
        ),
        dtype=np.float64,
    )


def canonical_resolved_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()
