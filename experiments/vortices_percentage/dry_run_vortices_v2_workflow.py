#!/usr/bin/env python3
"""Print the frozen prospective Vortices V2 workflow without executing it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from selection_contract import (
    CONFIG_PATH,
    load_selection_config,
    observation_bank_identity,
    sha256_file,
    validate_selection_config,
)


HERE = Path(__file__).resolve().parent


def workflow_manifest(config_path: Path = CONFIG_PATH) -> dict:
    config = load_selection_config(config_path)
    validate_selection_config(config)
    destinations = config["artifact_destinations"]
    root = HERE.parent.parent / destinations["root"]
    seeds = list(config["reference_replicates"]["training_seeds"])
    references = [
        {
            "seed": seed,
            "rollout_seed": seed + 3001,
            "input_endpoint_sha256": config["reference_replicates"]["endpoint_dataset"]["sha256"],
            "output": str(root / destinations["references"].replace("<seed>", str(seed))),
            "future_checkpoint_sha256": "PENDING_UNTIL_TRAINED",
            "future_rollout_sha256": "PENDING_UNTIL_QUALIFIED",
        }
        for seed in seeds
    ]
    selection_bank = observation_bank_identity(config, "selection")
    validation_bank = observation_bank_identity(config, "validation")
    stages = [
        {
            "id": "A-C",
            "name": "train_three_references",
            "depends_on": ["frozen_protocol", "common_endpoint_dataset"],
            "inputs": {"training_seeds": seeds, "training": config["reference_replicates"]["training"]},
            "outputs": references,
        },
        {
            "id": "D",
            "name": "qualify_all_three_references",
            "depends_on": ["A-C"],
            "inputs": {"qualification": config["reference_replicates"]["qualification"]},
            "outputs": [str(root / destinations["reference_qualification"].replace("<seed>", str(seed))) for seed in seeds],
            "failure": "stop_without_replacement_seed",
        },
        {
            "id": "E",
            "name": "freeze_common_bandwidth",
            "depends_on": ["D"],
            "inputs": {"rule": config["common_bandwidth"], "qualified_reference_count": 3},
            "outputs": [str(HERE.parent.parent / config["common_bandwidth"]["receipt_path"])],
        },
        {
            "id": "F",
            "name": "generate_one_shared_selection_bank",
            "depends_on": ["E"],
            "inputs": selection_bank,
            "outputs": [str(root / destinations["selection_bank"])],
        },
        {
            "id": "G",
            "name": "population_selection",
            "depends_on": ["D", "E"],
            "inputs": config["optimization"]["population"],
            "outputs": [str(root / destinations["population"])],
        },
        {
            "id": "H",
            "name": "law_selection_and_new_risk_anchor",
            "depends_on": ["F", "G"],
            "inputs": {"optimizer": config["optimization"]["law"], "selection_prefix": [0, 64]},
            "outputs": [str(root / destinations["law"])],
        },
        {
            "id": "I",
            "name": "nested_tangent_and_full_selection",
            "depends_on": ["H"],
            "inputs": {
                "allowances_percent": config["risk_and_geometry"]["risk_allowance_percentages"],
                "tangent": config["optimization"]["tangent"],
                "full": config["optimization"]["full"],
                "proxy": config["optimization"]["full_search_proxy"],
            },
            "outputs": [str(root / destinations["tangent_full"])],
        },
        {
            "id": "J",
            "name": "freeze_all_sensor_winners",
            "depends_on": ["I"],
            "inputs": {"tie_tolerance": config["optimization"]["candidate_handling"]["tie_tolerance"]},
            "outputs": [str(root / destinations["frozen_winners"])],
        },
        {
            "id": "K",
            "name": "generate_one_shared_validation_bank",
            "depends_on": ["J"],
            "inputs": validation_bank,
            "outputs": [str(root / destinations["validation_bank"])],
        },
        {
            "id": "L",
            "name": "evaluate_all_methods_all_references",
            "depends_on": ["K"],
            "inputs": {"references": seeds, "shared_trials": 1024, "scientific_evaluator": config["scientific_evaluator"]},
            "outputs": [str(root / destinations["validation_results"])],
        },
        {
            "id": "M",
            "name": "frozen_simultaneous_bootstrap",
            "depends_on": ["L"],
            "inputs": config["validation"],
            "outputs": [str(root / destinations["final_inference"])],
        },
        {
            "id": "N",
            "name": "generate_final_report",
            "depends_on": ["M"],
            "inputs": {"statistical_gates": config["validation"]["pass_gates"]},
            "outputs": [str(root / destinations["final_report"])],
        },
    ]
    return {
        "schema_version": 1,
        "mode": "DRY_RUN_NO_SCIENTIFIC_EXECUTION",
        "selection_config": str(config_path.resolve()),
        "selection_config_sha256": sha256_file(config_path),
        "numerical_config_sha256": sha256_file(HERE / "config.json"),
        "reference_seeds": seeds,
        "selection_bank": selection_bank,
        "validation_bank": validation_bank,
        "stages": stages,
        "files_written": [],
        "scientific_actions_evaluated": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    if not args.dry_run:
        raise SystemExit("refusing execution: this freeze runner supports --dry-run only")
    print(json.dumps(workflow_manifest(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

