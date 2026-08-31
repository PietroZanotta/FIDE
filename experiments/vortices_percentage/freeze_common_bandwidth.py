#!/usr/bin/env python3
"""Freeze the median Scott bandwidth from exactly three qualified V2 references."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from core import frozen_reference_scott_bandwidth
from selection_contract import (
    CONFIG_PATH,
    canonical_json_sha256,
    load_selection_config,
    sha256_file,
    validate_selection_config,
)


def _resolve(receipt: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    repository_candidate = CONFIG_PATH.parent.parent.parent / path
    return repository_candidate if repository_candidate.exists() else receipt.parent / path


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def freeze_common_bandwidth(
    receipt_paths: list[Path],
    output: Path,
    *,
    config_path: Path = CONFIG_PATH,
) -> dict[str, Any]:
    config = load_selection_config(config_path)
    validate_selection_config(config)
    expected_seeds = list(map(int, config["reference_replicates"]["training_seeds"]))
    expected_architecture_hash = canonical_json_sha256(
        config["reference_replicates"]["training"]
    )
    expected_endpoint_hash = str(
        config["reference_replicates"]["endpoint_dataset"]["sha256"]
    )
    expected_numerical_hash = sha256_file(config_path.parent / "config.json")
    if len(receipt_paths) != 3:
        raise ValueError("exactly three qualification receipts are required")

    rows = []
    seen_seeds: set[int] = set()
    common_architecture_hash: str | None = None
    common_endpoint_hash: str | None = None
    common_numerical_hash: str | None = None
    for receipt_path in map(Path, receipt_paths):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        required = set(config["reference_replicates"]["qualification"]["receipt_required_fields"])
        missing = sorted(required - set(receipt))
        if missing:
            raise ValueError(f"qualification receipt missing {missing}: {receipt_path}")
        if receipt.get("qualified") is not True:
            raise ValueError(f"reference is not qualified: {receipt_path}")
        seed = int(receipt["training_seed"])
        if seed not in expected_seeds or seed in seen_seeds:
            raise ValueError(f"unexpected or duplicate reference seed {seed}")
        seen_seeds.add(seed)
        architecture_hash = str(receipt["training_architecture_sha256"])
        endpoint_hash = str(receipt["endpoint_data_sha256"])
        numerical_hash = str(receipt["numerical_method_config_sha256"])
        if architecture_hash != expected_architecture_hash:
            raise ValueError("reference training architecture hash is not frozen")
        if endpoint_hash != expected_endpoint_hash:
            raise ValueError("reference endpoint hash is not frozen")
        if numerical_hash != expected_numerical_hash:
            raise ValueError("reference numerical-method hash is not frozen")
        if str(receipt["qualification_receipt_path"]) not in {
            str(receipt_path), str(receipt_path.resolve())
        }:
            raise ValueError("qualification receipt path does not identify itself")
        checkpoint_path = _resolve(receipt_path, str(receipt["checkpoint_path"]))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        if sha256_file(checkpoint_path) != str(receipt["checkpoint_sha256"]):
            raise ValueError(f"checkpoint hash mismatch: {checkpoint_path}")
        common_architecture_hash = common_architecture_hash or architecture_hash
        common_endpoint_hash = common_endpoint_hash or endpoint_hash
        common_numerical_hash = common_numerical_hash or numerical_hash
        if architecture_hash != common_architecture_hash:
            raise ValueError("reference training architectures differ")
        if endpoint_hash != common_endpoint_hash:
            raise ValueError("reference endpoint datasets differ")
        if numerical_hash != common_numerical_hash:
            raise ValueError("reference numerical-method configurations differ")

        rollout_path = _resolve(receipt_path, str(receipt["rollout_bank_path"]))
        if not rollout_path.is_file():
            raise FileNotFoundError(rollout_path)
        if sha256_file(rollout_path) != str(receipt["rollout_bank_sha256"]):
            raise ValueError(f"rollout hash mismatch: {rollout_path}")
        with np.load(rollout_path, allow_pickle=False) as bank:
            nodes = np.asarray(bank["nodes"], dtype=np.float64)
            weights = np.asarray(bank["weights"], dtype=np.float64)
        bandwidth, by_time = frozen_reference_scott_bandwidth(nodes, weights)
        if not np.isfinite(bandwidth) or bandwidth <= 0.0:
            raise ValueError(f"invalid Scott bandwidth for seed {seed}")
        if not np.isclose(
            bandwidth, float(receipt["scott_bandwidth"]), rtol=0.0, atol=5e-15
        ):
            raise ValueError(f"declared Scott bandwidth mismatch for seed {seed}")
        np.testing.assert_allclose(
            by_time,
            np.asarray(receipt["scott_bandwidth_by_time"], dtype=np.float64),
            rtol=0.0,
            atol=5e-15,
        )
        rows.append(
            {
                "training_seed": seed,
                "qualification_receipt": str(receipt_path.resolve()),
                "qualification_receipt_sha256": sha256_file(receipt_path),
                "training_config_sha256": str(receipt["training_config_sha256"]),
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "rollout_bank": str(rollout_path.resolve()),
                "rollout_bank_sha256": sha256_file(rollout_path),
                "scott_bandwidth": bandwidth,
                "scott_bandwidth_by_time": by_time.tolist(),
            }
        )

    if sorted(seen_seeds) != sorted(expected_seeds):
        raise ValueError("qualification receipts do not cover the three frozen seeds")
    rows.sort(key=lambda row: expected_seeds.index(row["training_seed"]))
    common = float(np.median([row["scott_bandwidth"] for row in rows]))
    payload = {
        "schema_version": 1,
        "status": "FROZEN_COMMON_REFERENCE_ONLY_BANDWIDTH",
        "selection_config": str(Path(config_path).resolve()),
        "selection_config_sha256": sha256_file(Path(config_path)),
        "training_seeds": expected_seeds,
        "training_architecture_sha256": common_architecture_hash,
        "endpoint_data_sha256": common_endpoint_hash,
        "numerical_method_config_sha256": common_numerical_hash,
        "per_reference": rows,
        "rule": "median_of_exactly_three_qualified_per_reference_median_21_time_weighted_2d_scott_bandwidths",
        "common_physical_bandwidth": common,
        "immutable": True,
        "action_or_risk_inputs_used": False,
    }
    if Path(output).exists():
        existing = json.loads(Path(output).read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("common-bandwidth receipt already exists with different content")
        return existing
    _atomic_json(Path(output), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-receipt", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    payload = freeze_common_bandwidth(
        args.reference_receipt, args.output, config_path=args.config
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
