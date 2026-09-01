from __future__ import annotations

"""Build and freeze the v6 endpoint-only multi-reference ensemble.

Design references are available to selection. Evaluation references are trained
only after both beta arms have been frozen and are never imported by selection.
"""

import argparse
import copy
import json
from pathlib import Path
import shutil
from typing import Any

from build_prospective_data import build
from common import SCRIPT_DIR, config_hash, fingerprint, load_config, write_json_atomic
from mfsi.cache import file_sha256
from train_reference import train_and_rollout


DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "production_v6_common.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "prospective_v6_beta_ablation"


def v6_paths(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir).resolve()
    return {
        "root": root,
        "shared": root / "shared",
        "endpoint": root / "shared" / "endpoint_reference",
        "prospective": root / "shared" / "prospective",
        "shared_results": root / "shared" / "results",
        "design_references": root / "shared" / "references" / "design",
        "evaluation_references": root / "shared" / "references" / "evaluation",
        "arms": root / "arms",
        "results": root / "results",
        "hidden": root / "hidden_validation",
    }


def prepare_common_inputs(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    paths = v6_paths(output_dir)
    if paths["hidden"].exists() and any(paths["hidden"].iterdir()):
        raise RuntimeError("v6 input preparation refuses to run after hidden data exists")
    receipt = build(cfg, paths["shared"])
    paths["shared_results"].mkdir(parents=True, exist_ok=True)
    binding = {
        "schema_version": 6,
        "role": "v6_shared_endpoint_and_aggregate_inputs",
        "config_hash": config_hash(cfg),
        "endpoint_sha256": file_sha256(paths["endpoint"] / "endpoint_data.npz"),
        "aggregate_sha256": file_sha256(paths["prospective"] / "aggregate_predictions.npz"),
        "build_receipt_sha256": file_sha256(paths["prospective"] / "build_receipt.json"),
        "raw_intermediate_states_persisted": False,
    }
    binding_path = paths["shared_results"] / "input_binding.json"
    if binding_path.exists():
        existing = json.loads(binding_path.read_text(encoding="utf-8"))
        if existing != binding:
            raise RuntimeError("existing v6 common input binding is incompatible")
    else:
        write_json_atomic(binding_path, binding)
    return {"build_receipt": receipt, "binding": binding}


def _reference_cfg(cfg: dict[str, Any], reference_id: str, training_seed: int, rollout_seed: int):
    result = copy.deepcopy(cfg)
    result["name"] = f"{cfg['name']}_{reference_id}"
    result["seed"] = int(training_seed)
    result["reference"]["seed_offset"] = int(rollout_seed) - int(training_seed)
    result["v6_reference"] = {
        "reference_id": str(reference_id),
        "training_seed": int(training_seed),
        "rollout_seed": int(rollout_seed),
    }
    return result


def _copy_endpoint(source: Path, reference_root: Path) -> Path:
    target = reference_root / "endpoint_reference" / "endpoint_data.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_sha256(target) != file_sha256(source):
            raise RuntimeError(f"reference endpoint copy is incompatible: {target}")
    else:
        shutil.copy2(source, target)
    return target


def train_reference_split(
    cfg: dict[str, Any], output_dir: str | Path, split: str
) -> dict[str, Any]:
    if split not in {"design", "evaluation"}:
        raise ValueError("split must be design or evaluation")
    paths = v6_paths(output_dir)
    endpoint = paths["endpoint"] / "endpoint_data.npz"
    if not endpoint.exists():
        raise FileNotFoundError("prepare v6 common inputs before references")
    if paths["hidden"].exists() and any(paths["hidden"].iterdir()):
        raise RuntimeError("reference training refuses to run after hidden data exists")
    if split == "evaluation":
        combined = paths["results"] / "combined_frozen_manifest.json"
        if not combined.exists():
            raise RuntimeError("evaluation references require both v6 arms to be frozen")
    ids = list(cfg["v6"][f"{split}_reference_ids"])
    seeds = list(cfg["v6"][f"{split}_reference_training_seeds"])
    rollout_seed = int(cfg["v6"][f"{split}_reference_rollout_seed"])
    if len(ids) != len(seeds) or len(set(ids)) != len(ids) or len(set(seeds)) != len(seeds):
        raise ValueError(f"invalid {split} reference registry")
    base = paths[f"{split}_references"]
    manifest_path = paths["shared_results"] / f"{split}_reference_manifest.json"
    signature = {
        "schema_version": 6,
        "split": split,
        "config_hash": config_hash(cfg),
        "endpoint_sha256": file_sha256(endpoint),
        "ids": ids,
        "training_seeds": [int(x) for x in seeds],
        "rollout_seed": rollout_seed,
        "source_sha256": file_sha256(Path(__file__)),
        "combined_manifest_sha256": (
            file_sha256(paths["results"] / "combined_frozen_manifest.json")
            if split == "evaluation" else None
        ),
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("signature") == signature:
            for row in existing["references"]:
                if file_sha256(row["checkpoint"]) != row["checkpoint_sha256"]:
                    raise RuntimeError("frozen reference checkpoint hash changed")
                if file_sha256(row["rollout"]) != row["rollout_sha256"]:
                    raise RuntimeError("frozen reference rollout hash changed")
            print(f"[v6-reference] reusing frozen {split} ensemble", flush=True)
            return existing
        raise RuntimeError(f"existing {split} reference manifest is incompatible")

    references = []
    for reference_id, training_seed in zip(ids, seeds):
        reference_root = base / str(reference_id)
        _copy_endpoint(endpoint, reference_root)
        ref_cfg = _reference_cfg(cfg, str(reference_id), int(training_seed), rollout_seed)
        receipt_path = reference_root / "endpoint_reference" / "reference_receipt.json"
        if receipt_path.exists():
            prior = json.loads(receipt_path.read_text(encoding="utf-8"))
            if prior.get("config_hash") != config_hash(ref_cfg):
                raise RuntimeError(f"incompatible interrupted reference run: {reference_id}")
        receipt = train_and_rollout(ref_cfg, reference_root)
        checkpoint = reference_root / "endpoint_reference" / "reference_checkpoint.npz"
        rollout = reference_root / "endpoint_reference" / "reference_rollout.npz"
        references.append({
            "reference_id": str(reference_id),
            "training_seed": int(training_seed),
            "rollout_seed": rollout_seed,
            "endpoint_sha256": file_sha256(endpoint),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint),
            "rollout": str(rollout.resolve()),
            "rollout_sha256": file_sha256(rollout),
            "receipt": str(receipt_path.resolve()),
            "receipt_sha256": file_sha256(receipt_path),
            "training_elapsed_seconds": float(receipt["elapsed_seconds"]),
        })
        print(f"[v6-reference] {split} {reference_id} frozen", flush=True)
    manifest = {
        "schema_version": 6,
        "status": "frozen_before_hidden_validation",
        "role": f"v6_{split}_reference_ensemble",
        "signature": signature,
        "references": references,
        "no_seed_replacement": True,
        "common_endpoint_particles": True,
        "common_rollout_indices_within_split": True,
        "selection_access_allowed": split == "design",
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def load_reference_manifest(output_dir: str | Path, split: str) -> dict[str, Any]:
    path = v6_paths(output_dir)["shared_results"] / f"{split}_reference_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"missing frozen {split} reference manifest")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("prepare", "design", "evaluation"), required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.stage == "prepare":
        result = prepare_common_inputs(cfg, args.output_dir)
    else:
        result = train_reference_split(cfg, args.output_dir, args.stage)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

