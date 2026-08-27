"""Read-only discovery and isolated materialization of production artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

import numpy as np

from .workflow import OUTPUT_ROOT, require_output_path, write_json

PRODUCTION_ROOT = OUTPUT_ROOT / "production_galerkin"

FROZEN_FILES = (
    "reference.npz",
    "reference_manifest.json",
    "truth_banks.npz",
    "reference_bank_projection.npz",
    "reference_bank_ritz_train.npz",
    "reference_bank_ritz_audit.npz",
    "reference_bank_validation_fit.npz",
    "reference_bank_validation_audit.npz",
)

REFERENCE_FILES = (
    "result.json",
    "bank_manifest.json",
    "ritz_full.npz",
    "ritz_law.npz",
    "ritz_validation_full.npz",
    "ritz_validation_law.npz",
    "three_percent_validation.json",
)


def require_production_output_path(path: Path) -> Path:
    resolved = require_output_path(path)
    root = PRODUCTION_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"production output must be beneath {root}, got {resolved}")
    return resolved


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_artifact_sets(search_roots: Iterable[Path]) -> list[Path]:
    """Return complete-looking roots without writing or opening scientific arrays."""

    discovered: set[Path] = set()
    for raw_root in search_roots:
        root = Path(raw_root).resolve()
        if not root.is_dir():
            continue
        candidates = [root] if (root / "truth_banks.npz").is_file() else []
        candidates.extend(path.parent for path in root.rglob("truth_banks.npz"))
        for candidate in candidates:
            if all((candidate / name).is_file() for name in FROZEN_FILES):
                discovered.add(candidate.resolve())
    return sorted(discovered)


def _npz_shapes(path: Path) -> dict[str, list[Any]]:
    with np.load(path, allow_pickle=False) as arrays:
        return {
            name: [list(arrays[name].shape), str(arrays[name].dtype)]
            for name in arrays.files
            if not name.startswith("__") and name != "metadata_json"
        }


def inspect_production_source(source: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate identities, shapes, configuration, and metadata without writing."""

    source = Path(source).resolve()
    expected_particles = int(cfg["physics"]["n_particles"])
    expected_times = int(cfg["physics"]["time_nodes"])
    expected = {
        "truth_banks.npz": {
            "times": [expected_times],
            "design": [expected_times, int(cfg["banks"]["truth_design_samples"]), expected_particles, 2],
            "validation": [expected_times, int(cfg["banks"]["truth_validation_samples"]), expected_particles, 2],
            "endpoint0": [int(cfg["banks"]["endpoint_samples"]), expected_particles, 2],
            "endpoint1": [int(cfg["banks"]["endpoint_samples"]), expected_particles, 2],
        },
        **{
            f"reference_bank_{name}.npz": {
                "configurations": [expected_times, int(cfg["banks"][f"{name}_samples"]), expected_particles, 2],
                "velocity": [expected_times, int(cfg["banks"][f"{name}_samples"]), expected_particles, 2],
                "base_weights": [expected_times, int(cfg["banks"][f"{name}_samples"]),],
            }
            for name in (
                "projection", "ritz_train", "ritz_audit",
                "validation_fit", "validation_audit",
            )
        },
    }
    missing = [name for name in FROZEN_FILES + REFERENCE_FILES if not (source / name).is_file()]
    mismatches: list[dict[str, Any]] = []
    shapes: dict[str, Any] = {}
    for name, expected_keys in expected.items():
        path = source / name
        if not path.is_file():
            continue
        actual = _npz_shapes(path)
        shapes[name] = actual
        for key, expected_shape in expected_keys.items():
            actual_shape = actual.get(key, [None])[0]
            if actual_shape != expected_shape:
                mismatches.append({
                    "file": name, "array": key,
                    "expected_shape": expected_shape, "actual_shape": actual_shape,
                })
    result_metadata: dict[str, Any] = {}
    result_path = source / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        production_cfg = result.get("config", {})
        core_keys = (
            "seed", "physics", "measurement", "moment_reconstruction", "banks",
            "reference_training", "projection", "forcing", "deep_ritz", "certificates",
        )
        config_matches = all(production_cfg.get(key) == cfg.get(key) for key in core_keys)
        full = result.get("full_3_percent", {})
        law = result.get("law_anchor", {})
        result_metadata = {
            "schema_version": result.get("schema_version"),
            "git_commit": result.get("git_commit"),
            "config_hash": result.get("config_hash"),
            "core_config_matches_isolated_authoritative_config": config_matches,
            "selected_eta": full.get("eta"),
            "selected_risk": full.get("selection_risk"),
            "selected_action": full.get("selection_action"),
            "risk_limit": full.get("risk_limit"),
            "law_eta": law.get("eta"),
            "law_risk": law.get("risk"),
        }
        if not config_matches:
            mismatches.append({"file": "result.json", "field": "core config", "actual": "mismatch"})
    inventory = []
    for name in FROZEN_FILES + REFERENCE_FILES:
        path = source / name
        if path.is_file():
            stat = path.stat()
            inventory.append({
                "name": name,
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": file_sha256(path),
            })
    return {
        "source": str(source),
        "complete": not missing and not mismatches,
        "missing": missing,
        "mismatches": mismatches,
        "array_shapes": shapes,
        "result_metadata": result_metadata,
        "inventory": inventory,
        "detector_noise": {
            "stored_as_file": False,
            "convention": "deterministically reconstructed from config seed, observation offset, shape, and float64 JAX PRNG",
        },
        "whitening": {
            "stored_as_file": False,
            "convention": "deterministically recomputed from the frozen truth-design feature bank",
        },
    }


def materialize_production_source(
    source_audit: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Copy a validated source once; existing identical files are left untouched."""

    if not source_audit.get("complete"):
        raise RuntimeError("cannot materialize an incomplete production artifact source")
    destination = require_production_output_path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    source = Path(source_audit["source"])
    copied: list[str] = []
    reused: list[str] = []
    for row in source_audit["inventory"]:
        name = row["name"]
        source_path = source / name
        destination_path = destination / name
        if destination_path.exists():
            if file_sha256(destination_path) != row["sha256"]:
                raise RuntimeError(f"refusing to overwrite nonmatching artifact: {destination_path}")
            reused.append(name)
            continue
        shutil.copy2(source_path, destination_path)
        if file_sha256(destination_path) != row["sha256"]:
            raise RuntimeError(f"copied artifact hash mismatch: {destination_path}")
        copied.append(name)
    result = {
        **source_audit,
        "destination": str(destination),
        "copied": copied,
        "reused_identical": reused,
        "source_read_only": True,
    }
    write_json(destination / "isolated_artifact_manifest.json", result)
    return result


def run_production_preflight(
    cfg: dict[str, Any], source: Path, output_dir: Path
) -> dict[str, Any]:
    output_dir = require_production_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = inspect_production_source(source, cfg)
    result: dict[str, Any] = {"preflight": audit, "passed": bool(audit["complete"])}
    if audit["complete"]:
        result["materialization"] = materialize_production_source(
            audit, PRODUCTION_ROOT / "artifacts"
        )
    write_json(output_dir / "result.json", result)
    return result


__all__ = [
    "FROZEN_FILES", "PRODUCTION_ROOT", "discover_artifact_sets",
    "file_sha256", "inspect_production_source", "materialize_production_source",
    "require_production_output_path", "run_production_preflight",
]
