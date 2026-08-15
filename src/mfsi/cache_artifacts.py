from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mfsi.cache import (
    file_sha256,
    fingerprint,
    load_npz_cache,
    save_npz_cache,
    write_json_atomic,
)

try:
    from .law_objectives import TrialBank
except ImportError:  # direct script execution
    from law_objectives import TrialBank


def reference_signature(cfg: Mapping[str, Any], checkpoint: str | Path) -> str:
    return fingerprint({
        "schema": 1,
        "reference": cfg["reference"],
        "checkpoint_sha256": file_sha256(checkpoint),
    })


def save_reference_bank(
    output_dir: str | Path,
    *,
    cfg: Mapping[str, Any],
    checkpoint: str | Path,
    times,
    nodes,
    velocity,
    base_weights,
    in_domain_mask,
    in_domain_base_mass,
) -> Path:
    output_dir = Path(output_dir)
    sig = reference_signature(cfg, checkpoint)
    return save_npz_cache(
        output_dir / "reference_bank.npz",
        {
            "times": times,
            "nodes": nodes,
            "velocity": velocity,
            "base_weights": base_weights,
            "in_domain_mask": in_domain_mask,
            "in_domain_base_mass": in_domain_base_mass,
        },
        signature=sig,
        metadata={
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_sha256": file_sha256(checkpoint),
        },
    )


def load_reference_bank(
    output_dir: str | Path,
    *,
    cfg: Mapping[str, Any],
    checkpoint: str | Path,
):
    return load_npz_cache(
        Path(output_dir) / "reference_bank.npz",
        signature=reference_signature(cfg, checkpoint),
    )


def trial_signature(
    cfg: Mapping[str, Any],
    *,
    namespace: int,
    trials: int,
) -> str:
    return fingerprint({
        "schema": 1,
        "population": cfg["population"],
        "measurement": cfg["measurement"],
        "seed": cfg["seed"],
        "namespace": int(namespace),
        "trials": int(trials),
    })


def save_trial_bank(
    output_dir: str | Path,
    name: str,
    bank: TrialBank,
    *,
    cfg: Mapping[str, Any],
    namespace: int,
) -> Path:
    trials = int(np.asarray(bank.masses).shape[0])
    sig = trial_signature(cfg, namespace=namespace, trials=trials)
    return save_npz_cache(
        Path(output_dir) / f"{name}_bank.npz",
        {
            "masses": bank.masses,
            "sample_indices": bank.sample_indices,
            "detector_z": bank.detector_z,
            **({"alphas": bank.alphas} if bank.alphas is not None else {}),
        },
        signature=sig,
        metadata={"namespace": int(namespace), "trials": trials},
    )


def load_trial_bank(
    output_dir: str | Path,
    name: str,
    *,
    cfg: Mapping[str, Any],
    namespace: int,
    trials: int,
) -> TrialBank | None:
    loaded = load_npz_cache(
        Path(output_dir) / f"{name}_bank.npz",
        signature=trial_signature(cfg, namespace=namespace, trials=trials),
    )
    if loaded is None:
        return None
    arrays, _ = loaded
    return TrialBank(
        masses=arrays["masses"],
        sample_indices=arrays["sample_indices"],
        detector_z=arrays["detector_z"],
        alphas=arrays.get("alphas"),
    )


def write_manifest(
    output_dir: str | Path,
    *,
    cfg: Mapping[str, Any],
    checkpoint: str | Path,
) -> Path:
    output_dir = Path(output_dir)
    files = {}
    for name in (
        "reference.npz",
        "reference_bank.npz",
        "selection_bank.npz",
        "validation_bank.npz",
        "result.json",
        "result.candidate_summary.csv",
        "result.validation_trials.csv",
        "selected_designs.npz",
    ):
        path = output_dir / name
        if path.exists():
            files[name] = {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    return write_json_atomic(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "config_hash": fingerprint(cfg),
            "reference_checkpoint": str(Path(checkpoint).resolve()),
            "reference_checkpoint_sha256": file_sha256(checkpoint),
            "files": files,
        },
    )
