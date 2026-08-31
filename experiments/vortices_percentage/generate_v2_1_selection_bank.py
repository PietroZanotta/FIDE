#!/usr/bin/env python3
"""Generate the one shared, prospective V2.1 namespace-11 selection bank."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUTPUT = HERE / "outputs" / "prospective_v2_1" / "selection"
for path in (REPO / "src", REPO / "experiments" / "vortices_percentage", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment import make_observation_bank  # noqa: E402
from mfsi.cache import fingerprint  # noqa: E402
from v2_1_contract import CONFIG, load_resolved_config, sha256_file  # noqa: E402


MANIFEST = HERE / "VORTICES_V2_1_FREEZE_MANIFEST.json"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def verify_manifest() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_PROSPECTIVE_BEFORE_V2_1_SELECTION_BANK":
        raise RuntimeError("V2.1 manifest is not frozen")
    standalone_inputs = {
        CONFIG: "0536565142fc1954b4d944153f9d76e969ac72d66762e3173f7e925fe0bf7211",
        HERE / "base_experiment_config.json": "8f57f167675718b19d7ffc1741a8175adbe22069ff4043634b62df8dcf100ed0",
        HERE / "experiment.py": "5bcd5b3c96668cabf6d7a8b2b1944f48f490635763b997172584328551a9a4c4",
        HERE / "inputs" / "truth_bank.npz": "d897ff7fc44c0b85d7bb5391c0cc25895b4301e9c2ce00184697a1899d853b5b",
    }
    for path, expected in standalone_inputs.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"standalone frozen input hash mismatch: {path}")
    return manifest


def main() -> None:
    manifest = verify_manifest()
    config, overlay = load_resolved_config()
    banks = config["observation_banks"]
    identity_tuple = (
        int(banks["generation_seed"]),
        int(banks["selection_namespace"]),
        int(banks["selection_master_trials"]),
        int(banks["finite_particles"]),
        list(map(int, banks["acquisition_indices_on_21_node_grid"])),
        int(banks["observables"]),
    )
    expected = (10, 11, 128, 2000, [0, 2, 5, 8, 10, 12, 15, 18, 20], 4)
    if identity_tuple != expected:
        raise RuntimeError(f"selection-bank identity changed: {identity_tuple!r}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    bank_path = OUTPUT / "shared_selection_bank.npz"
    receipt_path = OUTPUT / "shared_selection_bank_receipt.json"
    if bank_path.exists() or receipt_path.exists():
        raise RuntimeError("V2.1 bank or receipt already exists; refusing regeneration")

    seed, namespace, trials, finite_n, acquisition, observables = identity_tuple
    bank = make_observation_bank(
        seed=seed,
        namespace=namespace,
        trials=trials,
        acquisition_k=len(acquisition),
        finite_n=finite_n,
        truth_particle_count=50000,
        n_observables=observables,
    )
    indices = np.asarray(bank.sample_indices, dtype=np.int32)
    detector = np.asarray(bank.detector_z, dtype=np.float64)
    trial_ids = np.arange(trials, dtype=np.int32)
    identity = {
        "schema_version": 1,
        "data_role": "SELECTION",
        "method_version": "V2.1",
        "generation_seed": seed,
        "namespace": namespace,
        "trials": trials,
        "finite_particles": finite_n,
        "truth_particle_count": 50000,
        "acquisition_indices": acquisition,
        "observables": observables,
        "detector_noise_standard_deviation": float(banks["detector_noise_standard_deviation"]),
        "reference_training_seeds": config["reference_replicates"]["training_seeds"],
        "shared_across_all_references_and_methods": True,
        "rng": "numpy.default_rng(SeedSequence([generation_seed,namespace]))",
    }
    signature = fingerprint(identity)
    temporary = bank_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        sample_indices=indices,
        detector_z=detector,
        trial_ids=trial_ids,
        acquisition_indices=np.asarray(acquisition, dtype=np.int32),
        identity_json=np.asarray(json.dumps(identity, sort_keys=True)),
        signature=np.asarray(signature),
    )
    os.replace(temporary, bank_path)
    receipt = {
        **identity,
        "status": "FROZEN_SHARED_V2_1_SELECTION_BANK",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bank_path": str(bank_path.resolve()),
        "bank_sha256": sha256_file(bank_path),
        "signature": signature,
        "sample_indices_shape": list(indices.shape),
        "sample_indices_dtype": str(indices.dtype),
        "detector_z_shape": list(detector.shape),
        "detector_z_dtype": str(detector.dtype),
        "trial_ids": trial_ids.tolist(),
        "selection_config_sha256": sha256_file(CONFIG),
        "freeze_manifest_sha256": sha256_file(MANIFEST),
        "validation_namespace_used": False,
        "stress_test_namespace_used": False,
        "config_status_at_generation": overlay["status"],
    }
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
