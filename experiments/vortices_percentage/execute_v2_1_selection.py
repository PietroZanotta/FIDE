#!/usr/bin/env python3
"""Execute the frozen V2.1 feasibility-first prospective selection."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

import jax.numpy as jnp
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OLD_SELECTION = HERE / "outputs" / "prospective_v2" / "selection"
OUTPUT = HERE / "outputs" / "prospective_v2_1" / "selection"
HARNESS_PATH = HERE / "v2_selection_harness.py"
MANIFEST = HERE / "VORTICES_V2_1_FREEZE_MANIFEST.json"
PROVENANCE = HERE / "VORTICES_V2_1_RANDOMNESS_PROVENANCE.md"

V1_DIR = HERE
for path in (REPO / "src", REPO / "experiments", V1_DIR, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bounded_reference import BoxTransformedReferenceFlow  # noqa: E402
from experiment import ObservationTrialBank, VortexExperiment  # noqa: E402
from core import DevelopmentContext  # noqa: E402
from v2_1_contract import (  # noqa: E402
    CONFIG,
    canonical_resolved_sha256,
    generated_starts,
    load_resolved_config,
    sha256_file,
)


def load_harness():
    spec = importlib.util.spec_from_file_location("frozen_v2_selection_harness_for_v2_1", HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen V2 harness: {HARNESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest() -> dict[str, Any]:
    manifest = load_json(MANIFEST)
    if manifest.get("status") != "FROZEN_PROSPECTIVE_BEFORE_V2_1_SELECTION_BANK":
        raise RuntimeError("V2.1 freeze manifest is not frozen")
    standalone_inputs = {
        HERE / "base_experiment_config.json": "8f57f167675718b19d7ffc1741a8175adbe22069ff4043634b62df8dcf100ed0",
        HERE / "experiment.py": "5bcd5b3c96668cabf6d7a8b2b1944f48f490635763b997172584328551a9a4c4",
        HERE / "bounded_reference.py": "bb9bb091329cf1cda54252d4b86463c900307f5ed7b983fd30de959ffa4d7cbe",
        HERE / "inputs" / "truth_bank.npz": "d897ff7fc44c0b85d7bb5391c0cc25895b4301e9c2ce00184697a1899d853b5b",
        HERE / "inputs" / "reference_endpoints.npz": "ad4006927e268c52f621c16c773f0600d803370bd21fb5e0816d82a70dbdfbba",
    }
    for path, expected in standalone_inputs.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"standalone input mismatch: {path}: {actual} != {expected}")
    return manifest


def configure_harness(harness, config: dict[str, Any]) -> tuple[dict[str, Any], float]:
    manifest = verify_manifest()
    bank = OUTPUT / "shared_selection_bank.npz"
    receipt_path = OUTPUT / "shared_selection_bank_receipt.json"
    if not bank.is_file() or not receipt_path.is_file():
        raise RuntimeError("V2.1 selection bank and receipt must exist")
    receipt = load_json(receipt_path)
    if (receipt.get("status"), receipt.get("generation_seed"), receipt.get("namespace"), receipt.get("trials")) != (
        "FROZEN_SHARED_V2_1_SELECTION_BANK", 10, 11, 128
    ):
        raise RuntimeError("V2.1 selection bank receipt identity mismatch")
    if sha256_file(bank) != receipt["bank_sha256"]:
        raise RuntimeError("V2.1 selection bank hash mismatch")
    if receipt["selection_config_sha256"] != sha256_file(CONFIG):
        raise RuntimeError("V2.1 selection bank/config mismatch")
    if receipt["freeze_manifest_sha256"] != sha256_file(MANIFEST):
        raise RuntimeError("V2.1 selection bank/manifest mismatch")

    harness.HERE = OUTPUT
    harness.CONFIG = CONFIG
    harness.SEEDS = PROVENANCE
    harness.BANK_PATH = bank
    harness.BANK_RECEIPT = receipt_path
    harness.BASE_HASHES = {
        CONFIG: sha256_file(CONFIG),
        PROVENANCE: sha256_file(PROVENANCE),
        bank: receipt["bank_sha256"],
        receipt_path: sha256_file(receipt_path),
    }
    return manifest, float(manifest["common_physical_bandwidth"])


def load_experiments(config: dict[str, Any], bank_path: Path):
    v1_dir = V1_DIR
    reference_root = HERE / "outputs" / "prospective_v2" / "references"
    cfg = load_json(v1_dir / "base_experiment_config.json")
    with np.load(v1_dir / "inputs" / "truth_bank.npz", allow_pickle=False) as raw:
        times = np.asarray(raw["times"], dtype=np.float64)
        truth = jnp.asarray(raw["particles"], dtype=jnp.float64)
    with np.load(bank_path, allow_pickle=False) as raw:
        bank = ObservationTrialBank(
            jnp.asarray(raw["sample_indices"], dtype=jnp.int32),
            jnp.asarray(raw["detector_z"], dtype=jnp.float64),
        )
        np.testing.assert_array_equal(raw["trial_ids"], np.arange(128))
    experiments, contexts = [], []
    for seed in config["reference_replicates"]["training_seeds"]:
        root = reference_root / f"reference_seed_{seed}"
        qualification = load_json(root / "qualification_receipt.json")
        if qualification["status"] != "PASS":
            raise RuntimeError(f"reference {seed} is not qualified")
        reference = BoxTransformedReferenceFlow.from_npz(
            root / "reference.npz",
            substeps_per_interval=int(cfg["reference"]["rk4_substeps_per_time_interval"]),
        )
        with np.load(root / "reference_bank.npz", allow_pickle=False) as raw:
            np.testing.assert_allclose(raw["times"], times, rtol=0, atol=0)
            nodes = jnp.asarray(raw["nodes"], dtype=jnp.float64)
            velocity = jnp.asarray(raw["velocity"], dtype=jnp.float64)
            weights = jnp.asarray(raw["weights"], dtype=jnp.float64)
        experiment = VortexExperiment(
            cfg,
            reference,
            truth_particles=truth,
            reference_nodes=nodes,
            reference_velocity=velocity,
            reference_weights=weights,
        )
        experiments.append(experiment)
        contexts.append(DevelopmentContext(experiment, bank, times, cfg, root, bank_path, 11))
    return experiments, contexts, bank


def full_allowance(harness, config, schedule, evaluator, starts, old, population, law, tangent, allowance, incumbent):
    out = OUTPUT / "allowances" / f"risk_{str(allowance).replace('.', 'p')}pct" / "full.json"
    if out.exists():
        return load_json(out)
    L_max = float(population["L_max"])
    R_max = float(law["risk_caps"][str(allowance)])
    centers = [population["winner"]["eta"], law["winner"]["eta"], tangent["winner"]["eta"]]
    if incumbent is not None:
        centers.append(incumbent["winner"]["eta"])
    pool = {}
    for index, eta in enumerate(starts):
        harness.add_candidate(pool, eta, f"generated_{index:02d}")
    for label, eta in zip(("new_population", "new_law", "current_tangent", "previous_tighter_full"), centers):
        harness.add_candidate(pool, eta, label)
    for row in old:
        harness.add_candidate(pool, row["eta"], row["label"])
    rounds = []
    seeds = list(map(int, schedule["full_local_cloud_seeds_by_allowance_and_round"][str(allowance)]))
    for round_index, (scale, seed) in enumerate(zip((0.06, 0.03, 0.015), seeds), 1):
        local = harness.deterministic_local_cloud(
            centers,
            count_per_center=10,
            scale=scale,
            seed=seed,
            box=config["risk_and_geometry"]["center_box"],
        )
        for index, eta in enumerate(local):
            harness.add_candidate(pool, eta, f"full_round_{round_index}_local_{index:03d}")
        rounds.append({"round": round_index, "scale": scale, "seed": seed, "generated": len(local), "pool_after": len(pool)})

    candidates = list(pool.values())
    def exact_screen(row):
        population_record = evaluator.population(row["eta"])
        risk_record = evaluator.risk(row["eta"])
        feasible = bool(
            population_record["valid"]
            and population_record["value"] <= L_max + 1e-12
            and risk_record["valid"]
            and risk_record["value"] <= R_max + 1e-12
        )
        return {**row, "exact_population": population_record, "exact_risk": risk_record, "valid": feasible}

    audits = []
    with ThreadPoolExecutor(max_workers=4) as workers:
        started = time.perf_counter()
        print(f"[V2.1 Full {allowance}% exact feasibility] 0/{len(candidates)}", flush=True)
        for completed, row in enumerate(workers.map(exact_screen, candidates), 1):
            audits.append(row)
            if completed == 1 or completed == len(candidates) or completed % max(1, len(candidates) // 10) == 0:
                elapsed = time.perf_counter() - started
                print(f"[V2.1 Full {allowance}% exact feasibility] {completed}/{len(candidates)} elapsed={elapsed:.1f}s", flush=True)
    feasible = [row for row in audits if row["valid"]]
    if not feasible:
        raise RuntimeError(f"SCIENTIFIC_SELECTION_FAIL:no risk-feasible Full candidate at {allowance}%")

    def proxy(row):
        return {**row, "proxy": evaluator.full(row["eta"], 32, (64, 32), decomposition=False)}
    proxy_rows = []
    with ThreadPoolExecutor(max_workers=4) as workers:
        for row in workers.map(proxy, feasible):
            proxy_rows.append(row)
    proxy_rows.sort(key=lambda row: (not row["proxy"]["valid"], row["proxy"]["value"] if row["proxy"]["valid"] else float("inf"), row["candidate_id"]))

    mandatory_etas = (
        [law["winner"]["eta"], tangent["winner"]["eta"]]
        if incumbent is None
        else [incumbent["winner"]["eta"], tangent["winner"]["eta"]]
    )
    mandatory_keys = []
    for eta in mandatory_etas:
        key = harness.candidate_key(eta)
        if key not in mandatory_keys:
            mandatory_keys.append(key)
    by_key = {harness.candidate_key(row["eta"]): row for row in proxy_rows}
    missing = [key for key in mandatory_keys if key not in by_key]
    if missing:
        raise RuntimeError(f"NUMERICAL_CONSISTENCY_FAIL:mandatory Full finalist is not feasible: {missing}")
    promoted = [by_key[key] for key in mandatory_keys]
    for row in proxy_rows:
        if harness.candidate_key(row["eta"]) not in {harness.candidate_key(item["eta"]) for item in promoted}:
            promoted.append(row)
        if len(promoted) >= min(8, len(proxy_rows)):
            break

    evaluator.prewarm_reflected_kernels((256, 128))
    def exact_full(row):
        return {**row, "final": evaluator.full(row["eta"], 128, (256, 128), decomposition=True)}
    with ThreadPoolExecutor(max_workers=min(4, len(promoted))) as workers:
        finals = list(workers.map(exact_full, promoted))
    valid = [row for row in finals if row["final"]["valid"]]
    if len(valid) != len(finals):
        raise RuntimeError(f"NUMERICAL_CONSISTENCY_FAIL:invalid exact Full finalist at {allowance}%")
    challenger = min(valid, key=lambda row: row["final"]["value"])
    if incumbent is None:
        winner = challenger
        law_row = by_key[harness.candidate_key(law["winner"]["eta"])]
        law_final = next(row for row in finals if harness.candidate_key(row["eta"]) == harness.candidate_key(law_row["eta"]))
        if winner["final"]["value"] > law_final["final"]["value"] + 1e-6:
            raise RuntimeError("NUMERICAL_CONSISTENCY_FAIL:selected Full exceeds mandatory Law")
    else:
        incumbent_row = next(row for row in valid if harness.candidate_key(row["eta"]) == harness.candidate_key(incumbent["winner"]["eta"]))
        winner = challenger if challenger["final"]["value"] < incumbent_row["final"]["value"] - 1e-6 else incumbent_row
        if winner["final"]["value"] > incumbent_row["final"]["value"] + 1e-6:
            raise RuntimeError("NUMERICAL_CONSISTENCY_FAIL:Full incumbent nesting failed")
    result = {
        "status": "PASS",
        "stage": "full_feasibility_first",
        "allowance_percent": allowance,
        "rounds": rounds,
        "candidate_count": len(candidates),
        "feasible_count": len(feasible),
        "risk_audits": audits,
        "proxy_feasible_only": proxy_rows,
        "mandatory_finalist_keys": [list(key) for key in mandatory_keys],
        "finalists": finals,
        "winner": winner,
    }
    harness.atomic_json(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", choices=("population", "law", "tangent", "full"), default="full")
    args = parser.parse_args()
    harness = load_harness()
    config, overlay = load_resolved_config()
    manifest, bandwidth = configure_harness(harness, config)
    experiments, contexts, bank = load_experiments(config, harness.BANK_PATH)
    evaluator = harness.Evaluator(config, experiments, contexts, bank, bandwidth)
    starts = generated_starts(config)
    old = config["optimization"]["old_v1_proposals"]
    schedule = {
        "tangent_local_cloud_seed_by_allowance": overlay["two_digit_randomness"]["tangent_local_cloud_seed_by_allowance"],
        "full_local_cloud_seeds_by_allowance_and_round": overlay["two_digit_randomness"]["full_local_cloud_seeds_by_allowance_and_round"],
    }
    population = harness.stage_population(config, experiments, evaluator, starts, old)
    if args.through == "population":
        return
    law = harness.stage_law(config, experiments, evaluator, starts, old, population)
    allowances = list(map(float, config["risk_and_geometry"]["risk_allowance_percentages"]))
    law["risk_caps"] = {str(p): float(law["R_star"] + p / 100.0 * abs(law["R_star"])) for p in allowances}
    harness.atomic_json(OUTPUT / "law" / "current_result.json", law)
    if args.through == "law":
        return
    tangent_results, full_results = [], []
    tangent_incumbent = full_incumbent = None
    for index, allowance in enumerate(allowances):
        tangent = harness.tangent_allowance(
            config, schedule, experiments, evaluator, starts, old, population, law,
            allowance, index, tangent_incumbent,
        )
        tangent_results.append(tangent)
        tangent_incumbent = tangent
        if args.through == "tangent":
            continue
        full = full_allowance(
            harness, config, schedule, evaluator, starts, old, population, law,
            tangent, allowance, full_incumbent,
        )
        full_results.append(full)
        full_incumbent = full
        better_risk = min(
            (row["exact_risk"]["value"], row["eta"])
            for row in full["risk_audits"]
            if row["exact_risk"]["valid"]
        )
        if better_risk[0] < float(law["R_star"]) - float(config["optimization"]["law"]["anchor_consistency_tolerance"]):
            raise RuntimeError(f"LAW_ANCHOR_REFINEMENT_REQUIRED:{better_risk[0]}:{json.dumps(better_risk[1])}")
    if args.through == "tangent":
        return
    winners = {
        "schema_version": 1,
        "status": "FROZEN_V2_1_SELECTION_WINNERS",
        "method_version": "V2.1",
        "data_role": "SELECTION",
        "resolved_config_sha256": canonical_resolved_sha256(config),
        "selection_config_sha256": sha256_file(CONFIG),
        "freeze_manifest_sha256": sha256_file(MANIFEST),
        "selection_bank_sha256": harness.BASE_HASHES[harness.BANK_PATH],
        "common_bandwidth": bandwidth,
        "population": population,
        "law": law,
        "tangent": tangent_results,
        "full": full_results,
        "stress_test_namespace_used": False,
        "validation_namespace_used": False,
    }
    harness.atomic_json(OUTPUT / "frozen_winners.json", winners)
    print(json.dumps({"status": "PASS", "frozen_winners": str(OUTPUT / "frozen_winners.json")}, indent=2))


if __name__ == "__main__":
    main()
