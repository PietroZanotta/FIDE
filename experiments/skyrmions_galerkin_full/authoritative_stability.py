"""Resumable paired-restart gate for authoritative fixed-design comparisons."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from mfsi.cache import fingerprint

from .deep_ritz import load_ritz_checkpoint, solve_deep_ritz
from .fast_production import require_fast_output_path
from .fast_workflow import _authoritative_signature
from .full_gradient import forcing_state, reconstruct_moments
from .production_artifacts import PRODUCTION_ROOT, file_sha256
from .production_workflow import load_production_data
from .workflow import (
    authoritative_evaluate, inner_config, save_candidate_checkpoint, write_json,
)


def run_authoritative_solver_benchmark(
    cfg: dict[str, Any], artifact_dir: Path, output_dir: Path,
) -> dict[str, Any]:
    """Compare host-loop and compiled full-bank solvers on production shapes."""

    output_dir = require_fast_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_production_data(cfg, artifact_dir)
    eta = jnp.asarray(cfg["envelope"]["eta0"], dtype=jnp.float64)
    reconstruction = reconstruct_moments(eta, data.selection_problem)
    state = forcing_state(
        eta, data.selection_problem, data.ritz_train_bank, reconstruction
    )
    initial, initial_metadata = load_ritz_checkpoint(artifact_dir / "ritz_full.npz")
    benchmark_cfg = deepcopy(cfg)
    benchmark_cfg["deep_ritz"].update({
        "adam_steps": 40,
        "lbfgs_iterations": 8,
        "log_every": 10,
    })
    scientific = inner_config(
        benchmark_cfg, "full",
        sample_count=int(data.ritz_train_bank.configurations.shape[1]),
    )

    def run(compiled: bool):
        local = replace(scientific, compiled_full_bank=compiled)
        started = time.perf_counter()
        solve = solve_deep_ritz(
            data.ritz_train_bank.configurations, state.projection.weights,
            state.forcing, data.selection_problem.times,
            data.selection_problem.time_weights, local,
            initial_params=initial,
        )
        flat, _ = ravel_pytree(solve.params)
        flat.block_until_ready()
        return solve, flat, time.perf_counter() - started

    old_first, old_flat, old_first_seconds = run(False)
    _, _, old_steady_seconds = run(False)
    fast_first, fast_flat, fast_first_seconds = run(True)
    _, _, fast_steady_seconds = run(True)
    parameter_relative = float(
        jnp.linalg.norm(fast_flat - old_flat)
        / jnp.maximum(jnp.linalg.norm(old_flat), 1.0e-30)
    )
    objective_absolute = abs(
        float(fast_first.lbfgs_final_objective)
        - float(old_first.lbfgs_final_objective)
    )
    equivalent = bool(parameter_relative <= 1.0e-10 and objective_absolute <= 1.0e-11)

    batch_base = replace(
        scientific, adam_steps=8, lbfgs_iterations=0,
        compiled_full_bank=False,
    )

    def run_batch(batch_size: int):
        local = replace(batch_base, lbfgs_batch_size=batch_size)
        started = time.perf_counter()
        solve = solve_deep_ritz(
            data.ritz_train_bank.configurations, state.projection.weights,
            state.forcing, data.selection_problem.times,
            data.selection_problem.time_weights, local,
            initial_params=initial,
        )
        flat, _ = ravel_pytree(solve.params)
        flat.block_until_ready()
        return solve, flat, time.perf_counter() - started

    reference_batch, reference_batch_flat, reference_batch_first = run_batch(512)
    _, _, reference_batch_steady = run_batch(512)
    batch_rows = [{
        "batch_size": 512,
        "first_seconds": reference_batch_first,
        "steady_seconds": reference_batch_steady,
        "parameter_relative_difference": 0.0,
        "objective_absolute_difference": 0.0,
        "equivalent": True,
    }]
    for batch_size in (1024, 2048):
        candidate, candidate_flat, first_seconds = run_batch(batch_size)
        _, _, steady_seconds = run_batch(batch_size)
        parameter_difference = float(
            jnp.linalg.norm(candidate_flat - reference_batch_flat)
            / jnp.maximum(jnp.linalg.norm(reference_batch_flat), 1.0e-30)
        )
        objective_difference = abs(
            float(candidate.adam_final_objective)
            - float(reference_batch.adam_final_objective)
        )
        batch_rows.append({
            "batch_size": batch_size,
            "first_seconds": first_seconds,
            "steady_seconds": steady_seconds,
            "parameter_relative_difference": parameter_difference,
            "objective_absolute_difference": objective_difference,
            "equivalent": bool(
                parameter_difference <= 1.0e-10
                and objective_difference <= 1.0e-11
            ),
        })
    eligible_batches = [row for row in batch_rows if row["equivalent"]]
    selected_batch = min(eligible_batches, key=lambda row: row["steady_seconds"])
    result = {
        "ran": True,
        "passed": equivalent,
        "platform": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "production_shape": list(data.ritz_train_bank.configurations.shape),
        "adam_steps": 40,
        "lbfgs_iterations": 8,
        "initial_checkpoint_metadata": initial_metadata,
        "old": {
            "first_seconds": old_first_seconds,
            "steady_seconds": old_steady_seconds,
            "objective": old_first.lbfgs_final_objective,
        },
        "compiled": {
            "first_seconds": fast_first_seconds,
            "steady_seconds": fast_steady_seconds,
            "objective": fast_first.lbfgs_final_objective,
        },
        "steady_speedup": old_steady_seconds / max(fast_steady_seconds, 1.0e-30),
        "batch_size_sweep": batch_rows,
        "selected_equivalent_batch_size": selected_batch["batch_size"],
        "selected_batch_speedup": (
            reference_batch_steady / max(selected_batch["steady_seconds"], 1.0e-30)
        ),
        "parameter_relative_difference": parameter_relative,
        "objective_absolute_difference": objective_absolute,
        "equivalence_thresholds": {
            "parameter_relative": 1.0e-10,
            "objective_absolute": 1.0e-11,
        },
    }
    write_json(output_dir / "result.json", result)
    return result


def _restart_signature(
    cfg: dict[str, Any], artifact_dir: Path, eta: jax.Array, *,
    restart: int, initial_checkpoint: Path | None,
) -> str:
    local_files = (
        Path(__file__), Path(__file__).with_name("deep_ritz.py"),
        Path(__file__).with_name("workflow.py"),
    )
    return fingerprint({
        "kind": "paired_authoritative_restart_v1",
        "eta_float64": jax.device_get(jnp.asarray(eta, dtype=jnp.float64)).tolist(),
        "restart": int(restart),
        "artifact_manifest_sha256": file_sha256(
            artifact_dir / "isolated_artifact_manifest.json"
        ),
        "initial_checkpoint_sha256": (
            file_sha256(initial_checkpoint) if initial_checkpoint is not None else None
        ),
        "solver": cfg["deep_ritz"],
        "certificates": cfg["certificates"],
        "projection": cfg["projection"],
        "forcing": cfg["forcing"],
        "implementation_sha256": {
            path.name: file_sha256(path) for path in local_files
        },
    })


def _cached_restart(
    cfg: dict[str, Any], artifact_dir: Path, data: Any, eta: jax.Array,
    cache_dir: Path, *, restart: int, initial_checkpoint: Path | None,
) -> tuple[dict[str, Any], bool, float]:
    cache_dir = require_fast_output_path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = _restart_signature(
        cfg, artifact_dir, eta, restart=restart,
        initial_checkpoint=initial_checkpoint,
    )
    result_path = cache_dir / "result.json"
    metadata_path = cache_dir / "metadata.json"
    if result_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature:
            return (
                json.loads(result_path.read_text(encoding="utf-8")),
                True,
                float(metadata["elapsed_seconds"]),
            )

    params = None
    checkpoint_metadata = None
    if initial_checkpoint is not None:
        params, checkpoint_metadata = load_ritz_checkpoint(initial_checkpoint)
    started = time.perf_counter()
    evaluation = authoritative_evaluate(
        eta, cfg, data, allowance_percent=3.0,
        initial_params=params, validation=False,
    )
    elapsed = time.perf_counter() - started
    write_json(result_path, evaluation.payload)
    if evaluation.params is not None:
        save_candidate_checkpoint(
            cache_dir / "checkpoint.npz", evaluation,
            role=f"paired_authoritative_restart_{restart}",
        )
    write_json(metadata_path, {
        "signature": signature,
        "elapsed_seconds": elapsed,
        "platform": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "restart": int(restart),
        "seed": int(cfg["deep_ritz"]["seed"]),
        "compiled_full_bank": bool(
            cfg["deep_ritz"].get("compiled_full_bank", False)
        ),
        "initialization": "frozen_checkpoint" if initial_checkpoint else "seeded_fresh",
        "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
        "initial_checkpoint_metadata": checkpoint_metadata,
    })
    return evaluation.payload, False, elapsed


def ordering_summary(
    pairs: list[dict[str, Any]], *, minimum_improvement: float,
) -> dict[str, Any]:
    valid_pairs = [
        row for row in pairs
        if row["incumbent"].get("valid", False)
        and row["challenger"].get("valid", False)
    ]
    differences = [
        float(row["challenger"]["action"]) - float(row["incumbent"]["action"])
        for row in valid_pairs
    ]
    all_valid = len(valid_pairs) == len(pairs) and bool(pairs)
    improvement = bool(
        all_valid and all(value < -minimum_improvement for value in differences)
    )
    regression = bool(
        all_valid and all(value > minimum_improvement for value in differences)
    )
    if improvement:
        decision = "stable_improvement"
    elif regression:
        decision = "stable_regression"
    else:
        decision = "indeterminate"
    valid_incumbent = [
        float(row["incumbent"]["action"]) for row in valid_pairs
    ]
    valid_challenger = [
        float(row["challenger"]["action"]) for row in valid_pairs
    ]
    return {
        "decision": decision,
        "passed": bool(improvement or regression),
        "all_pairs_valid": all_valid,
        "pair_count": len(pairs),
        "valid_pair_count": len(valid_pairs),
        "minimum_improvement": float(minimum_improvement),
        "action_differences_challenger_minus_incumbent": differences,
        "negative_difference_count": sum(value < -minimum_improvement for value in differences),
        "positive_difference_count": sum(value > minimum_improvement for value in differences),
        "ambiguous_difference_count": sum(
            abs(value) <= minimum_improvement for value in differences
        ),
        "best_valid_incumbent_action": min(valid_incumbent) if valid_incumbent else None,
        "best_valid_challenger_action": min(valid_challenger) if valid_challenger else None,
    }


def _validated_legacy_warm_pair(
    cfg: dict[str, Any], artifact_dir: Path,
    incumbent_eta: jax.Array, challenger_eta: jax.Array,
) -> tuple[dict[str, Any], dict[str, Any], float, float] | None:
    """Reuse the already completed GPU warm-start pair after hash validation."""

    root = artifact_dir.parent.parent / "fast_production_3pct" / (
        "gpu_authoritative_checkpoint/paired"
    )
    members = (("eta0", incumbent_eta), ("tiny", challenger_eta))
    loaded = []
    expected_checkpoint = file_sha256(artifact_dir / "ritz_full.npz")
    for name, eta in members:
        member = root / name
        result_path, metadata_path = member / "result.json", member / "metadata.json"
        if not result_path.is_file() or not metadata_path.is_file():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_signature = _authoritative_signature(
            cfg, artifact_dir, eta, validation=False
        )
        if (
            metadata.get("signature") != expected_signature
            or metadata.get("initial_checkpoint_sha256") != expected_checkpoint
        ):
            return None
        loaded.append((
            json.loads(result_path.read_text(encoding="utf-8")),
            float(metadata["elapsed_seconds"]),
        ))
    return loaded[0][0], loaded[1][0], loaded[0][1], loaded[1][1]


def run_authoritative_stability(
    cfg: dict[str, Any], artifact_dir: Path, output_dir: Path, *,
    restart_count: int,
) -> dict[str, Any]:
    """Compare eta0 and the prior tiny update under common initializations."""

    if restart_count < 2:
        raise ValueError("restart_count must be at least two")
    output_dir = require_fast_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    crosscheck = PRODUCTION_ROOT / "authoritative_crosscheck"
    incumbent = json.loads((crosscheck / "eta0.json").read_text(encoding="utf-8"))
    challenger = json.loads((crosscheck / "eta1.json").read_text(encoding="utf-8"))
    incumbent_eta = jnp.asarray(incumbent["eta"], dtype=jnp.float64)
    challenger_eta = jnp.asarray(challenger["eta"], dtype=jnp.float64)
    initial_checkpoint = artifact_dir / "ritz_full.npz"
    data = load_production_data(cfg, artifact_dir)
    pairs: list[dict[str, Any]] = []
    base_seed = int(cfg["deep_ritz"]["seed"])
    benchmark_path = output_dir.parent / "authoritative_acceleration" / "result.json"
    if not benchmark_path.is_file():
        raise RuntimeError("authoritative solver acceleration benchmark is missing")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if not benchmark.get("passed", False):
        raise RuntimeError("authoritative solver acceleration equivalence did not pass")
    selected_batch_size = int(benchmark["selected_equivalent_batch_size"])
    new_calls = 0
    cache_hits = 0
    legacy_warm_pair = _validated_legacy_warm_pair(
        cfg, artifact_dir, incumbent_eta, challenger_eta
    ) if jax.default_backend() == "gpu" else None
    for restart in range(restart_count):
        restart_cfg = deepcopy(cfg)
        restart_cfg["deep_ritz"]["seed"] = base_seed + 10007 * restart
        # The compiled scan is verified but measured no material speedup. Keep
        # the established host-loop reduction order for the scientific study.
        restart_cfg["deep_ritz"]["compiled_full_bank"] = False
        restart_cfg["deep_ritz"]["lbfgs_batch_size"] = selected_batch_size
        restart_cfg["deep_ritz"]["authoritative_restarts"] = 1
        checkpoint = initial_checkpoint if restart == 0 else None
        pair_dir = output_dir / f"restart_{restart:03d}"
        if restart == 0 and legacy_warm_pair is not None:
            left, right, left_elapsed, right_elapsed = legacy_warm_pair
            left_hit = right_hit = True
            legacy_source = str(
                artifact_dir.parent.parent / "fast_production_3pct"
                / "gpu_authoritative_checkpoint" / "paired"
            )
        else:
            left, left_hit, left_elapsed = _cached_restart(
                restart_cfg, artifact_dir, data, incumbent_eta,
                pair_dir / "incumbent", restart=restart,
                initial_checkpoint=checkpoint,
            )
            right, right_hit, right_elapsed = _cached_restart(
                restart_cfg, artifact_dir, data, challenger_eta,
                pair_dir / "challenger", restart=restart,
                initial_checkpoint=checkpoint,
            )
            legacy_source = None
        cache_hits += int(left_hit) + int(right_hit)
        new_calls += int(not left_hit) + int(not right_hit)
        pairs.append({
            "restart": restart,
            "seed": int(restart_cfg["deep_ritz"]["seed"]),
            "initialization": "frozen_checkpoint" if checkpoint else "seeded_fresh",
            "incumbent": left,
            "challenger": right,
            "incumbent_cache_hit": left_hit,
            "challenger_cache_hit": right_hit,
            "incumbent_elapsed_seconds": left_elapsed,
            "challenger_elapsed_seconds": right_elapsed,
            "legacy_cache_source": legacy_source,
            "action_difference_challenger_minus_incumbent": (
                float(right["action"]) - float(left["action"])
                if left.get("valid", False) and right.get("valid", False) else None
            ),
        })
        partial = {
            "ran": True, "in_progress": True,
            "platform": jax.default_backend(), "device": str(jax.devices()[0]),
            "requested_restart_count": restart_count, "completed_pairs": len(pairs),
            "pairs": pairs,
        }
        write_json(output_dir / "result.json", partial)

    tolerance = float(cfg["envelope"].get("minimum_improvement", 1.0e-6))
    summary = ordering_summary(pairs, minimum_improvement=tolerance)
    result = {
        "ran": True,
        "in_progress": False,
        "platform": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "compiled_full_bank": False,
        "lbfgs_batch_size": selected_batch_size,
        "requested_restart_count": restart_count,
        "new_authoritative_calls": new_calls,
        "cache_hits": cache_hits,
        "incumbent_eta": jax.device_get(incumbent_eta).tolist(),
        "challenger_eta": jax.device_get(challenger_eta).tolist(),
        "pairs": pairs,
        "ordering": summary,
        "eligible_for_further_eta_refinement": bool(
            summary["decision"] == "stable_improvement"
        ),
        "original_production_incumbent_modified": False,
    }
    write_json(output_dir / "result.json", result)
    return result


__all__ = [
    "ordering_summary", "run_authoritative_solver_benchmark",
    "run_authoritative_stability",
]
