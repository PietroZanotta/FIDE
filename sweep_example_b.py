#!/usr/bin/env python3
"""Train/evaluate Experiment B across independent training and evaluation seeds.

This is intentionally separate from example_b.py's single-run output path: sweep
checkpoints/results are isolated under the requested output directory and never
replace the packaged/current single-run checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import example_b as exb
from backend_runtime import normalize_backend


def parse_seeds(text: str) -> list[int]:
    vals = [x for x in re.split(r"[\s,]+", text.strip()) if x]
    if not vals:
        raise argparse.ArgumentTypeError("seed list cannot be empty")
    return [int(x) for x in vals]


def save_model(path: Path, model) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        reference_params=np.asarray(exb.core.flatten_mlp(model[0])),
        potential_params=np.asarray(exb.core.flatten_mlp(model[1])),
        state_dim=np.array(exb.STATE_DIM, dtype=np.int32),
    )


def load_model(path: Path):
    data = np.load(path)
    return (
        exb.unflatten(jnp.asarray(data["reference_params"]), exb.REFERENCE_HIDDEN, exb.STATE_DIM),
        exb.unflatten(jnp.asarray(data["potential_params"]), exb.RITZ_HIDDEN, 1),
    )


def budgets(mode: str) -> dict:
    if mode == "smoke":
        return dict(
            ref_steps=8, ref_batch=256,
            ritz_steps=6, n_times=2, ptime=64, pool=1, refresh=0, endpoint_particles=64,
            holdout_n=512,
            n_particles=128, flow_steps=8, mgd_steps=16, target_bank=192,
        )
    if mode == "quick":
        return dict(
            ref_steps=120, ref_batch=768,
            ritz_steps=100, n_times=4, ptime=128, pool=3, refresh=50, endpoint_particles=128,
            holdout_n=2000,
            n_particles=512, flow_steps=40, mgd_steps=120, target_bank=768,
        )
    return dict(
        ref_steps=1800, ref_batch=3072,
        ritz_steps=1600, n_times=10, ptime=384, pool=8, refresh=250, endpoint_particles=384,
        holdout_n=12000,
        n_particles=3072, flow_steps=160, mgd_steps=800, target_bank=4096,
    )


def aggregate(records: list[dict]) -> dict:
    """Summarize a complete training-seed x evaluation-seed design.

    The Cartesian cells are repeated measurements, not iid replications.  We
    retain their pooled distribution as a descriptive quantity, but scientific
    uncertainty comes from independently resampling the two seed dimensions.
    """
    return crossed_seed_analysis(records)["methods"]


BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260809


def _sample_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1) if values.size > 1 else 0.0),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _crossed_bootstrap_stats(matrix: np.ndarray, train_draws: np.ndarray,
                             eval_draws: np.ndarray) -> dict:
    draws = matrix[train_draws[:, :, None], eval_draws[:, None, :]].mean(axis=(1, 2))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "method": "crossed_seed_percentile_bootstrap",
        "n_bootstrap": int(draws.size),
        "mean": float(matrix.mean()),
        "bootstrap_se": float(draws.std(ddof=1)),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def _variance_components(matrix: np.ndarray) -> dict:
    """Balanced two-way random-effects method-of-moments diagnostics."""
    n_train, n_eval = matrix.shape
    grand = matrix.mean()
    train_means = matrix.mean(axis=1)
    eval_means = matrix.mean(axis=0)
    residual = matrix - train_means[:, None] - eval_means[None, :] + grand
    if n_train < 2 or n_eval < 2:
        return {
            "training_seed": 0.0, "evaluation_seed": 0.0,
            "residual_or_interaction": 0.0,
            "grand_mean_se": 0.0,
        }
    ms_train = n_eval * np.var(train_means, ddof=1)
    ms_eval = n_train * np.var(eval_means, ddof=1)
    ms_residual = np.sum(residual * residual) / ((n_train - 1) * (n_eval - 1))
    var_train = max(float((ms_train - ms_residual) / n_eval), 0.0)
    var_eval = max(float((ms_eval - ms_residual) / n_train), 0.0)
    var_residual = float(ms_residual)
    mean_se = math.sqrt(
        var_train / n_train + var_eval / n_eval
        + var_residual / (n_train * n_eval)
    )
    return {
        "training_seed": var_train,
        "evaluation_seed": var_eval,
        "residual_or_interaction": var_residual,
        "grand_mean_se": mean_se,
    }


def _matrix(records: list[dict], method: str, metric: str,
            train_seeds: list[int], eval_seeds: list[int]) -> np.ndarray:
    values = {
        (int(row["train_seed"]), int(row["eval_seed"])): float(row[metric])
        for row in records if row["method"] == method
    }
    missing = [
        (train_seed, eval_seed)
        for train_seed in train_seeds for eval_seed in eval_seeds
        if (train_seed, eval_seed) not in values
    ]
    if missing:
        raise ValueError(f"incomplete crossed design for {method}/{metric}: missing {missing[:5]}")
    return np.asarray([
        [values[(train_seed, eval_seed)] for eval_seed in eval_seeds]
        for train_seed in train_seeds
    ], dtype=np.float64)


def _matrix_summary(matrix: np.ndarray, train_draws: np.ndarray,
                    eval_draws: np.ndarray) -> dict:
    return {
        "all_train_eval_cells": _sample_stats(matrix.ravel()),
        "training_seed_means": _sample_stats(matrix.mean(axis=1)),
        "evaluation_seed_means": _sample_stats(matrix.mean(axis=0)),
        "crossed_seed_bootstrap": _crossed_bootstrap_stats(matrix, train_draws, eval_draws),
        "random_effect_variance_components": _variance_components(matrix),
    }


def crossed_seed_analysis(records: list[dict]) -> dict:
    if not records:
        raise ValueError("cannot aggregate an empty sweep")
    train_seeds = sorted({int(row["train_seed"]) for row in records})
    eval_seeds = sorted({int(row["eval_seed"]) for row in records})
    methods = sorted({row["method"] for row in records})
    metrics = [
        key for key, value in records[0].items()
        if key not in {"train_seed", "eval_seed", "method"}
        and isinstance(value, (int, float))
    ]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    train_draws = rng.integers(0, len(train_seeds), (BOOTSTRAP_REPLICATES, len(train_seeds)))
    eval_draws = rng.integers(0, len(eval_seeds), (BOOTSTRAP_REPLICATES, len(eval_seeds)))

    matrices: dict[tuple[str, str], np.ndarray] = {}
    method_output = {}
    for method in methods:
        entry = {"n_train_eval_cells": len(train_seeds) * len(eval_seeds)}
        for metric in metrics:
            matrix = _matrix(records, method, metric, train_seeds, eval_seeds)
            matrices[(method, metric)] = matrix
            entry[metric] = _matrix_summary(matrix, train_draws, eval_draws)
        method_output[method] = entry

    contrasts = {}
    focal = "mfsi_learned_safe"
    if focal in methods:
        for baseline in ("moment_tangent", "mgd_style", "raw_si"):
            if baseline not in methods:
                continue
            name = f"{focal}_minus_{baseline}"
            contrasts[name] = {}
            for metric in metrics:
                difference = matrices[(focal, metric)] - matrices[(baseline, metric)]
                contrasts[name][metric] = _matrix_summary(difference, train_draws, eval_draws)

    return {
        "design": "complete_crossed_training_seed_by_evaluation_seed",
        "inferential_unit": "training and evaluation seed dimensions; cells are not iid",
        "bootstrap": {
            "method": "independently resample training-seed rows and evaluation-seed columns",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "methods": method_output,
        "paired_contrasts": contrasts,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Experiment-B training-seed x evaluation-seed sweep")
    p.add_argument("--train-seeds", required=True, type=parse_seeds)
    p.add_argument("--eval-seeds", required=True, type=parse_seeds)
    p.add_argument("--out", type=Path, default=Path("results/multiseed/example_b"))
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="short debug budgets; not paper metrics")
    mode.add_argument("--smoke", action="store_true", help="minimal plumbing-only budgets")
    p.add_argument("--force", action="store_true", help="rerun completed training/evaluation pairs instead of resuming")
    p.add_argument("--backend", choices=("tesseract", "jax"), default=normalize_backend(None),
                   help="component execution backend for evaluation; default: tesseract")
    args = p.parse_args()

    run_mode = "smoke" if args.smoke else "quick" if args.quick else "full"
    b = budgets(run_mode)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "training_seeds": args.train_seeds,
        "evaluation_seeds": args.eval_seeds,
        "mode": run_mode,
        "backend": args.backend,
        "budgets": b,
    }, indent=2))

    records: list[dict] = []
    for train_seed in args.train_seeds:
        print(f"\n=== training seed {train_seed} ===", flush=True)
        train_dir = out / f"train_{train_seed}"
        train_dir.mkdir(parents=True, exist_ok=True)
        model_path = train_dir / "model.npz"
        training_path = train_dir / "training.json"
        master = jax.random.PRNGKey(train_seed)
        kref, kritz, khold = jax.random.split(master, 3)

        if model_path.exists() and training_path.exists() and not args.force:
            print("  reusing completed training checkpoint", flush=True)
            model = load_model(model_path)
        else:
            ref, ref_hist, ref_train = exb.train_reference(
                kref, steps=b["ref_steps"], batch_size=b["ref_batch"],
                eval_every=max(1, min(100, b["ref_steps"])),
            )
            pot, ritz_hist, ritz_train = exb.train_ritz(
                kritz, ref,
                steps=b["ritz_steps"], n_times=b["n_times"],
                particles_per_time=b["ptime"], pool_size=b["pool"],
                refresh_every=b["refresh"],
                eval_every=max(1, min(100, b["ritz_steps"])),
                lbfgs_maxiter=0 if run_mode != "full" else 2,
                endpoint_particles=b["endpoint_particles"],
            )
            model = (ref, pot)
            save_model(model_path, model)
            ref_holdout = exb.reference_holdout(khold, ref, n=b["holdout_n"])
            training_path.write_text(json.dumps({
                "train_seed": train_seed,
                "mode": run_mode,
                "reference_training": ref_train,
                "ritz_training": ritz_train,
                "reference_holdout": ref_holdout,
                "reference_history": ref_hist,
                "ritz_history": ritz_hist,
            }, indent=2))

        for eval_seed in args.eval_seeds:
            print(f"  evaluation seed {eval_seed}", flush=True)
            eval_dir = train_dir / f"eval_{eval_seed}"
            eval_dir.mkdir(parents=True, exist_ok=True)
            result_path = eval_dir / "example_b_results.json"
            reuse = False
            if result_path.exists() and not args.force:
                payload = json.loads(result_path.read_text())
                reuse = payload.get("backend") == args.backend
            if reuse:
                print("    reusing completed evaluation", flush=True)
                summary = payload["benchmark_summary"]
            else:
                summary, per_method, target_rows, _ = exb.benchmark(
                    jax.random.PRNGKey(eval_seed), model,
                    n_particles=b["n_particles"], flow_steps=b["flow_steps"],
                    mgd_steps=b["mgd_steps"], target_bank=b["target_bank"],
                    backend=args.backend,
                )
                payload = {
                    "train_seed": train_seed,
                    "eval_seed": eval_seed,
                    "mode": run_mode,
                    "backend": args.backend,
                    "benchmark_summary": summary,
                    "benchmark_per_time": per_method,
                    "projected_target_diagnostics": target_rows,
                }
                result_path.write_text(json.dumps(payload, indent=2))
            for method, vals in summary.items():
                records.append({"train_seed": train_seed, "eval_seed": eval_seed,
                                "method": method, **vals})

    # Complete per-run table.
    with (out / "per_run.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)

    analysis = crossed_seed_analysis(records)
    agg = analysis["methods"]
    result = {
        "mode": run_mode,
        "backend": args.backend,
        "n_training_seeds": len(args.train_seeds),
        "n_evaluation_seeds": len(args.eval_seeds),
        "n_train_eval_cells": len(args.train_seeds) * len(args.eval_seeds),
        "design": analysis["design"],
        "inferential_unit": analysis["inferential_unit"],
        "bootstrap": analysis["bootstrap"],
        "methods": agg,
        "paired_contrasts": analysis["paired_contrasts"],
    }
    (out / "aggregate.json").write_text(json.dumps(result, indent=2))

    flat_rows = []
    for method, vals in sorted(agg.items()):
        row = {"method": method, "n_train_eval_cells": vals["n_train_eval_cells"]}
        for metric, stats in vals.items():
            if metric == "n_train_eval_cells":
                continue
            cells = stats["all_train_eval_cells"]
            bootstrap = stats["crossed_seed_bootstrap"]
            row[f"{metric}_mean"] = cells["mean"]
            row[f"{metric}_cell_std"] = cells["std"]
            row[f"{metric}_crossed_ci95_low"] = bootstrap["ci95_low"]
            row[f"{metric}_crossed_ci95_high"] = bootstrap["ci95_high"]
        flat_rows.append(row)
    with (out / "aggregate.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0]))
        writer.writeheader(); writer.writerows(flat_rows)

    print("\n=== crossed-seed aggregate: mean interior MMD ===")
    for method, values in sorted(agg.items()):
        stats = values["mean_interior_mmd"]["crossed_seed_bootstrap"]
        print(
            f"{method:20s} {stats['mean']:.6f} "
            f"({stats['ci95_low']:.6f}, {stats['ci95_high']:.6f})"
        )
    print("\n=== paired safe-MFSI contrasts ===")
    for name, values in analysis["paired_contrasts"].items():
        stats = values["mean_interior_mmd"]["crossed_seed_bootstrap"]
        print(
            f"{name:44s} {stats['mean']:.6f} "
            f"({stats['ci95_low']:.6f}, {stats['ci95_high']:.6f})"
        )
    print(f"\nauthoritative crossed analysis: {out / 'aggregate.json'}")


if __name__ == "__main__":
    main()
