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
    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_method[row["method"]].append(row)
    metrics = [
        k for k, v in records[0].items()
        if k not in {"train_seed", "eval_seed", "method"} and isinstance(v, (int, float))
    ]
    out = {}
    for method, rows in sorted(by_method.items()):
        entry = {"n_runs": len(rows)}
        for metric in metrics:
            x = np.asarray([float(r[metric]) for r in rows], dtype=float)
            entry[metric] = {
                "mean": float(x.mean()),
                "std": float(x.std(ddof=1) if len(x) > 1 else 0.0),
                "min": float(x.min()),
                "max": float(x.max()),
            }
        out[method] = entry
    return out


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

    agg = aggregate(records)
    result = {
        "mode": run_mode,
        "backend": args.backend,
        "n_training_seeds": len(args.train_seeds),
        "n_evaluation_seeds": len(args.eval_seeds),
        "n_train_eval_pairs": len(args.train_seeds) * len(args.eval_seeds),
        "methods": agg,
    }
    (out / "aggregate.json").write_text(json.dumps(result, indent=2))

    flat_rows = []
    for method, vals in sorted(agg.items()):
        row = {"method": method, "n_runs": vals["n_runs"]}
        for metric, stats in vals.items():
            if metric == "n_runs":
                continue
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
        flat_rows.append(row)
    with (out / "aggregate.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0]))
        writer.writeheader(); writer.writerows(flat_rows)

    print("\n=== aggregate ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
