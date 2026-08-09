#!/usr/bin/env python3
"""Print compact terminal tables for MFSI Experiment A/B outputs.

The script is dependency-free (stdlib only).  For Experiment B multi-seed
results it prefers the live backend-specific sweep at

    results/multiseed/example_b/<backend>/aggregate.json

and understands the uncertainty-aware schema produced by sweep_example_b.py.
Legacy CSV/reference snapshots are supported only as fallbacks and are labelled
explicitly so they cannot be mistaken for a newly completed sweep.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable


def fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "PASS" if v else "FAIL"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            return str(v)
        av = abs(v)
        if av == 0:
            return "0"
        if av < 1e-3 or av >= 1e4:
            return f"{v:.3e}"
        return f"{v:.5f}"
    return str(v)


def fmt_ci(lo: Any, hi: Any) -> str:
    if lo is None or hi is None:
        return "-"
    return f"[{fmt(float(lo))}, {fmt(float(hi))}]"


def table(title: str, headers: list[str], rows: Iterable[Iterable[Any]]) -> None:
    rows_s = [[fmt(x) for x in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in rows_s:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    total = sum(widths) + 3 * (len(widths) - 1)
    print("\n" + title)
    print("=" * max(len(title), total))
    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("-+-".join("-" * w for w in widths))
    for row in rows_s:
        print(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def print_experiment_a(root: Path) -> None:
    csv_path = root / "method_benchmark_summary.csv"
    val_path = root / "learned_validation.json"

    if csv_path.exists():
        rows = read_csv(csv_path)
        table(
            "Experiment A — matched law-path benchmark",
            [
                "method", "mean W1", "max W1", "mean W2", "mean KS",
                "max moment-2 err", "4th-moment RMSE", "endpoint W1",
            ],
            [
                [
                    r["method"], f(r, "mean_interior_w1"), f(r, "max_interior_w1"),
                    f(r, "mean_interior_w2"), f(r, "mean_interior_ks"),
                    f(r, "max_second_moment_error"), f(r, "rmse_fourth_moment_vs_target"),
                    f(r, "endpoint_w1_t1"),
                ]
                for r in rows
            ],
        )
    else:
        print(f"\n[Experiment A] missing {csv_path}")

    if val_path.exists():
        d = load_json(val_path)
        safe = d.get("safe_learned_flow", {}).get("summary", {})
        ref = d.get("reference_validation", {})
        vjp = d.get("particle_implicit_vjp", {})
        criteria = d.get("criteria", {})
        table(
            "Experiment A — learned-pipeline diagnostics",
            ["diagnostic", "value"],
            [
                ["reference FM MSE", ref.get("fm_mse")],
                ["FM / zero-predictor MSE", ref.get("fm_mse_ratio_to_zero")],
                ["implicit VJP rel. error", vjp.get("relative_error")],
                ["max calibration residual", safe.get("max_calibration_residual")],
                ["min ESS fraction", safe.get("min_ess_fraction")],
                ["min covariance rank", safe.get("min_covariance_rank")],
                ["max covariance condition", safe.get("max_covariance_condition")],
                ["median weak residual", safe.get("median_weak_form_residual")],
                ["max weak residual", safe.get("max_weak_form_residual")],
                ["median projected MMD", safe.get("median_projected_mmd")],
                ["max projected MMD", safe.get("max_projected_mmd")],
                ["max generated moment error", safe.get("max_generated_moment_error")],
                ["all criteria", d.get("all_passed")],
            ],
        )
        if criteria:
            table(
                "Experiment A — validation gates",
                ["criterion", "status"],
                [[k, v] for k, v in criteria.items()],
            )


def print_experiment_b(root: Path) -> None:
    bdir = root / "example_b"
    csv_path = bdir / "benchmark_summary.csv"
    json_path = bdir / "example_b_results.json"

    if csv_path.exists():
        rows = read_csv(csv_path)
        table(
            "Experiment B — projected-law benchmark",
            ["method", "mean MMD", "max MMD", "max moment err", "angular err", "endpoint MMD", "runtime (s)"],
            [
                [
                    r["method"], f(r, "mean_interior_mmd"), f(r, "max_interior_mmd"),
                    f(r, "max_moment_error"), f(r, "mean_interior_angular_error"),
                    f(r, "endpoint_t1_mmd"), f(r, "runtime_s"),
                ]
                for r in rows
            ],
        )
    else:
        print(f"\n[Experiment B] missing {csv_path}")

    if json_path.exists():
        d = load_json(json_path)
        hold = d.get("projection_and_ritz_holdout", [])
        ref = d.get("reference_holdout", {})
        parity = d.get("tesseract_kernel_parity", {})
        if hold:
            table(
                "Experiment B — held-out projection / Deep-Ritz diagnostics",
                ["t", "calib residual", "ESS", "rank", "condition", "weak residual"],
                [
                    [
                        x.get("t"), x.get("calibration_residual"), x.get("ess_fraction"),
                        x.get("rank"), x.get("condition"), x.get("weak_form_residual"),
                    ]
                    for x in hold
                ],
            )
            table(
                "Experiment B — diagnostic summary",
                ["diagnostic", "value"],
                [
                    ["reference FM MSE", ref.get("fm_mse")],
                    ["FM / zero-predictor MSE", ref.get("ratio")],
                    ["max calibration residual", max(x["calibration_residual"] for x in hold)],
                    ["min ESS fraction", min(x["ess_fraction"] for x in hold)],
                    ["min covariance rank", min(x["rank"] for x in hold)],
                    ["max covariance condition", max(x["condition"] for x in hold)],
                    ["median weak residual", median(x["weak_form_residual"] for x in hold)],
                    ["max weak residual", max(x["weak_form_residual"] for x in hold)],
                    ["max Tesseract-kernel parity error", max(parity.values()) if parity else None],
                ],
            )


def live_multiseed_dir(root: Path, backend: str) -> Path:
    return root / "multiseed" / "example_b" / backend


def _metric_stats(method_data: dict[str, Any], metric: str, unit: str) -> dict[str, Any]:
    metric_data = method_data.get(metric, {})
    out = metric_data.get(unit, {}) if isinstance(metric_data, dict) else {}
    return out if isinstance(out, dict) else {}


def _cell_stats(method_data: dict[str, Any], metric: str) -> dict[str, Any]:
    return (
        _metric_stats(method_data, metric, "all_train_eval_cells")
        or _metric_stats(method_data, metric, "all_train_eval_pairs")
    )


def _stats(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    if not values:
        return {"n": 0, "mean": math.nan, "std": math.nan, "se": math.nan,
                "ci95_low": math.nan, "ci95_high": math.nan}
    mean = sum(values) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    se = std / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "se": se,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
    }


def paired_training_seed_stats(per_run_path: Path) -> dict[str, dict[str, float | int]]:
    """Legacy fallback for paired MFSI-safe minus baseline MMD.

    New sweeps persist a crossed-seed bootstrap and never use this calculation.
    It is retained only so older partial artifacts remain printable.
    """
    if not per_run_path.exists():
        return {}

    rows = read_csv(per_run_path)
    required = {"train_seed", "eval_seed", "method", "mean_interior_mmd"}
    if not rows or not required.issubset(rows[0]):
        return {}

    by_train_method: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        by_train_method[(int(row["train_seed"]), row["method"])].append(float(row["mean_interior_mmd"]))

    train_seeds = sorted({k[0] for k in by_train_method})
    out: dict[str, dict[str, float | int]] = {}
    for baseline in ("moment_tangent", "mgd_style", "raw_si"):
        diffs: list[float] = []
        for train_seed in train_seeds:
            a = by_train_method.get((train_seed, "mfsi_learned_safe"))
            b = by_train_method.get((train_seed, baseline))
            if a and b:
                diffs.append(sum(a) / len(a) - sum(b) / len(b))
        if diffs:
            out[f"mfsi_learned_safe_minus_{baseline}"] = _stats(diffs)
    return out


def print_live_multiseed(root: Path, backend: str) -> bool:
    sweep_dir = live_multiseed_dir(root, backend)
    aggregate_json = sweep_dir / "aggregate.json"
    aggregate_csv = sweep_dir / "aggregate.csv"

    if not aggregate_json.exists():
        return False

    d = load_json(aggregate_json)
    methods = d.get("methods", {})
    if not isinstance(methods, dict) or not methods:
        print(f"\n[Multi-seed] malformed or empty {aggregate_json}")
        return True

    table(
        f"Experiment B — live multi-seed sweep ({aggregate_json.relative_to(root)})",
        ["field", "value"],
        [
            ["backend", d.get("backend")],
            ["mode", d.get("mode")],
            ["training seeds", d.get("n_training_seeds")],
            ["evaluation seeds", d.get("n_evaluation_seeds")],
            ["train/eval cells (descriptive)", d.get("n_train_eval_cells", d.get("n_train_eval_pairs"))],
            ["design", d.get("design")],
        ],
    )

    rows = []
    for method, md in sorted(methods.items()):
        mmd_all = _cell_stats(md, "mean_interior_mmd")
        mmd_train = _metric_stats(md, "mean_interior_mmd", "training_seed_means")
        mmd_eval = _metric_stats(md, "mean_interior_mmd", "evaluation_seed_means")
        mmd_crossed = _metric_stats(md, "mean_interior_mmd", "crossed_seed_bootstrap")
        max_mmd = _cell_stats(md, "max_interior_mmd")
        angular = _cell_stats(md, "mean_interior_angular_error")
        moment = _cell_stats(md, "max_moment_error")
        rows.append([
            method,
            mmd_all.get("mean"),
            fmt_ci(mmd_crossed.get("ci95_low"), mmd_crossed.get("ci95_high")),
            mmd_train.get("std"),
            mmd_eval.get("std"),
            mmd_all.get("std"),
            max_mmd.get("mean"),
            angular.get("mean"),
            moment.get("mean"),
        ])

    table(
        "Experiment B — crossed training/evaluation-seed aggregate",
        [
            "method", "mean MMD", "crossed 95% CI", "train std", "eval std",
            "cell std", "mean max MMD", "angular err", "mean max moment err",
        ],
        rows,
    )

    paired = d.get("paired_contrasts", {})
    persisted_crossed = isinstance(paired, dict) and bool(paired)
    if not persisted_crossed:
        paired = paired_training_seed_stats(sweep_dir / "per_run.csv")
    if paired:
        paired_rows = []
        for name, contrast_data in paired.items():
            st = (
                _metric_stats(contrast_data, "mean_interior_mmd", "crossed_seed_bootstrap")
                if persisted_crossed else contrast_data
            )
            baseline = name.removeprefix("mfsi_learned_safe_minus_")
            paired_rows.append([
                f"safe MFSI - {baseline}",
                f"{d.get('n_training_seeds')}×{d.get('n_evaluation_seeds')}",
                st.get("mean"), st.get("bootstrap_se", st.get("std")),
                fmt_ci(st.get("ci95_low"), st.get("ci95_high")),
            ])
        table(
            "Experiment B — paired MMD differences, crossed-seed bootstrap",
            ["comparison", "seed design", "mean ΔMMD", "bootstrap SE", "95% CI"],
            paired_rows,
        )
        print("  Negative ΔMMD favors safe MFSI.")

    if aggregate_csv.exists():
        print(f"\nLive aggregate CSV : {aggregate_csv}")
    print(f"Live aggregate JSON: {aggregate_json}")
    return True


def find_legacy_multiseed(root: Path, backend: str) -> tuple[Path | None, bool]:
    candidates = [
        root / "multiseed" / "example_b" / backend / "multiseed_summary.csv",
        root / "multiseed" / "example_b" / "summary.csv",
        root / "multiseed" / "example_b" / "multiseed_summary.csv",
    ]
    for path in candidates:
        if path.exists():
            return path, False

    reference = root / "reference" / "example_b" / "multiseed_summary.csv"
    if reference.exists():
        return reference, True
    return None, False


def print_legacy_multiseed(root: Path, backend: str) -> None:
    path, is_reference = find_legacy_multiseed(root, backend)
    if path is None:
        print(f"\n[Multi-seed] no live sweep found for backend '{backend}'.")
        return

    if is_reference:
        print(
            "\nWARNING: no live backend-specific aggregate.json was found; "
            "showing the frozen reference snapshot instead."
        )

    rows = read_csv(path)
    required = {"method", "mean_interior_mmd_mean", "mean_interior_mmd_std"}
    if not rows or not required.issubset(rows[0]):
        print(f"\n[Multi-seed] unsupported legacy schema: {path}")
        return

    table(
        f"Experiment B — legacy multi-seed aggregate ({path.relative_to(root)})",
        ["method", "mean MMD", "std MMD", "mean max MMD", "angular err", "mean max moment err"],
        [
            [
                r["method"], f(r, "mean_interior_mmd_mean"), f(r, "mean_interior_mmd_std"),
                f(r, "mean_max_interior_mmd"), f(r, "mean_interior_angular_error"),
                f(r, "mean_max_moment_error"),
            ]
            for r in rows
        ],
    )


def print_multiseed(root: Path, backend: str) -> None:
    if not print_live_multiseed(root, backend):
        print_legacy_multiseed(root, backend)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, default=Path("results"))
    ap.add_argument("--backend", choices=("tesseract", "jax"), default="tesseract")
    ap.add_argument("--no-multiseed", action="store_true")
    args = ap.parse_args()
    root = args.results_root.resolve()

    print("\nMFSI RESULTS REPORT")
    print("===================")
    print(f"results: {root}")
    print(f"requested backend: {args.backend}")

    print_experiment_a(root)
    print_experiment_b(root)
    if not args.no_multiseed:
        print_multiseed(root, args.backend)


if __name__ == "__main__":
    main()
