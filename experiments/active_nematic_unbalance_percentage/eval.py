"""Saved-result evaluator for two-species finite-measure action and risk."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "outputs" / "run" / "manifest.json"
DESIGNS = ("law", "tangent", "unbalanced_full")
LABELS = {"law": "Law", "tangent": "Tangent", "unbalanced_full": "Full"}
COLORS = {"law": "#2878B5", "tangent": "#E29D26", "unbalanced_full": "#D1495B"}


def _metric(summary, name):
    row = summary["metrics"][name]
    return f"{row['mean']:.7g} +/- {row['se']:.3g} (SE)"


def evaluate(path: Path) -> int:
    result = json.loads(path.read_text())
    validation = result["validation"]
    metrics = validation["metrics"]
    print("=" * 88)
    print("ACTIVE NEMATIC — TWO-SPECIES UNBALANCED MFSI")
    print("=" * 88)
    print(f"file: {path}")
    print(f"reference seeds: base={result['reference_seed']} plus={result['plus_reference_seed']} minus={result['minus_reference_seed']}")
    print(f"physical interval: {result['physical_interval']}")
    print(f"reaction kappa: {result['reaction_kappa']}")
    print(f"valid trials: {validation['valid_trials']}/{validation['trials']}")
    print("\nFinite-measure risk")
    for name in ("law_risk_total", "law_risk_plus", "law_risk_minus", "shape_mmd_plus", "shape_mmd_minus", "mass_error_plus", "mass_error_minus"):
        print(f"  {name:<34} {_metric(validation, name)}")
    print("\nUnbalanced Full action")
    for name in (
        "full_unbalanced_action_total", "full_unbalanced_action_plus", "full_unbalanced_action_minus",
        "move_action_plus", "reaction_action_plus", "move_action_minus", "reaction_action_minus",
        "reaction_fraction_plus", "reaction_fraction_minus", "reaction_fraction_total",
    ):
        print(f"  {name:<34} {_metric(validation, name)}")
    print("\nNumerical diagnostics")
    for name in ("max_calibration_residual", "min_ess_fraction", "max_screened_pde_relative_residual"):
        print(f"  {name:<34} {_metric(validation, name)}")
    balance = result["charge_balance_diagnostics"]
    print(f"\ncharge balance: {'PASS' if balance['passed'] else 'FAIL'}; max violation={balance['maximum_violation']:.7g}")
    decomposed = (
        result.get("species_weight_plus", 1.0)
        * (metrics["move_action_plus"]["mean"] + metrics["reaction_action_plus"]["mean"])
        + result.get("species_weight_minus", 1.0)
        * (metrics["move_action_minus"]["mean"] + metrics["reaction_action_minus"]["mean"])
    )
    total = metrics["full_unbalanced_action_total"]["mean"]
    failures = []
    if validation["valid_trials"] != validation["trials"]:
        failures.append("not every validation trial is valid")
    if not balance["passed"]:
        failures.append("charge-balance guard failed")
    if not np.isclose(decomposed, total, rtol=2.0e-8, atol=1.0e-10):
        failures.append("species move/reaction decomposition does not equal total")
    if failures:
        print("\nFAILURES")
        for failure in failures:
            print(f"  - {failure}")
        return 2
    print("\nSaved result passes declared structural checks.")
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _result_paths(input_path: Path) -> list[Path]:
    payload = _read_json(input_path)
    if "runs" not in payload:
        return [input_path]
    paths = []
    for row in payload["runs"]:
        candidate = Path(row["result"])
        if not candidate.is_file():
            candidate = input_path.parent / candidate.parent.name / candidate.name
        if not candidate.is_file():
            raise FileNotFoundError(f"manifest result does not exist: {row['result']}")
        paths.append(candidate.resolve())
    return paths


def _mean_se(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "se": float(np.std(array, ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "n": int(len(array)),
    }


def _summary_values(result: dict[str, Any], design: str) -> dict[str, float]:
    summary = result["validation_designs"][design]["summary"]
    metrics = summary["metrics"]
    move = metrics["move_action_plus"]["mean"] + metrics["move_action_minus"]["mean"]
    reaction = metrics["reaction_action_plus"]["mean"] + metrics["reaction_action_minus"]["mean"]
    return {
        "law_risk": float(metrics["law_risk_total"]["mean"]),
        "law_risk_se": float(metrics["law_risk_total"]["se"]),
        "full_action": float(metrics["full_unbalanced_action_total"]["mean"]),
        "full_action_se": float(metrics["full_unbalanced_action_total"]["se"]),
        "move_action": float(move),
        "reaction_action": float(reaction),
        "reaction_fraction": float(metrics["reaction_fraction_total"]["mean"]),
        "valid_trials": int(summary["valid_trials"]),
        "trials": int(summary["trials"]),
    }


def _selected_full_audit(result: dict[str, Any], design: str) -> dict[str, Any] | None:
    eta = np.asarray(result["designs"][design], dtype=np.float64)
    rows = result["selection_candidates"]["full"]
    matches = [
        row for row in rows
        if np.allclose(np.asarray(row["eta"], dtype=np.float64), eta, rtol=0.0, atol=1.0e-8)
    ]
    if not matches:
        return None
    return min(matches, key=lambda row: float(row["audit"]["value"]))


def build_aggregate_statistics(
    results: list[dict[str, Any]], paths: list[Path]
) -> dict[str, Any]:
    rows = []
    for result, path in zip(results, paths, strict=True):
        designs = {name: _summary_values(result, name) for name in DESIGNS}
        law, full = designs["law"], designs["unbalanced_full"]
        law_selection = _selected_full_audit(result, "law")
        full_selection = _selected_full_audit(result, "unbalanced_full")
        selection_reduction = (
            1.0
            - float(full_selection["audit"]["value"])
            / float(law_selection["audit"]["value"])
            if law_selection is not None and full_selection is not None
            else None
        )
        reductions = {
            "selection_full_action": selection_reduction,
            "validation_full_action": 1.0 - full["full_action"] / law["full_action"],
            "validation_move_action": 1.0 - full["move_action"] / law["move_action"],
            "validation_reaction_action": 1.0 - full["reaction_action"] / law["reaction_action"],
        }
        rows.append({
            "reference_seed": int(result["reference_seed"]),
            "result": str(path),
            "risk_star": float(result["risk_star"]),
            "risk_max": float(result["risk_max"]),
            "selection_certified": bool(result["selection_certified"]),
            "designs": designs,
            "selection": {
                "law_full_action": (
                    float(law_selection["audit"]["value"])
                    if law_selection is not None else None
                ),
                "full_full_action": (
                    float(full_selection["audit"]["value"])
                    if full_selection is not None else None
                ),
                "law_risk": (
                    float(law_selection["law_screen"]["value"])
                    if law_selection is not None else None
                ),
                "full_risk": (
                    float(full_selection["law_screen"]["value"])
                    if full_selection is not None else None
                ),
            },
            "reductions": reductions,
        })
    aggregate = {
        key: _mean_se([
            row["reductions"][key]
            for row in rows
            if row["reductions"][key] is not None
        ])
        for key in rows[0]["reductions"]
    }
    return {
        "schema_version": 1,
        "experiment": "active_nematic_unbalance_percentage_evaluation",
        "rows": rows,
        "aggregate_reductions": aggregate,
        "all_selection_certified": all(row["selection_certified"] for row in rows),
        "all_validation_trials_valid": all(
            design["valid_trials"] == design["trials"]
            for row in rows for design in row["designs"].values()
        ),
    }


def print_aggregate(stats: dict[str, Any]) -> None:
    print("\n" + "=" * 88)
    print("THREE-REFERENCE DESIGN COMPARISON")
    print("=" * 88)
    for row in stats["rows"]:
        law = row["designs"]["law"]
        tangent = row["designs"]["tangent"]
        full = row["designs"]["unbalanced_full"]
        red = row["reductions"]
        selection_text = (
            f"{100 * red['selection_full_action']:.2f}%"
            if red["selection_full_action"] is not None
            else "n/a"
        )
        print(
            f"seed {row['reference_seed']}: "
            f"A Law/Tangent/Full={law['full_action']:.6g}/"
            f"{tangent['full_action']:.6g}/{full['full_action']:.6g}; "
            f"selection reduction={selection_text}; "
            f"validation reduction={100*red['validation_full_action']:.2f}%"
        )
    print("\nAcross-reference reductions (mean +/- SE across reference seeds)")
    for key, label in (
        ("selection_full_action", "selection Full action"),
        ("validation_full_action", "validation Full action"),
        ("validation_move_action", "validation move action"),
        ("validation_reaction_action", "validation reaction action"),
    ):
        row = stats["aggregate_reductions"][key]
        if row is None:
            print(f"  {label:<30} n/a (not evaluated by fast selection)")
            continue
        print(
            f"  {label:<30} {100*row['mean']:.2f}% +/- {100*row['se']:.2f}% "
            f"(range {100*row['min']:.2f}% to {100*row['max']:.2f}%)"
        )


def save_figure(stats: dict[str, Any], output: Path, *, show: bool = False) -> None:
    import matplotlib.pyplot as plt

    rows = stats["rows"]
    seeds = [str(row["reference_seed"])[-2:] for row in rows]
    x = np.arange(len(rows), dtype=np.float64)
    width = 0.24
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), constrained_layout=True)

    ax = axes[0, 0]
    for offset, design in zip((-width, 0.0, width), DESIGNS, strict=True):
        values = [row["designs"][design]["full_action"] for row in rows]
        errors = [row["designs"][design]["full_action_se"] for row in rows]
        ax.bar(x + offset, values, width, yerr=errors, capsize=3, label=LABELS[design], color=COLORS[design])
    ax.set_xticks(x, seeds)
    ax.set_xlabel("Reference seed suffix")
    ax.set_ylabel("Validation Full action")
    ax.set_title("Independent validation")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    selection = 100 * np.asarray([
        row["reductions"]["selection_full_action"]
        if row["reductions"]["selection_full_action"] is not None
        else np.nan
        for row in rows
    ])
    validation = 100 * np.asarray([row["reductions"]["validation_full_action"] for row in rows])
    if np.any(np.isfinite(selection)):
        ax.bar(x - width / 2, selection, width, label="Selection", color="#6B8E23")
        ax.bar(x + width / 2, validation, width, label="Validation", color="#8B5A9F")
        ax.legend(frameon=False)
    else:
        ax.bar(x, validation, width, color="#8B5A9F")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, seeds)
    ax.set_ylabel("Full-vs-Law reduction (%)")
    ax.set_title("Independent validation reduction")

    ax = axes[1, 0]
    move = 100 * np.asarray([row["reductions"]["validation_move_action"] for row in rows])
    reaction = 100 * np.asarray([row["reductions"]["validation_reaction_action"] for row in rows])
    ax.bar(x - width / 2, move, width, label="Move", color="#4C78A8")
    ax.bar(x + width / 2, reaction, width, label="Reaction", color="#F58518")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, seeds)
    ax.set_xlabel("Reference seed suffix")
    ax.set_ylabel("Validation reduction (%)")
    ax.set_title("Action decomposition")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for row in rows:
        risks = [row["designs"][design]["law_risk"] for design in DESIGNS]
        actions = [row["designs"][design]["full_action"] for design in DESIGNS]
        ax.plot(risks, actions, color="#777777", alpha=0.55, linewidth=1.0)
        for design, risk, action in zip(DESIGNS, risks, actions, strict=True):
            ax.scatter(risk, action, color=COLORS[design], s=42, label=LABELS[design] if row is rows[0] else None)
    ax.set_xlabel("Validation finite-measure risk")
    ax.set_ylabel("Validation Full action")
    ax.set_title("Risk/action geometry")
    ax.legend(frameon=False)

    fig.suptitle("Percentage-budget unbalanced active-nematic MFSI", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    paths = _result_paths(args.result)
    status = max(evaluate(path) for path in paths)
    results = [_read_json(path) for path in paths]
    if not all("validation_designs" in result for result in results):
        return status
    stats = build_aggregate_statistics(results, paths)
    print_aggregate(stats)
    output_dir = args.output_dir or args.result.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "evaluation_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nstats: {stats_path}")
    if not args.no_figure:
        figure_path = output_dir / "evaluation.png"
        save_figure(stats, figure_path, show=args.show)
        print(f"figure: {figure_path}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
