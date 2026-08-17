from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys

import jax

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
jax.config.update("jax_enable_x64", True)

from mfsi.config import load_config
from mfsi.cache import fingerprint
from experiment import run_experiment

PARETO_METHODOLOGY_VERSION = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--eps", nargs="+", type=float, default=[5e-4, 1e-3, 2e-3, 4e-3])
    p.add_argument("--source-run", type=Path, default=SCRIPT_DIR / "outputs" / "run")
    p.add_argument("--output", type=Path, default=SCRIPT_DIR / "outputs" / "pareto")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def _tag(eps: float) -> str:
    return f"epsR_{eps:.7f}".replace(".", "p").replace("-", "m")


def _link(src: Path, dst: Path) -> None:
    if not src.exists() or dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _seed(source: Path, target: Path) -> None:
    for name in (
        "truth_bank.npz", "reference_endpoints.npz", "reference.npz", "reference_bank.npz",
        "selection_bank.npz", "validation_bank.npz",
    ):
        _link(source / name, target / name)


def _best_archived_law_seed(source: Path, output: Path) -> list[float] | None:
    """Use old runs only as candidate generators; every seed is re-audited."""
    paths = [source / "result.json", *sorted(output.glob("epsR_*/result.json"))]
    candidates: list[tuple[float, list[float], Path]] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            l_max = float(result["law_screens"]["L_max"])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        for stage in ("law", "tangent", "full"):
            for row in result.get("selection_audit", {}).get(stage, []):
                try:
                    exact_l = float(row["exact_L"])
                    exact_r = float(row["exact_R"])
                    eta = [float(value) for value in row["eta"]]
                except (KeyError, TypeError, ValueError):
                    continue
                if row.get("valid") and exact_l <= l_max + 1.0e-12 and math.isfinite(exact_r):
                    candidates.append((exact_r, eta, path))
    if not candidates:
        return None
    risk, eta, path = min(candidates, key=lambda item: item[0])
    print(
        f"[pareto] seeding Law-anchor refinement with archived R={risk:.8g} "
        f"from {path}",
        flush=True,
    )
    return eta


def _audited_full_action(result: dict, selected: list[float]) -> float:
    for row in result.get("selection_audit", {}).get("full", []):
        if row.get("eta") == selected and row.get("valid"):
            value = float(row["objective"])
            if math.isfinite(value):
                return value
    raise RuntimeError("design is missing its exact Full-action audit row")


def _save(rows: list[dict], output: Path) -> None:
    with (output / "pareto.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "pareto.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    try:
        from visualize_pareto import save_pareto_figure

        save_pareto_figure(rows, output / "pareto.png")
    except Exception as exc:
        print(f"[pareto] plot skipped: {exc}", flush=True)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = load_config(SCRIPT_DIR / "config.json", smoke=False)
    archive_seed = _best_archived_law_seed(args.source_run, args.output)
    shared_anchor = None
    incumbent_full_eta = None
    incumbent_full_action = math.inf
    rows = []
    for eps in sorted(set(float(x) for x in args.eps)):
        if eps < 0.0:
            raise ValueError("epsilon_R must be nonnegative")
        point = args.output / _tag(eps)
        point.mkdir(parents=True, exist_ok=True)
        result_path = point / "result.json"
        cfg = json.loads(json.dumps(base))
        cfg["law"]["epsilon_r"] = eps
        cfg["optimization"]["pareto_methodology_version"] = PARETO_METHODOLOGY_VERSION
        if shared_anchor is not None:
            cfg["optimization"]["fixed_law_anchor"] = shared_anchor
        elif archive_seed is not None:
            cfg["optimization"]["law_anchor_seed_eta"] = archive_seed
        if incumbent_full_eta is not None:
            cfg["optimization"]["pareto_incumbent_full_eta"] = incumbent_full_eta
        cached = None
        if result_path.exists() and not args.force:
            candidate = json.loads(result_path.read_text(encoding="utf-8"))
            if candidate.get("config_hash") == fingerprint(cfg):
                cached = candidate
            else:
                print(f"[pareto] stale config -> rerunning {point}", flush=True)
        if cached is None:
            _seed(args.source_run, point)
            print(f"[pareto] epsilon_R={eps:g} -> {point}", flush=True)
            result = run_experiment(cfg, point, smoke=False)
        else:
            print(f"[pareto] reusing compatible {result_path}", flush=True)
            result = cached
        current_anchor = {
            "eta": result["selection"]["law_optimum"],
            "R_star": float(result["law_screens"]["R_star"]),
        }
        if shared_anchor is None:
            shared_anchor = current_anchor
            print(
                f"[pareto] froze common exact Law anchor "
                f"R*={shared_anchor['R_star']:.8g}",
                flush=True,
            )
        elif abs(current_anchor["R_star"] - shared_anchor["R_star"]) > float(
            cfg["optimization"].get("law_anchor_consistency_tol", 1.0e-10)
        ):
            raise RuntimeError(
                "Pareto points do not share one Law anchor: "
                f"expected {shared_anchor['R_star']:.12g}, "
                f"got {current_anchor['R_star']:.12g} at epsilon_R={eps:g}"
            )
        full = result["selection_certificates"]["full"]
        if float(full["R_excess_from_star"]) < -float(
            cfg["optimization"].get("law_anchor_consistency_tol", 1.0e-10)
        ):
            raise RuntimeError(
                f"invalid Pareto point at epsilon_R={eps:g}: a Full design "
                "strictly beats the frozen Law-risk anchor"
            )
        full_action_selection = _audited_full_action(
            result, result["selection"]["full_optimum"]
        )
        law_action_selection = _audited_full_action(
            result, result["selection"]["law_optimum"]
        )
        action_tol = 1.0e-10 * max(
            1.0, abs(incumbent_full_action) if math.isfinite(incumbent_full_action) else 1.0
        )
        if (
            math.isfinite(incumbent_full_action)
            and full_action_selection > incumbent_full_action + action_tol
        ):
            raise RuntimeError(
                f"non-nested Pareto action at epsilon_R={eps:g}: exact action "
                f"increased from {incumbent_full_action:.12g} to "
                f"{full_action_selection:.12g}"
            )
        if full_action_selection < incumbent_full_action:
            incumbent_full_action = full_action_selection
            incumbent_full_eta = result["selection"]["full_optimum"]
        val = result["validation"]
        law_validation = val["law"]
        full_validation = val["full"]
        contrast = result.get("contrasts", {}).get("full_vs_law_full_action_reduction", {})
        rows.append({
            "epsilon_r": eps,
            "R_star": result["law_screens"]["R_star"],
            "R_max": result["law_screens"]["R_max"],
            "full_R_selection": full["R_selection"],
            "full_R_excess_selection": full["R_excess_from_star"],
            "full_A_selection": full_action_selection,
            "law_A_selection": law_action_selection,
            "full_certified": full["certified"],
            "anchor_refinement_passes": result.get("law_anchor", {}).get(
                "anchor_refinement_passes", 0
            ),
            "law_R_validation": law_validation["law_risk"]["mean"],
            "law_R_validation_se": law_validation["law_risk"]["se"],
            "law_A_validation": law_validation["full_action"]["mean"],
            "law_A_validation_se": law_validation["full_action"]["se"],
            "law_valid_fraction": law_validation["valid_fraction"],
            "full_R_validation": full_validation["law_risk"]["mean"],
            "full_R_validation_se": full_validation["law_risk"]["se"],
            "full_A_validation": full_validation["full_action"]["mean"],
            "full_A_validation_se": full_validation["full_action"]["se"],
            "full_valid_fraction": full_validation["valid_fraction"],
            "validation_action_reduction": contrast.get("ratio_of_means_reduction"),
            "full_centers": json.dumps(result["selection_centers"]["full"]),
            "result": str(result_path),
        })
        _save(rows, args.output)
    print(f"results={args.output / 'pareto.csv'}", flush=True)
    print(f"plot={args.output / 'pareto.png'}", flush=True)


if __name__ == "__main__":
    main()
