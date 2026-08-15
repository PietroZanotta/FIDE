from __future__ import annotations

import argparse
import csv
import json
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
from experiment import run_experiment


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


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = load_config(SCRIPT_DIR / "config.json", smoke=False)
    rows = []
    for eps in sorted(set(float(x) for x in args.eps)):
        if eps < 0.0:
            raise ValueError("epsilon_R must be nonnegative")
        point = args.output / _tag(eps)
        point.mkdir(parents=True, exist_ok=True)
        result_path = point / "result.json"
        if args.force and result_path.exists():
            result_path.unlink()
        if not result_path.exists():
            _seed(args.source_run, point)
            cfg = json.loads(json.dumps(base))
            cfg["law"]["epsilon_r"] = eps
            print(f"[pareto] epsilon_R={eps:g} -> {point}", flush=True)
            result = run_experiment(cfg, point, smoke=False)
        else:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        full = result["selection_certificates"]["full"]
        val = result["validation"]
        contrast = result.get("contrasts", {}).get("full_vs_law_full_action_reduction", {})
        rows.append({
            "epsilon_r": eps,
            "R_star": result["law_screens"]["R_star"],
            "R_max": result["law_screens"]["R_max"],
            "full_R_selection": full["R_selection"],
            "full_R_excess_selection": full["R_excess_from_star"],
            "full_certified": full["certified"],
            "law_A_validation": val["law"]["full_action"]["mean"],
            "full_A_validation": val["full"]["full_action"]["mean"],
            "validation_action_reduction": contrast.get("ratio_of_means_reduction"),
            "full_centers": json.dumps(result["selection_centers"]["full"]),
            "result": str(result_path),
        })
        with (args.output / "pareto.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        (args.output / "pareto.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"results={args.output / 'pareto.csv'}", flush=True)


if __name__ == "__main__":
    main()
