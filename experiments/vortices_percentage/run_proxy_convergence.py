from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys

import jax
import numpy as np

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
    p.add_argument(
        "--fidelity", nargs="+",
        default=["2:5:48:24:2e-6:100", "2:7:64:32:1e-6:120"],
        help="trials:times:nx:ny:cg_tol:cg_maxiter",
    )
    p.add_argument("--source-run", type=Path, default=SCRIPT_DIR / "outputs" / "run")
    p.add_argument("--output", type=Path, default=SCRIPT_DIR / "outputs" / "proxy_convergence")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def _parse(s: str):
    x = s.split(":")
    if len(x) != 6:
        raise ValueError("expected trials:times:nx:ny:cg_tol:cg_maxiter")
    return int(x[0]), int(x[1]), int(x[2]), int(x[3]), float(x[4]), int(x[5])


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


def _center_distance(a, b) -> float:
    # Sensors are labelled during optimization. This reports the maximum matched
    # Euclidean shift; reporting-only permutation matching can be added later.
    aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.linalg.norm(aa - bb, axis=-1)))


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = load_config(SCRIPT_DIR / "config.json", smoke=False)
    rows = []
    for spec_s in args.fidelity:
        trials, times, nx, ny, tol, maxiter = _parse(spec_s)
        point = args.output / f"p{trials}_t{times}_g{nx}x{ny}"
        point.mkdir(parents=True, exist_ok=True)
        result_path = point / "result.json"
        if args.force and result_path.exists():
            result_path.unlink()
        if not result_path.exists():
            _seed(args.source_run, point)
            cfg = json.loads(json.dumps(base))
            o = cfg["optimization"]
            o["full_gradient_trials"] = trials
            o["full_gradient_time_n"] = times
            o["full_gradient_grid_nx"] = nx
            o["full_gradient_grid_ny"] = ny
            o["full_gradient_cg_tol"] = tol
            o["full_gradient_cg_maxiter"] = maxiter
            print(f"[proxy-audit] {spec_s} -> {point}", flush=True)
            result = run_experiment(cfg, point, smoke=False)
        else:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        rows.append({
            "proxy_trials": trials, "proxy_times": times, "proxy_nx": nx, "proxy_ny": ny,
            "proxy_cg_tol": tol, "proxy_cg_maxiter": maxiter,
            "full_centers": result["selection_centers"]["full"],
            "full_R_selection": result["selection_certificates"]["full"]["R_selection"],
            "full_certified": result["selection_certificates"]["full"]["certified"],
            "full_A_validation": result["validation"]["full"]["full_action"]["mean"],
            "result": str(result_path),
        })
    if rows:
        base_centers = rows[0]["full_centers"]
        base_A = float(rows[0]["full_A_validation"])
        for r in rows:
            r["max_center_shift_from_first"] = _center_distance(base_centers, r["full_centers"])
            r["full_A_relative_change_from_first"] = float(r["full_A_validation"]) / base_A - 1.0
            r["full_centers"] = json.dumps(r["full_centers"])
        with (args.output / "proxy_convergence.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        (args.output / "proxy_convergence.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"results={args.output / 'proxy_convergence.csv'}", flush=True)


if __name__ == "__main__":
    main()
