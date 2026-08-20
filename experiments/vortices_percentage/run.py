from __future__ import annotations

import argparse
from pathlib import Path
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

CONFIG_PATH = SCRIPT_DIR / "config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--reference-seed",
        type=int,
        help="override only the learned reference-flow training seed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write an isolated run instead of outputs/run or outputs/smoke",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    if args.reference_seed is not None:
        cfg.setdefault("reference_training", {})["seed"] = int(
            args.reference_seed
        )
    mode = "smoke" if args.smoke else "run"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else SCRIPT_DIR / "outputs" / mode
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "vortices_double_gyre "
        f"smoke={args.smoke} "
        f"reference_seed={cfg['reference_training'].get('seed', cfg['seed'])} "
        f"output={output_dir}",
        flush=True,
    )
    payload = run_experiment(cfg, output_dir, smoke=args.smoke)
    if payload.get("smoke"):
        print("[smoke] complete", flush=True)
        print(f"  centers={payload['smoke_centers']}", flush=True)
        m = payload["smoke_metrics"]
        for key in ("law_risk", "tangent_action", "full_action", "max_calibration_residual", "min_ess_fraction", "max_poisson_relative_residual"):
            print(f"  {key}={m.get(key)}", flush=True)
        print(f"  valid={m.get('valid')}", flush=True)
    else:
        print("[run] complete", flush=True)
        print(f"  population_centers={payload['selection_centers']['population']}", flush=True)
        print(f"  law_centers={payload['selection_centers']['law']}", flush=True)
        print(f"  tangent_centers={payload['selection_centers']['tangent']}", flush=True)
        print(f"  full_centers={payload['selection_centers']['full']}", flush=True)
        s = payload["law_screens"]
        print(f"  L*={s['L_star']:.8g} L_max={s['L_max']:.8g}", flush=True)
        print(f"  R*={s['R_star']:.8g} R_max={s['R_max']:.8g}", flush=True)
        print(
            "  max_relative_risk_violation="
            f"{100.0 * cfg['law']['max_relative_risk_violation']:.4g}%",
            flush=True,
        )
    print(f"results={output_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
