from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mfsi.config import load_config
from experiment import run_experiment

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(CONFIG_PATH, smoke=args.smoke)

    mode = "smoke" if args.smoke else "run"
    output_dir = SCRIPT_DIR / "outputs" / mode
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"toy_example smoke={args.smoke} output={output_dir}", flush=True)
    payload = run_experiment(cfg, output_dir, smoke=args.smoke)

    if payload.get("smoke", False):
        eta = payload["smoke_design_deg"]
        metrics = payload["smoke_metrics"]
        print("[smoke] complete", flush=True)
        print(f"  eta_deg={eta}", flush=True)
        print(f"  law_risk={metrics['law_risk']:.8g}", flush=True)
        print(f"  tangent_action={metrics['tangent_action']:.8g}", flush=True)
        print(f"  full_action={metrics['full_action']:.8g}", flush=True)
        print(f"  max_calibration_residual={metrics['max_calibration_residual']:.3e}", flush=True)
        print(f"  min_ess_fraction={metrics['min_ess_fraction']:.6f}", flush=True)
        print(f"  max_poisson_relative_residual={metrics['max_poisson_relative_residual']:.3e}", flush=True)
        print(f"  valid={metrics['valid']}", flush=True)
    else:
        sel = payload["selection"]
        screens = payload["law_screens"]
        print("[run] complete", flush=True)
        print(f"  population_optimum_deg={sel['population_optimum_deg']}", flush=True)
        print(f"  law_optimum_deg={sel['law_optimum_deg']}", flush=True)
        print(f"  tangent_optimum_deg={sel['tangent_optimum_deg']}", flush=True)
        print(f"  full_optimum_deg={sel['full_optimum_deg']}", flush=True)
        print(
            f"  L*={screens['L_star']:.8g}  L_max={screens['L_max']:.8g}",
            flush=True,
        )
        print(
            f"  R*={screens['R_star']:.8g}  R_max={screens['R_max']:.8g}",
            flush=True,
        )
        print(
            "  max_relative_risk_violation="
            f"{100.0 * cfg['law']['max_relative_risk_violation']:.4g}%",
            flush=True,
        )

    print(f"results={output_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
