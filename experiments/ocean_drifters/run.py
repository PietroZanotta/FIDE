from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
DEFAULT_CONFIG = SCRIPT_DIR / "config.json"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mfsi.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen NOAA ocean-drifter MFSI experiment.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        choices=("projection", "risk", "benchmark", "plots", "tangent_action", "full_action", "solver_repair", "final_evaluation"),
        default="benchmark",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    os.environ.setdefault("JAX_PLATFORMS", str(cfg.get("runtime", {}).get("jax_platforms", "cpu")))

    try:
        from .experiment import run_experiment
    except ImportError:  # direct ``python experiments/ocean_drifters/run.py`` invocation
        from experiment import run_experiment

    if args.output is None:
        root = Path(cfg["output"]["root"])
        output_dir = (REPO_ROOT / root if not root.is_absolute() else root) / (
            "smoke" if args.smoke else args.stage
        )
    else:
        output_dir = args.output.resolve()
    print(
        f"noaa_ocean_drifters stage={args.stage} smoke={args.smoke} output={output_dir}",
        flush=True,
    )
    payload = run_experiment(cfg, output_dir, smoke=args.smoke, stage=args.stage)
    if payload.get("smoke"):
        print("[smoke] complete", flush=True)
        print(json.dumps(payload["smoke_projection"], indent=2), flush=True)
        print(f"  risk={payload['smoke_metrics']['R_star']:.8g}", flush=True)
    else:
        print(f"[{args.stage}] complete", flush=True)
        print(
            f"  admissible_layouts={payload['numerical_admissibility']['admissible_layout_count']}",
            flush=True,
        )
        if "risk" in payload:
            print(f"  best={payload['risk']['best_design_id']}", flush=True)
            print(f"  R*={payload['risk']['R_star']:.8g}", flush=True)
    print(f"results={output_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
