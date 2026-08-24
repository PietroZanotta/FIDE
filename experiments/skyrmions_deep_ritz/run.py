from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import jax

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

jax.config.update("jax_enable_x64", True)

from mfsi.config import load_config
from experiments.skyrmions_deep_ritz.experiment import run_experiment


def _deep_update(base, override):
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _load_preflight(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    overlay = raw.pop("preflight", {})
    raw.pop("smoke", None)
    return _deep_update(raw, overlay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Many-body skyrmion FIDE / Deep Ritz experiment")
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--smoke", action="store_true")
    profile.add_argument("--preflight", action="store_true")
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.json")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    cfg = _load_preflight(args.config) if args.preflight else load_config(args.config, smoke=args.smoke)
    mode = "smoke" if args.smoke else "preflight" if args.preflight else "run"
    output = args.output_dir or SCRIPT_DIR / "outputs" / mode
    result = run_experiment(cfg, output, smoke=args.smoke)
    print(f"result={output / 'result.json'}")
    print(f"milestone_success={result['milestone_success']}")
    if not args.smoke and not args.preflight and not result["milestone_success"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
