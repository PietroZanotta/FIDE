from __future__ import annotations

"""Run the exact-protocol prospective-v5 replication on the frozen v4 engine."""

import argparse
from pathlib import Path

from common import SCRIPT_DIR, load_config
from run_v4 import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=SCRIPT_DIR / "configs" / "production_v5.json",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--stage", choices=("all", "prepare", "select", "validate"), default="all"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.get("name") != "prospective_v5_robust_full_replication":
        raise RuntimeError("run_v5 requires the sealed prospective-v5 identity")
    if cfg.get("replication", {}).get("parent_hidden_validation_use") != "forbidden":
        raise RuntimeError("v5 must explicitly forbid use of the v4 hidden bank")
    output = args.output_dir or SCRIPT_DIR / "outputs" / str(cfg["name"])
    run(args.config, output, args.stage)


if __name__ == "__main__":
    main()
