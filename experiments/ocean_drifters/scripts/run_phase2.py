#!/usr/bin/env python3
"""Run the reproducible NOAA drifter MFSI Phase-2 pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-reference", action="store_true")
    parser.add_argument("--skip-repeated-cv", action="store_true")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    commands = [
        [sys.executable, str(directory / "freeze_cohort.py")],
        [sys.executable, str(directory / "train_reference.py")]
        + (["--force"] if args.force_reference else []),
        [sys.executable, str(directory / "build_sensor_bank.py")],
        [sys.executable, str(directory / "evaluate_iprojection.py")],
        [sys.executable, str(directory / "diagnose_reference_support.py")],
    ]
    if not args.skip_repeated_cv:
        commands.append(
            [sys.executable, str(directory / "evaluate_repeated_cv.py")]
            + (["--force"] if args.force_reference else [])
        )
    commands.append([sys.executable, str(directory / "validate_phase2.py")])
    for command in commands:
        print(f"[phase2] running {Path(command[1]).name}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
