#!/usr/bin/env python3
"""Compare advanced level-2 JAX and served-Tesseract summaries."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "level2_suite"
KEYS = (
    "optimized_integrated_fresh_energy", "optimized_min_fresh_ess",
    "max_fresh_calibration_residual", "gradient_relative_error",
)


def main():
    worst = 0.0
    for experiment in ("finite_neural", "manybody"):
        paths = {backend: BASE / experiment / backend / "results.json" for backend in ("jax", "tesseract")}
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise SystemExit("missing result(s): " + ", ".join(missing))
        reports = {backend: json.loads(path.read_text()) for backend, path in paths.items()}
        print(f"\n{experiment} JAX/Tesseract parity")
        print(f"{'metric':42s} {'jax':>14s} {'tesseract':>14s} {'abs diff':>12s}")
        for key in KEYS:
            left = float(reports["jax"]["metrics"][key])
            right = float(reports["tesseract"]["metrics"][key])
            difference = abs(left - right)
            worst = max(worst, difference)
            print(f"{key:42s} {left:14.6e} {right:14.6e} {difference:12.3e}")
    if worst > 2e-8:
        raise SystemExit(f"backend parity failed: maximum absolute difference {worst:.3e}")
    print("\nadvanced level-2 backend parity: PASS")


if __name__ == "__main__":
    main()
