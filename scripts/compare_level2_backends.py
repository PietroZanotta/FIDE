#!/usr/bin/env python3
"""Compare completed direct-JAX and served-Tesseract level-2 summaries."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "level2_schedule"


def main() -> None:
    paths = {name: BASE / name / "level2_results.json" for name in ("jax", "tesseract")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit("missing backend result(s): " + ", ".join(missing))
    reports = {name: json.loads(path.read_text()) for name, path in paths.items()}
    keys = (
        "optimized_beta",
        "optimized_objective",
        "optimized_integrated_correction_energy",
        "optimized_min_ess_fraction",
        "gradient_relative_error",
    )
    print("\nLevel-2 JAX/Tesseract parity")
    print(f"{'metric':42s} {'jax':>14s} {'tesseract':>14s} {'abs diff':>12s}")
    worst = 0.0
    for key in keys:
        jax_value = float(reports["jax"]["metrics"][key])
        tess_value = float(reports["tesseract"]["metrics"][key])
        difference = abs(jax_value - tess_value)
        worst = max(worst, difference)
        print(f"{key:42s} {jax_value:14.6e} {tess_value:14.6e} {difference:12.3e}")
    if worst > 2e-9:
        raise SystemExit(f"backend parity failed: maximum absolute difference {worst:.3e}")
    print("backend parity: PASS")


if __name__ == "__main__":
    main()
