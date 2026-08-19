"""Saved-result evaluator for two-species finite-measure action and risk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def _metric(summary, name):
    row = summary["metrics"][name]
    return f"{row['mean']:.7g} +/- {row['se']:.3g} (SE)"


def evaluate(path: Path) -> int:
    result = json.loads(path.read_text())
    validation = result["validation"]
    metrics = validation["metrics"]
    print("=" * 88)
    print("ACTIVE NEMATIC — TWO-SPECIES UNBALANCED MFSI")
    print("=" * 88)
    print(f"file: {path}")
    print(f"reference seeds: base={result['reference_seed']} plus={result['plus_reference_seed']} minus={result['minus_reference_seed']}")
    print(f"physical interval: {result['physical_interval']}")
    print(f"reaction kappa: {result['reaction_kappa']}")
    print(f"valid trials: {validation['valid_trials']}/{validation['trials']}")
    print("\nFinite-measure risk")
    for name in ("law_risk_total", "law_risk_plus", "law_risk_minus", "shape_mmd_plus", "shape_mmd_minus", "mass_error_plus", "mass_error_minus"):
        print(f"  {name:<34} {_metric(validation, name)}")
    print("\nUnbalanced Full action")
    for name in (
        "full_unbalanced_action_total", "full_unbalanced_action_plus", "full_unbalanced_action_minus",
        "move_action_plus", "reaction_action_plus", "move_action_minus", "reaction_action_minus",
        "reaction_fraction_plus", "reaction_fraction_minus", "reaction_fraction_total",
    ):
        print(f"  {name:<34} {_metric(validation, name)}")
    print("\nNumerical diagnostics")
    for name in ("max_calibration_residual", "min_ess_fraction", "max_screened_pde_relative_residual"):
        print(f"  {name:<34} {_metric(validation, name)}")
    balance = result["charge_balance_diagnostics"]
    print(f"\ncharge balance: {'PASS' if balance['passed'] else 'FAIL'}; max violation={balance['maximum_violation']:.7g}")
    decomposed = (
        result.get("species_weight_plus", 1.0)
        * (metrics["move_action_plus"]["mean"] + metrics["reaction_action_plus"]["mean"])
        + result.get("species_weight_minus", 1.0)
        * (metrics["move_action_minus"]["mean"] + metrics["reaction_action_minus"]["mean"])
    )
    total = metrics["full_unbalanced_action_total"]["mean"]
    failures = []
    if validation["valid_trials"] != validation["trials"]:
        failures.append("not every validation trial is valid")
    if not balance["passed"]:
        failures.append("charge-balance guard failed")
    if not np.isclose(decomposed, total, rtol=2.0e-8, atol=1.0e-10):
        failures.append("species move/reaction decomposition does not equal total")
    if failures:
        print("\nFAILURES")
        for failure in failures:
            print(f"  - {failure}")
        return 2
    print("\nSaved result passes declared structural checks.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    return evaluate(args.result)


if __name__ == "__main__":
    raise SystemExit(main())
