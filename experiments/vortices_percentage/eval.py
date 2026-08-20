from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT = SCRIPT_DIR / "outputs" / "run" / "result.json"


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _num(x: Any) -> str:
    if not _finite(x):
        return "n/a"
    x = float(x)
    return f"{x:.4e}" if x != 0.0 and (abs(x) < 1e-4 or abs(x) >= 1e5) else f"{x:.7g}"


def _metric(block: dict[str, Any], key: str) -> str:
    m = block.get(key, {})
    if not _finite(m.get("mean")):
        return "n/a"
    return f"{_num(m['mean'])} ± {_num(m.get('se'))} (SE, n={m.get('n', 0)})"


def _tail_ratio(block: dict[str, Any], key: str) -> float:
    metric = block.get(key, {})
    median = metric.get("median")
    maximum = metric.get("max")
    if not _finite(median) or not _finite(maximum) or float(median) <= 0.0:
        return float("nan")
    return float(maximum) / float(median)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", nargs="?", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()
    data = json.loads(args.result.read_text(encoding="utf-8"))
    print("=" * 88)
    print("VORTICES / DOUBLE-GYRE — SAVED RESULT")
    print("=" * 88)
    print(f"file:       {args.result}")
    print(f"schema:     {data.get('schema_version')}")
    print(f"config hash:{data.get('config_hash', 'n/a')}")
    print(f"smoke:      {data.get('smoke')}")
    if data.get("smoke"):
        print(f"centers:    {data.get('smoke_centers')}")
        m = data.get("smoke_metrics", {})
        for k in ("law_risk", "tangent_action", "full_action", "max_calibration_residual", "min_ess_fraction", "max_poisson_relative_residual"):
            print(f"  {k:<34} {_num(m.get(k))}")
        print(f"  {'valid':<34} {m.get('valid')}")
        return 0 if m.get("valid") else 2

    s = data.get("law_screens", {})
    print("\nInformation screens")
    print(f"  L*={_num(s.get('L_star'))}   Lmax={_num(s.get('L_max'))}   epsilon_L={_num(s.get('epsilon_l'))}")
    print(f"  R*={_num(s.get('R_star'))}   Rmax={_num(s.get('R_max'))}   epsilon_R={_num(s.get('epsilon_r'))}")
    relative_limit = data.get("config", {}).get("law", {}).get("max_relative_risk_violation")
    if _finite(relative_limit):
        print(f"  maximum relative R violation: {100.0 * float(relative_limit):.4g}%")

    print("\nSelected centers")
    for name in ("population", "law", "tangent", "full"):
        print(f"  {name:<12} {data.get('selection_centers', {}).get(name)}")

    failures = []
    cert = data.get("selection_certificates", {})
    if cert:
        print("\nSelection-bank certification")
        for name in ("population", "law", "tangent", "full"):
            c = cert.get(name, {})
            print(
                f"  {name:<12} L={_num(c.get('L_selection')):<12} "
                f"R={_num(c.get('R_selection')):<12} required={'+'.join(c.get('required_screens', [])):<5} "
                f"{'PASS' if c.get('certified') else 'FAIL'}"
            )
            if not c.get("certified"):
                failures.append(f"{name} failed selection certificate")

    print("\nIndependent validation")
    action_tail_limit = float(
        data.get("config", {})
        .get("validity", {})
        .get("max_action_to_median_ratio", 5.0)
    )
    for name in ("population", "law", "tangent", "full"):
        block = data.get("validation", {}).get(name, {})
        tangent_tail = _tail_ratio(block, "tangent_action")
        full_tail = _tail_ratio(block, "full_action")
        print(
            f"  {name:<12} R={_metric(block, 'law_risk'):<30} "
            f"Atan={_metric(block, 'tangent_action'):<30} A={_metric(block, 'full_action'):<30} "
            f"valid={100.0*float(block.get('valid_fraction', 0.0)):.1f}%"
        )
        print(
            f"  {'':<12} action max/median: "
            f"Atan={_num(tangent_tail)} A={_num(full_tail)} "
            f"(limit={_num(action_tail_limit)})"
        )
        # The population design is an exact-moment oracle baseline and is only
        # certified against L.  Finite/noisy validity is diagnostic for it; the
        # DG-Obs law and transport designs must pass the finite validation gate.
        if name != "population" and float(block.get("valid_fraction", 0.0)) < 0.95:
            failures.append(f"{name} validation valid fraction below 0.95")
        if name != "population":
            for metric_name, ratio in (
                ("tangent action", tangent_tail),
                ("full action", full_tail),
            ):
                if _finite(ratio) and ratio > action_tail_limit:
                    failures.append(
                        f"{name} {metric_name} max/median ratio exceeds "
                        f"{action_tail_limit:g}"
                    )
        lb = block.get("tangent_lower_bound_check", {})
        if _finite(lb.get("max_violation")) and float(lb["max_violation"]) > float(lb.get("tolerance", 1e-6)):
            failures.append(f"{name} violated Atan <= A beyond tolerance")

    contrast = data.get("contrasts", {}).get("full_vs_law_full_action_reduction", {})
    print("\nFull vs Law")
    print(f"  paired n:                  {contrast.get('n', 0)}")
    print(f"  ratio-of-means reduction: {_num(contrast.get('ratio_of_means_reduction'))}")
    print(f"  mean paired A reduction:  {_num(contrast.get('mean_paired_difference'))}")

    if failures:
        print("\nFAILURES")
        for f in failures:
            print(f"  - {f}")
        return 2
    print("\nSaved result passes declared structural checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
