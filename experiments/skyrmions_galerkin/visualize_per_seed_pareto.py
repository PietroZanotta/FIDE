"""Visualize an independent single-seed B1 Galerkin Pareto frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from . import per_seed_pareto


ROOT = Path(__file__).resolve().parent


def load_records(seed_id: str) -> list[dict]:
    path = per_seed_pareto.seed_output_root(seed_id) / "pareto.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("seed_id") != seed_id:
        raise ValueError(f"seed mismatch in {path}")
    return payload["allowances"]


def make_figure(seed_id: str, rows: list[dict]) -> plt.Figure:
    allowances = [float(row["allowance_percent"]) for row in rows]
    law_risk = float(rows[0]["Law"]["risk"])
    figure, (action_axis, risk_axis) = plt.subplots(
        1, 2, figsize=(10.4, 4.2), constrained_layout=True
    )
    figure.suptitle(
        f"Skyrmions B1 Galerkin · {seed_id} Pareto frontier",
        fontsize=14, fontweight="bold",
    )
    styles = {
        "Tangent": ("#6A3D9A", "o", "tangent_action"),
        "Full": ("#D55E00", "s", "full_action"),
    }
    for method, (color, marker, action_key) in styles.items():
        actions = [
            None if row[method] is None else float(row[method][action_key])
            for row in rows
        ]
        action_axis.plot(
            allowances, actions, color=color, marker=marker,
            linewidth=2.1, markersize=6, label=method,
        )
        risk_change = [
            None if row[method] is None else
            100.0 * (float(row[method]["risk"]) / law_risk - 1.0)
            for row in rows
        ]
        risk_axis.plot(
            allowances, risk_change, color=color, marker=marker,
            linewidth=2.1, markersize=6, label=method,
        )
    action_axis.set_title("Certified action")
    action_axis.set_xlabel("Risk allowance p (%)")
    action_axis.set_ylabel("Action")
    risk_axis.plot(
        allowances, allowances, color="#555555", linestyle="--",
        linewidth=1.5, label="risk ceiling",
    )
    risk_axis.axhline(0.0, color="#999999", linewidth=0.8)
    risk_axis.set_title("Realized risk change")
    risk_axis.set_xlabel("Risk allowance p (%)")
    risk_axis.set_ylabel("Change from this seed's Law (%)")
    for axis in (action_axis, risk_axis):
        axis.set_xticks(allowances)
        axis.grid(True, alpha=0.22, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    figure.text(
        0.5, -0.01,
        "Independent single-reference selection · negative risk change = below Law · validation not accessed",
        ha="center", fontsize=8.5, color="#555555",
    )
    return figure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", choices=per_seed_pareto.SUPPORTED_SEEDS, default="B1_seed0")
    parser.add_argument("--output-stem", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    output_stem = args.output_stem or (
        ROOT / "figures" / f"skyrmion_b1_galerkin_pareto_{args.seed}_v1"
    )
    output_stem = output_stem.expanduser().resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    rows = load_records(args.seed)
    figure = make_figure(args.seed, rows)
    outputs = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    figure.savefig(outputs[0], dpi=args.dpi, bbox_inches="tight")
    figure.savefig(outputs[1], bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(figure)
    for path in outputs:
        print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
