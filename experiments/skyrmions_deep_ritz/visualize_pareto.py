from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot an already-computed Pareto sweep")
    parser.add_argument("pareto", nargs="?", type=Path, default=SCRIPT_DIR / "outputs" / "pareto" / "pareto.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.pareto.read_text(encoding="utf-8"))
    rows = data["rows"]
    status = "Certified" if data.get("certified", False) else "Exploratory"
    output = args.output or args.pareto.parent / "figures"
    output.mkdir(parents=True, exist_ok=True)
    allowance = np.asarray([row["allowance_percent"] for row in rows])
    risk = np.asarray([row["selection_risk"] for row in rows])
    action = np.asarray([row["selection_action"] for row in rows])
    selection_reduction = 100.0 * np.asarray([row["action_reduction_vs_law"] for row in rows])
    validation_reduction = 100.0 * np.asarray(
        [row["validation_action_reduction_vs_law"] for row in rows]
    )

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    scatter = ax.scatter(risk, action, c=allowance, cmap="viridis", s=55)
    ax.plot(risk, action, color="0.55", linewidth=1)
    ax.set(
        xlabel="selection scientific risk",
        ylabel="authoritative Full action",
        title=f"{status} skyrmion Pareto front",
    )
    fig.colorbar(scatter, ax=ax, label="allowed extra risk (%)")
    fig.tight_layout()
    fig.savefig(output / "pareto_risk_vs_action.png", dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(allowance, selection_reduction, "o-", label="selection bank")
    ax.plot(allowance, validation_reduction, "s--", label="independent validation bank")
    ax.set(
        xlabel="allowed extra scientific risk (%)",
        ylabel="Full action reduction vs Law (%)",
        title=f"{status} action reduction vs risk budget",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "action_reduction_vs_allowance.png", dpi=190)
    plt.close(fig)
    print(f"figures={output}")


if __name__ == "__main__":
    main()
