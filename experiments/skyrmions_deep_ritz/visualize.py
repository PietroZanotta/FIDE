"""Pure post-processing for saved 3% or Pareto results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", nargs="?", type=Path, default=SCRIPT_DIR / "outputs" / "run" / "result.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    output = args.output or args.result.parent / "figures"
    output.mkdir(parents=True, exist_ok=True)

    candidates = [row for row in result["authoritative_candidates"] if np.isfinite(row["action"])]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.scatter([r["risk"] for r in candidates], [r["action"] for r in candidates], c=[r["valid"] for r in candidates], cmap="coolwarm")
    ax.set(xlabel="selection scientific risk", ylabel="certified Full action", title="Risk vs Full Deep Ritz action")
    _save(fig, output / "risk_vs_action.png")

    truth_path = args.result.parent / "truth_banks.npz"
    with np.load(truth_path) as bank:
        configurations = np.asarray(bank["design"])
    middle = configurations[len(configurations) // 2].reshape(-1, 2)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), sharex=True, sharey=True)
    for ax, (label, eta) in zip(axes, (("Law", result["law_anchor"]["eta"]), ("Full Deep Ritz", result["full_3_percent"]["eta"]))):
        ax.hist2d(middle[:, 0], middle[:, 1], bins=(45, 24), range=((0, 2), (0, 1)), cmap="magma")
        centers = np.asarray(eta).reshape(-1, 2)
        ax.scatter(centers[:, 0], centers[:, 1], marker="x", s=80, c="cyan", linewidths=2)
        ax.set(title=label, xlim=(0, 2), ylim=(0, 1), xlabel="x")
    axes[0].set_ylabel("y")
    _save(fig, output / "sensor_layouts.png")

    history = result["full_3_percent"]["optimization"]["history"]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    adam = [row for row in history if row["phase"] == "adam"]
    lbfgs = [row for row in history if row["phase"] == "lbfgs"]
    if adam:
        ax.plot([row["step"] for row in adam], [row["objective"] for row in adam], "o-", label="Adam")
    if lbfgs:
        offset = adam[-1]["step"] if adam else 0
        ax.plot([offset + row["iteration"] for row in lbfgs], [row["objective"] for row in lbfgs], "o-", label="L-BFGS")
    ax.set(xlabel="optimization iteration", ylabel="Ritz objective", title="Deep Ritz convergence")
    ax.legend()
    _save(fig, output / "ritz_convergence.png")

    cert = result["full_3_percent"]["certificate"]
    names = ["weak", "energy", "gauge", "moment-rate"]
    values = [cert["maximum_weak_residual"], cert["maximum_energy_residual"], cert["maximum_gauge_residual"], cert["maximum_moment_rate_residual"]]
    thresholds = cert["thresholds"]
    limits = [thresholds["maximum_weak_residual"], thresholds["maximum_energy_residual"], thresholds["maximum_gauge_residual"], thresholds["maximum_moment_rate_residual"]]
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.bar(names, np.asarray(values) / np.asarray(limits))
    ax.axhline(1, color="black", linestyle="--")
    ax.set(ylabel="residual / threshold", title="Authoritative certificate summary")
    _save(fig, output / "certificates.png")

    timings = result["timings_seconds"]
    labels = [key for key in timings if key != "total"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(labels, [timings[key] for key in labels])
    ax.tick_params(axis="x", rotation=35)
    ax.set(ylabel="seconds", title="Runtime profile")
    _save(fig, output / "runtime_profile.png")
    print(f"figures={output}")


if __name__ == "__main__":
    main()

