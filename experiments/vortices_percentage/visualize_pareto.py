"""Visualize the vortices percentage-risk Pareto sweep."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from percentage_pareto_visualization import load_rows, make_figure, save_figure

DEFAULT_INPUT = SCRIPT_DIR / "outputs" / "pareto"


def save_pareto_figure(rows: list[dict], output: Path, *, dpi: int = 220) -> Path:
    return save_figure(rows, output, experiment_label="Vortices / double gyre", dpi=dpi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        rows, resolved = load_rows(args.input)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(f"Could not read Pareto data {args.input}: {exc}") from exc
    output = args.output or resolved.with_name("pareto.png")
    if args.show:
        fig = make_figure(rows, experiment_label="Vortices / double gyre")
        fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
        print(f"saved={Path(output).resolve()}", flush=True)
        plt.show()
        plt.close(fig)
    else:
        save_pareto_figure(rows, output, dpi=args.dpi)
        print(f"saved={Path(output).resolve()}", flush=True)


if __name__ == "__main__":
    main()
