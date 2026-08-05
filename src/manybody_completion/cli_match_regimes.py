"""Rank cross-family samples with similar pair moments and distinct angular structure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .datasets import select_matched_cross_regime_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--max-pairs", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with np.load(args.dataset, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    matches = select_matched_cross_regime_pairs(arrays, max_pairs=args.max_pairs)
    rendered = json.dumps(matches, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote matches: {args.output}")


if __name__ == "__main__":
    main()
