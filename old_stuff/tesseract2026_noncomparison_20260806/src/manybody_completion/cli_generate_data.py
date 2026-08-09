"""Command-line entry point for deterministic dataset generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_yaml
from .datasets import generate_dataset, save_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    arrays, metadata = generate_dataset(config)
    dataset_path, metadata_path = save_dataset(arrays, metadata, args.output)
    print(f"wrote dataset: {dataset_path}")
    print(f"wrote metadata: {metadata_path}")
    print(f"coordinates shape: {arrays['coordinates'].shape}")
    print(f"pair moments shape: {arrays['pair_moments'].shape}")


if __name__ == "__main__":
    main()
