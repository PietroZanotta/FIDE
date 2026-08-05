"""Generate the fixed S1--S3 smoke-problem fixture archive."""

from __future__ import annotations

import argparse
from pathlib import Path

from .problem_instances import build_smoke_problem_instances, save_smoke_problem_instances


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/smoke_problems.npz"))
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    args = parser.parse_args()
    arrays, metadata = build_smoke_problem_instances(args.seed, args.dtype)
    archive, manifest = save_smoke_problem_instances(arrays, metadata, args.output)
    print(f"wrote problem archive: {archive}")
    print(f"wrote problem manifest: {manifest}")


if __name__ == "__main__":
    main()
