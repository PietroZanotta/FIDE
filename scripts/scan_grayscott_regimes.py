#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.grayscott.benchmark_design import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("scan")
    main()
