#!/usr/bin/env python3
"""Audit-only entry point for rapid schema/data checks."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, load_data, repo_root  # noqa: E402
from run_analysis import audit, mkdirs  # noqa: E402

config = load_config()
root = repo_root()
output = root / config["output"]
mkdirs(output)
audit(load_data(root / config["input"]), root / config["input"], output)
print(output / "data_audit.md")
