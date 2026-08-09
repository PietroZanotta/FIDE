#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from manybody_completion.experiment import build_markdown_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    Path(args.output).write_text(build_markdown_summary(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
