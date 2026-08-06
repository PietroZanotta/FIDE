#!/usr/bin/env python3
"""Patch a JAX-recipe tesseract_api.py to enable one-entry VJP caching."""

from __future__ import annotations

import argparse
from pathlib import Path

IMPORT = "from tesseract_core.runtime.experimental import set_jax_vjp_cache_size"
CALL = "set_jax_vjp_cache_size(1)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        text = path.read_text(encoding="utf-8")
        if IMPORT not in text:
            text = IMPORT + "\n" + text
        if CALL not in text:
            lines = text.splitlines()
            insert_at = 1
            while insert_at < len(lines) and lines[insert_at].startswith(("import ", "from ")):
                insert_at += 1
            lines.insert(insert_at, CALL)
            text = "\n".join(lines) + "\n"
        path.write_text(text, encoding="utf-8")
        print(f"patched {path}")


if __name__ == "__main__":
    main()
