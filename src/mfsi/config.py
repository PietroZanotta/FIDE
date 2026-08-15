from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(path: str | Path, *, smoke: bool = False) -> dict[str, Any]:
    """Load one experiment JSON config and optionally apply its ``smoke`` overlay.

    The CLI should expose only ``--smoke``. All scientific and numerical values live
    in the experiment config file.
    """
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    smoke_override = config.pop("smoke", {})
    if smoke:
        config = _deep_update(config, smoke_override)
    return config
