"""Canonical standalone V2.1 implementation of the vortices experiment."""
from __future__ import annotations

from typing import Any

__all__ = [
    "V2_VERSION",
    "config_fingerprint",
    "frozen_common_reference_scott_bandwidth",
    "frozen_reference_scott_bandwidth",
    "rasterize_v2",
    "solve_v2",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from . import core

    return getattr(core, name)
