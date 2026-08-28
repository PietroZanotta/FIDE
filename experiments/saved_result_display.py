"""Small, dependency-free formatting helpers for saved-result evaluators."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Sequence


MISSING = "—"


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def number(value: Any, digits: int = 6) -> str:
    if not finite(value):
        return MISSING
    value = float(value)
    if value != 0.0 and (abs(value) < 1e-4 or abs(value) >= 1e5):
        return f"{value:.4e}"
    return f"{value:.{digits}f}"


def percent(value: Any, digits: int = 2) -> str:
    if not finite(value):
        return MISSING
    value = float(value)
    if abs(value) < 0.5 * 10 ** (-(digits + 2)):
        value = 0.0
    return f"{100.0 * value:.{digits}f}%"


def sample_sd_from_se(standard_error: Any, sample_count: Any) -> float | None:
    """Recover sample SD only when the saved record supplies both SE and n."""
    if not finite(standard_error) or not finite(sample_count):
        return None
    sample_count = int(sample_count)
    if sample_count < 1:
        return None
    return float(standard_error) * math.sqrt(sample_count)


def source_label(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repository_root.resolve()))
    except ValueError:
        return str(resolved)


def print_heading(experiment: str, scope: str, sources: Iterable[str]) -> None:
    print(experiment)
    print(scope)
    for source in sources:
        print(f"source: {source}")
    print()


def print_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    rendered = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered:
        if len(row) != len(headers):
            raise ValueError("table row has a different length from its header")
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rendered:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))


def print_uncertainty_note(text: str) -> None:
    print()
    print(f"uncertainty: {text}")
