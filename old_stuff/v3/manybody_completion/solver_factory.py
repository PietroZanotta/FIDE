"""Create the configured scientific solver backend."""

from __future__ import annotations

from .tesseract_backend import LocalSolverBackend


def create_solver_backend(config: dict | None = None) -> LocalSolverBackend:
    # The compact repository always has a fully functional local backend.  The
    # packaged Tesseract APIs call the same functions and can be served by an
    # external runtime when installed.
    _ = config
    return LocalSolverBackend()
