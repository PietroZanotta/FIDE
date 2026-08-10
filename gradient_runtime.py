"""Backend adapter for complete differentiable MFSI objectives.

Unlike ``backend_runtime.py`` (small forward component maps), this adapter
dispatches whole gradient-bearing optimizers.  In Tesseract mode the objective,
reverse pass, optimizer, and checkpoint selection all execute inside the
Pasteur/ISI Labs Tesseract Core container.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

from backend_runtime import _post, normalize_backend


ROOT = Path(__file__).resolve().parent
ENGINES = {
    "rollout": {
        "api": ROOT / "tesseracts" / "rollout_gradient_engine" / "tesseract_api.py",
        "url_env": "MFSI_ROLLOUT_GRADIENT_TESSERACT_URL",
    },
    "fiber": {
        "api": ROOT / "tesseracts" / "fiber_gradient_engine" / "tesseract_api.py",
        "url_env": "MFSI_FIBER_GRADIENT_TESSERACT_URL",
    },
}
_MODULES: dict[str, Any] = {}


def _local_api(engine: str):
    if engine not in _MODULES:
        path = ENGINES[engine]["api"]
        specification = importlib.util.spec_from_file_location(
            f"mfsi_{engine}_gradient_api", path
        )
        if specification is None or specification.loader is None:
            raise RuntimeError(f"cannot load gradient engine: {path}")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        _MODULES[engine] = module
    return _MODULES[engine]


def run_gradient_engine(
    engine: str, payload: dict[str, Any], backend: str | None = None
) -> dict[str, Any]:
    if engine not in ENGINES:
        raise ValueError(f"unknown gradient engine {engine!r}")
    backend = normalize_backend(backend)
    if backend == "jax":
        return _local_api(engine).apply_payload(payload)
    url_name = ENGINES[engine]["url_env"]
    url = os.environ.get(url_name)
    if not url:
        raise RuntimeError(
            f"{url_name} is missing; invoke through "
            "scripts/_run_gradient_tesseracts.sh"
        )
    return _post(url, "apply", {"inputs": payload}, timeout=1800.0)
