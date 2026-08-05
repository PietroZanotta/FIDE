"""Report the JAX and Tesseract environment without requiring Docker."""

from __future__ import annotations

import importlib.metadata
import shutil

import jax


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def main() -> None:
    print(f"jax:             {jax.__version__}")
    print(f"jax backend:     {jax.default_backend()}")
    print(f"jax devices:     {jax.devices()}")
    print(f"tesseract-core:  {_version('tesseract-core')}")
    print(f"tesseract-jax:   {_version('tesseract-jax')}")
    print(f"docker:          {shutil.which('docker') or 'NOT FOUND'}")


if __name__ == "__main__":
    main()
