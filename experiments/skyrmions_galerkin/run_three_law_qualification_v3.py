"""Confirm the v2-selected common three-Law task on untouched larger banks."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

from . import three_law_qualification as base
from . import three_law_qualification_v3 as qualification
from .galerkin_only import execution_device


def main() -> None:
    cfg = base.load_default_config()
    print(f"device={execution_device()}", flush=True)
    with jax.default_device(execution_device()):
        result = qualification.run(cfg, print)
    print(f"status={result['status']}", flush=True)
    print(f"output_root={qualification.OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
