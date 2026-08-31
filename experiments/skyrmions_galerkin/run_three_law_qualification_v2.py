"""Run the corrective lower-K three-Law common-task qualification."""

from __future__ import annotations

import argparse

import jax

jax.config.update("jax_enable_x64", True)

from . import three_law_qualification as base
from . import three_law_qualification_v2 as qualification
from .galerkin_only import execution_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("protocol", "development", "all"), default="all")
    args = parser.parse_args()
    cfg = base.load_default_config()
    print(f"device={execution_device()}", flush=True)
    with jax.default_device(execution_device()):
        if args.stage == "protocol":
            result = qualification.freeze_protocol(cfg)
        elif args.stage == "development":
            result = qualification.run_development(cfg, print)
        else:
            result = qualification.run(cfg, print)
    print(f"status={result.get('status', 'COMPLETE')}", flush=True)
    print(f"output_root={qualification.OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
