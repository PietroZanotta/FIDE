"""Run the prospective common-task qualification for all three B1 Laws."""

from __future__ import annotations

import argparse

import jax

jax.config.update("jax_enable_x64", True)

from . import three_law_qualification as qualification
from .galerkin_only import execution_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("protocol", "laws", "development", "all"),
        default="all",
    )
    args = parser.parse_args()
    cfg = qualification.load_default_config()
    print(f"device={execution_device()}", flush=True)
    with jax.default_device(execution_device()):
        if args.stage == "protocol":
            result = qualification.freeze_protocol(cfg)
        elif args.stage == "laws":
            result = qualification.fit_all_laws(cfg, print)
        elif args.stage == "development":
            result = qualification.run_development(cfg, print)
        else:
            result = qualification.run(cfg, print)
    print(f"status={result.get('status', 'COMPLETE')}", flush=True)
    print(f"output_root={qualification.OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
