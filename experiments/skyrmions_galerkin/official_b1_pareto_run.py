"""Resumable CLI for official B1 Galerkin Pareto v1."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
from mfsi.config import load_config

from . import official_b1_pareto as study
from .galerkin_only import execution_device


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--mode",required=True,choices=("freeze-protocol","generate-data","law","candidates","screen","tangent","full","cross","freeze-selection","generate-validation","validate","report","all")); parser.add_argument("--config",type=Path,default=study.CONFIG_PATH); args=parser.parse_args(); cfg=load_config(args.config)
    routes={"freeze-protocol":study.freeze_protocol,"generate-data":study.generate_banks,"law":study.reconstruct_law,"candidates":study.generate_candidates,"screen":study.screen_candidates,"tangent":study.select_tangent,"full":study.select_full,"cross":study.cross_evaluate,"freeze-selection":study.freeze_selection,"generate-validation":study.generate_validation,"validate":study.validate,"report":study.write_report}
    order=("freeze-protocol","generate-data","law","candidates","screen","tangent","full","cross","freeze-selection","generate-validation","validate","report")
    with jax.default_device(execution_device()):
        for mode in (order if args.mode=="all" else (args.mode,)):
            print(f"starting={mode}",flush=True); result=routes[mode](cfg); print(f"completed={mode} passed={result.get('passed',True)}",flush=True)


if __name__ == "__main__": main()
