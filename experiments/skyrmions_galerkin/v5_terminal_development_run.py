"""Runner for the development-only V4 terminal-failure diagnosis."""
from __future__ import annotations
import argparse
import time
import jax
from . import v5_terminal_development as study

def progress(message: str) -> None: print(message, flush=True)

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--mode",choices=("freeze","banks","risk","algebra","finalize","all"),required=True); args=parser.parse_args()
    study.activate(); routes={"freeze":study.freeze,"banks":study.generate_banks,"risk":study.run_risk,"algebra":study.run_algebra,"finalize":study.finalize}
    order=("freeze","banks","risk","algebra","finalize") if args.mode=="all" else (args.mode,)
    with jax.default_device(jax.devices("gpu")[0]):
        for name in order:
            started=time.perf_counter(); print(f"starting={name}",flush=True); routes[name](progress); print(f"completed={name} wall_seconds={time.perf_counter()-started:.3f}",flush=True)

if __name__=="__main__": main()

