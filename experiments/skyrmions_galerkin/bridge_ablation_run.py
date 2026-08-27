from __future__ import annotations

import time

from .bridge_ablation import console_report, run_all


def _progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    run_all(progress=_progress)
    print(console_report(), flush=True)


if __name__ == "__main__":
    main()
