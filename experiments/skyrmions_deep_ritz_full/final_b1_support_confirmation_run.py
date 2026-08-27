from __future__ import annotations

import time

from .final_b1_support_confirmation import console_report, run_all


def main() -> None:
    run_all(progress=lambda message: print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True))
    print(console_report(), flush=True)


if __name__ == "__main__": main()
