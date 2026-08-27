from __future__ import annotations

from common import SCRIPT_DIR
from run_experiment import run


if __name__ == "__main__":
    run(
        SCRIPT_DIR / "configs" / "smoke.json",
        SCRIPT_DIR / "outputs" / "smoke",
    )

