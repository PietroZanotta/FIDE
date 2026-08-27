from pathlib import Path

from common import SCRIPT_DIR
from run_v4 import run


if __name__ == "__main__":
    run(
        SCRIPT_DIR / "configs" / "smoke_v4.json",
        SCRIPT_DIR / "outputs" / "prospective_v4_robust_full_smoke",
        "all",
    )
