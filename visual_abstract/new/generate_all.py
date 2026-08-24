from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "01_system_flow.py",
    "02_measurements.py",
    "03_moment_fiber.py",
    "04_transportability.py",
    "05_design_balance.py",
]

for name in SCRIPTS:
    subprocess.run([sys.executable, str(HERE / name)], check=True, cwd=HERE)

print("Generated:")
for name in SCRIPTS:
    print(" -", HERE / name.replace(".py", ".pdf"))
