# 1. Install Python dependencies + Tesseract Core CLI/runtime
chmod +x scripts/install.sh
./scripts/install.sh

# 2. Build both actual Tesseracts
source .venv/bin/activate
chmod +x scripts/build_tesseracts.sh
./scripts/build_tesseracts.sh

# 3. Run with Tesseract backend — default
chmod +x scripts/_run_with_backend.sh
chmod +x scripts/run_example_a.sh
chmod +x scripts/run_example_b.sh
./scripts/run_example_a.sh
./scripts/run_example_b.sh

# Equivalent, explicit
./scripts/run_example_a.sh --backend tesseract
./scripts/run_example_b.sh --backend tesseract

# Direct-JAX reference path
./scripts/run_example_a.sh --backend jax
./scripts/run_example_b.sh --backend jax

# Generate report
chmod +x scripts/run_experiments_and_report.sh 
./scripts/run_experiments_and_report.sh --backend jax

chmod +x scripts/show_results.sh --backend jax
./scripts/show_results.sh
