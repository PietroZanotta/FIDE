# Experiment runbook

Run every command below from the repository root:

```bash
cd /home/zanot/projects/tesseract2026
```

## 1. Install

For both direct JAX and Tesseract/Docker execution:

```bash
./scripts/install.sh
source .venv/bin/activate
```

Docker must be installed and its daemon running for Tesseract runs. The direct
JAX backend does not need Docker. A JAX-only environment can be installed with:

```bash
./scripts/install.sh --jax-only
source .venv/bin/activate
```

Keep the virtual environment activated for the remaining commands so the
`tesseract` executable in `.venv/bin/` is available.

## 2. What the levels and experiment names mean

- **Experiment A** is the exact one-dimensional diagnostic.
- **Experiment B** is the two-dimensional equal-covariance multimodal study.
- **Level 0** is the core MFSI method used by A/B: stop-gradient calibration,
  fiber calculus, and the learned Deep-Ritz correction.
- **Level 1** is the implicit multiplier derivative. Its finite-difference
  check is part of Experiment A validation.
- **Level 2** is the new small fiber-adapted-path study. It optimizes the
  stochastic-interpolant noise amplitude `beta` to minimize integrated exact
  correction energy while penalizing relative ESS below `0.60`.

The new level-2 code is isolated from A/B. It has its own Tesseract image,
runner, port, and output directories.

## 3. Experiments A and B (levels 0 and 1)

Tesseract is the default component backend:

```bash
./scripts/build_tesseracts.sh
./scripts/run_example_a.sh
./scripts/run_example_b.sh
```

The same experiment paths with the direct in-process JAX backend:

```bash
./scripts/run_example_a.sh --backend jax
./scripts/run_example_b.sh --backend jax
```

Useful development variants:

```bash
# A: retrain, refine, or validate without the matched benchmark
./scripts/run_example_a.sh --retrain
./scripts/run_example_a.sh --refine
./scripts/run_example_a.sh --validate-only

# B: short smoke run or full retraining with an explicit seed
./scripts/run_example_b.sh --backend jax --quick --seed 123
./scripts/run_example_b.sh --retrain --seed 20260808
```

Run the existing full A/B workflow:

```bash
./scripts/run_all.sh                    # Tesseract backend
./scripts/run_all.sh --backend jax      # direct JAX backend
```

Run A/B and print the main tables:

```bash
./scripts/run_experiments_and_report.sh
./scripts/run_experiments_and_report.sh --backend jax
```

Print already-generated results without rerunning anything:

```bash
./scripts/show_results.sh --backend tesseract
./scripts/show_results.sh --backend jax
```

## 4. Level-2 fiber-adapted schedule

### Fast JAX smoke run

```bash
./scripts/run_level2.sh --backend jax --quick
```

### Standard JAX run

```bash
./scripts/run_level2.sh --backend jax
```

### Standard Tesseract run

The runner builds only `mfsi-fiber-path-adapter:latest` if it is missing,
serves it on port `18083`, calls the REST `/apply` endpoint, and tears it down:

```bash
./scripts/run_level2.sh --backend tesseract
```

To build the image explicitly first:

```bash
./scripts/build_level2_tesseract.sh
./scripts/run_level2.sh --backend tesseract
```

### Run both implementations and enforce parity

```bash
./scripts/run_level2_both.sh
```

For a faster parity smoke test:

```bash
./scripts/run_level2_both.sh --quick
```

If both results already exist, compare without rerunning:

```bash
python scripts/compare_level2_backends.py
```

Use another served-component port if `18083` is occupied:

```bash
MFSI_LEVEL2_TESSERACT_PORT=19083 \
  ./scripts/run_level2.sh --backend tesseract
```

Add `--no-plots` to any level-2 run when only numerical validation is needed.

## 5. Level-2 outputs and acceptance criteria

Backends never overwrite one another:

```text
results/level2_schedule/jax/
results/level2_schedule/tesseract/
```

Each directory contains:

- `level2_results.json` — configuration, metrics, and pass/fail gates;
- `level2_arrays.npz` — full time curves, optimization trace, densities, and
  objective landscape;
- `level2_schedule_summary.png` — objective landscape, optimization trace,
  correction-energy profile, and ESS profile;
- `level2_density_paths.png` — raw reference versus I-projected density before
  and after schedule adaptation.

The run fails if the optimized schedule does not substantially lower
correction energy, misses the ESS floor, loses moment calibration, or if the
implicit gradient disagrees with a central finite difference.

The expected controlled result is `beta` near `1`: at this value the chosen
noise schedule exactly restores unit raw-bridge variance, so the reference
path already stays on the mean/second-moment fiber and needs essentially no
dynamic correction. This known optimum is shown in the plot but is not used by
the optimizer.

## 5a. Advanced level-2 two-experiment suite

The scalar schedule study above remains available. A separate advanced suite
adds two more demanding tests:

1. `finite_neural`: the 2D ring-to-four-lobe bridge uses 512 training particles
   per time, an independent 1024-particle validation bank, a three-parameter
   time-dependent schedule, and a 64-feature neural Ritz potential.
2. `manybody`: 16 particles in two dimensions give a 32-dimensional microscopic
   state. It uses 72 training configurations and 160 fresh configurations per
   time, periodic radial-pair neural features, three pair-gyration constraints,
   and held-out fourfold order.

The neural potential is a one-hidden-layer random-feature network: its hidden
features are fixed reproducibly and its output layer is learned from empirical
Deep-Ritz normal equations. Schedule differentiation passes through the neural
solve and the implicit I-projection derivative.

Run the standard suite with direct JAX:

```bash
./scripts/run_level2_suite.sh --backend jax
```

Run the actual served Tesseracts:

```bash
./scripts/build_level2_suite_tesseracts.sh
./scripts/run_level2_suite.sh --backend tesseract
```

Run both implementations and require numerical parity:

```bash
./scripts/run_level2_suite_both.sh
```

Useful variants:

```bash
# plumbing-only budgets
./scripts/run_level2_suite.sh --backend jax --quick

# run only one study
./scripts/run_level2_suite.sh --backend jax --experiment finite_neural
./scripts/run_level2_suite.sh --backend jax --experiment manybody

# compare existing backend outputs
python scripts/compare_level2_suite_backends.py
```

The advanced Tesseracts use ports `18084` and `18085`. Override them with
`MFSI_FINITE_NEURAL_TESSERACT_PORT` and
`MFSI_MANYBODY_TESSERACT_PORT`, respectively.

Outputs are separated by experiment and backend:

```text
results/level2_suite/finite_neural/jax/
results/level2_suite/finite_neural/tesseract/
results/level2_suite/manybody/jax/
results/level2_suite/manybody/tesseract/
```

Each directory contains `results.json`, `arrays.npz`, a six-panel dashboard,
and fresh-bank path snapshots. Standard-mode gates require lower correction
energy and forcing power on the independent bank, improved ESS, fresh-bank calibration, an
implicit/finite-difference gradient match, and positive initial-path neural
Ritz gain. The many-body study additionally requires a 32D state and substantial
held-out structural motion.

Quick mode checks plumbing, calibration, gradients, energy, and ESS using much
smaller banks; neural generalization is intentionally a standard-budget gate.

## 5b. Paper-facing level-2 study (the six-point experiment)

This additional workflow is isolated from every experiment above. It addresses
the six publication gaps directly:

1. five independent finite endpoint banks, with hand-constant, optimized
   scalar, and nested three-parameter schedules;
2. a fully trained two-hidden-layer invariant MLP potential and corrected ODE;
3. generated-versus-independent projected-law MMD on radial descriptors plus
   held-out fourfold order;
4. a validation-selected correction amplitude with an exact zero fallback and
   a separately reported held-out Ritz gain;
5. 32 particles in two dimensions (64D state), three smooth radial-pair
   constraints, finite-temperature energy-relaxed endpoints, and a hidden q4
   phase change;
6. raw and instantaneous tangent-projection baselines, paired seed-level 95%
   intervals, wall time, and matched Heun NFE.

The schedule model is selected on a bank distinct from both its optimization
bank and final test bank. The common endpoint radial target is found inside the
intersection of the two finite empirical convex hulls, so exact endpoint
calibration does not depend on shared configurations. Quick mode is one seed;
standard mode is five seeds (`401` through `405`).

Fast direct-JAX plumbing run:

```bash
./scripts/run_level2_paper_study.sh --backend jax --quick
```

Five-bank standard run with confidence intervals and figures:

```bash
./scripts/run_level2_paper_study.sh --backend jax
```

Build and exercise the invariant correction through a served Tesseract while
the training/orchestration remains in JAX:

```bash
./scripts/build_level2_paper_tesseract.sh
./scripts/run_level2_paper_study.sh --backend tesseract --quick
./scripts/run_level2_paper_study.sh --backend tesseract
```

The Tesseract wrapper uses port `18086`; override it with
`MFSI_PAPER_LEVEL2_TESSERACT_PORT`. To rebuild summaries and plots from complete
seed files without repeating the simulation:

```bash
./scripts/run_level2_paper_study.sh --backend jax --aggregate-existing
```

Outputs never overwrite between component backends:

```text
results/level2_paper_study/jax/
results/level2_paper_study/tesseract/
```

Each contains `seed_<seed>.json`, `summary.json`,
`paper_level2_summary.png`, `paper_level2_path_diagnostics.png`, and (when the
five-bank diagnostics are present) `paper_level2_failure_diagnostics.png`. The JSON
records raw seed values, marginal intervals, paired-effect intervals, schedule
selection decisions, endpoint residuals, neural gate/gain, wall time, NFE, and
JAX/component parity. It also records the N=32 local-to-rollout diagnostic
bundle: held-out gain at off-grid times, feature-space rollout shift, matched
24/48/96-step Heun comparisons, a gate selected on a separate rollout bank,
and a fixed-budget angular-augmented invariant MLP. It additionally compares
the six-time regular-grid Ritz model with an 18-time stratified random-time
model at identical initialization, optimizer-step, and total-configuration
budgets. Rollout-adaptation readiness depends on reducing the on/off-grid Ritz gap by at
least 50% without lowering mean off-grid gain; tangent MMD is explicitly absent
from that criterion. These probes are secondary diagnostics and do not replace
or tune against the primary radial/Ritz-gated result. Interior-time MMD is the
primary path-law metric;
endpoint MMD is reported separately because the endpoint banks themselves are
given to the bridge.

## 6. Validation, ablations, and paper workflow

Validate the existing JAX kernels and the original two Tesseracts:

```bash
./scripts/run_tesseracts.sh --build
./scripts/run_validations.sh --backend jax
```

Run the paper-facing level-0 workflow:

```bash
./scripts/run_part0_paper.sh                   # Tesseract
./scripts/run_part0_paper.sh --backend jax
./scripts/run_part0_paper.sh --backend jax --quick
```

Run prescribed level-0 ablations:

```bash
./scripts/run_part0_ablations.sh
./scripts/run_part0_ablations.sh --quick
```

Run Experiment-B seed/evaluation uncertainty:

```bash
./scripts/run_multiseed_b.sh \
  --train-seeds "101 102 103 104 105" \
  --eval-seeds  "201 202 203 204 205"
```

Use `--backend jax` for the corresponding direct-JAX evaluation path. The
multi-seed commands are much more expensive than the level-2 schedule study.

## 7. Reproducibility and safety notes

- The level-2 time grid, quadrature grid, starting parameter, optimizer budget,
  ESS threshold, and finite-difference step are recorded in its JSON output.
- Level-2 uses deterministic quadrature and no random seed.
- The multiplier derivative is implicit; Newton iterates are not unrolled in
  the reported gradient.
- Tesseract and direct JAX execute the same `apply_jax` scientific recipe, and
  `compare_level2_backends.py` enforces numerical agreement.
- Quick mode is for plumbing checks. Use standard mode for figures/results.
- Generated results are written only below `results/level2_schedule/`.
- The existing A/B checkpoints, scripts, Tesseracts, and result directories are
  not read, rewritten, or deleted by the level-2 runner.
- The advanced suite is likewise isolated: it does not participate in
  `run_all.sh` or the Part-0 reproduction workflow unless invoked explicitly.
- The paper-facing study is also opt-in and writes only below
  `results/level2_paper_study/`; it does not load or modify level-0/1
  checkpoints or outputs.
