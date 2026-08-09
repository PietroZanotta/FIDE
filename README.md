# Moment-Fiber Stochastic Interpolants (MFSI)

Runnable research repository for the two low-dimensional experiments in the current MFSI draft.

The repository supports **two execution backends for the learned MFSI component maps**:

- `tesseract` — **default**. Uses the Pasteur/ISI Labs **Tesseract Core** tool: the two projects in `tesseracts/` are built with `tesseract build`, served as Docker containers with `tesseract serve`, and called over their REST `/apply` endpoints.
- `jax` — executes the same JAX kernels directly in-process. This is useful for development, debugging, and faster local runs.

There is no Tesseract Python-SDK orchestration in the experiment path.

> **Training vs backend execution.** Neural-network optimization (flow matching and Deep-Ritz AdamW/L-BFGS) remains native JAX. The backend flag controls execution of the two scientific maps used in generation/evaluation: `ReferenceTransport` and `MomentFiberRealizer`. This keeps the Tesseracts context-free scientific components rather than turning them into training jobs.

The learned deterministic pipeline is

```text
endpoint samples
    -> flow-matched reference velocity u_theta(t,x)
    -> ordinary stochastic-interpolant particles
    -> empirical I-projection / lambda_t
    -> C_t, lambda_dot_t, h_t
    -> Deep-Ritz scalar potential psi_omega(t,x)
    -> v_t = u_theta - grad_x psi_omega
    -> optional population-moment safety correction
    -> ODE generation
```

## Quick start

### 1. Install dependencies

```bash
./scripts/install.sh
```

This creates `.venv/` and installs JAX plus Tesseract Core. Docker is a separate system requirement for the default Tesseract backend.

For a JAX-only development environment:

```bash
./scripts/install.sh --jax-only
```

### 2. Build the two Tesseracts

```bash
./scripts/build_tesseracts.sh
```

This runs Tesseract Core's CLI build on

```text
tesseracts/reference_transport/
tesseracts/moment_fiber_realizer/
```

and produces

```text
mfsi-reference-transport:latest
mfsi-moment-fiber-realizer:latest
```

The experiment shell wrappers will also build the images automatically when `--backend tesseract` is selected and the images are missing.

### 3. Run the experiments

Tesseract is the default:

```bash
./scripts/run_example_a.sh
./scripts/run_example_b.sh
```

Explicitly select a backend with

```bash
./scripts/run_example_a.sh --backend tesseract
./scripts/run_example_b.sh --backend tesseract

# direct in-process reference implementation
./scripts/run_example_a.sh --backend jax
./scripts/run_example_b.sh --backend jax
```

Run the full workflow:

```bash
./scripts/run_all.sh                       # Tesseract backend
./scripts/run_all.sh --backend jax         # direct JAX backend
```

Run **Experiments A and B and then print all main result tables directly in the terminal**:

```bash
./scripts/run_experiments_and_report.sh              # default: Tesseract backend
./scripts/run_experiments_and_report.sh --backend jax
```

Useful variants:

```bash
# retrain both experiments, then report
./scripts/run_experiments_and_report.sh --retrain

# quick/debug B run but normal A, then report
./scripts/run_experiments_and_report.sh --backend jax --quick-b

# do not rerun anything; only format the current results/ files
./scripts/run_experiments_and_report.sh --report-only

# equivalent report-only convenience command
./scripts/show_results.sh --backend jax
```

The terminal report includes the Experiment-A matched benchmark, A learned-pipeline diagnostics and validation gates, the Experiment-B projected-law benchmark, B held-out projection/Deep-Ritz diagnostics, and an available multi-seed aggregate.

## What “Tesseract backend” means here

The shell wrapper starts the two built Tesseract Core images with persistent `tesseract serve` processes, exports their local REST addresses, runs the experiment, and tears the containers down afterward.

Conceptually:

```text
experiment Python process
      |
      | POST /apply
      v
+----------------------------+
| mfsi-reference-transport   |
| Tesseract Core container   |
+----------------------------+
      |
      | u_theta(t,x)
      v
+----------------------------+
| mfsi-moment-fiber-realizer |
| Tesseract Core container   |
+----------------------------+
      |
      | projected state / corrected velocity
      v
experiment orchestration
```

The Tesseract backend therefore exercises the **actual container boundary**. It does not import the Tesseract API files into the experiment process and does not use `Tesseract.from_*`.

Because these toy neural kernels are very small, REST/container overhead can dominate runtime. Use `--backend jax` when benchmarking the numerical method itself; use `--backend tesseract` when validating the intended componentized execution/deployment path.

Ports can be overridden if needed:

```bash
MFSI_REFERENCE_TESSERACT_PORT=19081 \
MFSI_FIBER_TESSERACT_PORT=19082 \
./scripts/run_example_b.sh
```

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── requirements-tesseract.txt
│
├── mfsi_components.py
│   Core 1D MFSI mathematics and reusable learned utilities.
│
├── example_b.py
│   Complete 2D Experiment-B definition, training and benchmark.
│
├── backend_runtime.py
│   Backend adapter. In Tesseract mode it talks to the two served
│   Tesseract Core containers over HTTP; in JAX mode the experiment uses
│   direct kernels.
│
├── mgd.py
│   Interacting-particle Moment Guided Diffusion implementation.
│
├── validate_pipeline.py
│   Experiment-A learned-pipeline validation using future-compatible criteria.
│
├── benchmark_methods.py
│   Experiment-A matched benchmark. Learned MFSI generation can run through
│   Tesseract Core or direct JAX.
│
├── validate_mgd.py
│   Independent MGD validation.
│
├── validate_tesseracts.py
│   Pure-JAX kernel parity plus real `tesseract run` container checks when the
│   images are available. No Python SDK route.
│
├── ablate_and_benchmark.py
│   Implicit/unrolled/stop-gradient checks and local JAX microbenchmarks.
│
├── tesseracts/
│   ├── reference_transport/
│   │   ├── tesseract_api.py
│   │   ├── tesseract_config.yaml
│   │   └── tesseract_requirements.txt
│   └── moment_fiber_realizer/
│       ├── tesseract_api.py
│       ├── tesseract_config.yaml
│       └── tesseract_requirements.txt
│
├── scripts/
│   ├── install.sh
│   ├── build_tesseracts.sh
│   ├── run_example_a.sh
│   ├── run_example_b.sh
│   ├── run_experiments_and_report.sh
│   ├── show_results.sh
│   ├── report_results.py
│   ├── run_multiseed_b.sh
│   ├── run_tesseracts.sh
│   ├── run_mgd_validation.sh
│   ├── run_ablations.sh
│   ├── run_validations.sh
│   ├── run_all.sh
│   ├── reset_checkpoints.sh
│   ├── sweep_example_b.py
│   ├── _run_with_backend.sh
│   └── _common.sh
│
├── checkpoints/
│   ├── example_a.npz
│   └── example_b.npz
│
└── results/
    ├── reference/        provenance snapshot
    ├── example_b/        active Experiment-B outputs
    ├── multiseed/        robustness sweeps
    └── ...
```

## Installation

### Normal installation: JAX + Tesseract Core

```bash
./scripts/install.sh
```

Choose another Python executable/version:

```bash
./scripts/install.sh --python 3.12
./scripts/install.sh --python /path/to/python
```

### JAX only

```bash
./scripts/install.sh --jax-only
```

Manual equivalent:

```bash
pip install -r requirements.txt
pip install -r requirements-tesseract.txt
```

Tesseract container builds additionally require a working Docker engine.

## Build / inspect the Tesseracts

Build both images:

```bash
./scripts/build_tesseracts.sh
```

Request a cache-free build when supported by the installed Tesseract Core release:

```bash
./scripts/build_tesseracts.sh --no-cache
```

Check direct-kernel parity and, if the images are available, actual container execution:

```bash
./scripts/run_tesseracts.sh
```

Build first, then validate:

```bash
./scripts/run_tesseracts.sh --build
```

`validate_tesseracts.py` invokes built images with `tesseract run ... apply ...`; it does not use the Python SDK.

## Experiment A

Evaluate the current checkpoint and run the matched benchmark using the default Tesseract backend:

```bash
./scripts/run_example_a.sh
```

Direct JAX:

```bash
./scripts/run_example_a.sh --backend jax
```

Retrain from scratch, then evaluate:

```bash
./scripts/run_example_a.sh --retrain
```

Continue training with oracle-free holdout selection:

```bash
./scripts/run_example_a.sh --refine
```

Validation only:

```bash
./scripts/run_example_a.sh --validate-only
```

The training/validation phase is native JAX. The backend flag is used by the learned-component generation benchmark.

Experiment A is the exact low-dimensional diagnostic problem. Learned-model acceptance uses quantities that remain available later:

- held-out stochastic-interpolant / flow-matching regression;
- fresh empirical moment calibration;
- ESS, covariance rank and conditioning;
- implicit derivative vs finite difference;
- held-out Deep-Ritz weak-form residual;
- generated-vs-independently-projected sample discrepancy;
- generated population-moment drift.

The exact 1D path remains debug/reporting information and is not a learned-model selection criterion.

## Experiment B

Default Tesseract backend:

```bash
./scripts/run_example_b.sh
```

Direct JAX:

```bash
./scripts/run_example_b.sh --backend jax
```

Retrain and evaluate:

```bash
./scripts/run_example_b.sh --retrain --seed 20260808
```

Short debug run:

```bash
./scripts/run_example_b.sh --backend jax --quick --seed 123
```

Skip plots:

```bash
./scripts/run_example_b.sh --no-plots
```

Experiment B uses two smooth distributions in `R^2` with the same population mean and covariance:

- `Q_minus`: centers continuously uniform on a ring plus isotropic Gaussian thickness;
- `Q_plus`: four axis-aligned lobe centers plus the same Gaussian thickness.

Measured observables are

```text
Phi(x,y) = (x, y, x^2, xy, y^2)
target   = (0, 0, 1, 0, 1)
```

Angular Fourier features are evaluation-only diagnostics. At every evaluation time the target law is estimated from a fresh stochastic-interpolant bank followed by a fresh empirical I-projection.

The comparison contains raw SI, moment tangent, MGD-style guidance, learned MFSI, and learned MFSI plus the optional population safety layer.

## Multiple training seeds × multiple evaluation seeds

Recommended paper-facing robustness command:

```bash
./scripts/run_multiseed_b.sh \
  --train-seeds "101 102 103 104 105" \
  --eval-seeds  "201 202 203 204 205"
```

This defaults to the **Tesseract backend for evaluation**. Training is still ordinary native-JAX neural optimization.

Run the same Cartesian product with direct JAX evaluation:

```bash
./scripts/run_multiseed_b.sh \
  --backend jax \
  --train-seeds "101 102 103 104 105" \
  --eval-seeds  "201 202 203 204 205"
```

Short debug sweep:

```bash
./scripts/run_multiseed_b.sh \
  --backend jax \
  --train-seeds "101 102" \
  --eval-seeds  "201 202" \
  --quick
```

Plumbing-only sweep:

```bash
./scripts/run_multiseed_b.sh \
  --backend jax \
  --train-seeds "101 102" \
  --eval-seeds  "201 202" \
  --smoke
```

The sweep is resumable by default and stores each checkpoint/evaluation pair separately. The default output directory is backend-specific (`.../tesseract/` or `.../jax/`) so cached evaluations from one backend are never mistaken for the other:

```text
results/multiseed/example_b/<backend>/
├── config.json
├── aggregate.json
├── aggregate.csv
├── per_run.csv
├── train_<seed>/
│   ├── model.npz
│   ├── training.json
│   ├── eval_<seed>/example_b_results.json
│   └── ...
└── ...
```

Use `--force` to recompute completed pairs.

## Terminal result tables

Run both low-dimensional experiments and finish with terminal tables:

```bash
./scripts/run_experiments_and_report.sh
```

The command uses the same backend convention as the individual experiment wrappers: `tesseract` by default, or `--backend jax`. It supports `--retrain`, `--retrain-a`, `--refine-a`, `--retrain-b`, `--quick-b`, `--no-plots`, and `--seed N`.

To print the current result files without rerunning the experiments:

```bash
./scripts/show_results.sh
./scripts/show_results.sh --backend jax
```

The reporting code is `scripts/report_results.py` and uses only the Python standard library. It reads the CSV/JSON files already produced by the experiments, so the displayed numbers are exactly the persisted experiment outputs rather than a second computation. If a multi-seed Example-B aggregate exists under `results/multiseed/example_b/<backend>/`, it is printed automatically. The packaged provenance aggregate under `results/reference/example_b/` is used as a fallback when no active sweep exists. Use `--no-multiseed` to omit that table.

## Other commands

Independent MGD validation:

```bash
./scripts/run_mgd_validation.sh
```

Differentiation ablations and JAX microbenchmarks:

```bash
./scripts/run_ablations.sh
```

Validation bundle:

```bash
./scripts/run_validations.sh
```

Run everything:

```bash
./scripts/run_all.sh
./scripts/run_all.sh --backend jax
./scripts/run_all.sh --retrain
```

## Tesseract contracts

### Tesseract 1 — `reference_transport`

```text
(x, t, flattened reference-network parameters)
    -> u_theta(t,x)
```

### Tesseract 2 — `moment_fiber_realizer`

```text
x
u_theta(t,x)
Phi(x)
J_Phi(x) u_theta(t,x)
target moments
base log weights
t
flattened Deep-Ritz parameters
    -> lambda_t
       projected weights
       moments / covariance
       lambda_dot_t
       h_t
       -grad psi_omega
       corrected velocity
       ESS / calibration / rank / condition diagnostics
```

`Phi(x)` and `J_Phi(x)u` cross the Tesseract boundary as inputs rather than being hard-coded, so the same two projects support Experiments A and B and can be reused for later observable families.

The calibration derivative uses an implicit linearization rather than backpropagating through every Newton iteration. The JAX-backed Tesseract API exposes apply/JVP/VJP/Jacobian endpoints through Tesseract Core's runtime recipes.

## Checkpoints and results

Packaged models:

```text
checkpoints/example_a.npz
checkpoints/example_b.npz
```

Restore the packaged models to the active result locations with

```bash
./scripts/reset_checkpoints.sh
```

`results/reference/` is a provenance snapshot only; active experiments do not read it.

## Reproducibility notes

- JAX 64-bit mode is enabled in the numerical scripts.
- Shell entry points set a non-interactive Matplotlib backend.
- Training and evaluation seeds are explicit.
- Multi-seed B keeps each training checkpoint and every evaluation result isolated.
- Learned-model selection uses flow-matching/Deep-Ritz holdouts and future-available diagnostics, not the Example-A oracle.
- The backend is recorded in Experiment-A/Experiment-B result JSON and multi-seed sweep configuration.
- Tesseract mode defaults to ports `18081` and `18082`; override them with environment variables if needed.

---

## Part-0 paper reproduction workflow

The low-dimensional paper program (Experiments A and B, prescribed ablations,
uncertainty reporting, figures, and the Tesseract systems table) can be run with:

```bash
./scripts/run_part0_paper.sh
```

The default scientific component backend is the **Pasteur/ISI Labs Tesseract
Core** backend. Use direct JAX explicitly when desired:

```bash
./scripts/run_part0_paper.sh --backend jax
```

A shorter debugging pass is:

```bash
./scripts/run_part0_paper.sh --backend jax --quick
```

`--quick` is only a plumbing/debug budget. Do not quote its neural-ablation or
multi-seed numbers as paper results.

For independent training-seed and evaluation-seed uncertainty in Experiment B:

```bash
./scripts/run_part0_paper.sh \
  --multiseed \
  --train-seeds "101 102 103 104 105" \
  --eval-seeds  "201 202 203 204 205"
```

The sweep reports three distinct uncertainty summaries:

1. descriptive variability across all train/evaluation pairs;
2. variability across **training-seed means** after averaging evaluation seeds;
3. variability across **evaluation-seed means** after averaging training seeds.

The paper-facing uncertainty interval is the training-seed interval. The sweep
also reports paired MFSI-safe minus moment-tangent differences on identical
train/evaluation pairs.

### New Part-0 outputs

Experiment A now writes:

- `results/learned_validation.json`: includes the empirical-versus-quadrature
  checks for `lambda`, `lambda_dot`, and the forcing `h`;
- `results/example_a_component_validation.png`: component-level oracle figure;
- `results/example_a_density_overlay.png`: central-time density overlay against the projected target;
- `results/method_benchmark_metrics.json`: includes integrated learned correction
  energy, integrated projection distortion, and explicit NFE/step counts;
- `results/method_benchmark.png`: law-path/fourth-moment benchmark figure.

Experiment B now writes:

- `results/example_b/example_b_results.json`: includes time-resolved correction
  energy and projection distortion plus integrated path functionals;
- `results/example_b/benchmark_summary.csv`: includes NFE and integration steps;
- `results/example_b/fiber_profile.png`: measured moments stay flat while held-out
  angular descriptors move;
- `results/example_b/projection_diagnostics.png`: correction energy, projection
  distortion, ESS, and covariance conditioning;
- `results/example_b/path_mmd.png`: projected-law two-sample discrepancy;
- `results/example_b/snapshots_t075.png`: representative generated populations.

A multi-seed sweep additionally writes `uncertainty_mmd.png` in its output directory.

Prescribed ablations are produced by:

```bash
./scripts/run_part0_ablations.sh          # full paper-facing budgets
./scripts/run_part0_ablations.sh --quick  # implementation smoke/debug only
```

Outputs:

- `results/ablation_metrics.json`: stop/unrolled/implicit and bridge-design tests;
- `results/part0_ablations/part0_ablations.json`;
- `results/part0_ablations/part0_ablations.png`.

The Part-0 ablation report covers correction-network capacity, Ritz batch size,
covariance-rank truncation, reference coupling, stochastic-interpolant noise,
safety-layer use, and stop-gradient versus implicit differentiation.

### Tesseract systems table

Build the two actual Pasteur/ISI Labs Tesseract Core images with:

```bash
./scripts/build_tesseracts.sh
```

Then benchmark the real CLI/Docker execution path with:

```bash
./scripts/run_tesseract_systems.sh
```

To include image rebuild time:

```bash
./scripts/run_tesseract_systems.sh --rebuild
```

The systems benchmark measures image size, one-shot `tesseract run` latency,
`tesseract serve` startup, first served `/apply`, warm median `/apply`, p95
latency, and numerical parity. These systems timings are kept separate from the
scientific local-JAX wall-clock comparison because HTTP/container overhead is a
different question from algorithmic compute.

If Tesseract Core or Docker is unavailable, the script writes an explicit
`available: false` result instead of substituting SDK or JAX timing numbers.

### Reporting only

After any subset of runs, print all available tables with:

```bash
./scripts/show_results.sh --backend jax
# or
./scripts/show_results.sh --backend tesseract
```

The reporter includes A/B benchmark tables, empirical-vs-quadrature A checks,
correction energy, projection distortion, NFE, held-out Ritz/projection
statistics, multi-seed uncertainty, prescribed ablations, and the Tesseract
systems table when available.
