# MFSI execution architecture

## Backend policy

Tesseract Core is the default execution backend. Direct JAX is retained as the
reference implementation and regression backend.

Use a Tesseract when a scientific component owns a meaningful differentiable
map. Put the complete objective and reverse pass on the same side of the
component boundary. Do not create an RPC boundary around individual arithmetic
operations, optimizer updates, or ODE steps.

| Work | Boundary | Default |
|---|---|---|
| Neural model fitting | Ordinary training job | JAX |
| Reference/fiber forward evaluation | Context-free component map | Tesseract |
| Level-2 schedule adaptation | Complete objective and optimizer | Tesseract |
| Stage-3 rollout adaptation | Complete Heun rollout, loss, gradient, optimizer, selection | Tesseract |
| Stage-4 fiber design | Complete fiber construction, implicit I-projection gradient, optimizer, selection | Tesseract |
| Plotting, aggregation, intervals, provenance | Experiment orchestration | Python/JAX host |

The Tesseract recipes are JAX-backed. “Tesseract backend” therefore means the
same numerical recipe is built and served by the Pasteur/ISI Labs Tesseract Core
runtime across a real container boundary. It does not mean a different set of
scientific equations.

## Repository map

```text
backend_runtime.py             forward-component REST adapter
gradient_runtime.py            complete differentiable-objective adapter
backend_experiment_runner.py   backend-aware Stage 3/4 orchestration

tesseracts/
  reference_transport/         learned reference field
  moment_fiber_realizer/       I-projection, forcing, Ritz correction
  fiber_path_adapter/          scalar Level-2 gradient engine
  finite_neural_path/          finite-neural Level-2 gradient engine
  manybody_neural_path/        many-body Level-2 gradient engine
  paper_level2_correction/     frozen invariant correction evaluation
  rollout_gradient_engine/     end-to-end Stage-3 gradient engine
  fiber_gradient_engine/       end-to-end Stage-4 gradient engine

scripts/
  run_*                        user entry points
  build_*tesseract*.sh         image builders
  _run_*                       internal serve/teardown helpers

results/
  backend_smoke/               JAX/Tesseract gradient parity artifacts
  backend_runs/<study>/<mode>/ backend-separated new Stage 3/4 runs
  stage*/                      frozen historical/confirmatory artifacts
```

The original Stage 3/4 Python drivers are deliberately not rewritten. Stage 4B
records their hashes as confirmatory provenance. Backend-aware wrappers replace
only the optimizer call at runtime and write to `results/backend_runs/` by
default, keeping the archived confirmation intact.

## Commands

Build and compare the complete gradient engines:

```bash
./scripts/build_gradient_tesseracts.sh
./scripts/run_gradient_smoke_both.sh
```

Run a backend-aware gradient experiment. Tesseract is the default:

```bash
./scripts/run_stage3_rollout_adaptation.sh
./scripts/run_stage4_fiber_design.sh
./scripts/run_stage4b_confirmatory.sh
```

Select direct JAX explicitly:

```bash
./scripts/run_stage4_fiber_design.sh --backend jax
```

Use `--legacy-output` only for an intentional regression of the historical
result paths. Confirmatory reruns should use the original frozen Python driver
so its provenance remains literal.

## Parity rule

JAX and Tesseract smoke inputs are generated on CPU and use pinned JAX, NumPy,
SciPy, and Matplotlib versions. The gate compares complete optimization traces,
gradient norms, candidate parameters, selection objectives, selected
checkpoints, and selected parameters. The maximum absolute difference must not
exceed `2e-9`.
