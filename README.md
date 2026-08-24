# MFSI sensor-design experiments

Research code for measurement-conditioned flow sensor design. The repository
compares population, finite-law, tangent, and full correction objectives while
keeping endpoint-trained reference dynamics and validation data frozen.

The reusable JAX-first implementation lives in `src/mfsi`; each scientific
system is an adapter under `experiments`. Optional C++/OpenMP Tesseract modules
accelerate information projection and weighted-Poisson solves.

## Experiments

| Experiment | System | Documentation |
|---|---|---|
| Toy percentage-risk sweep | Two sensors on an analytic Gaussian-mixture path | [README](experiments/toy_example_percentage/README.md) |
| Vortices percentage-risk sweep | Four sensors in a time-dependent double gyre | [README](experiments/vortices_percentage/README.md) |
| Active-nematic unbalance sweep | Sensors for an unbalanced active-nematic model | [README](experiments/active_nematic_unbalance_percentage/README.md) |
| Skyrmion Deep Ritz sweep | Four sensors for a 16-particle, 32-D periodic system | [README](experiments/skyrmions_deep_ritz/README.md) |

The toy and vortices directories include their selected, publication-facing
result artifacts. The skyrmion directory includes the authoritative Pareto
summaries, the standalone 3% validation record, and publication figures; large
checkpoints, frozen banks, search caches, and rerun products are intentionally
excluded.

## Repository layout

```text
src/mfsi/       reusable flow, projection, reconstruction, action, and I/O code
experiments/    scientific systems, configurations, runners, and selected results
native/         optional Tesseract C++/OpenMP backends
tests/          unit and integration tests
visual_abstract/ scripts and rendered assets for the visual summary
```

## Installation

Python 3.11 or newer is required. Create an isolated environment and install the
package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install experiment-specific dependencies as needed:

```bash
python -m pip install -e '.[skyrmions,active-nematic]'
python -m pip install -e '.[tesseract-cpp]'
```

JAX accelerator installation is platform-specific; follow the JAX guidance for
the CUDA or CPU build appropriate to the machine.

## Tests and smoke runs

Run the test suite from the repository root:

```bash
python -m pytest -q
```

Each active experiment has a smoke profile:

```bash
python experiments/toy_example_percentage/run.py --smoke
python experiments/vortices_percentage/run.py --smoke
python experiments/active_nematic_unbalance_percentage/run.py --smoke
python experiments/skyrmions_deep_ritz/run.py --smoke
```

Full scientific runs are substantially more expensive. Read the experiment
README and inspect its checked-in `config.json` before launching one.

## Native accelerators

The native projects are optional and retain Python/JAX fallbacks where the
experiment supports them. Build instructions and contracts are documented in:

- [batched I-projection](native/iprojection_tesseract/README.md)
- [weighted Poisson](native/poisson_tesseract/README.md)
- [active-nematic 3-D Poisson](native/active_nematic_poisson3d_tesseract/README.md)
- [active-nematic screened Poisson](native/active_nematic_unbalanced_screened_tesseract/README.md)
- [variational Poisson](native/variational_poisson_tesseract/README.md)

Generated environments, native build trees, caches, logs, archived scratch
directories, and uncurated experiment outputs are excluded by `.gitignore`.
