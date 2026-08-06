# Many-body scientific comparison package

This repository implements an exact periodic homometric benchmark and compares
observation-only inverse methods with population-informed conditional equivariant
flow matching.

## Methods

### Observation-only track

- Reverse Monte Carlo using the configured smooth pair-moment condition.
- Iterative Boltzmann inversion using a disclosed Gaussian-smoothed radial pair
  histogram. This target is richer than the RBF condition and is never presented
  as an equal-information comparison.

### Population-informed track

- Conditional equivariant flow matching with soft moment and physical penalties,
  without solver differentiation.
- The same soft model followed by standardized Post-hoc repair.
- Full-E2E flow matching fine-tuned through relaxation and projection.
- Base/Post-hoc/Relax-E2E/Full-E2E as a mechanistic gradient-routing ablation.

## Higher-order conditional UQ

Every method is evaluated on the predictive distribution of held-out angular
features conditional on the one shared pair vector. Reports include predictive
intervals, coverage, interval scores, multivariate energy score, A/B/Far
probability intervals, mode-calibration error, and entropy. Multi-seed aggregation
adds aleatoric/epistemic variance decomposition.

See `HIGHER_ORDER_CONDITIONAL_UQ.md` and `METHOD_COMPARISON_PROTOCOL.md`.

## Install

The target environment is pinned to JAX 0.8.1:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

For the user's CUDA JAX installation:

```bash
python -m pip install -e '.[cuda,test]'
```

## Complete bounded smoke setup

The reliable quick smoke reuses the included validated flow artifact, then reruns
RMC, IBI, standardized repair, higher-order UQ, the comparison report, and the
independent composed-gradient probe:

```bash
./scripts/run_whole_scientific_setup.sh
```

To retrain a deliberately tiny flow and every gradient route from scratch:

```bash
./scripts/run_from_scratch_acceptance.sh
```

The from-scratch acceptance run is intended for CUDA or a patient CPU host; nested
JAX solver compilation can dominate wall-clock time even at tiny array sizes.

Outputs are written to `artifacts/scientific_comparison_smoke/`:

- `scientific_comparison_report.json`;
- `scientific_comparison_summary.csv`;
- `scientific_comparison_arrays.npz`;
- `SCIENTIFIC_COMPARISON_SUMMARY.md`;
- `flow_ablation/` with the complete routing ablation artifacts;
- `composed_gradient_probe.json`.

## Registered multi-seed study

After checking the smoke output:

```bash
./scripts/run_registered_scientific_study.sh
```

The registered configuration uses eight independently trained seeds. Final method
claims use training seeds—not replicas or generated ensembles—as inferential units.
The aggregate report includes seed-level primary effects and higher-order
aleatoric/epistemic decomposition.

## Current smoke interpretation

The included smoke run favors Full-E2E over soft/Post-hoc flow in the point
estimates for repair burden, projected pair error, mode balance, and higher-order
reference distance. Its repair confidence interval crosses zero, so this is not a
superiority result. RMC matches the reduced pair condition effectively but does not
recover the registered mode population; IBI likewise cannot identify the homometric
mixture from pair information.
