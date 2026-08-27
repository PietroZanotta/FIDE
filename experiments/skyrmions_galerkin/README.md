# Skyrmion B1 FIDE/MFSI experiment

This directory contains the completed official B1 Galerkin Pareto study for the
skyrmion benchmark. The current documentation is intentionally small:

- [`OFFICIAL_B1_GALERKIN_PARETO_V1_COMPLETE_STUDY.md`](OFFICIAL_B1_GALERKIN_PARETO_V1_COMPLETE_STUDY.md) is the comprehensive implementation, results, interpretation, audit, and reproduction record.
- [`OFFICIAL_B1_GALERKIN_PARETO_V1_PROTOCOL.md`](OFFICIAL_B1_GALERKIN_PARETO_V1_PROTOCOL.md) is the prospectively frozen protocol.
- [`outputs/official_b1_galerkin_pareto_v1/report.md`](outputs/official_b1_galerkin_pareto_v1/report.md) is the compact generated run report.
- [`outputs/official_b1_galerkin_pareto_v1/final_summary.json`](outputs/official_b1_galerkin_pareto_v1/final_summary.json) is the machine-readable final summary.
- [`outputs/official_b1_galerkin_pareto_v1/selection/selection_manifest.json`](outputs/official_b1_galerkin_pareto_v1/selection/selection_manifest.json) is the sealed selection manifest.

The official run selected a frozen 6% candidate and returned `PASS`. All 18
validation rows passed both the preregistered strict gate and the diagnostic
`p+5pp` gate. The outcome is scientifically valid for the frozen B1 protocol;
its limitations and the observed Full-versus-Law response plateau are discussed
in the complete study record.

## Reproduction entry points

Run from the repository root with the project environment:

```bash
.venv/bin/python -m experiments.skyrmions_galerkin.official_b1_pareto_run --mode freeze-protocol
.venv/bin/python -m experiments.skyrmions_galerkin.official_b1_pareto_run --mode generate-data
.venv/bin/python -m experiments.skyrmions_galerkin.official_b1_pareto_run --mode law
.venv/bin/python -m experiments.skyrmions_galerkin.official_b1_pareto_run --mode candidates
.venv/bin/python -m experiments.skyrmions_galerkin.official_b1_pareto_run --mode screen
.venv/bin/python -m experiments.skyrmions_galerkin.official_b1_pareto_run --mode tangent
.venv/bin/python -m experiments.skyrmions_galerkin.official_b1_pareto_run --mode full
.venv/bin/python -m experiments.skyrmions_galerkin.official_b1_pareto_run --mode cross
.venv/bin/python -m experiments.skyrmions_galerkin.official_b1_pareto_run --mode freeze-selection
.venv/bin/python -m experiments.skyrmions_galerkin.official_b1_pareto_run --mode generate-validation
.venv/bin/python -m experiments.skyrmions_galerkin.official_b1_pareto_run --mode validate
.venv/bin/python -m experiments.skyrmions_galerkin.official_b1_pareto_run --mode report
```

Focused tests:

```bash
.venv/bin/python -m pytest -q \
  experiments/skyrmions_galerkin/test_official_b1_pareto.py \
  experiments/skyrmions_galerkin/test_final_b1_support_confirmation.py \
  experiments/skyrmions_galerkin/test_single_reference_b1_preflight.py
```

Historical and superseded study documents are preserved, without modification,
under [`old_stuff/`](old_stuff/). Production code, sealed artifacts, and output
directories remain in their original locations.
