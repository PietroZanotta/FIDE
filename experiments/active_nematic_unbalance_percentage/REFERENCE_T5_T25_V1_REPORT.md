# Active-nematic endpoint reference study: physical t=5 to t=25

## Status

REFERENCE TRAINING COMPLETE

This is an isolated development reference study. It does not replace the
official physical-t=21 to t=31 references, does not modify the frozen Pareto
inputs, and has not accessed Pareto validation data.

## Prospective change

The study derives its configuration from `config_more_training_v2.json`. The
only scientific changes are:

- experiment name;
- physical saved endpoints: `[5, 25]` instead of `[21, ..., 31]`;
- defect-population endpoints: `[5, 25]` instead of `[21, ..., 31]`.

The configuration loader additionally records `smoke: false`; this is execution
metadata, not a scientific change. Physics, all 96 deterministic realization
seeds, the explicit 64/16/16 split, defect extraction, reference architecture,
optimizer, endpoint sampling, KDE quadrature, and learned-reference seeds are
unchanged.

## Endpoint feasibility

Physical t=0 was rejected because all 96 realizations contain exactly zero
accepted plus and minus defects. Physical t=5 is supported:

| species | training mass t=5 | training mass t=25 | training defects t=5 | training defects t=25 | nonempty training runs t=5 / t=25 |
|---|---:|---:|---:|---:|---:|
| plus | 2.78125 | 11.65625 | 178 | 746 | 60 / 64 |
| minus | 2.78125 | 11.65625 | 178 | 746 | 60 / 64 |

Across all 96 runs, each species has mean mass 2.8125 at t=5 and 11.739583 at
t=25. Plus/minus counts match exactly, and the charge-balance audit passes.

## Reference implementation

Three base reference seeds were trained: 20260818, 20260819, and 20260820.
Each base seed produces independent plus and minus flows, for six checkpoints
in total. Every checkpoint declares physical interval `[5.0, 25.0]`, endpoint-
only training, and no use of intermediate physical marginals.

The unchanged training setup is:

- periodic state `(x, y, beta)` with periods `(32, 32, 2 pi)`;
- four hidden layers of width 128;
- 12,000 Adam steps, batch size 2,048;
- learning rate 0.001 with cosine decay to 5%;
- gradient clipping at norm 10;
- linear shortest-periodic-arc bridge with noise standard deviation 0.15;
- 50,000 empirical endpoint particles per species;
- 8,192 rollout-bank particles per species;
- spatial KDE jitter 1.0 and beta jitter 0.25;
- 16 RK4 substeps over the normalized reference interval;
- separate Fisher--Rao mass schedule from 2.78125 to 11.65625 for each species.

The generalization of `normalized_times` maps the first and last declared
physical times to 0 and 1. Regression tests verify that it produces exactly the
historical normalization for 21, 26, 31 and the prospective normalization for
5, 15, 25.

## Endpoint-law results

The audit uses the production 48 x 48 x 24 periodic histogram and the declared
multiscale periodic MMD-squared kernel.

| seed | species | initial MMD2 | target MMD2 | no-transport target MMD2 | target reduction vs no transport |
|---:|---|---:|---:|---:|---:|
| 20260818 | plus | 0.00175279 | 0.00195127 | 0.00339644 | 42.55% |
| 20260818 | minus | 0.00182905 | 0.00200703 | 0.00331455 | 39.45% |
| 20260819 | plus | 0.00175279 | 0.00167474 | 0.00339644 | 50.69% |
| 20260819 | minus | 0.00182905 | 0.00205826 | 0.00331455 | 37.90% |
| 20260820 | plus | 0.00175279 | 0.00171643 | 0.00339644 | 49.46% |
| 20260820 | minus | 0.00182905 | 0.00192259 | 0.00331455 | 42.00% |

All six learned flows improve the target endpoint relative to leaving the
initial reference distribution unmoved. The improvement range is 37.90% to
50.69%, with a six-flow mean of 43.68%. Target error remains of the same order
as the initial KDE-to-empirical representation error.

The paper visualization's descriptive smoothed-histogram overlaps are 0.65 to
0.69 for target spatial marginals. Target beta overlap is 0.65 to 0.67 for plus
and 0.82 to 0.83 for minus. Thus the weakest resolved component is the plus-
species target orientation marginal. These overlaps are visual diagnostics,
not replacements for the production MMD audit.

Final logged conditional flow-matching losses range from 139.729 to 151.794,
down from first logged values of 173.737 to 177.241. Because those losses use
different stochastic minibatches, endpoint MMD is the primary fit diagnostic.

## Interpretation

The t=5 to t=25 reference problem is feasible and every trained seed learns a
meaningful endpoint transport. It is harder than the t=21 to t=31 stationary-
regime problem: t=5 has a smaller and sparser defect population, mass grows by
more than fourfold, and plus-orientation fit is visibly weaker. These references
are suitable for isolated development analysis but should not replace the
official references without a separate prospectively frozen downstream study.

The learned intermediate rollout is a reference bridge. It must not be
interpreted as a prediction of intermediate physical marginals, which were
neither supplied nor accessed during training.

## Reproduction and artifacts

Run or resume the complete workflow with:

```bash
.venv/bin/python \
  experiments/active_nematic_unbalance_percentage/train_reference_endpoints.py \
  --start 5 --end 25 \
  --output experiments/active_nematic_unbalance_percentage/outputs/reference_t5_t25_v1
```

A completed replay takes approximately 4.4 seconds because it verifies and
reuses the physical bank, defect bank, and six individual flow checkpoints.

Primary artifacts:

- `outputs/reference_t5_t25_v1/effective_config.json`
- `outputs/reference_t5_t25_v1/provenance.json`
- `outputs/reference_t5_t25_v1/endpoint_support.json`
- `outputs/reference_t5_t25_v1/defect_bank_audit.json`
- `outputs/reference_t5_t25_v1/reference_manifest.json`
- `outputs/reference_t5_t25_v1/reference_endpoint_audit.json`
- `outputs/reference_t5_t25_v1/reference_seed_*/{plus,minus}_reference.npz`
- `outputs/reference_t5_t25_v1/reference_seed_*/{plus,minus}_reference_bank.npz`
- `figures/active_nematic_reference_endpoint_fit_t5_t25_v1.png`
- `figures/active_nematic_reference_endpoint_fit_t5_t25_v1.pdf`

NO intermediate-truth training

NO Pareto validation access

NO official reference replacement
