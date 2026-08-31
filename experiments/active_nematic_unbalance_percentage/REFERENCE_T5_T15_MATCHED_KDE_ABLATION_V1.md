# Active-nematic t=5→15 matched-KDE reference ablation

## Status and scope

This is a development-only, endpoint-only reference-flow ablation. It changes
one semantic choice: the distribution used for the initial endpoint during
conditional flow matching. It does not run sensor design, Law, Tangent, Full,
or validation evaluation, and it does not modify or resume the paused Pareto
study.

The result is strong evidence that the poor learned target endpoint was not
caused by the periodic bridge, Fisher–Rao mass interpolation, RK4 rollout, MMD
implementation, or plotting code. The dominant issue was a train/rollout
distribution mismatch at the initial endpoint.

## Scientific question

The original reference flow was trained on an unjittered empirical initial
endpoint law but was rolled out from a periodic KDE law with spatial jitter
standard deviation 1.0 and beta jitter standard deviation 0.25. Therefore the
network was evaluated from an initial distribution it had not seen during
training.

This ablation asks whether the target fit improves when training and rollout
use the same initial density semantics, holding the physical data, target
samples, architecture, optimizer, loss, seeds, endpoint interval, and rollout
particles fixed.

## Frozen design

| Field | Empirical-source baseline | Matched-KDE arm |
|---|---|---|
| Physical interval | t=5→15 | t=5→15 |
| Physical realizations | same 96-run bank | same 96-run bank |
| Training split | same 64 runs | same 64 runs |
| Reference seeds | 20260818, 20260819, 20260820 | same |
| Species | plus and minus | same |
| Training initial law | empirical resample | periodic KDE resample |
| Training target law | empirical resample | bitwise-identical samples |
| Rollout initial law | periodic KDE | periodic KDE |
| Rollout initial particles | frozen | bitwise identical for all six pairs |
| Architecture, optimizer, CFM loss | frozen | unchanged |
| Intermediate truth | not used | not used |
| Validation | not accessed | not accessed |

The physical bank SHA-256 is
`8d19e44a3431769d6ca04083edaa0180b714a441cc6f1735d6c26b9a43b3b811` in
both arms. The extracted two-species defect bank SHA-256 is
`1733bf01b3d982cc34b00fd17c98b1084b18460cd9e33e53854f7818f08e01a8`
in both arms. Both files were hard-linked from the baseline output after
compatibility validation, so no physical simulation or defect extraction was
repeated.

The matched source is constructed by first reproducing the original empirical
endpoint source with the original seed, retaining its target samples, and then
replacing only its initial samples with an independent draw from the declared
periodic KDE. The comparison script reconstructs both sources and confirms
that both plus and minus target arrays are bitwise identical while their
initial arrays differ.

## Implementation

`run.py` now supports
`reference_training.initial_endpoint_density_model`, with `empirical` as the
backward-compatible default and `periodic_kde` as the prospective alternative.
For `periodic_kde`, the training source uses the same position and beta jitter
parameters as the rollout bank.

Checkpoint metadata now records the training and rollout initial-density
semantics. Loading refuses a checkpoint whose stored semantics differ from the
effective configuration. Existing checkpoints without the new field are
interpreted as empirical-source checkpoints, preserving the old outputs.

`train_reference_endpoints.py` exposes the choice through
`--initial-training-density periodic_kde` and can safely reuse a compatible
physical/defect bank through `--reuse-bank-from`. The output configuration and
provenance are frozen before training.

The deterministic comparison is implemented in
`compare_reference_initial_density.py`. It verifies bank hashes, reconstructs
the endpoint training samples, checks the frozen rollout particles, computes
the production 3-D periodic multiscale MMD², computes the visualization's
smoothed marginal overlaps, and writes the complete paired result as JSON.

## Seedwise results

Lower target MMD² is better. The reduction is paired within the same reference
seed and species.

| seed | species | baseline target MMD² | matched-KDE target MMD² | reduction |
|---:|---|---:|---:|---:|
| 20260818 | plus | 0.001751123 | 0.000607223 | 65.32% |
| 20260819 | plus | 0.001684701 | 0.000650724 | 61.37% |
| 20260820 | plus | 0.001700005 | 0.000617698 | 63.66% |
| 20260818 | minus | 0.001880443 | 0.000670589 | 64.34% |
| 20260819 | minus | 0.001822086 | 0.000605594 | 66.76% |
| 20260820 | minus | 0.001829991 | 0.000637587 | 65.16% |

All six paired comparisons improve. The smallest reduction is 61.37% and the
largest is 66.76%.

## Aggregate results

Values are the mean ± sample standard deviation over three reference seeds for
each species. The combined row contains all six seed/species pairs.

| panel | baseline target MMD² | matched-KDE target MMD² | paired reduction |
|---|---:|---:|---:|
| plus | 0.001711943 ± 0.000034783 | 0.000625215 ± 0.000022704 | 63.45% ± 1.98% |
| minus | 0.001844173 ± 0.000031658 | 0.000637923 ± 0.000032499 | 65.42% ± 1.23% |
| combined | 0.001778058 ± 0.000078296 | 0.000631569 ± 0.000026021 | 64.44% ± 1.83% |

The target fit also improves in both independently displayed marginals:

| panel | spatial overlap baseline → matched | beta overlap baseline → matched |
|---|---:|---:|
| plus | 0.6757 ± 0.0047 → 0.7484 ± 0.0030 | 0.6733 ± 0.0085 → 0.9082 ± 0.0108 |
| minus | 0.6697 ± 0.0022 → 0.7489 ± 0.0018 | 0.8318 ± 0.0069 → 0.9329 ± 0.0048 |

In particular, the plus-species beta mismatch that was visually dominant in
the empirical-source references is largely removed. The improvement occurs in
the production joint 3-D MMD² as well, so it is not merely an artifact of the
separate spatial and beta marginal plots.

## Initial endpoint and controlled-comparison checks

The rollout initial particles are bitwise identical between arms for every
seed/species pair. Consequently, the initial endpoint audit is unchanged:

| species | initial MMD² | spatial overlap | beta overlap |
|---|---:|---:|---:|
| plus | 0.001752787 | 0.590639 | 0.765511 |
| minus | 0.001829055 | 0.582230 | 0.913351 |

This is important for interpretation. The ablation improves how the flow maps
the already-frozen KDE initial law to the target; it does not make the KDE
initial law itself closer to the raw empirical t=5 population. The relatively
low initial spatial overlap remains a separate bandwidth/modeling issue.

## Training-loss interpretation

The final raw CFM loss increases from 145.92 ± 5.50 to 158.97 ± 3.79 when all
six flows are pooled. This does not contradict the endpoint result and should
not be used to rank the two arms directly. KDE jitter changes the distribution
and magnitude of shortest-arc bridge displacements, so the two losses average
squared velocity errors under different training distributions. The frozen
endpoint MMD² and overlaps are the appropriate common evaluation here.

## Conclusion

The matched-KDE arm is unambiguously better for the declared KDE rollout
semantics on t=5→15. The evidence supports the diagnosis of initial-law
covariate shift: training on raw empirical p0 and evaluating from KDE p0 was a
material implementation-semantic defect, even though the individual formulas
were correct.

This result does not by itself qualify a new official active-nematic reference
or justify resuming the Pareto sweep. It is an isolated development ablation on
t=5→15, not a confirmation on the production interval, and it does not assess
Law/Tangent/Full scientific risk. The next controlled step should be chosen
explicitly:

1. keep the periodic-KDE rollout and use matched-KDE training, then separately
   tune or qualify the KDE bandwidth to improve the initial endpoint; or
2. use an empirical rollout law consistently in both training and evaluation,
   accepting its different support and concentration properties.

No existing official reference, saved Pareto result, or paused process was
overwritten or resumed.

## Reproduction

```bash
.venv/bin/python experiments/active_nematic_unbalance_percentage/train_reference_endpoints.py \
  --start 5 --end 15 \
  --initial-training-density periodic_kde \
  --reuse-bank-from experiments/active_nematic_unbalance_percentage/outputs/reference_t5_t15_v1 \
  --output experiments/active_nematic_unbalance_percentage/outputs/reference_t5_t15_matched_kde_v1

.venv/bin/python experiments/active_nematic_unbalance_percentage/visualize_reference_endpoints.py \
  --frozen-inputs experiments/active_nematic_unbalance_percentage/outputs/reference_t5_t15_matched_kde_v1 \
  --output-stem experiments/active_nematic_unbalance_percentage/figures/active_nematic_reference_endpoint_fit_t5_t15_matched_kde_v1

.venv/bin/python experiments/active_nematic_unbalance_percentage/compare_reference_initial_density.py
```

Verification after implementation: 26 focused active-nematic tests completed
successfully.
