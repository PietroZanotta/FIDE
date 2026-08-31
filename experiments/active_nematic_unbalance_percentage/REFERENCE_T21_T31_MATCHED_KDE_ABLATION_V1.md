# Active-nematic t=21→31 matched-KDE reference ablation

## Outcome

Matching the flow-training initial law to the periodic-KDE rollout law improves
the original t=21→31 reference ensemble. The effect is smaller than in the
t=5→15 transient study, but it is consistent: target joint MMD² decreases for
all three seeds and both species.

This remains a reference-only development result. No Law, Tangent, Full,
sensor optimization, Pareto selection, or validation calculation was run. The
existing `more_training_v2_pareto` process and its frozen results were not
modified or resumed.

## Controlled design

The comparison changes only
`reference_training.initial_endpoint_density_model` from `empirical` to
`periodic_kde`. The matched arm uses the already declared rollout bandwidths:
spatial jitter standard deviation 1.0 and beta jitter standard deviation 0.25.

The following quantities are unchanged:

- t=21 and t=31 training endpoint populations;
- all 96 deterministic physical realizations and the 64/16/16 split;
- all intermediate physical and defect-bank time slices t=21,…,31;
- three reference seeds and their plus/minus offsets;
- target endpoint training samples, verified bitwise for both species;
- network architecture, optimizer, CFM loss, and training budget;
- the 8,192 rollout initial particles, verified bitwise for all six pairs;
- Fisher–Rao mass schedules, whose common SHA-256 is
  `91eb18e5bdb7f74b479785b1c57a5ee1b4fcdb96d99f0f9caae12a8a3cdbe2fb`.

The reused physical bank SHA-256 is
`512fe2f077c336696ad4085de2207ec4072d24e1e3559466d335c4c00d9ebd9b`.
The reused two-species defect bank SHA-256 is
`a2f3ee9b3d896e5e4f3aff56851b1ac24cc36e0b10151b58e0c8eec83ea44f26`.

Training remains endpoint-only. Intermediate states are retained solely so
the learned flows can be rolled out on the complete eleven-node grid required
by downstream measurements and action evaluation. No intermediate truth was
used in the CFM objective or in this endpoint comparison.

## Joint endpoint MMD²

Lower is better. Every row is paired by seed and species.

| seed | species | empirical-source | matched-KDE | paired reduction |
|---:|---|---:|---:|---:|
| 20260818 | plus | 0.000560722 | 0.000542716 | 3.21% |
| 20260819 | plus | 0.000585570 | 0.000525968 | 10.18% |
| 20260820 | plus | 0.000595519 | 0.000524682 | 11.89% |
| 20260818 | minus | 0.000594803 | 0.000527764 | 11.27% |
| 20260819 | minus | 0.000629009 | 0.000554745 | 11.81% |
| 20260820 | minus | 0.000608760 | 0.000553335 | 9.10% |

Aggregate values are mean ± sample standard deviation across the three seeds
within each species:

| panel | empirical-source target MMD² | matched-KDE target MMD² | paired reduction |
|---|---:|---:|---:|
| plus | 0.000580604 ± 0.000017922 | 0.000531122 ± 0.000010061 | 8.43% ± 4.60% |
| minus | 0.000610857 ± 0.000017199 | 0.000545281 ± 0.000015187 | 10.73% ± 1.43% |
| combined | 0.000595731 ± 0.000022834 | 0.000538202 ± 0.000013888 | 9.58% ± 3.30% |

The initial MMD² and rollout particles are identical between arms. Thus the
target change cannot be attributed to a different rollout sample, physical
bank, or target training sample.

## Marginal endpoint overlaps

The visualization uses smoothed periodic histogram intersection, where higher
is better.

| species | target spatial overlap, old → matched | target beta overlap, old → matched |
|---|---:|---:|
| plus | 0.7627 ± 0.0040 → 0.7708 ± 0.0025 | 0.9381 ± 0.0044 → 0.9424 ± 0.0017 |
| minus | 0.7596 ± 0.0039 → 0.7671 ± 0.0008 | 0.9498 ± 0.0048 → 0.9460 ± 0.0012 |

The plus marginals and both spatial marginals improve. The minus beta marginal
decreases slightly by about 0.0038, while the authoritative joint 3-D MMD²
still improves for every minus seed. This is a mixed marginal redistribution,
not a joint-metric failure.

The matched arm's target ranges are:

- plus spatial overlap: 0.7683–0.7734;
- plus beta overlap: 0.9405–0.9434;
- minus spatial overlap: 0.7666–0.7680;
- minus beta overlap: 0.9447–0.9470.

These values are much stronger and less seed-sensitive than the problematic
t=5 transient references.

## CFM loss

The final raw CFM loss increases from 162.37 ± 2.35 to 166.67 ± 2.69 when all
six flows are pooled. The arms train under different initial bridge
distributions, so their raw conditional losses are not directly comparable.
The common frozen endpoint MMD² is the relevant paired diagnostic.

## Interpretation and recommendation

The original t=21→31 reference was already substantially better fitted than
the t=5 transient reference, which explains why the semantic correction yields
a 9.58% rather than a 64.44% reduction. Nevertheless, the correction moves the
joint endpoint metric in the favorable direction for every independent pair
and eliminates the conceptual train/rollout inconsistency.

The matched-KDE t=21→31 ensemble is therefore the preferable reference input
for a new 2% development run and, if that run is numerically healthy, a newly
frozen Pareto sweep. It must not be inserted into the existing Pareto output:
the learned paths have changed, so old 0.5%, 1%, and 2% receipts are not
scientifically reusable.

The candidate output already contains six rollout banks on all eleven times
from t=21 through t=31, each with 8,192 particles. It is structurally ready to
serve as the reference source for a new downstream study.

## Artifacts

- `outputs/reference_t21_t31_matched_kde_v1/effective_config.json`
- `outputs/reference_t21_t31_matched_kde_v1/reference_endpoint_audit.json`
- `outputs/reference_t21_t31_matched_kde_v1/initial_density_ablation_comparison.json`
- `outputs/reference_t21_t31_matched_kde_v1/reference_seed_*/{plus,minus}_reference.npz`
- `outputs/reference_t21_t31_matched_kde_v1/reference_seed_*/{plus,minus}_reference_bank.npz`
- `figures/active_nematic_reference_endpoint_fit_t21_t31_matched_kde_v1.png`
- `figures/active_nematic_reference_endpoint_fit_t21_t31_matched_kde_v1.pdf`

## Reproduction

```bash
.venv/bin/python experiments/active_nematic_unbalance_percentage/train_reference_endpoints.py \
  --start 21 --end 31 --time-step 1 \
  --initial-training-density periodic_kde \
  --reuse-bank-from experiments/active_nematic_unbalance_percentage/outputs/more_training_v2_source \
  --output experiments/active_nematic_unbalance_percentage/outputs/reference_t21_t31_matched_kde_v1

.venv/bin/python experiments/active_nematic_unbalance_percentage/visualize_reference_endpoints.py \
  --frozen-inputs experiments/active_nematic_unbalance_percentage/outputs/reference_t21_t31_matched_kde_v1 \
  --output-stem experiments/active_nematic_unbalance_percentage/figures/active_nematic_reference_endpoint_fit_t21_t31_matched_kde_v1

.venv/bin/python experiments/active_nematic_unbalance_percentage/compare_reference_initial_density.py \
  --baseline experiments/active_nematic_unbalance_percentage/outputs/more_training_v2_pareto/frozen_inputs \
  --candidate experiments/active_nematic_unbalance_percentage/outputs/reference_t21_t31_matched_kde_v1
```
