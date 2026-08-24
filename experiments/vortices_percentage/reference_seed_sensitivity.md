# Vortices reference-seed sensitivity audit

Date: 2026-08-19

## Conclusion

The vortices experiment has a small, statistically detectable dependence on
the learned-reference seed, but it does **not** show the severe sensitivity seen
in the active-nematic experiment.

Across reference seeds 20260815--20260817, the held-out Full-versus-Law action
reduction ranges from 50.54% to 54.15%, a range of 3.61 percentage points.  The
corresponding active-nematic range was 12.7%--74.5%, or 61.8 percentage points.
The vortices spread is therefore about 17 times smaller by this simple range
comparison.  All three vortices runs select the same Full sensor geometry and
pass every saved structural and validity check.

This is good evidence that the reported vortices gain is robust to the three
reference-training seeds tested.  It is not evidence that the endpoint-only
reference path is physically exact; that is a separate modeling question.

## Controlled experiment

The production run at seed 20260815 was reused, and two new full-budget runs
were performed at seeds 20260816 and 20260817.  Only
`reference_training.seed` changed.  The following saved inputs are byte-for-byte
identical across all three runs:

- the 50,000-particle truth bank;
- the 50,000 paired reference-training endpoints;
- the 16-trial law and 24-trial action selection bank;
- the 64-trial independent validation bank.

Each learned reference was rolled with 32,768 particles.  Optimizer starts,
sensor constraints, numerical backends, and validation trials were held fixed.
The two new runs were written under
`outputs/reference_seed_sensitivity/`; the production output was not modified.

## Results

The reduction is the paired ratio-of-means statistic

`1 - mean(Full-design action) / mean(Law-design action)`

on the common 64-trial validation bank.  Intervals are paired nonparametric
bootstrap intervals with 100,000 resamples.

| Reference seed | Law action | Full action | Reduction | Paired 95% interval | Full valid trials |
|---:|---:|---:|---:|---:|---:|
| 20260815 | 3.3629 | 1.6634 | 50.54% | 47.97%--53.16% | 64/64 |
| 20260816 | 3.4572 | 1.6567 | 52.08% | 49.85%--54.25% | 64/64 |
| 20260817 | 3.4507 | 1.5823 | 54.15% | 51.91%--56.35% | 64/64 |

Across seeds, the mean reduction is 52.25%, the sample standard deviation is
1.81 percentage points, and the range is 3.61 points.  Law action has 1.54%
coefficient of variation across seeds; Full action has 2.76%.

Because the same validation trials are reused, paired comparisons can resolve
small shifts more precisely than the intervals for any one seed:

| Comparison | Reduction difference | Paired 95% interval |
|---|---:|---:|
| 20260816 minus 20260815 | +1.54 points | +0.71 to +2.10 points |
| 20260817 minus 20260815 | +3.61 points | +3.00 to +4.05 points |
| 20260817 minus 20260816 | +2.07 points | +1.81 to +2.35 points |

Thus the seed effect is real at the resolution of this controlled experiment,
but practically modest compared with the approximately 52% action reduction
and dramatically smaller than the active-nematic instability.

## Geometry and reference-path evidence

The selected Full centers are identical in all three result files:

```text
[(0.440933, 0.427542),
 (1.685739, 0.383945),
 (0.634678, 0.606686),
 (1.654194, 0.755642)]
```

The Law designs move only slightly: their optimal-assignment RMS separation is
0.0065--0.0113 in a 2-by-1 domain.  Tangent designs separate by at most 0.0652.
There is no seed-specific jump to a remote Full-action basin like the one found
for active-nematic seed 20260820.

The learned references are not numerically identical.  On the common reference
particles, pairwise normalized velocity RMSE is 0.212--0.235 and final paired
position RMS is 0.052--0.060.  The action result is therefore stable despite
visible reference-field variation, not because training reproduced the same
network three times.

One qualification is important: the fixed candidate list contains the
production Full geometry as a provenance-tracked seed, and every run selected
that candidate.  This makes the action-gain comparison especially clean, since
the numerator geometry is held constant in practice.  It does not prove that
three completely de-novo searches with no seeded action candidate would all
discover the same geometry.  That is an optimizer-basin audit, distinct from
the present reference-seed audit.

## Why vortices is more stable than active nematics

The vortices reference-training problem is much better identified:

1. It uses 50,000 distinct initial particles and their physically paired final
   states from the deterministic double-gyre rollout.
2. Active nematics resampled from only 36 distinct initial positive-defect
   states and 107 final states, with at most 3,852 independent Cartesian pairs.
3. The vortices map is a smooth, fixed-mass Lagrangian flow.  The normalized
   active-defect law instead hides defect creation and annihilation and lacks
   persistent defect identities.
4. The vortices reference is transformed through box-logit coordinates, so
   different seeds cannot create competing boundary-escape artifacts.

These differences support the hypothesis that sparse, unpaired, dynamically
ambiguous endpoint data—not endpoint-only training by itself—is the main source
of the active-nematic instability.

## Remaining caveat and next test

The endpoint-only vortices reference can still disagree with the physical
double-gyre marginal path at intermediate times.  Seed stability only says the
three learned solutions lead to similar design conclusions.  A stronger model
audit should compare each learned marginal and velocity field directly with the
known double-gyre truth over all 21 times.

If discovery robustness also matters, rerun the three searches after removing
the provenance-seeded `full_seed_etas`, increase independent Full multistarts,
and compare the best certified basin.  This should be reported separately from
reference-model sensitivity.

## Reproduction

The runner now supports isolated reference-seed outputs:

```bash
OMP_NUM_THREADS=4 .venv/bin/python experiments/vortices/run.py \
  --reference-seed 20260816 \
  --output-dir experiments/vortices/outputs/reference_seed_sensitivity/reference_seed_20260816
```

The read-only summary audit is:

```bash
.venv/bin/python experiments/vortices/summarize_reference_seeds.py
```

It writes the machine-readable result to
`experiments/vortices/outputs/reference_seed_sensitivity/summary.json`.
