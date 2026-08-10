# Experiment D confirmatory evidence

## Verdict

The registered `R=3` experiment provides evidence for the fiber-oriented claim.
FIBER learned observably different subspaces from INFO and CV, passed all hard
feasibility gates, improved its method-blind local tangent-to-law criterion, and
also improved untouched nominal and rotated-condition law-level rollouts.

This is not evidence for every motivating conjecture. INFO was not more
endpoint-identifying than the other learned representations in the fresh
representative diagnostic, and CV's excellent reduced-flow closure did not
translate into the best projected-law dynamics.

## Protocol

- 10 independently learned/trained model seeds by 10 untouched evaluation banks;
- 500 nominal crossed cells: INFO, CV, FIBER, random `R=3`, and the capacity-unequal full-Phi5 reference;
- 600 held-out rotation cells: three learned objectives, two rotations, and the same 10 by 10 crossed design;
- 20,000 crossed-bootstrap replicates, independently resampling model-seed rows and evaluation-bank columns;
- all headline contrasts were specified as FIBER minus INFO and FIBER minus CV.

The 100 cells per objective are repeated crossed measurements, not 100 iid
replicates. All intervals below use the two-dimensional crossed bootstrap.

## Headline nominal results

Lower is better except for ESS.

| Metric | INFO mean | CV mean | FIBER mean | FIBER - INFO, 95% CI | FIBER - CV, 95% CI |
|---|---:|---:|---:|---:|---:|
| Local tangent-to-next-law MMD | 0.07558 | 0.08416 | 0.05829 | -0.01729 [-0.02338, -0.01114] | -0.02587 [-0.03074, -0.02006] |
| Tangent interior rollout MMD | 0.10114 | 0.08674 | 0.07484 | -0.02631 [-0.03969, -0.01375] | -0.01191 [-0.01976, -0.00404] |
| Safe-MFSI interior rollout MMD | 0.08502 | 0.08539 | 0.06791 | -0.01712 [-0.02421, -0.00958] | -0.01748 [-0.02698, -0.00769] |
| Tangent-MFSI velocity-gap MSE | 0.61962 | 0.45493 | 0.17207 | -0.44755 [-0.69334, -0.22422] | -0.28286 [-0.41994, -0.16745] |
| Minimum path ESS fraction | 0.36311 | 0.24520 | 0.80026 | +0.43715 [+0.30518, +0.55981] | +0.55506 [+0.40337, +0.67553] |

Relative to INFO, FIBER reduced mean local MMD by 22.9%, tangent rollout MMD
by 26.0%, safe-MFSI rollout MMD by 20.1%, and velocity-gap MSE by 72.2%.
Relative to CV, the reductions were 30.7%, 13.7%, 20.5%, and 62.2%.

The hidden-angular reconstruction error also favored FIBER descriptively:
0.05221 versus 0.06728 for INFO and 0.07371 for CV. Safe-rollout maximum
moment errors were similar and small (INFO 0.00164, CV 0.00169, FIBER 0.00159).

## Held-out rotations

The nominal conclusion persisted after applying the same held-out rotation to
both endpoint laws, recomputing the common target, freezing A, and retraining
the downstream model with matched compute.

At rotation pi/8, FIBER safe-MFSI MMD was 0.06908 versus 0.08527 (INFO) and
0.08432 (CV). At rotation pi/4 it was 0.06741 versus 0.08582 and 0.08519.
Every prespecified rotated FIBER-minus-INFO/CV interval for local MMD, tangent
rollout MMD, safe-MFSI MMD, velocity gap, and minimum ESS excluded zero in the
favorable direction. The complete intervals are in `results.json` under
`robustness_crossed_bootstrap`.

## Learned subspaces

Across the ten paired observable seeds:

| Pair | Projection distance mean [min, max] | Largest principal angle mean [min, max] |
|---|---:|---:|
| FIBER vs INFO | 0.595 [0.456, 0.682] | 71.1 deg [48.1, 86.3] |
| FIBER vs CV | 0.635 [0.586, 0.681] | 80.1 deg [73.5, 90.0] |
| INFO vs CV | 0.368 [0.087, 0.601] | 45.7 deg [8.5, 85.1] |

Thus the objectives did not collapse to a common `R=3` moment subspace.

## Objective success and feasibility

- INFO's own held-out endpoint-label AUROC averaged 0.695 across observable seeds.
- CV's own held-out reduced-flow closure R2 averaged 0.987 across seeds.
- All 10 FIBER checkpoints passed the hard gate. Validation minimum ESS ranged
  from 0.300 to 0.961, maximum calibration residual was below `1e-16`, and
  maximum covariance condition number was below 2.22.
- The representative untouched endpoint classifier AUROCs were essentially
  identical: INFO 0.6976, CV 0.6973, and FIBER 0.6971. This does not support the
  secondary hypothesis that INFO would be distinctly more identifying.
- Endpoint expectation gaps were finite-bank sampling errors and calibrated to
  below `7e-18`; Phi-space MMD and the fixed angular gap remained nonzero, as
  required for observational non-identifiability.

## Controls

The random `R=3` control had mean local/tangent/safe-MFSI MMD of
0.07076/0.10270/0.07851. FIBER improved all three. The full-Phi5 reference had
0.08347/0.08408/0.08332; it is reported only as a higher-capacity reference,
not as an equal-capacity comparator.

## Interpretation

These results match the brief's Outcome B: FIBER differs and improves local law
closure. They additionally show favorable local-to-global translation in both
tangent and learned-MFSI rollouts, a smaller tangent-MFSI gap, better ESS, and
generalization under both held-out rotations.

The defensible conclusion is specific: on this toy benchmark, under the shared
low-order `R=3` family and registered compute, optimizing population-law fiber
closure selected different and more transportable observable subspaces than
the INFO and CV objectives. It does not establish a general theorem or show
that INFO always maximizes identifying power.
