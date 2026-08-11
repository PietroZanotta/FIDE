# Observable-design mechanism analysis

This is post-hoc mechanism analysis of the frozen Experiment-D confirmatory checkpoints. No observable or downstream model was retrained, no registered metric was changed, and these diagnostics were not used for selection.

## Bottom line

FIBER selects lower-strength constraints on these reference banks and captures less of the bridge's standardized low-order mean departure. Mean standardized reference-fiber violation is `0.1617`, versus `0.4034` for INFO and `0.4748` for CV. Mean capture fraction is `0.334`, versus `0.676` and `0.822`.

The learned FIBER observables are not vacuous. Their mean reference Phi variance trace is `2.106` (INFO `2.097`, CV `2.139`), endpoint Phi-space MMD is `0.127`, and representative endpoint AUROC is `0.697`. FIBER therefore retains nontrivial sample-level information.

FIBER's mean reference-motion null fraction is `0.666`, compared with INFO `0.324` and CV `0.178`. FIBER allocates the largest mean share of reference low-order motion to its unresolved plane. This is a descriptive mechanism diagnostic, not causal proof.

## 1. Constraint strength and projection geometry

| Objective | Mean / mean seedwise max / integrated violation | Mean / integrated capture | Mean / min ESS | Mean KL | Mean lambda | Mean max weight | Mean entropy | Ref / projected Phi variance trace |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| INFO | 0.4034 / 0.8089 / 0.4983 | 0.676 / 0.702 | 0.696 / 0.349 | 0.1956 | 0.5236 | 3.26e-03 | 8.122 | 2.097 / 2.823 |
| CV | 0.4748 / 0.9517 / 0.5876 | 0.822 / 0.886 | 0.633 / 0.251 | 0.2549 | 0.5974 | 3.91e-03 | 8.063 | 2.139 / 3.040 |
| FIBER | 0.1617 / 0.3196 / 0.1959 | 0.334 / 0.264 | 0.913 / 0.780 | 0.0457 | 0.2654 | 1.07e-03 | 8.272 | 2.106 / 2.304 |
| RANDOM | 0.3515 / 0.7018 / 0.4331 | 0.593 / 0.584 | 0.736 / 0.401 | 0.1540 | 0.4689 | 2.98e-03 | 8.164 | 2.130 / 2.743 |
| FULL_PHI5 | 0.4803 / 0.9549 / 0.5922 | 1.000 / 1.000 | 0.633 / 0.252 | 0.2579 | 0.6072 | 3.80e-03 | 8.060 | 3.525 / 5.030 |

The descriptive evidence primarily supports smaller raw violation; the violation-adjusted model does not show an additional positive FIBER ESS coefficient. There is limited cross-objective overlap in violation, so the adjustment is partly extrapolative and does not establish conditioning at exactly matched violation.

## 2. Null-space stability and physical directions

### INFO

Within-objective null-projector distance: mean `0.736`, range `0.434`-`0.996`. Mean largest principal angle: `73.8` degrees. Mean-projector eigenvalues: `0.678, 0.528, 0.414, 0.285, 0.095`.

Consensus raw null directions (columns are a deterministic eigensystem representation of the consensus subspace; individual signs/basis rotations have no invariant meaning):

```text
  x1=+0.804, x2=-0.513, x1^2=-0.295, x1*x2=+0.041, x2^2=+0.044
  x1=+0.569, x2=+0.800, x1^2=+0.156, x1*x2=+0.075, x2^2=-0.084
```

### CV

Within-objective null-projector distance: mean `0.633`, range `0.096`-`0.981`. Mean largest principal angle: `68.9` degrees. Mean-projector eigenvalues: `0.775, 0.623, 0.453, 0.146, 0.004`.

Consensus raw null directions (columns are a deterministic eigensystem representation of the consensus subspace; individual signs/basis rotations have no invariant meaning):

```text
  x1=-0.227, x2=+0.113, x1^2=-0.676, x1*x2=+0.014, x2^2=+0.692
  x1=+0.883, x2=-0.393, x1^2=-0.200, x1*x2=+0.045, x2^2=+0.158
```

### FIBER

Within-objective null-projector distance: mean `0.672`, range `0.481`-`0.910`. Mean largest principal angle: `64.8` degrees. Mean-projector eigenvalues: `0.876, 0.484, 0.318, 0.246, 0.076`.

Consensus raw null directions (columns are a deterministic eigensystem representation of the consensus subspace; individual signs/basis rotations have no invariant meaning):

```text
  x1=-0.019, x2=+0.042, x1^2=+0.822, x1*x2=+0.128, x2^2=+0.553
  x1=-0.135, x2=-0.499, x1^2=-0.432, x1*x2=-0.179, x2^2=+0.717
```

The primary objects are the projectors and principal angles, not individual printed vectors. FIBER's mean-projector spectrum and pairwise distances quantify whether the unresolved plane is reproducible; see `nullspaces.json` and `nullspace_stability.png`.

FIBER's unresolved plane is only partially reproducible: its leading mean-projector eigenvalue is `0.876`, but the second is `0.484`; mean pairwise distance is `0.672` and mean largest angle is `64.8` degrees. Thus one unresolved direction is moderately stable, while the full two-plane is not tightly seed-stable and is not clearly more reproducible than CV.

The leading FIBER consensus raw direction is mostly a positive quadratic-scale combination (`x1^2` and `x2^2`). The second mixes `x2^2`, `-x1^2`, and `-x2`; because the second consensus eigenvalue is modest, that coordinate should be treated as exploratory rather than a seed-invariant physical law.

## 3. Motion along unresolved directions

The reference bridge places `66.6%` of its standardized low-order mean energy in FIBER's null space on average. The corresponding fractions are `32.4%` for INFO and `17.8%` for CV. The projected, tangent, and safe-MFSI trajectories in consensus-null coordinates are shown in `null_moment_trajectories.png`.

At the three interior times, where the reference departure is appreciable, the mean null fractions are FIBER `85.1%`, INFO `25.5%`, and CV `0.8%`. Endpoint fractions are less interpretable because the total standardized mean departure is close to zero there.

Mean standardized constrained / null norm along each path:

| Objective | Reference | I-projected | Tangent | Safe MFSI |
|---|---:|---:|---:|---:|
| INFO | 0.403 / 0.191 | 0.000 / 0.188 | 0.002 / 0.089 | 0.002 / 0.189 |
| CV | 0.475 / 0.037 | 0.000 / 0.053 | 0.002 / 0.011 | 0.002 / 0.055 |
| FIBER | 0.162 / 0.434 | 0.000 / 0.415 | 0.002 / 0.342 | 0.002 / 0.382 |
| RANDOM | 0.351 / 0.297 | 0.000 / 0.276 | 0.002 / 0.150 | 0.002 / 0.260 |
| FULL_PHI5 | 0.480 / 0.000 | 0.000 / 0.000 | 0.002 / 0.000 | 0.002 / 0.000 |

FIBER's R=3 advantage over full-Phi5 is consistent with allowing these low-order combinations to move while full-Phi5 has no low-order null space. This is supportive but insufficient to label full-Phi5 causally 'overconstrained'; architecture/optimization at different output dimension remains a confound.

## 4. Relationship to law-level performance

Across seed means, FIBER retains the confirmatory law advantage (safe-MFSI MMD `0.0679` versus INFO `0.0850` and CV `0.0854`). In the descriptive learned-seed regression `safe MMD ~ violation + objective`, the FIBER-vs-INFO coefficient is `-0.0093`. This is exploratory adjustment, not causal identification or a significance test.

For mean ESS under the same descriptive adjustment, the FIBER-vs-INFO coefficient is `-0.0149`. Its ESS advantage does not remain descriptively visible after the linear adjustment. Ten seeds per objective are too few for a strong adjusted claim.

All Pearson/Spearman correlations, including within-objective n=10 values, are in `mechanism_correlations.json`. They are exploratory and no p-values are interpreted.

## 5. Endpoint informativeness

| Objective | AUROC | Phi-space MMD | Expectation gap | Calibrated max gap | Angular gap |
|---|---:|---:|---:|---:|---:|
| INFO | 0.698 | 0.125 | 0.0364 | 6.29e-18 | 0.471 |
| CV | 0.697 | 0.120 | 0.0351 | 2.39e-18 | 0.471 |
| FIBER | 0.697 | 0.127 | 0.0334 | 2.82e-18 | 0.471 |

INFO is not relabeled as more identifying: the frozen representative AUROCs are essentially equal. FIBER remains sample-level discriminative despite matched measured expectations.

## Guarded conclusion

FIBER selects lower-strength constraints on these reference banks and captures less of the bridge's standardized low-order mean departure. Those constraints are nevertheless nonzero-variance and sample-level informative. FIBER allocates the largest mean share of reference low-order motion to its unresolved plane. The descriptive evidence primarily supports smaller raw violation; the violation-adjusted model does not show an additional positive FIBER ESS coefficient. The favorable FIBER safe-MFSI association remains in the descriptive linear adjustment. These observations support only a mechanism hypothesis; they do not establish causality or rule out objective-specific architecture/optimization effects.

This post-hoc analysis does not alter the registered confirmatory conclusion and does not prove causality.
