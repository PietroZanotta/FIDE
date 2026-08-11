# Strength-matched observable follow-up

This is a separate Experiment-E follow-up. Experiment-D objectives, checkpoints, metrics, and conclusions were not modified. Matching used fixed design-only banks; final evaluation banks never entered selection.

## Bottom line

The strength-matched control explains most of the original FIBER law-level advantage.

Against strength-matched random controls, FIBER-minus-control crossed contrasts are: local tangent MMD `+0.0005 [-0.0007, +0.0022]`, tangent rollout MMD `-0.0037 [-0.0118, +0.0030]`, safe-MFSI MMD `+0.0000 [-0.0052, +0.0054]`, velocity gap `-0.0940 [-0.2206, -0.0044]`, angular error `+0.0027 [-0.0061, +0.0130]`, and min ESS `-0.0016 [-0.0132, +0.0073]`. Negative is favorable for errors; positive is favorable for ESS.

FIBER mean strength is `0.0760` and matched-random mean strength is `0.0760`. The joint strength/informativeness control has mean strength `0.0769`; FIBER-minus-joint safe-MFSI is `-0.0004 [-0.0028, +0.0020]`.

## E1: strength-matched random subspaces

Five controls were nested within each of ten model seeds and averaged before crossed bootstrap. Exact 5% strength matching was available for every seed. FIBER safe-MFSI MMD is `0.0679` versus `0.0679` for strength-matched random; mean ESS is `0.913` versus `0.915`.

Across the 50 E1 controls, relative strength error has median `0.179%`, 95th percentile `1.143%`, and maximum `1.888%`.

## E2: joint strength/informativeness matching

Exact triple-tolerance matching supplied five controls for `10/10` seeds; the nearest-neighbor fallback was not used. Joint-control safe-MFSI MMD is `0.0683` and local tangent MMD is `0.0581`.

Across the 50 E2 controls, maximum relative errors were strength `4.972%`, variance `3.029%`, and endpoint Phi-MMD `6.151%` (prespecified limits 5%, 5%, and 10%).

## E3: strength-constrained FIBER

Accepted checkpoints within the prespecified 5% target tolerance: LOW `8/10`, MEDIUM `10/10`, HIGH `10/10`. Among accepted checkpoints, mean achieved strengths are `0.0737`, `0.2536`, and `0.4321`. Rejected checkpoints were retained in the achievement table but were not evaluated or counted as scientific replicates.

- FIBER_LOW: V `0.0737`, local MMD `0.0575`, safe-MFSI `0.0758`, tangent rollout `0.0944`, mean ESS `0.905`, velocity gap `0.2662`.
- FIBER_MEDIUM: V `0.2536`, local MMD `0.0673`, safe-MFSI `0.0936`, tangent rollout `0.1168`, mean ESS `0.747`, velocity gap `0.5146`.
- FIBER_HIGH: V `0.4321`, local MMD `0.0790`, safe-MFSI `0.0954`, tangent rollout `0.1055`, mean ESS `0.656`, velocity gap `0.7399`.

At nearly the original FIBER strength, FIBER-low minus original FIBER is `+0.0023 [+0.0013, +0.0035]` for local MMD, `+0.0231 [+0.0177, +0.0284]` for tangent rollout, and `+0.0113 [+0.0055, +0.0185]` for safe-MFSI. Medium and high constraints also worsen law-level metrics and ESS relative to original FIBER. This supplies evidence for a transportability/strength tradeoff, but it is not strictly monotone for every endpoint (for example, tangent rollout is lower at HIGH than at MEDIUM). The constrained runs do not support a medium/high-strength FIBER advantage.

The central frontier plots are descriptive. No method is called Pareto-optimal unless it is nondominated in the displayed empirical metrics.

## E4: reference-geometry sensitivity

The two alternate references are fixed smooth monotone time reparameterizations of the repository-validated default stochastic interpolant (smoothstep and cosine). They preserve the endpoint laws, endpoint coupling, standardization, and reference construction; their endpoint and derivative identities are covered by the follow-up tests.

Mean largest principal angles are default/smoothstep `21.0` degrees, default/cosine `21.2` degrees, and smoothstep/cosine `2.0` degrees.

Matched-reference means:

| reference | V(A) | local MMD | tangent rollout | safe-MFSI | angular error | mean ESS |
|---|---:|---:|---:|---:|---:|---:|
| default | 0.0760 | 0.0583 | 0.0748 | 0.0679 | 0.0522 | 0.913 |
| smoothstep | 0.0581 | 0.0564 | 0.0697 | 0.0606 | 0.0450 | 0.936 |
| cosine | 0.0583 | 0.0564 | 0.0692 | 0.0606 | 0.0444 | 0.936 |

Safe-MFSI cross-reference means (rows train the observable, columns evaluate after reference-specific downstream retraining):

| train \ eval | default | smoothstep | cosine |
|---|---:|---:|---:|
| default | 0.0679 | 0.0609 | 0.0610 |
| smoothstep | 0.0662 | 0.0606 | 0.0605 |
| cosine | 0.0657 | 0.0607 | 0.0606 |

The default-trained subspaces differ materially from those trained under either alternate schedule, while the two alternate schedules produce similar subspaces. Accordingly, the learned object is best described here as reference-aware rather than fully intrinsic. Cross-reference safe-MFSI remains stable within each evaluation geometry after downstream retraining, so observable transfer is robust despite the subspace shift. Subspace stability and performance transfer are kept separate because downstream potentials were retrained while A remained frozen.

## Interpretation guardrails

Matched candidates, particles, time points, and optimizer iterations were never treated as independent replicates. All correlations/frontiers are descriptive. Negative results are retained. No result changes the registered Experiment-D confirmatory analysis.
