# Observable-design toy report

Phase: **smoke**. This is a plumbing/debug run; no scientific claims are made.

## Reused implementation

The experiment imports the Experiment-B endpoint samplers, linear stochastic interpolant and derivative, frozen flow-matched reference velocity, implicit empirical I-projection, stable covariance solve, Deep-Ritz potential/integrand, Heun rollout, RBF-MMD bandwidth convention, and held-out angular Fourier features. Experiment B itself was not modified.

## Observable family and objectives

The raw dictionary is `b=[x1,x2,x1^2,x1*x2,x2^2]`. A design-only pooled bank fixes `c_b` and an invertible symmetric whitening `W`; every learned map is `Phi_A=A W (b-c_b)` with `A A^T=I_R` and target zero. INFO minimizes supervised endpoint-label cross-entropy, CV minimizes variance-normalized reduced-flow closure error, and FIBER minimizes delta-t-normalized weighted MMD from a calibrated tangent pushforward to an independently calibrated next law, subject to hard calibration/rank/ESS gates.

INFO is a supervised state-information proxy, not a mutual-information estimator. A discriminative distribution of `Phi_A(X)` is compatible with matched endpoint expectations.

## Run status

Completed observable families: info, cv, fiber, random, full_phi5.
Design raw endpoint discrepancy: `[-0.015414570009182066, 0.010443348888710033, 0.011150813936259274, -0.017599011392806112, -0.03163866844320684]`.

## Learned subspaces

Pairwise projection distances: `{"info": {"info": 0.0, "cv": 0.05103548525151122, "fiber": 0.013806953795700436}, "cv": {"info": 0.05103548525151122, "cv": 0.0, "fiber": 0.0525488298063667}, "fiber": {"info": 0.013806953795700436, "cv": 0.0525488298063667, "fiber": 0.0}}`.
Principal angles (radians): `{"info": {"info": [0.0, 2.1073424255447017e-08, 3.650024149988857e-08], "cv": [2.5809568279517847e-08, 0.0035602767260869816, 0.08843957116744047], "fiber": [0.0, 0.004904879661687845, 0.023408082776653828]}, "cv": {"info": [0.0, 0.0035602767259934305, 0.0884395711674367], "cv": [0.0, 2.1073424255447017e-08, 2.1073424255447017e-08], "fiber": [0.0, 0.0069861035938294834, 0.09087376091019025]}, "fiber": {"info": [0.0, 0.004904879661597305, 0.023408082776668056], "cv": [0.0, 0.0069861035938930515, 0.0908737609101878], "fiber": [0.0, 2.9802322387695312e-08, 2.9802322387695312e-08]}}`.

See `summary.csv` for endpoint ambiguity, fiber conditioning, local closure, velocity-gap, rollout, hidden-law, and robustness diagnostics. The seven prespecified figure families are emitted as PNGs.

## Prespecified scientific questions

### Question 1: do the objectives select different subspaces?

In this run the learned-subspace distances range from `0.01381` to `0.05255`. For a smoke run this is only a collapse/separation diagnostic, not a stable scientific conclusion.

### Question 2: information versus fiber conditioning

INFO endpoint AUROC is `0.5579` versus `0.5606` averaged over the other learned maps; its minimum path ESS is `0.3234` versus `0.3207` for the others.

### Question 3: reduced-flow closure

Fresh frozen-observable closure R2 values are `{"info": 0.2622605040789783, "cv": 0.2644378406853273, "fiber": 0.2619428017997898}`. These are reported separately from law-level MMD.

### Question 4: tangent-to-next-law closure

Mean local tangent MMD is `0.1149` for FIBER and `0.115` for INFO (FIBER minus INFO `-0.000114`).

### Question 5: local-to-global translation

FIBER's mean tangent-versus-MFSI velocity-gap MSE is `0.5649`; its tangent and safe-MFSI interior rollout MMDs are `0.1406` and `0.1432`. No direction is declared beneficial from this smoke cell.

### Question 6: held-out rotations

Frozen-A minimum endpoint ESS across the two rotations is `{"info": 0.9988780770389115, "cv": 0.9987665001190175, "fiber": 0.9988681889839006}`. This panel establishes feasibility only; it is not a frozen-network rollout claim.

## Gradient validation

Gradient-validation results are produced by `tests/test_observable_design_toy.py`; they are intentionally not fabricated into this run artifact. Run `python3 -m pytest tests/test_observable_design_toy.py -q` in the validated environment.

## Interpretation

No positive outcome is assumed. Inspect the pairwise angles and untouched-bank contrasts. A smoke run is only evidence that the protocol executes; confirmatory conclusions require the prespecified crossed model-seed/evaluation-bank run.

## Current scope note

The rotation panel reports frozen-A endpoint feasibility with the common rotated target recomputed. It is not labeled as a zero-shot downstream-network result. Full matched-compute downstream retraining per rotation is not yet implemented by this driver and remains required before a confirmatory robustness claim.
