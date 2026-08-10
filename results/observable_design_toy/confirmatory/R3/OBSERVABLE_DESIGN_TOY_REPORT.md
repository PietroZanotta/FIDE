# Observable-design toy report

Phase: **confirmatory**. Confirmatory protocol.

## Reused implementation

The experiment imports the Experiment-B endpoint samplers, linear stochastic interpolant and derivative, frozen flow-matched reference velocity, implicit empirical I-projection, stable covariance solve, Deep-Ritz potential/integrand, Heun rollout, RBF-MMD bandwidth convention, and held-out angular Fourier features. Experiment B itself was not modified.

## Observable family and objectives

The raw dictionary is `b=[x1,x2,x1^2,x1*x2,x2^2]`. A design-only pooled bank fixes `c_b` and an invertible symmetric whitening `W`; every learned map is `Phi_A=A W (b-c_b)` with `A A^T=I_R` and target zero. INFO minimizes supervised endpoint-label cross-entropy, CV minimizes variance-normalized reduced-flow closure error, and FIBER minimizes delta-t-normalized weighted MMD from a calibrated tangent pushforward to an independently calibrated next law, subject to hard calibration/rank/ESS gates.

INFO is a supervised state-information proxy, not a mutual-information estimator. A discriminative distribution of `Phi_A(X)` is compatible with matched endpoint expectations.

## Run status

Completed observable families: info, cv, fiber, random, full_phi5.
Design raw endpoint discrepancy: `[-0.00027820915092069366, 0.0007501167194185035, 0.004104747576545442, 0.00046164683130923306, -0.0036660524744061007]`.

## Learned subspaces

Pairwise projection distances: `{"info": {"info": 0.0, "cv": 0.14939690483624474, "fiber": 0.5625044595702594}, "cv": {"info": 0.14939690483624474, "cv": 0.0, "fiber": 0.5859380729394597}, "fiber": {"info": 0.5625044595702594, "cv": 0.5859380729394597, "fiber": 0.0}}`.
Principal angles (radians): `{"info": {"info": [0.0, 0.0, 2.5809568279517847e-08], "cv": [2.1073424255447017e-08, 0.15238656383048693, 0.21112602008756925], "fiber": [0.0, 0.27489356217465616, 1.2102640646866836]}, "cv": {"info": [2.9802322387695312e-08, 0.15238656383048693, 0.21112602008756925], "cv": [0.0, 2.1073424255447017e-08, 2.1073424255447017e-08], "fiber": [0.0, 0.2648711510932551, 1.3731421566514856]}, "fiber": {"info": [3.942476676500724e-08, 0.27489356217465205, 1.2102640646866836], "cv": [2.5809568279517847e-08, 0.26487115109325426, 1.3731421566514856], "fiber": [0.0, 2.1073424255447017e-08, 2.9802322387695312e-08]}}`.

See `summary.csv` for endpoint ambiguity, fiber conditioning, local closure, velocity-gap, rollout, hidden-law, and robustness diagnostics. The seven prespecified figure families are emitted as PNGs.

## Prespecified scientific questions

### Question 1: do the objectives select different subspaces?

In this run the learned-subspace distances range from `0.1494` to `0.5859`. For a smoke run this is only a collapse/separation diagnostic, not a stable scientific conclusion.

### Question 2: information versus fiber conditioning

INFO endpoint AUROC is `0.6976` versus `0.6972` averaged over the other learned maps; its minimum path ESS is `0.2291` versus `0.5236` for the others.

### Question 3: reduced-flow closure

Fresh frozen-observable closure R2 values are `{"info": 0.9732438342770904, "cv": 0.9759283217646587, "fiber": 0.9689868350867964}`. These are reported separately from law-level MMD.

### Question 4: tangent-to-next-law closure

Mean local tangent MMD is `0.05706` for FIBER and `0.08904` for INFO (FIBER minus INFO `-0.03198`).

### Question 5: local-to-global translation

FIBER's mean tangent-versus-MFSI velocity-gap MSE is `0.1609`; its tangent and safe-MFSI interior rollout MMDs are `0.05115` and `0.05688`. No direction is declared beneficial from this smoke cell.

### Question 6: held-out rotations

Frozen-A minimum endpoint ESS across the two rotations is `{"info": 0.9993844479967783, "cv": 0.9993951168859571, "fiber": 0.9995534360227178}`. Matched-retraining safe-MFSI interior MMDs by rotation are `{"info": [0.10086729638820018, 0.09617667810640847], "cv": [0.09196602018138145, 0.08559249431621319], "fiber": [0.049659595138951886, 0.05399799424413762]}`.

## Gradient validation

Gradient-validation results are produced by `tests/test_observable_design_toy.py`; they are intentionally not fabricated into this run artifact. Run `python3 -m pytest tests/test_observable_design_toy.py -q` in the validated environment.

## Interpretation

No positive outcome is assumed. Inspect the pairwise angles and untouched-bank contrasts. A smoke run is only evidence that the protocol executes; confirmatory conclusions require the prespecified crossed model-seed/evaluation-bank run.

## Current scope note

The rotation panel includes condition-specific matched-compute downstream retraining with A frozen; it is distinct from a strict zero-shot frozen-network test.
