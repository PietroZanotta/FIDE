# Observable-design toy report

Phase: **smoke**. This is a plumbing/debug run; no scientific claims are made.

## Reused implementation

The experiment imports the Experiment-B endpoint samplers, linear stochastic interpolant and derivative, frozen flow-matched reference velocity, implicit empirical I-projection, stable covariance solve, Deep-Ritz potential/integrand, Heun rollout, RBF-MMD bandwidth convention, and held-out angular Fourier features. Experiment B itself was not modified.

## Observable family and objectives

The raw dictionary is `b=[x1,x2,x1^2,x1*x2,x2^2]`. A design-only pooled bank fixes `c_b` and an invertible symmetric whitening `W`; every learned map is `Phi_A=A W (b-c_b)` with `A A^T=I_R` and target zero. INFO minimizes supervised endpoint-label cross-entropy, CV minimizes variance-normalized reduced-flow closure error, and FIBER minimizes delta-t-normalized weighted MMD from a calibrated tangent pushforward to an independently calibrated next law, subject to hard calibration/rank/ESS gates.

INFO is a supervised state-information proxy, not a mutual-information estimator. A discriminative distribution of `Phi_A(X)` is compatible with matched endpoint expectations.

## Run status

Completed observable families: info, cv, fiber.
Design raw endpoint discrepancy: `[-0.015414570009182067, 0.010443348888710027, 0.011150813936259607, -0.01759901139280611, -0.03163866844320662]`.

## Learned subspaces

Pairwise projection distances: `{"info": {"info": 0.0, "cv": 0.051035485251511005, "fiber": 0.013806953794782036}, "cv": {"info": 0.051035485251511005, "cv": 0.0, "fiber": 0.05254882980718779}, "fiber": {"info": 0.013806953794782036, "cv": 0.05254882980718779, "fiber": 0.0}}`.
Principal angles (radians): `{"info": {"info": [0.0, 0.0, 2.1073424255447017e-08], "cv": [0.0, 0.0035602767261805326, 0.08843957116743167], "fiber": [0.0, 0.00490487966100879, 0.023408082775107496]}, "cv": {"info": [0.0, 0.0035602767263988185, 0.0884395711674279], "cv": [0.0, 2.5809568279517847e-08, 2.5809568279517847e-08], "fiber": [0.0, 0.006986103593511643, 0.09087376091165345]}, "fiber": {"info": [2.1073424255447017e-08, 0.00490487966096352, 0.023408082775107496], "cv": [2.1073424255447017e-08, 0.006986103593432183, 0.09087376091165467], "fiber": [0.0, 0.0, 2.1073424255447017e-08]}}`.

See `summary.csv` for endpoint ambiguity, fiber conditioning, local closure, velocity-gap, rollout, hidden-law, and robustness diagnostics. The seven prespecified figure families are emitted as PNGs.

## Gradient validation

Gradient-validation results are produced by `tests/test_observable_design_toy.py`; they are intentionally not fabricated into this run artifact. Run `python3 -m pytest tests/test_observable_design_toy.py -q` in the validated environment.

## Interpretation

No positive outcome is assumed. Inspect the pairwise angles and untouched-bank contrasts. A smoke run is only evidence that the protocol executes; confirmatory conclusions require the prespecified crossed model-seed/evaluation-bank run.

## Current scope note

The rotation panel reports frozen-A endpoint feasibility with the common rotated target recomputed. It is not labeled as a zero-shot downstream-network result. Full matched-compute downstream retraining per rotation is reserved for confirmatory execution.
