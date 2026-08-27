# Final frozen 3% Galerkin cross-check

## Scope and repository isolation

This is a retrospective characterization of three already frozen geometries.
It performed no sensor optimization, added no starts, changed no dictionary or
hyperparameter, did not alter the risk allowance, and did not reopen selection.
The already-open validation banks were used only to compare the frozen pair.

The initial `git status --short` was:

```text
 M experiments/skyrmions_deep_ritz_full/README.md
 M experiments/skyrmions_deep_ritz_full/deep_ritz.py
 M experiments/skyrmions_deep_ritz_full/run.py
 M experiments/skyrmions_deep_ritz_full/workflow.py
?? experiments/skyrmions_deep_ritz_full/AUTHORITATIVE_GPU_ACCELERATION.md
?? experiments/skyrmions_deep_ritz_full/AUTHORITATIVE_STABILITY_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/FAST_PRODUCTION_3PCT_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/GALERKIN_ONLY_3PCT_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/authoritative_platform.py
?? experiments/skyrmions_deep_ritz_full/authoritative_stability.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only_data.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only_run.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only_workflow.py
?? experiments/skyrmions_deep_ritz_full/test_fast_production.py
?? experiments/skyrmions_deep_ritz_full/test_galerkin_only.py
```

All new code and machine-readable output are confined to this isolated
experiment. New numerical records are under `outputs/final_3pct_crosscheck/`.
The sealed `outputs/galerkin_only_3pct/selection/result.json` and
`validation/result.json` were read and hash-recorded but not overwritten.

## Frozen geometries

The precise geometries used throughout were:

```text
eta_Law = [
  0.890286510596537, 0.22728952886850587,
  1.3103688321444902, 0.8591631921629669,
  0.7975888227142434, 0.5357230013163333,
  1.6103431504475714, 0.583219225445585
]

eta0 = [
  0.8954153767761239, 0.20592631632470587,
  1.3343788098383822, 0.8654288352917223,
  0.7508355365766083, 0.5179100329264751,
  1.6423735249784726, 0.5883599695898114
]

eta_grad = [
  0.895371148114089, 0.205982940238786,
  1.334525121515147, 0.865464965382237,
  0.750749623351011, 0.518133188490931,
  1.642405611981796, 0.588309862016330
]
```

`eta_Law` and `eta0` come from the frozen configuration; `eta_grad` is the
selection winner frozen before validation. None was changed.

## Validation-risk protocol resolution

The original declared skyrmion validation rule is unambiguous:

```text
R_Full,val <= (1 + p/100 + validation_relative_slack) R_Law,val
validation_relative_slack = 0.05
```

At `p=3`, the predeclared validation multiplier is therefore `1.08`. The
`0.05` is an additive five percentage points in the multiplier, not 5% of the
already inflated 1.03 ceiling.

Concrete evidence predating the current Galerkin validation is:

- `experiments/skyrmions_deep_ritz/config.json:126-130` declares
  `risk_allowance_percent=3.0` and `validation_relative_slack=0.05`.
- `experiments/skyrmions_deep_ritz/experiment.py:1383-1385` implements
  `1 + allowance/100 + validation_relative_slack` against validation Law risk.
- `experiments/skyrmions_deep_ritz/README.md:823-830` reports that the original
  3% result passes the declared “3% plus 5% validation neighborhood.”
- The frozen production artifact embeds the same config and records
  `validation_contrast.scientific_risk_neighborhood_pass=true`.
- Local Git history shows the original config and implementation in commit
  `6eb04d8e` dated 2026-08-23, before this 2026-08-25 retrospective check.

No local manuscript TeX file is present. The original code, config, frozen
metadata, and local experiment report all agree. Thus this is not a post-hoc
protocol change. The earlier Galerkin-only report's use of `1.03` for validation
was an implementation/reporting error; its sealed actions and geometries remain
unchanged.

With directly recomputed validation Law risk
`R_Law,val = 5.357974522307318`:

| convention | multiplier | ceiling | eta0 (`5.548626547535`) | eta_grad (`5.550507348678`) |
|---|---:|---:|---|---|
| strict 3% comparison only | 1.03 | 5.518713757977 | fail | fail |
| predeclared 3% + 5pp validation slack | **1.08** | **5.786612484092** | **pass** | **pass** |

The final classification uses only the predeclared 1.08 rule. The strict 1.03
result is retained solely for transparency.

## Direct K=280 eta-gradient validation

The fixed-coefficient envelope gradient at eta0 was deterministic across two
identical evaluations and finite:

```text
[ 0.620870341622904, -0.478218577516537,
 -2.649835429074640, -0.879768128402833,
  0.516433358902474, -1.240854168159048,
 -0.373835755462651, -0.939225845709615]
```

The center train action was `0.293500059182`. Its timewise ranks were
`[279, 280, ..., 280]`, minimum rank fraction `0.996429`, identity residual
`3.70e-11`, worst range/stationarity residuals `1.43e-9`, and worst retained
condition `4.045e11`. The full selection certificate passed.

The five normalized directions exactly reuse the established production
directions. Every plus/minus point independently recomputed moments,
I-projection, forcing, K/f, rank-aware coefficients, and action. All 80 points
were geometry-valid, retained the center rank, and passed train/audit forcing
and algebra gates. Across them, worst train/audit projection residuals were
`9.55e-11` / `9.74e-11`, minimum train/audit ESS fractions `0.06442` /
`0.06257`, worst train/audit forcing means `1.47e-8` / `2.56e-8`, worst range
and stationarity residuals `1.58e-9`, and worst retained condition `4.070e11`.

### Direction 1

```text
v = [-0.167760524, -0.171419664, 0.261076945, -0.369159940,
     -0.473448647,  0.596764257, -0.291230976, -0.270064423]
AD = -1.01169530871
```

| epsilon | FD | absolute error | relative error | accepted |
|---:|---:|---:|---:|---|
| 1e-2 | -1.00617481112 | 5.52050e-3 | 5.45668e-3 | yes |
| 5e-3 | -1.01017874954 | 1.51656e-3 | 1.49903e-3 | yes |
| 3e-3 | -1.01096042133 | 7.34887e-4 | 7.26392e-4 | yes |
| 1e-3 | -1.01131139421 | 3.83915e-4 | 3.79476e-4 | yes |
| 5e-4 | -1.01149400020 | 2.01309e-4 | 1.98981e-4 | yes |
| 3e-4 | -1.01138257013 | 3.12739e-4 | 3.09123e-4 | yes |
| 1e-4 | -1.01111547733 | 5.79831e-4 | 5.73128e-4 | yes |
| 3e-5 | -1.01163211406 | 6.31946e-5 | 6.24641e-5 | yes |

### Direction 2

```text
v = [ 0.339070924,  0.247302041,  0.666630463,  0.085793478,
     -0.098288407, -0.307725385, -0.475065115, -0.205118324]
AD = -1.04835306303
```

| epsilon | FD | absolute error | relative error | accepted |
|---:|---:|---:|---:|---|
| 1e-2 | -1.03174147527 | 1.66116e-2 | 1.58454e-2 | yes |
| 5e-3 | -1.04426242146 | 4.09064e-3 | 3.90197e-3 | yes |
| 3e-3 | -1.04699188924 | 1.36117e-3 | 1.29839e-3 | yes |
| 1e-3 | -1.04825538754 | 9.76755e-5 | 9.31704e-5 | yes |
| 5e-4 | -1.04848368382 | 1.30621e-4 | 1.24581e-4 | yes |
| 3e-4 | -1.04856249385 | 2.09431e-4 | 1.99731e-4 | yes |
| 1e-4 | -1.04843786575 | 8.48027e-5 | 8.08848e-5 | yes |
| 3e-5 | -1.04844931240 | 9.62494e-5 | 9.18017e-5 | yes |

### Direction 3

```text
v = [-0.317189260,  0.623425629,  0.087234421, -0.010704887,
     -0.253697108, -0.287407416,  0.529538840,  0.275009802]
AD = -0.947449988038
```

| epsilon | FD | absolute error | relative error | accepted |
|---:|---:|---:|---:|---|
| 1e-2 | -0.941635445094 | 5.81454e-3 | 6.13704e-3 | yes |
| 5e-3 | -0.946338659871 | 1.11133e-3 | 1.17297e-3 | yes |
| 3e-3 | -0.947320531016 | 1.29457e-4 | 1.36637e-4 | yes |
| 1e-3 | -0.947821686537 | 3.71698e-4 | 3.92161e-4 | yes |
| 5e-4 | -0.947841898030 | 3.91910e-4 | 4.13476e-4 | yes |
| 3e-4 | -0.947951171163 | 5.01183e-4 | 5.28701e-4 | yes |
| 1e-4 | -0.947872294183 | 4.22306e-4 | 4.45531e-4 | yes |
| 3e-5 | -0.947784496682 | 3.34509e-4 | 3.52937e-4 | yes |

### Direction 4

```text
v = [-0.307435667,  0.603603030, -0.388662315,  0.179150455,
     -0.460796213,  0.264979403, -0.179992660, -0.207482098]
AD = 0.0881373493669
```

| epsilon | FD | absolute error | relative error | accepted |
|---:|---:|---:|---:|---|
| 1e-2 | 0.091200458621 | 3.06311e-3 | 3.35866e-2 | yes |
| 5e-3 | 0.088468193826 | 3.30844e-4 | 3.73970e-3 | yes |
| 3e-3 | 0.087914191065 | 2.23158e-4 | 2.53194e-3 | yes |
| 1e-3 | 0.087792174327 | 3.45175e-4 | 3.91633e-3 | yes |
| 5e-4 | 0.087898867963 | 2.38481e-4 | 2.70579e-3 | yes |
| 3e-4 | 0.087862499494 | 2.74850e-4 | 3.11843e-3 | yes |
| 1e-4 | 0.087804708822 | 3.32641e-4 | 3.77412e-3 | yes |
| 3e-5 | 0.087837074073 | 3.00275e-4 | 3.40690e-3 | yes |

### Direction 5

```text
v = [-0.731861458, -0.268706041, -0.021719638, -0.054337594,
     -0.264022868,  0.459984760,  0.238570228, -0.224814912]
AD = -0.805691233336
```

| epsilon | FD | absolute error | relative error | accepted |
|---:|---:|---:|---:|---|
| 1e-2 | -0.804910643715 | 7.80590e-4 | 9.68845e-4 | yes |
| 5e-3 | -0.805345018816 | 3.46215e-4 | 4.29711e-4 | yes |
| 3e-3 | -0.805490873904 | 2.00359e-4 | 2.48680e-4 | yes |
| 1e-3 | -0.805569802728 | 1.21431e-4 | 1.50716e-4 | yes |
| 5e-4 | -0.805416277025 | 2.74956e-4 | 3.41268e-4 | yes |
| 3e-4 | -0.805413753484 | 2.77480e-4 | 3.44400e-4 | yes |
| 1e-4 | -0.805512782433 | 1.78451e-4 | 2.21488e-4 | yes |
| 3e-5 | -0.805656108517 | 3.51248e-5 | 4.35959e-5 | yes |

All five directions pass: every direction has at least three consecutive sign
matches, at least two consecutive errors below 2%, an error-decrease regime,
and a point below the preferred 0.5% discrepancy. The observed small-epsilon
floor does not replace the clear preceding convergence regime.

**K=280 eta gradient: PASS (5/5 directions).**

## Cross-K frozen-pair ordering

In the diagnostic columns below, values are `eta0 / eta_grad`. Rank, range,
stationarity, and condition refer to the independently solved fit/train system;
weak/energy/gauge/moment refer to the named data view. Every eta0 and eta_grad
row passes all applicable gates.

### Selection train

| K | eta0 action | eta_grad action | delta | improvement | weak | energy | gauge | moment | rank frac. | range | stationarity | condition |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 160 | 0.264577196897 | 0.263925318141 | -0.000651878756 | 0.246385% | .023070/.022954 | 2.28e-9/4.35e-9 | 6.94e-18/9.78e-18 | .001061/.001060 | .99375/.99375 | 2.51e-9/2.52e-9 | 2.51e-9/2.52e-9 | 3.719e11/3.719e11 |
| 200 | 0.279589965806 | 0.278886992265 | -0.000702973542 | 0.251430% | .022353/.022242 | 2.00e-9/3.35e-9 | 1.42e-17/1.88e-17 | 3.16e-5/3.15e-5 | .995/.995 | 1.81e-9/1.81e-9 | 1.81e-9/1.81e-9 | 3.767e11/3.767e11 |
| 240 | 0.286457888355 | 0.285722618233 | -0.000735270122 | 0.256677% | .021847/.021738 | 1.84e-9/2.89e-9 | 1.24e-17/2.50e-17 | 1.06e-5/1.06e-5 | .99583/.99583 | 1.71e-9/1.72e-9 | 1.71e-9/1.72e-9 | 3.828e11/3.828e11 |
| 280 | 0.293500059182 | 0.292740724350 | -0.000759334833 | 0.258717% | .021387/.021276 | 2.41e-9/3.02e-9 | 1.25e-17/1.75e-17 | 1.09e-5/1.09e-5 | .99643/.99643 | 1.43e-9/1.44e-9 | 1.43e-9/1.44e-9 | 4.045e11/4.045e11 |

### Selection audit

| K | eta0 action | eta_grad action | delta | improvement | weak | energy | gauge | moment | rank frac. | range | stationarity | condition |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 160 | 0.265053280485 | 0.264396987699 | -0.000656292787 | 0.247608% | .074013/.073863 | .050369/.050051 | 1.54e-17/9.87e-18 | .007939/.007920 | .99375/.99375 | 2.51e-9/2.52e-9 | 2.51e-9/2.52e-9 | 3.719e11/3.719e11 |
| 200 | 0.281308617161 | 0.280598124478 | -0.000710492684 | 0.252567% | .071466/.071328 | .069929/.069539 | 1.98e-17/1.75e-17 | .008815/.008788 | .995/.995 | 1.81e-9/1.81e-9 | 1.81e-9/1.81e-9 | 3.767e11/3.767e11 |
| 240 | 0.289068593314 | 0.288318833317 | -0.000749759997 | 0.259371% | .069904/.069777 | .076748/.076349 | 1.08e-17/9.20e-18 | .009819/.009793 | .99583/.99583 | 1.71e-9/1.72e-9 | 1.71e-9/1.72e-9 | 3.828e11/3.828e11 |
| 280 | 0.296692769256 | 0.295913390075 | -0.000779379180 | 0.262689% | .069147/.069030 | .079867/.079468 | 9.97e-18/2.01e-17 | .010403/.010377 | .99643/.99643 | 1.43e-9/1.44e-9 | 1.43e-9/1.44e-9 | 4.045e11/4.045e11 |

### Validation fit

| K | eta0 action | eta_grad action | delta | improvement | weak | energy | gauge | moment | rank frac. | range | stationarity | condition |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 160 | 0.220014310911 | 0.219471628440 | -0.000542682471 | 0.246658% | .012218/.012087 | 6.67e-11/8.39e-11 | 2.06e-17/9.07e-18 | .000998/.000993 | .99375/.99375 | 5.83e-9/5.84e-9 | 5.83e-9/5.83e-9 | 4.116e11/4.117e11 |
| 200 | 0.231683522413 | 0.231110910033 | -0.000572612379 | 0.247153% | .011556/.011423 | 5.58e-11/6.70e-11 | 1.79e-17/2.15e-17 | 3.49e-5/3.48e-5 | .995/.995 | 4.43e-9/4.43e-9 | 4.43e-9/4.43e-9 | 4.168e11/4.168e11 |
| 240 | 0.235349713907 | 0.234762806535 | -0.000586907372 | 0.249377% | .012106/.011972 | 6.03e-11/5.56e-11 | 2.02e-17/3.10e-17 | 1.14e-5/1.14e-5 | .99583/.99583 | 4.69e-9/4.69e-9 | 4.69e-9/4.69e-9 | 4.234e11/4.235e11 |
| 280 | 0.239936893722 | 0.239336294326 | -0.000600599396 | 0.250316% | .011407/.011274 | 4.12e-11/5.74e-11 | 2.25e-17/9.23e-18 | 1.16e-5/1.16e-5 | .99643/.99643 | 3.52e-9/3.52e-9 | 3.52e-9/3.52e-9 | 4.475e11/4.476e11 |

### Validation audit

| K | eta0 action | eta_grad action | delta | improvement | weak | energy | gauge | moment | rank frac. | range | stationarity | condition |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 160 | 0.218649623471 | 0.218110228464 | -0.000539395008 | 0.246694% | .029612/.029524 | .058030/.057980 | 1.65e-17/2.11e-17 | .005209/.005187 | .99375/.99375 | 5.83e-9/5.84e-9 | 5.83e-9/5.83e-9 | 4.116e11/4.117e11 |
| 200 | 0.230333132985 | 0.229765306029 | -0.000567826955 | 0.246524% | .029152/.029065 | .061923/.061869 | 2.91e-17/1.69e-17 | .005789/.005766 | .995/.995 | 4.43e-9/4.43e-9 | 4.43e-9/4.43e-9 | 4.168e11/4.168e11 |
| 240 | 0.234046293780 | 0.233464122240 | -0.000582171540 | 0.248742% | .029261/.029172 | .064878/.064808 | 1.24e-17/2.87e-17 | .005792/.005769 | .99583/.99583 | 4.69e-9/4.69e-9 | 4.69e-9/4.69e-9 | 4.234e11/4.235e11 |
| 280 | 0.238797637727 | 0.238200776208 | -0.000596861519 | 0.249944% | .029148/.029060 | .066467/.066399 | 1.48e-17/3.22e-17 | .005786/.005764 | .99643/.99643 | 3.52e-9/3.52e-9 | 3.52e-9/3.52e-9 | 4.475e11/4.476e11 |

The pairwise ordering is therefore robust:

```text
delta_K < 0 for K = 160, 200, 240, 280
```

on selection train, selection audit, validation fit, and validation audit. The
incremental improvement stays in the narrow `0.246%–0.263%` range even though
the absolute Galerkin action continues to move with K.

## Common K=280 Law / eta0 / eta_grad comparison

All actions below use the identical K=280 fixed-feature Galerkin solver.

| data view | Law | eta0 | eta_grad | eta_grad over Law | eta0 over Law | eta_grad over eta0 |
|---|---:|---:|---:|---:|---:|---:|
| selection train | 0.374832445634 | 0.293500059182 | 0.292740724350 | **21.900911%** | 21.698331% | **0.258717%** |
| selection audit | 0.378294866798 | 0.296692769256 | 0.295913390075 | **21.777054%** | 21.571030% | **0.262689%** |
| validation fit | 0.312643615507 | 0.239936893722 | 0.239336294326 | **23.447567%** | 23.255463% | **0.250316%** |
| validation audit | 0.313203630213 | 0.238797637727 | 0.238200776208 | **23.946994%** | 23.756427% | **0.249944%** |

At K=280, all three designs pass geometry, projection/ESS/forcing, rank,
range/stationarity, conditioning, and restricted identity gates. Eta0 and
eta_grad pass fit and audit physical certificates on selection and validation.
Law passes both validation certificates and its selection-train certificate,
but its selection-audit energy residual is `0.108442 > 0.08`; this limitation
is reported rather than repaired. The Law algebra and forcing gates pass, and
the same-solver action comparison is numerically well defined, but the failed
selection-audit physical certificate weakens a fully certified interpretation
of the selection-side FIDE percentage. The validation-side Law comparisons are
fully certified.

K=280 selection audit diagnostics:

| design | risk | train/audit action | train/audit forcing valid | algebra | audit weak | audit energy | audit gauge | audit moment | audit certificate |
|---|---:|---:|---|---|---:|---:|---:|---:|---|
| Law | 5.186549474478 | .374832446/.378294867 | yes/yes | pass | .070193 | .108442 | 2.08e-17 | .013487 | **fail** |
| eta0 | 5.340106050966 | .293500059/.296692769 | yes/yes | pass | .069147 | .079867 | 9.97e-18 | .010403 | pass |
| eta_grad | 5.342099811291 | .292740724/.295913390 | yes/yes | pass | .069030 | .079468 | 2.01e-17 | .010377 | pass |

K=280 validation audit diagnostics:

| design | risk | fit/audit action | fit/audit forcing valid | algebra | audit weak | audit energy | audit gauge | audit moment | audit certificate |
|---|---:|---:|---|---|---:|---:|---:|---:|---|
| Law | 5.357974522307 | .312643616/.313203630 | yes/yes | pass | .036607 | .059739 | 1.27e-17 | .006788 | pass |
| eta0 | 5.548626547535 | .239936894/.238797638 | yes/yes | pass | .029148 | .066467 | 1.48e-17 | .005786 | pass |
| eta_grad | 5.550507348678 | .239336294/.238200776 | yes/yes | pass | .029060 | .066399 | 3.22e-17 | .005764 | pass |

The corresponding fit/audit projection residual, ESS, and pre-centering forcing
mean triples are:

| stage/design | fit projection / ESS / forcing mean | audit projection / ESS / forcing mean |
|---|---|---|
| selection Law | 9.14e-12 / .055569 / 2.00e-9 | 1.16e-11 / .057297 / 1.83e-9 |
| selection eta0 | 8.03e-11 / .071017 / 9.76e-9 | 1.47e-11 / .069163 / 2.49e-9 |
| selection eta_grad | 7.54e-11 / .071481 / 9.13e-9 | 1.33e-11 / .069644 / 2.24e-9 |
| validation Law | 8.42e-11 / .068987 / 2.68e-8 | 2.17e-11 / .078892 / 4.16e-9 |
| validation eta0 | 3.90e-11 / .089653 / 3.17e-8 | 3.09e-11 / .104392 / 4.03e-9 |
| validation eta_grad | 3.91e-11 / .090248 / 3.18e-8 | 3.10e-11 / .105001 / 4.05e-9 |

## Exact risk distances

| design | selection risk | selection ratio over Law | validation risk | validation ratio over Law |
|---|---:|---:|---:|---:|
| Law | 5.186549474478 | 0% | 5.357974522307 | 0% |
| eta0 | 5.340106050966 | 2.960669% | 5.548626547535 | 3.558285% |
| eta_grad | 5.342099811291 | **2.999110%** | 5.550507348678 | **3.593388%** |

For eta_grad, the distance to the selection 3% boundary is
`0.000889752` percentage points (`8.89752e-6` as a ratio). Its distance to the
actual predeclared 8% validation boundary is `4.406612` percentage points.
No significance or extra safety threshold is invented.

## Conclusions, limitations, and recommendation

The four requested questions are answered:

1. The original validation rule is 3% plus an additive 5 percentage-point
   validation slack, giving a 1.08 multiplier. Both eta0 and eta_grad pass it.
2. The exact K=280 gradient passes all five direct multi-epsilon AD/FD checks.
3. eta_grad beats eta0 at every K on all four train/audit/fit views.
4. With one K=280 solver, eta_grad improves over Law by 21.78–21.90% on
   selection and 23.45–23.95% on validation; its additional improvement over
   eta0 is 0.25–0.26%.

Limitations remain:

- absolute action convergence in K was not established; the result is robust
  pairwise ordering in the declared finite nested spaces;
- the incremental continuous improvement is small, though its sign and size are
  unusually stable across K and banks;
- K=280 eta0/eta_grad selection-audit energy residuals remain close to 0.08;
- K=280 Law fails the selection-audit energy certificate, so the selection-side
  Law percentage is a same-discretization diagnostic rather than a fully
  certified physical comparison; validation Law passes;
- validation was already open, so any future methodology redesign would need a
  fresh untouched validation bank;
- no Pareto sweep was run here.

Because the final classification is A, a full allowance sweep is scientifically
permissible. Whether it is worth running is a cost/communication decision: the
extra continuous gain is only about 0.25%, so a sweep is warranted if the paper
needs a continuous-refinement Pareto curve or wants to test whether this stable
increment accumulates at other allowances. It is not needed merely to support
the frozen 3% claim established here.

## Verification and final repository audit

The 9 new retrospective tests and 20 Galerkin-only tests pass, including direct
K=280 reproducibility, the centered-FD helper, exact nested dictionary reuse,
frozen-pair comparisons, protocol arithmetic, common-solver improvements,
selection immutability, output isolation, and live CPU/GPU equivalence. The
unchanged 62 historical isolated tests also pass. Total: **91 tests passed**.

The sealed selection SHA-256 remains
`32ef8fa0f76a5c0cdacc8381fb1345ad1315d3ec5ece8d400c225eff78d829c2`
and still exactly matches the hash embedded in the prior sealed validation.
The sealed validation SHA-256 is
`fe143132293aebdd379f958a2a6b350440c439193db9d790a0b3c83fbf7f9206`.
Their timestamps precede the first new cross-check output. Static source audit
finds no optimization, Deep Ritz, or sealed-output write call in the new
workflow. `git diff --check` passes.

Final `git status --short`:

```text
 M experiments/skyrmions_deep_ritz_full/README.md
 M experiments/skyrmions_deep_ritz_full/deep_ritz.py
 M experiments/skyrmions_deep_ritz_full/run.py
 M experiments/skyrmions_deep_ritz_full/workflow.py
?? experiments/skyrmions_deep_ritz_full/AUTHORITATIVE_GPU_ACCELERATION.md
?? experiments/skyrmions_deep_ritz_full/AUTHORITATIVE_STABILITY_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/FAST_PRODUCTION_3PCT_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/FINAL_3PCT_GALERKIN_CROSSCHECK.md
?? experiments/skyrmions_deep_ritz_full/GALERKIN_ONLY_3PCT_EVALUATION.md
?? experiments/skyrmions_deep_ritz_full/authoritative_platform.py
?? experiments/skyrmions_deep_ritz_full/authoritative_stability.py
?? experiments/skyrmions_deep_ritz_full/final_crosscheck.py
?? experiments/skyrmions_deep_ritz_full/final_crosscheck_run.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only_data.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only_run.py
?? experiments/skyrmions_deep_ritz_full/galerkin_only_workflow.py
?? experiments/skyrmions_deep_ritz_full/test_fast_production.py
?? experiments/skyrmions_deep_ritz_full/test_final_crosscheck.py
?? experiments/skyrmions_deep_ritz_full/test_galerkin_only.py
```

This task added only `FINAL_3PCT_GALERKIN_CROSSCHECK.md`,
`final_crosscheck.py`, `final_crosscheck_run.py`, `test_final_crosscheck.py`,
and ignored files under `outputs/final_3pct_crosscheck/`. Every task-created
path is under `experiments/skyrmions_deep_ritz_full/`; all other listed changes
were already present and preserved. No new optimization or Pareto sweep ran,
and no validation quantity entered selection.

A. 3% CONTINUOUS GALERKIN REFINEMENT ROBUSTLY VALIDATED
