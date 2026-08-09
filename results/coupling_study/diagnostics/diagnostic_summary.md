# Stage-2 coupling diagnosis

## 1. Question

Why did fiber-aware coupling fail to transfer beyond geometric OT?

## 2. Objective decomposition

On selection banks, fiber-aware minus geometric E_corr was `-0.00934091`, the scaled ESS-penalty change was `-0.00090081`, and the total-objective change was `-0.0102417`. On untouched evaluation banks the corresponding E_corr change was `0.00808065` and total-objective change was `0.0075883`.

The strict scalarization-mismatch signature (non-improving E_corr offset by ESS penalty) is not present in the selection-bank mean.

## 3. Soft-plan sampling analysis

Each fixed evaluation plan was realized `20` times with `256` pairs per time. Mean within-plan E_corr SD was `0.0481833` for geometric OT and `0.0491579` for fiber-aware (ratio `1.02`). Mean E_corr sampling bias was `0.0283187` and `0.019553`, respectively.

The plan-level held-out fiber advantage is absent, so categorical realization cannot explain away the primary failure.

The fixed neural fields were reconstructed once per bank and method, then only generation/oracle pairs were resampled. Mean within-plan MMD² SD was `0.0049654` for geometric and `0.00496288` for fiber-aware. The mean fiber-minus-geometric difference across each bank's resampling distribution was `0.00114991` (bank-level 95% interval `-0.0012631` to `0.00356292`). The largest deterministic reconstruction error for the original MMD² cell was `2.543e-08`.

Thus categorical realization materially affects the apparent strength of the MMD² result, but it is a secondary uncertainty rather than the cause of the plan-level correction failure.

## 4. Geometric OT effect

Geometric minus independent E_corr was `0.00427606` (`-0.00897367`, `0.0175258`), minimum ESS was `0.0275118` (`0.0142206`, `0.040803`), D_proj was `-0.0684343` (`-0.111162`, `-0.0257069`), and MMD² was `-0.00356274` (`-0.012399`, `0.00527357`).

Geometric OT robustly improves overlap and projection distortion, while its mean MMD² is lower but its five-bank interval crosses zero. Correction energy alone is not sufficient to rank coupling quality in this experiment.

## 5. Metric alignment

Paired-change Pearson and Spearman summaries are stored in `metric_alignment.json`. With n=5 per contrast they are exploratory only. No upstream metric has a stable association sign across all three contrasts. At the aggregate method level, geometric OT combines lower D_proj and higher ESS with lower mean MMD despite slightly higher E_corr; this supports D_proj/ESS as redesign hypotheses, not established predictors.

## 6. Diagnosis

Dominant supported diagnosis: **parameterization/generalization mismatch, with geometric OT already a strong comparator**. A secondary issue is finite-pair realization noise, especially for MMD². Fiber-aware minus geometric held-out plan-level E_corr was `0.00808065` and MMD² was `0.00362982`. The method's selection tendency did not generalize to the untouched plans.

## 7. Next experiment

Run exactly one richer coupling-only extension. Keep the geometric kernel and existing nine Phi interactions, and add a fixed 36-parameter bilinear interaction between the six upper-triangular entries of each endpoint's moment-response Gram matrix JPhi(X) JPhi(X)^T. This embedding is permutation/translation invariant, fiber-specific, and excludes q4 and all final-MMD descriptors. Keep the objective, frozen schedules, optimizer, bank splits, pair realization, and downstream evaluation unchanged; predeclare the 36 added parameters rather than searching architectures.

## 8. What NOT to do

Joint schedule-plus-coupling optimization is not currently justified. Do not expose q4 or final MMD² to optimization, do not alter the schedule in the next coupling test, and do not perform an architecture search. If the single prescribed coupling-only follow-up does not beat geometric OT on held-out fiber metrics without worsening MMD², stop fiber-aware coupling development for this paper.
