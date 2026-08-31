# Vortices V2.1 independent holdout confirmation: 0.5--2% front

Status: **PASS**

Scope: the selected Full geometries at `0.5%`, `1%`, and `2%`, compared with
the common selected Law geometry. No claim is made for `3%`, `4%`, or `5%`;
those selection branches remain paused and uncomputed.

## Confirmed result

The fresh 64-trial shared holdout used observation seed `22`, namespace `23`,
and bootstrap seed `24`. Each of the four designs was evaluated on all three
qualified references with the exact `256 x 128` V2 Full action. All 768 ordered
reference/design/trial evaluations passed every frozen numerical gate.

Full-action reduction relative to Law, with the primary familywise simultaneous
95% interval in brackets:

| Reference | 0.5% | 1% | 2% |
|---|---:|---:|---:|
| 0 | 7.97% [5.52%, 10.43%] | 11.81% [9.36%, 14.27%] | 15.40% [12.95%, 17.86%] |
| 1 | 8.30% [5.85%, 10.75%] | 12.35% [9.90%, 14.80%] | 15.82% [13.37%, 18.27%] |
| 2 | 8.55% [6.10%, 11.01%] | 12.42% [9.97%, 14.87%] | 15.91% [13.46%, 18.37%] |
| Equal-reference descriptive mean | **8.28%** | **12.19%** | **15.71%** |

The common max-deviation simultaneous half-width is `2.454` percentage
points. The maximum within-reference relative standard error is `2.442%`.

## Holdout Pareto coordinates

The secondary finite-risk cross-evaluation retained all 768 ordered values.

| Allowed extra risk | Holdout finite-risk change vs Law | Holdout Full action | Law action | Full-action reduction |
|---:|---:|---:|---:|---:|
| 0.5% | 0.303% | 1.4598 | 1.5915 | 8.28% |
| 1% | 0.850% | 1.3974 | 1.5915 | 12.19% |
| 2% | 1.599% | 1.3414 | 1.5915 | 15.71% |

These holdout risk values are independent cross-evaluations; the frozen
selection receipts remain the authoritative risk certificates.

## Prespecified gates

- Exactly three qualified references: PASS.
- Exactly 64 shared trials: PASS.
- All 768 exact evaluations numerically valid: PASS.
- All nine simultaneous lower bounds strictly positive: PASS.
- Simultaneous half-width at most `.05`: PASS (`.02454`).
- Every within-reference relative SE at most `.10`: PASS (`.02442` maximum).
- No trial top-up or outcome-dependent confirmatory amendment: PASS.

The earlier 1,024-trial bank is permanently retired without an action cell,
master evaluation receipt, inference, or inspected result. A development-only
fused-cell acceleration was 1.19x faster but changed floating-point solver
fields, so it failed exact equivalence and was not used. The accepted ordered
parallel exact solver retained its prior exact-equality qualification.

## Authoritative artifacts

- `outputs/prospective_v2_1_c3_64/execution_receipt.json`
- `outputs/prospective_v2_1_c3_64/shared_confirmatory_bank_receipt.json`
- `outputs/prospective_v2_1_c3_64/exact_action_evaluation_receipt.json`
- `outputs/prospective_v2_1_c3_64/simultaneous_inference.json`
- `outputs/prospective_v2_1_c3_64/finite_risk_evaluation_receipt.json`
- `plots/pareto_0p5_to_2pct.pdf`
- `plots/vortices_v2_1_full_0p5_paper.pdf`
- `plots/vortices_v2_1_full_1p0_paper.pdf`
- `plots/vortices_v2_1_full_2p0_paper.pdf`
- `plots/vortices_v2_1_full_2p0.gif`
