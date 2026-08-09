# MFSI Stage 2B: moment-response Gram coupling

## Question

Does adding local moment-fiber geometry to the endpoint representation improve held-out coupling quality beyond geometric OT and the Phi-only coupling?

## Isolated intervention

The geometric Sinkhorn kernel and nine standardized Phi endpoint interactions were retained. The only added representation is a 36-parameter bilinear interaction between the six unique entries of JPhi(X) JPhi(X)^T at the two endpoints. Those six entries use fixed pooled-endpoint centering and coordinatewise scaling, matching the existing Phi feature construction; the preprocessing has no learned parameters. No q4, final-MMD descriptors, neural descriptors, schedule parameters, objectives, optimizer settings, or downstream settings were changed.

## Aggregate metrics

| method | E_corr | min ESS | D_proj | MMD² |
|---|---:|---:|---:|---:|
| independent | 0.163469 | 0.116759 | 0.46964 | 0.017453 |
| geometric OT | 0.167745 | 0.144271 | 0.401206 | 0.0138903 |
| Phi-only | 0.175826 | 0.157872 | 0.413505 | 0.0175201 |
| Phi + Gram | 0.185004 | 0.142575 | 0.442565 | 0.0170152 |

## Primary paired effects

Gram minus geometric E_corr: `0.0172585` (95% interval `-0.0441596` to `0.0786765`).

Gram minus Phi-only E_corr: `0.0091778` (95% interval `-0.0320554` to `0.050411`).

Gram minus geometric MMD²: `0.0031249` (95% interval `-0.00178262` to `0.00803241`).

## Generalization across bank roles

| role | Gram - geometric E_corr | Gram - geometric ESS penalty | Gram - geometric total objective |
|---|---:|---:|---:|
| train | -0.0472724 | -0.00465469 | -0.051927 |
| selection | -0.00191206 | -0.0059952 | -0.00790725 |
| evaluation | 0.0172585 | 0.000399986 | 0.0176584 |

The richer plan lowered the mean objective on the training and selection banks, but the correction-energy effect reversed on the untouched evaluation banks. This is the same parameterization/generalization mismatch the experiment was designed to test.

## Numerical constraint check

The richer logits required 500 fixed inner log-Sinkhorn iterations instead of the Stage-2 default of 100 to converge to the same prescribed endpoint marginals. At 100 iterations the worst residual was `7.8e-6`; at 500 it was `1.09074e-10`. This changes only numerical convergence of the constraint solve: epsilon, target marginals, objective, Adam learning rate, and 60 optimization steps are unchanged. It is not a searched scientific hyperparameter.

## Interpretation and stop decision

The single richer representation does not provide held-out evidence of lower correction burden than either geometric OT or the Phi-only coupling. Its mean projected-law MMD² is also higher than geometric OT, although that paired interval includes zero.

**Stop condition reached:** fiber-aware coupling development should stop for this paper. Geometric OT remains the preferred coupling baseline for any later interaction study. Joint schedule-plus-coupling optimization was not implemented and is not justified by this result.
