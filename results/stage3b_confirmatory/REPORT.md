# MFSI Stage 3B: confirmatory rollout credit assignment

Ten new seeds (406-415) were executed under the predeclared Stage 3 settings. The original full three-parameter rollout method was unchanged. Two controls were added on the same new bank triples: a scalar full-rollout amplitude and the same three-parameter modulation with state-to-state gradients stopped at each Heun step.

## Untouched evaluation

| method | interior law MMD² | max moment error | interior Phi MMD² |
|---|---:|---:|---:|
| raw SI | 0.016167188 (0.012496429, 0.019837948) | 0.023803527 (0.01991622, 0.027690834) | 0.024801902 (0.020461826, 0.029141978) |
| tangent | 0.0086789437 (0.0044719138, 0.012885973) | 0.0020775115 (0.0014018397, 0.0027531833) | 0.0069018849 (0.0031067335, 0.010697036) |
| frozen neural | 0.013966723 (0.010356771, 0.017576676) | 0.01928118 (0.016493306, 0.022069054) | 0.020133008 (0.015420436, 0.02484558) |
| scalar adapted | 0.013438813 (0.0097297514, 0.017147874) | 0.018327174 (0.015670535, 0.020983813) | 0.01852957 (0.014021955, 0.023037186) |
| stopped-state | 0.013328115 (0.0095671589, 0.01708907) | 0.018422496 (0.015860074, 0.020984918) | 0.018449293 (0.014107855, 0.02279073) |
| full rollout | 0.01238723 (0.0086968179, 0.016077642) | 0.017715879 (0.014544532, 0.020887226) | 0.016550386 (0.01224662, 0.020854151) |

## Prespecified paired effects

full_minus_frozen: `-0.0015794936` (95% interval `-0.0027388068` to `-0.00042018036`).

full_minus_scalar: `-0.0010515831` (95% interval `-0.001770353` to `-0.00033281327`).

full_minus_stopped_state: `-0.00094088479` (95% interval `-0.0016024554` to `-0.00027931419`).

## Combined descriptive estimate

Across the original five and confirmatory ten seeds, full minus frozen interior MMD² is `-0.0022706793` (95% interval `-0.0034293834` to `-0.0011119752`). The new-ten result above remains the confirmatory test.

## Interpretation

The predeclared new-ten confirmation supports lower final-law MMD for full rollout adaptation versus the frozen correction. It also supports the two stronger ablations: the time-dependent modulation beats scalar rollout adaptation, and full temporal credit assignment beats the identical-forward stopped-state gradient control. Tangent remains lower in mean and is not claimed to be beaten.
