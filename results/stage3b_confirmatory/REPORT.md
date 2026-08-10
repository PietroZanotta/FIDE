# MFSI Stage 3B: confirmatory rollout credit assignment

Ten new seeds (406-415) were executed under the predeclared Stage 3 settings. The original full three-parameter rollout method was unchanged. Two controls were added on the same new bank triples: a scalar full-rollout amplitude and the same three-parameter modulation with state-to-state gradients stopped at each Heun step.

## Untouched evaluation

| method | interior law MMD² | max moment error | interior Phi MMD² |
|---|---:|---:|---:|
| raw SI | 0.016167188 (0.012496429, 0.019837948) | 0.023803527 (0.01991622, 0.027690834) | 0.024801902 (0.020461826, 0.029141978) |
| tangent | 0.0086789437 (0.0044719138, 0.012885973) | 0.0020775115 (0.0014018397, 0.0027531833) | 0.0069018849 (0.0031067335, 0.010697036) |
| frozen neural | 0.013966719 (0.010356771, 0.017576668) | 0.019281178 (0.016493306, 0.02206905) | 0.020133 (0.015420439, 0.024845562) |
| scalar adapted | 0.01343881 (0.0097297499, 0.017147871) | 0.018327176 (0.015670541, 0.020983811) | 0.01852957 (0.014021958, 0.023037182) |
| stopped-state | 0.013319995 (0.0095614027, 0.017078587) | 0.018417035 (0.015856357, 0.020977714) | 0.018430602 (0.014103733, 0.02275747) |
| full rollout | 0.012386303 (0.0086967935, 0.016075812) | 0.01771453 (0.0145427, 0.02088636) | 0.016551189 (0.012246136, 0.020856243) |

## Prespecified paired effects

full_minus_frozen: `-0.0015804164` (95% interval `-0.0027397105` to `-0.0004211224`).

full_minus_scalar: `-0.0010525077` (95% interval `-0.0017714313` to `-0.00033358415`).

full_minus_stopped_state: `-0.00093369206` (95% interval `-0.0015817451` to `-0.00028563906`).

## Combined descriptive estimate

Across the original five and confirmatory ten seeds, full minus frozen interior MMD² is `-0.0022712949` (95% interval `-0.0034298902` to `-0.0011126996`). The new-ten result above remains the confirmatory test.

## Interpretation

The predeclared new-ten confirmation supports lower final-law MMD for full rollout adaptation versus the frozen correction. It also supports the two stronger ablations: the time-dependent modulation beats scalar rollout adaptation, and full temporal credit assignment beats the identical-forward stopped-state gradient control. Tangent remains lower in mean and is not claimed to be beaten.
