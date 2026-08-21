# Corrected nested Toy Full sweep

**PASS** — selection is based only on the frozen 64-trial bank; validation is diagnostic and post-selection.

Authoritative Full rule: positive-support physical-`q_h`, directly deposited signed source, no density floor in the scientific operator, `101 x 101`, all 21 time nodes, frozen Scott bandwidth `0.417530106552`.

## Final corrected nested table

| Allow. | Full geometry (deg) | L | R | Risk inc. | Selection A_full | Validation A_full (SE) | Full vs Law validation | A_tan,h | A_hid,h | Gamma_h | L/R | Sel./Val. cert. | Changed? |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| 0.5% | `[23.0867771118319, 69.06085054685987]` | 0.0687593711 | 0.0662118856 | 0.040% | 27.5739506 | 26.6186459 (0.856) | 8.256% | 0.719446999 | 26.8545036 | 0.973908 | PASS | PASS | yes |
| 1% | `[20.791665181605865, 71.67431874734494]` | 0.0691746199 | 0.0666986087 | 0.775% | 21.0402594 | 20.310305 (0.634) | 29.998% | 0.878955905 | 20.1613035 | 0.958225 | PASS | PASS | yes |
| 2% | `[21.145218572199138, 72.02787213793822]` | 0.0691928978 | 0.0666834956 | 0.752% | 20.3224317 | 19.7469959 (0.542) | 31.940% | 0.876657084 | 19.4457746 | 0.956863 | PASS | PASS | yes |
| 3% | `[21.49877196279241, 72.3814255285315]` | 0.0692174816 | 0.0666745725 | 0.739% | 19.6714023 | 19.2430822 (0.461) | 33.677% | 0.874378712 | 18.7970236 | 0.955551 | PASS | PASS | yes |
| 4% | `[21.67554865808905, 72.55820222382813]` | 0.0692321109 | 0.0666724061 | 0.735% | 19.370319 | 19.0131486 (0.425) | 34.469% | 0.873247181 | 18.4970718 | 0.954918 | PASS | PASS | yes |
| 5% | `[21.67554865808905, 72.55820222382813]` | 0.0692321109 | 0.0666724061 | 0.735% | 19.370319 | 19.0131486 (0.425) | 34.469% | 0.873247181 | 18.4970718 | 0.954918 | PASS | PASS | yes |

## Nested-stage decisions

| Allow. | Previous action | Best audited action | Winner action | Replaced? | Nested? | Candidates / feasible / exact |
|---:|---:|---:|---:|:---:|:---:|---:|
| 1% | 27.5739506 | 21.0402594 | 21.0402594 | yes | PASS | 144 / 119 / 7 |
| 2% | 21.0402594 | 20.3224317 | 20.3224317 | yes | PASS | 198 / 139 / 13 |
| 3% | 20.3224317 | 19.6714023 | 19.6714023 | yes | PASS | 204 / 143 / 19 |
| 4% | 19.6714023 | 19.370319 | 19.370319 | yes | PASS | 209 / 147 / 25 |
| 5% | 19.370319 | 19.370319 | 19.370319 | no | PASS | 212 / 150 / 31 |

## Final checks

1. Corrected Full selection curve nested: **PASS**. Raw consecutive differences: `[-6.533691115285315, -0.7178277686258383, -0.6510293731814549, -0.3010833071670298, 0.0]`.
2. Full-vs-Law validation reductions: 0.5% = 8.256%, 1% = 29.998%, 2% = 31.940%, 3% = 33.677%, 4% = 34.469%, 5% = 34.469%.
3. Historical 2–5% geometries changed at: **[2.0, 3.0, 4.0, 5.0]**.
4. Any new candidate beats the corrected 1% incumbent: **YES**.
5. Tangent-vs-Full corrected ranking survives: **YES**.
6. All selection and validation decomposition checks resolved: **YES**.
7. Central FIDE result survives: **YES**.
8. Further targeted Full optimization required: **NO**.
9. Historical Pareto outputs and frozen inputs unchanged: **YES**.

## Numerical certification flags

Every selection and validation row passes all named checks: positive `q_h`, raster mass, signed-source compatibility, conductive-component compatibility, solver convergence, trial validity, physical Poisson residual, Full moment feasibility, Tangent moment feasibility, hidden nullspace, orthogonality, Pythagorean identity, and raw hierarchy. The CSV exposes each check as a separate `selection_flag_*` and `validation_flag_*` Boolean; the JSON retains the full-precision residual maxima and tolerances.

Full-precision summaries, certification maxima, hashes, and validation trials are in the companion JSON and CSV files. Every allowance, including the fixed 0.5% and 1% inputs, has candidate, audit, and validation records in its corresponding `risk_*pct/` subdirectory.
