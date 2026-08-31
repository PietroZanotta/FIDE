# Vortices V2.1 randomness provenance

Status: **FROZEN BEFORE ANY V2.1 SELECTION BANK**

All newly chosen V2.1 RNG seeds and observation namespaces are ordinary
two-digit integers. Historical reference seeds and failed-V2 identifiers remain
unchanged because altering them would destroy provenance and would require
retraining the frozen references.

The allocation rule is outcome-blind: assign the first unused values from the
declared role ranges before generating or inspecting any V2.1 observation.
Bank/global roles use `10:15`, Tangent allowance roles use `20:25`, and the 18
Full allowance/round roles use `30:47`. Repository and relevant Git-history
searches found no prior assignment of these exact field/value pairs in the
Vortices experiment.

| Role | Value |
|---|---:|
| Observation generation seed | 10 |
| Selection namespace | 11 |
| Development stress-test namespace | 12 |
| Final validation namespace | 13 |
| Optimizer root seed | 14 |
| Bootstrap seed | 15 |
| Tangent allowance seeds | 20, 21, 22, 23, 24, 25 |
| Full allowance/round seeds | 30 through 47, in allowance-major order |

No alternative seed schedule will be evaluated. The old V2 selection namespace
`410000101` and unused old validation namespace `410000102` are permanently
retired from V2.1.
