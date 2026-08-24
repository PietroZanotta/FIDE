# Certified skyrmion Tangent extension

The original Law/Full artifacts were retained verbatim. The Tangent curve was computed from frozen banks with closed-form Gram solves; no Deep Ritz model was rerun.

Tangent is a lower bound that enforces the selected moment rates only. Full Deep Ritz enforces the complete many-body continuity equation, so the two curves should not be described as equivalent realized transports.

| allowance | Tangent actual risk | Tangent validation action | Tangent gain | Full validation action | Full gain |
|---:|---:|---:|---:|---:|---:|
| 0.5% | 0.490% | 0.083387 | 17.09% | 0.297172 | 10.51% |
| 1% | 0.886% | 0.081550 | 18.92% | 0.266720 | 19.68% |
| 2% | 1.880% | 0.078080 | 22.37% | 0.260606 | 21.52% |
| 3% | 2.482% | 0.072732 | 27.68% | 0.230970 | 30.44% |
| 4% | 3.979% | 0.070593 | 29.81% | 0.230970 | 30.44% |
| 5% | 4.961% | 0.068152 | 32.24% | 0.230970 | 30.44% |

Saved feasible geometries rescored: 611.
New Tangent-only refined geometries retained: 85.
Full Deep Ritz rerun: False.
