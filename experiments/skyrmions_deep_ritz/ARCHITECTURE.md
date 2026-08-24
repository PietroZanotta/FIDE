# Architecture and compatibility map

The vortices experiment established the repository conventions used here:
configuration-driven runs, endpoint-only frozen references, deterministic common
banks, anchored spline moment reconstruction, hard empirical tilting, staged
risk/action screening, authoritative candidate re-audits, independent validation,
JSON/CSV manifests, timing records, and a gated percentage sweep.

The skyrmion experiment keeps those contracts but cannot reuse the vortices'
two-dimensional state assumptions. The compatibility boundary is therefore:

| Concern | Reused shared path | Skyrmion-local path |
|---|---|---|
| Configuration/smoke overlay | `mfsi.config` | `config.json` |
| Moment reconstruction | `mfsi.moments.AnchoredCubicSplineReconstructor` | observation bank construction |
| Hard I-projection | `mfsi.projection` / native Tesseract backend | strict acceptance wrapper |
| Caching and output | `mfsi.cache`, `mfsi.io` | role manifests/checkpoints |
| Hidden dynamics | — | fixed-`N` Thiele simulator |
| Frozen reference | endpoint-only training convention | equivariant many-body network |
| Measurements/risk | search conventions | invariant configuration observables |
| Full-law solve | unchanged weighted weak action | permutation-invariant JAX Deep Ritz |

Nothing in the existing reference, particle, raster, Poisson, selection, or
experiment modules is modified. The only repository-level packaging change is an
additive `skyrmions` optional dependency group for SciPy/Matplotlib.

The hot data layout is `[time, configuration, particle, xy]`. JAX vectorizes
over configurations and particles; Python loops exist only over outer candidate
designs and optimizer logging. Projection, Ritz optimization, Ritz audit, and
validation use different deterministic bank identifiers. Selection access to a
validation-role bank raises immediately.

The run sequence is risk-first: geometry → reconstruction → hard projection →
ESS/risk screen → cheap forcing proxy → authoritative Adam/L-BFGS solve → held-out
equation certificates. A failed inner solve has `valid=false` and infinite
eligibility cost; its kinetic energy is never used to rank winners.
