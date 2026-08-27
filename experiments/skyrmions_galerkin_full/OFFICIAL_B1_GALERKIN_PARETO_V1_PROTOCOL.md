# Official B1 Galerkin Pareto v1 Protocol

This is a new official skyrmion experiment, authorized only by the prospectively
frozen `PRODUCTION_LAUNCH_READY` result in
`outputs/skyrmion_b1_final_support_confirmation_v1/summary.json`.

The sole reference is the endpoint-qualified, particle-matched B1 checkpoint
with SHA-256 `1e13e2ea58df122702d4f555f8788a148b3150bbfbfc953cbac9f963c03d539b`.
It is not retrained. Configuration-level OT is disabled. All selection and
validation seeds, source hashes, bank sizes, numerical gates, candidate mixture,
start counts, incumbent rules, validation firewall, and risk arithmetic are
materialized in the immutable machine-readable `protocol.json` before official
data generation.

Fresh selection data comprise 6,000 truth trajectories and role-disjoint B1
reference banks of sizes 32,768 (Law search), 32,768 (risk anchor), 8,192
(screen), 32,768 (continuous search), 16,384 (periodic audit), and 65,536 each
(authoritative train and audit). The official Law is reconstructed from scratch
by scientific-risk search and independently anchored; development risk is never
an official ceiling. A fresh 4,096-member canonical pool uses a frozen
35/25/20/20 local, risk-tangent, periodic-path, and Sobol mixture.

Tangent and Full are independent allowance-by-allowance branches. Full is the
fixed-feature finite-dimensional K=280 Galerkin approximation, assembled with
JAX, using rank-aware pseudoinverses and fixed-coefficient envelope derivatives;
it is not an infinite-dimensional converged solution and is not Deep Ritz.
Previously certified winners are mandatory nested incumbents and replacement
requires improvement beyond the frozen tolerance.

Validation seeds are frozen in advance, but validation arrays are forbidden
until the complete selection hash exists. Fresh validation uses 5,000 truth,
16,384 reference-fit, and 16,384 independent reference-audit samples. The
predeclared validation rule is `R_method,val <= (1+p/100+0.05) R_Law,val`, with
strict nominal-p results reported separately. Validation cannot alter selection.
