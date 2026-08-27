# Galerkin resolution study protocol

This is a selection-development-only qualification study. The official Pareto
v1 protocol, report, and output tree are immutable failed records. No sensor
optimization, winner selection, validation construction, or validation access
is permitted.

Six exact frozen geometries are evaluated: Law, historical 0.5%, 1%, and 2%
Full designs, eta0 at 3%, and the continuously refined 3% eta_grad. Provenance,
exact coordinates, and source hashes are sealed in the machine protocol.

The primary study fixes K=280, dictionary SHA-256
`37e9b60fcb92c4e5a0ee7ec1651fb7f8889f7ac6bdb02d3bd314e9ef40833326`,
dictionary ordering and normalization, rank tolerance `1e-12`, all algebra
thresholds, and the physical certificate thresholds including Ritz energy
`0.08`. It varies only nested empirical support:

```text
train/audit = 8192/4096, 16384/8192, 16384/16384, 32768/16384.
```

Two maximum banks are generated from fresh independent initial draws and the
frozen reference dynamics, labeled `selection_development_only`. Smaller banks
are exact prefixes with new uniform weights. Basis evaluation, K/f assembly,
audit, and fixed-coefficient gradient rows are streamed in chunks; no 29-GiB
K=280 full basis cache is constructed.

Seeds use SHA-256 of
`<global_seed>:skyrmion:galerkin_resolution:v1:<role>:<maximum-size>`.
Exact initial-row overlap is checked among fresh banks and every permitted
historical selection bank. Historical validation arrays remain unopened;
disjointness from them is by the independent versioned seed namespace and fresh
continuous initial-state construction.

Primary classification A requires both Law and historical 0.5% final energy
residuals at or below 0.08, valid algebra, final consecutive train-action change
at most 3%, gradient cosine at least 0.995, and gradient relative change at most
5%. A completed ladder that fails these rules is B; inability to complete a
meaningful ladder for genuine resource reasons is C.

Only classification B unlocks the conditional study. It uses the same largest
32,768/16,384 support, exact nested K prefixes `[120,160,200,240,280]`, and
rank tolerances `[1e-10,1e-11,1e-12]`. The 0.08 threshold never changes. A
future fixed discretization must pass all geometries and gates, remain stable
across neighboring K, and avoid material rank-tolerance sensitivity; a smaller,
better-conditioned basis is preferred when scientifically equivalent.
Neighboring-K tolerances are 5% for action, cosine at least 0.995, and 10% for
relative gradient change. Rank-tolerance robustness requires at most 2% action
spread, 0.01 absolute energy-residual spread, and gradient cosine at least
0.995 across the three cutoffs.

Candidate v2 initialization logic may admit a start that passes exact risk,
geometry, projection/ESS/forcing/covariance, algebra, rank, range, and
stationarity even if its held-out physical certificate initially fails. Every
official endpoint/incumbent still requires the complete unchanged certificate.
This capability is documented and tested only; it is not executed as an eta
optimizer.

The future start-generator diagnostic uses only selection risk: 17-point
Law-to-history interpolations, 16 local perturbations per center at scale 0.01,
16 risk-tangent directions at four fixed radii, and 32 global designs. It
reports feasibility and diversity at all six allowances but never evaluates or
optimizes Full action.
