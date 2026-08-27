# Frozen K=280 empirical-quadrature extension protocol

## Scope

This is a selection-development-only qualification of one fixed finite
Galerkin problem. It cannot optimize eta, run a Pareto sweep, load validation,
change K, change the dictionary or normalization, change the rank cutoff, or
change any physical/numerical threshold. The previous official Pareto v1 and
the complete prior resolution study are immutable historical records.

The old 32,768/16,384 `complete_certificate` flags are audited before this
protocol is frozen. That audit decomposes every forcing, geometry, algebra,
rank, weak, energy, gauge, and moment gate and separately records the prior
support-convergence decision.

## Fixed scientific problem

- `K = 280`.
- Dictionary SHA-256:
  `37e9b60fcb92c4e5a0ee7ec1651fb7f8889f7ac6bdb02d3bd314e9ef40833326`.
- Dictionary ordering and eta-independent normalization are unchanged.
- Relative rank tolerance is exactly `1e-12`.
- The Ritz-energy threshold is exactly `0.08`.
- All projection, ESS, forcing, covariance, algebra, weak, gauge, and
  moment-rate thresholds come unchanged from `config.json`.
- The exact six geometries are copied from the immutable prior protocol:
  Law, historical 0.5%, historical 1%, historical 2%, eta0 at 3%, and the
  frozen continuously refined 3% eta.

## Frozen nested support ladder

The mandatory levels are, in order:

```text
32768 / 16384
32768 / 32768
65536 / 32768
65536 / 65536
```

The optional `131072 / 65536` level is predeclared now and runs only if the
mandatory qualification is not fully resolved. One deterministic maximum
train bank of 131,072 samples and one maximum audit bank of 65,536 samples are
generated so all smaller levels are exact prefixes.

Seeds are SHA-256-derived from the frozen global seed and the labels:

```text
<global_seed>:skyrmion:k280_quad_extension:v1:train:max
<global_seed>:skyrmion:k280_quad_extension:v1:audit:max
```

New train and audit initial states must be mutually disjoint and disjoint from
every permitted historical selection/development bank. Validation files are
not opened; their disjointness relies on the independent namespace and fresh
continuous draws.

## Streaming and diagnostics

Basis values, state gradients, Gram/load assembly, audit evaluation, and the
fixed-coefficient eta derivative use the existing chunked implementation. A
per-sample `K x K` Gram tensor is prohibited. Each row records complete core,
gradient, projection/forcing, eigen/rank/algebra, held-out physical, and
individual-gate diagnostics.

All consecutive nested-support comparisons record train/audit action changes,
gradient cosine and relative change, per-coordinate gradient differences, and
weak/energy/moment changes. The first audit-only, train-only, and second
audit-only steps are interpreted separately. An energy effect below `0.005` is
called negligible; otherwise one source is called dominant only when its
absolute effect exceeds the other by a predeclared factor of `1.5`.

## Qualification gates

At the final two relevant fitted supports, all six geometries must pass every
unchanged physical and numerical gate. Train and audit action relative changes
must each be at most `0.02`; `0.01` is preferred but not required. Gradient
cosine must be at least `0.995` and relative gradient change at most `0.05`.

For the mandatory ladder, train convergence is measured by
`32768/32768 -> 65536/32768`; audit convergence by
`65536/32768 -> 65536/65536`. Both train and audit action changes are checked
on both comparisons. If the optional level runs, the final comparison is
`65536/65536 -> 131072/65536`.

If magnitude remains above 5% while cosine is at least `0.999`, all material
component signs are stable, and both action changes are at most 1%, the result
is explicitly `DIRECTION STABLE, GRADIENT SCALE NOT FULLY CONVERGED`; it is not
accepted as converged.

## Conditional finite difference

Only if final physical validity and basic action stability pass, four fixed
geometries (Law, historical 0.5%, eta0 3%, eta-grad 3%) receive two
SHA-256-seeded normalized directions each. Centered finite differences use
`epsilon = 1e-3, 3e-4, 1e-4`. Every perturbation independently recomputes
projection, forcing, K/f, coefficients, action, ranks, and certificates. Every
perturbation must preserve rank and physical/numerical validity; signs must
agree, and every direction needs at least one relative error at most 1%
(`0.5%` preferred).

## Final decisions

- A requires physical, action, gradient, and unlocked AD/FD gates to pass.
- B requires physical/action/direction validity and passing AD/FD, but retains
  a failed 5% gradient-scale gate.
- C records action instability, physical failure, or derivative-audit failure.
- D is reserved for a genuine computational blocker.

No Pareto, Tangent, Full, or Law optimization is part of this protocol.
