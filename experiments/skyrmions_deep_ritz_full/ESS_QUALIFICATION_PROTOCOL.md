# ESS qualification and performance protocol

## Scope

This is a selection-development-only diagnostic of the unchanged relative-ESS
gate and of the cost of the fixed K=280 Galerkin workflow. It performs no eta
optimization, no Pareto sweep, and no validation access. The nonlinear Deep
Ritz solver is excluded. All records are write-once beneath
`outputs/ess_qualification/`.

## Frozen scientific constants

- Galerkin size: K=280.
- Dictionary SHA-256:
  `37e9b60fcb92c4e5a0ee7ec1651fb7f8889f7ac6bdb02d3bd314e9ef40833326`.
- Relative rank tolerance: `1e-12`.
- Minimum relative ESS: `0.05`.
- Maximum Ritz-energy residual: `0.08`.
- Allowances: `0.5, 1, 2, 3, 4, 5` percent.

## Independent-bank ladder and anchor decision

The main ladder is fixed before evaluation:

| N | independent replicates |
|---:|---:|
| 8192 | 4 |
| 16384 | 4 |
| 32768 | 3 |

Replicate seeds are derived from
`"<global_seed>:skyrmion:ess_qualification:v1:N<N>:rep<r>"` by SHA-256.
Reference dynamics, time grid, initial law, and reconstruction data are frozen;
the reference network is never retrained. Replicate 0 at each N is retained as
the deterministic staged-screening bank; the other banks are independent anchor
replicates. A 95% Student-t interval over independent replicate minima is used.
An anchor is CLEARLY ABOVE 0.05 if the lower interval endpoint is above 0.05,
CLEARLY BELOW if its upper endpoint is below 0.05, BORDERLINE if the interval
crosses 0.05 and the N=32768 mean is within 0.005 of it, and UNRESOLVED
otherwise. Fits in `1/N` and `1/sqrt(N)` are descriptive only.

## Candidate construction and staged scoring

The deterministic 337-design future-v2 pool construction from the frozen
resolution-study protocol is reused exactly: fixed anchors, 17-point Law-to-
anchor interpolants, 16 local perturbations per center at scale 0.01, 16
risk-tangent directions at four fixed radii and both signs, and 32 deterministic
global designs. Exact selection risk is evaluated on the frozen selection
projection bank. No Full action is evaluated in bulk.

Stage A scores all candidates at N=8192. Stage B deduplicates the union, across
allowances, of risk-feasible and projection-valid candidates with rESS at least
0.04 or in the top 32 rESS values for that allowance. Stage C scores only Stage
B candidates remaining at or above 0.045 at N=16384. The authoritative study
gate remains 0.05. A positive existence witness at N=32768 yields YES; absence
after staged screening is UNRESOLVED rather than an unjustified NO.

## Performance audit

The audit measures scalar and candidate-batched observable/reconstruction
preprocessing, information projection, forcing, cached K/f assembly, the
rank-aware solve, fixed-coefficient value/gradient, and held-out audit where
available. Only experiment-local, float64, semantics-preserving changes with
explicit equivalence checks may be adopted. Promising invasive changes are
reported rather than implemented.

## Prohibitions

No old or fresh validation quantity may be read; no eta Full optimization,
Pareto sweep, winner change, threshold change, K/rank/dictionary change,
historical-output overwrite, shared-code modification, or Deep Ritz invocation
is permitted.
