# Official fixed-feature Galerkin Pareto protocol

## Frozen methodological decision

The fixed-feature K=280 Galerkin finite-dimensional approximation is the
official skyrmion Full discretization for this sweep. The nonlinear Deep Ritz
solver is retired from continuous optimization, candidate ranking,
certification, and validation. Historical Deep Ritz artifacts remain read only.

This protocol covers the ordered nominal selection allowances
`[0.5, 1, 2, 3, 4, 5]%`. It is written before the first new optimization. Its
complete machine-readable representation and SHA-256 seal are
`outputs/official_galerkin_pareto/protocol.json` and `protocol_hash.txt`.

## Risk rules

Selection uses the exact scientific risk and the allowance-specific constraint

```text
R_sel(eta) <= (1 + p/100) R_Law,sel.
```

Fresh validation uses the already predeclared five-percentage-point
neighborhood

```text
R_val(eta) <= (1 + p/100 + 0.05) R_Law,val.
```

The strict `p%` validation comparison is also reported, but it is transparency
only and does not replace the declared `p+5pp` rule. In particular, the 3%
optimizer receives a 3% selection budget, never an 8% selection budget.

## Full discretization and hard gates

The solver uses the existing K=280 hybrid Fourier/pairwise dictionary, its exact
ordering and selection-train normalization, relative rank tolerance `1e-12`,
and all unchanged production Galerkin algebra, forcing, geometry, and held-out
certificate thresholds. The validated dictionary SHA-256 is
`37e9b60fcb92c4e5a0ee7ec1651fb7f8889f7ac6bdb02d3bd314e9ef40833326`.
K is not retuned. This remains a finite-dimensional Galerkin approximation;
absolute infinite-dimensional convergence is not claimed.

## Optimizer and replacement rule

Every start uses the validated periodic projected-gradient bounded trajectory:

- trust radius `2e-4`;
- initial step `5e-5`, halved on backtracking;
- at most eight accepted-step attempts and ten backtracks per attempt;
- successful next-step cap `7.5e-5`;
- complete selection certification every fourth accepted step and at endpoints;
- exact risk and exact minimum-separation checks at every proposal;
- unchanged rank, forcing, covariance, algebra, and certificate gates;
- replacement only for an authoritative action decrease greater than `1e-10`.

Risk and smooth-separation penalty weights are frozen at `100` as
globalization diagnostics. Exact hard gates remain authoritative. The preceding
winner is both a mandatory start and the retained incumbent at every larger
allowance, ensuring a nonincreasing reported selection action up to the fixed
replacement tolerance.

## Deterministic multistart rule

Before optimization, one manifest freezes a selection-only start algorithm.
It combines the mandatory preceding incumbent, Law, every exact-feasible
historical Pareto geometry without consulting its old validation result, two
deterministic local perturbations per historical center at scale `2e-4`, and
the two best exact-feasible designs from a fixed pool of 48 deterministic global
geometries ranked by K=280 selection action. At most eight deduplicated starts
are used per allowance. All seeds are SHA-256-derived from the global seed,
fixed labels, and protocol version `v1`.

## Finalist derivative audit and selection seal

After all six selection winners are chosen, each unique geometry receives one
deterministic centered-FD audit at epsilon `3e-4` and `1e-4`. Rank stability,
forcing/projection validity, sign agreement, and relative error at most 2% are
required. Failure stops before validation and does not substitute a winner.

Only then are `pareto_selection.json` and `manifest.json` written with
`selection_frozen=true` and `validation_accessed=false`, and their hashes are
sealed. No winning geometry can change afterward.

## Fresh validation construction

Fresh seeds for truth, reference-fit, reference-audit, and detector noise are
derived as

```text
SHA256("<global_seed>:skyrmion:official_galerkin_pareto:v1:<label>")
```

and converted deterministically to positive signed-32-bit integers. Their
labels, hashes, and integers are frozen in `protocol.json` before selection.
No fresh validation array is generated or opened before selection freezes.

After the selection seal, the workflow generates 5,000 independent truth
trajectories and two independent 16,384-configuration reference rollout banks
from the frozen reference dynamics, using 24 truth and 14 reference substeps.
Reference initial states are new independent draws from the same frozen truth
initial distribution. Exact initial-row overlap checks against every selection
and previously opened validation bank must be empty. Artifact hashes are sealed.

All six geometries and Law are then evaluated once without optimization using
the same K=280 solver. The production weighted empirical audit-sample standard
error is reported; no pseudo-replicates or post-hoc significance gate is used.

## Isolation and failure policy

All new code, records, caches, and reports are confined to
`experiments/skyrmions_deep_ritz_full/`; numerical outputs are confined to
`outputs/official_galerkin_pareto/`. Original experiments, `src/`, `native/`,
historical outputs, the sealed prior 3% result, and paper artifacts are read
only. Old validation is development data and cannot enter the new selection
path.

Every phase is resumable only when prerequisite hashes match. Reproduction,
dictionary, risk, geometry, rank, derivative, certification, isolation,
disjointness, or immutability failures stop closed; thresholds are never
loosened. The final classification is made separately at every allowance as
`PASS`, `VALIDATION RISK REVERSAL`, or `VALIDATION NUMERICAL FAILURE`.
