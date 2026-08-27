# Frozen v2 Full-Pool Audit Across 0.5%–5% Risk Allowances

Date: 2026-08-25

Status: **development-only diagnostic**. This is not an official Pareto v3
continuation, protocol, selection, validation, or certification result.

## Question

Across the complete frozen Pareto-v2 candidate pool, at which Law-relative
risk allowances, if any, do candidates pass the unchanged relative-ESS gate on
both the frozen v2 screen bank and the independent frozen v2 periodic-audit
bank?

The diagnostic used the exact allowance set

```text
[0.5, 1, 2, 3, 4, 5] percent
```

with frozen Law selection risk

```text
R_Law_sel = 5.186549474478042
```

and exact risk feasibility

```text
risk <= (1 + p/100) * R_Law_sel
```

No numerical slack or rounding was added to this comparison.

## Frozen scientific contract

The run preserved the existing scientific settings:

- candidate pool: 337 unique frozen v2 geometries;
- screen bank: frozen v2 `N=8192` bank;
- independent audit bank: frozen v2 `N=16384` periodic-audit bank;
- dictionary size: `K=280`;
- dtype: `float64`;
- minimum acceptable relative ESS: `0.05` exactly;
- projection residual tolerance: `2e-6`;
- forcing-mean tolerance: `2e-7`;
- maximum covariance condition number: `1e10`;
- robust relative ESS: `min(screen min-rESS, audit min-rESS)`.

Complete dual-bank eligibility required the exact risk gate, valid geometry,
all existing screen support gates, all existing audit support gates, and
minimum rESS of at least `0.05` on both banks.

The run did not access validation data, run Tangent optimization, construct a
Full Galerkin K/f system, create an official v3 protocol, freeze a selection,
or create official v3 banks.

## Main result

| Allowance | Exact risk ceiling | Inside ceiling | Screen feasible | Audit projection valid | Audit rESS >= 0.05 | Complete dual-bank eligible | Best audit min-rESS | Best candidate |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 0.5% | 5.21248222185043 | 194 | 193 | 193 | 0 | 0 | 0.0483757148952091 | `candidate_054` |
| 1% | 5.23841496922282 | 217 | 216 | 216 | 1 | 1 | 0.0505263148807359 | `candidate_078` |
| 2% | 5.29028046396760 | 244 | 242 | 242 | 12 | 12 | 0.0570084405273966 | `candidate_074` |
| 3% | 5.34214595871238 | 276 | 274 | 274 | 35 | 35 | 0.0600649291652449 | `candidate_093` |
| 4% | 5.39401145345716 | 294 | 292 | 292 | 53 | 53 | 0.0614956563448191 | `candidate_168` |
| 5% | 5.44587694820194 | 301 | 299 | 299 | 59 | 59 | 0.0822143994982666 | `candidate_080` |

For the best candidate at every allowance, the audit value controlled the
reported robust rESS. Therefore the best robust-rESS column is numerically
identical to the best audit min-rESS column in this table.

The first tested allowance with any complete dual-bank eligible candidate is
**1%**. Its sole eligible candidate is `candidate_078`:

```text
eta = [0.8966819529790524, 0.2280129421284253,
       1.3113395845179603, 0.8547679615341017,
       0.7946947652028741, 0.5384978540519456,
       1.6087997948136492, 0.5933231030708730]

scientific selection risk = 5.221025656272905
screen minimum rESS       = 0.0671850281392830
audit minimum rESS        = 0.0505263148807359
robust rESS               = 0.0505263148807359
```

Because the allowance sets are nested, the eligible counts at and above the
first viable allowance are:

```text
1% ->  1
2% -> 12
3% -> 35
4% -> 53
5% -> 59
```

The complete candidate identifiers, canonical geometries, risks, bank
diagnostics, deterministic rankings, and symmetry-aware diversity shortlists
are retained in `summary.json` rather than duplicated here.

## Audit-rESS distributions

These summaries are over the screen-feasible candidates at each allowance.

| Allowance | Minimum | p05 | p25 | Median | p75 | p95 | Maximum |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5% | 0.03589578 | 0.04362662 | 0.04460955 | 0.04469650 | 0.04509578 | 0.04722837 | 0.04837571 |
| 1% | 0.03589578 | 0.04333165 | 0.04461665 | 0.04471731 | 0.04566842 | 0.04824307 | 0.05052631 |
| 2% | 0.03547847 | 0.04202189 | 0.04459938 | 0.04472143 | 0.04599050 | 0.04993967 | 0.05700844 |
| 3% | 0.03547847 | 0.04209187 | 0.04461595 | 0.04478729 | 0.04682158 | 0.05493697 | 0.06006493 |
| 4% | 0.03547847 | 0.04221741 | 0.04463177 | 0.04484009 | 0.04745496 | 0.05641980 | 0.06149566 |
| 5% | 0.03547847 | 0.04227671 | 0.04463360 | 0.04488153 | 0.04783898 | 0.05789985 | 0.08221440 |

## Relationship to the frozen 0.5% result

The existing Pareto-v3 Phase-1 development diagnostic remains unchanged. All
193 previously evaluated 0.5%-screen-feasible rows were reused by exact
candidate identity after verifying their frozen source artifacts and
semantics. Their audit values reproduce exactly, including:

```text
eligible count at 0.5% = 0
best audit min-rESS     = 0.0483757148952091
best candidate         = candidate_054
```

The remaining 144 candidates were evaluated on the same frozen v2
periodic-audit bank. Their batched audit evaluation took
`12.321093077000114` seconds on the resumed GPU run.

## Artifact provenance

Machine-readable outputs:

- `outputs/official_galerkin_pareto_v3/diagnostic_v2_audit_all_allowances/summary.json`
- `outputs/official_galerkin_pareto_v3/diagnostic_v2_audit_all_allowances/inventory.json`

Important seals:

```text
new summary SHA-256:
3b9c43f4486bfc07708182285e4a66fe9a0dc550e9f7184e0e66e92ac4ec4867

frozen v2 protocol SHA-256:
22a33ce47b2a3cc17ff063d100b878ac32c3ef6cc1a2b3e10a6eb8cd076488f1

frozen v2 output-tree SHA-256:
e69e58bb0cd02967315b83634551ff66773740c2524f5a110542d8e71f95b723

frozen Phase-1 summary SHA-256:
fd856bf004932e467a7abf87e5f158864899d1afe7b592eef2bc01bae35d3d33

frozen Phase-1 inventory SHA-256:
20f7cf4c6c9db8efea82b7e4f2c84b144f9d5959f2d69506681dcfa6b0323078
```

The diagnostic was run with:

```bash
.venv/bin/python -m experiments.skyrmions_deep_ritz_full.pareto_v3_run \
  --mode diagnose-v2-audit-all-allowances
```

## Verification

- 30/30 focused Pareto-v3 tests passed.
- 65/65 combined Pareto-v2 and Pareto-v3 tests passed.
- Python compilation checks passed.
- `git diff --check` passed.
- A second invocation verified and returned the sealed result as a cache hit.
- The candidate pool contained exactly 337 unique identifiers and 337 unique
  geometries.
- The reproduced screen-feasible counts were exactly
  `[193, 216, 242, 274, 292, 299]`.

## Scientific interpretation and limits

Within this fixed candidate pool and these two frozen banks, the independent
rESS failure observed at 0.5% does not persist through the full tested risk
sweep. One candidate passes the unchanged dual-bank gate at 1%, and additional
candidates appear at larger allowances.

This result supports considering a future, separately specified and frozen
partial-curve protocol at wider Law-relative risk allowances. It does not
itself constitute that protocol and does not certify any candidate against
fresh validation data.

The immutable v2 screen and Phase-1 audit artifacts retain the minimum rESS but
not its controlling physical time-node index. Those indices were therefore
reported as unavailable rather than reconstructed or invented.
