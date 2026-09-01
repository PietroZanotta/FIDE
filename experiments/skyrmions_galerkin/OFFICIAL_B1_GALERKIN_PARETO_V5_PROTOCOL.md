You are freezing and executing a NEW prospective single-root-seed
Skyrmion Galerkin V5 authoritative Pareto experiment.

V4 is a terminal failed authority and MUST remain unchanged.

The V5 design follows the completed DEVELOPMENT diagnosis of V4.

======================================================================
V5 PURPOSE
==========

V4 successfully repaired the earlier held-out energy-certificate problem.

All V4 held-out energy residuals passed the frozen 0.08 gate.

V4 subsequently failed for two independent finite-sample reasons:

1. scientific-risk generalization at the tight 0.5% allowance;

2. empirical Galerkin retained-range/stationarity residuals slightly above the
   frozen 1e-8 gate.

Development has identified both mechanisms.

V5 repairs ONLY these finite-sample issues.

======================================================================
FINAL V5 CHANGES
================

CHANGE 1 — ROBUST SCIENTIFIC-RISK FEASIBILITY

Every Tangent/Full candidate must satisfy its nominal allowance on:

```
the main selection risk bank
```

AND

```
four independent pre-seal risk guard banks.
```

Thus there are FIVE risk roles defining candidate feasibility.

For every role b:

```
Delta_R_b(eta)
  =
  R_b(eta) / R_b(eta_Law) - 1.
```

Candidate eta is feasible at allowance p iff:

```
Delta_R_b(eta) <= p/100
```

for EVERY:

```
b in {
    selection,
    risk_guard_1,
    risk_guard_2,
    risk_guard_3,
    risk_guard_4
}
```

plus all support/geometry/numerical requirements.

No averaging of guard risks.

No majority vote.

No post-hoc safety margin.

This is a direct max-guard constrained feasible set.

---

CHANGE 2 — GALERKIN EMPIRICAL PRECISION

Every AUTHORITATIVE / CERTIFICATE-BEARING K=280 fit/audit role uses:

```
N = 131,072
```

where the certificate requires empirical K/f transfer.

In particular, use 131,072 for:

```
authoritative Full train/fit statistics
authoritative Full independent audit statistics
```

and:

```
final held-out reference-fit
final held-out reference-audit.
```

Search/proxy stages may retain their lower prospectively frozen sample counts.

Do not use a lower authoritative N merely because a particular row appears to
pass.

======================================================================
UNCHANGED SCIENCE
=================

Keep exactly:

```
K = 280

JAX backend

float64

native Galerkin prohibited

rank threshold:
    lambda_i > 1e-12 * lambda_max

energy certificate formula

energy threshold:
    0.08

range residual threshold:
    1e-8

stationarity residual threshold:
    1e-8

weak certificate threshold

gauge certificate threshold

moment-rate threshold

sensor definition

scientific risk functional

many-body risk features

whitening construction

information projection

forcing

candidate geometry constraints

V4 support-robust Law mechanism

Law reanchor logic

allowances:
    0.5%
    1%
    2%.
```

Do NOT run:

```
3%
4%
5%.
```

======================================================================
WHY M=4 RISK GUARDS
===================

The completed DEVELOPMENT diagnosis found that at 0.5%:

```
selection slack
  =
  approximately 0.000673
```

while the measured independent-role SD was approximately:

```
0.01144.
```

Thus risk variability was approximately 17 times the available selection
slack.

The V4 selected 0.5% geometries therefore occupied a scientifically unstable
boundary region.

A two-guard rule is the minimum repair, but V5 is intended as a one-shot
authoritative run.

Use four pre-seal guards to define a meaningfully robust feasible set.

The fact that the historical V4 finalist panel had no surviving non-Law 0.5%
candidate under M=4 is NOT a failure of V5.

Those historical finalists were optimized without the V5 robust constraint.

V5 must search under the robust constraint from the start.

The Law geometry itself is always a feasible Full baseline because its
role-wise relative risk is exactly zero.

Therefore:

```
if no improved robust candidate exists at 0.5%,
select Law.
```

Do NOT weaken the guard to force a non-Law 0.5% winner.

======================================================================
V5 AUTHORITY IS NEW
===================

Do NOT:

* modify V4;
* relabel V4 as passing;
* reuse V4 heldout as V5 authority;
* reuse V5-development roles as final validation;
* test alternate V5 roots;
* regenerate final heldout after results;
* increase N after heldout begins;
* relax any threshold;
* change K;
* switch to native Galerkin.

Create a completely isolated V5 output root.

======================================================================
PHASE 0 — HISTORICAL INTEGRITY
==============================

Hash and verify:

```
old B1 authority
V2/V2.1
V3/V3.4
V4
V4 terminal diagnosis
```

before generating V5 data.

All must remain immutable.

======================================================================
PHASE 1 — ONE NEW ROOT
======================

Choose exactly ONE fresh root seed.

Use an outcome-blind deterministic unused-seed rule.

Freeze it before any new scientific outcome.

Every role is a deterministic JAX fold_in child.

No alternate roots.

Create:

```
V5_RANDOMNESS_PROVENANCE.md
```

======================================================================
PHASE 2 — FREEZE V5
===================

Before generating candidate/risk outcomes create:

```
OFFICIAL_B1_GALERKIN_PARETO_V5_PROTOCOL.md

protocol_v5.json

freeze_manifest_v5.json
```

Freeze:

```
root seed
all role IDs
candidate universe
optimization budgets
Law rule
support guards
risk guards M=4
guard sample counts
authoritative N=131072
final heldout N=131072
K=280
rank rule
thresholds
allowances
incumbent rule
output paths.
```

No scientific parameter may change after this freeze.

======================================================================
PHASE 3 — PERFORMANCE / JAX CALL GRAPH
======================================

STATIC REQUIREMENT:

```
native Galerkin is unreachable.
```

All scientific Galerkin work must use JAX.

Run a static/call-graph test before data generation.

Use:

```
JAX_ENABLE_X64=1
```

and GPU.

======================================================================
PERFORMANCE IS CRITICAL
=======================

V5 approximately doubles the largest authoritative empirical roles, so
performance engineering matters.

Use the optimized sufficient-statistic implementation.

Required principles:

```
compile once, run many

static array shapes where practical

jax.jit

jax.vmap

jax.lax.scan

batched time nodes

chunked Gram/load accumulation

fused basis-gradient/sufficient-statistic kernels

content-addressed caches

no full-bank gradient tensor materialization

minimum host/device transfers.
```

Do not use Python loops for particle-level operations.

Do not use Python candidate loops where safe JAX batching is practical.

======================================================================
RISK-GUARD PERFORMANCE
======================

Risk evaluation is substantially cheaper than Full K=280 action.

Exploit that.

For the frozen candidate pool:

evaluate exact scientific risk on all five risk roles BEFORE expensive Full
action ranking.

Vectorize candidates within each role.

Cache:

```
Law risk per role
```

and:

```
exact candidate risk per role.
```

Risk is evaluated ONCE per candidate/role and reused across:

```
0.5%
1%
2%.
```

Because allowances are nested:

```
robust_feasible(.5)
    subset
robust_feasible(1)
    subset
robust_feasible(2).
```

Do not recompute risk per allowance.

======================================================================
RISK ROLE SAMPLE COUNTS
=======================

For each of the four pre-seal risk guards use:

```
truth N = 5,000

reference / projection N = 65,536
```

as established in development.

Use fresh role IDs derived from the new root.

These roles are:

```
SELECTION inputs / guards,
```

not:

```
scientific replicates
or
heldout validation.
```

The final heldout roles remain completely inaccessible.

======================================================================
PHASE 4 — LAW
=============

Retain the successful V4 support-robust Law procedure.

Law must pass the unchanged four support guards.

Risk guards do not alter the definition of Law risk.

However, after Law is sealed, compute its risk independently on each V5 risk
role:

```
R_selection(Law)
R_guard1(Law)
...
R_guard4(Law)
```

These become the role-specific baselines for Tangent/Full constraints.

======================================================================
LAW REANCHOR
============

Retain the existing material-risk challenger logic.

A proposed reanchor must:

```
improve exact Law risk beyond the frozen consistency tolerance
```

AND

```
pass the complete support-robust Law rule.
```

If reanchoring occurs:

```
seal the new Law
```

then recompute:

```
Law risk on all five risk roles
```

and therefore all role-specific percentage ceilings.

Then rerun every:

```
Tangent
Full
```

allowance from scratch.

Preserve all previous branches.

======================================================================
PHASE 5 — COMPLETE ROBUST CANDIDATE SCREEN
==========================================

Persist EVERY candidate.

For every candidate compute:

```
geometry validity
support validity
selection risk
guard1 risk
guard2 risk
guard3 risk
guard4 risk.
```

For allowance p compute:

```
feasible_p(eta)
```

only after all five exact risk values are known.

Do not Full-rank unknown-feasibility candidates.

======================================================================
ROLE-WISE RISK FORMULA
======================

For role b define:

```
relrisk_b(eta)
  =
  R_b(eta) / R_b(Law) - 1.
```

Require:

```
relrisk_b(eta) <= p/100
```

on ALL five risk roles.

Do NOT compare guard candidate risk against selection-bank Law risk.

Pair each candidate and Law on the SAME role.

This preserves shared-bank cancellation and matches the heldout validation
quantity.

======================================================================
PHASE 6 — SEARCH UNDER THE ROBUST CONSTRAINT
============================================

Do not merely apply M=4 to V4-style final candidates.

The candidate search itself must operate from the robust feasible pool.

For every allowance, provide starts from:

```
Law
current Tangent robust-feasible candidates
broad robust-feasible pool
previous tighter Full incumbent
prospectively declared diversity candidates.
```

At 0.5%:

```
Law is mandatory.
```

At 1% and 2%:

```
previous tighter Full winner is mandatory.
```

Law may remain a fallback at all allowances.

======================================================================
LOCAL PROPOSALS
===============

For every generated local Full/Tangent proposal:

evaluate exact five-role scientific risk BEFORE expensive authoritative Full
action.

If it fails any guard:

```
reject as scientifically infeasible.
```

Do not spend K=280 action compute on clearly guard-infeasible proposals.

This is both scientifically correct and performance efficient.

======================================================================
DO NOT FORCE 0.5% IMPROVEMENT
=============================

If the robust 0.5% feasible set contains no candidate with Full action below
Law:

```
Full 0.5% = Law.
```

That is a legitimate Pareto result.

Do not:

```
weaken M
add risk slack
choose another root
add a handcrafted candidate.
```

The aim is a valid robust Pareto curve, not a mandatory nonzero improvement.

======================================================================
PHASE 7 — TANGENT
=================

Run Tangent for exactly:

```
0.5%
1%
2%.
```

Tangent candidates must satisfy the identical five-role scientific-risk
constraints.

Do not use Tangent feasibility as a proxy for Full feasibility.

======================================================================
PHASE 8 — FULL
==============

Run Full for exactly:

```
0.5%
1%
2%.
```

Use JAX K=280.

Only robust-feasible candidates may be action-ranked.

The Full sequence must be nested/nonincreasing:

```
A_Full(.5)
  >=
A_Full(1)
  >=
A_Full(2)
```

within the frozen replacement tolerance.

======================================================================
PHASE 9 — AUTHORITATIVE CERTIFICATION
=====================================

For every final unique geometry perform exact certificate-bearing Galerkin
evaluation with:

```
fit N = 131,072

independent audit N = 131,072.
```

This applies to every final:

```
Law
Tangent
Full
```

geometry requiring a Full certificate.

Use independent roles.

Every frozen certificate must pass.

======================================================================
ALGEBRA CERTIFICATES
====================

Do NOT add iterative refinement as a scientific repair.

Development established that the failed range residual is:

```
||(I-P_range(K)) f|| / ||f||
```

and stationarity agrees because:

```
Ka + f = (I-P)f
```

under the truncated pseudoinverse.

The failure was empirical finite-N range error, not retained-space solve
roundoff.

Therefore V5 repair is:

```
N=131072,
```

not:

```
solver refinement
threshold relaxation
rank change.
```

Keep:

```
max range residual <= 1e-8

max stationarity residual <= 1e-8.
```

======================================================================
ENERGY CERTIFICATE
==================

Keep the successful V4 repair and exact formula.

Threshold:

```
<= 0.08.
```

Do not modify it.

N=131072 provides at least as much empirical precision as the successful V4
65k energy setup.

======================================================================
PHASE 10 — SELECTION SEAL
=========================

Before final heldout exists:

write and hash:

```
selection_seal.json
```

containing:

```
Law
support receipts
all five Law risk baselines
allowances
all candidate five-role risks
robust-feasible sets
Tangent winners
Full winners
action receipts
authoritative 131k certificates
reanchor decision
incumbent lineage.
```

Then independently reconstruct:

```
Law
risk baselines
robust feasibility
winners
nesting
certificates.
```

Require PASS.

======================================================================
PHASE 11 — FINAL HELDOUT
========================

ONLY after selection seal + independent reconstruction:

generate exactly one fresh final heldout set.

Use:

```
heldout reference-fit N   = 131,072

heldout reference-audit N = 131,072.
```

Final risk roles must be fresh and disjoint from:

```
selection
support guards
four risk guards
action search
authoritative certificates
every historical/development role.
```

======================================================================
HELDOUT SCIENTIFIC RISK
=======================

For candidate eta calculate paired heldout relative risk:

```
R_holdout(eta) / R_holdout(Law) - 1.
```

Require:

```
<= 0.005
<= 0.01
<= 0.02
```

for the corresponding allowance.

Law is identically the zero-relative-risk baseline.

Do not reinterpret the allowance after observation.

======================================================================
HELDOUT FULL
============

For every unique selected geometry perform full JAX K=280 heldout fit/audit
certification at:

```
131072 / 131072.
```

Require all frozen:

```
forcing
weak
energy
gauge
moment
range
stationarity
```

gates.

No post-heldout modification.

======================================================================
IF 0.5% FULL = LAW
==================

This is valid.

Heldout relative risk is then exactly zero because it is the identical
geometry evaluated against itself.

Report the lack of an improved robust design honestly.

Do not consider this a failed optimization.

======================================================================
IF V5 FAILS
===========

V5 is terminal.

Complete the report.

Do not:

```
switch M=4 to M=2
add more guards
change N
switch root
rerun final heldout
relax algebra threshold
relax risk
alter K.
```

Future diagnosis would require V6.

======================================================================
PERFORMANCE OPTIMIZATION
========================

Keep a machine-readable performance receipt.

Major strategies:

1. RISK FIRST

   Screen all five risk roles before K280 action.

2. ROLE REUSE

   One candidate-risk computation per role reused for all allowances.

3. DEDUPLICATION

   If Tangent and Full share geometry, evaluate Full certificates once.

4. STATIC 131K CHUNKING

   Use the proven chunked sufficient-statistic path.

5. K/F CACHE

   Content-address by:

   ```
   geometry
   role
   N
   K
   basis
   reference
   forcing config.
   ```

6. PROCESS LIFETIME

   Keep compiled JAX kernels resident when memory permits.

7. MEMORY

   Maintain GPU headroom.
   Reduce chunk/batch sizes on OOM rather than scientific N/K.

8. NO NATIVE SOLVER

   Never use it as performance fallback.

======================================================================
PERFORMANCE TARGET REPORT
=========================

Create:

```
V5_PERFORMANCE_REPORT.md
```

Report:

```
bank generation wall time
risk guard wall time
robust candidate count per allowance
Full search wall time
131k authoritative certificate time
heldout generation time
heldout certificate time
JIT compile counts
peak GPU memory
cache hit rate
total wall time.
```

======================================================================
FINAL OUTPUTS
=============

Create:

```
OFFICIAL_B1_GALERKIN_PARETO_V5_PROTOCOL.md

protocol_v5.json

V5_RANDOMNESS_PROVENANCE.md

freeze_manifest_v5.json

complete_candidate_risk_matrix.csv

robust_feasible_sets.json

selection_seal.json

independent_selection_verification.json

heldout_validation/results.json

V5_PERFORMANCE_REPORT.md

OFFICIAL_B1_GALERKIN_PARETO_V5_FINAL_RESULT.md

final_summary.json

final_rows.csv

terminal_inventory.json.
```

======================================================================
FINAL DECISION TABLE
====================

A. Historical V1--V4 authorities unchanged?
PASS / FAIL

B. One fresh V5 root frozen before outcomes?
PASS / FAIL

C. Native Galerkin unreachable?
PASS / FAIL

D. JAX float64 K=280 everywhere?
PASS / FAIL

E. K/rank rules unchanged?
PASS / FAIL

F. Energy formula and 0.08 threshold unchanged?
PASS / FAIL

G. Range/stationarity 1e-8 thresholds unchanged?
PASS / FAIL

H. Exactly four pre-seal risk guards?
PASS / FAIL

I. Risk guard truth N exactly 5000?
PASS / FAIL

J. Risk guard reference N exactly 65536?
PASS / FAIL

K. All candidate feasibility used selection + all four guards?
PASS / FAIL

L. Role-specific Law risk baseline used on every guard?
PASS / FAIL

M. Law support-robust procedure passed?
PASS / FAIL

N. Law consistency/reanchor gate passed?
PASS / FAIL

O. Only 0.5%, 1%, 2% run?
PASS / FAIL

P. Law mandatory Full baseline at 0.5%?
PASS / FAIL

Q. Previous Full incumbent mandatory at 1%/2%?
PASS / FAIL

R. All authoritative certificate-bearing fit N = 131072?
PASS / FAIL

S. All authoritative certificate-bearing audit N = 131072?
PASS / FAIL

T. Range certificate <=1e-8 for every sealed row?
PASS / FAIL

U. Stationarity certificate <=1e-8 for every sealed row?
PASS / FAIL

V. Energy certificate <=0.08 for every sealed row?
PASS / FAIL

W. Selection sealed before heldout?
PASS / FAIL

X. Final heldout fit/audit exactly 131072/131072?
PASS / FAIL

Y. Final heldout nominal risk passed for every row?
PASS / FAIL

Z. Final heldout Full certificates passed every row?
PASS / FAIL

AA. No post-heldout tuning?
PASS / FAIL

AB. V5 SINGLE-SEED K280 ROBUST PARETO AUTHORITY:
PASS / FAIL

======================================================================
INTERPRETATION
==============

V5 is not a new Full method.

It retains exactly the K=280 JAX Galerkin population estimand.

It changes only empirical robustness:

```
more independent evidence before declaring a candidate risk-feasible
```

and:

```
more empirical samples for certificate-bearing Galerkin statistics.
```

The intended constrained optimization is now:

```
establish robust scientific adequacy across five independent pre-seal
risk roles
```

then:

```
minimize K=280 Full action inside that robust feasible set.
```

The final heldout remains a single untouched test.
