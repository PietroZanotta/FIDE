# Skyrmion Pareto v2/v3: detailed engineering and scientific worklog

## Purpose of this document

This document is a detailed narrative of the recent skyrmion FIDE/MFSI work.
It explains what problem we are trying to solve, which numerical method is now
official, what native/Tesseract acceleration was added, how the official Pareto
v2 protocol was designed and executed, why it failed, what Pareto v3 was meant
to repair, what the v3 preflight diagnostic found, and what work is and is not
scientifically permissible next.

This is a worklog and orientation document. The immutable protocol/evaluation
documents and machine-readable JSON artifacts remain authoritative whenever a
hash or result is quoted. The most important primary records are:

- `OFFICIAL_GALERKIN_PARETO_V2_PROTOCOL.md`;
- `OFFICIAL_GALERKIN_PARETO_V2_EVALUATION.md`;
- `OFFICIAL_GALERKIN_PARETO_V3_EVALUATION.md`;
- `outputs/official_galerkin_pareto_v2/protocol.json`;
- `outputs/official_galerkin_pareto_v2/failure.json`;
- `outputs/official_galerkin_pareto_v3/diagnostic_v2_audit_map/summary.json`.

## Executive summary

The project seeks sensor geometries that reduce a dynamical-information action
while respecting an exact scientific-risk allowance. Three designs are meant
to be compared at each allowance:

1. **Law**, the existing frozen risk-defined sensor geometry;
2. **Tangent**, optimized using a finite observable-space tangent action;
3. **Full**, optimized using a fixed-feature, finite-dimensional K=280
   Galerkin approximation of the full weak problem.

The nonlinear empirical Deep Ritz route has been retired from official Full
optimization and certification. The Full method is now the fixed-feature K=280
Galerkin discretization, with a rank-aware coefficient solve and a
fixed-coefficient envelope derivative with respect to sensor geometry.

Two independent native performance components were added:

- `candidate_iprojection_tesseract`, which batches candidate-specific
  information-projection trajectories and is materially faster than making one
  native call per candidate;
- `galerkin_tesseract`, which computes additive Galerkin K/f statistics in
  C++/OpenMP/OpenBLAS but is slower than device-resident JAX on the current GPU
  once host/device transfers are included.

The original `native/iprojection_tesseract/` was restored exactly to
`origin/main`. Candidate batching lives in its own native directory because the
original component is shared by other experiments and was explicitly required
to remain unchanged.

Pareto v2 created fresh, disjoint selection banks and successfully screened a
337-candidate feasible-manifold pool. At 0.5%, 193 candidates passed screen-side
risk, geometry, projection, and rESS. V2 nevertheless promoted only four
starts. All four passed search-side checks but failed the independent N=16384
audit rESS gate. V2 therefore stopped before Full, before selection freeze, and
before validation.

Pareto v3 was proposed to repair that start-selection weakness prospectively:
before choosing a small start set, evaluate all 193 screen-feasible 0.5%
candidates on the independent audit bank and require both screen and audit
`rESS >= 0.05`. The required cheap diagnostic found that all 193 passed audit
projection, but **zero** passed audit rESS. The best audit rESS was
`0.0483757148952091`.

The v3 handoff predeclared that a zero count is classification C and requires a
stop before freezing a v3 protocol. That stop was honored. No official v3
selection banks, Tangent/Full branches, selection freeze, or validation arrays
exist.

## 1. Scientific problem

### 1.1 Microscopic state and sensor design

The skyrmion system contains 16 microscopic particles in two spatial
dimensions, so the microscopic state is 32-dimensional. A design contains four
two-dimensional periodic sensor centers:

```text
eta = [s1x, s1y, s2x, s2y, s3x, s3y, s4x, s4y] in R^8.
```

Each sensor measures a permutation-invariant local-density observable. Sensor
centers live in a periodic box and must satisfy the unchanged minimum-separation
constraint.

For a candidate geometry, the computational chain is conceptually:

```text
eta
  -> local-density features Phi_eta
  -> reconstructed observable trajectory c_eta(t), cdot_eta(t)
  -> empirical information projection
  -> projected reference weights Q_eta
  -> lambda_dot_eta
  -> continuity forcing h_eta
  -> Tangent or Full action and certificates.
```

Reference dynamics, reference trajectories, reference velocities, truth common
random numbers, detector-noise draws, and reference-network parameters are
frozen with respect to `eta`.

### 1.2 Scientific risk

Scientific risk is not redefined by the optimization method. For selection at
allowance `p`, feasibility is exactly:

```text
R_sel(eta) <= (1 + p/100) R_Law,sel.
```

The optimizer never receives validation slack. The validation convention, if
and only if selection has already been frozen, is:

```text
R_val(eta) <= (1 + p/100 + 0.05) R_Law,val.
```

Strict nominal-`p` validation status and the actual Law-relative validation-risk
increase are also intended to be reported. Validation cannot alter a selected
geometry.

For v2, the frozen Law risk was:

```text
R_Law,sel = 5.186549474478042.
```

The 0.5% selection ceiling was therefore:

```text
5.186549474478042 * 1.005 = 5.212482221850432.
```

### 1.3 Law geometry

Law is selected solely by the existing scientific-risk construction. It is not
reoptimized for Tangent or Full action. Its frozen coordinates are:

```text
[
  0.890286510596537,   0.227289528868506,
  1.310368832144490,   0.859163192162967,
  0.797588822714243,   0.535723001316333,
  1.610343150447571,   0.583219225445585
]
```

## 2. Information projection and rESS

### 2.1 Empirical information projection

The information projection finds exponential-tilt multipliers that match the
candidate observable targets under a fixed reference bank. Schematically:

```text
q_i(lambda) proportional to b_i exp(lambda . phi_i),
E_q[phi] = target.
```

The implementation uses Newton iterations with physical-time warm starts.
Derivatives are obtained from the implicit moment-covariance system. We do not
differentiate through Newton iteration history.

### 2.2 Relative ESS definition

For normalized projected weights `q_i` and normalized base weights `b_i`:

```text
ESS_projected = 1 / sum_i q_i^2,
ESS_base      = 1 / sum_i b_i^2,
rESS          = ESS_projected / ESS_base.
```

For uniform base weights, `ESS_base = N`, so `rESS = ESS/N`. The controlling
quantity is the minimum over the 13 physical time nodes. The threshold remains:

```text
minimum rESS >= 0.05.
```

This gate was never lowered or rounded. A larger sample count does not imply
that one finite-bank rESS realization must be numerically larger than another;
rESS is a distributional overlap quantity, not absolute ESS. Independent banks
can yield different finite-sample minima, especially when the controlling time
node changes.

## 3. Tangent and Full objectives

### 3.1 Tangent objective

The Tangent approximation uses the exact validated observable-space formula:

```text
r = E_Q[J Phi_eta u] - cdot_eta,
G = E_Q[J Phi_eta J Phi_eta^T],
A_Tangent = r^T G^dagger r.
```

It is computationally much cheaper than Full. It remains a distinct objective;
its raw numerical action must not be compared directly with a Full action as
though the two were the same physical quantity.

### 3.2 Fixed-feature Full Galerkin method

Deep Ritz is excluded from official continuous Full decisions. The official
Full discretization uses 280 frozen basis functions:

```text
psi_a(X) = sum_k a_k phi_k(X).
```

At each physical time:

```text
K_t[j,k] = E_Q[grad phi_j . grad phi_k],
f_t[j]   = E_Q[h_eta (phi_j - E_Q[phi_j])].
```

The coefficient system is solved by a rank-aware pseudoinverse/eigendecomposition:

```text
K_t a_t = -f_t.
```

The finite-dimensional action is:

```text
A_Full(eta) = sum_t omega_t a_t^T K_t a_t.
```

At the stationary finite-dimensional solution, the action identity is:

```text
A_Full = -2 J_G.
```

The eta derivative uses the fixed-coefficient envelope rule:

```text
grad_eta A_Full = -2 partial_eta J_G(a_fixed, eta).
```

The implementation never differentiates through the coefficient
eigendecomposition or pseudoinverse.

The scientific wording must remain “fixed-feature finite-dimensional K=280
Galerkin Full approximation.” No completed study establishes that K=280 is the
converged infinite-dimensional Full solution.

### 3.3 Frozen numerical constants

The relevant constants remained unchanged throughout:

| quantity | frozen value |
| --- | --- |
| basis size | `K = 280` |
| dictionary SHA-256 | `37e9b60f…833326` |
| relative rank tolerance | `1e-12` |
| minimum rESS | `0.05` |
| maximum held-out energy residual | `0.08` |
| allowances | `[0.5, 1, 2, 3, 4, 5]%` |

All projection, forcing, covariance, geometry, rank, range, stationarity,
symmetry, weak, gauge, moment-rate, and replacement thresholds were also left
unchanged.

## 4. Native and Tesseract engineering

### 4.1 Why two information-projection Tesseracts exist

There are now two deliberately separate I-projection native components.

`native/iprojection_tesseract/` is the original shared component from GitHub.
It handles ordinary trajectories with a shared feature tensor, conceptually
`phi[T,N,M]`, and includes existing hard/soft projection endpoints used by
other work.

`native/candidate_iprojection_tesseract/` is the new candidate-specific
accelerator. It accepts:

```text
phi              [C,T,N,M]
targets          [C,T,M]
log_base_weights [T,N].
```

Every candidate has its own feature and target trajectory. Candidates are
parallelized with OpenMP, while each candidate retains its own multiplier warm
start along physical time.

The two directories were kept separate because the user explicitly required
the original I-projection component to be restored. The final verification:

```text
git diff --exit-code origin/main -- native/iprojection_tesseract
```

returns success with no diff.

The candidate component builds a distinct extension module named:

```text
_candidate_iprojection_native
```

rather than shadowing `_iprojection_native`.

### 4.2 Python routing

`src/mfsi/projection_tesseract.py` retains the original native root and module
loader for ordinary trajectories and adds a separate candidate root, module
loader, availability check, direct forward path, and differentiable Tesseract
client.

`src/mfsi/projection.py` adds
`EmpiricalIProjector.project_candidate_trajectories`, which accepts the
candidate-specific tensor layout. The JAX fallback uses candidate `vmap`; the
`tesseract_cpp` route uses the isolated native component.

### 4.3 Candidate-projection performance

On actual skyrmion shapes around `[C=8,T=13,N=8192,M=4]`, candidate batching
reduced multiple scalar native calls by roughly 6.4–7.2x in the recorded
benchmarks. Results were numerically equivalent:

- maximum multiplier discrepancy: `0`;
- residuals around `1e-11` or smaller;
- deterministic repeatability;
- forward, implicit VJP, and JVP tests passed.

This is a meaningful high-value/low-risk optimization because Stage-A and
audit-aware screening evaluate hundreds of candidates but construct no
Galerkin system.

### 4.4 Independent Galerkin Tesseract

`native/galerkin_tesseract/` is another separate optional component. It computes
additive raw statistics for one Galerkin chunk:

```text
gram,
raw_load,
basis_mean,
forcing_sum.
```

Global centering is applied after chunk accumulation:

```text
load = raw_load - forcing_sum * basis_mean.
```

The implementation uses C++17, OpenMP, float64, and OpenBLAS. It deliberately
does not use `-ffast-math`.

Numerical equivalence was excellent, with discrepancies around `1e-13` to
`1e-12`. Performance on the RTX 5090 Laptop GPU system was not favorable,
however. Transfer-inclusive native/Tesseract throughput was only about
`0.137x` to `0.280x` the speed of direct JAX, or approximately 3.6–7.3 times
slower.

### 4.5 CPU/GPU communication limitation

The installed Tesseract-JAX path lowers through a Python callback. Inputs are
materialized on the host and outputs are copied back to the device. That means
the current Galerkin Tesseract cannot avoid CPU/GPU communication.

True device-resident native assembly would require a CUDA-aware JAX FFI/custom
call that accepts device buffers and launches CUDA kernels. That is a separate
engineering architecture, not a flag on the present callback implementation.
For this workload, JAX/XLA already provides the device-resident path and is
faster.

### 4.6 Galerkin assembly backend flag

The experiment configuration now contains:

```json
"production_galerkin": {
  "assembly_backend": "jax"
}
```

Accepted values are:

- `jax`: direct device-resident JAX/XLA assembly;
- `tesseract_cpp`: the independent host Tesseract assembly path.

The official v2 seal selected `jax`. Each Full evaluation records its selected
assembly backend. An invalid backend fails closed.

Changing this flag in the shared config after a frozen protocol would make its
code/config seal incompatible. Any development comparison should therefore use
a copied development config rather than mutate an official frozen run.

### 4.7 CUDA installation question

The candidate I-projection accelerator is CPU/OpenMP and does not require the
CUDA toolkit. The Galerkin Tesseract is also CPU/OpenMP/OpenBLAS. JAX GPU use
requires a functioning NVIDIA driver and compatible installed JAX CUDA runtime,
but local `nvcc` is not needed for either of these Tesseract extensions.

## 5. Development evidence before Pareto v2

The preceding studies established several important facts:

- Deep Ritz should not be used as the official Full oracle.
- The fixed K=280 action ordering and eta-gradient direction were stable enough
  to motivate a finite-dimensional multi-fidelity search.
- The 3% refined geometry consistently beat its unrefined eta0 counterpart on
  multiple K values and independent data views.
- Some low-risk anchors had borderline or failing finite-bank rESS, while many
  candidates elsewhere in the feasible manifold were witnesses above 0.05 on
  independent N=32768 development banks.
- K/f assembly and held-out Full auditing dominate Full runtime.
- It would be wasteful to use N=65536 at every optimizer step.

The ESS qualification's staged feasible-manifold study reported N=32768
risk+rESS witnesses at all allowances, including 30 at 0.5%. Those were
development-bank observations, not official v2 selection results.

This evidence supported a new official multi-fidelity architecture, but it did
not guarantee that any particular future independent audit bank would contain
the same witnesses.

## 6. Official Pareto v2 design

### 6.1 Intended workflow

V2 froze the following architecture before official computation:

```text
N=8192 screen
  -> N=32768 continuous search
  -> N=16384 periodic independent audit
  -> shortlist at most three Full finalists
  -> N=65536 train + N=65536 audit certification
  -> freeze every method/allowance winner
  -> generate fresh validation once.
```

The intended validation sizes were truth N=5000, reference fit N=16384, and
reference audit N=16384. Their seeds were sealed prospectively, but arrays were
not allowed to exist before the complete selection hash.

### 6.2 V2 protocol seals and preserved attempts

Several pre-selection attempts were preserved rather than overwritten:

1. a pre-start attempt terminated because a periodic-distance helper omitted
   its box argument;
2. a later seal was paused before selection to isolate and benchmark Galerkin
   Tesseract;
3. another pre-bank seal was superseded while candidate projection was moved to
   its own native directory and the Galerkin backend flag was added.

The final active v2 protocol version was:

```text
skyrmion_official_galerkin_pareto_v2_rerun3
```

with inner seal:

```text
22a33ce47b2a3cc17ff063d100b878ac32c3ef6cc1a2b3e10a6eb8cd076488f1.
```

The protocol and evaluation document hashes are:

| artifact | SHA-256 |
| --- | --- |
| v2 protocol Markdown | `f7353f82…c2f0e6` |
| v2 evaluation Markdown | `00965dbf…1487df` |
| v2 protocol JSON | `8360afb8…8dbf3b` |
| v2 failure JSON | `e73d4e3…9acf40` |

### 6.3 Official v2 selection banks

V2 generated five new deterministic, role-disjoint selection banks:

| role | N | seed | artifact SHA-256 | generation time |
| --- | ---: | ---: | --- | ---: |
| screen | 8,192 | 462821422 | `05ec181f…34ea0` | 12.124 s |
| search train | 32,768 | 1269066990 | `cc8be9fc…3f04` | 38.608 s |
| periodic audit | 16,384 | 1839249659 | `e0fcbb16…42915` | 19.778 s |
| authoritative train | 65,536 | 1331973801 | `59987834…0effa6` | 70.227 s |
| authoritative audit | 65,536 | 1100690834 | `3dd2f3a8…e2b294` | 69.308 s |

The bank manifest SHA-256 is:

```text
cc81d24b01ebfcf25eaa91604ba75dc40856299697af1d47e0738667029dc6d3.
```

Total recorded generation time was approximately 210 seconds. Reference
dynamics were frozen and were not retrained.

### 6.4 V2 candidate pool and screening

The deterministic pool contained 337 unique geometries drawn from:

- Law-to-history interpolations;
- deterministic local clouds around fixed centers;
- risk-tangent perturbations;
- a small global component.

Stage A used candidate-batched native I-projection on N=8192 and explicitly
constructed zero Full K/f systems.

Screen-feasible counts under exact risk, geometry, projection, and rESS were:

| allowance | candidates |
| ---: | ---: |
| 0.5% | 193 |
| 1% | 216 |
| 2% | 242 |
| 3% | 274 |
| 4% | 292 |
| 5% | 299 |

The candidate-pool artifact SHA-256 is:

```text
f3ca87f8c251c9bb5aec13e5bc77885cc643a98a9dd4554f4a86c800879d18ab.
```

### 6.5 V2 start rule

V2 selected four deterministic starts per allowance, prioritizing Law,
historically relevant candidates, high screen-side rESS, and max-min geometric
diversity. A previous certified winner would become the mandatory incumbent at
the next allowance.

At 0.5%, the four promoted starts were:

| role | risk | screen/search rESS used at relevant stages |
| --- | ---: | ---: |
| Law | 5.186549474 | about 0.0624 screen / 0.0576 search |
| historical strong | 5.203174625 | about 0.0645 / 0.0594 |
| screen-best rESS | 5.207900752 | about 0.0672 / 0.0615 |
| max-min diverse | 5.196816357 | about 0.0540 / 0.0537 |

All four were valid on the search context.

## 7. Why Pareto v2 failed

Before taking any Tangent optimizer step, v2 audited each start on the
independent N=16384 periodic bank. The results were:

| role | audit rESS | threshold | result |
| --- | ---: | ---: | --- |
| Law | 0.044666261 | 0.05 | fail |
| historical strong | 0.047229777 | 0.05 | fail |
| screen-best rESS | 0.048375715 | 0.05 | fail |
| max-min diverse | 0.042700268 | 0.05 | fail |

Their audit projection residuals, Gram conditions, and moment-rate residuals
passed. The independent rESS gate was the sole controlling failure.

Because no start was independently eligible, there was no Tangent finalist or
incumbent. The run raised:

```text
RuntimeError: no certified Tangent winner at 0.5%
```

V2 correctly failed closed. It did not:

- lower the rESS gate;
- add post-hoc starts;
- skip 0.5%;
- start Full after the whole-workflow failure;
- freeze a partial selection;
- generate or open validation;
- substitute a development or historical winner.

V2's formal classification is **NO CERTIFIED SELECTION WINNER** at the 0.5%
Tangent stage. Larger Tangent allowances and all Full allowances were not run.

## 8. Pareto v3 proposal

### 8.1 What v3 was meant to repair

The initial v3 hypothesis was that v2 may have chosen the wrong four starts
because it ranked candidates using a noisy N=8192 screen-side rESS estimate.
The proposed repair was not to lower a threshold. It was to test start
eligibility on both independent selection banks before promoting candidates.

For a candidate to become a v3 start, the intended dual-bank rule was:

```text
exact risk passes,
geometry passes,
screen projection passes,
screen rESS >= 0.05,
audit projection passes,
audit rESS >= 0.05.
```

The robust ranking statistic was frozen conceptually as:

```text
robust_rESS = min(screen_rESS, audit_rESS).
```

If robust starts existed, Tangent could use up to 10 and Full up to 6 per
allowance. Tangent and Full were to be independent branches so one branch's
failure would not erase the other.

### 8.2 Mandatory cheap decision gate

Before creating any official v3 bank or protocol, the handoff required a
development-only map using the already-failed v2 banks. The prospective
classification rule implemented before computation was:

- A: at least 10 dual-bank candidates;
- B: 1–9 dual-bank candidates;
- C: 0 dual-bank candidates.

Classification C explicitly required stopping before v3 selection. This rule
prevented a second expensive official run from being launched on an unsupported
start-selection premise.

## 9. V3 Phase-1 diagnostic implementation

The new experiment-local command is:

```bash
.venv/bin/python -m experiments.skyrmions_deep_ritz_full.pareto_v3_run \
  --mode diagnose-v2-audit-starts
```

It performs the following:

1. verifies fixed hashes of the v2 protocol Markdown, v2 evaluation Markdown,
   protocol JSON, and failure JSON;
2. inventories and hashes every active v2 output artifact;
3. proves that v2 has no selection hash and no fresh-validation directory;
4. reads the 193 candidates previously feasible at 0.5% on the v2 screen bank;
5. evaluates all 193 on the frozen v2 N=16384 periodic-audit bank using the
   candidate-batched native projection path;
6. computes audit projection, rESS, covariance, and forcing diagnostics;
7. constructs no Tangent optimization and no Full K/f system;
8. writes immutable development-only summary and inventory JSON files.

The v2 output-tree inventory digest is:

```text
e69e58bb0cd02967315b83634551ff66773740c2524f5a110542d8e71f95b723.
```

## 10. V3 diagnostic result

The result was classification C:

```text
C. NO ROBUST 0.5% START FOUND ON v2 BANKS.
```

The core counts were:

| quantity | count |
| --- | ---: |
| screen-feasible 0.5% candidates | 193 |
| audit projection valid | 193 |
| audit rESS >= 0.05 | 0 |
| audit projection+rESS+forcing+covariance valid | 0 |

The audit rESS distribution was:

| statistic | rESS |
| --- | ---: |
| minimum | 0.0358957832 |
| p05 | 0.0436266194 |
| p25 | 0.0446095515 |
| median | 0.0446964976 |
| p75 | 0.0450957830 |
| p95 | 0.0472283739 |
| maximum | 0.0483757149 |

The maximum audit projection residual over all 193 candidates was about
`9.86e-11`, far below `2e-6`. Maximum forcing mean was about `2.61e-8`, below
`2e-7`, and maximum covariance condition was about `4.58`, far below `1e10`.
The population-wide controlling gate was therefore specifically rESS.

The first five candidates ranked by `robust_rESS` were:

| rank | candidate | risk | screen rESS | audit rESS | robust rESS |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | candidate_054 | 5.207900752 | 0.067183582 | 0.048375715 | 0.048375715 |
| 2 | candidate_085 | 5.212220462 | 0.065376993 | 0.048311416 | 0.048311416 |
| 3 | candidate_053 | 5.201734445 | 0.066491908 | 0.047766111 | 0.047766111 |
| 4 | candidate_031 | 5.209314805 | 0.065787139 | 0.047760814 | 0.047760814 |
| 5 | candidate_083 | 5.206476414 | 0.064884923 | 0.047607171 | 0.047607171 |

The full top-20 table, exact geometries, and all per-candidate diagnostics are
in `OFFICIAL_GALERKIN_PARETO_V3_EVALUATION.md` and the machine-readable summary.

The batched diagnostic completed its candidate evaluation in about 11.2
seconds. It used no validation and no Full computation.

## 11. Meaning of the v3 result

The diagnostic disproves the narrow hypothesis that v2 failed only because it
selected the wrong four candidates from the 193 screen-feasible rows. On the
specific frozen v2 periodic-audit bank, none of those 193 rows reaches the
unchanged audit rESS threshold.

It does **not** prove any of the following broader statements:

- no possible 0.5% sensor design can satisfy rESS >= 0.05;
- the rESS threshold is scientifically wrong;
- the earlier N=32768 development witnesses were false;
- increasing absolute sample count must always decrease overlap;
- Tangent or Full optimization is intrinsically invalid.

It establishes a bank- and pool-specific fact: the proposed audit-aware start
repair has no eligible 0.5% seed on the two frozen v2 selection banks.

Because the handoff required a stop under classification C, no official v3
protocol was created. In particular, the following paths do not exist:

```text
outputs/official_galerkin_pareto_v3/protocol.json
outputs/official_galerkin_pareto_v3/banks/
outputs/official_galerkin_pareto_v3/selection/
outputs/official_galerkin_pareto_v3/fresh_validation/
```

There are no official v3 Tangent or Full winners and no v3 validation result.

## 12. Selection/validation firewall

The central data-separation rule has remained intact.

For v2:

- validation seed derivations were frozen before selection;
- validation arrays were forbidden until complete selection freeze;
- v2 failed before selection freeze;
- therefore validation arrays were never generated or opened.

For the attempted v3:

- Phase 1 was development-only and used v2 selection banks;
- classification C stopped the workflow before a v3 protocol;
- no v3 validation seed was turned into data;
- no selection or validation loader was invoked.

At no point was validation used to choose a start, tune a threshold, change K,
or substitute a winner.

## 13. Test and regression evidence

The current combined focused suite covered:

- original and candidate I-projection native paths;
- candidate forward/value/VJP/JVP equivalence;
- Galerkin JAX/Tesseract chunk equivalence;
- backend flag behavior and invalid-backend failure;
- all v2 protocol/firewall contracts;
- v3 v2-immutability and Phase-1 decision contracts.

The final relevant regression run reported:

```text
177 passed, 4 skipped in 23.43 seconds.
```

The 20 v3-specific Phase-0/1 tests verify:

- v2 fixed hashes;
- absence of v2 validation;
- the `official_pareto_v3` seed namespace;
- unchanged K, dictionary, rank, rESS, and energy values;
- exact risk arithmetic;
- exactly 193 input candidates and zero dual-rESS candidates;
- `robust_rESS = min(screen,audit)` for every row;
- all 193 audit projections pass;
- classification C prevents v3 continuation;
- no Tangent optimization or Full K/f construction occurred;
- no v3 protocol, bank, selection, or validation tree exists.

`git diff --check` passes. V2 hashes remained unchanged after the diagnostic.

## 14. Current repository map

### Shared/additive native infrastructure

```text
native/iprojection_tesseract/             original, clean vs origin/main
native/candidate_iprojection_tesseract/   independent candidate batching
native/galerkin_tesseract/                optional CPU Galerkin assembly
src/mfsi/projection.py                     candidate high-level API
src/mfsi/projection_tesseract.py           independent native routing
src/mfsi/galerkin_tesseract.py             optional Galerkin wrapper
tests/test_projection_tesseract.py         original + candidate tests
tests/test_galerkin_tesseract.py           Galerkin equivalence tests
```

### Pareto v2 code and evidence

```text
pareto_v2_common.py
pareto_v2_selection.py
pareto_v2_validation.py
pareto_v2_report.py
pareto_v2_run.py
test_pareto_v2.py
OFFICIAL_GALERKIN_PARETO_V2_PROTOCOL.md
OFFICIAL_GALERKIN_PARETO_V2_EVALUATION.md
outputs/official_galerkin_pareto_v2/
```

### Pareto v3 Phase-0/1 code and evidence

```text
pareto_v3_common.py
pareto_v3_diagnostic.py
pareto_v3_run.py
test_pareto_v3.py
OFFICIAL_GALERKIN_PARETO_V3_EVALUATION.md
outputs/official_galerkin_pareto_v3/diagnostic_v2_audit_map/
outputs/official_galerkin_pareto_v3/final_summary.json
```

There is intentionally no v3 protocol or optimizer implementation beyond the
gated diagnostic because classification C required stopping before those
stages.

## 15. Reproduction commands

### Build candidate I-projection

```bash
.venv/bin/cmake \
  -S native/candidate_iprojection_tesseract \
  -B native/candidate_iprojection_tesseract/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python"

.venv/bin/cmake --build \
  native/candidate_iprojection_tesseract/build -j 8
```

### Build optional Galerkin Tesseract

```bash
.venv/bin/cmake \
  -S native/galerkin_tesseract \
  -B native/galerkin_tesseract/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python"

.venv/bin/cmake --build native/galerkin_tesseract/build -j 8
```

### Re-read the immutable v3 diagnostic

```bash
.venv/bin/python -m experiments.skyrmions_deep_ritz_full.pareto_v3_run \
  --mode diagnose-v2-audit-starts
```

The command is resumable. When the saved diagnostic exists, it re-verifies the
v2 tree digest before returning the cached result.

### Run focused tests

```bash
.venv/bin/python -m pytest \
  experiments/skyrmions_deep_ritz_full/test_pareto_v3.py \
  experiments/skyrmions_deep_ritz_full/test_pareto_v2.py \
  tests/test_projection_tesseract.py \
  tests/test_galerkin_tesseract.py -q
```

## 16. Performance recommendations

### High value / low risk

1. Continue using candidate-batched native I-projection for large candidate
   maps. It materially reduces per-candidate native call overhead without
   changing converged solutions.
2. Reuse exact-eta projection/forcing artifacts only when bank, config, dtype,
   code, and eta hashes match exactly.
3. Keep fixed JIT shapes and deterministic candidate batches.
4. Deduplicate identical finalist geometries before any future high-fidelity
   evaluation.

### Medium value

1. Test nearby-eta multiplier trajectories as Newton initial guesses. This is
   acceptable only if the converged multiplier, residual, action, and gradient
   remain within frozen equivalence tolerances.
2. Investigate time-sharded fixed-basis cache/mmap designs. A full K280 basis
   cache is large, so memory pressure and transfer cost must be measured.
3. Tune JAX K/f chunk size prospectively. The current chunk size favors bounded
   memory; larger chunks may improve throughput but can increase GPU memory
   pressure.
4. Schedule multiple starts to reuse compiled kernels without concurrently
   contending for GPU memory.

### Not recommended

1. Do not put the current CPU Galerkin Tesseract in the GPU hot path. It is
   numerically valid but slower because of host callbacks and transfers.
2. Do not build per-sample KxK Gram tensors; memory scales disastrously.
3. Do not lower precision, relax rESS, relax energy, change K, or differentiate
   through eigensolves for speed.
4. Do not overlap uncontrolled CPU-native projection and GPU Full work until a
   deterministic contention profile justifies the complexity.

## 17. Scientifically valid next options

The existing v3 handoff is finished at classification C. Continuing it by
freezing a v3 protocol anyway would violate its prospective decision rule.

A future project must be separately named and prospectively frozen. Plausible
development questions include:

1. **Independent-bank robustness study.** Quantify how often 0.5% candidates
   pass rESS over multiple independently seeded banks at fixed N, without using
   validation and without selecting an official winner.
2. **Candidate-pool coverage study.** Determine whether the current local
   feasible-manifold generator undercovers regions with stable cross-bank
   overlap. Any expanded generator must be specified before evaluating a new
   official bank.
3. **Quadrature design study.** Investigate whether the current reference-bank
   proposal produces high-variance minimum-time rESS near 0.5%. Changing the
   proposal or estimator would be a methodological change requiring independent
   qualification.
4. **Predeclared replicate gate.** Consider whether future selection should
   require a quantile or worst-case rESS over multiple selection-audit banks.
   This must be justified scientifically and frozen before results.
5. **Partial-curve protocol.** A future protocol could independently classify
   each allowance/method and continue larger allowances after a low-allowance
   failure. That logic was proposed for v3 but was never activated because v3
   did not pass Phase 1.

None of these options authorizes mining validation, reusing validation as
selection data, lowering 0.05, or relabeling v2/v3 as successful.

## 18. Exact current status

As of this worklog:

- the original I-projection native folder is clean versus `origin/main`;
- candidate-specific projection has an independent native/Tesseract component;
- Galerkin assembly has an optional independent native/Tesseract component;
- JAX is the configured/default Galerkin K/f backend;
- v2 remains a frozen failed protocol;
- v2 official selection banks and screening evidence remain immutable;
- v2 never froze selection and never opened validation;
- the v3 development diagnostic is complete and immutable;
- v3 Phase 1 is classification C with 0/193 robust starts;
- no official v3 protocol or official v3 selection/validation result exists;
- the detailed v2 and v3 evaluation reports are the authoritative scientific
  records;
- the relevant regression suite passes 177 tests with 4 skips.

The central lesson is that the engineering acceleration worked, the scientific
firewalls worked, and the fail-closed protocol worked. The blocker is not code
speed or a projection failure. It is the absence, on the frozen v2 audit bank,
of any 0.5%-risk candidate in the declared pool whose independent rESS reaches
the unchanged 0.05 gate.
