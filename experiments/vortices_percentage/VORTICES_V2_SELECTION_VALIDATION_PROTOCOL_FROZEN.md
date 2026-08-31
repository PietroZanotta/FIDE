# Vortices V2 prospective selection and validation protocol — frozen version 1

**Freeze date:** 2026-08-30  
**Repository HEAD:** `1e65587684857a84e98fda181fe472d62f46f7a8`  
**Machine-readable selection config:** `VORTICES_V2_SELECTION_CONFIG.json`

> **NO V2 REFERENCE MODEL, SELECTION OUTCOME, OR VALIDATION OUTCOME HAD BEEN
> INSPECTED WHEN THIS PROTOCOL WAS FROZEN.**

This protocol becomes operational only after
`preflight_vortices_v2_freeze.py` passes against
`VORTICES_V2_FREEZE_MANIFEST.json`. The repository contained unrelated and
previously existing working-tree changes at freeze time. No commit or tag was
created automatically. The manifest therefore records exact hashes for every
scientific dependency and the preflight rejects any unrecorded change within
the V2, Toy, V1, or shared-dependency scopes. A clean reviewed commit/tag is
recommended before the separate scientific run.

## 1. Scope and immutable history

This is a prospective Vortices V2 experiment. It does not modify or supersede:

- the analytic Toy/Gaussian-mixture experiment;
- the Vortices V1 numerical results;
- the V1 2,048-trial numerical PASS/statistical FAIL;
- the first column-normalized V2 numerical-development record; or
- the continuity-commutator and final reflection/Neumann audits.

V1 trials and scientific outcomes cannot enter V2 inference. Ten historical
V1 geometries are permitted only as declared proposal candidates and must
compete normally under the new V2 banks, references, screens, and action.

## 2. Frozen numerical action

The numerical method is closed:

| Item | Frozen definition |
|:---|:---|
| Projection | hard empirical information projection |
| Defect | `s=partial_t q+div(q u)` |
| Scalar density | direct particle, cell-integrated even-reflection/Neumann Gaussian |
| Signed source | the identical reflected scalar kernel |
| Reference flux | `j_h=S_reflect(q u)`; odd in its normal coordinate and even tangentially |
| Regularized velocity | `u_h=j_h/q_h`, never `u_original` sampled on the grid |
| Correction | `delta_h=-grad psi_h` |
| Operator | `K(q_h)=-div(q_h grad)` |
| Equation | `K(q_h) psi_h=-s_h` |
| Correction boundary condition | homogeneous Neumann |
| Exact grid | `256 x 128` |
| Time nodes | all 21 equally spaced nodes |
| Reflected image pairs | 4 |
| Density floor | 0 |
| Precision | float64 |

No soft fiber, `lambda_dot`-only ridge, alternate reflected kernel, action cap,
or performance-selected bandwidth is allowed.

## 3. Fresh reference replicates

The frozen training seeds are

```text
310000101, 310000102, 310000103.
```

`REFERENCE_SEED_PROVENANCE.md` documents the repository and relevant-history
audit. Historical seeds `20260815`, `20260816`, and `20260817` are ineligible.

All references use the same 50,000-particle physical endpoint dataset, SHA-256
`ad4006927e268c52f621c16c773f0600d803370bd21fb5e0816d82a70dbdfbba`.
The dataset was produced from seed `20262816` with 512 truth RK4 steps. Holding
it common isolates learned-reference training randomness.

For a reference seed, that seed controls:

- neural-network initialization;
- training-time draws;
- independent initial- and final-endpoint index draws in every batch;
- the optimizer minibatch sequence; and
- bridge-normal draws, although bridge noise is exactly zero and makes those
  draws mathematically inactive.

The architecture and training schedule are common: four width-128 hidden
layers, SiLU, 12,000 steps, batch size 2,048, Adam
`(beta1,beta2,epsilon)=(.9,.999,1e-8)`, gradient clip 10, linear bridge, zero
bridge noise, and cosine learning-rate decay from `1e-3` to `5e-5`. Training
occurs in box-logit coordinates; velocity is mapped back by the exact chain
rule.

Each replicate receives an independent deterministic 32,768-particle rollout
initial sample. Its rollout seed is `training_seed+3001`, namely
`310003102`, `310003103`, or `310003104`. RK4 uses 16 substeps per scientific
interval. Once checkpoint and initial particles are fixed, rollout and
velocity evaluation have no randomness.

Each replicate must retain its checkpoint hash, seed, complete training-config
hash, common endpoint-data hash, rollout-bank hash, qualification receipt, and
per-reference Scott bandwidth. Reference-training uncertainty and observation
uncertainty remain separate.

## 4. Reference-only qualification

Qualification occurs before bandwidth freezing or sensor optimization and may
use no geometry, risk, Tangent/Full action, reduction, selection bank, or
validation bank. A reference passes only if:

1. its seed is the corresponding frozen seed;
2. checkpoint metadata records the box-logit transform, width 128, four hidden
   layers, and 12,000 completed steps;
3. the final logged step is 12,000, final CFM loss is finite and at most `10`,
   and final pre-clip gradient norm is finite and at most `100`;
4. all checkpoint parameters are finite;
5. rollout nodes and velocities are finite with shape `[21,32768,2]`;
6. weights have shape `[21,32768]`, sum to one within `1e-12`, and the minimum
   in-domain base mass is at least `.995`; and
7. the 21 per-time Scott bandwidths and their median are finite and positive.

The limits were fixed from architecture expectations and historical
reference-only diagnostics, not V2 scientific performance. Failure stops the
experiment. The failed seed remains failed and cannot be replaced by a fourth
seed after any reference or downstream result is observed.

## 5. Common physical bandwidth

For each qualified reference `r`, compute

```text
h_r = median over the 21 scientific times of
      weighted 2-D Scott bandwidth(reference rollout at that time).
```

Then, before any sensor optimization,

```text
h_common = median(h_1,h_2,h_3).
```

`freeze_common_bandwidth.py` reopens all three rollout banks, verifies every
hash and qualification receipt, recomputes all Scott values, and writes one
immutable receipt. It refuses any other seed set or reference count. The same
`h_common` is used for every reference, method, geometry, allowance,
observation trial, and grid. No action or risk enters this rule.

## 6. Shared observation hierarchy

The frozen unused namespaces are:

```text
selection namespace  = 410000101
validation namespace = 410000102
```

There is exactly one shared 128-trial selection bank and one disjoint shared
1,024-trial validation bank. A trial ID identifies the same 2,000 truth-particle
indices at the same nine acquisition indices
`[0,2,5,8,10,12,15,18,20]` and the same four detector-normal draws for every
method and all three references. Detector-noise standard deviation is `.005`;
endpoints remain exact.

Selection prefixes are deterministic:

| Use | Trial IDs |
|:---|:---|
| Tangent/Full prescreen | `0:32` |
| Law finite-risk selection | `0:64` |
| Tangent/Full final selection | `0:128` |

Validation is never generated until every Population, Law, Tangent, and Full
winner at every allowance is frozen.

## 7. Population and Law definitions

Sensors are four width-`.12` Gaussian probes. Centers lie in
`[.24,1.76] x [.24,.76]`, with minimum pairwise separation `.24`.

For geometry `eta`, `L_r(eta)` is the exact population multiscale MMD under
reference `r`, using bandwidths `.05,.10,.20,.40`. The common Population score
is `L(eta)=(1/3) sum_r L_r(eta)`. Numerical validity must pass separately for
all references. The new Population winner minimizes `L`, and
`L_max=L_star+.025`.

`R_r(eta)` is the arithmetic mean finite-law risk on shared trials `0:64` for
reference `r`; `R(eta)=(1/3) sum_r R_r(eta)`. Law minimizes `R` subject to the
common Population screen and per-reference numerical validity. Its newly
selected `R_star` is the sole V2 Law anchor. The old V1 Law risk is forbidden.

Risk caps are

```text
R_max(p)=R_star+(p/100)|R_star|,
p in {.5,1,2,3,4,5}.
```

## 8. Generated starts, constraints, and candidate identity

The optimizer root seed is `310000201`. A JAX uniform pool of 8,192 labeled
four-sensor layouts is drawn in the center box. Stable valid-first filtering
retains 64 separated starts. Every start remains in audit pools; only the
stage-specific leading counts are optimized.

During continuous optimization, sensors remain labeled to avoid
nondifferentiable permutation boundaries. For duplicate detection, audit, and
reporting, centers are lexicographically sorted by `(x,y)` and rounded to 12
decimal places. Duplicate provenance labels are merged and the first numerical
candidate is retained.

Adam uses `beta1=.9`, `beta2=.999`, `epsilon=1e-8`, no gradient clipping,
constraint penalty `10000`, feasibility tolerance `1e-6`, and a `1e12`
invalid-proxy penalty. Iterates are clipped to the center box. Minimum
separation receives the quadratic search penalty; authoritative audits reject
any violation beyond tolerance. Nonfinite candidates are recorded and
rejected, never replaced silently.

## 9. Complete stage budgets

| Stage | Optimizer | Optimized starts | Steps | LR | Local proposals | Audit/promotion |
|:---|:---|---:|---:|---:|:---|:---|
| Population | multistart Adam | 8 | 100 | `.01` | none | audit 20; require 6 valid |
| Law | multistart Adam | 6 | 50 | `.008` | none | 4 gradient trials; audit 24; require 8 valid; exact 64 |
| Tangent | multistart Adam | 4 | 50 | `.006` | 12 per Population/Law/incumbent center, scale `.08` | audit 30; prescreen 32; promote 10; exact 128 |
| Full | deterministic derivative-free reflected-proxy search | 64 global | 0 | n/a | 3 rounds, 10 per center at `.06,.03,.015` | audit 30; prescreen 32; promote 8; exact 128 once |

Law permits at most two complete anchor-refinement restarts if a later audited
candidate beats the provisional anchor. Consistency tolerance is `1e-5`.
Exhaustion aborts rather than publishing a false anchor.

Full local centers are the new Population, new Law, current Tangent, and
previous tighter Full incumbent. Tangent uses the analogous Population, Law,
and tighter Tangent centers. Each candidate evaluation and each stage is
atomically checkpointed. Resume requires exact config, input-bank, reference,
bandwidth, and code hashes; a mismatch fails closed and requires a new output
directory.

## 10. V2 Full search proxy

The only Full proxy is the reflected V2 action on `64 x 32`, using all 21 time
nodes, the exact projected law and forcing, four reflected image pairs, the
fixed common physical bandwidth, `K psi=-s`, and zero density floor. It cannot
call the V1 hard-bin raster, shrinking `.35`-cell bandwidth, column-normalized
V2 raster, or regularized V1 action.

Before freeze, `verify_v2_search_proxy.py` tested the base and fixed `±.015`
first-sensor coordinate perturbations for the three previously declared golden
mechanism trials: 15 candidates total. Against `256 x 128`, overall Pearson was
`.99999805`, overall Spearman was `1.0`, every within-case Spearman was `1.0`,
the lowest within-case Pearson was `.99996965`, and the maximum relative action
discrepancy was `1.4818%`. The predeclared correlation gates were `.95` and
`.90`. These are numerical ranking diagnostics on old development data, not V2
scientific evidence.

## 11. Tangent and Full selection objectives

For geometry `eta`, the final Full selection objective is

```text
A_full,select(eta)
  = (1/3) sum_r [(1/128) sum_i A_full(eta,r,i)].
```

The Tangent objective uses the identical equal-reference and shared-128-trial
arithmetic mean with the unchanged particle-space Tangent action. Every finite
valid action is retained.

Before selection, candidates must pass the common aggregate Population and
finite-risk screens. On each reference and trial they must also pass projection
calibration, ESS, empirical support, raster mass/source compatibility,
reflected continuity, one-component conductivity, physical Poisson residual,
moment feasibility, Full/Tangent/Hidden decomposition, and float64 gates.

Allowances run in ascending order. The previous tighter winner is mandatory
and is re-audited at the current caps. It is replaced only by a candidate valid
on all references whose declared objective is lower by more than `1e-6`.
Otherwise its geometry and action repeat exactly.

## 12. Historical V1 proposal list

These candidates have no incumbent or scientific privilege:

| Label | Flat labeled geometry `x1,y1,...,x4,y4` |
|:---|:---|
| `v1_population` | `1.067773587630,.388849765078,.446639685026,.76,1.76,.24,.254416359228,.576922102546` |
| `v1_law` | `1.077489121283,.388962931394,.479798455087,.76,1.76,.24,.257336093013,.602955827375` |
| `v1_0p5_tangent_full` | `1.072627025331,.396292769880,.486241749946,.76,1.76,.24,.271939753687,.618506353015` |
| `v1_1_tangent` | `1.054448525966,.390148811266,.451830212130,.748630244617,1.76,.24,.262004079048,.577847429925` |
| `v1_1_full` | `1.054873562018,.402549031149,.470011449284,.76,1.76,.24,.24,.610885563801` |
| `v1_2_3_tangent` | `1.058357235989,.379515462139,.475428730676,.76,1.757856801653,.24,.289138449825,.571883938578` |
| `v1_2_3_full` | `1.083266484208,.433572469145,.507291099938,.749698126097,1.76,.24,.280866535843,.639628829394` |
| `v1_4_tangent` | `1.070392506009,.426616046167,.490176312755,.700051644156,1.76,.24,.263510271002,.629298016600` |
| `v1_4_full` | `1.063709350804,.455019203406,.498536611401,.742034295263,1.759761794463,.24,.249705956459,.646381958974` |
| `v1_5_tangent_full` | `1.052135304565,.417897198208,.504910855829,.725622399888,1.749147171122,.246883061009,.245616671283,.652486278434` |

## 13. Numerical gates

All authoritative candidates must satisfy:

| Gate | Limit |
|:---|---:|
| Mass absolute error | `5e-13` |
| Integrated-source compatibility | `5e-12` |
| Physical Poisson relative residual | `2e-7` |
| Independent action discrepancy when audited | `2e-6` |
| Projection finite calibration residual | `1e-3` |
| Population calibration residual | `1e-5` |
| Minimum ESS fraction | `.03` |
| Minimum in-domain base mass | `.995` |
| Tangent compatibility / common-raster decomposition | `1e-6` absolute |
| Reflected boundary flux | `1e-14` absolute |
| Conductive components | exactly 1 |

The final numerical implementation additionally retains the frozen
manufactured and golden continuity gates documented in
`VORTICES_V2_NUMERICAL_REPAIR_FINAL.md`.

## 14. Validation and inference

After all winners are frozen, generate the one shared 1,024-trial validation
bank. For reference `r` and allowance `p`, the primary effect is

```text
D_r,p = 1 - mean_i A_full(r,p,i) / mean_i A_law(r,i).
```

There is no trimming, winsorization, censoring, median or log replacement, or
post-hoc action cap. Every finite numerically valid action is retained;
invalidity fails the numerical gate rather than deleting a trial.

The bootstrap uses 100,000 resamples and seed `821775`. Each bootstrap
replicate draws one common vector of 1,024 indices and applies it to all three
references, Law, and all six Full allowances. The 18 effects are computed from
that same draw. The simultaneous critical value is the 95th percentile of the
maximum absolute unstudentized deviation from the 18 observed effects.
`v2_inference.py` implements and tests this cross-method and cross-reference
pairing.

The claim passes only if:

1. exactly three frozen qualified references and 1,024 shared validation
   trials are present;
2. every scientific evaluation is numerically valid;
3. all 18 simultaneous lower bounds are strictly positive;
4. the common simultaneous half-width is at most `.05`;
5. every Law and Full arithmetic mean has within-reference relative SE at most
   `.10`; and
6. there is no V1 confirmation reuse, outcome-dependent stopping, or estimand
   change.

Three references times 1,024 shared trials are not 3,072 independent reference
replicates. Reports must show each reference stratum, the equal-reference
summary, and the between-reference range separately.

## 15. Artifact destinations and future sequence

All prospective outputs live under
`experiments/vortices_percentage_v2/outputs/prospective_v2/`; nothing is
written beneath V1 or Toy output trees.

1. `references/reference_seed_<seed>/`: checkpoint, rollout and qualification.
2. `freeze/common_bandwidth_receipt.json`: immutable common bandwidth.
3. `selection/shared_selection_bank.npz`: one master selection bank.
4. `selection/population/` and `selection/law/`: new anchors.
5. `selection/allowances/`: nested Tangent/Full searches.
6. `selection/frozen_winners.json`: all geometries frozen before validation.
7. `validation/shared_validation_bank.npz`: one independent bank.
8. `validation/results/`: all reference/method/trial receipts.
9. `validation/simultaneous_inference.json`: frozen inference.
10. `VORTICES_V2_FINAL_RESULT.md`: final result regardless of pass or fail.

`dry_run_vortices_v2_workflow.py --dry-run` prints every dependency, input,
future output, seed, namespace, sample count, and known hash without writing a
file or evaluating an action.

## 16. Stop rule

This document freezes researcher choices; it does not authorize outcome-driven
changes. After preflight review, the next scientific execution must be a
separate explicit command. This task stops before reference training,
selection-bank generation, optimization, or validation-bank generation.

