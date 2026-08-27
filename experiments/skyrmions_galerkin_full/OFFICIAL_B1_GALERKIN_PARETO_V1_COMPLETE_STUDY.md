# Official B1 Galerkin Pareto v1 — Complete Implementation and Scientific Record

Status: **COMPLETE**  
Record date: **2026-08-27**  
Experiment identifier: `skyrmion_official_b1_galerkin_pareto_v1`  
Official output root: [`outputs/official_b1_galerkin_pareto_v1/`](outputs/official_b1_galerkin_pareto_v1/)  
Final confirmation root: [`outputs/skyrmion_b1_final_support_confirmation_v1/`](outputs/skyrmion_b1_final_support_confirmation_v1/)  
Clean-room development root: [`outputs/skyrmion_galerkin_dev_single_reference_b1_preflight_v1/`](outputs/skyrmion_galerkin_dev_single_reference_b1_preflight_v1/)

## 1. Executive conclusion

The corrected single-reference B1 skyrmion FIDE/MFSI experiment completed its
prospectively gated development confirmation, official selection, selection
freeze, and fresh validation.

The central conclusions are:

1. The endpoint-qualified B1 reference checkpoint passed all eight frozen
   production-launch conditions. The development Law passed all 16 new
   individual confirmation banks, with minimum rESS `0.086257`, safely above
   both the scientific threshold `0.05` and the development launch margin
   `0.060`.
2. A completely fresh official Law was reconstructed from official selection
   data. Its scientific-risk anchor is
   `R_Law_official = 0.5205351891956636`; the development value
   `0.4979830145383645` was not reused as the production anchor.
3. A fresh, canonical, feasible pool of exactly 4,096 official candidates was
   generated. Of these, 3,970 passed both the N=8,192 screen and the independent
   N=16,384 periodic audit.
4. Tangent and fixed-feature K=280 Galerkin Full each produced certified
   selection winners at every allowance `p = 0.5, 1, 2, 3, 4, 5%`.
5. The complete selection was frozen before validation. The immutable selection
   SHA-256 is
   `ef268eb434386bf853289f01d668bf10b003b84963d844e2122b2e69d0edbe6a`.
6. All 18 fresh-validation rows—Law, Tangent, and Full at six allowances—passed
   numerical certification, strict nominal-p risk, and the separately reported
   prospectively frozen p+5 percentage-point validation rule.
7. Full validation reduction relative to validation Law is approximately
   `5.67–5.72%` at 0.5–4%, then `21.8225%` at 5%.
8. The nearly flat 0.5–4% achieved curve is an optimizer/protocol-limited local
   continuation, not evidence that the mathematical constrained K=280 Pareto
   frontier itself is flat. At 5%, Full changes to a different candidate branch
   and uses almost all of the available risk budget.

The word **Full** in this document always means the **fixed-feature,
finite-dimensional K=280 Galerkin approximation**. It is not a claim of
convergence to an infinite-dimensional Full solution. Deep Ritz did not enter
selection, certification, or validation.

## 2. Scope, object definitions, and nonclaims

The experiment optimizes four periodic local-density sensors. A geometry is an
eight-vector

```text
eta = [x1, y1, x2, y2, x3, y3, x4, y4]
```

on the rectangular torus `[0,2] x [0,1]`, with minimum pairwise periodic sensor
separation `0.20`. Sensor labels are scientifically unordered. Candidate
canonicalization therefore uses periodic wrapping and exhaustive
permutation-aware matching, rather than naïve flattened Euclidean comparison.

The observable associated with sensor `j` is the particle-averaged Gaussian
local density

```text
Phi_j(X; eta) = (1/16) sum_i exp(-|x_i-c_j|_T^2 / (2 sigma^2)),
sigma = 0.12,
```

where `|.|_T` uses the minimum-image displacement on the torus.

The experiment does not claim:

- that K=280 is an infinite-dimensional limit;
- that the 4,096-member pool exhausts feasible sensor geometry;
- that one accepted optimizer step finds a global constrained optimum;
- that raw Tangent action and raw Full action are directly comparable;
- that historical B0 or Deep Ritz action reductions carry over to B1;
- that the achieved 0.5–4% plateau is a fundamental physical plateau.

## 3. Physical benchmark and scientific risk

### 3.1 Truth dynamics

The truth model is the 16-particle driven Thiele-type point-skyrmion system in
[`domain.py`](domain.py). Its frozen physical parameters are:

| quantity | value |
|---|---:|
| particles | 16 |
| periodic box | `(2.0, 1.0)` |
| interaction strength | `0.035` |
| interaction length | `0.16` |
| pinning strength | `0.055` |
| pinning width | `0.10` |
| pinning centers | `(0.36,0.24)`, `(0.72,0.74)`, `(1.05,0.46)`, `(1.43,0.78)`, `(1.72,0.25)` |
| longitudinal drive | sigmoid ramp from `0.015` to `0.13` |
| transverse drive amplitude | `0.018` |
| dissipation | `1.0` |
| Magnus coefficient | `0.32` |
| initial jitter | `0.035` |
| truth noise standard deviation | `0.006` |
| retained time nodes | 13 on `[0,1]` |
| truth substeps per interval | 24 |

The initial configurations are jittered lattice configurations with randomly
permuted particle labels. Consequently, no downstream component may rely on a
fixed lattice-label order.

### 3.2 Scientific risk

Scientific risk is independent of the optimized sensor outputs. It uses nine
permutation-invariant many-body features from [`risk.py`](risk.py):

- four pair-distance radial channels centered at `0.10, 0.20, 0.32, 0.48`,
  each with width `0.055`;
- four structure-factor channels with wavevectors corresponding to
  `(1,0)`, `(0,1)`, `(1,1)`, and `(2,0)` on the periodic box;
- one mean local hexatic-order magnitude.

Let `Psi(X)` denote this nine-vector, `mu_truth(t)` its truth mean, and `M` the
inverse ridge-regularized covariance estimated from fresh official selection
truth. With projected reference weights `w_eta(t,n)`, risk is

```text
mu_eta(t) = sum_n w_eta(t,n) Psi(X_ref(t,n))

R(eta) = sum_t omega_t
         (mu_eta(t)-mu_truth(t))^T M (mu_eta(t)-mu_truth(t)),
```

using normalized trapezoidal time weights over the 13 nodes. Official
allowance feasibility is exact:

```text
R(eta) <= (1 + p/100) R_Law_official.
```

There is no selection-side slack.

## 4. B1 reference lineage

### 4.1 Why B1 exists

Earlier reference-seed and bridge studies showed that single-reference support
was sensitive to endpoint-coupling semantics. B1 retains endpoint-only
conditional-flow-matching training but performs exact particle matching within
each paired endpoint configuration. It does **not** perform configuration-level
optimal transport.

```text
B1 particle matching: YES
configuration OT: NO
intermediate truth used for reference training: NO
```

### 4.2 Clean-room data

The accepted reference was built in a development-only clean room:

| artifact | N | SHA-256 |
|---|---:|---|
| endpoint reference-training data | 12,000 | `41a2551c75cc26c5edfbaa59b1849e4280abe5ecb5a8caa192983d8e4ac45e3e` |
| endpoint qualification holdout | 4,096 | `c65b25bb04fc04ae56b83412bafac4e5b2abb0140eb0c071e586a112deb51622` |
| development selection truth | 6,000 | `957014cf63c062b37fafd890c2e23a211b4b1d90be153f0c282a2e769f4ea8ae` |

Only the endpoint datasets entered reference training or endpoint
qualification. The 6,000-trajectory development truth bank was selection-side
evidence and never a training target.

### 4.3 Accepted checkpoint

The first prospectively ordered training attempt passed every endpoint-only
qualification gate, so attempts B and C were never trained.

| item | value |
|---|---|
| accepted attempt | `A` |
| training seed | `76881925` |
| checkpoint SHA-256 | `1e13e2ea58df122702d4f555f8788a148b3150bbfbfc953cbac9f963c03d539b` |
| training-config SHA-256 | `28e2cc7ab81e26674792b83c4ad4d654402879fa426b6edf4bf6ac86c7e43e6d` |
| B1 bridge SHA-256 | `8a16d95d33cce1a25a1ea1d91d92769916580f609f86af871eeef9f4dffe9d8b` |
| CFM qualification loss | `0.08935777197172345` |
| endpoint Psi L2 | `0.01517284798055838` |
| endpoint whitened-Psi norm | `0.8567926261044824` |
| endpoint historical-Law Phi L2 | `0.0032513627066367676` |
| qualification decision used intermediate truth | `false` |

This checkpoint was the only reference eligible for confirmation and official
production. It was copied byte-for-byte into the official artifact namespace;
its hash remained unchanged. No production retraining occurred.

## 5. Final support confirmation and launch decision

The final confirmation used only the already frozen development B1 Law and
2,048-member development candidate pool. It did not optimize Law, create new
candidates, train a reference, run Tangent, run Full, or access validation.

### 5.1 Fresh confirmation banks

Eight new pairs were frozen under namespace
`skyrmion_b1_final_support_confirmation_v1`:

```text
screen N = 8,192
audit  N = 16,384
pairs      = 8
banks      = 16
```

All 16 seeds were new and immutable before generation. Every individual bank
used the accepted checkpoint above and the unchanged support thresholds:

| gate | threshold |
|---|---:|
| minimum rESS | `0.05` |
| maximum projection residual | `2e-6` |
| maximum forcing mean | `2e-7` |
| maximum covariance condition | `1e10` |

### 5.2 Development Law confirmation

| bank | minimum rESS | controlling node | node-7 rESS | node-7 lambda norm | node-7 top-1% mass |
|---|---:|---:|---:|---:|---:|
| screen_0 | 0.086257 | 7 | 0.086257 | 143.868 | 0.202907 |
| audit_0 | 0.110531 | 6 | 0.113082 | 153.236 | 0.205413 |
| screen_1 | 0.101275 | 6 | 0.103454 | 167.483 | 0.217046 |
| audit_1 | 0.105134 | 6 | 0.111842 | 162.386 | 0.209377 |
| screen_2 | 0.118882 | 6 | 0.123836 | 155.699 | 0.203104 |
| audit_2 | 0.110444 | 7 | 0.110444 | 148.901 | 0.212832 |
| screen_3 | 0.109564 | 6 | 0.119263 | 141.939 | 0.207355 |
| audit_3 | 0.110902 | 6 | 0.120874 | 149.875 | 0.207826 |
| screen_4 | 0.115013 | 6 | 0.118486 | 168.805 | 0.211775 |
| audit_4 | 0.106244 | 6 | 0.108680 | 152.580 | 0.208745 |
| screen_5 | 0.117893 | 6 | 0.129805 | 154.052 | 0.200223 |
| audit_5 | 0.116607 | 6 | 0.127321 | 162.465 | 0.202819 |
| screen_6 | 0.125606 | 6 | 0.127165 | 149.358 | 0.196272 |
| audit_6 | 0.112557 | 6 | 0.117821 | 152.489 | 0.211069 |
| screen_7 | 0.110561 | 6 | 0.116880 | 158.601 | 0.215036 |
| audit_7 | 0.121916 | 6 | 0.127434 | 154.763 | 0.201274 |

Summary:

```text
minimum Law rESS = 0.086257
median Law rESS  = 0.110731
scientific pass  = 16/16
```

### 5.3 Frozen candidate confirmation

| allowance | candidates inside development risk | survivors of all eight complete pairs |
|---:|---:|---:|
| 0.5% | 395 | 395 |
| 1% | 429 | 429 |
| 2% | 473 | 473 |
| 3% | 500 | 500 |
| 4% | 522 | 522 |
| 5% | 536 | 536 |

For the 0.5% set:

```text
robust-rESS p10                    = 0.08375957933404508
robust-rESS median                 = 0.08627492741278982
symmetry-aware diverse survivors  = 5
minimum pair survival fraction    = 1.0
maximum non-rESS failure fraction = 0.0
```

All eight frozen launch conditions passed. The classification was therefore
`PRODUCTION_LAUNCH_READY`.

### 5.4 Confirmation bookkeeping erratum

The seed manifest stored fully qualified derivation roles such as `screen_0`.
An initial artifact-label helper appended the pair index again, producing
temporary labels `screen_0_0` and `audit_0_0`. The run was interrupted before
any candidate evaluation. A source erratum then normalized the artifact role to
`screen`/`audit`. It explicitly records:

```text
scientific protocol changed: NO
seeds changed:               NO
sample sizes changed:        NO
pairing changed:             NO
thresholds changed:          NO
candidate evaluation begun:  NO
```

The two duplicate partial bank artifacts are retained for auditability. The
erratum is [`final_support_source_erratum.json`](outputs/skyrmion_b1_final_support_confirmation_v1/final_support_source_erratum.json).

## 6. Official protocol and reproducibility seals

The official machine-readable protocol was written before any official array.

| seal | SHA-256 |
|---|---|
| protocol payload | `78cd16a8e5c04c8848a54087719f8da32e8e0489b0f9ad0cfffbab0e38ba468e` |
| `protocol.json` file | `08a38dd0efa199cc8e6758f44eb956be18706e05a1946f1d30301b9128a96924` |
| accepted B1 checkpoint | `1e13e2ea58df122702d4f555f8788a148b3150bbfbfc953cbac9f963c03d539b` |
| K=280 dictionary | `37e9b60fcb92c4e5a0ee7ec1651fb7f8889f7ac6bdb02d3bd314e9ef40833326` |
| physical simulator | `515f6c98064aa4164b30f5f86d1e92f2ead4babf2ba70ee254b57a2799058af3` |
| scientific risk implementation | `69848cbb3c3f686b1292f8b1752179c0f651bcab1fc1a174d15f95857075553a` |
| official workflow implementation | `29fb2649ccbf34e203c27b318ef444320f43a1bf11920e863673b955ed9f067b` |
| official CLI | `5dbd685c58814e9da786e0212d85cafa4ebfade2529b07539878edd8964daa29` |
| inherited K=280 selection engine | `2f17048b93fe6c416471192bd67fd26bdd79149af634bffe1c43b8937f34cd96` |
| inherited validation engine | `21f6445c6c9bbf48295898cf0a439af72862a8e36ba95e81f137aed3414444c3` |

The protocol froze float64 arithmetic, JAX Galerkin assembly, K=280,
rank tolerance `1e-12`, rESS `0.05`, projection `2e-6`, forcing mean `2e-7`,
covariance condition `1e10`, and winner replacement tolerance `1e-10`.

## 7. Official data generation

### 7.1 Selection truth and whitening

Fresh official selection truth used 6,000 complete truth trajectories under
seed `1583265408`.

| artifact | SHA-256 |
|---|---|
| design truth NPZ | `b74d68374669acb063c610a20b0d8350daa7afca9e1f7217ef2b62572ca62993` |
| whitening matrix | `2d24d159279725524effbc6faa74c326df84fac9cfaafa0a235c43edc30b9357` |

The whitening matrix is the inverse of the empirical nine-feature covariance
plus a trace-scaled ridge of `1e-5`. It was built from official selection truth
and later reused unchanged in validation, preserving the prospective feature
metric.

### 7.2 Selection reference banks

All official banks were independent, role-disjoint rollouts of the same frozen
B1 checkpoint.

| role | N | seed | NPZ SHA-256 | generation s |
|---|---:|---:|---|---:|
| Law search | 32,768 | 1914645731 | `508e3d342581fe08ae9dcb99de51c233a461d0cfdea7ac5cda15cfe3df513270` | 42.120 |
| risk anchor | 32,768 | 1149463244 | `38475035ab1c09604bc60015d1e94f1d9a4cd604ad361eee4235cdde1ec5f651` | 40.251 |
| cheap screen | 8,192 | 1824724371 | `c4766d6a628a61fe4206dc6a3b93306f76f8efbb15562e54179ced2be1dc7c90` | 10.880 |
| continuous search | 32,768 | 1225510557 | `ece287bf2bcf225005c3e0c33cd3741c44c97059e294169be6525f510b222d60` | 40.055 |
| periodic audit | 16,384 | 1847370677 | `c4b21a6b8aceb998d703951316194775c7c50194c64331ba1896e40ae948b803` | 20.705 |
| authoritative train | 65,536 | 1160119042 | `84b722a93a61d8bfede83edc2182f2f5449f99703a90f1af473b59c14f9ba588` | 82.364 |
| authoritative audit | 65,536 | 1996580043 | `a4f8a14cefff1f3db73a883dae509e5d5e990eeafd74c447549751824a6bccc9` | 80.332 |

The bank manifest verifies pairwise-disjoint initial-state hashes and
`reference_retrained = false`.

## 8. Official Law reconstruction

The development Law was a mandatory context candidate, not an automatic
official winner. The official search was scientific-risk-only; Tangent and Full
did not define Law.

The frozen initial pool contained:

| component | count |
|---|---:|
| mandatory contexts | 2 |
| local cloud | 1,022 |
| Sobol global | 512 |
| total | 1,536 |

Three independently persisted refinement pools added `128` candidates each,
for `1,920` initial-plus-refined evaluations. The top supported candidates were
shortlisted on the Law-search bank; 24 finalists were evaluated on the
independent risk-anchor bank. The winner was candidate `candidate_00051` from
refinement round 3.

Development Law:

```text
[0.3579165299258919,
 0.24142147825748522,
 1.1169800086283912,
 0.8647621670744411,
 0.19330094464808556,
 0.5605533130981228,
 1.688901674991126,
 0.44300108779651315]

R_Law_B1_dev = 0.4979830145383645
```

Official Law:

```text
[0.33804197624141186,
 0.23852050068148709,
 1.10517864267751,
 0.8049414642903351,
 0.20889866775790386,
 0.5733995593368788,
 1.7310785419380261,
 0.4522625467670803]

R_Law_official = 0.5205351891956636
```

The Law had search-bank risk `0.49492172797803`; its authoritative official
anchor is the independently recomputed risk-anchor-bank value above.

| allowance | exact official ceiling |
|---:|---:|
| 0.5% | 0.5231378651416418 |
| 1% | 0.5257405410876201 |
| 2% | 0.5309458929795768 |
| 3% | 0.5361512448715334 |
| 4% | 0.5413565967634901 |
| 5% | 0.5465619486554467 |

The difference between development and official risk is expected: both truth
and reference risk-anchor banks were regenerated under fresh official seeds.
The official value—not the lower development value—governs every official
allowance.

## 9. Official candidate generation and support screen

The generator specification was sealed before candidate scoring. Exactly 4,096
canonical, feasible, unique candidates were produced:

| component | count | fraction |
|---|---:|---:|
| multi-scale local/context | 1,434 | 35.01% |
| risk-tangent directions | 1,024 | 25.00% |
| periodic paths | 819 | 20.00% |
| scrambled Sobol global | 819 | 20.00% |

The local scales were
`0.00025, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.04`.
Risk-tangent directions were made orthogonal to a central-finite-difference
official-risk gradient using epsilon `1e-4`, then sampled over radii from
`1e-4` through `0.055`. Periodic paths used low-discrepancy golden-ratio
interior points. Sobol proposals were filtered through exact periodic geometry.

Every candidate was evaluated on the official risk-anchor bank, N=8,192 screen,
and N=16,384 periodic audit. Full K/f systems were not assembled for the pool.

```text
pool count                  = 4096
dual-bank support passes    = 3970
dual-bank support fraction  = 0.96923828125
Full K/f solves at screen   = 0
```

| allowance | exact risk- and dual-bank-eligible | frozen starts |
|---:|---:|---:|
| 0.5% | 1,218 | 6 |
| 1% | 1,382 | 6 |
| 2% | 1,569 | 6 |
| 3% | 1,712 | 6 |
| 4% | 2,055 | 6 |
| 5% | 2,126 | 6 |

Starts were chosen deterministically from low risk, best robust rESS, and
symmetry-aware max-min diversity, with the previously certified method-specific
winner inserted as the mandatory incumbent at the next allowance.

## 10. Tangent and Full implementations

### 10.1 Tangent objective

For reconstructed sensor-moment derivatives `cdot_eta`, the Tangent branch uses
the exact four-observable Gram objective on its current reference bank:

```text
r = E_Q[J Phi_eta u] - cdot_eta
G = E_Q[J Phi_eta J Phi_eta^T]
A_Tangent = r^T G^dagger r.
```

Tangent uses six starts per allowance. Search occurs on N=32,768 with an
independent N=16,384 audit; finalists are recomputed on N=65,536/N=65,536.
Tangent action is meaningful within the Tangent branch but is not compared
numerically as if it were Full action.

### 10.2 Fixed-feature K=280 Full objective

The dictionary is eta-independent, permutation-invariant, and frozen at K=280.
At each time node,

```text
K_t[j,k] = E_Q[grad phi_j . grad phi_k]

f_t[j] = E_Q[h_eta (phi_j - E_Q[phi_j])]

K_t a_t = -f_t

A_Full(eta) = sum_t omega_t a_t^T K_t a_t.
```

The rank-aware eigensolve uses relative tolerance `1e-12`. The eta derivative
is the fixed-coefficient envelope derivative. The implementation never
differentiates through the eigendecomposition.

Full uses three starts per allowance. Search uses N=32,768 and N=16,384 audit;
authoritative finalists use N=65,536 train and N=65,536 independent audit.

### 10.3 Forcing and support construction

The empirical I-projection solves for exponential-tilt weights matching the
reconstructed sensor moments. The continuity forcing includes the projected
feature covariance, reference advective feature rates, and implicit
`lambda_dot`. Only floating-point gauge offset is centered away; material
pre-centering forcing means fail the gate.

Support validity requires all of:

- geometry validity and periodic minimum separation;
- projection calibration residual no greater than `2e-6`;
- minimum rESS at least `0.05`;
- forcing compatibility mean no greater than `2e-7`;
- covariance condition no greater than `1e10`.

### 10.4 Full certification

Authoritative Full certification additionally requires:

| diagnostic | threshold |
|---|---:|
| range residual | `1e-8` |
| stationarity residual | `1e-8` |
| energy identity relative error | `1e-8` |
| raw symmetry residual | `1e-12` |
| retained condition | `1e12` |
| minimum rank fraction | `0.5` |
| held-out weak residual | `0.12` |
| held-out energy residual | `0.08` |
| gauge residual | `1e-9` |
| moment-rate residual | `0.10` |

### 10.5 Optimizer budget

Both branches were prospectively limited to:

```text
maximum accepted-step attempts = 1
maximum backtracks             = 3
initial step                   = 5e-5
backtrack factor               = 0.5
trust radius                   = 2e-4
audit cadence                  = every accepted step
replacement tolerance         = 1e-10
Full rank must remain stable   = yes
```

This very limited local-search budget is crucial when interpreting the achieved
Pareto curve.

## 11. Exact selected geometries

### 11.1 Tangent winners

| p | start role | exact eta |
|---:|---|---|
| 0.5% | best rESS | `[0.3431880328966233, 0.23031471115667462, 1.1284319721022071, 0.7875476862113521, 0.19809693986545568, 0.5574428137522496, 1.7334201914061034, 0.4786126119007908]` |
| 1% | previous incumbent | `[0.34320487337666833, 0.23031294823406054, 1.1284493050272482, 0.7875117661202661, 0.19811927546147912, 0.5574403936333359, 1.733418269422276, 0.478623297276521]` |
| 2% | previous incumbent | `[0.3432216860325659, 0.23031122277915062, 1.1284666211833616, 0.7874758327897986, 0.19814162585355302, 0.5574379493420359, 1.733416349649366, 0.4786339792141799]` |
| 3% | previous incumbent | `[0.3432384707570572, 0.23030953470363005, 1.128483920450594, 0.7874398862050669, 0.19816399102594218, 0.5574354808005889, 1.7334144321030505, 0.4786446577069287]` |
| 4% | previous incumbent | `[0.3432552274402322, 0.23030788392041326, 1.1285012027051524, 0.7874039263462008, 0.19818637095989508, 0.5574329879298211, 1.7334125167990448, 0.47865533274667693]` |
| 5% | max-min diverse | `[0.3527907808415285, 0.23435045480900454, 1.0570214248498524, 0.7619982285537953, 0.2124846328771535, 0.5807824059163931, 1.7264648565966991, 0.47264463098370857]` |

### 11.2 Full winners

| p | start role | exact eta |
|---:|---|---|
| 0.5% | best rESS | `[0.3431898513878221, 0.23031461759178007, 1.1284475030118528, 0.7876058232778972, 0.1980936103910932, 0.5574433780001022, 1.7334208799744977, 0.4786162883639749]` |
| 1% | previous incumbent | `[0.3432085119493376, 0.2303127664097894, 1.1284803905654468, 0.7876280572795592, 0.19811261454705834, 0.557441525948614, 1.7334196465890208, 0.47863065169504615]` |
| 2% | previous incumbent | `[0.34322714630002554, 0.23031095801858, 1.1285132852559914, 0.7876502948703434, 0.19813163158849142, 0.5574396535441359, 1.7334184154470627, 0.4786450131055563]` |
| 3% | previous incumbent | `[0.34324575429092524, 0.23030919231727076, 1.1285461870558025, 0.787672536115981, 0.19815066147164057, 0.5574377607278941, 1.7334171865683154, 0.47865937256885677]` |
| 4% | previous incumbent | `[0.34326433578769877, 0.23030746920996392, 1.1285790959523583, 0.7876947810249513, 0.1981697041708202, 0.5574358474372549, 1.7334159599719392, 0.4786737300693168]` |
| 5% | best rESS | `[0.3387716017287543, 0.2151612381975228, 1.1179619178889415, 0.7944660919320236, 0.20816472792211582, 0.5702333911994388, 1.6845669858252275, 0.4580392450162957]` |

## 12. Selection results

### 12.1 Branch objectives

| p | Tangent risk | Tangent action | Full risk | Full K=280 action |
|---:|---:|---:|---:|---:|
| 0.5% | 0.520563486 | 0.0195764723 | 0.520563485 | 0.0331355737 |
| 1% | 0.520563505 | 0.0195732623 | 0.520563504 | 0.0331300282 |
| 2% | 0.520563525 | 0.0195700572 | 0.520563522 | 0.0331244926 |
| 3% | 0.520563545 | 0.0195668575 | 0.520563541 | 0.0331189658 |
| 4% | 0.520563565 | 0.0195636625 | 0.520563560 | 0.0331134484 |
| 5% | 0.537422694 | 0.0172665943 | 0.546174650 | 0.0323973139 |

Every selected risk satisfies its exact official ceiling. The branch-specific
actions should be read down their own columns, not compared across columns.

### 12.2 Common K=280 selection metric

The cross-evaluation recomputed K=280 Full action at Law, Tangent, and Full
geometries on the authoritative banks. Selected common-metric values are in
[`selection/cross_evaluation.json`](outputs/official_b1_galerkin_pareto_v1/selection/cross_evaluation.json).

Two points matter scientifically:

1. At 0.5–4%, Tangent and Full geometries are extremely close and have similar
   common K=280 actions near `0.03314`.
2. At 5%, Tangent's low Tangent action does not transfer to Full: its selection
   K=280 action is `0.0422675326`, worse than Law's `0.0370059473`. The Full
   branch instead selects a different geometry with K=280 action
   `0.0323973139`.

### 12.3 Selection freeze

Only after all six allowances in both branches and the common-metric
cross-evaluation completed were the selections frozen.

| artifact | SHA-256 |
|---|---|
| Tangent selection | `98cfdfd9c66f7711983e6de2fab0fdfbbf4211796994d3a619e214a850c570a5` |
| Full selection | `0a2c5030388abfe42f9a0dc056b003c8dc0bfbdbbc53a7d89622f33cacdb836d` |
| cross-evaluation | `4ccf6ff16a03f4f0e3169d2caabf6be7d1c15d230e8f0ea2e6e8e402901db86c` |
| frozen Pareto selection | `ef268eb434386bf853289f01d668bf10b003b84963d844e2122b2e69d0edbe6a` |
| selection manifest | `125313bd060bb7539442cc3538b1c16239f13e3b235024e91534c68171972b88` |

At selection freeze, `validation_accessed = false` and no fresh-validation file
existed.

## 13. Fresh validation implementation

Validation seeds were present in the protocol before selection, but arrays were
forbidden until the selection hash existed.

| role | N | seed |
|---|---:|---:|
| truth validation | 5,000 | 1142102445 |
| reference fit | 16,384 | 171692060 |
| reference audit | 16,384 | 1799149268 |
| measurement noise | acquisition-shaped | 865005081 |

The fresh artifact manifest verified that validation initial states were
pairwise distinct and disjoint from all official selection bank initial states.
Validation independently rebuilt:

- measurement targets and reconstructed derivatives;
- empirical projection and rESS;
- continuity forcing and covariance diagnostics;
- validation scientific risk;
- Tangent diagnostics;
- K=280 fit action and independent audit action;
- all Full algebraic and physical certificates.

Validation did not optimize eta and verified that the frozen winner-geometry
hash was unchanged.

The validation risk convention was frozen prospectively:

```text
strict: R_method,val <= (1+p/100) R_Law,val

declared p+5pp:
R_method,val <= (1+p/100+0.05) R_Law,val.
```

Both statuses are reported. No slack was invented after observing validation.

## 14. Complete validation results

Validation Law:

```text
R_Law,val              = 0.6179906681749475
K280 fit action         = 0.0362288485160045
K280 audit action       = 0.036314617738831455
action standard error   = 0.0001277024394379412
numerically certified   = YES
```

| p | method | validation risk | risk increase vs validation Law | K280 fit action | K280 audit action | reduction vs Law | action SE | strict p | p+5pp | class |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|
| 0.5% | Law | 0.617990668 | 0.000% | 0.036228849 | 0.036314618 | 0.000% | 0.000127702 | PASS | PASS | PASS |
| 0.5% | Tangent | 0.591978198 | -4.209% | 0.034174283 | 0.034202222 | 5.671% | 0.000096832 | PASS | PASS | PASS |
| 0.5% | Full | 0.591957096 | -4.213% | 0.034173752 | 0.034201682 | 5.673% | 0.000096826 | PASS | PASS | PASS |
| 1% | Law | 0.617990668 | 0.000% | 0.036228849 | 0.036314618 | 0.000% | 0.000127702 | PASS | PASS | PASS |
| 1% | Tangent | 0.591953798 | -4.213% | 0.034170566 | 0.034198529 | 5.681% | 0.000096824 | PASS | PASS | PASS |
| 1% | Full | 0.591911610 | -4.220% | 0.034169504 | 0.034197451 | 5.684% | 0.000096814 | PASS | PASS | PASS |
| 2% | Law | 0.617990668 | 0.000% | 0.036228849 | 0.036314618 | 0.000% | 0.000127702 | PASS | PASS | PASS |
| 2% | Tangent | 0.591929528 | -4.218% | 0.034166862 | 0.034194850 | 5.692% | 0.000096817 | PASS | PASS | PASS |
| 2% | Full | 0.591866272 | -4.228% | 0.034165268 | 0.034193232 | 5.696% | 0.000096801 | PASS | PASS | PASS |
| 3% | Law | 0.617990668 | 0.000% | 0.036228849 | 0.036314618 | 0.000% | 0.000127702 | PASS | PASS | PASS |
| 3% | Tangent | 0.591905387 | -4.221% | 0.034163170 | 0.034191183 | 5.702% | 0.000096810 | PASS | PASS | PASS |
| 3% | Full | 0.591821080 | -4.235% | 0.034161046 | 0.034189027 | 5.708% | 0.000096789 | PASS | PASS | PASS |
| 4% | Law | 0.617990668 | 0.000% | 0.036228849 | 0.036314618 | 0.000% | 0.000127702 | PASS | PASS | PASS |
| 4% | Tangent | 0.591881375 | -4.225% | 0.034159491 | 0.034187529 | 5.712% | 0.000096803 | PASS | PASS | PASS |
| 4% | Full | 0.591776035 | -4.242% | 0.034156837 | 0.034184834 | 5.719% | 0.000096777 | PASS | PASS | PASS |
| 5% | Law | 0.617990668 | 0.000% | 0.036228849 | 0.036314618 | 0.000% | 0.000127702 | PASS | PASS | PASS |
| 5% | Tangent | 0.625125582 | +1.154% | 0.038086748 | 0.038078757 | -5.128% | 0.000126226 | PASS | PASS | PASS |
| 5% | Full | 0.568814554 | -7.957% | 0.028322798 | 0.028387904 | 21.823% | 0.000071104 | PASS | PASS | PASS |

All validation classifications are `PASS`. In particular, every non-Law risk
also passes strict nominal-p; the p+5pp rule is not needed to rescue any row.

## 15. Validation certificate margins

All rows are numerically certified. Representative and extremal diagnostics are:

| quantity | Law | Full 0.5% | Full 5% | threshold |
|---|---:|---:|---:|---:|
| fit minimum rESS | 0.069664 | 0.096177 | 0.155954 | >=0.05 |
| audit minimum rESS | 0.081116 | 0.087289 | 0.158369 | >=0.05 |
| maximum projection residual | 8.75e-11 | 9.68e-11 | 7.65e-11 | <=2e-6 |
| maximum forcing mean | 1.47e-8 | 4.81e-8 | 1.42e-8 | <=2e-7 |
| maximum covariance condition | 6.281 | 7.286 | 9.256 | <=1e10 |
| held-out weak residual | 0.045051 | 0.039046 | 0.032155 | <=0.12 |
| held-out energy residual | 0.064770 | 0.053497 | 0.030519 | <=0.08 |
| gauge residual | 9.39e-18 | 8.02e-18 | 1.00e-17 | <=1e-9 |
| moment-rate residual | 9.92e-4 | 8.33e-4 | 7.52e-4 | <=0.10 |
| minimum rank fraction | 0.996429 | 0.996429 | 0.996429 | >=0.5 |
| worst retained condition | 9.17e11 | 8.76e11 | 8.67e11 | <=1e12 |

The retained Galerkin condition number is the closest algebraic certificate to
its threshold, but every reported row remains below `1e12`. The physical
residuals and support diagnostics have materially larger margins.

## 16. Scientific interpretation

### 16.1 The B1 correction solved the support-launch problem

The accepted reference is not merely a lower training-loss seed. It passed
endpoint-only qualification, then passed a prospectively independent 16-bank
confirmation with strong low-risk population support. The confirmation's
population-wide non-rESS failure fraction was exactly zero. This is the evidence
that authorized production—not the earlier development risk reduction alone.

### 16.2 Official Law differs from development Law for a valid reason

The official Law is displaced from the development Law and has a larger risk
anchor (`0.520535` versus `0.497983`). This is not a degradation relative to an
official fixed truth: the two risks are measured on independently generated
selection truth and reference anchor banks. The protocol explicitly required a
fresh official anchor to prevent optimistic development carryover.

### 16.3 Why Full appears flat from 0.5–4%

The achieved Full validation reductions are:

```text
0.5%  5.6725%
1%    5.6843%
2%    5.6960%
3%    5.7076%
4%    5.7192%
5%   21.8225%
```

From 1–4%, each winning Full trajectory starts from the previous incumbent and
takes the single accepted step permitted by the protocol. Consecutive eta
distances are exactly approximately `5e-5`. Their risks remain essentially
fixed near `0.5205635`, only about `0.0054%` above `R_Law_official`; hence they
do not consume the larger 1–4% budgets. Validation Full action changes only
from `0.0341737516` to `0.0341568366`.

At 5%, Full selects a different `best_rESS` start. Its eta is approximately
`0.0590` in flattened Euclidean distance from the 4% winner, and its selection
risk `0.54617465` is close to the 5% ceiling `0.54656195`. That new branch
produces validation action `0.0283227978` and the `21.8225%` reduction.

Therefore:

- there is a genuine plateau in the **achieved, protocol-limited curve**;
- there is no basis for claiming a flat **true constrained Pareto frontier**;
- a prospectively declared deeper/multistart search at 0.5–4% could test whether
  other branches exist within those budgets;
- the current official selection must not be altered retrospectively on that
  basis.

### 16.4 Tangent is useful but not a universal Full proxy

At 0.5–4%, Tangent and Full select nearby geometries and both validate at about
5.7% Full reduction. At 5%, Tangent strongly lowers its own objective but its
geometry has validation Full action `0.0380867482`, which is `5.128%` worse
than Law. It remains a valid Tangent winner and passes its risk/certificates,
but it is not a Full winner. This is direct evidence for retaining independent
Tangent and Full branches and cross-evaluating on a common metric.

### 16.5 The 5% Full result is the strongest achieved result

The 5% Full geometry simultaneously shows:

- strict validation risk pass;
- validation risk `7.957%` lower than validation Law despite being selected
  near the selection risk ceiling;
- validation Full reduction `21.823%`;
- fit/audit rESS `0.155954/0.158369`;
- lower weak and energy residuals than validation Law;
- complete algebraic and physical certification.

Its risk generalization is favorable but should still be understood as one
fresh validation realization, not a repeated-seed uncertainty study.

## 17. Limitations and recommended follow-up

1. **Finite basis.** Results apply to the frozen K=280 dictionary. They are not
   an infinite-dimensional convergence statement.
2. **Limited outer optimization.** One accepted-step attempt and three starts
   make the search deliberately bounded. This is the main limitation behind
   the 0.5–4% plateau interpretation.
3. **Single official data realization.** Selection and validation are strictly
   disjoint, but this study does not estimate between-seed variance of the final
   Pareto curve.
4. **Candidate-generator dependence.** The 4,096 pool is broad and mostly
   supported, but finite.
5. **Validation uncertainty.** Action standard errors are empirical
   audit-sample standard errors; no pseudo-replicate blocks were introduced.
6. **No retrospective reselection.** Any deeper-search study must use a new
   protocol and new selection namespace. It may compare against these frozen
   incumbents but must not rewrite this experiment.

A scientifically clean next study would prospectively freeze a deeper
multi-branch Full search at 0.5–4%, using these winners as incumbents, additional
risk-efficient starts, more accepted steps, and new selection/audit banks. Its
question should be whether the apparent low-allowance plateau is search-limited,
not whether the existing official result can be post hoc improved.

## 18. Resumability and execution audit

Every major stage is resumable and hash-checked:

- protocol freeze;
- design truth and each reference bank;
- Law pool, each refinement, and Law freeze;
- candidate generator and pool;
- support screen;
- each Tangent trajectory and allowance;
- each Full trajectory and authoritative geometry cache;
- cross-evaluation and selection freeze;
- validation generation and validation results;
- final report and inventory.

The long Full stage was cleanly interrupted at user request and later resumed.
At interruption, allowances 0.5%, 1%, and 2% and the first 3% trajectory were
already sealed. Resume verified and reused those artifacts, then completed the
remaining work.

During selection freeze, the inherited Pareto-v2 `selection_hash.txt` writer
correctly rejected the new B1 output root because of its old namespace guard.
The already-written selection content and manifest were not recomputed or
changed. Their digest was completed using the B1 workflow's sealed atomic
writer, and the complete freeze check then passed. No validation file existed
before that repair or before selection-hash verification.

## 19. Tests and integrity checks

The terminal focused regression suite covered:

- official B1 workflow and protocol semantics;
- final B1 confirmation;
- clean-room single-reference B1 preflight;
- inherited Pareto-v2 selection/validation engine;
- projection Tesseract contract;
- Galerkin Tesseract contract.

Result:

```text
103 passed in 30.10s
```

The three focused B1 modules were also run separately against the archived
layout:

```text
59 passed in 12.58s
```

All 25 archived documents were also compared byte-for-byte with their tracked
pre-move content.

Additional terminal checks:

```text
checkpoint hash audit:         PASS
K=280 dictionary hash audit:   PASS
protocol payload audit:        PASS
selection SHA audit:           PASS
validation geometry unchanged: PASS
126-artifact inventory audit:  PASS
git diff --check:               PASS
```

Key final digests:

| artifact | SHA-256 |
|---|---|
| fresh validation results | `c534b71afa164458b36bce945baf1b750c10b24bcd3c44b96a159a5db7a7b229` |
| final summary | `e7f38eeda7d1aefea1ed2bc701bc35b5926f3b8f504bc3a75df0392ee5ddd9d3` |
| inventory | `cbf0105359a4f8a446415de958f04b5f6d7a4b53b0806fb164417b548ffae6c5` |

Approximate on-disk sizes are 1.1 GiB for the clean-room B1 preflight, 1.4 GiB
for final confirmation, and 1.8 GiB for official production.

## 20. Implementation map

| responsibility | implementation |
|---|---|
| official protocol, data, Law, pool, orchestration adapters | [`official_b1_pareto.py`](official_b1_pareto.py) |
| resumable command-line stages | [`official_b1_pareto_run.py`](official_b1_pareto_run.py) |
| official contract tests | [`test_official_b1_pareto.py`](test_official_b1_pareto.py) |
| final support confirmation | [`final_b1_support_confirmation.py`](final_b1_support_confirmation.py) |
| clean-room B1 reference and development Law | [`single_reference_b1_preflight.py`](single_reference_b1_preflight.py) |
| physical simulator | [`domain.py`](domain.py) |
| sensor observables | [`measurements.py`](measurements.py) |
| continuity forcing and hard support gates | [`forcing.py`](forcing.py) |
| scientific Psi and risk | [`risk.py`](risk.py) |
| K=280 selection engine | [`pareto_v2_selection.py`](pareto_v2_selection.py) |
| fresh validation engine | [`pareto_v2_validation.py`](pareto_v2_validation.py) |
| K=280 basis and assembly | [`production_basis.py`](production_basis.py), [`production_galerkin.py`](production_galerkin.py) |

## 21. Artifact map

The principal human- and machine-readable records are:

- official protocol: [`protocol.json`](outputs/official_b1_galerkin_pareto_v1/protocol.json)
- official Law: [`official_law.json`](outputs/official_b1_galerkin_pareto_v1/law/official_law.json)
- candidate generator: [`generator_spec.json`](outputs/official_b1_galerkin_pareto_v1/candidate_pool/generator_spec.json)
- candidate pool: [`candidate_pool.json`](outputs/official_b1_galerkin_pareto_v1/candidate_pool/candidate_pool.json)
- dual-bank screen: [`screening/candidate_pool.json`](outputs/official_b1_galerkin_pareto_v1/screening/candidate_pool.json)
- Tangent selection: [`tangent/selection.json`](outputs/official_b1_galerkin_pareto_v1/tangent/selection.json)
- Full selection: [`full_search/selection.json`](outputs/official_b1_galerkin_pareto_v1/full_search/selection.json)
- common-metric cross-evaluation: [`cross_evaluation.json`](outputs/official_b1_galerkin_pareto_v1/selection/cross_evaluation.json)
- frozen selection: [`pareto_selection.json`](outputs/official_b1_galerkin_pareto_v1/selection/pareto_selection.json)
- selection manifest: [`selection_manifest.json`](outputs/official_b1_galerkin_pareto_v1/selection/selection_manifest.json)
- validation manifest: [`artifact_manifest.json`](outputs/official_b1_galerkin_pareto_v1/fresh_validation/artifact_manifest.json)
- validation results: [`results.json`](outputs/official_b1_galerkin_pareto_v1/fresh_validation/results.json)
- final summary: [`final_summary.json`](outputs/official_b1_galerkin_pareto_v1/final_summary.json)
- complete inventory: [`inventory.json`](outputs/official_b1_galerkin_pareto_v1/inventory.json)
- compact generated report: [`report.md`](outputs/official_b1_galerkin_pareto_v1/report.md)

Historical root-level Markdown studies have been retained, not deleted, under
[`old_stuff/`](old_stuff/). They are superseded context and do not define this
official B1 result. A few superseded protocol builders hash documents at their
former root-relative paths; reproducing one of those legacy studies therefore
requires temporarily restoring its referenced document from `old_stuff/`.
Archiving those documents does not alter the sealed B1 artifacts, whose active
protocol and source dependencies remain in place.

## 22. Reproduction commands

From the repository root, with the project virtual environment and GPU-enabled
JAX available:

```bash
.venv/bin/python -m experiments.skyrmions_deep_ritz_full.official_b1_pareto_run --mode freeze-protocol
.venv/bin/python -m experiments.skyrmions_deep_ritz_full.official_b1_pareto_run --mode generate-data
.venv/bin/python -m experiments.skyrmions_deep_ritz_full.official_b1_pareto_run --mode law
.venv/bin/python -m experiments.skyrmions_deep_ritz_full.official_b1_pareto_run --mode candidates
.venv/bin/python -m experiments.skyrmions_deep_ritz_full.official_b1_pareto_run --mode screen
.venv/bin/python -m experiments.skyrmions_deep_ritz_full.official_b1_pareto_run --mode tangent
.venv/bin/python -m experiments.skyrmions_deep_ritz_full.official_b1_pareto_run --mode full
.venv/bin/python -m experiments.skyrmions_deep_ritz_full.official_b1_pareto_run --mode cross
.venv/bin/python -m experiments.skyrmions_deep_ritz_full.official_b1_pareto_run --mode freeze-selection
.venv/bin/python -m experiments.skyrmions_deep_ritz_full.official_b1_pareto_run --mode generate-validation
.venv/bin/python -m experiments.skyrmions_deep_ritz_full.official_b1_pareto_run --mode validate
.venv/bin/python -m experiments.skyrmions_deep_ritz_full.official_b1_pareto_run --mode report
```

These commands are resumable. Existing artifacts are accepted only after their
upstream seals and hashes verify; authoritative completed artifacts are never
silently overwritten.
