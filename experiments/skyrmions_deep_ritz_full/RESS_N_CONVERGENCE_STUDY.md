# Nested-N rESS Convergence Study

Date completed: 2026-08-25

Status: **complete, development-only, no official launch authorized**

Development interpretation: **`MIXED_N_AND_PROPOSAL_EFFECT`**

## Executive conclusion

The instability observed in the preceding fresh-bank and replicate-gate studies
is real, and it appears at the reference-bank rESS gate. Increasing the number
of samples in each reference bank helps materially: across the 64-candidate
panel, the median bank-to-bank standard deviation of minimum rESS falls from
`0.00909` at N=8192 to `0.00545` at N=65536, a 40.0% reduction.

However, larger N does not make the unchanged `rESS >= 0.05` decision stable.
The final N=32768 to N=65536 transition still changes 28.0% of all pair-level
decisions, including 29.5% for the 55 high-pass candidates and 31.3% for Law's
16 pair decisions. Law converges near the boundary rather than clearly above
it: its N=65536 median is `0.05148` and p10 is `0.04576`.

The controlling physical-time region also remains structurally special. At
N=65536, node 7 controls 74.5% of the high-pass trajectories. Its median
top-1% projected-weight mass remains about 29.8%, its median calibrated lambda
norm remains about 89.7, and its median rESS is only `0.05658`, compared with
`0.49575` over the other nodes. Increasing N reduces the largest single weight,
as expected when more samples are available, but does not disperse the
aggregate top-1% mass or remove the strong exponential tilt.

The result is therefore neither “the old banks were merely too small” nor “N
does nothing.” It is a mixed effect: finite-N variability shrinks, while the
reference proposal and the low-risk targets retain borderline overlap at the
controlling time region.

No Tangent optimization, Full K/f assembly, K=280 eigensolve, Deep Ritz solve,
validation access, selection freeze, or official protocol was performed.

## Why Deep Ritz appeared to work before

Deep Ritz did not establish that the upstream support decision was stable over
independent reference-bank realizations. It was a downstream optimization and
audit route applied to a small set admitted under the bank structure of its
then-current protocol. A design can optimize successfully on one favorable
finite reference realization without demonstrating that its empirical rESS
will remain above 0.05 on another realization.

The later Galerkin selection work asks a deliberately stronger question. It
uses independent screen and audit banks and requires the unchanged support
criteria on both. The earlier v2 work already exposed the difference: the four
promoted 0.5% starts had screen/search rESS values around `0.0540` to `0.0672`,
but their independent audit rESS values were only `0.04270` to `0.04838`.
Projection residual, covariance, and forcing checks passed; independent audit
rESS alone rejected them before Tangent or Full work began.

The 32-pair fresh-bank study then measured this cross-bank variability over a
large frozen candidate pool instead of relying on a designated stage bank. The
present nested-N study goes one step further: it separates within-realization
sample-size behavior from variation between independent master realizations.
Thus, the apparent historical contrast is primarily a difference in what was
measured, not evidence that Deep Ritz used a superior support estimator.

## Prospectively frozen design

All source hashes, candidate identities, control-selection rules, bank seeds,
nested sample sizes, thresholds, and interpretation rules were sealed before
any master bank was generated.

### Candidate panel

The panel contains exactly 64 fixed geometries:

- Law (`candidate_000`), exactly once;
- all 55 non-Law candidates inside the exact 0.5% risk ceiling that passed at
  least 24 of 32 complete fresh-bank pairs;
- four symmetry-aware max-min controls from the 16–23/32 pass-count stratum;
- four symmetry-aware max-min controls from the 0–15/32 stratum.

The controls were selected algorithmically before seeing any nested-N result:

| Candidate | Role | Old pair passes /32 | Fixed risk | Law-relative increase |
|---|---|---:|---:|---:|
| `coverage_1671` | middle | 23 | 5.212176495 | 0.4941% |
| `coverage_0785` | middle | 22 | 5.211350604 | 0.4782% |
| `coverage_2827` | middle | 17 | 5.202376511 | 0.3052% |
| `coverage_3268` | middle | 16 | 5.210869316 | 0.4689% |
| `coverage_3334` | lower | 15 | 5.209216527 | 0.4370% |
| `coverage_3991` | lower | 0 | 5.130251237 | -1.0855% |
| `coverage_1077` | lower | 0 | 5.210292983 | 0.4578% |
| `coverage_1009` | lower | 0 | 5.195644620 | 0.1754% |

No candidate generation or geometry refinement occurred after the panel seal.

### Master banks and nesting

The experiment generated 16 independent master pairs, each containing roles A
and B. All 32 unique seeds were derived from the new development namespace and
serialized before generation. Every master bank contains 65,536 complete
float64 samples.

The N ladder is exactly:

```text
8192 -> 16384 -> 32768 -> 65536
```

Each level is the corresponding deterministic prefix of the same master bank.
Base weights are sliced with the prefix and renormalized independently at each
physical time. No independent per-N bank is generated, so adjacent-N deltas
measure nested within-realization changes.

The bank inventory contains 32 banks totaling 14,176,830,784 bytes. All 32
initial-state hashes are distinct. Bank generation took 2,150.90 seconds.

### Scientific evaluator

For every candidate, bank, and N, the exact existing empirical I-projection was
recomputed in float64 through the candidate-batched native Tesseract backend.
The fixed gates remained:

| Quantity | Threshold |
|---|---:|
| Minimum rESS | `0.05` |
| Projection residual | `2e-6` |
| Forcing mean | `2e-7` |
| Covariance condition | `1e10` |

The evaluator retains the complete 13-node trajectories for rESS, lambda norm,
maximum normalized projected weight, top-1% weight mass, sum of squared
weights, empirical `D2 = -log(rESS)`, covariance eigenvalues and condition,
projection residual, and forcing mean. The controlling index is the actual
argmin of the stored rESS trajectory.

A pair passes only when both A and B pass the unchanged individual-bank
support checks. A and B rESS values are never averaged into a new gate.

## Primary result 1: Law convergence

| N | Bank passes /32 | Pair passes /16 | Minimum | p10 | Median | p90 | Maximum | SD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 24 | 8 | 0.03210 | 0.04500 | 0.05682 | 0.06512 | 0.06983 | 0.00880 |
| 16384 | 22 | 7 | 0.03075 | 0.04291 | 0.05435 | 0.06095 | 0.06914 | 0.00786 |
| 32768 | 19 | 6 | 0.03411 | 0.04209 | 0.05207 | 0.05823 | 0.06150 | 0.00648 |
| 65536 | 20 | 5 | 0.03705 | 0.04576 | 0.05148 | 0.05711 | 0.06065 | 0.00543 |

Law's distribution narrows with N, but its center moves toward the gate rather
than clearly above it. Its pair-pass count falls from 8/16 to 5/16. This is not
evidence that increasing N causes rESS to decrease monotonically; finite nested
prefixes need not move monotonically. It is evidence that N=8192 was not simply
producing conservative false failures that larger banks repair.

The controlling index remains in the same physical region. Law is controlled
by node 7 on 25, 25, 22, and 20 of 32 banks as N increases; the remaining
counts are node 8.

## Primary result 2: high-pass panel

| N | Median bank-pass fraction | Median pair-pass fraction | >=8/16 | >=12/16 | >=14/16 | 16/16 |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 0.84375 | 0.68750 | 55 | 6 | 1 | 0 |
| 16384 | 0.81250 | 0.68750 | 55 | 13 | 1 | 0 |
| 32768 | 0.81250 | 0.68750 | 55 | 26 | 13 | 0 |
| 65536 | 0.78125 | 0.62500 | 50 | 11 | 0 | 0 |

N=32768 looks favorable under several pass-count summaries, but that apparent
improvement is not retained at N=65536. It would therefore be unsafe to stop at
N=32768 and declare convergence.

At N=65536, the distribution across candidates of their 32-bank median rESS
has median `0.05565`, while the distribution of candidate-specific bank p10
rESS has median `0.04862`. Thus the typical candidate center is above 0.05,
but its lower bank-realization tail remains below the gate.

No high-pass candidate passes all 16 pairs at any N. At N=65536, none passes
14/16 pairs and only 11 pass at least 12/16.

## Primary result 3: within-realization convergence

Median signed changes remain small and near zero; importantly, no monotonicity
assumption was imposed. Absolute changes for the high-pass panel are:

| Transition | Median signed delta | Median absolute delta | p90 absolute delta | Maximum absolute delta |
|---|---:|---:|---:|---:|
| 8192 -> 16384 | -0.0008440 | 0.0033883 | 0.0106827 | 0.0213727 |
| 16384 -> 32768 | -0.0004965 | 0.0038941 | 0.0126603 | 0.0227108 |
| 32768 -> 65536 | +0.0002683 | 0.0029901 | 0.0074239 | 0.0205968 |

The last transition is smaller in median and p90 magnitude than the first, but
changes of several thousandths remain scientifically large relative to a hard
0.05 boundary. The maximum last-transition change remains about 0.0206.

Across all 64 candidates, median between-bank standard deviation by N is:

| N | Median candidate bank SD |
|---:|---:|
| 8192 | 0.009085 |
| 16384 | 0.007690 |
| 32768 | 0.006314 |
| 65536 | 0.005455 |

The N=65536/N=8192 ratio is `0.6004`, satisfying the prospectively frozen
material-variance-reduction condition.

## Primary result 4: threshold-decision stability

| Transition | Group | Fail -> pass | Pass -> fail | Disagreements | Rate |
|---|---|---:|---:|---:|---:|
| 8192 -> 16384 | all | 119 | 121 | 240/1024 | 23.44% |
| 8192 -> 16384 | high-pass | 106 | 107 | 213/880 | 24.20% |
| 16384 -> 32768 | all | 211 | 163 | 374/1024 | 36.52% |
| 16384 -> 32768 | high-pass | 194 | 148 | 342/880 | 38.86% |
| 32768 -> 65536 | all | 89 | 198 | 287/1024 | 28.03% |
| 32768 -> 65536 | high-pass | 81 | 179 | 260/880 | 29.55% |

The final flip rate is above the frozen 10% stabilization criterion and is
higher, not lower, than the first transition's 23.44%. For Law specifically,
5/16 pair decisions flip in each of the last two transitions.

This is the most direct reason that a larger N alone cannot yet be used to
justify an official gate architecture. Estimator variance shrinks, but the
remaining rESS mass is close enough to 0.05 that modest nested changes continue
to change many Boolean outcomes.

## Primary result 5: node 7 and weight concentration

High-pass node-7 diagnostics are:

| N | Node-7 controlling fraction | Median rESS | Median lambda norm | Median max weight | Median top-1% mass | Median D2 |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 80.06% | 0.06157 | 90.160 | 0.016698 | 0.29519 | 2.7875 |
| 16384 | 79.43% | 0.05941 | 90.011 | 0.010447 | 0.29761 | 2.8233 |
| 32768 | 71.31% | 0.05844 | 89.941 | 0.006678 | 0.29731 | 2.8398 |
| 65536 | 74.49% | 0.05658 | 89.726 | 0.004553 | 0.29760 | 2.8722 |

For comparison, the N=65536 median rESS across the other nodes is `0.49575`.
The controlling region therefore does not migrate away as N grows.

The maximum individual projected weight declines substantially, which is a
normal benefit of a larger empirical sample. The aggregate top-1% mass does
not decline; it remains about 0.30. Lambda norm also remains nearly unchanged,
and D2 increases slightly as node-7 rESS moves down. Descriptively, larger N
reduces some sampling granularity while leaving the strong-tilt/overlap geometry
at node 7 intact.

## Frozen interpretation

The prospective checks evaluate as follows:

| Check | Result |
|---|---|
| Material variance reduction | yes |
| Adjacent-N threshold decisions stabilized | no |
| Law N=65536 median | 0.05148, in boundary band |
| High-pass median candidate p10 at N=65536 | 0.04862, in boundary band |
| Stable adequate support margin | no |
| Stable inadequate support margin | no |
| Node 7 remains dominant | yes, 74.49% |

The sealed development interpretation is:

```text
MIXED_N_AND_PROPOSAL_EFFECT
```

Increasing N materially improves the continuous estimator's dispersion, but
the unchanged threshold remains embedded in the resulting Law/high-pass
distributions and node 7 retains a concentrated exponential tilt. More samples
do not produce a sufficiently stable support decision.

## Recommended next scientific step

Do not launch an official Full run, do not select N=32768 based on its favorable
intermediate pass counts, and do not respond by adding more replicate votes
under the same unresolved support semantics.

The next action should be a separately specified development study of the
frozen reference proposal/sampling strategy around the node-7 physical-time
region. That study should distinguish whether the proposal underrepresents the
tilted target population or whether the fixed 0.05 gate is intrinsically close
to the scientifically relevant low-risk population. It must not retrospectively
alter this study's proposal, threshold, panel, or seeds.

## Performance

The study evaluated exactly `64 * 32 * 4 = 8192` candidate-bank-N trajectories.
After initial compilation, the candidate-batched native path scaled close to
linearly with N.

| N | Total projection and diagnostic time over 32 banks |
|---:|---:|
| 8192 | 17.49 s |
| 16384 | 22.03 s |
| 32768 | 37.09 s |
| 65536 | 67.50 s |

Total scientific evaluation time was 144.11 seconds. Summary construction took
0.34 seconds. Master-bank generation, not I-projection, dominated wall time.

The output namespace contains 128 compressed per-bank/N result files, 16 pair
inventories, 32 master banks, and about 14.19 GB overall. Results are resumable
and sealed per pair.

## Verification

- The pre-freeze evaluator reproduced prior sealed rESS values with maximum
  discrepancy `4.93e-16`.
- The candidate-batched native projection regression suite passed.
- All expected fresh-bank and replicate-gate source hashes matched.
- The candidate panel is exactly 64 and contains every required high-pass row.
- Candidate panel mtime precedes the manifest and all banks.
- The manifest contains 16 pairs, 32 unique seeds, and the exact N ladder.
- Every master-bank shape, float64 dtype, base-weight normalization, and hash
  was verified.
- All nested prefix tests passed.
- The controlling time index equals the stored rESS trajectory argmin.
- All 16 pair inventories and the final summary inventory were hash-verified.
- A sealed-summary cache reread reproduced the summary and inventory hashes.
- 60/60 focused convergence, preflight, fresh-bank, and native-projection tests
  passed.
- `git diff --check` passed before launch and is rerun at final handoff.

## Principal artifacts and hashes

Output root:

`experiments/skyrmions_deep_ritz_full/outputs/skyrmion_galerkin_dev_ress_n_convergence_v1/`

| Artifact | SHA-256 |
|---|---|
| `source_seal.json` | `c812abeaa6fdda3242ab175571f9b77ff2cf68372028aa3689571f8cf95b74db` |
| `candidate_panel.json` | `f2a6437899383072634c4c2c596e35c49275b6fb47ee9b05a3425d35a81a0189` |
| `master_bank_manifest.json` | `ca6fdeec773408a704c262c244d0b8783522dc0b70435dc205cd0d6cd6ea11fe` |
| `master_bank_inventory.json` | `79d26a677dd29637766112a54142d01c104714b88466e91853b66e095146cd13` |
| `law_convergence.json` | `1c7a2c7acd27166012a3889eaa5a656e91342485a0f1e0d3e15da72c1fc45d88` |
| `candidate_convergence.json` | `84c6b2bcbc8e2b9147b95be2be4640654773c5d3510d5601ec11a7e7cb180ec5` |
| `time7_diagnostics.json` | `b7d79707408ff48b47d04e6d8424151e01ada78f61f6314d1ca5c36544934d73` |
| `threshold_flip_summary.json` | `7b33ec79a901cca8e0a4bf14efb30b8745829539ce5a21be6d826721030c8232` |
| `summary.json` | `295975b93e8b4db11007de3f0adf1afd5c1b06df7efea194d9251cbefd0c3a2a` |
| `inventory.json` | `2d72c01654873c0ea5bb98500b227851e71d8e657e98fbeb9a20c32765ab0c4f` |

## Firewall statement

This study remains development-only. It created no official protocol, no
official bank, no candidate generation, no Tangent run, no Full branch, no K/f
assembly, no eigensolve, no Deep Ritz run, no selection hash, and no validation
access. Existing v2/v3, coverage, fresh-bank, and replicate-gate artifacts were
read only and retain their expected hashes.
