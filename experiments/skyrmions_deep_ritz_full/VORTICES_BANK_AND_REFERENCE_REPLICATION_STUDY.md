# Vortices Bank and Reference-Seed Design: Implications for Skyrmions

Date: 2026-08-26

Status: **development design study; skyrmion execution remains paused**

## Executive conclusion

The vortices reference-seed audit provides a useful controlled-experiment
pattern for the skyrmion problem:

1. freeze one physical truth bank and one endpoint-training dataset;
2. train several reference models while changing only the training seed;
3. roll every model from exactly the same initial reference particles;
4. reuse exactly the same selection and held-out evaluation randomness;
5. compare downstream scientific behavior, not training loss alone;
6. preserve each run in an isolated output namespace and compare hashes.

The three skyrmion retrainings completed on 2026-08-26 already implement the
first two items. They used one newly generated 12,000-configuration endpoint
dataset and three training seeds, with no other training-configuration change.
The computation was stopped after all three 6,000-step trainings. No new
reference rollout bank or rESS evaluation was completed.

A literal copy of the vortices setup would next create one matched reference
bank per model. That would isolate model-seed effects on one finite bank, but it
would not adequately address the skyrmion failure. The skyrmion evidence shows
that the `rESS >= 0.05` decision changes materially across reference-bank
realizations even at N=65,536. Therefore the appropriate replication is a
**crossed model-by-bank experiment**:

```text
three fresh reference models
    x the same 16 previously sealed A/B master-bank pairs
    x the same frozen 64-candidate panel
    x N = 65,536 and the unchanged support gates.
```

For each of the 32 initial-particle banks, all three models must receive the
same node-zero configurations. This retains the paired strength of the vortices
audit while measuring the source of instability that is specific to
skyrmions. The original production model can be included as a descriptive
baseline, but it must not be treated as a fourth exchangeable training-seed
replicate because it was trained on a different endpoint dataset.

Replicating the vortices control structure is therefore beneficial. Replicating
its single-reference-bank scope without the crossed bank ensemble would not be
sufficient.

## 1. The different meanings of "bank"

The repository uses the word "bank" for several scientifically different
objects. Keeping their roles separate is essential.

| Bank type | Contains | Scientific role |
|---|---|---|
| Physical truth bank | Samples from the known physical simulator over time | Oracle population, observations, and direct model diagnostics |
| Endpoint-training bank | Samples at `t=0` and `t=1` | Trains the endpoint-only learned reference |
| Reference rollout bank | Particles rolled through a learned reference flow, with velocities and base weights | Proposal population for empirical I-projection and action calculations |
| Selection observation bank | Frozen finite-particle indices and detector noise | Candidate optimization and selection only |
| Validation observation bank | Independent finite-particle indices and detector noise | Held-out evaluation after selection is frozen |
| Reference-bank replicate pair | Two independent reference rollout populations, conventionally A and B | Tests whether support decisions reproduce across finite proposal realizations |

The vortices reference-seed audit controlled the first five types. The later
skyrmion studies discovered a problem in the sixth type: support classification
depends too strongly on which finite reference rollout bank is used.

## 2. Current authoritative vortices status

The active vortices experiment is under `experiments/vortices_percentage/`.
Its current authoritative result is the corrected percentage-risk Pareto sweep
under `outputs/pareto/`, not the older single-run result under `outputs/run/`.

The current authoritative design uses:

- 50,000 physical truth particles at 21 time nodes;
- 50,000 endpoint-training particles;
- a learned endpoint-only reference with four hidden layers of width 128;
- 12,000 flow-matching steps and batch size 2,048;
- a 32,768-particle learned-reference rollout bank;
- 24 selection trials, of which the first 16 supply the Law trials;
- 64 independent validation trials;
- a corrected physical-density weighted-Poisson Full evaluator on a
  `128 x 64` grid at all 21 time nodes.

The corrected Full designs reduce validation Full action relative to the common
Law design by 59.17% to 70.18%, depending on risk allowance. This active result
is scientifically separate from the older three-seed audit described below.

## 3. How vortices constructs its banks

### 3.1 Physical truth bank

`ensure_truth_bank` in `experiment.py` fingerprints the truth configuration,
base seed, and time grid. It reuses a compatible cache or generates:

```text
seed = 20260815 + 1001
particles = 50,000
time nodes = 21
RK4 substeps per interval = 32
array shape = [21, 50000, 2]
```

Its SHA-256 in the audited runs is:

```text
d897ff7fc44c0b85d7bb5391c0cc25895b4301e9c2ce00184697a1899d853b5b
```

This bank is independent of the learned-reference training seed.

### 3.2 Endpoint-training bank

`ensure_reference_endpoints` uses a different fixed namespace:

```text
seed = 20260815 + 2001
particles = 50,000
endpoint rollout substeps = 512
x0 shape = [50000, 2]
x1 shape = [50000, 2]
```

It samples physical initial particles and rolls them through the known
double-gyre truth to `t=1`. The file is byte-identical across the three audited
reference seeds:

```text
ad4006927e268c52f621c16c773f0600d803370bd21fb5e0816d82a70dbdfbba
```

The saved rows are physically corresponding initial and final states. However,
the flow-matching objective does not exploit row identity: `sample_cfm_batch`
draws endpoint-zero and endpoint-one samples with independent random keys. Its
metadata records `independent_endpoint_pairing = True`. The skyrmion trainer
does the same by drawing independent `idx0` and `idx1` arrays.

Thus the endpoint data are physically generated, but the learned path is
identified from endpoint distributions, not physical Lagrangian pairs or
intermediate truth.

### 3.3 Learned reference

Vortices transforms the endpoint data to box-logit coordinates. The learned
latent flow is mapped back through a logistic diffeomorphism, keeping reference
particles inside the physical rectangle and preventing boundary-escape
artifacts.

The three audit configurations changed only:

```text
reference_training.seed = 20260815, 20260816, or 20260817.
```

Architecture, training budget, endpoint data, optimizer, bridge, and rollout
discretization were fixed. Each checkpoint stores the endpoint signature and
training signature, so an incompatible cache triggers retraining.

### 3.4 Learned-reference rollout bank

`ensure_reference_bank` uses:

```text
seed = 20260815 + 3001
particles = 32,768
time nodes = 21
reference RK4 substeps per interval = 16
```

For each model, it samples the same initial particles, rolls those particles
through that model, evaluates that model's velocity on its own trajectory, and
assigns uniform base weights. It saves:

```text
nodes:    [21, 32768, 2]
velocity: [21, 32768, 2]
weights:  [21, 32768]
```

The complete files differ because the learned trajectories differ. Direct
inspection confirms that their node-zero arrays are byte-identical, with common
node-zero SHA-256:

```text
f63c2ee26c7705025342e1862ec7e62bc87a9d4fe1102d94ad7910cd02255646
```

This is the core paired-control device: model differences are evaluated from
the same starting particles rather than independent Monte Carlo realizations.

### 3.5 Selection observation bank

The selection bank contains finite-observation randomness rather than learned
reference particles:

```text
sample_indices: [24, 9, 2000]
detector_z:      [24, 9, 4]
SHA-256: 0ae52680ba66f07e36e02a0d85d25847fc11dc2554fcb63f95cb4e7aa0636ef9
```

The first 16 trials are used for Law evaluation; all 24 are available for
action selection. The file is byte-identical across model seeds.

### 3.6 Validation observation bank

The independent validation bank has:

```text
sample_indices: [64, 9, 2000]
detector_z:      [64, 9, 4]
SHA-256: 63748a79d00bce58e6307f2070f29c480998de0a7c5c47b4fcb0788696dea894
```

Because all models use the same 64 validation trials, downstream differences
are paired. This removes much observation-trial variation from the comparison.

## 4. What the historical vortices audit tested

The audit asked:

> Given fixed endpoint data, truth, reference-bank initial particles,
> optimization design, selection randomness, and validation randomness, how
> much does the scientific conclusion change when only training randomness
> changes?

It did not test endpoint-dataset variability, reference-bank realization
variability, physical correctness of intermediate learned marginals, or fully
independent optimizer-basin discovery.

The historical results were:

| Seed | Law action | Full action | Full-vs-Law reduction |
|---:|---:|---:|---:|
| 20260815 | 3.3629 | 1.6634 | 50.54% |
| 20260816 | 3.4572 | 1.6567 | 52.08% |
| 20260817 | 3.4507 | 1.5823 | 54.15% |

The reduction range was 3.61 percentage points. Pairwise learned-reference
velocity normalized RMSE was 0.212 to 0.235, and final paired-position RMS was
0.052 to 0.060. The downstream conclusion was fairly stable even though the
learned fields were visibly different.

All three historical runs selected the same Full geometry. The candidate pool
included the production Full geometry as a provenance seed, so this was a clean
action comparison but not proof of independent discovery of the same basin.

## 5. Provenance issue found in this review

The preserved reference-seed summary still contains paths under the former
directory name `experiments/vortices/`. Its production entry points to an old
`outputs/run/result.json` that is absent from the current working tree. The
checked-out `experiments/vortices_percentage/outputs/run/result.json` is a
later/different receipt:

| Receipt | Reduction | First Full center |
|---|---:|---|
| Historical seed-audit production receipt | 50.54% | `(0.440933, 0.427542)` |
| Current `outputs/run/result.json` | 25.94% | `(1.050855, 0.419585)` |

The historical production receipt is recoverable at Git commit `c56f6087` and
exactly matches the preserved summary. The summary itself matches its Git
version byte-for-byte:

```text
bda90e9aa99942b2118aee9e831e06e84da47bd82fe477f0e2e401dccccc22f0
```

Therefore the historical conclusion is recoverable, but the current default
summary command is not self-contained: without explicit inputs it would resolve
the mutable current `outputs/run` receipt. The reproduction commands in
`reference_seed_sensitivity.md` also retain the old directory name.

A skyrmion replication should improve this discipline:

- use immutable, hash-addressed baseline receipts;
- copy or explicitly reference all compared receipts inside the study manifest;
- never resolve a baseline through a mutable generic `outputs/run` path;
- fail closed on any input-path or input-hash drift;
- retain a complete model/bank/result inventory.

The older audit also must not be cited as proof that the currently authoritative
corrected vortices Pareto curve is reference-seed robust. It demonstrates a
useful method and a historical single-run conclusion.

## 6. The skyrmion problem

The skyrmion issue is not merely potential model variation. The empirical
reference-proposal support decision is near a hard threshold and changes across
finite reference-bank realizations.

The completed nested-N study found:

| Quantity | Result |
|---|---:|
| Law median minimum rESS at N=65,536 | 0.05148 |
| Law p10 minimum rESS at N=65,536 | 0.04576 |
| Law pair passes at N=65,536 | 5/16 |
| High-pass median candidate p10 rESS | 0.04862 |
| High-pass candidates passing at least 12/16 pairs | 11/55 |
| High-pass candidates passing 14/16 pairs | 0/55 |
| Final N=32,768 to 65,536 decision-flip rate | 28.03% overall |
| Final decision-flip rate for the high-pass panel | 29.55% |
| Node-7 controlling fraction for the high-pass panel | 74.49% |
| Median top-1% projected-weight mass at node 7 | about 29.8% |

Increasing N reduced median between-bank standard deviation by about 40%, but
did not move the population safely away from `rESS = 0.05`. The current
interpretation is `MIXED_N_AND_PROPOSAL_EFFECT`.

Two explanations remain:

1. **Checkpoint/training explanation:** the original reference happened to
   learn a poor intermediate proposal near node 7; retraining may improve and
   stabilize overlap.
2. **Structural proposal explanation:** endpoint-only training does not
   identify the needed intermediate law, so retraining may reproduce the same
   insufficient overlap or produce seed-dependent paths.

## 7. Completed skyrmion retraining work

The frozen development namespace is:

```text
outputs/skyrmion_galerkin_dev_reference_retraining_ensemble_v1/
```

One shared endpoint dataset was generated:

```text
samples: 12,000 configurations
seed: 2026082601
SHA-256: a00cfe2b583dde69012c541b320b547322ebed6630c68435f4adb5d02771c32a
```

One disjoint physical evaluation trajectory was generated:

```text
samples: 6,000 configurations at each of 13 nodes
seed: 2026082602
SHA-256: 5390abfb026d4991b5d95632b85aeb20441089e046e5aff764be4ed6b59d5ea1
```

All models used width 64, three hidden layers, 6,000 steps, batch size 512,
learning rate `8e-4`, bridge noise 0.01, and otherwise identical settings.

| Model | Seed | Final loss | Time | Checkpoint SHA-256 |
|---|---:|---:|---:|---|
| `retrained_0` | 2026082611 | 4.412443 | 49.27 s | `03b4739e398ea59d1144b4eca3500517f407b0f1b6378ba7d72d951aa23e4c53` |
| `retrained_1` | 2026082612 | 4.263568 | 49.89 s | `da4b1aab0b96b5efedd89df3bed4e446b1d7530fa274a8785d6e00c67b3867d9` |
| `retrained_2` | 2026082613 | 4.337775 | 46.13 s | `4946e45a416173f950dbd871d9b62cb45cbfdf079e40bad2fc0c1981b5c8ca2c` |

The distinct hashes confirm three independent models. Training loss does not
establish which, if any, has correct intermediate physical behavior.

The paused runner planned one 65,536-particle matched bank for the original and
three new models, fixed-CRN held-out loss, truth-moment errors at nodes 6--8,
and rESS evaluation of the frozen 64-candidate panel. Execution was interrupted
after training. A first rollout began in memory, but no bank artifact was
written.

## 8. Direct comparison

| Design element | Vortices audit | Paused skyrmion study | Assessment |
|---|---|---|---|
| Shared endpoint data | 50,000 | 12,000 | Replicated |
| Only training seed changes | Yes | Yes | Replicated |
| Independent physical truth | Yes | Yes | Replicated |
| Same rollout initial particles across models | 32,768 | Planned 65,536 | Correct, not executed |
| Same downstream randomness | Shared selection and validation banks | Fixed targets and CRNs | Development analogue |
| Pairwise trajectory/velocity metrics | Yes | Not yet summarized | Add to companion audit |
| Full downstream optimization per model | Yes | Fixed support panel only | Intentionally narrower |
| Multiple rollout-bank realizations per model | No | No in current protocol | Required for skyrmions |
| Immutable baseline receipt | Historical path drift | Production hash sealed | Skyrmion must remain hash-addressed |

## 9. What should be copied directly

### Shared non-model inputs

All model comparisons must use byte-identical endpoint data, physical truth,
candidate panel, thresholds, time grid, initial configurations, and evaluation
randomness.

### Matched initial particles

Every model must be rolled from the same initial array for each bank identity.
Independent initial samples would mix model effects with Monte Carlo effects.

### Paired downstream comparisons

The primary unit should be a paired difference on the same candidate and bank:

```text
Delta rESS(i,j,c,b) = rESS(i,c,b) - rESS(j,c,b).
```

Use the same pairing for support Booleans, trajectory RMS, velocity normalized
RMSE, truth errors, and any later action comparison.

### Isolated outputs and scientific outcomes

Every result must name its checkpoint hash and initial-state hash. Conclusions
must depend on rESS/support and physical-path behavior, not training loss alone.

## 10. What must be adapted for skyrmions

### Cross models with the 16 sealed A/B master pairs

Vortices used one reference rollout bank per model. Skyrmions should instead
reuse the 16 sealed nested-N A/B master pairs, providing 32 initial arrays of
65,536 particles. For every array:

1. read its existing node-zero configurations;
2. roll all three fresh models from that exact array;
3. evaluate all 64 candidates with unchanged I-projection and support gates;
4. save result arrays and source hashes;
5. form A/B pair decisions separately for each model.

The production-flow N=65,536 results already exist for the same panel and bank
pairs. They can serve as a descriptive historical baseline after hash
verification; no production reroll is needed.

The primary datasets become:

```text
3 fresh models x 32 banks x 64 candidates x 13 nodes
3 fresh models x 16 A/B pair decisions x 64 candidates
```

### Do not treat the original model as a fourth replicate

The original checkpoint used older endpoint data. Comparisons among the three
fresh models estimate conditional training-seed sensitivity. Original-versus-
fresh differences combine endpoint-data and seed effects.

### Retain direct physical diagnostics

Three endpoint-only models may agree and still be jointly wrong. Use the sealed
6,000-sample physical trajectory at all 13 nodes, reporting candidate-feature
moment error and at least one prospectively fixed candidate-independent marginal
discrepancy. Nodes 6, 7, and 8 require special reporting.

### Keep support stability separate from optimizer discovery

The fixed 64-candidate panel tests support classification. A de-novo search asks
whether models drive optimization into the same basin and requires a separate,
prospectively frozen protocol.

## 11. Recommended companion protocol

The current one-bank retraining protocol is sealed and should not be edited.
Create a separate development-only companion protocol that consumes its three
checkpoints.

### Source freeze

Freeze and hash:

- all fresh checkpoints and training receipts;
- regenerated endpoint and physical-evaluation data;
- the frozen 64-candidate panel;
- the nested-N master manifest and all 32 node-zero hashes;
- original N=65,536 result receipts used as baselines;
- configuration, analysis source, and interpretation rules.

### Matched rollout

For each of 32 initial states and each fresh model:

- roll 65,536 configurations over 13 nodes;
- use 14 RK4 substeps per interval;
- evaluate velocity on the model's own trajectory;
- assign normalized uniform base weights;
- confirm node zero is byte-identical across the three models;
- record checkpoint, initial-state, trajectory, velocity, and result hashes.

Streaming bank-by-bank can avoid approximately 42 GB of persistent duplicate
model-bank storage. Retain the compact scientific result arrays and enough
hash-addressed source information to reproduce each rollout. If caches are
needed for resumability, inventory them explicitly.

### Unchanged support evaluation

Retain complete 13-node rESS, multiplier norm, projected-weight concentration,
projection residual, forcing residual, covariance condition, controlling node,
all validity Booleans, and combined support. Do not average A and B into a new
gate: a pair passes only when both individual banks pass.

### Required summaries

Report:

1. Law bank and pair passes for each model;
2. high-panel candidate bank and pair pass fractions;
3. paired model-to-model and fresh-to-original rESS deltas;
4. support-decision disagreement matrices;
5. model-by-bank interactions;
6. controlling-node frequencies;
7. node-7 rESS, lambda norm, maximum weight, and top-1% mass;
8. trajectory RMS, final-position RMS, and velocity normalized RMSE;
9. held-out physical moment and marginal errors;
10. fixed-CRN endpoint loss as a secondary diagnostic.

### Interpretation matrix

| Outcome | Interpretation |
|---|---|
| Fresh models agree and all remain cross-bank unstable | Structural endpoint-only proposal problem is reproduced |
| Fresh models agree and all stabilize support with adequate physical fit | Original checkpoint/data realization likely contributed materially |
| Fresh models disagree on the same banks | Learned proposal is training-seed sensitive |
| Physical fit improves but support stays near 0.05 | Better approximation does not remove the overlap boundary |
| Support improves while physical fit degrades | Apparent rESS repair is scientifically untrustworthy |
| Models agree but all disagree with truth | Shared model misspecification remains despite seed stability |

Numeric readiness thresholds must be frozen before rollout or evaluation.

## 12. Benefits

The crossed design separates:

- training-seed variability across models on the same bank;
- bank-realization variability across banks for the same model;
- model-by-bank interaction;
- historical data/checkpoint differences from the original model.

Matched initial particles provide high statistical efficiency, exactly as
shared validation trials did in vortices. The experiment directly tests whether
retraining consistently moves node-7 overlap, and it requires no further model
training.

## 13. Remaining limitations

- The three models share one endpoint dataset, so they do not measure endpoint-
  dataset variability.
- Endpoint-only training remains non-identifying at intermediate times.
- The known 16 master pairs are ideal for diagnosis but cannot serve as unseen
  official confirmation banks.
- A fixed panel does not test de-novo optimization-basin discovery.
- Three seeds reveal large instability but provide limited resolution for a
  full seed distribution.

## 14. Recommended staged decision

1. Do not resume the current single-bank run unchanged yet.
2. Preserve its sealed protocol, regenerated data, and checkpoints.
3. Create the crossed-bank companion protocol using the 16 sealed A/B pairs.
4. Run only development support and physical-path diagnostics.
5. If support stabilizes with improved physical fit, specify fresh confirmation
   pairs before changing any production checkpoint.
6. If support remains unstable, return to proposal/overlap design near node 7.
7. If models disagree, improve training identification or stability before any
   downstream official run.

## 15. Final assessment

Replicating vortices is beneficial for its shared data, matched particles,
paired evaluation, downstream metrics, and isolated artifacts. It converts the
retraining hypothesis into a controlled experiment.

The literal vortices scope is one dimension short: it compares model seeds on
one reference population, while skyrmion bank realization already controls the
`rESS >= 0.05` decision. The correct replication is therefore the matched
three-model by 16-pair diagnostic described here.

The completed skyrmion checkpoints should be retained. Their next meaningful
use is this crossed development diagnostic, not an isolated one-bank conclusion
or an immediate official run.

## 16. Principal files

### Vortices

- `experiments/vortices_percentage/experiment.py`
- `experiments/vortices_percentage/bounded_reference.py`
- `src/mfsi/flow_matching.py`
- `experiments/vortices_percentage/run.py`
- `experiments/vortices_percentage/summarize_reference_seeds.py`
- `experiments/vortices_percentage/reference_seed_sensitivity.md`
- `experiments/vortices_percentage/README.md`
- `experiments/vortices_percentage/outputs/reference_seed_sensitivity/summary.json`
- `experiments/vortices_percentage/outputs/pareto/frozen_inputs/manifest.json`

### Skyrmions

- `experiments/skyrmions_deep_ritz_full/reference.py`
- `experiments/skyrmions_deep_ritz_full/reference_retraining.py`
- `experiments/skyrmions_deep_ritz_full/reference_retraining_run.py`
- `experiments/skyrmions_deep_ritz_full/RESS_N_CONVERGENCE_STUDY.md`
- `experiments/skyrmions_deep_ritz_full/FRESH_BANK_ROBUSTNESS_STUDY.md`
- `experiments/skyrmions_deep_ritz_full/REPLICATE_GATE_PREFLIGHT_V2_REPORT.md`
- `experiments/skyrmions_deep_ritz_full/outputs/skyrmion_galerkin_dev_reference_retraining_ensemble_v1/protocol.json`
- `experiments/skyrmions_deep_ritz_full/outputs/skyrmion_galerkin_dev_ress_n_convergence_v1/summary.json`

## Firewall statement

This review was read-only with respect to scientific artifacts. It did not
resume reference-bank generation, evaluate a retrained checkpoint, modify the
production checkpoint, generate official data, access validation, run Tangent
or Full optimization, assemble K/f, run an eigensolve, run Deep Ritz, or freeze
a selection. The only new artifact is this Markdown design study.
