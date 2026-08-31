# Vortices V2 reference-seed provenance and prospective freeze

**Audit date:** 2026-08-30  
**Repository HEAD during audit:** `1e65587684857a84e98fda181fe472d62f46f7a8`

## Decision

The historical proposal `20260815, 20260816, 20260817` is ineligible. All
three values already produced inspected Vortices scientific results, and
`20260815` additionally generated the production V1 reference used throughout
the reflected V2 numerical-development replay.

The prospectively frozen reference-training seeds are instead

```text
310000101, 310000102, 310000103.
```

They are three consecutive values from a new range chosen without training a
model or inspecting a reference, sensor, risk, action, or validation outcome.

## Seed-by-seed provenance

| Seed | Where used | Purpose | Scientific outputs inspected? | Eligible as fresh V2 reference? | Reason |
|---:|:---|:---|:---:|:---:|:---|
| `20260815` | `experiments/vortices_percentage/config.json`; V1 run, corrected Pareto, validation and confirmatory artifacts; V2 development manifests | V1 production reference-training seed and the historical reference behind V2 numerical development | YES | NO | Previously trained; its sensor/action outcomes and V1 failure were inspected |
| `20260816` | `outputs/reference_seed_sensitivity/reference_seed_20260816/` and `reference_seed_sensitivity.md` | Historical full-budget learned-reference sensitivity replicate | YES | NO | Produced inspected Law/Full geometries, actions and reductions |
| `20260817` | `outputs/reference_seed_sensitivity/reference_seed_20260817/` and `reference_seed_sensitivity.md` | Historical full-budget learned-reference sensitivity replicate | YES | NO | Produced inspected Law/Full geometries, actions and reductions |
| `310000101` | Prospective freeze files only | Frozen V2 reference replicate 1 | NO | YES | No pre-freeze current-tree or relevant-history occurrence; fixed before training |
| `310000102` | Prospective freeze files only | Frozen V2 reference replicate 2 | NO | YES | No pre-freeze current-tree or relevant-history occurrence; fixed before training |
| `310000103` | Prospective freeze files only | Frozen V2 reference replicate 3 | NO | YES | No pre-freeze current-tree or relevant-history occurrence; fixed before training |

The common physical endpoint dataset is not a reference replicate. Its
generation seed is `20262816 = 20260815 + 2001`, and its SHA-256 is
`ad4006927e268c52f621c16c773f0600d803370bd21fb5e0816d82a70dbdfbba`.
It remains common across the three new trainings. Each training seed controls
network initialization and all training-batch randomness. Each replicate has
an independent deterministic rollout initial-particle seed equal to its
training seed plus `3001`:

```text
310003102, 310003103, 310003104.
```

## Audit coverage

The audit searched:

- the complete current worktree, including ignored and archived Vortices text
  artifacts, configs, scripts, Markdown, JSON, and CSV files;
- `experiments/vortices_percentage/`;
- `experiments/vortices_percentage_v2/`;
- the older `experiments/vortices_prospective/` lineage to avoid accidental
  reuse from adjacent Vortices work;
- the repository's `testing/FIDE/` mirrors; and
- relevant Git history with exact-number regular expressions and `git log -G`.

The historical audit located only reference-training seeds `20260815`,
`20260816`, and `20260817` in Vortices V1/V2 and its reference sensitivity
study. Git history records the base seed by commit
`9250d083890fbc5b9938d46210b733b491a849f4` and later Vortices updates; the
reference-sensitivity document is present at
`aaffc4d5ed778f885807703eac8c2104148c971b` and the current HEAD.

Before these freeze files were written, exact searches for `310000101`,
`310000102`, and `310000103` returned no occurrence in the current repository
or relevant Vortices Git history. After freezing, the preflight permits those
values only in the prospective V2 config, protocol, provenance, manifest,
preflight/tests, and dry-run scaffolding. An occurrence in any historical
result or output tree fails closed.

## Replacement prohibition

The three seeds are a fixed set, not a pool. If any replicate fails the frozen
reference-only qualification rules, that failure remains part of the
experiment and the scientific workflow stops. A fourth seed may not replace it
after reference or downstream outcomes have been inspected.

