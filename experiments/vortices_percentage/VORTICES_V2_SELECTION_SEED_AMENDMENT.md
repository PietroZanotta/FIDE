# Prospective Vortices V2 local-proposal seed amendment

## Status and scope

This is an additive prospective amendment to the frozen Vortices V2 selection
contract. It resolves only the missing RNG schedule for Tangent and Full local
proposal clouds. It does not change any reference, bandwidth, observation,
objective, geometry, optimizer budget, proposal count, proposal scale,
candidate gate, nesting rule, or validation rule.

The omission was detected after the single shared selection bank had been
generated but before any Population, Law, Tangent, or Full candidate objective
was evaluated. No scientific selection outcome was inspected before this
amendment was fixed.

The machine-readable authority is
[`VORTICES_V2_SELECTION_SEED_SCHEDULE.json`](VORTICES_V2_SELECTION_SEED_SCHEDULE.json).

## Frozen derivation

Let `root = 310000201`, the already-frozen optimizer root seed. Allowances and
Full local-search rounds use their zero-based positions in the already-frozen
ascending orders.

```text
tangent_seed(allowance_index)
  = root + 1000 + allowance_index

full_seed(allowance_index, round_index)
  = root + 2000 + 10 * allowance_index + round_index
```

The resulting literal seeds are:

| Allowance | Tangent | Full round 1 | Full round 2 | Full round 3 |
|---:|---:|---:|---:|---:|
| 0.5% | 310001201 | 310002201 | 310002202 | 310002203 |
| 1% | 310001202 | 310002211 | 310002212 | 310002213 |
| 2% | 310001203 | 310002221 | 310002222 | 310002223 |
| 3% | 310001204 | 310002231 | 310002232 | 310002233 |
| 4% | 310001205 | 310002241 | 310002242 | 310002243 |
| 5% | 310001206 | 310002251 | 310002252 | 310002253 |

Each seed is passed once to the frozen `deterministic_local_cloud` helper for
the corresponding allowance or allowance/round. The helper processes centers
in the frozen declared order and consumes one NumPy `default_rng` stream across
those centers. Candidate canonicalization and duplicate merging then follow
the original frozen contract.

## Existing bank and receipts

The existing shared selection bank remains immutable and is not regenerated:

```text
namespace = 410000101
trials = 128
SHA-256 = 1096a255beffa781ee5a9bec881a2778b11f3bf5b8674389d7120180f5280d3b
```

The original common-bandwidth and bank receipts retain their original base
selection-config hash. This amendment is an additive authority because editing
those immutable historical receipts would erase the chronology of the repair.

Validation namespace `410000102` remains unused.
