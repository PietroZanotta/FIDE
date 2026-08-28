# Published authoritative output bundle

This directory is the GitHub-visible output bundle for the robust
active-nematic Pareto experiment. It contains:

- all six allowance-specific `risk_*pct/result.json` receipts, including the
  frozen selection candidates and post-freeze validation results;
- the finalized Pareto tables, certification diagnostic, and report figure;
- the exact endpoint-only reference checkpoints and reference particle banks
  for all three reference seeds;
- the extracted two-species defect bank, selection/validation common-random-
  number banks, configurations, run-view manifest, and integrity manifests.

`frozen_inputs/physical_bank.npz` is intentionally not versioned. It is a
176 MB derived Q-tensor field bank, larger than GitHub's 100 MB ordinary-file
limit. It is not needed once `two_species_defect_bank.npz` has been frozen for
the authoritative Pareto calculation. It can be regenerated deterministically
from the versioned code, effective/source configuration, seeds, and solver
revision:

```bash
.venv/bin/python experiments/active_nematic_unbalance_percentage/run.py physical-bank \
  --output-dir experiments/active_nematic_unbalance_percentage/outputs/source
.venv/bin/python experiments/active_nematic_unbalance_percentage/run.py defects \
  --output-dir experiments/active_nematic_unbalance_percentage/outputs/source
```

The complete robust sweep and independent finalization commands are documented
in the experiment README. Regeneration is computationally expensive; the
published receipts are the immutable record used by the read-only evaluators.
