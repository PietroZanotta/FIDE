# Paper experiment entry points

| Study | Paper run | Fast validation |
|---|---|---|
| Experiment A | `./scripts/run_example_a.sh --backend jax` | `./.venv/bin/python validate_pipeline.py` |
| Experiment B | `./scripts/run_example_b.sh --backend jax` | `--quick` |
| Experiment-B crossed seeds | `./scripts/run_multiseed_b.sh --backend jax ...` | resumable cached cells |
| Scalar Level 2 | `./scripts/run_level2.sh --backend jax` | `--quick` |
| Advanced Level-2 suite | `./scripts/run_level2_suite.sh --backend jax` | `--quick` |
| Paper-facing N=32 Level 2 | `./scripts/run_level2_paper_study.sh --backend jax` | `--quick` |
| Stage 3 | frozen driver `stage3_rollout_adaptation.py` | `validate_stage3_rollout_adaptation.py` |
| Stage 3B | frozen driver `stage3b_confirmatory.py` | `validate_stage3b_confirmatory.py` |
| Stage 4 | frozen driver `stage4_fiber_design.py` | `validate_stage4_fiber_design.py` |
| Stage 4B | frozen driver `stage4b_fiber_design_confirmatory.py` | `validate_stage4b_fiber_design.py` |

Run the complete JAX regression suite with:

```bash
./scripts/run_paper_jax_regression.sh
```

That command intentionally uses the frozen drivers for confirmatory reruns.
The backend-aware Stage 3/4 wrappers are for new backend-separated runs.
