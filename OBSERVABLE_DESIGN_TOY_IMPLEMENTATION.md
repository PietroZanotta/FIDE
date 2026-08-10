# Observable-design toy implementation note

Experiment D is additive: `example_b.py` and its stored results are unchanged.

The new implementation reuses these Experiment-B functions directly:

- endpoints and interpolant: `sample_ring`, `sample_four_lobes`, `sample_bridge`, `sample_bridge_times`;
- raw dictionary and Jacobian: `phi`, `jphi`;
- frozen flow-matched reference: `reference_velocity` and the packaged reference parameters in `results/example_b/learned_mfsi_example_b.npz`;
- empirical I-projection and implicit derivative: `mfsi_components.empirical_fiber_state` and `calibrate_empirical_implicit`;
- stable rank-aware solves: `mfsi_components._stable_cov_solve`;
- neural/optimization components: `init_mlp`, `mlp_apply`, AdamW/cosine schedule, `potential_grad`, and `ritz_state_loss`;
- rollout and evaluation conventions: `integrate_field`, the three-scale Experiment-B RBF bandwidth rule, `angular_features`, and empirical endpoint whitening.

The interface change is confined to supplying generic `ph=Phi_A(x)` and
`jphi_u=J_Phi_A(x)u` arrays to the existing calibration routine. The Deep-Ritz
integrand already consumes generic calibrated weights and forcing, so its
mathematics was not reproduced.

`observable_design_toy.py` defines the fixed design standardization, shared QR
row-Stiefel parameterization, INFO/CV/FIBER objectives, diagnostics, generic
tangent/safety correction, matched downstream training, and plotting.
`expD_observable_design_toy.py` owns seed separation, caching, crossed
evaluation, bootstrap contrasts, and artifacts. The JSON-compatible YAML config
predeclares smoke and confirmatory budgets.

The FIBER checkpoint gate rejects calibration residual above `1e-6`, covariance
rank below `R`, non-finite conditioning, or minimum interior ESS below `0.20`.
An infeasible optimizer cannot replace the initialization.

Run:

```bash
.venv/bin/python expD_observable_design_toy.py \
  --config configs/observable_design_toy.yaml \
  --phase smoke
```

For a one-seed plumbing run, add `--seed 20260810`. Confirmatory mode executes
the predeclared 10 model seeds by 10 evaluation banks and must not be tuned from
its downstream results.
