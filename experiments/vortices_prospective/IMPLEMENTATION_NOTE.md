# Prospective vortices implementation note

The repository's active vortices implementation is
`experiments/vortices_percentage`, a finite-population double-gyre benchmark.
This experiment reuses its physical and numerical conventions, not the skyrmion
protocol:

- normalized-time double-gyre dynamics on `[0,2] x [0,1]`;
- the uniform-plus-four-truncated-Gaussians initial law;
- four labelled Gaussian point sensors of width `0.12`;
- sparse finite/noisy acquisition with exact endpoints;
- endpoint-anchored cubic penalized least-squares reconstruction;
- endpoint-only box-logit flow matching and a frozen reference rollout;
- empirical information projection and particle MFSI forcing;
- physical-density weighted-Poisson Full action on a rectangular grid.

The retrospective full-law MMD risks and target intermediate truth access are
deliberately not reused.  New prospective components are:

1. a response table containing only the population mean and second moment of
   every candidate Gaussian response on a spatial grid;
2. fixed aggregate QoI trajectories (centroid and raw second moments);
3. `TargetProspectiveData`, which exposes endpoints and aggregate products but
   has no hidden-state member or loader;
4. aggregate-QoI risk evaluated on the information-projected reference law;
5. Tangent selection and reduced-grid Full shortlisting inside the same risk set;
6. a frozen selection manifest written before validation can load or generate
   the disjoint hidden microscopic bank;
7. explicit predicted-versus-realized action and risk reporting.

The target regime keeps the established physical parameters (`A=0.1`,
`epsilon=0.25`, horizon/period `10`).  The primary question is quantity-specific
trust at that target regime; cross-parameter generalization is outside the first
implementation.

The predeclared scientific QoIs are `x/2`, `y`, `(x/2)^2`, `y^2`, and
`(x/2)y`.  They encode global centroid, spread, and covariance information, are
cheap, and cannot be identical to any optimized localized Gaussian sensor.

Data flow is one way:

```text
prospective simulator states (preprocessor memory only)
  -> aggregate response/QoI artifact
  -> selection

endpoint simulator states -> endpoint artifact -> frozen reference -> selection

frozen manifest -> hidden validation simulator/state cache -> validation only
```

No existing vortices or shared source file is modified by this experiment.
