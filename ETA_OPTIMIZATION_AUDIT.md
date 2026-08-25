# FIDE `eta` Optimization Audit

## Executive finding

The repository contains four active benchmark families. Three—Toy, Vortices,
and Active Nematic—differentiate a reduced-fidelity Full-action proxy with
respect to `eta` using reverse-mode JAX and custom implicit VJPs. The learned
reference dynamics are frozen because their trajectories and velocities are
materialized before the design objective is constructed.

The Skyrmion benchmark is materially different: it does **not** differentiate
Full action with respect to `eta`. It searches a finite, randomly refined set of
sensor designs, ranks them with a forcing-norm proxy, and then trains a separate
Deep Ritz network for each shortlisted fixed design.

This audit follows executed source paths rather than relying on comments or the
paper alone.

## 1. Optimization variable and call chain

| Benchmark | `eta` being searched | Parameterization and constraints | Optimizer/search |
|---|---|---|---|
| Toy | JAX vector of two angles, shape `(2,)` | Centers are `r(cos eta, sin eta)`. The evaluator applies modulo `2*pi` and sorting. Circular separation uses a squared-ReLU penalty. | Custom JAX Adam |
| Vortices | JAX vector `[x1,y1,...,x4,y4]`, shape `(8,)` | Direct physical coordinates. Sensors remain labelled. Iterates are clipped to the sensor box; box and separation constraints use squared-ReLU penalties. | Custom JAX Adam |
| Active nematic | JAX vector `[x1,y1,x2,y2]`, shape `(4,)` in the current config | Direct periodic coordinates. Observable evaluation applies modulo box size. Periodic separation and risk ceiling use squared-ReLU penalties. | Custom JAX Adam |
| Skyrmion | Candidate vector `[x1,y1,...,x4,y4]`, shape `(8,)` | Direct coordinates generated as random/local feasible candidates and filtered on the host. | No gradient optimizer over `eta`; finite candidate search |

The common optimizer constructs

```text
L_opt(eta) = L_primary(eta)
             + penalty * sum_j ReLU(g_j(eta) - upper_j)^2
```

in [`src/mfsi/design.py`](src/mfsi/design.py#L83), calls
`jax.value_and_grad(objective)` in `_adam_single` at
[`src/mfsi/design.py`](src/mfsi/design.py#L98), and updates the `eta` array
directly at line 116. No reference-network parameters appear in this optimizer
PyTree.

### Toy

`eta` is converted to physical centers in `GaussianSensor2D.features` at
[`src/mfsi/measurements.py`](src/mfsi/measurements.py#L26). Canonicalization is
modulo-plus-sort at lines 43–46, so it is nonsmooth at wrap and sensor-order
ties.

The stage-four call chain is:

```text
run.py
  -> run_experiment
  -> _select_action_design
  -> optimize_multistart_candidates
  -> _adam_single / jax.value_and_grad
  -> full_action_gradient
  -> _geometry
  -> _reconstruct_from_geometry
  -> _particle_forcing_trajectory
  -> rasterize_projected_particles
  -> Tesseract or JAX weighted-Poisson proxy
  -> mean time/trial action
```

The supplied objective is `full_action_gradient(eta, grad_action_bank) / anchor`
at [`experiments/toy_example_percentage/experiment.py`](experiments/toy_example_percentage/experiment.py#L3907).

### Vortices

`GaussianPointSensors2D` reshapes the optimized eight-vector directly into four
labelled centers at
[`src/mfsi/measurements.py`](src/mfsi/measurements.py#L72). Canonicalization is
the identity.

The geometry constraints are box and separation constraints at
[`experiments/vortices_percentage/selection.py`](experiments/vortices_percentage/selection.py#L120),
and per-iterate box clipping is at lines 131–146. Stage four supplies the
normalized reduced-grid Full proxy to Adam at line 628.

### Active nematic

The current configuration declares two sensors at
[`experiments/active_nematic_unbalance_percentage/config.json`](experiments/active_nematic_unbalance_percentage/config.json#L52).
`PeriodicGaussianSensors` reshapes the four-vector into centers and applies
periodic modulo at
[`experiments/active_nematic_unbalance_percentage/measurements.py`](experiments/active_nematic_unbalance_percentage/measurements.py#L47).

The optimizer uses a smooth periodic chord-distance separation constraint at
line 121 of that file and a normalized law-risk constraint constructed at
[`percentage_selection.py`](experiments/active_nematic_unbalance_percentage/percentage_selection.py#L123).
Its Full objective is `proxy_exp.mean_metric(..., "full_action")` at line 413.

### Skyrmion

`LocalDensitySensors.centers` accepts the physical eight-vector directly at
[`experiments/skyrmions_deep_ritz/measurements.py`](experiments/skyrmions_deep_ritz/measurements.py#L24).
Designs come from `random_sensor_designs` and `local_sensor_designs`, which
host-filter geometry at lines 50–103.

The main search enumerates candidates at
[`experiments/skyrmions_deep_ritz/experiment.py`](experiments/skyrmions_deep_ritz/experiment.py#L713).
The only gradient-optimized object in the authoritative Full stage is the Deep
Ritz network parameter PyTree, through Adam and L-BFGS at
[`deep_ritz.py`](experiments/skyrmions_deep_ritz/deep_ritz.py#L350).

## 2. What is frozen

For every benchmark:

- Reference neural-network parameters are absent from the design optimizer tree.
- Reference rollouts and velocities are computed or loaded before `eta` search.
- They are not recomputed when `eta` changes.
- There is no path from `eta` into the learned reference dynamics.
- An explicit `stop_gradient` is unnecessary because these arrays are already
  disconnected constants in the design objective.

Evidence:

- Toy trains the reference separately at
  [`experiment.py`](experiments/toy_example_percentage/experiment.py#L432),
  constructs `nodes` and `velocity` once at lines 578–598, and loads or passes
  those arrays into `ToyExperiment` at lines 3384–3441.
- Vortices trains or loads its reference at
  [`experiment.py`](experiments/vortices_percentage/experiment.py#L452), rolls
  the bank once at lines 489–530, and stores the arrays at lines 608–627.
- Active nematic creates separate frozen plus/minus rollouts at
  [`run.py`](experiments/active_nematic_unbalance_percentage/run.py#L189),
  specifically lines 280–295, and passes the arrays into the experiment at
  lines 332–355.
- Skyrmion marks its reference as frozen and constructs cached banks at
  [`experiment.py`](experiments/skyrmions_deep_ritz/experiment.py#L128) and
  lines 159–206.

Therefore, `D_eta` never differentiates reference parameters or reference
dynamics. It does differentiate expressions such as

```text
J_x Phi_eta(X_ref) * u_ref
```

through `J_x Phi_eta`, while `X_ref` and `u_ref` remain fixed. The Toy Full graph,
for example, passes eta-dependent `grad_nodes` together with stored
`reference_velocity` at
[`experiment.py`](experiments/toy_example_percentage/experiment.py#L2753).

## 3. Observable dependence on `eta`

Toy and Vortices use analytic JAX formulas for both `Phi_eta` and its state
Jacobian:

- Toy: `GaussianSensor2D.features` and `feature_gradients` at
  [`src/mfsi/measurements.py`](src/mfsi/measurements.py#L26).
- Vortices: `GaussianPointSensors2D.features` and `feature_gradients` at lines
  79–92 of the same file.
- Active nematic: periodic chord-Gaussian windows, features, and analytic state
  gradients at
  [`measurements.py`](experiments/active_nematic_unbalance_percentage/measurements.py#L57).
- Skyrmion: local-density features at
  [`measurements.py`](experiments/skyrmions_deep_ritz/measurements.py#L30), with
  state-directional derivatives using `jax.jvp` at line 37.

Thus `D_eta Phi_eta` and `D_eta J_x Phi_eta` are available within smooth regions.
Expected nonsmooth boundaries remain at modulo wraps, Toy sorting ties,
minimum-image boundaries, clipping operations, active-set changes, and density
clamps.

## 4. Reconstruction of `c_eta` and `cdot_eta`

Truth and common-random-number draws are fixed while `eta` varies. Only the
feature values at those fixed samples change.

- Toy uses fixed trial masses, sample indices, and detector noise at
  [`experiment.py`](experiments/toy_example_percentage/experiment.py#L962).
- Vortices uses fixed sample indices and noise at
  [`experiment.py`](experiments/vortices_percentage/experiment.py#L874).
- Active uses fixed per-species sample indices and noise at
  [`unbalanced_experiment.py`](experiments/active_nematic_unbalance_percentage/unbalanced_experiment.py#L433).
- Skyrmion uses a fixed prefix of truth configurations and fixed seeded noise at
  [`experiment.py`](experiments/skyrmions_deep_ritz/experiment.py#L209).

The data-dependent fits remain in JAX:

- Toy GLS builds and solves its normal equations at
  [`src/mfsi/moments.py`](src/mfsi/moments.py#L117), then evaluates both `c` and
  `c_dot` at lines 142–149.
- Vortices, Active, and Skyrmion use the anchored cubic-spline reconstruction.
  Its coefficients are solved in JAX and both outputs are evaluated at
  [`src/mfsi/moments.py`](src/mfsi/moments.py#L401).

SciPy/NumPy constructs fixed spline basis and roughness matrices, not the
data-dependent fitted coefficients, so it does not cut this gradient.

Toy additionally applies an eta-dependent, piecewise-differentiable moment
support projection. Its polytope depends on feature support at
[`src/mfsi/feasibility.py`](src/mfsi/feasibility.py#L40), and the selected
facet/vertex projection is implemented at lines 105–158. The projected
coefficient is used to recompute `c` and `c_dot` at
[`experiment.py`](experiments/toy_example_percentage/experiment.py#L1050).

## 5. Information projection

The JAX implementation uses

```text
w = softmax(log(base_weights) + Phi @ lambda)
F(lambda, Phi, c) = w @ Phi - c
```

at [`src/mfsi/projection.py`](src/mfsi/projection.py#L49).

Newton iterations are **not** autodifferentiated. `make_i_projection_solver`
declares a `jax.custom_vjp` at line 130. Its reverse rule:

1. recomputes weights and covariance;
2. solves `C.T @ adjoint = lambda_bar` at line 158;
3. applies the residual VJP with respect to `phi`, `log_base_weights`, and
   `target` at lines 160–169; and
4. returns zero cotangent only for the algorithmic warm start.

Consequently it implements the implicit relation

```text
D_eta lambda = -C^{-1} D_eta F
```

including both the eta dependence of `Phi_eta` and that of `c_eta`. After the
solve, weights, moments, covariance, residual, and ESS are recomputed with
ordinary JAX operations at
[`src/mfsi/projection.py`](src/mfsi/projection.py#L282). `lambda` is not detached.

The fallback's `custom_vjp` supports the reverse mode used by Adam, but direct
forward-mode `jax.jvp` through that function is not supported.

### Native/Tesseract projection

The native schema marks `phi`, base log-weights, and targets as differentiable
at
[`native/iprojection_tesseract/tesseract_api.py`](native/iprojection_tesseract/tesseract_api.py#L31).
Its VJP exposes cotangents for all three at lines 106–126.

The C++ VJP adds the implicit covariance ridge, solves the covariance adjoint,
returns target cotangents, and accumulates feature/base-weight cotangents at
[`bindings.cpp`](native/iprojection_tesseract/src/bindings.cpp#L526). The native
JVP forms `D F` from `phi_dot`, `log_dot`, and `target_dot`, then solves the
covariance system at lines 566–624. Differentiability is therefore supplied by
registered native JVP/VJP rules, not by tracing the C++ Newton loop.

## 6. `dot(lambda)` and forcing

The balanced shared implementation computes

```text
m_i = J_x Phi_eta(X_i) * u_i_ref
C * lambda_dot = c_dot - E_w[m] - Cov_w(Phi, lambda^T m)
h_i = (Phi_i - E_w[Phi])^T lambda_dot
      + lambda^T m_i - E_w[lambda^T m]
```

at [`src/mfsi/particles.py`](src/mfsi/particles.py#L64). The covariance solve is
a normal `jnp.linalg.solve` at lines 82–86, and forcing is constructed at lines
88–89.

Vortices implements the corresponding batched graph at
[`experiment.py`](experiments/vortices_percentage/experiment.py#L902), while
Active does so at
[`unbalanced_experiment.py`](experiments/active_nematic_unbalance_percentage/unbalanced_experiment.py#L713).
Active subsequently adds the target relative mass rate and subtracts the frozen
reference source rate at lines 757–761.

The graph retains derivatives through:

- covariance `C(eta)`;
- the entire RHS, including `c_dot`, weights, features, and feature Jacobians;
- `lambda_dot`;
- projected weights; and
- forcing `h`.

There is no `stop_gradient` in these constructions.

Skyrmion computes the same finite-dimensional forcing equations at
[`forcing.py`](experiments/skyrmions_deep_ritz/forcing.py#L78), but later NumPy
conversions and exceptions at lines 107–128 are hard candidate checks. There is
no outer eta gradient in that benchmark.

## 7. Full weighted solve

### Toy and Vortices

The balanced JAX equation is schematically

```text
[-div_h(q_eta grad_h) + gauge_strength * g_eta g_eta^T] psi_eta
    = q_eta h_eta,
g_eta = q_eta / ||q_eta||_2.
```

The weighted Laplacian is implemented at
[`src/mfsi/poisson.py`](src/mfsi/poisson.py#L48). RHS, gauge, CG solve, and action
are at lines 77–130.

The pure JAX path uses `jax.scipy.sparse.linalg.cg`, whose implicit derivative
propagates through both the RHS and closed-over operator parameters. The wrapper
is at [`src/mfsi/linear.py`](src/mfsi/linear.py#L12). It does not unroll CG in the
backward pass.

Thus, for `K psi = b`, the derivative includes both terms in

```text
K * D_eta(psi) = D_eta(b) - D_eta(K) * psi.
```

The eta-dependent gauge is included as well.

The native proxy sends `q + q_floor`, `-q*h`, and the normalized gauge to the
native operator at
[`src/mfsi/poisson_tesseract.py`](src/mfsi/poisson_tesseract.py#L101). Its custom
VJP solves an adjoint and returns operator-density, RHS, and gauge cotangents at
[`native/poisson_tesseract/tesseract_api.py`](native/poisson_tesseract/tesseract_api.py#L172).
Its JVP constructs the effective RHS from `q_dot`, `rhs_dot`, and `gauge_dot` at
lines 215–251.

The proxy action is evaluated using physical `q`, not `q + q_floor`; Toy does
this at
[`experiment.py`](experiments/toy_example_percentage/experiment.py#L2715).

### Active nematic

The screened periodic equation is

```text
[-div_h(q_operator grad_h) + q_operator/kappa] psi = q * h_unbalanced,
q_operator = q + q_floor.
```

There is no gauge because the reaction term removes the nullspace. The JAX
implementation is at
[`unbalanced_correction.py`](experiments/active_nematic_unbalance_percentage/unbalanced_correction.py#L154).
Here the floor enters the proxy operator itself, not merely its preconditioner.

The configured native path wraps Tesseract in a custom VJP at
[`screened_poisson3d_tesseract.py`](experiments/active_nematic_unbalance_percentage/screened_poisson3d_tesseract.py#L83).
The native API returns both operator-density and RHS cotangents at
[`tesseract_api.py`](native/active_nematic_unbalanced_screened_tesseract/tesseract_api.py#L138),
and its JVP forms the effective RHS from both perturbations at lines 168–194.

### Skyrmion Deep Ritz

There is no outer differentiation of the optimal potential with respect to
`eta`. For every fixed candidate, the code minimizes

```text
sum_t time_weight_t * [
    0.5 * E_Q[|grad psi_theta|^2]
    + E_Q[h * (psi_theta - E_Q[psi_theta])]
]
```

at [`deep_ritz.py`](experiments/skyrmions_deep_ritz/deep_ritz.py#L195). Adam and
L-BFGS differentiate this objective with respect to the Ritz-network parameters
at lines 350–515.

For `D_eta A`, none of unrolled optimization, implicit optimum differentiation,
the envelope theorem, or stopped-gradient Ritz parameters applies: the code
never asks for that outer derivative.

## 8. End-to-end dependency table

The `Differentiated?` column describes the Toy, Vortices, and Active Full-gradient
paths. Skyrmion evaluates the eta-dependent quantities for fixed candidates but
does not differentiate an outer design objective.

| Quantity | Depends on `eta`? | Differentiated? | Mechanism |
|---|---:|---:|---|
| `Phi_eta` | Yes | Yes | JAX sensor formulas |
| `J_x Phi_eta` | Yes | Yes | Analytic JAX formulas |
| `c_eta` | Yes | Yes | Observations followed by JAX GLS/spline solve |
| `cdot_eta` | Yes | Yes | Same fit with differentiable derivative basis |
| `lambda_eta` | Yes | Yes | Implicit covariance custom VJP |
| Projected weights | Yes | Yes | JAX softmax of features and multiplier |
| Raster `q_eta` | Yes | Yes | Fixed bins with eta-dependent weights |
| `lambda_dot_eta` | Yes | Yes | `jnp.linalg.solve(C(eta), rhs(eta))` |
| `h_eta` | Yes | Yes | Ordinary JAX forcing algebra |
| Poisson operator `K_eta` | Yes | Yes | Implicit CG or native operator VJP |
| Poisson RHS `b_eta` | Yes | Yes | Implicit CG or native RHS cotangent |
| `psi_eta` | Yes | Yes | Implicit linear-solve derivative |
| Reference samples | No | No | Precomputed/cached constants |
| Reference velocities | No | No | Precomputed constants used with differentiable feature Jacobians |
| Reference NN parameters | No | No | Absent from design optimizer tree |

Hard particle-cell binning does not block eta derivatives because reference
positions are fixed and eta enters the raster through weights and forcing; see
[`src/mfsi/raster.py`](src/mfsi/raster.py#L41).

## 9. Optimization versus certification

| Check | What the gradient optimizer sees | Hard/post-gradient treatment |
|---|---|---|
| Geometry | Squared-ReLU penalties; Vortices also clips each iterate | Candidate feasibility rechecked; Skyrmion uses only hard generation/filtering |
| Risk ceiling | Smooth normalized squared-ReLU penalty in Toy, Vortices, and Active | Exact risk ceiling applied again during finalist selection |
| Calibration | Toy/Vortices use `where(valid, A, A + constant)`, not a smooth residual penalty. Active ignores it in `mean_metric`. | Exact hard rejection in all benchmarks |
| ESS | Same hard constant branch in Toy/Vortices; absent from Active objective | Exact audit/filter; hard Skyrmion support screen |
| Convex hull/support | Toy's approximate support projection is piecewise differentiable | Exact Toy/Vortex support checks are authoritative; Skyrmion fails hard on calibration |
| Covariance conditioning | Ridge enters smooth solves | Explicit Vortex exact eigenvalue gate; Skyrmion hard condition check; no explicit Active condition gate |
| Density positivity | Active includes a piecewise mass clamp | Toy authoritative positive-density/mass/source checks; Vortices handles disconnected zero-conductance regions |
| Forcing compatibility | Weighted forcing mean is centered in the graph | Authoritative source/moment checks; Skyrmion hard mean check |
| Poisson residual | Computed but not a smooth penalty | Authoritative hard gate; Active audit checks screened-PDE residual |
| Moment-rate residual | Not in the optimizer loss | Authoritative Toy/Vortex and Deep Ritz certificate |
| Deep Ritz weak/energy/gauge certificates | Not in eta search | Held-out hard certification |
| Incumbent retention/replacement | Not differentiated | Explicit post-gradient candidate retention/replacement |

Toy's `_validity` accepts `projection_distance` but does not use it at
[`experiment.py`](experiments/toy_example_percentage/experiment.py#L2293).
Approximate support is enforced by projecting the bridge, while exact hull and
identifiability checks are deferred to authoritative evaluation.

Active's optimizer objective simply returns the mean requested metric at
[`unbalanced_experiment.py`](experiments/active_nematic_unbalance_percentage/unbalanced_experiment.py#L873).
Calibration, ESS, and PDE diagnostics become hard booleans only in
`audit_metric` at lines 1047–1100.

## 10. Proxy versus authoritative Full objective

| Benchmark | Gradient/search Full | Authoritative Full | Is authoritative Full differentiated? |
|---|---|---|---:|
| Toy | 4 CRN trials, 7/21 time nodes, `41x41`, regularized native/JAX proxy | Full time grid, `51x51`, positive-support raster, robust projection, physical-`q` direct Poisson; 12-trial prescreen then larger-bank rescore | No |
| Vortices | 2 trials, 7/21 times, `64x32`, regularized native/JAX proxy | 24-trial action bank, 21 times, `128x64`, robust projection and physical-`q` direct Poisson; 4-trial prescreen | No |
| Active | 4 trials, `24x24x12`, looser CG | Complete audit banks, `48x48x24`, tighter CG; same screened equation/backend | No |
| Skyrmion | Finite-candidate forcing-norm proxy `E_Q[h^2]` | Fresh Deep Ritz training and held-out certification per shortlisted candidate | No eta differentiation at either stage |

Toy settings are at
[`config.json`](experiments/toy_example_percentage/config.json#L82) and lines
117–127. Vortex settings are at
[`config.json`](experiments/vortices_percentage/config.json#L128) and lines
212–224. Active settings are at
[`config.json`](experiments/active_nematic_unbalance_percentage/config.json#L110)
and lines 142–152.

Skyrmion's proxy is explicitly `E_Q[h^2]` at
[`experiment.py`](experiments/skyrmions_deep_ritz/experiment.py#L729) and is used
only to rank the shortlist at lines 946–960.

Final selected designs can therefore differ from proxy rankings in every
benchmark. Authoritative rescoring is intentionally decisive.

## 11. Numerical gradient sanity check

A temporary script used the Toy production-selected design

```text
eta = (0.3703323583, 1.2015904228)
v   = (0.8686085208, -0.4954989784)
```

with the production frozen reference and selection bank, but reduced to one
trial, three time nodes, an `11x11` grid, and the JAX projection/Poisson
fallbacks.

At `epsilon = 3e-4`:

| Quantity | AD directional derivative | Centered finite difference | Relative discrepancy |
|---|---:|---:|---:|
| mean `c^2` | -0.02145961353 | -0.02145960936 | `1.94e-7` |
| mean `cdot^2` | -0.07835964738 | -0.07835961763 | `3.80e-7` |
| mean `lambda^2` | -83.68053926 | -83.68001566 | `6.26e-6` |
| mean `h^2` | -1,079,376.4094 | -1,079,377.7704 | `1.26e-6` |
| Full proxy | 2,046.4840813 | 2,046.4746127 | `4.63e-6` |

For `epsilon = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4)`, the relative-error sequences
were:

- `c`: `2.16e-4, 1.94e-5, 2.16e-6, 1.94e-7, 2.16e-8`
- `cdot`: `4.22e-4, 3.80e-5, 4.22e-6, 3.80e-7, 4.22e-8`
- `lambda`: `3.50e-3, 3.07e-4, 2.57e-5, 6.26e-6, 9.07e-6`
- forcing: `1.26e-3, 1.13e-4, 1.27e-5, 1.26e-6, 2.56e-7`
- Full: `7.68e-4, 7.02e-5, 9.20e-6, 4.63e-6, 7.50e-6`

The small-epsilon Full plateau is consistent with iterative CG tolerance and
floating-point accumulation, rather than a missing derivative term.

The checked-in Vortex smoke result independently reports finite-risk AD/FD
relative errors of `9.63e-8` and `2.15e-6` at two coordinates in
[`gradient_smoke.json`](experiments/vortices_percentage/outputs/gradient_smoke/gradient_smoke.json#L58).

A fresh native/Tesseract finite-difference run was **not established** because
the optional runtime was unavailable in the temporary environment. The native
feature, target, operator, RHS, and gauge derivative terms were instead verified
directly in their registered JVP/VJP implementations.

## A. Actual computational graph

For Toy, Vortices, and Active, the smooth-region graph is:

```text
eta
 -> (Phi_eta(X_truth), Phi_eta(X_ref), J_x Phi_eta(X_ref))
 -> fixed-CRN observations
 -> (c_eta, cdot_eta)
 -> lambda_eta by implicitly differentiated I-projection
 -> (weights_eta, C_eta, m_eta, lambda_dot_eta, h_eta)
 -> raster (q_eta, h_eta)
 -> psi_eta by implicitly differentiated weighted solve
 -> A_proxy(eta)
 -> Adam plus smooth constraint penalties
 -> nondifferentiated authoritative rescoring/certification
```

Skyrmion instead uses:

```text
eta_candidate
 -> (c_eta, cdot_eta, lambda_eta, h_eta)
 -> E_Q[h_eta^2] shortlist proxy
 -> minimize the Deep Ritz objective over network parameters
 -> held-out certification
```

There is no `D_eta A` in the Skyrmion benchmark.

## B. Gradient semantics

The implementation differentiates every smooth eta-dependent stage from
observable evaluation through reconstruction, implicit calibration, projected
weights, covariance/RHS construction, forcing, rasterization, Poisson operator,
Poisson RHS, solution, and proxy action.

Implicit differentiation is used for:

1. the covariance-adjoint information-projection VJP; and
2. the implicit CG/native adjoint of the linear Poisson solve.

Reference samples, velocities, and neural-network parameters are frozen
constants. Frozen velocities still participate in differentiable products with
`J_x Phi_eta`.

Authoritative host-side exact projection and physical direct-Poisson paths are
not differentiated. Skyrmion's Deep Ritz optimum is not differentiated with
respect to `eta`.

## C. Potential manuscript inaccuracies or ambiguities

- “Reference dynamics are frozen during design optimization” is accurate. It
  would be clearer to say they are precomputed/cached and disconnected, rather
  than explicitly wrapped in `stop_gradient`.
- “Information-projection derivatives use implicit covariance solves” is
  accurate for both JAX and Tesseract.
- “All other smooth operations use automatic differentiation” is accurate only
  for the Toy/Vortex/Active proxy graphs. It needs caveats for native custom
  VJPs, modulo/sort/clipping/argmin boundaries, boolean validity branches, and
  host-side authoritative code.
- “The weighted-Poisson solution is implicitly differentiated” is accurate for
  gradient proxies, but not authoritative direct rescoring and not Skyrmion
  candidate selection.
- “Hard certification is outside the gradient computation” is broadly accurate,
  but Toy/Vortices embed calibration/ESS validity as a discontinuous constant
  branch, while geometry/risk constraints enter as differentiable penalties.
- A repository-wide claim that Full action is optimized through `D_eta A` would
  be false for Skyrmion.
- A claim that candidate optimization and final Full evaluation use the same
  numerical objective would be false. All four benchmarks use proxy ranking
  followed by higher-fidelity or fundamentally different authoritative
  rescoring.

## D. Minimal methodology text

During sensor-design optimization, the reference neural flow is trained first
and its particle trajectories, velocities, and base weights are materialized and
held fixed. For each candidate `eta`, JAX evaluates the sensor observables and
their state gradients on fixed truth and reference samples, reconstructs the
finite-observation moment curve and its time derivative with a differentiable
anchored GLS or spline solve, and calibrates the empirical reference law by
information projection. The multiplier derivative is supplied by a custom
implicit rule that solves the observable covariance system and propagates
cotangents through both the eta-dependent features and reconstructed targets.

For Toy, Vortices, and Active Nematic, the calibrated weights and frozen
reference velocities are used to form the eta-dependent advective moments,
multiplier time derivative, continuity forcing, raster density/source, and a
reduced-fidelity weighted-Poisson or screened-Poisson action. The linear solve is
differentiated implicitly with respect to both its operator and RHS, and
reverse-mode JAX differentiates the resulting scalar proxy with respect to
`eta`; multistart Adam adds squared-ReLU geometry and law-risk penalties.
Candidate selection is then repeated with nondifferentiated higher-fidelity
feasibility checks and authoritative action evaluation. In the Skyrmion
benchmark, `eta` is instead searched over a finite random/local candidate pool:
a forcing-norm proxy shortlists designs, and a separate Deep Ritz network is
trained and certified for each fixed candidate; no Full-action gradient with
respect to `eta` is computed.
