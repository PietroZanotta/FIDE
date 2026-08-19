# Two-species unbalanced active-nematic MFSI

This is an isolated variant of `experiments/active_nematic`. The balanced
experiment, vortices, shared `src/mfsi` files, and existing native tesseracts
are not modified or imported with changed semantics.

## Physical solver and provenance

The isolated physical bank uses a 2N-padded pseudospectral evaluation for all
nonlinear terms and ETD2 time integration. The 2x padding is required by the
cubic terms; the focused solver tests compare it with an independent 4N
reference, check second-order temporal convergence, verify the screened-Stokes
residual and incompressibility, and check passive free-energy decay.

Every physical bank stores `solver_revision=dealiased-2x-etd2-v1`. Legacy or
mismatched banks are rejected instead of being silently reused. Consequently,
changing the physical solver requires regeneration from the `physical-bank`
stage onward.

## Scientific state

The two finite positive measures are represented separately:

- `mu_plus` on `(x, y, beta_plus)`, where `beta_plus` is the +1/2 comet/vector
  polarity;
- `mu_minus` on `(x, y, beta_minus)`, where `beta_minus` is a -1/2 triatic
  texture phase.

For visualization only, one minus arm is `beta_minus/3 mod 2*pi/3`; all three
arms must be drawn. Learning and measurements use the branch-free pair
`(cos(beta_minus), sin(beta_minus))`.

One accepted defect from a bank subset containing `N_runs` physical
realizations has finite-measure weight `1/N_runs`. Thus total mass is mean
defect count per realization. Resampling normalized shape particles never
changes physical mass.

## Charge coupling and mass paths

At every saved time the bank checks

`Delta(t) = M_plus(t) - M_minus(t)`

against the declared expected imbalance and `charge_balance_tolerance`. A
failure reports time, both masses, imbalance, and rejected texture/core counts;
it is never silently repaired.

Observed masses are parameterized through pair mass

`m=(M_plus+M_minus)/2`, `M_plus=m+Delta/2`, `M_minus=m-Delta/2`.

The target pair mass is reconstructed by an anchored spline in `log(m)`. The
endpoint reference uses the analytic Fisher--Rao schedule

`sqrt(m_ref(tau))=(1-tau)sqrt(m_21)+tau sqrt(m_31)`.

## Reference

There are two existing-style normalized `PeriodicReferenceFlow` objects, one
per species. Each is trained only on its own `t=21` and `t=31` normalized
endpoint shapes over the single full interval. Times 22--30 never enter
reference training. They remain available for observations, risk, and audits.
The reference reaction is spatially uniform:

`g_ref,s = dot(M_ref,s)/M_ref,s`.

There is no learned reaction-rate head.

The reference particle bank is a declared periodic KDE quadrature of the
empirical training endpoint. Its spatial and phase widths are configured by
`bank_position_jitter_std` and `bank_beta_jitter_std`. This represents the
underlying continuous defect law and avoids the artificial moment-convex-hull
barrier caused by duplicating a few empirical atoms thousands of times. The
KDE random bank is fixed across learned-reference seeds, and hidden
intermediate marginals are still never used.

## Unbalanced residual and screened correction

The existing normalized particle machinery returns `h_shape` under

`partial_t q + div(q u) = q h_shape`.

The finite-measure residual is

`h_ub = h_shape + dot(M)/M - g_ref`.

The correction convention is

`div(mu delta) - mu alpha = -mu h_ub`.

Minimizing

`integral mu (|delta|^2 + kappa alpha^2)`

with `delta=grad(psi)` and `alpha=psi/kappa` gives

`-div(q grad(psi)) + q psi/kappa = q h_ub`.

This screened operator is SPD and has no gauge nullspace. The production
backend is the separate
`native/active_nematic_unbalanced_screened_tesseract` C++17/OpenMP Tesseract;
`full_action.backend=jax_screened` retains the experiment-local JAX fallback.
Independent trajectory times are sent in one native batch. The existing
`native/active_nematic_poisson3d_tesseract` is not modified.

The reported species action is

`A_s = M_s integral q_s |grad psi_s|^2 + M_s integral q_s psi_s^2/kappa`.

Transport, reaction, and reaction fraction are serialized separately. Total
action is the configured weighted sum of plus and minus actions.

## Tangent model and risk

For raw finite-measure observables the tangent Gram is

`G_ub = integral [J Phi J Phi^T + Phi Phi^T/kappa] d mu`.

The global mass observable `Phi=1` is appended with zero geometry derivative;
it does not consume a movable sensor. Tangent transport and reaction pieces are
reported independently.

The scientific risk is the unnormalized RKHS embedding distance

`D_k(mu,nu)=<mu,mu>_k+<nu,nu>_k-2<mu,nu>_k`,

computed separately for both charges and aggregated with declared weights.
Normalized shape MMD and squared mass error are diagnostics only; selection
uses finite-measure risk.

## Law-level interpretation

These trajectories are law-level trajectories, not persistent defect
identities. Source correction permits population creation/destruction while
transport correction moves the finite law. The method does not explicitly pair
individual birth or annihilation events.

## Staged commands

Do not skip the bank audit. Recommended order:

```bash
# 1. Focused unit specifications
.venv/bin/cmake \
  -S native/active_nematic_unbalanced_screened_tesseract \
  -B native/active_nematic_unbalanced_screened_tesseract/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python"
.venv/bin/cmake --build \
  native/active_nematic_unbalanced_screened_tesseract/build -j "$(nproc)"
.venv/bin/pytest -q \
  experiments/active_nematic_unbalanced/test_active_nematic_solver.py \
  experiments/active_nematic_unbalanced/test_unbalanced_core.py \
  experiments/active_nematic_unbalanced/test_screened_poisson3d_tesseract.py

# 2. Isolation and backward compatibility
.venv/bin/pytest -q \
  experiments/active_nematic_unbalanced/test_unbalanced_core.py::test_import_isolation_leaves_vortices_config_unchanged \
  tests/test_active_nematic.py tests/test_active_nematic_mfsi.py tests/test_vortex_support.py

# 3. Physical bank, signed extraction, and charge audit
.venv/bin/python experiments/active_nematic_unbalanced/run.py physical-bank
.venv/bin/python experiments/active_nematic_unbalanced/run.py defects
.venv/bin/python experiments/active_nematic_unbalanced/run.py defect-audit

# 4. One full-interval reference smoke test
.venv/bin/python experiments/active_nematic_unbalanced/run.py reference \
  --smoke --reference-seeds 20260818 \
  --input-dir experiments/active_nematic_unbalanced/outputs/run \
  --output-dir experiments/active_nematic_unbalanced/outputs/reference_smoke

# 5. One fixed-design unbalanced action evaluation
.venv/bin/python experiments/active_nematic_unbalanced/run.py fixed-design \
  --reference-seeds 20260818

# 6. Three-reference audit at the fixed geometry
.venv/bin/python experiments/active_nematic_unbalanced/run.py reference
.venv/bin/python experiments/active_nematic_unbalanced/run.py fixed-design
.venv/bin/python experiments/active_nematic_unbalanced/run.py reference-audit

# 7. Only after those pass, a small design smoke run
.venv/bin/python experiments/active_nematic_unbalanced/run.py design \
  --smoke --reference-seeds 20260818 \
  --input-dir experiments/active_nematic_unbalanced/outputs/run \
  --output-dir experiments/active_nematic_unbalanced/outputs/design_smoke
```

Smoke and production outputs live under this experiment's own `outputs/`
directory. No command defaults to the balanced active-nematic output tree.

For a solver revision, use a fresh directory so stale results remain visibly
separate. The current production run is staged as:

```bash
OUT=experiments/active_nematic_unbalanced/outputs/run_dealiased_etd2
.venv/bin/python experiments/active_nematic_unbalanced/run.py physical-bank --output-dir "$OUT"
.venv/bin/python experiments/active_nematic_unbalanced/run.py defects --output-dir "$OUT"
.venv/bin/python experiments/active_nematic_unbalanced/run.py defect-audit --output-dir "$OUT"
.venv/bin/python experiments/active_nematic_unbalanced/run.py reference --output-dir "$OUT"
.venv/bin/python experiments/active_nematic_unbalanced/run.py fixed-design --output-dir "$OUT"
.venv/bin/python experiments/active_nematic_unbalanced/run.py reference-audit --output-dir "$OUT"
.venv/bin/python experiments/active_nematic_unbalanced/run.py design --output-dir "$OUT"
```

Optimization gives all three objectives a broad multistart stage followed by a
two-candidate quasi-Newton refinement. Broad frozen gradient banks contain 16
law, 16 tangent, and 8 Full trials. Law and tangent refinement use all 32
selection trials. Full refinement uses 16 trials on a 36 x 36 x 18 grid after
the 8-trial 24 x 24 x 12 broad proxy. Exact Full candidate audits use all 32
trials on the production 48 x 48 x 24 grid, and independent validation retains
all 32 validation trials.

Action refinements use exact 32-trial law risk as their constraint. They target
a `1e-5` safety margin inside `risk_star + epsilon_r`; an endpoint that crosses
that inner boundary is bisected back along the proposed step to the last exact
feasible point. Every seed is retained, so an unsuccessful or inferior endpoint
cannot erase an incumbent. See
[`optimization_capacity_diagnosis.md`](optimization_capacity_diagnosis.md) for
the failure analysis and acceptance criteria.

Only a law-stage audit can define `risk_star`. If an action-stage geometry has
lower exact law risk, it becomes a restart for a new law optimization using all
32 selection trials; tangent and Full are then rerun under the refined risk
ceiling. A Full-action score can therefore never define the law optimum.
