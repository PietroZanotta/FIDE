from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from .feasibility import common_beta_support_polytope_2d, project_metric_polytope_2d
from .moments import QuadraticBridgeConfig, fit_quadratic_bridge_gls
from .raster import gaussian_kernel_2d

Array = jax.Array


class TrialBank(NamedTuple):
    """Dense common-random-number bank used by the toy experiment."""

    masses: Array          # [R,T,G], probability masses on the scientific grid
    sample_indices: Array  # [R,K,N]
    detector_z: Array      # [R,K,M]
    alphas: Array          # [R]


@dataclass(frozen=True)
class FastLawConfig:
    finite_n: int
    obs_noise_std: float
    variance_floor: float = 1.0e-10
    raster_bandwidth: float = 0.0
    raster_truncate: float = 4.0
    feasibility_margin: float = 0.0
    feasibility_tol: float = 1.0e-9
    max_finite_calibration_resid: float = 1.0e-3
    max_population_calibration_resid: float = 1.0e-5
    min_ess_fraction: float = 0.03
    min_in_domain_base_mass: float = 0.995


class FastToyLawEvaluator:
    """Efficient differentiable law-search evaluator for stages 1 and 2.

    This object is an optimization/search path, not the authoritative scientific
    acceptance boundary. Final candidates are re-evaluated with exact ConvexHull
    feasibility and strict all-trial validity in ``ToyExperiment``.

    The important distinction is computational:

    * sensor geometry is formed once per eta;
    * CRN trials are batched;
    * Newton multipliers are warm-started along time;
    * all time nodes are still calibrated for the finite validity gate;
    * MMD/KDE work is done only at nodes that actually enter the law objective;
    * the fixed truth-side kernel convolution in MMD is precomputed once;
    * a fixed-direction support polygon provides a smooth outer-approximation search
      surrogate for common physical/particle beta feasibility;
    * the two-dimensional GLS coefficient is projected for every trial.

    No tangent forcing or Poisson quantity is evaluated here because neither is
    part of L(eta) or R(eta).
    """

    def __init__(
        self,
        *,
        family,
        projector,
        grid,
        times: Array,
        time_weights: Array,
        acq_idx: Array,
        heldout_idx: Array,
        population_idx: Array,
        reference_nodes: Array,
        reference_base_weights: Array,
        reference_in_domain: Array,
        population_masses: Array,       # [A,T,G]
        population_alpha_weights: Array,
        selection_bank: TrialBank,
        mmd_kernel: Array,
        support_directions: Array,
        moment_cfg: QuadraticBridgeConfig,
        cfg: FastLawConfig,
    ):
        self.family = family
        self.projector = projector
        self.grid = grid
        self.times = jnp.asarray(times, dtype=jnp.float64)
        self.time_weights = jnp.asarray(time_weights, dtype=jnp.float64)
        self.acq_idx = jnp.asarray(acq_idx, dtype=jnp.int32)
        self.heldout_idx = jnp.asarray(heldout_idx, dtype=jnp.int32)
        self.population_idx = jnp.asarray(population_idx, dtype=jnp.int32)
        self.reference_nodes = jnp.asarray(reference_nodes, dtype=jnp.float64)
        self.reference_in_domain = jnp.asarray(reference_in_domain, dtype=bool)
        self.population_masses = jnp.asarray(population_masses, dtype=jnp.float64)
        self.population_alpha_weights = jnp.asarray(population_alpha_weights, dtype=jnp.float64)
        self.selection_bank = TrialBank(*[jnp.asarray(x) for x in selection_bank])
        self.mmd_kernel = jnp.asarray(mmd_kernel, dtype=jnp.float64)
        self.support_directions = jnp.asarray(support_directions, dtype=jnp.float64)
        self.moment_cfg = moment_cfg
        self.cfg = cfg

        base = jnp.asarray(reference_base_weights, dtype=jnp.float64)
        if base.ndim == 1:
            base = jnp.broadcast_to(base[None, :], self.reference_in_domain.shape)
        base = jnp.where(self.reference_in_domain, base, 0.0)
        self.reference_base_mass = jnp.sum(base, axis=-1)
        self.reference_base_weights = base / jnp.maximum(
            jnp.sum(base, axis=-1, keepdims=True), 1.0e-300
        )
        self.reference_cell_idx = self.grid.flat_bin_index(self.reference_nodes)

        bw = cfg.raster_bandwidth if cfg.raster_bandwidth > 0.0 else 0.35 * grid.dx
        self.raster_kernel = gaussian_kernel_2d(bw / grid.dx, cfg.raster_truncate)

        # Boolean masks and normalized quadrature weights are fixed forever.
        T = int(self.times.shape[0])
        held_mask = jnp.zeros((T,), dtype=bool).at[self.heldout_idx].set(True)
        pop_mask = jnp.zeros((T,), dtype=bool).at[self.population_idx].set(True)
        self.heldout_mask = held_mask
        self.population_mask = pop_mask

        held_w = jnp.where(held_mask, self.time_weights, 0.0)
        self.heldout_weights_all = held_w / jnp.maximum(jnp.sum(held_w), 1.0e-300)
        pop_w = jnp.where(pop_mask, self.time_weights, 0.0)
        self.population_weights_all = pop_w / jnp.maximum(jnp.sum(pop_w), 1.0e-300)

        # MMD truth-side convolutions do not depend on eta.  Precompute Kp and
        # <p,Kp>, leaving only one kernel application (Kq) per projected law.
        self.population_truth_k, self.population_truth_self = self._precompute_truth_mmd(
            self.population_masses
        )
        self.selection_truth_k, self.selection_truth_self = self._precompute_truth_mmd(
            self.selection_bank.masses
        )

    # ------------------------------------------------------------------
    # Fixed-grid helpers
    # ------------------------------------------------------------------

    def _precompute_truth_mmd(self, masses: Array) -> tuple[Array, Array]:
        n = self.grid.n
        grids = masses.reshape(masses.shape[:-1] + (n, n))

        def one(p):
            kp = jsp.signal.fftconvolve(p, self.mmd_kernel, mode="same")
            return kp, jnp.sum(p * kp)

        flat = grids.reshape((-1, n, n))
        kp, self_term = jax.vmap(one)(flat)
        return (
            kp.reshape(grids.shape),
            self_term.reshape(grids.shape[:-2]),
        )

    def _mmd_with_precomputed_truth(self, qmass: Array, kp: Array, pself: Array) -> Array:
        kq = jsp.signal.fftconvolve(qmass, self.mmd_kernel, mode="same")
        val = jnp.sum(qmass * kq) - 2.0 * jnp.sum(qmass * kp) + pself
        return jnp.maximum(val, 0.0)

    def _raster_mass_one(self, weights: Array, cell_idx: Array) -> Array:
        mass = jnp.zeros(self.grid.n * self.grid.n, dtype=jnp.float64)
        mass = mass.at[cell_idx].add(weights)
        mass = mass.reshape((self.grid.n, self.grid.n))
        mass = jsp.signal.convolve2d(mass, self.raster_kernel, mode="same")
        return mass / jnp.maximum(jnp.sum(mass), 1.0e-300)

    def _sensor_tensors(self, eta: Array) -> tuple[Array, Array]:
        eta = self.family.canonicalize(eta)
        return (
            self.family.features(self.grid.flat_points(), eta),
            self.family.features(self.reference_nodes, eta),
        )

    # ------------------------------------------------------------------
    # Batched I-projection scan
    # ------------------------------------------------------------------

    def _project_batch_one_time(
        self,
        phi_t: Array,
        base_t: Array,
        target_t: Array,      # [B,M]
        lam_t: Array,         # [B,M]
    ):
        def one(target, lam0):
            st = self.projector.project(phi_t, base_t, target, lam0=lam0)
            return st.lam, st.weights, jnp.linalg.norm(st.residual), st.ess_fraction

        return jax.vmap(one)(target_t, lam_t)

    def _law_scan(
        self,
        *,
        phi_ref: Array,             # [T,N,M]
        targets: Array,             # [B,T,M]
        truth_k: Array,             # [B,T,n,n]
        truth_self: Array,          # [B,T]
        score_mask: Array,          # [T]
        score_weights: Array,       # [T]
        validity_tol: float,
    ) -> Array:
        """Calibrate every time node, score only requested nodes, return [B]."""
        projection = self.projector.project_trajectory(
            phi_ref, self.reference_base_weights, targets
        )
        B = int(targets.shape[0])
        loss0 = jnp.zeros((B,), dtype=jnp.float64)
        max_resid = jnp.max(jnp.linalg.norm(projection.residual, axis=-1), axis=1)
        min_ess = jnp.min(projection.ess_fraction, axis=1)

        # Scan by time rather than by trial.  Every trial at a fixed time shares
        # phi_t/base_t/cell indices, which is the GPU-friendly batching direction.
        def step(loss, xs):
            weights, cell_t, kp_t, pself_t, do_score, w_t = xs

            def score(_):
                qmass = jax.vmap(self._raster_mass_one, in_axes=(0, None))(weights, cell_t)
                mmd = jax.vmap(self._mmd_with_precomputed_truth)(qmass, kp_t, pself_t)
                return loss + w_t * mmd

            loss = jax.lax.cond(do_score, score, lambda _: loss, operand=None)
            return loss, None

        xs = (
            jnp.swapaxes(projection.weights, 0, 1),
            self.reference_cell_idx,
            jnp.swapaxes(truth_k, 0, 1),
            jnp.swapaxes(truth_self, 0, 1),
            score_mask,
            score_weights,
        )
        loss, _ = jax.lax.scan(step, loss0, xs)
        valid = (
            (max_resid <= float(validity_tol))
            & (min_ess >= float(self.cfg.min_ess_fraction))
            & (jnp.min(self.reference_base_mass) >= float(self.cfg.min_in_domain_base_mass))
        )
        return jnp.where(valid, loss, loss + 1.0e3)

    # ------------------------------------------------------------------
    # Population law objective
    # ------------------------------------------------------------------

    def _population_loss_from_tensors(self, phi_grid: Array, phi_ref: Array) -> Array:
        targets = jnp.einsum("atg,gm->atm", self.population_masses, phi_grid)
        losses = self._law_scan(
            phi_ref=phi_ref,
            targets=targets,
            truth_k=self.population_truth_k,
            truth_self=self.population_truth_self,
            score_mask=self.population_mask,
            score_weights=self.population_weights_all,
            validity_tol=self.cfg.max_population_calibration_resid,
        )
        return jnp.sum(self.population_alpha_weights * losses)

    def population_loss(self, eta: Array) -> Array:
        phi_grid, phi_ref = self._sensor_tensors(eta)
        return self._population_loss_from_tensors(phi_grid, phi_ref)

    # ------------------------------------------------------------------
    # Finite-resource law objective
    # ------------------------------------------------------------------

    def _finite_measurements(self, phi_grid: Array):
        bank = self.selection_bank
        acq_mass = bank.masses[:, self.acq_idx, :]  # [R,K,G]
        exact = jnp.einsum("rkg,gm->rkm", acq_mass, phi_grid)

        second = jnp.einsum("rkg,gi,gj->rkij", acq_mass, phi_grid, phi_grid)
        cov = second - jnp.einsum("rki,rkj->rkij", exact, exact)
        eye = jnp.eye(phi_grid.shape[-1], dtype=jnp.float64)
        V = cov / float(self.cfg.finite_n)
        V = V + (
            float(self.cfg.obs_noise_std) ** 2 + float(self.cfg.variance_floor)
        ) * eye

        vals = phi_grid[bank.sample_indices]  # [R,K,N,M]
        empirical = jnp.mean(vals[..., : self.cfg.finite_n, :], axis=2)
        y = empirical + float(self.cfg.obs_noise_std) * bank.detector_z
        endpoint = (self.acq_idx == 0) | (self.acq_idx == self.times.shape[0] - 1)
        y = jnp.where(endpoint[None, :, None], exact, y)
        return y, V, exact[:, 0], exact[:, -1]

    def _reconstruct_batch(self, phi_grid: Array, phi_ref: Array):
        y, V, c0, c1 = self._finite_measurements(phi_grid)
        t_acq = self.times[self.acq_idx]

        def fit_one(y_i, V_i, c0_i, c1_i):
            return fit_quadratic_bridge_gls(
                t_acq, y_i, V_i, c0_i, c1_i, self.times, self.moment_cfg
            )

        fits = jax.vmap(fit_one)(y, V, c0, c1)

        # The toy endpoints are common across alpha/trials.  Build the common
        # differentiable support-polygon feasibility surrogate once per eta.
        A, b, endpoint_violation = common_beta_support_polytope_2d(
            directions=self.support_directions,
            times=self.times,
            c0=c0[0],
            c1=c1[0],
            physical_features=phi_grid,
            particle_features_by_time=phi_ref,
            particle_mask_by_time=self.reference_in_domain,
            margin=float(self.cfg.feasibility_margin),
        )

        def project_one(beta, information):
            return project_metric_polytope_2d(
                beta,
                information,
                A,
                b,
                tol=float(self.cfg.feasibility_tol),
            )

        projections = jax.vmap(project_one)(fits.beta, fits.information)
        beta = projections.beta
        t = self.times
        z = t * (1.0 - t)
        c = (
            (1.0 - t[None, :, None]) * c0[:, None, :]
            + t[None, :, None] * c1[:, None, :]
            + z[None, :, None] * beta[:, None, :]
        )
        return c, projections.distance, endpoint_violation

    def _finite_risk_from_tensors(self, phi_grid: Array, phi_ref: Array) -> Array:
        targets, projection_distance, endpoint_violation = self._reconstruct_batch(phi_grid, phi_ref)
        losses = self._law_scan(
            phi_ref=phi_ref,
            targets=targets,
            truth_k=self.selection_truth_k,
            truth_self=self.selection_truth_self,
            score_mask=self.heldout_mask,
            score_weights=self.heldout_weights_all,
            validity_tol=self.cfg.max_finite_calibration_resid,
        )
        # Projection distance is diagnostic; the projection itself restores common
        # feasibility. A positive endpoint violation means the common feasible set
        # is inconsistent and should never silently pass.
        endpoint_bad = endpoint_violation > float(self.cfg.feasibility_tol)
        losses = jnp.where(endpoint_bad, losses + 1.0e3, losses)
        return jnp.mean(losses)

    def finite_risk(self, eta: Array) -> Array:
        phi_grid, phi_ref = self._sensor_tensors(eta)
        return self._finite_risk_from_tensors(phi_grid, phi_ref)

    def population_and_finite(self, eta: Array) -> tuple[Array, Array]:
        """Compute L and R with shared eta-dependent sensor geometry."""
        phi_grid, phi_ref = self._sensor_tensors(eta)
        return (
            self._population_loss_from_tensors(phi_grid, phi_ref),
            self._finite_risk_from_tensors(phi_grid, phi_ref),
        )

    def finite_penalized_by_population(
        self,
        eta: Array,
        *,
        population_limit: float,
        penalty: float,
    ) -> Array:
        """Stage-2 optimizer objective with one shared L/R computation graph."""
        L, R = self.population_and_finite(eta)
        violation = jax.nn.relu(L - float(population_limit))
        return R + float(penalty) * violation * violation
