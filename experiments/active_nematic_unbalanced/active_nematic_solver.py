# """Minimal 2-D periodic active-nematic solver.

# Model
# -----
# We evolve a symmetric traceless 2-D nematic tensor

#     Q = [[q1, q2],
#          [q2,-q1]]

# with Beris-Edwards dynamics coupled to an incompressible, overdamped
# Stokes flow.  The implementation follows the simplified active-nematic
# model

#     d_t Q + u.grad Q = S(Q, grad u) - (1/gamma) [delta F/delta Q]^TS
#     eta Lap u - friction u = grad p + div(alpha Q)
#     div u = 0

# with

#     F = int (A |Q|^2 + C |Q|^4 + L |grad Q|^2) dx.

# The Stokes equation is solved exactly mode-by-mode in Fourier space.
# The elastic Laplacian in Q is treated semi-implicitly; all nonlinear
# terms are explicit Euler.  This is intended as a transparent research
# prototype / training-bank generator, not a production CFD solver.

# References for the model form are given in Velez-Ceron et al., arXiv:2409.15479.
# """

# from __future__ import annotations

# from dataclasses import asdict, dataclass
# import json
# from pathlib import Path
# from typing import Dict, Optional, Tuple

# import numpy as np


# Array = np.ndarray


# @dataclass(frozen=True)
# class ActiveNematicParams:
#     """Physical and numerical parameters for :class:`ActiveNematic2D`.

#     ``activity`` is the coefficient ``alpha`` in
#     ``sigma_active = alpha Q``.  The force in the momentum balance is
#     ``div(sigma_active)``; changing the sign of ``activity`` reverses that
#     forcing.  ``gamma`` is the rotational viscosity, while ``friction`` is the
#     substrate/screening coefficient ``gamma_f``.
#     """
#     # Grid / domain
#     n: int = 128
#     box_size: float = 32.0

#     # Landau-de Gennes / rotational dynamics
#     A: float = -1.0
#     C: float = 1.0
#     elastic_L: float = 0.20
#     gamma: float = 1.0
#     flow_alignment: float = 0.70

#     # Overdamped hydrodynamics
#     viscosity: float = 1.0
#     friction: float = 0.10
#     activity: float = 1.50

#     # Time integration
#     dt: float = 0.02

#     # Initial condition
#     init_angle_noise: float = 0.18
#     init_amplitude_noise: float = 0.03
#     init_smoothing_length: float = 2.0

#     def __post_init__(self) -> None:
#         if int(self.n) < 8 or int(self.n) % 2:
#             raise ValueError("n must be an even integer >= 8")
#         if self.box_size <= 0.0 or self.dt <= 0.0:
#             raise ValueError("box_size and dt must be positive")
#         if self.C <= 0.0 or self.elastic_L <= 0.0 or self.gamma <= 0.0:
#             raise ValueError("C, elastic_L, and gamma must be positive")
#         if self.viscosity <= 0.0 or self.friction < 0.0:
#             raise ValueError("viscosity must be positive and friction nonnegative")
#         if self.friction == 0.0:
#             raise ValueError("the initial screened-Stokes benchmark requires friction > 0")
#         if self.init_smoothing_length <= 0.0:
#             raise ValueError("init_smoothing_length must be positive")
#         if self.init_angle_noise < 0.0 or self.init_amplitude_noise < 0.0:
#             raise ValueError("initial-condition noise amplitudes must be nonnegative")


# class ActiveNematic2D:
#     """Pseudo-spectral 2-D active nematic on a periodic square."""

#     def __init__(self, params: ActiveNematicParams, seed: int = 0):
#         self.p = params
#         self.seed = int(seed)
#         self.rng = np.random.default_rng(self.seed)
#         self.n = params.n
#         self.Lbox = params.box_size
#         self.dx = self.Lbox / self.n

#         # Fourier wave numbers in physical units.
#         k = 2.0 * np.pi * np.fft.fftfreq(self.n, d=self.dx)
#         self.kx, self.ky = np.meshgrid(k, k, indexing="ij")
#         self.k2 = self.kx**2 + self.ky**2
#         self.nonzero = self.k2 > 0.0

#         # Standard 2/3 de-aliasing mask for nonlinear products.
#         mode = np.fft.fftfreq(self.n) * self.n
#         mx, my = np.meshgrid(mode, mode, indexing="ij")
#         cutoff = self.n / 3.0
#         self.dealias = (np.abs(mx) <= cutoff) & (np.abs(my) <= cutoff)

#         x = np.linspace(0.0, self.Lbox, self.n, endpoint=False)
#         self.x, self.y = np.meshgrid(x, x, indexing="ij")

#         self.q1, self.q2 = self._initial_condition()
#         self.t = 0.0

#     # ---------- spectral helpers ----------
#     def _fft(self, a: Array) -> Array:
#         return np.fft.fft2(a)

#     def _ifft_real(self, ah: Array) -> Array:
#         return np.fft.ifft2(ah).real

#     def derivative(self, a: Array, axis: int) -> Array:
#         ah = self._fft(a)
#         kh = self.kx if axis == 0 else self.ky
#         return self._ifft_real(1j * kh * ah)

#     def laplacian(self, a: Array) -> Array:
#         return self._ifft_real(-self.k2 * self._fft(a))

#     def _smooth_random_field(self, scale: float) -> Array:
#         noise = self.rng.standard_normal((self.n, self.n))
#         nh = self._fft(noise)
#         # Gaussian low-pass filter exp(-k^2 ell^2 / 2).
#         filt = np.exp(-0.5 * self.k2 * scale**2)
#         out = self._ifft_real(nh * filt)
#         std = out.std()
#         return out / std if std > 0 else out

#     def _initial_condition(self) -> Tuple[Array, Array]:
#         p = self.p
#         # For the chosen free energy, uniform equilibrium has S0=sqrt(-A/C).
#         S0 = np.sqrt(max(-p.A / p.C, 1e-12))
#         theta = p.init_angle_noise * self._smooth_random_field(p.init_smoothing_length)
#         amp = S0 * (
#             1.0 + p.init_amplitude_noise * self._smooth_random_field(p.init_smoothing_length)
#         )
#         amp = np.maximum(0.15 * S0, amp)

#         # Q = S (nn - I/2): q1=(S/2)cos(2theta), q2=(S/2)sin(2theta).
#         q1 = 0.5 * amp * np.cos(2.0 * theta)
#         q2 = 0.5 * amp * np.sin(2.0 * theta)
#         return q1, q2

#     # ---------- physical fields ----------
#     def order_parameter(self) -> Array:
#         """Return scalar nematic order S = 2 sqrt(q1^2 + q2^2)."""
#         return 2.0 * np.sqrt(self.q1**2 + self.q2**2)

#     def director_angle(self) -> Array:
#         """Return director angle theta in [-pi/2, pi/2)."""
#         return 0.5 * np.arctan2(self.q2, self.q1)

#     def velocity(self) -> Tuple[Array, Array]:
#         """Solve incompressible screened Stokes equation in Fourier space.

#         0 = -grad p + eta Lap u - friction u + div(activity * Q),
#         div u = 0.

#         The k=0 velocity is fixed to zero.
#         """
#         p = self.p
#         q1h = self._fft(self.q1)
#         q2h = self._fft(self.q2)

#         # f_i = d_j (alpha Q_ij).
#         # Qxx=q1, Qxy=q2, Qyy=-q1.
#         fxh = 1j * p.activity * (self.kx * q1h + self.ky * q2h)
#         fyh = 1j * p.activity * (self.kx * q2h - self.ky * q1h)

#         # Leray projection P = I - kk^T/k^2 removes pressure.
#         kdotf = self.kx * fxh + self.ky * fyh
#         pfx = fxh.copy()
#         pfy = fyh.copy()
#         pfx[self.nonzero] -= self.kx[self.nonzero] * kdotf[self.nonzero] / self.k2[self.nonzero]
#         pfy[self.nonzero] -= self.ky[self.nonzero] * kdotf[self.nonzero] / self.k2[self.nonzero]
#         pfx[~self.nonzero] = 0.0
#         pfy[~self.nonzero] = 0.0

#         denom = p.viscosity * self.k2 + p.friction
#         uh = np.zeros_like(pfx)
#         vh = np.zeros_like(pfy)
#         # Fourier transform of the declared momentum balance gives
#         # (eta k^2 + friction) u_hat = P f_hat.
#         uh[self.nonzero] = pfx[self.nonzero] / denom[self.nonzero]
#         vh[self.nonzero] = pfy[self.nonzero] / denom[self.nonzero]
#         # If friction > 0, the zero mode is still set to zero by gauge choice.

#         return self._ifft_real(uh), self._ifft_real(vh)

#     def active_force(self) -> Tuple[Array, Array]:
#         """Return f_active = div(activity * Q).

#         This is the direct force density supplied by the active nematic stress
#         before the incompressibility/pressure projection.
#         """
#         a = self.p.activity
#         fx = a * (self.derivative(self.q1, 0) + self.derivative(self.q2, 1))
#         fy = a * (self.derivative(self.q2, 0) - self.derivative(self.q1, 1))
#         return fx, fy

#     def pressure(self) -> Array:
#         """Reconstruct the zero-mean Stokes pressure from the active force.

#         In ``0=-grad p+eta Lap u-friction u+f``, the longitudinal force is
#         canceled by ``grad p``.  The k=0 pressure gauge is fixed to zero.
#         """
#         q1h = self._fft(self.q1)
#         q2h = self._fft(self.q2)
#         fxh = 1j * self.p.activity * (self.kx * q1h + self.ky * q2h)
#         fyh = 1j * self.p.activity * (self.kx * q2h - self.ky * q1h)
#         kdotf = self.kx * fxh + self.ky * fyh
#         ph = np.zeros_like(kdotf)
#         ph[self.nonzero] = -1j * kdotf[self.nonzero] / self.k2[self.nonzero]
#         ph[~self.nonzero] = 0.0
#         return self._ifft_real(ph)

#     def molecular_field(self) -> Tuple[Array, Array]:
#         """Return H=-delta F/delta Q in the q1,q2 representation.

#         The solver advances Q with H/gamma plus flow/advection terms.
#         """
#         p = self.p
#         qnorm2 = 2.0 * (self.q1**2 + self.q2**2)
#         bulk = -(2.0 * p.A + 4.0 * p.C * qnorm2)
#         h1 = bulk * self.q1 + 2.0 * p.elastic_L * self.laplacian(self.q1)
#         h2 = bulk * self.q2 + 2.0 * p.elastic_L * self.laplacian(self.q2)
#         return h1, h2

#     def vorticity(self, u: Optional[Array] = None, v: Optional[Array] = None) -> Array:
#         if u is None or v is None:
#             u, v = self.velocity()
#         return self.derivative(v, 0) - self.derivative(u, 1)

#     def _beris_edwards_S(self, u: Array, v: Array) -> Tuple[Array, Array]:
#         """Compute the q1,q2 components of generalized tensor advection S."""
#         lam = self.p.flow_alignment

#         ux = self.derivative(u, 0)
#         uy = self.derivative(u, 1)
#         vx = self.derivative(v, 0)
#         vy = self.derivative(v, 1)

#         # E=(grad u + grad u^T)/2, Omega=(grad u - grad u^T)/2.
#         Exx = ux
#         Eyy = vy
#         Exy = 0.5 * (uy + vx)
#         Omxy = 0.5 * (uy - vx)

#         # B = Q + I/2.
#         Bxx = self.q1 + 0.5
#         Bxy = self.q2
#         Byy = -self.q1 + 0.5

#         # Aplus = lambda E + Omega, Aminus = lambda E - Omega.
#         Ap_xx = lam * Exx
#         Ap_xy = lam * Exy + Omxy
#         Ap_yx = lam * Exy - Omxy
#         Ap_yy = lam * Eyy

#         Am_xx = lam * Exx
#         Am_xy = lam * Exy - Omxy
#         Am_yx = lam * Exy + Omxy
#         Am_yy = lam * Eyy

#         # M = Aplus B + B Aminus.
#         Mxx = Ap_xx * Bxx + Ap_xy * Bxy + Bxx * Am_xx + Bxy * Am_yx
#         Mxy = Ap_xx * Bxy + Ap_xy * Byy + Bxx * Am_xy + Bxy * Am_yy
#         Myx = Ap_yx * Bxx + Ap_yy * Bxy + Bxy * Am_xx + Byy * Am_yx
#         Myy = Ap_yx * Bxy + Ap_yy * Byy + Bxy * Am_xy + Byy * Am_yy

#         # grad(u):Q = E:Q because Omega:Q = 0 for symmetric Q.
#         contraction = ux * self.q1 + uy * self.q2 + vx * self.q2 - vy * self.q1
#         Mxx -= 2.0 * lam * Bxx * contraction
#         Mxy -= 2.0 * lam * Bxy * contraction
#         Myx -= 2.0 * lam * Bxy * contraction
#         Myy -= 2.0 * lam * Byy * contraction

#         # Numerical TS projection. q1 is (Sxx-Syy)/2; q2=(Sxy+Syx)/2.
#         s1 = 0.5 * (Mxx - Myy)
#         s2 = 0.5 * (Mxy + Myx)
#         return s1, s2

#     def step(self, nsteps: int = 1) -> None:
#         """Advance by nsteps using semi-implicit Euler for Q diffusion."""
#         p = self.p
#         diff_coeff = 2.0 * p.elastic_L / p.gamma
#         denom = 1.0 + p.dt * diff_coeff * self.k2

#         for _ in range(nsteps):
#             u, v = self.velocity()
#             s1, s2 = self._beris_edwards_S(u, v)

#             q1x, q1y = self.derivative(self.q1, 0), self.derivative(self.q1, 1)
#             q2x, q2y = self.derivative(self.q2, 0), self.derivative(self.q2, 1)
#             adv1 = u * q1x + v * q1y
#             adv2 = u * q2x + v * q2y

#             # |Q|^2 = Tr(Q^T Q) = 2(q1^2+q2^2).
#             qnorm2 = 2.0 * (self.q1**2 + self.q2**2)

#             # -1/gamma * delta F/delta Q, excluding the Laplacian part
#             # handled implicitly. For F=A|Q|^2+C|Q|^4+L|grad Q|^2:
#             # delta F/delta Q = 2A Q + 4C|Q|^2 Q - 2L Lap Q.
#             bulk_factor = -(2.0 * p.A + 4.0 * p.C * qnorm2) / p.gamma
#             rhs1 = -adv1 + s1 + bulk_factor * self.q1
#             rhs2 = -adv2 + s2 + bulk_factor * self.q2

#             q1h = (self._fft(self.q1) + p.dt * self._fft(rhs1)) / denom
#             q2h = (self._fft(self.q2) + p.dt * self._fft(rhs2)) / denom
#             q1h *= self.dealias
#             q2h *= self.dealias

#             self.q1 = self._ifft_real(q1h)
#             self.q2 = self._ifft_real(q2h)
#             self.t += p.dt

#             if not (np.isfinite(self.q1).all() and np.isfinite(self.q2).all()):
#                 raise FloatingPointError(
#                     "Non-finite Q encountered. Reduce dt/activity or increase viscosity/friction."
#                 )

#     def snapshot(self) -> Dict[str, Array]:
#         u, v = self.velocity()
#         fx, fy = self.active_force()
#         h1, h2 = self.molecular_field()
#         return {
#             "t": np.array(self.t),
#             "q1": self.q1.copy(),
#             "q2": self.q2.copy(),
#             "u_x": u,
#             "u_y": v,
#             # Compatibility aliases used by the original exploratory plotter.
#             "u": u,
#             "v": v,
#             "speed": np.hypot(u, v),
#             "pressure": self.pressure(),
#             "vorticity": self.vorticity(u, v),
#             "active_force_x": fx,
#             "active_force_y": fy,
#             "H1": h1,
#             "H2": h2,
#             "S": self.order_parameter(),
#             "theta": self.director_angle(),
#         }

#     def state_dict(self) -> Dict[str, Array]:
#         """Return the minimal restart state; derived fields remain reproducible."""
#         return {
#             "schema_version": np.asarray(1, dtype=np.int64),
#             "t": np.asarray(self.t, dtype=np.float64),
#             "q1": self.q1.copy(),
#             "q2": self.q2.copy(),
#             "seed": np.asarray(self.seed, dtype=np.int64),
#             "params_json": np.asarray(json.dumps(asdict(self.p), sort_keys=True)),
#         }

#     def load_state_dict(self, state: Dict[str, Array]) -> None:
#         """Restore ``t,q1,q2`` after validating the configured grid."""
#         q1 = np.asarray(state["q1"], dtype=np.float64)
#         q2 = np.asarray(state["q2"], dtype=np.float64)
#         if q1.shape != (self.n, self.n) or q2.shape != q1.shape:
#             raise ValueError(f"restart fields must both have shape {(self.n, self.n)}")
#         if not (np.isfinite(q1).all() and np.isfinite(q2).all()):
#             raise ValueError("restart fields must be finite")
#         self.q1 = q1.copy()
#         self.q2 = q2.copy()
#         self.t = float(np.asarray(state["t"]))

#     def save_state(self, path: str | Path) -> None:
#         """Save a compressed, self-describing restart checkpoint."""
#         path = Path(path)
#         path.parent.mkdir(parents=True, exist_ok=True)
#         np.savez_compressed(path, **self.state_dict())

#     def run(
#         self,
#         t_final: float,
#         save_every: Optional[float] = None,
#         output: Optional[str | Path] = None,
#     ) -> Dict[str, Array]:
#         """Run to t_final; optionally save a time bank to a compressed NPZ."""
#         if t_final <= self.t:
#             raise ValueError("t_final must exceed the current simulation time")

#         if save_every is None:
#             nsteps = int(np.ceil((t_final - self.t) / self.p.dt))
#             self.step(nsteps)
#             return self.snapshot()

#         if save_every < self.p.dt:
#             raise ValueError("save_every must be >= dt")

#         times, q1s, q2s = [], [], []
#         next_save = self.t
#         while self.t < t_final - 0.5 * self.p.dt:
#             if self.t >= next_save - 0.5 * self.p.dt:
#                 times.append(self.t)
#                 q1s.append(self.q1.copy())
#                 q2s.append(self.q2.copy())
#                 next_save += save_every
#             self.step(1)

#         times.append(self.t)
#         q1s.append(self.q1.copy())
#         q2s.append(self.q2.copy())

#         bank = {
#             "schema_version": np.asarray(1, dtype=np.int64),
#             "t": np.asarray(times),
#             "q1": np.asarray(q1s),
#             "q2": np.asarray(q2s),
#             "box_size": np.array(self.Lbox),
#             "dt": np.array(self.p.dt),
#             "seed": np.asarray(self.seed, dtype=np.int64),
#             "params_json": np.asarray(json.dumps(asdict(self.p), sort_keys=True)),
#         }
#         if output is not None:
#             output = Path(output)
#             output.parent.mkdir(parents=True, exist_ok=True)
#             np.savez_compressed(output, **bank)
#         return bank



"""Minimal 2-D periodic active-nematic solver.

Model
-----
We evolve a symmetric traceless 2-D nematic tensor

    Q = [[q1, q2],
         [q2,-q1]]

with Beris-Edwards dynamics coupled to an incompressible, overdamped
screened-Stokes flow.  The reduced model is

    d_t Q + u.grad Q = S(Q, grad u) - (1/gamma) [delta F/delta Q]^TS
    eta Lap u - friction u = grad p + div(alpha Q)
    div u = 0

with

    F = int (A |Q|^2 + C |Q|^4 + L |grad Q|^2) dx.

This is the commonly used reduced active-nematic model in which passive
nematic/back-flow stresses are omitted from the momentum equation.  The
screened-Stokes equation is solved exactly mode-by-mode in Fourier space.
Cubic nonlinearities are evaluated pseudospectrally on a 2N x 2N padded
grid and truncated back to N x N, avoiding aliasing into the resolved band.
The Q equation uses a second-order exponential time-differencing Runge-Kutta
(ETD2) update with the elastic Laplacian treated exactly.

See, e.g., the Beris-Edwards/screened-Stokes reduction in Schimming, Reichhardt & Reichhardt,
arXiv:2409.15479 (published as Phys. Rev. E 111, 035404 (2025)).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


Array = np.ndarray
ACTIVE_NEMATIC_SOLVER_REVISION = "dealiased-2x-etd2-v1"


@dataclass(frozen=True)
class ActiveNematicParams:
    """Physical and numerical parameters for :class:`ActiveNematic2D`.

    ``activity`` is the coefficient ``alpha`` in the reduced momentum balance

        eta Lap u - friction u = grad p + div(alpha Q).

    Thus changing the sign of ``activity`` reverses the active forcing.
    ``gamma`` is the rotational viscosity, while ``friction`` is the
    substrate/screening coefficient.  ``friction=0`` is allowed; on a periodic
    domain the spatially uniform velocity mode is then fixed to zero.
    """

    # Grid / domain
    n: int = 128
    box_size: float = 32.0

    # Landau-de Gennes / rotational dynamics
    A: float = -1.0
    C: float = 1.0
    elastic_L: float = 0.20
    gamma: float = 1.0
    flow_alignment: float = 0.70

    # Overdamped hydrodynamics
    viscosity: float = 1.0
    friction: float = 0.10
    activity: float = 1.50

    # Time integration
    dt: float = 0.02

    # Initial condition
    init_angle_noise: float = 0.18
    init_amplitude_noise: float = 0.03
    init_smoothing_length: float = 2.0

    def __post_init__(self) -> None:
        if int(self.n) < 8 or int(self.n) % 2:
            raise ValueError("n must be an even integer >= 8")
        if self.box_size <= 0.0 or self.dt <= 0.0:
            raise ValueError("box_size and dt must be positive")
        if self.C <= 0.0 or self.elastic_L <= 0.0 or self.gamma <= 0.0:
            raise ValueError("C, elastic_L, and gamma must be positive")
        if self.viscosity <= 0.0 or self.friction < 0.0:
            raise ValueError("viscosity must be positive and friction nonnegative")
        if self.init_smoothing_length <= 0.0:
            raise ValueError("init_smoothing_length must be positive")
        if self.init_angle_noise < 0.0 or self.init_amplitude_noise < 0.0:
            raise ValueError("initial-condition noise amplitudes must be nonnegative")


class ActiveNematic2D:
    """Pseudo-spectral 2-D active nematic on a periodic square."""

    def __init__(self, params: ActiveNematicParams, seed: int = 0):
        self.p = params
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.n = params.n
        self.Lbox = params.box_size
        self.dx = self.Lbox / self.n

        # Base-grid Fourier wave numbers in physical units.
        k = 2.0 * np.pi * np.fft.fftfreq(self.n, d=self.dx)
        self.kx, self.ky = np.meshgrid(k, k, indexing="ij")
        self.k2 = self.kx**2 + self.ky**2
        self.nonzero = self.k2 > 0.0
        self._inverse_k2 = np.zeros_like(self.k2)
        self._inverse_k2[self.nonzero] = 1.0 / self.k2[self.nonzero]
        self._inverse_stokes = np.zeros_like(self.k2)
        self._inverse_stokes[self.nonzero] = 1.0 / (
            params.viscosity * self.k2[self.nonzero] + params.friction
        )

        # Retain the legacy public-ish attribute for compatibility.  The ETD2
        # solver below does not rely on this 2/3 mask: nonlinearities are
        # instead evaluated on a 2N padded grid, which is sufficient for the
        # cubic terms present in this model.
        mode = np.fft.fftfreq(self.n) * self.n
        mx, my = np.meshgrid(mode, mode, indexing="ij")
        cutoff = self.n / 3.0
        self.dealias = (np.abs(mx) <= cutoff) & (np.abs(my) <= cutoff)

        # For even grids the Nyquist line has no distinct positive-frequency
        # partner.  Setting those lines to zero is the standard spectral
        # convention and makes interpolation to the padded grid unambiguous.
        self._state_mask = ~(
            np.isclose(np.abs(mx), self.n / 2.0)
            | np.isclose(np.abs(my), self.n / 2.0)
        )

        # 2N grid used only for alias-free nonlinear evaluation.  A factor of
        # two is sufficient here because the highest polynomial degree in Q is
        # cubic.  The resolved N-grid coefficients are truncated back exactly.
        self._npad = 2 * self.n
        self._pad_scale = (self._npad / self.n) ** 2
        kp = 2.0 * np.pi * np.fft.fftfreq(
            self._npad, d=self.Lbox / self._npad
        )
        self._kx_pad, self._ky_pad = np.meshgrid(kp, kp, indexing="ij")
        self._k2_pad = self._kx_pad**2 + self._ky_pad**2
        self._nonzero_pad = self._k2_pad > 0.0
        self._inverse_k2_pad = np.zeros_like(self._k2_pad)
        self._inverse_k2_pad[self._nonzero_pad] = (
            1.0 / self._k2_pad[self._nonzero_pad]
        )
        self._inverse_stokes_pad = np.zeros_like(self._k2_pad)
        self._inverse_stokes_pad[self._nonzero_pad] = 1.0 / (
            params.viscosity * self._k2_pad[self._nonzero_pad]
            + params.friction
        )
        self._etd_coefficient_cache: dict[float, Tuple[Array, Array, Array]] = {}

        x = np.linspace(0.0, self.Lbox, self.n, endpoint=False)
        self.x, self.y = np.meshgrid(x, x, indexing="ij")

        self.q1, self.q2 = self._initial_condition()
        self.q1 = self._project_state(self.q1)
        self.q2 = self._project_state(self.q2)
        self.t = 0.0

    # ---------- spectral helpers ----------
    def _fft(self, a: Array) -> Array:
        return np.fft.fft2(a)

    def _ifft_real(self, ah: Array) -> Array:
        return np.fft.ifft2(ah).real

    @staticmethod
    def _ifft_real_pair(ah: Array, bh: Array) -> Tuple[Array, Array]:
        """Invert two Hermitian spectra with one complex inverse FFT."""
        pair = np.fft.ifft2(ah + 1j * bh)
        return pair.real, pair.imag

    def _state_hat(self, a: Array) -> Array:
        """FFT of a base-grid state with ambiguous Nyquist lines removed."""
        ah = self._fft(a)
        ah *= self._state_mask
        return ah

    def _project_state(self, a: Array) -> Array:
        return self._ifft_real(self._state_hat(a))

    def _pad_hat(self, ah: Array) -> Array:
        """Embed an N x N spectrum in a 2N x 2N spectrum.

        NumPy's FFT normalization requires an area factor so that inverse FFTs
        on the padded grid preserve physical-space amplitudes.  Nyquist lines
        are assumed to have been zeroed by :meth:`_state_hat`.
        """
        n, m = self.n, self._npad
        start = (m - n) // 2
        out_shift = np.zeros((m, m), dtype=np.complex128)
        out_shift[start : start + n, start : start + n] = np.fft.fftshift(ah)
        return np.fft.ifftshift(out_shift) * self._pad_scale

    def _truncate_hat(self, ah_pad: Array) -> Array:
        """Truncate a 2N x 2N spectrum to the resolved N x N band."""
        n, m = self.n, self._npad
        start = (m - n) // 2
        block = np.fft.fftshift(ah_pad)[
            start : start + n, start : start + n
        ]
        out = np.fft.ifftshift(block) / self._pad_scale
        out *= self._state_mask
        return out

    @staticmethod
    def _phi1(z: Array) -> Array:
        out = np.empty_like(z, dtype=np.float64)
        small = np.abs(z) < 1e-7
        zs = z[small]
        out[small] = 1.0 + 0.5 * zs + zs**2 / 6.0 + zs**3 / 24.0
        out[~small] = np.expm1(z[~small]) / z[~small]
        return out

    @staticmethod
    def _phi2(z: Array) -> Array:
        out = np.empty_like(z, dtype=np.float64)
        small = np.abs(z) < 1e-5
        zs = z[small]
        out[small] = 0.5 + zs / 6.0 + zs**2 / 24.0 + zs**3 / 120.0
        out[~small] = (np.expm1(z[~small]) - z[~small]) / (z[~small] ** 2)
        return out

    def _etd_coefficients(self, h: float) -> Tuple[Array, Array, Array]:
        """Cache the fixed-step ETD2 multipliers used by every realization."""
        key = float(h)
        cached = self._etd_coefficient_cache.get(key)
        if cached is None:
            diffusion = 2.0 * self.p.elastic_L / self.p.gamma
            z = -key * diffusion * self.k2
            cached = (np.exp(z), self._phi1(z), self._phi2(z))
            self._etd_coefficient_cache[key] = cached
        return cached

    def derivative(self, a: Array, axis: int) -> Array:
        ah = self._fft(a)
        kh = self.kx if axis == 0 else self.ky
        return self._ifft_real(1j * kh * ah)

    def laplacian(self, a: Array) -> Array:
        return self._ifft_real(-self.k2 * self._fft(a))

    def _smooth_random_field(self, scale: float) -> Array:
        noise = self.rng.standard_normal((self.n, self.n))
        nh = self._fft(noise)
        # Gaussian low-pass filter exp(-k^2 ell^2 / 2).
        filt = np.exp(-0.5 * self.k2 * scale**2)
        out = self._ifft_real(nh * filt)
        std = out.std()
        return out / std if std > 0 else out

    def _initial_condition(self) -> Tuple[Array, Array]:
        p = self.p
        # For F=A|Q|^2+C|Q|^4+..., Q=S(nn-I/2) gives S0=sqrt(-A/C).
        S0 = np.sqrt(max(-p.A / p.C, 1e-12))
        theta = p.init_angle_noise * self._smooth_random_field(
            p.init_smoothing_length
        )
        amp = S0 * (
            1.0
            + p.init_amplitude_noise
            * self._smooth_random_field(p.init_smoothing_length)
        )
        amp = np.maximum(0.15 * S0, amp)

        # Q = S (nn - I/2): q1=(S/2)cos(2theta), q2=(S/2)sin(2theta).
        q1 = 0.5 * amp * np.cos(2.0 * theta)
        q2 = 0.5 * amp * np.sin(2.0 * theta)
        return q1, q2

    # ---------- Stokes helpers ----------
    @staticmethod
    def _project_force(
        fxh: Array,
        fyh: Array,
        kx: Array,
        ky: Array,
        inverse_k2: Array,
    ) -> Tuple[Array, Array]:
        """Apply the Fourier-space Leray projector P=I-kk^T/k^2."""
        kdotf = kx * fxh + ky * fyh
        pfx = fxh - kx * kdotf * inverse_k2
        pfy = fyh - ky * kdotf * inverse_k2
        return pfx, pfy

    def _velocity_hat_from_qhat(
        self,
        q1h: Array,
        q2h: Array,
        *,
        kx: Array,
        ky: Array,
        inverse_k2: Array,
        inverse_stokes: Array,
    ) -> Tuple[Array, Array]:
        """Solve the declared screened-Stokes equation for Fourier velocity."""
        p = self.p
        fxh = 1j * p.activity * (kx * q1h + ky * q2h)
        fyh = 1j * p.activity * (kx * q2h - ky * q1h)
        pfx, pfy = self._project_force(
            fxh, fyh, kx, ky, inverse_k2
        )

        # eta Lap u - Gamma u = grad p + f
        # => -(eta k^2+Gamma) u_hat = P f_hat
        # => u_hat = -P f_hat/(eta k^2+Gamma).
        return -pfx * inverse_stokes, -pfy * inverse_stokes

    # ---------- physical fields ----------
    def order_parameter(self) -> Array:
        """Return scalar nematic order S = 2 sqrt(q1^2 + q2^2)."""
        return 2.0 * np.sqrt(self.q1**2 + self.q2**2)

    def director_angle(self) -> Array:
        """Return director angle theta in [-pi/2, pi/2)."""
        return 0.5 * np.arctan2(self.q2, self.q1)

    def velocity(self) -> Tuple[Array, Array]:
        """Solve incompressible screened Stokes flow in Fourier space.

        eta Lap u - friction u = grad p + div(activity * Q),
        div u = 0.

        The k=0 velocity is fixed to zero, including when ``friction=0``.
        """
        q1h = self._state_hat(self.q1)
        q2h = self._state_hat(self.q2)
        uh, vh = self._velocity_hat_from_qhat(
            q1h,
            q2h,
            kx=self.kx,
            ky=self.ky,
            inverse_k2=self._inverse_k2,
            inverse_stokes=self._inverse_stokes,
        )
        return self._ifft_real_pair(uh, vh)

    def active_force(self) -> Tuple[Array, Array]:
        """Return f_active = div(activity * Q) before pressure projection."""
        a = self.p.activity
        fx = a * (
            self.derivative(self.q1, 0) + self.derivative(self.q2, 1)
        )
        fy = a * (
            self.derivative(self.q2, 0) - self.derivative(self.q1, 1)
        )
        return fx, fy

    def pressure(self) -> Array:
        """Reconstruct the zero-mean pressure for the declared Stokes equation.

        For

            eta Lap u - friction u = grad p + f,

        the longitudinal part satisfies ``grad p = -P_L f``.  The k=0
        pressure gauge is fixed to zero.
        """
        q1h = self._state_hat(self.q1)
        q2h = self._state_hat(self.q2)
        fxh = 1j * self.p.activity * (
            self.kx * q1h + self.ky * q2h
        )
        fyh = 1j * self.p.activity * (
            self.kx * q2h - self.ky * q1h
        )
        kdotf = self.kx * fxh + self.ky * fyh
        ph = np.zeros_like(kdotf)
        ph[self.nonzero] = (
            1j * kdotf[self.nonzero] / self.k2[self.nonzero]
        )
        return self._ifft_real(ph)

    def molecular_field(self) -> Tuple[Array, Array]:
        """Return H=-delta F/delta Q in the q1,q2 representation.

        The cubic bulk contribution is evaluated on the 2N padded grid so this
        diagnostic matches the de-aliased nonlinear term used by ``step``.
        """
        p = self.p
        q1h = self._state_hat(self.q1)
        q2h = self._state_hat(self.q2)
        q1hp = self._pad_hat(q1h)
        q2hp = self._pad_hat(q2h)
        q1p, q2p = self._ifft_real_pair(q1hp, q2hp)

        qnorm2 = 2.0 * (q1p**2 + q2p**2)
        bulk = -(2.0 * p.A + 4.0 * p.C * qnorm2)
        b1h = self._truncate_hat(np.fft.fft2(bulk * q1p))
        b2h = self._truncate_hat(np.fft.fft2(bulk * q2p))

        h1h = b1h - 2.0 * p.elastic_L * self.k2 * q1h
        h2h = b2h - 2.0 * p.elastic_L * self.k2 * q2h
        return self._ifft_real_pair(h1h, h2h)

    def vorticity(
        self, u: Optional[Array] = None, v: Optional[Array] = None
    ) -> Array:
        if u is None or v is None:
            u, v = self.velocity()
        return self.derivative(v, 0) - self.derivative(u, 1)

    def _beris_edwards_S(
        self, u: Array, v: Array
    ) -> Tuple[Array, Array]:
        """Compute de-aliased q1,q2 components of generalized advection S."""
        if u.shape != (self.n, self.n) or v.shape != (self.n, self.n):
            raise ValueError(f"u and v must have shape {(self.n, self.n)}")

        lam = self.p.flow_alignment
        q1h = self._state_hat(self.q1)
        q2h = self._state_hat(self.q2)
        uhp = self._pad_hat(self._state_hat(u))
        vhp = self._pad_hat(self._state_hat(v))
        q1p, q2p = self._ifft_real_pair(
            self._pad_hat(q1h), self._pad_hat(q2h)
        )

        ux, uy = self._ifft_real_pair(
            1j * self._kx_pad * uhp, 1j * self._ky_pad * uhp
        )
        vx, vy = self._ifft_real_pair(
            1j * self._kx_pad * vhp, 1j * self._ky_pad * vhp
        )

        Exx = ux
        Eyy = vy
        Exy = 0.5 * (uy + vx)
        Omxy = 0.5 * (uy - vx)

        Bxx = q1p + 0.5
        Bxy = q2p
        Byy = -q1p + 0.5

        Ap_xx = lam * Exx
        Ap_xy = lam * Exy + Omxy
        Ap_yx = lam * Exy - Omxy
        Ap_yy = lam * Eyy

        Am_xx = lam * Exx
        Am_xy = lam * Exy - Omxy
        Am_yx = lam * Exy + Omxy
        Am_yy = lam * Eyy

        Mxx = Ap_xx * Bxx + Ap_xy * Bxy + Bxx * Am_xx + Bxy * Am_yx
        Mxy = Ap_xx * Bxy + Ap_xy * Byy + Bxx * Am_xy + Bxy * Am_yy
        Myx = Ap_yx * Bxx + Ap_yy * Bxy + Bxy * Am_xx + Byy * Am_yx
        Myy = Ap_yx * Bxy + Ap_yy * Byy + Bxy * Am_xy + Byy * Am_yy

        contraction = ux * q1p + uy * q2p + vx * q2p - vy * q1p
        Mxx -= 2.0 * lam * Bxx * contraction
        Mxy -= 2.0 * lam * Bxy * contraction
        Myx -= 2.0 * lam * Bxy * contraction
        Myy -= 2.0 * lam * Byy * contraction

        s1h = self._truncate_hat(np.fft.fft2(0.5 * (Mxx - Myy)))
        s2h = self._truncate_hat(np.fft.fft2(0.5 * (Mxy + Myx)))
        return self._ifft_real_pair(s1h, s2h)

    # ---------- time integration ----------
    def _nonlinear_rhs_hat(
        self, q1: Array, q2: Array
    ) -> Tuple[Array, Array]:
        """Return the explicit RHS N(Q) in dQ/dt = D Lap Q + N(Q).

        All products are evaluated on the 2N padded grid.  Because this model
        contains terms of polynomial degree at most three in Q, 2x padding
        prevents unresolved cubic modes from aliasing into the N-grid band.
        """
        p = self.p
        q1h = self._state_hat(q1)
        q2h = self._state_hat(q2)
        q1hp = self._pad_hat(q1h)
        q2hp = self._pad_hat(q2h)

        q1p, q2p = self._ifft_real_pair(q1hp, q2hp)

        uhp, vhp = self._velocity_hat_from_qhat(
            q1hp,
            q2hp,
            kx=self._kx_pad,
            ky=self._ky_pad,
            inverse_k2=self._inverse_k2_pad,
            inverse_stokes=self._inverse_stokes_pad,
        )
        up, vp = self._ifft_real_pair(uhp, vhp)

        q1x, q1y = self._ifft_real_pair(
            1j * self._kx_pad * q1hp, 1j * self._ky_pad * q1hp
        )
        q2x, q2y = self._ifft_real_pair(
            1j * self._kx_pad * q2hp, 1j * self._ky_pad * q2hp
        )
        adv1 = up * q1x + vp * q1y
        adv2 = up * q2x + vp * q2y

        ux, uy = self._ifft_real_pair(
            1j * self._kx_pad * uhp, 1j * self._ky_pad * uhp
        )
        vx, vy = self._ifft_real_pair(
            1j * self._kx_pad * vhp, 1j * self._ky_pad * vhp
        )

        Exx = ux
        Eyy = vy
        Exy = 0.5 * (uy + vx)
        Omxy = 0.5 * (uy - vx)
        lam = p.flow_alignment

        Bxx = q1p + 0.5
        Bxy = q2p
        Byy = -q1p + 0.5

        Ap_xx = lam * Exx
        Ap_xy = lam * Exy + Omxy
        Ap_yx = lam * Exy - Omxy
        Ap_yy = lam * Eyy

        Am_xx = lam * Exx
        Am_xy = lam * Exy - Omxy
        Am_yx = lam * Exy + Omxy
        Am_yy = lam * Eyy

        Mxx = Ap_xx * Bxx + Ap_xy * Bxy + Bxx * Am_xx + Bxy * Am_yx
        Mxy = Ap_xx * Bxy + Ap_xy * Byy + Bxx * Am_xy + Bxy * Am_yy
        Myx = Ap_yx * Bxx + Ap_yy * Bxy + Bxy * Am_xx + Byy * Am_yx
        Myy = Ap_yx * Bxy + Ap_yy * Byy + Bxy * Am_xy + Byy * Am_yy

        contraction = ux * q1p + uy * q2p + vx * q2p - vy * q1p
        Mxx -= 2.0 * lam * Bxx * contraction
        Mxy -= 2.0 * lam * Bxy * contraction
        Myx -= 2.0 * lam * Bxy * contraction
        Myy -= 2.0 * lam * Byy * contraction

        s1 = 0.5 * (Mxx - Myy)
        s2 = 0.5 * (Mxy + Myx)

        qnorm2 = 2.0 * (q1p**2 + q2p**2)
        bulk_factor = -(2.0 * p.A + 4.0 * p.C * qnorm2) / p.gamma

        rhs1p = -adv1 + s1 + bulk_factor * q1p
        rhs2p = -adv2 + s2 + bulk_factor * q2p
        return (
            self._truncate_hat(np.fft.fft2(rhs1p)),
            self._truncate_hat(np.fft.fft2(rhs2p)),
        )

    def _advance_one(self, h: float) -> None:
        """Advance one ETD2 step of size ``h`` without changing the API."""
        if h <= 0.0:
            raise ValueError("time step must be positive")

        E, phi1, phi2 = self._etd_coefficients(h)

        q1h = self._state_hat(self.q1)
        q2h = self._state_hat(self.q2)
        n1h, n2h = self._nonlinear_rhs_hat(self.q1, self.q2)

        # ETD Euler predictor followed by the second-order ETD2 correction.
        a1h = E * q1h + h * phi1 * n1h
        a2h = E * q2h + h * phi1 * n2h
        a1h *= self._state_mask
        a2h *= self._state_mask
        a1, a2 = self._ifft_real_pair(a1h, a2h)

        na1h, na2h = self._nonlinear_rhs_hat(a1, a2)
        q1new = a1h + h * phi2 * (na1h - n1h)
        q2new = a2h + h * phi2 * (na2h - n2h)
        q1new *= self._state_mask
        q2new *= self._state_mask

        self.q1, self.q2 = self._ifft_real_pair(q1new, q2new)
        self.t += h

        if not (np.isfinite(self.q1).all() and np.isfinite(self.q2).all()):
            raise FloatingPointError(
                "Non-finite Q encountered. Reduce dt/activity or increase "
                "viscosity/friction."
            )

    def step(self, nsteps: int = 1) -> None:
        """Advance by ``nsteps`` using second-order ETD2 time integration."""
        for _ in range(nsteps):
            self._advance_one(self.p.dt)

    # ---------- output / restart ----------
    def snapshot(self) -> Dict[str, Array]:
        u, v = self.velocity()
        fx, fy = self.active_force()
        h1, h2 = self.molecular_field()
        return {
            "t": np.array(self.t),
            "q1": self.q1.copy(),
            "q2": self.q2.copy(),
            "u_x": u,
            "u_y": v,
            # Compatibility aliases used by the original exploratory plotter.
            "u": u,
            "v": v,
            "speed": np.hypot(u, v),
            "pressure": self.pressure(),
            "vorticity": self.vorticity(u, v),
            "active_force_x": fx,
            "active_force_y": fy,
            "H1": h1,
            "H2": h2,
            "S": self.order_parameter(),
            "theta": self.director_angle(),
        }

    def state_dict(self) -> Dict[str, Array]:
        """Return the minimal restart state; derived fields remain reproducible."""
        return {
            "schema_version": np.asarray(1, dtype=np.int64),
            "t": np.asarray(self.t, dtype=np.float64),
            "q1": self.q1.copy(),
            "q2": self.q2.copy(),
            "seed": np.asarray(self.seed, dtype=np.int64),
            "params_json": np.asarray(json.dumps(asdict(self.p), sort_keys=True)),
        }

    def load_state_dict(self, state: Dict[str, Array]) -> None:
        """Restore ``t,q1,q2`` after validating the configured grid."""
        q1 = np.asarray(state["q1"], dtype=np.float64)
        q2 = np.asarray(state["q2"], dtype=np.float64)
        if q1.shape != (self.n, self.n) or q2.shape != q1.shape:
            raise ValueError(f"restart fields must both have shape {(self.n, self.n)}")
        if not (np.isfinite(q1).all() and np.isfinite(q2).all()):
            raise ValueError("restart fields must be finite")
        # Project away only the ambiguous Nyquist lines; all resolved modes are
        # otherwise preserved exactly.
        self.q1 = self._project_state(q1)
        self.q2 = self._project_state(q2)
        self.t = float(np.asarray(state["t"]))

    def save_state(self, path: str | Path) -> None:
        """Save a compressed, self-describing restart checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.state_dict())

    def _advance_to(self, target: float) -> None:
        """Advance exactly to ``target`` using full dt steps plus one remainder."""
        tol = 64.0 * np.finfo(float).eps * max(1.0, abs(target), abs(self.t))
        while self.t + self.p.dt < target - tol:
            self._advance_one(self.p.dt)
        rem = target - self.t
        if rem > tol:
            self._advance_one(rem)
        # Eliminate harmless accumulated roundoff in reported output times.
        self.t = float(target)

    def run(
        self,
        t_final: float,
        save_every: Optional[float] = None,
        output: Optional[str | Path] = None,
    ) -> Dict[str, Array]:
        """Run exactly to ``t_final``; optionally save a time bank to NPZ."""
        if t_final <= self.t:
            raise ValueError("t_final must exceed the current simulation time")

        if save_every is None:
            self._advance_to(float(t_final))
            return self.snapshot()

        if save_every < self.p.dt:
            raise ValueError("save_every must be >= dt")

        times = [self.t]
        q1s = [self.q1.copy()]
        q2s = [self.q2.copy()]
        next_save = self.t + save_every
        tol = 64.0 * np.finfo(float).eps * max(1.0, abs(t_final))

        while next_save < t_final - tol:
            self._advance_to(float(next_save))
            times.append(self.t)
            q1s.append(self.q1.copy())
            q2s.append(self.q2.copy())
            next_save += save_every

        if self.t < t_final - tol:
            self._advance_to(float(t_final))
        if abs(times[-1] - self.t) > tol:
            times.append(self.t)
            q1s.append(self.q1.copy())
            q2s.append(self.q2.copy())

        bank = {
            "schema_version": np.asarray(1, dtype=np.int64),
            "t": np.asarray(times),
            "q1": np.asarray(q1s),
            "q2": np.asarray(q2s),
            "box_size": np.array(self.Lbox),
            "dt": np.array(self.p.dt),
            "seed": np.asarray(self.seed, dtype=np.int64),
            "params_json": np.asarray(json.dumps(asdict(self.p), sort_keys=True)),
        }
        if output is not None:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output, **bank)
        return bank
