"""Minimal 2-D periodic active-nematic solver.

Model
-----
We evolve a symmetric traceless 2-D nematic tensor

    Q = [[q1, q2],
         [q2,-q1]]

with Beris-Edwards dynamics coupled to an incompressible, overdamped
Stokes flow.  The implementation follows the simplified active-nematic
model

    d_t Q + u.grad Q = S(Q, grad u) - (1/gamma) [delta F/delta Q]^TS
    eta Lap u - friction u = grad p + div(alpha Q)
    div u = 0

with

    F = int (A |Q|^2 + C |Q|^4 + L |grad Q|^2) dx.

The Stokes equation is solved exactly mode-by-mode in Fourier space.
The elastic Laplacian in Q is treated semi-implicitly; all nonlinear
terms are explicit Euler.  This is intended as a transparent research
prototype / training-bank generator, not a production CFD solver.

References for the model form are given in the accompanying ChatGPT
response; see in particular the Beris-Edwards/Stokes equations in
Velez-Ceron et al., arXiv:2409.15479.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class ActiveNematicParams:
    """Physical and numerical parameters for :class:`ActiveNematic2D`.

    ``activity`` is the coefficient ``alpha`` in
    ``sigma_active = alpha Q``.  The force in the momentum balance is
    ``div(sigma_active)``; changing the sign of ``activity`` reverses that
    forcing.  ``gamma`` is the rotational viscosity, while ``friction`` is the
    substrate/screening coefficient ``gamma_f``.
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
        if self.friction == 0.0:
            raise ValueError("the initial screened-Stokes benchmark requires friction > 0")
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

        # Fourier wave numbers in physical units.
        k = 2.0 * np.pi * np.fft.fftfreq(self.n, d=self.dx)
        self.kx, self.ky = np.meshgrid(k, k, indexing="ij")
        self.k2 = self.kx**2 + self.ky**2
        self.nonzero = self.k2 > 0.0

        # Standard 2/3 de-aliasing mask for nonlinear products.
        mode = np.fft.fftfreq(self.n) * self.n
        mx, my = np.meshgrid(mode, mode, indexing="ij")
        cutoff = self.n / 3.0
        self.dealias = (np.abs(mx) <= cutoff) & (np.abs(my) <= cutoff)

        x = np.linspace(0.0, self.Lbox, self.n, endpoint=False)
        self.x, self.y = np.meshgrid(x, x, indexing="ij")

        self.q1, self.q2 = self._initial_condition()
        self.t = 0.0

    # ---------- spectral helpers ----------
    def _fft(self, a: Array) -> Array:
        return np.fft.fft2(a)

    def _ifft_real(self, ah: Array) -> Array:
        return np.fft.ifft2(ah).real

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
        # For the chosen free energy, uniform equilibrium has S0=sqrt(-A/C).
        S0 = np.sqrt(max(-p.A / p.C, 1e-12))
        theta = p.init_angle_noise * self._smooth_random_field(p.init_smoothing_length)
        amp = S0 * (
            1.0 + p.init_amplitude_noise * self._smooth_random_field(p.init_smoothing_length)
        )
        amp = np.maximum(0.15 * S0, amp)

        # Q = S (nn - I/2): q1=(S/2)cos(2theta), q2=(S/2)sin(2theta).
        q1 = 0.5 * amp * np.cos(2.0 * theta)
        q2 = 0.5 * amp * np.sin(2.0 * theta)
        return q1, q2

    # ---------- physical fields ----------
    def order_parameter(self) -> Array:
        """Return scalar nematic order S = 2 sqrt(q1^2 + q2^2)."""
        return 2.0 * np.sqrt(self.q1**2 + self.q2**2)

    def director_angle(self) -> Array:
        """Return director angle theta in [-pi/2, pi/2)."""
        return 0.5 * np.arctan2(self.q2, self.q1)

    def velocity(self) -> Tuple[Array, Array]:
        """Solve incompressible screened Stokes equation in Fourier space.

        0 = -grad p + eta Lap u - friction u + div(activity * Q),
        div u = 0.

        The k=0 velocity is fixed to zero.
        """
        p = self.p
        q1h = self._fft(self.q1)
        q2h = self._fft(self.q2)

        # f_i = d_j (alpha Q_ij).
        # Qxx=q1, Qxy=q2, Qyy=-q1.
        fxh = 1j * p.activity * (self.kx * q1h + self.ky * q2h)
        fyh = 1j * p.activity * (self.kx * q2h - self.ky * q1h)

        # Leray projection P = I - kk^T/k^2 removes pressure.
        kdotf = self.kx * fxh + self.ky * fyh
        pfx = fxh.copy()
        pfy = fyh.copy()
        pfx[self.nonzero] -= self.kx[self.nonzero] * kdotf[self.nonzero] / self.k2[self.nonzero]
        pfy[self.nonzero] -= self.ky[self.nonzero] * kdotf[self.nonzero] / self.k2[self.nonzero]
        pfx[~self.nonzero] = 0.0
        pfy[~self.nonzero] = 0.0

        denom = p.viscosity * self.k2 + p.friction
        uh = np.zeros_like(pfx)
        vh = np.zeros_like(pfy)
        # Fourier transform of the declared momentum balance gives
        # (eta k^2 + friction) u_hat = P f_hat.
        uh[self.nonzero] = pfx[self.nonzero] / denom[self.nonzero]
        vh[self.nonzero] = pfy[self.nonzero] / denom[self.nonzero]
        # If friction > 0, the zero mode is still set to zero by gauge choice.

        return self._ifft_real(uh), self._ifft_real(vh)

    def active_force(self) -> Tuple[Array, Array]:
        """Return f_active = div(activity * Q).

        This is the direct force density supplied by the active nematic stress
        before the incompressibility/pressure projection.
        """
        a = self.p.activity
        fx = a * (self.derivative(self.q1, 0) + self.derivative(self.q2, 1))
        fy = a * (self.derivative(self.q2, 0) - self.derivative(self.q1, 1))
        return fx, fy

    def pressure(self) -> Array:
        """Reconstruct the zero-mean Stokes pressure from the active force.

        In ``0=-grad p+eta Lap u-friction u+f``, the longitudinal force is
        canceled by ``grad p``.  The k=0 pressure gauge is fixed to zero.
        """
        q1h = self._fft(self.q1)
        q2h = self._fft(self.q2)
        fxh = 1j * self.p.activity * (self.kx * q1h + self.ky * q2h)
        fyh = 1j * self.p.activity * (self.kx * q2h - self.ky * q1h)
        kdotf = self.kx * fxh + self.ky * fyh
        ph = np.zeros_like(kdotf)
        ph[self.nonzero] = -1j * kdotf[self.nonzero] / self.k2[self.nonzero]
        ph[~self.nonzero] = 0.0
        return self._ifft_real(ph)

    def molecular_field(self) -> Tuple[Array, Array]:
        """Return H=-delta F/delta Q in the q1,q2 representation.

        The solver advances Q with H/gamma plus flow/advection terms.
        """
        p = self.p
        qnorm2 = 2.0 * (self.q1**2 + self.q2**2)
        bulk = -(2.0 * p.A + 4.0 * p.C * qnorm2)
        h1 = bulk * self.q1 + 2.0 * p.elastic_L * self.laplacian(self.q1)
        h2 = bulk * self.q2 + 2.0 * p.elastic_L * self.laplacian(self.q2)
        return h1, h2

    def vorticity(self, u: Optional[Array] = None, v: Optional[Array] = None) -> Array:
        if u is None or v is None:
            u, v = self.velocity()
        return self.derivative(v, 0) - self.derivative(u, 1)

    def _beris_edwards_S(self, u: Array, v: Array) -> Tuple[Array, Array]:
        """Compute the q1,q2 components of generalized tensor advection S."""
        lam = self.p.flow_alignment

        ux = self.derivative(u, 0)
        uy = self.derivative(u, 1)
        vx = self.derivative(v, 0)
        vy = self.derivative(v, 1)

        # E=(grad u + grad u^T)/2, Omega=(grad u - grad u^T)/2.
        Exx = ux
        Eyy = vy
        Exy = 0.5 * (uy + vx)
        Omxy = 0.5 * (uy - vx)

        # B = Q + I/2.
        Bxx = self.q1 + 0.5
        Bxy = self.q2
        Byy = -self.q1 + 0.5

        # Aplus = lambda E + Omega, Aminus = lambda E - Omega.
        Ap_xx = lam * Exx
        Ap_xy = lam * Exy + Omxy
        Ap_yx = lam * Exy - Omxy
        Ap_yy = lam * Eyy

        Am_xx = lam * Exx
        Am_xy = lam * Exy - Omxy
        Am_yx = lam * Exy + Omxy
        Am_yy = lam * Eyy

        # M = Aplus B + B Aminus.
        Mxx = Ap_xx * Bxx + Ap_xy * Bxy + Bxx * Am_xx + Bxy * Am_yx
        Mxy = Ap_xx * Bxy + Ap_xy * Byy + Bxx * Am_xy + Bxy * Am_yy
        Myx = Ap_yx * Bxx + Ap_yy * Bxy + Bxy * Am_xx + Byy * Am_yx
        Myy = Ap_yx * Bxy + Ap_yy * Byy + Bxy * Am_xy + Byy * Am_yy

        # grad(u):Q = E:Q because Omega:Q = 0 for symmetric Q.
        contraction = ux * self.q1 + uy * self.q2 + vx * self.q2 - vy * self.q1
        Mxx -= 2.0 * lam * Bxx * contraction
        Mxy -= 2.0 * lam * Bxy * contraction
        Myx -= 2.0 * lam * Bxy * contraction
        Myy -= 2.0 * lam * Byy * contraction

        # Numerical TS projection. q1 is (Sxx-Syy)/2; q2=(Sxy+Syx)/2.
        s1 = 0.5 * (Mxx - Myy)
        s2 = 0.5 * (Mxy + Myx)
        return s1, s2

    def step(self, nsteps: int = 1) -> None:
        """Advance by nsteps using semi-implicit Euler for Q diffusion."""
        p = self.p
        diff_coeff = 2.0 * p.elastic_L / p.gamma
        denom = 1.0 + p.dt * diff_coeff * self.k2

        for _ in range(nsteps):
            u, v = self.velocity()
            s1, s2 = self._beris_edwards_S(u, v)

            q1x, q1y = self.derivative(self.q1, 0), self.derivative(self.q1, 1)
            q2x, q2y = self.derivative(self.q2, 0), self.derivative(self.q2, 1)
            adv1 = u * q1x + v * q1y
            adv2 = u * q2x + v * q2y

            # |Q|^2 = Tr(Q^T Q) = 2(q1^2+q2^2).
            qnorm2 = 2.0 * (self.q1**2 + self.q2**2)

            # -1/gamma * delta F/delta Q, excluding the Laplacian part
            # handled implicitly. For F=A|Q|^2+C|Q|^4+L|grad Q|^2:
            # delta F/delta Q = 2A Q + 4C|Q|^2 Q - 2L Lap Q.
            bulk_factor = -(2.0 * p.A + 4.0 * p.C * qnorm2) / p.gamma
            rhs1 = -adv1 + s1 + bulk_factor * self.q1
            rhs2 = -adv2 + s2 + bulk_factor * self.q2

            q1h = (self._fft(self.q1) + p.dt * self._fft(rhs1)) / denom
            q2h = (self._fft(self.q2) + p.dt * self._fft(rhs2)) / denom
            q1h *= self.dealias
            q2h *= self.dealias

            self.q1 = self._ifft_real(q1h)
            self.q2 = self._ifft_real(q2h)
            self.t += p.dt

            if not (np.isfinite(self.q1).all() and np.isfinite(self.q2).all()):
                raise FloatingPointError(
                    "Non-finite Q encountered. Reduce dt/activity or increase viscosity/friction."
                )

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
        self.q1 = q1.copy()
        self.q2 = q2.copy()
        self.t = float(np.asarray(state["t"]))

    def save_state(self, path: str | Path) -> None:
        """Save a compressed, self-describing restart checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.state_dict())

    def run(
        self,
        t_final: float,
        save_every: Optional[float] = None,
        output: Optional[str | Path] = None,
    ) -> Dict[str, Array]:
        """Run to t_final; optionally save a time bank to a compressed NPZ."""
        if t_final <= self.t:
            raise ValueError("t_final must exceed the current simulation time")

        if save_every is None:
            nsteps = int(np.ceil((t_final - self.t) / self.p.dt))
            self.step(nsteps)
            return self.snapshot()

        if save_every < self.p.dt:
            raise ValueError("save_every must be >= dt")

        times, q1s, q2s = [], [], []
        next_save = self.t
        while self.t < t_final - 0.5 * self.p.dt:
            if self.t >= next_save - 0.5 * self.p.dt:
                times.append(self.t)
                q1s.append(self.q1.copy())
                q2s.append(self.q2.copy())
                next_save += save_every
            self.step(1)

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

