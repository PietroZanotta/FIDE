"""Topological-defect extraction and lightweight tracking for a 2-D Q field.

For

    Q = [[q1, q2],
         [q2,-q1]],

psi = q1 + i q2 has phase 2*theta, where theta is the head-tail-symmetric
nematic director angle. A +/-1/2 nematic defect therefore gives winding +/-1
of psi around a small closed loop.

This module provides:
  * topological charge from periodic plaquette winding,
  * sub-grid core localization by solving the bilinear q1=q2=0 problem,
  * one signed annular texture-phase fit for both charges,
  * conservative frame-to-frame tracking with births/deaths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment

Array = np.ndarray


@dataclass
class Defect:
    x: float
    y: float
    charge: float
    polarity: Optional[float] = None          # +1/2 vector angle, radians
    polarity_coherence: Optional[float] = None
    triatic_orientation: Optional[float] = None  # -1/2 arm angle modulo 2pi/3
    triatic_coherence: Optional[float] = None
    orientation_phase_beta: Optional[float] = None
    orientation_coherence: Optional[float] = None
    core_residual: float = np.nan
    i: int = -1
    j: int = -1
    track_id: Optional[int] = None

    @property
    def plaquette_index(self) -> tuple[int, int]:
        return self.i, self.j

    @property
    def triatic_arm_angle(self) -> Optional[float]:
        """One of the three equivalent -1/2 arms, modulo ``2*pi/3``."""
        if self.charge >= 0.0 or self.orientation_phase_beta is None:
            return None
        return float((self.orientation_phase_beta / 3.0) % (2.0 * np.pi / 3.0))


@dataclass(frozen=True)
class SignedTextureFit:
    """Branch-free phase fit for ``arg(q1+i q2) ~= winding*phi + beta``."""

    beta: float
    coherence: float
    sample_count: int


def _wrap_pi(a: Array | float) -> Array | float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _periodic_delta(a: Array | float, b: float, L: float) -> Array | float:
    """Shortest signed displacement a-b on a periodic interval of length L."""
    return (a - b + 0.5 * L) % L - 0.5 * L


def plaquette_winding(q1: Array, q2: Array) -> Array:
    """Integer winding of psi=q1+i q2 on every periodic plaquette.

    +1 -> +1/2 nematic defect, -1 -> -1/2 nematic defect.
    """
    if q1.shape != q2.shape or q1.ndim != 2 or q1.shape[0] != q1.shape[1]:
        raise ValueError("q1 and q2 must be same-shape square 2-D arrays")

    phi = np.arctan2(q2, q1)
    p00 = phi
    p10 = np.roll(phi, -1, axis=0)
    p11 = np.roll(p10, -1, axis=1)
    p01 = np.roll(phi, -1, axis=1)

    total = (
        _wrap_pi(p10 - p00)
        + _wrap_pi(p11 - p10)
        + _wrap_pi(p01 - p11)
        + _wrap_pi(p00 - p01)
    )
    return np.rint(total / (2.0 * np.pi)).astype(int)


def _bilinear_value_and_jac(corners: Tuple[float, float, float, float], a: float, b: float):
    """Bilinear value and derivatives in local cell coordinates a,b in [0,1].

    corners are (f00, f10, f11, f01).
    """
    f00, f10, f11, f01 = corners
    val = (1-a)*(1-b)*f00 + a*(1-b)*f10 + a*b*f11 + (1-a)*b*f01
    da = -(1-b)*f00 + (1-b)*f10 + b*f11 - b*f01
    db = -(1-a)*f00 - a*f10 + a*f11 + (1-a)*f01
    return val, da, db


def _refine_core_bilinear(q1: Array, q2: Array, i: int, j: int, L: float) -> Tuple[float, float, float]:
    """Refine a winding-cell core by a bounded bilinear root solve.

    The final coordinate is always an optimized sub-grid point.  In particular,
    failure of an unconstrained Newton step does not silently substitute the
    plaquette center.
    """
    n = q1.shape[0]
    dx = L / n
    ip, jp = (i + 1) % n, (j + 1) % n

    c1 = (q1[i,j], q1[ip,j], q1[ip,jp], q1[i,jp])
    c2 = (q2[i,j], q2[ip,j], q2[ip,jp], q2[i,jp])

    def residual(z: Array) -> Array:
        f1, _, _ = _bilinear_value_and_jac(c1, float(z[0]), float(z[1]))
        f2, _, _ = _bilinear_value_and_jac(c2, float(z[0]), float(z[1]))
        return np.asarray([f1, f2], dtype=np.float64)

    def jacobian(z: Array) -> Array:
        _, f1a, f1b = _bilinear_value_and_jac(c1, float(z[0]), float(z[1]))
        _, f2a, f2b = _bilinear_value_and_jac(c2, float(z[0]), float(z[1]))
        return np.asarray([[f1a, f1b], [f2a, f2b]], dtype=np.float64)

    # Multiple deterministic starts handle distorted bilinear cells without
    # turning the reported coordinate into a start-dependent heuristic.
    starts = ((0.5, 0.5), (0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8))
    fits = [
        least_squares(
            residual,
            np.asarray(start, dtype=np.float64),
            jac=jacobian,
            bounds=(0.0, 1.0),
            xtol=1.0e-13,
            ftol=1.0e-13,
            gtol=1.0e-13,
            max_nfev=80,
        )
        for start in starts
    ]
    best = min(fits, key=lambda fit: float(np.linalg.norm(fit.fun)))
    a, b = map(float, best.x)
    f1, _, _ = _bilinear_value_and_jac(c1, a, b)
    f2, _, _ = _bilinear_value_and_jac(c2, a, b)
    residual = float(np.hypot(f1, f2))

    x = ((i + a) * dx) % L
    y = ((j + b) * dx) % L
    return x, y, residual


def fit_signed_texture_phase(
    q1: Array,
    q2: Array,
    x0: float,
    y0: float,
    charge: float,
    L: float,
    rmin_cells: float = 2.0,
    rmax_cells: float = 6.0,
) -> Optional[SignedTextureFit]:
    """Fit canonical defect texture on an annulus around a refined core.

    ``winding=2*charge`` is +1 or -1.  For +1/2, beta is the vector/comet
    polarity angle.  For -1/2, beta is a triatic texture phase; beta/3 is only
    a derived arm angle modulo 2*pi/3 and is not a polarity.
    """
    n = q1.shape[0]
    dx = L / n
    coords = np.arange(n) * dx
    X, Y = np.meshgrid(coords, coords, indexing="ij")
    rx = _periodic_delta(X, x0, L)
    ry = _periodic_delta(Y, y0, L)
    r = np.hypot(rx, ry)
    phi = np.arctan2(ry, rx)
    phase = np.arctan2(q2, q1)
    S = 2.0 * np.sqrt(q1*q1 + q2*q2)

    mask = (r >= rmin_cells*dx) & (r <= rmax_cells*dx) & (S > 1e-10)
    if np.count_nonzero(mask) < 8:
        return None

    winding = 1.0 if charge > 0.0 else -1.0
    z = np.exp(1j * (phase[mask] - winding * phi[mask]))

    # Down-weight the low-order inner region and emphasize the well-ordered annulus.
    w = S[mask]
    meanz = np.sum(w*z) / np.sum(w)
    coherence = float(np.abs(meanz))
    beta = float(np.angle(meanz) % (2.0 * np.pi))
    return SignedTextureFit(beta, coherence, int(np.count_nonzero(mask)))


def _texture_fit(
    q1: Array,
    q2: Array,
    x0: float,
    y0: float,
    charge: float,
    L: float,
    rmin_cells: float = 2.0,
    rmax_cells: float = 6.0,
) -> Tuple[Optional[float], Optional[float]]:
    """Backward-compatible display-angle adapter around the signed beta fit."""
    fit = fit_signed_texture_phase(
        q1, q2, x0, y0, charge, L, rmin_cells, rmax_cells
    )
    if fit is None:
        return None, None
    angle = fit.beta if charge > 0.0 else (fit.beta / 3.0) % (2.0 * np.pi / 3.0)
    return float(angle), fit.coherence


def extract_defects(
    q1: Array,
    q2: Array,
    box_size: float,
    min_core_order: Optional[float] = None,
    fit_rmin_cells: float = 2.0,
    fit_rmax_cells: float = 6.0,
) -> List[Defect]:
    """Extract +/-1/2 defects from a periodic Q field."""
    winding = plaquette_winding(q1, q2)
    n = q1.shape[0]
    S = 2.0 * np.sqrt(q1**2 + q2**2)
    defects: List[Defect] = []

    for i, j in zip(*np.nonzero(winding)):
        w = int(winding[i, j])
        if abs(w) != 1:
            # |w|>1 means multiple charge in one cell / under-resolution.
            continue

        if min_core_order is not None:
            ip, jp = (i + 1) % n, (j + 1) % n
            if min(S[i,j], S[ip,j], S[ip,jp], S[i,jp]) > min_core_order:
                continue

        x, y, residual = _refine_core_bilinear(q1, q2, i, j, box_size)
        charge = 0.5 * w

        polarity = polarity_coherence = None
        triatic = triatic_coherence = None
        texture = fit_signed_texture_phase(
            q1, q2, x, y, charge, box_size,
            rmin_cells=fit_rmin_cells,
            rmax_cells=fit_rmax_cells,
        )
        beta = None if texture is None else texture.beta
        coh = None if texture is None else texture.coherence
        if charge > 0:
            polarity, polarity_coherence = beta, coh
        else:
            triatic = None if beta is None else (beta / 3.0) % (2.0 * np.pi / 3.0)
            triatic_coherence = coh

        defects.append(
            Defect(
                x=x,
                y=y,
                charge=charge,
                polarity=polarity,
                polarity_coherence=polarity_coherence,
                triatic_orientation=triatic,
                triatic_coherence=triatic_coherence,
                orientation_phase_beta=beta,
                orientation_coherence=coh,
                core_residual=residual,
                i=int(i),
                j=int(j),
            )
        )

    return defects


def periodic_distance(a: Defect, b: Defect, L: float) -> float:
    dx = _periodic_delta(a.x, b.x, L)
    dy = _periodic_delta(a.y, b.y, L)
    return float(np.hypot(dx, dy))


class DefectTracker:
    """Conservative one-frame tracker with explicit births and deaths.

    Matching is charge-preserving and uses a global Hungarian assignment under
    a maximum periodic displacement. Unmatched current defects are births;
    unmatched previous defects are deaths. No gap-closing is attempted.
    """

    def __init__(self, box_size: float, max_displacement: float):
        self.L = float(box_size)
        self.max_displacement = float(max_displacement)
        if self.L <= 0.0 or self.max_displacement <= 0.0:
            raise ValueError("box_size and max_displacement must be positive")
        self._next_id = 0
        self.previous: List[Defect] = []

    def update(self, current: List[Defect]) -> Dict[str, List[int]]:
        births: List[int] = []
        deaths: List[int] = []
        matched_prev = set()
        matched_cur = set()

        for sign in (+1, -1):
            prev_idx = [k for k,d in enumerate(self.previous) if np.sign(d.charge) == sign]
            cur_idx = [k for k,d in enumerate(current) if np.sign(d.charge) == sign]
            if not prev_idx or not cur_idx:
                continue
            C = np.array([
                [periodic_distance(self.previous[ip], current[ic], self.L) for ic in cur_idx]
                for ip in prev_idx
            ])
            # Invalid edges receive a penalty larger than any collection of valid
            # matches. Dummy rows/columns then let the Hungarian solve prefer an
            # unmatched birth/death over an over-threshold displacement.
            nr, nc = C.shape
            size = nr + nc
            unmatched = self.max_displacement + np.finfo(float).eps
            augmented = np.full((size, size), 4.0 * unmatched, dtype=float)
            augmented[:nr, :nc] = np.where(C <= self.max_displacement, C, 4.0 * unmatched)
            augmented[:nr, nc : nc + nr] = np.eye(nr) * unmatched + (1.0 - np.eye(nr)) * 4.0 * unmatched
            augmented[nr : nr + nc, :nc] = np.eye(nc) * unmatched + (1.0 - np.eye(nc)) * 4.0 * unmatched
            augmented[nr:, nc:] = 0.0
            rr, cc = linear_sum_assignment(augmented)
            for r, c in zip(rr, cc):
                if r < nr and c < nc and C[r, c] <= self.max_displacement:
                    ip, ic = prev_idx[r], cur_idx[c]
                    current[ic].track_id = self.previous[ip].track_id
                    matched_prev.add(ip)
                    matched_cur.add(ic)

        for k, d in enumerate(current):
            if k not in matched_cur:
                d.track_id = self._next_id
                births.append(self._next_id)
                self._next_id += 1

        for k, d in enumerate(self.previous):
            if k not in matched_prev and d.track_id is not None:
                deaths.append(d.track_id)

        self.previous = current
        return {"births": births, "deaths": deaths}


def defects_to_array(defects: Iterable[Defect]) -> Array:
    """Columns: x,y,charge,beta,coherence,polarity,triatic_arm,
    core_residual,plaquette_i,plaquette_j,track_id.
    """
    rows = []
    for d in defects:
        rows.append([
            d.x, d.y, d.charge,
            np.nan if d.orientation_phase_beta is None else d.orientation_phase_beta,
            np.nan if d.orientation_coherence is None else d.orientation_coherence,
            np.nan if d.polarity is None else d.polarity,
            np.nan if d.triatic_orientation is None else d.triatic_orientation,
            d.core_residual,
            d.i,
            d.j,
            np.nan if d.track_id is None else d.track_id,
        ])
    return np.asarray(rows, dtype=float) if rows else np.empty((0,11), dtype=float)
