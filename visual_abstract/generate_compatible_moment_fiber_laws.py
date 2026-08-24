#!/usr/bin/env python3
"""Generate transparent vector-PDF thumbnails of genuinely moment-compatible 2D laws.

Every constructed law is analytically standardized to satisfy

    E[X] = (0, 0)
    Cov(X) = I_2

so all outputs lie in the same moment fiber for

    Phi(x, y) = (x, y, x^2, x*y, y^2)
    c          = (0, 0, 1,   0,   1).

The laws nevertheless have very different higher-order / geometric structure.

Outputs:
    law_gaussian.pdf
    law_bimodal.pdf
    law_four_lobed.pdf
    law_ring.pdf
    law_skewed_trimodal.pdf
    law_crescent.pdf

Dependencies:
    numpy, scipy, matplotlib
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# --------------------------- user controls ---------------------------
OUT_DIR = Path("moment_fiber_laws")
SEED = 19
N_PARTICLES = 900
SHOW_PARTICLES = True
SHOW_CONTOURS = True
FILL_CONTOURS = False

FIGSIZE = (2.15, 2.15)
XLIM = (-3.0, 3.0)
YLIM = (-3.0, 3.0)
GRID_N = 260

PARTICLE_SIZE = 3.0
PARTICLE_ALPHA = 0.18
CONTOUR_LINEWIDTH = 1.15
CONTOUR_LEVEL_FRACTIONS = (0.12, 0.25, 0.43, 0.63, 0.82)
# --------------------------------------------------------------------


def mixture_moments(weights, means, covs):
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    means = np.asarray(means, float)
    covs = np.asarray(covs, float)

    mean = np.einsum("k,kd->d", weights, means)
    cov = np.zeros((2, 2))
    for w, mu, C in zip(weights, means, covs):
        d = mu - mean
        cov += w * (C + np.outer(d, d))
    return mean, cov


def inverse_sqrt_spd(C):
    vals, vecs = np.linalg.eigh(C)
    if np.any(vals <= 0):
        raise ValueError("Covariance must be positive definite.")
    return vecs @ np.diag(vals ** -0.5) @ vecs.T


def whiten_mixture(weights, means, covs):
    """Affine-standardize an arbitrary Gaussian mixture to mean 0, covariance I."""
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    means = np.asarray(means, float)
    covs = np.asarray(covs, float)

    m, C = mixture_moments(weights, means, covs)
    A = inverse_sqrt_spd(C)

    new_means = np.array([A @ (mu - m) for mu in means])
    new_covs = np.array([A @ S @ A.T for S in covs])
    return weights, new_means, new_covs


def iso_cov(sigma2, k):
    return np.repeat((sigma2 * np.eye(2))[None, :, :], k, axis=0)


def build_raw_laws():
    laws = {}

    # 1) Unimodal baseline.
    laws["gaussian"] = (
        [1.0],
        [[0.0, 0.0]],
        [np.eye(2)],
    )

    # 2) Strongly bimodal.
    laws["bimodal"] = (
        [0.5, 0.5],
        [[-1.5, 0.0], [1.5, 0.0]],
        [np.diag([0.16, 0.65]), np.diag([0.16, 0.65])],
    )

    # 3) Four separated lobes / cross geometry.
    a = 1.55
    four_means = [[a, 0], [-a, 0], [0, a], [0, -a]]
    laws["four_lobed"] = (
        np.full(4, 0.25),
        four_means,
        iso_cov(0.10, 4),
    )

    # 4) Ring-like law approximated by many narrow Gaussian components.
    K = 14
    theta = np.linspace(0, 2*np.pi, K, endpoint=False)
    r = 1.85
    ring_means = np.column_stack([r*np.cos(theta), r*np.sin(theta)])
    laws["ring"] = (
        np.full(K, 1.0 / K),
        ring_means,
        iso_cov(0.075, K),
    )

    # 5) Asymmetric / skewed trimodal structure.
    laws["skewed_trimodal"] = (
        [0.56, 0.29, 0.15],
        [[-1.25, -0.35], [1.10, -0.05], [0.20, 1.65]],
        [
            [[0.28, 0.12], [0.12, 0.20]],
            [[0.22, -0.08], [-0.08, 0.30]],
            [[0.13, 0.03], [0.03, 0.18]],
        ],
    )

    # 6) Curved / crescent-like law: a chain of local Gaussian components.
    theta = np.linspace(-0.95*np.pi, 0.35*np.pi, 9)
    crescent_means = np.column_stack([
        1.7*np.cos(theta),
        1.15*np.sin(theta),
    ])
    # Slightly nonuniform weights make it visibly non-Gaussian before whitening.
    crescent_weights = np.exp(-0.5 * ((np.arange(9) - 4.5) / 2.5)**2)
    crescent_weights /= crescent_weights.sum()
    crescent_covs = []
    for th in theta:
        # Local covariance elongated along the arc tangent.
        tangent = np.array([-np.sin(th), np.cos(th)])
        normal = np.array([np.cos(th), np.sin(th)])
        C = 0.16*np.outer(tangent, tangent) + 0.035*np.outer(normal, normal)
        crescent_covs.append(C)
    laws["crescent"] = (
        crescent_weights,
        crescent_means,
        crescent_covs,
    )

    return laws


def mixture_pdf(points, weights, means, covs):
    z = np.zeros(points.shape[0])
    for w, mu, C in zip(weights, means, covs):
        z += w * multivariate_normal(mean=mu, cov=C).pdf(points)
    return z


def sample_mixture(rng, n, weights, means, covs):
    comp = rng.choice(len(weights), size=n, p=weights)
    out = np.empty((n, 2))
    for k in range(len(weights)):
        idx = np.flatnonzero(comp == k)
        if len(idx):
            out[idx] = rng.multivariate_normal(means[k], covs[k], size=len(idx))
    return out


def analytic_moment_vector(weights, means, covs):
    m, C = mixture_moments(weights, means, covs)
    second = C + np.outer(m, m)
    # E[x], E[y], E[x^2], E[xy], E[y^2]
    return np.array([m[0], m[1], second[0, 0], second[0, 1], second[1, 1]])


def save_law(name, weights, means, covs, rng):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    gx = np.linspace(*XLIM, GRID_N)
    gy = np.linspace(*YLIM, GRID_N)
    X, Y = np.meshgrid(gx, gy)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    Z = mixture_pdf(pts, weights, means, covs).reshape(X.shape)
    levels = np.asarray(CONTOUR_LEVEL_FRACTIONS) * Z.max()

    if SHOW_CONTOURS:
        if FILL_CONTOURS:
            ax.contourf(X, Y, Z, levels=np.r_[levels, Z.max() * 1.001], alpha=0.10)
        ax.contour(X, Y, Z, levels=levels, linewidths=CONTOUR_LINEWIDTH)

    if SHOW_PARTICLES:
        samples = sample_mixture(rng, N_PARTICLES, weights, means, covs)
        ax.scatter(
            samples[:, 0], samples[:, 1],
            s=PARTICLE_SIZE,
            alpha=PARTICLE_ALPHA,
            linewidths=0,
        )

    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    path = OUT_DIR / f"law_{name}.pdf"
    fig.savefig(path, format="pdf", transparent=True, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    raw_laws = build_raw_laws()

    target = np.array([0.0, 0.0, 1.0, 0.0, 1.0])
    print("Target moment vector [E x, E y, E x^2, E xy, E y^2]:")
    print(target)
    print()

    for name, (weights, means, covs) in raw_laws.items():
        weights, means, covs = whiten_mixture(weights, means, covs)
        phi_mean = analytic_moment_vector(weights, means, covs)
        err = np.max(np.abs(phi_mean - target))
        if err > 1e-10:
            raise RuntimeError(f"{name}: moment standardization error {err:g}")

        path = save_law(name, weights, means, covs, rng)
        print(f"{name:16s}  moments={np.array2string(phi_mean, precision=6)}  -> {path}")


if __name__ == "__main__":
    main()
