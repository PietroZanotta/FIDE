#!/usr/bin/env python3
"""Generate transparent vector-PDF snapshots for an abstract evolving scientific system.

Visual language: particle cloud + density contours + a sparse velocity field.
The dynamics use the standard time-dependent double-gyre vector field, which gives
an immediate vortex/transport cue while remaining visually generic.

Outputs (by default):
    unknown_system_t0.pdf
    unknown_system_t1.pdf
    unknown_system_t2.pdf

Dependencies:
    numpy, scipy, matplotlib
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# --------------------------- user controls ---------------------------
OUT_DIR = Path("unknown_system_pdfs")
SEED = 7
N_PARTICLES = 1800
SNAPSHOT_TIMES = (0.0, 5.0, 10.0)  # physical time tau
DT = 0.01

# Double-gyre parameters (standard form)
A = 0.10
EPS = 0.25
PERIOD = 10.0
OMEGA = 2.0 * np.pi / PERIOD

# Figure geometry. Every output has the same page size / coordinate frame.
FIGSIZE = (4.0, 2.0)
XLIM = (0.0, 2.0)
YLIM = (0.0, 1.0)

# Layer switches
SHOW_PARTICLES = True
SHOW_CONTOURS = True
SHOW_VELOCITY = True
FILL_CONTOURS = False

# Visual density. Colors are intentionally left to Matplotlib defaults.
PARTICLE_SIZE = 3.2
PARTICLE_ALPHA = 0.22
CONTOUR_LINEWIDTH = 1.15
CONTOUR_LEVEL_FRACTIONS = (0.16, 0.30, 0.48, 0.68, 0.86)
QUIVER_ALPHA = 0.50
QUIVER_GRID = (17, 9)
QUIVER_SCALE = 4.2
# --------------------------------------------------------------------


def double_gyre_velocity(xy, tau):
    """Velocity field on [0,2] x [0,1]. xy has shape (..., 2)."""
    x = xy[..., 0]
    y = xy[..., 1]

    a = EPS * np.sin(OMEGA * tau)
    b = 1.0 - 2.0 * a
    f = a * x**2 + b * x
    dfdx = 2.0 * a * x + b

    u = -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * y)
    v = np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * y) * dfdx
    return np.stack([u, v], axis=-1)


def rk4_step(xy, tau, dt):
    k1 = double_gyre_velocity(xy, tau)
    k2 = double_gyre_velocity(xy + 0.5 * dt * k1, tau + 0.5 * dt)
    k3 = double_gyre_velocity(xy + 0.5 * dt * k2, tau + 0.5 * dt)
    k4 = double_gyre_velocity(xy + dt * k3, tau + dt)
    out = xy + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # The analytic field is tangent to the box boundary. Clipping only removes
    # tiny numerical excursions from finite-step integration.
    out[..., 0] = np.clip(out[..., 0], *XLIM)
    out[..., 1] = np.clip(out[..., 1], *YLIM)
    return out


def sample_initial_population(rng, n):
    """A generic multi-population initial law with a small diffuse background."""
    centers = np.array([
        [0.42, 0.28],
        [0.76, 0.73],
        [1.28, 0.30],
        [1.62, 0.69],
    ])
    weights = np.array([0.27, 0.18, 0.225, 0.225])
    background_weight = 0.10
    weights = weights / weights.sum() * (1.0 - background_weight)

    counts = rng.multinomial(n, np.r_[weights, background_weight])
    chunks = []

    for center, count in zip(centers, counts[:-1]):
        # Rejection sampling avoids boundary pile-up.
        accepted = []
        while sum(len(a) for a in accepted) < count:
            batch = rng.normal(center, [0.075, 0.060], size=(max(64, count), 2))
            good = (
                (batch[:, 0] >= XLIM[0]) & (batch[:, 0] <= XLIM[1]) &
                (batch[:, 1] >= YLIM[0]) & (batch[:, 1] <= YLIM[1])
            )
            accepted.append(batch[good])
        chunks.append(np.concatenate(accepted, axis=0)[:count])

    n_bg = counts[-1]
    background = np.column_stack([
        rng.uniform(*XLIM, n_bg),
        rng.uniform(*YLIM, n_bg),
    ])
    chunks.append(background)
    return np.concatenate(chunks, axis=0)


def evolve_and_capture(x0, snapshot_times):
    """Integrate once and capture particles at requested times."""
    targets = sorted(snapshot_times)
    captures = {}
    xy = x0.copy()
    tau = 0.0

    if np.isclose(targets[0], 0.0):
        captures[targets[0]] = xy.copy()

    target_idx = 1 if np.isclose(targets[0], 0.0) else 0
    tmax = max(targets)

    while tau < tmax - 1e-12:
        step = min(DT, tmax - tau)
        xy = rk4_step(xy, tau, step)
        tau += step

        while target_idx < len(targets) and tau >= targets[target_idx] - 0.5*DT:
            captures[targets[target_idx]] = xy.copy()
            target_idx += 1

    return [captures[t] for t in snapshot_times]


def kde_on_grid(xy, nx=220, ny=120):
    gx = np.linspace(*XLIM, nx)
    gy = np.linspace(*YLIM, ny)
    X, Y = np.meshgrid(gx, gy)
    pts = np.vstack([X.ravel(), Y.ravel()])
    kde = gaussian_kde(xy.T, bw_method=0.16)
    Z = kde(pts).reshape(X.shape)
    return X, Y, Z


def add_velocity_layer(ax, tau):
    nx, ny = QUIVER_GRID
    # Slight inset keeps arrowheads away from the crop edge.
    xs = np.linspace(XLIM[0] + 0.08, XLIM[1] - 0.08, nx)
    ys = np.linspace(YLIM[0] + 0.07, YLIM[1] - 0.07, ny)
    X, Y = np.meshgrid(xs, ys)
    V = double_gyre_velocity(np.stack([X, Y], axis=-1), tau)

    # Suppress very small arrows to keep the graphic sparse.
    speed = np.linalg.norm(V, axis=-1)
    mask = speed > np.quantile(speed, 0.22)
    ax.quiver(
        X[mask], Y[mask], V[..., 0][mask], V[..., 1][mask],
        angles="xy", scale_units="xy", scale=QUIVER_SCALE,
        width=0.0042, headwidth=3.2, headlength=4.0,
        alpha=QUIVER_ALPHA,
    )


def save_snapshot(xy, tau, output_path):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    if SHOW_CONTOURS:
        X, Y, Z = kde_on_grid(xy)
        levels = np.asarray(CONTOUR_LEVEL_FRACTIONS) * Z.max()
        if FILL_CONTOURS:
            ax.contourf(X, Y, Z, levels=np.r_[levels, Z.max() * 1.001], alpha=0.10)
        ax.contour(X, Y, Z, levels=levels, linewidths=CONTOUR_LINEWIDTH)

    if SHOW_PARTICLES:
        ax.scatter(
            xy[:, 0], xy[:, 1],
            s=PARTICLE_SIZE,
            alpha=PARTICLE_ALPHA,
            linewidths=0,
        )

    if SHOW_VELOCITY:
        add_velocity_layer(ax, tau)

    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Vector PDF, transparent page, fixed-size crop for easy later alignment.
    fig.savefig(
        output_path,
        format="pdf",
        transparent=True,
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    x0 = sample_initial_population(rng, N_PARTICLES)
    snapshots = evolve_and_capture(x0, SNAPSHOT_TIMES)

    for i, (tau, xy) in enumerate(zip(SNAPSHOT_TIMES, snapshots)):
        path = OUT_DIR / f"unknown_system_t{i}.pdf"
        save_snapshot(xy, tau, path)
        print(f"wrote {path}  (tau={tau:g})")


if __name__ == "__main__":
    main()
