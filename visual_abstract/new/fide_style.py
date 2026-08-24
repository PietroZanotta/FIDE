from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Ellipse, Circle, Polygon, FancyArrowPatch, Arc

# Shared palette chosen to mimic the purple / blue / green watercolor aesthetic.
PURPLE = "#5B35A8"
VIOLET = "#7A63C7"
LAVENDER = "#C9C0EB"
BLUE = "#4C8BB8"
TEAL = "#2C8F86"
GREEN = "#5C9B66"
PALE_GREEN = "#CFE3D2"
INK = "#262B42"
LIGHT = "#F7F5FB"


def setup_canvas(width=10, height=4, xlim=(0, 10), ylim=(-2, 2)):
    """Create a transparent, text-free canvas."""
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "path.simplify": False,
    })
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("auto")
    ax.axis("off")
    return fig, ax


def save(fig, filename):
    """Save as a transparent vector PDF with tight bounds."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        filename,
        format="pdf",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def cubic_path(points, **kwargs):
    """Create a single cubic Bezier path from four 2D control points."""
    p = np.asarray(points, dtype=float)
    path = MplPath(p, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    return PathPatch(path, fill=False, **kwargs)


def poly_bezier(xs, ys, color=PURPLE, lw=1.0, alpha=0.35, zorder=2):
    """Smooth-ish polyline using a dense interpolation; suitable for soft flow fields."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    t = np.linspace(0, 1, len(xs))
    ti = np.linspace(0, 1, 220)
    # Low-order polynomial interpolation is stable for our small point counts.
    deg = min(5, len(xs) - 1)
    yy = np.polyval(np.polyfit(t, ys, deg), ti)
    xx = np.interp(ti, t, xs)
    return plt.Line2D(xx, yy, color=color, lw=lw, alpha=alpha,
                      solid_capstyle="round", zorder=zorder)


def watercolor_band(ax, x, center, width_fn, color, rng, layers=45,
                    alpha=0.045, lw_range=(0.5, 2.0), jitter=0.12, zorder=1):
    """Approximate watercolor with many translucent vector strokes."""
    x = np.asarray(x)
    base = np.asarray(center)
    width = np.asarray(width_fn(x) if callable(width_fn) else width_fn)
    for _ in range(layers):
        local = base + rng.normal(0, jitter, size=x.size) * width
        # smooth random perturbation
        kernel = np.ones(11) / 11
        local = np.convolve(local, kernel, mode="same")
        offset = rng.normal(0, 0.42) * width
        y = local + offset
        ax.plot(x, y, color=color,
                lw=rng.uniform(*lw_range), alpha=alpha,
                solid_capstyle="round", zorder=zorder)


def arrow(ax, start, end, color=INK, lw=1.8, alpha=0.9,
          mutation_scale=12, zorder=6, linestyle="-"):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=mutation_scale,
        linewidth=lw, color=color, alpha=alpha, linestyle=linestyle,
        shrinkA=0, shrinkB=0, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def add_soft_ellipse(ax, xy, width, height, color, alpha=0.12,
                     edge_alpha=0.7, lw=1.2, angle=0, zorder=2):
    e = Ellipse(xy, width, height, angle=angle,
                facecolor=color, edgecolor=color,
                alpha=alpha, lw=lw, zorder=zorder)
    ax.add_patch(e)
    # crisp edge separately, so interior remains soft
    edge = Ellipse(xy, width, height, angle=angle,
                   facecolor="none", edgecolor=color,
                   alpha=edge_alpha, lw=lw, zorder=zorder + 0.1)
    ax.add_patch(edge)
    return e


def scatter_particles(ax, x, y, rng, n=100, color=PURPLE,
                      size=(7, 28), alpha=(0.35, 0.9), zorder=5,
                      second_color=None):
    x = np.asarray(x)
    y = np.asarray(y)
    ids = rng.integers(0, len(x), n)
    xp = x[ids] + rng.normal(0, 0.03 * (x.max() - x.min()), n)
    yp = y[ids] + rng.normal(0, 0.15, n)
    sizes = rng.uniform(size[0], size[1], n)
    if second_color is None:
        colors = color
    else:
        mix = rng.random(n)
        colors = [color if m < 0.72 else second_color for m in mix]
    ax.scatter(xp, yp, s=sizes, c=colors, alpha=rng.uniform(alpha[0], alpha[1], n),
               linewidths=0.4, edgecolors=INK, zorder=zorder)


def bell_curve_icon(ax, x0, y0, sx=0.45, sy=0.35, color=PURPLE, zorder=8):
    x = np.linspace(-2.8, 2.8, 120)
    y = np.exp(-0.5 * x**2)
    ax.plot(x0 + sx * x / 2.8, y0 + sy * y,
            color=color, lw=1.6, zorder=zorder)
    ax.plot([x0 - sx, x0 + sx], [y0, y0], color=INK, lw=0.8, alpha=0.6, zorder=zorder)


def histogram_icon(ax, x0, y0, sx=0.55, sy=0.55, color=PURPLE, zorder=8):
    heights = np.array([0.25, 0.5, 0.9, 1.0, 0.72, 0.45, 0.22])
    w = sx / len(heights)
    for i, h in enumerate(heights):
        ax.add_patch(Polygon([
            (x0 - sx/2 + i*w, y0),
            (x0 - sx/2 + (i+0.78)*w, y0),
            (x0 - sx/2 + (i+0.78)*w, y0 + h*sy),
            (x0 - sx/2 + i*w, y0 + h*sy),
        ], closed=True, facecolor=color, edgecolor=color, alpha=0.36 + 0.05*i,
           lw=0.7, zorder=zorder))
    ax.plot([x0 - sx/2, x0 + sx/2], [y0, y0], color=INK, lw=0.8, alpha=0.6, zorder=zorder)


def dot_grid_icon(ax, x0, y0, sx=0.62, sy=0.45, color=PURPLE, zorder=8):
    gx, gy = np.meshgrid(np.linspace(-0.45, 0.45, 5), np.linspace(-0.3, 0.3, 4))
    r = np.sqrt(gx.ravel()**2 + gy.ravel()**2)
    sizes = 12 + 32*np.clip(0.6-r, 0, None)
    ax.scatter(x0 + sx*gx.ravel(), y0 + sy*gy.ravel(), s=sizes,
               facecolors=color, edgecolors=INK, linewidths=0.35,
               alpha=0.75, zorder=zorder)


def radial_icon(ax, x0, y0, r=0.34, color=PURPLE, zorder=8):
    for k, a in enumerate([0.05, 0.11, 0.18, 0.25, 0.33]):
        ax.add_patch(Circle((x0, y0), r*(1-k*0.15),
                            facecolor=color, edgecolor="none",
                            alpha=a, zorder=zorder+k*0.01))
    ax.add_patch(Circle((x0, y0), r, facecolor="none", edgecolor=color,
                        lw=1.0, alpha=0.65, zorder=zorder+1))


def lightbulb_icon(ax, x0, y0, scale=1.0, color=PURPLE, zorder=10):
    ax.add_patch(Circle((x0, y0+0.16*scale), 0.18*scale,
                        facecolor="none", edgecolor=color, lw=1.8, zorder=zorder))
    ax.plot([x0-0.09*scale, x0+0.09*scale], [y0-0.04*scale, y0-0.04*scale],
            color=color, lw=1.6, zorder=zorder)
    ax.plot([x0-0.07*scale, x0+0.07*scale], [y0-0.09*scale, y0-0.09*scale],
            color=color, lw=1.4, zorder=zorder)
    for ang in np.linspace(0.1, np.pi-0.1, 7):
        p1 = (x0 + 0.28*scale*np.cos(ang), y0+0.16*scale + 0.28*scale*np.sin(ang))
        p2 = (x0 + 0.38*scale*np.cos(ang), y0+0.16*scale + 0.38*scale*np.sin(ang))
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=1.0, alpha=0.75, zorder=zorder)


def flow_icon(ax, x0, y0, scale=1.0, color=GREEN, zorder=10):
    xs = np.linspace(-0.55, 0.55, 120)
    for off in [-0.16, 0, 0.16]:
        ys = 0.15*np.sin(2.2*np.pi*(xs+0.15)) + off
        ax.plot(x0+scale*xs, y0+scale*ys, color=color,
                lw=1.4 if off == 0 else 0.9,
                alpha=0.85 if off == 0 else 0.55, zorder=zorder)
    arrow(ax, (x0+0.30*scale, y0+0.14*scale),
          (x0+0.49*scale, y0+0.07*scale), color=color,
          lw=1.2, mutation_scale=9, zorder=zorder+1)
