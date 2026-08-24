from pathlib import Path
import numpy as np
from matplotlib.patches import Circle
from fide_style import *

OUT = Path(__file__).with_name("01_system_flow.pdf")
rng = np.random.default_rng(20260822)
fig, ax = setup_canvas(width=10.5, height=4.5, xlim=(0, 10.5), ylim=(-2.2, 2.2))

x = np.linspace(0.45, 10.1, 260)
center = 0.28*np.sin(1.28*x) + 0.12*np.sin(2.7*x+0.5)
width = 1.45*np.exp(-0.32*x) + 0.32

# soft watercolor body
watercolor_band(ax, x, center, lambda _: width, LAVENDER, rng,
                layers=65, alpha=0.04, lw_range=(0.6, 2.5), jitter=0.18)
watercolor_band(ax, x, center-0.05, lambda _: 0.72*width, BLUE, rng,
                layers=34, alpha=0.035, lw_range=(0.5, 1.8), jitter=0.15)
watercolor_band(ax, x, center+0.02, lambda _: 0.55*width, GREEN, rng,
                layers=28, alpha=0.03, lw_range=(0.5, 1.6), jitter=0.12)

# principal trajectories
for phase, col, lw, a in [(0.0, PURPLE, 2.2, 0.9), (0.55, BLUE, 1.3, 0.7), (-0.42, TEAL, 1.2, 0.65)]:
    y = center + 0.23*np.sin(0.95*x + phase)*np.clip(width, 0.3, None)
    ax.plot(x, y, color=col, lw=lw, alpha=a, solid_capstyle="round", zorder=4)

# dotted centerline
xd = np.linspace(0.7, 10.0, 58)
yd = np.interp(xd, x, center)
ax.scatter(xd, yd, s=np.linspace(6, 18, len(xd)), c=PURPLE, alpha=0.75, linewidths=0, zorder=6)

# particles: large diffuse population at the left, sparse downstream
n = 180
xp = rng.beta(1.2, 2.6, n)*4.4 + 0.25
yp_center = np.interp(xp, x, center)
yp_width = np.interp(xp, x, width)
yp = yp_center + rng.normal(0, 0.55, n)*yp_width
sizes = rng.uniform(7, 34, n)
cols = [PURPLE if v < 0.75 else (BLUE if v < 0.9 else GREEN) for v in rng.random(n)]
ax.scatter(xp, yp, s=sizes, c=cols, alpha=rng.uniform(0.35,0.86,n),
           linewidths=0.45, edgecolors=INK, zorder=7)

# a few downstream particles
for xx in np.linspace(5.6, 9.5, 13):
    yy = np.interp(xx, x, center) + rng.normal(0, 0.18)
    rr = rng.uniform(0.028, 0.065)
    ax.add_patch(Circle((xx, yy), rr, facecolor=TEAL, edgecolor=INK,
                        lw=0.45, alpha=0.72, zorder=7))

save(fig, OUT)
print(OUT)
