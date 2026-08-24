from pathlib import Path
import numpy as np
from matplotlib.patches import Polygon, Circle
from fide_style import *

OUT = Path(__file__).with_name("02_measurements.pdf")
rng = np.random.default_rng(20260823)
fig, ax = setup_canvas(width=10.5, height=5.0, xlim=(0, 10.5), ylim=(-2.4, 2.4))

# incoming population / flow
x = np.linspace(0.2, 5.4, 180)
center = 0.12*np.sin(1.7*x) - 0.1
width = 0.9 + 0.22*np.sin(0.85*x+0.6)
watercolor_band(ax, x, center, lambda _: width, LAVENDER, rng,
                layers=48, alpha=0.045, lw_range=(0.6,2.1), jitter=0.18)
watercolor_band(ax, x, center-0.05, lambda _: 0.55*width, TEAL, rng,
                layers=22, alpha=0.035, lw_range=(0.5,1.5), jitter=0.14)

# particles before measurement
n = 125
xp = rng.uniform(0.4, 5.1, n)
yp = np.interp(xp, x, center) + rng.normal(0, 0.42, n)
cols = [PURPLE if v < .75 else BLUE for v in rng.random(n)]
ax.scatter(xp, yp, s=rng.uniform(7,30,n), c=cols,
           alpha=rng.uniform(.35,.85,n), edgecolors=INK, linewidths=.35, zorder=6)

# measurement plane
plane_x = 5.65
plane = Polygon([(plane_x, -2.0), (plane_x+0.72,-1.55),
                 (plane_x+0.72,1.72), (plane_x,2.12)], closed=True,
                facecolor=LAVENDER, edgecolor=PURPLE, alpha=0.22, lw=1.7, zorder=7)
ax.add_patch(plane)
for yy in np.linspace(-1.45,1.55,5):
    arrow(ax, (5.15, yy*0.76), (5.72, yy*0.78), color=PURPLE,
          lw=1.1, alpha=0.8, mutation_scale=9, zorder=8)

# particles sampled on the plane
for _ in range(22):
    xx = rng.uniform(5.76, 6.18)
    yy = rng.uniform(-1.35, 1.45)
    ax.add_patch(Circle((xx,yy), rng.uniform(0.025,0.055),
                        facecolor=PURPLE, edgecolor=INK, lw=.35, alpha=.72, zorder=9))

# four clean measurement channels / summaries
levels = [1.55, 0.55, -0.45, -1.45]
for i, yy in enumerate(levels):
    col = PURPLE if i < 2 else (BLUE if i == 2 else TEAL)
    arrow(ax, (6.35, yy), (8.25, yy), color=col, lw=1.5,
          alpha=0.8, mutation_scale=11, zorder=7)

# icons, all graphical; no labels
histogram_icon(ax, 9.15, 1.30, sx=.85, sy=.58, color=PURPLE)
bell_curve_icon(ax, 9.15, 0.35, sx=.78, sy=.48, color=PURPLE)
dot_grid_icon(ax, 9.15, -0.50, sx=.92, sy=.72, color=BLUE)
radial_icon(ax, 9.15, -1.55, r=.42, color=PURPLE)

save(fig, OUT)
print(OUT)
