from pathlib import Path
import numpy as np
from matplotlib.patches import Ellipse
from fide_style import *

OUT = Path(__file__).with_name("03_moment_fiber.pdf")
rng = np.random.default_rng(20260824)
fig, ax = setup_canvas(width=6.0, height=6.2, xlim=(-3.2, 3.2), ylim=(-3.2, 3.2))

# ruled hyperboloid / hourglass-like compatible-law manifold
zvals = np.linspace(-2.75, 2.75, 46)
for z in zvals:
    w = 0.72 + 0.16*z*z
    alpha = 0.10 + 0.04*(abs(z)/2.75)
    ax.add_patch(Ellipse((0,z), 2*w, 0.34 + 0.05*abs(z),
                         facecolor="none", edgecolor=VIOLET,
                         lw=0.62, alpha=alpha, zorder=1))

# diagonal ruling lines
for side in [-1,1]:
    for u in np.linspace(-1,1,27):
        z = np.linspace(-2.75, 2.75, 180)
        w = 0.72 + 0.16*z*z
        x = side*w*(0.62 + 0.38*np.cos(0.7*z + u*1.4))
        # slight criss-cross to evoke fiber geometry
        x += 0.20*u*np.sin(0.9*z)
        ax.plot(x, z, color=VIOLET, lw=0.55, alpha=0.18, zorder=1)

# emphasized central moment fiber
add_soft_ellipse(ax, (0,0), 3.55, 0.78, PURPLE, alpha=0.13, edge_alpha=0.85, lw=1.6, zorder=4)
for _ in range(72):
    # uniform-ish cloud in elliptical disk
    r = np.sqrt(rng.random())
    th = rng.uniform(0, 2*np.pi)
    xx = 1.55*r*np.cos(th)
    yy = 0.27*r*np.sin(th)
    ax.scatter([xx],[yy], s=rng.uniform(8,22), c=PURPLE,
               alpha=rng.uniform(.45,.9), edgecolors=INK, linewidths=.35, zorder=6)

# subtle top/bottom rims
for z in [-2.78,2.78]:
    ax.add_patch(Ellipse((0,z), 5.2, .52, facecolor="none",
                         edgecolor=VIOLET, lw=1.0, alpha=.38, zorder=2))

save(fig, OUT)
print(OUT)
