from pathlib import Path
import numpy as np
from matplotlib.patches import Polygon, Circle
from fide_style import *

OUT = Path(__file__).with_name("05_design_balance.pdf")
rng = np.random.default_rng(20260826)
fig, ax = setup_canvas(width=7.2, height=5.4, xlim=(-3.6, 3.6), ylim=(-2.7, 2.7))

# central stand / fulcrum
ax.plot([-2.5, 2.5], [0.72, 0.72], color=INK, lw=2.0, alpha=.78, zorder=6)
ax.add_patch(Circle((0,0.72), .14, facecolor=VIOLET, edgecolor=INK, lw=1.0, alpha=.8, zorder=8))
ax.plot([0,0], [0.6,-1.55], color=INK, lw=2.2, alpha=.72, zorder=5)
ax.add_patch(Polygon([(-.62,-1.65),(.62,-1.65),(0,-.72)], closed=True,
                     facecolor=LAVENDER, edgecolor=PURPLE, lw=1.4, alpha=.48, zorder=4))
ax.add_patch(Ellipse((0,-1.7), 1.55,.25, facecolor=LAVENDER,
                     edgecolor=PURPLE, lw=1.2, alpha=.35, zorder=3))

# suspension strings and pans
for sgn in [-1,1]:
    xh = 2.15*sgn
    ax.plot([xh, xh-0.58*sgn], [0.68,-0.75], color=INK, lw=1.0, alpha=.6, zorder=5)
    ax.plot([xh, xh+0.58*sgn], [0.68,-0.75], color=INK, lw=1.0, alpha=.6, zorder=5)
    panx = xh
    ax.add_patch(Polygon([(panx-0.72,-0.77),(panx+0.72,-0.77),(panx+0.53,-1.02),(panx-0.53,-1.02)],
                         closed=True, facecolor=LAVENDER, edgecolor=PURPLE,
                         lw=1.2, alpha=.4, zorder=4))

# left: information / distribution shape in pan
xx = np.linspace(-3.0,-1.35,150)
for mu, sig, a in [(-2.45,.28,.32),(-2.05,.42,.26),(-2.7,.55,.18)]:
    yy = -0.79 + .63*np.exp(-0.5*((xx-mu)/sig)**2)
    ax.fill_between(xx, -0.78, yy, color=PURPLE, alpha=a, linewidth=0, zorder=6)
    ax.plot(xx, yy, color=PURPLE, lw=1.0, alpha=.7, zorder=7)
for _ in range(12):
    xp = rng.uniform(-2.85,-1.55)
    yp = -0.62 + rng.uniform(0,.62)*np.exp(-((xp+2.3)/.7)**2)
    ax.scatter([xp],[yp],s=rng.uniform(8,22),c=PURPLE,alpha=.75,
               edgecolors=INK,linewidths=.3,zorder=8)
lightbulb_icon(ax, -2.15, 1.45, scale=1.45, color=PURPLE)

# right: dynamic feasibility / flow field in pan and above
rx = np.linspace(1.35,3.0,130)
for off in [-.17,0,.17]:
    ry = -0.62 + .12*np.sin(5*(rx-1.25))+off
    ax.plot(rx, ry, color=GREEN, lw=1.35 if off==0 else .8,
            alpha=.8 if off==0 else .5, zorder=7)
flow_icon(ax, 2.15, 1.42, scale=1.45, color=GREEN)

save(fig, OUT)
print(OUT)
