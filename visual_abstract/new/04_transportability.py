from pathlib import Path
import numpy as np
from fide_style import *

OUT = Path(__file__).with_name("04_transportability.pdf")
rng = np.random.default_rng(20260825)
fig, ax = setup_canvas(width=10.5, height=4.4, xlim=(0, 10.5), ylim=(-2.15, 2.15))

x = np.linspace(0.35, 10.0, 240)
center = 0.42*np.sin(0.83*x+0.45)
width = 0.72 + 0.18*np.cos(0.55*x)
watercolor_band(ax, x, center, lambda _: width, PALE_GREEN, rng,
                layers=64, alpha=0.055, lw_range=(0.6,2.0), jitter=0.16)
watercolor_band(ax, x, center, lambda _: 0.55*width, GREEN, rng,
                layers=28, alpha=0.035, lw_range=(0.5,1.5), jitter=0.13)

# background vector flow: many short green arrows following local tangent
for yyoff in np.linspace(-0.65,0.65,7):
    for xx in np.linspace(1.0,9.2,9):
        y0 = np.interp(xx, x, center) + yyoff*0.7
        dydx = 0.42*0.83*np.cos(0.83*xx+0.45)
        dx = 0.38
        dy = dydx*dx
        arrow(ax, (xx, y0), (xx+dx, y0+dy), color=GREEN,
              lw=.75, alpha=.5, mutation_scale=7, zorder=3)

# reference path (dashed) and projected / corrected path (solid)
y_ref = center + 0.58*np.sin(0.42*x+0.65)
y_proj = center - 0.18*np.sin(0.73*x+1.5)
ax.plot(x, y_ref, color=VIOLET, lw=1.8, alpha=.75, ls=(0,(4,4)), zorder=7)
ax.plot(x, y_proj, color=PURPLE, lw=2.8, alpha=.95, zorder=8)

# sparse anchor points on projected path
for xx in [0.55, 2.0, 4.15, 6.35, 8.35, 9.75]:
    yy = np.interp(xx, x, y_proj)
    ax.scatter([xx],[yy], s=72, facecolors=LIGHT, edgecolors=PURPLE,
               linewidths=1.8, zorder=10)

# final direction arrow
arrow(ax, (9.15, np.interp(9.15,x,y_proj)),
      (10.05, np.interp(10.05,x,y_proj)), color=PURPLE,
      lw=2.3, mutation_scale=16, zorder=11)

save(fig, OUT)
print(OUT)
