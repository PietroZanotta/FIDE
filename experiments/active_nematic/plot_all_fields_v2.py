"""Comprehensive visualization for one 2-D active-nematic realization.

The script writes TWO figures:

1) <output>                    : all physical / diagnostic fields used by the model
2) <output_stem>_defects.png   : defect-focused global view + local texture gallery

The field figure contains
  q1=Qxx, q2=Qxy, scalar order S, director texture + defects,
  molecular-field components H1,H2 and |H|, director angle theta,
  fluid ux, uy, speed, vorticity,
  pressure, active-force components fx,fy and |f_active|.

Usage
-----
    python plot_all_fields_v2.py --bank outputs/run/physical_bank.npz --run 0 --frame -1
    python plot_all_fields_v2.py --n 96 --time 25 --output active_nematic_fields.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

try:
    from .active_nematic_solver import ActiveNematic2D, ActiveNematicParams
    from .defect_extractor import Defect, extract_defects
    from .domain import PhysicalBank
except ImportError:  # pragma: no cover - direct plotting-script convention.
    from active_nematic_solver import ActiveNematic2D, ActiveNematicParams
    from defect_extractor import Defect, extract_defects
    from domain import PhysicalBank

VISUALIZER_VERSION = "all-fields-v2"


def _robust_absmax(a: np.ndarray, pct: float = 99.0) -> float:
    m = float(np.percentile(np.abs(a[np.isfinite(a)]), pct))
    return max(m, 1e-12)


def _periodic_delta(a, b, L):
    return (a - b + 0.5 * L) % L - 0.5 * L


def _plot_signed(fig, ax, a, L, title, label, cmap="coolwarm"):
    vmax = _robust_absmax(a)
    im = ax.imshow(a.T, origin="lower", extent=(0,L,0,L), cmap=cmap,
                   vmin=-vmax, vmax=vmax, interpolation="bilinear")
    ax.set_title(title)
    ax.set_aspect("equal")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    cb.set_label(label)
    return im


def _plot_positive(fig, ax, a, L, title, label, cmap="viridis"):
    vmax = max(float(np.percentile(a[np.isfinite(a)], 99.0)), 1e-12)
    im = ax.imshow(a.T, origin="lower", extent=(0,L,0,L), cmap=cmap,
                   vmin=0.0, vmax=vmax, interpolation="bilinear")
    ax.set_title(title)
    ax.set_aspect("equal")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    cb.set_label(label)
    return im


def _draw_director(ax, x, y, theta, stride, seg, color="white", alpha=0.78, lw=0.65):
    xs = x[::stride, ::stride]
    ys = y[::stride, ::stride]
    th = theta[::stride, ::stride]
    dx = seg*np.cos(th)
    dy = seg*np.sin(th)
    for x0,y0,ddx,ddy in zip(xs.ravel(), ys.ravel(), dx.ravel(), dy.ravel()):
        ax.plot([x0-ddx,x0+ddx], [y0-ddy,y0+ddy], color=color, alpha=alpha, lw=lw)


def _draw_minus_arms(ax, d: Defect, radius=0.7, color="deepskyblue", lw=1.4):
    phi0 = 0.0 if d.triatic_orientation is None else d.triatic_orientation
    for k in range(3):
        a = phi0 + 2*np.pi*k/3
        ax.plot([d.x, d.x + radius*np.cos(a)], [d.y, d.y + radius*np.sin(a)],
                color=color, lw=lw, zorder=7)


def _overlay_defects(ax, defects: Iterable[Defect], arrow_length=0.95, labels=None):
    defects = list(defects)
    plus = [d for d in defects if d.charge > 0]
    minus = [d for d in defects if d.charge < 0]

    if plus:
        ax.scatter([d.x for d in plus], [d.y for d in plus], s=52,
                   facecolors="none", edgecolors="red", linewidths=1.4, zorder=6)
        for d in plus:
            if d.polarity is not None:
                ax.arrow(d.x, d.y,
                         arrow_length*np.cos(d.polarity), arrow_length*np.sin(d.polarity),
                         color="red", width=0.014, head_width=0.16,
                         length_includes_head=True, zorder=7)
    if minus:
        ax.scatter([d.x for d in minus], [d.y for d in minus], s=18,
                   c="deepskyblue", zorder=6)
        for d in minus:
            _draw_minus_arms(ax, d, radius=0.55)

    if labels:
        for d, text in labels:
            ax.text(d.x+0.25, d.y+0.25, text, fontsize=8, weight="bold",
                    color="white", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.55))
    return plus, minus


def _nearest_best(defects, n=4, positive=True):
    ds = [d for d in defects if (d.charge > 0) == positive]
    def score(d):
        c = d.polarity_coherence if positive else d.triatic_coherence
        return -1.0 if c is None else c
    return sorted(ds, key=score, reverse=True)[:n]


def _patch_data(a, d: Defect, L: float, half_cells: int):
    n = a.shape[0]
    dx = L/n
    i0 = int(np.round(d.x/dx)) % n
    j0 = int(np.round(d.y/dx)) % n
    ii_raw = np.arange(i0-half_cells, i0+half_cells+1)
    jj_raw = np.arange(j0-half_cells, j0+half_cells+1)
    ii = ii_raw % n
    jj = jj_raw % n
    patch = a[np.ix_(ii,jj)]
    xg = ii*dx
    yg = jj*dx
    xr = _periodic_delta(xg, d.x, L)
    yr = _periodic_delta(yg, d.y, L)
    Xr,Yr = np.meshgrid(xr,yr,indexing="ij")
    return patch, Xr, Yr


def _plot_defect_patch(ax, d: Defect, S, theta, L, half_cells=10):
    Sp,X,Y = _patch_data(S,d,L,half_cells)
    thp,_,_ = _patch_data(theta,d,L,half_cells)
    extent = (Y.min(), Y.max(), X.min(), X.max())  # replaced below by pcolormesh-free imshow extents
    # Arrays use [x,y], so transpose for display. Coordinates are nearly uniform around the core.
    xmin,xmax = X[:,0].min(), X[:,0].max()
    ymin,ymax = Y[0,:].min(), Y[0,:].max()
    ax.imshow(Sp.T, origin="lower", extent=(xmin,xmax,ymin,ymax), cmap="viridis",
              vmin=0.0, vmax=max(np.percentile(S,99),1e-12), interpolation="bilinear")

    stride = 2
    xr = X[::stride,::stride]
    yr = Y[::stride,::stride]
    th = thp[::stride,::stride]
    seg = 0.22*(xmax-xmin)/5.0
    dxl = seg*np.cos(th); dyl = seg*np.sin(th)
    for x0,y0,ddx,ddy in zip(xr.ravel(),yr.ravel(),dxl.ravel(),dyl.ravel()):
        ax.plot([x0-ddx,x0+ddx],[y0-ddy,y0+ddy],color="white",lw=0.8,alpha=0.85)

    if d.charge > 0:
        ax.scatter([0],[0],s=65,facecolors="none",edgecolors="red",linewidths=1.6,zorder=8)
        if d.polarity is not None:
            ell = 0.28*(xmax-xmin)
            ax.arrow(0,0,ell*np.cos(d.polarity),ell*np.sin(d.polarity),color="red",
                     width=0.025,head_width=0.18,length_includes_head=True,zorder=9)
        coh = d.polarity_coherence
        ttl = rf"$+1/2$   coherence={coh:.2f}" if coh is not None else r"$+1/2$"
    else:
        ax.scatter([0],[0],s=22,c="deepskyblue",zorder=8)
        phi0 = 0.0 if d.triatic_orientation is None else d.triatic_orientation
        ell = 0.24*(xmax-xmin)
        for k in range(3):
            a = phi0+2*np.pi*k/3
            ax.plot([0,ell*np.cos(a)],[0,ell*np.sin(a)],color="deepskyblue",lw=1.7,zorder=9)
        coh = d.triatic_coherence
        ttl = rf"$-1/2$   coherence={coh:.2f}" if coh is not None else r"$-1/2$"
    ax.set_title(ttl,fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")


def make_field_figure(sim, defects, output: Path):
    p = sim.p
    L = p.box_size
    q1,q2 = sim.q1,sim.q2
    S = sim.order_parameter()
    theta = sim.director_angle()
    u,v = sim.velocity()
    speed = np.hypot(u,v)
    omega = sim.vorticity(u,v)
    pressure = sim.pressure()
    fx,fy = sim.active_force()
    fmag = np.hypot(fx,fy)
    h1,h2 = sim.molecular_field()
    hmag = np.sqrt(2.0*(h1*h1+h2*h2))

    fig,axs = plt.subplots(4,4,figsize=(17,16),constrained_layout=True)

    # Row 1: primary nematic state.
    _plot_signed(fig,axs[0,0],q1,L,r"$Q_{xx}=q_1$",r"$q_1$")
    _plot_signed(fig,axs[0,1],q2,L,r"$Q_{xy}=q_2$",r"$q_2$")
    _plot_positive(fig,axs[0,2],S,L,r"Scalar order $S=2\sqrt{q_1^2+q_2^2}$","S")
    im = axs[0,3].imshow(S.T,origin="lower",extent=(0,L,0,L),cmap="viridis",
                         vmin=0,vmax=max(np.percentile(S,99),1e-12),interpolation="bilinear")
    _draw_director(axs[0,3],sim.x,sim.y,theta,stride=max(1,p.n//26),seg=0.38*L/26)
    _overlay_defects(axs[0,3],defects)
    axs[0,3].set_title("Director texture + detected defects")
    axs[0,3].set_aspect("equal")
    cb=fig.colorbar(im,ax=axs[0,3],fraction=0.045,pad=0.025); cb.set_label("S")

    # Row 2: orientational relaxation diagnostics.
    _plot_signed(fig,axs[1,0],h1,L,r"Molecular field $H_{xx}=H_1$",r"$H_1$")
    _plot_signed(fig,axs[1,1],h2,L,r"Molecular field $H_{xy}=H_2$",r"$H_2$")
    _plot_positive(fig,axs[1,2],hmag,L,r"Molecular-field magnitude $|H|$",r"$|H|$")
    imth=axs[1,3].imshow(theta.T,origin="lower",extent=(0,L,0,L),cmap="twilight_shifted",
                         vmin=-np.pi/2,vmax=np.pi/2,interpolation="nearest")
    axs[1,3].set_title(r"Director angle $\theta$ (axial: $\theta\equiv\theta+\pi$)")
    axs[1,3].set_aspect("equal")
    cb=fig.colorbar(imth,ax=axs[1,3],fraction=0.045,pad=0.025); cb.set_label(r"$\theta$ [rad]")

    # Row 3: physical fluid.
    _plot_signed(fig,axs[2,0],u,L,r"Fluid velocity $u_x$",r"$u_x$")
    _plot_signed(fig,axs[2,1],v,L,r"Fluid velocity $u_y$",r"$u_y$")
    _plot_positive(fig,axs[2,2],speed,L,r"Fluid speed $|u|$",r"$|u|$",cmap="magma")
    _plot_signed(fig,axs[2,3],omega,L,r"Vorticity $\omega=\partial_xu_y-\partial_yu_x$",r"$\omega$")

    # Row 4: incompressibility / active coupling.
    _plot_signed(fig,axs[3,0],pressure,L,"Stokes pressure (zero-mean gauge)","p")
    _plot_signed(fig,axs[3,1],fx,L,r"Active force $f_x=[\nabla\cdot(\alpha Q)]_x$",r"$f_x$")
    _plot_signed(fig,axs[3,2],fy,L,r"Active force $f_y=[\nabla\cdot(\alpha Q)]_y$",r"$f_y$")
    _plot_positive(fig,axs[3,3],fmag,L,r"Active-force magnitude $|\nabla\cdot(\alpha Q)|$",r"$|f_{active}|$",cmap="magma")
    fs=max(1,p.n//18)
    den=max(np.percentile(fmag,90),1e-12)
    axs[3,3].quiver(sim.x[::fs,::fs],sim.y[::fs,::fs],fx[::fs,::fs]/den,fy[::fs,::fs]/den,
                    color="white",alpha=0.55,angles="xy",scale_units="xy",scale=1.8,width=0.0025)

    for ax in axs.flat:
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.suptitle(
        f"Active-nematic physical fields at t={sim.t:.2f}  |  "
        f"{sum(d.charge>0 for d in defects)} (+1/2), {sum(d.charge<0 for d in defects)} (-1/2)",
        fontsize=16,
    )
    fig.savefig(output,dpi=170)
    plt.close(fig)


def make_defect_figure(sim, defects, output: Path, n_gallery=4):
    p=sim.p; L=p.box_size
    S=sim.order_parameter(); theta=sim.director_angle()
    selected_plus=_nearest_best(defects,n_gallery,True)
    selected_minus=_nearest_best(defects,n_gallery,False)
    selected=selected_plus+selected_minus
    all_plus = [defect for defect in defects if defect.charge > 0.0]
    all_minus = [defect for defect in defects if defect.charge < 0.0]
    labels = [(d, f"P{k+1}") for k,d in enumerate(all_plus)] + \
             [(d, f"M{k+1}") for k,d in enumerate(all_minus)]

    fig=plt.figure(figsize=(15,10),constrained_layout=True)
    gs=fig.add_gridspec(2,n_gallery+2,width_ratios=[1.55,1.55]+[1]*n_gallery)
    ax=fig.add_subplot(gs[:,0:2])
    im=ax.imshow(S.T,origin="lower",extent=(0,L,0,L),cmap="viridis",
                 vmin=0,vmax=max(np.percentile(S,99),1e-12),interpolation="bilinear")
    _draw_director(ax,sim.x,sim.y,theta,stride=max(1,p.n//30),seg=0.36*L/30)
    _overlay_defects(ax,defects,labels=labels)
    half=10*L/p.n
    for d in selected:
        # Visual box only; if near a periodic seam the local gallery is still correct.
        if half < d.x < L-half and half < d.y < L-half:
            ax.add_patch(Rectangle((d.x-half,d.y-half),2*half,2*half,fill=False,
                                   edgecolor="white",linewidth=0.8,alpha=0.6))
    ax.set_title("Global nematic texture: red = +1/2 polarity, blue = -1/2 triatic arms")
    ax.set_xlabel("x");ax.set_ylabel("y");ax.set_aspect("equal")
    cb=fig.colorbar(im,ax=ax,fraction=0.035,pad=0.02);cb.set_label("S")

    for k in range(n_gallery):
        ap=fig.add_subplot(gs[0,k+2])
        if k<len(selected_plus):
            _plot_defect_patch(ap,selected_plus[k],S,theta,L)
            ap.text(0.03,0.95,f"P{k+1}",transform=ap.transAxes,va="top",ha="left",color="white",weight="bold")
        else: ap.axis("off")
        am=fig.add_subplot(gs[1,k+2])
        if k<len(selected_minus):
            _plot_defect_patch(am,selected_minus[k],S,theta,L)
            am.text(0.03,0.95,f"M{k+1}",transform=am.transAxes,va="top",ha="left",color="white",weight="bold")
        else: am.axis("off")

    fig.suptitle("Defect diagnostic: local winding texture and fitted orientations",fontsize=15)
    fig.savefig(output,dpi=180)
    plt.close(fig)


def main():
    print(f"Running active-nematic visualizer: {VISUALIZER_VERSION}")
    parser=argparse.ArgumentParser()
    parser.add_argument("--n",type=int,default=128)
    parser.add_argument("--time",type=float,default=25.0)
    parser.add_argument("--seed",type=int,default=7)
    parser.add_argument("--bank",type=Path,default=None,help="saved physical_bank.npz")
    parser.add_argument("--run",type=int,default=0,help="realization index in --bank")
    parser.add_argument("--frame",type=int,default=-1,help="saved-time index in --bank")
    parser.add_argument("--output",default="active_nematic_fields.png")
    parser.add_argument("--gallery-per-charge",type=int,default=4)
    args=parser.parse_args()

    if args.bank is None:
        p=ActiveNematicParams(n=args.n)
        sim=ActiveNematic2D(p,seed=args.seed)
        sim.run(args.time)
    else:
        bank = PhysicalBank.load(args.bank)
        if not -len(bank.seeds) <= args.run < len(bank.seeds):
            raise SystemExit(f"--run must index {len(bank.seeds)} realizations")
        if not -len(bank.times) <= args.frame < len(bank.times):
            raise SystemExit(f"--frame must index {len(bank.times)} saved times")
        p = bank.params
        sim = ActiveNematic2D(p, seed=int(bank.seeds[args.run]))
        sim.load_state_dict({
            "t": bank.times[args.frame],
            "q1": bank.q1[args.run, args.frame],
            "q2": bank.q2[args.run, args.frame],
        })
    defects=extract_defects(sim.q1,sim.q2,p.box_size)

    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    defect_output=output.with_name(output.stem+"_defects"+output.suffix)
    make_field_figure(sim,defects,output)
    make_defect_figure(sim,defects,defect_output,args.gallery_per_charge)

    plus=[d for d in defects if d.charge>0]; minus=[d for d in defects if d.charge<0]
    pc=[d.polarity_coherence for d in plus if d.polarity_coherence is not None]
    mc=[d.triatic_coherence for d in minus if d.triatic_coherence is not None]
    print(f"saved {output}")
    print(f"saved {defect_output}")
    print(f"defects: +1/2={len(plus)}, -1/2={len(minus)}")
    if pc: print(f"median +1/2 polarity coherence = {np.median(pc):.3f}")
    if mc: print(f"median -1/2 triatic coherence = {np.median(mc):.3f}")
    if defects: print(f"max refined-core residual = {max(d.core_residual for d in defects):.3e}")


if __name__=="__main__":
    main()
