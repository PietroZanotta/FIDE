"""Regression tests for defect extraction, sub-grid localization, orientation fits,
and periodic frame-to-frame tracking.
"""

import numpy as np

from experiments.active_nematic.defect_extractor import (
    Defect,
    DefectTracker,
    extract_defects,
    plaquette_winding,
)


def periodic_delta(a, b, L):
    return (a - b + 0.5 * L) % L - 0.5 * L


def synthetic_field(n=256, L=2*np.pi, x0=0.73, y0=1.11, beta=0.7):
    x=np.linspace(0,L,n,endpoint=False)
    X,Y=np.meshgrid(x,x,indexing="ij")
    # Near (x0,y0): psi=q1+i q2 ~ exp(i beta)[(x-x0)+i(y-y0)],
    # i.e. a canonical +1/2 defect with polarity beta.
    sx=np.sin(X-x0); sy=np.sin(Y-y0)
    q1=0.5*(sx*np.cos(beta)-sy*np.sin(beta))
    q2=0.5*(sx*np.sin(beta)+sy*np.cos(beta))
    return q1,q2


def test_analytic_defects():
    L=2*np.pi; x0=.73; y0=1.11
    for beta in [0.0,0.3,1.0,2.4,-1.2]:
        q1,q2=synthetic_field(L=L,x0=x0,y0=y0,beta=beta)
        w=plaquette_winding(q1,q2)
        ds=extract_defects(q1,q2,L)
        assert sorted(d.charge for d in ds)==[-0.5,-0.5,0.5,0.5]
        assert int(w.sum())==0
        assert int(np.max(np.abs(w)))==1
        d=min(ds,key=lambda z: periodic_delta(z.x,x0,L)**2+periodic_delta(z.y,y0,L)**2)
        assert d.charge==0.5
        poserr=np.hypot(periodic_delta(d.x,x0,L),periodic_delta(d.y,y0,L))
        assert poserr < 1e-5, poserr
        assert d.polarity is not None
        angerr=np.arctan2(np.sin(d.polarity-beta),np.cos(d.polarity-beta))
        assert abs(angerr)<1e-5, (beta,d.polarity,angerr)
        assert d.polarity_coherence is not None and d.polarity_coherence>0.999
        assert d.core_residual<1e-12


def test_periodic_tracking_birth_death():
    L=10.0
    tr=DefectTracker(L,max_displacement=0.8)
    f0=[Defect(9.8,2.0,+0.5), Defect(5.0,5.0,-0.5)]
    ev0=tr.update(f0)
    assert len(ev0["births"])==2 and not ev0["deaths"]
    plus_id=f0[0].track_id; minus_id=f0[1].track_id

    # +1/2 crosses periodic seam; -1/2 disappears; new -1/2 is born.
    f1=[Defect(0.2,2.1,+0.5), Defect(8.0,8.0,-0.5)]
    ev1=tr.update(f1)
    assert f1[0].track_id==plus_id
    assert minus_id in ev1["deaths"]
    assert f1[1].track_id in ev1["births"]


def run_tests():
    test_analytic_defects()
    test_periodic_tracking_birth_death()
    print("PASS: charges, sub-grid cores, +1/2 polarity, periodic tracking, births/deaths")


if __name__=="__main__":
    run_tests()
