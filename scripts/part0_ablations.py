#!/usr/bin/env python3
"""Paper Part-0 ablations for MFSI Experiments A/B.

Covers the §6.5 axes requested in the draft:
  correction-network capacity, Ritz batch size, covariance rank truncation,
  reference coupling, SI noise level, safety-layer use, and stop-vs-implicit AD.

The neural capacity/batch ablations use Example A because its learned reference
checkpoint is already available and the held-out weak-form metric does not use
its analytic oracle. Geometry ablations use empirical projection diagnostics.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import jax, jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import mfsi_components as core
import example_b as exb
import ablate_and_benchmark as grad_ab

jax.config.update("jax_enable_x64", True)
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"/"part0_ablations"; OUT.mkdir(parents=True, exist_ok=True)
A=0.8
MODEL=ROOT/"results"/"learned_mfsi_example_a.npz"


def weak_summary(potential, ref, key):
    model=core.LearnedMFSIModel(ref,potential)
    times=jnp.linspace(.1,.9,5)
    rows=core.heldout_learned_diagnostics(key,model,A,times,bank_particles=128,mmd_particles=64)
    weak=np.asarray([r["weak_form_residual"] for r in rows])
    return {"median_weak_form_residual":float(np.median(weak)),"max_weak_form_residual":float(np.max(weak))}


def neural_ablations(key, quick=False, part="all"):
    if not MODEL.exists(): raise FileNotFoundError("run Experiment A validation first")
    base=core.load_learned_model(MODEL); keys=iter(jax.random.split(key,16))
    steps=4 if quick else 240
    caps=[(32,32),(64,64)] if quick else [(32,32),(64,64,64),tuple(core.RITZ_HIDDEN)]
    capacity=[]
    if part in {"all","capacity"}:
        for hidden in caps:
            pot,_,_=core.train_deep_ritz(next(keys),base.reference_params,A,steps=steps,n_times=3 if quick else 6,
                particles_per_time=64 if quick else 128,hidden=hidden,bank_pool_size=1 if quick else 3,bank_refresh_every=0 if quick else max(steps//2,1),
                validation_times=3 if quick else 5,validation_particles=64 if quick else 96,lbfgs_maxiter=0)
            capacity.append({"hidden":list(hidden),"parameter_count":int(core.flatten_mlp(pot).size),**weak_summary(pot,base.reference_params,next(keys))})
    batches=[]
    if part in {"all","batch"}:
        for n in ((64,128) if quick else (64,128,256)):
            pot,_,_=core.train_deep_ritz(next(keys),base.reference_params,A,steps=steps,n_times=3 if quick else 6,
                particles_per_time=n,hidden=(64,64) if quick else core.RITZ_HIDDEN,bank_pool_size=1 if quick else 3,bank_refresh_every=0 if quick else max(steps//2,1),
                validation_times=3 if quick else 5,validation_particles=64 if quick else 96,lbfgs_maxiter=0)
            batches.append({"particles_per_time":n,**weak_summary(pot,base.reference_params,next(keys))})
    return {"capacity":capacity,"ritz_batch_size":batches,"training_steps":steps}


def _projection_stats(x, ph, target):
    logw=jnp.zeros(x.shape[0],dtype=x.dtype)
    lam=core.calibrate_empirical_implicit(logw,ph,target)
    w,m,cov=core.empirical_tilt_from_lambda(lam,logw,ph)
    dist=jnp.sum(w*jnp.log(jnp.maximum(w*x.shape[0],1e-300)))
    ess=1/(x.shape[0]*jnp.sum(w*w))
    return w,cov,{"ess_fraction":float(ess),"projection_distortion":float(dist),"calibration_residual":float(jnp.linalg.norm(m-target))}


def coupling_ablation(key,n=8192):
    rows=[]
    for rho in (0.0,0.5,0.9):
        vals=[]
        for i,t in enumerate(np.linspace(.1,.9,5)):
            k0,k1,ks=jax.random.split(jax.random.fold_in(key,i+int(100*rho)),3)
            z0=jax.random.normal(k0,(n,),dtype=jnp.float64); z1=jax.random.normal(k1,(n,),dtype=jnp.float64)
            sign=jnp.where(jax.random.bernoulli(ks,.5,(n,)),1.,-1.)
            x1=sign*A+jnp.sqrt(1-A*A)*(rho*z0+jnp.sqrt(1-rho*rho)*z1)
            x=(1-t)*z0+t*x1
            _,_,st=_projection_stats(x,core.phi(x),core.TARGET); vals.append(st)
        rows.append({"rho":rho,"min_ess_fraction":min(v["ess_fraction"] for v in vals),
                     "mean_projection_distortion":float(np.mean([v["projection_distortion"] for v in vals]))})
    return rows


def noise_ablation(key,n=8192):
    rows=[]
    for beta in (0.0,0.5,1.0):
        params=None if beta==0 else jnp.array([core.inverse_softplus(beta)],dtype=jnp.float64)
        vals=[]
        for i,t in enumerate(np.linspace(.1,.9,5)):
            x,_=core.sample_reference_bridge(jax.random.fold_in(key,100+i+int(beta*10)),jnp.asarray(t),n,A,params)
            _,_,st=_projection_stats(x,core.phi(x),core.TARGET); vals.append(st)
        rows.append({"beta":beta,"min_ess_fraction":min(v["ess_fraction"] for v in vals),
                     "mean_projection_distortion":float(np.mean([v["projection_distortion"] for v in vals]))})
    return rows


def rank_ablation(key,n=4096):
    x,_=exb.sample_bridge(key,jnp.asarray(.5),n)
    ph=exb.phi(x); target=exb.TARGET
    w,cov,_=_projection_stats(x,ph,target)
    # Add a near-duplicate measured coordinate to make the identifiable-subspace issue visible.
    dup=ph[:,2]+1e-3*x[:,0]**3; ph_aug=jnp.concatenate([ph,dup[:,None]],axis=1)
    targ_aug=jnp.concatenate([target,jnp.array([float(w@dup)])])
    centered=ph_aug-(w@ph_aug); cov_aug=(centered.T*w)@centered
    rhs=jnp.arange(1,7,dtype=jnp.float64)/6
    rows=[]
    for rcond in (1e-12,1e-9,1e-6,1e-3):
        sol,rank,cond=core._stable_cov_solve(cov_aug,rhs,rcond=rcond,damping=0.0)
        rows.append({"rcond":rcond,"effective_rank":int(rank),"condition":float(cond),"solution_norm":float(jnp.linalg.norm(sol)),
                     "augmented_target_dim":int(targ_aug.size)})
    return rows


def safety_ablation():
    path=ROOT/"results"/"example_b"/"example_b_results.json"
    if not path.exists(): return {"available":False}
    r=json.loads(path.read_text())["benchmark_summary"]
    return {"available":True,"mfsi_learned":r["mfsi_learned"],"mfsi_learned_safe":r["mfsi_learned_safe"]}


def differentiation_ablation():
    # Keep this expensive compile in its own script/process. Reuse its persisted
    # result here so the full Part-0 ablation report stays memory-stable.
    for path in (ROOT/"results/ablation_metrics.json", ROOT/"results/reference/ablation_metrics.json"):
        if path.exists():
            r=json.loads(path.read_text())
            return {"source":str(path.relative_to(ROOT)),"relative_errors":r["relative_errors"],
                    "initial":r["initial"],"optimized_implicit":r["optimized_implicit"],"optimized_stop":r["optimized_stop"]}
    raise FileNotFoundError("run ./scripts/run_ablations.sh before Part-0 ablations")


def make_plot(result):
    fig,axes=plt.subplots(2,3,figsize=(12,7.2))
    c=result["neural"]["capacity"]; axes[0,0].plot([z["parameter_count"] for z in c],[z["median_weak_form_residual"] for z in c],marker='o'); axes[0,0].set_xscale('log'); axes[0,0].set_title('Correction capacity')
    b=result["neural"]["ritz_batch_size"]; axes[0,1].plot([z["particles_per_time"] for z in b],[z["median_weak_form_residual"] for z in b],marker='o'); axes[0,1].set_title('Ritz batch size')
    r=result["rank_truncation"]; axes[0,2].semilogx([z["rcond"] for z in r],[z["solution_norm"] for z in r],marker='o'); axes[0,2].set_title('Rank truncation stress')
    q=result["reference_coupling"]; axes[1,0].plot([z["rho"] for z in q],[z["min_ess_fraction"] for z in q],marker='o'); axes[1,0].set_title('Reference coupling / overlap')
    n=result["si_noise_level"]; axes[1,1].plot([z["beta"] for z in n],[z["min_ess_fraction"] for z in n],marker='o'); axes[1,1].set_title('SI noise level / overlap')
    d=result["differentiation"]["relative_errors"]; axes[1,2].bar(['implicit-FD','stop-unrolled'],[d['correction_implicit_vs_fd'],d['correction_stop_vs_unrolled']]); axes[1,2].set_yscale('log'); axes[1,2].set_title('Calibration differentiation')
    fig.tight_layout(); fig.savefig(OUT/'part0_ablations.png',dpi=180); plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--quick',action='store_true')
    p.add_argument('--section',choices=('all','capacity','batch','geometry','safety','differentiation'),default='all',
                   help='run one ablation section; results are merged into the existing JSON')
    args=p.parse_args()
    ks=jax.random.split(jax.random.PRNGKey(62119),5)
    path=OUT/'part0_ablations.json'
    result=json.loads(path.read_text()) if path.exists() else {}
    result['mode']='quick' if args.quick else 'full'
    sec=args.section
    if sec in {'all','capacity','batch'}:
        part='all' if sec=='all' else sec
        nr=neural_ablations(ks[0],args.quick,part=part)
        old=result.get('neural',{})
        if nr['capacity']: old['capacity']=nr['capacity']
        if nr['ritz_batch_size']: old['ritz_batch_size']=nr['ritz_batch_size']
        old['training_steps']=nr['training_steps']; result['neural']=old
    if sec in {'all','geometry'}:
        result['rank_truncation']=rank_ablation(ks[1],n=1024 if args.quick else 4096)
        result['reference_coupling']=coupling_ablation(ks[2],n=1024 if args.quick else 8192)
        result['si_noise_level']=noise_ablation(ks[3],n=1024 if args.quick else 8192)
    if sec in {'all','safety'}: result['safety_layer']=safety_ablation()
    if sec in {'all','differentiation'}: result['differentiation']=differentiation_ablation()
    path.write_text(json.dumps(result,indent=2))
    required={'neural','rank_truncation','reference_coupling','si_noise_level','safety_layer','differentiation'}
    if required.issubset(result): make_plot(result)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
