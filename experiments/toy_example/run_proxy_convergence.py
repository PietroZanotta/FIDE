"""Confirm that the selected Full design is stable as the stage-4 proxy is refined.

This is a numerical-method audit, not a new scientific estimand.  The authoritative
selection action always remains the full configured Poisson/time discretization.
Validation is intentionally kept small here; use the main run for headline CI.
"""
from __future__ import annotations
import argparse,csv,json,os,shutil,sys
from pathlib import Path
from typing import Any

SCRIPT_DIR=Path(__file__).resolve().parent; REPO_ROOT=SCRIPT_DIR.parent.parent; SRC_DIR=REPO_ROOT/"src"
if str(SRC_DIR) not in sys.path: sys.path.insert(0,str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path: sys.path.insert(0,str(SCRIPT_DIR))
from mfsi.config import load_config
from experiment import run_experiment


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--fidelity", nargs="+", default=["4:7:41:1e-6:120","6:9:51:3e-7:180"], help="trials:times:grid:cg_tol:cg_maxiter")
    p.add_argument("--source-run",type=Path,default=SCRIPT_DIR/"outputs"/"run")
    p.add_argument("--output",type=Path,default=SCRIPT_DIR/"outputs"/"proxy_convergence")
    p.add_argument("--validation-trials",type=int,default=8)
    p.add_argument("--force",action="store_true")
    return p.parse_args()


def parse_spec(s:str):
    a=s.split(":");
    if len(a)!=5: raise ValueError(f"bad fidelity {s!r}; expected trials:times:grid:cg_tol:cg_maxiter")
    return int(a[0]),int(a[1]),int(a[2]),float(a[3]),int(a[4])


def link(src:Path,dst:Path):
    if not src.exists() or dst.exists(): return
    dst.parent.mkdir(parents=True,exist_ok=True)
    try: os.link(src,dst)
    except OSError: shutil.copy2(src,dst)


def seed(source:Path,target:Path,cache_source:Path|None):
    for n in ("reference.npz","reference_bank.npz","selection_bank.npz"):
        link(source/n,target/n)
    if cache_source:
        for n in ("population_selection.json","finite_law_selection.json"):
            link(cache_source/"cache"/n,target/"cache"/n)


def angular_distance_deg(a,b):
    vals=[]
    for x,y in zip(a,b):
        d=abs(float(x)-float(y))%360.0; vals.append(min(d,360.0-d))
    return max(vals)


def row(res:dict[str,Any],spec,point:Path):
    tr,tn,gn,tol,mi=spec; cert=res.get("selection_certificates",{}); full=cert.get("full",{}); law=cert.get("law",{}); pa=res.get("full_proxy_agreement",{}); f=res.get("full_search_funnel",{})
    return {"proxy_trials":tr,"proxy_times":tn,"proxy_grid_n":gn,"proxy_cg_tol":tol,"proxy_cg_maxiter":mi,
            "full_theta1_deg":res["selection"]["full_optimum_deg"][0],"full_theta2_deg":res["selection"]["full_optimum_deg"][1],
            "full_R_selection":full.get("R_selection"),"full_R_slack":full.get("R_slack_to_max"),"full_certified":full.get("certified"),
            "law_A_selection":law.get("full_action_selection"),"full_A_selection":full.get("full_action_selection"),
            "selection_action_reduction":(1.0-float(full["full_action_selection"])/float(law["full_action_selection"])) if full.get("full_action_selection") is not None and law.get("full_action_selection") else None,
            "proxy_pearson":pa.get("pearson"),"proxy_spearman":pa.get("spearman_rank"),"proxy_same_best":pa.get("same_best_candidate"),
            "exact_full_finalists":f.get("exact_full_finalists"),"result":str(point/"result.json")}


def main():
    args=parse_args(); args.output.mkdir(parents=True,exist_ok=True); base=load_config(SCRIPT_DIR/"config.json",smoke=False); rows=[]; cache_source=None
    for spec_s in args.fidelity:
        spec=parse_spec(spec_s); tr,tn,gn,tol,mi=spec; point=args.output/f"p{tr}_t{tn}_g{gn}"; point.mkdir(parents=True,exist_ok=True)
        if args.force:
            for n in ("result.json","result.candidate_summary.csv","result.full_proxy_vs_full.csv","result.validation_trials.csv","manifest.json"):
                try:(point/n).unlink()
                except FileNotFoundError:pass
        if not (point/"result.json").exists():
            seed(args.source_run,point,cache_source)
            cfg=json.loads(json.dumps(base)); o=cfg["optimization"]
            o["full_gradient_trials"]=tr; o["full_gradient_time_n"]=tn; o["full_gradient_grid_n"]=gn; o["full_gradient_cg_tol"]=tol; o["full_gradient_cg_maxiter"]=mi
            cfg["randomness"]["validation_trials"]=max(1,args.validation_trials); cfg["randomness"]["bootstrap_reps"]=200
            print("="*78,flush=True); print(f"[proxy-audit] {spec_s} -> {point}",flush=True); res=run_experiment(cfg,point,smoke=False)
            if cache_source is None: cache_source=point
        else:
            res=json.loads((point/"result.json").read_text());
            if cache_source is None: cache_source=point
        rows.append(row(res,spec,point))
    if rows:
        base_eta=[rows[0]["full_theta1_deg"],rows[0]["full_theta2_deg"]]; base_A=float(rows[0]["full_A_selection"])
        for r in rows:
            r["max_angle_shift_from_first_deg"]=angular_distance_deg(base_eta,[r["full_theta1_deg"],r["full_theta2_deg"]]); r["full_A_relative_change_from_first"]=(float(r["full_A_selection"])/base_A-1.0)
        with (args.output/"proxy_convergence.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        (args.output/"proxy_convergence.json").write_text(json.dumps(rows,indent=2)+"\n")
        print(); print("Proxy convergence summary")
        for r in rows: print(f"  {r['proxy_trials']}x{r['proxy_times']}x{r['proxy_grid_n']}  eta=({r['full_theta1_deg']:.3f},{r['full_theta2_deg']:.3f})  A_exact={r['full_A_selection']:.6g}  shift={r['max_angle_shift_from_first_deg']:.3g}deg  dA={100*r['full_A_relative_change_from_first']:.2f}%  rho={r['proxy_spearman']}")
        print(f"saved: {args.output/'proxy_convergence.csv'}")

if __name__=="__main__": main()
