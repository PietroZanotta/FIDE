"""Run an epsilon_R Pareto sweep with common reference and common random numbers.

Each point runs the same scientific pipeline with only law.epsilon_r changed.
Stage-1/2 caches are reusable because epsilon_R is downstream of the Law optimum.
The exact selection-bank frontier and independent-validation check are written to
outputs/pareto/pareto.csv, pareto.json, and (if matplotlib is installed) pareto.png.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path: sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path: sys.path.insert(0, str(SCRIPT_DIR))

from mfsi.config import load_config
from experiment import run_experiment


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser()
    p.add_argument("--eps", nargs="+", type=float, default=[2e-4,5e-4,1e-3,2e-3], help="additive epsilon_R values")
    p.add_argument("--source-run", type=Path, default=SCRIPT_DIR/"outputs"/"run", help="existing compatible run used only to seed reference/CRN artifacts")
    p.add_argument("--output", type=Path, default=SCRIPT_DIR/"outputs"/"pareto")
    p.add_argument("--force", action="store_true", help="rerun points even when result.json exists")
    return p.parse_args()


def _tag(eps: float) -> str:
    return f"epsR_{eps:.7f}".replace(".","p").replace("-","m")


def _link_or_copy(src: Path, dst: Path) -> None:
    if not src.exists() or dst.exists(): return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try: os.link(src,dst)
    except OSError: shutil.copy2(src,dst)


def _seed_artifacts(source: Path, target: Path, *, include_stage12: bool) -> None:
    for name in ("reference.npz","reference_bank.npz","selection_bank.npz","validation_bank.npz"):
        _link_or_copy(source/name,target/name)
    if include_stage12:
        for name in ("population_selection.json","finite_law_selection.json"):
            _link_or_copy(source/"cache"/name,target/"cache"/name)


def _row(result: dict[str,Any], eps: float, point_dir: Path) -> dict[str,Any]:
    sc=result.get("selection_certificates",{})
    full=sc.get("full",{})
    law=sc.get("law",{})
    val=result.get("validation",{})
    contrast=result.get("contrasts",{}).get("full_vs_law_full_action_reduction",{})
    boot=result.get("contrasts",{}).get("full_vs_law_ratio_of_means_bootstrap_95",{})
    pa=result.get("full_proxy_agreement",{})
    funnel=result.get("full_search_funnel",{})
    return {
        "epsilon_r": eps,
        "R_star": result["law_screens"]["R_star"],
        "R_max": result["law_screens"]["R_max"],
        "full_theta1_deg": result["selection"]["full_optimum_deg"][0],
        "full_theta2_deg": result["selection"]["full_optimum_deg"][1],
        "law_theta1_deg": result["selection"]["law_optimum_deg"][0],
        "law_theta2_deg": result["selection"]["law_optimum_deg"][1],
        "full_R_selection": full.get("R_selection"),
        "full_R_excess_selection": full.get("R_excess_from_star"),
        "full_R_slack_selection": full.get("R_slack_to_max"),
        "full_L_selection": full.get("L_selection"),
        "full_certified": full.get("certified"),
        "law_L_selection": law.get("L_selection"),
        "law_R_selection": law.get("R_selection"),
        "law_A_selection": law.get("full_action_selection"),
        "full_A_selection": full.get("full_action_selection"),
        "law_R_validation": val.get("law",{}).get("law_risk",{}).get("mean"),
        "full_R_validation": val.get("full",{}).get("law_risk",{}).get("mean"),
        "law_A_validation": val.get("law",{}).get("full_action",{}).get("mean"),
        "full_A_validation": val.get("full",{}).get("full_action",{}).get("mean"),
        "validation_action_reduction": contrast.get("ratio_of_means_reduction"),
        "validation_ci_lower": boot.get("lower"),
        "validation_ci_upper": boot.get("upper"),
        "proxy_spearman": pa.get("spearman_rank"),
        "proxy_same_best": pa.get("same_best_candidate"),
        "exact_full_finalists": funnel.get("exact_full_finalists"),
        "result": str(point_dir/"result.json"),
    }


def _save(rows: list[dict[str,Any]], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out/"pareto.json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")
    if rows:
        with (out/"pareto.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    try:
        import matplotlib.pyplot as plt
        xs=[float(r["full_R_excess_selection"]) for r in rows if r.get("full_certified") and r.get("full_R_excess_selection") is not None]
        ys=[float(r["full_A_selection"]) for r in rows if r.get("full_certified") and r.get("full_R_excess_selection") is not None]
        labels=[float(r["epsilon_r"]) for r in rows if r.get("full_certified") and r.get("full_R_excess_selection") is not None]
        if xs:
            fig,ax=plt.subplots(figsize=(6.5,4.5)); ax.plot(xs,ys,"o-")
            for x,y,e in zip(xs,ys,labels): ax.annotate(f"eps={e:g}",(x,y),xytext=(5,5),textcoords="offset points",fontsize=8)
            ax.set_xlabel(r"Exact selection risk excess $R(\eta)-R^\star$")
            ax.set_ylabel(r"Exact selection full action $A_{\rm full}(\eta)$")
            ax.set_title("MFSI information–transport Pareto sweep"); ax.grid(True,alpha=.25); fig.tight_layout(); fig.savefig(out/"pareto.png",dpi=180); plt.close(fig)
    except Exception as exc:
        print(f"[pareto] plot skipped: {exc}",flush=True)


def main() -> None:
    args=parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    base_cfg=load_config(SCRIPT_DIR/"config.json",smoke=False)
    eps_values=sorted(set(float(x) for x in args.eps))
    if any(x<0 for x in eps_values): raise ValueError("epsilon_R must be nonnegative")
    rows=[]; stage12_source: Path|None=None
    for i,eps in enumerate(eps_values):
        point=args.output/_tag(eps); point.mkdir(parents=True,exist_ok=True)
        if not (point/"result.json").exists() or args.force:
            if args.force and (point/"result.json").exists():
                for name in ("result.json","result.candidate_summary.csv","result.full_proxy_vs_full.csv","result.validation_trials.csv","manifest.json"):
                    try: (point/name).unlink()
                    except FileNotFoundError: pass
            _seed_artifacts(args.source_run,point,include_stage12=False)
            if stage12_source is not None: _seed_artifacts(stage12_source,point,include_stage12=True)
            cfg=json.loads(json.dumps(base_cfg)); cfg["law"]["epsilon_r"]=eps
            print("="*78,flush=True); print(f"[pareto] epsilon_R={eps:g} -> {point}",flush=True)
            result=run_experiment(cfg,point,smoke=False)
            if stage12_source is None: stage12_source=point
        else:
            print(f"[pareto] reusing {point/'result.json'}",flush=True)
            result=json.loads((point/"result.json").read_text(encoding="utf-8"))
            if stage12_source is None: stage12_source=point
        rows.append(_row(result,eps,point)); _save(rows,args.output)
        r=rows[-1]
        print(f"[pareto] eps={eps:g} ΔR_sel={r.get('full_R_excess_selection')} A_sel={r.get('full_A_selection')} A_val_reduction={r.get('validation_action_reduction')}",flush=True)
    # Add the Law optimum as the epsilon_R=0 Pareto anchor without paying for an
    # extra full stage-4 run.  By definition its exact selection risk is R_star.
    if rows and not any(abs(float(r["epsilon_r"])) < 1e-15 for r in rows):
        r0=dict(rows[0])
        r0.update({
            "epsilon_r": 0.0,
            "R_max": r0["R_star"],
            "full_theta1_deg": r0["law_theta1_deg"],
            "full_theta2_deg": r0["law_theta2_deg"],
            "full_R_selection": r0.get("law_R_selection", r0["R_star"]),
            "full_R_excess_selection": 0.0,
            "full_R_slack_selection": 0.0,
            "full_L_selection": r0.get("law_L_selection"),
            "full_certified": True,
            "full_A_selection": r0.get("law_A_selection"),
            "full_R_validation": r0.get("law_R_validation"),
            "full_A_validation": r0.get("law_A_validation"),
            "validation_action_reduction": 0.0,
            "validation_ci_lower": 0.0,
            "validation_ci_upper": 0.0,
            "proxy_spearman": None,
            "proxy_same_best": None,
            "exact_full_finalists": None,
            "result": "law_anchor_from_first_pareto_run",
        })
        rows=[r0]+rows
        _save(rows,args.output)
    print(f"[pareto] complete: {args.output/'pareto.csv'}",flush=True)
    print(f"[pareto] plot:     {args.output/'pareto.png'}",flush=True)

if __name__=="__main__": main()
