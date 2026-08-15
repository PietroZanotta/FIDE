"""Evaluate a saved toy-example result without rerunning the science.

Also re-certifies the selected designs against the exact selection-bank L/R
screens when `selection_certificates` are present in result.json, or (for older
saved runs) when result.candidate_summary.csv is available beside the JSON.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT = SCRIPT_DIR / "outputs" / "run" / "result.json"


def _finite(x: Any) -> bool:
    try: return math.isfinite(float(x))
    except (TypeError, ValueError): return False


def _num(x: Any, digits: int = 6) -> str:
    if not _finite(x): return "n/a"
    x = float(x)
    if x == 0.0: return "0"
    if abs(x) < 1e-4 or abs(x) >= 1e5: return f"{x:.4e}"
    return f"{x:.{digits}g}"


def _pct(x: Any, digits: int = 2) -> str:
    return "n/a" if not _finite(x) else f"{100.0*float(x):.{digits}f}%"


def _eta(x: Any) -> str:
    return "n/a" if not isinstance(x, (list, tuple)) or len(x) != 2 else f"({float(x[0]):.3f}°, {float(x[1]):.3f}°)"


def _metric(summary: dict[str, Any], key: str) -> str:
    b = summary.get(key, {})
    mean, se, n = b.get("mean"), b.get("se"), b.get("n", 0)
    if not _finite(mean): return "n/a"
    return f"{_num(mean)} ± {_num(se)}  (SE, n={n})" if _finite(se) else f"{_num(mean)}  (n={n})"


def _line(c: str = "-", n: int = 78) -> str: return c*n


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f: return json.load(f)


def _certificates(data: dict[str, Any], result_path: Path) -> dict[str, Any]:
    cert = data.get("selection_certificates")
    if isinstance(cert, dict) and cert:
        return cert
    # Backward-compatible certification of already-completed runs: experiment.py
    # has long saved the exact selection values in candidate_summary.csv.
    csv_path = result_path.with_name("result.candidate_summary.csv")
    if not csv_path.exists(): return {}
    screens = data.get("law_screens", {})
    Ls, Lm, Rs, Rm = map(float, [screens["L_star"], screens["L_max"], screens["R_star"], screens["R_max"]])
    out = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["design"]
            L, R = float(row["population_loss_selection"]), float(row["finite_risk_selection"])
            req = ["L"] if name in ("population", "law") else ["L", "R"]
            pL, pR = L <= Lm + 1e-12, R <= Rm + 1e-12
            out[name] = {
                "required_screens": req,
                "L_selection": L, "L_star": Ls, "L_max": Lm,
                "L_excess_from_star": L-Ls, "L_slack_to_max": Lm-L, "passes_L": pL,
                "R_selection": R, "R_star": Rs, "R_max": Rm,
                "R_excess_from_star": R-Rs, "R_slack_to_max": Rm-R, "passes_R": pR,
                "full_action_selection": float(row.get("full_action_selection", "nan")),
                "tangent_action_selection": float(row.get("tangent_action_selection", "nan")),
                "certified": bool(pL and (pR if "R" in req else True)),
            }
    return out


def _eval_smoke(data: dict[str, Any], path: Path) -> int:
    print(_line("=")); print("TOY EXAMPLE — SMOKE RESULT"); print(_line("="))
    print(f"file:   {path}"); print(f"design: {_eta(data.get('smoke_design_deg'))}")
    m=data.get("smoke_metrics",{}); print()
    for label,key in [("law risk","law_risk"),("tangent action","tangent_action"),("full action","full_action"),("max calibration residual","max_calibration_residual"),("max Poisson rel. residual","max_poisson_relative_residual")]:
        print(f"  {label:<28} {_num(m.get(key))}")
    print(f"  min ESS fraction             {_pct(m.get('min_ess_fraction'))}")
    print(f"  scientific validity          {'PASS' if m.get('valid') else 'FAIL'}")
    return 0 if m.get("valid") else 2


def _eval_run(data: dict[str, Any], path: Path) -> int:
    print(_line("=")); print("TOY EXAMPLE — SAVED RUN EVALUATION"); print(_line("="))
    print(f"file:          {path}"); print(f"schema:        {data.get('schema_version','n/a')}"); print(f"config hash:   {data.get('config_hash','n/a')}")
    ref=data.get("reference",{}); print(f"reference:     {ref.get('checkpoint','n/a')}"); print(f"min base mass: {_pct(ref.get('min_in_domain_base_mass'),3)}")
    s=data.get("law_screens",{}); print(); print("Lexicographic selection screens"); print(_line())
    print(f"  L*      = {_num(s.get('L_star'))}"); print(f"  L_max   = {_num(s.get('L_max'))}   (epsilon_L={_num(s.get('epsilon_l'))})")
    print(f"  R*      = {_num(s.get('R_star'))}"); print(f"  R_max   = {_num(s.get('R_max'))}   (epsilon_R={_num(s.get('epsilon_r'))})")
    sel=data.get("selection",{}); print(); print("Selected designs"); print(_line())
    for name,key in [("population","population_optimum_deg"),("law","law_optimum_deg"),("tangent","tangent_optimum_deg"),("full","full_optimum_deg")]: print(f"  {name:<12} {_eta(sel.get(key))}")

    cert=_certificates(data,path); failures=[]; warnings=[]
    if cert:
        print(); print("Selection-bank certification"); print(_line())
        print(f"{'design':<12} {'L':>11} {'L slack':>11} {'R':>11} {'R slack':>11} {'required':>10} {'status':>9}")
        for name in ("population","law","tangent","full"):
            c=cert.get(name); 
            if not c: continue
            req="+".join(c.get("required_screens",[])); status="PASS" if c.get("certified") else "FAIL"
            print(f"{name:<12} {_num(c.get('L_selection')):>11} {_num(c.get('L_slack_to_max')):>11} {_num(c.get('R_selection')):>11} {_num(c.get('R_slack_to_max')):>11} {req:>10} {status:>9}")
            if name in ("tangent","full") and not c.get("certified",False): failures.append(f"{name}: failed authoritative selection-bank L/R certificate")
    else:
        warnings.append("selection-bank certificate unavailable (no selection_certificates and no candidate summary CSV)")

    pa=data.get("full_proxy_agreement",{}); print(); print("Stage-4 proxy/full agreement"); print(_line())
    print(f"  exact finalists compared    {pa.get('candidate_count','n/a')}"); print(f"  Pearson correlation         {_num(pa.get('pearson'))}"); print(f"  Spearman rank correlation   {_num(pa.get('spearman_rank'))}"); print(f"  same best candidate         {'YES' if pa.get('same_best_candidate') else 'NO'}")
    if _finite(pa.get("spearman_rank")) and float(pa["spearman_rank"]) < .5: warnings.append(f"stage-4 proxy/full Spearman={float(pa['spearman_rank']):.3f} < 0.5")
    if not pa.get("same_best_candidate",False): warnings.append("stage-4 proxy and exact full action choose different best exact finalists; exact rescoring remains authoritative")

    v=data.get("validation",{}); print(); print("Independent validation"); print(_line())
    print(f"{'design':<12} {'eta':<24} {'law MMD²':<27} {'tangent action':<27} {'full action':<27} {'valid':>7}")
    for name in ("population","law","tangent","full"):
        b=v.get(name); 
        if not b: continue
        print(f"{name:<12} {_eta(b.get('eta_deg')):<24} {_metric(b,'law_risk'):<27} {_metric(b,'tangent_action'):<27} {_metric(b,'full_action'):<27} {_pct(b.get('valid_fraction'),1):>7}")
        vf=b.get("valid_fraction");
        if _finite(vf) and float(vf)<.95: failures.append(f"{name}: validation valid_fraction={float(vf):.3f} < 0.95")

    lawR=v.get("law",{}).get("law_risk",{}).get("mean"); print(); print("Validation law penalty relative to Law")
    if _finite(lawR) and abs(float(lawR))>1e-300:
        for name in ("population","tangent","full"):
            x=v.get(name,{}).get("law_risk",{}).get("mean")
            if _finite(x): print(f"  {name:<10} {_pct(float(x)/float(lawR)-1):>9}")
    print("  Note: validation is out-of-sample; formal L/R feasibility is certified above on the selection bank.")

    c=data.get("contrasts",{}); pr=c.get("full_vs_law_full_action_reduction",{}); boot=c.get("full_vs_law_ratio_of_means_bootstrap_95",{})
    rr=pr.get("ratio_of_means_reduction"); print(); print("Primary Full vs Law validation contrast"); print(_line())
    print(f"  ratio-of-means action reduction   {_pct(rr)}"); print(f"  mean paired action reduction      {_pct(pr.get('mean_paired_reduction'))} ± {_pct(pr.get('se_paired_reduction'))} (SE, n={pr.get('n',0)})"); print(f"  bootstrap 95% CI                 [{_pct(boot.get('lower'))}, {_pct(boot.get('upper'))}] (reps={boot.get('reps',0)})")
    if not _finite(rr): failures.append("Full-vs-Law action reduction unavailable")

    print(); print("Interpretation"); print(_line())
    if _finite(rr):
        if float(rr)>0: print(f"  • Full uses {_pct(rr)} less mean validation action than Law.")
        elif float(rr)<0: print(f"  • Full uses {_pct(-float(rr))} more mean validation action than Law.")
        else: print("  • Full and Law have equal mean validation action.")
    lo,hi=boot.get("lower"),boot.get("upper")
    if _finite(lo) and _finite(hi): print("  • The saved bootstrap 95% CI is entirely above zero." if float(lo)>0 else "  • The saved bootstrap 95% CI crosses/includes zero." if float(hi)>=0 else "  • The saved bootstrap 95% CI is entirely below zero.")
    if cert.get("full"):
        fc=cert["full"]; print(f"  • Full selection certificate: ΔR=R-R*={_num(fc.get('R_excess_from_star'))}, slack to R_max={_num(fc.get('R_slack_to_max'))}, {'PASS' if fc.get('certified') else 'FAIL'}.")

    print();
    if warnings or failures:
        print("Checks"); print(_line()); [print(f"  WARN  {x}") for x in warnings]; [print(f"  FAIL  {x}") for x in failures]
    else: print("Checks: PASS — exact selection certificates and saved validation summaries are complete.")
    print(); print("Recommended headline quantities"); print(_line()); print(f"  Law eta:   {_eta(sel.get('law_optimum_deg'))}"); print(f"  Full eta:  {_eta(sel.get('full_optimum_deg'))}"); print(f"  Law R val: {_metric(v.get('law',{}),'law_risk')}"); print(f"  Full R val:{_metric(v.get('full',{}),'law_risk')}"); print(f"  Law A val: {_metric(v.get('law',{}),'full_action')}"); print(f"  Full A val:{_metric(v.get('full',{}),'full_action')}"); print(f"  A reduction: {_pct(rr)}"); print(f"  95% CI: [{_pct(boot.get('lower'))}, {_pct(boot.get('upper'))}]")
    return 2 if failures else 0


def main() -> int:
    if len(sys.argv)>2: print(f"usage: {Path(sys.argv[0]).name} [result.json]",file=sys.stderr); return 64
    path=(Path(sys.argv[1]).expanduser().resolve() if len(sys.argv)==2 else DEFAULT_RESULT)
    try: data=_load(path)
    except Exception as e: print(f"error: could not read {path}: {e}",file=sys.stderr); return 1
    return _eval_smoke(data,path) if data.get("smoke",False) else _eval_run(data,path)

if __name__=="__main__": raise SystemExit(main())
