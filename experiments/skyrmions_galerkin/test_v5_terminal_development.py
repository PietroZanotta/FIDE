from __future__ import annotations
from . import v5_terminal_development as study

def test_fixed_schedule_and_rules():
    assert study.RISK_ROLE_COUNT == 8
    assert study.N_VALUES == (32768, 65536, 131072)
    assert study.K == 280 and study.RANK_TOLERANCE == 1e-12
    assert study._call_graph()["passed"]

def test_selected_rows_are_seven_unique_geometries():
    rows=study._selected()
    assert len(rows)==7 and len({x["eta_sha256"] for x in rows})==7
    assert next(x for x in rows if x["method"]=="Law")["selection_relative"]==0.0

