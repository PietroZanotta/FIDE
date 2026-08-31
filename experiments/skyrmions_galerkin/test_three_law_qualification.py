from __future__ import annotations

import copy

from . import three_law_qualification as qualification


def _row(K: int, tolerance: float, *, complete: bool = True,
         action: float = 1.0, energy: float = 0.05) -> dict:
    return {
        "K": K,
        "rank_tolerance": tolerance,
        "train_action": action,
        "gradient": [1.0, 0.0],
        "complete_certificate": complete,
        "algebra": {"valid": complete},
        "heldout_certificate": {"maximum_energy_residual": energy},
    }


def _flow(flow_id: str) -> dict:
    rows = []
    for K in qualification.K_LADDER:
        for tolerance in qualification.RANK_TOLERANCES:
            rows.append(_row(K, tolerance, action=1.0 + K * 1.0e-6))
    return {"flow_id": flow_id, "rows": rows}


def test_common_rule_selects_smallest_K_that_passes_all_flows() -> None:
    flows = [_flow(flow_id) for flow_id in qualification.FLOW_IDS]
    for flow in flows:
        for row in flow["rows"]:
            if row["K"] == qualification.K_LADDER[0]:
                row["complete_certificate"] = False
                row["algebra"]["valid"] = False
    result = qualification.qualify_rows(flows)
    assert result["development_qualified"]
    assert result["recommended_K"] == qualification.K_LADDER[1]


def test_one_failed_law_prevents_common_qualification_at_that_K() -> None:
    flows = [_flow(flow_id) for flow_id in qualification.FLOW_IDS]
    broken = copy.deepcopy(flows)
    for row in broken[-1]["rows"]:
        if row["K"] == qualification.K_LADDER[0]:
            row["complete_certificate"] = False
            row["algebra"]["valid"] = False
    result = qualification.qualify_rows(broken)
    first = result["qualification_candidates"][0]
    assert not first["qualified"]
    assert result["recommended_K"] == qualification.K_LADDER[1]


def test_rank_tolerance_instability_fails_closed() -> None:
    flows = [_flow(flow_id) for flow_id in qualification.FLOW_IDS]
    first_K = qualification.K_LADDER[0]
    row = next(
        item
        for item in flows[0]["rows"]
        if item["K"] == first_K
        and item["rank_tolerance"] == qualification.RANK_TOLERANCES[0]
    )
    row["train_action"] = 1.2
    result = qualification.qualify_rows(flows)
    assert not result["qualification_candidates"][0]["qualified"]
    assert result["recommended_K"] == qualification.K_LADDER[1]
