from __future__ import annotations

from . import three_law_qualification as base
from . import three_law_qualification_v2 as qualification


def _flow(flow_id: str) -> dict:
    rows = []
    for K in qualification.K_LADDER:
        for tolerance in qualification.RANK_TOLERANCES:
            rows.append(
                {
                    "K": K,
                    "rank_tolerance": tolerance,
                    "train_action": 1.0 + K * 1.0e-6,
                    "gradient": [1.0, 0.0],
                    "complete_certificate": tolerance == base.DEFAULT_RANK_TOLERANCE,
                    "algebra": {"valid": tolerance == base.DEFAULT_RANK_TOLERANCE},
                    "heldout_certificate": {"maximum_energy_residual": 0.05},
                    "train_forcing": {"valid": True},
                    "audit_forcing": {"valid": True},
                }
            )
    return {"flow_id": flow_id, "rows": rows}


def test_alternate_tolerances_are_output_diagnostics_not_hard_gate() -> None:
    result = qualification.qualify_rows(
        [_flow(flow_id) for flow_id in base.FLOW_IDS]
    )
    assert result["development_qualified"]
    assert result["recommended_K"] == qualification.K_LADDER[0]


def test_selected_tolerance_still_fails_closed() -> None:
    flows = [_flow(flow_id) for flow_id in base.FLOW_IDS]
    row = next(
        item
        for item in flows[-1]["rows"]
        if item["K"] == qualification.K_LADDER[0]
        and item["rank_tolerance"] == base.DEFAULT_RANK_TOLERANCE
    )
    row["complete_certificate"] = False
    row["algebra"]["valid"] = False
    result = qualification.qualify_rows(flows)
    assert not result["qualification_candidates"][0]["qualified"]
    assert result["recommended_K"] == qualification.K_LADDER[1]
