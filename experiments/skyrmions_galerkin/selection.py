from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class BankArtifact:
    role: str
    identifier: str
    payload: Any


class BankRegistry:
    """Role-aware bank access that makes selection/validation leakage testable."""

    def __init__(self, artifacts: Mapping[str, BankArtifact]):
        self._artifacts = dict(artifacts)
        self.access_log: list[dict[str, str]] = []

    def get(self, name: str, *, consumer: str) -> Any:
        artifact = self._artifacts[name]
        if consumer == "selection" and artifact.role == "validation":
            raise RuntimeError(
                f"bank leakage blocked: selection attempted to consume validation bank {artifact.identifier!r}"
            )
        self.access_log.append(
            {"name": name, "role": artifact.role, "identifier": artifact.identifier, "consumer": consumer}
        )
        return artifact.payload

    def manifest(self) -> list[dict[str, str]]:
        return [
            {"name": name, "role": item.role, "identifier": item.identifier}
            for name, item in sorted(self._artifacts.items())
        ]


def nested_certified_selection(
    candidates: list[dict[str, Any]],
    *,
    anchor_risk: float,
    allowances_percent: list[float],
    action_tolerance: float = 1.0e-10,
) -> list[dict[str, Any]]:
    """Select nested Pareto winners while retaining each certified incumbent.

    Values are never edited to manufacture monotonicity.  The prior winner is
    explicitly checked for eligibility at the next (larger) budget.
    """

    if anchor_risk <= 0.0:
        raise ValueError("anchor_risk must be positive")
    ordered = sorted(float(p) for p in allowances_percent)
    rows: list[dict[str, Any]] = []
    incumbent: dict[str, Any] | None = None
    for allowance in ordered:
        limit = (1.0 + allowance / 100.0) * anchor_risk
        eligible = [
            row for row in candidates
            if bool(row.get("valid"))
            and float(row.get("risk", float("inf"))) <= limit
        ]
        if incumbent is not None and incumbent not in eligible:
            raise RuntimeError("nested incumbent unexpectedly became ineligible at a larger allowance")
        if not eligible:
            raise RuntimeError(f"no certified candidate within {allowance:g}% risk allowance")
        winner = min(
            eligible,
            key=lambda row: (float(row["action"]), float(row["risk"]), str(row.get("id", ""))),
        )
        if incumbent is not None and float(winner["action"]) > float(incumbent["action"]) + action_tolerance:
            winner = incumbent
        incumbent = winner
        rows.append({
            "allowance_percent": allowance,
            "risk_limit": limit,
            "winner_id": winner.get("id"),
            "risk": float(winner["risk"]),
            "action": float(winner["action"]),
            "candidate": winner,
        })
    return rows

