import pytest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.skyrmions_deep_ritz.selection import (
    BankArtifact,
    BankRegistry,
    nested_certified_selection,
)
from experiments.skyrmions_deep_ritz.experiment import (
    _selection_signature_config,
    _validation_signature_config,
)


def test_validation_bank_cannot_leak_into_selection():
    registry = BankRegistry({
        "design": BankArtifact("design", "design-1", {"x": 1}),
        "validation": BankArtifact("validation", "validation-1", {"x": 2}),
    })
    assert registry.get("design", consumer="selection") == {"x": 1}
    with pytest.raises(RuntimeError, match="bank leakage blocked"):
        registry.get("validation", consumer="selection")
    assert registry.get("validation", consumer="validation") == {"x": 2}


def test_nested_pareto_retains_certified_incumbent():
    candidates = [
        {"id": "law", "risk": 10.0, "action": 8.0, "valid": True},
        {"id": "one", "risk": 10.08, "action": 6.0, "valid": True},
        {"id": "three", "risk": 10.25, "action": 4.0, "valid": True},
        {"id": "invalid-cheap", "risk": 10.0, "action": 0.0, "valid": False},
    ]
    rows = nested_certified_selection(
        candidates, anchor_risk=10.0, allowances_percent=[0.5, 1.0, 3.0]
    )
    assert [row["winner_id"] for row in rows] == ["law", "one", "three"]
    assert [row["action"] for row in rows] == sorted(
        [row["action"] for row in rows], reverse=True
    )


def test_cache_signatures_ignore_only_irrelevant_pareto_configuration():
    cfg = {
        "execution_profile": "authoritative",
        "physics": {"n_particles": 16},
        "banks": {
            "ritz_train_samples": 4096,
            "ritz_audit_samples": 4096,
            "validation_fit_samples": 8192,
            "validation_audit_samples": 4096,
        },
        "deep_ritz": {"lbfgs_iterations": 160},
        "search": {"risk_allowance_percent": 3.0},
    }
    changed_allowance = {
        **cfg,
        "search": {"risk_allowance_percent": 5.0},
    }
    assert _selection_signature_config(cfg) == _selection_signature_config(changed_allowance)
    assert _validation_signature_config(cfg) == _validation_signature_config(changed_allowance)

    changed_validation_bank = {
        **cfg,
        "banks": {**cfg["banks"], "validation_fit_samples": 16384},
    }
    assert _selection_signature_config(cfg) == _selection_signature_config(changed_validation_bank)
    assert _validation_signature_config(cfg) != _validation_signature_config(changed_validation_bank)

    changed_solver = {
        **cfg,
        "deep_ritz": {"lbfgs_iterations": 320},
    }
    assert _selection_signature_config(cfg) != _selection_signature_config(changed_solver)
    assert _validation_signature_config(cfg) != _validation_signature_config(changed_solver)
