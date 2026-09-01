from __future__ import annotations

import copy
import ast
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for path in (HERE, REPO / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import load_config


def test_v5_is_exact_v4_protocol_except_preregistered_identity_and_seeds():
    v4 = load_config(HERE / "configs" / "production_v4.json")
    v5 = load_config(HERE / "configs" / "production_v5.json")
    assert v5["name"] == "prospective_v5_robust_full_replication"
    assert v5["replication"]["parent_hidden_validation_use"] == "forbidden"
    assert set(v4["seeds"].values()).isdisjoint(v5["seeds"].values())

    normalized_v4 = copy.deepcopy(v4)
    normalized_v5 = copy.deepcopy(v5)
    normalized_v4.pop("schema_version")
    normalized_v5.pop("schema_version")
    normalized_v4.pop("name")
    normalized_v5.pop("name")
    normalized_v5.pop("replication")
    normalized_v4.pop("seeds")
    normalized_v5.pop("seeds")
    normalized_v4["v4"].pop("artifact_source_run")
    normalized_v5["v4"].pop("artifact_source_run")
    assert normalized_v5 == normalized_v4


def test_v5_selection_and_validation_seed_roles_are_disjoint():
    cfg = load_config(HERE / "configs" / "production_v5.json")
    selection = {
        value for key, value in cfg["seeds"].items() if not key.startswith("validation_")
    }
    validation = {
        value for key, value in cfg["seeds"].items() if key.startswith("validation_")
    }
    assert selection.isdisjoint(validation)
    assert len(selection) == 9
    assert len(validation) == 4


def test_v5_tangent_supplement_is_preregistered_and_selection_has_no_validation_import():
    protocol = json.loads(
        (HERE / "configs" / "tangent_supplement_v5.json").read_text(encoding="utf-8")
    )
    primary = load_config(HERE / "configs" / "production_v5.json")
    assert protocol["mode"] == "prospective"
    assert protocol["starts"] == 24
    assert protocol["steps"] == 60
    assert set(protocol["seeds"].values()).isdisjoint(primary["seeds"].values())
    tree = ast.parse((HERE / "tangent_supplement_select.py").read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "v4_validate" not in imports
    assert "tangent_supplement_validate" not in imports
