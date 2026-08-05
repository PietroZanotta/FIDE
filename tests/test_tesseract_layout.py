import ast
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]
TESSERACT = ROOT / "tesseracts" / "physical_relaxation"


def test_physical_relaxation_tesseract_has_complete_build_context():
    required = {
        "tesseract_api.py",
        "tesseract_config.yaml",
        "tesseract_requirements.txt",
        "README.md",
    }
    assert required.issubset({path.name for path in TESSERACT.iterdir()})
    assert (TESSERACT / "examples" / "s1_gradient_payload.json").exists()

    config = yaml.safe_load((TESSERACT / "tesseract_config.yaml").read_text())
    assert config["name"] == "manybody-physical-relaxation"
    assert config["version"] == "0.1.0"

    requirements = (TESSERACT / "tesseract_requirements.txt").read_text()
    assert "jax==0.8.1" in requirements
    assert "equinox" in requirements
    assert "../.." not in requirements
    assert config["build_config"]["package_data"] == [
        ["../../src/manybody_completion", "/tesseract/manybody_completion"]
    ]


def test_physical_relaxation_api_defines_required_and_gradient_endpoints():
    module = ast.parse((TESSERACT / "tesseract_api.py").read_text())
    functions = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
    assert {
        "apply_jit",
        "apply",
        "abstract_eval",
        "jacobian",
        "jacobian_vector_product",
        "vector_jacobian_product",
    }.issubset(functions)


MOMENT_TESSERACT = ROOT / "tesseracts" / "moment_projection"


def test_moment_projection_tesseract_has_complete_build_context():
    required = {
        "tesseract_api.py",
        "tesseract_config.yaml",
        "tesseract_requirements.txt",
        "README.md",
    }
    assert required.issubset({path.name for path in MOMENT_TESSERACT.iterdir()})

    config = yaml.safe_load((MOMENT_TESSERACT / "tesseract_config.yaml").read_text())
    assert config["name"] == "manybody-moment-projection"
    assert config["version"] == "0.1.0"
    assert config["metadata"]["basis_pruning"] == "explicit_mask"

    requirements = (MOMENT_TESSERACT / "tesseract_requirements.txt").read_text()
    assert "jax==0.8.1" in requirements
    assert "equinox" in requirements
    assert "../.." not in requirements
    assert config["build_config"]["package_data"] == [
        ["../../src/manybody_completion", "/tesseract/manybody_completion"]
    ]
    payload_path = MOMENT_TESSERACT / "examples" / "s2_payload.json"
    assert payload_path.exists()
    assert set(json.loads(payload_path.read_text())) == {"inputs"}


def test_moment_projection_api_defines_required_and_gradient_endpoints():
    module = ast.parse((MOMENT_TESSERACT / "tesseract_api.py").read_text())
    functions = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
    assert {
        "apply_jit",
        "apply",
        "abstract_eval",
        "jacobian",
        "jacobian_vector_product",
        "vector_jacobian_product",
    }.issubset(functions)
