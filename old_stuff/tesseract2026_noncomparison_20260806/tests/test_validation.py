from pathlib import Path

from manybody_completion.config import load_yaml
from manybody_completion.datasets import generate_dataset, save_dataset
from manybody_completion.validation import validate_dataset


def test_generated_tiny_dataset_passes_numerical_validation(tmp_path: Path):
    config = load_yaml(Path(__file__).parents[1] / "configs" / "tiny_smoke.yaml")
    arrays, metadata = generate_dataset(config)
    dataset_path, _ = save_dataset(arrays, metadata, tmp_path / "tiny.npz")
    report = validate_dataset(dataset_path)
    assert report["numerical_validation_passed"]
    assert report["recomputation_max_abs_errors"]["pair_moments"] == 0.0
    assert max(report["invariance_max_abs_errors"].values()) < 1e-12
    assert report["ambiguity_diagnostics"]["status"] == "calibration_required"
