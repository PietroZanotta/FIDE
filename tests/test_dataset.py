from pathlib import Path

import numpy as np

from manybody_completion.config import load_yaml
from manybody_completion.datasets import generate_dataset, select_matched_cross_regime_pairs


def test_tiny_dataset_shapes_and_reproducibility():
    config_path = Path(__file__).parents[1] / "configs" / "tiny_smoke.yaml"
    config = load_yaml(config_path)
    arrays_a, metadata_a = generate_dataset(config)
    arrays_b, metadata_b = generate_dataset(config)
    assert arrays_a["coordinates"].shape == (4, 4, 6, 2)
    assert arrays_a["pair_moments"].shape == (4, 6)
    assert arrays_a["angular_moments"].shape == (4, 3)
    np.testing.assert_array_equal(arrays_a["coordinates"], arrays_b["coordinates"])
    assert metadata_a["seed"] == metadata_b["seed"] == 7
    matches = select_matched_cross_regime_pairs(arrays_a, max_pairs=2)
    assert len(matches) == 2
    assert all(item["regime_a"] != item["regime_b"] for item in matches)
