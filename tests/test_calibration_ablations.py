import pytest

from experiments.calibration.run_calibration_ablations import (
    _validate_shared_metadata,
)


def test_shared_metadata_allows_last_bit_float_noise():
    reference = {
        "indices": [0, 2, 4],
        "angular_mean": [-0.05960949698055760, 0.06029942212031079],
        "backend": "jax",
    }
    candidate = {
        "indices": [0, 2, 4],
        "angular_mean": [-0.05960949698055759, 0.06029942212031078],
        "backend": "jax",
    }

    _validate_shared_metadata(reference, candidate, "post_hoc")


def test_shared_metadata_keeps_indices_exact():
    with pytest.raises(ValueError, match=r"shared\.indices\[2\]"):
        _validate_shared_metadata(
            {"indices": [0, 2, 4]},
            {"indices": [0, 2, 5]},
            "post_hoc",
        )
