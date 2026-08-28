from __future__ import annotations

import numpy as np
import pytest

from experiments.active_nematic_unbalance_percentage.domain import SplitConfig, make_run_split


def test_explicit_split_preserves_declared_roles() -> None:
    config = SplitConfig(
        train_runs=4,
        design_runs=2,
        validation_runs=2,
        train_indices=(0, 2, 4, 6),
        design_indices=(1, 3),
        validation_indices=(5, 7),
    )
    split = make_run_split(config)
    assert np.array_equal(split.train, [0, 2, 4, 6])
    assert np.array_equal(split.design, [1, 3])
    assert np.array_equal(split.validation, [5, 7])


def test_explicit_split_must_be_a_complete_partition() -> None:
    with pytest.raises(ValueError, match="partition"):
        SplitConfig(
            train_runs=2,
            design_runs=1,
            validation_runs=1,
            train_indices=(0, 4),
            design_indices=(1,),
            validation_indices=(2,),
        )
