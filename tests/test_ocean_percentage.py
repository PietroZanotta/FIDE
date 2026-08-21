from __future__ import annotations

import numpy as np

from experiments.ocean_drifters_percentage.run_pareto import choose_methods


def test_ocean_percentage_uses_common_certified_candidate_set() -> None:
    risk = np.asarray([1.0, 1.04, 1.08, 1.09])
    eligible = np.asarray([True, True, True, False])
    actions = {
        0: {"tangent_action": 8.0, "full_action": 12.0, "certified": True},
        1: {"tangent_action": 6.0, "full_action": 10.0, "certified": True},
        2: {"tangent_action": 5.0, "full_action": 11.0, "certified": True},
        3: {"tangent_action": 1.0, "full_action": 1.0, "certified": True},
    }

    selected = choose_methods(risk, actions, eligible)

    assert selected == {"law": 0, "tangent": 2, "full": 1}
    assert actions[selected["full"]]["full_action"] <= actions[
        selected["tangent"]
    ]["full_action"]


def test_ocean_percentage_excludes_uncertified_layouts() -> None:
    risk = np.asarray([1.0, 1.01])
    eligible = np.asarray([True, True])
    actions = {
        0: {"tangent_action": 4.0, "full_action": 5.0, "certified": True},
        1: {"tangent_action": 1.0, "full_action": 2.0, "certified": False},
    }

    assert choose_methods(risk, actions, eligible) == {
        "law": 0,
        "tangent": 0,
        "full": 0,
    }
