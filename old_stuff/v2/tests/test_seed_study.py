from __future__ import annotations

from manybody_completion.seed_study import aggregate_reports


def _report(seed: int, shift: float) -> dict:
    def method(ess: float, score: float) -> dict:
        return {
            "moment_error": 0.01 + shift,
            "ess_fraction": ess,
            "mode_probability_error": 0.02,
            "hidden_energy_score": score,
            "hidden_energy_distance": 0.03,
            "joint_total_variation": 0.04,
            "higher_order_conditional_uq": {
                "triplet_mean": shift,
                "triplet_variance": 0.5,
            },
        }

    return {
        "metadata": {"seed": seed},
        "methods": {
            "Full-E2E": method(0.8 + shift, 0.1 - shift),
            "Calibrated-StopGrad": method(0.7, 0.12),
        },
    }


def test_seed_aggregation_preserves_pairing() -> None:
    aggregate = aggregate_reports([_report(1, 0.0), _report(2, 0.02)])
    assert aggregate["seed_count"] == 2
    assert aggregate["paired_full_minus_stopgrad"]["ess_fraction"]["mean"] > 0
    assert aggregate["methods"]["Full-E2E"]["moment_error"]["count"] == 2
