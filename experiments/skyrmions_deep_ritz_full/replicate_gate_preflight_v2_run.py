"""CLI for the development-only replicate-gate preflight v2."""

from __future__ import annotations

import argparse

from .replicate_gate_preflight_v2 import OUTPUT_ROOT, analyze, freeze_design


def _print(result: dict) -> None:
    print("SOURCE VERIFIED")
    print(f"candidate count: {result['candidate_count']}")
    print(f"fresh development pairs: {result['fresh_development_pairs']}")
    print(f"new scientific evaluations: {result['new_scientific_evaluations']}")
    print("75% FAMILY")
    labels = ("3/4", "6/8", "12/16")
    print(f"{'Metric':<34} {'3/4':>12} {'6/8':>12} {'12/16':>12}")
    family = result["primary_75_percent_family"]
    metrics = (
        ("0.5% empty-set rate", "0.5", "empty_set_rate"),
        ("0.5% p10 survivor count", "0.5", "p10_survivors"),
        ("0.5% median survivor count", "0.5", "median_survivors"),
        ("1% empty-set rate", "1.0", "empty_set_rate"),
        ("1% p10 survivor count", "1.0", "p10_survivors"),
        ("2% empty-set rate", "2.0", "empty_set_rate"),
        ("0.5% median Jaccard", "0.5", "median_jaccard"),
        ("1% median Jaccard", "1.0", "median_jaccard"),
        ("2% median Jaccard", "2.0", "median_jaccard"),
        ("3% median Jaccard", "3.0", "median_jaccard"),
        ("0.5% expected survivors", "0.5", "expected_survivors"),
        ("1% expected survivors", "1.0", "expected_survivors"),
        (">=10 diverse-start availability", "0.5", "fraction_ge10_diverse"),
    )
    for title, allowance, key in metrics:
        print(f"{title:<34} " + " ".join(f"{family[label][allowance][key]:>12.5g}" for label in labels))
    print(f"{'relative pair cost':<34} " + " ".join(f"{family[label]['relative_pair_cost']:>11}x" for label in labels))
    print("STRICTNESS COMPARISON")
    strict = result["strictness_comparison"]
    for loose, hard in (("6/8", "7/8"), ("12/16", "14/16")):
        print(f"{loose} vs {hard}")
        for allowance in ("0.5", "1.0", "2.0", "3.0"):
            print(
                f"  {allowance}%: empty={strict[loose][allowance]['empty_set_rate']:.5g}/"
                f"{strict[hard][allowance]['empty_set_rate']:.5g} "
                f"median={strict[loose][allowance]['median_survivors']:.1f}/"
                f"{strict[hard][allowance]['median_survivors']:.1f}"
            )
    recommendation = result["recommendation"]
    print("RECOMMENDATION:")
    print(recommendation["recommendation"])
    print(f"M: {recommendation['recommended_M']}")
    print(f"required passes: {recommendation['recommended_required_passes']}")
    print(f"fraction: {recommendation['recommended_fraction']}")
    print("per-bank rESS threshold: 0.05 unchanged")
    print(f"reason: {recommendation['reason']}")
    print("NO new scientific banks generated")
    print("NO candidate generation")
    print("NO Tangent")
    print("NO Full")
    print("NO validation")
    print("NO official protocol created")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze", "analyze", "run"), default="run")
    args = parser.parse_args()
    if args.mode == "freeze":
        result = freeze_design()
        print("SOURCE VERIFIED")
        print(f"candidate count: {result['source']['candidate_count']}")
        print(f"fresh development pairs: {result['source']['fresh_development_pairs']}")
        print("new scientific evaluations: 0")
    else:
        result = analyze()
        _print(result)
    print(f"mode={args.mode}")
    print(f"output_root={OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
