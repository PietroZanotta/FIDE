from __future__ import annotations

"""Attach an externally frozen Tangent geometry for a post-hoc cross-evaluation."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from common import SCRIPT_DIR, artifact_dirs, load_config, write_json_atomic
from mfsi.cache import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-protocol", required=True, type=Path)
    parser.add_argument("--source-supplement", required=True, type=Path)
    args = parser.parse_args()
    protocol_path = args.target_protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["mode"] != "posthoc":
        raise RuntimeError("external Tangent adoption is only allowed for post-hoc analysis")
    output = (SCRIPT_DIR / protocol["primary_output"]).resolve()
    dirs = artifact_dirs(output)
    primary_path = dirs["results"] / "frozen_manifest.json"
    source_path = args.source_supplement.resolve()
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("status") != "frozen_before_hidden_validation":
        raise RuntimeError("source Tangent geometry was not frozen before its hidden validation")
    supplement_dir = dirs["results"] / "tangent_supplement"
    supplement_dir.mkdir(parents=True, exist_ok=True)
    target = supplement_dir / "frozen_manifest.json"
    if target.exists():
        raise RuntimeError("target Tangent supplement already exists")
    receipt = {
        "schema_version": 1,
        "experiment": protocol["name"],
        "mode": "posthoc",
        "status": "posthoc_cross_evaluation_of_externally_frozen_v5_tangent_geometry",
        "interpretation": (
            "Post-hoc v4 cross-evaluation of the Tangent geometry selected and frozen "
            "by the v5 Tangent supplement before v5 hidden generation. This does not "
            "alter or extend the original v4 prospective claim."
        ),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_input_hashes": {
            "primary_manifest_sha256": file_sha256(primary_path),
            "protocol_sha256": file_sha256(protocol_path),
            "source_tangent_manifest_sha256": file_sha256(source_path),
            "source_sha256": file_sha256(Path(__file__)),
        },
        "hidden_data_imported_or_loaded_by_selection": False,
        "primary_hidden_existed_at_selection": True,
        "protocol": protocol,
        "risk_anchor": primary["selection_metrics"]["law_risk"],
        "risk_ceiling": primary["selection_metrics"]["risk_ceiling"],
        "gradient_starts": source["gradient_starts"],
        "distinct_candidates": source["distinct_candidates"],
        "risk_feasible_candidates": source["risk_feasible_candidates"],
        "selected": source["selected"],
        "source_tangent_manifest": str(source_path),
    }
    write_json_atomic(target, receipt)
    print(json.dumps({"target": str(target), "eta": receipt["selected"]["eta"]}, sort_keys=True))


if __name__ == "__main__":
    main()
