"""Read-only postmortem of the sealed V3.4 held-out Full failure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import jax

from . import official_b1_pareto_v3_chunked_guard_run as runtime


study = runtime.study
OUTPUT = study.OUTPUT_ROOT
DIAGNOSTIC_PATH = OUTPUT / "heldout_validation" / "failure_diagnostic.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(relative: str) -> Any:
    return json.loads((OUTPUT / relative).read_text())


def _failed_certificate_gates(certificate: dict[str, Any]) -> list[dict[str, Any]]:
    names = (
        ("maximum_weak_residual", "maximum_weak_residual"),
        ("maximum_energy_residual", "maximum_energy_residual"),
        ("maximum_gauge_residual", "maximum_gauge_residual"),
        ("maximum_moment_rate_residual", "maximum_moment_rate_residual"),
    )
    return [
        {
            "gate": gate,
            "observed": float(certificate[value]),
            "threshold": float(certificate["thresholds"][gate]),
            "ratio_to_threshold": float(
                certificate[value] / certificate["thresholds"][gate]
            ),
        }
        for value, gate in names
        if float(certificate[value]) > float(certificate["thresholds"][gate])
    ]


def main() -> None:
    runtime.activate()
    selection = _read("selection/selection_seal.json")
    heldout = _read("heldout_validation/results.json")
    heldout_manifest = _read("heldout_validation/manifest.json")
    persisted = {
        (row["method"], row["allowance_percent"]): row
        for row in heldout["rows"]
    }
    methods_by_geometry: dict[str, list[dict[str, Any]]] = {}
    unique_rows: dict[str, dict[str, Any]] = {}
    for row in selection["rows"]:
        key = row["eta_sha256"]
        unique_rows.setdefault(key, row)
        methods_by_geometry.setdefault(key, []).append({
            "method": row["method"],
            "allowance_percent": row["allowance_percent"],
        })

    data = study.base._heldout_data()
    context = study.base.JaxGalerkinContext(
        study.base.effective_config(),
        data,
        study.base.DICTIONARY_PATH,
        chunk_size=int(study.base.require_protocol()["solver"]["chunk_size"]),
    )
    geometries = []
    for key, row in unique_rows.items():
        evaluation = context.evaluate(row["eta"], gradient=False)
        audit, audit_seconds = context.audit(evaluation.payload)
        public = study.base._public_timed(evaluation)
        representative = persisted[(row["method"], row["allowance_percent"])]
        risk_difference = float(public["risk"] - representative["heldout_risk"])
        train_action_difference = float(
            public["action"] - representative["heldout_train_K280_action"]
        )
        audit_action_difference = float(
            audit["heldout_certificate"]["action"]
            - representative["heldout_audit_K280_action"]
        )
        certificate = audit["heldout_certificate"]
        geometries.append({
            "eta_sha256": key,
            "eta": row["eta"],
            "selected_as": methods_by_geometry[key],
            "persisted_heldout_full_certificate_pass": representative[
                "heldout_full_certificate_pass"
            ],
            "recomputed": {
                "heldout_risk": public["risk"],
                "train_K280_action": public["action"],
                "audit_K280_action": certificate["action"],
                "algebra_valid": public["algebra_valid"],
                "geometry_valid": public["geometry_valid"],
                "train_forcing": public["train_forcing_audit"],
                "search_valid": public["search_valid"],
                "audit_forcing": audit["audit_forcing"],
                "heldout_certificate": certificate,
                "overall_full_valid": audit["valid"],
                "audit_seconds": audit_seconds,
            },
            "failed_full_certificate_gates": _failed_certificate_gates(certificate),
            "reproduction_differences": {
                "heldout_risk": risk_difference,
                "train_K280_action": train_action_difference,
                "audit_K280_action": audit_action_difference,
            },
            "reproduced_persisted_values": bool(
                abs(risk_difference) <= 1.0e-12
                and abs(train_action_difference) <= 1.0e-12
                and abs(audit_action_difference) <= 1.0e-12
                and bool(audit["valid"])
                == bool(representative["heldout_full_certificate_pass"])
            ),
        })

    payload = {
        "schema_version": 1,
        "classification": "READ_ONLY_POSTMORTEM_AFTER_TERMINAL_AUTHORITY",
        "optimization_run": False,
        "selection_changed": False,
        "authority_changed": False,
        "root_seed": 20261003,
        "alternate_root_seeds_tested": [],
        "selection_seal_sha256": _sha256(
            OUTPUT / "selection" / "selection_seal.json"
        ),
        "heldout_manifest_sha256": _sha256(
            OUTPUT / "heldout_validation" / "manifest.json"
        ),
        "heldout_results_sha256": _sha256(
            OUTPUT / "heldout_validation" / "results.json"
        ),
        "heldout_manifest_passed": heldout_manifest["passed"],
        "persisted_heldout_result_passed": heldout["passed"],
        "unique_geometry_count": len(geometries),
        "all_persisted_values_reproduced": all(
            row["reproduced_persisted_values"] for row in geometries
        ),
        "geometries": geometries,
    }
    study.base.atomic_json(DIAGNOSTIC_PATH, payload)
    print(json.dumps({
        "diagnostic": str(DIAGNOSTIC_PATH),
        "unique_geometry_count": len(geometries),
        "all_persisted_values_reproduced": payload[
            "all_persisted_values_reproduced"
        ],
        "failed_gates": {
            row["eta_sha256"]: row["failed_full_certificate_gates"]
            for row in geometries
        },
    }, indent=2))


if __name__ == "__main__":
    with jax.default_device((jax.devices("gpu") or jax.devices())[0]):
        main()
