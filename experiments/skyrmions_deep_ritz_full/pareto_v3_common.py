"""Prospective constants, seals, and atomic I/O for skyrmion Pareto v3."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from mfsi.cache import fingerprint

from .production_artifacts import PRODUCTION_ROOT, file_sha256


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs" / "official_galerkin_pareto_v3"
V2_OUTPUT_ROOT = ROOT / "outputs" / "official_galerkin_pareto_v2"
DIAGNOSTIC_ROOT = OUTPUT_ROOT / "diagnostic_v2_audit_map"
ALL_ALLOWANCES_DIAGNOSTIC_ROOT = (
    OUTPUT_ROOT / "diagnostic_v2_audit_all_allowances"
)
PROTOCOL_PATH = OUTPUT_ROOT / "protocol.json"
PROTOCOL_HASH_PATH = OUTPUT_ROOT / "protocol_hash.txt"
PROTOCOL_DOCUMENT = ROOT / "OFFICIAL_GALERKIN_PARETO_V3_PROTOCOL.md"
REPORT_PATH = ROOT / "OFFICIAL_GALERKIN_PARETO_V3_EVALUATION.md"
ARTIFACT_DIR = PRODUCTION_ROOT / "artifacts"
DICTIONARY_PATH = (
    ROOT / "outputs" / "galerkin_only_3pct" / "cache" / "dictionaries"
    / "dictionary_K280.npz"
)
EXPECTED_DICTIONARY_SHA256 = (
    "37e9b60fcb92c4e5a0ee7ec1651fb7f8889f7ac6bdb02d3bd314e9ef40833326"
)
VERSION = "skyrmion_official_galerkin_pareto_v3"
ALLOWANCES = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
K = 280
RANK_TOLERANCE = 1.0e-12
MINIMUM_RESS = 0.05
MAXIMUM_ENERGY_RESIDUAL = 0.08
BANK_SIZES = {
    "screen": 8192,
    "search_train": 32768,
    "periodic_audit": 16384,
    "authoritative_train": 65536,
    "authoritative_audit": 65536,
}
VALIDATION_SIZES = {
    "truth": 5000,
    "reference_fit": 16384,
    "reference_audit": 16384,
}
V2_EXPECTED = {
    "protocol_document": "f7353f821e194ea86a3dc1f891633fcecb77b0909d6c17229942de2032c2f0e6",
    "evaluation_document": "00965dbf9bf78763f0a32ae1a184010dce33042d290b385f60560fe6841487df",
    "protocol_json": "8360afb812b4036cbadaa0e2ca4f12d92c10ffa6900c1a22c5f231eea18dbf3b",
    "failure_json": "e73d4e3ac89f9562bc6038c8ea4c3d0d9bb040b78dfffa477a982585329acf40",
    "inner_protocol": "22a33ce47b2a3cc17ff063d100b878ac32c3ef6cc1a2b3e10a6eb8cd076488f1",
}
V3_PHASE1_EXPECTED = {
    "summary_json": "fd856bf004932e467a7abf87e5f158864899d1afe7b592eef2bc01bae35d3d33",
    "inventory_json": "20f7cf4c6c9db8efea82b7e4f2c84b144f9d5959f2d69506681dcfa6b0323078",
}
HISTORICAL_REPORTS = (
    "FINAL_3PCT_GALERKIN_CROSSCHECK.md",
    "GALERKIN_RESOLUTION_STUDY.md",
    "GALERKIN_K280_QUADRATURE_QUALIFICATION.md",
    "ESS_QUALIFICATION_AND_PERFORMANCE.md",
    "OFFICIAL_GALERKIN_PARETO_V2_PROTOCOL.md",
    "OFFICIAL_GALERKIN_PARETO_V2_EVALUATION.md",
)


def canonical(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical(payload)).hexdigest()


def output_path(path: Path) -> Path:
    resolved, root = Path(path).resolve(), OUTPUT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Pareto-v3 output must be beneath {root}: {resolved}")
    return resolved


def atomic_json(path: Path, payload: Any, *, immutable: bool = False) -> None:
    path = output_path(path)
    if immutable and path.exists():
        raise RuntimeError(f"refusing to overwrite immutable Pareto-v3 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_text(path: Path, value: str, *, immutable: bool = False) -> None:
    path = output_path(path)
    if immutable and path.exists():
        raise RuntimeError(f"refusing to overwrite immutable Pareto-v3 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def slug(value: float) -> str:
    return str(float(value)).replace(".", "p").removesuffix("p0")


def eta_key(eta: Any) -> str:
    return payload_sha256([float(value) for value in eta])[:20]


def selection_ceiling(law_risk: float, allowance: float) -> float:
    return float(law_risk) * (1.0 + float(allowance) / 100.0)


def validation_ceiling(law_risk: float, allowance: float) -> float:
    return float(law_risk) * (1.0 + float(allowance) / 100.0 + 0.05)


def derive_seed(global_seed: int, scope: str, label: str) -> dict[str, Any]:
    text = f"{int(global_seed)}:skyrmion:official_pareto_v3:{scope}:{label}"
    digest = hashlib.sha256(text.encode()).hexdigest()
    return {
        "label": label,
        "derivation_text": text,
        "sha256": digest,
        "seed": int(digest[:16], 16) % (2**31 - 1),
    }


def hashes(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(OUTPUT_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in paths
    ]


def signature(protocol: dict[str, Any], kind: str, extra: Any = None) -> str:
    return fingerprint(
        {
            "kind": kind,
            "protocol_sha256": protocol["protocol_sha256"],
            "dictionary_sha256": file_sha256(DICTIONARY_PATH),
            "K": K,
            "dtype": "float64",
            "extra": extra,
        }
    )


def verify_v2_frozen() -> dict[str, Any]:
    paths = {
        "protocol_document": ROOT / "OFFICIAL_GALERKIN_PARETO_V2_PROTOCOL.md",
        "evaluation_document": ROOT / "OFFICIAL_GALERKIN_PARETO_V2_EVALUATION.md",
        "protocol_json": V2_OUTPUT_ROOT / "protocol.json",
        "failure_json": V2_OUTPUT_ROOT / "failure.json",
    }
    actual = {name: file_sha256(path) for name, path in paths.items()}
    for name, digest in actual.items():
        if digest != V2_EXPECTED[name]:
            raise RuntimeError(f"frozen v2 {name} hash changed")
    protocol = read_json(paths["protocol_json"])
    if protocol.get("protocol_sha256") != V2_EXPECTED["inner_protocol"]:
        raise RuntimeError("frozen v2 inner protocol seal changed")
    if (V2_OUTPUT_ROOT / "selection" / "selection_hash.txt").exists():
        raise RuntimeError("v2 unexpectedly contains a selection freeze")
    if (V2_OUTPUT_ROOT / "fresh_validation").exists():
        raise RuntimeError("v2 unexpectedly contains fresh validation")
    tree = [
        {
            "path": str(path.relative_to(V2_OUTPUT_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(V2_OUTPUT_ROOT.rglob("*"))
        if path.is_file()
    ]
    return {
        "v2_frozen": True,
        "expected_hashes": V2_EXPECTED,
        "verified_hashes": actual,
        "output_tree": tree,
        "output_tree_sha256": payload_sha256(tree),
        "selection_frozen": False,
        "validation_accessed": False,
        "classification": "FAILED PROTOCOL",
    }


def verify_v3_phase1_frozen() -> dict[str, Any]:
    """Verify the immutable development-only 0.5% diagnostic and its source."""
    paths = {
        "summary_json": DIAGNOSTIC_ROOT / "summary.json",
        "inventory_json": DIAGNOSTIC_ROOT / "v2_inventory.json",
    }
    actual = {name: file_sha256(path) for name, path in paths.items()}
    for name, digest in actual.items():
        if digest != V3_PHASE1_EXPECTED[name]:
            raise RuntimeError(f"frozen v3 Phase-1 {name} hash changed")
    v2 = verify_v2_frozen()
    summary = read_json(paths["summary_json"])
    inventory = read_json(paths["inventory_json"])
    if summary.get("v2_output_tree_sha256") != v2["output_tree_sha256"]:
        raise RuntimeError("v3 Phase-1 source v2 tree no longer matches")
    if inventory.get("output_tree_sha256") != v2["output_tree_sha256"]:
        raise RuntimeError("v3 Phase-1 v2 inventory no longer matches")
    if summary.get("audit_ress_valid_count") != 0:
        raise RuntimeError("v3 Phase-1 classification-C evidence changed")
    return {
        "phase1_frozen": True,
        "verified_hashes": actual,
        "v2_output_tree_sha256": v2["output_tree_sha256"],
        "summary": summary,
    }
