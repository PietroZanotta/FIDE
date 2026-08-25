"""Command-line entry point for the isolated continuous-Full experiment."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
for search_path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

jax.config.update("jax_enable_x64", True)

from mfsi.config import load_config

from .workflow import (
    OUTPUT_ROOT,
    CandidateEvaluation,
    certify_designs,
    optimize_continuously,
    prepare_experiment,
    require_output_path,
    run_gradient_check,
    save_candidate_checkpoint,
    write_json,
)
from .rigorous_gradient_check import run_rigorous_gradient_check
from .galerkin_workflow import (
    run_galerkin_convergence,
    run_galerkin_fixed,
    run_galerkin_gradient_check,
    run_galerkin_refinement,
)
from .production_artifacts import (
    PRODUCTION_ROOT,
    inspect_production_source,
    run_production_preflight,
)
from .production_galerkin import run_production_galerkin_convergence
from .production_gradient import run_production_gradient_check
from .production_refinement import run_production_refinement
from .production_authoritative import run_production_authoritative_crosscheck
from .production_workflow import run_production_reproduction
from .fast_production import FAST_ROOT
from .fast_workflow import (
    run_fast_benchmark,
    run_authoritative_selection,
    run_fast_multistart,
    run_fast_refinement,
    run_fast_validation,
    run_gradient_convergence,
    run_local_gradient_audit,
)


def _deep_update(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_rigorous_config(path: Path, *, smoke: bool) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rigorous_smoke = raw.pop("rigorous_smoke", {})
    raw.pop("smoke", None)
    raw.pop("preflight", None)
    return _deep_update(raw, rigorous_smoke) if smoke else raw


def _default_output(mode: str, smoke: bool) -> Path:
    if mode == "production-preflight":
        return PRODUCTION_ROOT / "preflight"
    if mode == "reproduce-production":
        return PRODUCTION_ROOT / "reproduction"
    if mode == "galerkin-production-convergence":
        return PRODUCTION_ROOT / "convergence"
    if mode == "galerkin-production-gradient-check":
        return PRODUCTION_ROOT / "gradient_checks"
    if mode == "galerkin-production-refine":
        return PRODUCTION_ROOT / "refinement"
    if mode == "production-authoritative-crosscheck":
        return PRODUCTION_ROOT / "authoritative_crosscheck"
    fast_leaves = {
        "benchmark-production-galerkin": "profiling",
        "production-gradient-convergence": "gradient_convergence",
        "production-local-gradient-audit": "local_gradient_audit",
        "production-refine-3pct": "trajectories",
        "production-multistart-3pct": "multistart",
        "production-authoritative-3pct": "authoritative",
        "production-validate-3pct": "validation",
    }
    if mode in fast_leaves:
        return FAST_ROOT / fast_leaves[mode]
    if mode.startswith("galerkin-"):
        leaf = {
            "galerkin-fixed": "fixed_eta",
            "galerkin-convergence": "convergence",
            "galerkin-gradient-check": "gradient_checks",
            "galerkin-refine": "refinement",
        }[mode]
        return OUTPUT_ROOT / "galerkin" / leaf / ("smoke" if smoke else "full")
    if mode == "rigorous-gradient-check":
        return OUTPUT_ROOT / "gradient_checks" / (
            "rigorous_smoke" if smoke else "rigorous_full"
        )
    if mode == "gradient-check":
        return OUTPUT_ROOT / "gradient_checks" / ("smoke" if smoke else "full")
    if mode == "certify":
        return OUTPUT_ROOT / "validation" / ("smoke" if smoke else "full")
    return OUTPUT_ROOT / ("smoke" if smoke else "selection")


def _result_etas(path: Path) -> list[jax.Array]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "selected" not in payload or "eta" not in payload["selected"]:
        raise ValueError(f"{path} does not contain selected.eta")
    return [jnp.asarray(payload["selected"]["eta"], dtype=jnp.float64)]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continuous sensor refinement with a fixed-theta Ritz envelope gradient"
    )
    parser.add_argument(
        "--mode", required=True,
        choices=(
            "smoke", "gradient-check", "rigorous-gradient-check", "optimize", "certify",
            "galerkin-fixed", "galerkin-convergence", "galerkin-gradient-check",
            "galerkin-refine",
            "production-preflight",
            "reproduce-production",
            "galerkin-production-convergence",
            "galerkin-production-gradient-check",
            "galerkin-production-refine",
            "production-authoritative-crosscheck",
            "benchmark-production-galerkin",
            "production-gradient-convergence",
            "production-local-gradient-audit",
            "production-refine-3pct",
            "production-multistart-3pct",
            "production-authoritative-3pct",
            "production-validate-3pct",
        ),
    )
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.json")
    parser.add_argument(
        "--smoke-profile", action="store_true",
        help="use reduced deterministic banks for non-smoke modes",
    )
    parser.add_argument("--allowance", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--frozen-source", type=Path,
        help="read an existing artifact directory; artifacts are copied into local outputs",
    )
    parser.add_argument(
        "--input-result", type=Path,
        help="optimization result whose selected eta should be freshly certified",
    )
    parser.add_argument(
        "--eta", type=float, nargs=8,
        help="explicit sensor coordinates for certify mode",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    smoke = bool(args.mode == "smoke" or args.smoke_profile)
    cfg = (
        _load_rigorous_config(args.config, smoke=smoke)
        if args.mode == "rigorous-gradient-check"
        else load_config(args.config, smoke=smoke)
    )
    output_dir = require_output_path(
        args.output_dir or _default_output(args.mode, smoke)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in {
        "benchmark-production-galerkin", "production-gradient-convergence",
        "production-local-gradient-audit", "production-refine-3pct",
        "production-multistart-3pct", "production-authoritative-3pct",
        "production-validate-3pct",
    }:
        artifact_dir = PRODUCTION_ROOT / "artifacts"
        if args.frozen_source is not None:
            source_audit = inspect_production_source(args.frozen_source, cfg)
            if not source_audit["complete"]:
                raise SystemExit("the supplied production artifact set is incomplete")
        if not (artifact_dir / "isolated_artifact_manifest.json").is_file():
            raise SystemExit("production artifacts are not materialized; run production-preflight")
        runners = {
            "benchmark-production-galerkin": run_fast_benchmark,
            "production-gradient-convergence": run_gradient_convergence,
            "production-local-gradient-audit": run_local_gradient_audit,
            "production-refine-3pct": run_fast_refinement,
            "production-multistart-3pct": run_fast_multistart,
        }
        if args.mode == "production-validate-3pct":
            if args.input_result is None:
                raise SystemExit("--input-result is required for production-validate-3pct")
            result = run_fast_validation(
                cfg, artifact_dir, output_dir, selection_result=args.input_result
            )
            print(f"result={output_dir / 'result.json'}")
            print(f"passed={result.get('ran', False)}")
            if not result.get("ran", False):
                raise SystemExit(2)
            return
        if args.mode == "production-authoritative-3pct":
            result = run_authoritative_selection(cfg, artifact_dir, output_dir)
            print(f"result={output_dir / 'result.json'}")
            print(f"passed={result.get('ran', False)}")
            if not result.get("ran", False):
                raise SystemExit(2)
            return
        result = runners[args.mode](cfg, artifact_dir, output_dir)
        print(f"result={output_dir / 'result.json'}")
        print(f"passed={result.get('passed', result.get('ran', False))}")
        if not result.get("passed", result.get("ran", False)):
            raise SystemExit(2)
        return
    if args.mode == "production-preflight":
        if args.frozen_source is None:
            raise SystemExit("--frozen-source is required for production-preflight")
        result = run_production_preflight(cfg, args.frozen_source, output_dir)
        print(f"result={output_dir / 'result.json'}")
        print(f"production_artifacts_complete={result['passed']}")
        if not result["passed"]:
            raise SystemExit(2)
        return
    if args.mode == "reproduce-production":
        if args.frozen_source is not None:
            preflight = run_production_preflight(
                cfg, args.frozen_source, PRODUCTION_ROOT / "preflight"
            )
            if not preflight["passed"]:
                raise SystemExit(2)
        artifact_dir = PRODUCTION_ROOT / "artifacts"
        if not (artifact_dir / "isolated_artifact_manifest.json").is_file():
            raise SystemExit(
                "production artifacts are not materialized; pass --frozen-source"
            )
        result, _ = run_production_reproduction(cfg, artifact_dir, output_dir)
        print(f"result={output_dir / 'result.json'}")
        print(f"gate_a_passed={result['gate_a_passed']}")
        if not result["gate_a_passed"]:
            raise SystemExit(2)
        return
    if args.mode == "galerkin-production-convergence":
        artifact_dir = PRODUCTION_ROOT / "artifacts"
        if args.frozen_source is not None:
            source_audit = inspect_production_source(args.frozen_source, cfg)
            if not source_audit["complete"]:
                raise SystemExit("the supplied production artifact set is incomplete")
            if not (artifact_dir / "isolated_artifact_manifest.json").is_file():
                preflight = run_production_preflight(
                    cfg, args.frozen_source, PRODUCTION_ROOT / "preflight"
                )
                if not preflight["passed"]:
                    raise SystemExit(2)
        if not (artifact_dir / "isolated_artifact_manifest.json").is_file():
            raise SystemExit(
                "production artifacts are not materialized; pass --frozen-source"
            )
        result, _ = run_production_galerkin_convergence(
            cfg, artifact_dir, output_dir
        )
        print(f"result={output_dir / 'result.json'}")
        print(f"basis_convergence_passed={result.get('basis_convergence_passed', False)}")
        if not result.get("basis_convergence_passed", False):
            raise SystemExit(2)
        return
    if args.mode == "galerkin-production-gradient-check":
        artifact_dir = PRODUCTION_ROOT / "artifacts"
        if args.frozen_source is not None:
            source_audit = inspect_production_source(args.frozen_source, cfg)
            if not source_audit["complete"]:
                raise SystemExit("the supplied production artifact set is incomplete")
        if not (artifact_dir / "isolated_artifact_manifest.json").is_file():
            raise SystemExit(
                "production artifacts are not materialized; run production-preflight"
            )
        result = run_production_gradient_check(cfg, artifact_dir, output_dir)
        print(f"result={output_dir / 'result.json'}")
        print(f"gradient_check_passed={result['passed']}")
        if not result["passed"]:
            raise SystemExit(2)
        return
    if args.mode == "galerkin-production-refine":
        artifact_dir = PRODUCTION_ROOT / "artifacts"
        if args.frozen_source is not None:
            source_audit = inspect_production_source(args.frozen_source, cfg)
            if not source_audit["complete"]:
                raise SystemExit("the supplied production artifact set is incomplete")
        if not (artifact_dir / "isolated_artifact_manifest.json").is_file():
            raise SystemExit(
                "production artifacts are not materialized; run production-preflight"
            )
        result = run_production_refinement(
            cfg, artifact_dir, output_dir, allowance_percent=args.allowance
        )
        print(f"result={output_dir / 'result.json'}")
        print(f"refinement_ran={result['ran']}")
        if not result["ran"]:
            raise SystemExit(2)
        return
    if args.mode == "production-authoritative-crosscheck":
        artifact_dir = PRODUCTION_ROOT / "artifacts"
        if args.frozen_source is not None:
            source_audit = inspect_production_source(args.frozen_source, cfg)
            if not source_audit["complete"]:
                raise SystemExit("the supplied production artifact set is incomplete")
        if not (artifact_dir / "isolated_artifact_manifest.json").is_file():
            raise SystemExit(
                "production artifacts are not materialized; run production-preflight"
            )
        result = run_production_authoritative_crosscheck(
            cfg, artifact_dir, output_dir, allowance_percent=args.allowance
        )
        print(f"result={output_dir / 'result.json'}")
        print(f"authoritative_crosscheck_ran={result['ran']}")
        if not result["ran"]:
            raise SystemExit(2)
        return
    if args.mode.startswith("galerkin-"):
        artifacts = OUTPUT_ROOT / "galerkin" / "artifacts" / (
            "smoke" if smoke else "full"
        )
        legacy_artifacts = OUTPUT_ROOT / "artifacts" / ("smoke" if smoke else "full")
        frozen_source = args.frozen_source
        if frozen_source is None and legacy_artifacts.is_dir():
            frozen_source = legacy_artifacts
    else:
        artifacts = (
            output_dir / "artifacts"
            if args.mode == "rigorous-gradient-check"
            else OUTPUT_ROOT / "artifacts" / ("smoke" if smoke else "full")
        )
        frozen_source = args.frozen_source
    data = prepare_experiment(cfg, artifacts, frozen_source=frozen_source)

    if args.mode == "galerkin-fixed":
        result = run_galerkin_fixed(cfg, data, output_dir)
        print(f"result={output_dir / 'result.json'}")
        print(f"physical_valid={result['physical_valid']}")
        if not result["physical_valid"]:
            raise SystemExit(2)
        return

    if args.mode == "galerkin-convergence":
        result = run_galerkin_convergence(cfg, data, output_dir)
        print(f"result={output_dir / 'result.json'}")
        print(f"basis_convergence_passed={result['basis_convergence_passed']}")
        if not result["basis_convergence_passed"]:
            raise SystemExit(2)
        return

    if args.mode == "galerkin-gradient-check":
        result = run_galerkin_gradient_check(cfg, data, output_dir)
        print(f"result={output_dir / 'result.json'}")
        print(f"gradient_check_passed={result['passed']}")
        if not result["passed"]:
            raise SystemExit(2)
        return

    if args.mode == "galerkin-refine":
        result = run_galerkin_refinement(
            cfg, data, output_dir, allowance_percent=args.allowance
        )
        print(f"result={output_dir / 'result.json'}")
        print(f"refinement_ran={result['ran']}")
        if not result["ran"]:
            raise SystemExit(2)
        return

    if args.mode == "rigorous-gradient-check":
        result = run_rigorous_gradient_check(cfg, data, output_dir)
        print(f"result={output_dir / 'summary.json'}")
        print(f"conclusion={result['conclusion']}")
        if not result["passed"]:
            raise SystemExit(2)
        return

    if args.mode == "gradient-check":
        result, theta = run_gradient_check(
            cfg, data, inner_mode="smoke" if smoke else "full"
        )
        result["mode"] = "gradient-check"
        result["profile"] = "smoke" if smoke else "authoritative"
        write_json(output_dir / "result.json", result)
        evaluation = CandidateEvaluation(theta, {
            "eta": result["eta0"], "risk": None,
            "action": result["envelope_value"],
        })
        save_candidate_checkpoint(
            output_dir / "theta_center.npz", evaluation, role="gradient_check_center"
        )
        print(f"result={output_dir / 'result.json'}")
        print(f"gradient_check_passed={result['passed']}")
        if not result["passed"]:
            raise SystemExit(2)
        return

    if args.mode in {"smoke", "optimize"}:
        result = optimize_continuously(
            cfg,
            data,
            output_dir,
            allowance_percent=args.allowance,
            inner_mode="smoke" if smoke else "full",
        )
        print(f"result={output_dir / 'result.json'}")
        print(f"accepted={result['accepted']}")
        if not result["accepted"]:
            raise SystemExit(2)
        return

    if args.eta is not None:
        etas = [jnp.asarray(args.eta, dtype=jnp.float64)]
    elif args.input_result is not None:
        etas = _result_etas(args.input_result)
    else:
        etas = [jnp.asarray(cfg["envelope"]["eta0"], dtype=jnp.float64)]
    result = certify_designs(
        cfg, data, output_dir, etas, allowance_percent=args.allowance
    )
    print(f"result={output_dir / 'result.json'}")
    print(f"all_valid={result['all_valid']}")
    if not result["all_valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
