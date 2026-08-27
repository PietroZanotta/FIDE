"""Strictly gated tiny production Galerkin sensor refinement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from .deep_ritz import CertificateConfig
from .full_gradient import (
    forcing_state,
    minimum_sensor_separation,
    periodic_branch_distance,
    reconstruct_moments,
    smooth_separation_penalty,
    wrap_periodic,
)
from .production_artifacts import require_production_output_path
from .production_basis import load_dictionary
from .production_galerkin import audit_hybrid_solutions, make_basis_evaluators
from .production_gradient import (
    evaluate_local_eta,
    precompute_fixed_potential_rows,
    production_hybrid_envelope_value_and_grad,
)
from .production_workflow import load_production_data
from .workflow import selection_risk, write_json


def run_production_refinement(
    cfg: dict[str, Any], artifact_dir: Path, output_dir: Path,
    *, allowance_percent: float,
) -> dict[str, Any]:
    output_dir = require_production_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.parent
    gradient_path = root / "gradient_checks" / "result.json"
    convergence_path = root / "convergence" / "result.json"
    if not gradient_path.is_file() or not convergence_path.is_file():
        raise RuntimeError("production convergence/gradient prerequisite is missing")
    gradient_result = json.loads(gradient_path.read_text(encoding="utf-8"))
    convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
    if not gradient_result.get("passed") or not convergence.get("basis_convergence_passed"):
        result = {
            "ran": False,
            "reason": "production Galerkin convergence or gradient prerequisite failed",
            "outcome_classification": gradient_result.get(
                "outcome_classification",
                "B. PRODUCTION GALERKIN SOLVER VALID, ETA GRADIENT NOT YET VALIDATED",
            ),
        }
        write_json(output_dir / "result.json", result)
        return result
    data = load_production_data(cfg, artifact_dir)
    dictionary = load_dictionary(
        root / "convergence" / "features" / "hybrid_dictionary.npz",
        box=tuple(cfg["physics"]["box"]),
    )
    evaluators = make_basis_evaluators(
        dictionary, int(data.selection_problem.times.shape[0])
    )
    eta = jnp.asarray(cfg["envelope"]["eta0"], dtype=jnp.float64)
    start_eta = eta
    start_payload, current_solve = evaluate_local_eta(
        eta, cfg, data, dictionary, evaluators
    )
    law_eta = jnp.asarray(cfg["envelope"]["law_eta"], dtype=jnp.float64)
    law_risk = float(selection_risk(law_eta, data))
    risk_limit = law_risk * (1.0 + float(allowance_percent) / 100.0)
    settings = cfg["production_galerkin"]["refinement"]
    history = []
    for step in range(int(settings["steps"])):
        potential_rows, kinetic_rows = precompute_fixed_potential_rows(
            dictionary, current_solve.coefficients, data, evaluators,
            chunk_size=int(cfg["production_galerkin"]["chunk_size"]),
        )
        _, action_gradient = production_hybrid_envelope_value_and_grad(
            eta, current_solve.coefficients, data, potential_rows, kinetic_rows
        )

        def penalty(design):
            risk_hinge = jax.nn.relu(selection_risk(design, data) / risk_limit - 1.0)
            return (
                float(settings["risk_penalty"]) * risk_hinge * risk_hinge
                + float(settings["separation_penalty"])
                * smooth_separation_penalty(design, data.selection_problem.family)
            )

        penalty_value, penalty_gradient = jax.value_and_grad(penalty)(eta)
        total_gradient = action_gradient + penalty_gradient
        accepted = False
        attempts = []
        accepted_payload = start_payload
        accepted_solve = current_solve
        accepted_risk = float(selection_risk(eta, data))
        for backtrack in range(int(settings["backtracking_steps"])):
            learning_rate = float(settings["learning_rate"]) * (0.5 ** backtrack)
            proposal = wrap_periodic(
                eta - learning_rate * total_gradient,
                data.selection_problem.family,
            )
            geometry = bool(jax.device_get(
                data.selection_problem.family.geometry_valid(proposal)
            ))
            if not geometry:
                attempts.append({
                    "backtrack": backtrack, "learning_rate": learning_rate,
                    "accepted": False, "reason": "geometry",
                })
                continue
            proposal_risk = float(selection_risk(proposal, data))
            if proposal_risk > risk_limit:
                attempts.append({
                    "backtrack": backtrack, "learning_rate": learning_rate,
                    "accepted": False, "reason": "exact risk",
                    "risk": proposal_risk,
                })
                continue
            proposal_payload, proposal_solve = evaluate_local_eta(
                proposal, cfg, data, dictionary, evaluators
            )
            ranks_stable = bool(jnp.array_equal(
                current_solve.numerical_rank, proposal_solve.numerical_rank
            ))
            action_decreased = bool(
                proposal_payload["action"]
                < start_payload["action"] - float(settings["minimum_action_decrease"])
                if step == 0
                else proposal_payload["action"]
                < history[-1]["action_after"] - float(settings["minimum_action_decrease"])
            )
            accepted = bool(
                proposal_payload["hard_gates_passed"]
                and ranks_stable and action_decreased
            )
            attempts.append({
                "backtrack": backtrack, "learning_rate": learning_rate,
                "accepted": accepted,
                "risk": proposal_risk,
                "action": proposal_payload["action"],
                "ranks_stable": ranks_stable,
                "hard_gates_passed": proposal_payload["hard_gates_passed"],
            })
            if accepted:
                eta = proposal
                accepted_payload = proposal_payload
                accepted_solve = proposal_solve
                accepted_risk = proposal_risk
                break
        history.append({
            "step": step,
            "accepted": accepted,
            "eta": jax.device_get(eta).tolist(),
            "action_after": accepted_payload["action"],
            "risk_after": accepted_risk,
            "action_gradient": jax.device_get(action_gradient).tolist(),
            "penalty_value": float(penalty_value),
            "penalty_gradient": jax.device_get(penalty_gradient).tolist(),
            "total_gradient_norm": float(jnp.linalg.norm(total_gradient)),
            "attempts": attempts,
        })
        if not accepted:
            break
        current_solve = accepted_solve
    end_payload, end_solve = evaluate_local_eta(
        eta, cfg, data, dictionary, evaluators
    )
    reconstruction = reconstruct_moments(eta, data.selection_problem)
    audit_state = forcing_state(
        eta, data.selection_problem, data.ritz_audit_bank, reconstruction
    )
    certificate = audit_hybrid_solutions(
        dictionary,
        end_solve.coefficients[None],
        data,
        eta,
        reconstruction,
        audit_state,
        CertificateConfig(**cfg["production_galerkin"]["certificate_thresholds"]),
        chunk_size=int(cfg["production_galerkin"]["chunk_size"]),
    )[0]
    end_risk = float(selection_risk(eta, data))
    improved = bool(
        end_payload["action"] < start_payload["action"]
        and end_risk <= risk_limit
        and end_payload["hard_gates_passed"]
        and certificate["valid"]
    )
    result = {
        "ran": True,
        "steps_requested": int(settings["steps"]),
        "steps_accepted": sum(bool(row["accepted"]) for row in history),
        "start_eta": jax.device_get(start_eta).tolist(),
        "end_eta": jax.device_get(eta).tolist(),
        "start_galerkin_action": start_payload["action"],
        "end_galerkin_action": end_payload["action"],
        "galerkin_action_reduction": start_payload["action"] - end_payload["action"],
        "end_selection_risk": end_risk,
        "law_risk": law_risk,
        "risk_ceiling": risk_limit,
        "end_minimum_separation": float(minimum_sensor_separation(
            eta, data.selection_problem.family
        )),
        "end_periodic_branch_distance": float(periodic_branch_distance(
            eta, data.selection_problem.family
        )),
        "end_local_diagnostics": end_payload,
        "end_held_out_certificate": certificate,
        "history": history,
        "eligible_for_authoritative_crosscheck": improved,
        "authoritative_improvement_claimed": False,
        "outcome_classification": (
            "A. PRODUCTION GALERKIN SOLVER AND ETA GRADIENT VALIDATED"
        ),
    }
    write_json(output_dir / "result.json", result)
    return result


__all__ = ["run_production_refinement"]
