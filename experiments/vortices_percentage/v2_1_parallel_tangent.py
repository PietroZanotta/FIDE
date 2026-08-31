"""Ordered candidate-level parallelism for V2.1 exact Tangent selection.

This module changes orchestration only: every candidate is still evaluated by
the frozen ``Evaluator.tangent`` endpoint, and results are consumed in the
same candidate order as the serial selection harness.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time
from typing import Any, Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np


def ordered_parallel_map(
    fn: Callable[[Any], Any],
    rows: Iterable[Any],
    *,
    workers: int,
    label: str,
) -> list[Any]:
    """Evaluate independent rows concurrently and return input-order results."""
    items = list(rows)
    if not items:
        return []
    count = len(items)
    started = time.perf_counter()
    print(f"[{label}] 0/{count}", flush=True)
    out = []
    with ThreadPoolExecutor(max_workers=min(int(workers), count)) as executor:
        for completed, result in enumerate(executor.map(fn, items), 1):
            out.append(result)
            if completed == 1 or completed == count or completed % max(1, count // 10) == 0:
                elapsed = time.perf_counter() - started
                rate = completed / max(elapsed, 1.0e-12)
                remaining = (count - completed) / max(rate, 1.0e-12)
                print(
                    f"[{label}] {completed}/{count} elapsed={elapsed:.1f}s "
                    f"eta={remaining:.1f}s",
                    flush=True,
                )
    return out


def install_parallel_tangent_allowance(harness: Any, *, workers: int = 4) -> None:
    """Replace only the harness's Tangent candidate-loop orchestration."""

    def tangent_allowance(
        config,
        schedule,
        exps,
        evaluator,
        starts,
        old,
        population,
        law,
        allowance,
        index,
        incumbent,
    ):
        del index
        out = (
            harness.HERE
            / "allowances"
            / f"risk_{str(allowance).replace('.', 'p')}pct"
            / "tangent.json"
        )
        if out.exists():
            return harness.load_json(out)

        pop_fast, risk4, tan4 = harness.fast_functions(exps, evaluator.bank)
        geometry, projector = harness.geometry_tools(config)
        L_max = float(population["L_max"])
        R_max = float(law["risk_caps"][str(allowance)])
        pop_anchor = float(jax.jit(pop_fast)(jnp.asarray(population["winner"]["eta"])))
        risk_anchor = float(jax.jit(risk4)(jnp.asarray(law["winner"]["eta"])))
        lscale = max(float(config["risk_and_geometry"]["population_slack"]), 1.0e-10)
        rscale = max(R_max - float(law["R_star"]), 1.0e-10)
        constraints = geometry + (
            (
                lambda eta: jnp.maximum(
                    (pop_fast(eta) - (pop_anchor + lscale)) / lscale,
                    (risk4(eta) - (risk_anchor + rscale)) / rscale,
                ),
                0.0,
            ),
        )
        centers = [population["winner"]["eta"], law["winner"]["eta"]]
        if incumbent is not None:
            centers.append(incumbent["winner"]["eta"])
        local = harness.deterministic_local_cloud(
            centers,
            count_per_center=12,
            scale=0.08,
            seed=int(
                schedule["tangent_local_cloud_seed_by_allowance"][str(allowance)]
            ),
            box=config["risk_and_geometry"]["center_box"],
        )
        spec = config["optimization"]["tangent"]
        optimizer_starts = centers + local + list(starts)
        optimized = harness.optimize_multistart_candidates(
            tan4,
            jnp.asarray(optimizer_starts[: int(spec["optimized_starts"])]),
            harness.optimizer_cfg(config, "tangent"),
            constraints=constraints,
            canonicalize=exps[0].family.canonicalize,
            project_iterate=projector,
            vectorize_starts=False,
        )
        pool = {}
        for label, eta in zip(
            ("new_population", "new_law", "previous_tighter_tangent"),
            centers,
        ):
            harness.add_candidate(pool, eta, label)
        for candidate_index, eta in enumerate(local):
            harness.add_candidate(
                pool, eta, f"tangent_local_{candidate_index:03d}"
            )
        for candidate_index, eta in enumerate(starts):
            harness.add_candidate(pool, eta, f"generated_{candidate_index:02d}")
        for candidate_index, row in enumerate(optimized):
            harness.add_candidate(
                pool, row.eta, f"tangent_adam_{candidate_index:02d}"
            )
        for row in old:
            harness.add_candidate(pool, row["eta"], row["label"])
        ranked = harness.fast_rank(
            pool, tan4, constraints, f"tangent {allowance}% fast rank"
        )
        audits, feasible = harness.exact_risk_screen(
            ranked,
            evaluator,
            L_max,
            R_max,
            int(spec["exact_risk_audit_candidates"]),
            incumbent["winner"]["eta"] if incumbent else None,
        )
        if not feasible:
            raise RuntimeError(
                f"no risk-feasible Tangent candidate at {allowance}%"
            )

        def prescreen_one(row):
            record = evaluator.tangent(row["eta"], 32)
            return dict(row, prescreen=record) if record["valid"] else None

        prescreen = [
            row
            for row in ordered_parallel_map(
                prescreen_one,
                feasible,
                workers=workers,
                label=f"tangent {allowance}% 32-trial parallel prescreen",
            )
            if row is not None
        ]
        prescreen.sort(key=lambda row: row["prescreen"]["value"])
        promoted = prescreen[: int(spec["promoted_candidates"])]
        if incumbent is not None and all(
            harness.candidate_key(row["eta"])
            != harness.candidate_key(incumbent["winner"]["eta"])
            for row in promoted
        ):
            promoted.append(
                next(
                    row
                    for row in prescreen
                    if harness.candidate_key(row["eta"])
                    == harness.candidate_key(incumbent["winner"]["eta"])
                )
            )

        def final_one(row):
            return dict(row, final=evaluator.tangent(row["eta"], 128))

        finals = ordered_parallel_map(
            final_one,
            promoted,
            workers=workers,
            label=f"tangent {allowance}% 128-trial parallel final",
        )
        valid = [row for row in finals if row["final"]["valid"]]
        if incumbent is None:
            winner = min(valid, key=lambda row: row["final"]["value"])
        else:
            incumbent_row = next(
                row
                for row in valid
                if harness.candidate_key(row["eta"])
                == harness.candidate_key(incumbent["winner"]["eta"])
            )
            challenger = min(valid, key=lambda row: row["final"]["value"])
            winner = (
                challenger
                if challenger["final"]["value"]
                < incumbent_row["final"]["value"] - 1.0e-6
                else incumbent_row
            )
        result = {
            "status": "PASS",
            "stage": "tangent",
            "allowance_percent": allowance,
            "winner": winner,
            "risk_audits": audits,
            "prescreen": prescreen,
            "finalists": finals,
            "seed": int(
                schedule["tangent_local_cloud_seed_by_allowance"][str(allowance)]
            ),
        }
        harness.atomic_json(out, result)
        return result

    harness.tangent_allowance = tangent_allowance

