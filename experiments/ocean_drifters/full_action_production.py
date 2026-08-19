"""Resumable production full-action sweep for the ocean-drifter experiment."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from mfsi.cache import file_sha256, fingerprint, write_json_atomic

try:
    from .action import _read_csv, _write_csv
    from .full_action import OceanWeightedPoissonPilot
except ImportError:  # direct ``python experiments/ocean_drifters/run.py`` invocation
    from action import _read_csv, _write_csv
    from full_action import OceanWeightedPoissonPilot


class OceanFullActionProduction:
    """Run the pilot-authorized method over all tangent-ready layout-times."""

    def __init__(self, experiment, analysis_dir: Path, output_dir: Path):
        self.experiment = experiment
        self.analysis = Path(analysis_dir)
        self.output = Path(output_dir)
        self.tables = self.analysis / "tables"
        self.cfg = experiment.cfg["action"]["full_action_production"]
        self.action_cfg = experiment.cfg["action"]
        self.runner = OceanWeightedPoissonPilot(
            experiment, analysis_dir, output_dir
        )

        pilot_rows = _read_csv(
            experiment._resolve(self.action_cfg["poisson_pilot_table"])
        )
        if not pilot_rows or not all(
            row.get("pilot_full_action_valid") == "True" for row in pilot_rows
        ):
            raise RuntimeError(
                "the frozen variational pilot has not authorized production"
            )

        tangent_path = experiment._resolve(
            self.action_cfg["tangent_readiness_table"]
        )
        tangent_rows = _read_csv(tangent_path)
        ready_rows = [row for row in tangent_rows if row.get("valid") == "True"]
        self.designs = np.asarray(
            [int(row["design_index"]) for row in ready_rows], dtype=int
        )
        expected_layouts = int(self.cfg["layout_count"])
        if len(self.designs) != expected_layouts:
            raise RuntimeError(
                f"production freeze expects {expected_layouts} tangent-ready "
                f"layouts, found {len(self.designs)}"
            )
        if len(self.runner.times) != int(self.cfg["time_count"]):
            raise RuntimeError("production time grid differs from its frozen contract")

        self.integrated_tangent = {
            int(row["design_index"]): float(row["tangent_action"])
            for row in ready_rows
        }
        self.runner.designs = self.designs.copy()
        self.runner.local_by_design = {
            int(design): int(
                np.flatnonzero(self.runner.all_designs == design)[0]
            )
            for design in self.designs
        }
        self.runner.integrated_tangent = dict(self.integrated_tangent)
        self.runner._build_soft_moment_penalties()

        self.resolution = tuple(
            int(value) for value in self.cfg["grid_resolution"]
        )
        pilot_fine = tuple(
            int(value)
            for value in self.runner.cfg["grid_resolutions"][-1]
        )
        if self.resolution != pilot_fine:
            raise RuntimeError("production must use the pilot-authorized fine grid")
        self.checkpoints = self.output / "cache/full_action_production"
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.output / "progress.json"
        # Only numerical production fields participate in checkpoint identity.
        # Reporting paths and diagnostic-only thresholds may be added without
        # invalidating already completed scientific solves.
        checkpoint_contract = {
            key: self.cfg[key]
            for key in (
                "layout_count",
                "time_count",
                "grid_resolution",
                "design_batch_size",
                "source_chunk_size",
                "adaptive_through_source_index",
                "minimum_valid_layout_fraction",
                "time_table",
                "summary_table",
            )
        }
        self.input_signature = fingerprint({
            "schema": 1,
            "production": checkpoint_contract,
            "variational": self.action_cfg["variational_poisson"],
            "adaptive": self.action_cfg["adaptive_variational_quadrature"],
            "soft_projection": self.action_cfg["soft_moment_projection"],
            "designs": self.designs.tolist(),
            "tangent_table": file_sha256(tangent_path),
            "pilot_table": file_sha256(
                experiment._resolve(self.action_cfg["poisson_pilot_table"])
            ),
            "moment_cache": file_sha256(experiment._resolve(
                "experiments/ocean_drifters/cache/action_moments_positive_kernel.npz"
            )),
            "reference": file_sha256(
                experiment.paths["reference_checkpoint"]
            ),
            "endpoint": file_sha256(
                experiment.paths["conditioned_endpoint_estimator"]
            ),
            "final_test_accessed": False,
        })

    def _checkpoint_path(self, source: int) -> Path:
        return self.checkpoints / f"time_{source:03d}.json"

    def _checkpoint_signature(self, source: int) -> str:
        return fingerprint({
            "input_signature": self.input_signature,
            "source_time_index": int(source),
        })

    def _load_checkpoint(self, source: int) -> list[dict[str, Any]] | None:
        path = self._checkpoint_path(source)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        rows = payload.get("rows", [])
        if (
            payload.get("signature") != self._checkpoint_signature(source)
            or payload.get("final_test_accessed") is not False
            or len(rows) != len(self.designs)
            or {int(row["design_index"]) for row in rows}
            != set(self.designs.tolist())
        ):
            return None
        return rows

    def _save_checkpoint(
        self, source: int, rows: list[dict[str, Any]]
    ) -> None:
        write_json_atomic(self._checkpoint_path(source), {
            "schema_version": 1,
            "signature": self._checkpoint_signature(source),
            "source_time_index": int(source),
            "row_count": len(rows),
            "rows": rows,
            "final_test_accessed": False,
        })

    def _write_progress(self, completed_sources: int, started: float) -> None:
        write_json_atomic(self.progress_path, {
            "schema_version": 1,
            "stage": "full_action_production",
            "completed_time_count": int(completed_sources),
            "total_time_count": len(self.runner.times),
            "completed_case_count": int(completed_sources * len(self.designs)),
            "total_case_count": int(len(self.runner.times) * len(self.designs)),
            "elapsed_seconds": time.perf_counter() - started,
            "resumable": True,
            "final_test_accessed": False,
        })

    def _apply_required_adaptive_audit(
        self,
        row: dict[str, Any],
        system: dict[str, Any],
        points: np.ndarray,
        dx: float,
        base_log_mass: np.ndarray,
        velocity: np.ndarray,
    ) -> None:
        audit = self.runner._adaptive_variational_audit(
            system, points, dx, base_log_mass, velocity
        )
        row.update({
            "adaptive_quadrature_attempted": True,
            "adaptive_quadrature_required_by_production_freeze": True,
            **audit,
        })
        row["full_action_density"] = audit["adaptive_action_density"]
        row["maximum_relative_rank_action_change"] = audit[
            "adaptive_maximum_relative_rank_action_change"
        ]
        row["rank_sensitivity_valid"] = audit[
            "adaptive_rank_sensitivity_valid"
        ]
        tangent = float(row["tangent_action_density"])
        action = float(row["full_action_density"])
        tolerance = float(
            self.runner.variational_cfg[
                "tangent_full_inequality_relative_tolerance"
            ]
        )
        row["tangent_full_inequality_valid"] = bool(
            np.isfinite(action)
            and tangent
            <= action + tolerance * max(abs(action), abs(tangent), 1.0)
        )
        row["solve_accepted_before_refinement"] = bool(
            audit["adaptive_quadrature_valid"]
            and audit["adaptive_solver_success"]
            and row["rank_sensitivity_valid"]
            and row["tangent_full_inequality_valid"]
        )

    def _run_source_chunk(
        self, sources: np.ndarray
    ) -> dict[int, list[dict[str, Any]]]:
        self.runner.source_indices = np.asarray(sources, dtype=int)
        points, dx, log_base, velocity = self.runner._reference_grid(
            self.resolution,
            source_indices=self.runner.source_indices,
            cache_namespace="full_action_production_reference",
        )
        time_local = {
            int(source): local for local, source in enumerate(sources)
        }
        rows_by_source: dict[int, list[dict[str, Any]]] = {
            int(source): [] for source in sources
        }
        batch_size = int(self.cfg["design_batch_size"])
        adaptive_max_source = int(self.cfg["adaptive_through_source_index"])
        adaptive_workers = int(self.cfg.get("adaptive_worker_count", 1))
        use_adaptive = bool(np.any(sources <= adaptive_max_source))
        executor = (
            ThreadPoolExecutor(max_workers=adaptive_workers)
            if use_adaptive and adaptive_workers > 1 else None
        )
        if executor is not None:
            # Initialize each cached JAX callable once on the main thread so
            # workers never race to create duplicate compiled functions.
            for source in sources:
                if int(source) <= adaptive_max_source:
                    self.runner._reference_at_points(points[:1], int(source))
        try:
            for start in range(0, len(self.designs), batch_size):
                batch = self.designs[start : start + batch_size]
                self.runner.designs = batch
                systems = self.runner._systems_for_grid(
                    self.resolution, points, dx, log_base, velocity
                )
                rows, _ = self.runner._solve_variational(systems, dx)
                pairs = list(zip(systems, rows, strict=True))
                if executor is not None:
                    futures = []
                    for system, row in pairs:
                        source = int(system["source_time_index"])
                        if source <= adaptive_max_source:
                            local = time_local[source]
                            futures.append(executor.submit(
                                self._apply_required_adaptive_audit,
                                row,
                                system,
                                points,
                                dx,
                                log_base[local],
                                velocity[local],
                            ))
                    for future in futures:
                        future.result()
                for system, row in pairs:
                    source = int(system["source_time_index"])
                    local = time_local[source]
                    if source <= adaptive_max_source:
                        if executor is None:
                            self._apply_required_adaptive_audit(
                                row,
                                system,
                                points,
                                dx,
                                log_base[local],
                                velocity[local],
                            )
                        quadrature_method = "adaptive_nested_local_refinement"
                        stability_valid = bool(row["adaptive_quadrature_valid"])
                    else:
                        row.update({
                            "adaptive_quadrature_attempted": False,
                            "adaptive_quadrature_required_by_production_freeze": False,
                            "adaptive_quadrature_valid": False,
                        })
                        quadrature_method = "pilot_authorized_fine_structured_grid"
                        stability_valid = True
                    row.update({
                        "production_quadrature_method": quadrature_method,
                        "production_quadrature_stability_valid": stability_valid,
                        "pilot_method_authorized": True,
                        "production_case_valid": bool(
                            stability_valid
                            and row["solve_accepted_before_refinement"]
                        ),
                        "density_modified": False,
                        "operator_floor": 0.0,
                        "final_test_accessed": False,
                    })
                    rows_by_source[source].append(row)
                print(
                    f"[ocean full production] times={sources[0]}-{sources[-1]} "
                    f"layouts={start + 1}-{min(start + batch_size, len(self.designs))}"
                    f"/{len(self.designs)}",
                    flush=True,
                )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
        self.runner.designs = self.designs.copy()
        return rows_by_source

    def _summaries(
        self, rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        times = np.asarray(self.runner.times, dtype=np.float64)
        inequality_tolerance = float(
            self.runner.variational_cfg[
                "tangent_full_inequality_relative_tolerance"
            ]
        )
        for design in self.designs:
            local = sorted(
                (
                    row for row in rows
                    if int(row["design_index"]) == int(design)
                ),
                key=lambda row: int(row["source_time_index"]),
            )
            full_density = np.asarray(
                [float(row["full_action_density"]) for row in local]
            )
            tangent_density = np.asarray(
                [float(row["tangent_action_density"]) for row in local]
            )
            full_action = (
                float(np.trapezoid(full_density, times))
                if len(local) == len(times) and np.isfinite(full_density).all()
                else math.nan
            )
            grid_tangent_action = (
                float(np.trapezoid(tangent_density, times))
                if len(local) == len(times) and np.isfinite(tangent_density).all()
                else math.nan
            )
            canonical_tangent = self.integrated_tangent[int(design)]
            integrated_lower_bound = bool(
                np.isfinite(full_action)
                and grid_tangent_action
                <= full_action
                + inequality_tolerance
                * max(abs(full_action), abs(grid_tangent_action), 1.0)
            )
            valid_count = sum(
                bool(row["production_case_valid"]) for row in local
            )
            layout_valid = bool(
                len(local) == len(times)
                and valid_count == len(times)
                and integrated_lower_bound
            )
            summaries.append({
                "design_index": int(design),
                "design_id": self.experiment.sensor_bank.design_ids[design],
                "production_time_count": len(local),
                "valid_time_count": valid_count,
                "solver_success_time_count": sum(
                    bool(row["solver_success"]) for row in local
                ),
                "rank_sensitivity_valid_time_count": sum(
                    bool(row["rank_sensitivity_valid"]) for row in local
                ),
                "pointwise_tangent_full_valid_time_count": sum(
                    bool(row["tangent_full_inequality_valid"]) for row in local
                ),
                "adaptive_required_time_count": sum(
                    bool(row["adaptive_quadrature_required_by_production_freeze"])
                    for row in local
                ),
                "adaptive_valid_time_count": sum(
                    bool(row["adaptive_quadrature_valid"]) for row in local
                ),
                "canonical_tangent_action": canonical_tangent,
                "production_grid_tangent_action": grid_tangent_action,
                "production_full_action": full_action,
                "integrated_tangent_full_inequality_valid": integrated_lower_bound,
                "full_action_valid": layout_valid,
                "density_modified": False,
                "operator_floor": 0.0,
                "final_test_accessed": False,
            })
        valid = [row for row in summaries if row["full_action_valid"]]
        if valid:
            ranked = sorted(
                valid, key=lambda row: row["production_full_action"]
            )
            rank_by_design = {
                int(row["design_index"]): rank
                for rank, row in enumerate(ranked, start=1)
            }
            optimum = ranked[0]
            for row in summaries:
                row["production_full_action_optimum"] = (
                    row["design_index"] == optimum["design_index"]
                )
                row["production_full_action_rank"] = rank_by_design.get(
                    int(row["design_index"]), ""
                )
        else:
            for row in summaries:
                row["production_full_action_optimum"] = False
                row["production_full_action_rank"] = ""
        return summaries

    def _diagnostics(
        self, rows: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Flag numerical failures and robust within-time action outliers."""
        threshold = float(self.cfg["log_action_outlier_robust_z_threshold"])
        robust_score: dict[tuple[int, int], float] = {}
        for source in range(len(self.runner.times)):
            local = [
                row for row in rows
                if int(row["source_time_index"]) == source
                and np.isfinite(float(row["full_action_density"]))
                and float(row["full_action_density"]) > 0.0
            ]
            values = np.log10(np.asarray([
                float(row["full_action_density"]) for row in local
            ]))
            if not len(values):
                continue
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            scale = max(1.4826 * mad, 1.0e-12)
            for row, value in zip(local, values, strict=True):
                robust_score[(
                    int(row["design_index"]), source
                )] = abs(float(value) - median) / scale

        diagnostics: list[dict[str, Any]] = []
        reason_counts = {
            "solver": 0,
            "rank_sensitivity": 0,
            "tangent_full_inequality": 0,
            "adaptive_quadrature": 0,
            "log_action_outlier": 0,
        }
        for row in rows:
            reasons: list[str] = []
            if not bool(row["solver_success"]):
                reasons.append("solver")
            if not bool(row["rank_sensitivity_valid"]):
                reasons.append("rank_sensitivity")
            if not bool(row["tangent_full_inequality_valid"]):
                reasons.append("tangent_full_inequality")
            if (
                bool(row["adaptive_quadrature_required_by_production_freeze"])
                and not bool(row["adaptive_quadrature_valid"])
            ):
                reasons.append("adaptive_quadrature")
            key = (
                int(row["design_index"]), int(row["source_time_index"])
            )
            score = robust_score.get(key, math.nan)
            outlier = bool(np.isfinite(score) and score > threshold)
            if outlier:
                reasons.append("log_action_outlier")
            for reason in set(reasons):
                reason_counts[reason] += 1
            if reasons:
                diagnostics.append({
                    "design_index": int(row["design_index"]),
                    "design_id": row["design_id"],
                    "source_time_index": int(row["source_time_index"]),
                    "day": float(row["day"]),
                    "production_case_valid": bool(row["production_case_valid"]),
                    "diagnostic_reasons": ";".join(reasons),
                    "full_action_density": float(row["full_action_density"]),
                    "tangent_action_density": float(row["tangent_action_density"]),
                    "log_action_robust_z_score": score,
                    "maximum_relative_rank_action_change": float(
                        row["maximum_relative_rank_action_change"]
                    ),
                    "scaled_weak_relative_residual": float(
                        row["scaled_weak_relative_residual"]
                    ),
                    "compatibility_relative_residual": float(
                        row["compatibility_relative_residual"]
                    ),
                    "adaptive_quadrature_required": bool(
                        row["adaptive_quadrature_required_by_production_freeze"]
                    ),
                    "adaptive_quadrature_valid": bool(
                        row["adaptive_quadrature_valid"]
                    ),
                    "density_modified": False,
                    "operator_floor": 0.0,
                    "final_test_accessed": False,
                })
        return diagnostics, reason_counts

    def _plots(
        self,
        rows: list[dict[str, Any]],
        summaries: list[dict[str, Any]],
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure_dir = self.experiment._resolve(self.cfg["figure_directory"])
        figure_dir.mkdir(parents=True, exist_ok=True)
        ordered = sorted(
            summaries,
            key=lambda row: (
                not bool(row["full_action_valid"]),
                float(row["production_full_action"])
                if np.isfinite(float(row["production_full_action"])) else math.inf,
            ),
        )
        fig, axis = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
        colors = [
            "#2b6cb0" if row["full_action_valid"] else "#c53030"
            for row in ordered
        ]
        axis.bar(
            np.arange(len(ordered)),
            [float(row["production_full_action"]) for row in ordered],
            color=colors,
        )
        axis.set(
            xlabel="production rank (valid layouts first)",
            ylabel="integrated full action",
        )
        axis.grid(axis="y", alpha=0.2)
        fig.savefig(figure_dir / "integrated_full_action_ranked.png", dpi=190)
        plt.close(fig)

        row_by_key = {
            (int(row["design_index"]), int(row["source_time_index"])): row
            for row in rows
        }
        matrix = np.asarray([
            [
                math.log10(max(float(row_by_key[(int(design), source)][
                    "full_action_density"
                ]), 1.0e-300))
                for source in range(len(self.runner.times))
            ]
            for design in self.designs
        ])
        invalid_y, invalid_x = np.nonzero(np.asarray([
            [
                not bool(row_by_key[(int(design), source)][
                    "production_case_valid"
                ])
                for source in range(len(self.runner.times))
            ]
            for design in self.designs
        ]))
        fig, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
        image = axis.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
        if len(invalid_x):
            axis.scatter(invalid_x, invalid_y, s=9, c="#e53e3e", marker="x")
        axis.set(xlabel="time index", ylabel="tangent-ready layout index")
        fig.colorbar(image, ax=axis, label="log10 full-action density")
        fig.savefig(figure_dir / "full_action_density_heatmap.png", dpi=190)
        plt.close(fig)

    def _finalize_artifacts(
        self,
        rows: list[dict[str, Any]],
        summaries: list[dict[str, Any]],
        diagnostics: list[dict[str, Any]],
        reason_counts: dict[str, int],
        result: dict[str, Any],
        time_table: Path,
        summary_table: Path,
    ) -> None:
        diagnostics_table = self.experiment._resolve(
            self.cfg["diagnostics_table"]
        )
        _write_csv(diagnostics_table, diagnostics)
        self._plots(rows, summaries)
        freeze_path = self.experiment._resolve(self.cfg["freeze_manifest"])
        selection_frozen = bool(result["full_action_valid"])
        write_json_atomic(freeze_path, {
            "schema_version": 1,
            "production_input_signature": self.input_signature,
            "production_layout_count": result["production_layout_count"],
            "production_time_count": result["production_time_count"],
            "production_case_count": result["production_case_count"],
            "valid_layout_count": result["valid_layout_count"],
            "valid_layout_fraction": result["valid_layout_fraction"],
            "minimum_valid_layout_fraction": result[
                "minimum_valid_layout_fraction"
            ],
            "full_action_valid": result["full_action_valid"],
            "selection_frozen": selection_frozen,
            "selected_design_index": (
                result["production_full_action_optimum_design_index"]
                if selection_frozen else None
            ),
            "selected_design_id": (
                result["production_full_action_optimum_design_id"]
                if selection_frozen else None
            ),
            "selected_full_action": (
                result["production_full_action_optimum"]
                if selection_frozen else None
            ),
            "failure_reason_counts": reason_counts,
            "artifacts": {
                "time_table": {
                    "path": str(time_table.relative_to(self.experiment._resolve("."))),
                    "sha256": file_sha256(time_table),
                },
                "summary_table": {
                    "path": str(summary_table.relative_to(self.experiment._resolve("."))),
                    "sha256": file_sha256(summary_table),
                },
                "diagnostics_table": {
                    "path": str(diagnostics_table.relative_to(self.experiment._resolve("."))),
                    "sha256": file_sha256(diagnostics_table),
                },
            },
            "thresholds_changed_after_production_started": False,
            "density_modified": False,
            "operator_floor": 0.0,
            "final_test_accessed": False,
        })
        report_path = self.experiment._resolve(self.cfg["report"])
        failed_layouts = sum(not row["full_action_valid"] for row in summaries)
        report_path.write_text(
            "# Ocean-drifter full-action production report\n\n"
            "## Decision\n\n"
            f"The frozen 68x181 variational production sweep is "
            f"{'valid' if result['full_action_valid'] else 'not valid'}: "
            f"{result['valid_layout_count']}/68 layouts pass every required "
            f"time-local gate ({result['valid_layout_fraction']:.2%}), against "
            "the frozen 95% requirement.\n\n"
            f"Failed layouts: {failed_layouts}. Failure reasons by layout-time "
            f"case: `{json.dumps(reason_counts, sort_keys=True)}`.\n\n"
            "The robust log-action outlier diagnostic uses a predeclared "
            f"within-time MAD score threshold of {self.cfg['log_action_outlier_robust_z_threshold']}. "
            "It is diagnostic only and never removes a case.\n\n"
            "## Frozen choice\n\n"
            + (
                f"`{result['production_full_action_optimum_design_id']}` is "
                f"frozen with integrated full action "
                f"{result['production_full_action_optimum']:.17g}.\n\n"
                if selection_frozen else
                "No production choice was frozen because the validity gate did not pass.\n\n"
            )
            + "No thresholds were changed, no density/operator floor was used, "
            "and the 69 final-test trajectories remain locked and untouched.\n",
            encoding="utf-8",
        )

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        total_times = len(self.runner.times)
        completed = {
            source: rows
            for source in range(total_times)
            if (rows := self._load_checkpoint(source)) is not None
        }
        self._write_progress(len(completed), started)
        adaptive_max_source = int(self.cfg["adaptive_through_source_index"])
        regular_chunk_size = int(self.cfg["source_chunk_size"])
        source = 0
        while source < total_times:
            if source in completed:
                source += 1
                continue
            chunk_size = 1 if source <= adaptive_max_source else regular_chunk_size
            candidates = np.arange(source, min(source + chunk_size, total_times))
            pending = np.asarray(
                [value for value in candidates if int(value) not in completed],
                dtype=int,
            )
            if not len(pending):
                source += chunk_size
                continue
            local_rows = self._run_source_chunk(pending)
            for local_source in pending:
                local_source = int(local_source)
                rows = local_rows[local_source]
                if len(rows) != len(self.designs):
                    raise RuntimeError("production source checkpoint is incomplete")
                self._save_checkpoint(local_source, rows)
                completed[local_source] = rows
                self._write_progress(len(completed), started)
                print(
                    f"[ocean full production] checkpoint time={local_source + 1}"
                    f"/{total_times} cases={len(completed) * len(self.designs)}"
                    f"/{total_times * len(self.designs)}",
                    flush=True,
                )
            source += chunk_size

        rows = [
            row
            for source in range(total_times)
            for row in completed[source]
        ]
        rows.sort(key=lambda row: (
            int(row["design_index"]), int(row["source_time_index"])
        ))
        summaries = self._summaries(rows)
        time_table = self.experiment._resolve(self.cfg["time_table"])
        summary_table = self.experiment._resolve(self.cfg["summary_table"])
        _write_csv(time_table, rows)
        _write_csv(summary_table, summaries)
        valid = [row for row in summaries if row["full_action_valid"]]
        valid_fraction = len(valid) / max(len(summaries), 1)
        required_fraction = float(self.cfg["minimum_valid_layout_fraction"])
        full_action_valid = valid_fraction >= required_fraction
        optimum = next(
            (
                row for row in summaries
                if row["production_full_action_optimum"]
            ),
            None,
        )
        result = {
            "schema_version": 1,
            "poisson_backend": self.runner.poisson_backend,
            "production_layout_count": len(self.designs),
            "production_time_count": total_times,
            "production_case_count": len(rows),
            "valid_layout_count": len(valid),
            "valid_layout_fraction": valid_fraction,
            "minimum_valid_layout_fraction": required_fraction,
            "full_action_valid": full_action_valid,
            "production_full_action_optimum_design_index": (
                int(optimum["design_index"]) if optimum else None
            ),
            "production_full_action_optimum_design_id": (
                optimum["design_id"] if optimum else None
            ),
            "production_full_action_optimum": (
                float(optimum["production_full_action"]) if optimum else None
            ),
            "resumable": True,
            "density_modified": False,
            "operator_floor": 0.0,
            "final_test_accessed": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        diagnostics, reason_counts = self._diagnostics(rows)
        result["diagnostic_row_count"] = len(diagnostics)
        result["failure_reason_counts"] = reason_counts
        self._finalize_artifacts(
            rows,
            summaries,
            diagnostics,
            reason_counts,
            result,
            time_table,
            summary_table,
        )
        write_json_atomic(self.progress_path, {
            **result,
            "stage": "full_action_production",
            "completed_time_count": total_times,
            "completed_case_count": len(rows),
        })
        return result


def run_full_action_production(
    experiment, analysis_dir: Path, output_dir: Path
) -> dict[str, Any]:
    return OceanFullActionProduction(experiment, analysis_dir, output_dir).run()
