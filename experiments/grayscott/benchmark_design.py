"""Design-only regime scan and endpoint calibration for Experiment C."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .calibration import calibrate_shared_target
from .morphology_metrics import (
    metric_rows,
    pooled_otsu_threshold,
    summarize_rows,
    weighted_metric_mean,
)
from .observables import ShellDefinition, field_observables, fit_standardization
from .simulator import generate_initial_conditions, simulate


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "expC_grayscott_design.yaml"


def _json_default(value):
    if isinstance(value, (np.ndarray, jax.Array)):
        return np.asarray(value).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n")


def _read_config(path: Path) -> dict:
    # JSON is a strict subset of YAML, keeping the config dependency-free.
    return json.loads(path.read_text())


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _provenance(config: dict, config_path: Path) -> dict:
    design = set(range(
        int(config["seeds"]["design_initial_condition_start"]),
        int(config["seeds"]["design_initial_condition_start"])
        + int(config["seeds"]["design_initial_condition_count"]),
    ))
    training = set(map(int, config["seeds"]["training_model"]))
    evaluation = set(map(int, config["seeds"]["final_evaluation_bank"]))
    if design & training or design & evaluation or training & evaluation:
        raise ValueError("design, training, and final evaluation seed roles must be disjoint")
    return {
        "config_path": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "device": str(jax.devices()[0]),
        "simulator_dtype": "float32",
        "calibration_dtype": "float64",
        "seed_roles": {
            "design_initial_conditions": sorted(design),
            "training_model": sorted(training),
            "final_evaluation_bank": sorted(evaluation),
        },
        "anti_leakage_check": True,
    }


def _shells(config: dict) -> ShellDefinition:
    obs = config["observables"]
    return ShellDefinition(
        tuple(map(float, obs["shell_centers_cycles_per_pixel"])),
        tuple(map(float, obs["shell_widths_cycles_per_pixel"])),
    )


def _regimes(config: dict) -> list[dict]:
    regimes = [dict(row) for row in config.get("candidate_regimes", [])]
    used = {(round(float(row["feed"]), 8), round(float(row["kill"]), 8)) for row in regimes}
    grids = [config.get("candidate_grid", {}), config.get("candidate_local_grid", {})]
    for grid in grids:
        for feed in grid.get("feed_values", []):
            for kill in grid.get("kill_values", []):
                key = (round(float(feed), 8), round(float(kill), 8))
                if key in used:
                    continue
                regimes.append({
                    "id": f"grid_F{int(round(10000 * feed)):04d}_k{int(round(10000 * kill)):04d}",
                    "feed": float(feed), "kill": float(kill),
                })
                used.add(key)
    if not regimes:
        raise ValueError("no candidate regimes configured")
    return regimes


def _make_montage(fields: np.ndarray, regimes: list[dict], output: Path) -> None:
    sample_indices = np.linspace(0, fields.shape[1] - 1, 6).round().astype(int)
    vmin, vmax = np.quantile(fields, [0.005, 0.995])
    page_size = 20
    for page, start in enumerate(range(0, len(regimes), page_size), start=1):
        page_regimes = regimes[start:start + page_size]
        page_fields = fields[start:start + page_size]
        fig, axes = plt.subplots(
            len(page_regimes), len(sample_indices), figsize=(10.5, 1.75 * len(page_regimes))
        )
        axes = np.asarray(axes).reshape(len(page_regimes), len(sample_indices))
        for row, regime in enumerate(page_regimes):
            for column, sample_index in enumerate(sample_indices):
                axes[row, column].imshow(
                    page_fields[row, sample_index, 0], cmap="magma", vmin=vmin, vmax=vmax,
                    interpolation="nearest",
                )
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
                if column == 0:
                    axes[row, column].set_ylabel(
                        f"{regime['id']}\nF={regime['feed']:.4f}\nk={regime['kill']:.4f}",
                        rotation=0, ha="right", va="center", fontsize=8,
                    )
        fig.suptitle(f"Experiment C design-only Gray–Scott scan, page {page} (fixed indices)")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.992))
        page_output = output if page == 1 else output.with_name(
            f"{output.stem}_page_{page:02d}{output.suffix}"
        )
        fig.savefig(page_output, dpi=180, bbox_inches="tight")
        plt.close(fig)


def _classification(summaries: list[dict], config: dict) -> None:
    keys = [
        "minority_component_count_mean", "absolute_euler_characteristic_mean",
        "interface_length_mean", "anisotropy_mean",
    ]
    eligible = [row for row in summaries if row["design_regime_gate_pass"]]
    if len(eligible) < 3:
        raise RuntimeError("fewer than three stable patterned regimes passed design gates")
    matrix = np.asarray([[row[key] for key in keys] for row in eligible], dtype=np.float64)
    center = matrix.mean(axis=0)
    scale = np.maximum(matrix.std(axis=0, ddof=1), 1e-12)
    z = (matrix - center) / scale
    # Spot/hole populations have many disconnected domains in their minority
    # phase regardless of V polarity. Labyrinths have fewer, longer interfaces.
    scores = z[:, 0] + z[:, 1] - 0.35 * z[:, 2] - 0.15 * z[:, 3]
    low = np.quantile(scores, float(config["classification"]["labyrinth_quantile"]))
    high = np.quantile(scores, float(config["classification"]["spot_quantile"]))
    for row in summaries:
        if not row["design_regime_gate_pass"]:
            row["morphology_score"] = float("nan")
            row["empirical_class"] = "rejected_unstable"
    for row, score in zip(eligible, scores):
        row["morphology_score"] = float(score)
        if score >= high:
            row["empirical_class"] = "spot_like"
        elif score <= low:
            row["empirical_class"] = "labyrinth_like"
        else:
            row["empirical_class"] = "ambiguous"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_scan(config_path: Path = DEFAULT_CONFIG) -> dict:
    config_path = config_path.resolve()
    config = _read_config(config_path)
    output = ROOT / config["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics").mkdir(exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    shutil.copyfile(config_path, output / "design_config.yaml")
    provenance = _provenance(config, config_path)
    _write_json(output / "run_metadata.json", provenance)

    seed_start = int(config["seeds"]["design_initial_condition_start"])
    count = int(config["seeds"]["design_initial_condition_count"])
    seeds = np.arange(seed_start, seed_start + count, dtype=np.int64)
    grid = config["grid"]
    ic = config["initial_conditions"]
    initial_u, initial_v, ic_metadata = generate_initial_conditions(
        seeds,
        height=int(grid["height"]), width=int(grid["width"]),
        blob_count=tuple(ic["blob_count"]), radius_range=tuple(ic["radius_range"]),
        u_depletion_range=tuple(ic["u_depletion_range"]),
        v_amplitude_range=tuple(ic["v_amplitude_range"]), noise_std=float(ic["noise_std"]),
    )
    regimes = _regimes(config)
    regime_count = len(regimes)
    tiled_u = np.tile(initial_u[None], (regime_count, 1, 1, 1, 1)).reshape((-1,) + initial_u.shape[1:])
    tiled_v = np.tile(initial_v[None], (regime_count, 1, 1, 1, 1)).reshape((-1,) + initial_v.shape[1:])
    feeds = np.repeat([float(row["feed"]) for row in regimes], count)
    kills = np.repeat([float(row["kill"]) for row in regimes], count)
    simulator = config["simulator"]
    final_u, final_v = simulate(
        tiled_u, tiled_v, feed=feeds, kill=kills,
        diffusion_u=float(simulator["diffusion_u"]),
        diffusion_v=float(simulator["diffusion_v"]), dt=float(simulator["dt"]),
        physical_time=float(simulator["physical_time"]), spacing=float(grid["spacing"]),
    )
    final_u = np.asarray(final_u).reshape((regime_count, count) + initial_u.shape[1:])
    final_v = np.asarray(final_v).reshape((regime_count, count) + initial_v.shape[1:])
    if not np.isfinite(final_v).all():
        raise FloatingPointError("Gray–Scott scan produced non-finite states")

    threshold = pooled_otsu_threshold(final_v)
    shells = _shells(config)
    features = np.asarray(field_observables(
        jnp.asarray(final_v), shells, tuple(config["observables"]["components"])
    ))
    sample_rows, regime_rows, all_metric_rows = [], [], []
    for regime_index, regime in enumerate(regimes):
        rows = metric_rows(final_v[regime_index], threshold)
        all_metric_rows.append(rows)
        summary = {**regime, **summarize_rows(rows)}
        summary.update({
            f"phi_{name}_mean": float(features[regime_index, :, feature_index].mean())
            for feature_index, name in enumerate(config["observables"]["components"])
        })
        summary["field_min"] = float(final_v[regime_index].min())
        summary["field_max"] = float(final_v[regime_index].max())
        summary["within_regime_diversity"] = float(np.mean(np.std(final_v[regime_index, :, 0], axis=0)))
        regime_rows.append(summary)
        for sample_index, (seed, metrics) in enumerate(zip(seeds, rows)):
            row = {"regime_id": regime["id"], "feed": regime["feed"], "kill": regime["kill"],
                   "sample_index": sample_index, "initial_condition_seed": int(seed), **metrics}
            row.update({f"phi_{name}": float(features[regime_index, sample_index, idx])
                        for idx, name in enumerate(config["observables"]["components"])})
            sample_rows.append(row)
    convergence_count = int(simulator["convergence_samples"])
    fine_dt = float(simulator["convergence_dt"])
    conv_u = np.tile(initial_u[:convergence_count][None], (regime_count, 1, 1, 1, 1)).reshape(
        (-1,) + initial_u.shape[1:]
    )
    conv_v = np.tile(initial_v[:convergence_count][None], (regime_count, 1, 1, 1, 1)).reshape(
        (-1,) + initial_v.shape[1:]
    )
    fine_u, fine_v = simulate(
        conv_u, conv_v, feed=np.repeat([r["feed"] for r in regimes], convergence_count),
        kill=np.repeat([r["kill"] for r in regimes], convergence_count),
        diffusion_u=float(simulator["diffusion_u"]), diffusion_v=float(simulator["diffusion_v"]),
        dt=fine_dt, physical_time=float(simulator["physical_time"]), spacing=float(grid["spacing"]),
    )
    fine_v = np.asarray(fine_v).reshape((regime_count, convergence_count) + initial_v.shape[1:])
    coarse_v = final_v[:, :convergence_count]
    convergence_rows = []
    for index, regime in enumerate(regimes):
        rmse = np.sqrt(np.mean((coarse_v[index] - fine_v[index]) ** 2, axis=(1, 2, 3)))
        scale = np.maximum(np.sqrt(np.mean(fine_v[index] ** 2, axis=(1, 2, 3))), 1e-12)
        convergence_rows.append({
            "regime_id": regime["id"], "coarse_dt": simulator["dt"], "fine_dt": fine_dt,
            "mean_relative_field_rmse": float(np.mean(rmse / scale)),
            "max_relative_field_rmse": float(np.max(rmse / scale)),
        })
    classification = config["classification"]
    for row, convergence in zip(regime_rows, convergence_rows):
        regime_index = next(i for i, regime in enumerate(regimes) if regime["id"] == row["id"])
        sample_std = np.std(final_v[regime_index, :, 0], axis=(-2, -1))
        row["pattern_presence_fraction"] = float(np.mean(sample_std >= 0.02))
        row["mean_timestep_relative_rmse"] = convergence["mean_relative_field_rmse"]
        row["worst_timestep_relative_rmse"] = convergence["max_relative_field_rmse"]
        failures = []
        if row["pattern_presence_fraction"] < float(classification["minimum_pattern_presence_fraction"]):
            failures.append("pattern_presence")
        if row["mean_timestep_relative_rmse"] > float(classification["maximum_mean_timestep_relative_rmse"]):
            failures.append("mean_timestep_convergence")
        if row["worst_timestep_relative_rmse"] > float(classification["maximum_worst_timestep_relative_rmse"]):
            failures.append("worst_timestep_convergence")
        row["design_regime_gate_pass"] = not failures
        row["design_regime_failure_reason"] = ";".join(failures)
    _classification(regime_rows, config)

    np.savez_compressed(
        output / "design_banks.npz", initial_u=initial_u, initial_v=initial_v,
        endpoint_u=final_u, endpoint_v=final_v, features=features, seeds=seeds,
        regime_ids=np.asarray([r["id"] for r in regimes]), threshold=np.asarray(threshold),
    )
    _write_json(output / "initial_condition_metadata.json", ic_metadata)
    _write_csv(output / "metrics" / "design_morphology_metrics.csv", sample_rows)
    _write_csv(output / "design_candidates.csv", regime_rows)
    _write_csv(output / "metrics" / "timestep_convergence.csv", convergence_rows)
    _write_json(output / "design_scan_summary.json", {
        "global_threshold": threshold,
        "shells": {"centers": shells.centers, "widths": shells.widths},
        "regimes": regime_rows,
        "timestep_convergence": convergence_rows,
    })
    _make_montage(final_v, regimes, output / "figures" / "regime_scan_montage.png")
    return {"output": str(output), "threshold": threshold, "regimes": regime_rows}


def _morphology_scale(all_rows: list[list[dict]], keys: list[str]) -> np.ndarray:
    values = np.asarray([[row[key] for key in keys] for rows in all_rows for row in rows])
    return np.maximum(values.std(axis=0, ddof=1), 1e-12)


def run_endpoint_selection(config_path: Path = DEFAULT_CONFIG) -> dict:
    config_path = config_path.resolve()
    config = _read_config(config_path)
    output = ROOT / config["output_directory"]
    bank_path = output / "design_banks.npz"
    summary_path = output / "design_scan_summary.json"
    if not bank_path.exists() or not summary_path.exists():
        raise FileNotFoundError("run the design scan before endpoint selection")
    bank = np.load(bank_path)
    scan = json.loads(summary_path.read_text())
    features = np.asarray(bank["features"], dtype=np.float64)
    fields = np.asarray(bank["endpoint_v"])
    threshold = float(bank["threshold"])
    regime_ids = list(map(str, bank["regime_ids"]))
    by_id = {row["id"]: row for row in scan["regimes"]}
    morphology = [metric_rows(fields[index], threshold) for index in range(len(regime_ids))]
    morph_keys = [
        "minority_component_count", "euler_characteristic", "interface_length",
        "anisotropy", "heldout_spectrum_1",
    ]
    morph_scale = _morphology_scale(morphology, morph_keys)
    feature_center, feature_scale = fit_standardization(features.reshape(-1, features.shape[-1]))
    standardized = (features - feature_center) / feature_scale
    spot_indices = [i for i, rid in enumerate(regime_ids) if by_id[rid]["empirical_class"] == "spot_like" and by_id[rid]["design_regime_gate_pass"]]
    labyrinth_indices = [i for i, rid in enumerate(regime_ids) if by_id[rid]["empirical_class"] == "labyrinth_like" and by_id[rid]["design_regime_gate_pass"]]
    gates = config["endpoint_gates"]
    rows, records = [], {}
    for minus_index in spot_indices:
        for plus_index in labyrinth_indices:
            calibrated = calibrate_shared_target(standardized[minus_index], standardized[plus_index])
            wm, wp = calibrated["minus"]["weights"], calibrated["plus"]["weights"]
            hidden_m = weighted_metric_mean(morphology[minus_index], wm)
            hidden_p = weighted_metric_mean(morphology[plus_index], wp)
            difference = np.asarray([hidden_p[key] - hidden_m[key] for key in morph_keys])
            effect = float(np.linalg.norm(difference / morph_scale) / np.sqrt(len(morph_keys)))
            residual = max(calibrated["minus"]["max_abs_residual"], calibrated["plus"]["max_abs_residual"])
            minimum_ess = min(calibrated["minus"]["ess_fraction"], calibrated["plus"]["ess_fraction"])
            reasons = []
            if residual > float(gates["max_standardized_calibration_residual"]):
                reasons.append("calibration_residual")
            if minimum_ess < float(gates["minimum_ess_fraction"]):
                reasons.append("endpoint_ess")
            if effect < float(gates["minimum_morphology_effect_size"]):
                reasons.append("morphology_effect")
            pair_id = f"{regime_ids[minus_index]}__{regime_ids[plus_index]}"
            target_physical = calibrated["target"] * feature_scale + feature_center
            score = effect + 0.5 * minimum_ess - 0.05 * max(
                calibrated["minus"]["lambda_norm"], calibrated["plus"]["lambda_norm"]
            )
            row = {
                "pair_id": pair_id, "spot_regime": regime_ids[minus_index],
                "labyrinth_regime": regime_ids[plus_index],
                "spot_feed": by_id[regime_ids[minus_index]]["feed"],
                "spot_kill": by_id[regime_ids[minus_index]]["kill"],
                "labyrinth_feed": by_id[regime_ids[plus_index]]["feed"],
                "labyrinth_kill": by_id[regime_ids[plus_index]]["kill"],
                "calibration_residual_standardized": residual,
                "spot_ess_fraction": calibrated["minus"]["ess_fraction"],
                "labyrinth_ess_fraction": calibrated["plus"]["ess_fraction"],
                "minimum_ess_fraction": minimum_ess,
                "spot_weight_entropy_fraction": calibrated["minus"]["weight_entropy_fraction"],
                "labyrinth_weight_entropy_fraction": calibrated["plus"]["weight_entropy_fraction"],
                "spot_max_weight": calibrated["minus"]["max_weight"],
                "labyrinth_max_weight": calibrated["plus"]["max_weight"],
                "spot_lambda_norm": calibrated["minus"]["lambda_norm"],
                "labyrinth_lambda_norm": calibrated["plus"]["lambda_norm"],
                "spot_covariance_condition": calibrated["minus"]["covariance_condition"],
                "labyrinth_covariance_condition": calibrated["plus"]["covariance_condition"],
                "calibrated_morphology_effect_size": effect,
                "endpoint_gate_pass": not reasons,
                "failure_reason": ";".join(reasons),
                "method_blind_endpoint_score": score,
            }
            rows.append(row)
            records[pair_id] = {
                "target_standardized": calibrated["target"], "target_physical": target_physical,
                "feature_center": feature_center, "feature_scale": feature_scale,
                "spot_weights": wm, "labyrinth_weights": wp,
                "spot_hidden_mean": hidden_m, "labyrinth_hidden_mean": hidden_p,
                "morphology_keys": morph_keys, "morphology_scale": morph_scale,
                "diagnostics": row,
            }
    rows.sort(key=lambda row: (not row["endpoint_gate_pass"], -row["method_blind_endpoint_score"]))
    _write_csv(output / "endpoint_calibration_candidates.csv", rows)
    passing = [row for row in rows if row["endpoint_gate_pass"]]
    provisional = passing[0] if passing else None
    failure_summary = None
    if not passing and rows:
        lowest_residual = min(rows, key=lambda row: row["calibration_residual_standardized"])
        morphology_passing = [
            row for row in rows
            if row["calibrated_morphology_effect_size"] >= float(gates["minimum_morphology_effect_size"])
        ]
        best_overlap_with_morphology = (
            min(morphology_passing, key=lambda row: row["calibration_residual_standardized"])
            if morphology_passing else None
        )
        failure_summary = {
            "lowest_residual_pair": lowest_residual,
            "best_overlap_among_morphology_passing_pairs": best_overlap_with_morphology,
        }
    selection = {
        "status": "phase_2_pass" if passing else "phase_2_failed",
        "is_frozen_benchmark": False,
        "reason": (
            "provisional endpoint shortlist only; interior projection and tangent blind-spot gates remain"
            if passing else "no candidate passed the predeclared endpoint gates"
        ),
        "provisional_pair": provisional,
        "pair_details": records.get(provisional["pair_id"]) if provisional else None,
        "failure_summary": failure_summary,
        "passing_pair_count": len(passing),
        "evaluated_pair_count": len(rows),
        "selection_uses_learned_method_results": False,
    }
    _write_json(output / "selected_endpoint_summary.json", selection)
    _write_design_report(output, config, scan, rows, selection)
    return selection


def _write_design_report(output: Path, config: dict, scan: dict, rows: list[dict], selection: dict) -> None:
    lines = [
        "# Gray–Scott Experiment C design report", "",
        "This report contains design-split morphology discovery and shared-target endpoint calibration only. No learned MFSI or tangent rollout result was computed or used.", "",
        f"Global pooled threshold: `{scan['global_threshold']:.8g}`.", "",
        "## Regime scan", "",
        "| id | F | k | class | regime gate | minority components | Euler | interface | anisotropy | diversity |", "|---|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in scan["regimes"]:
        lines.append(
            f"| {row['id']} | {row['feed']:.4f} | {row['kill']:.4f} | {row['empirical_class']} | {row['design_regime_gate_pass']} | "
            f"{row['minority_component_count_mean']:.3g} | {row['euler_characteristic_mean']:.3g} | "
            f"{row['interface_length_mean']:.3g} | {row['anisotropy_mean']:.3g} | {row['within_regime_diversity']:.3g} |"
        )
    lines += ["", "The class labels are empirical score strata, not assumed labels for `(F,k)` values. Visual confirmation is retained in the paginated `figures/regime_scan_montage*.png` files.", "", "## Endpoint calibration", "",
              "| pair | residual (std.) | min ESS | morphology effect | pass | reason |", "|---|---:|---:|---:|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['pair_id']} | {row['calibration_residual_standardized']:.3g} | "
            f"{row['minimum_ess_fraction']:.3g} | {row['calibrated_morphology_effect_size']:.3g} | "
            f"{row['endpoint_gate_pass']} | {row['failure_reason'] or '—'} |"
        )
    lines += ["", "## Decision", "", f"Status: `{selection['status']}`.", "", selection["reason"] + ".", ""]
    if selection["provisional_pair"]:
        lines.append(f"Provisional endpoint pair: `{selection['provisional_pair']['pair_id']}`.")
        lines.append("")
    lines += [
        "The pair is not a frozen Experiment C benchmark. Per protocol, reference-flow quality, interior I-projection ESS, projected morphology shift, and tangent blind-spot diagnostics must pass before `benchmark_selection.yaml` is created or final MFSI training begins.", "",
        "Timestep convergence diagnostics are recorded in `metrics/timestep_convergence.csv`; all raw sample metrics and failed pairs are retained.", "",
    ]
    (ROOT / "GRAYSCOTT_DESIGN_REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("scan", "select", "design"), nargs="?", default="design")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if args.command in ("scan", "design"):
        result = run_scan(args.config)
        print(json.dumps({"scan": result}, indent=2, default=_json_default))
    if args.command in ("select", "design"):
        result = run_endpoint_selection(args.config)
        print(json.dumps({"selection": result}, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
