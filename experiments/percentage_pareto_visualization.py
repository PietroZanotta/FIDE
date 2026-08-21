"""Shared publication-style visualization for percentage-risk Pareto sweeps."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PAPER = "#F5F2EB"
PANEL = "#FCFBF8"
INK = "#252A33"
MUTED = "#66707C"
LAW = "#2878B5"
TANGENT = "#E39D24"
FULL = "#D1495B"
DOMINATED = "#D19A2A"

METHODS = ("law", "tangent", "full")
METHOD_LABELS = {"law": "Law", "tangent": "Tangent", "full": "Full"}
METHOD_COLORS = {"law": LAW, "tangent": TANGENT, "full": FULL}
METHOD_MARKERS = {"law": "s", "tangent": "^", "full": "o"}


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        if not _finite(row.get("risk_allowance_percent")):
            if _finite(row.get("risk_allowance_fraction")):
                row["risk_allowance_percent"] = 100.0 * float(row["risk_allowance_fraction"])
            elif _finite(row.get("epsilon_r")) and _finite(row.get("R_star")):
                scale = abs(float(row["R_star"]))
                row["risk_allowance_percent"] = 100.0 * float(row["epsilon_r"]) / scale if scale else 0.0
        row["full_certified"] = _as_bool(row.get("full_certified", False))
        if all(
            _finite(row.get(key))
            for key in ("risk_allowance_percent", "R_star", "full_R_excess_selection", "full_A_selection")
        ):
            normalized.append(row)
    normalized.sort(key=lambda row: float(row["risk_allowance_percent"]))
    if not normalized:
        raise ValueError("no complete percentage-risk Pareto rows were found")
    return normalized


def load_rows(path: Path) -> tuple[list[dict[str, Any]], Path]:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / ("pareto.csv" if (path / "pareto.csv").exists() else "pareto.json")
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    elif path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"expected a JSON list in {path}")
    else:
        raise ValueError("input must be a Pareto directory, .csv, or .json")
    return normalize_rows(rows), path


def _resolve_result_path(row: dict[str, Any], pareto_source: Path) -> Path:
    """Resolve a point result even when a Pareto directory was moved."""
    source = pareto_source if pareto_source.is_dir() else pareto_source.parent
    candidates: list[Path] = []
    raw = row.get("result")
    if raw:
        path = Path(str(raw)).expanduser()
        candidates.extend((path, source / path, source / path.name))
    percent = float(row["risk_allowance_percent"])
    tag = f"risk_{f'{percent:g}'.replace('.', 'p').replace('-', 'm')}pct"
    candidates.append(source / tag / "result.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"could not locate result.json for the {percent:g}% Pareto point under {source}"
    )


def load_point_results(
    rows: list[dict[str, Any]], pareto_source: Path
) -> list[tuple[dict[str, Any], Path, dict[str, Any]]]:
    """Load detailed point results used by method tables and sensor plots."""
    loaded = []
    for row in normalize_rows(rows):
        path = _resolve_result_path(row, Path(pareto_source).expanduser().resolve())
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read Pareto point {path}: {exc}") from exc
        loaded.append((row, path, data))
    return loaded


def _candidate_actions(result_path: Path) -> dict[str, dict[str, Any]]:
    path = result_path.with_name("result.candidate_summary.csv")
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {str(row.get("design")): row for row in csv.DictReader(handle)}


def _selection_coordinates(data: dict[str, Any], method: str) -> Any:
    centers = data.get("selection_centers", {}).get(method)
    if centers is not None:
        return centers
    selection = data.get("selection", {})
    for suffix in ("_optimum_deg", "_optimum"):
        value = selection.get(f"{method}{suffix}")
        if value is not None:
            return value
    return None


def method_records(
    rows: list[dict[str, Any]], pareto_source: Path
) -> list[dict[str, Any]]:
    """Return comparable Law/Tangent/Full metrics for every allowance."""
    records: list[dict[str, Any]] = []
    for pareto_row, result_path, data in load_point_results(rows, pareto_source):
        allowance = float(pareto_row["risk_allowance_percent"])
        screens = data.get("law_screens", {})
        r_star = float(screens["R_star"])
        r_scale = max(abs(r_star), np.finfo(float).tiny)
        validation = data.get("validation", {})
        law_validation = validation.get("law", {})
        law_risk = law_validation.get("law_risk", {}).get("mean")
        law_action = law_validation.get("full_action", {}).get("mean")
        candidates = _candidate_actions(result_path)
        for method in METHODS:
            cert = data.get("selection_certificates", {}).get(method, {})
            block = validation.get(method, {})
            candidate = candidates.get(method, {})
            selection_risk = cert.get("R_selection", candidate.get("finite_risk_selection"))
            selection_action = cert.get(
                "full_action_selection", candidate.get("full_action_selection")
            )
            tangent_action = cert.get(
                "tangent_action_selection", candidate.get("tangent_action_selection")
            )
            validation_risk = block.get("law_risk", {}).get("mean")
            validation_action = block.get("full_action", {}).get("mean")
            risk_change = (
                100.0 * (float(validation_risk) - float(law_risk)) / abs(float(law_risk))
                if _finite(validation_risk) and _finite(law_risk) and float(law_risk) != 0.0
                else None
            )
            action_reduction = (
                100.0 * (float(law_action) - float(validation_action)) / abs(float(law_action))
                if _finite(validation_action) and _finite(law_action) and float(law_action) != 0.0
                else None
            )
            records.append(
                {
                    "risk_allowance_percent": allowance,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "R_star": r_star,
                    "R_max": screens.get("R_max"),
                    "selection_R": selection_risk,
                    "selection_R_increase_percent": (
                        100.0 * (float(selection_risk) - r_star) / r_scale
                        if _finite(selection_risk)
                        else None
                    ),
                    "selection_budget_used_percent": (
                        100.0 * (float(selection_risk) - r_star) / (allowance * r_scale / 100.0)
                        if _finite(selection_risk) and allowance > 0.0
                        else None
                    ),
                    "selection_L": cert.get("L_selection"),
                    "selection_tangent_action": tangent_action,
                    "selection_full_action": selection_action,
                    "certified": _as_bool(cert.get("certified", False)),
                    "validation_R_mean": validation_risk,
                    "validation_R_se": block.get("law_risk", {}).get("se"),
                    "validation_R_change_vs_law_percent": risk_change,
                    "validation_full_action_mean": validation_action,
                    "validation_full_action_se": block.get("full_action", {}).get("se"),
                    "validation_full_action_reduction_vs_law_percent": action_reduction,
                    "validation_valid_fraction": block.get("valid_fraction"),
                    "sensor_coordinates": json.dumps(_selection_coordinates(data, method)),
                    "result": str(result_path),
                }
            )
    return records


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def save_method_tables(records: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    """Write machine-readable and compact paper-ready comparison tables."""
    if not records:
        raise ValueError("no method records to tabulate")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_fields = (
        "risk_allowance_percent", "method", "R_star", "R_max", "selection_R",
        "selection_R_increase_percent", "selection_budget_used_percent", "selection_L",
        "selection_tangent_action", "selection_full_action", "certified",
        "sensor_coordinates", "result",
    )
    validation_fields = (
        "risk_allowance_percent", "method", "validation_R_mean", "validation_R_se",
        "validation_R_change_vs_law_percent", "validation_full_action_mean",
        "validation_full_action_se", "validation_full_action_reduction_vs_law_percent",
        "validation_valid_fraction", "result",
    )
    paths = [
        output_dir / "pareto_methods_selection.csv",
        output_dir / "pareto_methods_validation.csv",
    ]
    for path, fields in zip(paths, (selection_fields, validation_fields), strict=True):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                {field: _csv_value(record.get(field)) for field in fields} for record in records
            )

    markdown = output_dir / "pareto_methods_tables.md"
    lines = [
        "# Law, Tangent, and Full Pareto comparison",
        "",
        "## Selection-bank certificate",
        "",
        "| Allowance | Method | Risk increase | Budget used | Full action | Certified |",
        "|---:|:---|---:|---:|---:|:---:|",
    ]
    for record in records:
        lines.append(
            f"| {float(record['risk_allowance_percent']):g}% | {record['method_label']} | "
            f"{_format_table(record['selection_R_increase_percent'], '.3f', '%')} | "
            f"{_format_table(record['selection_budget_used_percent'], '.1f', '%')} | "
            f"{_format_table(record['selection_full_action'], '.5g')} | "
            f"{'yes' if record['certified'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Independent validation",
            "",
            "| Allowance | Method | Law risk (mean ± SE) | Full action (mean ± SE) | Action reduction vs Law | Valid |",
            "|---:|:---|---:|---:|---:|---:|",
        ]
    )
    for record in records:
        risk = _format_mean_se(record["validation_R_mean"], record["validation_R_se"])
        action = _format_mean_se(
            record["validation_full_action_mean"], record["validation_full_action_se"]
        )
        lines.append(
            f"| {float(record['risk_allowance_percent']):g}% | {record['method_label']} | "
            f"{risk} | {action} | "
            f"{_format_table(record['validation_full_action_reduction_vs_law_percent'], '.2f', '%')} | "
            f"{_format_table(_percent(record['validation_valid_fraction']), '.1f', '%')} |"
        )
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths.append(markdown)
    return paths


def _percent(value: Any) -> float | None:
    return 100.0 * float(value) if _finite(value) else None


def _format_table(value: Any, spec: str, suffix: str = "") -> str:
    return f"{format(float(value), spec)}{suffix}" if _finite(value) else "—"


def _format_mean_se(mean: Any, se: Any) -> str:
    if not _finite(mean):
        return "—"
    if not _finite(se):
        return f"{float(mean):.5g}"
    return f"{float(mean):.5g} ± {float(se):.2g}"


def _nondominated(x: np.ndarray, y: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    keep = np.zeros(len(x), dtype=bool)
    for i in range(len(x)):
        if not eligible[i]:
            continue
        x_tol = 1.0e-9 * max(1.0, abs(float(x[i])))
        y_tol = 1.0e-9 * max(1.0, abs(float(y[i])))
        dominates = eligible & (x <= x[i] + x_tol) & (y <= y[i] + y_tol) & (
            (x < x[i] - x_tol) | (y < y[i] - y_tol)
        )
        # If multiple allowances select the identical design, retain the first
        # (smallest allowance; rows are sorted) as the frontier representative.
        duplicate_before = eligible[:i] & np.isclose(x[:i], x[i]) & np.isclose(y[:i], y[i])
        keep[i] = not np.any(dominates) and not np.any(duplicate_before)
    return keep


def _allowance_group_label(values: list[float]) -> str:
    """Compact repeated-design labels, e.g. 0.5%, 2–5%."""
    values = sorted(values)
    parts: list[str] = []
    index = 0
    while index < len(values):
        end = index
        while (
            end + 1 < len(values)
            and values[end].is_integer()
            and values[end + 1].is_integer()
            and values[end + 1] == values[end] + 1.0
        ):
            end += 1
        if end > index:
            parts.append(f"{values[index]:g}–{values[end]:g}%")
        else:
            parts.append(f"{values[index]:g}%")
        index = end + 1
    return ", ".join(parts)


def _style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.2,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.edgecolor": "#B7B5AF",
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#555D67",
        "ytick.color": "#555D67",
        "figure.facecolor": PAPER,
        "axes.facecolor": PANEL,
        "savefig.facecolor": PAPER,
    })


def _scatter_status(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    colors: list[Any],
    certified: np.ndarray,
    efficient: np.ndarray,
) -> None:
    dominated = certified & ~efficient
    failed = ~certified
    ax.scatter(x[dominated], y[dominated], s=68, marker="o", facecolor=PANEL,
               edgecolor=DOMINATED, linewidth=1.8, zorder=4)
    ax.scatter(x[efficient], y[efficient], s=78, marker="o",
               c=np.asarray(colors, dtype=object)[efficient].tolist(),
               edgecolor="white", linewidth=1.1, zorder=5)
    ax.scatter(x[failed], y[failed], s=74, marker="X", facecolor="#858B94",
               edgecolor="white", linewidth=0.8, zorder=5)


def make_figure(rows: list[dict[str, Any]], *, experiment_label: str) -> plt.Figure:
    rows = normalize_rows(rows)
    allowance = np.asarray([float(row["risk_allowance_percent"]) for row in rows])
    anchor = np.asarray([abs(float(row["R_star"])) for row in rows])
    achieved = 100.0 * np.asarray([float(row["full_R_excess_selection"]) for row in rows]) / anchor
    selection_action = np.asarray([float(row["full_A_selection"]) for row in rows])
    certified = np.asarray([bool(row["full_certified"]) for row in rows])
    efficient = _nondominated(achieved, selection_action, certified)

    norm = mpl.colors.Normalize(vmin=float(np.min(allowance)), vmax=float(np.max(allowance)) if np.ptp(allowance) else float(np.min(allowance)) + 1.0)
    cmap = mpl.colormaps["viridis"]
    colors = [cmap(norm(value)) for value in allowance]
    law_selection_action = np.median([
        float(row["law_A_selection"]) for row in rows if _finite(row.get("law_A_selection"))
    ])

    _style()
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.5))
    fig.subplots_adjust(left=0.065, right=0.97, bottom=0.19, top=0.72, wspace=0.28)
    selection, validation, response = axes

    order = np.argsort(achieved)
    selection.plot(achieved[order], selection_action[order], color="#9DA2A8", lw=1.0, zorder=1)
    frontier_order = np.argsort(achieved[efficient])
    if np.count_nonzero(efficient) > 1:
        selection.plot(achieved[efficient][frontier_order], selection_action[efficient][frontier_order], color=FULL, lw=2.2, zorder=2)
    _scatter_status(selection, achieved, selection_action, colors, certified, efficient)
    selection.scatter([0.0], [law_selection_action], marker="*", s=160, facecolor=LAW,
                      edgecolor="white", linewidth=0.9, zorder=6)
    selection.axhline(law_selection_action, color=LAW, ls=":", lw=1.1, alpha=0.75)
    groups: list[tuple[float, float, list[float]]] = []
    for x, y, pct in zip(achieved, selection_action, allowance):
        matched = next(
            (group for group in groups if np.isclose(group[0], x) and np.isclose(group[1], y)),
            None,
        )
        if matched is None:
            groups.append((float(x), float(y), [float(pct)]))
        else:
            matched[2].append(float(pct))
    label_offsets = ((7, 8), (7, -18), (-38, 8), (-38, -18))
    for i, (x, y, percentages) in enumerate(groups):
        selection.annotate(
            _allowance_group_label(percentages), (x, y),
            xytext=label_offsets[i % len(label_offsets)],
            textcoords="offset points", fontsize=8, color=INK,
            bbox={"boxstyle": "round,pad=0.16", "facecolor": PANEL,
                  "edgecolor": "none", "alpha": 0.82},
        )
    selection.set_xlabel(r"Achieved selection-risk increase  $100(R-R^*)/|R^*|$  (%)")
    selection.set_ylabel(r"Exact selection full action  $A_{\rm full}$")
    selection.set_title("A   Certified selection frontier", loc="left")

    usable_validation = np.asarray([
        all(_finite(row.get(key)) for key in ("law_R_validation", "full_R_validation", "law_A_validation", "full_A_validation"))
        for row in rows
    ])
    validation_x = np.full(len(rows), np.nan)
    validation_y = np.full(len(rows), np.nan)
    for i, row in enumerate(rows):
        if usable_validation[i]:
            law_risk = float(row["law_R_validation"])
            law_action = float(row["law_A_validation"])
            validation_x[i] = 100.0 * (float(row["full_R_validation"]) - law_risk) / abs(law_risk)
            validation_y[i] = 100.0 * (law_action - float(row["full_A_validation"])) / abs(law_action)
    if np.any(usable_validation):
        val_order = np.argsort(validation_x[usable_validation])
        validation.plot(validation_x[usable_validation][val_order], validation_y[usable_validation][val_order], color="#9DA2A8", lw=1.0, zorder=1)
        validation.scatter(validation_x[usable_validation], validation_y[usable_validation], s=76,
                           c=np.asarray(colors, dtype=object)[usable_validation].tolist(),
                           edgecolor="white", linewidth=1.1, zorder=4)
    validation.scatter([0.0], [0.0], marker="*", s=160, facecolor=LAW, edgecolor="white", linewidth=0.9, zorder=6)
    validation.axhline(0.0, color=LAW, ls=":", lw=1.0, alpha=0.7)
    validation.axvline(0.0, color=LAW, ls=":", lw=1.0, alpha=0.7)
    validation.set_xlabel("Validation-risk change vs Law  (%)")
    validation.set_ylabel("Validation full-action reduction vs Law  (%)")
    validation.set_title("B   Independent validation", loc="left")

    reduction = np.asarray([
        100.0 * float(row["validation_action_reduction"])
        if _finite(row.get("validation_action_reduction")) else np.nan
        for row in rows
    ])
    usable_reduction = np.isfinite(reduction)
    if np.any(usable_reduction):
        response.plot(allowance[usable_reduction], reduction[usable_reduction], color=FULL, lw=2.0, zorder=2)
        for i in np.flatnonzero(usable_reduction):
            lower = rows[i].get("validation_ci_lower")
            upper = rows[i].get("validation_ci_upper")
            yerr = None
            if _finite(lower) and _finite(upper):
                yerr = np.asarray([[reduction[i] - 100.0 * float(lower)], [100.0 * float(upper) - reduction[i]]])
            response.errorbar(allowance[i], reduction[i], yerr=yerr, fmt="o", ms=7.5,
                              color=colors[i], ecolor=colors[i], elinewidth=1.4,
                              capsize=3.5, markeredgecolor="white", markeredgewidth=1.0, zorder=4)
    response.axhline(0.0, color=LAW, ls=":", lw=1.0, alpha=0.7)
    response.set_xlabel("Allowed extra risk  (%)")
    response.set_ylabel("Validation full-action reduction vs Law  (%)")
    response.set_title("C   Benefit across the risk allowance", loc="left")

    for ax in axes:
        ax.grid(color="#AEB2B8", lw=0.6, alpha=0.28)
        ax.margins(x=0.12, y=0.18)

    legend = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor=LAW,
               markeredgecolor="white", markersize=11, label="Law baseline"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=FULL,
               markeredgecolor="white", markersize=7.5, label="nondominated + certified"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PANEL,
               markeredgecolor=DOMINATED, markeredgewidth=1.6, markersize=7.5,
               label="certified but dominated"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#858B94",
               markeredgecolor="white", markersize=7.5, label="failed certificate"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.815), ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle(f"{experiment_label} · percentage-risk information–transport frontier",
                 x=0.065, y=0.965, ha="left", fontsize=17, fontweight="bold", color=INK)
    fig.text(0.065, 0.91,
             "Full action is minimized under an allowed percentage increase from the common exact Law-risk anchor.",
             ha="left", fontsize=9.5, color=MUTED)
    fig.text(0.5, 0.055,
             "Selection-bank certification is authoritative; validation metrics are independent out-of-sample checks.",
             ha="center", fontsize=8.3, color=MUTED)
    colorbar_ax = fig.add_axes((0.78, 0.835, 0.18, 0.022))
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_ax, orientation="horizontal")
    colorbar.set_label("allowed extra risk (%)", labelpad=2, fontsize=8)
    colorbar.ax.tick_params(labelsize=7.5)
    colorbar.outline.set_visible(False)
    return fig


def _method_arrays(
    records: list[dict[str, Any]], method: str, field: str
) -> tuple[np.ndarray, np.ndarray]:
    selected = sorted(
        (record for record in records if record.get("method") == method),
        key=lambda record: float(record["risk_allowance_percent"]),
    )
    x = np.asarray([float(record["risk_allowance_percent"]) for record in selected])
    y = np.asarray(
        [float(record[field]) if _finite(record.get(field)) else np.nan for record in selected]
    )
    return x, y


def _plot_methods(
    ax: plt.Axes,
    records: list[dict[str, Any]],
    field: str,
    *,
    se_field: str | None = None,
) -> None:
    for method in METHODS:
        x, y = _method_arrays(records, method, field)
        usable = np.isfinite(y)
        if not np.any(usable):
            continue
        color = METHOD_COLORS[method]
        ax.plot(
            x[usable], y[usable], marker=METHOD_MARKERS[method], ms=6.2,
            color=color, lw=1.9, markeredgecolor="white", markeredgewidth=0.8,
            label=METHOD_LABELS[method], zorder=3,
        )
        if se_field is not None:
            _, se = _method_arrays(records, method, se_field)
            band = usable & np.isfinite(se)
            if np.any(band):
                ax.fill_between(
                    x[band], y[band] - 1.96 * se[band], y[band] + 1.96 * se[band],
                    color=color, alpha=0.12, linewidth=0, zorder=1,
                )


def make_method_figure(
    records: list[dict[str, Any]], *, experiment_label: str
) -> plt.Figure:
    """Four directly comparable Law/Tangent/Full Pareto panels."""
    if not records:
        raise ValueError("no method records to plot")
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.5))
    fig.subplots_adjust(left=0.085, right=0.975, bottom=0.105, top=0.83, hspace=0.36, wspace=0.27)
    selection_action, selection_risk, validation_action, validation_risk = axes.ravel()

    _plot_methods(selection_action, records, "selection_full_action")
    selection_action.set_ylabel(r"Exact selection full action  $A_{\rm full}$")
    selection_action.set_title("A   Common-metric selection action", loc="left")

    _plot_methods(selection_risk, records, "selection_R_increase_percent")
    allowances = sorted({float(record["risk_allowance_percent"]) for record in records})
    selection_risk.plot(allowances, allowances, color="#4B9A73", ls="--", lw=1.2, label="risk limit")
    selection_risk.fill_between(allowances, 0.0, allowances, color="#4B9A73", alpha=0.07)
    selection_risk.set_ylabel(r"Selection risk increase  $100(R-R^*)/|R^*|$  (%)")
    selection_risk.set_title("B   Use of the finite-law allowance", loc="left")

    _plot_methods(
        validation_action, records, "validation_full_action_mean",
        se_field="validation_full_action_se",
    )
    validation_action.set_ylabel(r"Validation full action  $A_{\rm full}$")
    validation_action.set_title("C   Independent validation action", loc="left")
    validation_action.text(
        0.02, 0.04, "shading: 95% normal mean interval", transform=validation_action.transAxes,
        fontsize=7.7, color=MUTED,
    )

    _plot_methods(validation_risk, records, "validation_R_change_vs_law_percent")
    validation_risk.axhline(0.0, color=LAW, ls=":", lw=1.1, alpha=0.75)
    validation_risk.set_ylabel("Validation-risk change vs Law  (%)")
    validation_risk.set_title("D   Independent validation risk", loc="left")

    for ax in axes.ravel():
        ax.set_xlabel("Allowed extra risk  (%)")
        ax.grid(color="#AEB2B8", lw=0.6, alpha=0.28)
        ax.margins(x=0.08, y=0.15)
    handles = [
        Line2D([0], [0], marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
               markeredgecolor="white", lw=1.9, markersize=7, label=METHOD_LABELS[method])
        for method in METHODS
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.89), ncol=3, frameon=False)
    fig.suptitle(
        f"{experiment_label} · Law, Tangent, and Full across the Pareto sweep",
        x=0.085, y=0.965, ha="left", fontsize=17, fontweight="bold", color=INK,
    )
    fig.text(
        0.085, 0.92,
        "Every method is evaluated with the same finite-law risk and full-action metrics; validation uses the frozen out-of-sample bank.",
        ha="left", fontsize=9.3, color=MUTED,
    )
    return fig


def save_method_figure(
    records: list[dict[str, Any]], output: Path, *, experiment_label: str, dpi: int = 220
) -> Path:
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = make_method_figure(records, experiment_label=experiment_label)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def save_figure(rows: list[dict[str, Any]], output: Path, *, experiment_label: str, dpi: int = 220) -> Path:
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = make_figure(rows, experiment_label=experiment_label)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output
