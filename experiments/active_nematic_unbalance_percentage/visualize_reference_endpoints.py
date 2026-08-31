"""Visualize every frozen active-nematic reference law at both endpoints.

Each panel compares one learned endpoint-only reference distribution with the
empirical endpoint distribution from the exact 64-run training split used to
fit it.  The main axes compare the periodic spatial marginals: filled density
is training truth and contours are the learned reference.  The inset compares
the periodic beta-orientation marginals.  Endpoint finite masses are reported
separately because the learned flow represents normalized shape while its
Fisher--Rao schedule represents mass.

This is deterministic, read-only post-processing.  It does not simulate,
train, optimize, validate, or modify the running Pareto study.  Every run saves
both PNG and PDF outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter, gaussian_filter1d


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FROZEN = (
    SCRIPT_DIR / "outputs" / "more_training_v2_pareto" / "frozen_inputs"
)
DEFAULT_OUTPUT_STEM = (
    SCRIPT_DIR / "figures" / "active_nematic_reference_endpoint_fit_v2"
)

PAPER_BACKGROUND = "#FFFFFF"
PANEL_BACKGROUND = "#FFFFFF"
TRUTH_CMAP = LinearSegmentedColormap.from_list(
    "training_endpoint_density",
    ("#FBFAF6", "#DDD5E7", "#9A89AF", "#4B3E62"),
)
SPECIES_COLORS = {"plus": "#C65355", "minus": "#337C91"}
SPECIES_LABELS = {"plus": "+ DEFECT", "minus": "− DEFECT"}
ENDPOINT_LABELS = ("INITIAL", "TARGET")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-inputs", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--spatial-bins", type=int, default=96)
    parser.add_argument("--orientation-bins", type=int, default=96)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.facecolor": PANEL_BACKGROUND,
            "axes.edgecolor": "#C9C3B8",
            "axes.linewidth": 0.65,
            "figure.facecolor": PAPER_BACKGROUND,
            "savefig.facecolor": PAPER_BACKGROUND,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _training_endpoint_states(
    saved: np.lib.npyio.NpzFile,
    species: str,
    train_runs: np.ndarray,
    time_index: int,
) -> np.ndarray:
    states = np.asarray(saved[f"{species}_states"], dtype=np.float64)
    offsets = np.asarray(saved[f"{species}_offsets"], dtype=np.int64)
    chunks = [
        states[offsets[run, time_index] : offsets[run, time_index + 1]]
        for run in train_runs
    ]
    nonempty = [chunk for chunk in chunks if len(chunk)]
    if not nonempty:
        raise ValueError(f"empty {species} training endpoint at index {time_index}")
    return np.concatenate(nonempty, axis=0)


def _spatial_density(
    states: np.ndarray,
    weights: np.ndarray,
    *,
    box_size: float,
    bins: int,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    histogram, _, _ = np.histogram2d(
        np.mod(states[:, 0], box_size),
        np.mod(states[:, 1], box_size),
        bins=bins,
        range=((0.0, box_size), (0.0, box_size)),
        weights=weights,
    )
    density = gaussian_filter(histogram, sigma=1.35, mode="wrap")
    density /= np.sum(density)
    return density.T


def _orientation_density(
    states: np.ndarray,
    weights: np.ndarray,
    *,
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    histogram, edges = np.histogram(
        np.mod(states[:, 2], 2.0 * np.pi),
        bins=bins,
        range=(0.0, 2.0 * np.pi),
        weights=weights,
    )
    density = gaussian_filter1d(histogram, sigma=1.35, mode="wrap")
    density /= np.sum(density)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, density


def _highest_density_levels(density: np.ndarray) -> list[float]:
    """Return contour thresholds enclosing 90%, 75%, and 50% of mass."""
    values = np.asarray(density, dtype=np.float64).ravel()
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    cumulative /= cumulative[-1]
    levels = []
    for enclosed_mass in (0.90, 0.75, 0.50):
        index = min(int(np.searchsorted(cumulative, enclosed_mass)), len(ordered) - 1)
        levels.append(float(ordered[index]))
    levels = sorted(set(levels))
    if len(levels) < 3:
        maximum = float(np.max(density))
        levels = [0.15 * maximum, 0.35 * maximum, 0.60 * maximum]
    return levels


def _distribution_overlap(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first / np.sum(first)
    second = second / np.sum(second)
    return float(np.minimum(first, second).sum())


def _prepare_data(args: argparse.Namespace) -> dict[str, Any]:
    frozen = args.frozen_inputs.expanduser().resolve()
    manifest = _load_json(frozen / "view_manifest.json")
    train_runs = np.asarray(manifest["train_runs"], dtype=np.int64)
    bank_path = frozen / "two_species_defect_bank.npz"
    reference_dirs = sorted(frozen.glob("reference_seed_*"))
    if not reference_dirs:
        raise FileNotFoundError(f"no frozen reference seeds under {frozen}")

    with np.load(bank_path, allow_pickle=False) as bank:
        times = np.asarray(bank["times"], dtype=np.float64)
        box_size = float(bank["box_size"])
        truth: dict[tuple[str, int], dict[str, Any]] = {}
        for species in ("plus", "minus"):
            counts = np.asarray(bank[f"{species}_counts"], dtype=np.float64)
            for endpoint, time_index in enumerate((0, len(times) - 1)):
                states = _training_endpoint_states(
                    bank, species, train_runs, time_index
                )
                uniform = np.full(len(states), 1.0 / len(states), dtype=np.float64)
                truth[(species, endpoint)] = {
                    "states": states,
                    "mass": float(np.mean(counts[train_runs, time_index])),
                    "spatial": _spatial_density(
                        states,
                        uniform,
                        box_size=box_size,
                        bins=args.spatial_bins,
                    ),
                    "orientation": _orientation_density(
                        states, uniform, bins=args.orientation_bins
                    ),
                }

    references = []
    for directory in reference_dirs:
        seed = int(directory.name.removeprefix("reference_seed_"))
        mass_schedule = _load_json(directory / "reference_mass_schedule.json")
        species_rows = {}
        for species in ("plus", "minus"):
            with np.load(
                directory / f"{species}_reference_bank.npz", allow_pickle=False
            ) as saved:
                nodes = np.asarray(saved["nodes"], dtype=np.float64)
                weights = np.asarray(saved["weights"], dtype=np.float64)
            endpoints = []
            for endpoint, time_index in enumerate((0, len(nodes) - 1)):
                spatial = _spatial_density(
                    nodes[time_index],
                    weights[time_index],
                    box_size=box_size,
                    bins=args.spatial_bins,
                )
                orientation = _orientation_density(
                    nodes[time_index],
                    weights[time_index],
                    bins=args.orientation_bins,
                )
                truth_row = truth[(species, endpoint)]
                endpoints.append(
                    {
                        "spatial": spatial,
                        "orientation": orientation,
                        "mass": float(mass_schedule[f"mass_{species}"][time_index]),
                        "spatial_overlap": _distribution_overlap(
                            truth_row["spatial"], spatial
                        ),
                        "orientation_overlap": _distribution_overlap(
                            truth_row["orientation"][1], orientation[1]
                        ),
                    }
                )
            species_rows[species] = endpoints
        references.append({"seed": seed, "species": species_rows})

    return {
        "frozen": frozen,
        "train_runs": train_runs,
        "times": times,
        "box_size": box_size,
        "truth": truth,
        "references": references,
    }


def _clean_spatial_axis(axis: plt.Axes, box_size: float) -> None:
    axis.set_xlim(0.0, box_size)
    axis.set_ylim(0.0, box_size)
    axis.set_aspect("equal")
    axis.set_xticks((0.0, 0.5 * box_size, box_size))
    axis.set_yticks((0.0, 0.5 * box_size, box_size))
    axis.tick_params(length=2.2, width=0.55, labelsize=6.8, colors="#68656A")
    for spine in axis.spines.values():
        spine.set_color("#C9C3B8")


def make_figure(data: dict[str, Any]) -> plt.Figure:
    _style()
    references = data["references"]
    box_size = float(data["box_size"])
    all_truth = [row["spatial"] for row in data["truth"].values()]
    color_max = float(np.quantile(np.concatenate([row.ravel() for row in all_truth]), 0.995))
    norm = PowerNorm(gamma=0.62, vmin=0.0, vmax=color_max)

    figure, axes = plt.subplots(
        len(references),
        4,
        figsize=(14.7, 10.8),
        constrained_layout=False,
    )
    axes = np.asarray(axes).reshape(len(references), 4)
    figure.subplots_adjust(
        left=0.105,
        right=0.925,
        bottom=0.08,
        top=0.865,
        wspace=0.10,
        hspace=0.27,
    )
    image = None
    columns = (
        ("plus", 0),
        ("plus", 1),
        ("minus", 0),
        ("minus", 1),
    )
    for row_index, reference in enumerate(references):
        for column_index, (species, endpoint) in enumerate(columns):
            axis = axes[row_index, column_index]
            truth = data["truth"][(species, endpoint)]
            learned = reference["species"][species][endpoint]
            image = axis.imshow(
                truth["spatial"],
                origin="lower",
                extent=(0.0, box_size, 0.0, box_size),
                cmap=TRUTH_CMAP,
                norm=norm,
                interpolation="bilinear",
            )
            coordinates = (
                np.arange(learned["spatial"].shape[0]) + 0.5
            ) * box_size / learned["spatial"].shape[0]
            axis.contour(
                coordinates,
                coordinates,
                learned["spatial"],
                levels=_highest_density_levels(learned["spatial"]),
                colors=SPECIES_COLORS[species],
                linewidths=(0.85, 1.15, 1.7),
                alpha=0.95,
            )
            _clean_spatial_axis(axis, box_size)

            inset = axis.inset_axes((0.525, 0.67, 0.43, 0.255))
            theta_truth, density_truth = truth["orientation"]
            theta_reference, density_reference = learned["orientation"]
            inset.fill_between(
                theta_truth,
                density_truth,
                color="#81758E",
                alpha=0.28,
                linewidth=0.0,
            )
            inset.plot(
                theta_reference,
                density_reference,
                color=SPECIES_COLORS[species],
                linewidth=1.35,
            )
            inset.set_xlim(0.0, 2.0 * np.pi)
            inset.set_ylim(bottom=0.0)
            inset.set_xticks((0.0, np.pi, 2.0 * np.pi), ("0", "π", "2π"))
            inset.set_yticks(())
            inset.tick_params(length=1.8, width=0.45, labelsize=5.5, pad=1.2)
            inset.set_facecolor((0.98, 0.97, 0.94, 0.88))
            for spine in inset.spines.values():
                spine.set_linewidth(0.45)
                spine.set_color("#BDB6AE")

            axis.set_xlabel(
                (
                    f"mass truth/reference  {truth['mass']:.2f} / {learned['mass']:.2f}"
                    f"   ·   overlap xy/β  {learned['spatial_overlap']:.2f} / "
                    f"{learned['orientation_overlap']:.2f}"
                ),
                fontsize=6.6,
                color="#5F5B62",
                labelpad=2.0,
            )
            if row_index == 0:
                physical_time = data["times"][0 if endpoint == 0 else -1]
                axis.set_title(
                    f"{SPECIES_LABELS[species]}  ·  {ENDPOINT_LABELS[endpoint]}  ·  t={physical_time:g}",
                    fontsize=10.2,
                    fontweight="bold",
                    color=SPECIES_COLORS[species],
                    pad=7.0,
                )

        figure.text(
            0.067,
            axes[row_index, 0].get_position().y0
            + 0.5 * axes[row_index, 0].get_position().height,
            f"REFERENCE\n{reference['seed']}",
            rotation=90,
            ha="center",
            va="center",
            fontsize=9.4,
            fontweight="bold",
            color="#4A3A62",
        )

    if image is None:
        raise RuntimeError("no endpoint panels were generated")
    color_axis = figure.add_axes((0.945, 0.28, 0.012, 0.37))
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label(
        "normalized training-endpoint spatial density",
        fontsize=8.2,
        color="#555A62",
    )
    colorbar.set_ticks(())
    colorbar.outline.set_visible(False)

    figure.suptitle(
        "Endpoint fit of the active-nematic reference laws",
        x=0.105,
        y=0.95,
        ha="left",
        fontsize=21,
        fontweight="bold",
        color="#20242B",
    )
    legend = (
        Patch(facecolor="#81758E", alpha=0.48, edgecolor="none", label="64-run training truth"),
        Line2D([0], [0], color="#555A62", linewidth=1.7, label="learned reference contours / β line"),
    )
    figure.legend(
        handles=legend,
        loc="upper right",
        bbox_to_anchor=(0.925, 0.94),
        frameon=False,
        ncol=2,
        fontsize=8.3,
        handlelength=2.0,
        columnspacing=1.4,
    )
    return figure


def _output_paths(stem: Path) -> tuple[Path, Path]:
    stem = stem.expanduser().resolve()
    if stem.suffix.lower() in {".png", ".pdf"}:
        stem = stem.with_suffix("")
    return stem.with_suffix(".png"), stem.with_suffix(".pdf")


def main() -> int:
    args = _parse_args()
    data = _prepare_data(args)
    figure = make_figure(data)
    png, pdf = _output_paths(args.output_stem)
    png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, dpi=args.dpi, bbox_inches="tight", pad_inches=0.10)
    figure.savefig(pdf, bbox_inches="tight", pad_inches=0.10)
    print(f"saved {png}")
    print(f"saved {pdf}")
    for reference in data["references"]:
        for species in ("plus", "minus"):
            for endpoint, label in enumerate(("initial", "target")):
                row = reference["species"][species][endpoint]
                print(
                    f"seed={reference['seed']} species={species} endpoint={label} "
                    f"spatial_overlap={row['spatial_overlap']:.6f} "
                    f"orientation_overlap={row['orientation_overlap']:.6f}"
                )
    if args.show:
        plt.show()
    else:
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
