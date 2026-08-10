"""Periodic morphology metrics used only for design and held-out evaluation."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def pooled_otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """A single Otsu threshold for pooled design fields (never per image)."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    counts, edges = np.histogram(flat, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    probability = counts.astype(np.float64) / max(counts.sum(), 1)
    omega = np.cumsum(probability)
    mu = np.cumsum(probability * centers)
    total = mu[-1]
    between = (total * omega - mu) ** 2 / np.maximum(omega * (1.0 - omega), 1e-30)
    if len(between) > 1:
        between[-1] = -np.inf
    return float(centers[int(np.argmax(between))])


class _UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[item] != item:
            nxt = int(self.parent[item])
            self.parent[item] = root
            item = nxt
        return root

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def periodic_component_count(mask: np.ndarray) -> int:
    """Four-connected component count on a torus."""
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    uf = _UnionFind(height * width)
    for y, x in np.argwhere(mask):
        index = int(y * width + x)
        for ny, nx in ((y, (x + 1) % width), ((y + 1) % height, x)):
            if mask[ny, nx]:
                uf.union(index, int(ny * width + nx))
    return len({uf.find(int(y * width + x)) for y, x in np.argwhere(mask)})


def periodic_euler_characteristic(mask: np.ndarray) -> int:
    """Cubical-complex Euler characteristic of foreground pixels on a torus."""
    faces = np.asarray(mask, dtype=bool)
    # Each unique periodic edge/vertex is counted if incident to a foreground face.
    horizontal_edges = faces | np.roll(faces, 1, axis=0)
    vertical_edges = faces | np.roll(faces, 1, axis=1)
    vertices = (
        faces | np.roll(faces, 1, axis=0) | np.roll(faces, 1, axis=1)
        | np.roll(np.roll(faces, 1, axis=0), 1, axis=1)
    )
    return int(vertices.sum() - horizontal_edges.sum() - vertical_edges.sum() + faces.sum())


def periodic_interface_length(mask: np.ndarray, normalized: bool = True) -> float:
    mask = np.asarray(mask, dtype=bool)
    length = np.count_nonzero(mask != np.roll(mask, 1, axis=0))
    length += np.count_nonzero(mask != np.roll(mask, 1, axis=1))
    return float(length / mask.size if normalized else length)


def structure_tensor_anisotropy(field: np.ndarray, epsilon: float = 1e-12) -> float:
    field = np.asarray(field, dtype=np.float64)
    gx = 0.5 * (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1))
    gy = 0.5 * (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0))
    jxx, jyy, jxy = np.mean(gx * gx), np.mean(gy * gy), np.mean(gx * gy)
    trace = jxx + jyy
    gap = np.sqrt(max((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
    return float(gap / (trace + epsilon))


def radial_spectrum(field: np.ndarray, bands: tuple[tuple[float, float], ...]) -> list[float]:
    field = np.asarray(field, dtype=np.float64)
    height, width = field.shape
    fy, fx = np.fft.fftfreq(height), np.fft.fftfreq(width)
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    power = np.abs(np.fft.fft2(field, norm="ortho")) ** 2 / (height * width)
    output = []
    for low, high in bands:
        band = (radius >= low) & (radius < high)
        output.append(float(power[band].sum()))
    return output


def field_metrics(
    field: np.ndarray,
    threshold: float,
    heldout_bands: tuple[tuple[float, float], ...] = ((0.20, 0.30), (0.30, 0.50)),
) -> dict[str, float]:
    field = np.asarray(field, dtype=np.float64)
    mask = field >= threshold
    components = periodic_component_count(mask)
    background_components = periodic_component_count(~mask)
    minority_components = components if mask.mean() <= 0.5 else background_components
    euler = periodic_euler_characteristic(mask)
    spectrum = radial_spectrum(field, heldout_bands)
    row = {
        "component_count": float(components),
        "background_component_count": float(background_components),
        "minority_component_count": float(minority_components),
        "phase_component_max": float(max(components, background_components)),
        "euler_characteristic": float(euler),
        "absolute_euler_characteristic": float(abs(euler)),
        "interface_length": periodic_interface_length(mask),
        "area_fraction": float(mask.mean()),
        "minkowski_area": float(mask.mean()),
        "minkowski_perimeter": periodic_interface_length(mask),
        "minkowski_euler": float(euler),
        "anisotropy": structure_tensor_anisotropy(field),
    }
    row.update({f"heldout_spectrum_{i + 1}": value for i, value in enumerate(spectrum)})
    return row


def metric_rows(fields: np.ndarray, threshold: float) -> list[dict[str, float]]:
    array = np.asarray(fields)
    if array.ndim != 4 or array.shape[1] != 1:
        raise ValueError("fields must have shape [B, 1, H, W]")
    return [field_metrics(field[0], threshold) for field in array]


def summarize_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot summarize an empty metric list")
    output = {}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        output[f"{key}_mean"] = float(np.mean(values))
        output[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return output


def weighted_metric_mean(rows: list[dict[str, float]], weights: np.ndarray) -> dict[str, float]:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    return {
        key: float(weights @ np.asarray([row[key] for row in rows], dtype=np.float64))
        for key in rows[0]
    }
